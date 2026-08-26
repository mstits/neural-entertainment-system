"""Phase 2 trainer for the discrete-time hazard model (no emulator).

Context (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md, v18/v19/v20):
Phase 1's micro-forking collector (scripts/hazard_collect.py, a
concurrent lane) writes an npz of (obs, action, died, steps_to_event,
censored) rows. This script fits a small MLP to those rows under a
discrete-time survival likelihood (src/training/hazard_model.py) and
reports THE GATE: Uno's C-index with inverse-probability-of-censoring
weighting (IPCW) on a held-out split. The synthesis doc's kill
criterion is explicit -- C-index < 0.85 means the tile observation does
not resolve threats well enough to act on, and the substrate stops here
without proceeding to Phase 3 (policy integration / hazard-masked PPO).

This script never touches the emulator: it loads an npz from disk,
trains a feedforward net against it on CPU, and writes a checkpoint +
a metrics report. `--data` must point at an already-collected npz;
producing one is scripts/hazard_collect.py's job, not this script's.

The held-out split is BY SOURCE STATE, not by row: rows forked from the
same saved state during micro-forking share almost everything about
that state and are not independent samples, so a row-level split would
let the same source state's outcome leak across train and val through
its siblings. See `src.training.hazard_model.infer_source_groups` for
how the group id is recovered from this exact npz schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.hazard_model import (  # noqa: E402
    DEFAULT_HIDDEN,
    DEFAULT_HIDDEN_LAYERS,
    NUM_ACTIONS,
    OBS_DIM,
    HazardMLP,
    build_time_bin_edges,
    concordance_index_ipcw,
    cumulative_risk_score,
    discrete_time_survival_nll,
    discretize_time,
    encode_input,
    infer_source_groups,
    SOURCE_ID_KEYS,
)

REQUIRED_KEYS = ("obs", "action", "died", "steps_to_event", "censored")

# scripts/hazard_collect.py's build_provenance() continuation_mode values
# (CONTINUATION_NOOP, CONTINUATION_POLICY). Duplicated as literals rather
# than imported so this script keeps its "never touches the emulator"
# guarantee -- hazard_collect.py pulls in src.emulation at module scope.
KNOWN_CONTINUATION_MODES = ("noop-after-intervention-tick", "policy-conditioned")


# ---------------------------------------------------------------------------
# Data loading + validation
# ---------------------------------------------------------------------------


def _check_data_provenance(path: str, npz) -> dict:
    """Surface whether `path` carries hazard_collect.py's meta_json
    provenance and an unrecognized continuation_mode. A stale
    pre-fork-horizon-fix npz, or any hand-built file, can satisfy every
    REQUIRED_KEYS/shape check below while never having asserted its
    continuation_mode -- this makes that state visible on stdout and in
    hazard_report.json instead of a normal-looking run."""
    if "meta_json" not in npz:
        print(f"[train_hazard] WARNING: {path} has no meta_json provenance "
              f"-- continuation_mode is UNVERIFIED (expected one of "
              f"{KNOWN_CONTINUATION_MODES}); this may be a stale "
              f"pre-fork-horizon-fix or hand-built dataset.", file=sys.stderr)
        return {"meta_json_present": False, "continuation_mode": None,
                "verified": False}
    raw = npz["meta_json"]
    try:
        meta = json.loads(str(raw))
        continuation_mode = meta.get("continuation_mode")
    except (json.JSONDecodeError, TypeError, AttributeError):
        continuation_mode = None
    verified = continuation_mode in KNOWN_CONTINUATION_MODES
    if not verified:
        print(f"[train_hazard] WARNING: {path}'s meta_json has "
              f"continuation_mode={continuation_mode!r}, not one of "
              f"{KNOWN_CONTINUATION_MODES} -- UNVERIFIED provenance.",
              file=sys.stderr)
    return {"meta_json_present": True, "continuation_mode": continuation_mode,
            "verified": verified}


def load_and_validate(path: str) -> dict:
    npz = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in npz]
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) {missing}; hazard_collect.py's "
            f"schema is {REQUIRED_KEYS}")
    data = {k: np.asarray(npz[k]) for k in REQUIRED_KEYS}
    data["_provenance"] = _check_data_provenance(path, npz)
    # Carry over an explicit source-state id column if the collector
    # wrote one (e.g. hazard_collect.py's `source_state_idx`) so
    # infer_source_groups can use it instead of falling back to
    # hashing `obs` -- dropping it here would silently force every
    # run onto the fallback path even when the real column exists.
    for key in SOURCE_ID_KEYS:
        if key in npz:
            data[key] = np.asarray(npz[key])
    n = data["obs"].shape[0]
    if data["obs"].ndim != 2 or data["obs"].shape[1] != OBS_DIM:
        raise ValueError(f"{path}: obs must be (N, {OBS_DIM}), got {data['obs'].shape}")
    for k in ("action", "died", "steps_to_event", "censored"):
        if data[k].shape[0] != n:
            raise ValueError(
                f"{path}: {k} has {data[k].shape[0]} rows, obs has {n}")
    n_mismatch = int(np.sum((1 - data["censored"]) != data["died"]))
    if n_mismatch:
        print(f"[train_hazard] warning: {n_mismatch}/{n} rows have "
              f"died != (1 - censored); using `censored` as the survival "
              f"loss's authoritative indicator (uncensored = 1 - censored).",
              file=sys.stderr)
    return data


def split_by_source_state(data: dict, val_frac: float, seed: int):
    """Group rows by inferred source state, then hold out whole groups
    so no source state straddles train/val."""
    groups = infer_source_groups(data)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_val_groups = max(1, int(round(len(shuffled) * val_frac))) if len(shuffled) > 1 else 0
    val_groups = set(shuffled[:n_val_groups].tolist())
    val_mask = np.isin(groups, list(val_groups))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx, len(unique_groups), n_val_groups


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def run_epoch(model, X, bin_idx, censored, *, batch_size: int,
              optimizer=None, generator: Optional[torch.Generator] = None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    n = X.shape[0]
    if train_mode:
        perm = torch.randperm(n, generator=generator)
    else:
        perm = torch.arange(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        logits = model(X[idx])
        loss = discrete_time_survival_nll(logits, bin_idx[idx], censored[idx])
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * len(idx)
    return total_loss / n


def evaluate_gate(model, X, steps_to_event: np.ndarray, censored: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        risk = cumulative_risk_score(logits).cpu().numpy()
    events = 1.0 - np.asarray(censored, dtype=np.float64)
    return concordance_index_ipcw(steps_to_event, events, risk)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Discrete-time hazard model trainer (Phase 2, no "
                     "emulator) -- fits src.training.hazard_model.HazardMLP "
                     "against a hazard_collect.py npz and reports the "
                     "IPCW C-index gate.")
    ap.add_argument("--data", required=True,
                     help="npz from scripts/hazard_collect.py "
                          f"(keys: {REQUIRED_KEYS}).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bins", type=int, default=20,
                     help="Number of discrete-time hazard bins covering "
                          "the observation horizon.")
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--hidden-layers", type=int, default=DEFAULT_HIDDEN_LAYERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True,
                     help="Output dir: writes hazard_model.pt (checkpoint) "
                          "and hazard_report.json (loss curve + gate "
                          "result) here.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                     help="Fraction of SOURCE-STATE GROUPS (not rows) "
                          "held out for the gate evaluation.")
    ap.add_argument("--horizon", type=float, default=None,
                     help="Override the modeled horizon (steps). Default: "
                          "max(steps_to_event) observed in --data.")
    ap.add_argument("--gate", type=float, default=0.85,
                     help="IPCW C-index threshold this run is judged "
                          "against (docs/proposals/"
                          "RESEARCH_SYNTHESIS_2026-08-17.md's Phase 2 gate).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Load + validate data, print the split, "
                          "construct the model -- no training, no writes.")
    return ap


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        data = load_and_validate(args.data)
    except (ValueError, FileNotFoundError, OSError, zipfile.BadZipFile) as e:
        print(f"[train_hazard] {e}", file=sys.stderr)
        return 2

    provenance = data.pop("_provenance")
    print(f"[train_hazard] data provenance: meta_json_present="
          f"{provenance['meta_json_present']} "
          f"continuation_mode={provenance['continuation_mode']} "
          f"verified={provenance['verified']}")

    n = data["obs"].shape[0]
    horizon = float(args.horizon) if args.horizon else float(
        max(1.0, data["steps_to_event"].max()))
    edges = build_time_bin_edges(horizon, args.bins)

    train_idx, val_idx, n_groups, n_val_groups = split_by_source_state(
        data, args.val_frac, args.seed)
    print(f"[train_hazard] {n} rows, {n_groups} source-state group(s) "
          f"({n_val_groups} held out for val); train={len(train_idx)} rows, "
          f"val={len(val_idx)} rows")
    if len(train_idx) == 0 or len(val_idx) == 0:
        print("[train_hazard] train or val split is empty after grouping by "
              "source state -- need more distinct source states, or a "
              "smaller/larger --val-frac.", file=sys.stderr)
        return 2

    model = HazardMLP(input_dim=OBS_DIM + NUM_ACTIONS, hidden=args.hidden,
                       n_hidden_layers=args.hidden_layers, n_bins=args.bins)
    print(f"[train_hazard] HazardMLP: input_dim={model.input_dim} "
          f"hidden={model.hidden} n_hidden_layers={model.n_hidden_layers} "
          f"n_bins={model.n_bins} params={model.n_params()} horizon={horizon}")

    if args.dry_run:
        print("[train_hazard] --dry-run: data validated, split computed, "
              "model constructed -- no training or writes performed.")
        return 0

    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))

    X_all = encode_input(data["obs"], data["action"], num_actions=NUM_ACTIONS)
    bin_idx_all = torch.as_tensor(discretize_time(data["steps_to_event"], edges))
    censored_all = torch.as_tensor(data["censored"].astype(np.float32))

    X_train = X_all[train_idx]
    bin_train = bin_idx_all[train_idx]
    cens_train = censored_all[train_idx]
    X_val = X_all[val_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, X_train, bin_train, cens_train,
                                batch_size=args.batch, optimizer=optimizer,
                                generator=gen)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"[train_hazard] epoch {epoch}/{args.epochs} "
                  f"train_loss={train_loss:.4f}")

    gate_result = evaluate_gate(
        model, X_val,
        data["steps_to_event"][val_idx], data["censored"][val_idx])
    c_index = gate_result["c_index"]
    passed = (not np.isnan(c_index)) and c_index >= args.gate
    verdict = "PASS" if passed else "FAIL"
    print(f"[train_hazard] GATE: IPCW C-index={c_index:.4f} "
          f"(threshold {args.gate}) -> {verdict}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "hazard_model.pt"
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": model.input_dim, "hidden": model.hidden,
            "n_hidden_layers": model.n_hidden_layers, "n_bins": model.n_bins,
            "horizon": horizon, "num_actions": NUM_ACTIONS,
        },
        "provenance": "hazard_model_phase2",
    }, str(tmp))
    os.replace(tmp, ckpt_path)

    report = {
        "data": str(args.data), "n_rows": n, "n_source_groups": n_groups,
        "n_val_groups": n_val_groups, "n_train_rows": len(train_idx),
        "n_val_rows": len(val_idx), "epochs": args.epochs, "lr": args.lr,
        "batch": args.batch, "bins": args.bins, "hidden": args.hidden,
        "hidden_layers": args.hidden_layers, "seed": args.seed,
        "horizon": horizon, "gate_threshold": args.gate,
        "history": history, "gate": gate_result, "gate_verdict": verdict,
        "data_provenance": provenance,
    }
    report_path = out_dir / "hazard_report.json"
    tmp_r = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp_r, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_r, report_path)

    print(f"[train_hazard] wrote checkpoint -> {ckpt_path}")
    print(f"[train_hazard] wrote report -> {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
