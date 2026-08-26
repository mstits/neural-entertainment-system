#!/usr/bin/env python3
"""Offline instrument sweep over banked training iterates — NO training.

WHY THIS EXISTS. `configs/mario_1_1_backward_v4.yaml` closes with a
registered action: *"Score the late `vanilla_ppo_iter_*.pt` checkpoints too —
the binary accept gate can only re-snapshot once, so the winner file is not
guaranteed to be the best policy the run produced."* One of them ever got
scored (iter 260 -> greedy 0.00, in
`checkpoints/mario_1_1_backward_v4/eval.jsonl`). The other eighteen, plus
twenty-four B4 iterates
(`checkpoints/mario_1_1_backward/prevrun_20260808_124336/`, iters 10-240),
are still sitting on disk unmeasured. This script is that sweep.

FIVE INSTRUMENTS, four of them free (no emulator, no ROM, no training):

  1. greedy/sampled paired clear-rate curve, under ONE named harness.
     THIS ONE COSTS EMULATION, so it is OPT-IN (`--eval`). By default the
     script STUB-COSTS it: it reports the episode count and the projected
     wall clock instead of running it, so the free instruments can be read
     in seconds and the paid one is a deliberate decision.
  2. top-two logit margin `M(s)` at supplied states — "is the argmax a tie?"
     measured, rather than asserted from an unsourced threshold.
  3. dormant-unit fraction (Sokar et al., ICML 2023) — the plasticity story,
     testable against the collapse arc rather than assumed.
  4. effective rank of the representation (or of a weight matrix) — the other
     half of the plasticity read.
  5. ``||theta_t - theta_0||`` parameter drift — and, as a by-product, the
     honest discriminator for the "the instrument was the bug" hypothesis:
     if a scored network is parameter-identical to another, no eval result
     can differ between them, and drift = 0 says the weights never moved.

Everything that can be a pure function of arrays is one, so the arithmetic is
unit-testable with duck-typed stubs and no checkpoints. The torch- and
emulator-touching parts are thin shells around those functions.

USAGE

    # free instruments over both banked runs, JSON to stdout
    .venv/bin/python scripts/score_banked_iterates.py \\
        --iterates 'checkpoints/mario_1_1_backward/prevrun_20260808_124336/vanilla_ppo_iter_*.pt' \\
        --iterates 'checkpoints/mario_1_1_backward_v4/vanilla_ppo_iter_*.pt' \\
        --states runs/instruments/stall_states.npy \\
        --out runs/instruments/b4_v4_sweep.json

    # ...and the paid one (subprocesses scripts/eval_game.py per iterate)
    ... --eval --profile configs/mario_1_1_backward_v4.yaml \\
        --start-state runs/live_show/smb_4_4_micro/entrance_start.state \\
        --episodes 30 --sticky-prob 0.25 --start-jitter 16 --eval-seed 7
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ===========================================================================
# PURE INSTRUMENTS — numpy in, numbers out. No torch, no I/O, no globals.
# ===========================================================================

def top_two_margin(logits: Any) -> np.ndarray:
    """Per-state gap between the best and second-best action logit.

    `logits` is `(N, A)` (a single `(A,)` row is accepted and treated as
    `N=1`). Returns an `(N,)` float array of ``max - second_max``, which is
    exactly the quantity an "argmax tie" claim is about: a near-zero margin
    means the greedy action is decided by numerical noise, while the sampled
    policy still visits both branches.

    A single-action space has no second-best; the margin is then +inf, since
    an argmax over one action cannot tie.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"logits must be (N, A) or (A,), got {arr.shape}")
    if arr.shape[1] < 2:
        return np.full(arr.shape[0], np.inf)
    part = np.partition(arr, -2, axis=1)
    return part[:, -1] - part[:, -2]


def margin_summary(logits: Any, *, tie_threshold: float = 0.15) -> dict:
    """Distributional read of :func:`top_two_margin` (pure).

    `tie_threshold` is REPORTED, not believed: the fraction of states under
    it is emitted alongside the quantiles so the threshold can be calibrated
    against a healthy iterate instead of asserted a priori.
    """
    m = top_two_margin(logits)
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return {"n_states": int(m.size), "mean": None, "median": None,
                "p05": None, "p25": None, "min": None,
                "tie_threshold": float(tie_threshold), "tie_fraction": None}
    return {
        "n_states": int(m.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "min": float(np.min(finite)),
        "tie_threshold": float(tie_threshold),
        "tie_fraction": float(np.mean(finite < float(tie_threshold))),
    }


def dormant_fraction(activations: Any, *, tau: float = 0.0) -> float:
    """Fraction of units that are DORMANT under the Sokar et al. score.

    (Sokar, Agarwal, Castro & Evci, "The Dormant Neuron Phenomenon in Deep
    Reinforcement Learning", ICML 2023.) For a layer with activations `(N,
    U)`, unit i's score is its mean absolute activation divided by the layer's
    mean over all units:

        s_i = E_x|h_i(x)| / ( (1/U) * sum_j E_x|h_j(x)| )

    and unit i is tau-dormant when ``s_i <= tau``. `tau = 0.0` is the strict
    definition (the unit is exactly dead). A layer whose activations are all
    zero has an undefined normalizer; every unit is then dormant by
    definition, which is what this returns (1.0) rather than a NaN.
    """
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] == 0:
        raise ValueError(f"activations must be (N, U), got {a.shape}")
    per_unit = np.mean(np.abs(a), axis=0)
    denom = float(np.mean(per_unit))
    if denom <= 0.0:
        return 1.0
    scores = per_unit / denom
    return float(np.mean(scores <= float(tau)))


def effective_rank(matrix: Any, *, delta: float = 0.01) -> dict:
    """Two effective-rank readings of a matrix (pure).

    * ``srank`` — the smallest k whose top-k singular values carry at least
      ``1 - delta`` of the total singular mass (Kumar et al.'s implicit
      under-parameterisation measure). Integer, easy to read across a run.
    * ``erank`` — ``exp(H(p))`` where ``p`` is the singular spectrum
      normalized to a distribution (Roy & Vetterli). Continuous, so it moves
      before ``srank`` snaps.

    An all-zero matrix has no spectrum: both are 0.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {m.shape}")
    sv = np.linalg.svd(m, compute_uv=False)
    total = float(np.sum(sv))
    if total <= 0.0:
        return {"srank": 0, "erank": 0.0, "n_singular_values": int(sv.size)}
    cumulative = np.cumsum(sv) / total
    srank = int(np.searchsorted(cumulative, 1.0 - float(delta)) + 1)
    p = sv / total
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "srank": min(srank, int(sv.size)),
        "erank": float(np.exp(entropy)),
        "n_singular_values": int(sv.size),
    }


def param_drift(theta_t: dict, theta_0: dict) -> dict:
    """``||theta_t - theta_0||`` over the shared tensors (pure-ish).

    Duck-typed: values may be numpy arrays or anything with a
    ``detach().cpu().numpy()`` chain (i.e. torch tensors). Returns the global
    L2 distance, the relative drift against ``||theta_0||``, the count of
    compared tensors, and the single largest per-tensor contribution — which
    is what tells you whether a run moved its trunk or only its heads.

    A tensor present in both state_dicts but with a different shape (e.g. a
    resized hidden layer) cannot be compared and is excluded from the sum —
    but it is named in ``skipped_tensors`` rather than silently vanishing, so
    a drift computed over a shrunken trunk cannot be mistaken for one over
    the whole network.

    ``identical`` is the discriminator for the "was the scored network the
    trained network?" question: a drift of exactly 0 means the two
    state_dicts are the same weights, and no eval difference between them can
    be real. It is never true while a tensor was skipped for a shape
    mismatch — networks that disagree on shape are not identical, whatever
    the comparable tensors say.
    """
    total_sq = 0.0
    base_sq = 0.0
    per_tensor: dict = {}
    skipped: list = []
    shared = [k for k in theta_t if k in theta_0]
    for k in shared:
        a = _to_numpy(theta_t[k]).astype(np.float64).ravel()
        b = _to_numpy(theta_0[k]).astype(np.float64).ravel()
        if a.shape != b.shape:
            skipped.append(k)
            continue
        d = float(np.sum((a - b) ** 2))
        per_tensor[k] = float(np.sqrt(d))
        total_sq += d
        base_sq += float(np.sum(b ** 2))
    l2 = float(np.sqrt(total_sq))
    base = float(np.sqrt(base_sq))
    largest = max(per_tensor.items(), key=lambda kv: kv[1]) if per_tensor else None
    return {
        "l2": l2,
        "relative": (l2 / base) if base > 0 else None,
        "n_tensors": len(per_tensor),
        "identical": bool(l2 == 0.0 and per_tensor and not skipped),
        "largest_tensor": largest[0] if largest else None,
        "largest_tensor_l2": largest[1] if largest else None,
        "skipped_tensors": sorted(skipped),
        "n_skipped": len(skipped),
    }


def _to_numpy(v: Any) -> np.ndarray:
    """numpy view of an array-like or a torch tensor (duck-typed)."""
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy())
    return np.asarray(v)


def iterate_number(path: Any) -> Optional[int]:
    """The iteration number encoded in a ``vanilla_ppo_iter_NNNNN.pt`` name."""
    m = re.search(r"iter[_-]?(\d+)", str(path))
    return int(m.group(1)) if m else None


def sort_iterates(paths: Iterable[Any]) -> list:
    """Iterate paths ordered by their iteration number (unnumbered last)."""
    return sorted(paths, key=lambda p: (iterate_number(p) is None,
                                        iterate_number(p) or 0, str(p)))


def eval_cost_estimate(
    n_iterates: int, episodes: int, *, modes: int = 2,
    seconds_per_episode: float = 4.5, workers: int = 1,
) -> dict:
    """STUB COST for the paid instrument (pure arithmetic, no emulation).

    `seconds_per_episode` defaults to the measured worst case for this
    lineage: a stuck argmax burning the 1500-step cap
    (`configs/mario_1_1_backward_v4.yaml`, ~4.5 s). Reported so the decision
    to spend the emulation is made against a number rather than a vibe.
    """
    eps = int(n_iterates) * int(episodes) * int(modes)
    serial = eps * float(seconds_per_episode)
    return {
        "n_iterates": int(n_iterates),
        "episodes_per_cell": int(episodes),
        "modes": int(modes),
        "total_episodes": eps,
        "seconds_per_episode": float(seconds_per_episode),
        "serial_seconds": serial,
        "workers": max(1, int(workers)),
        "wall_seconds_estimate": serial / max(1, int(workers)),
    }


# ===========================================================================
# THIN SHELLS — torch / emulator. Everything above stays importable without
# either, so the pure suite runs with numpy alone.
# ===========================================================================

def load_iterate(path: Any) -> dict:
    """The policy `state_dict` inside a trainer checkpoint (CPU tensors)."""
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        for key in ("net_state_dict", "state_dict", "model_state_dict"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
        if all(hasattr(v, "shape") for v in blob.values()):
            return blob
    raise ValueError(f"{path}: no recognisable policy state_dict in checkpoint")


def build_tile_net(state_dict: dict):
    """Reconstruct a `TilePolicyNetwork` sized from the weights themselves.

    The trainer's checkpoints carry only `net_state_dict`, so the widths are
    read off `fc1/fc2/actor` rather than from a profile — which also means the
    instruments cannot be silently run against a differently-shaped net.
    """
    from src.models.tile_policy import TilePolicyNetwork

    try:
        hidden, feature = tuple(_to_numpy(state_dict["fc1.weight"]).shape)
        trunk, _ = tuple(_to_numpy(state_dict["fc2.weight"]).shape)
        actions, _ = tuple(_to_numpy(state_dict["actor.weight"]).shape)
    except KeyError as e:  # pragma: no cover - guarded by the caller
        raise ValueError(
            f"state_dict is not a TilePolicyNetwork (missing {e})"
        ) from e
    net = TilePolicyNetwork(
        num_actions=int(actions), feature_dim=int(feature),
        hidden_dim=int(hidden), trunk_dim=int(trunk),
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def tile_forward_with_trunk(net, states: Any) -> tuple:
    """`(logits, trunk_activations)` for `(N, feature_dim)` states.

    The trunk activation is the post-SiLU representation the heads read, i.e.
    the layer whose dormancy and effective rank the plasticity story is about.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.asarray(states, dtype=np.float32))
    with torch.no_grad():
        h1 = F.silu(net.norm1(net.fc1(x)))
        h2 = F.silu(net.norm2(net.fc2(h1)))
        logits = net.actor(h2)
    return _to_numpy(logits), _to_numpy(h2), _to_numpy(h1)


def score_one_iterate(
    path: Any,
    *,
    states: Optional[np.ndarray],
    theta_0: Optional[dict],
    tie_threshold: float,
    dormant_tau: float,
    srank_delta: float,
    forward: Optional[Callable] = None,
) -> dict:
    """Every FREE instrument for one iterate. Emulator-free by construction.

    `forward` is injectable (duck-typed: ``states -> (logits, trunk, hidden)``)
    so the sweep can be exercised end to end without torch or a checkpoint.
    """
    row: dict = {"path": str(path), "iter": iterate_number(path)}
    sd = load_iterate(path) if forward is None else {}
    if theta_0 is not None and sd:
        row["drift"] = param_drift(sd, theta_0)
    if states is None:
        return row
    fwd = forward
    if fwd is None:
        net = build_tile_net(sd)

        def fwd(s, _net=net):
            return tile_forward_with_trunk(_net, s)

    logits, trunk, hidden = fwd(states)
    row["margin"] = margin_summary(logits, tie_threshold=tie_threshold)
    row["dormant"] = {
        "trunk": dormant_fraction(trunk, tau=dormant_tau),
        "hidden": dormant_fraction(hidden, tau=dormant_tau),
        "tau": float(dormant_tau),
    }
    row["effective_rank"] = {
        "trunk": effective_rank(trunk, delta=srank_delta),
        "hidden": effective_rank(hidden, delta=srank_delta),
        "delta": float(srank_delta),
    }
    return row


def score_curve_cell(
    path: Any, *, profile: str, rom: Optional[str], start_state: Optional[str],
    episodes: int, max_steps: int, sticky_prob: float, start_jitter: int,
    eval_seed: int, eval_workers: int, action_select: str,
) -> dict:
    """ONE paid cell: subprocess `eval_game.py` for this iterate + mode.

    Uses the same harness for every cell (``--sequential --level-clear`` from
    an explicit start state) because the point of the curve is that greedy and
    sampled are PAIRED — a curve assembled from two harnesses is not a curve.
    """
    import yaml

    from src.training import cold_probe

    cfg = yaml.safe_load(Path(profile).read_text())
    sd = load_iterate(path)
    net = build_tile_net(sd)
    res = cold_probe.probe(
        net, cfg, episodes=episodes, sequential=True, level_clear=True,
        start_state=start_state, max_steps=max_steps,
        rom_path=rom or cfg.get("rom_path"),
        game=str(cfg.get("name", "mario")),
        sticky_prob=sticky_prob, start_jitter=start_jitter,
        eval_seed=eval_seed, eval_workers=eval_workers,
        eval_rng="per-episode" if eval_workers > 1 else None,
        action_select=action_select,
    )
    return {
        "action_select": action_select,
        "clear_rate": res.get("cold_seq_clear_rate"),
        "n_episodes": res.get("cold_n_episodes"),
        "mean_length": res.get("cold_mean_length"),
        "status": res.get("cold_status"),
        "error": res.get("cold_error"),
        "protocol": {
            "sticky_prob": float(sticky_prob),
            "start_jitter": int(start_jitter),
            "eval_seed": int(eval_seed),
            "eval_workers": int(eval_workers),
            "max_steps": int(max_steps),
            "harness": "eval_game.py --sequential --level-clear",
        },
    }


def expand_iterates(patterns: Sequence[str]) -> list:
    """Glob every `--iterates` pattern, de-duplicate, order by iteration."""
    hits: list = []
    seen: set = set()
    for pat in patterns:
        for p in glob.glob(pat):
            rp = str(Path(p).resolve())
            if rp not in seen:
                seen.add(rp)
                hits.append(Path(p))
    return sort_iterates(hits)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline instrument sweep over banked iterates "
                    "(no training; emulation is opt-in).",
    )
    ap.add_argument("--iterates", action="append", default=[], metavar="GLOB",
                    help="Glob of checkpoints to score. Repeatable.")
    ap.add_argument("--baseline", default=None, metavar="CKPT",
                    help="theta_0 for the drift instrument. Default: the "
                         "lowest-numbered iterate in the sweep.")
    ap.add_argument("--states", default=None, metavar="NPY",
                    help="(N, feature_dim) float array of observations — the "
                         "states the logit margin / dormancy / effective rank "
                         "are measured AT. Omit to run drift only.")
    ap.add_argument("--tie-threshold", type=float, default=0.15,
                    help="Reported (not believed) margin tie threshold; "
                         "calibrate it against a healthy iterate.")
    ap.add_argument("--dormant-tau", type=float, default=0.0,
                    help="Sokar tau. 0.0 = the strict (exactly dead) rule.")
    ap.add_argument("--srank-delta", type=float, default=0.01)
    ap.add_argument("--out", default=None, metavar="JSON",
                    help="Write the sweep here as well as to stdout.")
    # --- the paid instrument -------------------------------------------
    ap.add_argument("--eval", action="store_true",
                    help="Actually run the greedy/sampled curve (subprocesses "
                         "eval_game.py per iterate). OFF by default: the "
                         "sweep stub-costs it instead.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--sticky-prob", type=float, default=0.25)
    ap.add_argument("--start-jitter", type=int, default=16)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--seconds-per-episode", type=float, default=4.5,
                    help="Cost model for the stub (measured worst case).")
    args = ap.parse_args(argv)

    paths = expand_iterates(args.iterates)
    if not paths:
        print("no iterates matched", file=sys.stderr)
        return 2

    states = None
    if args.states:
        states = np.asarray(np.load(args.states), dtype=np.float32)

    theta_0 = None
    base_path = args.baseline or (str(paths[0]) if paths else None)
    if base_path:
        theta_0 = load_iterate(base_path)

    rows = []
    for p in paths:
        try:
            row = score_one_iterate(
                p, states=states, theta_0=theta_0,
                tie_threshold=args.tie_threshold, dormant_tau=args.dormant_tau,
                srank_delta=args.srank_delta,
            )
            if args.eval:
                if not args.profile:
                    print("--eval needs --profile", file=sys.stderr)
                    return 2
                row["curve"] = [
                    score_curve_cell(
                        p, profile=args.profile, rom=args.rom,
                        start_state=args.start_state, episodes=args.episodes,
                        max_steps=args.max_steps, sticky_prob=args.sticky_prob,
                        start_jitter=args.start_jitter, eval_seed=args.eval_seed,
                        eval_workers=args.eval_workers, action_select=mode,
                    )
                    for mode in ("greedy", "sampled")
                ]
        except Exception as exc:
            # One bad iterate (truncated .pt, unrecognised architecture) must
            # not cost every iterate already scored this run: report it in
            # its own row and keep going, so stdout/--out still land.
            print(f"{p}: {exc}", file=sys.stderr)
            row = {"path": str(p), "iter": iterate_number(p), "error": str(exc)}
        rows.append(row)

    out = {
        "baseline": base_path,
        "n_iterates": len(rows),
        "states": args.states,
        "n_states": int(states.shape[0]) if states is not None else 0,
        "eval_ran": bool(args.eval),
        "iterates": rows,
    }
    if not args.eval:
        out["curve_stub_cost"] = eval_cost_estimate(
            len(rows), args.episodes,
            seconds_per_episode=args.seconds_per_episode,
            workers=args.eval_workers,
        )
        out["curve_stub_note"] = (
            "greedy/sampled curve NOT run (--eval omitted); the cost above is "
            "an estimate, not a measurement"
        )
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
