"""Adam optimizer-state analysis across the v27/v28 peak-instability checkpoints.

Loads vanilla_ppo_iter_*.pt (net + optimizer state_dict) for a set of runs,
reconstructs the exact net + RND-predictor + Adam optimizer used at train
time (src/training/trainer.py:_build_ppo_optimizer), round-trips the saved
optimizer state into it, and extracts per-parameter first/second-moment
statistics and the effective per-element learning rate
(lr / (sqrt(v_hat) + eps)) at every checkpointed iteration.

Output: one row per (run, iter, param_name) written to adam_state_report.csv,
plus a round-trip verification log and a printed summary, both saved
alongside this script.

Run from the repo root:
    .venv/bin/python runs/peak_instability/adam_optimizer_state/analyze_adam_state.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tile_policy import TilePolicyNetwork  # noqa: E402
from src.models.tile_rnd import TileRND  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

RUNS = {
    # assigned minimum
    "v28_seed3": "checkpoints/mario_1_1_v28_capacity_seed3",
    "v27_seed0": "checkpoints/mario_1_1_v27_recovery_seed0",
    # corroboration set (N=8 discipline -- don't generalize off 2 runs)
    "v27_seed2": "checkpoints/mario_1_1_v27_recovery_seed2",
    "v28_seed2": "checkpoints/mario_1_1_v28_capacity_seed2",
}

# peak iter per run, read straight from winners/best.json (authoritative,
# per the telemetry-trap note -- NOT derived from the training log).
def peak_iter(run_dir: Path) -> int:
    bj = json.loads((run_dir / "winners" / "best.json").read_text())
    return int(bj["source_iter"])


def infer_policy_dims(net_sd: dict) -> dict:
    hidden_dim, feature_dim = net_sd["fc1.weight"].shape
    trunk_dim = net_sd["fc2.weight"].shape[0]
    num_actions = net_sd["actor.weight"].shape[0]
    return dict(
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        trunk_dim=int(trunk_dim),
        num_actions=int(num_actions),
    )


def build_net_rnd_optim(net_sd: dict, rnd_sd: dict | None, lr: float):
    """Rebuild the exact objects _build_ppo_optimizer would have made,
    then load state into them. Returns (net, rnd_or_None, optim, names)
    where `names` lines up 1:1 with the optimizer's flat param list
    (and therefore with integer keys in optimizer_state_dict()["state"])."""
    dims = infer_policy_dims(net_sd)
    net = TilePolicyNetwork(
        num_actions=dims["num_actions"],
        feature_dim=dims["feature_dim"],
        hidden_dim=dims["hidden_dim"],
        trunk_dim=dims["trunk_dim"],
    )
    missing_net = net.load_state_dict(net_sd, strict=True)  # raises on mismatch

    rnd = None
    if rnd_sd is not None:
        pred_w0 = rnd_sd["predictor.net.0.weight"]
        pred_w6 = rnd_sd["predictor.net.6.weight"]
        rnd_hidden = int(pred_w0.shape[0])
        rnd_feat = int(pred_w6.shape[0])
        rnd = TileRND(
            feature_dim=dims["feature_dim"], feat_dim=rnd_feat, obs_clip=5.0
        )
        # hidden_dim is hardcoded 128 in TileRND/_TileRNDEncoder; verify
        # the checkpoint agrees before trusting the reconstruction.
        actual_hidden = rnd.predictor.net[0].out_features
        if actual_hidden != rnd_hidden:
            raise ValueError(
                f"RND hidden-dim mismatch: checkpoint has {rnd_hidden}, "
                f"TileRND default builds {actual_hidden}"
            )
        rnd.load_state_dict(rnd_sd, strict=True)  # raises on mismatch

    params = list(net.parameters())
    if rnd is not None:
        params += list(rnd.predictor.parameters())

    try:
        optim = torch.optim.Adam(params, lr=lr, fused=True)
        fused_ok = True
    except (RuntimeError, ValueError, TypeError):
        optim = torch.optim.Adam(params, lr=lr)
        fused_ok = False

    names = [n for n, _ in net.named_parameters()]
    if rnd is not None:
        names += [f"rnd.predictor.{n}" for n, _ in rnd.predictor.named_parameters()]

    return net, rnd, optim, names, fused_ok


def round_trip_check(ckpt: dict, lr: float) -> dict:
    """Load the saved optimizer_state_dict into a freshly-built optimizer
    over freshly-built (but state_dict-loaded) net+RND, and verify it
    round-trips: same param count, same tensor values bit-for-bit after
    a save/reload cycle, and a real .step() call does not raise."""
    net_sd = ckpt["net_state_dict"]
    rnd_sd = ckpt.get("rnd_state_dict")
    net, rnd, optim, names, fused_ok = build_net_rnd_optim(net_sd, rnd_sd, lr)

    saved_opt_sd = ckpt["optimizer_state_dict"]
    n_saved_params = len(saved_opt_sd["param_groups"][0]["params"])
    n_built_params = sum(1 for _ in optim.param_groups[0]["params"])
    count_match = n_saved_params == n_built_params == len(names)

    optim.load_state_dict(saved_opt_sd)  # raises on shape/key mismatch

    # bit-exact re-serialization check
    reloaded = optim.state_dict()
    exact = True
    for idx in saved_opt_sd["state"]:
        a = saved_opt_sd["state"][idx]
        b = reloaded["state"][idx]
        for key in ("exp_avg", "exp_avg_sq"):
            if not torch.equal(a[key], b[key]):
                exact = False
        if float(a["step"]) != float(b["step"]):
            exact = False

    # exercise a real step: fabricate a nonzero gradient on every param
    # (ones, not zero -- fused Adam's step-count bump only happens with
    # non-None .grad, and we want the identical code path train time used)
    # and confirm no exception + finite output.
    step_ok = True
    try:
        for p in optim.param_groups[0]["params"]:
            p.grad = torch.ones_like(p)
        optim.step()
        for p in optim.param_groups[0]["params"]:
            if not torch.isfinite(p).all():
                step_ok = False
    except Exception:
        step_ok = False

    return dict(
        count_match=count_match,
        n_saved_params=n_saved_params,
        n_built_params=n_built_params,
        n_names=len(names),
        state_dict_load_ok=True,  # reaching here means load_state_dict didn't raise
        bit_exact_reserialize=exact,
        post_load_step_ok=step_ok,
        fused_ok=fused_ok,
        names=names,
    )


def extract_moment_stats(ckpt: dict, names: list[str], beta2: float = 0.999,
                          eps: float = 1e-8, lr: float = 3e-4) -> list[dict]:
    opt_sd = ckpt["optimizer_state_dict"]
    pg = opt_sd["param_groups"][0]
    rows = []
    for name, idx in zip(names, pg["params"]):
        st = opt_sd["state"][idx]
        step = float(st["step"])
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
        bias2 = 1.0 - beta2 ** step if step > 0 else 1.0
        v_hat = v / bias2
        denom = v_hat.sqrt() + eps
        eff_lr = lr / denom  # elementwise effective step-size scale
        rows.append(dict(
            param=name,
            numel=m.numel(),
            step=step,
            m_mean_abs=float(m.abs().mean()),
            m_rms=float(m.pow(2).mean().sqrt()),
            v_mean=float(v.mean()),
            v_min=float(v.min()),
            v_max=float(v.max()),
            eff_lr_mean=float(eff_lr.mean()),
            eff_lr_median=float(eff_lr.median()),
            eff_lr_min=float(eff_lr.min()),
            eff_lr_max=float(eff_lr.max()),
            frac_v_near_zero=float((v < 1e-10).float().mean()),
        ))
    return rows


def main():
    lr = 3e-4  # confirmed from param_groups[0]["lr"] across checked checkpoints
    all_rows = []
    roundtrip_log = []

    for run_key, rel_dir in RUNS.items():
        run_dir = REPO_ROOT / rel_dir
        pk_iter = peak_iter(run_dir)
        ckpt_paths = sorted(
            run_dir.glob("vanilla_ppo_iter_*.pt"),
            key=lambda p: int(re.search(r"(\d+)\.pt$", p.name).group(1)),
        )
        did_roundtrip = False
        for cp in ckpt_paths:
            it = int(re.search(r"(\d+)\.pt$", cp.name).group(1))
            ckpt = torch.load(cp, map_location="cpu", weights_only=False)
            actual_lr = ckpt["optimizer_state_dict"]["param_groups"][0]["lr"]

            if not did_roundtrip:
                rt = round_trip_check(ckpt, actual_lr)
                roundtrip_log.append(dict(run=run_key, iter=it, **{
                    k: v for k, v in rt.items() if k != "names"
                }))
                names = rt["names"]
                did_roundtrip = True
            else:
                net_sd = ckpt["net_state_dict"]
                rnd_sd = ckpt.get("rnd_state_dict")
                dims = infer_policy_dims(net_sd)
                # names are architecture-derived and identical across
                # iters within a run; skip rebuilding, just reuse.

            stats = extract_moment_stats(ckpt, names, lr=actual_lr)
            for row in stats:
                row.update(run=run_key, iter=it, is_peak=(it == pk_iter))
                all_rows.append(row)

    # --- write CSV ---
    csv_path = OUT_DIR / "adam_state_report.csv"
    fieldnames = [
        "run", "iter", "is_peak", "param", "numel", "step",
        "m_mean_abs", "m_rms", "v_mean", "v_min", "v_max",
        "eff_lr_mean", "eff_lr_median", "eff_lr_min", "eff_lr_max",
        "frac_v_near_zero",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fieldnames})

    rt_path = OUT_DIR / "roundtrip_log.json"
    rt_path.write_text(json.dumps(roundtrip_log, indent=2))

    print(f"wrote {len(all_rows)} rows to {csv_path}")
    print(f"wrote round-trip log to {rt_path}")
    print()
    print("=== ROUND-TRIP SUMMARY ===")
    for r in roundtrip_log:
        print(r)


if __name__ == "__main__":
    main()
