"""Direct checkpoint autopsy: measure the NETWORK, not its telemetry.

For every one of the 8 from-scratch runs (v27 seed0-3, v27 recovery
naming; v28 seed0-3, v28 capacity naming) and every saved checkpoint
(vanilla_ppo_iter_000{10..240}.pt, ~24 per run), this script:

  (a) loads the real net via build_tile_policy_from_checkpoint (the
      SAME dispatch scripts/eval_game.py uses -- shape-inferred
      hidden/trunk dims, so it works unmodified across v27's 48k
      (hidden=64) and v28's 72k (hidden=96) architectures);
  (b) runs it forward (no_grad, net.eval()) on the ONE fixed real
      observation batch captured by capture_obs_batch.py (1536 real
      stacked-RAM-tile vectors from the actual entrance start-state,
      same batch for every checkpoint in every run);
  (c) computes action-distribution entropy, argmax action-usage
      (literal collapse-to-N-actions), logit magnitude/saturation, and
      per-layer weight L2 norms plus checkpoint-to-checkpoint weight
      movement (raw delta norm + cosine similarity, so "still
      changing" and "still changing in a NEW direction" are reported
      separately -- a net can have nonzero delta while merely scaling
      an already-fixed direction, which is a distinct failure mode
      from actually exploring weight space).

Output: one row per (run, iter) to autopsy_metrics.csv, plus a full
action-histogram dump to action_histograms.json (every checkpoint,
every run -- small, 6 ints each).

Run:
    .venv/bin/python runs/peak_instability/checkpoint_autopsy/checkpoint_autopsy.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CKPT_ROOT = REPO_ROOT / "checkpoints"
NUM_ACTIONS = 6
FEATURE_DIM = 712
ITERS = list(range(10, 241, 10))

RUNS = [
    "mario_1_1_v27_recovery_seed0",
    "mario_1_1_v27_recovery_seed1",
    "mario_1_1_v27_recovery_seed2",
    "mario_1_1_v27_recovery_seed3",
    "mario_1_1_v28_capacity_seed0",
    "mario_1_1_v28_capacity_seed1",
    "mario_1_1_v28_capacity_seed2",
    "mario_1_1_v28_capacity_seed3",
]

# Layers common to TilePolicyNetwork regardless of hidden/trunk width
# (build_tile_policy_from_checkpoint infers width from these same
# tensors' shapes, so every run in RUNS produces exactly this key set).
LAYER_KEYS = [
    "fc1.weight", "fc1.bias",
    "norm1.weight", "norm1.bias",
    "fc2.weight", "fc2.bias",
    "norm2.weight", "norm2.bias",
    "actor.weight", "actor.bias",
    "critic.weight", "critic.bias",
]


def flat_all(sd: dict) -> torch.Tensor:
    return torch.cat([sd[k].detach().reshape(-1).float() for k in LAYER_KEYS])


def layer_norms(sd: dict) -> dict[str, float]:
    return {k: float(sd[k].detach().float().norm(2).item()) for k in LAYER_KEYS}


def entropy_of_probs(probs: torch.Tensor) -> torch.Tensor:
    # probs: (N, A). Clamp to avoid log(0) on a fully saturated softmax.
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def main() -> None:
    obs = np.load(OUT_DIR / "obs_batch.npy")
    x = torch.from_numpy(obs).float()
    print(f"obs batch: {x.shape}")

    rows: list[dict] = []
    action_histograms: dict[str, dict[int, list[int]]] = {}

    for run in RUNS:
        run_dir = CKPT_ROOT / run
        if not run_dir.exists():
            print(f"SKIP missing run dir: {run_dir}")
            continue
        action_histograms[run] = {}
        prev_sd: dict | None = None
        prev_iter: int | None = None
        for it in ITERS:
            ckpt_path = run_dir / f"vanilla_ppo_iter_{it:05d}.pt"
            if not ckpt_path.exists():
                continue
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            net, is_recurrent = build_tile_policy_from_checkpoint(
                state, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
            )
            assert not is_recurrent, f"{run} iter{it}: unexpected recurrent net"
            load_res = net.load_state_dict(state["net_state_dict"], strict=False)
            net.eval()
            sd = {k: v.clone() for k, v in net.state_dict().items()}

            with torch.no_grad():
                logits, _value = net.forward_ac(x)
                probs = torch.softmax(logits, dim=-1)

            ent = entropy_of_probs(probs)
            max_prob, argmax_idx = probs.max(dim=-1)
            hist = np.bincount(
                argmax_idx.numpy(), minlength=NUM_ACTIONS
            ).astype(int).tolist()
            n_actions_used = int(sum(1 for c in hist if c > 0))

            lnorms = layer_norms(sd)
            total_l2 = float(flat_all(sd).norm(2).item())

            row: dict = {
                "run": run,
                "iter": it,
                "hidden_dim": net.hidden_dim,
                "trunk_dim": net.trunk_dim,
                "num_params": int(sum(p.numel() for p in net.parameters())),
                "missing_keys": len(load_res.missing_keys),
                "unexpected_keys": len(load_res.unexpected_keys),
                "total_weight_l2": total_l2,
                "mean_entropy": float(ent.mean().item()),
                "std_entropy": float(ent.std().item()),
                "min_entropy": float(ent.min().item()),
                "max_entropy": float(ent.max().item()),
                "max_entropy_possible": float(np.log(NUM_ACTIONS)),
                "mean_max_prob": float(max_prob.mean().item()),
                "frac_saturated_p99": float((max_prob > 0.99).float().mean().item()),
                "frac_saturated_p999": float((max_prob > 0.999).float().mean().item()),
                "n_actions_used_argmax": n_actions_used,
                "frac_actions_used_argmax": n_actions_used / NUM_ACTIONS,
                "argmax_hist": hist,
                "dominant_action_frac": max(hist) / float(len(argmax_idx)),
                "mean_abs_logit": float(logits.abs().mean().item()),
                "max_abs_logit": float(logits.abs().max().item()),
                "std_logit": float(logits.std().item()),
            }
            for k in LAYER_KEYS:
                row[f"l2__{k}"] = lnorms[k]

            if prev_sd is not None:
                delta_total = float(
                    (flat_all(sd) - flat_all(prev_sd)).norm(2).item()
                )
                row["delta_iters"] = it - prev_iter
                row["delta_total_l2"] = delta_total
                row["delta_total_l2_relative"] = (
                    delta_total / row["total_weight_l2"]
                    if row["total_weight_l2"] > 0 else float("nan")
                )
                for k in LAYER_KEYS:
                    a = sd[k].detach().reshape(-1).float()
                    b = prev_sd[k].detach().reshape(-1).float()
                    d = float((a - b).norm(2).item())
                    row[f"delta_l2__{k}"] = d
                    denom = (a.norm(2) * b.norm(2)).item()
                    cos = float((a @ b).item() / denom) if denom > 0 else float("nan")
                    row[f"cos__{k}"] = cos
            else:
                row["delta_iters"] = None
                row["delta_total_l2"] = None
                row["delta_total_l2_relative"] = None
                for k in LAYER_KEYS:
                    row[f"delta_l2__{k}"] = None
                    row[f"cos__{k}"] = None

            rows.append(row)
            action_histograms[run][it] = hist
            prev_sd = sd
            prev_iter = it
            print(
                f"{run} iter={it:>3} H={row['mean_entropy']:.4f} "
                f"n_act_used={n_actions_used} dom_frac={row['dominant_action_frac']:.3f} "
                f"||W||={total_l2:.3f} "
                f"dW/W={row['delta_total_l2_relative']}"
            )

    if not rows:
        print("NO ROWS PRODUCED -- check RUNS/CKPT_ROOT paths")
        return

    # CSV: drop the list-valued field (argmax_hist lives in the JSON dump).
    fieldnames = [k for k in rows[0].keys() if k != "argmax_hist"]
    csv_path = OUT_DIR / "autopsy_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})
    print(f"wrote {csv_path} ({len(rows)} rows)")

    hist_path = OUT_DIR / "action_histograms.json"
    with open(hist_path, "w") as f:
        json.dump(action_histograms, f, indent=2)
    print(f"wrote {hist_path}")


if __name__ == "__main__":
    main()
