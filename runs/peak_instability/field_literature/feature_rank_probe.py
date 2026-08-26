"""Retrospective feature-rank / representation-collapse probe.

Tests one specific external-literature prediction against data already on
disk, with zero new training: Moalla et al. 2024 ("No Representation, No
Trust: Connecting Representation, Collapse, and Trust Issues in PPO",
arXiv:2405.00662) report that PPO's late-training performance collapse is
preceded by a decline in the effective rank of the penultimate-layer
feature representation, and that this rank collapse is measurable
independent of dormant-unit counts (their "capacity loss" is explicitly
distinct from ReDo-style dormancy).

This project's own ReDo audit already found zero dormant units across all
8 runs (see docs/research and the whole_field_sweep/checkpoint_autopsy
siblings). Rank collapse is a DIFFERENT, unmeasured quantity: it can fall
even while every unit is technically "alive" (nonzero mean activation) if
the units become linearly redundant. This script checks whether that is
happening here.

Method, reusing the exact loader + fixed observation batch the sibling
checkpoint_autopsy already validated (same net reconstruction, same 1536
real entrance-state observations, so results are directly comparable to
autopsy_metrics.csv):

  1. Load each saved checkpoint (iter 10..240 step 10, ~24/run) with
     build_tile_policy_from_checkpoint (shape-inferred widths).
  2. Forward the fixed obs batch through fc1->norm1->SiLU->fc2->norm2->SiLU
     to get the trunk_dim-wide representation fed to the actor/critic
     heads (this project's "penultimate layer").
  3. Compute effective rank via the Kumar et al. 2020 / Moalla et al. 2024
     definition: srank_delta(h) = min{ k : sum(top-k sigma^2) / sum(sigma^2)
     > 1 - delta }, delta = 0.01, on the (1536 x trunk_dim) feature matrix.
     Also report the participation-ratio rank (sum(sigma^2))^2 / sum(sigma^4)
     as a smoother companion statistic.
  4. Report both against the checkpoint's iter, alongside the honest
     peak/final markers already established (winners/best.json + the
     curriculum_ladder sibling's honest_peak/honest_final).

Run:
    .venv/bin/python runs/peak_instability/field_literature/feature_rank_probe.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CKPT_ROOT = REPO_ROOT / "checkpoints"
AUTOPSY_DIR = REPO_ROOT / "runs" / "peak_instability" / "checkpoint_autopsy"
NUM_ACTIONS = 6
FEATURE_DIM = 712
ITERS = list(range(10, 241, 10))
DELTA = 0.01

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


def srank(sigma: np.ndarray, delta: float) -> int:
    sq = sigma.astype(np.float64) ** 2
    total = sq.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(sq)
    k = int(np.searchsorted(cum, (1.0 - delta) * total) + 1)
    return min(k, len(sigma))


def participation_ratio(sigma: np.ndarray) -> float:
    sq = sigma.astype(np.float64) ** 2
    denom = np.sum(sq ** 2)
    if denom <= 0:
        return 0.0
    return float((np.sum(sq) ** 2) / denom)


def trunk_features(net, x: torch.Tensor) -> torch.Tensor:
    h = F.silu(net.norm1(net.fc1(x)))
    h = F.silu(net.norm2(net.fc2(h)))
    return h


def main() -> None:
    obs = np.load(AUTOPSY_DIR / "obs_batch.npy")
    x = torch.from_numpy(obs).float()
    print(f"obs batch: {x.shape}")

    rows = []
    for run in RUNS:
        run_dir = CKPT_ROOT / run
        for it in ITERS:
            ckpt_path = run_dir / f"vanilla_ppo_iter_{it:05d}.pt"
            if not ckpt_path.exists():
                continue
            try:
                net, is_recurrent = build_tile_policy_from_checkpoint(
                    ckpt_path, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
                    load_weights=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {run} iter {it}: {exc}")
                continue
            if is_recurrent:
                print(f"SKIP {run} iter {it}: recurrent net, probe assumes stateless")
                continue
            net.eval()
            with torch.no_grad():
                h = trunk_features(net, x)
            trunk_dim = h.shape[1]
            hc = h - h.mean(dim=0, keepdim=True)
            sigma = torch.linalg.svdvals(hc).numpy()
            s_rank = srank(sigma, DELTA)
            pr = participation_ratio(sigma)
            feat_norm = h.norm(dim=1).mean().item()
            rows.append({
                "run": run,
                "iter": it,
                "trunk_dim": trunk_dim,
                "srank_delta01": s_rank,
                "srank_frac_of_trunk": s_rank / trunk_dim,
                "participation_ratio": round(pr, 4),
                "pr_frac_of_trunk": round(pr / trunk_dim, 4),
                "mean_feature_norm": round(feat_norm, 4),
                "top_sigma": round(float(sigma[0]), 4) if len(sigma) else 0.0,
            })
            print(f"{run} iter {it}: srank={s_rank}/{trunk_dim} pr={pr:.2f} "
                  f"||h||={feat_norm:.3f}")

    out_csv = OUT_DIR / "feature_rank.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
