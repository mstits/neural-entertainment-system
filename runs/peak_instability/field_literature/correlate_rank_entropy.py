"""Correlate the feature-rank probe against iteration and ppo_entropy.

Reads runs/peak_instability/field_literature/feature_rank.csv (produced by
feature_rank_probe.py) and each run's metrics.jsonl, and reports, per run:
  - Pearson r(srank, iter)      -- does effective rank trend with training time?
  - Pearson r(srank, ppo_entropy) -- does effective rank track entropy (and in
    which direction)?
  - srank at the honest-peak iter (from the curriculum_ladder sibling's
    peak_iter_bestjson) vs at the final logged checkpoint.

This is the direct test of the Moalla et al. 2024 (arXiv:2405.00662)
prediction that penultimate-layer feature rank DECLINES as PPO performance
collapses. A negative r(srank, iter) / falling srank-at-final would support
it; the result found here (see field_literature_findings output) is the
opposite sign in all 8/8 runs.

Run:
    .venv/bin/python runs/peak_instability/field_literature/correlate_rank_entropy.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RANK_CSV = Path(__file__).resolve().parent / "feature_rank.csv"

# From runs/peak_instability/curriculum_ladder/ladder_summary.json
# (peak_iter_bestjson == winners/best.json's authoritative peak iter).
PEAK_ITERS = {
    "mario_1_1_v27_recovery_seed0": 60,
    "mario_1_1_v27_recovery_seed1": 50,
    "mario_1_1_v27_recovery_seed2": 90,
    "mario_1_1_v27_recovery_seed3": 60,
    "mario_1_1_v28_capacity_seed0": 70,
    "mario_1_1_v28_capacity_seed1": 60,
    "mario_1_1_v28_capacity_seed2": 120,
    "mario_1_1_v28_capacity_seed3": 90,
}


def main() -> None:
    rows = list(csv.DictReader(open(RANK_CSV)))
    runs = sorted(set(r["run"] for r in rows))

    header = f"{'run':32s} {'r(srank,iter)':>14s} {'r(srank,entropy)':>18s} {'srank@peak':>11s} {'srank@final':>12s}"
    print(header)
    print("-" * len(header))

    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        iters = np.array([int(r["iter"]) for r in rr])
        sranks = np.array([int(r["srank_delta01"]) for r in rr])

        metrics_path = REPO_ROOT / "checkpoints" / run / "metrics.jsonl"
        ent_by_gen: dict[int, float] = {}
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                g, e = d.get("generation"), d.get("ppo_entropy")
                if g is not None and e is not None:
                    ent_by_gen[int(g)] = float(e)

        ents = np.array([ent_by_gen.get(i, np.nan) for i in iters])
        mask = ~np.isnan(ents)
        corr_iter = np.corrcoef(iters, sranks)[0, 1]
        corr_ent = (
            np.corrcoef(sranks[mask], ents[mask])[0, 1] if mask.sum() > 2 else float("nan")
        )

        pk = PEAK_ITERS[run]
        srank_peak = sranks[iters == pk][0] if pk in iters else None
        srank_final = sranks[iters == iters.max()][0]
        print(f"{run:32s} {corr_iter:14.3f} {corr_ent:18.3f} {srank_peak!s:>11s} {srank_final:>12d}")


if __name__ == "__main__":
    main()
