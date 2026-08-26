#!/usr/bin/env python3
"""Cross-run (N=8) correlations between value-loss summary stats and honest
peak score. Purely descriptive -- N=8 is too small for significance testing,
report as a lead not a finding.

Usage:
    .venv/bin/python runs/peak_instability/value_critic_health/correlations.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "runs" / "peak_instability" / "value_critic_health" / "summary.csv"


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy)


def rank(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for r, i in enumerate(idx):
        ranks[i] = r
    return ranks


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    for r in rows:
        for k in r:
            if k != "run":
                r[k] = float(r[k])

    honest_peak = [r["honest_peak"] for r in rows]
    fields = {
        "vloss_at_peak_mean": "value loss at the honest-peak checkpoint",
        "vloss_global_min": "lowest value loss anywhere in the run",
        "vloss_first10_mean": "value loss during the shared early cold-start window",
        "ent_at_peak_mean": "entropy at the honest-peak checkpoint (cross-check only, not this dimension's axis)",
    }

    print(f"runs (n=8): {[r['run'] for r in rows]}")
    print(f"honest_peak: {honest_peak}")
    print()
    for field, desc in fields.items():
        xs = [r[field] for r in rows]
        p = pearson(xs, honest_peak)
        s = spearman(xs, honest_peak)
        print(f"{field:22s} ({desc})")
        print(f"    pearson r = {p:+.3f}   spearman r = {s:+.3f}")
    print()
    print("N=8 -- report as a lead, not a finding. Direction of causation is not")
    print("established: a lower value-loss floor could reflect a healthier critic")
    print("enabling better policy learning, OR simply reflect that a policy that")
    print("(for other reasons) finds a more consistent/higher-reward behavior")
    print("produces lower-variance return targets that are trivially easier to fit.")


if __name__ == "__main__":
    main()
