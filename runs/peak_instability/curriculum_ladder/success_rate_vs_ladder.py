#!/usr/bin/env python3
"""Authoritative version of the entrance-success-curve check, using the
trainer's own per-iter `success_rate` field from metrics.jsonl instead of
differencing the log's cumulative entrance N/T counters.

Why this is better evidence: `success_rate` is emitted natively per
iteration (not reconstructed), so it is not subject to the two-space
regex bug or cumulative-counter noise that the hand-rolled log parser
has to fight. It is NOT directly comparable across the ladder-descent
phase (iters where tau > 0: the task gets HARDER each rung as tau
decreases, so a falling success_rate there is *expected curriculum
behavior*, not decay) -- it only becomes a fixed-task measure once
AT-ENTRANCE (tau=0), which is exactly the regime this investigation
cares about.

For each run this reports:
  - entrance_iter_tau0        (from the [backward] log, tau first hits 0)
  - post-entrance success_rate series: argmax iter/value, and the value
    at the run's final iter
  - whether that native peak iter lands near the honest-eval peak_iter
    already recorded in winners/best.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")

TAU_RE = re.compile(r"\[backward\] iter (\d+): tau=(\d+)/")

RUNS = [
    ("v27_seed0", "runs/v27_fresh_recovery/train_seed0.log", "checkpoints/mario_1_1_v27_recovery_seed0/metrics.jsonl", 60),
    ("v27_seed1", "runs/v27_fresh_recovery/train_seed1.log", "checkpoints/mario_1_1_v27_recovery_seed1/metrics.jsonl", 50),
    ("v27_seed2", "runs/v27_fresh_recovery/train_seed2.log", "checkpoints/mario_1_1_v27_recovery_seed2/metrics.jsonl", 90),
    ("v27_seed3", "runs/v27_fresh_recovery/train_seed3.log", "checkpoints/mario_1_1_v27_recovery_seed3/metrics.jsonl", 60),
    ("v28_seed0", "runs/v28_capacity/train_seed0.log", "checkpoints/mario_1_1_v28_capacity_seed0/metrics.jsonl", 70),
    ("v28_seed1", "runs/v28_capacity/train_seed1.log", "checkpoints/mario_1_1_v28_capacity_seed1/metrics.jsonl", 60),
    ("v28_seed2", "runs/v28_capacity/train_seed2.log", "checkpoints/mario_1_1_v28_capacity_seed2/metrics.jsonl", 120),
    ("v28_seed3", "runs/v28_capacity/train_seed3.log", "checkpoints/mario_1_1_v28_capacity_seed3/metrics.jsonl", 90),
]


def entrance_iter_of(log_path: Path) -> int:
    for line in open(log_path):
        m = TAU_RE.search(line)
        if m and int(m.group(2)) == 0:
            return int(m.group(1))
    raise ValueError(f"no tau=0 found in {log_path}")


def main():
    out = {}
    for name, log_rel, met_rel, honest_peak_iter in RUNS:
        log_path = REPO / log_rel
        met_path = REPO / met_rel
        entrance_iter = entrance_iter_of(log_path)

        rows = [json.loads(l) for l in open(met_path) if l.strip()]
        rows.sort(key=lambda r: r["generation"])
        post = [r for r in rows if r["generation"] >= entrance_iter]

        sr = [(r["generation"], r.get("success_rate")) for r in post if r.get("success_rate") is not None]
        if not sr:
            out[name] = {"error": "no success_rate rows post-entrance"}
            continue
        peak_gen, peak_val = max(sr, key=lambda x: x[1])
        final_gen, final_val = sr[-1]

        # iters at/above a "working policy" threshold (>=0.5 success), a
        # coarse read on how long the policy actually held the level.
        iters_ge_50 = [g for g, v in sr if v >= 0.5]
        iters_ge_90 = [g for g, v in sr if v >= 0.9]

        out[name] = {
            "entrance_iter_tau0": entrance_iter,
            "honest_eval_peak_iter": honest_peak_iter,
            "native_success_rate_peak_iter": peak_gen,
            "native_success_rate_peak_val": round(peak_val, 4),
            "native_success_rate_final_iter": final_gen,
            "native_success_rate_final_val": round(final_val, 4),
            "delta_honest_peak_minus_native_peak": honest_peak_iter - peak_gen,
            "n_iters_success_ge_0.5": len(iters_ge_50),
            "first_iter_success_ge_0.5": min(iters_ge_50) if iters_ge_50 else None,
            "last_iter_success_ge_0.5": max(iters_ge_50) if iters_ge_50 else None,
            "n_iters_success_ge_0.9": len(iters_ge_90),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
