#!/usr/bin/env python3
"""Policy-update-magnitude / trust-region dimension, peak-instability investigation.

Extracts ppo_loss, ppo_policy_loss, ppo_value_loss, ppo_entropy from all 8
metrics.jsonl runs (v27_recovery seed0-3, v28_capacity seed0-3), joins each
run's authoritative peak iter from checkpoints/<run>/winners/best.json, and
computes windowed statistics + a policy/entropy correlation used to check
whether "policy_loss -> ~0 late in training" is a distinct trust-region
signature or a trivial restatement of the entropy collapse.

NOTE ON WHAT IS NOT HERE, ON PURPOSE:
No clip_fraction, no KL divergence, no grad_norm field exists anywhere in
metrics.jsonl or in the training logs for these runs (verified by grep over
both train_seed*.log files and the ppo_losses()/ppo_updater.py source).
This script cannot compute those because the training code never computed
them either. Everything below is inferred from the four loss/entropy
scalars that DO exist.

IMPORTANT INSTRUMENTATION CAVEAT (read before trusting any single-iter
value here): ppo_updater.py holds only the FINAL minibatch's loss tensors
per outer iteration ("Hold the final minibatch's loss tensors ... only the
last is logged" -- see the comment above the epoch loop) out of
K=10 epochs x ~240 minibatches (rollout 1024*60=61440 valid steps /
mb_size 256) = ~2400 gradient steps per iteration. Every ppo_policy_loss /
ppo_value_loss / ppo_entropy / ppo_loss value in metrics.jsonl is therefore
a ONE-OUT-OF-~2400 SAMPLE of that iteration's updates, not a mean or max.
A spike in this field is real evidence something extreme happened in at
least the last minibatch; the ABSENCE of a spike says nothing about the
other ~2399 minibatches that iteration ran and never got observed.

Usage:
    .venv/bin/python runs/peak_instability/policy_update_magnitude/extract_policy_update_stats.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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
FIELDS = ["generation", "ppo_loss", "ppo_policy_loss", "ppo_value_loss", "ppo_entropy"]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def pearson(xs, ys):
    pairs = [
        (x, y) for x, y in zip(xs, ys)
        if x is not None and y is not None
        and isinstance(x, (int, float)) and isinstance(y, (int, float))
        and math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = mean(xs2), mean(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x in xs2) ** 0.5
    dy = sum((y - my) ** 2 for y in ys2) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def frac_near_zero(xs, eps):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return None
    return sum(1 for x in xs if abs(x) < eps) / len(xs)


def main():
    results = {}
    for run in RUNS:
        metrics_path = REPO / "checkpoints" / run / "metrics.jsonl"
        best_path = REPO / "checkpoints" / run / "winners" / "best.json"
        rows = read_jsonl(metrics_path)
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        peak_iter = best.get("source_iter")
        peak_val = best.get("metric_value")

        by_gen = {r.get("generation"): r for r in rows if "generation" in r}
        max_gen = max(by_gen) if by_gen else None

        def window(lo, hi):
            return [by_gen[g] for g in sorted(by_gen) if lo <= g < hi]

        pre = window(0, peak_iter) if peak_iter is not None else []
        near = window(peak_iter, peak_iter + 40) if peak_iter is not None else []
        late = window(max_gen - 40, max_gen + 1) if max_gen is not None else []

        def stats_block(block, field):
            vals = [r.get(field) for r in block]
            return {
                "n": len(vals),
                "mean": mean(vals),
                "std": std(vals),
                "mean_abs": mean([abs(v) for v in vals if v is not None]),
            }

        full_policy = [r.get("ppo_policy_loss") for r in rows]
        full_entropy = [r.get("ppo_entropy") for r in rows]
        full_value = [r.get("ppo_value_loss") for r in rows]
        full_loss = [r.get("ppo_loss") for r in rows]
        full_gen = [r.get("generation") for r in rows]

        # decomposition sanity check at a handful of iters:
        # ppo_loss ?= policy_loss + value_coef*value_loss - entropy_coef*entropy
        # value_coef=0.25, entropy_coef=0.005 for all 8 runs (config-verified)
        decomp_residuals = []
        for r in rows:
            pl, vl, ent, tot = (
                r.get("ppo_policy_loss"), r.get("ppo_value_loss"),
                r.get("ppo_entropy"), r.get("ppo_loss"),
            )
            if None not in (pl, vl, ent, tot):
                pred = pl + 0.25 * vl - 0.005 * ent
                decomp_residuals.append(abs(pred - tot))

        results[run] = {
            "peak_iter": peak_iter,
            "peak_honest_metric": peak_val,
            "max_gen_logged": max_gen,
            "n_rows": len(rows),
            "windows": {
                "pre_peak_[0,peak)": {
                    f: stats_block(pre, f) for f in ["ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_loss"]
                },
                "near_peak_[peak,peak+40)": {
                    f: stats_block(near, f) for f in ["ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_loss"]
                },
                "late_[max-40,max]": {
                    f: stats_block(late, f) for f in ["ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_loss"]
                },
            },
            "frac_policy_loss_abs_lt_0.01": {
                "pre_peak": frac_near_zero([r.get("ppo_policy_loss") for r in pre], 0.01),
                "near_peak": frac_near_zero([r.get("ppo_policy_loss") for r in near], 0.01),
                "late": frac_near_zero([r.get("ppo_policy_loss") for r in late], 0.01),
            },
            "frac_policy_loss_abs_lt_0.005": {
                "pre_peak": frac_near_zero([r.get("ppo_policy_loss") for r in pre], 0.005),
                "near_peak": frac_near_zero([r.get("ppo_policy_loss") for r in near], 0.005),
                "late": frac_near_zero([r.get("ppo_policy_loss") for r in late], 0.005),
            },
            "corr_abs_policy_loss_vs_entropy_full_run": pearson(
                [abs(v) if v is not None else None for v in full_policy], full_entropy
            ),
            "corr_policy_loss_vs_entropy_full_run": pearson(full_policy, full_entropy),
            "corr_value_loss_vs_entropy_full_run": pearson(full_value, full_entropy),
            "decomposition_check_mean_abs_residual": mean(decomp_residuals),
            "decomposition_check_max_abs_residual": max(decomp_residuals) if decomp_residuals else None,
            "decomposition_check_n": len(decomp_residuals),
        }

    out_path = REPO / "runs" / "peak_instability" / "policy_update_magnitude" / "policy_update_stats.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
