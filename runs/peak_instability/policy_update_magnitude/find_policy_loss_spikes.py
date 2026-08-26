#!/usr/bin/env python3
"""Scan all 8 runs' metrics.jsonl for catastrophic ppo_policy_loss spikes.

Typical ppo_policy_loss magnitude across all 8 runs is O(0.05) (PPO's
clipped surrogate is supposed to keep it small and bounded). This script
flags any iteration where |ppo_policy_loss| exceeds THRESH -- three to five
orders of magnitude above baseline -- and reports its position relative to
that run's authoritative peak iter (checkpoints/<run>/winners/best.json)
and its entropy at that iteration.

Mechanism this is checking for: PPO's clipped surrogate
min(ratio*A, clip(ratio)*A) is only bounded BELOW-in-magnitude when A>0.
When A<0 and ratio grows past 1+eps (policy drifting toward an action the
advantage says was bad), ratio*A < clip(ratio)*A (more negative), so the
min() selects the UNCLIPPED, unbounded term -- policy_loss = -ratio*A can
grow without limit as ratio -> large. A near-deterministic policy (low
entropy) is exactly the regime where log-probs sit near extreme values, so
a small parameter step across a decision boundary produces a huge ratio
swing. This predicts spikes should cluster at LOW entropy and AFTER the
honest peak (once the policy has already sharpened) -- both checked below.

Usage:
    .venv/bin/python runs/peak_instability/policy_update_magnitude/find_policy_loss_spikes.py
"""
from __future__ import annotations

import json
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
THRESH = 0.5  # ~10x the typical mean_abs(ppo_policy_loss) of ~0.05


def main():
    out = {}
    for run in RUNS:
        rows = [
            json.loads(ln)
            for ln in (REPO / "checkpoints" / run / "metrics.jsonl").open()
            if ln.strip()
        ]
        rows.sort(key=lambda r: r["generation"])
        best = json.loads(
            (REPO / "checkpoints" / run / "winners" / "best.json").read_text()
        )
        peak = best["source_iter"]
        spikes = [
            {
                "iter": r["generation"],
                "vs_peak": r["generation"] - peak,
                "pre_or_post_peak": "PRE" if r["generation"] < peak else "POST",
                "ppo_policy_loss": r.get("ppo_policy_loss"),
                "ppo_entropy": r.get("ppo_entropy"),
                "ppo_value_loss": r.get("ppo_value_loss"),
            }
            for r in rows
            if abs(r.get("ppo_policy_loss") or 0.0) > THRESH
        ]
        out[run] = {
            "peak_iter": peak,
            "peak_honest_metric": best["metric_value"],
            "threshold": THRESH,
            "n_spikes": len(spikes),
            "spikes": spikes,
        }
        print(f"{run}: peak_iter={peak} peak_honest={best['metric_value']:.3f} "
              f"n_spikes(|policy_loss|>{THRESH})={len(spikes)}")
        for s in spikes:
            print(f"    iter {s['iter']:>4d} ({s['pre_or_post_peak']}-peak, "
                  f"{s['vs_peak']:+d}): policy_loss={s['ppo_policy_loss']:>10.4f}  "
                  f"entropy={s['ppo_entropy']:.4f}  value_loss={s['ppo_value_loss']:.2f}")

    out_path = REPO / "runs" / "peak_instability" / "policy_update_magnitude" / "policy_loss_spikes.json"
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
