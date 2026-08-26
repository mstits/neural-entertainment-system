#!/usr/bin/env python3
"""Second-pass script: for each of the 8 runs, find the iteration where
vanilla_ppo_intrinsic_mean first drops to <=10% of its iter-0 value, and
compare that iteration to the run's own peak iteration (winners/best.json
source_iter). Also samples the intrinsic/rnd_loss noise floor after iter
100 to check for late resurgence, and expresses iter-0 intrinsic as a
percentage of the iter-0 per-step extrinsic forward reward
(reward_forward / rollout_steps) for scale context.

Output written to full_table.txt in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

CKPT_ROOT = Path("/Users/stits/Documents/macos-emulation-and-training/checkpoints")
ROLLOUT_STEPS = 1024  # configs/mario_1_1_v{27,28}_seed*.yaml: reinforce.rollout_steps

RUNS = [
    ("v27", "seed0", "mario_1_1_v27_recovery_seed0", 60),
    ("v27", "seed1", "mario_1_1_v27_recovery_seed1", 50),
    ("v27", "seed2", "mario_1_1_v27_recovery_seed2", 90),
    ("v27", "seed3", "mario_1_1_v27_recovery_seed3", 60),
    ("v28", "seed0", "mario_1_1_v28_capacity_seed0", 70),
    ("v28", "seed1", "mario_1_1_v28_capacity_seed1", 60),
    ("v28", "seed2", "mario_1_1_v28_capacity_seed2", 120),
    ("v28", "seed3", "mario_1_1_v28_capacity_seed3", 90),
]  # peak iters cross-checked against each run's winners/best.json source_iter


def main() -> None:
    deltas = []
    print(f"{'run':12s} {'peak_it':>7s} {'x10pct_it':>9s} {'delta(x-peak)':>13s} "
          f"{'intr0':>8s} {'fwd0/step':>10s} {'intr0/fwd0_pct':>14s} "
          f"{'max_intr_after_100':>19s} {'max_rndloss_after_100':>21s}")
    for major, seed, name, peak in RUNS:
        rows = []
        with open(CKPT_ROOT / name / "metrics.jsonl") as f:
            for line in f:
                rows.append(json.loads(line))
        im0 = rows[0]["vanilla_ppo_intrinsic_mean"]
        fwd0_step = rows[0]["reward_forward"] / ROLLOUT_STEPS
        thresh10 = im0 * 0.10
        cross10 = next(
            r["generation"] for r in rows
            if r["vanilla_ppo_intrinsic_mean"] <= thresh10
        )
        max_intr_late = max(
            r["vanilla_ppo_intrinsic_mean"] for r in rows if r["generation"] > 100
        )
        max_rndloss_late = max(
            r["vanilla_ppo_rnd_loss"] for r in rows if r["generation"] > 100
        )
        delta = cross10 - peak
        deltas.append(delta)
        pct = 100.0 * im0 / fwd0_step
        print(f"{major}_{seed:5s} {peak:7d} {cross10:9d} {delta:13d} "
              f"{im0:8.5f} {fwd0_step:10.5f} {pct:14.2f} "
              f"{max_intr_late:19.7f} {max_rndloss_late:21.7f}")

    deltas.sort()
    print()
    print("deltas (cross10pct_iter - peak_iter), sorted:", deltas)
    print("median delta:", deltas[len(deltas) // 2])
    print("all negative (crossed before peak) in all 8:", all(d < 0 for d in deltas))


if __name__ == "__main__":
    main()
