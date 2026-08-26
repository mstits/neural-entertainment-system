#!/usr/bin/env python3
"""Extract RND intrinsic-reward / predictor-loss trajectories for the
8 from-scratch runs (v27 x4 seeds, v28 x4 seeds) and relate them to the
peak iteration recorded in each run's winners/best.json.

Reads only checked-in artifacts (metrics.jsonl, winners/best.json,
configs/*.yaml). No new training compute.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/Users/stits/Documents/macos-emulation-and-training")
CKPT_ROOT = ROOT / "checkpoints"

RUNS = [
    ("v27", "seed0", "mario_1_1_v27_recovery_seed0"),
    ("v27", "seed1", "mario_1_1_v27_recovery_seed1"),
    ("v27", "seed2", "mario_1_1_v27_recovery_seed2"),
    ("v27", "seed3", "mario_1_1_v27_recovery_seed3"),
    ("v28", "seed0", "mario_1_1_v28_capacity_seed0"),
    ("v28", "seed1", "mario_1_1_v28_capacity_seed1"),
    ("v28", "seed2", "mario_1_1_v28_capacity_seed2"),
    ("v28", "seed3", "mario_1_1_v28_capacity_seed3"),
]

FIELDS = [
    "generation",
    "vanilla_ppo_rnd_loss",
    "vanilla_ppo_intrinsic_mean",
    "vanilla_ppo_count_bonus_mean",
    "reward_forward",
    "reward_completion",
    "ppo_entropy",
    "vanilla_ppo_max_x",
]


def load_metrics(run_dir: Path) -> list[dict]:
    rows = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({k: d.get(k) for k in FIELDS})
    return rows


def load_peak(run_dir: Path) -> dict:
    with open(run_dir / "winners" / "best.json") as f:
        return json.load(f)


def nearest_row(rows: list[dict], gen: int) -> dict:
    return min(rows, key=lambda r: abs((r["generation"] or -999999) - gen))


def main() -> None:
    out = {}
    print(f"{'run':22s} {'peak_it':>7s} {'rnd_loss@10':>12s} {'rnd_loss@peak':>13s} "
          f"{'rnd_loss@240':>12s} {'intr@10':>9s} {'intr@peak':>10s} {'intr@240':>9s} "
          f"{'intr_peak/intr10':>16s} {'intr240/intr10':>15s}")
    for major, seed, ckpt_name in RUNS:
        run_dir = CKPT_ROOT / ckpt_name
        rows = load_metrics(run_dir)
        peak = load_peak(run_dir)
        peak_it = peak["source_iter"]

        r10 = nearest_row(rows, 10)
        rpeak = nearest_row(rows, peak_it)
        r240 = nearest_row(rows, 240)

        rl10 = r10["vanilla_ppo_rnd_loss"]
        rlpeak = rpeak["vanilla_ppo_rnd_loss"]
        rl240 = r240["vanilla_ppo_rnd_loss"]

        i10 = r10["vanilla_ppo_intrinsic_mean"]
        ipeak = rpeak["vanilla_ppo_intrinsic_mean"]
        i240 = r240["vanilla_ppo_intrinsic_mean"]

        ratio_peak = (ipeak / i10) if i10 else float("nan")
        ratio_240 = (i240 / i10) if i10 else float("nan")

        key = f"{major}_{seed}"
        out[key] = {
            "peak_iter": peak_it,
            "peak_metric_value": peak["metric_value"],
            "rows": rows,
        }

        print(f"{key:22s} {peak_it:7d} {rl10:12.6f} {rlpeak:13.6f} {rl240:12.6f} "
              f"{i10:9.5f} {ipeak:10.5f} {i240:9.5f} {ratio_peak:16.4f} {ratio_240:15.4f}")

    with open(Path(__file__).parent / "rnd_raw_rows.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
