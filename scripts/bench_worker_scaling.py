#!/usr/bin/env python3
"""
Scaling bench — measure per-worker throughput as the pool grows.

On M4 Max (up to 16 cores), rayon's default thread pool may or may
not use every core. This script runs the same per-step workload at
worker counts {4, 8, 12, 16, 20, 24, 32} and reports:

  - steps/sec (wall-clock, the aggregate that matters for training)
  - per-worker steps/sec (how much each worker contributes)
  - efficiency (actual / ideal linear scaling)

If per-worker sps stays roughly flat up to 16 and tanks past 16 →
we're already maxed on cores; more workers just add contention.

If per-worker sps drops starting at 12 → scheduler is saturating
earlier; may be worth tuning rayon's RAYON_NUM_THREADS.

Usage:
    .venv/bin/python scripts/bench_worker_scaling.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nes_core


def bench_at_scale(rom: str, workers: int, frame_skip: int, steps: int) -> float:
    pool = nes_core.Pool(rom_path=rom, num_workers=workers, frame_skip=frame_skip)
    pool.reset_all()
    # 2-step warm-up to settle mapper / APU state + rayon thread-pool.
    for _ in range(2):
        pool.step_all([0] * workers)
    t0 = time.perf_counter()
    for _ in range(steps):
        pool.step_all([0] * workers)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    rom = str(_REPO / "roms" / "zelda.nes")
    if not Path(rom).exists():
        print(f"ROM not found: {rom}", file=sys.stderr)
        return 1

    steps = 100
    frame_skip = 16
    counts = [1, 4, 8, 12, 16, 20, 24, 32]

    print(f"== worker-scaling bench ==")
    print(f"  rom:        {rom}")
    print(f"  steps:      {steps} per config")
    print(f"  frame_skip: {frame_skip}")
    print(f"  import os; os.cpu_count() = {__import__('os').cpu_count()}")
    print()

    print(f"{'workers':>8}{'ms':>10}{'total sps':>14}{'per-worker sps':>18}{'efficiency':>14}")
    print("-" * 64)
    baseline_per_worker = None
    for n in counts:
        ms = bench_at_scale(rom, n, frame_skip, steps)
        total_sps = (n * steps) / (ms / 1000.0)
        per_worker = total_sps / n
        if baseline_per_worker is None and n == 1:
            baseline_per_worker = per_worker
        eff = (per_worker / baseline_per_worker * 100.0) if baseline_per_worker else 100.0
        print(f"{n:8d}{ms:10.0f}{total_sps:14.0f}{per_worker:18.1f}{eff:13.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
