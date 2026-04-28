#!/usr/bin/env python3
"""Head-to-head throughput: nes-py (C++ reference) vs nes_core (Rust/ASM).

Reports frames/sec and multiples-of-realtime for:
  1. Single env, frame_skip=1 (raw per-frame stepping)
  2. Single env, frame_skip=4 (typical RL-training cadence)
  3. Parallel: 12 nes_core workers (Pool) vs 12 nes-py subprocesses

The Pool number is the one that actually drives training throughput.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ROM = str(REPO / "roms" / "CONTRA.NES")
FRAMES_SINGLE = 1200
STEPS_PARALLEL = 200


def bench_nes_py_single(frame_skip: int) -> float:
    from nes_py import NESEnv

    env = NESEnv(ROM)
    env.reset()
    t0 = time.perf_counter()
    steps = FRAMES_SINGLE // frame_skip
    for _ in range(steps):
        for _ in range(frame_skip):
            env.step(0)
    el = time.perf_counter() - t0
    env.close()
    return FRAMES_SINGLE / el


def bench_nes_core_single(frame_skip: int) -> float:
    import nes_core

    env = nes_core.NESEnvironment(ROM, frame_skip=frame_skip)
    env.reset()
    t0 = time.perf_counter()
    steps = FRAMES_SINGLE // frame_skip
    for _ in range(steps):
        env.step(0)
    el = time.perf_counter() - t0
    return FRAMES_SINGLE / el


def _nes_py_worker(rom: str, n_steps: int, frame_skip: int, done_ev: mp.Event):
    """nes-py worker for multiprocess parallel test."""
    from nes_py import NESEnv

    env = NESEnv(rom)
    env.reset()
    for _ in range(n_steps):
        for _ in range(frame_skip):
            env.step(0)
    env.close()
    done_ev.set()


def bench_nes_py_parallel(num_workers: int, frame_skip: int) -> float:
    ctx = mp.get_context("spawn")
    evs = [ctx.Event() for _ in range(num_workers)]
    procs = [
        ctx.Process(target=_nes_py_worker, args=(ROM, STEPS_PARALLEL, frame_skip, evs[i]))
        for i in range(num_workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    el = time.perf_counter() - t0
    total_frames = num_workers * STEPS_PARALLEL * frame_skip
    return total_frames / el


def bench_nes_core_pool(num_workers: int, frame_skip: int) -> float:
    import nes_core

    p = nes_core.Pool(rom_path=ROM, num_workers=num_workers, frame_skip=frame_skip)
    p.reset_all()
    # Warm-up
    for _ in range(2):
        p.step_all([0] * num_workers)
    t0 = time.perf_counter()
    for _ in range(STEPS_PARALLEL):
        p.step_all([0] * num_workers)
    el = time.perf_counter() - t0
    total_frames = num_workers * STEPS_PARALLEL * frame_skip
    return total_frames / el


def fmt(fps: float) -> str:
    return f"{fps:>10.0f} fps  ({fps/60:>6.1f}× realtime)"


def main() -> int:
    print(f"ROM:          {ROM}")
    print(f"CPU count:    {os.cpu_count()}")
    print()

    print("─── Single env, frame_skip=1 (raw per-frame stepping) ───")
    py_fs1 = bench_nes_py_single(1)
    print(f"  nes-py:    {fmt(py_fs1)}")
    core_fs1 = bench_nes_core_single(1)
    print(f"  nes_core:  {fmt(core_fs1)}    [{core_fs1/py_fs1:.2f}× nes-py]")
    print()

    print("─── Single env, frame_skip=4 (typical RL cadence) ───")
    py_fs4 = bench_nes_py_single(4)
    print(f"  nes-py:    {fmt(py_fs4)}")
    core_fs4 = bench_nes_core_single(4)
    print(f"  nes_core:  {fmt(core_fs4)}    [{core_fs4/py_fs4:.2f}× nes-py]")
    print()

    print("─── 12 parallel envs, frame_skip=4 (training workload) ───")
    py_par = bench_nes_py_parallel(12, 4)
    print(f"  nes-py ×12 (multiprocessing): {fmt(py_par)}")
    core_par = bench_nes_core_pool(12, 4)
    print(f"  nes_core Pool ×12:            {fmt(core_par)}    [{core_par/py_par:.2f}× nes-py]")
    print()

    print("─── 12 parallel envs, frame_skip=16 (aggressive RL) ───")
    py_par16 = bench_nes_py_parallel(12, 16)
    print(f"  nes-py ×12 (multiprocessing): {fmt(py_par16)}")
    core_par16 = bench_nes_core_pool(12, 16)
    print(f"  nes_core Pool ×12:            {fmt(core_par16)}    [{core_par16/py_par16:.2f}× nes-py]")
    print()

    print("─── Summary (total frames/sec delta) ───")
    print(f"  fs=1  single-env: {core_fs1/py_fs1:5.2f}×")
    print(f"  fs=4  single-env: {core_fs4/py_fs4:5.2f}×")
    print(f"  fs=4  12-parallel: {core_par/py_par:5.2f}×")
    print(f"  fs=16 12-parallel: {core_par16/py_par16:5.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
