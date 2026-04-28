#!/usr/bin/env python3
"""Head-to-head throughput sweep: nes-py vs nes_core, 1..N workers.

Reports aggregate frames/sec and per-worker fps for each worker count.

Usage:
    .venv/bin/python scripts/bench_vs_nes_py_sweep.py
    .venv/bin/python scripts/bench_vs_nes_py_sweep.py --workers 16 --steps 300

`--rom` defaults to SMB. `--frame-skip` defaults to 4 (typical RL).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

DEFAULT_ROM = str(REPO / "roms" / "Super Mario Bros. (World).nes")


def _nes_py_worker(rom: str, n_steps: int, frame_skip: int, ready_ev, start_ev, done_ev):
    """nes-py worker process. Resets, signals ready, waits for start, runs."""
    from nes_py import NESEnv

    env = NESEnv(rom)
    env.reset()
    # 2-step warm-up so all workers are past mapper / APU init when timing starts.
    for _ in range(2):
        for _ in range(frame_skip):
            env.step(0)
    ready_ev.set()
    start_ev.wait()
    for _ in range(n_steps):
        for _ in range(frame_skip):
            env.step(0)
    env.close()
    done_ev.set()


def bench_nes_py_parallel(rom: str, num_workers: int, n_steps: int, frame_skip: int) -> float:
    """Aggregate frames/sec for `num_workers` nes-py processes."""
    ctx = mp.get_context("spawn")
    ready = [ctx.Event() for _ in range(num_workers)]
    start = ctx.Event()
    done = [ctx.Event() for _ in range(num_workers)]
    procs = [
        ctx.Process(
            target=_nes_py_worker,
            args=(rom, n_steps, frame_skip, ready[i], start, done[i]),
        )
        for i in range(num_workers)
    ]
    for p in procs:
        p.start()
    for ev in ready:
        ev.wait()
    t0 = time.perf_counter()
    start.set()
    for ev in done:
        ev.wait()
    el = time.perf_counter() - t0
    for p in procs:
        p.join()
    total_frames = num_workers * n_steps * frame_skip
    return total_frames / el


def bench_nes_core_pool(rom: str, num_workers: int, n_steps: int, frame_skip: int) -> float:
    """Aggregate frames/sec for an N-worker nes_core Pool."""
    import nes_core

    pool = nes_core.Pool(rom_path=rom, num_workers=num_workers, frame_skip=frame_skip)
    pool.reset_all()
    # 2-step warm-up so rayon thread pool / mapper / APU are settled.
    for _ in range(2):
        pool.step_all([0] * num_workers)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        pool.step_all([0] * num_workers)
    el = time.perf_counter() - t0
    total_frames = num_workers * n_steps * frame_skip
    return total_frames / el


def fmt_fps(fps: float) -> str:
    return f"{fps:>9,.0f}".replace(",", " ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default=DEFAULT_ROM)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--frame-skip", type=int, default=4)
    args = p.parse_args()

    if not Path(args.rom).exists():
        print(f"ROM not found: {args.rom}", file=sys.stderr)
        return 1

    print(f"ROM:        {Path(args.rom).name}")
    print(f"steps/cfg:  {args.steps}")
    print(f"frame_skip: {args.frame_skip}")
    print(f"CPU count:  {os.cpu_count()}")
    print()

    header = (
        f"{'N':>3}  "
        f"{'nes-py fps':>11}  {'core fps':>11}  "
        f"{'×nes-py':>8}  "
        f"{'py per-w':>9}  {'core per-w':>11}  "
        f"{'×realtime':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n in range(1, args.max_workers + 1):
        py = bench_nes_py_parallel(args.rom, n, args.steps, args.frame_skip)
        core = bench_nes_core_pool(args.rom, n, args.steps, args.frame_skip)
        ratio = core / py
        per_py = py / n
        per_core = core / n
        rtmult = core / 60.0
        results.append((n, py, core, ratio, per_py, per_core, rtmult))
        print(
            f"{n:>3}  "
            f"{fmt_fps(py)}  {fmt_fps(core)}  "
            f"{ratio:>7.2f}×  "
            f"{fmt_fps(per_py)}  {fmt_fps(per_core)}  "
            f"{rtmult:>8.1f}×"
        )

    # Compact summary
    print()
    print("Summary (× nes-py):", "  ".join(f"N={n}:{r:.2f}×" for n, _, _, r, *_ in results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
