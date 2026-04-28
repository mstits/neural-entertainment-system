"""POC multiprocess Pool for nes_core, A/B benched vs the Rust Pool.

Architecture:
  * N Python subprocess workers, each holds its own nes_core.NESEnvironment
  * Master sends actions via mp.Pipe (one per worker)
  * Workers ack done after each step
  * Frames are written to a shared memory block, viewed by the master
    as a single (N, 240, 256, 3) uint8 ndarray

The point: if Rayon thread-pool overhead at 12+ workers is the actual
bottleneck (vs nes-py's near-linear multiprocess scaling), this should
show measurable per-step wins at N>=8.

Usage:
    .venv/bin/python scripts/poc_multiprocess_pool.py --workers 12 --steps 200
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

DEFAULT_ROM = str(REPO / "roms" / "Super Mario Bros. (World).nes")
FRAME_H = 240
FRAME_W = 256
FRAME_BYTES = FRAME_H * FRAME_W * 3


def _worker_main(rom_path: str, frame_skip: int, conn, shm_name: str, frame_offset: int):
    """Subprocess body. Owns one NESEnvironment, services actions in a loop."""
    import nes_core
    from multiprocessing import shared_memory

    env = nes_core.NESEnvironment(rom_path=rom_path, frame_skip=frame_skip)
    env.reset()

    shm = shared_memory.SharedMemory(name=shm_name)
    frame_view = np.ndarray(
        (FRAME_H, FRAME_W, 3),
        dtype=np.uint8,
        buffer=shm.buf,
        offset=frame_offset,
    )

    try:
        while True:
            cmd = conn.recv()
            if cmd is None:
                break
            if cmd == "reset":
                frame, _ = env.reset(), False
                # nes-core reset returns frame; copy
                if hasattr(frame, "shape") and frame.shape == (FRAME_H, FRAME_W, 3):
                    frame_view[:] = frame
                conn.send(False)
            else:
                action = int(cmd)
                frame, done = env.step(action)
                # Copy frame bytes into shared memory
                frame_view[:] = frame
                conn.send(done)
    finally:
        shm.close()


class MultiprocessPool:
    """Mirror of nes_core.Pool API using mp processes + shared memory."""

    def __init__(self, rom_path: str, num_workers: int, frame_skip: int = 4):
        self.num_workers = num_workers
        self.frame_skip = frame_skip
        self.rom_path = rom_path

        from multiprocessing import shared_memory

        total_bytes = num_workers * FRAME_BYTES
        self.shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.frames = np.ndarray(
            (num_workers, FRAME_H, FRAME_W, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )

        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.workers = []
        for i in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            self.parent_conns.append(parent_conn)
            offset = i * FRAME_BYTES
            p = ctx.Process(
                target=_worker_main,
                args=(rom_path, frame_skip, child_conn, self.shm.name, offset),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            child_conn.close()

    def step_all(self, actions):
        # Phase 1: send all actions
        for conn, a in zip(self.parent_conns, actions):
            conn.send(int(a))
        # Phase 2: collect all dones
        dones = [conn.recv() for conn in self.parent_conns]
        return self.frames, dones

    def reset_all(self):
        for conn in self.parent_conns:
            conn.send("reset")
        for conn in self.parent_conns:
            conn.recv()

    def shutdown(self):
        for conn in self.parent_conns:
            try:
                conn.send(None)
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=2)
            if w.is_alive():
                w.terminate()
        self.shm.close()
        self.shm.unlink()


def bench_multiprocess(rom: str, n: int, steps: int, frame_skip: int) -> float:
    pool = MultiprocessPool(rom, n, frame_skip)
    pool.reset_all()
    actions = [0] * n
    # Warm up
    for _ in range(5):
        pool.step_all(actions)
    t0 = time.perf_counter()
    for _ in range(steps):
        pool.step_all(actions)
    el = time.perf_counter() - t0
    pool.shutdown()
    return (n * steps * frame_skip) / el


def bench_rust_pool(rom: str, n: int, steps: int, frame_skip: int) -> float:
    import nes_core
    pool = nes_core.Pool(rom_path=rom, num_workers=n, frame_skip=frame_skip)
    pool.reset_all()
    for _ in range(5):
        pool.step_all([0] * n)
    t0 = time.perf_counter()
    for _ in range(steps):
        pool.step_all([0] * n)
    el = time.perf_counter() - t0
    return (n * steps * frame_skip) / el


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

    print(f"{'N':>3} {'rust fps':>10} {'mp fps':>10} {'mp ratio':>9} {'rust per-w':>11} {'mp per-w':>9}")
    print("-" * 60)
    for n in [1, 2, 4, 6, 8, 10, 12]:
        try:
            rust_fps = bench_rust_pool(args.rom, n, args.steps, args.frame_skip)
        except Exception as e:
            rust_fps = 0
            print(f"  rust pool failed at N={n}: {e}", file=sys.stderr)
        try:
            mp_fps = bench_multiprocess(args.rom, n, args.steps, args.frame_skip)
        except Exception as e:
            mp_fps = 0
            print(f"  mp pool failed at N={n}: {e}", file=sys.stderr)
        ratio = mp_fps / rust_fps if rust_fps > 0 else 0
        print(
            f"{n:>3d} {rust_fps:>10.0f} {mp_fps:>10.0f} {ratio:>8.2f}× "
            f"{rust_fps / max(n, 1):>11.0f} {mp_fps / max(n, 1):>9.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
