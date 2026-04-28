#!/usr/bin/env python3
"""
Bench the trainer's _evaluate_batch with async_pipeline on vs off.

Runs the full training path (forward + step_all + reward compute +
trajectory append + narrator observe + ...) for a fixed number of
steps and compares wall time. Async enables 1-step observation lag
so the policy forward overlaps with pool.step_all.

Usage:
    .venv/bin/python scripts/bench_trainer_async.py [--steps 500] [--workers 16]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from src.training.trainer import Trainer, load_game_profile


def bench(async_pipeline: bool, steps: int, workers: int) -> float:
    profile = load_game_profile(str(_REPO / "configs" / "zelda.yaml"))
    profile["async_pipeline"] = async_pipeline
    # Shorten to keep bench reasonable
    profile["max_episode_steps"] = steps
    trainer = Trainer(
        rom_path=str(_REPO / "roms" / "zelda.nes"),
        game_profile=profile,
        num_instances=workers,
        population_size=workers,
        checkpoint_dir="/tmp/bench_trainer_async",
        env_spec="nes_core:NESEnvironment",
    )
    trainer._running = True
    # Manually build + seed the pool; we don't want to run full training.
    from src.training.trainer import _make_pool
    trainer.pool = _make_pool(
        env_spec=trainer.env_spec,
        rom_path=trainer.rom_path,
        num_workers=trainer.num_instances,
        frame_skip=trainer.frame_skip,
        start_state_path=trainer.start_state_path,
    )
    trainer.pool.start()
    # GA init — we just need a population of networks.
    trainer.ga.initialize()
    pop = trainer.ga.population

    # Warm up (1 small batch — JIT the vmap path).
    trainer.max_episode_steps = 20
    _ = trainer._evaluate_batch(pop[:workers])

    # Actual timed run.
    trainer.max_episode_steps = steps
    t0 = time.perf_counter()
    _ = trainer._evaluate_batch(pop[:workers])
    dur = time.perf_counter() - t0
    trainer.pool.shutdown()
    return (steps * workers) / dur


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    print(f"bench_trainer_async: {args.workers} workers × {args.steps} steps × zelda")
    off_sps = bench(async_pipeline=False, steps=args.steps, workers=args.workers)
    on_sps = bench(async_pipeline=True, steps=args.steps, workers=args.workers)
    print(f"  async=False: {off_sps:7.0f} sps")
    print(f"  async=True:  {on_sps:7.0f} sps  ({100*(on_sps-off_sps)/off_sps:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
