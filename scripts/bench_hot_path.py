#!/usr/bin/env python3
"""
Per-component trainer hot-path benchmark.

Runs a representative load on an M4 Max and reports wall-clock time
spent in each layer:

  - Pool.step_all (Rust: emulator step + preprocess + RAM snapshot)
  - Reward compute (Rust via nes_core.build_reward_function)
  - Policy forward (PyTorch MPS)
  - Audio drain+push (Rust + sounddevice-compatible callback)
  - Narrator observe
  - DepthTracker observe
  - Python bookkeeping (stacker push, action table lookup)

Output is a single table. Use as the baseline for optimization work —
"which layer is eating the budget" questions answerable without
guessing.

Usage:
    .venv/bin/python scripts/bench_hot_path.py [--rom roms/zelda.nes]
                                               [--profile configs/zelda.yaml]
                                               [--workers 16]
                                               [--steps 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import yaml

import nes_core
from src.emulation.frame_utils import FrameStacker
from src.models.policy_network import PolicyNetwork, get_best_device
from src.training.depth_tracker import DepthTracker
from src.training.narrator import Narrator
from src.utils.reward_functions import build_reward_function


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", default=str(_REPO / "roms" / "zelda.nes"))
    p.add_argument("--profile", default=str(_REPO / "configs" / "zelda.yaml"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument(
        "--headless",
        action="store_true",
        default=bool(__import__("os").environ.get("NES_BENCH_HEADLESS")),
        help="Call pool.set_headless(true) — skips the 184 KB RGB "
             "unpack per step. Use in PGO profile-gen runs so the "
             "headless code path gets covered by the emitted profile.",
    )
    args = p.parse_args()

    rom = Path(args.rom)
    if not rom.exists():
        print(f"ROM not found: {rom}", file=sys.stderr)
        return 1

    profile_data = yaml.safe_load(Path(args.profile).read_text())
    action_space = profile_data["action_space"]
    num_actions = len(action_space)

    device = get_best_device()
    print(f"== hot-path bench ==")
    print(f"  rom:       {rom}")
    print(f"  profile:   {profile_data.get('name', '?')}")
    print(f"  workers:   {args.workers}")
    print(f"  steps:     {args.steps}")
    print(f"  frame_skip:{args.frame_skip}")
    print(f"  device:    {device}")
    print()

    # Initialise the pieces the trainer assembles.
    pool = nes_core.Pool(
        rom_path=str(rom),
        num_workers=args.workers,
        frame_skip=args.frame_skip,
    )
    if args.headless and hasattr(pool, "set_headless"):
        pool.set_headless(True)
        print(f"  headless:  True (skipping RGB unpack)")
    net = PolicyNetwork(num_actions=num_actions).to(device).eval()
    rewards = [build_reward_function(profile_data) for _ in range(args.workers)]
    for r in rewards:
        r.reset()
    stackers = [FrameStacker(stack_size=4) for _ in range(args.workers)]
    narrator = Narrator(rng_seed=0)
    depth = DepthTracker(profile_data.get("name", "generic"))

    # Warm-up: reset all + initial stacker fill. This prevents the first
    # step from skewing the numbers.
    t0 = _now_ms()
    init = pool.reset_all()
    init_dt = _now_ms() - t0
    for i, (frame, pp, _ram, _done) in enumerate(init):
        stackers[i].reset(frame, pp)

    # Timing buckets.
    buckets: dict[str, float] = {
        "pool_step": 0.0,
        "stacker_push": 0.0,
        "policy_forward": 0.0,
        "reward_compute": 0.0,
        "narrator": 0.0,
        "depth_tracker": 0.0,
        "audio_drain": 0.0,
    }

    # Fixed actions (cycle through action space) — we're benching layers,
    # not policy exploration. Simpler + more reproducible.
    action_cycle = np.arange(num_actions, dtype=np.int64)

    prev_breakdown = [dict(r.breakdown) for r in rewards]

    for step_idx in range(args.steps):
        # --- Pool.step_all -----------------------------------------
        action_idx = action_cycle[step_idx % num_actions]
        bitmask = 0
        for b in action_space[int(action_idx)]:
            # Quick map — we don't care about exact buttons here.
            bitmask = int(action_idx) & 0xFF
        actions = [bitmask] * args.workers
        t = _now_ms()
        results = pool.step_all(actions)
        buckets["pool_step"] += _now_ms() - t

        # --- Stacker push ------------------------------------------
        t = _now_ms()
        stacked = np.zeros((args.workers, 4, 84, 84), dtype=np.uint8)
        for i, (frame, pp, _ram, _done) in enumerate(results):
            stacked[i] = stackers[i].push(frame, pp)
        buckets["stacker_push"] += _now_ms() - t

        # --- Policy forward ---------------------------------------
        t = _now_ms()
        with torch.no_grad():
            obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
            logits = net(obs_t)
            acts = logits.argmax(dim=1).cpu().numpy()
        if device == "mps":
            torch.mps.synchronize()
        buckets["policy_forward"] += _now_ms() - t
        del acts  # not used further

        # --- Reward compute (all workers) --------------------------
        t = _now_ms()
        for i, (_frame, _pp, ram, _done) in enumerate(results):
            rewards[i].compute(bytes(ram), action=bitmask)
        buckets["reward_compute"] += _now_ms() - t

        # --- Narrator observe -------------------------------------
        t = _now_ms()
        for i in range(args.workers):
            new_bd = dict(rewards[i].breakdown)
            narrator.observe(i, f"w{i}", prev_breakdown[i], new_bd)
            prev_breakdown[i] = new_bd
        buckets["narrator"] += _now_ms() - t

        # --- DepthTracker observe ---------------------------------
        t = _now_ms()
        for i, (_frame, _pp, ram, _done) in enumerate(results):
            depth.observe(bytes(ram), i, f"w{i}", 0)
        buckets["depth_tracker"] += _now_ms() - t

        # --- Audio drain ------------------------------------------
        t = _now_ms()
        for i in range(args.workers):
            _ = pool.drain_audio(i)
        buckets["audio_drain"] += _now_ms() - t

    total = sum(buckets.values())
    print(f"{'layer':22}{'ms':>12}{'% of total':>12}{'ms/step':>12}")
    print("-" * 58)
    for name, ms in sorted(buckets.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * ms / max(total, 1e-9)
        per = ms / args.steps
        print(f"{name:22}{ms:12.1f}{pct:12.1f}{per:12.3f}")
    print("-" * 58)
    print(f"{'TOTAL':22}{total:12.1f}{100.0:12.1f}{total / args.steps:12.3f}")
    print(f"{'pool.reset_all (init)':22}{init_dt:12.1f}")
    print()
    worker_steps = args.workers * args.steps
    print(f"worker-steps: {worker_steps}")
    print(f"sps (total/wall): {worker_steps / (total / 1000.0):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
