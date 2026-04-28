#!/usr/bin/env python3
"""
Emulator phase breakdown — isolate CPU+APU cost vs PPU rendering cost.

The main hot-path bench showed `pool_step` is 89% of trainer wall-time.
This drill-down splits the 20 ms/step emulator budget into:

  - Full-render cost (CPU + APU + PPU pixel work)
  - Skip-render cost (CPU + APU + PPU state ticks only, no pixel work)
  - Difference = pure PPU pixel rendering

Why: Batch 6's per-scanline PPU rewrite targets the pixel-rendering
path. This bench quantifies how much headroom actually exists there.
If skip-render ≈ full-render, the CPU is the bottleneck and Batch 6
won't help. If full-render >> skip-render, Batch 6 is the right lever.

Usage: .venv/bin/python scripts/bench_emulator_phases.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nes_core


def bench_single_env(rom: str, frame_skip: int, steps: int, skip_render_all: bool) -> float:
    """Step a single NESEnvironment `steps` times at `frame_skip` and
    return wall-clock ms.

    `skip_render_all=True` uses a workaround: set frame_skip=1 and
    rely on the env's internal per-frame handling. We rely on the fact
    that `env.step()` only renders the FINAL frame of each `frame_skip`
    window — so at frame_skip=1 every call renders one full frame, at
    frame_skip=16 only 1 of 16 frames is full-rendered.

    To isolate pure skip-render, call with frame_skip that has many
    non-rendered frames (e.g. 256); the full render of the last frame
    becomes a minor fraction.
    """
    del skip_render_all  # kept for docstring clarity; we vary frame_skip instead
    env = nes_core.NESEnvironment(rom, frame_skip=frame_skip)
    env.reset()
    # Warm-up — 1 reset frame is already rendered; do 5 steps to
    # settle mapper state / APU.
    for _ in range(5):
        env.step(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(0)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    rom = str(_REPO / "roms" / "zelda.nes")
    if not Path(rom).exists():
        print(f"ROM not found: {rom}", file=sys.stderr)
        return 1

    steps = 400
    print(f"== emulator phase bench ==")
    print(f"  rom:      {rom}")
    print(f"  steps:    {steps} per config")
    print()

    # At frame_skip=1, EVERY step produces a rendered frame. Each step
    # = 1 frame = CPU + APU + PPU(full render).
    t_fs1 = bench_single_env(rom, frame_skip=1, steps=steps, skip_render_all=False)

    # At frame_skip=16, 1 of every 16 frames is rendered. Each step =
    # 16 frames = CPU×16 + APU×16 + PPU(skip)×15 + PPU(full)×1.
    t_fs16 = bench_single_env(rom, frame_skip=16, steps=steps, skip_render_all=False)

    # At frame_skip=64, same math, different ratio. Mostly
    # isolates CPU+APU+PPU(skip).
    t_fs64 = bench_single_env(rom, frame_skip=64, steps=steps, skip_render_all=False)

    # Compute per-frame costs from the two data points.
    # At frame_skip=16, time per step = 15 * (CPU+APU+PPU_skip) + 1 * (CPU+APU+PPU_full)
    #                                 = 16 * (CPU+APU+PPU_skip) + (PPU_full - PPU_skip)
    # At frame_skip=1,  time per step = CPU+APU+PPU_full
    # Let A = CPU+APU+PPU_skip, B = PPU_full - PPU_skip.
    # Then t_fs1 / steps = A + B, t_fs16 / steps = 16*A + B
    # → A = (t_fs16/steps - t_fs1/steps) / 15
    # → B = t_fs1/steps - A
    t1 = t_fs1 / steps
    t16 = t_fs16 / steps
    t64 = t_fs64 / steps
    a_us = ((t16 - t1) / 15) * 1000  # per-frame CPU+APU+PPU_skip (μs)
    b_us = (t1 - (t16 - t1) / 15) * 1000  # per-frame pure PPU pixel work (μs)
    # Cross-check via fs=64: should be ≈ 63*a + b
    predicted_fs64 = 63 * (a_us / 1000) + (b_us / 1000)

    print(f"{'config':20}{'ms total':>14}{'ms/step':>12}{'frames/step':>14}")
    print("-" * 60)
    print(f"{'frame_skip=1':20}{t_fs1:14.1f}{t1:12.3f}{'1':>14}")
    print(f"{'frame_skip=16':20}{t_fs16:14.1f}{t16:12.3f}{'16':>14}")
    print(f"{'frame_skip=64':20}{t_fs64:14.1f}{t64:12.3f}{'64':>14}")
    print()
    print(f"per-frame decomposition:")
    print(f"  CPU + APU + PPU(skip):    {a_us:7.1f} μs/frame")
    print(f"  PPU pixel rendering:      {b_us:7.1f} μs/frame  (pure rendering cost)")
    print(f"  Full-render frame total:  {a_us + b_us:7.1f} μs/frame")
    print()
    print(f"cross-check (predicted fs=64): {predicted_fs64:.3f} ms/step "
          f"vs measured {t64:.3f} ms/step")
    print()

    # Headline: what % of a train-step is PPU pixel work?
    # At fs=16, per step: CPU+APU+PPU_skip × 16 + (PPU_full - PPU_skip) × 1
    step_total_us = 16 * a_us + b_us
    pixel_pct = 100.0 * b_us / step_total_us
    print(f"At frame_skip=16:")
    print(f"  full step cost:          {step_total_us:7.1f} μs")
    print(f"  PPU pixel rendering:     {b_us:7.1f} μs = {pixel_pct:.1f}%  "
          "← ceiling on per-scanline PPU rewrite savings")
    print(f"  CPU+APU+PPU(state):      {16 * a_us:7.1f} μs = "
          f"{100.0 - pixel_pct:.1f}% ← everything-else floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
