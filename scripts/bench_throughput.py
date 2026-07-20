#!/usr/bin/env python3
"""
Pure-emulation throughput benchmark — how many NES frames/second the
core produces, headless, with no policy in the loop.

This exists to put a real number behind the "~30k fps" claim. It steps
`nes_core.Pool` with no-op actions (no policy forward pass), in
headless mode (no RGB framebuffer), and reports for each worker count:

  pool_calls_per_s  step_all() invocations per second (one call steps
                    every worker once)
  env_steps_per_s   pool_calls_per_s x workers — the training sample
                    rate ("steps/s")
  nes_frames_per_s  env_steps_per_s x frame_skip — raw NES video frames
                    emulated per second ("fps")
  realtime_x        nes_frames_per_s / 60.0988 — multiple of wall-clock
                    NES speed

Two observation modes are swept so the number is honestly bounded:
  raw       headless + skip_preprocess — pure emulator advance, the
            true fps ceiling
  preprocess headless with the 84x84 grayscale kernel on — the actual
            training data-collection path (still no policy)

Worker counts default to 1/8/16/60. 60 on a 16-core machine is
deliberately oversubscribed to show the past-saturation behaviour.

Usage:
    .venv/bin/python scripts/bench_throughput.py
    .venv/bin/python scripts/bench_throughput.py --workers 1 8 16 60 --frame-skip 4
    .venv/bin/python scripts/bench_throughput.py --out runs/emulator_bench_2026-07-20.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nes_core

DEFAULT_ROM = _REPO / "roms" / "Super Mario Bros. (World).nes"
DEFAULT_STATE = _REPO / "roms" / "Super Mario Bros. (World)_start.state.bin"

# NTSC NES frame rate (CPU 1.7898 MHz / 29780.5 cycles per frame).
NES_HZ = 60.0988


def _bench_config(rom: str, state: str, workers: int, frame_skip: int,
                  skip_preprocess: bool, budget_s: float,
                  warmup: int, min_calls: int) -> dict:
    """Run one (workers, frame_skip, mode) config for ~budget_s and
    return a throughput dict."""
    pool = nes_core.Pool(rom_path=rom, num_workers=workers,
                         frame_skip=frame_skip, start_state_path=state)
    pool.set_headless(True)
    if skip_preprocess:
        pool.set_skip_preprocess(True)
    pool.reset_all()
    noop = np.zeros(workers, dtype=np.uint8)

    for _ in range(warmup):
        pool.step_all(noop)

    calls = 0
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    # Batch the deadline check so perf_counter overhead is amortised
    # across a handful of steps rather than paid every call.
    while True:
        for _ in range(8):
            pool.step_all(noop)
        calls += 8
        if calls >= min_calls and time.perf_counter() >= deadline:
            break
    dt = time.perf_counter() - t0

    pool.shutdown()

    calls_per_s = calls / dt
    env_steps_per_s = calls_per_s * workers
    nes_frames_per_s = env_steps_per_s * frame_skip
    return {
        "workers": workers,
        "frame_skip": frame_skip,
        "mode": "raw" if skip_preprocess else "preprocess",
        "wall_s": dt,
        "pool_calls": calls,
        "pool_calls_per_s": calls_per_s,
        "env_steps_per_s": env_steps_per_s,
        "nes_frames_per_s": nes_frames_per_s,
        "realtime_x": nes_frames_per_s / NES_HZ,
    }


def _print_row(r: dict) -> None:
    print(f"{r['workers']:>7d}{r['frame_skip']:>4d}  {r['mode']:<10s}"
          f"{r['pool_calls_per_s']:>12.0f}"
          f"{r['env_steps_per_s']:>14.0f}"
          f"{r['nes_frames_per_s']:>16.0f}"
          f"{r['realtime_x']:>11.0f}x")


def run(rom: str, state: str, worker_counts, frame_skip: int,
        budget_s: float, warmup: int, min_calls: int, modes) -> dict:
    print("== pure-emulation throughput bench ==")
    print(f"  rom:        {rom}")
    print(f"  frame_skip: {frame_skip}")
    print(f"  budget:     {budget_s:.1f}s per config")
    print(f"  cpu_count:  {os.cpu_count()}")
    try:
        l1, l5, l15 = os.getloadavg()
        warn = "  <-- CONTENDED, numbers are a lower bound" if l1 > (os.cpu_count() or 1) else ""
        print(f"  loadavg:    {l1:.1f} {l5:.1f} {l15:.1f}{warn}")
    except (OSError, AttributeError):
        pass
    print(f"  NES rate:   {NES_HZ} fps (realtime)")
    print()
    print(f"{'workers':>7}{'fs':>4}  {'mode':<10}"
          f"{'calls/s':>12}{'env-steps/s':>14}{'nes-frames/s':>16}{'realtime':>12}")
    print("-" * 76)

    rows = []
    for skip_pp in modes:
        for n in worker_counts:
            r = _bench_config(rom, state, n, frame_skip, skip_pp,
                              budget_s, warmup, min_calls)
            rows.append(r)
            _print_row(r)
    print()

    # Headline: peak raw NES fps across the sweep.
    raw_rows = [r for r in rows if r["mode"] == "raw"] or rows
    peak = max(raw_rows, key=lambda r: r["nes_frames_per_s"])
    print(f"  peak raw NES fps: {peak['nes_frames_per_s']:.0f} "
          f"at {peak['workers']} workers "
          f"({peak['realtime_x']:.0f}x realtime)")

    return {
        "rom": rom,
        "start_state": state,
        "frame_skip": frame_skip,
        "budget_s": budget_s,
        "nes_hz": NES_HZ,
        "rows": rows,
        "peak_raw_nes_frames_per_s": peak["nes_frames_per_s"],
        "peak_raw_workers": peak["workers"],
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _merge_out(out_path: Path, section_key: str, payload: dict) -> None:
    doc = {}
    if out_path.exists():
        try:
            doc = json.loads(out_path.read_text())
        except Exception:
            doc = {}
    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):
        load1 = load5 = load15 = None
    doc.setdefault("machine", {
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
    })
    payload["loadavg_1_5_15"] = [load1, load5, load15]
    payload["contended"] = bool(load1 is not None
                                and load1 > (os.cpu_count() or 1))
    doc["git_commit"] = _git_commit()
    doc[section_key] = payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"  wrote {section_key} -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 60])
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--budget", type=float, default=3.0,
                    help="seconds to run each config")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--min-calls", type=int, default=200)
    ap.add_argument("--only-raw", action="store_true",
                    help="skip the preprocess-on pass")
    ap.add_argument("--out", default=None,
                    help="merge results into this JSON file")
    args = ap.parse_args()

    for pth in (args.rom, args.state):
        if not Path(pth).exists():
            print(f"missing: {pth}", file=sys.stderr)
            return 1

    modes = [True] if args.only_raw else [True, False]  # raw first
    result = run(args.rom, args.state, args.workers, args.frame_skip,
                 args.budget, args.warmup, args.min_calls, modes)

    if args.out:
        _merge_out(Path(args.out), "throughput", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
