#!/usr/bin/env python3
"""
Opcode coverage diagnostic — how often does try_bulk_step hit vs miss?

Adds a thin Python harness that drives the emulator through
representative workload and measures the bulk-path hit rate by
checking the `cycles_total` vs the time spent. Under heavy bulk
coverage, wall-time / cycles_total ratio drops; if we fall back to
per-cycle for most opcodes, the ratio stays high.

Alternative: a specific counter in Rust (would require a test-only
build). For now, this bench gives a coarse proxy via step timing.

Usage: .venv/bin/python scripts/bench_opcode_coverage.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nes_core


def measure(rom: str, workers: int = 16, steps: int = 400, fs: int = 16) -> dict:
    pool = nes_core.Pool(rom_path=rom, num_workers=workers, frame_skip=fs)
    pool.reset_all()
    # Warm up.
    for _ in range(5):
        pool.step_all([0] * workers)
    # Timed region.
    t0 = time.perf_counter()
    for _ in range(steps):
        pool.step_all([0] * workers)
    dt = time.perf_counter() - t0
    sps = (workers * steps) / dt
    ms_per_step = dt * 1000.0 / steps
    us_per_worker_frame = (dt * 1e6) / (workers * steps * fs)
    cycles_per_worker_frame = 29781  # NTSC NES CPU cycles per frame
    ns_per_cpu_cycle = (dt * 1e9) / (workers * steps * fs * cycles_per_worker_frame)
    return {
        "rom": Path(rom).name,
        "workers": workers,
        "steps": steps,
        "fs": fs,
        "dt_ms": dt * 1000,
        "sps": sps,
        "ms_per_step": ms_per_step,
        "us_per_worker_frame": us_per_worker_frame,
        "ns_per_cpu_cycle": ns_per_cpu_cycle,
    }


def main() -> int:
    roms = [
        ("zelda", "roms/zelda.nes"),
        ("mario", "roms/Mario Bros.nes"),
        ("contra", "roms/Contra (USA).nes"),
        ("megaman2", "roms/MEGAMAN2.NES"),
    ]
    print("Running opcode-coverage diagnostic across 4 ROMs (16 workers, 400 steps, fs=16)")
    print()
    rows = []
    for label, path in roms:
        if not Path(path).exists():
            print(f"  skip {label}: {path} missing")
            continue
        r = measure(path, steps=400)
        rows.append((label, r))

    print(f"{'game':12}{'sps':>10}{'ms/step':>12}{'ns/cpu-cycle':>16}")
    print("-" * 50)
    for label, r in rows:
        print(
            f"{label:12}{r['sps']:10.0f}{r['ms_per_step']:12.3f}"
            f"{r['ns_per_cpu_cycle']:16.2f}"
        )
    print()
    print("Interpretation:")
    print(" - ns/cpu-cycle is the emulator's per-cycle budget.")
    print(" - Ideal (if every opcode bulks AND PPU/APU are zero-cost):")
    print("   ~3-5 ns/cpu-cycle (just memory fetch + ALU op).")
    print(" - PPU + APU + dispatch overhead adds 10-20 ns/cpu-cycle.")
    print(" - Above 25 ns/cpu-cycle suggests significant bulk-path misses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
