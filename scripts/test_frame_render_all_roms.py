#!/usr/bin/env python3
"""
Regression test: every ROM in `roms/` renders non-zero frames after
30 pool steps. Protects the bulk-step path from the
NMI-line-propagation-missing class of bug that caused black-screens
in live Zelda training (and would have caught it before the GUI).

Usage:
    .venv/bin/python scripts/test_frame_render_all_roms.py

Exit 0 if all ROMs render; exit 1 if any produce all-zero frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

import nes_core


def test_rom(rom_path: str, steps: int = 30, fs: int = 16) -> tuple[bool, float]:
    """Run `steps` × fs-skipped steps, return (rendered, nonzero_pct).
    `rendered` is True iff the last frame has >5% non-zero pixels."""
    pool = nes_core.Pool(rom_path=rom_path, num_workers=1, frame_skip=fs)
    pool.reset_all()
    results = None
    for _ in range(steps):
        results = pool.step_all([0])
    if results is None:
        return False, 0.0
    frame, _pp, _ram, _done = results[0]
    frame_np = np.asarray(frame)
    nonzero = (frame_np != 0).sum()
    pct = 100.0 * nonzero / frame_np.size
    return pct > 5.0, pct


def main() -> int:
    roms = sorted((_REPO / "roms").glob("*.nes"))
    # Filter out obviously-broken variants.
    roms = [p for p in roms if ".ines1_start" not in p.name]
    if not roms:
        print("No ROMs found in roms/")
        return 0

    print(f"Testing {len(roms)} ROMs (30 steps × fs=16 each)")
    print()
    print(f"{'ROM':45}{'frames render?':>20}{'nonzero %':>12}")
    print("-" * 77)
    failed: list[str] = []
    for rom in roms:
        try:
            ok, pct = test_rom(str(rom))
        except Exception as exc:
            print(f"{rom.name:45}{'ERROR':>20}{str(exc)[:12]:>12}")
            failed.append(rom.name)
            continue
        status = "OK" if ok else "ALL-BLACK"
        print(f"{rom.name:45}{status:>20}{pct:11.1f}%")
        if not ok:
            failed.append(rom.name)

    print()
    if failed:
        print(f"FAILED: {len(failed)} ROMs: {failed}")
        return 1
    print(f"ALL {len(roms)} ROMs render non-zero frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
