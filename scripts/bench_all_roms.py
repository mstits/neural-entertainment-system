#!/usr/bin/env python3
"""Sweep every ROM in roms/, measure FPS + correctness signals."""
from __future__ import annotations
import glob
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import nes_core

FRAMES = 300
roms = sorted(glob.glob(str(REPO / "roms" / "*.nes")) + glob.glob(str(REPO / "roms" / "*.NES")))

print(f"{'ROM':<42} {'Map':>3} {'FPS':>7} {'×60Hz':>7} {'Pixels':>6}")
print("─" * 72)
for rom in roms:
    try:
        with open(rom, "rb") as f:
            hdr = f.read(16)
        mapper = (hdr[6] >> 4) | (hdr[7] & 0xF0)
        env = nes_core.NESEnvironment(rom)
        env.reset()
        start = time.perf_counter()
        for _ in range(FRAMES):
            env.step(0)
        elapsed = time.perf_counter() - start
        frame = env.get_frame()
        nonzero = int((frame != 0).any(axis=2).sum())
        fps = FRAMES / elapsed
        name = Path(rom).name[:42]
        print(f"{name:<42} {mapper:>3} {fps:>7.0f} {fps/60:>7.2f} {nonzero:>6}")
    except Exception as e:
        print(f"{Path(rom).name:<42} ERR  {e}")
