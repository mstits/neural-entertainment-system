#!/usr/bin/env python3
"""Compare ASM-path frame pixels byte-by-byte against pure Rust.
Requires two wheel builds: this script runs within ONE — the user
toggles by rebuilding. Instead we do it inline: import once, capture
frames, then run the ROM a second time with known expected hashes
from pure Rust (pre-computed by the user toggling asm_cpu off).

Simplified: just compute a pixel hash after N frames and print it
along with cycle count. Run twice (ASM on/off), compare hashes.
"""
from __future__ import annotations
import hashlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import nes_core

rom = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "roms" / "zelda.ines1.nes")
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300

env = nes_core.NESEnvironment(rom)
env.reset()

hashes = []
for i in range(frames):
    env.step(0)
    if (i + 1) % max(1, frames // 10) == 0 or i == frames - 1:
        f = env.get_frame()
        h = hashlib.sha256(f.tobytes()).hexdigest()[:12]
        hashes.append((i + 1, h))

print(f"ROM: {rom}")
print(f"Frames: {frames}")
for i, h in hashes:
    print(f"  frame {i:5}: sha256={h}")
