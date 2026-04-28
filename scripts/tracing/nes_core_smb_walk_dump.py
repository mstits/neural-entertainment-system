"""nes_core counterpart to mesen_smb_walk.lua.

Same input scenario (50 idle, 6 Start, 50 idle, 200 Right), per-frame
RAM dump to /tmp/nes_core_smb_walk_ram.bin. Output binary is
306 frames × 2KB = 626688 bytes, suitable for byte-by-byte diff
against the Mesen tape.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nes_core

ROM = REPO / "roms" / "Super Mario Bros. (World).nes"
OUT = Path("/tmp/nes_core_smb_walk_ram.bin")
TOTAL_FRAMES = 306


def input_for_frame(f: int) -> int:
    if 50 <= f < 56:
        return 0x08  # Start
    if 106 <= f < 306:
        return 0x80  # Right
    return 0


def main() -> int:
    env = nes_core.NESEnvironment(rom_path=str(ROM), frame_skip=1)
    # reset_no_advance leaves CPU at cycle 7 (post-reset state) and
    # does NOT pre-advance a frame. Mesen's `endFrame` callback fires
    # at the END of each rendered frame; the 1st endFrame is after 1
    # frame of emulation. We match by stepping once per dump.
    env.reset_no_advance()

    with OUT.open("wb") as fh:
        for f in range(TOTAL_FRAMES):
            env.step(input_for_frame(f))
            ram = bytes(env.get_ram(addr) for addr in range(0x800))
            fh.write(ram)

    print(f"Wrote {OUT.stat().st_size} bytes ({TOTAL_FRAMES} frames * 2048)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
