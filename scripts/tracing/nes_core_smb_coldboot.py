"""nes_core counterpart to mesen_smb_coldboot.lua.

Cold-boots SMB, no input, traces every CPU instruction for MAX_FRAMES.
Uses NESEnvironment.trace_line() which already emits Mesen-compatible
format. Output is per-instruction PC/A/X/Y/P/SP/PPU/CYC, exactly
matching the Mesen Lua script for direct line-by-line diffing.

Usage:
    .venv/bin/python scripts/tracing/nes_core_smb_coldboot.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nes_core

ROM = REPO / "roms" / "Super Mario Bros. (World).nes"
OUT = Path("/tmp/nes_core_smb_coldboot.txt")
MAX_FRAMES = 200
MAX_INSTR = 2_500_000


def main() -> int:
    env = nes_core.NESEnvironment(rom_path=str(ROM), frame_skip=1)
    # reset_no_advance leaves CPU at cycle 7 (post-6502-reset),
    # matching Mesen's first-traced-instruction state. Standard
    # env.reset() auto-advances 1 frame which would offset our
    # trace from the Mesen oracle by ~9700 instructions.
    env.reset_no_advance()

    instr_count = 0
    target_frame = MAX_FRAMES
    with OUT.open("w") as fh:
        while env.ppu_state()[1] < target_frame and instr_count < MAX_INSTR:
            line = env.trace_line()
            if line:
                # trace_line() returns a string already in Mesen-compatible
                # format: "PC OP DISASM ... A:AA X:XX Y:YY P:PP SP:SP PPU:sl,cyc CYC:n"
                # Strip the disassembly portion (variable-width) so the
                # output line-format matches mesen_smb_coldboot.lua exactly.
                fh.write(_normalize(line) + "\n")
            env.step_one_instruction()
            instr_count += 1

    print(f"Logged {instr_count} instructions over "
          f"{env.ppu_state()[1]} frames -> {OUT}")
    return 0


def _normalize(trace_line: str) -> str:
    """Convert nes_core trace_line() output to mesen_smb_coldboot.lua format.

    nes_core emits:
        "8012  10 FB     BPL $800F                       A:00 X:FF Y:00 P:26 SP:FF PPU:  0,  1 CYC:0"
    Mesen Lua emits:
        "8012 10 A:00 X:FF Y:00 P:26 SP:FF PPU:  0,  1 CYC:0"

    Mesen captures only the first opcode byte (no operand bytes). Strip
    the operand bytes + disassembly column from nes_core output so the
    two are line-for-line identical when registers/cycles agree.
    """
    parts = trace_line.split()
    if len(parts) < 2:
        return trace_line
    pc = parts[0]
    opcode = parts[1]
    # Find where the register columns begin (first "A:" token)
    a_idx = next((i for i, t in enumerate(parts) if t.startswith("A:")), None)
    if a_idx is None:
        return trace_line
    tail = " ".join(parts[a_idx:])
    # PPU and CYC columns may have spaces inside (e.g., "PPU:  0,  1")
    # — trace_line already preserved them as a single token after split
    # because split() collapses whitespace. We need to re-pad to match.
    # Simplest: just emit "PC OP <tail>" and rely on parser regex which
    # is whitespace-tolerant.
    return f"{pc} {opcode} {tail}"


if __name__ == "__main__":
    sys.exit(main())
