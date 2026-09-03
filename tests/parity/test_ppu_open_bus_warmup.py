"""Regression test: PPU open-bus latch (`ppu_gen_latch`) is filled by
all $2000-$3FFF writes, including writes that occur during the
~29658-cycle post-reset PPU warmup period.

Reproducer: SMB cold-boot sequence at $8000-$800D writes 0x10 to
PPUCTRL ($2000), then reads PPUSTATUS ($2002). Real hardware (and
Mesen) returns 0x10 because the open-bus low 5 bits = 0x10 & 0x1F.
A pre-fix nes_core early-returned during the warmup before updating
ppu_gen_latch, returning 0x00 instead — confirmed by Mesen-vs-nes_core
side-by-side trace at instruction index 7 of cold-boot SMB.

Wiki ref: https://www.nesdev.org/wiki/PPU_registers#PPUSTATUS
Mesen ref: third_party/Mesen2/Core/NES/NesPpu.cpp `WriteRam` calls
`SetOpenBus(0xFF, value)` unconditionally before per-register
handling.
"""
from __future__ import annotations

from pathlib import Path
import pytest

import nes_core

from tests.skip_gates import requires

REPO = Path(__file__).resolve().parents[2]
SMB_ROM_REL = "roms/Super Mario Bros. (World).nes"
SMB_ROM = REPO / SMB_ROM_REL


# roms/* is gitignored (.gitignore:71), so a clean clone has no SMB dump
# to cold-boot and the miss surfaces as a bare nes_core RuntimeError.
@pytest.mark.parity
@requires(SMB_ROM_REL)
def test_ppustatus_returns_open_bus_during_warmup():
    """SMB cold-boot: instruction 7 (`LDA $2002` after `STA $2000 #$10`)
    must put 0x10 into A, not 0x00.

    Walks the CPU one instruction at a time from cold boot until A
    reflects the LDA $2002 result, then asserts.
    """
    env = nes_core.NESEnvironment(rom_path=str(SMB_ROM), frame_skip=1)
    env.reset_no_advance()

    # Cold-boot SMB byte-stream:
    #   8000  78        SEI
    #   8001  D8        CLD
    #   8002  A9 10     LDA #$10
    #   8004  8D 00 20  STA $2000   <-- fills open bus with 0x10
    #   8007  A2 FF     LDX #$FF
    #   8009  9A        TXS
    #   800A  AD 02 20  LDA $2002   <-- must read 0x10 (open bus)
    #   800D  10 FB     BPL $-5
    #
    # Step 7 instructions (SEI, CLD, LDA #$10, STA $2000, LDX #$FF,
    # TXS, LDA $2002), then check A.
    for _ in range(7):
        env.step_one_instruction()

    pc, a, x, y, sp, p, _nmi = env.cpu_state()

    assert pc == 0x800D, f"expected PC=$800D after 7 instr, got ${pc:04X}"
    assert a == 0x10, (
        f"expected A=0x10 from LDA $2002 open-bus (last $20xx write was "
        f"STA $2000 #$10), got A=0x{a:02X}. The PPU $2002 read must "
        f"return open-bus bits 4-0 = last value placed on the data bus, "
        f"even during the 29658-cycle post-reset warmup. See "
        f"nes_core/src/ppu.rs::write_byte — `ppu_gen_latch` must be "
        f"updated BEFORE any early-return for warmup-restricted writes."
    )
