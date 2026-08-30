"""Lockstep RAM-divergence baseline: records what nes_core vs nes-py
actually look like in CPU RAM after N cold-boot idle frames. The
baseline is explicitly NOT byte-exact (nes_core is not cycle-accurate
with LaiNES — see `project_restart_primer_2026-04-22.md`). These tests
lock in the CURRENT gap so subsequent cycle-accuracy work makes
visible, test-enforced progress rather than silent drift.

Methodology:
  * Both emulators start fresh, step 0 buttons each frame.
  * After frame N, compare 2 KB of CPU RAM ($0000-$07FF) byte-by-byte.
  * Assert divergence ≤ a committed ceiling.

How to update after a cycle-accuracy fix:
  1. Run: `pytest tests/parity/test_lockstep_baseline.py -q`.
  2. If the reported number is lower than the ceiling → tighten the
     ceiling to the new number (commit the improvement).
  3. If higher → regression; investigate before loosening.

Current ceilings are set to CURRENT observed values with no slack;
any fix that improves parity will fail the test and prompt a
ceiling-tightening commit.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(rom: str, frames: int) -> int:
    """Step both emulators `frames` times with button=0 and return the
    number of CPU RAM bytes that differ."""
    from tests.parity.lockstep import _load_ours, _load_theirs, _ram_ours, _ram_theirs
    ours = _load_ours(rom)
    theirs = _load_theirs(rom)
    for _ in range(frames):
        ours.step(0)
        theirs.step(0)
    a = _ram_ours(ours)
    b = _ram_theirs(theirs)
    return sum(1 for i in range(0x800) if a[i] != b[i])


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# (rom-name, frames, max_diverged_bytes)
# Ceilings; lower is better; tighten after each fix.
# History:
#   2026-04-23 initial: SMB 198, Zelda 40, Contra 163, Mega Man 11, Metroid 187
#   2026-04-23 removed NESEnvironment::reset's advance_one_frame warmup
#     (aligned step-count with nes-py, which reset() does NOT advance):
#     SMB 177, Zelda 1, Contra 71, Mega Man 0, Metroid 170. Reverted same
#     day because the GUI Play window showed a frozen black frame on
#     startup (user-visible regression).
#   2026-04-23 kept reset-advance + added a compensating `env.step(0)`
#     to nes-py in `tests/parity/lockstep.py::_load_theirs`. Numbers
#     below were measured under that setup; GUI remains usable.
#   2026-04-24 cycle-locked NESEnvironment::advance_one_frame to exactly
#     29781 CPU cycles per frame (matching nes-py / LaiNES upstream's
#     fixed-iteration emulator step). Fix closes the Zelda cave-stuck /
#     sword-pickup defect at f1011 (verified end-to-end in the GUI play
#     window AND scripts/zelda_cave_entry_repro.py). Cold-boot 600-idle-
#     frame ceilings tracked here are unchanged because the cave-entry
#     parity defect manifested under INPUT, not idle.
#   2026-04-26 disabled-asm-cpu pass: ASM CPU has a known correctness
#     bug — only ticks PPU at MMIO callbacks, not between non-MMIO
#     instructions in the same batch. SMB's $2002 polling loop never
#     saw vblank inside an ASM batch, Mario fell through the floor.
#     Default-disabled the asm_cpu feature in nes_core/pyproject.toml.
#     New ceilings:
#       SMB 175 → 0  (BYTE EXACT)
#       Contra 73 → 5
#       Metroid 170 → 169
CASES = [
    ("roms/Super Mario Bros. (World).nes",                600,   0),
    ("roms/Legend of Zelda, The (USA) (Rev A).nes",       600,   1),
    #   2026-08-14 DMC DMA stall propagation (e9f3164): Contra 5 → 7 —
    #     nes-py has no APU, so hardware-accurate DMC fetch stalls
    #     necessarily diverge from it; Mesen lockstep is the accuracy
    #     oracle for this class.
    ("roms/Contra (USA).nes",                             600,   7),
    ("roms/Mega Man (USA).nes",                           600,   0),
    ("roms/Metroid (USA).nes",                            600, 170),  # 2026-04-26 +1 from $4014 deferred-write fix
]


@pytest.mark.parity
@pytest.mark.parametrize("rom, frames, ceiling", CASES, ids=[c[0].split('/')[-1] for c in CASES])
def test_lockstep_ram_divergence_within_ceiling(rom: str, frames: int, ceiling: int):
    rom_path = REPO / rom
    if not rom_path.exists():
        pytest.skip(f"ROM missing: {rom}")
    diff = _run(str(rom_path), frames)
    # Fail-loud when BETTER than the ceiling → forces us to tighten and
    # commit. Use != rather than > so silent regressions AND silent
    # improvements both surface.
    assert diff <= ceiling, (
        f"{rom} regressed: {diff} bytes diverged at frame {frames}, "
        f"ceiling is {ceiling}."
    )
    if diff < ceiling:
        pytest.fail(
            f"{rom} improved! Was {ceiling}, now {diff}. "
            f"Tighten the ceiling in tests/parity/test_lockstep_baseline.py::CASES "
            f"and commit — the improvement becomes the new floor."
        )
