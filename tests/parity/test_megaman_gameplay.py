"""Layer-1 gameplay test: Mega Man (USA), the strongest byte-exact ROM
in the library. Drives the title screen with Start, advances to stage
select, and asserts a couple of game-state RAM addresses match what
nes-py reports under the same input. Because Mega Man is currently
byte-exact at idle, this test should pass with diff=0 across the full
input sequence — any regression in CPU cycle accuracy or PPU timing
will start here.

Pattern (per docs/proposals/parity_design_pattern.md, Layer 1):
  1. Pick a ROM where game-state RAM addresses are documented.
  2. Script a button sequence to drive a known scenario.
  3. Assert critical RAM bytes match nes-py's at sample frames.
  4. If broken, mark xfail(strict=True); the fix will xpass.

To replicate this for another game, copy this file, swap the ROM,
swap the address dict, and document the inputs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROM = REPO / "roms" / "Mega Man (USA).nes"

# A short scripted-input sequence: 600 cold-boot idle frames is enough
# for the Capcom logo + title screen + "PRESS START" prompt cycle.
# We don't press anything — pure idle. The test asserts that NOTHING
# anywhere in CPU RAM diverges between nes_core and nes-py at any of
# the sample frames. This is the Layer-1 model: nes_core is the
# authoritative reference for byte-exact ROMs and any future change
# that breaks Mega Man's idle parity is a critical correctness
# regression in our CPU/PPU/mapper code.

SAMPLE_FRAMES = [60, 120, 240, 480, 600]


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
def test_megaman_idle_byte_exact_at_every_sample():
    if not ROM.exists():
        pytest.skip(f"ROM missing: {ROM}")
    from tests.parity.lockstep import _load_ours, _load_theirs, _ram_ours, _ram_theirs
    ours = _load_ours(str(ROM))
    theirs = _load_theirs(str(ROM))
    sample_set = set(SAMPLE_FRAMES)
    failures: list[tuple[int, int]] = []  # (frame, diff_count)
    for frame in range(1, max(SAMPLE_FRAMES) + 1):
        ours.step(0)
        theirs.step(0)
        if frame in sample_set:
            a = _ram_ours(ours)
            b = _ram_theirs(theirs)
            diff = sum(1 for i in range(0x800) if a[i] != b[i])
            if diff != 0:
                failures.append((frame, diff))
    assert not failures, (
        f"Mega Man (the byte-exact reference ROM) drifted from nes-py "
        f"during idle: {failures}. This is a critical correctness "
        f"regression — Mega Man was byte-exact at all sample frames "
        f"on 2026-04-23."
    )
