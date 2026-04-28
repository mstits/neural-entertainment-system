"""Layer-4 long-soak: 10 ROMs that stay byte-exact with nes-py for
2400 cold-boot idle frames (40 seconds of emulated time). This is
the strongest correctness guarantee in the suite — every CPU/PPU/
APU/mapper interaction over the full 40-second window must produce
identical RAM byte-for-byte.

Surfaced 2026-04-23 by sweeping the byte_exact fleet at 120/600/2400
sample points; these 10 ROMs maintain byte-exact through the full
window. The other 13 byte-exact-at-120f ROMs (e.g. Bionic Commando,
Tetris, Dr. Mario) drift slightly by 600f or 2400f — still in the
"tight" bucket, but no longer perfect. Tracked separately by
test_byte_exact_fleet.py at the 120f sample.

Slow test (~25s); marked `@pytest.mark.slow` so a developer can
opt out via `-m "not slow"` for the inner loop.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

LONG_SOAK_BYTE_EXACT_ROMS = [
    "Bad Street Brawler (USA).nes",
    "BreakThru (USA).nes",
    "Dragon Warrior III (USA).nes",
    "Final Fantasy (USA).nes",
    "Godzilla 2 - War of the Monsters (USA).nes",
    "Kid Kool and the Quest for the Seven Wonder Herbs (USA).nes",
    "Lee Trevino's Fighting Golf (USA).nes",
    "Mega Man (USA).nes",
    "Rally Bike (USA).nes",
    "Terra Cresta (USA).nes",
]

LONG_SOAK_FRAMES = 2400


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize("rom", LONG_SOAK_BYTE_EXACT_ROMS)
def test_rom_byte_exact_through_long_soak(rom: str):
    """Hardest correctness wall: this ROM stays byte-exact with nes-py
    through 2400 idle frames (40 seconds). Any change that introduces
    even a single byte of drift over the full window fails — a deep
    correctness regression."""
    rom_path = REPO / "roms" / rom
    if not rom_path.exists():
        pytest.skip(f"ROM missing: {rom}")
    from tests.parity.lockstep import _load_ours, _load_theirs, _ram_ours, _ram_theirs
    ours = _load_ours(str(rom_path))
    theirs = _load_theirs(str(rom_path))
    for _ in range(LONG_SOAK_FRAMES):
        ours.step(0)
        theirs.step(0)
    a = _ram_ours(ours)
    b = _ram_theirs(theirs)
    mismatches = [i for i in range(0x800) if a[i] != b[i]]
    assert not mismatches, (
        f"{rom}: {len(mismatches)} bytes diverged after {LONG_SOAK_FRAMES} idle frames. "
        f"This ROM was byte-exact at 2400f on 2026-04-23. "
        f"First 5 diffs: {[(hex(x), hex(a[x]), hex(b[x])) for x in mismatches[:5]]}"
    )
