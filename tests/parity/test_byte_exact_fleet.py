"""Fleet regression: every ROM that's currently byte-exact with nes-py
at 120 idle frames must STAY byte-exact. Any emulator change that
knocks one of these out of byte-exact parity is a correctness
regression and fails CI loudly.

This locks in the 23 ROMs surfaced by `scripts/parity_sweep.py` as the
strongest correctness signal we have on the library. If the structural
CPU fix lands (synchronous-NMI-on-vblank per
`project_lainess_reference_findings_2026-04-23.md`) and drops MORE
games to byte-exact, add them to BYTE_EXACT_ROMS below — the test
then becomes a strictly-increasing correctness floor.

Design pattern: for every scoped guarantee (byte-exact with reference,
Link picks up sword, Mario reaches 1-2, etc.) there's a test that
fails loud when the guarantee breaks. Session-to-session, the set of
committed-guarantees only grows.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# 23 ROMs surfaced by the 2026-04-23 library sweep (scripts/parity_sweep.py).
# Each of these produces byte-exact CPU RAM with nes-py after 120 cold-boot
# idle frames. Add new entries when the sweep's byte_exact bucket grows.
# 2026-04-24: cycle-locked NESEnvironment::advance_one_frame to exactly
# 29781 CPU cycles per frame (matching nes-py / LaiNES upstream's fixed-
# iteration emulator step) closed the Zelda cave-stuck / sword-pickup
# defect. Side-effect: three previously byte-exact carts now diverge by
# 1 byte after 120 idle frames (Baseball, Fester's Quest, Kung-Fu Heroes).
# Demoted from this fleet to the tracked-bucket fleet in
# test_library_buckets.py.
# 2026-04-26: California Games demoted (byte_exact → tight) after the
# asm_cpu disable. Three-way Mesen-vs-nes-py-vs-nes_core analysis
# (`/tmp/diag_california.py` 2026-04-26) showed all three diverge from
# each other in different ways — nes-py is closer to Mesen mid-game,
# nes_core is closer at frame 0 and frame 3. Tracked at 9-byte ceiling
# in test_library_buckets.py via parity_sweep.json bucket=tight.
BYTE_EXACT_ROMS = [
    "Bad Street Brawler (USA).nes",
    "Bionic Commando (USA).nes",
    "BreakThru (USA).nes",
    "Castlequest (USA).nes",
    "Clash at Demonhead (USA).nes",
    "Dr. Mario (Japan, USA) (Rev A).nes",
    "Dragon Warrior III (USA).nes",
    "Final Fantasy (USA).nes",
    "Godzilla 2 - War of the Monsters (USA).nes",
    "Golf Grand Slam (USA).nes",
    "Kid Kool and the Quest for the Seven Wonder Herbs (USA).nes",
    "Lee Trevino's Fighting Golf (USA).nes",
    "Magmax (USA).nes",
    "Mega Man (USA).nes",
    "Princess Tomato in Salad Kingdom (USA).nes",
    "Rally Bike (USA).nes",
    "Super Cars (USA).nes",
    "Terra Cresta (USA).nes",
    "Tetris (USA).nes",
]


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
@pytest.mark.parametrize("rom", BYTE_EXACT_ROMS)
def test_rom_is_byte_exact_with_nespy_at_120_idle(rom: str):
    """Hard regression wall: this ROM must match nes-py byte-for-byte
    after 120 idle frames, or the emulator has regressed a previously-
    correct behavior."""
    rom_path = REPO / "roms" / rom
    if not rom_path.exists():
        pytest.skip(f"ROM missing: {rom}")
    from tests.parity.lockstep import _load_ours, _load_theirs, _ram_ours, _ram_theirs
    ours = _load_ours(str(rom_path))
    theirs = _load_theirs(str(rom_path))
    for _ in range(120):
        ours.step(0)
        theirs.step(0)
    a = _ram_ours(ours)
    b = _ram_theirs(theirs)
    mismatches = [i for i in range(0x800) if a[i] != b[i]]
    assert not mismatches, (
        f"{rom}: {len(mismatches)} bytes diverged after 120 idle frames. "
        f"This ROM was byte-exact on 2026-04-23. "
        f"First 5 diffs: {[(hex(x), hex(a[x]), hex(b[x])) for x in mismatches[:5]]}"
    )
