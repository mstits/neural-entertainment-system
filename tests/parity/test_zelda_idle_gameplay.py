"""Layer-1 gameplay test for Zelda at idle. The 1-byte residual (per
test_lockstep_baseline.py — at $01FD, a stack push slot) does NOT
affect Zelda's gameplay-state addresses at idle: mode, submode, Link
position, sword inventory, hearts, rupees all match nes-py byte-exact
at the 360-frame sample.

The test_zelda_input_replay.py companion drives a 3354-frame scripted
overworld walk that DOES diverge cycle-accuracy-cumulatively (Link
walks one tile less than nes-py over 1011 frames; sword pickup fails
at frame 1759). That's the test which xpasses when the structural
NMI fix lands. THIS test passes today and locks in that the
fundamental game state in nes_core matches nes-py from cold boot —
the divergence is in cycle-counted side state, not game logic.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROM = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"

# LABELS ARE DECORATIVE, NOT A SEMANTICS CLAIM — sibling of
# test_zelda_input_replay.py's ZELDA_GAMEPLAY_BYTES; keep both in sync,
# fixing one without the other is a half-fix. This is a fidelity/parity
# harness: the assertion is raw-byte agreement between nes_core and
# nes-py, not a claim about what any byte means. 0x0010, 0x0070, 0x00EB,
# and 0x066F are numerically identical to entries `configs/zelda.yaml`
# QUARANTINES as unverified external knowledge (q_dungeon_level,
# q_link_x, q_world_map_x, q_current_hearts/q_max_hearts — see that
# file's quarantine block). That quarantine retracts these labels as
# TRAINING semantics — nothing here feeds a reward function or a win
# predicate.
ZELDA_GAMEPLAY_BYTES = {
    0x0010: "game_mode",
    0x0011: "game_submode",
    0x0070: "link_y_position",
    0x00EB: "link_x_position",
    0x0657: "inventory_sword",
    0x066F: "current_hearts",
    0x0670: "max_hearts",
    0x0675: "current_rupees",
}


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
def test_zelda_gameplay_state_matches_nespy_after_360f_idle():
    if not ROM.exists():
        pytest.skip(f"ROM missing: {ROM}")
    from tests.parity.lockstep import _load_ours, _load_theirs
    ours = _load_ours(str(ROM))
    theirs = _load_theirs(str(ROM))
    for _ in range(360):
        ours.step(0)
        theirs.step(0)
    diffs = []
    for addr, name in ZELDA_GAMEPLAY_BYTES.items():
        o = int(ours.get_ram(addr))
        t = int(theirs.ram[addr])
        if o != t:
            diffs.append(f"${addr:04X} ({name}): ours=0x{o:02X} theirs=0x{t:02X}")
    if diffs:
        pytest.fail("Zelda game-state diverges from nes-py:\n  " + "\n  ".join(diffs))
