"""TDD reproduction: feed the same 3354-frame button sequence
(roms/zelda_start_419.state.bin, a legacy action-replay recording)
into both nes_core and nes-py cold-boot, and check that critical
Zelda game-state RAM bytes agree at sample-frame boundaries.

Test RED (expected to fail): nes_core's 40-byte RAM divergence from
nes-py (per tests/parity/test_lockstep_baseline.py) compounds under
active input — the 3354-frame playthrough drives Link through the
title screen, name entry, file select, and into the overworld. By
frame 3354, game state bytes (Link X/Y, room ID, sword inventory,
etc.) will almost certainly differ.

The test names the specific RAM addresses that matter for gameplay
correctness (per the NESdev/Zelda-speedrun RAM map). When it fails,
we know the first scoped divergence and can bisect from there. When
it passes, the gameplay-sensitive subset of emulation is in parity
with nes-py even if total RAM is not.

This is Prove-It: the test MUST fail on HEAD; the fix isn't landed
until this passes.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ZELDA_ROM = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"
ZELDA_TAPE = REPO / "roms" / "zelda_start_419.state.bin"

# Canonical Zelda RAM addresses — gameplay-critical.
# Sources:
#   * datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/RAM_map
#   * nesworld / zelda speedrun community annotations
ZELDA_GAMEPLAY_BYTES: dict[int, str] = {
    0x00EB: "link_x_position_pixels",
    0x0070: "link_y_position_pixels",
    0x00EB: "link_x_tile",
    0x10:   "game_mode",
    0x11:   "game_submode",
    0x657:  "inventory_sword",       # 0=none, 1=wooden, 2=white, 3=magical
    0x658:  "inventory_bombs",
    0x66F:  "current_hearts_halves",
    0x670:  "max_hearts_halves",
    0x675:  "current_rupees",
}


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _load_both(rom: str):
    from tests.parity.lockstep import _load_ours, _load_theirs
    return _load_ours(rom), _load_theirs(rom)


def _replay_and_sample(sample_frames: list[int]) -> dict[int, tuple[dict[int, int], dict[int, int]]]:
    """Step both emulators with the legacy Zelda replay's button
    sequence, snapshotting each ZELDA_GAMEPLAY_BYTES address at each
    frame in `sample_frames`. Returns {frame: (ours, theirs)}."""
    buttons = list(ZELDA_TAPE.read_bytes())
    ours, theirs = _load_both(str(ZELDA_ROM))
    out: dict[int, tuple[dict[int, int], dict[int, int]]] = {}
    sample_set = set(sample_frames)
    for i, b in enumerate(buttons):
        ours.step(int(b) & 0xFF)
        theirs.step(int(b) & 0xFF)
        if i in sample_set:
            out[i] = (
                {a: int(ours.get_ram(a)) for a in ZELDA_GAMEPLAY_BYTES},
                {a: int(theirs.ram[a]) for a in ZELDA_GAMEPLAY_BYTES},
            )
    return out


def _describe_diff(ours: dict[int, int], theirs: dict[int, int]) -> list[str]:
    lines = []
    for addr, name in ZELDA_GAMEPLAY_BYTES.items():
        a, b = ours[addr], theirs[addr]
        if a != b:
            lines.append(f"    ${addr:04X} ({name}): ours=0x{a:02X}  theirs=0x{b:02X}")
    return lines


@pytest.mark.parity
@pytest.mark.xfail(
    strict=True,
    reason=(
        "nes_core has a residual ~1 byte ($01FD, a stack slot) "
        "RAM divergence from nes-py at 600 idle Zelda frames. Under "
        "the 3354-frame scripted replay the divergence compounds: "
        "game_submode ($0011) first differs at frame 1011; Link Y "
        "first differs at frame 1011 also; by frame 3353 nes-py's "
        "Link has picked up the sword ($0657=1) and nes_core's has "
        "not ($0657=0). Root cause is a CPU/PPU cycle-accuracy gap "
        "that drives a flags-register divergence at some interrupt "
        "push. See docs/proposals/archive/cycle_accuracy_plan.md; this test "
        "becomes XPASS (strict=True → FAIL) once the gap is closed."
    ),
)
def test_zelda_critical_ram_tracks_nespy_through_replay():
    """RED on HEAD: gameplay-critical Zelda RAM must match nes-py at
    the end of a 3354-frame scripted replay that drives Link from
    cold-boot into the overworld. Fails on master 2026-04-23 because
    cycle-accuracy drift accumulates over the 55-second playthrough.
    """
    if not ZELDA_ROM.exists() or not ZELDA_TAPE.exists():
        pytest.skip(f"Zelda ROM or tape missing ({ZELDA_ROM}, {ZELDA_TAPE})")

    sample_frames = [500, 1000, 2000, 3353]
    snapshots = _replay_and_sample(sample_frames)

    diffs_by_frame: dict[int, list[str]] = {}
    for frame in sample_frames:
        ours, theirs = snapshots[frame]
        d = _describe_diff(ours, theirs)
        if d:
            diffs_by_frame[frame] = d

    if diffs_by_frame:
        lines = ["Zelda critical RAM diverges from nes-py:"]
        for frame, d in diffs_by_frame.items():
            lines.append(f"  frame {frame}:")
            lines.extend(d)
        pytest.fail("\n".join(lines))
