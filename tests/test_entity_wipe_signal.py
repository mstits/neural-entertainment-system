"""entity_wipe_windows — the object-array-collapse signal, divorced from the
coord signal's position-reset precondition.

scripts.clear_detect.coord_entity_windows requires BOTH a position readout
dropping sharply toward a level-start value AND a contiguous RAM block
collapsing to (near) zero. Verified at the top level: the position half is
not merely untuned on non-SMB profiles, it is either unreachable (an
odometer profile re-anchors WITHOUT integrating across a scene cut, so a
real stage wipe never reads as a backward gx move) or INVERTED (a player
retreating toward the origin satisfies "dropped by >= 300 and landed <=
200" just as well as a level-start teleport does — see
test_the_old_coord_did_fire_on_backward_walk below). entity_wipe_windows is
the wipe half alone, with no position argument at all.

This file is self-contained: every RAM fixture is built here rather than
imported from tests/test_confluence_v2.py, so this signal's tests do not
depend on that file's fixtures surviving unchanged underneath a sibling's
concurrent edits.

Every test uses a plain 2 KiB uint8 array as its RAM history — no address
is claimed to mean anything beyond "some contiguous run of bytes". That is
the purity litmus this signal has to pass: a party who has never seen any
of the narrated games (death, room transition, combat blip below) can
derive "an object array being cleared" from the byte pattern alone.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.clear_detect import (
    ENTITY_WIPE_EXCLUDE_DEFAULT,
    ENTITY_WIPE_MIN_BYTES,
    ENTITY_WIPE_TOL,
    coord_entity_windows,
    entity_wipe_windows,
)

# Module defaults entity_wipe_windows is exercised against below.
WINDOW = 60
STRIDE = 15


def _ram(n: int) -> np.ndarray:
    return np.zeros((n, 2048), dtype=np.uint8)


def _permanent_wipe(at: int, n_bytes: int = ENTITY_WIPE_MIN_BYTES,
                     n: int = 120, lo: int = 0x60) -> np.ndarray:
    """Occupied for i < at, wiped and never refilled for i >= at — a
    contiguous run of `n_bytes` bytes at [lo, lo+n_bytes)."""
    ram = _ram(n)
    ram[:at, lo:lo + n_bytes] = 9
    return ram


def _transient_wipe(at: int, dur: int, n: int = 200,
                     lo: int = 0x60, hi: int = 0x70) -> np.ndarray:
    """Occupied throughout except a `dur`-frame gap starting at `at` —
    entities vanish and then come back, the shape of a squad despawning
    without the level itself ending."""
    ram = _ram(n)
    ram[:, lo:hi] = 9
    ram[at:at + dur, lo:hi] = 0
    return ram


# --------------------------------------------------------------------------
# The signal must be able to say "no": prove it CAN return false before
# trusting anything it says elsewhere.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("occupied", [True, False])
def test_entity_wipe_does_not_fire_when_the_mechanism_is_absent(occupied: bool) -> None:
    """Nothing collapses: RAM is either occupied the whole time or empty the
    whole time. Neither is a wipe, so the signal must be silent in both
    cases — this is the test that would have caught a signal that fires on
    ANY nonzero byte, or on any zero byte, rather than on a transition."""
    ram = _ram(WINDOW)
    if occupied:
        ram[:, 0x60:0x60 + ENTITY_WIPE_MIN_BYTES] = 9
    assert entity_wipe_windows(ram) == []


def test_entity_wipe_ignores_a_fill_the_reverse_of_a_wipe() -> None:
    """Bytes going empty -> occupied (an object table being POPULATED, e.g.
    entities spawning in) must not satisfy a signal that is specifically
    about occupied -> empty. Same magnitude of change as a real wipe, the
    opposite direction."""
    ram = _ram(WINDOW)
    ram[WINDOW // 2:, 0x60:0x60 + ENTITY_WIPE_MIN_BYTES] = 9
    assert entity_wipe_windows(ram) == []


# --------------------------------------------------------------------------
# The min_bytes boundary, at every window alignment.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("offset", range(STRIDE))
def test_entity_wipe_boundary_is_exercised_in_both_directions(offset: int) -> None:
    """A constant that is never approached from below is a constant nothing
    tests. `at` is walked across a full stride period (offset 0..14) so the
    transition lands at every phase the window/stride grid can put it at;
    at every one of those alignments, a run one byte short of min_bytes
    must never fire and a run of exactly min_bytes must always fire."""
    at = 40 + offset
    below = _permanent_wipe(at, n_bytes=ENTITY_WIPE_MIN_BYTES - 1)
    at_min = _permanent_wipe(at, n_bytes=ENTITY_WIPE_MIN_BYTES)
    assert entity_wipe_windows(below) == []
    assert entity_wipe_windows(at_min) != []


# --------------------------------------------------------------------------
# Anti-vacuity control + the documented false-positive classes. Each of
# these is a POSITIVE assertion that the signal DOES fire — proof that the
# suppression tests above are measuring something, and a permanent record
# of why this signal may never be a standalone or majority-carrier vote.
# --------------------------------------------------------------------------

def test_entity_wipe_still_fires_on_the_gradius_death_stream() -> None:
    """DEATH (the 2026-08-06 Gradius finding): a shmup respawn wipes the
    entity slot table exactly like a stage load does. This is the
    anti-vacuity control for the whole file — if this ever goes green by
    the signal being silent, the boundary tests above are measuring
    nothing. lives/position are irrelevant to this signal by construction;
    only the RAM shape is exercised here."""
    ram = _permanent_wipe(at=200, n_bytes=16, n=400)
    assert entity_wipe_windows(ram) != []


def test_entity_wipe_fires_on_a_room_transition_shape() -> None:
    """ROOM TRANSITION (the Kirby 3-fires-in-24s finding): an entirely
    ordinary room change wipes the same table a clear would. No lives are
    lost, no game-ending event has happened."""
    ram = _permanent_wipe(at=58, n_bytes=12, n=200)
    assert entity_wipe_windows(ram) != []


def test_entity_wipe_fires_on_a_combat_blip_shape() -> None:
    """COMBAT BLIP: a wave of enemies despawns together and the level keeps
    going — entities return afterward, never a permanent collapse. `dur`
    is held >= stride so the fire does not depend on where in the
    window/stride grid the blip happens to land (a `dur` shorter than the
    stride is alignment-lottery territory, exactly like the coord signal's
    own documented blip sensitivity, and is not the property under test
    here)."""
    ram = _transient_wipe(at=60, dur=STRIDE + 5, lo=0x60, hi=0x70)
    assert entity_wipe_windows(ram) != []


# --------------------------------------------------------------------------
# Calibration knobs: region / exclude.
# --------------------------------------------------------------------------

def test_stack_page_is_excluded_by_default() -> None:
    """The stack page ($0100-$01FF) oscillates with call/return traffic and
    can manufacture a nonzero->zero run with no object semantics at all —
    an unexamined false-positive source in coord_entity_windows, which
    scans the undifferentiated full 2 KiB. entity_wipe_windows must not
    inherit it: a wipe confined entirely to the stack page must be silent
    under the default exclude list."""
    ram = _permanent_wipe(at=40, n_bytes=20, lo=0x100)
    assert entity_wipe_windows(ram) == []


def test_exclude_can_be_overridden_to_widen_the_scan() -> None:
    """The stack-page exclusion is a default, not a hardcoded blind spot —
    a caller that explicitly passes an empty exclude list gets the region
    scanned."""
    ram = _permanent_wipe(at=40, n_bytes=20, lo=0x100)
    assert entity_wipe_windows(ram, exclude=[]) != []


def test_region_restricts_the_scan_to_the_declared_bytes() -> None:
    """A profile that narrows `region` to where its own null measurement
    shows real object-array structure must have a wipe OUTSIDE that region
    ignored, and the identical wipe INSIDE it detected."""
    ram = _permanent_wipe(at=40, n_bytes=20, lo=0x300)
    assert entity_wipe_windows(ram, region=[(0x60, 0x70)]) == []
    assert entity_wipe_windows(ram, region=[(0x300, 0x320)]) != []


# --------------------------------------------------------------------------
# The falsifier for the divorce itself: the position-reset half is not
# just defeated on this shape, it has no argument left to be defeated ON.
# --------------------------------------------------------------------------

def test_the_old_coord_did_fire_on_backward_walk() -> None:
    """Pins the measured defect being fixed, so it stays in the record.
    coord_entity_windows's "reset" check is `gx_after <= 200 and
    (gx_before - gx_after) >= 300` — a description that a player WALKING
    BACKWARDS toward the origin satisfies just as well as a level-start
    teleport does. The identical wipe wrapped in genuine forward progress
    (the honest direction) does not satisfy it at all: retreat was the
    only surviving trigger shape."""
    ram = _permanent_wipe(at=1, n_bytes=ENTITY_WIPE_MIN_BYTES, n=WINDOW)
    gx_backward = np.array([1200 - 20 * i for i in range(WINDOW)])
    gx_forward = np.array([20 + 20 * i for i in range(WINDOW)])
    assert coord_entity_windows(ram, gx_backward) == [(0, WINDOW)]
    assert coord_entity_windows(ram, gx_forward) == []


def test_position_reset_half_is_gone() -> None:
    """entity_wipe_windows fires on the exact same RAM shape the previous
    test proves is retreat-only under the old joint check — unconditionally,
    because there is no position argument here for a trajectory's direction
    to matter to. The signature itself proves the deletion, not just this
    call's result."""
    ram = _permanent_wipe(at=1, n_bytes=ENTITY_WIPE_MIN_BYTES, n=WINDOW)
    assert entity_wipe_windows(ram) == [(0, WINDOW)]
    params = inspect.signature(entity_wipe_windows).parameters
    assert "gx" not in params and "gx_series" not in params


# --------------------------------------------------------------------------
# Defaults match the historical coord_entity_windows constants (same tol /
# min_bytes / window / stride) so this signal is a drop-in replacement for
# the wipe half wherever it is wired, not a silently different one.
# --------------------------------------------------------------------------

def test_defaults_match_the_historical_coord_constants() -> None:
    assert ENTITY_WIPE_MIN_BYTES == 8
    assert ENTITY_WIPE_TOL == 4
    assert ENTITY_WIPE_EXCLUDE_DEFAULT == [(0x0100, 0x0200)]
