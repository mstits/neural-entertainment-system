"""Selector that turns a full minted backward ladder into the sparse
online-campaign restart set (x ~= 2500/2000/1500/1000/500/0).

Pure-logic tests only — no emulator, no state blobs. The selection rules
under test:

* target 0 is the TRUE ENTRANCE: the ladder's first entry (tape step 0),
  never a gx==0 entry from a re-based later segment;
* every other target resolves to the nearest-gx entry INSIDE the
  bottleneck segment (the earliest segment whose max gx covers the
  largest target) — 1-2 re-bases its odometer twice, so "gx 500" is
  ambiguous without the segment rule;
* a target the segment cannot cover within tolerance raises instead of
  silently minting a rung somewhere else;
* the result is sorted by tape step with strictly increasing gx, so the
  TauScheduler's tau indexes it back-to-front (tau_init -1 == the
  deepest rung, x ~= 2500).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.backward_curriculum import StateEntry  # noqa: E402
from scripts.select_restart_states import (  # noqa: E402
    pick_bottleneck_segment, select_entries, split_segments,
)

TARGETS = (2500, 2000, 1500, 1000, 500, 0)


def _ladder() -> list[StateEntry]:
    """Synthetic 1-2-shaped ladder: intro / underground / exit re-base."""
    entries = []
    # Segment A: overworld intro, area 1, gx 0..144.
    for i in range(10):
        entries.append(StateEntry(step=i, frame=i * 4, gx=i * 16, area=1,
                                  file=f"s_{i:06d}.state"))
    # Segment B: underground main area, area 2, gx 0..2673 (odometer
    # re-based at the pipe; area byte moves with it).
    for i in range(100):
        step = 10 + i
        entries.append(StateEntry(step=step, frame=step * 4, gx=i * 27,
                                  area=2, file=f"s_{step:06d}.state"))
    # Segment C: exit area, area 2 THROUGHOUT, gx re-based 0..800 (the
    # 1-2 step-971 shape: same area byte, coordinate wraps).
    for i in range(41):
        step = 110 + i
        entries.append(StateEntry(step=step, frame=step * 4, gx=i * 20,
                                  area=2, file=f"s_{step:06d}.state"))
    return entries


def test_split_segments_finds_three():
    segs = split_segments(_ladder())
    assert len(segs) == 3
    assert [s[0].step for s in segs] == [0, 10, 110]


def test_bottleneck_segment_is_earliest_covering_max_target():
    segs = split_segments(_ladder())
    seg = pick_bottleneck_segment(segs, max_target=2500)
    assert seg[0].step == 10 and seg[-1].gx == 2673


def test_bottleneck_segment_uncoverable_raises():
    segs = split_segments(_ladder())
    with pytest.raises(ValueError):
        pick_bottleneck_segment(segs, max_target=5000)


def test_select_nearest_in_bottleneck_segment():
    chosen = select_entries(_ladder(), TARGETS)
    by_target = {t: e for t, e in chosen}
    assert set(by_target) == set(TARGETS)
    # Non-zero targets land in segment B (steps 10..109) at the nearest
    # multiple of 27.
    for t in (2500, 2000, 1500, 1000, 500):
        e = by_target[t]
        assert 10 <= e.step <= 109
        assert abs(e.gx - t) <= 27 // 2 + 1


def test_entrance_is_the_ladder_head_not_a_rebased_zero():
    chosen = select_entries(_ladder(), TARGETS)
    entrance = dict(chosen)[0]
    assert entrance.step == 0 and entrance.area == 1


def test_result_sorted_by_step_with_increasing_gx_after_entrance():
    chosen = select_entries(_ladder(), TARGETS)
    steps = [e.step for _, e in chosen]
    assert steps == sorted(steps)
    gxs = [e.gx for _, e in chosen]
    assert gxs[1:] == sorted(gxs[1:])
    # Deepest rung last: tau_init -1 must mean "the x~=2500 stage".
    assert chosen[-1][0] == 2500


def test_target_beyond_tolerance_raises():
    # Segment B tops out at 2673; a 2800 target has no rung within
    # tolerance and must be a loud error, not a silent nearest-pick.
    with pytest.raises(ValueError):
        select_entries(_ladder(), (2800, 0), tolerance_px=64)


def test_empty_ladder_raises():
    with pytest.raises(ValueError):
        select_entries([], TARGETS)
