"""Unit tests for the backward-curriculum trailing-entropy guard.

The guard exists because of one measured failure (B4 v1, 2026-08-08): the
1-1 reverse curriculum's policy entropy fell 0.19 -> 0.04 after ~iter 130
and the collapsed argmax stopped taking the final flag jump the sampled
policy still took (cold greedy 0.02 vs sampled 0.50/0.63). These tests
pin the decision logic — arm, hold, hysteretic release — without an
emulator; `tests/test_vanilla_ppo_backward_guard_smoke.py` covers the
trainer wiring that applies it.

The properties that matter here are the ones a live run cannot re-litigate:
* it never arms on a single noisy iter (min_samples + trailing mean),
* it never flaps (the disarm band sits strictly above the arm band),
* an absent config block builds nothing at all, which is what makes the
  feature opt-in and every pre-guard lineage still valid.
"""

from __future__ import annotations

import pytest

from src.training.trainer import (
    GUARD_ARM,
    GUARD_DISARM,
    BackwardEntropyGuard,
)


# ---- construction / opt-in ------------------------------------------


@pytest.mark.parametrize("cfg", [None, {}, [], 0, "", {"enabled": False},
                                 {"enabled": False, "floor": 0.08,
                                  "boost": 3.0}])
def test_absent_or_disabled_config_builds_nothing(cfg) -> None:
    """No block => no guard => the trainer's entropy_coef path is the
    pre-guard path exactly. This is the opt-in contract."""
    assert BackwardEntropyGuard.from_config(cfg) is None


def test_config_present_but_incomplete_raises() -> None:
    """A half-written block must not degrade to a silent no-op — that is
    this project's dead-knob bug class (config_schema.py)."""
    with pytest.raises(ValueError, match="floor"):
        BackwardEntropyGuard.from_config({"boost": 3.0})
    with pytest.raises(ValueError, match="boost"):
        BackwardEntropyGuard.from_config({"floor": 0.08})


def test_from_config_reads_every_knob() -> None:
    g = BackwardEntropyGuard.from_config({
        "floor": 0.09, "boost": 2.5, "trailing": 7, "min_samples": 3,
        "recover_mult": 2.0, "armed_history": 11,
    })
    assert g is not None
    assert (g.floor, g.boost, g.trailing, g.min_samples) == (0.09, 2.5, 7, 3)
    assert g.recover_mult == 2.0
    assert g.recover_floor == pytest.approx(0.18)
    # armed_history bounds the rolling window the kill criterion reads.
    for _ in range(50):
        g.observe(0.5)
    assert g.history_len == 11


@pytest.mark.parametrize("kwargs", [
    {"floor": 0.0, "boost": 3.0},
    {"floor": -0.1, "boost": 3.0},
    {"floor": 0.08, "boost": 0.5},     # would sharpen a collapsing policy
    {"floor": 0.08, "boost": 3.0, "trailing": 0},
    {"floor": 0.08, "boost": 3.0, "min_samples": 0},
    {"floor": 0.08, "boost": 3.0, "recover_mult": 0.9},   # bands invert
    {"floor": 0.08, "boost": 3.0, "armed_history": 0},
])
def test_out_of_range_values_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        BackwardEntropyGuard(**kwargs)


def test_min_samples_cannot_exceed_the_window() -> None:
    """A sample floor longer than the trailing window could never be met,
    so the guard would be permanently inert. Clamp instead."""
    g = BackwardEntropyGuard(0.08, 3.0, trailing=4, min_samples=99)
    assert g.min_samples == 4
    for _ in range(4):
        ev = g.observe(0.001)
    assert ev == GUARD_ARM


# ---- arming ---------------------------------------------------------


def test_does_not_arm_before_min_samples() -> None:
    """One catastrophic minibatch entropy is noise, not a collapse."""
    g = BackwardEntropyGuard(0.08, 3.0, trailing=10, min_samples=5)
    for _ in range(4):
        assert g.observe(0.0) is None
        assert not g.armed
        assert g.multiplier == 1.0
    assert g.observe(0.0) == GUARD_ARM
    assert g.armed
    assert g.multiplier == 3.0


def test_healthy_entropy_never_arms() -> None:
    g = BackwardEntropyGuard(0.08, 3.0, trailing=10, min_samples=5)
    for _ in range(200):
        assert g.observe(0.19) is None
    assert not g.armed
    assert g.arms == 0
    assert g.armed_frac == 0.0


def test_arm_event_fires_once_then_the_guard_just_holds() -> None:
    g = BackwardEntropyGuard(0.08, 3.0, trailing=5, min_samples=5)
    events = [g.observe(0.02) for _ in range(20)]
    assert events[:5] == [None, None, None, None, GUARD_ARM]
    assert all(e is None for e in events[5:])
    assert g.armed and g.arms == 1
    # Held armed the whole time after the transition.
    assert g.multiplier == 3.0


def test_b4_receipt_trace_arms_after_the_collapse_not_before() -> None:
    """Replay of the measured B4 v1 shape: healthy at 0.19 through the
    ladder, then 0.04 from ~iter 130. The guard must sleep through the
    first phase and arm shortly into the second."""
    g = BackwardEntropyGuard.from_config({"floor": 0.08, "boost": 3.0})
    assert g is not None
    for _ in range(130):
        assert g.observe(0.19) is None
    armed_at = None
    for i in range(130, 260):
        if g.observe(0.04) == GUARD_ARM:
            armed_at = i
            break
    assert armed_at is not None, "guard slept through a 0.19 -> 0.04 collapse"
    # Trailing-mean smoothing costs a handful of iters; it must not cost
    # dozens, or the boost lands after the policy is already welded shut.
    assert 130 < armed_at <= 145, armed_at
    assert g.armed and g.multiplier == 3.0


# ---- hysteresis / release -------------------------------------------


def test_a_poke_above_the_floor_does_not_disarm() -> None:
    """Disarming at `floor` exactly would flap once per iter around the
    boundary, whipsawing entropy_coef by the full boost each time."""
    g = BackwardEntropyGuard(0.08, 3.0, trailing=4, min_samples=4,
                             recover_mult=1.25)
    for _ in range(4):
        g.observe(0.02)
    assert g.armed
    # Trailing mean climbs over the floor (0.08) but stays under the
    # recover band (0.10).
    for _ in range(4):
        assert g.observe(0.09) is None
    assert g.trailing_mean == pytest.approx(0.09)
    assert g.armed, "disarmed inside the hysteresis band"


def test_real_recovery_disarms_once() -> None:
    g = BackwardEntropyGuard(0.08, 3.0, trailing=4, min_samples=4,
                             recover_mult=1.25)
    for _ in range(4):
        g.observe(0.02)
    assert g.armed
    # 0.11 clears the recover band (0.10) as a VALUE but not as a mean
    # until it has displaced the whole collapsed window — the release is
    # keyed on the trailing mean, not on one good iter.
    events = [g.observe(0.11) for _ in range(8)]
    assert events.count(GUARD_DISARM) == 1
    assert events[:3] == [None, None, None], events
    assert events[3] == GUARD_DISARM, events
    assert not g.armed
    assert g.multiplier == 1.0


def test_arm_disarm_arm_is_countable() -> None:
    g = BackwardEntropyGuard(0.08, 3.0, trailing=2, min_samples=2)
    seq = [0.01, 0.01, 0.9, 0.9, 0.01, 0.01]
    events = [g.observe(x) for x in seq]
    assert [e for e in events if e] == [GUARD_ARM, GUARD_DISARM, GUARD_ARM]
    assert g.arms == 2
    assert g.armed


# ---- window bookkeeping ---------------------------------------------


def test_trailing_window_forgets_old_samples() -> None:
    """A healthy prefix must not hold the guard off forever once the
    policy has actually collapsed."""
    g = BackwardEntropyGuard(0.08, 3.0, trailing=3, min_samples=3)
    for _ in range(20):
        g.observe(0.5)
    assert not g.armed
    assert g.observe(0.0) is None       # mean still 0.33
    assert g.observe(0.0) is None       # mean still 0.17
    assert g.observe(0.0) == GUARD_ARM  # window is all-collapse now
    assert g.samples == 3
    assert g.trailing_mean == pytest.approx(0.0)


def test_trailing_mean_is_zero_before_any_sample_but_cannot_arm() -> None:
    g = BackwardEntropyGuard(0.08, 3.0, trailing=10, min_samples=5)
    assert g.trailing_mean == 0.0
    assert g.samples == 0
    assert not g.armed


def test_armed_fraction_tracks_the_kill_criterion_window() -> None:
    """B5 registers "armed for >50% of 50 consecutive iters" as a kill
    criterion, so the guard has to be able to answer that directly."""
    g = BackwardEntropyGuard(0.08, 3.0, trailing=2, min_samples=2,
                             armed_history=10)
    for _ in range(10):
        g.observe(0.01)
    assert g.history_len == 10
    assert g.armed_recent == 9        # first iter was pre-min_samples
    assert g.armed_frac == pytest.approx(0.9)
    for _ in range(5):
        g.observe(0.9)
    # The window only remembers `armed_history` iters, so the fraction
    # decays as the recovery ages instead of reporting the whole run.
    assert g.armed_recent == 5
    assert g.armed_frac == pytest.approx(0.5)
    for _ in range(5):
        g.observe(0.9)
    assert g.history_len == 10
    assert g.armed_recent == 0
    assert g.armed_frac == 0.0
