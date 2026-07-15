"""Tests for the DEFERRED Go-Explore unstick burst (Lane 5) decisions in
`src/training/oneshot_curriculum.py`.

The burst is a bounded, reversible archive subroutine armed only when a rung
stalls; the campaign spec (§Q3) requires it to be capped, self-retracting,
and to harvest at most one deeper seed. Those decisions are factored into
four pure functions so the gates the spec names are exercised here without
booting an emulator or a real `GoExploreArchive`:

  * the stall detector fires at N, never at N-1 (and honors the enable /
    reaches / consolidation / final-rung guards);
  * the diverted env quota is capped (no permanent Go-Explore takeover);
  * the burst clock retracts on exactly its Nth tick;
  * the harvest returns at most one seed, and only one strictly deeper than
    the stalled frontier.
"""

from __future__ import annotations

from src.training.oneshot_curriculum import (
    burst_quota,
    burst_tick,
    harvest_burst_seed,
    stall_ready,
)


# ---- stall detector: fires at N, not N-1 ----------------------------------


def _stall(iters, patience=60, **kw):
    base = dict(
        enabled=True, reaches=True, frontier=4, ladder_size=18, blocked=False
    )
    base.update(kw)
    return stall_ready(iters, patience, **base)


def test_stall_detector_fires_at_patience_not_one_short():
    # The exact boundary the gate names: N-1 does not fire, N does.
    assert _stall(59, patience=60) is False
    assert _stall(60, patience=60) is True
    assert _stall(61, patience=60) is True


def test_stall_detector_off_unless_enabled():
    # Off unless the profile set the knob, regardless of how long it stalled.
    assert _stall(500, patience=60, enabled=False) is False


def test_stall_detector_requires_the_rung_to_be_reached():
    # The burst is "find the next state," not "can't play this rung": a rung
    # the pool never even reaches is not a Go-Explore candidate.
    assert _stall(500, patience=60, reaches=False) is False


def test_stall_detector_blocked_during_consolidation_or_active_burst():
    # `blocked` folds in "consolidating" and "a burst is already running".
    assert _stall(500, patience=60, blocked=True) is False


def test_stall_detector_never_at_the_final_rung():
    # No deeper rung to seed -> nothing to harvest, so never arm.
    assert _stall(500, patience=60, frontier=17, ladder_size=18) is False
    # One rung short of the top still fires (rung 17 remains to be seeded).
    assert _stall(500, patience=60, frontier=16, ladder_size=18) is True


# ---- quota cap: no permanent Go-Explore takeover --------------------------


def test_burst_quota_is_a_fraction_of_the_pool():
    assert burst_quota(24, 0.25, 8) == 6   # round(24 * 0.25)


def test_burst_quota_is_hard_capped():
    # Even at frac=1.0 the cap bounds the diversion (curriculum keeps the
    # majority) — this is what prevents a permanent GE takeover.
    assert burst_quota(24, 1.0, 8) == 8
    assert burst_quota(24, 0.9, 8) == 8


def test_burst_quota_never_exceeds_the_pool():
    assert burst_quota(24, 1.0, 100) == 24


def test_burst_quota_at_least_one_but_zero_on_empty_pool():
    assert burst_quota(24, 0.0, 8) == 1
    assert burst_quota(0, 0.5, 8) == 0


# ---- retraction: the burst self-terminates on its Nth tick ----------------


def test_burst_tick_counts_down_and_retracts_on_the_last():
    rem, retract = burst_tick(3)
    assert (rem, retract) == (2, False)
    rem, retract = burst_tick(rem)
    assert (rem, retract) == (1, False)
    rem, retract = burst_tick(rem)
    assert (rem, retract) == (0, True)   # retracts on the last iter


def test_burst_of_n_iters_retracts_on_exactly_iter_n():
    burst_iters = 30
    rem = burst_iters
    retracted_at = None
    for i in range(1, burst_iters + 5):
        rem, retract = burst_tick(rem)
        if retract:
            retracted_at = i
            break
    assert retracted_at == burst_iters   # never earlier, never permanent


def test_burst_tick_is_idempotent_once_exhausted():
    # A stray tick past zero still reports retract (and never goes negative).
    assert burst_tick(0) == (0, True)
    assert burst_tick(-5) == (0, True)


# ---- harvest-one-seed rule ------------------------------------------------


def test_harvest_returns_the_single_deepest_state_past_frontier():
    cells = [
        (2, b"shallow"),
        (5, b"deep_a"),
        (5, b"deep_b"),
        (3, b"mid"),
        (7, None),          # deepest region but no blob -> ineligible
    ]
    result = harvest_burst_seed(cells, frontier=4)
    # Only regions strictly > 4 with a blob qualify; deepest such is region 5.
    assert result is not None
    region, state = result
    assert region == 5
    assert state == b"deep_a"   # first max wins (stable)


def test_harvest_returns_none_when_nothing_is_deeper():
    # Everything at or below the stalled frontier is no help.
    assert harvest_burst_seed([(1, b"x"), (4, b"y"), (4, b"z")], frontier=4) is None


def test_harvest_skips_stateless_cells_even_if_deepest():
    # A None-state deeper cell can't seed; fall through to the deepest with a
    # blob.
    assert harvest_burst_seed([(9, None), (5, b"z")], frontier=4) == (5, b"z")


def test_harvest_of_empty_archive_is_none():
    assert harvest_burst_seed([], frontier=4) is None


def test_harvest_yields_at_most_one_seed():
    # Many deeper candidates -> the burst still emits exactly one.
    cells = [(6, b"a"), (7, b"b"), (8, b"c"), (8, b"d")]
    result = harvest_burst_seed(cells, frontier=4)
    assert result == (8, b"c")   # single deepest, first-seen on a tie
