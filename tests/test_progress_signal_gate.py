"""scripts/progress_signal_gate.py — assess() must be able to tell a raw
odometer axis that runs BACKWARD under the resolved sign (1942's PPU y
counting down during forward flight) from one that runs forward, even
though both rebase (caller shifts by the trace's own min) into an
identically-shaped nonnegative trace. Without `raw_direction`, assess()
only ever looks at set-based distinct/tail counts, which are blind to
time order, so a backward-running axis used to pass with the exact same
verdict as a healthy one — the 1942 pre-fix false PASS.
"""
from __future__ import annotations

from scripts.progress_signal_gate import (
    assess,
    first_exhaustion_index,
    note_camera_static,
    truncate_at_exhaustion,
)


def _rebased_1942_shape() -> list[int]:
    """The trace shape both directions rebase to: 788 distinct steps
    covering 0..787, matching the real 1942 gate receipt (distinct=600
    over 1200 steps, max=787 — the ramp here is denser only for a
    tighter, deterministic test)."""
    return list(range(788))


def test_negative_raw_direction_fails_even_on_a_clean_rebased_trace():
    trace = _rebased_1942_shape()
    v = assess(trace, None, True, raw_direction=-787)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE"
    assert any("DECREASES" in f for f in v["instrument_findings"])


def test_positive_raw_direction_on_the_identical_trace_still_passes():
    trace = _rebased_1942_shape()
    v = assess(trace, None, True, raw_direction=787)
    assert v["passed"] is True
    assert v["instrument_findings"] == []


def test_raw_direction_is_the_only_thing_distinguishing_the_two_verdicts():
    trace = _rebased_1942_shape()
    forward = assess(trace, None, True, raw_direction=787)
    backward = assess(trace, None, True, raw_direction=-787)
    # Same steps/distinct/min/max/tail_distinct — the pre-fix bug was
    # exactly that these fields, and passed/verdict derived only from
    # them, could not distinguish the two directions at all.
    for key in ("steps", "distinct", "min", "max", "tail_distinct"):
        assert forward[key] == backward[key]
    assert forward["passed"] != backward["passed"]


def test_raw_direction_defaults_to_none_and_does_not_flag():
    # The RAM-byte call site (progress_signal_gate.main's non-odometer
    # branch) does not pass raw_direction — must stay unaffected.
    trace = _rebased_1942_shape()
    v = assess(trace, None, True)
    assert v["passed"] is True
    assert v["instrument_findings"] == []


# --- D6: the vacuous camera-static override -------------------------------
#
# main()'s odometer branch calls note_camera_static() whenever the raw
# rx/ry range is zero — the trace is then a flat run of 1200 zeros, which
# assess() alone already flags via MIN_DISTINCT (distinct=1 < 32). The
# bug this guards against: an earlier version of that call site deleted
# the "too coarse" finding and forced passed=True whenever OAM churn
# showed the agent moving, so a permanently-flat progress column
# certified as "SIGNAL SOUND" — the exact legend_of_zelda receipt
# (distinct=1, min=0, max=0, oam_churn=967, passed=True). A gate that
# cannot fail on a constant-zero column has proven nothing.

def _flat_zero_verdict() -> dict:
    # Same shape assess() sees for a camera that never moved: 1200
    # identical readings, no high-byte wrap, no death signal, net motion
    # of zero under the resolved axis sign.
    return assess([0] * 1200, None, True, raw_direction=0)


def test_flat_progress_column_fails_before_camera_static_override():
    # Baseline: assess() alone already rejects this — confirms the
    # degenerate shape is caught upstream of the override.
    v = _flat_zero_verdict()
    assert v["passed"] is False
    assert any("too coarse" in f for f in v["instrument_findings"])


def test_camera_static_override_does_not_launder_an_active_agent():
    # This is the legend_of_zelda receipt exactly: agent demonstrably
    # active (oam_churn=967 over 1199 possible transitions) on an axis
    # that never moved. Must still fail.
    v = note_camera_static(_flat_zero_verdict(), oam_churn=967, steps=1200)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE — camera static"
    assert any("camera never moved" in f for f in v["instrument_findings"])


def test_camera_static_override_does_not_launder_an_inert_agent():
    v = note_camera_static(_flat_zero_verdict(), oam_churn=0, steps=1200)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE — camera static"


def test_camera_static_override_never_erases_the_coarseness_finding():
    # The pre-fix bug's defining move was `[f for f in instrument_findings
    # if "too coarse" not in f]` — silently deleting the one finding that
    # would have failed it. The fixed function must never shrink
    # instrument_findings, only add to it.
    before = _flat_zero_verdict()
    n_before = len(before["instrument_findings"])
    after = note_camera_static(before, oam_churn=967, steps=1200)
    assert len(after["instrument_findings"]) > n_before
    assert all(
        any(f in g for g in after["instrument_findings"])
        for f in before["instrument_findings"]
    )


# --- D5: the death-blind hold ---------------------------------------------
#
# scripts/progress_signal_gate.py's 1200-step forward hold never watched
# the lives byte at all. A 1-life game (Arkanoid) that loses its only ball
# ~halfway through the hold spent the rest of it stepping a non-interactive
# post-game-over screen, and an incidental byte flip on that dead screen
# got certified as a real "room transition" (reproduced live: $0010 reads
# 0 for all 601 genuinely-live steps and 1 for every step after death,
# corroborating the independent s6_instrument.json finding the discovery
# receipt itself contradicted). first_exhaustion_index/truncate_at_exhaustion
# are the fix: find where the lives trace records a death (GenericGame's
# own modular check) AND never recovers (the trailing quarter is frozen at
# one value — reused, not reinvented: it is the same "flat for the last
# quarter" heuristic assess() already applies to the progress trace), and
# drop everything from there on before assessing.

def test_no_death_recorded_returns_none():
    # Lives never drops — nothing to truncate.
    trace = [3] * 1200
    assert first_exhaustion_index(trace, start_lives=3) is None
    assert truncate_at_exhaustion(1200, trace, 3) == 1200


def test_death_that_never_recovers_is_exhaustion():
    # Arkanoid's exact shape: alive at 1 for 601 steps, drops to 0, then
    # (after this ROM's attract-mode placeholder kicks in) sits at a
    # DIFFERENT nonzero constant for the rest of the hold. A naive
    # "stuck at 0 forever" test misses this; the trailing-quarter freeze
    # check catches it regardless of what the frozen value is.
    trace = [1] * 601 + [0] * 100 + [14] * 499
    idx = first_exhaustion_index(trace, start_lives=1)
    assert idx == 601
    assert truncate_at_exhaustion(len(trace), trace, 1) == 601


def test_single_death_followed_by_continued_variation_is_not_exhaustion():
    # A multi-life game where the FIRST death respawns and play keeps
    # producing varied readings afterward — an ordinary scripted-probe
    # event, not a game-over tail. Must not be flagged.
    import itertools

    respawned = list(itertools.islice(itertools.cycle([2, 5, 9, 3, 7]), 900))
    trace = [3] * 300 + respawned
    assert first_exhaustion_index(trace, start_lives=3) is None
    assert truncate_at_exhaustion(len(trace), trace, 3) == len(trace)


def test_death_followed_by_a_frozen_tail_shorter_than_the_quarter_window():
    # The freeze must hold for the WHOLE trailing quarter — a value that
    # dips to 0 and then keeps moving before the window's own tail
    # re-stabilizes at something else again is not exhaustion either.
    trace = [1] * 700 + [0] * 50 + [3] * 50 + [0] * 200 + [1] * 200
    assert first_exhaustion_index(trace, start_lives=1) is None


def test_start_lives_zero_or_none_never_truncates():
    trace = [0] * 1200
    assert first_exhaustion_index(trace, start_lives=0) is None
    assert truncate_at_exhaustion(1200, trace, 0) == 1200
    assert truncate_at_exhaustion(1200, trace, None) == 1200


def test_no_lives_trace_never_truncates():
    # The no-lives-byte-declared call sites in main() pass lives_trace as
    # None outright; must be a no-op, not a crash.
    assert truncate_at_exhaustion(1200, None, 1) == 1200


def test_wrap_style_death_is_still_detected():
    # GenericGame.is_dead's own modular formula catches a lives byte that
    # displays REMAINING lives and wraps 0 -> 255 on the terminal death
    # (the Ninja Gaiden shape) exactly as it catches a plain decrement —
    # this helper must agree, since it reuses the same check.
    trace = [1] * 500 + [255] * 700
    idx = first_exhaustion_index(trace, start_lives=1)
    assert idx == 500
