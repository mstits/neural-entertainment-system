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

from scripts.progress_signal_gate import assess


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
