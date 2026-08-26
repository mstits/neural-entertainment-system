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

from scripts.progress_signal_gate import assess, note_camera_static


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
