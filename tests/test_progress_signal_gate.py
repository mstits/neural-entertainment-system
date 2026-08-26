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
    STASIS_MIN_MEDIAN_CHURN,
    assess,
    assess_hold,
    first_exhaustion_index,
    first_stasis_index,
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


def test_a_lives_byte_flat_at_zero_records_no_death():
    # d == 0 for every step: nothing to detect from the lives byte alone.
    # (The frozen-surface detector is what covers this shape; see
    # test_stasis_catches_the_terminal_death_the_lives_byte_misses.)
    trace = [0] * 1200
    assert first_exhaustion_index(trace, start_lives=0) is None
    assert truncate_at_exhaustion(1200, trace, 0) == 1200
    assert truncate_at_exhaustion(1200, trace, None) == 1200


def test_start_lives_zero_is_not_exempt_from_truncation():
    """Regression for the pinned-inapplicable D5 fix.

    `first_exhaustion_index` used to bail on `start_lives in (None, 0)`
    before looking at the trace, and a test asserted that as correct. On
    the D5 sweep's OWN receipt (runs/onboard_wave6_d5_sweep_v3.json) four
    profiles flagged contaminated — bad_dudes, ducktales_2, ninja_gaiden,
    paperboy — all read `lives_at_start: 0`, so the fix could not apply to
    any of them. The modular check handles the case fine: a 0 -> 255
    underflow is `(0 - 255) % 256 == 1`.
    """
    trace = [0] * 192 + [255] * 1008          # the bad_dudes shape
    assert first_exhaustion_index(trace, start_lives=0) == 192
    assert truncate_at_exhaustion(len(trace), trace, 0) == 192


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


# ==========================================================================
# THE D1 CLASS INSIDE THIS GATE: a terminal death the lives byte never
# reports. `first_exhaustion_index` reuses `GenericGame.is_dead`'s modular
# check `1 <= (start - cur) % 256 <= 8`, which defect D1 proved blind to a
# death that leaves the byte UNCHANGED — on ninja_gaiden_ii the byte at
# $004C sat flat at 1 straight through GAME OVER. The gate needs a second,
# independent detector or it certifies a frozen GAME OVER screen as live
# play, which is exactly the contamination D5 exists to remove.
# ==========================================================================

def _live_churn(n: int, level: int = 200) -> list[int]:
    """Ordinary play: a busy surface, a few hundred bytes moving per step."""
    return [level + (i % 17) for i in range(n)]


def test_stasis_catches_the_terminal_death_the_lives_byte_misses() -> None:
    churn = _live_churn(800) + [0] * 400          # froze at step 800
    lives_flat = [1] * 1200                       # the NG2 shape: d == 0

    assert first_exhaustion_index(lives_flat, start_lives=1) is None, (
        "premise check: the lives byte must be blind to this, or this test "
        "is not exercising the D1 class")

    idx, reason = first_stasis_index(churn)
    assert idx == 800, reason
    assert truncate_at_exhaustion(1200, lives_flat, 1, churn) == 800


def test_ordinary_play_never_looks_like_stasis() -> None:
    """The false-positive direction. A gate that fires on live play would
    truncate every hold on the roster and fail everything."""
    churn = _live_churn(1200)
    idx, reason = first_stasis_index(churn)
    assert idx is None, reason
    assert truncate_at_exhaustion(1200, [3] * 1200, 3, churn) == 1200


def test_a_brief_pause_is_not_an_absorbing_state() -> None:
    """Fades, door transitions and boss intros freeze the surface for a
    moment and then resume. Only a tail that never restarts counts."""
    churn = _live_churn(500) + [0] * 100 + _live_churn(600)
    idx, reason = first_stasis_index(churn)
    assert idx is None, reason


def test_a_short_frozen_tail_is_not_enough() -> None:
    churn = _live_churn(1150) + [0] * 50          # under the 25% floor
    idx, _ = first_stasis_index(churn)
    assert idx is None


def test_a_profile_too_quiet_to_judge_disarms_out_loud() -> None:
    """Refuse rather than guess: under the churn floor a frozen surface
    and ordinary play are not separable, so the test must say so instead
    of manufacturing a verdict."""
    churn = [2] * 600 + [0] * 600                 # median 2 < floor
    idx, reason = first_stasis_index(churn)
    assert idx is None
    assert "DISARMED" in reason
    assert str(STASIS_MIN_MEDIAN_CHURN) in reason


def test_stasis_tolerance_scales_with_the_profiles_own_churn() -> None:
    """A busy game's "frozen" is not a quiet game's "frozen". Tolerance is
    5% of the profile's own live median, so residual counter ticks on a
    dead screen still read as frozen on a busy profile."""
    churn = _live_churn(600, level=1000) + [7] * 600   # 7 <= 5% of ~1000
    idx, reason = first_stasis_index(churn)
    assert idx == 600, reason


def test_the_earliest_of_the_two_detectors_wins() -> None:
    lives = [1] * 300 + [0] * 900                 # death recorded at 300
    churn = _live_churn(800) + [0] * 400          # surface froze at 800
    assert truncate_at_exhaustion(1200, lives, 1, churn) == 300


# ==========================================================================
# assess_hold(): the wiring itself. Every judgement between the last
# emulator step and the printed verdict used to live inline in main(),
# where deleting `xy = xy[:keep]` restored the death-blind hold and failed
# nothing. These drive the production path.
# ==========================================================================

def _rising(n: int) -> list[int]:
    return [i * 3 for i in range(n)]


def test_assess_hold_drops_the_dead_tail_on_the_ram_path() -> None:
    live, dead = 400, 800
    trace = _rising(live) + [9999] * dead
    lives = [1] * live + [0] * dead
    v = assess_hold(ram_trace=trace, lives_trace=lives, lives0=1,
                    churn=_live_churn(live) + [0] * dead,
                    has_high_byte=True, requested_steps=live + dead)
    assert v["steps"] == live, "the post-death tail was assessed as live play"
    assert v["dropped_tail_steps"] == dead
    assert v["max"] == trace[live - 1]


def test_assess_hold_drops_the_dead_tail_on_the_odometer_path() -> None:
    live, dead = 400, 800
    xy = [(i * 4, 0) for i in range(live)] + [(99999, 0)] * dead
    lives = [1] * live + [0] * dead
    v = assess_hold(xy=xy, lives_trace=lives, lives0=1,
                    churn=_live_churn(live) + [0] * dead,
                    oam_changed=[True] * (live + dead),
                    progress_cfg={"source": "odometer", "axis": "x"},
                    requested_steps=live + dead)
    assert v["steps"] == live
    assert v["dropped_tail_steps"] == dead
    assert v["odometer_range"]["x"] == (live - 1) * 4
    assert v["oam_churn"] == live


def test_assess_hold_fails_a_frozen_tail_the_lives_byte_never_saw() -> None:
    """The D1 class end to end: surface froze, declared death observable
    said nothing. That is an INSTRUMENT fault — the profile cannot see its
    own terminal states — so the gate must refuse, not quietly truncate."""
    live, dead = 400, 800
    v = assess_hold(ram_trace=_rising(live) + [9999] * dead,
                    lives_trace=[1] * (live + dead), lives0=1,
                    churn=_live_churn(live) + [0] * dead,
                    has_high_byte=True, requested_steps=live + dead)
    assert v["passed"] is False
    assert "blind" in v["verdict"]
    assert any("no death recorded" in f.lower() or "D1" in f
               for f in v["instrument_findings"])
    assert v["exhaustion"]["stasis_index"] == live
    assert v["exhaustion"]["lives_index"] is None


def test_assess_hold_does_not_cry_blind_when_the_lives_byte_did_its_job() -> None:
    """Negative control for the finding above. If it fired whenever a hold
    ended on a dead screen it would flag every 1-life profile on the
    roster and mean nothing."""
    live, dead = 400, 800
    v = assess_hold(ram_trace=_rising(live) + [9999] * dead,
                    lives_trace=[1] * live + [0] * dead, lives0=1,
                    churn=_live_churn(live) + [0] * dead,
                    has_high_byte=True, requested_steps=live + dead)
    assert "blind" not in v["verdict"]
    assert v["exhaustion"]["lives_index"] == live


def test_assess_hold_still_blocks_a_static_camera() -> None:
    """D6, driven through the production path rather than by calling
    note_camera_static() directly."""
    v = assess_hold(xy=[(0, 0)] * 1200, lives_trace=[3] * 1200, lives0=3,
                    churn=_live_churn(1200),
                    oam_changed=[True] * 1200,
                    progress_cfg={"source": "odometer", "axis": "x"},
                    requested_steps=1200)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE — camera static"
    assert v["odometer_range"] == {"x": 0, "y": 0}


def test_assess_hold_passes_a_clean_hold() -> None:
    """Anti-vacuity: the checks above are only meaningful if this function
    can still return a positive."""
    v = assess_hold(ram_trace=_rising(1200), lives_trace=[3] * 1200, lives0=3,
                    churn=_live_churn(1200), has_high_byte=True,
                    requested_steps=1200)
    assert v["passed"] is True, v["instrument_findings"]
    assert v["dropped_tail_steps"] == 0
    assert v["verdict"].startswith("SIGNAL SOUND")
