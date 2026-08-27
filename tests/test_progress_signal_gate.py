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


# ==========================================================================
# THE TRUNCATION-ORDER DEFECT (2026-08-26). `assess()` computed its
# RESOLUTION instrument finding on the window that survives D5
# truncation, and stated it as a threshold claim:
#
#   "only 20 distinct values in 69 steps (< 32) — too coarse to be a
#    search gradient"
#
# A 69-sample window cannot demonstrate a 32-distinct threshold. That
# sentence measures how fast the scripted forward hold died, not the
# signal's resolution, and on that inference Contra — which has a real
# 16-bit progress pair {lo:0x65, hi:0x64} — was excluded from the
# roster. The honest verdict on a window that short is INCONCLUSIVE:
# blocked, but not condemned. Same VOID-vs-FAIL distinction CLAIMS.md
# draws for evals.
#
# The fix is deliberately ONE-SIDED and these tests pin both sides:
# observing >= MIN_DISTINCT levels is a positive demonstration and needs
# no window floor (Rygar's 116-in-138 still passes); observing fewer only
# means something if the window could have shown them.
# ==========================================================================

from scripts.progress_signal_gate import (  # noqa: E402
    INCONCLUSIVE_VERDICT,
    MIN_ASSESSABLE_STEPS,
    MIN_DISTINCT,
    exit_code,
    probe_actions,
    select_longest_live_window,
    steps_to_distinct,
)


#: The literal 69-step live window contra.yaml produces under the hold
#: probe, lifted from the sweep's own raw traces
#: (docs/receipts/progress_gate_window_sweep_2026-08-26.json records the
#: derived row: 69 live steps, 20 distinct, range 0..70, of a requested
#: 1200 — 1131 dropped as a post-death tail). 27 steps of standing still
#: while the level scrolls in, 18 steps of real advance at +4 a step, and
#: then dead against the first wall. This is the trace the old gate
#: called "too coarse to be a search gradient".
CONTRA_LIVE_WINDOW = (
    [0] * 27 + [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57,
                61, 65, 69] + [70] * 24)


def _contra_shape() -> list[int]:
    trace = list(CONTRA_LIVE_WINDOW)
    assert len(trace) == 69 and len(set(trace)) == 20 and max(trace) == 70
    return trace


def test_contras_short_window_is_inconclusive_not_unusable():
    v = assess(_contra_shape(), 2, True)
    assert v["verdict"] == INCONCLUSIVE_VERDICT, v["verdict"]
    assert v["instrument_findings"] == [], (
        "a 69-sample window cannot support a 32-distinct threshold claim, "
        "so nothing here is an instrument FAULT")
    assert any("cannot demonstrate" in f for f in v["inconclusive_findings"])
    assert v["passed"] is False, "INCONCLUSIVE is not a pass"


def test_the_identical_shortfall_on_a_long_window_is_still_unusable():
    """The negative control, and the reason the fix is not a blanket
    amnesty. Same 20 distinct values, but stretched over 1200 live steps:
    now the window CAN carry the claim, so the claim is made."""
    trace = [v for v in _contra_shape() for _ in range(18)][:1200]
    assert len(trace) == 1200 and len(set(trace)) == 20
    v = assess(trace, 2, True)
    assert v["verdict"] == "SIGNAL UNUSABLE"
    assert any("too coarse" in f for f in v["instrument_findings"])
    assert v["inconclusive_findings"] == []


def test_a_short_window_that_demonstrates_resolution_still_passes():
    """Rygar's receipt: 138 live steps, 116 distinct. The window floor
    gates the FAILING direction only — 116 observed levels is a positive
    demonstration and no floor can retract it. If this ever flips, the
    fix has become a blanket 'short window => no verdict' rule, which
    would void the profile the Rygar campaign actually ran on."""
    trace = list(range(116)) + list(range(22))
    assert len(trace) == 138 and len(set(trace)) == 116
    v = assess(trace, 1, True, raw_direction=467)
    assert v["passed"] is True, v["instrument_findings"]
    assert v["verdict"].startswith("SIGNAL SOUND")
    assert v["inconclusive_findings"] == []


def test_a_demonstrated_fault_outranks_an_unsupportable_window():
    """Precedence. A window too short to judge resolution is still long
    enough to read the start state: a death byte at 0 underflows whatever
    the window length is, so that profile is UNUSABLE, not INCONCLUSIVE.
    Evidence beats absence of evidence."""
    v = assess(_contra_shape(), 0, True)
    assert v["verdict"] == "SIGNAL UNUSABLE"
    assert any("underflow" in f for f in v["instrument_findings"])
    # ...and the window finding is still recorded, not swallowed.
    assert v["inconclusive_findings"]


def test_min_window_zero_reproduces_every_pre_fix_verdict():
    """Historical comparability: `min_window=0` is the old assessor,
    exactly. scripts/progress_gate_sweep.py diffs the two columns with
    this and nothing else, so if it ever stopped reproducing the old
    behaviour the whole 'N verdicts changed' number would be measuring
    two moving parts."""
    legacy = assess(_contra_shape(), 2, True, min_window=0)
    assert legacy["verdict"] == "SIGNAL UNUSABLE"
    assert any("too coarse" in f for f in legacy["instrument_findings"])
    assert legacy["inconclusive_findings"] == []


def test_the_floor_can_never_convert_a_fail_into_a_pass():
    """The property that makes the floor safe to raise: it only ever
    moves a verdict from FAIL to VOID. Swept over every floor from 0 to
    twice the calibrated value on the shape that fails for resolution."""
    trace = _contra_shape()
    for floor in range(0, 2 * MIN_ASSESSABLE_STEPS, 17):
        assert assess(trace, 2, True, min_window=floor)["passed"] is False


def test_exit_code_separates_void_from_fail():
    """The distinction has to reach a caller or it is decoration."""
    assert exit_code(assess(list(range(1200)), 3, True)) == 0
    assert exit_code(assess([0] * 1200, 3, True)) == 1        # too coarse
    assert exit_code(assess(_contra_shape(), 2, True)) == 2   # window short


def test_steps_to_distinct_is_the_calibration_measurement():
    assert steps_to_distinct(list(range(1000)), 32) == 32
    assert steps_to_distinct([i // 4 for i in range(1000)], 32) == 125
    assert steps_to_distinct([7] * 1000, 32) is None


def test_min_assessable_steps_is_at_least_the_slowest_measured_signal():
    """MIN_ASSESSABLE_STEPS is calibrated, not chosen: it is the largest
    `steps_to_min_distinct` over every roster profile whose signal DOES
    reach MIN_DISTINCT levels (Kung Fu, 187). A floor below that would
    call a signal too coarse in a window where a measured, real signal
    had not yet cleared the bar itself."""
    from pathlib import Path
    import json
    receipt = (Path(__file__).resolve().parent.parent /
               "docs/receipts/progress_gate_window_sweep_2026-08-26.json")
    data = json.loads(receipt.read_text())
    assert MIN_ASSESSABLE_STEPS >= data["calibration"]["max"], (
        f"floor {MIN_ASSESSABLE_STEPS} is below the slowest measured "
        f"time-to-{MIN_DISTINCT} on the roster "
        f"({data['calibration']['max']}, "
        f"{data['calibration']['slowest_profile']})")


# --- the same defect on the camera-static branch --------------------------

def test_camera_static_on_a_short_window_is_inconclusive():
    """`note_camera_static` made the identical unsupportable claim, and
    on the requested step count rather than the live one: "camera never
    moved over 1200 steps" about a hold that died at step 22."""
    v = note_camera_static(_flat_zero_verdict(), oam_churn=10, steps=1200,
                           live_steps=22)
    assert v["passed"] is False
    assert v["verdict"].startswith(INCONCLUSIVE_VERDICT)
    assert not any("camera never moved" in f
                   for f in v["instrument_findings"])
    assert any("22 live steps" in f for f in v["inconclusive_findings"])


def test_camera_static_on_a_long_window_is_still_unusable():
    """Negative control: the D6 regression case (legend_of_zelda,
    oam_churn=967 over a full 1200-step live window) must be unaffected."""
    v = note_camera_static(_flat_zero_verdict(), oam_churn=967, steps=1200,
                           live_steps=1200)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE — camera static"
    assert any("camera never moved" in f for f in v["instrument_findings"])


def test_assess_hold_reports_inconclusive_through_the_production_path():
    """End to end on the wiring `main()` actually uses: a hold that dies
    at step 69 of 1200 on a coarse-looking signal."""
    live = _contra_shape()
    dead = 1131
    v = assess_hold(ram_trace=live + [999] * dead,
                    lives_trace=[2] * len(live) + [0] * dead, lives0=2,
                    churn=_live_churn(len(live)) + [0] * dead,
                    has_high_byte=True, requested_steps=len(live) + dead)
    assert v["steps"] == 69
    assert v["dropped_tail_steps"] == dead
    assert v["verdict"] == INCONCLUSIVE_VERDICT
    assert v["passed"] is False
    assert exit_code(v) == 2


def test_assess_hold_still_fails_a_long_coarse_hold():
    """Anti-vacuity for the above: the production path must still be able
    to return SIGNAL UNUSABLE for coarseness, or the fix has disarmed the
    check instead of scoping it."""
    trace = [i // 60 for i in range(1200)]
    v = assess_hold(ram_trace=trace, lives_trace=[3] * 1200, lives0=3,
                    churn=_live_churn(1200), has_high_byte=True,
                    requested_steps=1200)
    assert v["verdict"] == "SIGNAL UNUSABLE"
    assert exit_code(v) == 1


# ==========================================================================
# THE PROBE. The gate's original probe holds one direction from a
# `lives=1` start state and does not dodge, so on any game with an early
# hazard it walks into it and the "live window" it reports is a property
# of the probe. Rygar: 138 steps held forward, against a measured median
# 677 for uniform-random and 3,865-6,018 solver actions in one life.
# ==========================================================================

def test_hold_probe_is_the_unchanged_default():
    assert probe_actions(8, 5, 3) == [3, 3, 3, 3, 3]
    assert probe_actions(8, 5, 3, "hold", seed=99) == [3, 3, 3, 3, 3]


def test_random_probe_actually_varies():
    """A 'random' probe that emitted one action every step would be the
    old probe wearing a hat, and every window measurement taken with it
    would be meaningless."""
    a = probe_actions(8, 400, 1, "random", seed=0)
    assert len(a) == 400
    assert len(set(a)) == 8, "did not reach every action in 400 draws"
    assert max(a.count(i) for i in range(8)) < 400


def test_random_probe_is_seed_reproducible_and_seed_sensitive():
    assert probe_actions(8, 200, 1, "random", seed=7) == \
        probe_actions(8, 200, 1, "random", seed=7)
    assert probe_actions(8, 200, 1, "random", seed=7) != \
        probe_actions(8, 200, 1, "random", seed=8)


def test_random_probe_respects_the_action_space_bounds():
    for n in (2, 3, 11):
        assert set(probe_actions(n, 500, 0, "random", seed=1)) <= set(range(n))


def test_episode_selection_takes_the_longest_live_window():
    assert select_longest_live_window([120, 900, 40]) == 1
    # Ties go to the earliest seed, so the choice is deterministic.
    assert select_longest_live_window([900, 900, 40]) == 0


def test_the_axis_sign_check_is_disarmed_by_an_undirected_probe():
    """A uniform-random policy commands no direction, so the sign of net
    odometer motion is a property of the dice. Answering the 1942
    axis-sign question from it would report or miss that fault at random,
    so the check refuses instead — the same discipline
    `first_stasis_index` uses when a profile is too quiet to judge."""
    backward = [(0, -i * 4) for i in range(600)]
    cfg = {"source": "odometer", "axis": "y"}
    directed = assess_hold(xy=backward, lives_trace=[3] * 600, lives0=3,
                           churn=_live_churn(600), oam_changed=[True] * 600,
                           progress_cfg=cfg, requested_steps=600)
    assert directed["passed"] is False
    assert any("DECREASES" in f for f in directed["instrument_findings"])

    undirected = assess_hold(xy=backward, lives_trace=[3] * 600, lives0=3,
                             churn=_live_churn(600), oam_changed=[True] * 600,
                             progress_cfg=cfg, requested_steps=600,
                             directed=False)
    assert not any("DECREASES" in f
                   for f in undirected["instrument_findings"]), (
        "an undirected probe must not answer the axis-sign question at all")


def test_an_undirected_probe_cannot_report_a_shortfall_as_a_fault():
    """The mirror image of the truncation-order defect, found by running
    the new probe: bionic_commando's odometer shows 122 distinct levels
    over a 1200-step forward hold and 21 over a 1200-step uniform-random
    rollout (docs/receipts/progress_gate_random_probe_2026-08-26.json).
    The window is not the problem there — 1200 steps is the full hold —
    the POLICY is. So an undirected probe may add evidence and never
    subtract a certification."""
    wandering = [i % 21 for i in range(1200)]
    directed = assess(wandering, 3, True)
    assert directed["verdict"] == "SIGNAL UNUSABLE"
    assert any("too coarse" in f for f in directed["instrument_findings"])

    undirected = assess(wandering, 3, True, directed=False)
    assert undirected["verdict"] == INCONCLUSIVE_VERDICT
    assert undirected["instrument_findings"] == []
    assert any("commanded no direction" in f
               for f in undirected["inconclusive_findings"])
    assert undirected["passed"] is False, "still not a pass"


def test_an_undirected_probe_can_still_demonstrate_resolution():
    """Anti-vacuity for the rule above. If disarming the fault direction
    also disarmed the positive direction, the random probe could never
    settle anything and would not be worth running — Contra's 346
    distinct over a 721-step random window is exactly the evidence it
    exists to produce."""
    v = assess(list(range(346)) * 2, 2, True, directed=False)
    assert v["passed"] is True, v["inconclusive_findings"]
    assert v["verdict"].startswith("SIGNAL SOUND")


def test_an_undirected_probe_cannot_report_a_static_camera_as_a_fault():
    base = assess([0] * 1200, None, True, raw_direction=0, directed=False)
    v = note_camera_static(base, oam_churn=967, steps=1200,
                           live_steps=1200, directed=False)
    assert v["passed"] is False
    assert v["verdict"].startswith(INCONCLUSIVE_VERDICT)
    assert v["instrument_findings"] == []


def test_an_undirected_probe_still_reports_window_independent_faults():
    """Scope check: directedness gates the two findings that depend on
    the agent having gone somewhere. A death byte reading 0 at the start
    state, and a byte that reached 200 with no pair, are facts the probe
    cannot manufacture or hide, so they still block."""
    v = assess([i * 3 for i in range(400)], 0, False, directed=False)
    assert v["passed"] is False
    assert v["verdict"] == "SIGNAL UNUSABLE"
    assert any("underflow" in f for f in v["instrument_findings"])
    assert any("no paired high byte" in f for f in v["instrument_findings"])
