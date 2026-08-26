"""UNREACHABLE is not a miss: the guard that separates VOID from FAIL.

WHAT WENT WRONG, so the tests can be aimed at it. The 41-profile
adjudication closed 4 CONFIRMED / 41 VOID / 0 FAIL. FAIL is exactly zero
because not one gap profile was ever measured by an instrument demonstrated
capable of returning a positive on that profile. Two receipts in this repo
show the mechanism:

  * runs/clear_control_2026-08-26/cv_odometer_swap.json — one witnessed
    Castlevania clear, one detector, ONE key changed (progress ->
    {source: odometer}). Hits 3/3 -> 0/3, coord checks 4/22 -> 0/22,
    largest single-step drop 592 -> 4 against the >= 300 `coord` requires.
    The detector constructed happily and returned a silent miss on a clear
    it had just found under the other arm.
  * runs/clear_control_2026-08-26/bb_offline_r99.json — two Bubble Bobble
    rows whose progress observable spans ONE unit, summarised as
    `n_valid: 2, hit_rate: 0.0, hit_rate_pass: false`.

Both are VOID-shaped measurements written down in FAIL-shaped numbers, and
41 of them accumulated before anyone noticed. So every test below is aimed
at one question:

    WHAT WOULD THIS TEST REPORT IF THE MECHANISM WERE ABSENT?

Answered leg by leg, because the discipline this week is that a test which
passes both before and after a change has certified nothing:

  * the construction refusal legs fail without `from_profile`/`UnfireableHook`
    — before this change the detector constructed on ANY profile, which is
    how a harness ran 30 and 220 checks against a 300-unit requirement and
    wrote `streaming_hit: false`;
  * the denominator legs fail against the old fold, which was
    `n_valid = sum(1 for r in per_run if "error" not in r)` followed by
    `hit_rate = n_hit / n_valid if n_valid else 0.0`;
  * the DEGENERATE leg fails against the old ceiling, which hardcoded
    `max_vote = 1.0 + coord + apu` — `tally` an unconditional vote with no
    separation test anywhere in the tree, despite firing on 22/22, 28/28,
    43/43 Castlevania checks and 30/30 Bubble Bobble checks;
  * the Rule-5 leg fails without the required-class term: three
    corroborators summing to THRESHOLD with coord=0 is precisely the
    frame-320 false positive, 1736 frames early;
  * and the last two are the anti-vacuity guards — three inputs, three
    answers, plus a mutation table that requires the verdict to MOVE. A
    function hardcoded to UNREACHABLE fails them, and so does one hardcoded
    to FIREABLE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from clear_reachability import (  # noqa: E402
    ALIVE,
    CLEAR,
    DEAD,
    DEGENERATE,
    ERROR,
    FIREABLE,
    LIVE,
    MAX_NULL_RATE,
    NO_CLEAR,
    NOT_WIRED,
    OFFLINE,
    COLLAPSING_SLOTS,
    SLOTS,
    shelf_specs,
    UNDER_WARMUP,
    UNREACHABLE,
    clear_quorum,
    launch_banner,
)

clear_detect = pytest.importorskip(
    "clear_detect", reason="needs the compiled nes_core extension")
StreamingConfluenceDetector = clear_detect.StreamingConfluenceDetector
UnfireableHook = clear_detect.UnfireableHook
summarize_runs = clear_detect.summarize_runs


def _profile(**solve) -> dict:
    """A solve block declaring the confluence hook, one knob overridable."""
    base = {
        "rom": "roms/does-not-need-to-exist.nes",
        "progress": {"lo": 0x0040, "hi": 0x0041},
        "y": 0x0002,
        "level_key": [],
        "lives": 0x0032,
        "clear": {"mode": "confluence"},
    }
    base.update(solve)
    return {"solve": base}


# ==========================================================================
# 1. The refusal itself.
# ==========================================================================

@pytest.mark.parametrize("label,progress,why", [
    # Measured spans from runs/clear_control_2026-08-26/: bubble_bobble's
    # progress readout moves 98..99 (largest drop 1) and tetris_b's 0..32,
    # against the >= 300 drop `coord` requires.
    ("bubble_bobble", {"lo": 0x0401}, "single RAM byte"),
    ("tetris_b", {"lo": 0x00D8}, "single RAM byte"),
    # The cv_odometer_swap mutation: ONE key changed, 3/3 hits -> 0/3,
    # largest single-step drop 592 -> 4.
    ("castlevania_odo", {"source": "odometer", "axis": "x"}, "odo_fold_frame"),
    # Same monotone shape, different integral.
    ("fight_gate", {"source": "fight_gate", "foe_hp": 0x0398}, "fight_gate"),
])
def test_a_profile_whose_signals_cannot_reach_quorum_reports_unreachable_not_a_miss(
        label, progress, why) -> None:
    prof = _profile(progress=progress)
    q = clear_quorum(prof)

    assert q.verdict == UNREACHABLE, label
    assert q.signal_state["coord"].state == DEAD
    assert why in q.signal_state["coord"].reason, "the reason names the mechanism"
    assert "coord" not in q.eligible
    assert q.slots["S_TRANSITION"] == []

    # (a) The detector REFUSES TO CONSTRUCT, so no harness can ever again
    #     run 30 checks on bubble_bobble and write `streaming_hit: false`.
    with pytest.raises(UnfireableHook) as exc:
        StreamingConfluenceDetector.from_profile(prof, lambda r: 0)
    assert why in str(exc.value), "the refusal carries its own evidence"

    # (b) The receipt writer keeps it OUT of the denominator.
    rep = summarize_runs([{"profile": prof, "detected_frame": None,
                           "true_clear_frame": 2056,
                           "within_tolerance": False}])
    assert rep["n_unreachable"] == 1
    assert rep["n_valid"] == 0
    assert rep["hit_rate"] is None            # NOT 0.0
    assert rep["hit_rate_pass"] is None       # NOT False
    assert rep["per_run"][0]["verdict"] == UNREACHABLE
    assert rep["verdict"] == UNREACHABLE


def test_the_refusal_names_the_reason_rather_than_returning_a_bare_false() -> None:
    """A silent no-op is the failure mode, not a bad exit code. The message
    has to carry the whole per-signal table so a reader can see the ceiling
    without re-deriving it."""
    with pytest.raises(UnfireableHook) as exc:
        StreamingConfluenceDetector.from_profile(
            _profile(progress={"source": "odometer"}), lambda r: 0)
    msg = str(exc.value)
    assert "UNREACHABLE" in msg
    assert "coord" in msg and "DEAD" in msg
    assert "tally" in msg, "the eligible signals are on the record too"


# ==========================================================================
# 2. Rule 2 — separation, not merely liveness.
# ==========================================================================

def test_a_signal_that_fires_on_every_null_check_is_degenerate_not_a_vote() -> None:
    """Measured: `tally` fired 22/22, 28/28 and 43/43 Castlevania checks
    (cv_live.json, cv_odometer_swap.json) and 30/30 Bubble Bobble checks
    (bb_live.json). The old ceiling counted it as a full castable vote, so
    every one of those profiles advertised a 2-of-2 confluence that was
    really coord-alone wearing a two-signal label."""
    q = clear_quorum(_profile(), null_rates={"tally": 1.0})
    assert q.signal_state["tally"].state == DEGENERATE
    assert "tally" not in q.eligible
    assert q.slots["S_CADENCE"] == []
    assert q.ceiling == 1.0                    # coord alone
    assert q.verdict == UNREACHABLE            # 1 < min_signals 2


def test_an_unmeasured_null_is_reported_as_unmeasured_not_guessed() -> None:
    """The COORD_RESET_DROP_MIN lesson, applied to Rule 2. No profile
    carries a measured null FIRE-RATE today: scripts/clear_calibrate.py
    exists as of the wire-up but measures the scene_cut gate, not a
    per-signal fire-rate. Inventing a plausible rate would be the same
    defect that produced an SMB-shaped constant applied to 45 games.
    Absent a measurement a signal is ALIVE-but-unseparated, and says so."""
    q = clear_quorum(_profile())
    assert q.verdict == FIREABLE
    assert q.signal_state["tally"].state == ALIVE
    assert q.signal_state["tally"].null_rate is None
    assert "UNMEASURED" in q.signal_state["tally"].reason


def test_the_null_rate_threshold_is_a_real_boundary() -> None:
    """A gate nobody probed at its edge is a gate nobody has tested."""
    below = clear_quorum(_profile(), null_rates={"tally": MAX_NULL_RATE / 2})
    at = clear_quorum(_profile(), null_rates={"tally": MAX_NULL_RATE})
    assert below.signal_state["tally"].state == ALIVE
    assert at.signal_state["tally"].state == DEGENERATE


def test_a_profile_may_declare_its_own_measured_nulls() -> None:
    """Where the numbers will live once they are measured: in the profile,
    next to everything else that was measured for that profile."""
    q = clear_quorum(_profile(null_rates={"tally": 0.9}))
    assert q.signal_state["tally"].state == DEGENERATE


# ==========================================================================
# 3. Rule 5 — corroborators alone cannot carry a clear.
# ==========================================================================

def test_corroborators_alone_cannot_reach_quorum() -> None:
    """runs/clear_control_2026-08-26/bb_offline_r99.json fired at frame 320
    on {audio: 1, tally: 1, lock: 1, coord: 0} — three corroborators
    summing to exactly THRESHOLD=0.75 with zero transition evidence — 1736
    frames before the true clear at 2056. Under the required class the
    offline roster cannot reach its bar on that shape however many
    corroborators agree."""
    q = clear_quorum(_profile(progress={"lo": 0x0401}), roster=OFFLINE)
    assert q.ceiling >= q.required, "the SUM still reaches the bar"
    assert q.verdict == UNREACHABLE, "and the required class still refuses it"
    assert "TRANSITION EVIDENCE" in q.reason
    assert "frame 320" in q.reason


def test_the_input_lock_probe_is_never_transition_evidence() -> None:
    """Its own docstring: it "must never be the sole term in a clear vote".
    It is an arming condition and a corroborator, and counting it as a full
    quarter-vote is half of what carried frame 320."""
    q = clear_quorum(_profile(), roster=OFFLINE)
    assert q.signal_state["lock"].state == ALIVE
    assert q.signal_state["lock"].transition_evidence is False
    assert q.signal_state["input_lock"].transition_evidence is False
    assert q.signal_state["coord"].transition_evidence is True


def test_arming_another_corroborator_cannot_make_firing_easier() -> None:
    """The additive trap the v3 apu_weight docstring warns about in
    writing: at min_signals=2, arming a third additive signal lets
    tally+apu carry a clear with coord silent. Behavioural, on the real
    detector, over a stream with no coord signature anywhere."""
    n = 400
    stream = [_ram(gx=600 + 2 * i, timer=250 - i // 8, score=i // 8)
              for i in range(n)]
    masks = [0b00001 if i < 200 else 0 for i in range(n)]
    det = StreamingConfluenceDetector(_progress, apu_weight=1.0)
    fired = None
    for i, ram in enumerate(stream):
        if det.push(ram, masks[i]):
            fired = i
            break
    assert fired is None, "cadence corroborators alone must not fire"
    assert det.n_checks > 0, "the detector must actually have run"
    assert det.n_required_class_vetoes > 0, (
        "and it must have been the required class that stopped it, not "
        "the stream failing to produce a vote at all")


def test_the_same_detector_still_fires_when_transition_evidence_is_present() -> None:
    """The over-correction guard for the test above. Rule 5 must refuse the
    frame-320 shape and NOTHING else — a detector that never fires is not a
    fix, it is the 41-VOID roster with extra steps."""
    stream = _load_stream()
    det = StreamingConfluenceDetector(_progress)
    fired = None
    for i, ram in enumerate(stream):
        if det.push(ram):
            fired = i
            break
    assert fired is not None
    assert det.n_required_class_vetoes == 0


# ==========================================================================
# 4. Rule 1 — a profile votes solely on signals that CAN fire for it.
# ==========================================================================

def test_a_degenerate_signal_contributes_nothing_to_the_live_vote() -> None:
    """Eligibility is not merely a report. A signal the table marks
    DEGENERATE must stop being counted as a corroborator that happened to
    be quiet, in the vote itself — otherwise the table is a document and
    the arithmetic is unchanged, which is the "six signals wired to
    nothing" defect in a new place."""
    stream = _load_stream()
    prof = _profile()

    # min_signals 1 so the vote can be carried by coord alone, isolating
    # what `tally` contributes.
    open_prof = _profile(clear={"mode": "confluence", "min_signals": 1})
    q_open = clear_quorum(open_prof)
    q_degen = clear_quorum(open_prof, null_rates={"tally": 1.0})
    assert q_open.verdict == FIREABLE and q_degen.verdict == FIREABLE

    def _first(q):
        det = StreamingConfluenceDetector(
            _progress, min_signals=1, eligibility=q)
        for i, ram in enumerate(stream):
            if det.push(ram):
                return i, det
        return None, det

    i_open, det_open = _first(q_open)
    i_degen, det_degen = _first(q_degen)
    assert i_open is not None and i_degen is not None
    # `tally` fires on this stream from the first check; with it eligible a
    # min_signals=1 vote is carried by cadence alone and Rule 5 vetoes it,
    # so the fire has to wait for coord either way -- but the eligible run
    # must record cadence votes it had to veto, and the degenerate run
    # must record none, because the signal is not counted at all.
    assert det_open.n_required_class_vetoes > det_degen.n_required_class_vetoes
    assert clear_quorum(prof).ceiling == 2.0
    assert clear_quorum(prof, null_rates={"tally": 1.0}).ceiling == 1.0


def test_an_ineligible_transition_signal_cannot_unlock_the_required_class() -> None:
    """The double-standard guard. A signal marked DEGENERATE contributes 0
    to the sum; if it could still satisfy Rule 5 by firing, the mask would
    be removing it from one half of the vote and leaving it in the other.
    Both halves read the same eligibility."""
    stream = _load_stream()
    prof = _profile(clear={"mode": "confluence", "min_signals": 1})

    def _first(**rates):
        q = clear_quorum(prof, null_rates=rates or None)
        det = StreamingConfluenceDetector(
            _progress, min_signals=1, eligibility=q)
        for i, ram in enumerate(stream):
            if det.push(ram):
                return i, det
        return None, det

    i_live, _ = _first()
    assert i_live is not None, "coord eligible: the stream really does fire"
    i_dead, det_dead = _first(coord=1.0)
    assert i_dead is None, (
        "coord DEGENERATE: it cannot carry the required class either")
    assert det_dead.n_required_class_vetoes > 0
    assert det_dead.warmup_observations() == 1 << 30, (
        "and the warm-up budget says unsatisfiable rather than naming a "
        "number a caller could meet and then read the silence as a verdict")


def test_the_six_shelf_signals_are_wired_but_dead_until_a_profile_arms_them() -> None:
    """Defect D2, closed. entity_wipe, room_fp_transition, input_lock,
    lock_release_novelty, oam_quiesce and scene_cut were built and tested
    on 2026-08-26 and, for one day, reached neither production path. They
    now reach both, so NOT_WIRED is no longer the honest answer for them
    — but WIRED IS NOT ARMED, and a profile that declares none of them
    must be byte-identical to the day before the wire-up. DEAD-with-the-
    key-named is the answer that says both things at once."""
    q = clear_quorum(_profile())
    shelf = [s.name for s in shelf_specs(LIVE)]
    assert set(shelf) == {
        "scene_cut", "room_fp_transition", "input_lock",
        "oam_quiesce", "entity_wipe", "lock_release_novelty"}
    for name in shelf:
        st = q.signal_state[name]
        assert st.state == DEAD, name
        assert st.weight == 1.0, "wired signals cast a live-roster vote"
        assert f"clear.signals.{name}" in st.reason, "the key is named"
        assert name not in q.eligible
    # ...and an unarmed profile's ceiling has not moved a particle.
    assert q.ceiling == 2.0


@pytest.mark.parametrize("name,knobs", [
    ("scene_cut", {"scene_min": 1, "blank_min": 1}),
    ("oam_quiesce", {}),
    ("entity_wipe", {"min_bytes": 12}),
    ("room_fp_transition", {"mask": [[0, 960]]}),
    ("input_lock", {}),
    ("lock_release_novelty", {"lock_max": 90, "m": 60}),
])
def test_arming_one_signal_moves_that_signal_and_nothing_else(name, knobs) -> None:
    """The anti-vacuity leg for the arming rule: six inputs, six different
    tables. A `signal_config` hardcoded to None fails every row; one
    hardcoded to a dict fails the unarmed test above."""
    prof = _profile(clear={"mode": "confluence", "signals": {name: knobs}})
    q = clear_quorum(prof)
    assert q.signal_state[name].state == ALIVE, q.signal_state[name].reason
    assert name in q.eligible
    for other in (s.name for s in shelf_specs(LIVE)):
        if other != name:
            assert q.signal_state[other].state == DEAD, other


@pytest.mark.parametrize("name,partial,missing", [
    ("scene_cut", {"blank_min": 1}, "scene_min"),
    ("entity_wipe", {"tol": 4}, "min_bytes"),
    ("room_fp_transition", {"settle": 3}, "mask"),
    ("lock_release_novelty", {"m": 60}, "lock_max"),
])
def test_arming_without_the_measured_constant_is_dead_not_defaulted(
        name, partial, missing) -> None:
    """COORD_RESET_DROP_MIN, generalized. Each of these signals refuses a
    global default IN ITS OWN DOCSTRING; a profile that arms one without
    measuring must not quietly inherit a number that was never about it.
    Delete the REQUIRED_KNOBS check and all four rows go ALIVE."""
    prof = _profile(clear={"mode": "confluence", "signals": {name: partial}})
    st = clear_quorum(prof).signal_state[name]
    assert st.state == DEAD
    assert missing in st.reason
    assert f"clear.signals.{name}" in st.reason


def test_a_signal_can_be_declined_explicitly_and_say_so() -> None:
    """`enabled: false` and absence produce the same arithmetic and
    different reasons — the profile that thought about it and the profile
    that never did are not the same fact."""
    off = clear_quorum(_profile(
        clear={"mode": "confluence",
               "signals": {"oam_quiesce": {"enabled": False}}}))
    absent = clear_quorum(_profile())
    assert off.signal_state["oam_quiesce"].state == DEAD
    assert absent.signal_state["oam_quiesce"].state == DEAD
    assert off.ceiling == absent.ceiling
    assert "declines" in off.signal_state["oam_quiesce"].reason
    assert "not declared" in absent.signal_state["oam_quiesce"].reason


def test_a_disarmed_apu_vote_is_dead_not_silently_counted() -> None:
    q = clear_quorum(_profile())
    assert q.signal_state["apu"].state == DEAD
    assert "apu_weight is 0" in q.signal_state["apu"].reason
    armed = clear_quorum(_profile(
        clear={"mode": "confluence", "apu_weight": 1.0}))
    assert armed.signal_state["apu"].state == ALIVE
    assert armed.ceiling == 3.0


# ==========================================================================
# 5. Rule 3 — slots are recorded; the tripwire that forces the arithmetic.
# ==========================================================================

def test_every_signal_carries_the_question_it_answers() -> None:
    q = clear_quorum(_profile(), roster=OFFLINE)
    for name, st in q.signal_state.items():
        assert st.slot in SLOTS, name
    assert q.signal_state["coord"].slot == "S_TRANSITION"
    assert q.signal_state["tally"].slot == "S_CADENCE"
    assert q.signal_state["oam_quiesce"].slot == "S_DESPAWN"
    assert q.signal_state["entity_wipe"].slot == "S_DESPAWN"
    assert q.signal_state["lock_release_novelty"].slot == "S_IRREVERSIBLE"


@pytest.mark.parametrize("roster", [LIVE, OFFLINE])
def test_the_tripwire_fired_and_every_non_cadence_slot_now_has_two_members(
        roster) -> None:
    """THE TRIPWIRE THAT FIRED, kept as the reason the arithmetic changed.

    Its predecessor asserted `len(members) <= 1` for every non-cadence
    slot and said in its own docstring what to do the day that stopped
    being true: "the ceiling has to become a slot-min, and this test is
    what says so". Wiring the shelf put a second WIRED member in three of
    them at once, it went red, and the arithmetic moved -- ceiling and
    vote alike (clear_reachability.slot_ceiling,
    StreamingConfluenceDetector._fold). This is the same fact asserted
    from the other side: if a future change un-wires them back to one
    member each, this goes red in turn and the collapse gets re-argued
    rather than inherited as cargo.

    S_CADENCE is exempt and stays exempt deliberately: {tally, apu} each
    cast a full vote today, and collapsing them would silently change what
    `min_signals: 3, apu_weight: 1.0` means with nothing measured to
    justify it."""
    q = clear_quorum(_profile(clear={"mode": "confluence", "apu_weight": 1.0}),
                     roster=roster)
    wired_per_slot: dict[str, list[str]] = {k: [] for k in SLOTS}
    for name, st in q.signal_state.items():
        if st.state != NOT_WIRED:
            wired_per_slot[st.slot].append(name)
    for slot in ("S_TRANSITION", "S_DESPAWN"):
        assert len(wired_per_slot[slot]) >= 2, (
            f"{slot} has {wired_per_slot[slot]}: with one member the slot "
            "collapse is a no-op and should be re-argued, not inherited")
        assert slot in COLLAPSING_SLOTS
    # S_ARMING gains its second member only on the OFFLINE roster: the
    # live detector has no env handle for a probe, so `lock` is not one of
    # its signals (StreamingConfluenceDetector's own "Availability note").
    if roster == OFFLINE:
        assert sorted(wired_per_slot["S_ARMING"]) == ["input_lock", "lock"]
    assert "S_ARMING" in COLLAPSING_SLOTS
    assert "S_CADENCE" not in COLLAPSING_SLOTS


def test_correlated_evidence_in_one_slot_casts_one_vote_not_two() -> None:
    """RULE 3, as arithmetic, on the ceiling. OamQuiesceSignal's own
    docstring: "a caller wiring both into a vote MUST treat {entity_wipe,
    oam_quiesce} as ONE corroborating slot, never as two independent
    votes -- counting them separately double-counts a single piece of
    evidence." Its false-positive set is a strict SUPERSET of
    entity_wipe's, so the two agreeing is one observation, not two.

    Revert slot_ceiling to a plain sum and `both` reads 4.0."""
    one = clear_quorum(_profile(clear={
        "mode": "confluence",
        "signals": {"entity_wipe": {"min_bytes": 12}}}))
    both = clear_quorum(_profile(clear={
        "mode": "confluence",
        "signals": {"entity_wipe": {"min_bytes": 12}, "oam_quiesce": {}}}))
    assert one.ceiling == 3.0            # coord + tally + S_DESPAWN
    assert both.ceiling == 3.0           # ...and the second member adds 0
    assert sorted(both.slots["S_DESPAWN"]) == ["entity_wipe", "oam_quiesce"]


def test_three_views_of_one_screen_wipe_cannot_be_a_three_signal_confluence() -> None:
    """The same rule where it bites hardest: coord, scene_cut and
    room_fp_transition all answer "a scene committed", and a stage wipe
    moves all three at once. Summed, that reaches min_signals: 3 on its
    own with no corroboration of any kind."""
    q = clear_quorum(_profile(clear={
        "mode": "confluence", "min_signals": 3,
        "signals": {"scene_cut": {"scene_min": 1, "blank_min": 1},
                    "room_fp_transition": {"mask": [[0, 960]]}}}))
    assert sorted(q.slots["S_TRANSITION"]) == [
        "coord", "room_fp_transition", "scene_cut"]
    assert q.ceiling == 2.0, "S_TRANSITION casts one vote, tally the other"
    assert q.verdict == UNREACHABLE, (
        "and a bar of 3 is therefore honestly out of reach, rather than "
        "met by one screen wipe counted three times")


# ==========================================================================
# 6. The denominator.
# ==========================================================================

def test_a_reachable_profile_still_produces_a_real_hit_rate() -> None:
    """THE OVER-CORRECTION GUARD, and the reason the refusal above is worth
    anything: a fold that returned None for everything would pass every
    denominator test on this page and measure nothing."""
    prof = _profile()
    rep = summarize_runs([
        {"profile": prof, "within_tolerance": True,
         "false_positive_crossings": 0},
        {"profile": prof, "within_tolerance": False,
         "false_positive_crossings": 0},
    ])
    assert rep["n_valid"] == 2
    assert rep["n_unreachable"] == 0
    assert rep["hit_rate"] == 0.5
    assert rep["hit_rate_pass"] is False       # a real FAIL, not a VOID
    assert rep["false_positive_gate_pass"] is True
    assert [r["verdict"] for r in rep["per_run"]] == [CLEAR, NO_CLEAR]


def test_an_errored_row_is_not_a_miss_either() -> None:
    prof = _profile()
    rep = summarize_runs([
        {"profile": prof, "error": "missing solution files"},
        {"profile": prof, "within_tolerance": True,
         "false_positive_crossings": 0},
    ])
    assert rep["n_error"] == 1 and rep["n_valid"] == 1
    assert rep["hit_rate"] == 1.0
    assert rep["per_run"][0]["verdict"] == ERROR


def test_a_replay_shorter_than_the_warmup_is_not_a_verdict() -> None:
    """The counterfactual gate already proved a too-short replay's silence
    is a structural no-op — 52-observation branches against the 100 the
    `min_signals: 3, apu_weight: 1.0` configuration needs — but the number
    had no name in any output."""
    prof = _profile()
    rep = summarize_runs([{"profile": prof, "within_tolerance": False,
                           "n_observations": 52,
                           "warmup_observations": 100}])
    assert rep["per_run"][0]["verdict"] == UNDER_WARMUP
    assert rep["n_under_warmup"] == 1
    assert rep["n_valid"] == 0
    assert rep["hit_rate"] is None


def test_the_false_positive_gate_does_not_pass_vacuously() -> None:
    """`all([])` is True. A gate over zero measurable rows reporting PASS is
    the same defect as a hit rate of 0.0 over zero measurable rows, and two
    vacuous gates shipped this week."""
    rep = summarize_runs([{"profile": _profile(progress={"lo": 0x0401}),
                           "within_tolerance": False,
                           "false_positive_crossings": 0}])
    assert rep["false_positive_gate_pass"] is None


def test_the_receipt_carries_the_eligibility_table() -> None:
    """Enforcement point 4: every receipt writer carries the verdict AND
    the reasons, because the census's failure mode was a number travelling
    without its meaning."""
    rep = summarize_runs([{"profile": _profile(progress={"lo": 0x0401}),
                           "within_tolerance": False}])
    q = rep["clear_quorum"]
    assert q["verdict"] == UNREACHABLE
    assert q["signals"]["coord"]["state"] == DEAD
    assert "255" in q["signals"]["coord"]["reason"]
    json.dumps(rep)          # a receipt has to survive being written down


# ==========================================================================
# 7. Enforcement point 3 — the table on stdout at every launch.
# ==========================================================================

def test_the_launch_banner_prints_the_per_signal_table() -> None:
    """The Gradius failure was not that the 2026-08-24 progress swap was
    undetectable. Nothing ever printed "coord is DEAD, ceiling 1 of 2"
    while 54+ minutes a day burned."""
    banner = launch_banner(
        _profile(progress={"lo": 0x0070}), "configs/example.yaml")
    assert banner is not None
    assert "configs/example.yaml" in banner
    assert "UNREACHABLE" in banner
    assert "coord" in banner and DEAD in banner
    assert "ceiling=1" in banner


def test_the_banner_prints_the_table_even_when_the_hook_is_fine() -> None:
    """A tripwire that only prints on the bad day is a tripwire nobody has
    ever seen work. The ceiling is a number on stdout at EVERY launch."""
    banner = launch_banner(_profile(), "configs/contra.yaml")
    assert banner is not None
    assert "FIREABLE" in banner
    assert "ceiling=2" in banner


# ==========================================================================
# 8. Anti-vacuity: the verdict is not a constant, in either direction.
# ==========================================================================

def test_the_verdict_is_not_a_constant() -> None:
    alive = _profile()
    assert clear_quorum(alive).verdict == FIREABLE
    assert StreamingConfluenceDetector.from_profile(alive, len) is not None
    assert clear_quorum(_profile(progress={"source": "odometer"})).verdict \
        == UNREACHABLE
    assert clear_quorum(alive, null_rates={"tally": 1.0}).verdict \
        == UNREACHABLE


@pytest.mark.parametrize("mutation,expected", [
    ({"progress": {"source": "odometer"}}, UNREACHABLE),
    ({"progress": {"source": "fight_gate"}}, UNREACHABLE),
    ({"progress": {"lo": 0x40}}, UNREACHABLE),
    ({"progress": {"lo": 0x40, "hi": 0x41}}, FIREABLE),
    ({"progress": {"tiles": [0x324, 0x325]}}, FIREABLE),
    ({"null_rates": {"tally": 1.0, "coord": 1.0}}, UNREACHABLE),
    ({"null_rates": {"coord": 1.0}}, UNREACHABLE),
    ({"clear": {"mode": "confluence", "min_signals": 3}}, UNREACHABLE),
])
def test_breaking_one_thing_moves_the_quorum_verdict(mutation, expected) -> None:
    """Break exactly one thing about an accepted profile and require the
    verdict to move. Delete the odometer rule and rows 1 and 6 go red;
    hardcode UNREACHABLE and rows 4, 5 go red. The function is provably
    not a constant in either direction."""
    assert clear_quorum(_profile()).verdict == FIREABLE
    assert clear_quorum(_profile(**mutation)).verdict == expected


# ==========================================================================
# 9. The control this was built from: same tapes, same detector, one key.
# ==========================================================================

#: Checked into docs/receipts/ ON PURPOSE. This lived under runs/, which is
#: gitignored, so the one end-to-end control tying the adjudicator to real
#: banked tapes SKIPPED on any fresh checkout — and a skip is not a pass.
#: Every other test in this module runs on synthetic profile dicts; this is
#: the only one whose input is a measurement.
CONTROL = REPO / "docs" / "receipts" / "clear_control" / "cv_odometer_swap_2026-08-26.json"


def test_the_control_receipt_is_checked_in() -> None:
    """Fail, never skip. If the receipt goes missing the control below
    would silently stop running, which is how the odometer arm's 0/3 got
    filed as a miss in the first place."""
    assert CONTROL.exists(), (
        f"{CONTROL.relative_to(REPO)} is missing — the clear-quorum "
        f"control has no input and this module is down to synthetic dicts")



def test_the_cv_odometer_swap_control_now_reports_unreachable_not_a_miss() -> None:
    """The banked receipt, replayed through the adjudicator rather than
    through the emulator. Its own numbers are the specification: under the
    RAM pair the detector hit 3/3 with coord firing 4 checks out of 22/28/43
    and a largest single-step drop of 592; under the odometer it hit 0/3
    with coord firing on 0 checks and a largest drop of 4, against the >= 300
    it requires.

    Before this change both arms produced the same shape of output — a
    number — and only the value differed, which is how a 0/3 on an
    instrument that could not have scored got filed next to a 3/3 on one
    that could. Now the arms produce different KINDS of answer."""
    rows = json.loads(CONTROL.read_text())
    prof = yaml.safe_load((REPO / "configs" / "castlevania.yaml").read_text())

    odo = dict(prof)
    odo["solve"] = dict(prof["solve"])
    odo["solve"]["progress"] = {"source": "odometer", "axis": "x"}

    ram_arm = [r for r in rows if r["source"] == "ram_pair"]
    odo_arm = [r for r in rows if r["source"] == "odometer"]
    assert len(ram_arm) == len(odo_arm) == 3

    # The receipt's own evidence that the two arms are not comparable.
    assert all(r["hit"] for r in ram_arm)
    assert not any(r["hit"] for r in odo_arm)
    assert all(r["coord_checks"] > 0 for r in ram_arm)
    assert all(r["coord_checks"] == 0 for r in odo_arm)
    assert max(r["max_single_step_drop"] for r in odo_arm) < 300

    # The verdict each arm gets today.
    assert clear_quorum(prof).verdict == FIREABLE
    q_odo = clear_quorum(odo)
    assert q_odo.verdict == UNREACHABLE
    assert "odo_fold_frame" in q_odo.signal_state["coord"].reason

    # And the detector that produced the 0/3 can no longer be built.
    StreamingConfluenceDetector.from_profile(prof, lambda r: 0)
    with pytest.raises(UnfireableHook):
        StreamingConfluenceDetector.from_profile(odo, lambda r: 0)

    # The 0/3 itself, refolded: three rows, none of them a miss.
    rep = summarize_runs([{"profile": odo, "within_tolerance": r["hit"]}
                          for r in odo_arm])
    assert rep["n_unreachable"] == 3 and rep["n_valid"] == 0
    assert rep["hit_rate"] is None
    # ...against the arm that really did measure something.
    rep_ram = summarize_runs([{"profile": prof, "within_tolerance": r["hit"]}
                              for r in ram_arm])
    assert rep_ram["n_valid"] == 3 and rep_ram["hit_rate"] == 1.0


# ==========================================================================
# 10. The shipped corpus.
# ==========================================================================

def _confluence_profiles() -> list[Path]:
    out = []
    for p in sorted((REPO / "configs").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:
            continue
        solve = d.get("solve") if isinstance(d, dict) else None
        clear = (solve or {}).get("clear") if isinstance(solve, dict) else None
        if isinstance(clear, dict) and clear.get("mode") == "confluence":
            out.append(p)
    return out


@pytest.mark.parametrize("path", _confluence_profiles(), ids=lambda p: p.name)
def test_every_shipped_confluence_profile_can_reach_its_own_quorum(path) -> None:
    """The regression bar. A profile may honestly have no clear predicate;
    it may not ADVERTISE a confluence vote whose eligible signals cannot
    reach the bar. Gradius shipped exactly that from 2026-08-24 to
    2026-08-26 and nobody noticed for eighteen days."""
    prof = yaml.safe_load(path.read_text())
    q = clear_quorum(prof)
    assert q.verdict == FIREABLE, f"{path.name}: {q.reason}"
    StreamingConfluenceDetector.from_profile(prof, lambda r: 0)


def test_the_confluence_corpus_is_not_empty() -> None:
    """Anti-vacuity on the sweep above: a parametrized test over zero
    profiles passes and asserts nothing."""
    assert len(_confluence_profiles()) >= 2


# ==========================================================================
# Stream fixtures (the RAM shapes the two live signals vote on).
# ==========================================================================

A_GX_LO, A_GX_HI, A_LIVES, A_Y, A_ROOM = 0x03E, 0x03F, 0x032, 0x002, 0x075
A_TIMER, A_SCORE = 0x07F, 0x080
ENTITY_LO, ENTITY_HI = 0x100, 0x110


def _ram(gx: int, lives: int = 3, room: int = 1, timer: int = 200,
         score: int = 0, entities: bool = True, y: int = 100) -> bytes:
    buf = bytearray(2048)
    buf[A_GX_LO] = gx & 0xFF
    buf[A_GX_HI] = (gx >> 8) & 0xFF
    buf[A_LIVES], buf[A_Y], buf[A_ROOM] = lives, y, room
    buf[A_TIMER], buf[A_SCORE] = timer, score
    for a in range(ENTITY_LO, ENTITY_HI):
        buf[a] = 9 if entities else 0
    return bytes(buf)


def _progress(ram) -> int:
    return int(ram[A_GX_LO]) | int(ram[A_GX_HI]) << 8


def _load_stream(n: int = 400, at: int = 200) -> list[bytes]:
    """A position reset toward a level start plus an entity slot wipe: the
    RAM shape both live signals vote on. Reused verbatim from
    tests/test_detector_v3.py so the two suites cannot drift about what a
    load looks like."""
    return [_ram(gx=(600 + 2 * i if i < at else 20 + 2 * (i - at)),
                 entities=i < at, timer=250 - i // 8, score=i // 8)
            for i in range(n)]


def test_the_fixture_stream_really_does_produce_both_votes() -> None:
    """The fixtures above are load-bearing for four behavioural tests; a
    stream that silently stopped producing a coord signature would make all
    of them pass for the wrong reason."""
    hist = np.stack([np.frombuffer(r, dtype=np.uint8) for r in _load_stream()])
    gx = np.array([_progress(r) for r in _load_stream()], dtype=np.int64)
    assert clear_detect.score_tally_windows(hist)
    assert clear_detect.coord_entity_windows(hist, gx)
