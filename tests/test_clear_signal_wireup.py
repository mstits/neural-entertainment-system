"""The six signals reach the vote, and each one can be made to redden a test.

WHAT WENT WRONG, so the tests can be aimed at it. entity_wipe,
room_fp_transition, input_lock, lock_release_novelty, oam_quiesce and
scene_cut were built, tested and receipted on 2026-08-26, and reached no
production path at all: the live vote was `tally + coord >= min_signals`
and the offline harness weighted only audio+tally+lock+coord. Six working
signals, zero of them able to change any verdict, while the roster closed
4 CONFIRMED / 41 VOID / 0 FAIL for want of an instrument that could return
a positive.

THE QUESTION EVERY TEST HERE IS AIMED AT is the one the whole week's
defects share: WHAT WOULD THIS REPORT IF THE MECHANISM WERE ABSENT? Answered
leg by leg, and every leg names the mutation that reddens it:

  * the arming legs fail if `signal_config` returns None (nothing ever
    arms) or a dict (everything is always armed) -- the vacuous-gate shape
    in both directions;
  * the fold legs fail if StreamingConfluenceDetector._fold sums a
    collapsing slot instead of taking its max: two views of one screen wipe
    become a two-signal confluence;
  * the Rule-5 legs fail if the required-class term is removed: corroborators
    carry a clear on their own, which is the measured frame-320 false
    positive;
  * the death-discriminant legs fail if RoomFpTransitionSignal stops reading
    lives: a Zelda death fade and a dungeon entry are the same event to
    every other signal in the file;
  * and the byte-identity leg fails if an unarmed profile's detector stops
    taking the shipped integer path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from clear_reachability import (  # noqa: E402
    ALIVE,
    DEAD,
    FIREABLE,
    LIVE,
    OFFLINE,
    SHELF_WEIGHT,
    UNREACHABLE,
    clear_quorum,
    signal_config,
    slot_ceiling,
)

clear_detect = pytest.importorskip(
    "clear_detect", reason="needs the compiled nes_core extension")
StreamingConfluenceDetector = clear_detect.StreamingConfluenceDetector
ShelfSignalConfigError = clear_detect.ShelfSignalConfigError
build_shelf_signals = clear_detect.build_shelf_signals
RoomFpTransitionSignal = clear_detect.RoomFpTransitionSignal


# ==========================================================================
# Fixtures: the RAM shapes the two shipped live signals vote on, reused
# verbatim from tests/test_clear_quorum_reachability.py so the suites
# cannot drift about what a load looks like.
# ==========================================================================

A_GX_LO, A_GX_HI, A_LIVES, A_Y, A_ROOM = 0x03E, 0x03F, 0x032, 0x002, 0x075
A_TIMER, A_SCORE = 0x07F, 0x080
ENTITY_LO, ENTITY_HI = 0x100, 0x110


def _ram(gx: int, lives: int = 3, timer: int = 200, score: int = 0,
         entities: bool = True) -> bytes:
    buf = bytearray(2048)
    buf[A_GX_LO] = gx & 0xFF
    buf[A_GX_HI] = (gx >> 8) & 0xFF
    buf[A_LIVES], buf[A_Y], buf[A_ROOM] = lives, 100, 1
    buf[A_TIMER], buf[A_SCORE] = timer, score
    for a in range(ENTITY_LO, ENTITY_HI):
        buf[a] = 9 if entities else 0
    return bytes(buf)


def _progress(ram) -> int:
    return int(ram[A_GX_LO]) | int(ram[A_GX_HI]) << 8


def _flat_stream(n: int = 400) -> list[bytes]:
    """Ordinary forward play: no coord signature, entities never wiped."""
    return [_ram(gx=600 + 2 * i, timer=250 - i // 8, score=i // 8)
            for i in range(n)]


def _profile(**clear) -> dict:
    base = {"mode": "confluence"}
    base.update(clear)
    return {"solve": {"rom": "roms/none.nes",
                      "progress": {"lo": 0x03E, "hi": 0x03F},
                      "y": A_Y, "lives": A_LIVES, "level_key": [],
                      "clear": base}}


# ==========================================================================
# 1. Wired means the vote can see it. Armed means this profile turned it on.
# ==========================================================================

def test_an_unarmed_profile_builds_no_shelf_signal_at_all() -> None:
    """BYTE-IDENTITY, at the construction boundary. Nothing in the shipped
    corpus declares `clear.signals`, so nothing may be constructed for it —
    a detector that built six silent signal objects would still change the
    warm-up budget, the reset() semantics and the receipt shape."""
    prof = _profile()
    assert build_shelf_signals(prof, clear_quorum(prof)) == {}
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    assert det.shelf_stats() == {}
    assert det.warmup_observations() == 20, "the shipped number, unmoved"


def test_signal_config_is_not_a_constant_in_either_direction() -> None:
    """The vacuous-gate guard on the arming reader itself. A version
    hardcoded to None fails the second leg; one hardcoded to a dict fails
    the first and third."""
    assert signal_config(_profile(), "scene_cut") is None
    armed = _profile(signals={"scene_cut": {"scene_min": 2, "blank_min": 1}})
    assert signal_config(armed, "scene_cut") == {"scene_min": 2, "blank_min": 1}
    off = _profile(signals={"scene_cut": {"enabled": False}})
    assert signal_config(off, "scene_cut") is None


def test_a_misspelled_knob_is_refused_rather_than_ignored() -> None:
    """A knob that silently does nothing is the vacuous pattern in
    miniature: the profile reads as configured, the receipt reads as
    calibrated, and the number that came out was the default nobody
    chose."""
    prof = _profile(signals={"scene_cut": {"scene_min": 1, "blank_min": 1,
                                            "scene_minimum": 4}})
    with pytest.raises(ShelfSignalConfigError) as exc:
        build_shelf_signals(prof, clear_quorum(prof))
    assert "scene_minimum" in str(exc.value)


@pytest.mark.parametrize("name,cfg", [
    ("scene_cut", {"scene_min": 1, "blank_min": 1}),
    ("oam_quiesce", {}),
    ("room_fp_transition", {"mask": [[0, 960]]}),
    ("input_lock", {}),
    ("lock_release_novelty", {"lock_max": 60, "m": 30}),
])
def test_each_armed_signal_is_constructed_and_reaches_the_detector(name, cfg) -> None:
    """Five inputs, five different detectors. Delete the `shelf=` argument
    from from_profile and every row goes red."""
    prof = _profile(signals={name: cfg})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    assert name in det.shelf_stats(), name
    assert clear_quorum(prof).signal_state[name].state == ALIVE


def test_entity_wipe_reaches_the_detector_as_a_window_scan() -> None:
    """entity_wipe is a function over the rolling RAM buffer, not an object
    with state, so it rides as kwargs rather than as a signal instance --
    and it still has to show up in the receipt, or "armed" and "counted"
    could quietly differ."""
    prof = _profile(signals={"entity_wipe": {"min_bytes": 8}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    assert det.shelf_stats()["entity_wipe"]["kwargs"] == {"min_bytes": 8}


# ==========================================================================
# 2. RULE 3 — a slot casts one vote, in the ARITHMETIC and not just the doc.
# ==========================================================================

def test_two_despawn_signals_agreeing_cast_one_vote_not_two() -> None:
    """OamQuiesceSignal's own docstring: its false-positive set is a strict
    SUPERSET of entity_wipe's, so "a caller wiring both into a vote MUST
    treat {entity_wipe, oam_quiesce} as ONE corroborating slot".

    Behavioural, on the real fold. Change `max` to `sum` in
    StreamingConfluenceDetector._fold and this reads 2.0."""
    prof = _profile(signals={"entity_wipe": {"min_bytes": 8},
                             "oam_quiesce": {}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    assert det._fold({"entity_wipe": 1}) == 1.0
    assert det._fold({"entity_wipe": 1, "oam_quiesce": 1}) == 1.0
    # ...and two DIFFERENT slots still add, or the collapse would be a mute.
    assert det._fold({"entity_wipe": 1, "tally": 1}) == 2.0


def test_the_fold_and_the_ceiling_agree_on_every_armed_combination() -> None:
    """A ceiling folded differently from the vote it bounds is not an upper
    bound, it is a second opinion — and the whole refusal machinery rests
    on it being an upper bound. Asserted against ALL signals firing at
    once, which is exactly what the ceiling claims to bound."""
    prof = _profile(signals={"entity_wipe": {"min_bytes": 8},
                             "oam_quiesce": {},
                             "scene_cut": {"scene_min": 1, "blank_min": 1},
                             "input_lock": {},
                             "lock_release_novelty": {"lock_max": 60, "m": 30}})
    q = clear_quorum(prof)
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    everything = {n: 1 for n in det.VOTING_SIGNALS}
    assert det._fold(everything) == q.ceiling
    assert q.ceiling == slot_ceiling(q.signal_state.values())


# ==========================================================================
# 3. RULE 5 — the six do not all unlock the required class.
# ==========================================================================

@pytest.mark.parametrize("name,cfg,is_transition", [
    ("scene_cut", {"scene_min": 1, "blank_min": 1}, True),
    ("room_fp_transition", {"mask": [[0, 960]]}, True),
    ("lock_release_novelty", {"lock_max": 60, "m": 30}, True),
    ("oam_quiesce", {}, False),
    ("entity_wipe", {"min_bytes": 8}, False),
    ("input_lock", {}, False),
])
def test_only_the_transition_class_can_satisfy_rule_five(
        name, cfg, is_transition) -> None:
    """Six inputs, two answers, and which is which is the load-bearing
    claim. "Something emptied" and "input stopped responding" are what a
    DEATH looks like; only "a scene committed" / "the world did not come
    back" is evidence a level ended. Flip any row and the shape that
    produced the frame-320 false positive becomes reachable again."""
    prof = _profile(signals={name: cfg})
    st = clear_quorum(prof).signal_state[name]
    assert st.transition_evidence is is_transition, name
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    assert (name in det._required) is is_transition


def test_corroborators_alone_still_cannot_carry_a_clear_when_armed() -> None:
    """The frame-320 shape, rebuilt out of the NEW signals. A despawn plus
    a cadence vote sums to the bar; with no transition evidence anywhere it
    must not fire. Delete the required-class term in push() and this goes
    red."""
    prof = _profile(min_signals=2, signals={"oam_quiesce": {}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    # tally fires on this stream from the first check; oam is forced high.
    det._shelf["oam_quiesce"] = _AlwaysOn()
    fired = None
    for i, ram in enumerate(_flat_stream()):
        if det.push(ram):
            fired = i
            break
    assert fired is None, "despawn + cadence is not transition evidence"
    assert det.n_checks > 0, "the detector really ran"
    assert det.n_required_class_vetoes > 0, (
        "and it was the required class that stopped it, not a stream that "
        "never produced a vote")


def test_the_same_stream_fires_once_a_transition_signal_is_armed() -> None:
    """THE OVER-CORRECTION GUARD. A detector that never fires is not a fix,
    it is the 41-VOID roster with extra steps. Same stream, same bar, one
    transition-class signal added."""
    prof = _profile(min_signals=2,
                    signals={"oam_quiesce": {},
                             "scene_cut": {"scene_min": 1, "blank_min": 1}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    det._shelf["oam_quiesce"] = _AlwaysOn()
    det._shelf["scene_cut"] = _AlwaysOn()
    fired = None
    for i, ram in enumerate(_flat_stream()):
        if det.push(ram):
            fired = i
            break
    assert fired is not None


class _AlwaysOn:
    """A stand-in whose vote is 1 from the first observation — isolates the
    FOLD from the question of whether a particular signal's own detector
    fires, which its own test module already pins."""

    def push(self, *a, **k) -> None:
        pass

    def vote(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def stats(self) -> dict:
        return {"always_on": True}


# ==========================================================================
# 4. The modalities: absent is silent, never a fabricated zero.
# ==========================================================================

def test_an_armed_signal_whose_modality_is_never_supplied_votes_nothing() -> None:
    """The direction that matters. A caller that armed oam_quiesce and
    forgot to plumb `oam=` must get a STRICTER detector and a receipt
    saying `n: 0`, never the loudest possible collapse manufactured out of
    an absent measurement."""
    prof = _profile(signals={"oam_quiesce": {}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    for ram in _flat_stream(200):
        det.push(ram)
    assert det.shelf_stats()["oam_quiesce"]["n"] == 0
    assert det.shelf_stats()["oam_quiesce"]["votes"] == 0


def test_supplying_the_modality_makes_the_same_signal_accumulate() -> None:
    """Anti-vacuity on the test above: a signal that could never
    accumulate would pass it for the wrong reason."""
    prof = _profile(signals={"oam_quiesce": {}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    for ram in _flat_stream(200):
        det.push(ram, oam=(20, 8))
    assert det.shelf_stats()["oam_quiesce"]["n"] == 200


def test_an_input_lock_verdict_is_held_until_the_next_one() -> None:
    """The streaming form of InputLockTrack.vote_at: the offline harness
    holds each probe's verdict until the next probe, and the live form has
    to mean the same thing or a receipt cannot compare them."""
    prof = _profile(signals={"input_lock": {}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    lock = det._shelf["input_lock"]
    stream = _flat_stream(60)
    det.push(stream[0], locked=True)
    assert lock.vote() == 1
    for ram in stream[1:20]:
        det.push(ram)
    assert lock.vote() == 1, "held across observations that supply nothing"
    det.push(stream[20], locked=False)
    assert lock.vote() == 0, "and a supplied UNLOCKED verdict drops it"
    assert lock.stats()["n_supplied"] == 2, (
        "only the two real verdicts count as measurements; the 19 silent "
        "observations are not 19 probes reading UNLOCKED")


def test_a_staleness_cap_is_opt_in_and_off_by_default() -> None:
    """`hold` caps how long a stale LOCKED verdict is trusted. It is 0 by
    default because the right cap is the caller's own probe stride, and
    nobody has measured a global one — the COORD_RESET_DROP_MIN rule
    applied to a knob nobody would otherwise notice."""
    prof = _profile(signals={"input_lock": {"hold": 5}})
    det = StreamingConfluenceDetector.from_profile(prof, _progress)
    lock = det._shelf["input_lock"]
    stream = _flat_stream(40)
    det.push(stream[0], locked=True)
    for ram in stream[1:4]:
        det.push(ram)
    assert lock.vote() == 1
    for ram in stream[4:10]:
        det.push(ram)
    assert lock.vote() == 0, "a verdict older than the cap is not evidence"


# ==========================================================================
# 5. THE DEATH DISCRIMINANT — Zelda as the negative control.
# ==========================================================================

def _nt(seed: int) -> bytes:
    """A distinct 2 KB nametable page per room identity."""
    return bytes((seed * 37 + i) & 0xFF for i in range(2048))


def _settle(sig, page: int, *, scenes=(0, 0, 0), lives=None):
    """Push one page for `len(scenes)` observations, scripting the scene
    ordinal ACROSS the churn window. `classify_transition` reads the
    onset-to-settle DELTA, not the absolute value, so a warp has to be a
    RISE during the churn -- (0, 1, 2) is the measured Zelda death flash
    (odometer modal 16 -> 272 -> 16, flat at settle; scene +2)."""
    for scene in scenes:
        sig.push(_nt(page), (0, 0), scene, lives=lives)


#: The measured Zelda death-flash / dungeon-entry churn: scene +2, odo flat.
WARP = (0, 1, 2)


def test_a_zelda_shaped_death_fade_and_a_room_entry_are_the_same_shape() -> None:
    """THE NEGATIVE CONTROL, stated as a positive assertion so it cannot
    quietly stop being true. `classify_transition` sees delta-scene >= 2
    with the odometer flat and calls it `warp` for BOTH — the measured
    Zelda death flash and a dungeon-entry fade are indistinguishable on
    this surface. Any discrimination has to come from somewhere else."""
    from go_explore_solve import classify_transition
    death = classify_transition((0, 0), 2)
    entry = classify_transition((0, 0), 2)
    assert death[0] == entry[0] == "warp"


def test_the_death_fade_does_not_vote_when_lives_fell() -> None:
    """The discriminant itself. Identical stream, identical classified
    kind, identical novelty — only the lives byte differs. Delete the
    lives comparison in RoomFpTransitionSignal.push and this goes red
    while its twin below stays green, which is exactly the pair a
    non-vacuous guard needs."""
    sig = RoomFpTransitionSignal(kind=("pan", "fade", "warp"))
    _settle(sig, 1, lives=3)                       # baseline room
    _settle(sig, 2, scenes=WARP, lives=2)          # ...and a life was lost
    assert sig.n_settles == 2
    assert sig.last_kind == "warp"
    assert sig.n_death_vetoes == 1
    assert sig.vote() == 0
    assert sig.stats()["lives_seen"] is True


def test_the_identical_transition_votes_when_lives_held() -> None:
    """The over-correction guard: a discriminant that vetoed everything
    would pass the test above and detect nothing."""
    sig = RoomFpTransitionSignal(kind=("pan", "fade", "warp"))
    _settle(sig, 1, lives=3)
    _settle(sig, 2, scenes=WARP, lives=3)
    assert sig.last_kind == "warp"
    assert sig.n_death_vetoes == 0
    assert sig.vote() == 1


def test_the_identity_is_still_interned_across_a_death() -> None:
    """Discovery is not a vote. A room first seen on the way to a death is
    still a room this episode has visited, or the next arrival there would
    read as novel and the veto would have bought a false positive one
    transition later."""
    sig = RoomFpTransitionSignal(kind=("pan", "fade", "warp"))
    _settle(sig, 1, lives=3)
    _settle(sig, 2, scenes=WARP, lives=2)
    assert sig.n_rooms() == 2, "the death room was interned anyway"
    _settle(sig, 1, scenes=(2, 3, 4), lives=2)
    _settle(sig, 2, scenes=(4, 5, 6), lives=2)
    assert sig.last_novel is False, "and coming back is not novel"
    assert sig.vote() == 0


def test_a_caller_that_never_supplies_lives_says_so_rather_than_passing() -> None:
    """A guard that cannot see its input must not report PASS. Four vacuous
    gates shipped the week this was written, and every one of them was a
    check whose subject was absent."""
    sig = RoomFpTransitionSignal(kind=("pan", "fade", "warp"))
    _settle(sig, 1)
    _settle(sig, 2, scenes=WARP)
    assert sig.vote() == 1, "inert, not silently vetoing"
    assert sig.stats()["lives_seen"] is False, (
        "and the receipt says the discriminant never saw its input")


def test_warp_is_still_outside_the_default_kind_set() -> None:
    """The other half of the Zelda guard, unchanged: the death-flash
    signature does not vote at all unless a profile opts `warp` back in."""
    sig = RoomFpTransitionSignal()
    _settle(sig, 1, lives=3)
    _settle(sig, 2, scenes=WARP, lives=3)
    assert sig.last_kind == "warp"
    assert sig.vote() == 0


# ==========================================================================
# 6. The offline roster carries the same three rules.
# ==========================================================================

def test_the_offline_fold_is_the_same_arithmetic_as_the_ceiling() -> None:
    prof = _profile(signals={"scene_cut": {"scene_min": 1, "blank_min": 1},
                             "oam_quiesce": {}})
    q = clear_quorum(prof, roster=OFFLINE)
    n = 4
    ones = {name: np.ones(n, dtype=np.int8) for name in
            ("audio", "tally", "lock", "coord", "scene_cut", "oam_quiesce")}
    weighted, transition = clear_detect.fold_offline_votes(ones, q, n)
    assert float(weighted[0]) == pytest.approx(q.ceiling)
    assert transition.all()


def test_the_offline_fold_refuses_the_frame_320_shape() -> None:
    """runs/clear_control_2026-08-26/bb_offline_r99.json crossed at frame
    320 on {audio: 1, tally: 1, lock: 1, coord: 0} -- three corroborators
    summing to exactly THRESHOLD with zero transition evidence, 1736 frames
    before the true clear at 2056."""
    prof = _profile(signals={"scene_cut": {"scene_min": 1, "blank_min": 1}})
    q = clear_quorum(prof, roster=OFFLINE)
    n = 2
    votes = {"audio": np.ones(n, np.int8), "tally": np.ones(n, np.int8),
             "lock": np.ones(n, np.int8), "coord": np.zeros(n, np.int8),
             "scene_cut": np.zeros(n, np.int8)}
    weighted, transition = clear_detect.fold_offline_votes(votes, q, n)
    assert float(weighted[0]) >= clear_detect.THRESHOLD, "the sum reaches it"
    assert not transition.any(), "and the required class refuses it"


def test_the_shelf_weight_matches_the_roster_it_votes_in() -> None:
    """A shelf signal casts exactly what a shipped signal of the same
    roster casts: a whole vote live, a quarter-vote offline. A per-profile
    weight knob is a dial for manufacturing a fire and deliberately does
    not exist."""
    assert SHELF_WEIGHT[LIVE] == 1.0
    assert SHELF_WEIGHT[OFFLINE] == clear_detect.WEIGHTS["coord"]
    prof = _profile(signals={"scene_cut": {"scene_min": 1, "blank_min": 1}})
    assert clear_quorum(prof).signal_state["scene_cut"].weight == 1.0
    assert clear_quorum(prof, roster=OFFLINE).signal_state[
        "scene_cut"].weight == 0.25


# ==========================================================================
# 7. The banked receipts — the only inputs on this page that are
#    MEASUREMENTS rather than synthetic dicts.
#
# Checked into docs/receipts/ on purpose: runs/ is gitignored, so a control
# that lives there SKIPS on a fresh checkout, and a skip is not a pass.
# ==========================================================================

import json  # noqa: E402

RECEIPTS = REPO / "docs" / "receipts" / "clear_control"
SMB_BEFORE = RECEIPTS / "smb_before_signal_wireup_2026-08-26.json"
SMB_AFTER = RECEIPTS / "smb_after_signal_wireup_2026-08-26.json"
BB_AFTER = RECEIPTS / "bubble_bobble_after_signal_wireup_2026-08-26.json"
TETRIS_AFTER = RECEIPTS / "tetris_b_after_signal_wireup_2026-08-26.json"


@pytest.mark.parametrize("path", [SMB_BEFORE, SMB_AFTER, BB_AFTER,
                                   TETRIS_AFTER],
                         ids=lambda p: p.name)
def test_the_wireup_receipts_are_checked_in(path) -> None:
    """Fail, never skip. If a receipt goes missing the controls below stop
    running silently, which is the shape of every defect this module is
    aimed at."""
    assert path.exists(), f"{path.relative_to(REPO)} is missing"


def _rows(path):
    d = json.loads(path.read_text())
    return d, {r["run"]: r for r in d["per_run"]}


def test_smb_is_byte_identical_across_the_wireup() -> None:
    """GATE (a). SMB scores 5/5 with 0 false positives and its runs are
    banked; the wire-up had to prove that, not assume it. Both receipts are
    real `clear_detect.py --test` runs over the same five traces, and every
    field but wall-clock and the new telemetry is compared.

    SCOPE CORRECTION (2026-08-26 ledger audit). This control was described
    in the wire-up commit as "field-for-field identical either side of the
    change" and that overstates it in two ways worth naming, because the
    whole point of this file is that a control's shape is part of its
    result. (1) Three fields differ in every row -- `armed_signals`,
    `shelf_stats` and `n_required_class_vetoes` are null in the BEFORE
    receipt and populated in the AFTER one; they are in `ignore` below, so
    "identical" always meant "identical modulo the new telemetry".
    (2) The BEFORE receipt already carries those three keys as null
    (generated_at 13:21:38, before the AFTER run at 13:42:58), which means
    it was produced by code that already had the new columns, with the
    signals unwired -- so what this pins is ARMED-vs-UNARMED under one code
    path, not new-code-vs-old-code. That is still the control that matters
    (SMB arms nothing, and the assertions below prove the identity comes
    from that rather than from luck), but it is a narrower statement than
    the commit message made and the narrower one is the true one."""
    before, b_rows = _rows(SMB_BEFORE)
    after, a_rows = _rows(SMB_AFTER)
    assert before["n_hit_within_tolerance"] == after["n_hit_within_tolerance"] == 5
    assert before["hit_rate"] == after["hit_rate"] == 1.0
    assert before["total_false_positive_crossings"] == 0
    assert after["total_false_positive_crossings"] == 0
    assert set(b_rows) == set(a_rows) and len(a_rows) == 5
    ignore = {"wall_s", "armed_signals", "shelf_stats",
              "n_required_class_vetoes"}
    for run, brow in b_rows.items():
        arow = a_rows[run]
        assert {k: v for k, v in brow.items() if k not in ignore} == \
               {k: v for k, v in arow.items() if k not in ignore}, run
    for row in a_rows.values():
        assert row["armed_signals"] == [], (
            "and it stayed identical because SMB arms nothing, not because "
            "the new signals happened to be quiet")
        assert row["n_required_class_vetoes"] == 0, (
            "Rule 5 removed nothing on SMB: coord fires at every crossing")


def test_bubble_bobble_now_fires_on_two_witnessed_clears() -> None:
    """GATE (b), the half that passed. Bubble Bobble round 99-0 -> 99-1 is
    a witnessed clear the detector could not see: its progress observable
    spans 98..99 (largest drop 1) against the >= 300 backward drop `coord`
    requires, so the only transition-evidence signal on the roster was
    arithmetically dead and the harness wrote `n_valid: 2, hit_rate: 0.0`.

    With scene_cut armed at its own measured gate the same two tapes are
    detected at +4 and +18 frames -- and the frame-320 false positive
    (audio + tally + lock == THRESHOLD with coord == 0, 1736 frames early)
    is gone, because three corroborators can no longer carry a clear."""
    d, rows = _rows(BB_AFTER)
    assert d["n_valid"] == 2 and d["n_unreachable"] == 0
    assert d["hit_rate"] == 1.0 and d["hit_rate_pass"] is True
    assert d["total_false_positive_crossings"] == 0
    for row in rows.values():
        assert row["within_tolerance"] is True
        assert row["verdict"] == "CLEAR"
        assert row["armed_signals"] == ["scene_cut"]
        assert row["contributions_at_detect"]["scene_cut"] == 1
        assert row["contributions_at_detect"]["coord"] == 0, (
            "carried by the new transition signal, not by the dead one")
        assert 0 < row["delta_frames"] <= 20


def test_tetris_b_is_now_measured_late_rather_than_measured_blind() -> None:
    """GATE (b), the half that did NOT pass, stated plainly rather than
    dressed up.

    The banked 4,329-action B-type win used to score `hit_rate: 0.0` from
    an instrument with no live transition signal at all. It now DETECTS the
    real transition -- the screen blanking to the SUCCESS curtain -- and
    lands at true_clear + 162, outside the harness's 120-frame tolerance.
    The blank folds themselves are at +151 and +152 (measured, see
    docs/receipts/clear_control/tetris_b_scene_cut_null_2026-08-26.json),
    so no stride or window change brings it inside; the clear PREDICATE
    fires when the line quota reaches 0 and the screen only turns over
    ~2.5 s later.

    TOLERANCE_FRAMES = 120 was calibrated on SMB's fast flagpole cut.
    Widening it until this row turns green would be manufacturing the hit,
    which is the exact class of move this campaign exists to stop, so the
    row is left as a real NO_CLEAR and this test pins the honest shape:
    measured, not blind; late, not silent."""
    d, rows = _rows(TETRIS_AFTER)
    row = next(iter(rows.values()))
    assert d["n_valid"] == 1 and d["n_unreachable"] == 0
    assert row["verdict"] == "NO_CLEAR", "measured, not VOID"
    assert row["within_tolerance"] is False
    assert row["detected_frame"] is not None, "the instrument DID fire"
    assert row["delta_frames"] == 162
    assert row["delta_frames"] > d["tolerance_frames"]
    assert row["contributions_at_detect"]["scene_cut"] == 1
    # ...and Rule 5 is what kept the in-tolerance despawn from carrying it.
    assert row["n_required_class_vetoes"] == 60, (
        "oam_quiesce collapsed at +28, inside tolerance, and was refused: "
        "'the sprites went away' is not 'a scene committed'")
    assert row["shelf_stats"]["oam_quiesce"]["n_triggers"] == 1


# ==========================================================================
# 8. The LIVE path: which hardware surfaces a run actually samples.
#
# The hot loop reads a surface only when some armed signal declared it —
# the same discipline `resolve_apu_sampling` already applied to the APU
# mask, generalized. A `hasattr` per worker per step is not free, and a
# modality sampled for a hook that cannot use it is pure cost.
# ==========================================================================

from go_explore_solve import (  # noqa: E402
    make_game,
    resolve_clear_modalities,
)


def _game(**clear):
    prof = _profile(**clear)
    prof["solve"]["rom"] = "roms/nonexistent.nes"
    return make_game(prof)


def test_a_profile_that_arms_nothing_samples_no_extra_surface() -> None:
    """BYTE-IDENTITY at the hot loop. Every profile in the shipped corpus
    is this one."""
    assert _game().clear_modalities() == frozenset()


@pytest.mark.parametrize("name,cfg,wants", [
    ("oam_quiesce", {}, {"_oam"}),
    ("scene_cut", {"scene_min": 1, "blank_min": 1}, {"_odo"}),
    ("room_fp_transition", {"mask": [[0, 960]]}, {"_odo", "_nt"}),
    ("input_lock", {}, {"_locked"}),
    ("lock_release_novelty", {"lock_max": 60, "m": 30}, {"_locked"}),
    ("entity_wipe", {"min_bytes": 8}, set()),
])
def test_arming_a_signal_asks_for_exactly_the_surface_it_reads(
        name, cfg, wants) -> None:
    """Six inputs, five different answers (entity_wipe reads the RAM the
    hook already has, so it asks for nothing). A `clear_modalities`
    hardcoded to the empty set fails five rows; one hardcoded to
    everything fails the unarmed test above."""
    assert _game(signals={name: cfg}).clear_modalities() == frozenset(wants)


def test_a_non_confluence_hook_asks_for_nothing_even_when_armed() -> None:
    """`clear.signals` says which signals the INSTRUMENT may use on this
    profile; the offline harness fuses them over any trace. A byte_change
    hook reads its own byte and never consults them, so sampling for it
    would be per-worker per-step cost buying nothing.

    This is not hypothetical: configs/bubble_bobble.yaml and
    configs/tetris_b.yaml are exactly this shape."""
    g = _game(mode="byte_change", addr=0x04A1, direction="up",
              signals={"scene_cut": {"scene_min": 1, "blank_min": 1}})
    assert g._conf_shelf_names == frozenset({"scene_cut"})
    assert g.clear_modalities() == frozenset()


class _PoolWithout:
    """A core missing one accessor — an older wheel, which is a real state
    this repo has shipped through more than once."""

    def __init__(self, *missing):
        self._missing = set(missing)

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        return lambda *a, **k: None


def test_a_core_missing_an_accessor_is_reported_rather_than_crashing(capsys) -> None:
    """The degrade path. An AttributeError once per worker per step is not
    an option, and neither is silently sampling nothing and reading the
    resulting quiet as a measurement."""
    g = _game(signals={"room_fp_transition": {"mask": [[0, 960]]}})
    full = resolve_clear_modalities(g, _PoolWithout())
    assert full == frozenset({"_odo", "_nt"})
    partial = resolve_clear_modalities(g, _PoolWithout("peek_nametables"))
    assert partial == frozenset({"_odo"})
    out = capsys.readouterr().out
    assert "peek_nametables" in out and "vote 0" in out


# ==========================================================================
# A ZERO THAT COULD NOT HAVE BEEN ANYTHING ELSE
#
# The 2026-08-26 ledger audit landed one finding against the wire-up:
# Bubble Bobble's headline `total_false_positive_crossings: 0` was scored
# on the SAME two tapes the scene_cut gate was calibrated on. The gate is
# "the smallest integer strictly above the null measured on those tapes",
# so it cannot fire below threshold there -- FP=0 was arithmetic, and the
# receipt reported it in the same field, with the same words, that an
# out-of-sample zero would have used.
#
# That is this project's signature defect wearing its politest face: a
# number that cannot come out any other way, presented as a measurement.
# `clear_detect.calibration_provenance` now computes the overlap from the
# banked calibration receipts and writes it beside the count every time.
# These tests exist so the disclosure is not itself a constant.
# ==========================================================================


def test_the_bubble_bobble_zero_is_disclosed_as_in_sample() -> None:
    """The finding, pinned. Both scored tapes are calibration tapes, the
    receipt says so by name, and the note refuses the stronger reading."""
    d, _ = _rows(BB_AFTER)
    prov = d["calibration_provenance"]
    assert prov["n_scored"] == 2
    assert prov["n_in_sample"] == 2, (
        "both BB tapes were calibrated on AND scored -- if this ever drops "
        "to 0 the zero became a real out-of-sample number and the entry in "
        "CLAIMS.md should be upgraded, not left as-is")
    assert prov["in_sample_by_signal"]["scene_cut"] == sorted(
        r["run"] for r in d["per_run"])
    assert "bubble_bobble_scene_cut_null_2026-08-26.json" in \
        prov["calibration_receipts_consulted"]
    assert "BY CONSTRUCTION" in prov["note"]
    # ...and the number it qualifies is still there, unchanged.
    assert d["total_false_positive_crossings"] == 0


def test_the_disclosure_is_not_a_constant() -> None:
    """ANTI-VACUITY. A block that said "in-sample" on every input would be
    the same defect one level up. Three inputs, three different answers:
    a calibrated tape, an uncalibrated tape on the same profile, and a
    profile with no calibration receipt at all."""
    calibrated = [{"run": "runs/bubble_bobble/r99_fixed/solutions/sol_000"}]
    other_tape = [{"run": "runs/bubble_bobble/chain_day2f/lvl_00_69/"
                          "solutions/sol_000"}]

    hit = clear_detect.calibration_provenance(
        calibrated, "configs/bubble_bobble.yaml")
    assert hit["n_in_sample"] == 1 and "BY CONSTRUCTION" in hit["note"]

    # Same profile, same consulted receipt -- a tape it was never measured
    # on. This is the live test's fixture (chain_day2f), which is exactly
    # why that one carries out-of-sample weight and this one does not.
    miss = clear_detect.calibration_provenance(
        other_tape, "configs/bubble_bobble.yaml")
    assert miss["calibration_receipts_consulted"] == \
        hit["calibration_receipts_consulted"], "the receipt WAS read"
    assert miss["n_in_sample"] == 0
    assert "out-of-sample" in miss["note"]

    # No calibration receipt for the profile at all.
    none = clear_detect.calibration_provenance(calibrated, "configs/nope.yaml")
    assert none["calibration_receipts_consulted"] == []
    assert none["n_in_sample"] == 0


def test_smb_carries_the_disclosure_and_it_comes_out_empty() -> None:
    """The banked negative, on a real receipt rather than a fixture. SMB
    arms nothing and was never calibrated, so its five-trace 0 false
    positives is an out-of-sample zero -- and the receipt now says which
    kind of zero it is instead of leaving a reader to assume."""
    d, _ = _rows(SMB_AFTER)
    prov = d["calibration_provenance"]
    assert prov["n_scored"] == 5
    assert prov["n_in_sample"] == 0
    assert "out-of-sample" in prov["note"]
    assert d["total_false_positive_crossings"] == 0


def test_every_wireup_receipt_carries_the_block() -> None:
    """Absence must be visible. A receipt with no `calibration_provenance`
    is one whose zero is unqualified, which is the state this fixes; the
    BEFORE receipt predates the field and is the one documented exception."""
    for path in (SMB_AFTER, BB_AFTER, TETRIS_AFTER):
        d = json.loads(path.read_text())
        assert "calibration_provenance" in d, path.name
        assert set(d["calibration_provenance"]) >= {
            "in_sample_runs", "n_in_sample", "note",
            "calibration_receipts_consulted"}, path.name
