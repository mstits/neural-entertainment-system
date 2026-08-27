"""The Rygar transition witness, and the clear predicate it refuses to mint.

WHAT IS UNDER TEST, AND WHAT IS NOT
-----------------------------------
Under test: `scripts/transition_witness.py`, a room-transition counter built on
`odo_blank` -- the PPU blank-fold counter that shadows `odometer_scene` on this
profile. It counts transitions, rejects death fades, and says whether the
arriving area is new.

NOT under test, because it does not exist and this suite is the record of why:
a Rygar *clear* predicate. `TestNaiveClearPredicatesAreRefuted` below runs the
two obvious candidates over Rygar's own deepest banked tape and measures how
many wins each would fabricate (55 and 28, on a trajectory whose first-visit
frontier never moved past 4,608 px). `docs/receipts/rygar/
clear_predicate_REFUTED.md` carries the argument.

ANTI-VACUITY
------------
Four vacuous gates shipped in one week, so nothing here is allowed to pass by
describing an absent mechanism.

  * Every guard has a removal test in `TestGuardsAreLoadBearing` that asserts
    the mechanism MISFIRES once the guard is taken out -- with the measured
    misfire count, not a smoke assertion. A guard that could be deleted
    without breaking a test would not be a guard.
  * `TestMechanismAbsent` pins what the class reports when the counter is off
    or frozen: UNAVAILABLE / UNINSTRUMENTED, never a silent zero.
  * The banked streams in `docs/receipts/rygar/transition_streams.json` are
    real per-observation measurements, and `TestAgainstTheEmulator`
    regenerates them from the ROM and asserts byte equality, so the fixture
    cannot drift away from the machine it claims to describe.

Ledger: EXHIBITION. Every measurement here comes from Go-Explore search output
and scripted/random rollouts. No policy was trained for this game and no
honest-protocol evaluation was run, so nothing here may be described with "the
AI learned", "the AI plays", or "the AI beat" (CLAIMS.md).
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.transition_witness import (
    DEATH, IN_BLANK, NOVEL, REVISIT, RYGAR_AREA_KEY_ADDRS,
    RYGAR_DEATH_DEBOUNCE, RYGAR_MIN_BLANK_FRAMES, SPLICE, TRANSITIONS,
    UNAVAILABLE, TransitionWitness, rle, run_stream, rygar_witness, unrle,
)

REPO = Path(__file__).resolve().parent.parent
STREAMS = REPO / "docs/receipts/rygar/transition_streams.json"
TAPE = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"
PROFILE = REPO / "configs/rygar.yaml"

#: Measured on the banked R1 tape (6,018 actions, 6,019 observations).
N_TRANSITIONS = 55
N_NOVEL = 2          # the start area, plus the two the tape ever reaches
N_REVISIT = 53
N_AREAS = 3          # matches RYGAR_CAMPAIGN_2026-08-26.md §6's hand count
#: Measured across 18 rollouts (right / noop / left + 15 seeded random).
N_DEATH_ROLLOUTS = 18
N_DEATH_FADES = 36   # the death fade and the game-over reload, 2 per rollout


@pytest.fixture(scope="module")
def streams() -> dict:
    assert STREAMS.exists(), (
        f"{STREAMS} is missing. It is deliberately tracked rather than left "
        "under gitignored runs/; regenerate with "
        "`python scripts/transition_witness.py bank`.")
    return json.loads(STREAMS.read_text())


@pytest.fixture(scope="module")
def tape_rows(streams) -> list:
    return list(unrle(streams["tape"]["rows"]))


@pytest.fixture(scope="module")
def death_rows(streams) -> list:
    return [(d["policy"], list(unrle(d["rows"]))) for d in streams["deaths"]]


def _summary(witness, rows) -> dict:
    run_stream(witness, rows)
    return witness.summary()


def _events(witness, rows) -> list:
    run_stream(witness, rows)
    return witness.events


def _totals(rows_by_policy, **kw) -> collections.Counter:
    tot = collections.Counter()
    for _, rows in rows_by_policy:
        s = _summary(rygar_witness(**kw), rows)
        for k in ("transitions", "novel", "revisit", "deaths", "short_runs"):
            tot[k] += s[k]
    return tot


class TestTheBankedStreamsAreWhatTheyClaim:
    """The fixture is a measurement, so it gets checked like one."""

    def test_provenance_pins_the_rom_and_the_tape(self, streams):
        p = streams["provenance"]
        assert len(p["rom_sha256"]) == 64
        assert len(p["start_state_sha256"]) == 64
        assert p["tape"].endswith("r1_tape_gx6242.json")
        assert p["frame_skip"] == 4
        assert p["lives_at_start"] == 1
        assert p["area_key_addrs"] == [f"0x{a:04X}"
                                       for a in RYGAR_AREA_KEY_ADDRS]
        assert streams["ledger"] == "EXHIBITION"

    def test_the_tape_stream_reproduces_the_receipt_totals(self, streams,
                                                           tape_rows):
        rec = json.loads(TAPE.read_text())
        inv = rec["invariants_for_the_guard"]
        assert len(tape_rows) == inv["n_actions"] + 1
        assert streams["tape"]["terminal_odometer_x"] == inv[
            "terminal_odometer_x"] == 6242
        # The counter this instrument replaces reads zero across the whole
        # tape: `odometer_scene` is structurally blind on this profile.
        assert streams["tape"]["odometer_scene"] == inv[
            "scene_cuts_total"] == 0
        assert streams["tape"]["terminal_odo_blank"] == 4329, (
            "the RAW counter is 4,329 BLANK FRAMES, not 4,329 transitions -- "
            "it must always be edge-detected into runs before it is counted")

    def test_the_stream_carries_the_55_two_step_lives_blips(self, tape_rows):
        # The receipt's liveness invariant, re-derived from this stream, so
        # the death tests below rest on measured input rather than on a
        # number copied between files.
        hist, run = collections.Counter(), 0
        for row in tape_rows:
            if row[1]:
                run += 1
            elif run:
                hist[run] += 1
                run = 0
        if run:
            hist[run] += 1
        assert dict(hist) == {2: N_TRANSITIONS}, (
            "Rygar's lives byte blips to 0 for exactly 2 observations at "
            "every door; if that changed, RYGAR_DEATH_DEBOUNCE must be "
            "re-derived, not left at 3")

    def test_the_rle_round_trips(self, streams, tape_rows):
        assert rle(tape_rows) == [list(r) for r in streams["tape"]["rows"]]


class TestTheWitnessOnRealRygar:
    """Positives and negatives, both from Rygar's own measurements."""

    def test_it_counts_every_one_of_the_55_transitions(self, tape_rows):
        s = _summary(rygar_witness(), tape_rows)
        assert s["verdict"] == "OK"
        assert s["signal_observed"] is True
        assert s["transitions"] == N_TRANSITIONS
        assert s["deaths"] == 0
        assert s["splices"] == 0
        assert s["open_run"] is False
        # The one sub-threshold run is the 9-blank-frame boot fade.
        assert s["short_runs"] == 1

    def test_it_reads_three_areas_matching_the_hand_count(self, tape_rows):
        s = _summary(rygar_witness(), tape_rows)
        assert len(s["areas"]) == N_AREAS, (
            "RYGAR_CAMPAIGN_2026-08-26.md §6 counted 'at least 3 visually "
            "distinct areas' BY EYE and recorded that no instrument in the "
            "pipeline could count them. This is that instrument, and it "
            "must keep agreeing with the eye.")
        assert s["areas"] == [(0, 14), (0, 29), (3, 16)]

    def test_the_two_step_blip_is_never_read_as_a_death(self, tape_rows):
        # 55 transitions each carry a 2-observation lives-0 blip. If the
        # debounce mistook one for a death the transition would be lost.
        s = _summary(rygar_witness(), tape_rows)
        assert s["deaths"] == 0
        assert s["transitions"] == N_TRANSITIONS

    def test_it_does_not_fire_on_a_real_death(self, death_rows):
        assert len(death_rows) == N_DEATH_ROLLOUTS
        tot = _totals(death_rows)
        assert tot["transitions"] == 0, (
            "18 rollouts that all die, and not one produced a transition")
        assert tot["novel"] == 0
        assert tot["deaths"] == N_DEATH_FADES
        # One boot fade per rollout, below the floor and carrying no death.
        assert tot["short_runs"] == N_DEATH_ROLLOUTS

    def test_every_rollout_actually_died(self, death_rows):
        # Without this the death test above could pass on rollouts that
        # never died -- the exact shape of a vacuous negative control.
        for policy, rows in death_rows:
            longest = run = 0
            for row in rows:
                run = run + 1 if row[1] else 0
                longest = max(longest, run)
            assert longest >= 500, (
                f"{policy} never sustained a death, so it cannot serve as a "
                f"death negative (longest dead run {longest})")

    def test_the_death_fade_and_the_door_do_not_overlap_in_length(
            self, tape_rows, death_rows):
        # The floor is only defensible if the two clusters are separated.
        door = [e["frames"] for e in
                _events(rygar_witness(), tape_rows) if e["verdict"]
                in TRANSITIONS]
        fade = []
        for _, rows in death_rows:
            fade += [e["frames"] for e in _events(rygar_witness(), rows)
                     if e["verdict"] == DEATH]
        assert set(door) == {78, 79}
        assert set(fade) == {14}
        assert max(fade) < RYGAR_MIN_BLANK_FRAMES < min(door)
        assert RYGAR_MIN_BLANK_FRAMES / max(fade) > 2.5
        assert min(door) / RYGAR_MIN_BLANK_FRAMES > 1.9


class TestTheRatchet:
    """The shape that would fool a naive transition count, from the tape."""

    def test_53_of_the_55_transitions_arrive_somewhere_already_seen(
            self, tape_rows):
        s = _summary(rygar_witness(), tape_rows)
        assert s["novel"] == N_NOVEL
        assert s["revisit"] == N_REVISIT
        assert s["novel"] + s["revisit"] == N_TRANSITIONS

    def test_no_transition_after_the_second_is_ever_novel(self, tape_rows):
        ev = [e for e in _events(rygar_witness(), tape_rows)
              if e["verdict"] in TRANSITIONS]
        assert [e["verdict"] for e in ev[:2]] == [NOVEL, NOVEL]
        assert all(e["verdict"] == REVISIT for e in ev[2:])
        # 27 round trips through one door: the areas alternate perfectly.
        areas = [tuple(e["area"]) for e in ev]
        assert areas[0::2] == [(3, 16)] * 28
        assert areas[1::2] == [(0, 29)] * 27

    def test_the_ratchet_is_the_receipts_27_door_cycles(self, tape_rows):
        rec = json.loads(TAPE.read_text())["invariants_for_the_guard"]
        ev = [e for e in _events(rygar_witness(), tape_rows)
              if e["verdict"] in TRANSITIONS]
        # 27 arrivals into the far area == the receipt's 27 zero-gain
        # post-door segments, independently re-derived from the blank
        # counter rather than from the odometer integral.
        far = sum(1 for e in ev if tuple(e["area"]) == (0, 29))
        assert far == rec["post_door_segments_dx_zero"] == 27


class TestMechanismAbsent:
    """What it reports when the mechanism is not there."""

    def test_a_disabled_odometer_reports_unavailable_not_zero(self,
                                                              tape_rows):
        w = rygar_witness(odometer_enabled=False)
        verdicts = run_stream(w, tape_rows)
        assert set(verdicts) == {UNAVAILABLE}
        s = w.summary()
        assert s["verdict"] == "UNAVAILABLE"
        assert s["observations"] == 0
        assert s["transitions"] == 0, (
            "the count is zero, which is exactly why the verdict field has "
            "to carry UNAVAILABLE -- a consumer reading only `transitions` "
            "would conclude Rygar has no doors")

    def test_a_none_reading_is_not_a_zero(self):
        w = rygar_witness()
        assert w.push(None, False, (0, 14)) == UNAVAILABLE
        assert w.summary()["observations"] == 0

    def test_a_frozen_counter_reports_uninstrumented(self):
        w = rygar_witness()
        for _ in range(500):
            w.push(0, False, (0, 14))
        s = w.summary()
        assert s["verdict"] == "UNINSTRUMENTED"
        assert s["signal_observed"] is False
        assert s["transitions"] == 0

    def test_a_moving_counter_with_no_transitions_is_distinguishable(self):
        # The counter moved (a boot fade) and closed below the floor. That
        # IS a measurement of "no transitions", and it must not be confused
        # with the frozen-counter case above.
        w = rygar_witness()
        w.push(0, False, (0, 14))
        for b in (4, 8, 9):
            w.push(b, False, (0, 14))
        w.push(9, False, (0, 14))
        s = w.summary()
        assert s["verdict"] == "OK"
        assert s["signal_observed"] is True
        assert s["transitions"] == 0 and s["short_runs"] == 1

    def test_min_blank_frames_has_no_default(self):
        with pytest.raises(TypeError):
            TransitionWitness(frames_per_step=4)
        with pytest.raises(ValueError):
            TransitionWitness(min_blank_frames=0, frames_per_step=4)


class TestSplices:
    """`odo_blank` rides inside OdoState, so it is per-trajectory and NOT a
    monotone global. Any consumer that assumes monotonicity across a cell
    restore is wrong."""

    def test_a_decrease_is_a_splice_not_a_transition(self):
        w = rygar_witness()
        w.push(273, False, (3, 16))
        assert w.push(90, False, (3, 16)) == SPLICE
        assert w.summary()["transitions"] == 0
        assert w.summary()["splices"] == 1

    def test_a_jump_larger_than_one_step_of_frames_is_a_splice(self):
        # The dangerous direction: restoring a DEEPER cell makes the counter
        # leap forward, which without this guard reads as one enormous
        # blank run and fabricates a transition.
        w = rygar_witness()
        w.push(90, False, (3, 16))
        assert w.push(90 + 4, False, (3, 16)) == IN_BLANK
        w2 = rygar_witness()
        w2.push(90, False, (3, 16))
        assert w2.push(90 + 5, False, (3, 16)) == SPLICE
        w3 = rygar_witness()
        w3.push(90, False, (3, 16))
        assert w3.push(273, False, (3, 16)) == SPLICE
        assert w3.summary()["transitions"] == 0

    def test_a_splice_mid_run_drops_the_run_rather_than_closing_it(self):
        w = rygar_witness()
        w.push(0, False, (3, 16))
        for b in range(4, 60, 4):        # a run building toward the floor
            assert w.push(b, False, (3, 16)) == IN_BLANK
        assert w.push(0, False, (0, 29)) == SPLICE
        w.push(0, False, (0, 29))
        s = w.summary()
        assert s["transitions"] == 0 and s["short_runs"] == 0, (
            "a run whose two ends came from different trajectories must be "
            "discarded, not classified")


class TestGuardsAreLoadBearing:
    """Each guard removed, with the measured cost of removing it.

    A test that still passes with the mechanism deleted proves nothing, so
    each case here asserts the MISFIRE, not the absence of one."""

    def test_removing_the_length_floor_turns_boot_fades_into_transitions(
            self, death_rows):
        tot = _totals(death_rows, min_blank_frames=1)
        assert tot["transitions"] == N_DEATH_ROLLOUTS, (
            "with the floor at 1 the 9-blank-frame boot fade, which carries "
            "no death, is banked as a room transition in every rollout")
        # Calibrated, the same streams give zero.
        assert _totals(death_rows)["transitions"] == 0

    def test_removing_both_death_guards_banks_every_death_fade(
            self, death_rows):
        tot = _totals(death_rows, min_blank_frames=1, death_debounce=10 ** 9)
        assert tot["transitions"] == N_DEATH_ROLLOUTS + N_DEATH_FADES == 54
        assert _totals(death_rows)["transitions"] == 0

    def test_the_debounce_value_is_load_bearing_in_both_directions(
            self, tape_rows):
        # Too low, and every real transition is destroyed: the 2-observation
        # door blip is read as a death.
        s = _summary(rygar_witness(death_debounce=2), tape_rows)
        assert s["transitions"] == 0 and s["deaths"] == N_TRANSITIONS
        assert s["novel"] == 0
        # At the shipped value all 55 survive.
        assert _summary(rygar_witness(), tape_rows)["transitions"] == \
            N_TRANSITIONS
        assert RYGAR_DEATH_DEBOUNCE == 3

    def test_removing_the_novelty_memory_reinstates_the_ratchet(self,
                                                                tape_rows):
        # A witness with no memory of where it has been calls all 55 door
        # crossings new -- which is precisely the naive count this whole
        # instrument exists to avoid.
        w = rygar_witness()
        novel = 0
        for row in tape_rows:
            w.seen.clear()
            if w.push(row[0], bool(row[1]), tuple(row[2:])) == NOVEL:
                novel += 1
        assert novel == N_TRANSITIONS == 55
        assert _summary(rygar_witness(), tape_rows)["novel"] == N_NOVEL == 2

    def test_removing_the_edge_detection_reads_the_raw_counter(self,
                                                               tape_rows):
        # The historical error this instrument was nearly built on: reading
        # `odo_blank` as a count. It is 4,329 on this tape -- 78x the true
        # transition count -- because it counts blank FRAMES.
        assert tape_rows[-1][0] == 4329
        assert 4329 / N_TRANSITIONS > 78


class TestNaiveClearPredicatesAreRefuted:
    """The gating task's answer, in code.

    Rygar cannot recognise winning. These are the two predicates that the
    transition counter makes available, and both fabricate wins on the
    deepest tape this project has ever produced for this game -- a tape
    whose first-visit frontier never moved past 4,608 px."""

    def test_a_transition_happened_would_bank_55_fabricated_wins(
            self, tape_rows):
        fired = sum(1 for v in run_stream(rygar_witness(), tape_rows)
                    if v in TRANSITIONS)
        assert fired == N_TRANSITIONS == 55

    def test_a_level_key_advance_on_the_area_byte_would_bank_28(
            self, tape_rows):
        # `GenericGame.is_clear` fires on `level_key(ram) > start_key`. Wire
        # the discovered area byte as a level_key and this is what happens.
        start = tuple(tape_rows[0][2:])
        keys, prev = [], None
        for row in tape_rows:
            k = tuple(row[2:])
            if k != prev:
                keys.append(k)
                prev = k
        forward = sum(1 for a, b in zip(keys, keys[1:]) if b > a)
        assert len(keys) - 1 == N_TRANSITIONS
        assert forward == 28
        # Same count for a one-byte `level_key: [0x14]`, which is the shape a
        # profile would actually carry.
        one = [k[0] for k in keys]
        assert sum(1 for a, b in zip(one, one[1:]) if b > a) == 28
        # And the predicate *latches*, so a solver checking it every
        # observation banks on most of the tape either way.
        held = sum(1 for row in tape_rows if tuple(row[2:]) > start)
        held_one = sum(1 for row in tape_rows if (row[2],) > (start[0],))
        assert (held, held_one) == (4978, 3725)
        assert held / len(tape_rows) > 0.8
        assert held_one / len(tape_rows) > 0.6

    def test_the_witness_is_two_orders_off_from_either_naive_count(
            self, tape_rows):
        s = _summary(rygar_witness(), tape_rows)
        assert s["novel"] == N_NOVEL
        assert N_TRANSITIONS / s["novel"] > 27
        # 2 novel areas over 6,018 actions is not a clear either. It is the
        # honest count of new ground, and it is why no predicate is wired.

    def test_the_profile_still_carries_no_clear_predicate(self):
        solve = yaml.safe_load(PROFILE.read_text())["solve"]
        assert solve.get("level_key") == [], (
            "configs/rygar.yaml grew a level_key. Read "
            "docs/receipts/rygar/clear_predicate_REFUTED.md before wiring "
            "one: on the banked tape the discovered area byte advances 28 "
            "times without the frontier moving.")
        assert "clear" not in solve and "finale" not in solve

    def test_the_refutation_receipt_is_tracked_and_says_declined(self):
        doc = REPO / "docs/receipts/rygar/clear_predicate_REFUTED.md"
        assert doc.exists()
        text = doc.read_text()
        for needed in ("REFUSED", "EXHIBITION", "4,608", "55", "28"):
            assert needed in text, f"the receipt must state {needed!r}"


class TestAgainstTheEmulator:
    """Regenerates the banked streams from the ROM. Skips when the ROM is
    absent -- `roms/` is gitignored and not distributable -- but every class
    above runs on any checkout, so a missing ROM cannot silence this file."""

    @staticmethod
    def _prof():
        prof = yaml.safe_load(PROFILE.read_text())
        rom = REPO / prof["solve"]["rom"]
        start = REPO / prof["start_state_path"]
        if not rom.exists() or not start.exists():
            pytest.skip(f"ROM or start state absent ({rom.name})")
        return prof, rom, start

    def test_the_rom_still_hashes_to_the_banked_provenance(self, streams):
        prof, rom, start = self._prof()
        p = streams["provenance"]
        assert hashlib.sha256(rom.read_bytes()).hexdigest() == p["rom_sha256"]
        assert hashlib.sha256(
            start.read_bytes()).hexdigest() == p["start_state_sha256"]

    def test_the_banked_tape_stream_reproduces_from_the_rom(self, streams):
        prof, rom, start = self._prof()
        import numpy as np
        import nes_core
        from src.training.profile_utils import action_space_to_bitmasks

        rec = json.loads(TAPE.read_text())
        bm = action_space_to_bitmasks(prof["action_space"])
        pool = nes_core.Pool(rom_path=str(rom), num_workers=1, frame_skip=4)
        rows = []
        try:
            pool.set_headless(True)
            pool.set_odometer_enabled(True)
            pool.reset_all()
            pool.load_worker_state(0, start.read_bytes())

            def rd(out):
                r = np.frombuffer(out[0][2], np.uint8)
                rows.append([int(pool.get_odometer_blank_per_worker()[0]),
                             int(1 <= (1 - int(r[0x0303])) % 256 <= 8)]
                            + [int(r[a]) for a in RYGAR_AREA_KEY_ADDRS])

            rd(pool.step_all(np.zeros(1, dtype=np.uint8)))
            for a in rec["actions"]:
                rd(pool.step_all(np.array([bm[int(a)]], dtype=np.uint8)))
            x = int(pool.get_odometer_per_worker()[0][0])
            scene = int(pool.get_odometer_scene_per_worker()[0])
        finally:
            pool.shutdown()
        assert x == streams["tape"]["terminal_odometer_x"] == 6242
        assert scene == 0
        assert rle(rows) == [list(r) for r in streams["tape"]["rows"]], (
            "the banked stream no longer matches the emulator -- re-bank it "
            "and re-adjudicate every number in this file")

    def test_savestate_pressure_fabricates_no_transition(self):
        # Go-Explore restores constantly, and `odo_blank` travels inside the
        # savestate. 602 restores at the clear detector's own probe shape.
        prof, rom, start = self._prof()
        import numpy as np
        import nes_core
        from src.training.profile_utils import action_space_to_bitmasks

        rec = json.loads(TAPE.read_text())
        bm = action_space_to_bitmasks(prof["action_space"])
        pool = nes_core.Pool(rom_path=str(rom), num_workers=1, frame_skip=4)
        w = rygar_witness()
        restores = 0
        try:
            pool.set_headless(True)
            pool.set_odometer_enabled(True)
            pool.reset_all()
            pool.load_worker_state(0, start.read_bytes())

            def push(out):
                r = np.frombuffer(out[0][2], np.uint8)
                w.push(int(pool.get_odometer_blank_per_worker()[0]),
                       1 <= (1 - int(r[0x0303])) % 256 <= 8,
                       tuple(int(r[a]) for a in RYGAR_AREA_KEY_ADDRS))

            push(pool.step_all(np.zeros(1, dtype=np.uint8)))
            for i, a in enumerate(rec["actions"]):
                if i % 20 == 0:
                    snap = pool.save_worker_state(0)
                    for probe in (1, 2):
                        pool.step_all(np.array([bm[probe]], dtype=np.uint8))
                        pool.load_worker_state(0, snap)
                        restores += 1
                push(pool.step_all(np.array([bm[int(a)]], dtype=np.uint8)))
            x = int(pool.get_odometer_per_worker()[0][0])
        finally:
            pool.shutdown()
        assert restores == 602
        assert x == 6242, "the probe must not perturb the trajectory"
        s = w.summary()
        assert s["transitions"] == N_TRANSITIONS
        assert s["novel"] == N_NOVEL
        assert s["areas"] == [(0, 14), (0, 29), (3, 16)]
