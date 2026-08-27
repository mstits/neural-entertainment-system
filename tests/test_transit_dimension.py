"""The transition dimension a Go-Explore search may key and score on.

WHAT IS UNDER TEST
------------------
`scripts/transition_witness.transit_dimension` and `lex_score` -- the two
functions that stand between a working room counter and the solver's cell key
(`scripts/go_explore_solve.py:4866`, the `sect` slot) and score
(`:4954`, `score = sect * 10000 + gx`).

`tests/test_transition_witness.py` already proves the COUNTER is sound on
Rygar's own rollouts. This file proves the DIMENSION is safe to search on,
which is a different and stronger claim: a counter that is right 55 times can
still destroy a search if all 55 are round trips through one door.

ANTI-VACUITY
------------
The dimension's whole job is to NOT move most of the time, so a passing test
here is worthless unless it also fails when the mechanism is taken out. Every
class below pairs a "must not fire" case with the measured number the reverted
mechanism produces instead:

  * `TestTheRatchetCannotFarmTheDimension` -- 27 round trips through ONE door
    advance the dimension ONCE. Revert to the raw transition count and the
    same tape advances it 55 times.
  * `TestDeathsCannotMoveTheDimension` -- the death fades across 18 real
    rollouts move it zero times, and TWO independent guards are why: remove
    the length floor and those rollouts produce 18 transitions where they
    produced none. (The dimension itself still holds there, because a Rygar
    death reloads into the area it died in -- so the floor and the debounce
    are pinned on synthetic streams built from Rygar's own measured fade
    lengths, where a short fade DOES arrive somewhere new.)
  * `TestARestoredLineageCannotRebankItsParentsRooms` -- a lineage restored
    with its parent's `seen` set re-crosses every door on the tape and banks
    nothing. Drop the seed (the per-trajectory carry) and it re-banks both.
  * `TestZeroIsAmbiguous` -- the dimension reads 0 both for "no transitions"
    and for "the odometer was never on", so the wiring is required to consult
    the witness verdict. This is pinned rather than papered over.

Ledger: EXHIBITION. Every number here comes from Go-Explore search output and
scripted/random rollouts. No policy was trained for this game (CLAIMS.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.transition_witness import (
    NOVEL, TRANSITIONS, TRANSIT_SCORE_WEIGHT, TransitionWitness, lex_score,
    run_stream, rygar_witness, transit_dimension, unrle,
)

REPO = Path(__file__).resolve().parent.parent
STREAMS = REPO / "docs/receipts/rygar/transition_streams.json"
TAPE = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"

#: Rygar's own measurements. The two px figures are DERIVED from the tape's
#: own invariant block rather than copied, so a re-banked tape moves them.
N_TRANSITIONS = 55      # raw door crossings on the banked R1 tape
N_NOVEL = 2             # arrivals somewhere this trajectory had not been
_INV = json.loads(TAPE.read_text())["invariants_for_the_guard"]
VERIFIED_FRONTIER_PX = _INV["artifact_free_depth"]              # 4,608
#: Everything the headline odometer credits past the artifact-free ceiling:
#: pure re-anchor ratchet, 27 crossings of ONE door.
RATCHET_PX = _INV["terminal_odometer_x"] - VERIFIED_FRONTIER_PX  # 1,634


@pytest.fixture(scope="module")
def streams() -> dict:
    assert STREAMS.exists(), (
        f"{STREAMS} is missing; regenerate with "
        "`python scripts/transition_witness.py bank`.")
    return json.loads(STREAMS.read_text())


@pytest.fixture(scope="module")
def tape_rows(streams) -> list:
    return list(unrle(streams["tape"]["rows"]))


@pytest.fixture(scope="module")
def death_rows(streams) -> list:
    return [(d["policy"], list(unrle(d["rows"]))) for d in streams["deaths"]]


def _dimension(rows, **kw) -> list:
    """The shipped dimension over one lineage's observation stream."""
    return transit_dimension(run_stream(rygar_witness(**kw), rows))


def _raw_dimension(rows, **kw) -> list:
    """THE REVERTED MECHANISM: the same stream with the novelty gate taken
    out, i.e. counting every transition instead of every new area. Built
    here and never shipped, so the file that contains it is the file that
    refutes it."""
    out, n = [], 0
    for v in run_stream(rygar_witness(**kw), rows):
        if v in TRANSITIONS:
            n += 1
        out.append(n)
    return out


class TestTheRatchetCannotFarmTheDimension:
    """27 round trips through one door are not 27 rooms."""

    def test_the_banked_tape_advances_the_dimension_exactly_twice(self,
                                                                  tape_rows):
        d = _dimension(tape_rows)
        assert len(d) == len(tape_rows)
        assert d[-1] == N_NOVEL

    def test_the_dimension_is_monotone_inside_a_lineage(self, tape_rows):
        # A lexicographic leading term is only meaningful if it cannot go
        # down while one trajectory runs; `seen` only grows, so it cannot.
        d = _dimension(tape_rows)
        assert all(b - a in (0, 1) for a, b in zip(d, d[1:]))

    def test_reverting_the_novelty_gate_advances_it_55_times(self, tape_rows):
        # The mechanism removed. This is what a raw-count key/score would
        # have keyed and paid on, and it is 27x the honest number.
        assert _raw_dimension(tape_rows)[-1] == N_TRANSITIONS

    def test_the_two_differ_by_the_receipts_27_door_cycles(self, tape_rows):
        raw, novel = _raw_dimension(tape_rows)[-1], _dimension(tape_rows)[-1]
        assert raw - novel == 53
        assert _INV["post_door_segments_dx_zero"] == 27
        assert (raw - 1) // 2 == 27, (
            "the tape's transitions past the first are 27 round trips "
            "through ONE door; if that changed the weight below must be "
            "re-derived from the new ratchet, not left as written")


class TestTheScoreCannotBeOutbidByWalking:
    """Sizing the weight against Rygar's own frontier and its own ratchet."""

    def test_one_new_room_outranks_the_entire_verified_frontier(self):
        assert lex_score(1, 0) > lex_score(0, VERIFIED_FRONTIER_PX)

    def test_one_new_room_outranks_the_frontier_plus_the_whole_ratchet(self):
        # The odometer already pays +53..+64 px per door crossing. The
        # transition term has to dominate the artifact it sits next to,
        # or arming it just amplifies the ratchet.
        assert lex_score(1, 0) > lex_score(
            0, VERIFIED_FRONTIER_PX + RATCHET_PX)

    def test_the_ratchet_is_a_rounding_error_against_one_room(self):
        assert RATCHET_PX < TRANSIT_SCORE_WEIGHT // 4

    def test_the_raw_count_would_invert_the_objective(self, tape_rows):
        # 55 crossings of one door would score 550,000 -- 55 rooms' worth
        # of credit for a trajectory that never left the corridor. The
        # honest dimension scores the same tape at 2.
        raw = _raw_dimension(tape_rows)[-1]
        assert lex_score(raw, 0) > lex_score(50, VERIFIED_FRONTIER_PX)
        assert lex_score(_dimension(tape_rows)[-1], 0) < lex_score(3, 0)

    def test_weight_zero_disarms_the_score_arm_without_touching_the_key(self):
        # The key-only arm the Rygar campaign never ran: same dimension,
        # no score term, so the two mechanisms are separable rather than
        # confounded in one wire.
        assert lex_score(7, 123, weight=0) == 123


def _fade(frames: int, *, dead: bool, arrive, start=(0, 14), per_step=4):
    """A synthetic observation stream: settled in `start`, one blank run of
    `frames` frames, settled in `arrive`.

    Synthetic ON PURPOSE, and only for the guard-removal cases. The banked
    death rollouts (below) cannot exercise the length floor at the DIMENSION
    level because Rygar's death fade always reloads into the area it died
    in -- the area key is constant `(0, 14)` across all 18 of them -- so the
    novelty gate rejects them no matter what the floor is set to. Pinning
    the floor therefore needs a stream where a short fade DOES arrive
    somewhere new, which is the shape a respawn-into-another-area would
    have. The two lengths are Rygar's own measurements (14-frame death fade,
    78-frame door), not invented numbers.
    """
    rows = [(0, 0) + tuple(start)] * 3
    b, left = 0, frames
    while left > 0:
        step = min(per_step, left)
        b += step
        left -= step
        rows.append((b, int(dead)) + tuple(start))
    rows += [(b, 0) + tuple(arrive)] * 3
    return rows


class TestDeathsCannotMoveTheDimension:
    """THE MUST-NOT-FIRE CASE, guarded twice over."""

    def test_no_death_rollout_ever_advances_the_dimension(self, death_rows):
        for policy, rows in death_rows:
            d = _dimension(rows)
            assert d[-1] == 0, f"{policy} rollout banked a phantom room"
            assert set(d) == {0}

    def test_removing_the_length_floor_banks_all_18_death_fades(self,
                                                                death_rows):
        # MEASURED, on the real rollouts: at the calibrated floor the death
        # fades produce zero transitions; with the floor removed they
        # produce one apiece. The floor is what rejects them.
        assert sum(_raw_dimension(r)[-1] for _, r in death_rows) == 0
        assert sum(_raw_dimension(r, min_blank_frames=1)[-1]
                   for _, r in death_rows) == len(death_rows)

    def test_the_novelty_gate_is_a_second_independent_guard(self, death_rows):
        # ...and even with the floor gone, the DIMENSION still does not move
        # on these rollouts, because a Rygar death reloads into the area it
        # died in. Two guards, either of which alone suffices here. Recorded
        # so the next reader does not mistake the floor for the only one.
        assert all(_dimension(r, min_blank_frames=1)[-1] == 0
                   for _, r in death_rows)

    def test_a_short_fade_into_a_new_area_is_rejected_by_the_floor(self):
        rows = _fade(14, dead=False, arrive=(1, 21))
        assert _dimension(rows)[-1] == 0
        # THE MECHANISM REVERTED: drop the floor and the same 14-frame fade
        # banks a room.
        assert _dimension(rows, min_blank_frames=1)[-1] == 1

    def test_a_full_length_fade_with_a_death_in_it_is_rejected_by_the_debounce(
            self):
        rows = _fade(78, dead=True, arrive=(1, 21))
        assert _dimension(rows)[-1] == 0
        # THE MECHANISM REVERTED: push the debounce out of reach and a
        # death long enough to clear the floor banks a room.
        assert _dimension(rows, death_debounce=10 ** 6)[-1] == 1

    def test_a_full_length_fade_into_a_new_area_is_what_does_fire(self):
        # The positive control for both negatives above: same shape, real
        # door length, no death. If this did not fire, the two rejections
        # would be vacuous.
        assert _dimension(_fade(78, dead=False, arrive=(1, 21)))[-1] == 1


class TestARestoredLineageCannotRebankItsParentsRooms:
    """Per-trajectory semantics. `odo_blank` rides inside OdoState, so a
    restored cell resumes with the SAVING lineage's count -- and must resume
    with its `seen` set too, or every child re-banks its parent's rooms and
    the dimension becomes a treadmill one restore deep."""

    def test_a_child_seeded_with_its_parents_areas_banks_nothing(self,
                                                                 tape_rows):
        parent = rygar_witness()
        run_stream(parent, tape_rows)
        child = rygar_witness(seen=parent.seen)
        assert transit_dimension(run_stream(child, tape_rows))[-1] == 0

    def test_dropping_the_carry_lets_the_child_rebank_both_rooms(self,
                                                                 tape_rows):
        # The mechanism removed: a fresh witness at every restore. The same
        # doors pay again, once per re-root, forever.
        assert transit_dimension(run_stream(rygar_witness(), tape_rows))[-1] \
            == N_NOVEL

    def test_a_savestate_splice_cannot_fabricate_an_arrival(self):
        # A restore drives the raw counter DOWN (or forward by more than one
        # step of frames). Neither may close a run: the two ends came from
        # different timelines.
        w = rygar_witness()
        rows = [(0, 0, 1, 1)]
        rows += [(4 * i, 0, 1, 1) for i in range(1, 30)]   # 116 blank frames
        rows += [(7, 0, 9, 9)]                             # restore: count drops
        rows += [(7, 0, 9, 9)] * 5
        assert transit_dimension(run_stream(w, rows))[-1] == 0
        assert w.splices == 1

    def test_the_count_is_not_the_raw_counter(self, streams, tape_rows):
        # 4,329 blank FRAMES, 55 runs, 2 rooms. Three different numbers, and
        # only the last one may enter a cell key.
        assert streams["tape"]["terminal_odo_blank"] == 4329
        assert _dimension(tape_rows)[-1] == N_NOVEL


class TestZeroIsAmbiguous:
    """The dimension alone cannot tell "nowhere new" from "never measured",
    so the wiring is REQUIRED to consult the witness verdict. Pinned here so
    a future consumer cannot read a silent zero as a measurement."""

    def test_an_absent_odometer_and_a_quiet_lineage_read_identically(self):
        off = TransitionWitness(min_blank_frames=40, frames_per_step=4,
                                odometer_enabled=False)
        quiet = rygar_witness()
        rows = [(0, 0, 1, 1)] * 200
        assert transit_dimension(run_stream(off, rows))[-1] == 0
        assert transit_dimension(run_stream(quiet, rows))[-1] == 0
        # ...and the ONLY thing that separates them is the verdict.
        assert off.summary()["verdict"] == "UNAVAILABLE"
        assert quiet.summary()["verdict"] == "UNINSTRUMENTED"

    def test_a_moving_counter_with_no_new_rooms_is_a_real_zero(self,
                                                              death_rows):
        w = rygar_witness()
        run_stream(w, death_rows[0][1])
        s = w.summary()
        assert s["verdict"] == "OK"
        assert s["novel"] == 0 and s["deaths"] > 0
