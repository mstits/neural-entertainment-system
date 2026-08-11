"""Tests for scripts/go_explore_chain.py's entrance extraction.

The chain driver builds its OWN Pool to replay a solution and snapshot
the next level's entrance. That pool must run the same machine the
solve ran on, or the entrance blob it banks came from a machine no
later run reproduces — the receipted cv_chain_hw2 failure mode. A real
Pool needs a ROM, so `Pool` and `make_game` are monkeypatched here and
the assertions are on call ORDER and on the sidecar that lands next to
the blob.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.go_explore_chain as chain

PROFILE = {"action_space": [[], ["right"]], "frame_skip": 4}


class _Clock:
    """Shared step counter: the fake pool ticks it, the fake game reads
    it, so the level key advances as a function of steps replayed."""

    def __init__(self) -> None:
        self.n = 0


class _FakeGame:
    rom = "roms/does-not-need-to-exist.nes"

    def __init__(self, clock: _Clock, advance_at: int) -> None:
        self.clock, self.advance_at = clock, advance_at

    def level_key(self, ram):
        return (1,) if self.clock.n >= self.advance_at else (0,)

    def lives(self, ram):
        return 3

    def is_dead(self, ram, start_lives):
        return False


class _FakePool:
    """Records every configuration call in order so a test can assert
    that hw flags land BEFORE reset_all().

    Also models save/restore honestly: `history` is the class-wide list
    of every action bitmask actually stepped, a savestate blob records
    its length, and reloading that blob truncates back to it. That makes
    a rewind observable to the fake game, which is what the
    blip-desync regression turns on."""

    instances: list = []
    clock: _Clock = _Clock()
    history: list = []
    MARK = b"ENTRANCE-BLOB@"

    def __init__(self, rom_path, num_workers, frame_skip):
        self.calls: list = []
        self.frame_skip = frame_skip
        _FakePool.instances.append(self)

    def __getattr__(self, name):
        if not name.startswith("set_hw_"):
            raise AttributeError(name)
        return lambda on: self.calls.append(name)

    def set_headless(self, on):
        self.calls.append("set_headless")

    def reset_all(self):
        self.calls.append("reset_all")

    def load_worker_state(self, wid, blob):
        self.calls.append("load_worker_state")
        if blob.startswith(_FakePool.MARK):        # rewind to a saved mark
            del _FakePool.history[int(blob[len(_FakePool.MARK):]):]

    def step_all(self, x):
        _FakePool.clock.n += 1
        _FakePool.history.append(int(x[0]))
        return [(None, None, bytearray(2048))]

    def save_worker_state(self, wid):
        # Recorded too: one save per judged settle (plus one on a bank),
        # so a test can count how many settles a scan actually paid for.
        self.calls.append("save_worker_state")
        return _FakePool.MARK + str(len(_FakePool.history)).encode()

    def shutdown(self):
        self.calls.append("shutdown")


class _TraceGame:
    """A game whose level key advances only when the EXACT expected
    action history has been applied as a prefix.

    This is the property a destructive settle destroys: NOOPs injected
    mid-replay sit where solution actions belong, so the prefix can
    never match again and the real clear is never observed — exactly
    how Bubble Bobble round 67 lost its 67->68 transition.
    `blip_at` reproduces the trigger: a one-step transient level-key
    reading at that history length, standing in for $0401's momentary
    dip to the previous round's value."""

    rom = "roms/does-not-need-to-exist.nes"

    def __init__(self, want, blip_at=None, dead=False) -> None:
        self.want, self.blip_at, self.dead = list(want), blip_at, dead

    def level_key(self, ram):
        h = _FakePool.history
        if self.blip_at is not None and len(h) == self.blip_at:
            return (9,)
        n = len(self.want)
        return (1,) if len(h) >= n and h[:n] == self.want else (0,)

    def lives(self, ram):
        return 3

    def is_dead(self, ram, start_lives):
        return self.dead


@pytest.fixture
def replay(monkeypatch):
    """Returns a runner for extract_next_entrance against the fakes.
    `advance_at` = the step on which the level key moves forward; a huge
    value means the replay never sees a transition."""
    clock = _Clock()
    _FakePool.instances.clear()
    _FakePool.clock = clock
    _FakePool.history = []
    monkeypatch.setattr(chain, "Pool", _FakePool)
    monkeypatch.setattr(chain, "action_space_to_bitmasks", lambda sp: [0, 1])

    def run(out_path, hw_flags=None, advance_at=3, **extra):
        clock.n = 0
        monkeypatch.setattr(
            chain, "make_game", lambda profile: _FakeGame(clock, advance_at))
        kw = {} if hw_flags is None else {"hw_flags": hw_flags}
        return chain.extract_next_entrance(
            PROFILE, b"ROOT", [1] * 12, out_path,
            settle=1, stable_for=1, settle_cap=4, **kw, **extra)

    return run


def test_default_call_sets_no_hw_flags(replay, tmp_path):
    # Byte-identical to every existing chain run: the new parameter
    # defaults to empty, so not one set_hw_* call is made.
    replay(tmp_path / "entrance_after_1-1.state")
    calls = _FakePool.instances[0].calls
    assert [c for c in calls if c.startswith("set_hw_")] == []


def test_hw_flags_are_applied_before_reset_all(replay, tmp_path):
    # Order is load-bearing: reset_alignment set AFTER reset_all boots a
    # different power-on lineage (divergence at frame 11, $006F/$01FE).
    replay(tmp_path / "entrance_after_1-1.state",
           hw_flags=["reset_alignment", "mmio_read_timing"])
    calls = _FakePool.instances[0].calls
    assert calls.index("set_hw_reset_alignment") < calls.index("reset_all")
    assert calls.index("set_hw_mmio_read_timing") < calls.index("reset_all")
    assert calls.index("reset_all") < calls.index("load_worker_state")


def test_banked_entrance_gets_a_lineage_sidecar(replay, tmp_path):
    out = tmp_path / "entrances" / "entrance_after_1-1.state"
    path, key = replay(out, hw_flags=["mmio_read_timing", "nmi_poll_timing"])
    assert path == str(out)
    assert out.read_bytes().startswith(b"ENTRANCE-BLOB")
    rec = json.loads((tmp_path / "entrances" /
                      "entrance_after_1-1.state.json").read_text())
    assert rec["hw_flags"] == ["mmio_read_timing", "nmi_poll_timing"]
    assert rec["frame_skip"] == 4
    assert rec["blob"] == "entrance_after_1-1.state"
    assert rec["settled_key"] == list(key)


def test_no_sidecar_is_written_when_no_transition_is_found(replay, tmp_path):
    out = tmp_path / "entrance_after_1-1.state"
    assert replay(out, advance_at=10**6) == (None, None)
    assert not out.exists()
    assert not (tmp_path / "entrance_after_1-1.state.json").exists()


# ---- destructive-settle rewind + post-trace scan window ---------------
#
# Bubble Bobble round 67 (2026-08-09): $0401 blipped down to the previous
# round's value at replay step 3, the reject path settled 53 NOOPs into
# the pool WITHOUT rewinding, and the 200 remaining solution actions then
# replayed against an idled machine. The genuine 67->68 transition on the
# very last action was never observed and the chain halted with "no
# forward transition captured after 67".

ACTIONS = [1] * 12
WANT = [0] + ACTIONS          # rooting NOOP, then the solution trace


@pytest.fixture
def replay_trace(monkeypatch):
    """Runner for extract_next_entrance against a game that advances only
    on an exact, uninterrupted action history."""
    _FakePool.instances.clear()
    _FakePool.clock = _Clock()
    monkeypatch.setattr(chain, "Pool", _FakePool)
    monkeypatch.setattr(chain, "action_space_to_bitmasks", lambda sp: [0, 1])

    def run(out_path, want=WANT, blip_at=None, dead=False, **kw):
        _FakePool.history = []
        monkeypatch.setattr(chain, "make_game",
                            lambda profile: _TraceGame(want, blip_at, dead))
        return chain.extract_next_entrance(
            PROFILE, b"ROOT", ACTIONS, out_path,
            settle=1, stable_for=1, settle_cap=4, **kw)

    return run


def test_clean_replay_reaches_the_transition(replay_trace, tmp_path):
    # Control: with no blip the trace advances the key on its last action.
    assert replay_trace(tmp_path / "e.state")[1] == (1,)


def test_a_transient_key_blip_does_not_desync_the_rest_of_the_replay(
        replay_trace, tmp_path):
    # THE REGRESSION. The blip fires at history length 3 (the second
    # solution action), is settled, and is correctly rejected — after
    # which the remaining ten actions must still land on the exact
    # history the transition needs. Pre-fix this returned (None, None).
    out = tmp_path / "e.state"
    path, key = replay_trace(out, blip_at=3)
    assert key == (1,)
    assert path == str(out)
    assert _FakePool.history[:len(WANT)] == WANT


def test_a_rejected_settle_rewinds_the_pool_to_the_trigger(
        replay_trace, tmp_path):
    # The rewind is a real save/restore round-trip, not a bookkeeping
    # adjustment: exactly one reload beyond the root load.
    replay_trace(tmp_path / "e.state", blip_at=3)
    calls = _FakePool.instances[0].calls
    assert calls.count("load_worker_state") == 2


def test_a_doomed_settle_also_rewinds(replay_trace, tmp_path):
    # is_dead rejection takes the same path and must rewind too, or a
    # death-animation blip truncates the search the same way. Here the
    # only trigger is the genuine transition, so the result is a clean
    # "nothing banked" with the settle's NOOPs undone.
    out = tmp_path / "e.state"
    assert replay_trace(out, dead=True) == (None, None)
    assert not out.exists()
    assert _FakePool.instances[0].calls.count("load_worker_state") == 2
    assert _FakePool.history == WANT      # settle NOOPs rewound, nothing extra


# `transit_timeout_frames`: a key that only turns over during an
# interlude the solution trace does not cover. Needs 5 NOOP steps past
# the trace; frame_skip is 4, so the knob is in FRAMES.
TAIL_WANT = WANT + [0] * 5


def test_no_post_trace_scan_by_default(replay_trace, tmp_path):
    out = tmp_path / "e.state"
    assert replay_trace(out, want=TAIL_WANT) == (None, None)
    assert not out.exists()


def test_transit_timeout_frames_extends_the_scan_past_the_trace(
        replay_trace, tmp_path):
    path, key = replay_trace(tmp_path / "e.state", want=TAIL_WANT,
                             transit_timeout_frames=20)   # 20 // 4 = 5 steps
    assert key == (1,)
    assert path is not None


def test_transit_timeout_frames_is_divided_by_frame_skip(
        replay_trace, tmp_path):
    # 16 frames is only 4 steps at frame_skip 4 — one short. If the knob
    # were read as steps this would pass and the unit would be a lie.
    assert replay_trace(tmp_path / "e.state", want=TAIL_WANT,
                        transit_timeout_frames=16) == (None, None)


# ---- chain-scope sequence invariant -----------------------------------
#
# "Settled, not the key we started on, not dead" is too weak a definition
# of forward progress: a room the campaign already passed through
# satisfies all three, so a re-entry banks as a new stage and the chain
# claims progress it never made (the Kirby fabrication class). The
# campaign's own visited-key history is the missing term.


def test_empty_history_leaves_the_scan_exactly_as_it_was(replay, tmp_path):
    # A NEW campaign's first level: the guard is on but the history is
    # empty, so nothing is refusable and the bank is the old bank.
    out = tmp_path / "e.state"
    assert replay(out, visited=())[1] == (1,)
    assert out.exists()


def test_a_settled_key_already_in_the_history_is_refused(replay, tmp_path):
    # THE INVARIANT. The replay's only transition settles on (1,), which
    # this campaign already banked — it is a revisit, not a new level.
    out = tmp_path / "e.state"
    assert replay(out, visited=[(1,)]) == (None, None)
    assert not out.exists()
    assert not (tmp_path / "e.state.json").exists()


def test_a_new_settled_key_still_banks_with_a_nonempty_history(
        replay, tmp_path):
    # The other half of the contract: a genuine forward transit is
    # untouched by a history that does not contain it.
    out = tmp_path / "e.state"
    path, key = replay(out, visited=[(7,), (8,)])
    assert (path, key) == (str(out), (1,))


def test_history_keys_match_across_a_json_round_trip(replay, tmp_path):
    # The persisted history comes back as LISTS; the adapters emit
    # tuples. If the guard compared raw shapes it would silently never
    # match and the invariant would be decorative.
    assert replay(tmp_path / "e.state",
                  visited=json.loads("[[1]]")) == (None, None)


def test_allow_revisit_banks_the_revisited_key(replay, tmp_path):
    out = tmp_path / "e.state"
    path, key = replay(out, visited=[(1,)], allow_revisit=True)
    assert (path, key) == (str(out), (1,))


def test_a_refused_revisit_rewinds_the_pool(replay, tmp_path):
    # Same contract as the blip/doomed rejections: EVERY rejection must
    # undo its settle's NOOPs or the rest of the replay runs
    # desynchronised. Nothing banks here, so every save is a settle and
    # every settle owes a rewind, on top of the root load.
    replay(tmp_path / "e.state", visited=[(1,)])
    calls = _FakePool.instances[0].calls
    settles = calls.count("save_worker_state")
    assert settles and calls.count("load_worker_state") == settles + 1


def test_a_refusal_does_not_blacklist_the_keys_value_for_the_scan(
        replay, tmp_path):
    # A refusal is a verdict about one SETTLE, never about a raw reading.
    # Screening later actions against previously-refused keys would skip
    # them unjudged; here that shows up as one settle instead of eleven
    # (ten actions follow the refusal, and each must be judged on its
    # own). The cost of judging is the price of not losing a transit —
    # see the next test for what the skip actually loses.
    replay(tmp_path / "e.state", visited=[(1,)])
    assert _FakePool.instances[0].calls.count("save_worker_state") == 11


def test_settles_are_only_paid_on_actions_that_left_the_start_key(
        replay, tmp_path):
    # The bound on that cost: an action still reading start_key is never
    # settled, so the scan pays at most one settle per action, not one
    # per action unconditionally.
    replay(tmp_path / "e.state", visited=[(1,)], advance_at=10)
    # clock: 1 rooting step + 12 actions; the key moves at step 10, so
    # actions 9..12 (four of them) leave start_key and are judged.
    assert _FakePool.instances[0].calls.count("save_worker_state") == 4


def test_the_guard_judges_the_settled_key_not_a_transient_blip(
        replay_trace, tmp_path):
    # MIND THE BB RECEIPTS. Round 67's counter dips to the PREVIOUS
    # round's value for one step. That value is in the campaign history
    # by definition, so a guard reading the raw trigger would refuse the
    # genuine 67->68 transit. Only settled keys are compared: here (9,)
    # is in the history and blips at step 3, and the real transition to
    # (1,) must still bank.
    out = tmp_path / "e.state"
    path, key = replay_trace(out, blip_at=3, visited=[(9,)])
    assert (path, key) == (str(out), (1,))
    assert _FakePool.history[:len(WANT)] == WANT


def test_a_visited_valued_blip_on_the_last_action_still_banks(
        replay_trace, tmp_path):
    # Round 67's exact shape: the genuine transition fires on the VERY
    # LAST action, and the raw reading at that instant is the previous
    # round's number — a value the campaign has certainly visited. A
    # guard that screened the raw trigger would skip the last action and
    # report "no forward transition"; screening the settled key banks it.
    out = tmp_path / "e.state"
    assert replay_trace(out, blip_at=len(WANT),
                        visited=[(9,)]) == (str(out), (1,))


class _RevisitThenTransitGame:
    """A replay that drops into an already-banked room for real and then
    transits out of it for real, on the last action.

    Steps `lo..hi` read (9,) and SETTLE there: a genuine re-entry the
    invariant must refuse. At step `transit_at` the trace is complete and
    the reading is (9,) once more — this time only as the transient of
    the real transition, which settles into the NEW room (1,).

    Refusing (9,) and then screening later raw readings against it skips
    the last action unjudged and loses the transit: (None, None) out of a
    stage that really did reach a new room. That is the round-67
    lost-transition class re-created by an optimization, which is why a
    refusal is a verdict about one settle and never about a value."""

    rom = "roms/does-not-need-to-exist.nes"

    def __init__(self, want, lo, hi, transit_at) -> None:
        self.want, self.lo, self.hi = list(want), lo, hi
        self.transit_at = transit_at

    def level_key(self, ram):
        h = _FakePool.history
        n = len(h)
        if n == self.transit_at:
            return (9,)                      # the transit's transient reading
        if n > self.transit_at and h[:self.transit_at] == self.want:
            return (1,)                      # the new room, once settled
        if self.lo <= n <= self.hi:
            return (9,)                      # the re-entered, banked room
        return (0,)

    def lives(self, ram):
        return 3

    def is_dead(self, ram, start_lives):
        return False


def test_a_refused_key_reappearing_as_a_transit_trigger_still_banks(
        monkeypatch, tmp_path):
    # THE REPRO. (9,) is in the campaign history, settles stably at steps
    # 3-8 (refused, correctly), and is the raw reading again at step 13
    # where the genuine transit to (1,) fires. Pre-fix: (None, None) and
    # "no forward transition captured".
    _FakePool.instances.clear()
    _FakePool.clock = _Clock()
    _FakePool.history = []
    monkeypatch.setattr(chain, "Pool", _FakePool)
    monkeypatch.setattr(chain, "action_space_to_bitmasks", lambda sp: [0, 1])
    monkeypatch.setattr(chain, "make_game", lambda profile:
                        _RevisitThenTransitGame(WANT, 3, 8, len(WANT)))
    out = tmp_path / "e.state"
    assert chain.extract_next_entrance(
        PROFILE, b"ROOT", ACTIONS, out, settle=1, stable_for=1, settle_cap=4,
        visited=[(9,)]) == (str(out), (1,))
    # And every refusal rewound, so the trace the transition needs is
    # still the exact history that was replayed.
    assert _FakePool.history[:len(WANT)] == WANT


# ---- campaign key history (persistence + positioning) -----------------
#
# A recorded walk is an ORDERED sequence and a re-run does not
# necessarily resume at its end. Handing the whole thing back regardless
# dead-ends every re-run that starts anywhere else: the stage-0 solve
# succeeds, the extraction settles on a key the log already holds, the
# invariant refuses it, and the driver writes `next: null` — a burned
# per-level budget and a false campaign end, unattended. The history is
# therefore truncated at the start state's own settled key.


def test_key_history_round_trips_through_the_campaign_dir(tmp_path):
    recs = [{"key": [1, 2], "label": "1-2", "after": "1-1"}]
    p = chain.save_key_history(tmp_path, recs)
    assert p == tmp_path / chain.VISITED_FILE
    assert chain.load_key_history(tmp_path, (1, 2)) == recs


def test_history_of_a_fresh_campaign_dir_is_empty(tmp_path):
    assert chain.load_key_history(tmp_path, (1, 2)) == []


def test_history_stops_at_the_start_states_own_key(tmp_path):
    # Resuming from stage 1's entrance: stages 0-1 are behind this run,
    # stage 2 is a room it has NOT reached and must be free to re-bank.
    chain.save_key_history(tmp_path, [
        {"key": [1, 2], "label": "1-2", "after": "1-1"},
        {"key": [1, 3], "label": "1-3", "after": "1-2"},
        {"key": [1, 4], "label": "1-4", "after": "1-3"},
    ])
    assert [r["key"] for r in chain.load_key_history(tmp_path, (1, 3))] == [
        [1, 2], [1, 3]]


def test_an_unknown_root_key_yields_an_empty_history(tmp_path):
    # A hand-captured root has no sidecar key, so nothing can be PROVEN
    # to be behind this run. Guarding only what the run banks itself is
    # the pre-history behavior; it can never fabricate a campaign end.
    chain.save_key_history(tmp_path, [
        {"key": [1, 2], "label": "1-2", "after": "1-1"}])
    assert chain.load_key_history(tmp_path, None) == []


def test_a_root_key_absent_from_the_receipts_yields_an_empty_history(tmp_path):
    # Re-running from the campaign's ORIGINAL root: its key is not in the
    # walk (the walk is what came after it), so the run re-walks.
    chain.save_key_history(tmp_path, [
        {"key": [1, 2], "label": "1-2", "after": "1-1"}])
    assert chain.load_key_history(tmp_path, (1, 1)) == []


def test_history_rebuilds_from_chain_jsonl_when_the_file_is_missing(tmp_path):
    # A campaign dir written before the history file existed: the
    # campaign's own receipt is the fallback source of truth. The stall
    # record carries no next_wd and the torn line is unparseable; neither
    # may enter the history.
    (tmp_path / "chain.jsonl").write_text(
        json.dumps({"level": "1-1", "status": "solved",
                    "next_wd": [1, 2], "next_label": "1-2"}) + "\n"
        + json.dumps({"level": "1-2", "status": "solved",
                      "next_wd": [1, 3], "next_label": "1-3"}) + "\n"
        + json.dumps({"level": "1-3", "status": "stall"}) + "\n"
        + '{"level": "1-4", "next_wd": [1,')
    assert chain.load_key_history(tmp_path, (1, 3)) == [
        {"key": [1, 2], "label": "1-2", "after": "1-1"},
        {"key": [1, 3], "label": "1-3", "after": "1-2"},
    ]


def test_a_malformed_record_is_dropped_not_crashed_on(tmp_path):
    # The driver indexes every record's key. One hand-edited or
    # half-written entry must cost that entry, not the campaign.
    (tmp_path / chain.VISITED_FILE).write_text(json.dumps({"visited": [
        {"key": [1, 2], "label": "1-2", "after": "1-1"},
        {"label": "no key at all"}, "not even a record", {"key": "xyz"},
    ]}))
    assert chain.load_key_history(tmp_path, (1, 2)) == [
        {"key": [1, 2], "label": "1-2", "after": "1-1"}]


def test_an_unreadable_history_file_falls_back_to_the_chain_log(tmp_path):
    (tmp_path / chain.VISITED_FILE).write_text("{not json")
    (tmp_path / "chain.jsonl").write_text(
        json.dumps({"level": "1-1", "next_wd": [1, 2],
                    "next_label": "1-2"}) + "\n")
    assert [r["key"] for r in
            chain.load_key_history(tmp_path, (1, 2))] == [[1, 2]]


def test_a_file_that_does_not_reach_the_root_key_falls_back_to_the_log(
        tmp_path):
    # A dir already re-walked once has a history file SHORTER than its
    # own chain.jsonl. Resuming from an entrance the log still records
    # should keep that walk rather than throw it away.
    chain.save_key_history(tmp_path, [
        {"key": [1, 2], "label": "1-2", "after": "1-1"}])
    (tmp_path / "chain.jsonl").write_text(
        json.dumps({"level": "1-1", "next_wd": [1, 2],
                    "next_label": "1-2"}) + "\n"
        + json.dumps({"level": "1-2", "next_wd": [1, 3],
                      "next_label": "1-3"}) + "\n")
    assert [r["key"] for r in
            chain.load_key_history(tmp_path, (1, 3))] == [[1, 2], [1, 3]]


def test_root_key_record_reads_the_start_states_sidecar(tmp_path):
    blob = tmp_path / "entrance_after_1-1.state"
    blob.write_bytes(b"X")
    (tmp_path / "entrance_after_1-1.state.json").write_text(
        json.dumps({"hw_flags": [], "settled_key": [1, 2]}))
    assert chain.root_key_record(blob, "1-2") == {
        "key": [1, 2], "label": "1-2", "after": None}


@pytest.mark.parametrize("sidecar", [None, {"hw_flags": []}, "{bad"])
def test_root_key_record_is_none_without_a_trustworthy_key(tmp_path, sidecar):
    # A hand-captured blob has no sidecar key; the root is then simply
    # not in the history rather than a guessed one.
    blob = tmp_path / "root.state"
    blob.write_bytes(b"X")
    if sidecar is not None:
        (tmp_path / "root.state.json").write_text(
            sidecar if isinstance(sidecar, str) else json.dumps(sidecar))
    assert chain.root_key_record(blob, "1-1") is None


# ---- the driver loop: stage k+1 requires stage k banked ---------------


class _LabelGame:
    rom = "roms/does-not-need-to-exist.nes"

    @staticmethod
    def label(key):
        return "-".join(str(int(x)) for x in key)


def _drive_main(monkeypatch, tmp_path, *, keys, stall_at=None,
                no_transit_at=None, root_sidecar=None, prior_log=None,
                argv_extra=()):
    """Run chain.main() with the per-level solver subprocess and the
    entrance extractor faked out.

    `stall_at` = the stage whose solve produces no solution file;
    `no_transit_at` = the stage whose extraction finds no forward
    transition. Returns (calls, chain.jsonl records, visited records)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "root.state"
    root.write_bytes(b"ROOT")
    if root_sidecar is not None:
        (tmp_path / "root.state.json").write_text(json.dumps(root_sidecar))
    prof = tmp_path / "profile.yaml"
    prof.write_text("action_space: [[], [right]]\nframe_skip: 4\n")
    out = tmp_path / "campaign"
    out.mkdir(parents=True)
    if prior_log is not None:
        (out / "chain.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in prior_log))

    calls = []

    def fake_run(cmd, check=False):
        d = Path(cmd[cmd.index("--out") + 1])
        stage = int(d.name.split("_")[1])
        calls.append(("solve", stage, cmd[cmd.index("--root-state") + 1],
                      list(cmd)))
        if stage == stall_at:
            return None                       # solver found nothing
        (d / "solutions").mkdir(parents=True, exist_ok=True)
        np.save(d / "solutions" / "sol_0.actions.npy", np.array([1, 1, 1]))

    def fake_extract(profile, root_bytes, actions, out_path, **kw):
        # Models the real extractor's contract, invariant included: a
        # settled key already in the campaign history is refused, which
        # is what turns a mispositioned history into a false campaign end.
        stage = len([c for c in calls if c[0] == "extract"])
        visited = [tuple(k) for k in kw.get("visited", ())]
        calls.append(("extract", stage, visited, bool(kw.get("allow_revisit"))))
        if stage == no_transit_at:
            return None, None
        assert stage < len(keys), "extractor ran past the scripted stages"
        if keys[stage] in visited and not kw.get("allow_revisit"):
            return None, None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ENTRANCE")
        return str(out_path), keys[stage]

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    monkeypatch.setattr(chain, "extract_next_entrance", fake_extract)
    monkeypatch.setattr(chain, "make_game", lambda profile: _LabelGame())
    monkeypatch.setattr(chain, "resolve_hw_flags", lambda profile, s: [])
    monkeypatch.setattr(sys, "argv", [
        "go_explore_chain.py", "--start-state", str(root),
        "--start-label", "1-1", "--profile", str(prof), "--out", str(out),
        "--max-levels", str(len(keys) + 2), *argv_extra])

    assert chain.main() == 0
    recs = [json.loads(ln) for ln in
            (out / "chain.jsonl").read_text().splitlines() if ln.strip()]
    vp = out / chain.VISITED_FILE
    visited = json.loads(vp.read_text())["visited"] if vp.exists() else []
    return calls, recs, visited


def test_a_stalled_stage_stops_the_chain_before_the_next_one(
        monkeypatch, tmp_path):
    # Stage k+1 requires stage k banked. Stage 1's solve finds nothing,
    # so nothing after it may run: no stage-1 extraction, no stage-2
    # solve, and stage 1's key never enters the history.
    calls, recs, vis = _drive_main(monkeypatch, tmp_path,
                                   keys=[(1, 2), (1, 3)], stall_at=1)
    assert [c[1] for c in calls if c[0] == "solve"] == [0, 1]
    assert [c[1] for c in calls if c[0] == "extract"] == [0]
    assert [r["status"] for r in recs] == ["solved", "stall"]
    assert [v["key"] for v in vis] == [[1, 2]]


def test_a_stage_with_no_forward_transition_stops_the_chain(
        monkeypatch, tmp_path):
    # The other break: the level was solved but no NEW room was reached
    # (every candidate refused, or the trace never transits). The stage
    # is logged with next=None and the chain must not walk on.
    calls, recs, vis = _drive_main(monkeypatch, tmp_path,
                                   keys=[(1, 2), (1, 3)], no_transit_at=1)
    assert [c[1] for c in calls if c[0] == "solve"] == [0, 1]
    assert [c[1] for c in calls if c[0] == "extract"] == [0, 1]
    assert [r.get("next", "banked") for r in recs] == ["banked", None]
    assert [v["key"] for v in vis] == [[1, 2]]


def test_each_stage_sees_every_key_the_campaign_has_banked(
        monkeypatch, tmp_path):
    calls, recs, vis = _drive_main(monkeypatch, tmp_path,
                                   keys=[(1, 2), (1, 3)], stall_at=2)
    ex = [c for c in calls if c[0] == "extract"]
    assert ex[0][2] == []                    # new campaign: nothing visited
    assert ex[1][2] == [(1, 2)]              # stage 0's settled key
    assert [v["key"] for v in vis] == [[1, 2], [1, 3]]
    assert [v["after"] for v in vis] == ["1-1", "1-2"]
    assert [v["label"] for v in vis] == ["1-2", "1-3"]


def test_the_root_entrances_own_key_seeds_the_history(monkeypatch, tmp_path):
    # A chain-produced start state records its settled key, so a transit
    # BACK to the campaign root is refusable from the very first stage.
    calls, recs, vis = _drive_main(monkeypatch, tmp_path, keys=[(1, 2)],
                                   stall_at=1,
                                   root_sidecar={"settled_key": [1, 1]})
    ex = [c for c in calls if c[0] == "extract"]
    assert ex[0][2] == [(1, 1)]
    assert [v["key"] for v in vis] == [[1, 1], [1, 2]]


def test_a_true_continuation_keeps_the_history_it_already_banked(
        monkeypatch, tmp_path):
    # Resuming from the LAST banked entrance: its sidecar key is the end
    # of the recorded walk, so the WHOLE walk is behind this run and
    # stage 0 must see all of it — not just the entrance it stands on.
    calls, recs, vis = _drive_main(
        monkeypatch, tmp_path, keys=[(1, 4)], stall_at=1,
        root_sidecar={"settled_key": [1, 3]},
        prior_log=[{"level": "1-1", "next_wd": [1, 2], "next_label": "1-2"},
                   {"level": "1-2", "next_wd": [1, 3], "next_label": "1-3"}])
    ex = [c for c in calls if c[0] == "extract"]
    assert ex[0][2] == [(1, 2), (1, 3)]
    assert [v["key"] for v in vis] == [[1, 2], [1, 3], [1, 4]]


def test_rerunning_the_same_command_rewalks_instead_of_dead_ending(
        monkeypatch, tmp_path):
    # THE REPRO. Identical command after a crash: same root, same --out,
    # and a chain.jsonl that already records the key stage 0 produces.
    # Pre-fix the rebuilt history refused that key, the driver wrote
    # `next: null` and printed "no forward transition captured" — a
    # burned per-level solve budget and a false campaign end. The run
    # must simply re-bank the stage.
    calls, recs, vis = _drive_main(
        monkeypatch, tmp_path, keys=[(2,)], stall_at=1,
        prior_log=[{"level": "1-1", "status": "solved",
                    "next_wd": [2], "next_label": "2"}])
    assert recs[-2]["next_wd"] == [2]
    assert "next" not in recs[-2]
    assert [c[1] for c in calls if c[0] == "solve"] == [0, 1]
    assert [v["key"] for v in vis] == [[2]]


def test_redoing_an_earlier_stage_truncates_the_history_there(
        monkeypatch, tmp_path):
    # Deliberately re-doing stage 2 from its banked entrance. Stages 0-1
    # stay behind the run; stage 2's own product ([1, 4]) is a room this
    # run has NOT reached again, so it must be free to re-bank — pre-fix
    # the unpositioned history refused it and ended the campaign.
    calls, recs, vis = _drive_main(
        monkeypatch, tmp_path, keys=[(1, 4)], stall_at=1,
        root_sidecar={"settled_key": [1, 3]},
        prior_log=[{"level": "1-1", "next_wd": [1, 2], "next_label": "1-2"},
                   {"level": "1-2", "next_wd": [1, 3], "next_label": "1-3"},
                   {"level": "1-3", "next_wd": [1, 4], "next_label": "1-4"}])
    ex = [c for c in calls if c[0] == "extract"]
    assert ex[0][2] == [(1, 2), (1, 3)]
    assert recs[-2]["next_wd"] == [1, 4]
    assert [v["key"] for v in vis] == [[1, 2], [1, 3], [1, 4]]


def test_the_guard_is_on_by_default_and_off_under_allow_revisit(
        monkeypatch, tmp_path):
    calls, _, _ = _drive_main(monkeypatch, tmp_path, keys=[(1, 2)],
                              stall_at=1)
    assert [c[3] for c in calls if c[0] == "extract"] == [False]
    calls, _, _ = _drive_main(monkeypatch, tmp_path / "b", keys=[(1, 2)],
                              stall_at=1, argv_extra=("--allow-revisit",))
    assert [c[3] for c in calls if c[0] == "extract"] == [True]


# ---------------------------------------------------------------------
# gate-opener passthrough
#
# The receipted dead-flag lineage this closes: eval's --start-state and
# --sect-cap were both unreachable through their drivers for weeks, and
# --kill-key / --ortho still are. An arm the campaign driver cannot pass
# is an arm the campaign never runs, and nobody finds out until the
# receipts are read.
# ---------------------------------------------------------------------

GATE_FLAGS = ("--gate-opener", "--gate-sweep-frac", "--gate-sweep-roots",
              "--gate-sweep-repeats", "--gate-pin-secs",
              "--gate-target-typed", "--gate-band", "--gate-sham-roots",
              "--gate-axes", "--contact-bits")


def test_the_chain_forwards_every_gate_flag_to_the_per_level_solver(
        monkeypatch, tmp_path):
    axes = tmp_path / "gate_axes_cv.json"
    axes.write_text("{}")
    calls, _recs, _vis = _drive_main(
        monkeypatch, tmp_path, keys=[(1, 2)], stall_at=1,
        argv_extra=("--gate-opener", "enumerate", "--gate-pin-secs", "600",
                    "--gate-target-typed", "--gate-sweep-frac", "0.2",
                    "--gate-sweep-roots", "8", "--gate-sweep-repeats", "3",
                    "--gate-band", "12", "--gate-sham-roots", "2",
                    "--gate-axes", str(axes), "--contact-bits", "2"))
    cmd = next(c[3] for c in calls if c[0] == "solve")
    for flag in GATE_FLAGS:
        assert flag in cmd, f"{flag} never reached the solver"
    for flag, want in (("--gate-opener", "enumerate"),
                       ("--gate-pin-secs", "600.0"),
                       ("--gate-sweep-frac", "0.2"),
                       ("--gate-sweep-roots", "8"),
                       ("--gate-sweep-repeats", "3"),
                       ("--gate-band", "12"),
                       ("--gate-sham-roots", "2"),
                       ("--gate-axes", str(axes)),
                       ("--contact-bits", "2")):
        assert cmd[cmd.index(flag) + 1] == want, flag
    # Every forwarded flag is one the solver actually declares — a
    # passthrough that invents a spelling fails the subprocess silently.
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    for flag in GATE_FLAGS:
        assert f'"{flag}"' in src, flag


def test_a_default_chain_forwards_the_inert_arm_and_nothing_optional(
        monkeypatch, tmp_path):
    # Default-off identity at the driver: no sidecar, no attestation, no
    # pin — so the per-level solver parses exactly the arm it always did.
    calls, _recs, _vis = _drive_main(monkeypatch, tmp_path, keys=[(1, 2)],
                                     stall_at=1)
    cmd = next(c[3] for c in calls if c[0] == "solve")
    assert cmd[cmd.index("--gate-opener") + 1] == "off"
    assert "--gate-target-typed" not in cmd
    assert "--gate-axes" not in cmd
    assert "--gate-pin-secs" not in cmd


def test_arming_the_chain_without_a_pin_is_a_hard_error(monkeypatch,
                                                        tmp_path):
    # The same "v1 derives nothing" contract the solver enforces, at the
    # driver — otherwise a campaign arms with an unstated pin and the
    # receipts cannot say what the arm was waiting for.
    with pytest.raises(SystemExit):
        _drive_main(monkeypatch, tmp_path, keys=[(1, 2)], stall_at=0,
                    argv_extra=("--gate-opener", "enumerate"))
