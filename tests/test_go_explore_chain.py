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

    def run(out_path, hw_flags=None, advance_at=3):
        clock.n = 0
        monkeypatch.setattr(
            chain, "make_game", lambda profile: _FakeGame(clock, advance_at))
        kw = {} if hw_flags is None else {"hw_flags": hw_flags}
        return chain.extract_next_entrance(
            PROFILE, b"ROOT", [1] * 12, out_path,
            settle=1, stable_for=1, settle_cap=4, **kw)

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
