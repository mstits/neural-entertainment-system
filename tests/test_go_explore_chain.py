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
    that hw flags land BEFORE reset_all()."""

    instances: list = []
    clock: _Clock = _Clock()

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

    def step_all(self, x):
        _FakePool.clock.n += 1
        return [(None, None, bytearray(2048))]

    def save_worker_state(self, wid):
        return b"ENTRANCE-BLOB"

    def shutdown(self):
        self.calls.append("shutdown")


@pytest.fixture
def replay(monkeypatch):
    """Returns a runner for extract_next_entrance against the fakes.
    `advance_at` = the step on which the level key moves forward; a huge
    value means the replay never sees a transition."""
    clock = _Clock()
    _FakePool.instances.clear()
    _FakePool.clock = clock
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
    assert out.read_bytes() == b"ENTRANCE-BLOB"
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
