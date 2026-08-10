"""Tests for src/training/tape_replay.py — the one replay convention.

Everything here runs against a stub pool rather than a real ROM: the
point is the SEQUENCE of controller writes and state loads, which is
what the convention actually is, and which is what silently drifts. The
byte-level proof that the convention is right lives in the minted banks
(`checkpoints/backward_states/*/index.json`), which re-mint identically.

The seam test is the load-bearing one. A chain replay that reuses its
controller buffer across segments replays the previous segment's last
button into the next segment's settle frame; measured on the 30-round
Bubble Bobble chain that perturbs 12541 of 16078 rows, starting at the
first seam. It is invisible in single-tape use (mint, replay_to_demos),
which is exactly why it survived into the proof-of-concept.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.training import tape_replay
from src.training.tape_replay import (
    NOOP,
    RAM_SIZE,
    TapePlayer,
    TapeSegment,
    _level_key,
    machine_from_profile,
)


class StubPool:
    """Records every controller byte and state load in order."""

    def __init__(self):
        self.masks: list[int] = []
        self.loads: list[bytes] = []
        self.saves = 0
        self.shutdowns = 0
        self.frame = 0

    def step_all(self, buf):
        self.masks.append(int(buf[0]))
        self.frame += 1
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = self.frame % 256
        ram[1] = int(buf[0])
        return [(None, None, ram.tobytes())]

    def load_worker_state(self, idx, blob):
        self.loads.append(bytes(blob))

    def save_worker_state(self, idx):
        self.saves += 1
        return b"blob-%d" % self.frame

    def shutdown(self):
        self.shutdowns += 1


def _player(bitmasks=(0x00, 0x80, 0x81, 0x40)) -> TapePlayer:
    """A TapePlayer wired to a stub, bypassing Pool construction."""
    p = TapePlayer.__new__(TapePlayer)
    p.rom, p.frame_skip, p.hw_flags = "stub.nes", 4, ()
    p.bitmasks = list(bitmasks)
    p.pool = StubPool()
    p._buf = np.zeros(1, dtype=np.uint8)
    p._closed = False
    return p


# ---------------------------------------------------------------------
# The convention.
# ---------------------------------------------------------------------

def test_play_leads_with_a_noop_then_one_step_per_action():
    p = _player()
    rows = list(p.play(b"root", [1, 2, 3]))
    assert [step for step, _ in rows] == [0, 1, 2, 3]
    assert p.pool.loads == [b"root"]
    assert p.pool.masks == [NOOP, 0x80, 0x81, 0x40]
    assert all(r.shape == (RAM_SIZE,) and r.dtype == np.uint8 for _, r in rows)


def test_yielded_ram_is_the_memory_before_that_action_runs():
    """`step` doubles as "actions applied so far", which is what makes a
    state minted at step i restorable into "about to run action i"."""
    p = _player()
    rows = dict(p.play(b"root", [1, 2]))
    assert int(rows[0][1]) == NOOP        # after the no-op
    assert int(rows[1][1]) == 0x80        # after action 0
    assert int(rows[2][1]) == 0x81        # after action 1, i.e. the last row


def test_save_state_snapshots_where_play_paused():
    p = _player()
    saved = {}
    for step, _ in p.play(b"root", [1, 2, 3]):
        if step == 2:
            saved["blob"] = p.save_state(step)
            saved["masks_so_far"] = list(p.pool.masks)
    # Saved after the no-op + 2 actions, and before action 2 was written.
    assert saved["masks_so_far"] == [NOOP, 0x80, 0x81]
    assert p.pool.saves == 1 and saved["blob"].startswith(b"blob-")


def test_save_state_failure_names_the_step():
    p = _player()
    p.pool.save_worker_state = lambda idx: None
    with pytest.raises(RuntimeError, match="at step 7"):
        p.save_state(7)


def test_close_is_idempotent():
    p = _player()
    p.close()
    p.close()
    assert p.pool.shutdowns == 1
    with _player() as ctx:
        pass
    assert ctx.pool.shutdowns == 1


# ---------------------------------------------------------------------
# The seam. This is the regression that matters.
# ---------------------------------------------------------------------

def test_every_segment_seam_releases_the_controller(monkeypatch):
    """A chain seam is a settle frame with NO button held.

    Reusing the controller buffer across segments would make the seam
    replay the previous segment's last action instead — the artifact
    measured at 12541/16078 perturbed rows on the Bubble Bobble chain.
    """
    p = _player()
    monkeypatch.setattr(tape_replay, "TapePlayer", lambda **kw: p)
    segs = [TapeSegment(root=b"r0", actions=np.array([1, 2])),
            TapeSegment(root=b"r1", actions=np.array([3, 1])),
            TapeSegment(root=b"r2", actions=np.array([2]))]
    mem, masks, seams = tape_replay.replay_segments(
        segs, rom="stub.nes", bitmasks=p.bitmasks, frame_skip=4)

    assert p.pool.loads == [b"r0", b"r1", b"r2"]
    assert p.pool.masks == [NOOP, 0x80, 0x81,      # segment 0
                            NOOP, 0x40, 0x80,      # segment 1 — NOT 0x81
                            NOOP, 0x81]            # segment 2 — NOT 0x80
    assert seams == [0, 3, 6]
    for s in seams:
        assert masks[s] == 0, f"seam row {s} recorded a held button"
    assert mem.shape == (8, RAM_SIZE)


def test_replay_segments_records_the_bitmask_that_produced_each_row(monkeypatch):
    p = _player()
    monkeypatch.setattr(tape_replay, "TapePlayer", lambda **kw: p)
    segs = [TapeSegment(root=b"r0", actions=np.array([1, 3])),
            TapeSegment(root=b"r1", actions=np.array([2]))]
    _, masks, _ = tape_replay.replay_segments(
        segs, rom="stub.nes", bitmasks=p.bitmasks, frame_skip=4)
    assert list(masks) == [0, 0x80, 0x40, 0, 0x81]


def test_replay_tape_shape_is_one_row_per_action_plus_the_noop(monkeypatch):
    p = _player()
    monkeypatch.setattr(tape_replay, "TapePlayer", lambda **kw: p)
    out = tape_replay.replay_tape(rom="stub.nes", root=b"r", actions=[1, 2, 3],
                                  bitmasks=p.bitmasks, frame_skip=4)
    assert out.shape == (4, RAM_SIZE)
    assert list(out[:, 1]) == [NOOP, 0x80, 0x81, 0x40]


# ---------------------------------------------------------------------
# Provenance helpers.
# ---------------------------------------------------------------------

def test_machine_from_profile_prefers_the_cli_then_rom_path_then_solve_rom():
    base = {"frame_skip": 3, "action_space": [[], ["right"]]}
    rom, fs, bm, hw = machine_from_profile(dict(base, rom_path="roms/a.nes"))
    assert rom.endswith("roms/a.nes") and fs == 3 and len(bm) == 2 and hw == ()
    assert machine_from_profile(dict(base, rom="roms/b.nes"))[0].endswith("b.nes")
    assert machine_from_profile(
        dict(base, solve={"rom": "roms/c.nes"}))[0].endswith("c.nes")
    # An explicit rom beats every profile key.
    assert machine_from_profile(dict(base, rom_path="roms/a.nes"),
                                rom="roms/z.nes")[0].endswith("z.nes")


def test_machine_from_profile_names_the_profile_when_no_rom_is_declared():
    with pytest.raises(SystemExit, match=r"\[mint\] configs/x.yaml"):
        machine_from_profile({"action_space": [[]]},
                             profile_name="configs/x.yaml", who="mint")


def test_level_key_sorts_levels_naturally():
    names = ["lvl_1-10", "lvl_2-1", "lvl_1-2", "lvl_1-1", "lvl_bonus"]
    assert sorted(names, key=_level_key) == [
        "lvl_1-1", "lvl_1-2", "lvl_1-10", "lvl_2-1", "lvl_bonus"]


def test_root_state_for_falls_back_to_the_run_dir(tmp_path):
    run = tmp_path / "run"
    (run / "lvl_1-1" / "solutions").mkdir(parents=True)
    blob = run / "entrance.state"
    blob.write_bytes(b"x")
    sidecar = run / "lvl_1-1" / "solutions" / "sol_000.json"
    sidecar.write_text(json.dumps({"root_state": "/gone/entrance.state"}))
    assert tape_replay.root_state_for(sidecar, run) == blob
    sidecar.write_text(json.dumps({"root_state": "/gone/missing.state"}))
    with pytest.raises(FileNotFoundError):
        tape_replay.root_state_for(sidecar, run)


def test_solution_paths_pairs_the_tape_with_its_sidecar(tmp_path):
    act, side = tape_replay.solution_paths(tmp_path, "4-2", 3)
    assert act.name == "sol_003.actions.npy" and side.name == "sol_003.json"
    assert act.parent == tmp_path / "lvl_4-2" / "solutions"


def test_resolve_chain_shuffle_seed_builds_a_matched_length_control(tmp_path):
    """The negative control keeps every root and every action count — it
    is the same trajectory length, differently ordered."""
    acts = tmp_path / "a.npy"
    np.save(acts, np.array([0, 1, 2, 3, 1, 2], dtype=np.int64))
    root = tmp_path / "r.state"
    root.write_bytes(b"root")
    spec = [{"root": str(root), "actions": str(acts)}]
    plain = tape_replay.resolve_chain(spec)
    shuffled = tape_replay.resolve_chain(spec, shuffle_seed=7)
    assert len(plain[0]) == len(shuffled[0]) == 6
    assert sorted(plain[0].actions) == sorted(shuffled[0].actions)
    assert not np.array_equal(plain[0].actions, shuffled[0].actions)
    assert shuffled[0].root == plain[0].root
    assert "shuffled7" in shuffled[0].label
    # Deterministic for a given seed.
    assert np.array_equal(shuffled[0].actions,
                          tape_replay.resolve_chain(spec, shuffle_seed=7)[0].actions)


def test_resolve_chain_rejects_an_empty_spec():
    with pytest.raises(ValueError, match="empty chain"):
        tape_replay.resolve_chain([])
