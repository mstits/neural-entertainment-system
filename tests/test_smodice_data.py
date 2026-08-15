"""smodice_data commit logic — the two defects gen_iq_transitions.py carried.

The IQ-Learn transition set had done=0 on every row because the window
commit gate discarded non-surviving windows wholesale (survivorship bias),
and a window that merely ran out its 60 steps was indistinguishable from a
real terminal. These tests pin the corrected semantics with the emulator
fully stubbed (no ROM, no Pool): a death window keeps its terminal
transition (done=1, absorbing), a survived window is a timeout (done=0,
truncated=1 on its last row), the forward-progress test is a LABEL not a
discard gate, and the packed output is a ~50/50 progressed/failed
stratification with composition counts + solution sha256 provenance in the
npz metadata.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.smodice_data import (
    classify_window,
    pack_windows,
    record_window,
    save_dataset,
    solution_provenance,
    stratify_windows,
    window_kind,
)

FDIM = 8


class StubWorker:
    """Deterministic fake emulator: obs is a step-counter vector; dies at
    step `die_at` (1-indexed) if given, else survives every step."""

    def __init__(self, die_at=None):
        self.t = 0
        self.die_at = die_at

    def step(self, action):
        self.t += 1
        obs = np.full(FDIM, self.t % 100, dtype=np.int8)
        dead = self.die_at is not None and self.t >= self.die_at
        return obs, dead


def _window(die_at=None, n_actions=6, wid=0, progressed=False):
    w = StubWorker(die_at=die_at)
    rows, died = record_window(np.zeros(FDIM, dtype=np.int8),
                               [1] * n_actions, w.step)
    prog = progressed and not died
    return {"id": wid, "rows": rows, "progressed": prog,
            "kind": window_kind(died, prog)}


def test_death_window_keeps_terminal_transition():
    w = StubWorker(die_at=3)
    rows, died = record_window(np.zeros(FDIM, dtype=np.int8),
                               [1] * 10, w.step)
    assert died
    assert len(rows) == 3
    assert [r[3] for r in rows] == [0, 0, 1]
    assert [r[4] for r in rows] == [0, 0, 0]


def test_timeout_window_is_truncated_not_done():
    w = StubWorker(die_at=None)
    rows, died = record_window(np.zeros(FDIM, dtype=np.int8),
                               [1] * 6, w.step)
    assert not died
    assert len(rows) == 6
    assert [r[3] for r in rows] == [0] * 6
    assert [r[4] for r in rows] == [0, 0, 0, 0, 0, 1]


def test_transitions_chain_and_actions_recorded():
    w = StubWorker(die_at=None)
    actions = [2, 0, 5, 1]
    rows, _ = record_window(np.zeros(FDIM, dtype=np.int8), actions, w.step)
    assert [r[1] for r in rows] == actions
    for prev, nxt in zip(rows, rows[1:]):
        np.testing.assert_array_equal(prev[2], nxt[0])


def test_progress_is_label_not_gate():
    assert classify_window(died=False, d_start=100.0, d_end=98.0)
    assert not classify_window(died=False, d_start=100.0, d_end=99.5)
    assert not classify_window(died=True, d_start=100.0, d_end=50.0)
    assert window_kind(True, False) == "death"
    assert window_kind(False, True) == "progressed"
    assert window_kind(False, False) == "stall"


def test_stratify_fifty_fifty_with_deaths_and_stalls():
    windows = [_window(wid=i, progressed=True) for i in range(10)]
    windows += [_window(die_at=2, wid=10 + i) for i in range(3)]
    windows += [_window(wid=13, progressed=False)]
    kept, counts = stratify_windows(windows, np.random.default_rng(0))
    assert counts["progressed_total"] == 10
    assert counts["failed_total"] == 4
    assert counts["deaths_total"] == 3
    assert counts["stalls_total"] == 1
    assert counts["kept_progressed"] == 4
    assert counts["kept_failed"] == 4
    assert sum(1 for w in kept if w["progressed"]) == 4
    assert sum(1 for w in kept if not w["progressed"]) == 4
    arrays = pack_windows(kept)
    assert int(arrays["done"].sum()) == 3


def test_stratify_requires_both_classes():
    windows = [_window(wid=i, progressed=True) for i in range(4)]
    with pytest.raises(ValueError):
        stratify_windows(windows, np.random.default_rng(0))


def test_pack_schema_and_flags():
    windows = [_window(wid=0, progressed=True), _window(die_at=2, wid=1)]
    arrays = pack_windows(windows)
    assert arrays["state"].dtype == np.int8
    assert arrays["state"].shape == (8, FDIM)
    assert arrays["next_state"].shape == (8, FDIM)
    assert arrays["action"].dtype == np.int64
    assert arrays["done"].dtype == np.int8
    assert arrays["truncated"].dtype == np.int8
    assert int(arrays["done"].sum()) == 1
    assert int(arrays["truncated"].sum()) == 1
    assert not np.any((arrays["done"] == 1) & (arrays["truncated"] == 1))
    np.testing.assert_array_equal(arrays["window_id"],
                                  np.array([0] * 6 + [1] * 2))
    np.testing.assert_array_equal(arrays["window_progressed"],
                                  np.array([1] * 6 + [0] * 2, dtype=np.int8))
    np.testing.assert_array_equal(arrays["is_expert_window"],
                                  np.ones(8, dtype=np.int8))


def test_save_dataset_roundtrip_with_provenance(tmp_path):
    sol = tmp_path / "sol_000.npy"
    np.save(sol, np.arange(5, dtype=np.int64))
    prov = solution_provenance([sol])
    assert prov[str(sol)] == hashlib.sha256(sol.read_bytes()).hexdigest()

    windows = [_window(wid=0, progressed=True), _window(die_at=2, wid=1)]
    counts = {"windows_total": 2, "kept_progressed": 1, "kept_failed": 1}
    out = tmp_path / "data.npz"
    save_dataset(out, windows, counts, prov)

    d = np.load(out)
    for key in ("state", "action", "next_state", "done", "truncated",
                "is_expert_window", "window_id", "window_progressed",
                "meta_json"):
        assert key in d.files
    meta = json.loads(str(d["meta_json"]))
    assert meta["counts"]["windows_total"] == 2
    assert meta["provenance"][str(sol)] == prov[str(sol)]
