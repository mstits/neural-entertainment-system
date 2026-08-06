"""Tests for scripts/go_explore_solve.py's pure helper functions.

Kept separate from tests/test_go_explore.py (which covers the archive
in src/training/go_explore.py) since this module's Solver class needs a
real ROM/Pool to construct — only the standalone pure functions are
covered here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.go_explore_solve import Solver, update_stall


def _fresh_stall() -> dict:
    return {"last_cells": 0, "last_t": 0.0, "flat_windows": 0}


def test_update_stall_resets_on_growth() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    assert stall == {"last_cells": 10, "last_t": 60.0, "flat_windows": 0}


def test_update_stall_increments_on_flat_window() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    assert stall["flat_windows"] == 1


def test_update_stall_reaches_two_window_stall_threshold() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    update_stall(stall, 10, 180.0)
    assert stall["flat_windows"] == 2


def test_update_stall_recovers_once_new_cells_appear() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    assert stall["flat_windows"] == 1
    update_stall(stall, 11, 180.0)
    assert stall["flat_windows"] == 0
    assert stall["last_cells"] == 11


def test_update_stall_a_shrinking_count_still_counts_as_flat() -> None:
    # An archive never shrinks in practice, but the check is `<=` by
    # design (matches "no NEW cells", not "changed") — pin that here so
    # a future refactor can't accidentally flip it to strict equality
    # and start flagging every static run as instantly stalled.
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 5, 120.0)
    assert stall["flat_windows"] == 1


def _fake_solver(tmp_path, n_cells: int) -> SimpleNamespace:
    # Duck-typed stand-in for Solver: only the attributes progress_line()
    # actually reads. Regression test for a real bug (2026-08-06): the
    # stall-watchdog refactor left a stale bare `stall` reference in
    # progress_line()'s dict literal instead of `self._stall`, which
    # passed every existing test (none called progress_line() with real
    # data) and only surfaced live, on real search runs.
    return SimpleNamespace(
        archive=list(range(n_cells)),
        _stall={"last_cells": 0, "last_t": 0.0, "flat_windows": 0},
        max_area=0, max_gx_in_area={}, max_sect=0,
        n_solutions=0, best_sol_len=0, steps_done=100,
        door_weight=0, transition_macros=[],
        out=tmp_path,
    )


def test_progress_line_does_not_crash_and_reports_stall_state(tmp_path) -> None:
    fake = _fake_solver(tmp_path, n_cells=5)
    Solver.progress_line(fake, 10.0)
    lines = (tmp_path / "progress.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["stall_flat_windows"] == 0
