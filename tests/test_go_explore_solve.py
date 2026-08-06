"""Tests for scripts/go_explore_solve.py's pure helper functions.

Kept separate from tests/test_go_explore.py (which covers the archive
in src/training/go_explore.py) since this module's Solver class needs a
real ROM/Pool to construct — only the standalone pure functions are
covered here.
"""

from __future__ import annotations

from scripts.go_explore_solve import update_stall


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
