"""Tests for scripts/rg1e_edge_validity.py's pure edge-state resolution.

Only `resolve_exemplar_state` is covered here -- everything else in this
script drives a real `Solver`/`Pool` (ROM I/O) and is exercised by hand
against real RG-1 runs, not by this suite.
"""

from __future__ import annotations

from src.training.go_explore import Cell
from scripts.rg1e_edge_validity import resolve_exemplar_state


def test_resolve_exemplar_state_prefers_frozen_copy_over_drifted_cell():
    """`exemplar_state` -- the frozen bytes `RoomIndex.record_edge` copies
    at edge-commit time -- must win even when the archive's mutable cell
    at `exemplar_cell` has since been overwritten by GoExploreArchive's
    domination logic (a later, unrelated, higher-scoring visit to the same
    key). Falling back to `cells[...].state` here is exactly the
    archive-domination-drift confound this script's docstring exists to
    close."""
    key = ("sect0", "tb0", "kk0")
    cells = {key: Cell(key=key, state=b"DRIFTED-by-a-later-visit",
                       best_score=1.0, best_steps=1, visits=7)}
    e = {"exemplar_cell": key, "exemplar_actions": [0, 1],
        "exemplar_state": b"FROZEN-at-edge-commit-time"}

    assert resolve_exemplar_state(e, cells) == b"FROZEN-at-edge-commit-time"


def test_resolve_exemplar_state_falls_back_when_frozen_copy_absent():
    """An edge banked before the frozen-copy fix existed has
    `exemplar_state=None` on disk (RoomIndex.load's documented fallback
    signal) -- resolve_exemplar_state must still return the archive cell's
    state for that edge, same as the old mutable-key behavior."""
    key = ("sect1", "tb1", "kk1")
    cells = {key: Cell(key=key, state=b"only-copy-that-exists",
                       best_score=1.0, best_steps=1, visits=1)}
    e = {"exemplar_cell": key, "exemplar_actions": [0],
        "exemplar_state": None}

    assert resolve_exemplar_state(e, cells) == b"only-copy-that-exists"


def test_resolve_exemplar_state_none_when_cell_missing_and_no_frozen_copy():
    e = {"exemplar_cell": ("gone",), "exemplar_actions": [0],
        "exemplar_state": None}

    assert resolve_exemplar_state(e, {}) is None
