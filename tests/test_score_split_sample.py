"""The standing selection rule, pinned to the numbers it was read against.

The capacity fork (DIRECTION §4.2) was adjudicated on v27 split-sample
best-of-4 = 0.500 vs v28's banked 0.670. If the estimator's arithmetic
drifts, those readings silently change meaning — so the banked v27 ladder
CSV is the fixture and the exact seed scores are the assertion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from score_split_sample import (  # noqa: E402
    CURSE,
    score_ladder,
    score_run,
)

V27_CSV = REPO / "docs" / "receipts" / "v27_corrected_ladder" / "ladder.csv"


def test_reproduces_the_banked_v27_reading_exactly():
    res = score_ladder(V27_CSV)
    by = {s.run: s for s in res.seeds}
    assert pytest.approx(by["mario_1_1_v27_recovery_seed0"].score) == 0.110
    assert pytest.approx(by["mario_1_1_v27_recovery_seed1"].score) == 0.500
    assert pytest.approx(by["mario_1_1_v27_recovery_seed2"].score) == 0.460
    assert pytest.approx(by["mario_1_1_v27_recovery_seed3"].score) == 0.460
    assert pytest.approx(res.best) == 0.500
    assert res.best_run == "mario_1_1_v27_recovery_seed1"
    assert pytest.approx(res.best_adj) == 0.500 - CURSE


def test_selection_and_scoring_use_opposite_eval_seeds():
    """The whole point: the episode that selects a checkpoint never scores
    it. A grid where the es0 peak is a fluke (high on es0, low on es1)
    must NOT inherit the fluke."""
    grid = {(10, 0): 0.9, (10, 1): 0.1,   # es0 fluke
            (20, 0): 0.4, (20, 1): 0.5}
    s = score_run(grid, "r")
    assert s.sel_on_es0 == 10 and s.scored_on_es1 == pytest.approx(0.1)
    assert s.sel_on_es1 == 20 and s.scored_on_es0 == pytest.approx(0.4)
    assert s.score == pytest.approx(0.25)  # not 0.9's neighborhood


def test_ties_break_to_the_later_iteration():
    grid = {(10, 0): 0.5, (10, 1): 0.2,
            (20, 0): 0.5, (20, 1): 0.4}
    s = score_run(grid, "r")
    assert s.sel_on_es0 == 20, "equal es0 rates must select the later iter"


def test_a_hole_in_the_grid_refuses_rather_than_guesses():
    grid = {(10, 0): 0.5, (10, 1): 0.2, (20, 0): 0.5}  # (20,1) missing
    with pytest.raises(ValueError, match="missing eval seed"):
        score_run(grid, "r")


def test_a_bad_status_row_poisons_its_run():
    import csv as _csv
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        w = _csv.writer(f)
        w.writerow(["run", "iter", "eval_seed", "clear_rate", "status"])
        w.writerow(["r", 10, 0, 0.5, "ok"])
        w.writerow(["r", 10, 1, 0.0, "no_rom"])
        path = f.name
    with pytest.raises(ValueError, match="no_rom"):
        score_ladder(Path(path))


def test_the_estimator_is_not_a_constant():
    """Anti-vacuity: three grids, three different answers."""
    flat = {(10, 0): 0.0, (10, 1): 0.0, (20, 0): 0.0, (20, 1): 0.0}
    mid = {(10, 0): 0.5, (10, 1): 0.5, (20, 0): 0.1, (20, 1): 0.1}
    high = {(10, 0): 0.9, (10, 1): 0.9, (20, 0): 0.1, (20, 1): 0.1}
    scores = {score_run(g, "r").score for g in (flat, mid, high)}
    assert scores == {0.0, 0.5, 0.9}
