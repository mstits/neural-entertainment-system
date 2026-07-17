"""Tests for the Go-Explore cell archive (src/training/go_explore.py)."""

from __future__ import annotations

import pytest

import src.training.go_explore as go_explore
from src.training.go_explore import (
    EXPLORE_AFTER_FIRST_CLEAR,
    START_WINDOW_FRAMES,
    GoExploreArchive,
    filter_learnable,
    horizontal_neighbors,
    keep_exploring,
    learnable,
    ram_bytes_cell,
    ram_downsample_cell,
    start_window,
)


def _ram(**kv: int) -> bytes:
    """Build a 2 KB RAM snapshot with specific bytes set. Keys are hex
    address strings (e.g. "0x0760") so int(addr, 0) auto-detects base."""
    buf = bytearray(2048)
    for addr, val in kv.items():
        buf[int(addr, 0)] = val
    return bytes(buf)


# ---- cell functions -------------------------------------------------


def test_ram_bytes_cell_keys_on_selected_addresses() -> None:
    fn = ram_bytes_cell([0x0760, 0x006D])
    assert fn(_ram(**{"0x0760": 2, "0x006D": 5})) == (2, 5)
    # Bytes outside the address list don't affect the key.
    assert fn(_ram(**{"0x0760": 2, "0x006D": 5, "0x0400": 99})) == (2, 5)


def test_ram_bytes_cell_bucketing_collapses_nearby_positions() -> None:
    fn = ram_bytes_cell([0x0086], bucket=16)
    # x=0..15 all map to bucket 0; x=16 to bucket 1.
    assert fn(_ram(**{"0x0086": 5})) == (0,)
    assert fn(_ram(**{"0x0086": 15})) == (0,)
    assert fn(_ram(**{"0x0086": 16})) == (1,)


def test_ram_downsample_cell_is_game_agnostic() -> None:
    fn = ram_downsample_cell(stride=64, bucket=16)
    key = fn(_ram())
    # 2048 / 64 = 32 sampled bytes.
    assert len(key) == 2048 // 64
    assert all(v == 0 for v in key)


# ---- archive: insertion + domination --------------------------------


def test_new_ram_creates_new_cell() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    assert arc.record(_ram(**{"0x0760": 1}), b"stateA", score=10.0, steps=100)
    assert len(arc) == 1
    assert arc.total_new_cells == 1


def test_higher_score_dominates_and_replaces_state() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    ram = _ram(**{"0x0760": 1})
    arc.record(ram, b"worse", score=10.0, steps=100)
    improved = arc.record(ram, b"better", score=20.0, steps=200)
    assert improved is True
    cell = next(iter(arc.cells.values()))
    assert cell.best_score == 20.0
    assert cell.state == b"better"
    assert cell.visits == 2
    assert arc.total_improvements == 1


def test_lower_score_does_not_replace() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    ram = _ram(**{"0x0760": 1})
    arc.record(ram, b"good", score=20.0, steps=100)
    improved = arc.record(ram, b"bad", score=5.0, steps=50)
    assert improved is False
    cell = next(iter(arc.cells.values()))
    assert cell.state == b"good"
    assert cell.best_score == 20.0


def test_equal_score_fewer_steps_dominates() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    ram = _ram(**{"0x0760": 1})
    arc.record(ram, b"slow", score=10.0, steps=200)
    improved = arc.record(ram, b"fast", score=10.0, steps=120)
    assert improved is True
    cell = next(iter(arc.cells.values()))
    assert cell.state == b"fast"
    assert cell.best_steps == 120


# ---- selection ------------------------------------------------------


def test_select_on_empty_archive_returns_none() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    assert arc.select_return_cell() is None
    assert arc.select_return_states(4) == [None, None, None, None]


def test_selection_favors_frontier_cells() -> None:
    """A never-explored (frontier) cell carries a 2x weight bonus and a
    low chosen-count, so it is strongly preferred. The count-based weight
    then self-balances (Go-Explore spreads exploration across cells rather
    than fixating), so the less-chosen cell keeps getting picked more."""
    arc = GoExploreArchive(ram_bytes_cell([0x0760]), seed=0)
    arc.record(_ram(**{"0x0760": 1}), b"A", score=1.0, steps=10)
    arc.record(_ram(**{"0x0760": 2}), b"B", score=1.0, steps=10)
    # Make cell A explored + heavily chosen so only B is on the frontier.
    a_cell = arc.cells[(1,)]
    a_cell.explored = True
    a_cell.times_chosen = 50
    # The very first return target is the fresh frontier cell B.
    assert arc.select_return_cell().key == (2,)
    # Across many draws, B (which started far behind on chosen-count) is
    # picked more often than the heavily-chosen A.
    picks = [arc.select_return_cell().key for _ in range(200)]
    assert picks.count((2,)) > picks.count((1,))


def test_selection_marks_cell_explored_and_chosen() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]), seed=0)
    arc.record(_ram(**{"0x0760": 1}), b"A", score=1.0, steps=10)
    assert arc.frontier_size() == 1
    cell = arc.select_return_cell()
    assert cell.explored is True
    assert cell.times_chosen == 1
    assert arc.frontier_size() == 0


def test_select_return_states_returns_blobs() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]), seed=0)
    arc.record(_ram(**{"0x0760": 1}), b"A", score=1.0, steps=10)
    arc.record(_ram(**{"0x0760": 2}), b"B", score=1.0, steps=10)
    states = arc.select_return_states(8)
    assert len(states) == 8
    assert all(s in (b"A", b"B") for s in states)


# ---- stats + persistence -------------------------------------------


def test_stats_reflect_archive_state() -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    arc.record(_ram(**{"0x0760": 1}), b"A", score=5.0, steps=10)
    arc.record(_ram(**{"0x0760": 2}), b"B", score=9.0, steps=10)
    arc.record(_ram(**{"0x0760": 1}), b"A2", score=7.0, steps=10)  # improve cell 1
    s = arc.stats()
    assert s["cells"] == 2
    assert s["best_score"] == 9.0
    assert s["new_cells"] == 2
    assert s["improvements"] == 1
    assert s["records"] == 3


def test_save_load_roundtrip(tmp_path) -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    arc.record(_ram(**{"0x0760": 1}), b"A", score=5.0, steps=10)
    arc.record(_ram(**{"0x0760": 2}), b"B", score=9.0, steps=20)
    path = tmp_path / "archive.pkl"
    arc.save(path)
    assert (tmp_path / "archive.stats.json").exists()

    arc2 = GoExploreArchive(ram_bytes_cell([0x0760]))
    arc2.load(path)
    assert len(arc2) == 2
    assert arc2.cells[(2,)].state == b"B"
    assert arc2.cells[(2,)].best_score == 9.0
    assert arc2.cells[(1,)].best_steps == 10


# ---- QW3: neighbor-aware selection weight (W_location) --------------


def test_horizontal_neighbors_vary_last_component_only() -> None:
    # (area..., x_bucket) convention: same area, x bucket +/- 1.
    assert horizontal_neighbors((1, 0, 5)) == ((1, 0, 4), (1, 0, 6))
    assert horizontal_neighbors((7,)) == ((6,), (8,))


def test_horizontal_neighbors_unstructured_keys_have_no_adjacency() -> None:
    assert horizontal_neighbors("not-a-tuple") == ()
    assert horizontal_neighbors(()) == ()
    assert horizontal_neighbors((1, "x")) == ()


def test_w_location_boosts_cells_with_missing_neighbors() -> None:
    """Paper form (Ecoffet 2021): W_location = (2 - h)/10 with h = number
    of horizontal neighbors present. Three contiguous cells: the middle
    one has both neighbors (h=2, bonus 0), the edges have one (h=1,
    bonus 0.1). Count/frontier terms are equalized so only W_location
    differs."""
    arc = GoExploreArchive(ram_bytes_cell([0x0086]))
    for x in (1, 2, 3):
        arc.record(_ram(**{"0x0086": x}), b"s", score=1.0, steps=10)
    for c in arc.cells.values():
        c.explored = True  # frontier bonus off; times_chosen equal
    w_edge = arc._selection_weight(arc.cells[(1,)])
    w_mid = arc._selection_weight(arc.cells[(2,)])
    assert w_edge == pytest.approx(w_mid + 0.1)
    # An isolated cell misses both neighbors: h=0, bonus 0.2.
    arc.record(_ram(**{"0x0086": 100}), b"s", score=1.0, steps=10)
    lone = arc.cells[(100,)]
    lone.explored = True
    assert arc._selection_weight(lone) == pytest.approx(w_mid + 0.2)


def test_w_location_coefficient_is_zeroable(monkeypatch) -> None:
    arc = GoExploreArchive(ram_bytes_cell([0x0086]))
    arc.record(_ram(**{"0x0086": 1}), b"s", score=1.0, steps=10)
    arc.record(_ram(**{"0x0086": 2}), b"s", score=1.0, steps=10)
    for c in arc.cells.values():
        c.explored = True
    monkeypatch.setattr(go_explore, "W_LOCATION_COEF", 0.0)
    w1 = arc._selection_weight(arc.cells[(1,)])
    w2 = arc._selection_weight(arc.cells[(2,)])
    assert w1 == pytest.approx(1.0)  # pure count-based weight
    assert w2 == pytest.approx(1.0)


def test_custom_neighbor_fn_is_injectable() -> None:
    # A no-adjacency neighbor_fn disables the bonus per cell.
    arc = GoExploreArchive(ram_bytes_cell([0x0086]), neighbor_fn=lambda k: ())
    arc.record(_ram(**{"0x0086": 1}), b"s", score=1.0, steps=10)
    cell = arc.cells[(1,)]
    cell.explored = True
    assert arc._selection_weight(cell) == pytest.approx(1.0)


# ---- QW4: keep exploring after the first clear -----------------------


def test_keep_exploring_does_not_stop_at_first_clear() -> None:
    assert EXPLORE_AFTER_FIRST_CLEAR is True
    assert keep_exploring(0, 10) is True
    assert keep_exploring(1, 10) is True  # first clear is NOT terminal
    assert keep_exploring(9, 10) is True
    assert keep_exploring(10, 10) is False  # demo budget met
    assert keep_exploring(11, 10) is False


def test_keep_exploring_legacy_flag_stops_at_first_clear() -> None:
    assert keep_exploring(0, 10, explore_after_first_clear=False) is True
    assert keep_exploring(1, 10, explore_after_first_clear=False) is False


def test_post_clear_shorter_trajectory_keeps_improving_elite() -> None:
    """The archive contract behind QW4: at equal progress, later (post-
    clear) records with fewer steps still dominate — success never
    freezes an elite."""
    arc = GoExploreArchive(ram_bytes_cell([0x0760]))
    ram = _ram(**{"0x0760": 4})
    arc.record(ram, b"first_clear", score=100.0, steps=900)
    assert arc.record(ram, b"shorter_clear", score=100.0, steps=700) is True
    assert arc.record(ram, b"shortest", score=100.0, steps=500) is True
    cell = arc.cells[(4,)]
    assert cell.state == b"shortest"
    assert cell.best_steps == 500


# ---- backward-algorithm start window ---------------------------------


def test_start_window_slices_trailing_window_of_trajectory() -> None:
    traj = [f"blob{i}" for i in range(100)]
    # 160 frames at frame_skip 4 = 40 steps back from the anchor.
    win = start_window(traj, 60, window_frames=160, frames_per_step=4)
    assert win == traj[20:61]
    assert win[-1] == "blob60"  # anchor inclusive
    assert len(win) == 41


def test_start_window_defaults_to_nature_160_frames() -> None:
    assert START_WINDOW_FRAMES == 160
    traj = list(range(500))
    win = start_window(traj, 400)
    assert win == list(range(240, 401))


def test_start_window_clamps_at_trajectory_start() -> None:
    traj = list(range(10))
    win = start_window(traj, 3, window_frames=160, frames_per_step=4)
    assert win == [0, 1, 2, 3]


def test_start_window_negative_anchor_and_bounds() -> None:
    traj = list(range(50))
    win = start_window(traj, -1, window_frames=8, frames_per_step=4)
    assert win == [47, 48, 49]
    with pytest.raises(IndexError):
        start_window(traj, 50)
    with pytest.raises(IndexError):
        start_window(traj, -51)


def test_start_window_over_archive_cells_by_best_steps() -> None:
    """Archive form: cells ordered by best_steps; the window behind the
    anchor cell (in steps) is the candidate start set for backward
    robustification."""
    arc = GoExploreArchive(ram_bytes_cell([0x0086]))
    for x, steps in ((1, 10), (2, 50), (3, 90), (4, 100)):
        arc.record(_ram(**{"0x0086": x}), f"s{x}".encode(), score=float(x),
                   steps=steps)
    win = start_window(arc, -1, window_frames=60, frames_per_step=1)
    assert [c.best_steps for c in win] == [50, 90, 100]
    assert win[-1].state == b"s4"  # anchor = deepest cell
    # The raw cells dict is accepted too.
    win2 = start_window(arc.cells, -1, window_frames=60, frames_per_step=1)
    assert [c.key for c in win2] == [c.key for c in win]


def test_start_window_is_pure() -> None:
    traj = list(range(20))
    before = list(traj)
    start_window(traj, 10, window_frames=8, frames_per_step=1)
    assert traj == before
    with pytest.raises(ValueError):
        start_window(traj, 10, window_frames=-1)
    with pytest.raises(ValueError):
        start_window(traj, 10, frames_per_step=0)


# ---- Florensa learnability band --------------------------------------


def test_learnable_band_is_inclusive() -> None:
    assert learnable(0.1) is True
    assert learnable(0.5) is True
    assert learnable(0.9) is True
    assert learnable(0.05) is False  # hopeless
    assert learnable(0.95) is False  # mastered
    assert learnable(0.0) is False
    assert learnable(1.0) is False


def test_learnable_custom_bounds_and_validation() -> None:
    assert learnable(0.05, lo=0.02, hi=0.5) is True
    with pytest.raises(ValueError):
        learnable(0.5, lo=0.9, hi=0.1)
    with pytest.raises(ValueError):
        learnable(0.5, lo=-0.1)


def test_filter_learnable_retires_mastered_and_hopeless_starts() -> None:
    starts = ["hopeless", "hard", "mid", "easy", "mastered"]
    p = [0.0, 0.15, 0.5, 0.85, 0.99]
    kept = filter_learnable(starts, p)
    assert kept == ["hard", "mid", "easy"]
    # Inputs are not mutated.
    assert len(starts) == 5 and len(p) == 5


def test_filter_learnable_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        filter_learnable(["a", "b"], [0.5])
