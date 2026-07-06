"""Tests for the Go-Explore cell archive (src/training/go_explore.py)."""

from __future__ import annotations

from src.training.go_explore import (
    GoExploreArchive,
    ram_bytes_cell,
    ram_downsample_cell,
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
