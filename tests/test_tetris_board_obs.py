"""Tetris board-tensor observation tests — verify the playfield RAM
decode produces the expected binary occupancy grid plus scalars for
known synthetic game states."""

from __future__ import annotations

import numpy as np
import pytest

from src.emulation.tile_observations import get_extractor
from src.emulation.tile_observations.tetris import (
    FEATURE_DIM,
    TetrisBoardObservation,
    _BOARD_CELLS,
    _BOARD_COLS,
    _BOARD_ROWS,
    _EMPTY_CELL,
    _RAM_BOARD_BASE,
    _RAM_LEVEL,
    _RAM_LINES_LOW,
)


def _empty_board_ram() -> bytearray:
    """2 KB RAM whose playfield is all empty cells.

    An empty Tetris well is *not* all-zero RAM: every cell holds the
    `0xEF` empty sentinel, so a zeroed buffer would decode as fully
    filled. Build the genuinely-empty state explicitly.
    """
    ram = bytearray(2048)
    for i in range(_BOARD_CELLS):
        ram[_RAM_BOARD_BASE + i] = _EMPTY_CELL
    return ram


def _cell_index(row: int, col: int) -> int:
    return row * _BOARD_COLS + col


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def test_factory_returns_tetris_extractor() -> None:
    ext = get_extractor("tetris_board")
    assert isinstance(ext, TetrisBoardObservation)
    assert ext.feature_dim == FEATURE_DIM


def test_factory_feature_dim_is_202() -> None:
    ext = get_extractor("tetris_board")
    assert ext.feature_dim == 202  # 20*10 board + 2 scalars


def test_factory_unknown_still_raises() -> None:
    with pytest.raises(ValueError):
        get_extractor("not_a_real_encoder")


# ---------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------


def test_geometry_constants() -> None:
    assert _BOARD_ROWS == 20
    assert _BOARD_COLS == 10
    assert _BOARD_CELLS == 200
    assert FEATURE_DIM == _BOARD_CELLS + 2


def test_extract_returns_int8_vector() -> None:
    ext = TetrisBoardObservation()
    out = ext.extract(bytes(_empty_board_ram()))
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.int8


# ---------------------------------------------------------------------
# Board decode
# ---------------------------------------------------------------------


def test_empty_board_is_all_zeros() -> None:
    """A well full of `0xEF` cells decodes to an all-empty grid."""
    ext = TetrisBoardObservation()
    out = ext.extract(bytes(_empty_board_ram()))
    assert np.all(out[:_BOARD_CELLS] == 0)


def test_single_filled_cell() -> None:
    """One non-`0xEF` byte flips exactly one grid cell to filled."""
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    row, col = 5, 3
    ram[_RAM_BOARD_BASE + _cell_index(row, col)] = 0x7B  # any locked block
    out = ext.extract(bytes(ram))
    assert out[_cell_index(row, col)] == 1
    # Every other cell stays empty.
    assert int(out[:_BOARD_CELLS].sum()) == 1


def test_filled_bottom_row() -> None:
    """A full row of blocks lights the whole bottom row and nothing else."""
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    bottom = _BOARD_ROWS - 1
    for col in range(_BOARD_COLS):
        ram[_RAM_BOARD_BASE + _cell_index(bottom, col)] = 0x02
    out = ext.extract(bytes(ram))
    grid = out[:_BOARD_CELLS].reshape(_BOARD_ROWS, _BOARD_COLS)
    assert np.all(grid[bottom] == 1)
    # Rows above remain empty.
    assert np.all(grid[:bottom] == 0)


def test_zero_byte_counts_as_filled() -> None:
    """Only `0xEF` is empty; a `0x00` cell (not the sentinel) is filled.

    This guards the invariant that all-zero RAM is *not* an empty board.
    """
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    ram[_RAM_BOARD_BASE + _cell_index(0, 0)] = 0x00
    out = ext.extract(bytes(ram))
    assert out[_cell_index(0, 0)] == 1


def test_all_zero_ram_decodes_fully_filled() -> None:
    """A zeroed 2 KB buffer has no `0xEF` sentinels, so every board
    cell reads as filled — documents the sentinel semantics explicitly."""
    ext = TetrisBoardObservation()
    out = ext.extract(bytes(bytearray(2048)))
    assert np.all(out[:_BOARD_CELLS] == 1)


# ---------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------


def test_scalar_level() -> None:
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    ram[_RAM_LEVEL] = 9
    out = ext.extract(bytes(ram))
    assert out[_BOARD_CELLS + 0] == 9


def test_scalar_lines_low_byte() -> None:
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    ram[_RAM_LINES_LOW] = 42
    out = ext.extract(bytes(ram))
    assert out[_BOARD_CELLS + 1] == 42


def test_scalars_clamped_to_int8_ceiling() -> None:
    """A large raw byte (e.g. a BCD lines byte) clamps to 127 rather
    than wrapping to a negative int8."""
    ext = TetrisBoardObservation()
    ram = _empty_board_ram()
    ram[_RAM_LEVEL] = 0xFF
    ram[_RAM_LINES_LOW] = 0x99  # BCD 99
    out = ext.extract(bytes(ram))
    assert out[_BOARD_CELLS + 0] == 127
    assert out[_BOARD_CELLS + 1] == 127


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------


def test_short_ram_is_zero_extended() -> None:
    """A truncated buffer must not raise; missing bytes read as 0x00,
    which (not being the sentinel) decode as filled cells."""
    ext = TetrisBoardObservation()
    out = ext.extract(bytes(bytearray(16)))  # far shorter than the board
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.int8
    assert np.all(out[:_BOARD_CELLS] == 1)
