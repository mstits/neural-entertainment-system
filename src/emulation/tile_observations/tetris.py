"""Tetris-specific board-tensor observation decoder.

Reads CPU RAM directly to produce a flat binary occupancy grid of the
NES Tetris playfield plus a couple of cheap game-state scalars. This is
the Tetris analogue of `SMBTileObservation`: it replaces the 4×84×84
pixel CNN input with a compact, semantic RAM decode so the policy can
learn from "which cells are filled" instead of re-deriving the board
from convolutions over falling-block sprites.

Output layout (202 int8 features, all in range [-128, 127]):

* `[0:200]` — 20×10 playfield occupancy grid (row-major, top-down).
  Values:  0 = empty cell, 1 = filled cell (a locked block or the
  active piece). Row 0 is the top of the well; column 0 is the left
  wall. Index for board cell (row, col) is `row * 10 + col`.
* `[200]`  — current level (`$0044`), clamped to [0, 127].
* `[201]`  — lines-cleared low byte, clamped to [0, 127].

Why a board tensor beats the pixel CNN here: Tetris is a fully
observable grid game whose entire relevant state is 200 booleans.
Feeding those directly turns credit assignment ("this cell became
filled, that row cleared") into a handful of features instead of asking
the conv stack to segment 10×20 blocks out of a noisy frame every step.
Same MarI/O rationale that made `smb_tiles` learnable.

RAM addresses:

* Playfield base `$0400`, 20 rows × 10 cols = 200 bytes, row-major.
  A cell is EMPTY when its byte equals ``0xEF`` and FILLED otherwise.
  **UNVERIFIED:** the ``0xEF`` empty-cell sentinel carries the same
  caveat as the bespoke Tetris reward — it matches the commonly cited
  Nintendo-Tetris disassembly ($0400-$04C7, empty == $EF) but has not
  been lock-step confirmed against this emulator's RAM. If board decode
  looks inverted or shifted, this sentinel (or the base address) is the
  first thing to re-check.
* Level `$0044`.
* Lines low byte `$0050` — **UNVERIFIED** address; treated as a rough
  progress scalar only, so a wrong address degrades signal quality but
  cannot corrupt the board grid.

Performance: each call reads 202 RAM bytes and does one vectorized
`!= 0xEF` comparison over a 200-byte slice. Sub-microsecond; far below
per-step emulation time. Pure numpy — no Rust fast path needed (the
whole decode is a single slice + compare, unlike SMB's 169-cell gather).
"""

from __future__ import annotations

import numpy as np


# RAM addresses. Level is given; the board base is the widely documented
# Nintendo-Tetris playfield origin. See module docstring for the
# UNVERIFIED caveats on the empty-cell sentinel and the lines address.
#
# PROVENANCE (2026-08-27 engine purity pass). "Commonly cited
# Nintendo-Tetris disassembly" in the docstring above is an EXTERNAL
# source named in production code, so it is recorded here rather than
# left implicit. What is ours: Tetris-B is on the witness ledger — a
# pre-registered `byte_change` with a rendered LINES-000/SUCCESS frame —
# so a line clear has actually been observed here. What is NOT ours: the
# $0400 base and the 0xEF sentinel have never been lock-step confirmed
# against this emulator's RAM, which the docstring already says twice and
# which is an honest null, not a breach. The Rust twin of `_EMPTY_CELL`
# is tagged `PURITY: UNWITNESSED-EXTERNAL` (registry row
# `tetris_empty_cell`) where it is opt-in behind `board_shaping`; here it
# is used unconditionally, so the caveat matters MORE in this file, not
# less. Behaviour deliberately unchanged: re-pointing these addresses
# would change what every Tetris tile policy observes.
_RAM_BOARD_BASE = 0x0400  # 20×10 playfield, row-major, one byte per cell
_RAM_LEVEL      = 0x0044  # current level number
_RAM_LINES_LOW  = 0x0050  # lines-cleared low byte (UNVERIFIED address)

# Playfield geometry.
_BOARD_ROWS = 20
_BOARD_COLS = 10
_BOARD_CELLS = _BOARD_ROWS * _BOARD_COLS  # 200

# A cell holds this sentinel byte when it is empty. UNVERIFIED — see
# the module docstring; carries the same caveat as the Tetris reward.
_EMPTY_CELL = 0xEF

# Cell encoding in the output grid.
_CELL_EMPTY = 0
_CELL_FILLED = 1

# Total feature vector length: board grid + 2 scalars.
_NUM_SCALARS = 2
FEATURE_DIM = _BOARD_CELLS + _NUM_SCALARS  # 202

# Highest value representable in the int8 output; scalars are clamped to
# this so a large raw byte (e.g. a BCD lines byte up to 0x99) can't wrap
# to a negative int8.
_INT8_MAX = 127

# Smallest RAM length we need to read every field without bounds errors.
_MIN_RAM_LEN = _RAM_BOARD_BASE + _BOARD_CELLS  # 0x04C8 == 1224


class TetrisBoardObservation:
    """Implements the `TileObservation` protocol for NES Tetris.

    Constructed once per worker; `extract(ram)` is called every agent
    step. Internal state-free — the same instance can be reused across
    workers without sync issues.

    The board decode is a single vectorized comparison: slice the 200
    playfield bytes and test each against the empty-cell sentinel. No
    per-cell Python loop, no address arithmetic, no page wrap-around
    (unlike SMB, Tetris' well never scrolls), so a native fast path
    would buy nothing measurable here.
    """

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def extract(self, ram_bytes: bytes) -> np.ndarray:
        """Decode a CPU RAM snapshot into a flat 202-int8 feature vector.

        The playfield maps one RAM byte to one grid cell in row-major
        order, so the 200-byte slice at `$0400` drops straight into the
        first 200 output features after the empty-cell test. Cells read
        as filled whenever the byte differs from ``0xEF``.
        """
        ram = np.frombuffer(ram_bytes, dtype=np.uint8)
        # Zero-extend short buffers so a truncated RAM snapshot can't
        # raise on the board slice or scalar reads. Real CPU RAM is 2 KB,
        # comfortably past the 1224 bytes we touch; this only guards
        # against malformed callers / tiny synthetic fixtures.
        if ram.shape[0] < _MIN_RAM_LEN:
            padded = np.zeros(_MIN_RAM_LEN, dtype=np.uint8)
            padded[: ram.shape[0]] = ram
            ram = padded

        out = np.zeros(FEATURE_DIM, dtype=np.int8)

        # Board: one vectorized compare over the 200-byte playfield slice.
        # Filled == 1 wherever the cell byte is NOT the empty sentinel.
        board = ram[_RAM_BOARD_BASE : _RAM_BOARD_BASE + _BOARD_CELLS]
        out[:_BOARD_CELLS] = (board != _EMPTY_CELL).astype(np.int8)

        # Scalars. Clamp to the int8 ceiling so a large/BCD raw byte
        # can't wrap negative; both are unsigned progress hints.
        out[_BOARD_CELLS + 0] = min(int(ram[_RAM_LEVEL]), _INT8_MAX)
        out[_BOARD_CELLS + 1] = min(int(ram[_RAM_LINES_LOW]), _INT8_MAX)

        return out
