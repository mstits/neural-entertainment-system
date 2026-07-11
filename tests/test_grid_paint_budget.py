"""High-tile-count grid paint budget — guards the Qt main-thread cost.

Above FAST_PAINT_THRESHOLD tiles the grid (a) downgrades the final
scale pass to nearest-neighbor and (b) repaints alternating tile
parities per tick (~15 fps/tile). Small grids keep the full-quality
30 fps path. These tests pin both behaviors so a refactor can't
silently re-enable the smooth pass at 64 tiles (a real Qt-thread
saturation risk for stream layouts).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.gui.emulator_grid import (  # noqa: E402
    EmulatorGrid,
    FAST_PAINT_THRESHOLD,
)

_app = QApplication.instance() or QApplication([])


def _provider_for(n: int, seq: list[int]):
    frames = [
        np.full((240, 256, 3), i % 255, dtype=np.uint8) for i in range(n)
    ]

    def provider():
        seq[0] += 1
        return [(frames[i], seq[0]) for i in range(n)]

    return provider


def _painted(grid: EmulatorGrid) -> list[int]:
    return [
        i for i, t in enumerate(grid.frame_labels)
        if t._last_frame_id is not None
    ]


def test_small_grid_paints_every_tile_per_tick():
    grid = EmulatorGrid(num_instances=4)
    assert all(not t._fast_paint for t in grid.frame_labels)
    grid.set_frame_provider(_provider_for(4, [0]))
    grid._refresh_frames()
    assert _painted(grid) == [0, 1, 2, 3]
    grid.close()


def test_large_grid_budgets_paints_across_two_ticks():
    n = FAST_PAINT_THRESHOLD + 4
    grid = EmulatorGrid(num_instances=n)
    assert all(t._fast_paint for t in grid.frame_labels)
    grid.set_frame_provider(_provider_for(n, [0]))

    grid._refresh_frames()
    first = set(_painted(grid))
    # Exactly one parity class painted on the first tick.
    assert first, "first tick painted nothing"
    assert len(first) == n // 2
    assert len({i & 1 for i in first}) == 1

    grid._refresh_frames()
    # Second tick covers the other parity — every tile painted once.
    assert len(_painted(grid)) == n
    grid.close()
