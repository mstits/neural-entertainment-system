"""LIVE STATS GRAPH — search progress over time, for the stream.

Cells discovered and frontier depth are the two numbers that actually say
"is the search working" during a long wall-grinding stretch where every
worker tile looks identical. This renders them as two rolling trend
lines (independently min-max normalized so both stay visible regardless
of absolute scale) into a small RGBA panel, pure numpy — same
never-touch-the-archive, small-fixed-buffer discipline as show_heatmap.py.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

_BG_RGB = np.array([6, 7, 12], dtype=np.float32)
_GRID_RGB = np.array([28, 30, 40], dtype=np.float32)
_GX_RGB = np.array([90, 255, 190], dtype=np.float32)      # frontier depth
_CELLS_RGB = np.array([120, 170, 255], dtype=np.float32)  # cells discovered


class StatsGraphRenderer:
    """Feed with `push(frontier_gx, cells)`; call `render()` for an (H, W, 4)
    uint8 RGBA array. Cost is O(width) per render, independent of history
    length (a fixed-size rolling deque)."""

    def __init__(self, width: int = 512, height: int = 96, *,
                 max_points: int = 240):
        self.width = int(width)
        self.height = int(height)
        self.max_points = int(max_points)
        self.gx_hist: deque[float] = deque(maxlen=self.max_points)
        self.cells_hist: deque[float] = deque(maxlen=self.max_points)
        self._surface = np.empty((self.height, self.width, 4), dtype=np.uint8)
        self._surface[..., 3] = 255

    def push(self, frontier_gx: float, cells: float) -> None:
        self.gx_hist.append(float(frontier_gx))
        self.cells_hist.append(float(cells))

    def reset(self) -> None:
        self.gx_hist.clear()
        self.cells_hist.clear()

    @staticmethod
    def _normalize(hist: deque) -> Optional[np.ndarray]:
        if len(hist) < 2:
            return None
        vals = np.asarray(hist, dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            return np.full_like(vals, 0.5)
        return (vals - lo) / (hi - lo)

    def _draw_line(self, grid: np.ndarray, norm: np.ndarray,
                    color: np.ndarray, pad: int = 6) -> None:
        n = len(norm)
        usable_w = self.width - 1
        xs = (np.linspace(0, usable_w, n)).astype(np.int32)
        plot_h = self.height - 2 * pad
        ys = (self.height - pad - norm * plot_h).astype(np.int32)
        ys = np.clip(ys, 0, self.height - 1)
        # Connect consecutive points with a thin vertical fill per column
        # span (cheap: O(width), no per-pixel Bresenham needed at this size).
        for i in range(n - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            if x1 <= x0:
                continue
            col_ys = np.linspace(y0, y1, x1 - x0 + 1).astype(np.int32)
            for dx, y in enumerate(col_ys):
                x = x0 + dx
                grid[max(0, y - 1):y + 2, x] = color

    def render(self) -> np.ndarray:
        grid = np.empty((self.height, self.width, 3), dtype=np.float32)
        grid[:] = _BG_RGB
        # light horizontal gridlines (quarter marks).
        for frac in (0.25, 0.5, 0.75):
            row = int(self.height * frac)
            grid[row, :] = _GRID_RGB
        gx_norm = self._normalize(self.gx_hist)
        cells_norm = self._normalize(self.cells_hist)
        if cells_norm is not None:
            self._draw_line(grid, cells_norm, _CELLS_RGB)
        if gx_norm is not None:
            self._draw_line(grid, gx_norm, _GX_RGB)
        self._surface[..., :3] = np.clip(grid, 0, 255).astype(np.uint8)
        return self._surface

    def qimage(self):
        from PyQt6.QtGui import QImage
        arr = self.render()
        return QImage(arr.tobytes(), self.width, self.height,
                      4 * self.width, QImage.Format.Format_RGBA8888)
