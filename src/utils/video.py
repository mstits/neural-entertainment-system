"""
Write NES emulator frames to an MP4.

Used by the trainer to save the best-genome episode every N generations,
so you can watch the learning curve visually (not just metrics).

Uses imageio-ffmpeg (already a transitive dep of matplotlib) to avoid
adding a new requirement. Falls back to no-op if ffmpeg isn't available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


class VideoWriter:
    """Append-style MP4 writer for (240, 256, 3) uint8 RGB frames."""

    def __init__(self, path: str | Path, fps: int = 30) -> None:
        self.path = Path(path)
        self.fps = fps
        self._writer = None
        self._frames_written = 0
        self._enabled = True

    def _lazy_init(self, frame_shape: tuple[int, int, int]) -> None:
        if self._writer is not None or not self._enabled:
            return
        try:
            import imageio.v2 as imageio
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                str(self.path),
                fps=self.fps,
                format="FFMPEG",
                codec="libx264",
                quality=7,
                macro_block_size=1,  # NES frames aren't divisible by 16
            )
        except Exception as exc:
            log.warning("VideoWriter disabled (ffmpeg missing?): %s", exc)
            self._enabled = False

    def write(self, frame: np.ndarray) -> None:
        if not self._enabled:
            return
        if self._writer is None:
            self._lazy_init(frame.shape)
            if not self._enabled:
                return
        try:
            self._writer.append_data(np.ascontiguousarray(frame))
            self._frames_written += 1
        except Exception as exc:
            log.warning("video write failed, disabling: %s", exc)
            self._enabled = False

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        return self._frames_written
