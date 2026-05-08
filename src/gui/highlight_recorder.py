"""
Highlight auto-clip recorder.

Maintains a per-worker ring buffer of recent frames in the GUI process
(outside the trainer's hot path). When the narrator reports a
first-of-its-kind event — first dungeon entry, first triforce piece,
first heart container, first new item — we freeze the last ~10 seconds
of that worker's frames to an MP4 under `highlights/`.

The highlights directory is auto-created, filenames include the
worker_id, genome name, kind, and timestamp so a stream-VOD pipeline
can pick them up as they land.

Why in the GUI and not the trainer: MP4 encoding (cv2.VideoWriter) is
not free. Doing it in the trainer subprocess would add 20-50 ms to
whichever worker fired the event on the step the event was captured.
The GUI is already the consumer of the frames, it polls at 30 Hz, and
it can afford the encoding hiccup without perturbing training.

Cost (measured on an M-series Mac, 16-worker grid, 4-second ring at
15 fps): ~250 MB steady-state RAM for the ring (16 × 60 × 184 KB), and
~200 ms wall-clock per clip write. Both are amortized; a clip only
gets written on a rare event.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


# Default: 4 seconds at 15 fps = 60 frames per worker. Tunable via
# ctor. Memory per frame at 240x256x3 uint8 is 184 KB, so 60 frames =
# 11 MB/worker; 16 workers = 176 MB.
_DEFAULT_FPS = 15
_DEFAULT_SECONDS = 4


class HighlightRecorder:
    """Per-worker frame ring buffer with on-demand MP4 dump.

    Usage:
        rec = HighlightRecorder(num_workers=16, out_dir="highlights/")
        # On every grid-poll tick, ingest the latest frame for each tile:
        for wid, frame in enumerate(latest_frames):
            rec.ingest(wid, frame)
        # When the narrator reports a banner event:
        rec.capture(worker_id=3, kind="dungeon_enter",
                    genome_name="Brutus", label="first dungeon entry")
    """

    def __init__(
        self,
        num_workers: int,
        out_dir: str | Path = "highlights",
        fps: int = _DEFAULT_FPS,
        seconds: float = _DEFAULT_SECONDS,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.fps = int(fps)
        self.seconds = float(seconds)
        # Ring size = fps × seconds. New ingests push out the oldest.
        self._ring_n = max(1, int(self.fps * self.seconds))
        self._rings: list[deque[np.ndarray]] = [
            deque(maxlen=self._ring_n) for _ in range(self.num_workers)
        ]
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # Per-worker last-ingest timestamp so we down-sample to `fps`
        # even when the caller polls us faster (grid poll ~30 Hz).
        self._last_ingest: list[float] = [0.0] * self.num_workers
        # Cooldown per (worker_id, kind) to avoid writing 30 copies of
        # the same highlight if the same event fires repeatedly during
        # a single episode.
        self._capture_cooldown: dict[tuple[int, str], float] = {}
        # Minimum seconds between two captures of the same
        # (worker, kind). Generous default — a rare event should only
        # generate one MP4 per episode.
        self._capture_gap_s: float = 5.0

    # ---------------- ingest ----------------------------------------

    def ingest(self, worker_id: int, frame: np.ndarray) -> None:
        """Append one frame to the worker's ring if enough time has
        passed since the last ingest (downsamples to self.fps)."""
        if not (0 <= worker_id < self.num_workers):
            return
        if frame is None:
            return
        if frame.shape != (240, 256, 3) or frame.dtype != np.uint8:
            return  # silently ignore unexpected frames

        now = time.monotonic()
        min_gap = 1.0 / max(1, self.fps)
        if now - self._last_ingest[worker_id] < min_gap:
            return
        self._last_ingest[worker_id] = now
        # Copy the frame because the caller's buffer may be reused on
        # the next poll — without a copy we'd keep capturing the same
        # latest-frame reference self._ring_n times.
        self._rings[worker_id].append(frame.copy())

    # ---------------- capture ---------------------------------------

    def capture(
        self,
        worker_id: int,
        kind: str,
        genome_name: str,
        label: Optional[str] = None,
    ) -> Optional[Path]:
        """Flush the current ring buffer for `worker_id` to an MP4.
        Returns the output path on success, None on cooldown or empty
        ring or if OpenCV isn't available."""
        if not (0 <= worker_id < self.num_workers):
            return None
        ring = self._rings[worker_id]
        if not ring:
            return None

        key = (worker_id, kind)
        now = time.monotonic()
        last = self._capture_cooldown.get(key, 0.0)
        if now - last < self._capture_gap_s:
            return None
        self._capture_cooldown[key] = now

        # Filename: sortable timestamp + worker + kind + genome so the
        # highlights directory is legible when a VOD script walks it.
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in genome_name)
        fname = f"{ts}_w{worker_id:02d}_{kind}_{safe_name}.mp4"
        out_path = self._out_dir / fname

        frames = list(ring)  # snapshot; future ingests won't affect this clip
        ok = _write_mp4(frames, out_path, self.fps)
        if not ok:
            return None
        if label:
            log.info("Highlight captured: %s (%s)", out_path.name, label)
        else:
            log.info("Highlight captured: %s", out_path.name)
        return out_path

    def clear(self, worker_id: Optional[int] = None) -> None:
        """Drop buffered frames (all workers or just one)."""
        if worker_id is None:
            for r in self._rings:
                r.clear()
        elif 0 <= worker_id < self.num_workers:
            self._rings[worker_id].clear()


# ---------- MP4 encoder with graceful fallback ------------------------

def _write_mp4(frames: list[np.ndarray], out_path: Path, fps: int) -> bool:
    """Encode RGB frames to MP4 at `out_path`. Tries cv2, then falls
    back to piping raw frames into an `ffmpeg` subprocess (required on
    systems where cv2 isn't installed — macOS + brew install ffmpeg).
    Returns True on success."""
    if _try_cv2(frames, out_path, fps):
        return True
    if _try_ffmpeg_subprocess(frames, out_path, fps):
        return True
    log.warning(
        "No MP4 backend available (cv2 absent and ffmpeg not on PATH); "
        "highlight %s skipped", out_path.name,
    )
    return False


def _try_cv2(frames: list[np.ndarray], out_path: Path, fps: int) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def _try_ffmpeg_subprocess(
    frames: list[np.ndarray], out_path: Path, fps: int
) -> bool:
    """Pipe raw RGB frames to an ffmpeg subprocess via stdin. Works on
    any box with ffmpeg on PATH, no Python bindings needed."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    h, w, _ = frames[0].shape
    cmd = [
        ff, "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        str(out_path),
    ]
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for f in frames:
            # Ensure contiguous RGB bytes.
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=30)
        if rc != 0:
            err = (proc.stderr.read() or b"").decode(errors="replace")
            log.warning("ffmpeg write failed rc=%d: %s", rc, err[:300])
            return False
    except Exception as exc:
        log.warning("ffmpeg subprocess error: %s", exc)
        return False
    finally:
        # If we bailed early (broken pipe, timeout, frame iteration
        # raised), the ffmpeg child can be left running. Kill it so it
        # doesn't accumulate as a zombie across long training sessions.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
    return out_path.exists() and out_path.stat().st_size > 0
