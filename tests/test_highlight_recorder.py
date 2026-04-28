"""Tests for HighlightRecorder — ring-buffer behaviour + MP4 dump."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.gui.highlight_recorder import HighlightRecorder


def _fake_frame(tag: int = 0) -> np.ndarray:
    f = np.zeros((240, 256, 3), dtype=np.uint8)
    f[..., 0] = tag & 0xFF
    return f


def test_recorder_downsamples_to_fps(tmp_path: Path) -> None:
    rec = HighlightRecorder(num_workers=2, out_dir=tmp_path, fps=10, seconds=2)
    # Ingest 100 frames as fast as possible; only ~20 should stick (2s × 10fps).
    for i in range(100):
        rec.ingest(0, _fake_frame(i))
    # Give the wall clock a moment so the rate-limiter sees time pass.
    buffered = len(rec._rings[0])
    # Allow a small margin — first frame lands unconditionally,
    # subsequent ingests are throttled.
    assert 1 <= buffered <= 2, f"expected near-zero-elapsed ingests, got {buffered}"


def test_recorder_fills_ring_over_time(tmp_path: Path) -> None:
    rec = HighlightRecorder(num_workers=1, out_dir=tmp_path, fps=20, seconds=1)
    deadline = time.monotonic() + 1.5
    tick = 0
    while time.monotonic() < deadline:
        rec.ingest(0, _fake_frame(tick))
        tick += 1
        time.sleep(0.01)
    # Ring caps at fps × seconds = 20. After 1.5 s of sub-ms ingests
    # we should be at the cap (or close).
    assert 15 <= len(rec._rings[0]) <= 20


def test_capture_writes_mp4(tmp_path: Path) -> None:
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    rec = HighlightRecorder(num_workers=1, out_dir=tmp_path, fps=5, seconds=2)
    # Simulate frames arriving over time so the ring actually has contents.
    for i in range(10):
        rec.ingest(0, _fake_frame(i))
        time.sleep(0.25)
    path = rec.capture(0, kind="dungeon_enter", genome_name="Brutus",
                       label="first dungeon entry")
    assert path is not None
    assert path.exists()
    assert path.suffix == ".mp4"
    assert "Brutus" in path.name
    assert "dungeon_enter" in path.name


def test_capture_cooldown_suppresses_duplicates(tmp_path: Path) -> None:
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    rec = HighlightRecorder(num_workers=1, out_dir=tmp_path, fps=5, seconds=1)
    for i in range(5):
        rec.ingest(0, _fake_frame(i))
        time.sleep(0.22)
    rec._capture_gap_s = 10.0  # aggressive cooldown for the test
    first = rec.capture(0, kind="triforce", genome_name="Scout", label="t1")
    second = rec.capture(0, kind="triforce", genome_name="Scout", label="t2")
    assert first is not None
    assert second is None  # cooldown suppressed
