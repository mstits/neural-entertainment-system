"""Diff primitives: scanline divergence, frame_hash, failure dumps."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.parity.diff import (
    dump_failure_cross,
    dump_failure_golden,
    frame_hash,
    scanline_divergence,
)


def _zeros() -> np.ndarray:
    return np.zeros((240, 256, 3), dtype=np.uint8)


@pytest.mark.parity
def test_scanline_divergence_zero_when_identical():
    a = _zeros()
    b = _zeros()
    assert scanline_divergence(a, b).sum() == 0


@pytest.mark.parity
def test_scanline_divergence_counts_per_row():
    a = _zeros()
    b = _zeros()
    b[10, 0:5, 0] = 255  # 5 diverged pixels on row 10
    b[50, 100:250, 1] = 128  # 150 diverged pixels on row 50
    div = scanline_divergence(a, b)
    assert div.dtype == np.int32
    assert div.shape == (240,)
    assert div[10] == 5
    assert div[50] == 150
    assert div[11] == 0 and div[49] == 0 and div[51] == 0


@pytest.mark.parity
def test_scanline_divergence_any_channel_counts():
    a = _zeros()
    b = _zeros()
    b[5, 0, 2] = 1  # one pixel, only blue channel differs
    assert scanline_divergence(a, b)[5] == 1


@pytest.mark.parity
def test_frame_hash_deterministic():
    rng = np.random.default_rng(seed=42)
    f = rng.integers(0, 256, size=(240, 256, 3), dtype=np.uint8)
    h1 = frame_hash(f)
    h2 = frame_hash(f)
    assert h1 == h2
    assert 0 < h1 < (1 << 64)


@pytest.mark.parity
def test_frame_hash_changes_on_one_byte():
    a = _zeros()
    b = _zeros()
    b[0, 0, 0] = 1
    assert frame_hash(a) != frame_hash(b)


@pytest.mark.parity
def test_dump_failure_cross_writes_three_pngs(tmp_path):
    a = _zeros()
    b = _zeros()
    b[:, :, 0] = 255  # fully red
    diff_path = dump_failure_cross(tmp_path, "sample", a, b)
    assert diff_path.name == "sample_diff.png"
    for suffix in ("_ours.png", "_theirs.png", "_diff.png"):
        p = tmp_path / f"sample{suffix}"
        assert p.exists(), p
        img = np.asarray(Image.open(p).convert("RGB"))
        assert img.shape == (240, 256, 3)


@pytest.mark.parity
def test_dump_failure_cross_diff_png_paints_red_on_divergence(tmp_path):
    a = _zeros()
    b = _zeros()
    b[100, 128] = [123, 45, 67]
    dump_failure_cross(tmp_path, "s", a, b)
    diff_img = np.asarray(Image.open(tmp_path / "s_diff.png").convert("RGB"))
    # Diverged pixel is red; untouched pixels are grayscale (equal channels)
    assert tuple(diff_img[100, 128]) == (255, 0, 0)
    assert diff_img[0, 0, 0] == diff_img[0, 0, 1] == diff_img[0, 0, 2]


@pytest.mark.parity
def test_dump_failure_golden_writes_png_and_hash_file(tmp_path):
    a = _zeros()
    ours_path = dump_failure_golden(tmp_path, "s", a, 0xDEADBEEF, 0xCAFEBABE)
    assert ours_path.exists()
    info = (tmp_path / "s_hash.txt").read_text()
    assert "00000000deadbeef" in info
    assert "00000000cafebabe" in info
