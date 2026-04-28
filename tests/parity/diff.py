"""Diff primitives: scanline Hamming (cross-emulator) + 64-bit frame
hash (golden-hash mode), plus failure-artifact PNG dumps.

Frame hash uses BLAKE2b truncated to 64 bits rather than FNV-1a. Both
ends of the pipeline (author.py record, run.py replay) use this same
function, so the choice of algorithm only needs to be fast and
deterministic across platforms — the C-backed hashlib path is
~1000× faster than a Python-loop FNV-1a.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def scanline_divergence(a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
    """Pixel-differ count per scanline. A pixel differs if any channel does."""
    assert a_rgb.shape == b_rgb.shape == (240, 256, 3), (a_rgb.shape, b_rgb.shape)
    assert a_rgb.dtype == b_rgb.dtype == np.uint8
    return (a_rgb != b_rgb).any(axis=2).sum(axis=1).astype(np.int32)


def frame_hash(rgb: np.ndarray) -> int:
    """Stable 64-bit hash of a uint8 RGB frame."""
    assert rgb.dtype == np.uint8
    h = hashlib.blake2b(rgb.tobytes(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _diff_overlay(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """Red on diverged pixels, desaturated `ours` elsewhere."""
    diff_mask = (ours != theirs).any(axis=2)
    gray = ours.mean(axis=2).astype(np.uint8)
    out = np.stack([gray, gray, gray], axis=2)
    out[diff_mask] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def dump_failure_cross(
    out_dir: Path, prefix: str, ours: np.ndarray, theirs: np.ndarray,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ours_path = out_dir / f"{prefix}_ours.png"
    theirs_path = out_dir / f"{prefix}_theirs.png"
    diff_path = out_dir / f"{prefix}_diff.png"
    Image.fromarray(ours).save(ours_path)
    Image.fromarray(theirs).save(theirs_path)
    Image.fromarray(_diff_overlay(ours, theirs)).save(diff_path)
    return diff_path


def dump_failure_golden(
    out_dir: Path, prefix: str, ours: np.ndarray,
    expected_hash: int, actual_hash: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ours_path = out_dir / f"{prefix}_ours.png"
    info_path = out_dir / f"{prefix}_hash.txt"
    Image.fromarray(ours).save(ours_path)
    info_path.write_text(
        f"expected_hash={expected_hash:016x}\nactual_hash={actual_hash:016x}\n"
    )
    return ours_path
