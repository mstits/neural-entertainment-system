"""Pre-flight check: nes_core and nes-py must produce byte-identical
framebuffers on a known baseline. Both ship the Laines palette; if
either LUT shifts, this test fails loudly before any tape is consulted.

If this fails, the whole parity harness is invalid — raw RGB diff no
longer equals palette-index diff. Either re-align the LUTs or reintroduce
a normalization layer in `diff.py` before marking this skip.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINE_ROM = REPO / "roms" / "Mario Bros. (World).nes"
BASELINE_FRAME = 30


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
def test_laines_palette_matches_between_emulators():
    import nes_core
    import nes_py

    assert BASELINE_ROM.exists(), f"Baseline ROM missing: {BASELINE_ROM}"

    ours = nes_core.NESEnvironment(str(BASELINE_ROM), frame_skip=1)
    ours.reset()
    theirs = nes_py.NESEnv(str(BASELINE_ROM))
    theirs.reset()

    for _ in range(BASELINE_FRAME):
        ours.step(0)
        theirs.step(0)

    a = np.asarray(ours.get_frame())
    b = np.asarray(theirs.screen).copy()

    assert a.shape == b.shape == (240, 256, 3), (a.shape, b.shape)
    assert a.dtype == b.dtype == np.uint8, (a.dtype, b.dtype)

    diff_mask = (a != b).any(axis=2)
    diff_count = int(diff_mask.sum())
    if diff_count:
        ys, xs = np.where(diff_mask)
        sample_y, sample_x = int(ys[0]), int(xs[0])
        pytest.fail(
            f"Palette LUT drift: {diff_count}/{240 * 256} pixels differ "
            f"at Mario frame {BASELINE_FRAME}. "
            f"Sample (y={sample_y}, x={sample_x}): "
            f"nes_core={tuple(a[sample_y, sample_x])} "
            f"nes_py={tuple(b[sample_y, sample_x])}. "
            f"Either realign palette LUTs or reintroduce normalization in diff.py."
        )
