"""skip_preprocess correctness — guards the tile-mode perf win.

Tile-encoder games read `ram_snapshot`, never the 84x84 `preprocessed`
obs, so the pool can skip the gray+resize kernel (a measured +2.4% on
the tile-path emulation phase). This test guards the invariant that
makes the optimization safe: skipping the preprocess must NOT change
the emulation — RAM must be bit-identical to a non-skipping run on the
same ROM/state/actions — it only zeroes the obs the tile policy ignores.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nes_core import Pool  # noqa: E402

_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_SS = ROOT / "roms" / "Super Mario Bros. (World)_start.state.bin"

pytestmark = pytest.mark.skipif(
    not (_ROM.exists() and _SS.exists()),
    reason="SMB ROM / start state not available",
)


def _run(skip: bool):
    actions = np.ones(2, dtype=np.uint8)  # all "right"
    p = Pool(rom_path=str(_ROM), num_workers=2, frame_skip=4,
             start_state_path=str(_SS))
    p.set_headless(True)
    p.set_skip_preprocess(skip)
    res = None
    for _ in range(12):  # warm past the post-load blank frame
        res = p.step_all(actions)
    preprocessed = np.asarray(res[0][1]).copy()
    ram = bytes(res[0][2])
    p.shutdown()
    return preprocessed, ram


def test_skip_preprocess_preserves_emulation_but_zeroes_obs():
    pp_off, ram_off = _run(skip=False)
    pp_on, ram_on = _run(skip=True)

    # Emulation must be untouched: identical inputs -> identical RAM.
    assert ram_off == ram_on, (
        "skip_preprocess changed RAM — it must only skip obs post-"
        "processing, never affect emulation"
    )
    # Without skip the obs has real content; with skip it is all zeros.
    assert pp_off.any(), "non-skip preprocessed obs should have content"
    assert not pp_on.any(), "skip_preprocess should return an all-zero obs"
