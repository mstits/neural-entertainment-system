"""batched_render (Replace mode) safety — guards the per-game opt-in.

Replace mode skips per-pixel PPU work on scanlines the prev-frame
clean-history predicts unchanged. A mispredict may paint a 1-frame
stale row — acceptable for training obs, which is why profiles opt in
per-game. What must NEVER change is the emulation itself: sprite-0
latching and all register/NMI/IRQ semantics are preserved, so game
logic (and therefore RAM) must be bit-identical to Off mode on the
same ROM/state/actions. SMB polls sprite-0 every frame for its status
bar raster split, making it a strong canary for that invariant.
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


def _run(mode: str) -> bytes:
    actions = np.ones(2, dtype=np.uint8)  # all "right"
    p = Pool(rom_path=str(_ROM), num_workers=2, frame_skip=4,
             start_state_path=str(_SS))
    p.set_headless(True)
    p.set_batched_render_mode(mode)
    res = None
    for _ in range(60):  # 240 frames — plenty of sprite-0 polls
        res = p.step_all(actions)
    ram = bytes(res[0][2])
    p.shutdown()
    return ram


def test_replace_mode_preserves_emulation():
    ram_off = _run("off")
    ram_replace = _run("replace")
    assert ram_off == ram_replace, (
        "batched_render=replace changed RAM — it must only affect "
        "framebuffer writes, never game logic (sprite-0/NMI/IRQ timing)"
    )


def test_bogus_mode_rejected():
    p = Pool(rom_path=str(_ROM), num_workers=1, frame_skip=4)
    try:
        with pytest.raises(ValueError):
            p.set_batched_render_mode("bogus")
    finally:
        p.shutdown()
