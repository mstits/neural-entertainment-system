"""Prove-It test: `nes_core.NESEnvironment(rom, frame_skip=4)` must
produce the same final frame as stepping the same ROM at `frame_skip=1`
for an equivalent number of emulated frames.

Under the hood, `frame_skip>1` toggles `set_skip_render(true)` on all
but the last frame of each step batch, assuming "the rendered frame is
correct because the LAST frame in the batch runs the full tick path."
That assumption fails on any game whose CPU reads $2002 mid-frame to
time scroll splits (SMB HUD, Zelda HUD, MMC3 status-bar splits, etc.):
sprite-0 hit never fires during skipped frames, so the CPU takes
different branches, and by the full-render frame the scroll state is
corrupted.

Current nes_core on master: this test FAILS with ~59000 pixels differing.
After the skip_render correctness fix: this test PASSES with 0 diff.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SMB = REPO / "roms" / "Super Mario Bros. (World).nes"


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _run_smb_to_gameplay(frame_skip: int) -> np.ndarray:
    """240 emulated frames, Start pressed during outer steps whose
    start-frame falls in [60, 70). Returns final RGB frame."""
    import nes_core
    env = nes_core.NESEnvironment(str(SMB), frame_skip=frame_skip)
    env.reset()
    outer_steps = 240 // frame_skip
    for i in range(outer_steps):
        t = i * frame_skip
        btn = 8 if 60 <= t < 70 else 0  # Start = bit 3
        env.step(btn)
    return np.asarray(env.get_frame())


@pytest.mark.parity
def test_frame_skip_4_matches_frame_skip_1_on_smb_gameplay():
    full = _run_smb_to_gameplay(1)
    skip = _run_smb_to_gameplay(4)
    diff = int((full != skip).any(axis=2).sum())
    # Two fixes behind this being byte-exact:
    # 1. Commit 0cfe61e removed the tick_skip_render bypass that skipped
    #    sprite-0-hit / mapper-IRQ / scroll reloads on skipped frames
    #    (pre-fix 59,432 px → post-fix 50 px).
    # 2. Follow-up fix in Ppu::tick resets skip_render to false at the
    #    frame-wrap boundary so CPU-bulk-step overshoot ticks that cross
    #    into the new frame always render; the skip→full transition no
    #    longer leaves 50 px of row-0 stale frame_buffer content
    #    (post-fix 50 px → 0 px).
    assert diff == 0, (
        f"skip_render regression: {diff}/61440 pixels differ "
        f"(expected 0 post-fix; see tests/skip_render_parity.rs and "
        f"Ppu::tick's scanline>PRE_RENDER_SCANLINE block)."
    )


@pytest.mark.parity
def test_frame_skip_4_matches_frame_skip_1_on_smb_title_idle():
    """Idle SMB title @ 120 frames converges byte-exact post-fix.
    Pre-fix had 144 pixels drifting; fix in ppu.rs::tick eliminated it.
    """
    import nes_core
    def run(fs: int) -> np.ndarray:
        env = nes_core.NESEnvironment(str(SMB), frame_skip=fs)
        env.reset()
        for _ in range(120 // fs):
            env.step(0)
        return np.asarray(env.get_frame())

    full = run(1)
    skip = run(4)
    diff = int((full != skip).any(axis=2).sum())
    assert diff == 0, (
        f"skip_render drift on idle SMB title: {diff}/61440 pixels differ"
    )


@pytest.mark.parity
@pytest.mark.parametrize("frame_skip", [2, 3, 4, 8, 16])
def test_skip_render_byte_exact_across_frame_skip_values(frame_skip: int):
    """Every supported frame_skip must produce the same final frame as
    frame_skip=1 on the SMB title-idle scenario (240 frames, no input).

    Pre-fix: fs=2: 2 px, fs=4: 50 px, fs=8: 87 px on scanline 0, from
    the bulk-step CPU path ticking the PPU past a frame boundary
    while skip_render still held the outgoing frame's value. Fix:
    `Ppu::tick` clears skip_render in its `scanline>PRE_RENDER_SCANLINE`
    block so overshoot ticks always render.

    This test uses idle (no Start press) on purpose: with NESEnvironment::reset
    no longer pre-advancing a warmup frame (to align with nes-py), any scenario
    that presses a button on a specific frame range would see different game
    states at different fs values because the Start-window frames don't line up
    with the outer-step boundary of larger fs values (fs=16 lands its press
    window on different frames than fs=1).
    """
    import nes_core
    def run(fs: int) -> np.ndarray:
        env = nes_core.NESEnvironment(str(SMB), frame_skip=fs)
        env.reset()
        for _ in range(240 // fs):
            env.step(0)
        return np.asarray(env.get_frame())
    full = run(1)
    skip = run(frame_skip)
    diff = int((full != skip).any(axis=2).sum())
    assert diff == 0, (
        f"frame_skip={frame_skip} diverged by {diff}/61440 px from "
        f"frame_skip=1 on SMB title idle — skip_render regression."
    )
