"""Pin classify_motion's static/live split in scripts/rom_library_scan.py.

`ok` in the scanner means "no panic or timeout": a ROM frozen at boot
is `ok` too (Jackal, mapper 2, confirmed 2026-09-01). classify_motion
adds the missing check: hash the framebuffer every frame under a
Start/A-burst-plus-random input schedule; a hash that never moves is a
static screen, not a live boot.

classify_motion takes only `.reset()`/`.step()`/`.get_frame()` off its
`env` argument, so this pins it against a fake env with a synthetic
framebuffer source, no nes_core, no real ROM required.

Loaded straight from the script file (importlib), matching the
existing tests/test_rom_resolver.py convention: the module has no
`__init__.py`-package identity to import normally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_rom_library_scan():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rom_library_scan.py"
    spec = importlib.util.spec_from_file_location("rom_library_scan", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


rom_library_scan = _load_rom_library_scan()
classify_motion = rom_library_scan.classify_motion


class _FrozenEnv:
    """Synthetic static framebuffer source: every frame, regardless of
    input, is the exact same bytes. Mirrors a ROM boot-locked on a
    title screen that never advances no matter what's pressed."""

    def __init__(self, frame_bytes: bytes = b"\x00" * (240 * 256 * 3)):
        self._frame = frame_bytes
        self.reset_calls = 0
        self.steps = []

    def reset(self):
        self.reset_calls += 1

    def step(self, action):
        self.steps.append(action)

    def get_frame(self):
        return self._frame  # same object every call, never changes


class _CountingEnv(_FrozenEnv):
    """Static up through frame `flip_at`, then a different constant
    frame forever after: a live boot that starts responding late,
    e.g. past the random-input frames."""

    def __init__(self, flip_at: int, **kw):
        super().__init__(**kw)
        self._flip_at = flip_at
        self._frame_before = b"\x00" * (240 * 256 * 3)
        self._frame_after = b"\x01" * (240 * 256 * 3)
        self._frame_count = 0

    def step(self, action):
        super().step(action)
        self._frame_count += 1

    def get_frame(self):
        return self._frame_after if self._frame_count > self._flip_at else self._frame_before


_SCHEDULE = dict(
    period=120, start_burst_len=8, a_burst_start=60, a_burst_len=4,
    random_from=1500, random_prob=0.5, seed=314159265,
)


def test_classify_motion_static_synthetic():
    """A framebuffer source that never changes, run for a full
    3000-frame schedule (Start/A bursts, then random input from frame
    1500), classifies static: exactly the shape of the Jackal /
    Action 52 / NWC1990 / SMB+Tetris+NWC false-positive `ok`s."""
    env = _FrozenEnv()
    result = classify_motion(env, frames=3000, reset_first=True, **_SCHEDULE)

    assert result["motion"] == "static"
    assert result["distinct_hashes"] == 1
    assert result["first_change_frame"] is None
    assert result["frames_checked"] == 3000
    assert env.reset_calls == 1
    assert len(env.steps) == 3000
    # Start burst present in the schedule even though it never landed.
    assert 0x08 in env.steps
    assert 0x01 in env.steps


def test_classify_motion_live_synthetic():
    """Sanity check on the other side of the same function: a
    framebuffer that changes partway through must NOT be classified
    static, and first_change_frame must land where it actually
    changed."""
    env = _CountingEnv(flip_at=500)
    result = classify_motion(env, frames=3000, reset_first=True, **_SCHEDULE)

    assert result["motion"] == "live"
    assert result["distinct_hashes"] == 2
    assert result["first_change_frame"] == 500


def test_classify_motion_reset_first_false_skips_reset():
    env = _FrozenEnv()
    classify_motion(env, frames=10, reset_first=False, **_SCHEDULE)
    assert env.reset_calls == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
