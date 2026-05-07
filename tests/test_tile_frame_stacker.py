"""Pin behavior of the tile-mode frame stacker.

Tile mode stacks the last N tile feature vectors so the policy
sees temporal context (moving enemies, recent positions) in
addition to the current static grid. Both canonical SMB-RL
recipes that have empirically cleared 1-1 (uvipen, yumouwei)
use 4-frame stacks; yumouwei explicitly reports n_stack=1 fails.

These tests pin the stacker's invariants:

* Output shape is `(stack_size * feature_dim,)`.
* `reset(features)` produces a stack of `features` repeated.
* `push(features)` rotates oldest→newest with the new frame at the end.
* Wrap-around works correctly across multiple full cycles.
* Shape mismatches raise ValueError loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emulation.tile_observations.stacker import TileFrameStacker


def test_reset_replicates_initial_features() -> None:
    s = TileFrameStacker(stack_size=4, feature_dim=3)
    feat = np.array([1, 2, 3], dtype=np.int8)
    out = s.reset(feat)
    assert out.shape == (12,)
    assert out.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_push_advances_one_slot_per_call() -> None:
    s = TileFrameStacker(stack_size=4, feature_dim=2)
    s.reset(np.array([0, 0], dtype=np.int8))
    out = s.push(np.array([1, 1], dtype=np.int8))
    # Oldest still 0, newest now 1.
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    out = s.push(np.array([2, 2], dtype=np.int8))
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
    out = s.push(np.array([3, 3], dtype=np.int8))
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    out = s.push(np.array([4, 4], dtype=np.int8))
    assert out.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_push_wraps_correctly_after_full_cycle() -> None:
    """After pushing more frames than stack_size, the oldest slot rolls
    off and we keep the most-recent N frames."""
    s = TileFrameStacker(stack_size=3, feature_dim=1)
    s.reset(np.array([0], dtype=np.int8))
    for i in range(1, 10):
        s.push(np.array([i], dtype=np.int8))
    # After pushing 1..9 with stack=3, output should be [7, 8, 9].
    assert s._view_oldest_to_newest().tolist() == [7, 8, 9]


def test_reset_after_use_clears_state() -> None:
    """A new episode should cleanly re-seed the stacker, not carry
    stale frames from the previous episode."""
    s = TileFrameStacker(stack_size=4, feature_dim=2)
    s.reset(np.array([0, 0], dtype=np.int8))
    s.push(np.array([1, 1], dtype=np.int8))
    s.push(np.array([2, 2], dtype=np.int8))
    # Reset on episode boundary.
    out = s.reset(np.array([99, 99], dtype=np.int8))
    assert out.tolist() == [99, 99, 99, 99, 99, 99, 99, 99]


def test_stack_size_one_passes_features_through() -> None:
    """stack_size=1 should be a no-op identity: output equals input."""
    s = TileFrameStacker(stack_size=1, feature_dim=3)
    feat = np.array([5, 6, 7], dtype=np.int8)
    out = s.reset(feat)
    assert out.tolist() == [5, 6, 7]
    out = s.push(np.array([8, 9, 10], dtype=np.int8))
    assert out.tolist() == [8, 9, 10]


def test_shape_mismatch_raises() -> None:
    s = TileFrameStacker(stack_size=4, feature_dim=3)
    with pytest.raises(ValueError):
        s.reset(np.array([1, 2], dtype=np.int8))  # too short
    s.reset(np.array([1, 2, 3], dtype=np.int8))
    with pytest.raises(ValueError):
        s.push(np.array([1, 2, 3, 4], dtype=np.int8))  # too long


def test_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        TileFrameStacker(stack_size=0, feature_dim=10)
    with pytest.raises(ValueError):
        TileFrameStacker(stack_size=4, feature_dim=0)


def test_smb_default_dimensions() -> None:
    """Sanity: stack=4 with feature_dim=175 (SMB tile observation)
    produces 700-dim output, matching the policy input."""
    s = TileFrameStacker(stack_size=4, feature_dim=175)
    feat = np.zeros(175, dtype=np.int8)
    out = s.reset(feat)
    assert out.shape == (700,)


def test_view_returned_as_int8() -> None:
    """Output dtype must stay int8 — the policy network casts to
    float32 itself, but a wrong dtype here would break that path."""
    s = TileFrameStacker(stack_size=4, feature_dim=3)
    feat = np.array([-128, 0, 127], dtype=np.int8)
    out = s.reset(feat)
    assert out.dtype == np.int8
    assert out.tolist() == [-128, 0, 127, -128, 0, 127, -128, 0, 127, -128, 0, 127]
