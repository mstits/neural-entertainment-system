"""Tests for the frame / tile feature stackers.

TileFeatureStacker is the temporal-context layer that turns the
175-dim point-in-time SMB tile vector into a 700-dim 4-frame stack
(yumouwei recipe). Its ring-wrap correctness is load-bearing — a
bug here silently feeds the policy a scrambled history and the
training run just doesn't learn.
"""

from __future__ import annotations

import numpy as np

from src.emulation.frame_utils import TileFeatureStacker


FEATURE_DIM = 8
STACK_SIZE = 4


def _vec(label: int) -> np.ndarray:
    """Distinct, identifiable feature vector. label 1 → [1,1,1,...]."""
    return np.full((FEATURE_DIM,), label, dtype=np.int8)


def test_reset_fills_ring_with_initial_features() -> None:
    """reset(v0) must produce concat(v0, v0, v0, v0) — no zero
    padding leaking into the first observation of the episode."""
    s = TileFeatureStacker(stack_size=STACK_SIZE, feature_dim=FEATURE_DIM)
    out = s.reset(_vec(7))
    assert out.shape == (STACK_SIZE * FEATURE_DIM,)
    expected = np.concatenate([_vec(7)] * STACK_SIZE)
    np.testing.assert_array_equal(out, expected)


def test_push_until_full_preserves_oldest_to_newest_order() -> None:
    """After reset(v0) + push(v1..v4) the output must be concat(v1, v2, v3, v4)
    — i.e. the ring rotates so the oldest live frame is at offset 0."""
    s = TileFeatureStacker(stack_size=STACK_SIZE, feature_dim=FEATURE_DIM)
    s.reset(_vec(0))
    out = None
    for i in range(1, STACK_SIZE + 1):
        out = s.push(_vec(i))
    expected = np.concatenate([_vec(1), _vec(2), _vec(3), _vec(4)])
    np.testing.assert_array_equal(out, expected)


def test_push_past_capacity_wraps_correctly() -> None:
    """Push 6 distinct vectors into a stack of 4 — only the last 4
    should survive, still in oldest→newest order."""
    s = TileFeatureStacker(stack_size=STACK_SIZE, feature_dim=FEATURE_DIM)
    s.reset(_vec(0))
    out = None
    for i in range(1, 7):  # v1..v6
        out = s.push(_vec(i))
    expected = np.concatenate([_vec(3), _vec(4), _vec(5), _vec(6)])
    np.testing.assert_array_equal(out, expected)


def test_reset_after_push_clears_history() -> None:
    """A fresh reset mid-stream must wipe the entire window — episode
    boundaries can't leak the previous episode's frames into the next."""
    s = TileFeatureStacker(stack_size=STACK_SIZE, feature_dim=FEATURE_DIM)
    s.reset(_vec(0))
    for i in range(1, 5):
        s.push(_vec(i))
    out = s.reset(_vec(99))
    expected = np.concatenate([_vec(99)] * STACK_SIZE)
    np.testing.assert_array_equal(out, expected)


def test_vectorized_matches_per_slot_reference() -> None:
    """The optimized np.concatenate flatten must produce the same
    bytes as a per-slot Python reference, across every possible
    head position. Guards future micro-opts of _flatten_oldest_to_newest."""
    s = TileFeatureStacker(stack_size=STACK_SIZE, feature_dim=FEATURE_DIM)
    s.reset(_vec(0))
    # Walk the head pointer through every position in the ring.
    for i in range(1, STACK_SIZE + 1):
        s.push(_vec(i))
        # Reference: explicit per-slot copy in oldest→newest order.
        ref = np.empty(STACK_SIZE * FEATURE_DIM, dtype=np.int8)
        for k in range(STACK_SIZE):
            ref[k * FEATURE_DIM:(k + 1) * FEATURE_DIM] = s._ring[
                (s._head + k) % STACK_SIZE
            ]
        np.testing.assert_array_equal(s._flatten_oldest_to_newest(), ref)
