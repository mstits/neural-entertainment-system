"""Sequential replay buffer tests — append/wrap behavior, bulk-add
correctness around the ring boundary, and sample shape invariants."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.replay_buffer import SequentialReplayBuffer


def _fake_obs(value: int) -> np.ndarray:
    """Distinguishable obs so we can tell which step we're looking at."""
    return np.full((4, 84, 84), value & 0xFF, dtype=np.uint8)


def test_basic_append_and_size() -> None:
    rb = SequentialReplayBuffer(capacity=100, seed=0)
    assert len(rb) == 0
    for i in range(50):
        rb.add(_fake_obs(i), action=i % 8, reward=float(i), done=(i == 49))
    assert len(rb) == 50
    assert not rb.is_full


def test_wraparound_drops_oldest() -> None:
    rb = SequentialReplayBuffer(capacity=10, seed=0)
    for i in range(15):
        rb.add(_fake_obs(i), action=0, reward=float(i), done=False)
    assert rb.is_full
    assert len(rb) == 10
    # The oldest 5 (i=0..4) should be gone; latest 10 (i=5..14) should
    # be present somewhere in the ring. Check via reward values.
    assert set(rb._rewards.tolist()) == set(float(i) for i in range(5, 15))


def test_sample_shapes() -> None:
    rb = SequentialReplayBuffer(capacity=200, seed=42)
    for i in range(150):
        rb.add(_fake_obs(i), action=i % 4, reward=float(i), done=False)
    obs, act, rew, done = rb.sample(batch_size=8, seq_len=16)
    assert obs.shape == (8, 16, 4, 84, 84)
    assert act.shape == (8, 16)
    assert rew.shape == (8, 16)
    assert done.shape == (8, 16)
    assert obs.dtype == np.uint8


def test_sample_raises_when_too_short() -> None:
    rb = SequentialReplayBuffer(capacity=100, seed=0)
    for i in range(5):
        rb.add(_fake_obs(i), action=0, reward=0.0, done=False)
    with pytest.raises(RuntimeError):
        rb.sample(batch_size=2, seq_len=10)


def test_bulk_add_no_wrap() -> None:
    """Bulk add that fits within the unfilled tail."""
    rb = SequentialReplayBuffer(capacity=20, seed=0)
    obs = np.stack([_fake_obs(i) for i in range(10)])
    rb.add_batch(
        obs=obs,
        actions=np.arange(10, dtype=np.int64),
        rewards=np.arange(10, dtype=np.float32),
        dones=np.zeros(10, dtype=bool),
    )
    assert len(rb) == 10
    # First entry should be _fake_obs(0).
    assert rb._actions[0] == 0
    assert rb._actions[9] == 9


def test_bulk_add_wraps_correctly() -> None:
    """Bulk add that straddles the ring boundary should land both
    halves in the right physical slots."""
    rb = SequentialReplayBuffer(capacity=10, seed=0)
    # Pre-fill 8 slots so the next bulk add of 6 must wrap.
    for i in range(8):
        rb.add(_fake_obs(i), action=i, reward=float(i), done=False)
    obs = np.stack([_fake_obs(i) for i in range(100, 106)])
    rb.add_batch(
        obs=obs,
        actions=np.arange(100, 106, dtype=np.int64),
        rewards=np.arange(100, 106, dtype=np.float32),
        dones=np.zeros(6, dtype=bool),
    )
    assert len(rb) == 10
    # Cursor lands at (8 + 6) % 10 = 4.
    assert rb._cursor == 4
    # Slots 8, 9 hold actions 100, 101; slots 0..3 hold 102..105.
    assert rb._actions[8] == 100
    assert rb._actions[9] == 101
    assert rb._actions[0] == 102
    assert rb._actions[3] == 105


def test_bulk_add_validates_aligned_lengths() -> None:
    rb = SequentialReplayBuffer(capacity=10, seed=0)
    obs = np.zeros((5, 4, 84, 84), dtype=np.uint8)
    with pytest.raises(ValueError):
        rb.add_batch(
            obs=obs,
            actions=np.zeros(4, dtype=np.int64),  # mismatched length
            rewards=np.zeros(5, dtype=np.float32),
            dones=np.zeros(5, dtype=bool),
        )


def test_stats_dict() -> None:
    rb = SequentialReplayBuffer(capacity=100, seed=0)
    for _ in range(25):
        rb.add(_fake_obs(0), 0, 0.0, False)
    stats = rb.stats()
    assert stats["size"] == 25
    assert stats["capacity"] == 100
    assert abs(stats["fill_fraction"] - 0.25) < 1e-9


def test_seed_determinism() -> None:
    """Same seed → same sample indices."""
    rb1 = SequentialReplayBuffer(capacity=200, seed=7)
    rb2 = SequentialReplayBuffer(capacity=200, seed=7)
    for i in range(150):
        rb1.add(_fake_obs(i), 0, float(i), False)
        rb2.add(_fake_obs(i), 0, float(i), False)
    _, _, r1, _ = rb1.sample(batch_size=4, seq_len=8)
    _, _, r2, _ = rb2.sample(batch_size=4, seq_len=8)
    np.testing.assert_array_equal(r1, r2)
