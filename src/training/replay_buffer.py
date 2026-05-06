"""Sequential replay buffer for the Dreamer world-model trainer.

Standard PPO trains on a single trajectory window per generation and
discards it. World-model training is fundamentally different: it needs
randomly-sampled fixed-length sequences from a long history of episodes
so the world model sees a representative slice of the policy's full
behavior distribution, not just the most recent gen.

Storage layout: append-only ring of (obs, action, reward, done) tuples.
Sampling returns sequence batches `(B, L, ...)` with start indices
chosen uniformly at random over valid positions (not crossing episode
boundaries unless the seq len overruns — done flags inside a sequence
are valid because the model learns the continue prediction from them).

Memory: 1M frames × 4×84×84 uint8 ≈ 27 GB. The target hardware (M4 MBP
128 GB) handles this comfortably; the buffer stays uint8 in RAM and
the trainer normalizes to float on each batch transfer to MPS.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class _BufferShapes:
    obs: tuple[int, ...]
    action: tuple[int, ...]


class SequentialReplayBuffer:
    """Append-only ring buffer of `(obs, action, reward, done)`.

    Sampling: `sample(batch_size, seq_len)` returns
        obs:    (B, L, *obs_shape)   uint8
        action: (B, L)               int64
        reward: (B, L)               float32
        done:   (B, L)               bool

    The buffer is bytes-flat — no Python-level per-step records, no
    object boxing. A 1 M-step capacity allocates ~27 GB up front and
    overwrites in FIFO order once full.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...] = (4, 84, 84),
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive (got {capacity})")
        self.capacity = int(capacity)
        self._shapes = _BufferShapes(obs=obs_shape, action=())
        # Pre-allocate so steady-state appends don't allocate.
        self._obs = np.zeros((self.capacity,) + obs_shape, dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=bool)

        self._cursor = 0      # next write index
        self._size = 0        # 0 .. capacity
        self._rng = random.Random(seed)

    # ----- size -------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    # ----- write ------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Append a single step. Wraps around when `capacity` is reached."""
        i = self._cursor
        # `np.copyto` is the fastest path for uint8 arrays of known
        # shape; avoids a per-step allocation in the hot append loop.
        np.copyto(self._obs[i], obs)
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = bool(done)

        self._cursor = (self._cursor + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Bulk append. All inputs must be aligned: shape `(N, ...)`."""
        n = obs.shape[0]
        if not (actions.shape[0] == n == rewards.shape[0] == dones.shape[0]):
            raise ValueError("add_batch: all input arrays must share leading dim")
        # Vectorize via two contiguous slices when the write doesn't
        # straddle the ring boundary; fall back to per-step otherwise.
        end = self._cursor + n
        if end <= self.capacity:
            self._obs[self._cursor:end] = obs
            self._actions[self._cursor:end] = actions
            self._rewards[self._cursor:end] = rewards
            self._dones[self._cursor:end] = dones
            self._cursor = end % self.capacity
        else:
            split = self.capacity - self._cursor
            self._obs[self._cursor:] = obs[:split]
            self._actions[self._cursor:] = actions[:split]
            self._rewards[self._cursor:] = rewards[:split]
            self._dones[self._cursor:] = dones[:split]
            wrap = n - split
            self._obs[:wrap] = obs[split:]
            self._actions[:wrap] = actions[split:]
            self._rewards[:wrap] = rewards[split:]
            self._dones[:wrap] = dones[split:]
            self._cursor = wrap
        self._size = min(self._size + n, self.capacity)

    # ----- read -------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(obs, actions, rewards, dones)` shaped `(B, L, ...)`.

        Sequence start indices are chosen uniformly over the valid range
        [0, size - seq_len]. A sequence may include an episode boundary
        (a done flag mid-sequence) — Dreamer's continue-head learns
        precisely this signal, so episode boundaries inside a sequence
        carry information rather than being noise. The continue-head
        target is `1 - done`.
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive (got {seq_len})")
        if self._size < seq_len:
            raise RuntimeError(
                f"replay buffer has {self._size} steps; cannot sample "
                f"sequences of length {seq_len}"
            )
        # When the buffer wraps, naive index ranges can cross the
        # cursor. To keep this simple and correct: only sample in the
        # contiguous "logical" window. While not yet full, that's
        # [0, _size]; once full, the contiguous slice depends on
        # cursor position and we conservatively avoid spanning the
        # write boundary by clamping.
        max_start = self._size - seq_len
        starts = [self._rng.randint(0, max_start) for _ in range(batch_size)]
        # Convert "logical" starts to physical indices when the buffer
        # has wrapped. While not full, logical == physical. When full,
        # the oldest logical step is at `_cursor`.
        if self.is_full:
            base = self._cursor
        else:
            base = 0
        # Build per-sequence index arrays once, then vectorize lookup.
        # Use modular arithmetic so a sequence starting near the end of
        # the ring continues at the start.
        idx = np.empty((batch_size, seq_len), dtype=np.int64)
        for b, s in enumerate(starts):
            for t in range(seq_len):
                idx[b, t] = (base + s + t) % self.capacity
        flat = idx.reshape(-1)
        obs = self._obs[flat].reshape(batch_size, seq_len, *self._shapes.obs)
        actions = self._actions[flat].reshape(batch_size, seq_len)
        rewards = self._rewards[flat].reshape(batch_size, seq_len)
        dones = self._dones[flat].reshape(batch_size, seq_len)
        return obs, actions, rewards, dones

    # ----- bookkeeping -----------------------------------------------

    def stats(self) -> dict[str, int | float]:
        """Quick health snapshot for the dashboard."""
        return {
            "size": self._size,
            "capacity": self.capacity,
            "fill_fraction": float(self._size) / float(self.capacity),
        }
