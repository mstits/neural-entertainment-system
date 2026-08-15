"""Self-imitation buffer (`reinforce.sil`) — store what already worked.

Full (obs, action) trajectories of episodes that CLEAR the level — the
vanilla loop's completion-diff detector, the same trusted signal the clear
counter and PLR use — go into a small FIFO ring. Each PPO iteration with a
non-empty buffer adds a BC cross-entropy term on minibatches sampled
uniformly over the stored steps, coefficient `bc_coef`. Deliberately
minimal: uniform sampling, no prioritized replay, no value clipping — the
mechanism exists so a rare stochastic clear is consolidated instead of
washed out by the next thousand failures.

Tile-mode only by construction (the trainer gates the config): a stored
trajectory is ~step_count x feature_dim int8, so 64 clears of a 2000-step
level cost ~90 MB at the 700-dim stacked feature — pixel trajectories
would be two orders of magnitude bigger.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class SelfImitationBuffer:
    """FIFO trajectory store with uniform per-step sampling."""

    def __init__(self, capacity: int) -> None:
        self._trajs: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=max(1, int(capacity))
        )
        # Lifetime clear counter (survives evictions) — metrics surface.
        self.total_clears = 0
        # Flat sample cache, rebuilt lazily after any add/eviction.
        self._flat_obs: np.ndarray | None = None
        self._flat_act: np.ndarray | None = None
        self._dirty = False

    def add(self, obs: np.ndarray, actions: np.ndarray) -> None:
        """Store one cleared episode's (L, F) observations + (L,) actions."""
        obs = np.asarray(obs)
        actions = np.asarray(actions, dtype=np.int64)
        if actions.size == 0:
            return
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"trajectory length mismatch: {obs.shape[0]} obs vs "
                f"{actions.shape[0]} actions"
            )
        self._trajs.append((obs, actions))
        self.total_clears += 1
        self._dirty = True

    def __len__(self) -> int:
        return len(self._trajs)

    @property
    def n_steps(self) -> int:
        return sum(a.shape[0] for _, a in self._trajs)

    def _rebuild(self) -> None:
        self._flat_obs = np.concatenate([o for o, _ in self._trajs], axis=0)
        self._flat_act = np.concatenate([a for _, a in self._trajs], axis=0)
        self._dirty = False

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform with-replacement draw of `batch_size` (obs, action) pairs
        over all stored steps. Uses the global numpy RNG so seeded runs
        stay reproducible (house convention — the PPO minibatch
        permutation draws from the same stream)."""
        if not self._trajs:
            raise ValueError("sample() on an empty self-imitation buffer")
        if self._dirty or self._flat_obs is None:
            self._rebuild()
        idx = np.random.randint(0, self._flat_act.shape[0], size=int(batch_size))
        return self._flat_obs[idx], self._flat_act[idx]
