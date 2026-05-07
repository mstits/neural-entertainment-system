"""Frame stacking for tile-mode observations.

Per-worker rolling window of the last N tile feature vectors,
concatenated oldest→newest into a single flat array. Mirrors
`src/emulation/frame_utils.FrameStacker` for the pixel-mode
case but works on flat int8 vectors instead of (84, 84) frames.

Why frame stacking matters even for tile features (which already
include `vel_x`, `vel_y`, `on_ground` scalars):

* The tile grid encodes static positions of enemies/blocks but no
  motion direction. With a single frame, "Goomba at (5, 6)" doesn't
  tell the policy whether the Goomba is approaching, retreating, or
  stationary — they all look identical. Stacking N=4 frames makes
  the trajectory visible: (7,6) (6,6) (5,6) → Goomba moving left,
  about to be in jumping range.
* Both canonical SMB-RL recipes that have empirically cleared 1-1
  (uvipen + yumouwei) use 4-frame stacks. yumouwei explicitly
  reports n_stack=1 fails to clear, n_stack≥2 succeeds.
* The first-derivative scalars (vel_x/vel_y/on_ground) substitute
  for SOME of what stacking provides, but only for Mario himself
  — not for the moving entities the agent must avoid.

The deque is reset to all-current-frame on episode boundary so the
policy sees a "valid" stack from frame 0; otherwise the first 3
frames of every episode would have stale data from the previous
episode's tail.
"""

from __future__ import annotations

import numpy as np


class TileFrameStacker:
    """Ring-buffer of the last `stack_size` tile feature vectors.

    Shape contract: `extract`-style methods return a flat
    `(stack_size * feature_dim,)` int8 array, oldest→newest.
    """

    def __init__(self, stack_size: int, feature_dim: int) -> None:
        if stack_size < 1:
            raise ValueError(f"stack_size must be >= 1, got {stack_size}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        self.stack_size = int(stack_size)
        self.feature_dim = int(feature_dim)
        self.stacked_dim = self.stack_size * self.feature_dim
        # Ring buffer of raw per-frame feature vectors.
        self._ring = np.zeros((self.stack_size, self.feature_dim), dtype=np.int8)
        # Output buffer reordered oldest→newest. Returned by reference,
        # callers must not mutate.
        self._out = np.zeros(self.stacked_dim, dtype=np.int8)
        self._head = 0

    def _view_oldest_to_newest(self) -> np.ndarray:
        # Concatenate the ring entries starting at `_head` (oldest)
        # and wrapping around back to `_head - 1` (newest).
        for i in range(self.stack_size):
            slot = (self._head + i) % self.stack_size
            self._out[i * self.feature_dim:(i + 1) * self.feature_dim] = self._ring[slot]
        return self._out

    def reset(self, features: np.ndarray) -> np.ndarray:
        """Initialize the stack with `features` repeated `stack_size` times.

        Called at episode boundary so the policy sees a coherent stack
        from t=0 instead of stale data from the previous episode.
        """
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"features shape {features.shape} != ({self.feature_dim},)"
            )
        for i in range(self.stack_size):
            self._ring[i] = features
        self._head = 0
        return self._view_oldest_to_newest()

    def push(self, features: np.ndarray) -> np.ndarray:
        """Add a new frame and return the updated oldest→newest stack."""
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"features shape {features.shape} != ({self.feature_dim},)"
            )
        self._ring[self._head] = features
        self._head = (self._head + 1) % self.stack_size
        return self._view_oldest_to_newest()
