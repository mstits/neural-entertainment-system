"""
Frame preprocessing + input button constants.

Pure-Python helpers used by the trainer, GUI, and tests. No emulator
imports — safe to load without any backend installed.
"""

from __future__ import annotations

import numpy as np


FRAME_HEIGHT = 240
FRAME_WIDTH = 256


BUTTON_RIGHT  = 0b10000000
BUTTON_LEFT   = 0b01000000
BUTTON_DOWN   = 0b00100000
BUTTON_UP     = 0b00010000
BUTTON_START  = 0b00001000
BUTTON_SELECT = 0b00000100
BUTTON_B      = 0b00000010
BUTTON_A      = 0b00000001
BUTTON_NOOP   = 0b00000000


_LUMA_R, _LUMA_G, _LUMA_B = 0.299, 0.587, 0.114


def _grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    g = (
        frame[..., 0].astype(np.float32) * _LUMA_R
        + frame[..., 1].astype(np.float32) * _LUMA_G
        + frame[..., 2].astype(np.float32) * _LUMA_B
    )
    return np.clip(g, 0.0, 255.0).astype(np.uint8)


def _resize_area(gray: np.ndarray, target_size: int) -> np.ndarray:
    """Area-based downsample (pixel-area weighted average).

    Pure NumPy implementation — no cv2 / PIL dep. Matches cv2.INTER_AREA
    and PIL Image.BOX numerically for integer downsample ratios, which is
    all we ever hit (240→84, 256→84). The Rust preprocess path (used by
    the Pool) is the hot path; this helper is for one-off callers (tests,
    scripts) where NumPy is cheap enough.
    """
    h, w = gray.shape
    if h == target_size and w == target_size:
        return gray.copy()
    # Integer-ratio area average via reshape + mean. Non-integer ratios
    # fall back to a bilinear interpolation (good-enough for
    # non-hot-path callers).
    if h % target_size == 0 and w % target_size == 0:
        sh = h // target_size
        sw = w // target_size
        return (
            gray.reshape(target_size, sh, target_size, sw)
            .mean(axis=(1, 3))
            .astype(np.uint8)
        )
    # Fallback: nearest-neighbor is adequate for tests; the trainer
    # always goes through the Rust NEON path.
    y_idx = (np.arange(target_size) * h // target_size).astype(np.intp)
    x_idx = (np.arange(target_size) * w // target_size).astype(np.intp)
    return gray[y_idx[:, None], x_idx[None, :]].astype(np.uint8)


def preprocess_frame(frame: np.ndarray, target_size: int = 84) -> np.ndarray:
    """(240, 256, 3) uint8 RGB → (target_size, target_size) uint8 grayscale."""
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got {frame.dtype}")
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError(f"expected (240, 256, 3) frame, got {frame.shape}")
    gray = _grayscale(frame)
    return _resize_area(gray, target_size)


def stack_frames(frames, stack_size: int = 4, target_size: int = 84) -> np.ndarray:
    lst = list(frames)
    if not lst:
        return np.zeros((stack_size, target_size, target_size), dtype=np.uint8)
    if len(lst) < stack_size:
        # Each pad slot must be its own array — `[lst[0]] * N` would
        # create N references to the same buffer, so any later in-place
        # mutation of one slot would propagate to all of them. np.stack
        # itself copies, but the intermediate list is also handed back
        # for inspection in some call sites.
        pad = [lst[0].copy() for _ in range(stack_size - len(lst))]
        lst = pad + lst
    else:
        lst = lst[-stack_size:]
    return np.stack(lst, axis=0)


class FrameStacker:
    """Rolling window of preprocessed frames for one environment.

    Ring-buffer storage; `_view_oldest_to_newest()` returns a view
    reordered to oldest→newest (the CNN input layout). `_out` is
    returned by reference — callers must not mutate.

    `dtype` defaults to `np.uint8` (legacy observations, 0-255 range).
    Set to `np.float16` when the Rust pool emits already-normalized
    [0, 1] half-precision observations via `set_preprocess_f16(True)`.
    """

    def __init__(
        self,
        stack_size: int = 4,
        target_size: int = 84,
        dtype: np.dtype = np.uint8,
    ) -> None:
        self.stack_size = stack_size
        self.target_size = target_size
        self.dtype = np.dtype(dtype)
        self._ring = np.zeros((stack_size, target_size, target_size), dtype=self.dtype)
        self._out = np.zeros((stack_size, target_size, target_size), dtype=self.dtype)
        self._head = 0
        self._filled = 0

    def _view_oldest_to_newest(self) -> np.ndarray:
        n = self.stack_size
        for i in range(n):
            self._out[i] = self._ring[(self._head + i) % n]
        return self._out

    def reset(self, frame: np.ndarray, preprocessed: np.ndarray | None = None) -> np.ndarray:
        pp = preprocessed if preprocessed is not None else preprocess_frame(frame, self.target_size)
        for i in range(self.stack_size):
            self._ring[i] = pp
        self._head = 0
        self._filled = self.stack_size
        return self._view_oldest_to_newest()

    def push(self, frame: np.ndarray, preprocessed: np.ndarray | None = None) -> np.ndarray:
        pp = preprocessed if preprocessed is not None else preprocess_frame(frame, self.target_size)
        self._ring[self._head] = pp
        self._head = (self._head + 1) % self.stack_size
        if self._filled < self.stack_size:
            self._filled += 1
        return self._view_oldest_to_newest()


class TileFeatureStacker:
    """Rolling stack of tile-mode feature vectors for one environment.

    Tile-mode observations are 1-D feature vectors (e.g. SMB's 175-dim
    grid+scalars). Without stacking, the policy sees only the
    instantaneous state — no temporal information, no derivative of
    motion. yumouwei (2022) demonstrated empirically that stack=1
    fails to clear SMB 1-1 while stack=2 or stack=4 succeeds, even
    though our scalar features already include vel_x. Frame stacking
    captures longer-horizon temporal context (enemy motion, jump
    arcs) that point-in-time scalars don't.

    Output shape: (stack_size * feature_dim,) — a flat concatenation
    of [oldest, ..., newest] feature vectors. The MLP consumes the
    flat vector unchanged; its first Linear layer learns the temporal
    convolution implicitly.
    """

    def __init__(
        self,
        stack_size: int = 4,
        feature_dim: int = 175,
        dtype: np.dtype = np.int8,
    ) -> None:
        self.stack_size = stack_size
        self.feature_dim = feature_dim
        self.dtype = np.dtype(dtype)
        self._ring = np.zeros((stack_size, feature_dim), dtype=self.dtype)
        self._out = np.zeros((stack_size * feature_dim,), dtype=self.dtype)
        self._head = 0
        self._filled = 0

    def _flatten_oldest_to_newest(self) -> np.ndarray:
        # Oldest→newest = ring rotated so head sits at position 0,
        # then flattened. np.concatenate emits a single C-level memcpy
        # into _out (one allocation reused across calls) instead of
        # the prior stack_size Python iterations with per-slot slice
        # writes. Matters during BC pretrain where this fires once
        # per emitted (state, action) sample on 100k+-sample demos.
        head = self._head
        self._out[: (self.stack_size - head) * self.feature_dim] = (
            self._ring[head:].reshape(-1)
        )
        if head:
            self._out[(self.stack_size - head) * self.feature_dim:] = (
                self._ring[:head].reshape(-1)
            )
        return self._out

    def reset(self, features: np.ndarray) -> np.ndarray:
        """Fill the ring with the initial features (so stack[0..N-1] all
        equal the first observation). Matches FrameStacker.reset
        semantics — no zero-padding leaks at episode start."""
        for i in range(self.stack_size):
            self._ring[i] = features
        self._head = 0
        self._filled = self.stack_size
        return self._flatten_oldest_to_newest()

    def push(self, features: np.ndarray) -> np.ndarray:
        self._ring[self._head] = features
        self._head = (self._head + 1) % self.stack_size
        if self._filled < self.stack_size:
            self._filled += 1
        return self._flatten_oldest_to_newest()
