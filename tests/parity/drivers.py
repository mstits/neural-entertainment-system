"""Thin wrappers around nes_core and nes-py so the diff harness can talk
to both with the same (step, frame) interface.

Both emulators accept an 8-bit NES button bitmask:
    A=1, B=2, Select=4, Start=8, Up=16, Down=32, Left=64, Right=128
Both emit `(240, 256, 3)` uint8 RGB.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


class NESCoreDriver:
    def __init__(self, rom_path: str | Path):
        import nes_core
        self._env = nes_core.NESEnvironment(str(rom_path), frame_skip=1)
        self._env.reset()

    def step(self, buttons: int) -> None:
        self._env.step(int(buttons) & 0xFF)

    def frame(self) -> np.ndarray:
        return np.asarray(self._env.get_frame())


class NESPyDriver:
    def __init__(self, rom_path: str | Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import nes_py
        self._env = nes_py.NESEnv(str(rom_path))
        self._env.reset()

    def step(self, buttons: int) -> None:
        # nes-py step returns (obs, reward, done, info); we only need side effect.
        self._env.step(int(buttons) & 0xFF)

    def frame(self) -> np.ndarray:
        return np.asarray(self._env.screen).copy()
