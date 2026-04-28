"""nes-py wrapped to look like `nes_core.NESEnvironment` for the GUI
play window. nes_core's emulator core has a long-standing CPU/PPU
timing divergence from nes-py that breaks SMB / Zelda / Contra
gameplay (Mario falls through floor, Link can't transition screens).

Until the nes_core core bug is fixed (multi-day PPU/CPU/timing dive),
this adapter lets the Play window use nes-py — which we know plays
these games correctly — without touching the rest of the codebase.

API mirrors `nes_core.NESEnvironment`:
  * `__init__(rom_path, frame_skip=1)` — accepts the same kwargs we
    actually use; ignores anything we don't.
  * `reset()` → returns the first frame as a numpy uint8 (240, 256, 3).
  * `step(action_bitmask)` → returns `(frame, done)`, matching nes_core.
    Skips frames per `frame_skip`.
  * `get_frame()` → last rendered frame.
  * `get_ram_range(start, end)` → numpy uint8 view of RAM[start:end].
  * `get_ram(addr)` → single byte.

Button bitmask layout matches nes_core / `frame_utils.py`:
  RIGHT 0x80 LEFT 0x40 DOWN 0x20 UP 0x10
  START 0x08 SELECT 0x04 B 0x02 A 0x01

Toggle the GUI to use this via env var:
    NES_PY_BACKEND=1 python -m src.gui.main
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nes_py


class NESpyEnvironment:
    def __init__(self, rom_path, frame_skip: int = 1, **_unused):
        self._env = nes_py.NESEnv(str(rom_path))
        self._frame_skip = max(1, int(frame_skip))
        self._last_frame: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        frame = self._env.reset()
        # nes-py returns (240, 256, 3) uint8 RGB. Same as nes_core.
        self._last_frame = np.asarray(frame, dtype=np.uint8)
        return self._last_frame

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        frame = None
        done = False
        for _ in range(self._frame_skip):
            r = self._env.step(int(action))
            # gym 4-tuple: (obs, reward, done, info). Older nes-py.
            frame, _reward, done, _info = r if len(r) == 4 else (r[0], 0, r[2], {})
            if done:
                break
        self._last_frame = np.asarray(frame, dtype=np.uint8)
        return self._last_frame, bool(done)

    def get_frame(self) -> np.ndarray:
        if self._last_frame is None:
            return np.zeros((240, 256, 3), dtype=np.uint8)
        return self._last_frame

    def get_ram_range(self, start: int, end: int) -> np.ndarray:
        return np.asarray(self._env.ram[start:end], dtype=np.uint8)

    def get_ram(self, addr: int) -> int:
        return int(self._env.ram[addr])
