"""
Per-worker runtime instrumentation for diagnosing visual corruption and
"frozen Link" symptoms. Opt-in: set `NES_DEBUG=1` in the env.

What it does on every worker step:

  1. RAM fingerprint check on reset
       CRC32 the 2 KB of work RAM after each reset(). First reset is the
       golden hash. Mismatch = the env's reset path isn't preserving
       full state across generations. Logs a loud warning.

  2. Freeze detector
       Tracks Link's ($0070, $0084) position over a sliding 60-step
       window. If unchanged for the whole window despite directional
       actions being issued, dumps a "frozen worker" bundle: last 200
       actions, current RAM, current frame.

  3. Corrupt-frame detector
       A reference NES frame has tile indices drawn from the game's
       CHR-ROM. SimpleNES renders black rectangles (tile index 0) when
       asked to draw a tile that doesn't exist in the active CHR bank —
       that's the "solid black geometry on sand" signature users see.
       We count large contiguous black pixel regions; above a threshold
       we dump the frame + action context for later analysis.

Output goes to $NES_DEBUG_DIR (default /tmp/nes_debug/).
"""

from __future__ import annotations

import json
import os
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


def is_enabled() -> bool:
    return os.environ.get("NES_DEBUG", "").strip() not in ("", "0", "false", "False")


def debug_dir() -> Path:
    base = os.environ.get("NES_DEBUG_DIR", "/tmp/nes_debug")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


# NO GAME CONSTANTS LIVE HERE. This module used to carry
# `ZELDA_LINK_X = 0x0070` / `ZELDA_LINK_Y = 0x0084` and select them with
# `if "zelda" in self.game`, i.e. a display-name substring choosing two
# hand-authored addresses — both of which are recorded in this project's
# purity quarantine (q_link_x / q_link_y). Same defect class as BREACH
# PATH 1 in the reward dispatch: the name is not a declaration, and a
# quarantined address must not arrive by inference. The freeze check now
# runs only on coordinates a CALLER passes in explicitly, and is skipped
# otherwise.


# Bitmask bits copied from nes_environment so we can detect "the agent
# asked Link to move but he didn't". Keep in sync.
_BTN_RIGHT = 0x80
_BTN_LEFT  = 0x40
_BTN_DOWN  = 0x20
_BTN_UP    = 0x10
_DIRECTIONAL_MASK = _BTN_RIGHT | _BTN_LEFT | _BTN_DOWN | _BTN_UP


class WorkerDebug:
    """All per-worker diagnostics, wrapped in one object so the hot path
    stays readable in `_worker_main`. Create with `WorkerDebug.maybe(worker_id)`
    and it returns None if NES_DEBUG isn't set — the worker then
    doesn't pay any runtime cost."""

    FREEZE_WINDOW = 60          # steps over which "unchanged position" = frozen
    CORRUPT_BLACK_PIXEL_FRAC = 0.40  # >= 40% of frame is exact (0,0,0) → probably corrupt

    def __init__(self, worker_id: int, game: str = "unknown",
                 pos_addrs: Optional[tuple[int, int]] = None,
                 hud_rows: int = 0) -> None:
        """`game` is a LABEL for the log line and nothing else — it selects
        no behaviour. `pos_addrs` is the (x, y) RAM pair the freeze check
        watches; without it the freeze check is skipped rather than guessed
        at. `hud_rows` is how many top scanlines to exclude from the
        black-frame corruption metric (a legitimately dark HUD is not
        corruption); 0 means measure the whole frame."""
        self.worker_id = worker_id
        self.game = game.lower()
        self.pos_addrs = pos_addrs
        self.hud_rows = int(hud_rows)
        self._golden_ram_crc: Optional[int] = None
        self._reset_count = 0
        self._step_count = 0
        self._action_history: deque[int] = deque(maxlen=500)
        # Ring of recent Link positions + directional-action flags.
        self._pos_history: deque[tuple[int, int]] = deque(maxlen=self.FREEZE_WINDOW)
        self._directional_hist: deque[bool] = deque(maxlen=self.FREEZE_WINDOW)
        self._dir = debug_dir()
        self._log_path = self._dir / f"worker_{worker_id:02d}.log"
        self._frozen_dumps = 0
        self._corrupt_dumps = 0
        self._log(f"worker {worker_id} debug mode enabled, game={game}")

    @classmethod
    def maybe(cls, worker_id: int, game: str = "unknown",
              pos_addrs: Optional[tuple[int, int]] = None,
              hud_rows: int = 0) -> Optional["WorkerDebug"]:
        if not is_enabled():
            return None
        return cls(worker_id, game, pos_addrs=pos_addrs, hud_rows=hud_rows)

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] w{self.worker_id:02d} {msg}\n"
        try:
            with open(self._log_path, "a") as f:
                f.write(line)
        except Exception:
            pass

    def on_reset(self, ram: bytes) -> None:
        """Check RAM matches the golden snapshot. If not, state is
        drifting across resets — evidence of an env backup/restore
        fidelity bug."""
        self._reset_count += 1
        self._step_count = 0
        self._pos_history.clear()
        self._directional_hist.clear()
        crc = zlib.crc32(ram)
        if self._golden_ram_crc is None:
            self._golden_ram_crc = crc
            self._log(f"reset #{self._reset_count}: golden CRC=0x{crc:08x}")
            return
        if crc != self._golden_ram_crc:
            # The C-side `_restore()` is bit-exact across resets in
            # standalone tests (verified via 100 cycles × 1000 random
            # steps each → 0 drift). When this fires, it means the
            # emulator derailed mid-episode and the post-restore RAM
            # diverges from the golden snapshot — i.e., our derail
            # detector missed something or rebuild was needed.
            self._log(
                f"⚠ reset #{self._reset_count}: RAM CRC drift — "
                f"got 0x{crc:08x}, expected 0x{self._golden_ram_crc:08x}. "
                "Likely cause: emulator derailed (mapper state corrupted) "
                "and rebuild on next reset is needed."
            )

    def on_step(
        self,
        action_bitmask: int,
        ram: bytes,
        frame: np.ndarray,
    ) -> None:
        """Advance all detectors. Dumps are side-effectful file writes."""
        self._step_count += 1
        self._action_history.append(action_bitmask)

        # Freeze check. Runs only when the caller declared which two RAM
        # bytes hold the position — never inferred from the game name.
        if self.pos_addrs is not None:
            ax, ay = self.pos_addrs
            self._pos_history.append((ram[ax], ram[ay]))
            self._directional_hist.append(bool(action_bitmask & _DIRECTIONAL_MASK))

            if len(self._pos_history) == self.FREEZE_WINDOW:
                positions = set(self._pos_history)
                dir_inputs = sum(self._directional_hist)
                # "Frozen" = one position for the whole window AND the agent
                # issued at least half as many directional inputs as the
                # window size (so the agent was genuinely trying to move).
                if len(positions) == 1 and dir_inputs >= self.FREEZE_WINDOW // 2:
                    self._dump_frozen(ram, frame, list(positions)[0], dir_inputs)

        # Corruption check: % of exact-black pixels in the play area. A
        # legitimately dark HUD would inflate that fraction, so the caller
        # may declare how many top scanlines to skip; default 0 measures
        # the whole frame. The check fires only when "obviously corrupt" —
        # most games don't naturally produce frames that are 40%+
        # pure-black-(0,0,0) RGB pixels.
        play_area = frame[self.hud_rows:, :, :] if self.hud_rows else frame
        black_mask = np.all(play_area == 0, axis=-1)
        black_frac = float(black_mask.mean())
        if black_frac >= self.CORRUPT_BLACK_PIXEL_FRAC:
            self._dump_corrupt(ram, frame, black_frac)

    # ------------------------------------------------------------------

    def _dump_frozen(
        self,
        ram: bytes,
        frame: np.ndarray,
        stuck_pos: tuple[int, int],
        dir_inputs: int,
    ) -> None:
        # Avoid spamming: cap dumps per worker.
        if self._frozen_dumps >= 5:
            return
        self._frozen_dumps += 1
        tag = f"frozen_w{self.worker_id:02d}_r{self._reset_count}_s{self._step_count}"
        self._log(
            f"FROZEN at step {self._step_count}: Link stuck at {stuck_pos} "
            f"for {self.FREEZE_WINDOW} steps with {dir_inputs} directional inputs."
        )
        self._write_bundle(tag, ram, frame, note=f"stuck at {stuck_pos}")

    def _dump_corrupt(self, ram: bytes, frame: np.ndarray, black_frac: float) -> None:
        if self._corrupt_dumps >= 5:
            return
        self._corrupt_dumps += 1
        tag = f"corrupt_w{self.worker_id:02d}_r{self._reset_count}_s{self._step_count}"
        self._log(
            f"CORRUPT frame at step {self._step_count}: "
            f"{black_frac:.1%} of play area is solid black."
        )
        self._write_bundle(tag, ram, frame, note=f"black_frac={black_frac:.3f}")

    def _write_bundle(
        self,
        tag: str,
        ram: bytes,
        frame: np.ndarray,
        note: str,
    ) -> None:
        """Dump RAM + frame + recent actions so we can replay the sequence
        on a fresh emulator and see if it reproduces."""
        out = self._dir / tag
        out.mkdir(parents=True, exist_ok=True)
        # Frame as raw PNG via imageio (optional; fall back to .npy).
        # Catch BOTH ImportError (no imageio) AND any write-side failure
        # (disk full, weird shape) — diagnostics must never crash the
        # worker they're trying to diagnose.
        try:
            import imageio.v3 as iio
            iio.imwrite(out / "frame.png", frame)
        except Exception:
            try:
                np.save(out / "frame.npy", frame)
            except Exception:
                pass
        try:
            with open(out / "ram.bin", "wb") as f:
                f.write(ram)
        except Exception:
            pass
        try:
            # Mask each value to a uint8 — protects against any future
            # action-bitmask widening that would otherwise raise
            # `ValueError: bytes must be in range(0, 256)`.
            with open(out / "actions.bin", "wb") as f:
                f.write(bytes(b & 0xFF for b in self._action_history))
        except Exception:
            pass
        meta = {
            "worker": self.worker_id,
            "game": self.game,
            "reset_count": self._reset_count,
            "step_count": self._step_count,
            "note": note,
            "action_history_len": len(self._action_history),
        }
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
