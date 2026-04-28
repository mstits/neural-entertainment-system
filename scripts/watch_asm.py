#!/usr/bin/env python3
"""
Minimal visual watcher for the AArch64-ASM-accelerated NES core.

Opens a PyQt6 window, loads a ROM (Zelda by default), and runs it
through `nes_core.NESEnvironment`.

Keys:
    Arrow keys   D-pad
    Z / X        A / B
    Enter        Start
    Shift        Select
    Q            Quit
    P            Pause / resume
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import nes_core  # noqa: E402

# Poll physical macOS key state via Quartz — Qt's auto-repeat events
# on macOS are unreliable for held keys. Falls back to None-op if the
# user is on another platform or Quartz can't be imported.
try:
    from Quartz import (
        CGEventSourceKeyState,
        kCGEventSourceStateHIDSystemState,
    )

    def _key_down(vk: int) -> bool:
        return bool(CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, vk))
except ImportError:
    def _key_down(_vk: int) -> bool:
        return False

# macOS virtual key codes → (NES button bit).
# Bit layout matches `src/emulation/frame_utils.py` / nes-py convention:
# right=bit7, left=6, down=5, up=4, start=3, select=2, B=1, A=0.
KEYMAP: list[tuple[int, int]] = [
    (126, 0x10),  # Up
    (125, 0x20),  # Down
    (123, 0x40),  # Left
    (124, 0x80),  # Right
    (  6, 0x01),  # Z → A
    (  7, 0x02),  # X → B
    ( 36, 0x08),  # Return → Start
    ( 56, 0x04),  # Shift (left) → Select
    ( 60, 0x04),  # Shift (right) → Select
]
SCALE = 3


class Viewer(QMainWindow):
    def __init__(self, rom_path: str):
        super().__init__()
        with open(rom_path, "rb") as f:
            hdr = f.read(16)
        mapper = (hdr[6] >> 4) | (hdr[7] & 0xF0)
        prg_kb = hdr[4] * 16
        # Every mapper whose `prg_asm_ptr` returns `Some(...)` — see
        # `nes_core/src/mapper/*.rs`. These expose a flat 32KB PRG
        # window the AArch64 ASM core can fetch from directly.
        asm_mappers = {0, 1, 2, 4, 5, 7, 11, 21, 22, 23, 24, 25, 26, 66, 69, 71}
        backend = (
            "ASM-accelerated (AArch64)" if mapper in asm_mappers
            else "pure-Rust fallback"
        )
        self.setWindowTitle(
            f"NES — {Path(rom_path).name} — mapper {mapper} "
            f"({prg_kb}KB PRG) — {backend}"
        )

        self.env = nes_core.NESEnvironment(rom_path)
        self.env.reset()
        # Real-time pacing: env.step sleeps the remainder of 1/60s so
        # we stay locked to NES native framerate. Keeps APU samples
        # flowing at a steady cadence — the audio mixer's resampler
        # can ride that cleanly; without it, render-timer jitter
        # starves + overruns the ring buffer, causing the static
        # pops user reported.
        try:
            self.env.set_realtime_pace(True)
        except Exception:
            pass

        # Audio — the Rust core has a cpal/Core Audio mixer ready to go;
        # wire the APU's PCM output from this instance into it and start
        # the stream. Any failure falls back to silent (no crash).
        self.mixer = None
        try:
            self.mixer = nes_core.AudioMixer(num_instances=1)
            # Mode 1 typically = "solo slot 0"; tweak via GUI if not.
            self.mixer.set_volume(0.6)
            self.mixer.set_mode("solo-0")
            self.mixer.start()
        except Exception as e:
            sys.stderr.write(f"audio init failed: {e}\n")
            self.mixer = None

        self.paused = False
        self.held = 0

        self.label = QLabel()
        self.label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCentralWidget(self.label)
        self.label.setFixedSize(256 * SCALE, 240 * SCALE)

        self.setStatusBar(self.statusBar())
        self.resize(256 * SCALE, 240 * SCALE + 28)

        self.frame_count = 0
        self.fps_start = time.perf_counter()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._step_and_render)
        self.timer.start(16)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def _poll_buttons(self) -> int:
        """Rebuild the NES button bitmask from physical key state."""
        bits = 0
        # Only read keys when our window is active, so the user typing
        # in another app doesn't drive the NES.
        if self.isActiveWindow():
            for vk, btn in KEYMAP:
                if _key_down(vk):
                    bits |= btn
        return bits

    def _step_and_render(self) -> None:
        if self.paused:
            return

        action = self._poll_buttons()
        self.held = action

        self.env.step(action)
        # Push APU samples to the mixer (43653 Hz; resampler in mixer
        # converts to device rate).
        if self.mixer is not None:
            try:
                samples = self.env.get_audio()
                if samples is not None and len(samples) > 0:
                    self.mixer.push_audio(0, samples, 43653)
            except Exception:
                pass
        frame = self.env.get_frame()  # (240, 256, 3) u8
        arr = np.ascontiguousarray(frame)
        img = QImage(arr.data, 256, 240, 256 * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            256 * SCALE, 240 * SCALE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.label.setPixmap(pix)

        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_start
        if elapsed >= 0.25:
            fps = self.frame_count / elapsed
            names = []
            if action & 0x80: names.append("→")
            if action & 0x40: names.append("←")
            if action & 0x20: names.append("↓")
            if action & 0x10: names.append("↑")
            if action & 0x08: names.append("Start")
            if action & 0x04: names.append("Sel")
            if action & 0x02: names.append("B")
            if action & 0x01: names.append("A")
            txt = "+".join(names) if names else "—"
            self.statusBar().showMessage(
                f"{fps:6.1f} fps  ·  keys={txt}  ·  ASM"
            )
            self.frame_count = 0
            self.fps_start = now

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        k = ev.key()
        if k == Qt.Key.Key_Q:
            self.close()
            return
        if k == Qt.Key.Key_P:
            self.paused = not self.paused
            self.statusBar().showMessage(
                "PAUSED" if self.paused else "RESUMED"
            )
            return
        # NES buttons are read from physical key state each frame —
        # Qt's keyPress/Release events aren't reliable for held keys
        # on macOS due to auto-repeat.
        ev.accept()

    def keyReleaseEvent(self, ev: QKeyEvent) -> None:
        ev.accept()


def main() -> None:
    default = str(REPO / "roms" / "zelda.ines1.nes")
    rom = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(rom):
        print(f"ROM not found: {rom}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    w = Viewer(rom)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
