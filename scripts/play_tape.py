"""Live tape player — replay the completed-run controller tape in a real
window with real audio, for streaming through OBS.

Replays `runs/full_run/full_tape.npy` (per-frame-skip-4 button masks) from an
actual cold boot at native 60 fps with the Rust core's Core Audio mixer
playing the APU. Each tape mask is held for `frame_skip` frames, which is
emulation-identical to the pool replay that verified the tape. No scenes, no
overlays — OBS captures the window and the app audio; this is just the game,
playing itself.

Usage:
  python scripts/play_tape.py                       # window + audio, 60 fps
  python scripts/play_tape.py --scale 4             # bigger window
  python scripts/play_tape.py --headless-verify     # no window: fast replay,
                                                    # assert the ending
Keys: Space = pause/resume, Q = quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
TAPE = REPO / "runs/full_run/full_tape.npy"
FRAME_SKIP = 4


def headless_verify(tape) -> int:
    env = nes_core.NESEnvironment(ROM)
    env.reset()
    t0 = time.perf_counter()
    for i, mask in enumerate(tape):
        for _ in range(FRAME_SKIP):
            env.step(int(mask))
        if i % 4000 == 0:
            print(f"[verify] step {i}/{len(tape)} wd="
                  f"{int(env.get_ram(0x75F))+1}-{int(env.get_ram(0x75C))+1}",
                  flush=True)
    om = int(env.get_ram(0x770))
    dt = time.perf_counter() - t0
    print(f"[verify] done in {dt:.0f}s: opermode={om} "
          f"{'VICTORY — tape verified on the per-frame path' if om == 2 else '*** MISMATCH ***'}")
    return 0 if om == 2 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default=str(TAPE))
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--volume", type=float, default=0.6)
    ap.add_argument("--headless-verify", action="store_true")
    args = ap.parse_args()
    tape = np.load(args.tape)
    print(f"[tape] {len(tape)} masks ({len(tape)*FRAME_SKIP/60.0/60.0:.1f} min)")
    if args.headless_verify:
        return headless_verify(tape)

    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

    class Player(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Super Mario Bros — the complete run")
            self.env = nes_core.NESEnvironment(ROM)
            self.env.reset()
            try:
                self.env.set_realtime_pace(True)
            except Exception:
                pass
            self.mixer = None
            try:
                self.mixer = nes_core.AudioMixer(num_instances=1)
                self.mixer.set_volume(args.volume)
                self.mixer.set_mode("solo-0")
                self.mixer.start()
            except Exception as e:
                sys.stderr.write(f"audio init failed (silent playback): {e}\n")
            self.label = QLabel()
            self.setCentralWidget(self.label)
            self.label.setFixedSize(256 * args.scale, 240 * args.scale)
            self.resize(256 * args.scale, 240 * args.scale)
            self.idx = 0          # tape index
            self.phase = 0        # frame within the held mask
            self.paused = False
            self.done = False
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self._tick)
            self.timer.start(16)

        def keyPressEvent(self, ev):
            if ev.key() == Qt.Key.Key_Space:
                self.paused = not self.paused
            elif ev.key() == Qt.Key.Key_Q:
                self.close()

        def _tick(self):
            if self.paused:
                return
            mask = int(tape[self.idx]) if not self.done else 0
            self.env.step(mask)
            self.phase += 1
            if self.phase >= FRAME_SKIP:
                self.phase = 0
                if not self.done:
                    self.idx += 1
                    if self.idx >= len(tape):
                        self.done = True   # hold on the ending; keep rendering
            frame = np.asarray(self.env.get_frame())
            h, w = frame.shape[0], frame.shape[1]
            img = QImage(frame.astype(np.uint8).tobytes(), w, h,
                         3 * w, QImage.Format.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img).scaled(
                self.label.width(), self.label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))
            if self.idx % 900 == 0 and self.phase == 0:
                self.setWindowTitle(
                    f"Super Mario Bros — the complete run — World "
                    f"{int(self.env.get_ram(0x75F))+1}-"
                    f"{int(self.env.get_ram(0x75C))+1}")

    app = QApplication(sys.argv)
    p = Player()
    p.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
