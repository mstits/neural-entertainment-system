"""THE LIVE SHOW: the search system solving Super Mario Bros level by level,
in real time, in a window — for streaming via OBS and walking away.

What the audience sees, per level, for as long as it takes from 1-1 to 8-4:

  1. SEARCH — a live view of a solver worker exploring at full machine speed
     (thousands of steps/second; the window samples its frames at 60 fps),
     with a status line: current level, elapsed, frontier, archive size.
  2. VICTORY LAP — the moment the level falls, the discovered solution
     replays at native 1x speed WITH AUDIO (the Rust core's Core Audio
     mixer), start to flagpole. The payoff moment, every level.
  3. The next level's entrance is extracted from the clear (the same
     transition contract as the verified full run) and search resumes.

This is the EXHIBITION ledger performing live: the machine genuinely
DISCOVERING each solution on air — nothing is pre-recorded, and a restart
resumes from the last banked entrance. Runs hours to days; that's the show.

Usage:
  caffeinate -dis python scripts/live_solve_show.py            # from power-on
  python scripts/live_solve_show.py --resume                   # continue
  python scripts/live_solve_show.py --minutes-per-level 90
Keys: Q quits (progress is banked; --resume continues the campaign).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402
from scripts.go_explore_solve import Solver  # noqa: E402
from scripts.go_explore_chain import extract_next_entrance  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
PROFILE = REPO / "configs/smb_4_4_micro.yaml"
SHOW_DIR = REPO / "runs/live_show"


def solver_args(root_state: str, out: Path, minutes: float, workers: int):
    return SimpleNamespace(
        root_state=root_state, profile=str(PROFILE), out=str(out),
        workers=workers, minutes=minutes, want_solutions=1,
        burst=200, deep_bias=0.6, sticky=0.35, max_steps=4000,
        gx_bucket=16, y_band=32, swim_gx_ceiling=0, flush_secs=1200,
        seed=int(time.time()) % 100000,
    )


class Show:
    """Producer thread: runs the chain; the Qt window consumes its state."""

    def __init__(self, args):
        self.args = args
        self.profile = yaml.safe_load(PROFILE.read_text())
        self.bm = action_space_to_bitmasks(self.profile["action_space"])
        self.fs = int(self.profile.get("frame_skip", 4))
        self.mode = "boot"          # boot | search | lap | done
        self.level = "1-1"
        self.status = "starting"
        self.frame = None           # latest RGB frame (any mode)
        self.lap_env = None
        self.stop = False
        SHOW_DIR.mkdir(parents=True, exist_ok=True)
        self.state_file = SHOW_DIR / "progress.json"

    # -- power-on entrance (replayed live so even the boot is on air) ----
    def _power_on_entrance(self) -> bytes:
        env = nes_core.NESEnvironment(ROM)
        env.reset()
        try:
            env.set_realtime_pace(True)
        except Exception:
            pass
        seq = [0] * 244 + [0x08] * 12 + [0] * 148   # per-frame: title, START, settle
        for m in seq:
            env.step(int(m))
            self.frame = np.asarray(env.get_frame())
        # hand the settled state to the solver via a temp file
        blob = env.save_state()
        p = SHOW_DIR / "entrance_1-1.state"
        p.write_bytes(bytes(blob))
        return str(p)

    def _victory_lap(self, root_state: str, actions):
        """Replay the discovered solution at 1x with audio."""
        self.mode = "lap"
        env = nes_core.NESEnvironment(ROM)
        env.reset()
        env.load_state(Path(root_state).read_bytes())
        try:
            env.set_realtime_pace(True)
        except Exception:
            pass
        mixer = None
        try:
            mixer = nes_core.AudioMixer(num_instances=1)
            mixer.set_volume(self.args.volume)
            mixer.set_mode("solo-0")
            mixer.start()
        except Exception:
            pass
        env.step(0)                                  # rooting convention no-op
        for a in actions:
            if self.stop:
                break
            for _ in range(self.fs):                 # tape masks: frame_skip each
                env.step(int(self.bm[int(a)]))
                self.frame = np.asarray(env.get_frame())
        for _ in range(180):                         # linger on the clear
            if self.stop:
                break
            env.step(0)
            self.frame = np.asarray(env.get_frame())
        if mixer is not None:
            try:
                mixer.stop()
            except Exception:
                pass

    def run(self):
        # resume or power on
        prog = {}
        if self.args.resume and self.state_file.exists():
            prog = json.loads(self.state_file.read_text())
        if prog.get("entrance"):
            entrance, self.level = prog["entrance"], prog["level"]
            self.status = f"resuming at {self.level}"
        else:
            self.status = "power-on"
            entrance = self._power_on_entrance()
        while not self.stop and self.level != "done":
            self.mode = "search"
            self.status = f"searching {self.level}"
            out = SHOW_DIR / f"lvl_{self.level}"
            s = Solver(solver_args(entrance, out,
                                   self.args.minutes_per_level,
                                   self.args.workers))

            def hook(r0, sv, _self=self):
                if _self.stop:
                    sv.stop = True
                f = r0[0]
                if f is not None:
                    _self.frame = np.asarray(f)
                _self.status = (
                    f"searching {_self.level} — "
                    f"{sv.steps_done/1e6:.1f}M steps, "
                    f"frontier gx {sv.max_gx_in_area.get(sv.max_area, 0)}, "
                    f"{len(sv.archive)} cells")

            s.step_hook = hook
            s.pool.set_headless(False)               # render worker frames
            s.seed()
            s.explore()
            sols = sorted(out.glob("solutions/sol_*.actions.npy"))
            if not sols:
                self.status = f"{self.level}: budget spent — searching again"
                continue                              # fresh seed, another round
            actions = np.load(sols[0]).astype(int)
            self.status = f"{self.level} SOLVED — victory lap"
            self._victory_lap(entrance, actions)
            if self.level == "8-4":
                self.mode = "done"
                self.status = "THE GAME IS COMPLETE"
                break
            nxt, wd = extract_next_entrance(
                self.profile, Path(entrance).read_bytes(), actions,
                SHOW_DIR / f"entrance_after_{self.level}.state")
            if nxt is None:
                self.status = f"{self.level}: no transition?! retrying"
                continue
            entrance = nxt
            self.level = f"{wd[0]+1}-{wd[1]+1}"
            self.state_file.write_text(json.dumps(
                {"entrance": entrance, "level": self.level}))


def default_args(**overrides) -> SimpleNamespace:
    """The show's knobs with their defaults; kwargs override."""
    ns = SimpleNamespace(minutes_per_level=120.0, workers=12, scale=3,
                         volume=0.6, resume=False)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow,  # noqa: E402
                             QVBoxLayout, QWidget)


class LiveSolveWindow(QMainWindow):
    """The show as a window: owns the producer thread's lifecycle, so it
    embeds in the training GUI (a launcher button) or runs standalone."""

    def __init__(self, args=None, parent=None):
        super().__init__(parent)
        args = args or default_args()
        self.show_state = Show(args)
        self.setWindowTitle("Super Mario Bros — live solve")
        self.view = QLabel()
        self.view.setFixedSize(256 * args.scale, 240 * args.scale)
        self.caption = QLabel("starting…")
        self.caption.setStyleSheet(
            "font-family: Menlo; font-size: 14px; padding: 6px;")
        box = QWidget(); lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(self.view); lay.addWidget(self.caption)
        self.setCentralWidget(box)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._render)
        self.timer.start(16)
        self.thread = threading.Thread(target=self.show_state.run,
                                       daemon=True)
        self.thread.start()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Q:
            self.close()

    def closeEvent(self, ev):
        self.show_state.stop = True   # progress is banked; --resume continues
        self.timer.stop()
        super().closeEvent(ev)

    def _render(self):
        f = self.show_state.frame
        if f is not None and f.ndim == 3:
            h, w = f.shape[0], f.shape[1]
            img = QImage(f.astype(np.uint8).tobytes(), w, h, 3 * w,
                         QImage.Format.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(img).scaled(
                self.view.width(), self.view.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))
        tag = {"search": "SEARCHING (machine speed)",
               "lap": "SOLVED — victory lap (1x, live audio)",
               "boot": "POWER ON", "done": "COMPLETE"}.get(
                   self.show_state.mode, "")
        self.caption.setText(f"[{tag}]  {self.show_state.status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes-per-level", type=float, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--volume", type=float, default=0.6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = LiveSolveWindow(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
