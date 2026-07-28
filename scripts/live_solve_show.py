"""THE LIVE SHOW: the search system beating a game level by level, in real
time, in one window — built for streaming via OBS and walking away.

The window has two synchronized views:

  HERO CAM (left, big, LIVE AUDIO) — a dedicated 1x-paced emulator that
    continuously replays the swarm's current best attempt with the real
    APU soundtrack. However fast the search runs, there is always real
    gameplay at real speed with real audio on screen. When a level falls,
    the victory lap plays here, start to flagpole.
  THE SWARM (right, grid) — every solver worker's live frames at full
    machine speed: the actual parallel search, restarts and all.

Per level: the swarm searches; the hero cam narrates the deepest attempt
so far; on a clear, the hero cam plays the discovered solution; the next
level's entrance is extracted and the campaign continues. Progress banks
per level; a restart resumes the campaign. Runs hours to days.

Usage:
  make show                                   # SMB from power-on
  make show PROFILE=configs/castlevania.yaml  # any solve-ready game
  python scripts/live_solve_show.py --view solo --scale 4   # hero cam only
Keys: Q quits (progress banked; --resume continues).
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
from scripts.go_explore_solve import Solver, make_game  # noqa: E402
from scripts.go_explore_chain import extract_next_entrance  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

DEFAULT_PROFILE = REPO / "configs/smb_4_4_micro.yaml"
SHOW_ROOT = REPO / "runs/live_show"
SMB_FINALE_LABEL = "8-4"


def solver_args(profile_path: str, root_state: str, out: Path,
                minutes: float, workers: int):
    return SimpleNamespace(
        root_state=root_state, profile=str(profile_path), out=str(out),
        workers=workers, minutes=minutes, want_solutions=1,
        burst=200, deep_bias=0.6, sticky=0.35, max_steps=4000,
        gx_bucket=16, y_band=32, swim_gx_ceiling=0, flush_secs=1200,
        seed=int(time.time()) % 100000,
    )


def default_args(**overrides) -> SimpleNamespace:
    """The show's knobs with their defaults; kwargs override."""
    ns = SimpleNamespace(minutes_per_level=120.0, workers=12, scale=3,
                         volume=0.6, resume=False, view="swarm",
                         profile=str(DEFAULT_PROFILE))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class HeroCam(threading.Thread):
    """A 1x-paced emulator with live audio, owned by its own thread.

    Modes (set via methods, executed in the run loop):
      idle  — sit at the level entrance, stepping no-ops (ambient music).
      best  — replay the swarm's current deepest trace, refetching the
              latest best each cycle (the "current best attempt" cam).
      lap   — play a discovered solution once, then signal done.
    """

    def __init__(self, show: "Show"):
        super().__init__(daemon=True)
        self.show = show
        self.env = nes_core.NESEnvironment(show.game.rom)
        self.env.reset()
        try:
            self.env.set_realtime_pace(True)
        except Exception:
            pass
        self.mixer = None
        try:
            self.mixer = nes_core.AudioMixer(num_instances=1)
            self.mixer.set_volume(show.args.volume)
            self.mixer.set_mode("solo-0")
            self.mixer.start()
        except Exception as e:
            sys.stderr.write(f"hero audio unavailable (silent): {e}\n")
        self._root: bytes | None = None
        self._get_best = None           # callable -> (root_bytes, trace) | None
        self._lap = None                # (root_bytes, actions ndarray)
        self.lap_done = threading.Event()
        self.stop = False

    # -- producer-side controls -----------------------------------------
    def set_level(self, root_bytes: bytes, get_best) -> None:
        self._root = root_bytes
        self._get_best = get_best

    def play_lap(self, root_bytes: bytes, actions) -> None:
        self.lap_done.clear()
        self._lap = (root_bytes, np.asarray(actions, dtype=np.int64))

    # -- internals -------------------------------------------------------
    def _emit(self):
        self.show.frame = np.asarray(self.env.get_frame())

    def _play(self, root: bytes, actions, tag: str) -> bool:
        """Replay actions at 1x; returns False if interrupted."""
        self.env.load_state(root)
        self.env.step(0)                  # rooting convention no-op
        self._emit()
        for a in actions:
            if self.stop or (tag == "best" and self._lap is not None):
                return False
            for _ in range(self.show.fs):
                self.env.step(int(self.show.bm[int(a)]))
            self._emit()
        return True

    def run(self) -> None:
        while not self.stop:
            lap = self._lap
            if lap is not None:
                root, actions = lap
                self._play(root, actions, "lap")
                for _ in range(150):      # linger on the clear
                    if self.stop:
                        break
                    self.env.step(0)
                    self._emit()
                self._lap = None
                self.lap_done.set()
                continue
            best = self._get_best() if self._get_best else None
            if best is not None:
                self._play(best[0], best[1], "best")
                continue
            if self._root is not None:    # idle at the entrance
                self.env.load_state(self._root)
                self._emit()
                for _ in range(240):
                    if self.stop or self._lap is not None or (
                            self._get_best and self._get_best()):
                        break
                    self.env.step(0)
                    self._emit()
            else:
                time.sleep(0.1)


class Show:
    """Producer thread: runs the campaign; the Qt window consumes state."""

    def __init__(self, args):
        self.args = args
        self.profile_path = str(getattr(args, "profile", DEFAULT_PROFILE))
        self.profile = yaml.safe_load(Path(self.profile_path).read_text())
        self.game = make_game(self.profile)
        self.is_smb = "solve" not in self.profile
        self.bm = action_space_to_bitmasks(self.profile["action_space"])
        self.fs = int(self.profile.get("frame_skip", 4))
        self.mode = "boot"          # boot | search | lap | done
        self.level = "?"
        self.status = "starting"
        self.frame = None           # hero-cam frame (big view)
        self.frames = None          # per-worker frames (swarm grid)
        self.stop = False
        self.show_dir = SHOW_ROOT / Path(self.profile_path).stem
        self.show_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.show_dir / "progress.json"
        self.hero: HeroCam | None = None

    def _label_of_state(self, state_path: str) -> str:
        pool = nes_core.Pool(rom_path=self.game.rom, num_workers=1,
                             frame_skip=self.fs)
        pool.set_headless(True)
        pool.reset_all()
        pool.load_worker_state(0, Path(state_path).read_bytes())
        r = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
        key = self.game.level_key(r)
        pool.shutdown()
        return self.game.label(key)

    # -- campaign root ---------------------------------------------------
    # SMB: an actual power-on boot, replayed live so even the title screen
    # is on air. Other games: the profile's captured start state (their
    # title demos ignore input, which is why start states exist at all).
    def _campaign_root(self) -> str:
        if not self.is_smb:
            p = REPO / self.profile["start_state_path"]
            self.level = self._label_of_state(str(p))
            return str(p)
        env = nes_core.NESEnvironment(self.game.rom)
        env.reset()
        try:
            env.set_realtime_pace(True)
        except Exception:
            pass
        seq = [0] * 244 + [0x08] * 12 + [0] * 148   # per-frame: title, START, settle
        for m in seq:
            env.step(int(m))
            self.frame = np.asarray(env.get_frame())
        blob = env.save_state()
        p = self.show_dir / "entrance_start.state"
        p.write_bytes(bytes(blob))
        self.level = "1-1"
        return str(p)

    def run(self):
        prog = {}
        if self.args.resume and self.state_file.exists():
            prog = json.loads(self.state_file.read_text())
        if prog.get("entrance"):
            entrance, self.level = prog["entrance"], prog["level"]
            self.status = f"resuming at {self.level}"
        else:
            self.status = "starting campaign"
            entrance = self._campaign_root()
        self.hero = HeroCam(self)
        self.hero.start()
        while not self.stop:
            self.mode = "search"
            self.status = f"searching {self.level}"
            out = self.show_dir / f"lvl_{self.level}"
            s = Solver(solver_args(self.profile_path, entrance, out,
                                   self.args.minutes_per_level,
                                   self.args.workers))
            root_bytes = Path(entrance).read_bytes()

            # Hero-cam feed: the deepest archived trace, refreshed at most
            # every 2 s (a scan of the trace table, cheap at this size).
            best_cache = {"t": 0.0, "val": None}

            def get_best(sv=s, cache=best_cache):
                now = time.time()
                if now - cache["t"] < 2.0:
                    return cache["val"]
                cache["t"] = now
                try:
                    items = list(sv.traces.items())
                    if not items:
                        return cache["val"]
                    key = max(
                        items,
                        key=lambda kv: sv.archive.cells[kv[0]].best_score
                        if kv[0] in sv.archive.cells else -1)[0]
                    rec = sv.traces[key]
                    root_id, tb = rec[0], rec[1]
                    rb = Path(sv.roots[root_id]["path"]).read_bytes()
                    cache["val"] = (rb, np.frombuffer(tb, dtype=np.uint8))
                except Exception:
                    pass
                return cache["val"]

            self.hero.set_level(root_bytes, get_best)

            def hook(rs, sv, _self=self):
                if _self.stop:
                    sv.stop = True
                _self.frames = [r[0] for r in rs]
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
            self.mode = "lap"
            self.status = f"{self.level} SOLVED — victory lap"
            self.hero.play_lap(root_bytes, actions)
            while not self.hero.lap_done.wait(timeout=0.5):
                if self.stop:
                    break
            if self.is_smb and self.level == SMB_FINALE_LABEL:
                self.mode = "done"
                self.status = "THE GAME IS COMPLETE"
                break
            nxt, key = extract_next_entrance(
                self.profile, root_bytes, actions,
                self.show_dir / f"entrance_after_{self.level}.state")
            if nxt is None:
                self.mode = "done"
                self.status = (f"{self.level} SOLVED — no onward level "
                               "found (campaign end?)")
                break
            entrance = nxt
            self.level = self.game.label(key)
            self.state_file.write_text(json.dumps(
                {"entrance": entrance, "level": self.level}))
        if self.hero is not None:
            self.hero.stop = True


from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout,  # noqa: E402
                             QLabel, QMainWindow, QVBoxLayout, QWidget)


def _to_pixmap(f, w, h):
    fh, fw = f.shape[0], f.shape[1]
    img = QImage(f.astype(np.uint8).tobytes(), fw, fh, 3 * fw,
                 QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        w, h, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation)


class LiveSolveWindow(QMainWindow):
    """Hero cam (1x + audio) beside the live swarm grid."""

    def __init__(self, args=None, parent=None):
        super().__init__(parent)
        args = args or default_args()
        self.args = args
        self.show_state = Show(args)
        game_name = self.show_state.profile.get("name", "NES")
        self.setWindowTitle(f"{game_name} — live solve")

        self.hero_view = QLabel()
        self.hero_view.setFixedSize(256 * args.scale, 240 * args.scale)
        self.hero_tag = QLabel("")
        self.hero_tag.setStyleSheet(
            "font-family: Menlo; font-size: 13px; padding: 4px;")
        left = QWidget(); lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(0)
        lv.addWidget(self.hero_view); lv.addWidget(self.hero_tag)

        self.tiles: list[QLabel] = []
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
        row.addWidget(left)
        if args.view == "swarm":
            cols = 4
            rows = (args.workers + cols - 1) // cols
            tile_h = (240 * args.scale) // rows - 4
            tile_w = int(tile_h * 256 / 240)
            gridw = QWidget(); grid = QGridLayout(gridw)
            grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(4)
            for i in range(args.workers):
                t = QLabel()
                t.setFixedSize(tile_w, tile_h)
                t.setStyleSheet("background: #111;")
                grid.addWidget(t, i // cols, i % cols)
                self.tiles.append(t)
            row.addWidget(gridw)

        box = QWidget(); root = QVBoxLayout(box)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        rw = QWidget(); rw.setLayout(row)
        root.addWidget(rw)
        self.caption = QLabel("starting…")
        self.caption.setStyleSheet(
            "font-family: Menlo; font-size: 14px; padding: 6px;")
        root.addWidget(self.caption)
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
        self.show_state.stop = True   # progress banked; --resume continues
        if self.show_state.hero is not None:
            self.show_state.hero.stop = True
        self.timer.stop()
        super().closeEvent(ev)

    def _render(self):
        st = self.show_state
        f = st.frame
        if f is not None and f.ndim == 3:
            self.hero_view.setPixmap(_to_pixmap(
                f, self.hero_view.width(), self.hero_view.height()))
        if self.tiles and st.mode == "search" and st.frames:
            for t, wf in zip(self.tiles, st.frames):
                if wf is not None:
                    a = np.asarray(wf)
                    if a.ndim == 3:
                        t.setPixmap(_to_pixmap(a, t.width(), t.height()))
        hero_tags = {
            "boot": "POWER ON (real speed, live audio)",
            "search": "HERO CAM — current best attempt (real speed, live audio)",
            "lap": "VICTORY LAP — the discovered solution (real speed, live audio)",
            "done": "COMPLETE"}
        self.hero_tag.setText(hero_tags.get(st.mode, ""))
        tag = {"search": "swarm: full machine speed",
               "lap": "SOLVED", "boot": "booting", "done": ""}.get(st.mode, "")
        self.caption.setText(f"[{tag}]  {st.status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE),
                    help="Game profile; non-SMB profiles need a verified "
                         "`solve:` section (scripts/verify_ram_map.py).")
    ap.add_argument("--minutes-per-level", type=float, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--volume", type=float, default=0.6)
    ap.add_argument("--view", choices=("swarm", "solo"), default="swarm",
                    help="swarm = hero cam + all-worker grid; solo = hero only")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = LiveSolveWindow(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
