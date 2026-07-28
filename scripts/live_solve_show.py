"""THE LIVE SHOW: the search system beating a game level by level, in real
time, in one window — built for streaming via OBS and walking away.

The window has two synchronized views:

  HERO CAM (left, big) — a dedicated 1x-paced emulator that continuously
    replays the swarm's current best attempt. When a level falls, the
    victory lap plays here, start to flagpole, with its APU soundtrack
    featured.
  THE SWARM (right, grid) — every solver worker's live frames at full
    machine speed: the actual parallel search, restarts and all.

AUDIO — the CHORUS: during search you hear ALL workers' real APU output
mixed together at machine speed (pitch rises with search speed — that is
the authentic sound of the machine working; --chorus-pitch native gives
normal-pitch granular slices instead). On each victory lap the mix
crossfades to the hero cam's clean 1x soundtrack, then back to the
chorus. --audio hero|chorus|off selects other mixes. Audio production
is decoupled from pacing in the pool, so the chorus costs sample
generation only — never search speed.

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
        gx_bucket=16, y_band=32, swim_gx_ceiling=0,
        # NO mid-level archive flushes on stream: pickling a multi-GB
        # archive runs on the solver thread and froze every swarm tile
        # for minutes at a time (observed live: a 6.5-min stall at a
        # 1.9 GB archive, 90 min into 4-3). The show never resumes
        # mid-level — resume restarts from the banked entrance — so
        # the flush bought nothing.
        flush_secs=10 ** 9,
        seed=int(time.time()) % 100000,
    )


def default_args(**overrides) -> SimpleNamespace:
    """The show's knobs with their defaults; kwargs override."""
    ns = SimpleNamespace(minutes_per_level=120.0, workers=12, scale=3,
                         volume=0.6, resume=False, view="swarm",
                         audio="both", chorus_pitch="ff", chorus_voices=6,
                         profile=str(DEFAULT_PROFILE))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


APU_RATE = 43653  # native APU sample rate (see watch_asm.py)


class HeroCam(threading.Thread):
    """A 1x-paced emulator, the featured audio voice, in its own thread.

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
        mixer = self.show.mixer
        if mixer is not None:
            try:
                samples = self.env.get_audio()
                if samples is not None and len(samples) > 0:
                    mixer.push_audio(self.show.hero_voice, samples, APU_RATE)
            except Exception:
                pass

    def _play(self, root: bytes, actions, tag: str) -> bool:
        """Replay actions at 1x; returns False if interrupted."""
        self.env.load_state(root)
        # Rooting convention no-op = ONE POOL STEP = frame_skip frames.
        # A single-frame no-op leaves the replay 3 frames out of phase
        # with the solver's trajectory — enough to kill Mario mid-lap
        # (verified on a real solution: 1-frame no-op dies in 1-1,
        # 4-frame no-op reproduces the clear into 1-2 exactly).
        for _ in range(self.show.fs):
            self.env.step(0)
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
        # One shared mixer: worker voices 0..N-1 (the CHORUS — every
        # solver instance's real APU output) + the hero cam as voice N.
        # Phase presets crossfade between them via per-voice intensity.
        self.hero_voice = int(args.workers)
        self.mixer = None
        if getattr(args, "audio", "both") != "off":
            try:
                self.mixer = nes_core.AudioMixer(
                    num_instances=args.workers + 1)
                self.mixer.set_mode("all")
                self.mixer.set_volume(args.volume)
                self.mixer.start()
            except Exception as e:
                sys.stderr.write(f"audio unavailable (silent show): {e}\n")

    def _apply_mix(self, phase: str) -> None:
        """Crossfade chorus vs hero per show phase."""
        if self.mixer is None:
            return
        n = self.args.workers
        n_sing = min(int(getattr(self.args, "chorus_voices", 6)), n)
        want = getattr(self.args, "audio", "both")
        chorus_on = want in ("both", "chorus") and n_sing > 0
        hero_on = want in ("both", "hero")
        # 1/sqrt(voices) keeps the chorus from clipping the master.
        chorus_lvl = (1.0 / max(1.0, n_sing ** 0.5)) if chorus_on else 0.0
        if phase == "search":
            c, h = chorus_lvl, (0.15 if (hero_on and chorus_on)
                                else (1.0 if hero_on else 0.0))
        elif phase == "lap":
            c, h = (0.15 * chorus_lvl), (1.0 if self.mixer else 0.0)
        else:                              # boot / done: hero carries it
            c, h = 0.0, 1.0
        try:
            for i in range(n):
                self.mixer.set_instance_intensity(i, c)
            self.mixer.set_instance_intensity(self.hero_voice, h)
        except Exception:
            pass

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
        self._apply_mix("boot")
        chorus = (self.mixer is not None
                  and getattr(self.args, "audio", "both") in ("both", "chorus"))
        ff = getattr(self.args, "chorus_pitch", "ff") == "ff"
        while not self.stop:
            self.mode = "search"
            self.status = f"searching {self.level}"
            self._apply_mix("search")
            out = self.show_dir / f"lvl_{self.level}"
            s = Solver(solver_args(self.profile_path, entrance, out,
                                   self.args.minutes_per_level,
                                   self.args.workers))
            root_bytes = Path(entrance).read_bytes()
            n_voices = min(int(getattr(self.args, "chorus_voices", 6)),
                           self.args.workers)
            if chorus:
                # Audio production WITHOUT pacing (decoupled in the pool:
                # set_worker_audio overrides the pace<->audio welding).
                # Sample-gen scales with emulation speed though — every
                # singing worker synthesizes APU audio at 30-60x realtime
                # — so all-12-voices cost ~8x search throughput (measured
                # 380 vs 4,400 sps). Default 6 voices keeps the wall of
                # sound at roughly half the cost; --chorus-voices 12
                # brings back the full choir, 0 disables.
                for i in range(n_voices):
                    s.pool.set_worker_audio(i, True)
            pump = {"t": time.time()}

            def pump_chorus(sv, _self=self, pump=pump, ff=ff,
                            n_voices=n_voices):
                now = time.time()
                dt = now - pump["t"]
                if dt < 0.05:
                    return
                pump["t"] = now
                for i in range(n_voices):
                    try:
                        raw = sv.pool.drain_audio(i)
                        if not len(raw):
                            continue
                        arr = np.frombuffer(bytes(raw), dtype=np.int16)
                        if ff:
                            # True machine-speed audio: push at the rate
                            # the swarm actually produced it — the mixer
                            # resamples it into wall time (pitch rises
                            # with search speed; that IS the sound).
                            rate = min(int(len(arr) / max(dt, 1e-3)),
                                       APU_RATE * 64)
                            _self.mixer.push_audio(i, arr, max(rate, 8000))
                        else:
                            # Native pitch: keep the freshest slice that
                            # fits real time, drop the rest (granular).
                            keep = int(dt * APU_RATE)
                            _self.mixer.push_audio(i, arr[-keep:], APU_RATE)
                    except Exception:
                        pass

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

            def hook(rs, sv, _self=self, chorus=chorus):
                if _self.stop:
                    sv.stop = True
                _self.frames = [r[0] for r in rs]
                if chorus:
                    pump_chorus(sv)
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
            self._apply_mix("lap")
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
    ap.add_argument("--audio", choices=("both", "chorus", "hero", "off"),
                    default="both",
                    help="both = the swarm CHORUS during search + hero cam "
                         "featured on laps; chorus = swarm only; hero = "
                         "1x best-attempt cam only; off = silent")
    ap.add_argument("--chorus-pitch", choices=("ff", "native"), default="ff",
                    help="ff = true machine-speed audio (pitch rises with "
                         "search speed); native = normal-pitch granular "
                         "slices of each worker's latest audio")
    ap.add_argument("--chorus-voices", type=int, default=6,
                    help="How many workers sing (audio synthesis costs "
                         "search speed; 6 = wall of sound at ~half the "
                         "cost, 12 = full choir, 0 = none)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = LiveSolveWindow(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
