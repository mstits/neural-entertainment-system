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


# THE ESCALATION LADDER — the campaign's actual wall-breaking behavior,
# encoded. During the receipted campaign a stuck level got a human
# diagnosis and a recipe change; the show used to just re-roll the same
# config forever. Now each failed budget escalates: the campaign-proven
# search settings first (seed 0; note the campaign receipts recorded
# neither worker count nor profile, so "same trajectory" is not claimed
# — same recipe is), then a fresh seed, then micro-search cell
# granularity (the castle recipe: gx_bucket 8 / y_band 16 credits the
# small precise maneuvers castles demand), then micro with a doubled
# budget. Seeds are the attempt index — deterministic and receiptable.
# The forensics of the 8h stream day validated the short budgets:
# 1-3 and 4-3 both fell within ~10 min of a FRESH archive after 2h
# stuck on a stale one.
ESCALATION = [
    # Arm 0: the maze-coverage research recipes LEAD (user call,
    # 2026-07-28 after three legacy attempts burned an evening on
    # 4-4's seed-dependent cliff: expiries at 0.9M and 1.06M cells,
    # the second PAST the campaign seed's 966k breakthrough). R1
    # count-based selection is Go-Explore's own native prior; R2
    # boundary throttling stops hammering pinned walls — commissioned
    # for coverage walls, expected benign-to-helpful on momentum
    # walls. Every solution receipt stamps its arm, so the ledger
    # tracks exactly which recipe cleared what.
    {"name": "coverage recipes", "minutes": 90, "gx_bucket": 16,
     "y_band": 32, "sel_mode": "count", "frontier_throttle": 3},
    # Arm 1: ENTRANCE DIVERSIFICATION — re-solve the PREVIOUS level
    # with a fresh seed to mint a different legitimate entrance, then
    # restart the ladder from it. Rationale (measured 2026-07-28): two
    # entrances to the same level can differ in hidden game state (a
    # full-state diff of tonight's vs the campaign's 4-4 entrance shows
    # 23 RAM bytes of divergent hidden state; the campaign's exact
    # winning inputs DIE at action 650 from tonight's entrance) — so
    # per-entrance difficulty varies, and a wall that resists one
    # entrance instance may fall easily from another. Generic: no byte
    # interpretation, no route knowledge — just "come at the door from
    # a different day".
    {"name": "re-entrance", "reenter": True, "minutes": 25,
     "gx_bucket": 16, "y_band": 32, "sel_mode": "legacy",
     "frontier_throttle": 0},
    # Arm 2: the receipted campaign's legacy recipe.
    {"name": "campaign recipe", "minutes": 60, "gx_bucket": 16,
     "y_band": 32, "sel_mode": "legacy", "frontier_throttle": 0},
    # Arm 2: fresh-seed legacy roll (momentum walls fall to these
    # inside 10 min when they fall at all).
    {"name": "fresh seed", "minutes": 45, "gx_bucket": 16,
     "y_band": 32, "sel_mode": "legacy", "frontier_throttle": 0},
    # Arm 3: micro-search granularity (castle precision) + recipes.
    {"name": "micro + recipes", "minutes": 90, "gx_bucket": 8,
     "y_band": 16, "sel_mode": "count", "frontier_throttle": 3},
]


def solver_args(profile_path: str, root_state: str, out: Path,
                minutes: float, workers: int, attempt: int = 0):
    arm = ESCALATION[min(attempt, len(ESCALATION) - 1)]
    return SimpleNamespace(
        root_state=root_state, profile=str(profile_path), out=str(out),
        workers=workers,
        minutes=min(float(arm["minutes"]), minutes),
        want_solutions=1,
        # Campaign-proven search params (the solver defaults that beat
        # all 32 levels in-chain; sticky 0.5 sustains the full-speed
        # run-ups athletic gaps demand — the show's original 0.35
        # pinned two seeds at 4-3's momentum wall for 2h).
        burst=64, deep_bias=0.4, sticky=0.5, max_steps=4000,
        gx_bucket=int(arm["gx_bucket"]), y_band=int(arm["y_band"]),
        sel_mode=str(arm.get("sel_mode", "legacy")),
        frontier_throttle=int(arm.get("frontier_throttle", 0)),
        swim_gx_ceiling=0,
        # NO mid-level archive flushes on stream: pickling a multi-GB
        # archive on the solver thread froze every swarm tile for
        # minutes (observed live: 6.5-min stall at 1.9 GB). The show
        # never resumes mid-level, so flushes bought nothing.
        flush_secs=10 ** 9,
        seed=attempt,
    ), arm["name"]


def default_args(**overrides) -> SimpleNamespace:
    """The show's knobs with their defaults; kwargs override."""
    ns = SimpleNamespace(minutes_per_level=120.0, workers=12, scale=3,
                         volume=0.6, resume=False, view="swarm",
                         audio="both", chorus_pitch="ff", chorus_voices=-1,
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
        # PACING IS DONE IN PYTHON, NOT THE EMULATOR. set_realtime_pace
        # sleeps INSIDE env.step() while holding the GIL — measured 85%
        # solver-throughput loss (3,050 -> 469 sps) with a paced hero
        # thread running. A Python time.sleep releases the GIL, so the
        # solver runs free while the hero holds 60 fps wall time.
        self._next_frame_t = time.monotonic()
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
    def _pace(self, frames: int):
        """Hold 1x wall time with a GIL-releasing Python sleep."""
        self._next_frame_t += frames / 60.0
        delay = self._next_frame_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.25:            # fell behind (lap start etc.): resync
            self._next_frame_t = time.monotonic()

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
            self._pace(self.show.fs)
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
                    self._pace(1)
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
                    self._pace(1)
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
        cv = int(getattr(self.args, "chorus_voices", -1))
        n_sing = n if cv < 0 else min(cv, n)
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

    def _reenter(self, prev_entrance: str, prev_level: str,
                 arm: dict, attempt: int):
        """Entrance diversification: re-solve the PREVIOUS level with a
        fresh seed to mint a different legitimate entrance for the
        current one (different hidden-state stream, same maze). Returns
        the new entrance path or None."""
        self.mode = "search"
        self._apply_mix("search")
        self.status = (f"re-solving {prev_level} for a fresh "
                       f"{self.level} entrance [re-entrance arm]")
        out = self.show_dir / f"reenter_{prev_level}_a{attempt}"
        sargs, _ = solver_args(self.profile_path, prev_entrance, out,
                               float(arm["minutes"]), self.args.workers,
                               attempt=0)
        sargs.minutes = float(arm["minutes"])
        sargs.seed = 1000 + attempt
        sargs.sel_mode = "legacy"
        sargs.frontier_throttle = 0
        s = Solver(sargs)

        def hook(rs, sv, _self=self):
            if _self.stop:
                sv.stop = True
            _self.frames = [r[0] for r in rs]
            _self.status = (
                f"re-solving {prev_level} (fresh entrance for "
                f"{_self.level}) — {sv.steps_done/1e6:.1f}M steps, "
                f"{len(sv.archive)} cells")

        s.step_hook = hook
        s.pool.set_headless(False)
        s.seed()
        s.explore()
        sols = sorted(out.glob("solutions/sol_*.actions.npy"))
        if not sols or self.stop:
            return None
        actions = np.load(sols[0]).astype(int)
        nxt, key = extract_next_entrance(
            self.profile, Path(prev_entrance).read_bytes(), actions,
            self.show_dir / f"entrance_{self.level}_alt{attempt}.state")
        if nxt is None or self.game.label(key) != self.level:
            return None
        return nxt

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
        prev_entrance = prog.get("prev_entrance")
        prev_level = prog.get("prev_level")
        resume_attempt = int(prog.get("attempt", 0))
        if prog.get("entrance"):
            entrance, self.level = prog["entrance"], prog["level"]
            self.status = f"resuming at {self.level}"
        else:
            self.status = "starting campaign"
            entrance = self._campaign_root()

        def bank(attempt_n: int):
            self.state_file.write_text(json.dumps(
                {"entrance": entrance, "level": self.level,
                 "attempt": attempt_n,
                 "prev_entrance": prev_entrance,
                 "prev_level": prev_level}))
        self.hero = HeroCam(self)
        self.hero.start()
        self._apply_mix("boot")
        chorus = (self.mixer is not None
                  and getattr(self.args, "audio", "both") in ("both", "chorus"))
        ff = getattr(self.args, "chorus_pitch", "ff") == "ff"
        attempt = resume_attempt
        cur_level = self.level
        while not self.stop:
            if self.level != cur_level:
                cur_level, attempt = self.level, 0
            arm_probe = ESCALATION[min(attempt, len(ESCALATION) - 1)]
            if arm_probe.get("reenter"):
                if prev_entrance and prev_level and Path(prev_entrance).exists():
                    new_ent = self._reenter(prev_entrance, prev_level,
                                            arm_probe, attempt)
                    if new_ent is not None:
                        entrance = new_ent
                        attempt = 0          # fresh instance, fresh ladder
                        bank(attempt)
                        continue
                attempt += 1                  # no prev info / re-solve failed
                bank(attempt)
                continue
            self.mode = "search"
            self._apply_mix("search")
            out = self.show_dir / f"lvl_{self.level}"
            # A prior session may have dumped a solution and died before
            # banking the next entrance — don't burn a full budget
            # rediscovering it (audit finding: stale-solution edge).
            stale = sorted(out.glob("solutions/sol_*.actions.npy"))
            if stale:
                actions = np.load(stale[-1]).astype(int)
                self.mode = "lap"
                self.status = (f"{self.level}: banked solution found — "
                               "victory lap")
                self._apply_mix("lap")
                self.hero.play_lap(Path(entrance).read_bytes(), actions)
                while not self.hero.lap_done.wait(timeout=0.5):
                    if self.stop:
                        break
                if self.is_smb and self.level == SMB_FINALE_LABEL:
                    self.mode = "done"
                    self.status = "THE GAME IS COMPLETE"
                    break
                nxt, key = extract_next_entrance(
                    self.profile, Path(entrance).read_bytes(), actions,
                    self.show_dir / f"entrance_after_{self.level}.state")
                if nxt is not None:
                    prev_entrance, prev_level = entrance, self.level
                    entrance = nxt
                    self.level = self.game.label(key)
                    attempt = 0
                    bank(attempt)
                    continue
            sargs, arm_name = solver_args(
                self.profile_path, entrance, out,
                self.args.minutes_per_level, self.args.workers,
                attempt=attempt)
            self.status = (f"searching {self.level} "
                           f"[attempt {attempt + 1}: {arm_name}]")
            s = Solver(sargs)
            root_bytes = Path(entrance).read_bytes()
            cv = int(getattr(self.args, "chorus_voices", -1))
            n_voices = self.args.workers if cv < 0 else min(cv, self.args.workers)
            if chorus:
                # Audio production WITHOUT pacing (decoupled in the pool:
                # set_worker_audio overrides the pace<->audio welding).
                # Attribution bench 2026-07-28: full-choir synthesis costs
                # ~14% throughput (3,050 -> 2,615 sps) — the 8x collapse
                # once blamed on the chorus was the hero cam's in-emulator
                # pacing sleep holding the GIL (fixed: Python-side pacing).
                # Default = every worker sings.
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
                # MUST be O(frontier), never O(archive): the original
                # full-trace-table max froze the whole show at 4-4
                # (715k cells -> seconds of GIL-held scanning every 2s;
                # the solver stepped 264 times in a minute while the
                # tiles sat frozen). The solver's selection cache
                # already maintains the near-frontier cells — read
                # those (~dozens), not the world.
                now = time.time()
                if now - cache["t"] < 2.0:
                    return cache["val"]
                cache["t"] = now
                try:
                    band = (getattr(sv, "_sel_band24", None)
                            or getattr(sv, "_sel_deep", None))
                    if not band:
                        return cache["val"]
                    cell = max(band, key=lambda c: c.best_score)
                    rec = sv.traces.get(cell.key)
                    if rec is None:
                        return cache["val"]
                    root_id, tb = rec[0], rec[1]
                    rb = Path(sv.roots[root_id]["path"]).read_bytes()
                    cache["val"] = (rb, np.frombuffer(tb, dtype=np.uint8))
                except Exception:
                    pass
                return cache["val"]

            self.hero.set_level(root_bytes, get_best)

            inst = {"t": time.time(), "steps": 0, "sps": 0}

            def hook(rs, sv, _self=self, chorus=chorus, inst=inst):
                if _self.stop:
                    sv.stop = True
                _self.frames = [r[0] for r in rs]
                if chorus:
                    pump_chorus(sv)
                # INSTANTANEOUS sps (2s window) — the cumulative figure
                # decays slowly and masked full stalls on stream (the
                # audit found 640s windows at ~0 sps shown as "300+").
                now = time.time()
                if now - inst["t"] >= 2.0:
                    inst["sps"] = int((sv.steps_done - inst["steps"])
                                      / (now - inst["t"]))
                    inst["t"], inst["steps"] = now, sv.steps_done
                _self.status = (
                    f"searching {_self.level} "
                    f"[attempt {attempt + 1}: {arm_name}] — "
                    f"{sv.steps_done/1e6:.1f}M steps @ {inst['sps']}/s, "
                    f"frontier gx {sv.max_gx_in_area.get(sv.max_area, 0)}, "
                    f"{len(sv.archive)} cells")

            s.step_hook = hook
            s.pool.set_headless(False)               # render worker frames
            s.seed()
            s.explore()
            sols = sorted(out.glob("solutions/sol_*.actions.npy"))
            if not sols:
                attempt += 1
                bank(attempt)
                nxt_arm = ESCALATION[min(attempt, len(ESCALATION) - 1)]["name"]
                self.status = (f"{self.level}: budget spent — escalating "
                               f"to attempt {attempt + 1} ({nxt_arm})")
                continue
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
            prev_entrance, prev_level = entrance, self.level
            entrance = nxt
            self.level = self.game.label(key)
            attempt = 0
            bank(attempt)
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
        elif ev.key() == Qt.Key.Key_M:
            # Mute toggle — master volume only; search never pauses.
            st = self.show_state
            if st.mixer is not None:
                self._muted = not getattr(self, "_muted", False)
                try:
                    st.mixer.set_volume(
                        0.0 if self._muted else st.args.volume)
                except Exception:
                    pass
                self.caption.setText(
                    "[audio muted — M to unmute]" if self._muted
                    else "[audio on]")

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
    ap.add_argument("--chorus-voices", type=int, default=-1,
                    help="How many workers sing: -1 = all (default; "
                         "~14%% throughput cost), N = first N, 0 = none")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = LiveSolveWindow(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
