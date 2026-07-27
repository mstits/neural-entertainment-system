"""Honest Go-Explore level solver for SMB (research-grounded pipeline, Phase 1).

Runs first-return-then-explore (Ecoffet et al. 2021) PURELY from a level's
ENTRANCE — no hand-crafted prefixes, no handoff seeds, no policy net. Search
(not gradient descent) does the hard exploration that model-free PPO cannot:
the deterministic Rust pool + microsecond save/restore lets the archive return
to any frontier cell for free, so a long correct path is discovered
incrementally instead of re-explored from scratch every episode.

This is the SOLVER half of the Go-Explore -> distillation pipeline (Blueprint
§2C). Solved trajectories are dumped to <out>/solutions/ as action traces +
provenance; the distillation step (BC + DAgger + 25% sticky) teaches a policy
to reproduce them robustly, and the honest metric is cold sticky+jitter from
the entrance. The search is a TEACHER; the deployed agent is the learned net.

Differences from the 2-1 campaign harvester (scripts/go_explore_2_1.py), which
crossed into hand-authored territory:
  * root ONLY at the level entrance (--root-state); NO prefix/handoff lineages.
  * lives-based death detection ($075A) — robust across enemy/pit/time deaths
    and multi-area levels, vs the fragile gx-drop heuristic.
  * (area-order, gx) progress so multi-area levels (1-2's entrance->underground)
    keep pushing the true frontier, not just the entrance area.
  * WARP GUARD: a level "clear" counts only when the world/level advance in the
    natural forward sequence (world unchanged, displayed level +1, OR a castle
    world-advance) — a warp-zone pipe (world jumps) is NOT a legit clear.

Usage:
  python scripts/go_explore_solve.py --out runs/ge_1_2 \
      --root-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
      --profile configs/mario_1_2_solo.yaml --workers 8 --minutes 20
"""
from __future__ import annotations

import argparse
import json
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.go_explore import GoExploreArchive, keep_exploring  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
DEATH_STATES = (6, 11)  # RAM $000E player-state: dying / death-pit
NOOP = 0

# RAM addresses (SMB).
R_X_PAGE, R_X_LOW = 0x006D, 0x0086
R_PSTATE = 0x000E
R_YPOS = 0x00CE
R_PHASE = 0x0009
R_AREA = 0x0760
R_WORLD = 0x075F
R_LEVEL = 0x075C
R_LIVES = 0x075A


def _gx(ram) -> int:
    return (int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW])


def _wd(ram) -> tuple:
    """Displayed (world, level), both 0-indexed."""
    return (int(ram[R_WORLD]), int(ram[R_LEVEL]))


GX_BUCKET = 16   # overridable via --gx-bucket (micro-search: 8)
Y_BAND = 32      # overridable via --y-band (micro-search: 16)


def cell_fn(ram) -> tuple:
    # (area, step-phase, y-band, gx bucket). Phase = (frame>>2)&7 is
    # step-granular at frame_skip 4 (all 8 classes reachable via step-count
    # variance). gx bucket is LAST so the archive's horizontal_neighbors
    # frontier bonus applies unmodified.
    # vx sign disambiguates travel direction: doubling back through
    # previously-visited coordinates is a DISTINCT cell, so the archive
    # explores backtracking maneuvers instead of pruning them as loops
    # (heuristic-inversion recipe, maze consultation 2026-07-24).
    vx = int(np.int8(ram[0x57]))
    vsign = 0 if vx == 0 else (1 if vx > 0 else 2)
    return (int(ram[R_AREA]), (int(ram[R_PHASE]) >> 2) & 7, vsign,
            int(ram[R_YPOS]) // Y_BAND, _gx(ram) // GX_BUCKET)


def is_forward_clear(start_wd: tuple, ram) -> bool:
    """True iff (world, level) advanced in the NATURAL forward sequence.

    Legit 1-2 -> 1-3: world unchanged, level +1. Legit castle world advance
    (e.g. 1-4 -> 2-1): world +1 (level resets to 0). A WARP-ZONE pipe jumps
    the world by >1 or advances the world without having been on a castle —
    guarded out so warps never count as a clear (honesty requirement)."""
    sw, sl = start_wd
    w, l = _wd(ram)
    if w == sw and l == sl + 1:
        return True   # same-world level advance (the common case, incl. 1-2->1-3)
    if w == sw + 1 and l == 0 and sl >= 2:
        return True   # castle clear -> next world (only from an x-4, level idx>=2... )
    return False


def inverted_weights(action_space) -> list:
    """Exploration bias for the saturation window: reward the maneuvers the
    forward heuristic structurally prunes (leftward, downward)."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "left" in b:
            w += 3.0
        if "down" in b:
            w += 3.0
        if "A" in b:
            w += 1.0
        if "right" in b:
            w -= 0.5
        ws.append(max(w, 0.2))
    return ws


def action_weights(action_space) -> list:
    """Right-biased sampling weights over the profile's action space."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "right" in b:
            w += 2.0
        if "A" in b:
            w += 2.0  # jumps carry gaps/pipes
        if "B" in b:
            w += 1.0
        ws.append(w)
    return ws


class Solver:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        (self.out / "solutions").mkdir(parents=True, exist_ok=True)
        profile = yaml.safe_load(Path(args.profile).read_text())
        self.bitmasks = action_space_to_bitmasks(profile["action_space"])
        self.weights = np.array(action_weights(profile["action_space"]))
        self.weights /= self.weights.sum()
        self.inv_weights = np.array(inverted_weights(profile["action_space"]))
        self.inv_weights /= self.inv_weights.sum()
        self.pool = Pool(rom_path=ROM, num_workers=args.workers,
                         frame_skip=int(profile.get("frame_skip", 4)))
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self.rng = np.random.default_rng(args.seed)
        self.archive = GoExploreArchive(cell_fn, seed=args.seed)
        self.traces: dict = {}            # cell key -> (root_id, trace bytes)
        self.roots: dict = {}             # root_id -> {path, start_wd, lives}
        self.start_wd = (0, 0)
        self.start_lives = 0
        # Progress = (area order, gx). We track the DEEPEST area reached and
        # the max gx within it, so the deep-frontier bias follows Mario into
        # the underground instead of pinning on the entrance area.
        self.max_area = 0
        self.max_gx_in_area: dict = {}    # area -> max gx seen
        self._pin_time = time.time()      # last frontier advance (inversion gate)
        self._loop_dest_min = None        # min gx observed right after a loop
        self.max_sect = 0                 # deepest section-transit count seen
        self.n_solutions = 0
        self.sol_counter = 0
        self.best_sol_len = 10 ** 9
        self.steps_done = 0
        self.t0 = time.time()
        self.stop = False

    def _step0(self, a: int):
        acts = np.zeros(self.args.workers, dtype=np.uint8)
        acts[0] = self.bitmasks[a]
        return self.pool.step_all(acts)[0][2]

    # ---- record path -------------------------------------------------

    def observe(self, wid: int, ram, trace: list, steps: int,
                root_id: str, loops: int = 0,
                route_sig: tuple = (), sect: int = 0,
                psig: tuple = ()) -> str:
        """Record one reached state. Returns 'dead' | 'clear' | 'live'."""
        # Clear (warp-guarded) is checked FIRST.
        if is_forward_clear(self.start_wd, ram):
            self._dump_solution(root_id, trace, ram, steps)
            return "clear"
        # A non-forward world/level change with fewer lives, or an explicit
        # dying state, or any life lost = death. Lives-based detection is
        # robust across enemy/pit/time deaths and multi-area levels.
        if int(ram[R_LIVES]) < self.start_lives:
            return "dead"
        if int(ram[R_PSTATE]) in DEATH_STATES:
            return "dead"
        # A world/level change that ISN'T a forward clear = a warp (or a
        # backward reload): do not record it (would poison the archive with
        # off-level cells).
        if _wd(ram) != self.start_wd and not is_forward_clear(self.start_wd, ram):
            return "dead"
        gx = _gx(ram)
        if gx > 7000:
            # Transition-frame garbage read (page byte mid-load reads huge);
            # real SMB levels reach ~6,300 px (8-1) — the old 3900 cap silently
            # froze the 8-1 frontier at 3900 (states past it never archived).
            return "live"
        area = int(ram[R_AREA])
        if gx > self.max_gx_in_area.get(area, 0):
            self.max_gx_in_area[area] = gx
            self._pin_time = time.time()  # frontier moved: inversion stays off
        if area > self.max_area:
            self.max_area = area
        # Domination score = gx within the cell; deeper-x wins, ties to fewer
        # steps (elites keep shortening). Peek-then-record: only pay
        # save_worker_state for a new/dominating cell.
        # MAZE FIX (2026-07-24): the trajectory's loop-count is the LEADING
        # key component. Castle mazes (4-4/7-4/8-4) loop wrong paths back —
        # observable as a discontinuous gx->0 collapse in our own rollouts —
        # so gx alone aliases first-pass and looped-back states and the
        # frontier saturates (4-4: pinned at gx 2064, 0 solutions, 1.5M
        # records). With the counter, the same coordinates on different
        # maze phases are DIFFERENT cells and search explores each pass.
        # Search-derived (a property of the agent's own path), no internals.
        #
        # SECOND MAZE FIX (same day, v4): the loop-count alone still aliased
        # right-route and wrong-route states at the same coordinates — the
        # game tracks the current pass's route in internal state, and the
        # wall-2102 diagnostic showed the deepest lineages were wrong-route
        # spirals from which EVERY action loops back. (A raw RAM-hash key
        # separated routes but exploded the archive to 585k one-visit cells
        # — timers/enemies churn every frame.) The ROUTE SIGNATURE is the
        # compact middle path: the trajectory's own y-band at each 512-px
        # gx-boundary crossing since the last loop event — the observable
        # footprint of the route the current pass has taken. Derived purely
        # from our own rollout; bounded cardinality (<=4 entries, 4 bands).
        # v5: key on the DISCOVERED route-tracker bytes. Differential analysis
        # of our own rollouts (32 passes, /tmp/ram_diff2.py) found $0742 and
        # $07F8 change rarely, at single consistent fork positions — the
        # empirical signature of the game's route-tracking state. Keying on
        # their VALUES separates right-route from wrong-route states with
        # bounded cardinality (vs the raw-hash explosion / sig blindness).
        # Archive-eligibility gate (8-4 lesson): lineages beyond loop-phase 6
        # are wrong-route spirals — they reach deep gx but are unwinnable, and
        # archiving them bloated the archive to 3.6M cells (sps 3300->1700)
        # while poisoning deep-cell selection. They may still EXPLORE (their
        # forks feed discovery) but do not enter the selection pool.
        if loops > 6:
            return "live"
        if sect > self.max_sect:
            self.max_sect = sect
            self._pin_time = time.time()
        # v7: CONTENT-aware cells. Hypothesis shift — the seam wrap (gx->0)
        # may fire on the CORRECT route too, loading the NEXT section's
        # layout; coordinate keys then alias post-seam progress with
        # wrong-route repeats and every gx-based metric is blind to success
        # by construction. The tile buffer ($0500-$06BF, the drawn layout) is
        # the observable content signature: same coords + same section hash
        # alike, same coords + ADVANCED section hash differently. Bounded
        # cardinality (distinct layouts only), unlike the v4 full-RAM hash.
        # SECTION-aware key (8-4 fix, verified empirically 2026-07-25):
        # correct pipe transits change the section pointer $0750 while the
        # area byte $0760 stays constant — confirmed by differential
        # comparison of our own pre/post-gate archive states (2 -> 229).
        # The transit count leads the key and dominates the score, so
        # section progress is the frontier even when gx jumps backward.
        # psig = the last-4 section-pointer values = PIPE-PATH IDENTITY.
        # sect alone aliases different pipe sequences at equal transit
        # counts; 8-4's final checks discriminate by the route taken, so
        # each path is its own frontier (fix 2026-07-26).
        key = (sect, psig, loops, route_sig) + cell_fn(ram)
        score = sect * 10000 + gx
        cur = self.archive.cells.get(key)
        dom = (cur is None or score > cur.best_score + 1e-9
               or (abs(score - cur.best_score) <= 1e-9 and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, score, steps,
                                                        key=key):
                self.traces[key] = (root_id, bytes(trace), loops, route_sig, sect, psig)
        else:
            self.archive.record(ram, None, score, steps, key=key)
        return "live"

    def _dump_solution(self, root_id: str, trace: list, ram, steps) -> None:
        if len(trace) >= self.best_sol_len - 8:
            return  # keep only materially-shorter re-clears
        self.best_sol_len = len(trace)
        n = self.sol_counter
        self.sol_counter += 1
        self.n_solutions += 1
        base = self.out / "solutions" / f"sol_{n:03d}"
        np.save(str(base) + ".actions.npy", np.array(trace, dtype=np.int64))
        (base.parent / (base.name + ".json")).write_text(json.dumps({
            "provenance": "search",
            "root_id": root_id,
            "root_state": self.roots[root_id]["path"],
            "start_wd": list(self.start_wd),
            "clear_wd": list(_wd(ram)),
            "steps": steps, "actions": len(trace),
        }, indent=2) + "\n")
        print(f"[go_explore_solve] *** SOLUTION {n} *** root={root_id} "
              f"{len(trace)} actions, {self.start_wd}->{_wd(ram)}", flush=True)

    # ---- seeding: root ONLY (honest) ---------------------------------

    def seed(self) -> None:
        path = self.args.root_state
        self.pool.load_worker_state(0, Path(path).read_bytes())
        r = self._step0(NOOP)  # convention: load root, one NOOP, then actions
        self.start_wd = _wd(r)
        self.start_lives = int(r[R_LIVES])
        self.max_area = int(r[R_AREA])
        self.roots["entrance"] = {"path": str(path),
                                  "start_wd": list(self.start_wd),
                                  "lives": self.start_lives}
        self.observe(0, r, [], 0, "entrance")
        print(f"[seed] rooted at {path} wd={self.start_wd} lives="
              f"{self.start_lives} area={self.max_area}; archive="
              f"{json.dumps(self.archive.stats())}", flush=True)

    # ---- frontier selection ------------------------------------------

    def _refresh_sel_cache(self) -> None:
        """Rebuild the selection caches. A full archive scan ran on EVERY
        worker reassignment (every episode end) — O(330k) three times over
        (state filter, deep filter, weighted fallback) — starving the Rust
        pool at 1.6/16 cores with sps decayed 2378->475. Rebuild only when
        the archive grows 2% or the deepest area advances."""
        self._sel_cells = [c for c in self.archive.cells.values()
                           if c.state is not None]
        self._sel_n = len(self.archive.cells)
        self._sel_area = self.max_area
        deep = [c for c in self._sel_cells
                if c.key[0] == self.max_sect and c.key[-5] == self.max_area]
        self._sel_deep = deep
        if deep:
            minl = min(c.key[0] for c in deep)
            lowl = [c for c in deep if c.key[0] <= minl + 1]
            self._sel_topgx = max(c.key[-1] for c in deep)
            # Near-frontier bands precomputed (the per-call band filter over
            # `deep` was itself O(N) in one-area castle levels).
            _f24 = self._sel_topgx - 24
            self._sel_band24 = [c for c in deep if c.key[-1] >= _f24]
            self._sel_lowl_band24 = [c for c in lowl if c.key[-1] >= _f24]
        else:
            self._sel_topgx = 0
            self._sel_band24 = []
            self._sel_lowl_band24 = []

    def select(self):
        if (getattr(self, "_sel_cells", None) is None
                or len(self.archive.cells) > self._sel_n * 1.02
                or self._sel_area != self.max_area):
            self._refresh_sel_cache()
        cells = self._sel_cells
        if not cells:
            return None
        # Deep-frontier arm: bias toward cells in the DEEPEST area reached,
        # near its max gx — this follows Mario through area transitions.
        if self.rng.random() < self.args.deep_bias:
            pool_band = (self._sel_lowl_band24
                         if (self.rng.random() < 0.7 and self._sel_lowl_band24)
                         else self._sel_band24)
            if pool_band:
                floor = self._sel_topgx - int(self.rng.integers(0, 24))
                band = [c for c in pool_band if c.key[-1] >= floor]
                if band:
                    cell = band[int(self.rng.integers(len(band)))]
                    cell.times_chosen += 1
                    cell.explored = True
                    return cell
        # Uniform over the cached list (the archive's weighted selection was
        # itself an O(N) scan; the deep arm supplies the directed pressure).
        return cells[int(self.rng.integers(len(cells)))]

    def _assign(self, wid: int) -> dict:
        cell = self.select()
        if cell is None:
            # Fall back to the entrance root.
            self.pool.load_worker_state(wid, Path(self.args.root_state).read_bytes())
            return {"key": None, "root": "entrance", "trace": [], "steps": 0,
                    "left": self.args.burst, "loops": 0, "prev_gx": -1,
                    "sig": (), "sect": 0, "p0750": None, "psig": (),
                    "prev": int(self.rng.choice(len(self.weights), p=self.weights))}
        self.pool.load_worker_state(wid, cell.state)
        rec = self.traces[cell.key]
        root_id, tb, loops, sig = rec[0], rec[1], rec[2], rec[3]
        sect = rec[4] if len(rec) > 4 else 0
        psig = rec[5] if len(rec) > 5 else ()
        # prev_gx -1: no loop detection on the restore step (the load frame
        # reads transitional garbage; the first real step re-arms it).
        return {"key": cell.key, "root": root_id, "trace": list(tb),
                "steps": cell.best_steps, "left": self.args.burst,
                "loops": loops, "prev_gx": -1, "sig": sig,
                "sect": sect, "p0750": None, "psig": psig,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights))}

    def explore(self) -> None:
        args = self.args
        ctx = [self._assign(i) for i in range(args.workers)]
        acts = np.zeros(args.workers, dtype=np.uint8)
        self.t0 = last_progress = last_flush = time.time()
        deadline = self.t0 + args.minutes * 60 if args.minutes > 0 else None
        while not self.stop:
            for i, c in enumerate(ctx):
                # Heuristic inversion inside the self-measured saturation
                # window [pin-300, pin+60]: the frontier pin is where OUR
                # search saturates (live telemetry, not any external map);
                # inside it, sample from inverted weights so leftward /
                # downward entries get explored instead of pruned.
                # SATURATION-TRIGGERED inversion (fix 2026-07-25): always-on
                # inversion sabotaged standard levels — at 5-3's frontier the
                # solver sampled left/down exactly where a full-speed rightward
                # jump was needed (chain stall). The maze maneuver hunt now
                # arms only after the frontier has been pinned >=180 s.
                _pin = self.max_gx_in_area.get(self.max_area, 0)
                _floor = (self._loop_dest_min
                          if self._loop_dest_min is not None
                          else _pin - 300)
                _w = (self.inv_weights
                      if (_pin > 400 and c.get("gx", -1) >= 0
                          and _floor <= c["gx"] <= _pin + 60
                          and time.time() - self._pin_time >= 180.0)
                      else self.weights)
                a = c["prev"] if self.rng.random() < args.sticky else \
                    int(self.rng.choice(len(_w), p=_w))
                c["prev"] = a
                c["pending"] = a
                acts[i] = self.bitmasks[a]
            results = self.pool.step_all(acts)
            self.steps_done += args.workers
            for i, c in enumerate(ctx):
                ram = results[i][2]
                c["trace"].append(c["pending"])
                c["steps"] += 1
                c["left"] -= 1
                # Loop-back detection: a discontinuous backward gx jump on a
                # non-garbage frame advances the trajectory's maze phase
                # (capped so wrong-path spirals can't explode the archive).
                # Route signature: the y-band recorded at each 512-px gx
                # boundary the pass crosses (reset on loop-back) — the
                # observable footprint of THIS pass's route choice.
                _lgx = _gx(ram)
                c["gx"] = _lgx if _lgx <= 7000 else c.get("gx", -1)
                # ROOM IDENTITY (verified 2026-07-26: $074E changes 0.67/1k
                # steps, only at full-screen transitions — vs $0750's 6/1k
                # streaming churn). rid = (area type, swim flag, Bowser slot
                # present); a rid CHANGE is a true room transition.
                _bw = 1 if 0x2D in bytes(ram[0x14:0x1C]) else 0
                _rid = (int(ram[0x74E]), int(ram[0x1D]), _bw)
                _transit = (c["p0750"] is not None and _rid != c["p0750"])
                if _transit and c["sect"] < 16:
                    c["sect"] += 1
                    c["psig"] = _rid
                c["p0750"] = _rid
                if _lgx <= 7000:
                    if c["prev_gx"] >= 0:
                        if _transit:
                            c["prev_gx"] = _lgx   # section change: re-arm, no loop
                        elif _lgx < c["prev_gx"] - 100:
                            if c["loops"] < 8:
                                c["loops"] += 1
                            c["sig"] = ()   # new pass, fresh route
                        elif _lgx // 512 != c["prev_gx"] // 512:
                            c["sig"] = (c["sig"] + (int(ram[R_YPOS]) // 64,))[-4:]
                    c["prev_gx"] = _lgx
                if (self.args.swim_gx_ceiling > 0 and int(ram[0x1D]) == 1
                        and _lgx <= 7000
                        and _lgx > self.args.swim_gx_ceiling):
                    ctx[i] = self._assign(i)
                    continue
                status = self.observe(i, ram, c["trace"], c["steps"],
                                      c["root"], c["loops"], c["sig"],
                                      c["sect"], c["psig"])
                # Finisher extension: the level-END transition (exit pipe /
                # flag slide) can run many steps with gx frozen, so a burst
                # from the deepest cell can end just short of the wd advance.
                # A burst ending in the deepest-area top band gets one +200
                # extension so it can actually complete the clear.
                if (status == "live" and c["left"] <= 0
                        and not c.get("extended")
                        and int(ram[R_AREA]) == self.max_area
                        and _gx(ram) // 16 >= self.max_gx_in_area.get(self.max_area, 0) // 16 - 3):
                    c["left"] += 200
                    c["extended"] = True
                if status != "live" or c["left"] <= 0 or c["steps"] >= args.max_steps:
                    ctx[i] = self._assign(i)
            now = time.time()
            if now - last_progress >= 60:
                last_progress = now
                self.progress_line(now - self.t0)
            if now - last_flush >= args.flush_secs:
                last_flush = now
                self.flush()
            if deadline and now >= deadline:
                break
            if not keep_exploring(self.n_solutions, args.want_solutions):
                break

    # ---- reporting / persistence -------------------------------------

    def progress_line(self, elapsed: float) -> None:
        line = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed),
            "cells": len(self.archive),
            "max_area": self.max_area,
            "max_gx_in_max_area": self.max_gx_in_area.get(self.max_area, 0),
            "max_sect": self.max_sect,
            "solutions": self.n_solutions,
            "best_sol_actions": (self.best_sol_len if self.n_solutions else None),
            "steps": self.steps_done,
            "sps": round(self.steps_done / max(elapsed, 1e-9)),
        }
        with open(self.out / "progress.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[go_explore_solve] {json.dumps(line)}", flush=True)

    def flush(self) -> None:
        self.archive.save(self.out / "archive.pkl")
        with open(self.out / "traces.pkl", "wb") as f:
            pickle.dump(self.traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        (self.out / "roots.json").write_text(json.dumps(self.roots, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-state", required=True,
                    help="Level ENTRANCE save-state (honest root; no prefix).")
    ap.add_argument("--profile", required=True,
                    help="Source of action_space + frame_skip only.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--burst", type=int, default=64)
    ap.add_argument("--sticky", type=float, default=0.5)
    ap.add_argument("--deep-bias", type=float, default=0.4)
    ap.add_argument("--minutes", type=float, default=0,
                    help="Wall-clock budget (0 = until solution quota).")
    ap.add_argument("--want-solutions", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--flush-secs", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swim-gx-ceiling", type=int, default=0,
                    help="If >0: in swim rooms ($001D=1), lineages crossing "
                         "this gx are terminated — forces attempt density "
                         "onto the section's floor pipes instead of the "
                         "scroll-buffer wrap (8-4 water separating expt).")
    ap.add_argument("--gx-bucket", type=int, default=16,
                    help="Cell gx granularity px (micro-search: 8).")
    ap.add_argument("--y-band", type=int, default=32,
                    help="Cell y granularity px (micro-search: 16).")
    args = ap.parse_args()
    global GX_BUCKET, Y_BAND
    GX_BUCKET, Y_BAND = args.gx_bucket, args.y_band

    s = Solver(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(s, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(s, "stop", True))
    s.seed()
    s.flush()
    s.explore()
    s.flush()
    s.progress_line(time.time() - s.t0)
    print(f"[go_explore_solve] done: {json.dumps(s.archive.stats())}, "
          f"{s.n_solutions} solutions -> {s.out}", flush=True)
    s.pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
