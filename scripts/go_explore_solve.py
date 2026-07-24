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


def cell_fn(ram) -> tuple:
    # (area, step-phase, y-band, gx bucket). Phase = (frame>>2)&7 is
    # step-granular at frame_skip 4 (all 8 classes reachable via step-count
    # variance). gx bucket is LAST so the archive's horizontal_neighbors
    # frontier bonus applies unmodified.
    return (int(ram[R_AREA]), (int(ram[R_PHASE]) >> 2) & 7,
            int(ram[R_YPOS]) // 32, _gx(ram) // 16)


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
                root_id: str) -> str:
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
        if gx > 3900:
            return "live"   # transition-frame garbage read; skip recording
        area = int(ram[R_AREA])
        self.max_gx_in_area[area] = max(self.max_gx_in_area.get(area, 0), gx)
        if area > self.max_area:
            self.max_area = area
        # Domination score = gx within the cell; deeper-x wins, ties to fewer
        # steps (elites keep shortening). Peek-then-record: only pay
        # save_worker_state for a new/dominating cell.
        key = cell_fn(ram)
        cur = self.archive.cells.get(key)
        dom = (cur is None or gx > cur.best_score + 1e-9
               or (abs(gx - cur.best_score) <= 1e-9 and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, gx, steps):
                self.traces[key] = (root_id, bytes(trace))
        else:
            self.archive.record(ram, None, gx, steps)
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

    def select(self):
        cells = [c for c in self.archive.cells.values() if c.state is not None]
        if not cells:
            return None
        # Deep-frontier arm: bias toward cells in the DEEPEST area reached,
        # near its max gx — this follows Mario through area transitions.
        if self.rng.random() < self.args.deep_bias:
            deep = [c for c in cells if c.key[0] == self.max_area]
            if deep:
                topgx = max(c.key[3] for c in deep)
                floor = topgx - int(self.rng.integers(0, 24))
                band = [c for c in deep if c.key[3] >= floor]
                cell = band[int(self.rng.integers(len(band)))]
                cell.times_chosen += 1
                cell.explored = True
                return cell
        for _ in range(8):
            cell = self.archive.select_return_cell()
            if cell is not None and cell.state is not None:
                return cell
        return cells[int(self.rng.integers(len(cells)))]

    def _assign(self, wid: int) -> dict:
        cell = self.select()
        if cell is None:
            # Fall back to the entrance root.
            self.pool.load_worker_state(wid, Path(self.args.root_state).read_bytes())
            return {"key": None, "root": "entrance", "trace": [], "steps": 0,
                    "left": self.args.burst,
                    "prev": int(self.rng.choice(len(self.weights), p=self.weights))}
        self.pool.load_worker_state(wid, cell.state)
        root_id, tb = self.traces[cell.key]
        return {"key": cell.key, "root": root_id, "trace": list(tb),
                "steps": cell.best_steps, "left": self.args.burst,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights))}

    def explore(self) -> None:
        args = self.args
        ctx = [self._assign(i) for i in range(args.workers)]
        acts = np.zeros(args.workers, dtype=np.uint8)
        self.t0 = last_progress = last_flush = time.time()
        deadline = self.t0 + args.minutes * 60 if args.minutes > 0 else None
        while not self.stop:
            for i, c in enumerate(ctx):
                a = c["prev"] if self.rng.random() < args.sticky else \
                    int(self.rng.choice(len(self.weights), p=self.weights))
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
                status = self.observe(i, ram, c["trace"], c["steps"], c["root"])
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
    args = ap.parse_args()

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
