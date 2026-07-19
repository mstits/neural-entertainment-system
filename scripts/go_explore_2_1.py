"""Frontier-cell Go-Explore harvester for SMB 2-1.

First-return-then-explore (Ecoffet et al. 2021) over the deterministic
Rust pool: discretize RAM into cells, archive the best-known way to
reach each cell (save-state blob + action trace from a named root),
return to frontier cells by restoring their blob (1us), and explore
with short right-biased sticky random bursts. Progress is CUMULATIVE —
a cell reached once is reachable forever — which deletes the
restart-multiplicative prefix cost that starved PPO of attempts at the
gx1658 enemy-pack compound.

Cell key = (area $0760, phase, y band $00CE//32, gx//16), phase =
(RAM[$0009] >> 2) & 7 — STEP-granular: at frame_skip 4 the frame
counter advances exactly 4 per step, so (>>2)&7 moves 1 per step and
all 8 phase classes are reachable via step-count variance (the raw &7
key collapses to 2 residues per lineage and is degenerate). The gx
bucket is the LAST key component so the archive's default
horizontal_neighbors adjacency (W_location frontier bonus) applies
unmodified.

Every cell carries its full action trace from ITS OWN root (per-cell
root id; the ep231 lineage roots at seed_2_1_gx1421_runway.state, NOT
the handoff — traces only replay from the root they were recorded
from). Trace replay convention shared with replay_to_demos.py: load
root, one NOOP step, then the actions.

Seeding: (1) replay the banked ep231 prefix (RAM-probe verified to
cross the compound to ~gx1966); (2) no-op-offset runs at pre-compound
prefix cells — at frame_skip 4 each no-op STEP shifts the phase class
by exactly 1, so 7 no-op steps mint all 8 phase classes per position;
(3) a handoff_2-1.state lineage grown by scripted right-biased bursts
(no policy net — the harvester stays CPU-light, torch-free).

Solutions (displayed world/level tuple advances past the lineage
start) are dumped to <out>/solutions/ as actions + provenance:search
sidecars and the loop keeps running for more phase classes. Search is
a TEACHER — demos feed the DQfD anchor; the deployed agent remains the
learned net.

Single process, one Pool(num_workers=N) stepped in lockstep: the
archive needs no locks and rayon parallelizes emulation across
workers with the GIL released — the simplest correct sharing design
for a campaign tool. One JSON progress line per minute lands in
<out>/progress.jsonl for the main session to poll; SIGTERM/SIGINT
flush the archive and exit cleanly. Re-running with the same --out
resumes from the flushed archive.

Usage (smoke):
  python scripts/go_explore_2_1.py --out /tmp/ge_smoke \
      --workers 4 --minutes 1.5
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
PRECOMPOUND_GX = 1650   # phase-seed only before the enemy-pack compound
NOOP = 0


def _gx(ram) -> int:
    return (int(ram[0x006D]) << 8) | int(ram[0x0086])


def _wd(ram) -> tuple:
    # Displayed (world, level) tuple — advances past the lineage start
    # exactly when the level is cleared (beam_solver_v1 predicate).
    return (int(ram[0x075F]), int(ram[0x075C]))


def cell_fn(ram) -> tuple:
    return (int(ram[0x0760]), (int(ram[0x0009]) >> 2) & 7,
            int(ram[0x00CE]) // 32, _gx(ram) // 16)


def action_weights(action_space) -> list:
    """Right-biased sampling weights over the profile's action space."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "right" in b:
            w += 2.0
        if "A" in b:
            w += 2.0  # jumps carry the compound
        if "B" in b:
            w += 1.0
        ws.append(w)
    return ws


class Harvester:
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
        # cell key -> (root_id, action trace as bytes). Updated ONLY when
        # archive.record() accepts (new cell or domination improvement).
        self.traces: dict = {}
        self.roots: dict = {}  # root_id -> {"path", "start_wd"}
        self.main_area = 0    # area byte at the first root (set in seed)
        self.sol_seen: dict = {}  # (root_id, phase) -> best trace len
        self.sol_counter = 0  # solution file numbering (incl. improvements)
        self.n_solutions = 0  # DISTINCT (root_id, phase) combos
        self.max_gx = 0
        self.steps_done = 0
        self.t0 = time.time()
        self.stop = False

    # ---- shared record path -----------------------------------------

    def observe(self, wid: int, ram, trace: list, steps: int,
                root_id: str, prev_gx: int) -> str:
        """Record one reached state. Returns 'dead' | 'clear' | 'live'.

        Death detection is TWO-pronged. Enemy contact sets player-state
        $000E to 6/11 — the repo-standard predicate. Pit falls and
        time-ups DON'T (verified by RAM probe: ps stays 8 below the
        screen, then the engine reloads the level — gx collapses, area
        and displayed world/level unchanged — and respawns at the mid-
        level checkpoint on a NEW LIFE). The reload signature (big gx
        drop, same area, same wd) is treated as death so no multi-life
        state or trace ever enters the archive: every cell stays
        single-life-honest, which is what makes the emitted demos
        replayable receipts. Legitimate transitions are exempt: a level
        clear advances wd (checked first) and pipes/vines change the
        area byte."""
        if _wd(ram) > tuple(self.roots[root_id]["start_wd"]):
            self._dump_solution(root_id, trace, ram, steps)
            return "clear"
        if int(ram[0x000E]) in DEATH_STATES:
            return "dead"
        gx = _gx(ram)
        if prev_gx - gx > 400 and int(ram[0x0760]) == self.main_area:
            return "dead"  # in-level reload = pit/time death, new life
        if int(ram[0x0760]) == self.main_area:
            self.max_gx = max(self.max_gx, gx)
        # Peek-then-record (same pattern as the trainer's go_explore
        # hook): only pay save_worker_state for a new cell or a strict
        # domination improvement; otherwise record(None) for visit
        # bookkeeping. Score = gx, so within a cell deeper-x wins and
        # ties go to fewer steps — elites keep shortening post-clear.
        key = cell_fn(ram)
        cur = self.archive.cells.get(key)
        dom = (cur is None or gx > cur.best_score + 1e-9
               or (abs(gx - cur.best_score) <= 1e-9
                   and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, gx, steps):
                self.traces[key] = (root_id, bytes(trace))
        else:
            self.archive.record(ram, None, gx, steps)
        return "live"

    def _dump_solution(self, root_id: str, trace: list, ram, steps) -> None:
        # Dedupe per (root, phase-at-clear): once past the flag the
        # engine ignores input, so every burst from a castle-walk cell
        # re-clears with a near-identical trace. Keep one solution per
        # combo, replacing it only on a materially shorter trace; the
        # distinct-combo count is what keep_exploring gates on.
        phase = (int(ram[0x0009]) >> 2) & 7
        combo = (root_id, phase)
        best = self.sol_seen.get(combo)
        if best is not None and len(trace) >= best - 15:
            return
        new_combo = best is None
        self.sol_seen[combo] = len(trace)
        n = self.sol_counter
        self.sol_counter += 1
        if new_combo:
            self.n_solutions += 1
        base = self.out / "solutions" / f"sol_{n:03d}_ph{phase}"
        np.save(str(base) + ".actions.npy", np.array(trace, dtype=np.int64))
        (base.parent / (base.name + ".json")).write_text(json.dumps({
            "provenance": "search",
            "root_id": root_id,
            "root_state": self.roots[root_id]["path"],
            "steps": steps, "actions": len(trace), "phase_class": phase,
        }, indent=2) + "\n")
        print(f"[go_explore] SOLUTION {n} (root {root_id}, {len(trace)} "
              f"actions, phase {phase})", flush=True)

    # ---- seeding (serial, worker 0) ---------------------------------

    def _add_root(self, root_id: str, path: str):
        """Load a root, take the convention NOOP step, record its cell.
        Returns the post-noop ram."""
        self.pool.load_worker_state(0, Path(path).read_bytes())
        r = self._step0(NOOP)
        if not self.roots:
            self.main_area = int(r[0x0760])
        self.roots[root_id] = {"path": str(path), "start_wd": _wd(r),
                               "area": int(r[0x0760])}
        self.observe(0, r, [], 0, root_id, _gx(r))
        return r

    def _step0(self, a: int):
        acts = np.zeros(self.args.workers, dtype=np.uint8)
        acts[0] = self.bitmasks[a]
        return self.pool.step_all(acts)[0][2]

    def seed(self) -> None:
        args = self.args
        # (1) ep231 lineage: verified compound crossing from the runway
        # seed. Record every step; keep per-step blobs for phase seeding.
        ram = self._add_root("gx1421_runway", args.root_state)
        # '' disables the prefix lineage (same convention as
        # --handoff-state ''): the archive then seeds from the bare root
        # plus no-op phase offsets and exploration bursts — the mode for
        # a fresh root with no verified action prefix (e.g. a live seam
        # arrival state).
        prefix = (
            np.load(args.prefix_actions).astype(np.int64)
            if args.prefix_actions else np.array([], dtype=np.int64)
        )
        blobs, gxs = [], []
        trace: list = []
        prev = _gx(ram)
        for a in prefix:
            trace.append(int(a))
            ram = self._step0(int(a))
            self.observe(0, ram, trace, len(trace), "gx1421_runway", prev)
            prev = _gx(ram)
            blobs.append(self.pool.save_worker_state(0))
            gxs.append(prev)
        if len(prefix):
            end_gx = gxs[-1]
            # The hard 1900-2050 band was the ep231-specific desync guard;
            # arbitrary policy prefixes (e.g. the 2-2 winner's 2083-deep
            # line) legitimately end elsewhere. Desync still shows up as a
            # near-zero end gx — warn loudly rather than die.
            if end_gx < 200:
                print(f"[seed] WARNING: prefix replay ended at gx {end_gx} "
                      f"— likely de-synced from its root", flush=True)
            print(f"[seed] prefix: {len(prefix)} steps -> gx {end_gx}",
                  flush=True)
        # (2) no-op phase offsets: one 7-noop run per pre-compound
        # checkpoint covers offsets 1..7 = all 8 phase classes there.
        for i in range(0, len(prefix), args.phase_seed_stride):
            if gxs[i] >= PRECOMPOUND_GX:
                break
            self.pool.load_worker_state(0, blobs[i])
            t = trace[:i + 1]
            prev = gxs[i]
            for _ in range(7):
                t = t + [NOOP]
                ram = self._step0(NOOP)
                if self.observe(0, ram, t, len(t), "gx1421_runway",
                                prev) != "live":
                    break
                prev = _gx(ram)
        # (3) handoff lineage: scripted right-biased bursts, no policy.
        if args.handoff_state:
            root_ram = self._add_root("handoff_2_1", args.handoff_state)
            root_key = cell_fn(root_ram)
            for _ in range(args.handoff_seed_bursts):
                cell = self.archive.cells[root_key]
                self.pool.load_worker_state(0, cell.state)
                t: list = []
                pa = int(self.rng.choice(len(self.weights), p=self.weights))
                prev = _gx(root_ram)
                for _ in range(300):
                    a = pa if self.rng.random() < args.sticky else \
                        int(self.rng.choice(len(self.weights), p=self.weights))
                    pa = a
                    t.append(a)
                    ram = self._step0(a)
                    if self.observe(0, ram, t, len(t), "handoff_2_1",
                                    prev) != "live":
                        break
                    prev = _gx(ram)
        print(f"[seed] archive after seeding: "
              f"{json.dumps(self.archive.stats())}", flush=True)

    # ---- frontier selection -----------------------------------------

    def _at_deep_band(self, ram) -> bool:
        """Is this state within ~3 gx buckets of the deepest main-area
        cell? (the region where finishing a run matters most)."""
        if int(ram[0x0760]) != self.main_area:
            return False
        return _gx(ram) // 16 >= self.max_gx // 16 - 3

    def select(self):
        # Deep bias: some picks go straight to a high-gx band of the
        # MAIN area so the frontier keeps pushing right; the rest use
        # the archive's count-based + W_location weighting (spreads
        # over low-visit cells). The band floor is randomized over the
        # top ~24 buckets (~384px) so the deep arm keeps working the
        # compound approach even after the deepest cells sit in the
        # flag/castle-walk region. Both arms mark the pick
        # chosen/explored.
        cells = [c for c in self.archive.cells.values()
                 if c.state is not None]
        main = [c for c in cells if c.key[0] == self.main_area]
        if main and self.rng.random() < self.args.deep_bias:
            floor = (max(c.key[3] for c in main)
                     - int(self.rng.integers(0, 24)))
            band = [c for c in main if c.key[3] >= floor]
            cell = band[int(self.rng.integers(len(band)))]
            cell.times_chosen += 1
            cell.explored = True
            return cell
        for _ in range(8):
            cell = self.archive.select_return_cell()
            if cell is not None and cell.state is not None:
                return cell
        return cells[int(self.rng.integers(len(cells)))]

    # ---- lockstep explore loop --------------------------------------

    def _assign(self, wid: int) -> dict:
        cell = self.select()
        self.pool.load_worker_state(wid, cell.state)
        root_id, tb = self.traces[cell.key]
        return {"key": cell.key, "root": root_id, "trace": list(tb),
                "steps": cell.best_steps, "left": self.args.burst,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights)),
                "fresh": True}

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
                prev_gx = c.get("prev_gx", c["key"][3] * 16)
                status = self.observe(i, ram, c["trace"], c["steps"],
                                      c["root"], prev_gx)
                if c.pop("fresh", False) and status == "live" \
                        and int(ram[0x0760]) == c["key"][0]:
                    # Post-restore sanity: one live same-area step from
                    # the restored blob must land within +/-2 gx buckets
                    # of the cell key (catches restore corruption; dead/
                    # clear/area-transition steps are classified above).
                    # Multi-area levels (2-2 water) can read gx 0 for a
                    # frame on scene boundaries — quarantine the cell and
                    # keep harvesting instead of killing the process.
                    if abs(_gx(ram) // 16 - c["key"][3]) > 2:
                        print(f"[quarantine] worker {i}: restore drift "
                              f"gx {_gx(ram)} vs bucket {c['key'][3]} — "
                              f"cell dropped", flush=True)
                        self.archive.cells.pop(c["key"], None)
                        ctx[i] = self._assign(i)
                        continue
                c["prev_gx"] = _gx(ram)
                if (c["left"] <= 0 and not c.get("extended")
                        and self._at_deep_band(ram)):
                    # Finisher extension: the level-end sequence (flag
                    # slide + castle walk + fade) runs ~105 steps with
                    # gx frozen, so its few distinct keys saturate ~40
                    # steps in and a 48-step burst from the deepest
                    # recordable cell can NEVER reach the wd advance.
                    # A burst ending in the deepest main-area band gets
                    # one +160-step extension — the torch-free stand-in
                    # for the proposal's policy arm that "finishes any
                    # post-barrier cell".
                    c["left"] += 160
                    c["extended"] = True
                if (status != "live" or c["left"] <= 0
                        or c["steps"] >= args.max_steps):
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

    # ---- reporting / persistence ------------------------------------

    def _phases_at_max(self) -> list:
        # Phase classes present in the deepest main-area gx bucket pair
        # — the spec's frontier-diversity gate reads this line.
        keys = [k for k in self.archive.cells if k[0] == self.main_area]
        if not keys:
            return []
        top = max(k[3] for k in keys)
        return sorted({k[1] for k in keys if k[3] >= top - 1})

    def progress_line(self, elapsed: float) -> None:
        line = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed),
            "cells": len(self.archive),
            "max_gx": self.max_gx,
            "phase_classes_at_max_gx": self._phases_at_max(),
            "solutions": self.n_solutions,
            "records": self.archive.total_records,
            "steps": self.steps_done,
            "sps": round(self.steps_done / max(elapsed, 1e-9)),
        }
        with open(self.out / "progress.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[go_explore] {json.dumps(line)}", flush=True)

    def flush(self) -> None:
        self.archive.save(self.out / "archive.pkl")
        with open(self.out / "traces.pkl", "wb") as f:
            pickle.dump(self.traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        (self.out / "roots.json").write_text(
            json.dumps(self.roots, indent=2) + "\n")

    def resume(self) -> bool:
        if not ((self.out / "archive.pkl").exists()
                and (self.out / "traces.pkl").exists()):
            return False
        self.archive.load(self.out / "archive.pkl")
        with open(self.out / "traces.pkl", "rb") as f:
            self.traces = pickle.load(f)
        self.roots = {
            k: {"path": v["path"], "start_wd": tuple(v["start_wd"]),
                "area": int(v.get("area", 0))}
            for k, v in json.loads(
                (self.out / "roots.json").read_text()).items()
        }
        self.main_area = next(iter(self.roots.values()))["area"]
        self.max_gx = int(self.archive.best_score())
        # Rebuild the per-(root, phase) solution dedupe table from the
        # sidecars so a resumed run doesn't re-dump known combos.
        for p in sorted((self.out / "solutions").glob("sol_*.json")):
            d = json.loads(p.read_text())
            combo = (d["root_id"], int(d["phase_class"]))
            self.sol_seen[combo] = min(
                self.sol_seen.get(combo, 10 ** 9), int(d["actions"]))
        self.sol_counter = len(list(
            (self.out / "solutions").glob("sol_*.json")))
        self.n_solutions = len(self.sol_seen)
        print(f"[go_explore] resumed: {len(self.archive)} cells, "
              f"{self.n_solutions} solution combos", flush=True)
        return True


def main() -> int:
    hs = "checkpoints/harvested_seeds"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-state",
                    default=str(REPO / hs / "seed_2_1_gx1421_runway.state"))
    ap.add_argument("--prefix-actions",
                    default=str(REPO / hs / "ep231_prefix_actions.npy"))
    ap.add_argument("--handoff-state",
                    default=str(REPO / "checkpoints/handoffs/handoff_2-1.state"),
                    help="Second lineage root ('' disables it).")
    ap.add_argument("--profile",
                    default=str(REPO
                                / "checkpoints/mario_2_1_runB/weld_2_1_runB.yaml"),
                    help="Source of action_space + frame_skip only.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--burst", type=int, default=48)
    ap.add_argument("--sticky", type=float, default=0.5)
    ap.add_argument("--deep-bias", type=float, default=0.35)
    ap.add_argument("--minutes", type=float, default=0,
                    help="Wall-clock budget (0 = until solution quota).")
    ap.add_argument("--want-solutions", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--phase-seed-stride", type=int, default=24)
    ap.add_argument("--handoff-seed-bursts", type=int, default=4)
    ap.add_argument("--flush-secs", type=float, default=300)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    h = Harvester(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(h, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(h, "stop", True))
    if not h.resume():
        h.seed()
        h.flush()
    h.explore()
    h.flush()
    h.progress_line(time.time() - h.t0)
    print(f"[go_explore] done: {json.dumps(h.archive.stats())}, "
          f"{h.n_solutions} solutions -> {h.out}", flush=True)
    h.pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
