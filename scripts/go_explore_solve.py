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
import threading
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


class SmbGame:
    @staticmethod
    def score_bonus(ram) -> int:
        return 0

    """SMB adapter — wraps the module-level helpers verbatim, so solver
    behavior on SMB is byte-identical to the pre-adapter code (regression:
    seeded 1-1 run reproduces the same solution sha)."""
    rom = ROM
    progress_cap = 7000   # transition-frame garbage guard (8-1 lesson)

    def progress(self, ram) -> int:
        return _gx(ram)

    def level_key(self, ram) -> tuple:
        return _wd(ram)

    def lives(self, ram) -> int:
        return int(ram[R_LIVES])

    def area(self, ram) -> int:
        return int(ram[R_AREA])

    def y(self, ram) -> int:
        return int(ram[R_YPOS])

    def swim(self, ram) -> int:
        return int(ram[0x1D])

    def cell_fn(self, ram) -> tuple:
        return cell_fn(ram)

    def is_clear(self, start_key: tuple, ram) -> bool:
        return is_forward_clear(start_key, ram)

    def is_dead(self, ram, start_lives: int) -> bool:
        return (int(ram[R_LIVES]) < start_lives
                or int(ram[R_PSTATE]) in DEATH_STATES)

    def is_finale(self, start_key: tuple, ram) -> bool:
        # GAME-COMPLETE (8-4): the ending never advances the world/level
        # bytes — victory is operating mode $0770 == 2 with inputs locked.
        return int(ram[0x770]) == 2 and tuple(start_key) == (7, 3)

    def room_id(self, ram) -> tuple:
        # rid = (area type, swim flag, Bowser-slot) — verified 2026-07-26:
        # $074E changes 0.67/1k steps, only at full-screen transitions.
        bw = 1 if 0x2D in bytes(ram[0x14:0x1C]) else 0
        return (int(ram[0x74E]), int(ram[0x1D]), bw)

    @staticmethod
    def label(key: tuple) -> str:
        return f"{key[0] + 1}-{key[1] + 1}"


class GenericGame:
    """Profile-driven adapter: every address comes from the game profile's
    `solve:` section, whose bytes must be observationally verified first
    (scripts/verify_ram_map.py; receipts under docs/receipts/ram_verify/).

    Required solve keys: rom, progress {lo[, hi]}, y, level_key (list of
    addrs; any lexicographic advance = clear), lives (decrement = death).
    Optional: area, progress_cap, player_state + death_states, finale
    {addr, value, level_key} for the game's ending."""

    def __init__(self, profile: dict) -> None:
        s = profile["solve"]
        self.rom = str(REPO / s["rom"])
        p = s["progress"]
        self._plo = int(p["lo"])
        self._phi = int(p["hi"]) if "hi" in p else None
        self._y = int(s["y"])
        self._lk = [int(a) for a in s["level_key"]]
        self._lives = int(s["lives"])
        self._area = int(s["area"]) if "area" in s else None
        self.progress_cap = int(s.get("progress_cap", 30000))
        self._pstate = int(s["player_state"]) if "player_state" in s else None
        self._death_states = tuple(s.get("death_states", ()))
        self._finale = s.get("finale")
        # Boss-fight progress (optional): a fixed-camera boss room pins the
        # gx frontier and position cells saturate in minutes (block-3 lesson:
        # 4.2M steps, 5,235 cells, zero gradient). `boss: {hp, start}` keys
        # cells on the boss-HP byte and scores damage dealt, giving the
        # archive a fight dimension. The byte must be discovered by
        # differential analysis of our own rollouts and receipted like every
        # other solve address.
        b = s.get("boss") or {}
        self._boss_hp = int(b["hp"]) if "hp" in b else None
        self._boss_start = int(b.get("start", 255))
        # Movement-mode signature (optional): `state_sig: [{addr, match}]`.
        # Each entry contributes one bit: ram[addr] in match. Separates
        # aliased player modes at the same coordinates — the CV block-3
        # lesson: stair-climb states share (gx, y-band) with jump apexes,
        # tie-break to fewer steps, and the airborne dead-end wins the
        # cell; the staircase stays structurally invisible to the archive.
        self._state_sig = [(int(e["addr"]),
                            frozenset(int(v) for v in e.get("match", ())))
                           for e in s.get("state_sig", ())]
        # Room signature (optional): `room_sig: [addr,...]` — bytes stable
        # within a room and different across rooms (found by before/after
        # transition diff of our own climbs). Feeds room_id, so the sect/
        # psig transit machinery (built on SMB's $074E) counts CV room
        # progress even though gx resets in every room.
        self._room_sig = tuple(int(a) for a in s.get("room_sig", ()))

    def progress(self, ram) -> int:
        v = int(ram[self._plo])
        if self._phi is not None:
            v |= int(ram[self._phi]) << 8
        return v

    def level_key(self, ram) -> tuple:
        return tuple(int(ram[a]) for a in self._lk)

    def lives(self, ram) -> int:
        return int(ram[self._lives])

    def area(self, ram) -> int:
        return int(ram[self._area]) if self._area is not None else 0

    def y(self, ram) -> int:
        return int(ram[self._y])

    def swim(self, ram) -> int:
        return 0

    def cell_fn(self, ram) -> tuple:
        # Same arity as SMB's cell (selection caches index key[-5]/key[-1];
        # boss HP and the mode bits ride in free middle slots so those
        # stay intact).
        hp = int(ram[self._boss_hp]) if self._boss_hp is not None else 0
        sig = 0
        for i, (a, m) in enumerate(self._state_sig):
            if int(ram[a]) in m:
                sig |= 1 << i
        return (self.area(ram), hp, sig,
                self.y(ram) // Y_BAND, self.progress(ram) // GX_BUCKET)

    def score_bonus(self, ram) -> int:
        # Damage dealt dominates within-room gx differences (a whip hit is
        # worth more frontier than any sidestep) without touching the
        # cross-room score ordering scale.
        if self._boss_hp is None:
            return 0
        return max(0, self._boss_start - int(ram[self._boss_hp])) * 2000

    def is_clear(self, start_key: tuple, ram) -> bool:
        # Forward = lexicographic advance of the level key (stage counters
        # increment; a game-over reset reads backward and lands in `dead`).
        return self.level_key(ram) > tuple(start_key)

    def is_dead(self, ram, start_lives: int) -> bool:
        if self.lives(ram) < start_lives:
            return True
        return (self._pstate is not None
                and int(ram[self._pstate]) in self._death_states)

    def is_finale(self, start_key: tuple, ram) -> bool:
        f = self._finale
        return (bool(f) and tuple(start_key) == tuple(f["level_key"])
                and int(ram[int(f["addr"])]) == int(f["value"]))

    def room_id(self, ram) -> tuple:
        return (self.level_key(ram) + (self.area(ram),)
                + tuple(int(ram[a]) for a in self._room_sig))

    @staticmethod
    def label(key: tuple) -> str:
        return "-".join(str(x) for x in key)


def make_game(profile: dict):
    """SMB profiles carry no `solve:` section — they get the byte-exact
    SMB adapter. A profile with `solve:` opts into the generic path."""
    return GenericGame(profile) if "solve" in profile else SmbGame()


class Solver:
    def __init__(self, args) -> None:
        # Cell granularity globals are consumed by cell_fn at call time.
        # main() sets them for subprocess runs; IN-PROCESS constructions
        # (the live show) previously bypassed that, silently ignoring
        # args.gx_bucket/y_band — the escalation ladder's micro-search
        # arm would have been a no-op. Apply here so both paths agree
        # (subprocess path re-applies the same values: no change).
        global GX_BUCKET, Y_BAND
        GX_BUCKET = int(getattr(args, "gx_bucket", GX_BUCKET))
        Y_BAND = int(getattr(args, "y_band", Y_BAND))
        self.args = args
        self.out = Path(args.out)
        (self.out / "solutions").mkdir(parents=True, exist_ok=True)
        profile = yaml.safe_load(Path(args.profile).read_text())
        self.bitmasks = action_space_to_bitmasks(profile["action_space"])
        self.weights = np.array(action_weights(profile["action_space"]))
        self.weights /= self.weights.sum()
        self.inv_weights = np.array(inverted_weights(profile["action_space"]))
        self.inv_weights /= self.inv_weights.sum()
        self.game = make_game(profile)
        self.pool = Pool(rom_path=self.game.rom, num_workers=args.workers,
                         frame_skip=int(profile.get("frame_skip", 4)))
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self.rng = np.random.default_rng(args.seed)
        # Sustained-hold macros (generic mechanism, profile-selected): at a
        # small per-step probability a worker settles briefly then HOLDS one
        # input for N steps — producing the long consecutive holds that
        # stochastic sampling almost never emits (stair mounts, pipe
        # entries). The 8-4 pipe-entry macro was this mechanism hardcoded;
        # macro steps are recorded verbatim in traces like any other action.
        self.macros = []          # (action_idx, hold_steps, weight)
        self.macro_p = 0.0
        for m in profile.get("solve", {}).get("hold_macros", []):
            want = set(m["buttons"])
            idx = next((i for i, c in enumerate(profile["action_space"])
                        if set(c) == want), None)
            if idx is None:
                print(f"[solver] hold_macro {m['buttons']} not in "
                      f"action_space — skipped", flush=True)
                continue
            p = float(m.get("p", 0.02))
            self.macros.append((idx, int(m.get("steps", 20)), p))
            self.macro_p += p
        if self.macros:
            w = np.array([m[2] for m in self.macros])
            self._macro_weights = w / w.sum()
        self.archive = GoExploreArchive(self.game.cell_fn, seed=args.seed)
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
        # Maze-coverage recipes (research round 2), flag-gated so the
        # default path stays byte-identical to the receipted campaign:
        # sel_mode "count" swaps the uniform arm for Go-Explore's native
        # count-based prior W = 1/sqrt(times_chosen+1) * (score_norm +
        # 0.1) via O(1) rejection sampling (the archive's own weighted
        # selection was bypassed for being an O(N) scan — this restores
        # the prior without the scan). frontier_throttle N > 0 arms
        # DD-RRT-style boundary suppression: a cell whose bursts yield
        # nothing novel N times in a row is excluded from the deep-
        # frontier band (stop throwing bursts at the physical wall).
        self.sel_mode = str(getattr(args, "sel_mode", "legacy"))
        self.frontier_throttle = int(getattr(args, "frontier_throttle", 0))
        # R4 door discovery (research round 2): cut vertices of the archive
        # transition graph are the maze's doors — every route between the
        # regions they separate passes through them. Edges are recorded from
        # our own rollouts (cell-to-cell transitions, no game internals); an
        # async pass re-derives articulation points every door_interval and
        # the count arm up-weights door cells by door_weight. 0 = off, and
        # the default path stays byte-identical to the receipted campaign.
        self.door_weight = float(getattr(args, "door_weight", 0.0))
        self.door_interval = float(getattr(args, "door_interval", 45.0))
        self._key_ids: dict = {}          # cell key -> int id (interned)
        self._adj: dict = {}              # id -> set of neighbor ids
        self._doors: frozenset = frozenset()
        self._door_lock = threading.Lock()
        self._door_thread: threading.Thread | None = None
        self._last_door_t = time.time()
        self._recorded_new = False
        # Optional spectator hook: called every pool step with
        # (results, solver) — the FULL per-worker results list, so the
        # live show can render one worker or the whole swarm. None in
        # headless runs (zero overhead).
        self.step_hook = None

    def _step0(self, a: int):
        acts = np.zeros(self.args.workers, dtype=np.uint8)
        acts[0] = self.bitmasks[a]
        return self.pool.step_all(acts)[0][2]

    # ---- record path -------------------------------------------------

    def observe(self, wid: int, ram, trace: list, steps: int,
                root_id: str, loops: int = 0,
                route_sig: tuple = (), sect: int = 0,
                psig: tuple = (), ctx: dict | None = None) -> str:
        """Record one reached state. Returns 'dead' | 'clear' | 'live'."""
        # GAME-COMPLETE check (8-4 finale): the ending never advances the
        # world/level bytes (there is no next level) — the victory state is
        # operating mode $0770 == 2 with inputs locked (verified 2026-07-27,
        # THANK YOU MARIO screen). Without this the winning trajectory sits
        # in the archive invisible, as it did for 1.5 hours on the night the
        # game was first beaten.
        game = self.game
        if game.is_finale(self.start_wd, ram):
            self._dump_solution(root_id, trace, ram, steps)
            return "clear"
        # Clear (warp-guarded) is checked FIRST.
        if game.is_clear(self.start_wd, ram):
            self._dump_solution(root_id, trace, ram, steps)
            return "clear"
        # Any life lost, or an explicit dying state = death. Lives-based
        # detection is robust across enemy/pit/time deaths and multi-area
        # levels.
        if game.is_dead(ram, self.start_lives):
            return "dead"
        # A level-key change that ISN'T a forward clear = a warp (or a
        # backward reload / game-over reset): do not record it (would
        # poison the archive with off-level cells).
        if game.level_key(ram) != self.start_wd:
            return "dead"
        gx = game.progress(ram)
        if gx > game.progress_cap:
            # Transition-frame garbage read (page byte mid-load reads huge);
            # real SMB levels reach ~6,300 px (8-1) — the old 3900 cap silently
            # froze the 8-1 frontier at 3900 (states past it never archived).
            return "live"
        area = game.area(ram)
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
        key = (sect, psig, loops, route_sig) + game.cell_fn(ram)
        # R4 edge recording: a cell-to-cell transition in OUR OWN rollout is
        # an edge of the maze's traversal graph. Interned ids keep the
        # adjacency compact at castle-archive scale (1M+ cells).
        if ctx is not None and self.door_weight > 0:
            pk = ctx.get("cur_key")
            if pk is not None and pk != key:
                ids = self._key_ids
                ia = ids.get(pk)
                if ia is None:
                    ia = ids[pk] = len(ids)
                ib = ids.get(key)
                if ib is None:
                    ib = ids[key] = len(ids)
                with self._door_lock:
                    self._adj.setdefault(ia, set()).add(ib)
                    self._adj.setdefault(ib, set()).add(ia)
            ctx["cur_key"] = key
        score = sect * 10000 + gx + game.score_bonus(ram)
        cur = self.archive.cells.get(key)
        dom = (cur is None or score > cur.best_score + 1e-9
               or (abs(score - cur.best_score) <= 1e-9 and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, score, steps,
                                                        key=key):
                self.traces[key] = (root_id, bytes(trace), loops, route_sig, sect, psig)
                self._recorded_new = True
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
            # Full effective invocation (audit finding: receipts recorded
            # neither workers nor profile nor buckets, making parity
            # claims between runs unverifiable after the fact).
            "solver_args": {k: v for k, v in vars(self.args).items()
                            if isinstance(v, (str, int, float, bool))},
            "root_id": root_id,
            "root_state": self.roots[root_id]["path"],
            "start_wd": list(self.start_wd),
            "clear_wd": list(self.game.level_key(ram)),
            "steps": steps, "actions": len(trace),
        }, indent=2) + "\n")
        print(f"[go_explore_solve] *** SOLUTION {n} *** root={root_id} "
              f"{len(trace)} actions, {self.start_wd}->"
              f"{self.game.level_key(ram)}", flush=True)

    # ---- seeding: root ONLY (honest) ---------------------------------

    def seed(self) -> None:
        path = self.args.root_state
        self.pool.load_worker_state(0, Path(path).read_bytes())
        r = self._step0(NOOP)  # convention: load root, one NOOP, then actions
        self.start_wd = self.game.level_key(r)
        self.start_lives = self.game.lives(r)
        self.max_area = self.game.area(r)
        self.roots["entrance"] = {"path": str(path),
                                  "start_wd": list(self.start_wd),
                                  "lives": self.start_lives}
        self.observe(0, r, [], 0, "entrance")
        prev = getattr(self.args, "resume_archive", None)
        if prev:
            prev = Path(prev)
            self.archive.load(prev / "archive.pkl")
            with open(prev / "traces.pkl", "rb") as f:
                self.traces.update(pickle.load(f))
            saved_roots = json.loads((prev / "roots.json").read_text())
            for rid, info in saved_roots.items():
                self.roots.setdefault(rid, info)
            # Rebuild the frontier trackers the loaded cells imply.
            for c in self.archive.cells.values():
                area, gx = c.key[-5], c.key[-1] * GX_BUCKET
                sect = c.key[0]
                if area > self.max_area:
                    self.max_area = area
                if gx > self.max_gx_in_area.get(area, 0):
                    self.max_gx_in_area[area] = gx
                if sect > self.max_sect:
                    self.max_sect = sect
            print(f"[seed] RESUMED archive from {prev}: "
                  f"{len(self.archive.cells)} cells, "
                  f"{len(self.traces)} traces, max_area={self.max_area}, "
                  f"max_sect={self.max_sect}", flush=True)
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
        self._sel_maxscore = max(
            (c.best_score for c in self._sel_cells), default=1.0) or 1.0
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
            if pool_band and self.frontier_throttle > 0:
                # DD-RRT boundary suppression: cells whose bursts came
                # back empty `throttle` times in a row are walls — stop
                # sampling them; if the whole band is walls, fall through
                # to the count arm (least-visited exploration).
                pool_band = [c for c in pool_band
                             if getattr(c, "barren", 0) < self.frontier_throttle]
            if pool_band:
                floor = self._sel_topgx - int(self.rng.integers(0, 24))
                band = [c for c in pool_band if c.key[-1] >= floor]
                if band:
                    cell = band[int(self.rng.integers(len(band)))]
                    cell.times_chosen += 1
                    cell.explored = True
                    return cell
        if self.sel_mode == "count":
            # Count-based prior via O(1) rejection sampling: accept cell
            # with prob W/Wmax, W = 1/sqrt(times_chosen+1) * (score_norm
            # + 0.1), Wmax = 1.1. Expected O(few) draws; bounded at 64.
            # R4: door cells (articulation points of the transition graph)
            # get door_weight x — Wmax scales so the sampling stays exact.
            ms = self._sel_maxscore
            doors = self._doors
            dw = self.door_weight if doors else 0.0
            wmax = 1.1 * max(dw, 1.0)
            pick = None
            for _ in range(64):
                pick = cells[int(self.rng.integers(len(cells)))]
                # R2 extension (block-3 doomed-tip lesson): a cell whose
                # bursts die yielding nothing `throttle` times in a row is
                # skipped here too — deterministic death states (e.g. a
                # committed bat swoop) otherwise drain the count arm's
                # budget forever, since depth keeps their score high.
                if (self.frontier_throttle > 0
                        and getattr(pick, "barren", 0) >= self.frontier_throttle):
                    continue
                w = ((1.0 / (pick.times_chosen + 1) ** 0.5)
                     * (pick.best_score / ms + 0.1))
                if dw > 0 and self._key_ids.get(pick.key) in doors:
                    w *= dw
                if self.rng.random() < w / wmax:
                    break
            pick.times_chosen += 1
            pick.explored = True
            return pick
        # Legacy: uniform over the cached list (the archive's weighted
        # selection was an O(N) scan; the deep arm supplies direction).
        return cells[int(self.rng.integers(len(cells)))]

    # ---- R4: door discovery (articulation points, async) --------------

    @staticmethod
    def _articulation_points(adj: dict) -> set:
        """Iterative Hopcroft-Tarjan cut vertices of an undirected graph
        given as {node: iterable-of-neighbors}. Pure derived-from-rollouts
        structure — no game internals."""
        disc: dict = {}
        low: dict = {}
        ap: set = set()
        timer = 0
        for start in adj:
            if start in disc:
                continue
            disc[start] = low[start] = timer
            timer += 1
            root_children = 0
            stack = [(start, None, iter(adj.get(start, ())))]
            while stack:
                node, parent, it = stack[-1]
                pushed = False
                for nb in it:
                    if nb == parent:
                        continue
                    if nb in disc:
                        if disc[nb] < low[node]:
                            low[node] = disc[nb]
                    else:
                        disc[nb] = low[nb] = timer
                        timer += 1
                        if node == start:
                            root_children += 1
                        stack.append((nb, node, iter(adj.get(nb, ()))))
                        pushed = True
                        break
                if not pushed:
                    stack.pop()
                    if stack:
                        pnode = stack[-1][0]
                        if low[node] < low[pnode]:
                            low[pnode] = low[node]
                        if pnode != start and low[node] >= disc[pnode]:
                            ap.add(pnode)
            if root_children > 1:
                ap.add(start)
        return ap

    def _door_scan(self) -> None:
        """Snapshot the adjacency and publish the current door set. Runs in
        a daemon thread; the snapshot copy holds the edge lock briefly (the
        4-4 frozen-show lesson: never scan a live million-entry structure
        from the hot loop)."""
        try:
            with self._door_lock:
                snap = {k: tuple(v) for k, v in self._adj.items()}
            self._doors = frozenset(self._articulation_points(snap))
        except Exception as exc:
            print(f"[door_scan] failed (doors unchanged): {exc}", flush=True)

    def _maybe_scan_doors(self, now: float) -> None:
        if self.door_weight <= 0:
            return
        if now - self._last_door_t < self.door_interval:
            return
        if self._door_thread is not None and self._door_thread.is_alive():
            return
        self._last_door_t = now
        self._door_thread = threading.Thread(target=self._door_scan,
                                             daemon=True)
        self._door_thread.start()

    def _assign(self, wid: int, prev: dict | None = None) -> dict:
        # R2 bookkeeping: credit or debit the cell the finished burst
        # was rooted at. A burst that recorded nothing novel increments
        # the source cell's barren counter; any novelty resets it.
        if prev is not None and prev.get("key") is not None:
            src = self.archive.cells.get(prev["key"])
            if src is not None:
                if prev.get("yielded"):
                    src.barren = 0
                else:
                    src.barren = getattr(src, "barren", 0) + 1
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
                "sect": sect, "p0750": None, "psig": psig, "cur_key": cell.key,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights))}

    def explore(self) -> None:
        args = self.args
        game = self.game
        cap = game.progress_cap
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
                if (c.get("macro_left", 0) <= 0 and self.macros
                        and self.rng.random() < self.macro_p):
                    mi = int(self.rng.choice(len(self.macros),
                                             p=self._macro_weights))
                    c["macro_a"], c["macro_hold"] = self.macros[mi][:2]
                    c["macro_left"] = c["macro_hold"] + 6   # settle, then hold
                if c.get("macro_left", 0) > 0:
                    c["macro_left"] -= 1
                    a = NOOP if c["macro_left"] >= c["macro_hold"] \
                        else c["macro_a"]
                else:
                    a = c["prev"] if self.rng.random() < args.sticky else \
                        int(self.rng.choice(len(_w), p=_w))
                c["prev"] = a
                c["pending"] = a
                acts[i] = self.bitmasks[a]
            results = self.pool.step_all(acts)
            self.steps_done += args.workers
            if self.step_hook is not None:
                try:
                    self.step_hook(results, self)
                except Exception:
                    pass
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
                _lgx = game.progress(ram)
                c["gx"] = _lgx if _lgx <= cap else c.get("gx", -1)
                # ROOM IDENTITY (SMB: verified 2026-07-26, $074E changes
                # 0.67/1k steps, only at full-screen transitions). A rid
                # CHANGE is a true room transition.
                _rid = game.room_id(ram)
                _transit = (c["p0750"] is not None and _rid != c["p0750"])
                if _transit and c["sect"] < 16:
                    c["sect"] += 1
                    c["psig"] = _rid
                c["p0750"] = _rid
                if _lgx <= cap:
                    if c["prev_gx"] >= 0:
                        if _transit:
                            c["prev_gx"] = _lgx   # section change: re-arm, no loop
                        elif _lgx < c["prev_gx"] - 100:
                            if c["loops"] < 8:
                                c["loops"] += 1
                            c["sig"] = ()   # new pass, fresh route
                        elif _lgx // 512 != c["prev_gx"] // 512:
                            c["sig"] = (c["sig"] + (game.y(ram) // 64,))[-4:]
                    c["prev_gx"] = _lgx
                if (self.args.swim_gx_ceiling > 0 and game.swim(ram) == 1
                        and _lgx <= cap
                        and _lgx > self.args.swim_gx_ceiling):
                    ctx[i] = self._assign(i, prev=c)
                    continue
                status = self.observe(i, ram, c["trace"], c["steps"],
                                      c["root"], c["loops"], c["sig"],
                                      c["sect"], c["psig"], ctx=c)
                # Finisher extension: the level-END transition (exit pipe /
                # flag slide) can run many steps with gx frozen, so a burst
                # from the deepest cell can end just short of the wd advance.
                # A burst ending in the deepest-area top band gets one +200
                # extension so it can actually complete the clear.
                if (status == "live" and c["left"] <= 0
                        and not c.get("extended")
                        and game.area(ram) == self.max_area
                        and _lgx // 16 >= self.max_gx_in_area.get(self.max_area, 0) // 16 - 3):
                    c["left"] += 200
                    c["extended"] = True
                if status != "live" or c["left"] <= 0 or c["steps"] >= args.max_steps:
                    ctx[i] = self._assign(i, prev=c)
            now = time.time()
            self._maybe_scan_doors(now)
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
        if self.door_weight > 0:
            line["doors"] = len(self._doors)
            line["edges"] = sum(len(v) for v in self._adj.values()) // 2
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
    ap.add_argument("--sel-mode", choices=("legacy", "count"),
                    default="legacy",
                    help="count = Go-Explore count-based selection prior "
                         "via O(1) rejection sampling (maze-coverage R1).")
    ap.add_argument("--frontier-throttle", type=int, default=0,
                    help="If >0: exclude cells whose bursts came back "
                         "empty this many times in a row from the deep-"
                         "frontier band (DD-RRT boundary suppression, R2).")
    ap.add_argument("--door-weight", type=float, default=0.0,
                    help="If >0: up-weight archive cells that are cut "
                         "vertices (doors) of the rollout transition graph "
                         "by this factor in the count arm (Hopcroft-Tarjan "
                         "sub-goals, R4). Needs --sel-mode count. 5 = "
                         "research-recipe default.")
    ap.add_argument("--door-interval", type=float, default=45.0,
                    help="Seconds between async door (articulation-point) "
                         "recomputations (R4).")
    ap.add_argument("--resume-archive", type=str, default=None, metavar="DIR",
                    help="Resume from a prior run's flushed out-dir: load "
                         "archive.pkl + traces.pkl + roots.json and continue "
                         "exploring instead of starting from one root cell. "
                         "Iterating on a single wall keeps every hard-won "
                         "frontier cell (e.g. CV block-3's stair funnel).")
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
