"""Hands-off Go-Explore CHAIN: solve consecutive SMB levels from clean
entrances, extracting each next entrance from the prior level's clear.

For each level: subprocess go_explore_solve.py from the current entrance until
it produces a solution (a warp-guarded forward clear); replay that solution to
the level transition, settle a few frames, and snapshot the NEXT level's clean
entrance; repeat. Stops on a stall (a level not solved within the per-level
budget) or when max_levels is reached. Every entrance + solution is banked, so
the whole progression is receipted and resumable.

This demonstrates the search agent clearing level after level from power-on-style
clean entrances (no mid-level seeds), machine-busy and hands-off.

Usage:
  python scripts/go_explore_chain.py --start-state <entrance.state> \
      --start-label 2-2 --profile configs/mario_1_2_solo.yaml \
      --out runs/ge_chain --max-levels 12 --minutes-per-level 20 --workers 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from scripts.go_explore_solve import (  # noqa: E402
    apply_hw_flags, hw_provenance, make_game, resolve_hw_flags,
    write_state_sidecar,
)
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

SOLVE = str(REPO / "scripts" / "go_explore_solve.py")


def extract_next_entrance(profile, root_bytes: bytes, actions, out_path: Path,
                          settle: int = 8, stable_for: int = 45,
                          settle_cap: int = 600, hw_flags=(),
                          transit_timeout_frames: int = 0):
    """Replay `actions` from the root; at the first level-key transition
    that lands on a genuinely NEW, ALIVE, stable state, snapshot it as the
    next level's entrance. Returns (path, next_key) or (None, None) if no
    such transition is ever reached.

    The old fixed 8-step settle could snapshot MID-TRANSITION: Bubble
    Bobble's umbrella warp animates ~400 frames with the round counter
    ticking through values, and a mid-warp "entrance" seeded the next
    solve inside an animation whose counter blips stalled it outright
    (round 52, 2026-08-06). Stability-settling lands the snapshot in the
    destination level's actual play. `settle` no-ops still run first
    (rendering-stability, the original purpose).

    Two more failure shapes surfaced the same day settling on Bubble
    Bobble round 9's arcade GAME OVER / PASSWORD screen (a live-show
    stream stall of several minutes): (1) a transient level-key blip
    during a death animation can drift back to the SAME value it started
    at, which settles "stable" but isn't a forward transition at all —
    reject and keep scanning if the settled key == start_key; (2) the
    settled state can be dead (GAME OVER: the round HUD digit freezes at
    whatever it last showed, satisfying both the not-equal-to-start check
    on the transient blip and the later stability check, while lives has
    already hit 0) — reject via the game adapter's own is_dead() using
    lives captured at the true root, before any transition. A doomed
    landing is treated as if no transition were seen on this action
    range; the search resumes scanning the rest of `actions`.

    REJECTION MUST REWIND. Settling is destructive: it steps up to
    `settle + settle_cap` NOOPs into the pool. When the settle is then
    REJECTED, "keep scanning `actions`" is only honest if the pool is
    put back where the blip fired — otherwise the remaining actions
    replay against a machine that idled for hundreds of steps and the
    real clear never reproduces. Measured on Bubble Bobble round 67
    (2026-08-09): the round counter $0401 blips DOWN one step to the
    previous round's value at replay step 3, the settle burned 53 steps,
    settled back on the start key, and the 200 remaining actions then
    ran desynchronised — the genuine 67->68 transition on the very last
    action was invisible and the chain reported "no forward transition
    captured after 67". Round 66 survived the same blip only by luck:
    its blip landed at action 438 of 440, so the destructive settle
    coasted into the next round instead of wrecking a replay tail. The
    rewind is a save/restore round-trip around the settle.

    `transit_timeout_frames` extends the scan PAST the end of `actions`
    with NOOPs (converted to steps by `frame_skip`) for games whose
    level/round byte only turns over after an uncontrollable interlude
    that outlasts the solution trace. 0 (the default) stops at the last
    action, exactly as before.

    `hw_flags` must be the SAME set the solve ran under: this replay
    machine is what stamps the next level's entrance blob, so a
    mismatch bakes a foreign-machine state into the chain and every
    downstream level inherits it. Empty (the default) reproduces the
    pre-existing behavior exactly."""
    game = make_game(profile)
    bm = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))
    pool = Pool(rom_path=game.rom, num_workers=1, frame_skip=fs)
    apply_hw_flags(pool, hw_flags)   # before reset_all(); order is load-bearing
    pool.set_headless(True)
    pool.reset_all()
    pool.load_worker_state(0, root_bytes)

    def step(a):
        x = np.zeros(1, dtype=np.uint8)
        x[0] = bm[a]
        return pool.step_all(x)[0][2]

    r = step(0)
    start_key = game.level_key(r)
    start_lives = game.lives(r)

    def judge_transition():
        """NOOP-settle the candidate transition and rule on it. Returns
        the settled key on acceptance; None on rejection, having REWOUND
        the pool to the pre-settle point so the caller's replay of the
        remaining actions stays in sync."""
        mark = pool.save_worker_state(0)
        rr = None
        for _ in range(settle):
            rr = step(0)
        # Settle until the key stops moving (transition fully over).
        key = game.level_key(rr)
        stable = 0
        for _ in range(settle_cap):
            if stable >= stable_for:
                break
            rr = step(0)
            k = game.level_key(rr)
            if k == key:
                stable += 1
            else:
                key, stable = k, 0
        settled_key = game.level_key(rr)
        if settled_key == start_key or game.is_dead(rr, start_lives):
            pool.load_worker_state(0, mark)
            return None    # false/doomed transition — scan resumes in sync
        return settled_key

    # NOOP tail past the end of the trace, for level bytes that only turn
    # over after an interlude the solution does not cover. 0 = stop at the
    # last action (the pre-existing behavior).
    tail = max(0, int(transit_timeout_frames)) // max(1, fs)
    result = (None, None)
    for a in list(actions) + [0] * tail:
        r = step(int(a))
        if game.level_key(r) != start_key:
            settled_key = judge_transition()
            if settled_key is None:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(pool.save_worker_state(0))
            # Sidecar, not blob: record the machine this entrance was
            # produced on so the next solve can be run on the same one.
            write_state_sidecar(out_path, hw_provenance(hw_flags, fs),
                                {"settled_key": list(settled_key)})
            result = (str(out_path), settled_key)
            break
    pool.shutdown()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--start-label", required=True, help="e.g. 2-2")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-levels", type=int, default=12)
    ap.add_argument("--minutes-per-level", type=float, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # Coverage-recipe passthrough to each per-level solve. Defaults
    # reproduce the plain solver exactly (sel-mode legacy, gx-bucket 16,
    # y-band 32); pass --sel-mode count (+ finer buckets) to grind the
    # hard-exploration levels that walled the plain chain (e.g. Lost
    # Levels 1-2, which the coverage ladder cracked where defaults froze).
    ap.add_argument("--sel-mode", choices=("legacy", "count"), default="legacy")
    ap.add_argument("--gx-bucket", type=int, default=16)
    ap.add_argument("--y-band", type=int, default=32)
    # Burst length (steps) per exploration rollout. Default 64 = the solver
    # default. Raise for games with long uncontrollable transition
    # animations: Bubble Bobble's umbrella item warps forward across a
    # ~400-frame animation, and a 64-step (256-frame at frame_skip 4)
    # burst ends INSIDE it — every burst re-archives the same mid-warp
    # cell and the chain stalls at 1 cell forever (round 52, 2026-08-06).
    ap.add_argument("--burst", type=int, default=64)
    # On a per-level stall, retry the SAME level with a fresh seed
    # (seed+1, seed+2, ...) up to this many times before declaring a real
    # stall. A coverage-reachable wall the solver freezes at is often a
    # low-probability stochastic passage that different exploration
    # randomness finds — e.g. Lost Levels 2-1 froze at gx~2985 on seed 0
    # but cleared on seed 1 with identical params. 0 = no retries (old
    # behavior).
    ap.add_argument("--seed-retries", type=int, default=0)
    # Room-transition domination cap passthrough (see go_explore_solve.py's
    # --sect-cap). Default 16 = byte-identical to every existing SMB1
    # caller. This chain driver is the primary unattended-multi-level
    # caller the knob needs to reach but didn't (found by adversarial
    # review, 2026-08-06) — Lost Levels needed --sect-cap 64 by hand
    # outside the chain to avoid silently freezing half the cell key.
    ap.add_argument("--sect-cap", type=int, default=16)
    # nes_core hw timing flags for the WHOLE chain — the per-level solves
    # and this driver's own entrance-extraction replay must run the same
    # machine, or the entrance blob this banks came from a machine the
    # next solve never reproduces (the CV cv_chain_hw2 failure mode).
    # Default: whatever the profile pins, else NONE (byte-identical to
    # every existing chain run). See go_explore_solve.py --hw-flags.
    ap.add_argument("--hw-flags", type=str, default=None, metavar="a,b,c")
    # Entrance extraction keeps scanning for the forward level-key
    # transition this many FRAMES past the end of the solution trace,
    # stepping NOOPs. For games that end a level on an uncontrollable
    # interlude (bonus/round-clear screens) the byte can turn over after
    # the trace stops. 0 = stop at the last action (pre-existing
    # behavior). Frames, not steps: divided by the profile's frame_skip.
    ap.add_argument("--transit-timeout-frames", type=int, default=0)
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    hw_flags = resolve_hw_flags(profile, args.hw_flags)
    if hw_flags:
        print(f"[chain] hw flags: {hw_flags}", flush=True)
    game = make_game(profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chain_log = out / "chain.jsonl"

    cur_state = str(args.start_state)
    cur_label = args.start_label
    solved = []
    for i in range(args.max_levels):
        base_out = out / f"lvl_{i:02d}_{cur_label}"
        print(f"\n===== CHAIN level {i}: solving {cur_label} from "
              f"{cur_state} =====", flush=True)
        # Try seed, then seed+1..seed+seed_retries on a stall. Each attempt
        # gets its own out dir so a fresh seed searches from scratch (no
        # archive carry-over from the frozen attempt).
        sols = []
        lvl_out = base_out
        for attempt in range(args.seed_retries + 1):
            seed = args.seed + attempt
            lvl_out = base_out if attempt == 0 else Path(f"{base_out}_s{seed}")
            if attempt > 0:
                print(f"[chain] {cur_label} stalled; retry {attempt}/"
                      f"{args.seed_retries} with seed {seed}", flush=True)
            cmd = [
                sys.executable, SOLVE, "--out", str(lvl_out),
                "--root-state", cur_state, "--profile", args.profile,
                "--workers", str(args.workers),
                "--minutes", str(args.minutes_per_level),
                "--want-solutions", "1", "--seed", str(seed),
                "--sel-mode", args.sel_mode,
                "--gx-bucket", str(args.gx_bucket),
                "--y-band", str(args.y_band),
                "--burst", str(args.burst),
                "--sect-cap", str(args.sect_cap),
                # Pass the RESOLVED set explicitly rather than the raw
                # CLI string: the subprocess then cannot re-resolve to
                # something different from the machine this driver uses
                # for entrance extraction. Empty resolves to 'none',
                # which sets nothing (the pre-existing behavior).
                "--hw-flags", ",".join(hw_flags) or "none",
            ]
            subprocess.run(cmd, check=False)
            sols = sorted(lvl_out.glob("solutions/sol_*.actions.npy"))
            if sols:
                break
        if not sols:
            print(f"[chain] STALL: {cur_label} not solved in "
                  f"{args.minutes_per_level} min x {args.seed_retries + 1} seed(s) "
                  f"— chain stops at {len(solved)} levels.", flush=True)
            with open(chain_log, "a") as f:
                f.write(json.dumps({"level": cur_label, "status": "stall"}) + "\n")
            break
        actions = np.load(sols[0]).astype(np.int64)
        nxt_path = out / "entrances" / f"entrance_after_{cur_label}.state"
        npath, nwd = extract_next_entrance(
            profile, Path(cur_state).read_bytes(), actions, nxt_path,
            hw_flags=hw_flags,
            transit_timeout_frames=args.transit_timeout_frames)
        rec = {"level": cur_label, "status": "solved",
               "actions": int(len(actions)), "solution": str(sols[0]),
               "hw_flags": list(hw_flags)}
        solved.append(cur_label)
        print(f"[chain] SOLVED {cur_label} ({len(actions)} actions)", flush=True)
        if npath is None:
            rec["next"] = None
            with open(chain_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[chain] no forward transition captured after {cur_label}; "
                  f"chain stops ({len(solved)} solved).", flush=True)
            break
        rec["next_wd"] = list(nwd)
        rec["next_label"] = game.label(nwd)
        rec["next_entrance"] = npath
        with open(chain_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cur_state = npath
        cur_label = game.label(nwd)

    print(f"\n[chain] DONE: solved {len(solved)} consecutive levels: "
          f"{solved}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
