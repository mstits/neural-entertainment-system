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
from scripts.go_explore_solve import make_game  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

SOLVE = str(REPO / "scripts" / "go_explore_solve.py")


def extract_next_entrance(profile, root_bytes: bytes, actions, out_path: Path,
                          settle: int = 8):
    """Replay `actions` from the root; at the first level-key transition,
    settle `settle` no-op frames and snapshot the next level's entrance.
    Returns (path, next_key) or (None, None) if no transition seen."""
    game = make_game(profile)
    bm = action_space_to_bitmasks(profile["action_space"])
    pool = Pool(rom_path=game.rom, num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.reset_all()
    pool.load_worker_state(0, root_bytes)

    def step(a):
        x = np.zeros(1, dtype=np.uint8)
        x[0] = bm[a]
        return pool.step_all(x)[0][2]

    r = step(0)
    start_key = game.level_key(r)
    result = (None, None)
    for a in actions:
        r = step(int(a))
        if game.level_key(r) != start_key:
            for _ in range(settle):
                r = step(0)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(pool.save_worker_state(0))
            result = (str(out_path), game.level_key(r))
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
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    game = make_game(profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chain_log = out / "chain.jsonl"

    cur_state = str(args.start_state)
    cur_label = args.start_label
    solved = []
    for i in range(args.max_levels):
        lvl_out = out / f"lvl_{i:02d}_{cur_label}"
        print(f"\n===== CHAIN level {i}: solving {cur_label} from "
              f"{cur_state} =====", flush=True)
        cmd = [
            sys.executable, SOLVE, "--out", str(lvl_out),
            "--root-state", cur_state, "--profile", args.profile,
            "--workers", str(args.workers),
            "--minutes", str(args.minutes_per_level),
            "--want-solutions", "1", "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=False)
        sols = sorted(lvl_out.glob("solutions/sol_*.actions.npy"))
        if not sols:
            print(f"[chain] STALL: {cur_label} not solved in "
                  f"{args.minutes_per_level} min — chain stops at {len(solved)} "
                  f"levels.", flush=True)
            with open(chain_log, "a") as f:
                f.write(json.dumps({"level": cur_label, "status": "stall"}) + "\n")
            break
        actions = np.load(sols[0]).astype(np.int64)
        nxt_path = out / "entrances" / f"entrance_after_{cur_label}.state"
        npath, nwd = extract_next_entrance(profile, Path(cur_state).read_bytes(),
                                           actions, nxt_path)
        rec = {"level": cur_label, "status": "solved",
               "actions": int(len(actions)), "solution": str(sols[0])}
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
