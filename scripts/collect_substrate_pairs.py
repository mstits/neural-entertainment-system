"""Collect strict-success pairs for 1-3 / 1-4 — the substrate run's S1.

Registered by docs/proposals/SUBSTRATE_RUN_ADDENDUM_2026-08-29.md §2
BEFORE any of this ran: specialists are the eval harness's own banked
baseline specs (checkpoint / profile / cold-entrance start state from
`scripts/eval_shared_substrate.py`'s CONFIG["baseline"]), and every
collection knob is `scripts/interference_falsifier.py`'s registered
value VERBATIM (300 episodes, max_steps 3000, sticky 0.25, jitter 16,
sampled T=1.0, per-episode RNG, collect_seed 20260815, 5 lanes). The
1-1/1-2 pair files were collected 2026-08-16 under the identical knobs
and are reused as-is — this script exists only because the falsifier's
CONFIG declares two specialists and the substrate experiment needs
four.

Everything that touches the emulator or the npz format is IMPORTED
from the falsifier (`_collect_with_real_pool`,
`success_pairs_from_episodes`, `write_success_pairs`) so the four
levels' data share one code path, not a reimplementation. Output:
`runs/substrate_pairs/success_1_{3,4}.npz` plus a per-level receipt
JSON carrying the plan, the kept/total counts, and the npz sha256 at
write time (the addendum's SHAs-recorded-at-write-time requirement).

A `.run.lock` (src/utils/run_lock) guards against a duplicate
collection; the shared "emulator-pool" resource lock
(`src/utils/run_lock.acquire_resource`) guards against any OTHER
writer stepping the pool at the same time — see that module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_shared_substrate import (  # noqa: E402
    CONFIG as EVAL_CONFIG,
    _load_yaml,
)
from scripts.interference_falsifier import (  # noqa: E402
    CONFIG as FALSIFIER_CONFIG,
    _collect_with_real_pool,
    success_pairs_from_episodes,
    write_success_pairs,
)
from src.utils.run_lock import (  # noqa: E402
    acquire,
    acquire_resource,
    release,
    release_resource,
)

LEVELS = ("1-3", "1-4")
OUT_DIR = REPO / "runs" / "substrate_pairs"
POOL_RESOURCE = "emulator-pool"


def build_plan(level: str) -> dict:
    """The falsifier's plan shape, sourced per the addendum: specialist
    spec from the EVAL harness's banked baseline, knobs from the
    FALSIFIER's registration."""
    base = EVAL_CONFIG["baseline"][level]
    fc = FALSIFIER_CONFIG
    # Start state: the PROFILE's own start_state_path — the level's cold
    # entrance, exactly how the eval harness's build_level_command
    # resolves it, so collection entrance == eval entrance.
    profile_yaml = _load_yaml(REPO / base["profile"])
    return {
        "level": level,
        "mode": "collection",
        "profile": str(REPO / base["profile"]),
        "checkpoint": str(REPO / base["checkpoint"]),
        "start_state": str(REPO / profile_yaml["start_state_path"]),
        "episodes": fc["collect_episodes"],
        "max_steps": fc["collect_max_steps"],
        "sticky_prob": fc["collect_sticky"],
        "start_jitter": fc["collect_jitter"],
        "action_select": "sampled",
        "temperature": fc["collect_temperature"],
        "eval_rng": "per-episode",
        "eval_seed": fc["collect_seed"],
        "lanes": fc["collect_workers"],
        "keep": "strict_episode_success_only",
        "out": str(OUT_DIR / f"success_{level.replace('-', '_')}.npz"),
    }


def collect_level(plan: dict) -> dict:
    episodes = _collect_with_real_pool(plan)
    obs, act, traj_len, kept = success_pairs_from_episodes(episodes)
    if not kept:
        raise RuntimeError(
            f"{plan['level']}: 0 strict successes in {len(episodes)} "
            f"episodes — the specialist did not clear under the "
            f"collection protocol; S1 is VOID for this level, per the "
            f"addendum's VOID licence.")
    write_success_pairs(
        plan["out"],
        obs=obs, act=act, traj_len=traj_len,
        label_max_gx=np.array([e.get("max_gx", -1) for e in kept],
                              dtype=np.int64),
        label_episode_success=np.array(
            [bool(e["episode_success"]) for e in kept], dtype=bool),
        provenance=plan,
    )
    sha = hashlib.sha256(Path(plan["out"]).read_bytes()).hexdigest()
    receipt = {
        "type": "substrate_pairs_collection",
        "plan": plan,
        "episodes_total": len(episodes),
        "episodes_kept_strict": len(kept),
        "pairs": int(act.shape[0]),
        "npz_sha256": sha,
        "timestamp": time.time(),
    }
    rpath = Path(plan["out"]).with_suffix(".receipt.json")
    rpath.write_text(json.dumps(receipt, indent=1))
    print(f"[collect_substrate_pairs] {plan['level']}: "
          f"{len(kept)}/{len(episodes)} strict successes, "
          f"{act.shape[0]} pairs -> {plan['out']} (sha {sha[:16]}...)",
          flush=True)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="validate specs + print plans, no emulator")
    ap.add_argument("--level", action="append", choices=LEVELS,
                    help="collect only this level (repeatable); "
                         "default both")
    args = ap.parse_args(argv)
    levels = tuple(args.level) if args.level else LEVELS

    plans = [build_plan(lvl) for lvl in levels]
    missing = [p[k] for p in plans for k in
               ("profile", "checkpoint", "start_state")
               if not Path(p[k]).exists()]
    if missing:
        print(f"[collect_substrate_pairs] MISSING inputs: {missing}",
              file=sys.stderr)
        return 1
    for p in plans:
        print(json.dumps(p, indent=1))
    if args.dry_run:
        print("[collect_substrate_pairs] --dry-run: specs validated, "
              "plans printed, no emulator touched.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = OUT_DIR / ".run.lock"
    holder = acquire(lock, extra="collect_substrate_pairs")
    if holder is not None:
        print(f"[collect_substrate_pairs] lock held by live PID "
              f"{holder.pid} — refusing a duplicate collection.",
              file=sys.stderr)
        return 75
    try:
        pool_holder = acquire_resource(POOL_RESOURCE,
                                       extra="collect_substrate_pairs")
        if pool_holder is not None:
            print(f"[collect_substrate_pairs] {POOL_RESOURCE} lock held "
                  f"by live PID {pool_holder.pid} — refusing to step the "
                  "pool while another writer holds it.", file=sys.stderr)
            return 75
        try:
            for p in plans:
                collect_level(p)
        finally:
            release_resource(POOL_RESOURCE)
    finally:
        release(lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
