#!/usr/bin/env python3
"""Honest eval ladder driver — V31_REDO_SURGICAL_2026-08-27.md §6/§8.

Runs `scripts/eval_game.py` once per (seed, checkpoint iter, eval seed)
under the IMMUTABLE honest protocol (cold entrance, greedy, sticky 0.25,
jitter +-16, 50 episodes, per-episode RNG, max_steps 1500), saves each
receipt as
``<out-dir>/<profile-slug>_seed<S>_it<III>_es<E>.json`` — the naming
convention `scripts/score_cross_fit.py` reads — and skips any receipt
that already exists (so a partial ladder can be resumed without
re-spending eval time on seeds/iters already scored).

Runs the (seed, iter, eval_seed) grid in a bounded thread pool: each
eval_game.py subprocess is single-process CPU work, so this is
process-level parallelism (`--parallel N`), not the `--eval-workers`
flag (which controls INTRA-eval episode parallelism inside one
eval_game.py invocation and is passed through unchanged).
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.run_lock import acquire as _acquire_run_lock  # noqa: E402


def _checkpoint_path(seed_dir: Path, it: int) -> Path:
    return seed_dir / f"vanilla_ppo_iter_{it:05d}.pt"


def _receipt_path(out_dir: Path, prefix: str, seed: int, it: int, es: int) -> Path:
    return out_dir / f"{prefix}_seed{seed}_it{it:03d}_es{es}.json"


def ladder_lock_path(out_dir) -> Path:
    """Run-lock path for a ladder --out-dir (see src/utils/run_lock.py)."""
    return Path(out_dir) / ".run.lock"


def _run_one(
    *, profile: Path, checkpoint: Path, eval_seed: int, episodes: int,
    max_steps: int, eval_workers: int, receipt: Path,
) -> tuple[Path, bool, str]:
    if not checkpoint.is_file():
        return receipt, False, f"missing checkpoint: {checkpoint}"
    cmd = [
        sys.executable, "scripts/eval_game.py", "--game", "mario",
        "--profile", str(profile), "--checkpoint", str(checkpoint),
        "--episodes", str(episodes), "--max-steps", str(max_steps),
        "--action-select", "greedy", "--sticky-prob", "0.25",
        "--start-jitter", "16", "--eval-seed", str(eval_seed),
        "--eval-rng", "per-episode", "--eval-workers", str(eval_workers),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        return receipt, False, f"exit {proc.returncode}: {proc.stderr[-2000:]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return receipt, False, f"bad JSON stdout: {exc}: {proc.stdout[-500:]}"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(data, indent=2))
    return receipt, True, f"clear_rate={data.get('clear_rate')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--profile-template", required=True,
                     help="e.g. configs/mario_1_1_v31_redo_seed{seed}.yaml")
    ap.add_argument("--checkpoint-dir-template", required=True,
                     help="e.g. checkpoints/mario_1_1_v31_redo_seed{seed}")
    ap.add_argument("--receipt-prefix", required=True,
                     help="e.g. mario_1_1_v31_redo")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--iters", type=int, nargs="+",
                     default=list(range(10, 241, 10)))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--eval-workers", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=1,
                     help="concurrent eval_game.py subprocesses")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    jobs = []
    for seed in args.seeds:
        profile = REPO / args.profile_template.format(seed=seed)
        ckpt_dir = REPO / args.checkpoint_dir_template.format(seed=seed)
        for it in args.iters:
            checkpoint = _checkpoint_path(ckpt_dir, it)
            for es in args.eval_seeds:
                receipt = _receipt_path(
                    args.out_dir, args.receipt_prefix, seed, it, es,
                )
                if receipt.is_file():
                    continue
                jobs.append(dict(
                    profile=profile, checkpoint=checkpoint, eval_seed=es,
                    episodes=args.episodes, max_steps=args.max_steps,
                    eval_workers=args.eval_workers, receipt=receipt,
                ))

    print(f"[ladder] {len(jobs)} eval(s) queued "
          f"({len(args.seeds)} seeds x {len(args.iters)} iters x "
          f"{len(args.eval_seeds)} eval-seeds, minus already-scored)")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['receipt'].name}  <-  {j['checkpoint']}")
        return 0

    # Run-lock: two ladder drivers racing the same --out-dir can both
    # decide a receipt is missing and re-run it concurrently, and a
    # partial-write during that race corrupts the JSON `score_cross_fit.py`
    # reads next (the same defect class as the 2026-08-29 duplicate-
    # chain-watcher incident). A stale lock from a dead process is
    # reclaimed; a live one refuses. See src/utils/run_lock.py.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _lock = ladder_lock_path(args.out_dir)
    _holder = _acquire_run_lock(_lock, extra=args.receipt_prefix)
    if _holder is not None:
        print(f"[ladder] {args.out_dir} is locked by live PID "
              f"{_holder.pid} ({_lock}). Refusing to run two ladder "
              "drivers on one --out-dir.", file=sys.stderr)
        return 75
    atexit.register(lambda: _lock.exists() and _lock.unlink())

    n_ok = n_fail = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = {ex.submit(_run_one, **j): j for j in jobs}
        for fut in cf.as_completed(futs):
            receipt, ok, msg = fut.result()
            tag = "OK  " if ok else "FAIL"
            print(f"[ladder] {tag} {receipt.name}: {msg}")
            n_ok += ok
            n_fail += not ok

    print(f"[ladder] done: {n_ok} ok, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
