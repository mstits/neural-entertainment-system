#!/usr/bin/env python3
"""
Save/restore microbenchmark — measure the true cost of a full NES
state snapshot + restore.

The README/CLAIMS describe "microsecond save/restore at scale". This
script measures it end-to-end and reports mean/p50/p99 *nanoseconds*
per save and per restore, so the claim can be confirmed or corrected
with real numbers.

What is timed
-------------
A save is `Pool.save_worker_state(worker_id)`: bincode-serialize the
full `nes::State` (RAM + CPU + PPU + APU + mapper + input regs) and
hand Python an opaque `bytes` blob (~21 KB for SMB). A restore is
`Pool.load_worker_state(worker_id, blob)`: header-check, deserialize,
install. These are the exact calls the auto-curriculum snapshot path
and Go-Explore cell restore use.

Two phases are run:
  1. single-worker pool  — the cost of one snapshot in isolation
  2. N-worker pool        — the same op measured round-robin across a
                            full pool, i.e. "at scale"

A correctness gate runs first: save -> mutate (step) -> restore ->
step, and confirms the restored RAM matches the pre-mutation baseline.
A benchmark that times a no-op restore would be worthless, so this
proves the op actually does work before any timing is reported.

Usage:
    .venv/bin/python scripts/bench_save_restore.py
    .venv/bin/python scripts/bench_save_restore.py --iters 20000 --scale-workers 16
    .venv/bin/python scripts/bench_save_restore.py --out runs/emulator_bench_2026-07-20.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nes_core

DEFAULT_ROM = _REPO / "roms" / "Super Mario Bros. (World).nes"
DEFAULT_STATE = _REPO / "roms" / "Super Mario Bros. (World)_start.state.bin"


def _percentiles(samples_ns: np.ndarray) -> dict:
    """mean / p50 / p99 (and min/max) in nanoseconds for a 1-D array."""
    return {
        "mean_ns": float(np.mean(samples_ns)),
        "p50_ns": float(np.percentile(samples_ns, 50)),
        "p99_ns": float(np.percentile(samples_ns, 99)),
        "min_ns": float(np.min(samples_ns)),
        "max_ns": float(np.max(samples_ns)),
        "n": int(samples_ns.size),
    }


def _fmt(stats: dict) -> str:
    us = 1000.0
    return (
        f"mean {stats['mean_ns'] / us:8.3f} us   "
        f"p50 {stats['p50_ns'] / us:8.3f} us   "
        f"p99 {stats['p99_ns'] / us:8.3f} us   "
        f"(min {stats['min_ns'] / us:.3f} / max {stats['max_ns'] / us:.3f} us, "
        f"n={stats['n']})"
    )


def _correctness_gate(pool: nes_core.Pool, workers: int) -> dict:
    """Prove save/restore actually round-trips state before timing it.

    save -> step (RAM must change) -> restore -> step, then compare the
    post-restore RAM to the baseline captured right after save. Returns
    a small dict for the JSON so the gate result is recorded, not just
    asserted.
    """
    noop = np.zeros(workers, dtype=np.uint8)
    pool.reset_all()
    for _ in range(4):
        pool.step_all(noop)

    blob = bytes(pool.save_worker_state(0))
    # RAM snapshot at the saved point: step once, that step's result
    # carries the RAM the saved state produces on its first advance.
    baseline = bytes(pool.step_all(noop)[0][2])

    # Mutate away from the saved point.
    mutated_same = True
    for _ in range(30):
        r = pool.step_all(noop)
        if bytes(r[0][2]) != baseline:
            mutated_same = False
    ram_after_mutation = bytes(pool.step_all(noop)[0][2])

    # Restore and re-run the identical first advance.
    pool.load_worker_state(0, blob)
    restored = bytes(pool.step_all(noop)[0][2])

    round_trips = restored == baseline
    diverged = ram_after_mutation != baseline
    return {
        "blob_bytes": len(blob),
        "ram_round_trips_exactly": bool(round_trips),
        "state_actually_diverged_before_restore": bool(diverged),
        "passed": bool(round_trips and diverged),
    }


def _bench_phase(pool: nes_core.Pool, workers: int, iters: int,
                 warmup: int) -> dict:
    """Time `iters` saves then `iters` restores, round-robin over the
    pool's workers. Returns save/restore percentile dicts."""
    noop = np.zeros(workers, dtype=np.uint8)
    pool.reset_all()
    for _ in range(4):
        pool.step_all(noop)

    # A blob per worker so restores install a real, worker-appropriate
    # state (loading worker 3's blob into worker 3, etc.).
    blobs = [bytes(pool.save_worker_state(w)) for w in range(workers)]

    # ---- saves ----
    for i in range(warmup):
        pool.save_worker_state(i % workers)
    save_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        w = i % workers
        t0 = time.perf_counter_ns()
        pool.save_worker_state(w)
        save_ns[i] = time.perf_counter_ns() - t0

    # ---- restores ----
    for i in range(warmup):
        w = i % workers
        pool.load_worker_state(w, blobs[w])
    restore_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        w = i % workers
        blob = blobs[w]
        t0 = time.perf_counter_ns()
        pool.load_worker_state(w, blob)
        restore_ns[i] = time.perf_counter_ns() - t0

    return {
        "workers": workers,
        "iters": iters,
        "blob_bytes": len(blobs[0]),
        "save": _percentiles(save_ns),
        "restore": _percentiles(restore_ns),
    }


def _verdict(save_stats: dict, restore_stats: dict) -> str:
    worst = max(save_stats["p99_ns"], restore_stats["p99_ns"])
    if worst < 1_000:
        return "SUB-MICROSECOND (nanoseconds) — even at p99"
    if worst < 1_000_000:
        return "MICROSECONDS — confirmed (p99 under 1 ms)"
    return "MILLISECONDS — the 'microsecond' claim does NOT hold"


def run(rom: str, state: str, iters: int, warmup: int,
        scale_workers: int, frame_skip: int) -> dict:
    result = {
        "rom": rom,
        "start_state": state,
        "frame_skip": frame_skip,
        "iters_per_op": iters,
        "warmup": warmup,
    }

    print("== save/restore microbenchmark ==")
    print(f"  rom:        {rom}")
    print(f"  iters/op:   {iters}")
    print(f"  frame_skip: {frame_skip}")
    import os as _os
    try:
        l1, l5, l15 = _os.getloadavg()
        warn = "  <-- CONTENDED (tail latency inflated)" if l1 > (_os.cpu_count() or 1) else ""
        print(f"  loadavg:    {l1:.1f} {l5:.1f} {l15:.1f}{warn}")
    except (OSError, AttributeError):
        pass
    print()

    # ---- correctness gate (single worker) ----
    gate_pool = nes_core.Pool(rom_path=rom, num_workers=1,
                              frame_skip=frame_skip, start_state_path=state)
    gate_pool.set_headless(True)
    gate = _correctness_gate(gate_pool, 1)
    result["correctness_gate"] = gate
    print(f"  correctness gate: passed={gate['passed']} "
          f"(round-trips={gate['ram_round_trips_exactly']}, "
          f"diverged-before-restore={gate['state_actually_diverged_before_restore']}, "
          f"blob={gate['blob_bytes']} bytes)")
    if not gate["passed"]:
        print("  !! WARNING: save/restore did not round-trip — timings below "
              "may be timing a no-op.")
    print()

    # ---- phase 1: single-worker pool ----
    p1 = nes_core.Pool(rom_path=rom, num_workers=1, frame_skip=frame_skip,
                       start_state_path=state)
    p1.set_headless(True)
    single = _bench_phase(p1, 1, iters, warmup)
    single["verdict"] = _verdict(single["save"], single["restore"])
    result["single_worker"] = single
    print("  [single-worker pool]")
    print(f"    blob:    {single['blob_bytes']} bytes")
    print(f"    save:    {_fmt(single['save'])}")
    print(f"    restore: {_fmt(single['restore'])}")
    print(f"    verdict: {single['verdict']}")
    print()

    # ---- phase 2: N-worker pool ("at scale") ----
    scale = None
    if scale_workers and scale_workers > 1:
        pN = nes_core.Pool(rom_path=rom, num_workers=scale_workers,
                           frame_skip=frame_skip, start_state_path=state)
        pN.set_headless(True)
        scale = _bench_phase(pN, scale_workers, iters, warmup)
        scale["verdict"] = _verdict(scale["save"], scale["restore"])
        result["scale"] = scale
        print(f"  [{scale_workers}-worker pool — at scale]")
        print(f"    save:    {_fmt(scale['save'])}")
        print(f"    restore: {_fmt(scale['restore'])}")
        print(f"    verdict: {scale['verdict']}")
        print()

    return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _merge_out(out_path: Path, section_key: str, payload: dict) -> None:
    """Merge one section into the shared bench JSON, preserving other
    sections written by sibling bench scripts."""
    doc = {}
    if out_path.exists():
        try:
            doc = json.loads(out_path.read_text())
        except Exception:
            doc = {}
    import os as _os
    try:
        load1, load5, load15 = _os.getloadavg()
    except (OSError, AttributeError):
        load1 = load5 = load15 = None
    doc.setdefault("machine", {
        "cpu_count": _os.cpu_count(),
        "platform": sys.platform,
    })
    payload["loadavg_1_5_15"] = [load1, load5, load15]
    payload["contended"] = bool(load1 is not None
                                and load1 > (_os.cpu_count() or 1))
    doc["git_commit"] = _git_commit()
    doc[section_key] = payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"  wrote {section_key} -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--iters", type=int, default=20000,
                    help="save/restore iterations per op (>=10000)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--scale-workers", type=int, default=16,
                    help="worker count for the 'at scale' phase")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="merge results into this JSON file")
    args = ap.parse_args()

    for pth in (args.rom, args.state):
        if not Path(pth).exists():
            print(f"missing: {pth}", file=sys.stderr)
            return 1

    result = run(args.rom, args.state, args.iters, args.warmup,
                 args.scale_workers, args.frame_skip)

    if args.out:
        _merge_out(Path(args.out), "save_restore", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
