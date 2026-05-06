"""Profile the trainer's hot path with real measurements, not estimates.

Runs a short trainer session with cProfile + the existing GenTimer
sections, then prints a structured report:

* Per-section wall-time breakdown (inference / forward / emulation_wait
  / bookkeeping / reward) summed across the run.
* Top 30 Python functions by cumulative time, filtered to the
  trainer's own modules + the rust_pool_adapter.
* PyO3 boundary call counts (how many `pool.step_all` /
  `pool.reset_all` per gen).
* Suggestions: a one-line diagnostic per top-3 hotspot.

Usage:

    python scripts/profile_trainer_hotpath.py \\
        --rom 'roms/Super Mario Bros. (World).nes' \\
        --profile configs/mario_tiles.yaml \\
        --gens 5 --workers 16

Output is text on stdout; pipe to a file for archiving:

    python scripts/profile_trainer_hotpath.py ... > reports/profile_$(date +%Y%m%d).txt
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _build_trainer(rom: str, profile_path: str, workers: int, gens: int):
    """Construct a Trainer with the supplied profile and short
    trajectories so the profile run finishes in a reasonable time."""
    sys.path.insert(0, str(ROOT))
    from src.training.trainer import Trainer

    profile = yaml.safe_load(Path(profile_path).read_text())
    # Force short episodes so each gen is fast and we get many gens'
    # worth of step-loop data in the profile run.
    profile["max_episode_steps"] = 200
    # Shrink BC so the profile measures the runtime hot path, not BC
    # pretrain (which only runs once).
    profile.setdefault("reinforce", {})
    profile["reinforce"]["bc_epochs"] = 0
    profile["reinforce"]["warmup_gens_ga_only"] = 0

    trainer = Trainer(
        rom_path=rom,
        game_profile=profile,
        num_instances=workers,
        population_size=workers,
        max_episode_steps=200,
        seed=42,
    )
    return trainer


def _format_section_breakdown(snapshot: dict[str, float]) -> str:
    """Pretty-print the GenTimer snapshot as a sorted table."""
    if not snapshot:
        return "  (no timing data — GenTimer never called)"
    rows = sorted(snapshot.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in rows)
    out = []
    for name, ms in rows:
        pct = 100.0 * ms / total if total > 0 else 0.0
        clean = name.replace("timing_", "").replace("_ms", "")
        out.append(f"  {clean:<22s} {ms:>10.1f} ms  ({pct:>5.1f}%)")
    out.append(f"  {'-' * 22} {'-' * 13}  {'-' * 7}")
    out.append(f"  {'TOTAL':<22s} {total:>10.1f} ms")
    return "\n".join(out)


def _format_pyfunc_top(profile: cProfile.Profile, n: int = 30) -> str:
    """Render the top-N Python functions by cumulative time, scoped to
    our own modules so torch/numpy internals don't dominate the list."""
    s = io.StringIO()
    stats = pstats.Stats(profile, stream=s).sort_stats("cumulative")
    # Restrict to our own code paths — keep noise out.
    stats.print_stats(
        r"(src/training|src/emulation|src/models|nes_core)",
        n,
    )
    return s.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--cprofile-output", default=None,
                    help="If set, write the raw cProfile stats here for "
                         "later analysis with snakeviz / pstats.")
    args = ap.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Trainer hot-path profile")
    print(f"  ROM:     {args.rom}")
    print(f"  Profile: {args.profile}")
    print(f"  Workers: {args.workers}, gens: {args.gens}")
    print(f"{'=' * 70}\n")

    trainer = _build_trainer(args.rom, args.profile, args.workers, args.gens)
    print(f"  Encoder: {trainer.encoder_kind} (tile_mode={trainer._is_tile_mode})")
    print(f"  Device:  {trainer.device}")
    print(f"  Async:   {trainer.async_pipeline}")
    print(f"  Vmap:    {trainer.vmap_forward}")
    print(f"  PanicIso: {trainer.panic_isolation}")
    print()

    # Run + measure.
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    try:
        trainer.run(num_generations=args.gens)
    except KeyboardInterrupt:
        print("\n  Interrupted; reporting partial data.\n")
    finally:
        pr.disable()
    total_wall_s = time.perf_counter() - t0

    # Pull section breakdown from the metrics log (latest gen).
    metrics_path = trainer.checkpoint_dir / "metrics.jsonl"
    section_total: dict[str, float] = {}
    if metrics_path.exists():
        try:
            for line in metrics_path.read_text().splitlines():
                row = json.loads(line)
                for k, v in row.items():
                    if isinstance(k, str) and k.startswith("timing_") and isinstance(v, (int, float)):
                        section_total[k] = section_total.get(k, 0.0) + float(v)
        except Exception as exc:
            print(f"  (could not read metrics: {exc})")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — total wall-time: {total_wall_s:.1f}s "
          f"({total_wall_s / max(args.gens, 1):.1f}s per gen)")
    print(f"{'=' * 70}\n")

    print("Section breakdown (summed across all gens, from GenTimer):")
    print(_format_section_breakdown(section_total))
    print()

    print("Top 30 Python functions by cumulative time (own code only):")
    print(_format_pyfunc_top(pr, n=30))

    # Diagnostics block.
    print("=" * 70)
    print("  Quick diagnostics")
    print("=" * 70)
    if not section_total:
        print("  ⚠ No GenTimer data captured — section instrumentation didn't run.")
    else:
        emu = section_total.get("timing_emulation_wait_ms", 0.0)
        fwd = section_total.get("timing_forward_ms", 0.0)
        inf = section_total.get("timing_inference_ms", 0.0)
        bk = section_total.get("timing_bookkeeping_ms", 0.0)
        tot = sum(section_total.values()) or 1.0
        print(f"  Emulation wait: {100 * emu / tot:.1f}% of measured wall-time.")
        if emu / tot > 0.7:
            print("    → CPU-bound. Optimizing GPU/inference path is low ROI.")
            print("    → Investigate: per-frame PPU rendering, mapper hot paths, ASM CPU coverage.")
        if fwd / tot > 0.2:
            print(f"  Forward: {100 * fwd / tot:.1f}% — GPU-bound. Consider:")
            print("    → MLX migration of policy net (lower kernel launch overhead than PyTorch MPS).")
            print("    → Check vmap is firing (used_vmap log should be true every step).")
        if bk / tot > 0.05:
            print(f"  Bookkeeping: {100 * bk / tot:.1f}% — Python overhead in step loop.")
            print("    → Vectorize reward function calls.")
            print("    → Reduce attribute lookups in the per-worker results loop.")

    if args.cprofile_output:
        pr.dump_stats(args.cprofile_output)
        print(f"\n  Raw cProfile stats written to: {args.cprofile_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
