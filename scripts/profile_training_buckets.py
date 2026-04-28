"""Profile training wall clock from a metrics.jsonl run.

Reads the per-generation timing buckets the trainer emits
(`timing_emulation_wait_ms`, `timing_forward_ms`, `timing_inference_ms`,
`timing_bookkeeping_ms`) and reports:

1. Mean / p50 / p95 wall clock per generation
2. Bucket breakdown (which phase dominates)
3. Convergence trajectory (best fitness + success rate)
4. Wall-clock projection to N generations at observed rate

This is the "where does the time actually go" data the perf decision
should be made on, before committing to multi-week emulation work.

Usage:
    python scripts/profile_training_buckets.py [path/to/metrics.jsonl]
"""
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path
from typing import Any


def percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt_secs(ms: float) -> str:
    s = ms / 1000.0
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60.0
    if m < 60:
        return f"{m:.1f}m"
    h = m / 60.0
    if h < 24:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/metrics.jsonl")
    if not path.exists():
        print(f"FATAL: {path} not found", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print("no rows in metrics.jsonl")
        return 1

    n = len(rows)
    print(f"profile: {path}  (n={n} generations)")
    print()

    buckets = ("emulation_wait", "forward", "inference", "bookkeeping")
    bucket_data: dict[str, list[float]] = {b: [] for b in buckets}
    totals: list[float] = []
    for r in rows:
        per_gen = 0.0
        for b in buckets:
            v = float(r.get(f"timing_{b}_ms", 0.0))
            bucket_data[b].append(v)
            per_gen += v
        totals.append(per_gen)

    print(f"{'bucket':<18} {'mean':>10} {'p50':>10} {'p95':>10} {'%share':>10}")
    print("-" * 60)
    grand_mean = statistics.mean(totals)
    for b in buckets:
        xs = bucket_data[b]
        m = statistics.mean(xs)
        p50 = percentile(xs, 0.50)
        p95 = percentile(xs, 0.95)
        share = (m / grand_mean * 100.0) if grand_mean > 0 else 0.0
        print(f"{b:<18} {m:>9.1f}ms {p50:>9.1f}ms {p95:>9.1f}ms {share:>9.1f}%")
    print("-" * 60)
    print(
        f"{'TOTAL/gen':<18} "
        f"{grand_mean:>9.1f}ms "
        f"{percentile(totals, 0.50):>9.1f}ms "
        f"{percentile(totals, 0.95):>9.1f}ms"
    )

    print()
    print(f"convergence trajectory:")
    print(f"  gen 0  → best_fitness={rows[0].get('best_fitness', 0):.0f}  "
          f"success_rate={rows[0].get('success_rate', 0):.2%}")
    print(f"  gen {n-1} → best_fitness={rows[-1].get('best_fitness', 0):.0f}  "
          f"success_rate={rows[-1].get('success_rate', 0):.2%}")
    fits = [r.get("best_fitness", 0) for r in rows]
    print(f"  best ever: {max(fits):.0f}  (gen {fits.index(max(fits))})")

    print()
    print(f"wall-clock projection at {grand_mean:.0f}ms/gen:")
    for target in (100, 1000, 10_000, 100_000):
        wall = grand_mean * target
        print(f"  {target:>6} generations → {fmt_secs(wall):>8}")

    print()
    print("dominant bucket says you should attack:")
    means = [(b, statistics.mean(bucket_data[b])) for b in buckets]
    means.sort(key=lambda x: -x[1])
    for b, m in means[:2]:
        share = m / grand_mean * 100.0
        print(f"  {b}: {m:.0f}ms/gen ({share:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
