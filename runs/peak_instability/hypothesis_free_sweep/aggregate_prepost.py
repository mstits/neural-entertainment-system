"""Cross-run aggregation of analyze.py's per-field divergence, pre-peak vs post-peak.

For each of the 8 runs, split metrics.jsonl at the authoritative peak iter
(from checkpoints/<run>/winners/best.json, source_iter) using the exact same
split_rows/diverge functions scripts/analyze.py uses (cohort A = generation>peak,
i.e. post-peak; cohort B = generation<=peak, i.e. pre-peak inclusive of the peak
row itself). Then, for every numeric field, look across all 8 runs: how many
agree on the sign of the effect, and what is the magnitude.

This does NOT replace analyze.py -- it drives it (same functions, same cohort
split rule) eight times and aggregates what a human would otherwise have to do
by eyeballing eight printed tables side by side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import analyze  # noqa: E402

RUNS = [
    "mario_1_1_v27_recovery_seed0",
    "mario_1_1_v27_recovery_seed1",
    "mario_1_1_v27_recovery_seed2",
    "mario_1_1_v27_recovery_seed3",
    "mario_1_1_v28_capacity_seed0",
    "mario_1_1_v28_capacity_seed1",
    "mario_1_1_v28_capacity_seed2",
    "mario_1_1_v28_capacity_seed3",
]

TRIVIAL_FIELDS = {"generation", "timestamp"}


def peak_iter(run: str) -> int:
    d = json.loads((REPO / "checkpoints" / run / "winners" / "best.json").read_text())
    return int(d["source_iter"])


def main() -> None:
    per_run: dict[str, dict[str, analyze.FieldDivergence]] = {}
    peaks: dict[str, int] = {}
    for run in RUNS:
        peak = peak_iter(run)
        peaks[run] = peak
        rows = analyze.read_jsonl(REPO / "checkpoints" / run / "metrics.jsonl")
        cohort_a, cohort_b, excluded = analyze.split_rows(rows, "generation", ">", float(peak))
        results = analyze.diverge(cohort_a, cohort_b)
        per_run[run] = {r.field: r for r in results}
        print(f"{run}: peak_iter={peak} post_n={len(cohort_a)} pre_n={len(cohort_b)} excluded={excluded}", file=sys.stderr)

    all_fields = sorted(set().union(*[set(d.keys()) for d in per_run.values()]))

    # Build cross-run table
    print("\n" + "=" * 100)
    print("CROSS-RUN FIELD DIVERGENCE (post-peak vs pre-peak+peak), all 8 runs")
    print("=" * 100)
    header = f"{'field':<30}{'sign_agree':>11}{'dir':>6}{'median|d|':>11}{'min d':>9}{'max d':>9}  per-run d (v27s0 v27s1 v27s2 v27s3 v28s0 v28s1 v28s2 v28s3)"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for field in all_fields:
        effects = []
        missing_runs = []
        for run in RUNS:
            fd = per_run[run].get(field)
            if fd is None:
                missing_runs.append(run)
                effects.append(float("nan"))
                continue
            effects.append(fd.effect_size)
        finite = [e for e in effects if e == e and abs(e) != float("inf")]
        inf_pos = sum(1 for e in effects if e == float("inf"))
        inf_neg = sum(1 for e in effects if e == float("-inf"))
        nonzero = [e for e in effects if e == e and e != 0.0]
        neg = sum(1 for e in nonzero if e < 0)
        pos = sum(1 for e in nonzero if e > 0)
        n_signed = neg + pos
        agree = max(neg, pos)
        direction = "post<pre" if neg >= pos else "post>pre"
        if n_signed == 0:
            direction = "flat"
        mags = [abs(e) for e in effects if e == e and abs(e) != float("inf")]
        median_mag = sorted(mags)[len(mags) // 2] if mags else float("nan")
        finite_or_inf = [e if e == e else float("nan") for e in effects]
        min_d = min([e for e in finite_or_inf if e == e], default=float("nan"))
        max_d = max([e for e in finite_or_inf if e == e], default=float("nan"))
        per_run_str = " ".join(
            ("  nan" if e != e else ("  inf" if e == float("inf") else (" -inf" if e == float("-inf") else f"{e:5.2f}")))
            for e in effects
        )
        summary_rows.append((field, agree, n_signed, direction, median_mag, min_d, max_d, per_run_str, missing_runs))

    # rank by (agreement out of 8, then median magnitude) descending, trivial fields flagged but kept for completeness
    def sort_key(row):
        field, agree, n_signed, direction, median_mag, *_ = row
        mm = median_mag if median_mag == median_mag else -1
        return (agree, mm)

    summary_rows.sort(key=sort_key, reverse=True)

    for field, agree, n_signed, direction, median_mag, min_d, max_d, per_run_str, missing_runs in summary_rows:
        flag = " [TRIVIAL]" if field in TRIVIAL_FIELDS else ""
        miss = f"  MISSING:{missing_runs}" if missing_runs else ""
        print(f"{field:<30}{agree:>6}/{n_signed:<4}{direction:>9}{median_mag:>11.3f}{min_d:>9.2f}{max_d:>9.2f}  {per_run_str}{flag}{miss}")

    print("\npeak iters used:", peaks)


if __name__ == "__main__":
    main()
