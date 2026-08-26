"""BubbleUp-lite: a generic cohort differ over metrics.jsonl.

WHY THIS EXISTS. scoreboard.py, campaign_report.py and engine_status.py
each join run_manifest.json + metrics.jsonl + eval.jsonl and compute one
pre-decided thing. None of them let you ask a NEW question without
writing new Python, and none of them answer "what's different about the
generations where clear_rate dropped vs the ones where it didn't" across
every field a trainer happens to emit -- a human has to think of the
dimension (entropy? samples_per_sec? reward_time_penalty?) before they
can even check it.

This tool inverts that: give it a cohort split (any `field OP value`
predicate over the rows) and it computes, for EVERY numeric field it can
find across BOTH cohorts, how different that field's distribution looks
between them -- then prints the fields ranked by how divergent they are.
The field nobody thought to ask about is the one that surfaces at the
top.

Divergence metric: an effect size in the spirit of Cohen's d --
`(mean_A - mean_B) / pooled_std` -- rather than a raw mean delta, because
raw deltas are useless for ranking across fields on wildly different
scales (`ppo_entropy` lives in [0, 2], `vanilla_ppo_max_x` lives in the
thousands). Pooled std here is `sqrt((std_A^2 + std_B^2) / 2)`: a simple
average of the two cohorts' variances, not the sample-size-weighted
pooled variance from a t-test, because we're ranking fields for
attention, not running a hypothesis test. Two edge cases fall out of
that division:

  * Both cohorts constant at the same value -> zero variance, zero
    delta -> effect size 0 (nothing to see).
  * Both cohorts constant but at DIFFERENT values (a step function) ->
    zero variance, nonzero delta -> effect size +-inf. This is a real,
    maximally-strong signal (the field perfectly separates the cohorts)
    and is sorted to the very top rather than treated as an error.

A field present in only one cohort (this repo just added
`schema_version`; older runs never wrote it) is the same kind of
maximal-signal case and is also ranked at +-inf, tagged "A-only" /
"B-only" so it reads as "this field appeared/disappeared" rather than a
numeric shift.

Per the audit: rows from different dates carry different field sets, so
a field missing from some rows must never crash the tool -- every ranked
field reports `n/total` coverage per cohort alongside its stats, so a
stat computed on 12 of 250 rows is visibly not the same confidence as
one computed on 250 of 250.

Usage:
    python scripts/analyze.py --metrics checkpoints/foo/metrics.jsonl \\
        --split "generation>100" --top 15

    python scripts/analyze.py --run checkpoints/foo \\
        --split "clear_rate<0.1"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

Row = dict[str, Any]

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}
# Longest operators first so ">=" isn't chopped into ">" + "=".
_SPLIT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$"
)


# ---------------------------------------------------------------------------
# Loading (tolerant of the real-world mess the audit flagged)
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[Row]:
    """Read one JSON object per line, skipping blank/corrupt lines.

    A truncated write (process killed mid-flush) or a hand-edited line
    must not take down the whole analysis -- skip it and keep going,
    same contract as scoreboard.py's _read_jsonl.
    """
    if not path.exists():
        return []
    out: list[Row] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def resolve_metrics_path(metrics: Path | None, run: Path | None) -> Path:
    if metrics is not None:
        return metrics
    assert run is not None
    return run / "metrics.jsonl"


# ---------------------------------------------------------------------------
# Split predicate: "field>100", "clear_rate<0.1", "stage==default"
# ---------------------------------------------------------------------------


def parse_split(expr: str) -> tuple[str, str, Any]:
    """Parse a `field OP value` predicate string.

    Returns (field, op_symbol, value). `value` is coerced to float when
    it parses as one (the common case: thresholds on numeric fields);
    otherwise it's kept as a string so `--split "stage==default"` also
    works against categorical fields.
    """
    m = _SPLIT_RE.match(expr)
    if not m:
        raise ValueError(
            f"can't parse --split {expr!r}; expected FIELD OP VALUE, "
            f'e.g. "generation>100" or "clear_rate<0.1" '
            f"(op is one of {', '.join(_OPS)})"
        )
    field, op_symbol, raw_value = m.groups()
    try:
        value: Any = float(raw_value)
    except ValueError:
        value = raw_value.strip("\"'")
    return field, op_symbol, value


def split_rows(
    rows: list[Row], field: str, op_symbol: str, value: Any
) -> tuple[list[Row], list[Row], int]:
    """Partition rows into (matches predicate, doesn't) by `field OP value`.

    Rows missing `field` entirely are excluded from both cohorts rather
    than dumped into "doesn't match" -- a row that never recorded
    `clear_rate` isn't evidence about low clear rate, it's just missing
    data, and silently folding it into one side would bias that cohort.
    """
    op = _OPS[op_symbol]
    cohort_a: list[Row] = []
    cohort_b: list[Row] = []
    excluded = 0
    for row in rows:
        if field not in row or row[field] is None:
            excluded += 1
            continue
        row_value = row[field]
        try:
            if isinstance(value, (int, float)) and not isinstance(row_value, bool):
                matched = op(float(row_value), value)
            else:
                matched = op(row_value, value)
        except (TypeError, ValueError):
            excluded += 1
            continue
        (cohort_a if matched else cohort_b).append(row)
    return cohort_a, cohort_b, excluded


# ---------------------------------------------------------------------------
# Divergence: every numeric field, ranked by normalized effect size
# ---------------------------------------------------------------------------


def _is_number(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    )


@dataclass
class FieldDivergence:
    field: str
    n_a: int
    total_a: int
    n_b: int
    total_b: int
    mean_a: float | None
    mean_b: float | None
    median_a: float | None
    median_b: float | None
    delta_mean: float | None
    delta_median: float | None
    effect_size: float
    note: str = ""


def numeric_values(rows: list[Row], field: str) -> list[float]:
    return [float(r[field]) for r in rows if field in r and _is_number(r[field])]


def diverge(cohort_a: list[Row], cohort_b: list[Row]) -> list[FieldDivergence]:
    """Compute per-field divergence between two cohorts, unranked."""
    fields: set[str] = set()
    for row in cohort_a:
        fields.update(row.keys())
    for row in cohort_b:
        fields.update(row.keys())

    total_a, total_b = len(cohort_a), len(cohort_b)
    results: list[FieldDivergence] = []

    for field in sorted(fields):
        vals_a = numeric_values(cohort_a, field)
        vals_b = numeric_values(cohort_b, field)
        n_a, n_b = len(vals_a), len(vals_b)

        if n_a == 0 and n_b == 0:
            continue  # never a numeric field in either cohort -- not ours to rank

        if n_a == 0 or n_b == 0:
            # Present in exactly one cohort: the strongest possible
            # divergence signal (the field appeared/disappeared), e.g.
            # schema_version on runs that predate it.
            results.append(
                FieldDivergence(
                    field=field,
                    n_a=n_a,
                    total_a=total_a,
                    n_b=n_b,
                    total_b=total_b,
                    mean_a=float(np.mean(vals_a)) if vals_a else None,
                    mean_b=float(np.mean(vals_b)) if vals_b else None,
                    median_a=float(np.median(vals_a)) if vals_a else None,
                    median_b=float(np.median(vals_b)) if vals_b else None,
                    delta_mean=None,
                    delta_median=None,
                    effect_size=math.inf,
                    note="B-only" if n_a == 0 else "A-only",
                )
            )
            continue

        mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
        median_a, median_b = float(np.median(vals_a)), float(np.median(vals_b))
        std_a, std_b = float(np.std(vals_a)), float(np.std(vals_b))
        delta_mean = mean_a - mean_b
        delta_median = median_a - median_b
        pooled_std = math.sqrt((std_a**2 + std_b**2) / 2.0)

        if pooled_std > 0:
            effect_size = delta_mean / pooled_std
        elif delta_mean == 0:
            effect_size = 0.0
        else:
            # Both cohorts are constant but at different values: a
            # perfect step function, the strongest finite-data signal
            # there is.
            effect_size = math.inf if delta_mean > 0 else -math.inf

        results.append(
            FieldDivergence(
                field=field,
                n_a=n_a,
                total_a=total_a,
                n_b=n_b,
                total_b=total_b,
                mean_a=mean_a,
                mean_b=mean_b,
                median_a=median_a,
                median_b=median_b,
                delta_mean=delta_mean,
                delta_median=delta_median,
                effect_size=effect_size,
            )
        )

    return results


def rank(results: list[FieldDivergence], top: int) -> list[FieldDivergence]:
    return sorted(results, key=lambda r: abs(r.effect_size), reverse=True)[:top]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v: float | None, spec: str = ".4g", dash: str = "-") -> str:
    if v is None:
        return dash
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)


def render(
    metrics_path: Path,
    n_rows: int,
    field: str,
    op_symbol: str,
    value: Any,
    cohort_a: list[Row],
    cohort_b: list[Row],
    excluded: int,
    ranked: list[FieldDivergence],
    n_candidates: int,
    top: int,
) -> str:
    lines: list[str] = []
    lines.append(f"metrics: {metrics_path} ({n_rows} rows)")
    val_str = value if isinstance(value, str) else _fmt(value)
    lines.append(
        f"split: {field}{op_symbol}{val_str}  ->  "
        f"A (match)={len(cohort_a)} rows, B (rest)={len(cohort_b)} rows, "
        f"excluded (no {field})={excluded}"
    )
    if not cohort_a or not cohort_b:
        lines.append(
            "\nboth cohorts must be non-empty to compare distributions -- "
            "one side has 0 rows, nothing to rank."
        )
        return "\n".join(lines)

    lines.append(
        f"\nshowing top {min(top, n_candidates)} of {n_candidates} numeric "
        f"fields, ranked by |effect size| (Cohen's-d-style; pooled std "
        f"across both cohorts)\n"
    )

    header = (
        f"{'field':<32}{'n_A/tot':>10}{'n_B/tot':>10}"
        f"{'mean_A':>12}{'mean_B':>12}{'Δmean':>11}{'Δmedian':>11}"
        f"{'effect':>9}  note"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in ranked:
        lines.append(
            f"{r.field:<32}"
            f"{f'{r.n_a}/{r.total_a}':>10}"
            f"{f'{r.n_b}/{r.total_b}':>10}"
            f"{_fmt(r.mean_a):>12}"
            f"{_fmt(r.mean_b):>12}"
            f"{_fmt(r.delta_mean):>11}"
            f"{_fmt(r.delta_median):>11}"
            f"{_fmt(r.effect_size, '.3g'):>9}  {r.note}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics", type=Path, help="path to a metrics.jsonl file")
    src.add_argument(
        "--run", type=Path, help="run/checkpoint dir containing metrics.jsonl"
    )
    parser.add_argument(
        "--split",
        required=True,
        help='cohort predicate, e.g. "generation>100" or "clear_rate<0.1"',
    )
    parser.add_argument(
        "--top", type=int, default=20, help="max fields to print (default 20)"
    )
    args = parser.parse_args(argv)

    metrics_path = resolve_metrics_path(args.metrics, args.run)
    if not metrics_path.exists():
        print(f"analyze: no such file: {metrics_path}", file=sys.stderr)
        return 1

    try:
        field, op_symbol, value = parse_split(args.split)
    except ValueError as exc:
        print(f"analyze: {exc}", file=sys.stderr)
        return 2

    rows = read_jsonl(metrics_path)
    if not rows:
        print(f"analyze: no rows read from {metrics_path}", file=sys.stderr)
        return 1

    cohort_a, cohort_b, excluded = split_rows(rows, field, op_symbol, value)
    results = diverge(cohort_a, cohort_b) if cohort_a and cohort_b else []
    ranked = rank(results, args.top)

    print(
        render(
            metrics_path,
            len(rows),
            field,
            op_symbol,
            value,
            cohort_a,
            cohort_b,
            excluded,
            ranked,
            len(results),
            args.top,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
