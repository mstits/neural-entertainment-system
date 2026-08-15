"""Death-cluster and clear-predicate audit over eval records.

Pure accounting over already-recorded eval artifacts (eval.jsonl rows and
final_eval*.json files) — no emulator, no rollouts.

Why this exists: hand-tallied cluster claims produced two receipt defects on
the 1-2 campaign — a "zero deaths at exactly 2979" claim contradicted by the
records (2 sticky deaths at 2979 + 1 at 2980 in the rung probes), and "clears"
counted as episodes reaching flag-x (47/150) when the records' own clear_rate
(episode_success) sums to 36/150. This tool computes each quantity from the
records and names the predicate on every number:

- deaths          = episodes with max_gx < flag_x (per-x histogram + bands)
- flag_reached    = episodes with max_gx >= flag_x (an x-threshold event,
                    NOT a registered clear)
- clears_episode_success = round(clear_rate * n)      (flag/castle latch)
- clears_seq             = round(seq_clear_rate * n)  (level-chain advance)
- flag_without_episode_success = flag_reached - clears_episode_success

Usage:
    python scripts/death_cluster_audit.py \
        --eval-jsonl checkpoints/mario_1_2_online_v2/eval.jsonl \
        --rows 9,11,13,15,17 \
        --extra-json runs/online_1_2/final_eval_seed7.json \
        --flag-x 3266 --band 2950:3010 --exact 2675,2979,2980 \
        --out /tmp/audit.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional


def _count_from_rate(rate: Optional[float], n: int) -> Optional[int]:
    """Recover an integer count from a stored rate; guards float-floor."""
    if rate is None:
        return None
    return int(round(rate * n))


def audit_record(rec: dict, flag_x: int) -> dict:
    """Per-record accounting: deaths, flag-x reachers, and every clear
    predicate the record carries — kept separate by name."""
    gx = rec.get("max_gx_per_episode") or rec.get("max_x_per_episode")
    if gx is None:
        raise ValueError("record has no max_gx_per_episode/max_x_per_episode")
    n = len(gx)
    deaths = [int(x) for x in gx if x < flag_x]
    flag_reached = n - len(deaths)
    clears_es = _count_from_rate(rec.get("clear_rate"), n)
    clears_seq = _count_from_rate(rec.get("seq_clear_rate"), n)
    return {
        "n": n,
        "deaths": deaths,
        "flag_reached": flag_reached,
        "clears_episode_success": clears_es,
        "clears_seq": clears_seq,
        "flag_without_episode_success": (
            None if clears_es is None else flag_reached - clears_es),
    }


def aggregate_audit(records: list[dict], flag_x: int,
                    bands: list[tuple[int, int]],
                    exact: Optional[list[int]] = None) -> dict:
    """Aggregate audit_record over records; bands are inclusive [lo, hi]."""
    per_record = [audit_record(r, flag_x) for r in records]
    hist: Counter = Counter()
    for a in per_record:
        hist.update(a["deaths"])
    band_out = {}
    for lo, hi in bands:
        in_band = {x: c for x, c in sorted(hist.items()) if lo <= x <= hi}
        band_out[f"{lo}-{hi}"] = {
            "count": sum(in_band.values()),
            "histogram": {str(x): c for x, c in in_band.items()},
        }
    def _sum_opt(key):
        vals = [a[key] for a in per_record]
        return None if any(v is None for v in vals) else sum(vals)
    agg = {
        "flag_x": flag_x,
        "n_records": len(records),
        "n_episodes": sum(a["n"] for a in per_record),
        "deaths_total": sum(len(a["deaths"]) for a in per_record),
        "death_histogram": {str(x): c for x, c in sorted(hist.items())},
        "bands": band_out,
        "flag_reached": sum(a["flag_reached"] for a in per_record),
        "clears_episode_success": _sum_opt("clears_episode_success"),
        "clears_seq": _sum_opt("clears_seq"),
        "flag_without_episode_success": _sum_opt(
            "flag_without_episode_success"),
        "per_record": [
            {k: v for k, v in a.items() if k != "deaths"}
            for a in per_record
        ],
    }
    if exact:
        agg["exact"] = {str(x): int(hist.get(x, 0)) for x in exact}
    return agg


# ------------------------------------------------------------------- CLI

def _parse_band(s: str) -> tuple[int, int]:
    lo, hi = s.split(":")
    return int(lo), int(hi)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--rows", default=None,
                    help="comma-separated 0-based row indices; default all")
    ap.add_argument("--extra-json", nargs="*", default=[],
                    help="additional single-record eval JSON files")
    ap.add_argument("--flag-x", type=int, required=True,
                    help="max_gx at/above which an episode reached flag-x")
    ap.add_argument("--band", action="append", default=[],
                    help="inclusive death band lo:hi (repeatable)")
    ap.add_argument("--exact", default=None,
                    help="comma-separated exact x values to report counts for")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            Path(args.eval_jsonl).read_text().splitlines() if line.strip()]
    if args.rows is not None:
        idx = [int(t) for t in args.rows.split(",") if t.strip()]
        bad = [i for i in idx if not 0 <= i < len(rows)]
        if bad:
            raise SystemExit(
                f"--rows {bad} out of range (file has {len(rows)} rows)")
        rows = [rows[i] for i in idx]
    for p in args.extra_json:
        rows.append(json.loads(Path(p).read_text()))

    exact = ([int(t) for t in args.exact.split(",")] if args.exact else None)
    agg = aggregate_audit(rows, flag_x=args.flag_x,
                          bands=[_parse_band(b) for b in args.band],
                          exact=exact)
    text = json.dumps(agg, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
