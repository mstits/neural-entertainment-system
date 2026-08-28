#!/usr/bin/env python3
"""Phase M adjudication — V31_REDO_SURGICAL_2026-08-27.md §5.2/§10 A2.

Reads a 60-iteration Phase M run.log (exactly tau=0.10, seed 0, 64/32,
--no-resume --no-supervise --strict-config) and renders the GO/NO-GO
verdict on all five conditions. Any failure routes to the §9 ladder
(escalate to tau=0.125 on M1/M4 failure, de-escalate to tau=0.075 on
M2/M3 failure, STOP outright on M5 failure) rather than starting the
7-hour treatment campaign on a knife-edge dose that was never measured.
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"\[redo\] iter (\d+): dormant fc1 (\d+)/(\d+) fc2 (\d+)/(\d+) "
    r"recycled (\d+) cum (\d+) agree ([0-9.]+) max_dlogit ([0-9.eE+-]+)"
)
INDICES_RE = re.compile(
    r"\[redo\] recycled unit indices: fc1=(\[[^\]]*\]) fc2=(\[[^\]]*\])"
)
OVERDOSE_RE = re.compile(r"VOID-OVERDOSE")


def adjudicate(log_path: Path, *, ceiling: float = 0.25) -> dict:
    checks = []  # (iter, dormant_fc2, trunk, recycled)
    fc2_index_counts: dict[int, int] = {}
    tripped = False
    text = log_path.read_text(errors="replace")
    if OVERDOSE_RE.search(text):
        tripped = True
    for line in text.splitlines():
        if (m := ITER_RE.search(line)) is not None:
            it = int(m.group(1))
            d2, trunk = int(m.group(4)), int(m.group(5))
            recycled = int(m.group(6))
            checks.append((it, d2, trunk, recycled))
        elif (m := INDICES_RE.search(line)) is not None:
            try:
                fc2_list = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                fc2_list = []
            for idx in fc2_list:
                fc2_index_counts[idx] = fc2_index_counts.get(idx, 0) + 1

    n_checks = len(checks)
    firing = [c for c in checks if c[3] >= 1]
    m1 = len(firing) >= 1
    m2 = not tripped
    win_a = [d2 for (it, d2, trunk, r) in checks if 26 <= it <= 30]
    win_b = [d2 for (it, d2, trunk, r) in checks if 56 <= it <= 60]
    drift = None
    m3 = True
    if win_a and win_b:
        drift = statistics.median(win_b) - statistics.median(win_a)
        m3 = drift <= 2.0
    m4 = len(firing) >= 8
    distinct = len(fc2_index_counts)
    max_share = (
        max(fc2_index_counts.values()) / sum(fc2_index_counts.values())
        if fc2_index_counts else 0.0
    )
    m5 = distinct >= 4 and max_share <= 0.60

    trunk_dims = {trunk for (_, _, trunk, _) in checks}
    equilibrium_frac = None
    if len(firing) >= 5:
        tail_fracs = [d2 / trunk for (_, d2, trunk, _) in firing[-10:]]
        equilibrium_frac = statistics.median(tail_fracs)
    law_prediction_corroborated = (
        equilibrium_frac is not None and 0.20 <= equilibrium_frac <= 0.30
    )

    go = m1 and m2 and m3 and m4 and m5
    return {
        "log": str(log_path),
        "n_checks": n_checks,
        "n_firing": len(firing),
        "trunk_dims_seen": sorted(trunk_dims),
        "ceiling_tripped": tripped,
        "M1_fires": m1,
        "M2_under_ceiling": m2,
        "M3_drift_units": drift,
        "M3_not_drifting": m3,
        "M4_ge_8_firing": m4,
        "M5_distinct_fc2": distinct,
        "M5_max_index_share": max_share,
        "M5_healthy": m5,
        "equilibrium_frac_tail10": equilibrium_frac,
        "eq_law_prediction_0.20_0.30_corroborated": law_prediction_corroborated,
        "verdict": "GO" if go else "NO-GO",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("log", type=Path)
    args = ap.parse_args(argv)
    result = adjudicate(args.log)
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0 if result["verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
