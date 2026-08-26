#!/usr/bin/env python3
"""Second-order check: once tau=0 (AT-ENTRANCE), the curriculum stops moving
-- every subsequent iter samples episodes from the SAME fixed start state.
If the policy's raw success rate at that fixed state ALSO peaks-then-decays
over the ~220 static iters that follow, distribution shift cannot be the
mechanism for that portion of the decay (there is no more shift to have) --
something else is degrading performance on a stationary task. If instead
the windowed entrance rate stays flat/monotonic-non-decreasing, the honest
in-eval collapse would be less attributable to "on-policy training signal
degraded" and more to an eval-protocol difference (greedy+sticky+jitter
during honest eval vs. exploratory sampling used to log entrance N/T).

`entrance N/T` in the log is CUMULATIVE since run start. This script
differences it iter-over-iter to get a per-iter (attempts, successes) pair,
then applies a rolling window to smooth the small per-iter counts into a
success-rate curve, and reports where that curve peaks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")

LINE_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): "
    r"tau=(?P<tau>\d+)/(?P<tau0>\d+) .*?"
    r"advances=(?P<advances>\d+)\s+"
    r"(?:(?P<at_entrance>AT-ENTRANCE)\s+)?\| "
    r"entrance (?P<ent_n>\d+)/(?P<ent_d>\d+)=(?P<ent_rate>[\d.]+) \| "
    r"truncated (?P<truncated>\d+)"
)

RUNS = [
    ("v27_seed0", REPO / "runs/v27_fresh_recovery/train_seed0.log", 60),
    ("v27_seed1", REPO / "runs/v27_fresh_recovery/train_seed1.log", 50),
    ("v27_seed2", REPO / "runs/v27_fresh_recovery/train_seed2.log", 90),
    ("v27_seed3", REPO / "runs/v27_fresh_recovery/train_seed3.log", 60),
    ("v28_seed0", REPO / "runs/v28_capacity/train_seed0.log", 70),
    ("v28_seed1", REPO / "runs/v28_capacity/train_seed1.log", 60),
    ("v28_seed2", REPO / "runs/v28_capacity/train_seed2.log", 120),
    ("v28_seed3", REPO / "runs/v28_capacity/train_seed3.log", 90),
]

WINDOW = 20  # iters; smooths noisy per-iter counts (attempts/iter ~70-90 post-entrance)


def parse(path: Path):
    rows = []
    for line in open(path):
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "iter": int(d["iter"]),
                "tau": int(d["tau"]),
                "ent_n": int(d["ent_n"]),
                "ent_d": int(d["ent_d"]),
                "at_entrance": d["at_entrance"] is not None,
            }
        )
    rows.sort(key=lambda r: r["iter"])
    return rows


def main():
    out = {}
    for name, path, peak_iter in RUNS:
        rows = parse(path)
        n_marg = [rows[0]["ent_n"]] + [
            b["ent_n"] - a["ent_n"] for a, b in zip(rows, rows[1:])
        ]
        d_marg = [rows[0]["ent_d"]] + [
            b["ent_d"] - a["ent_d"] for a, b in zip(rows, rows[1:])
        ]
        entrance_iter = next(r["iter"] for r in rows if r["tau"] == 0)

        # rolling window success rate, restricted to the AT-ENTRANCE tail
        # (pre-entrance the denominator is sparse/near-zero and noisy)
        tail_start = entrance_iter
        iters = [r["iter"] for r in rows if r["iter"] >= tail_start]
        n_tail = n_marg[tail_start:]
        d_tail = d_marg[tail_start:]

        win_iters, win_rate = [], []
        for i in range(0, len(iters) - WINDOW + 1):
            n_sum = sum(n_tail[i : i + WINDOW])
            d_sum = sum(d_tail[i : i + WINDOW])
            if d_sum > 0:
                win_iters.append(iters[i + WINDOW // 2])
                win_rate.append(n_sum / d_sum)

        if win_rate:
            peak_idx = int(np.argmax(win_rate))
            curve_peak_iter = win_iters[peak_idx]
            curve_peak_rate = win_rate[peak_idx]
            final_rate = win_rate[-1]
        else:
            curve_peak_iter = curve_peak_rate = final_rate = None

        out[name] = {
            "entrance_iter_tau0": entrance_iter,
            "honest_eval_peak_iter": peak_iter,
            "entrance_success_curve_peak_iter": curve_peak_iter,
            "entrance_success_curve_peak_rate": (
                round(curve_peak_rate, 4) if curve_peak_rate is not None else None
            ),
            "entrance_success_curve_final_rate": (
                round(final_rate, 4) if final_rate is not None else None
            ),
            "curve_decays_after_peak": (
                (final_rate < curve_peak_rate * 0.5)
                if curve_peak_rate not in (None, 0)
                else None
            ),
            "window": WINDOW,
            "n_windowed_points": len(win_rate),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
