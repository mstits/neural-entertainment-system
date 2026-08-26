#!/usr/bin/env python3
"""Parse v27/v28 training logs into per-iteration proxy trailing-rate
series and winner (new-best) event tables.

The '[backward] iter N: ... trailing A/B=R' line is a LOWER BOUND on the
entrance_trailing_rate the winner-selector reads (a force-completion pass
runs after this line prints and before the winner-save block reads the
window -- see project telemetry-trap note). We use it as a per-iteration
noise proxy, not as the authoritative selection metric.
"""
import re
import json
import csv
import sys
from pathlib import Path

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
OUT = REPO / "runs/peak_instability/selection_noise"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = []
for s in range(4):
    RUNS.append(("v27", s, REPO / f"runs/v27_fresh_recovery/train_seed{s}.log",
                 REPO / f"checkpoints/mario_1_1_v27_recovery_seed{s}"))
for s in range(4):
    RUNS.append(("v28", s, REPO / f"runs/v28_capacity/train_seed{s}.log",
                 REPO / f"checkpoints/mario_1_1_v28_capacity_seed{s}"))

BACKWARD_RE = re.compile(
    r"\[backward\] iter (\d+): tau=(\d+)/(\d+) \(step (\d+) frame (\d+) gx (\d+)\) "
    r"trailing (\d+)/(\d+)=([\d.]+) \(advance at >=([\d.]+) over (\d+)\) "
    r"advances=(\d+)\s*(AT-ENTRANCE)? \| entrance (\d+)/(\d+)=([\d.]+) \| truncated (\d+)"
)
WINNER_RE = re.compile(
    r"\[winner\] (.+) new best (\w+)=([\d.]+) \(iter (\d+)\)"
)

def parse_log(path):
    rows = []
    winners = []
    with open(path) as f:
        for line in f:
            m = BACKWARD_RE.search(line)
            if m:
                (it, tau, tau_max, step, frame, gx, tr_s, tr_n, tr_rate,
                 adv_thr, adv_min, advances, at_entrance,
                 ent_s, ent_n, ent_rate, trunc) = m.groups()
                rows.append(dict(
                    iter=int(it), tau=int(tau), tau_max=int(tau_max),
                    trailing_succ=int(tr_s), trailing_n=int(tr_n),
                    trailing_rate=float(tr_rate), at_entrance=bool(at_entrance),
                    entrance_succ_cum=int(ent_s), entrance_n_cum=int(ent_n),
                    entrance_rate_cum=float(ent_rate), truncated_cum=int(trunc),
                ))
            m2 = WINNER_RE.search(line)
            if m2:
                game, metric_name, metric_val, it = m2.groups()
                winners.append(dict(
                    iter=int(it), metric_name=metric_name,
                    metric_value=float(metric_val),
                ))
    return rows, winners

all_rows = []
all_winner_events = []
for tag, seed, logpath, ckdir in RUNS:
    run_id = f"{tag}_seed{seed}"
    rows, winners = parse_log(logpath)
    for r in rows:
        r["run"] = run_id
        all_rows.append(r)
    for w in winners:
        w["run"] = run_id
        all_winner_events.append(w)
    bj = ckdir / "winners" / "best.json"
    best = json.load(open(bj)) if bj.exists() else None
    print(f"{run_id}: {len(rows)} backward-lines, {len(winners)} new-best events, "
          f"best.json peak={best['metric_value']:.4f} @ iter {best['source_iter']}")

# write CSVs
with open(OUT / "proxy_trailing_series.csv", "w", newline="") as f:
    fieldnames = ["run", "iter", "tau", "tau_max", "trailing_succ", "trailing_n",
                  "trailing_rate", "at_entrance", "entrance_succ_cum",
                  "entrance_n_cum", "entrance_rate_cum", "truncated_cum"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

with open(OUT / "winner_new_best_events.csv", "w", newline="") as f:
    fieldnames = ["run", "iter", "metric_name", "metric_value"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in all_winner_events:
        w.writerow(r)

print(f"\nwrote {OUT/'proxy_trailing_series.csv'}")
print(f"wrote {OUT/'winner_new_best_events.csv'}")
