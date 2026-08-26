#!/usr/bin/env python3
"""Consolidated selection-noise / winner's-curse analysis (peak-instability,
"is the peak real or selection noise" dimension). Reproducible from
proxy_trailing_series.csv (built by parse_logs.py) + best.json + the
honest-eval numbers supplied in the task brief + the two honest-eval gate
receipts read directly off disk for the eval-seed-split sanity check.
"""
import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
IN = REPO / "runs/peak_instability/selection_noise"

HONEST = {
    "v27_seed0": dict(peak_iter=60, honest_peak=0.040, honest_final=0.020),
    "v27_seed1": dict(peak_iter=50, honest_peak=0.290, honest_final=0.020),
    "v27_seed2": dict(peak_iter=90, honest_peak=0.530, honest_final=0.000),
    "v27_seed3": dict(peak_iter=60, honest_peak=0.170, honest_final=0.010),
    "v28_seed0": dict(peak_iter=70, honest_peak=0.450, honest_final=0.000),
    "v28_seed1": dict(peak_iter=60, honest_peak=0.230, honest_final=0.050),
    "v28_seed2": dict(peak_iter=120, honest_peak=0.370, honest_final=0.000),
    "v28_seed3": dict(peak_iter=90, honest_peak=0.670, honest_final=0.000),
}

rows = defaultdict(list)
with open(IN / "proxy_trailing_series.csv") as f:
    for r in csv.DictReader(f):
        rows[r["run"]].append(r)
for run in rows:
    rows[run].sort(key=lambda x: int(x["iter"]))

best = {}
for run in HONEST:
    tag, seed = run.split("_seed")
    sub = "recovery" if tag == "v27" else "capacity"
    ck = REPO / f"checkpoints/mario_1_1_{tag}_{sub}_seed{seed}"
    best[run] = json.load(open(ck / "winners" / "best.json"))

def full_checkpoint_rates(run):
    rs = rows[run]
    ck = [r for r in rs if int(r["iter"]) % 10 == 0 and int(r["iter"]) > 0
          and int(r["trailing_n"]) == 30]
    return [float(r["trailing_rate"]) for r in ck]

# ---------------------------------------------------------------- (a)
print("=" * 80)
print("(a) Window independence check: new qualifying entrance episodes per")
print("    10-iter checkpoint gap (must be >=30 for the two checkpoints'")
print("    30-sample windows to be fully disjoint)")
print("=" * 80)
for run, rs in rows.items():
    n_cum = {int(r["iter"]): int(r["entrance_n_cum"]) for r in rs}
    ckpts = list(range(10, 241, 10))
    deltas = [n_cum[b] - n_cum[a] for a, b in zip(ckpts, ckpts[1:])]
    print(f"{run:12s} min_delta={min(deltas):4d}  "
          f"(all {len(deltas)} gaps >=30: {all(d >= 30 for d in deltas)})")

print()
print("=" * 80)
print("(a) Checkpoint-spaced (10-iter) proxy trailing_rate noise vs binomial floor")
print("=" * 80)
print(f"{'run':12s} {'mean_p':>7s} {'sd_diff1/sqrt2':>15s} {'sd_diff2/sqrt6':>15s} "
      f"{'binom_sd(n=30)':>15s} {'ratio1':>7s} {'ratio2':>7s}")
r1s, r2s = [], []
for run in HONEST:
    rates = full_checkpoint_rates(run)
    mean_p = st.mean(rates)
    d1 = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
    sd1 = st.pstdev(d1) / math.sqrt(2)
    d2 = [rates[i + 1] - 2 * rates[i] + rates[i - 1] for i in range(1, len(rates) - 1)]
    sd2 = st.pstdev(d2) / math.sqrt(6)
    binom = math.sqrt(mean_p * (1 - mean_p) / 30)
    ratio1, ratio2 = sd1 / binom, sd2 / binom
    r1s.append(ratio1); r2s.append(ratio2)
    print(f"{run:12s} {mean_p:7.3f} {sd1:15.3f} {sd2:15.3f} {binom:15.3f} "
          f"{ratio1:7.2f} {ratio2:7.2f}")
print(f"\nmedian ratio (1st-diff, trend-contaminated upper bound): {st.median(r1s):.2f}")
print(f"median ratio (2nd-diff, linear-trend-robust):             {st.median(r2s):.2f}")

# ---------------------------------------------------------------- (b) MC
print()
print("=" * 80)
print("(b) Winner's-curse MC: E[max of K=24 disjoint Binomial(30,p)/30 draws]")
print("=" * 80)
random.seed(1234)
N_SIM = 20000
K = 24
print(f"{'p_true':>8s} {'E[max]':>8s} {'inflation':>10s} {'P(max>=.90)':>13s} {'P(max==1.0)':>13s}")
for p_true in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    maxes = []
    ge90 = eq100 = 0
    for _ in range(N_SIM):
        m = max(sum(1 for _ in range(30) if random.random() < p_true) / 30.0
                for _ in range(K))
        maxes.append(m)
        ge90 += m >= 0.90
        eq100 += m >= 0.999
    em = st.mean(maxes)
    print(f"{p_true:8.2f} {em:8.3f} {em - p_true:10.3f} {ge90 / N_SIM:13.3f} {eq100 / N_SIM:13.3f}")

print()
print("=" * 80)
print("(b) Direct empirical test: proxy-vs-honest gap at a NON-selected")
print("    checkpoint (iter=240, terminal) vs at the ARGMAX-selected peak")
print("=" * 80)
print(f"{'run':12s} {'proxy@240':>10s} {'honest_final':>12s} {'gap@240':>9s} "
      f"{'metric@peak':>12s} {'honest@peak':>12s} {'gap@peak':>9s}")
gaps240, gapspeak = [], []
for run, meta in HONEST.items():
    by_iter = {int(r["iter"]): float(r["trailing_rate"]) for r in rows[run]}
    p240 = by_iter[240]
    hf = meta["honest_final"]
    g240 = p240 - hf
    mv = best[run]["metric_value"]
    hp = meta["honest_peak"]
    gpk = mv - hp
    gaps240.append(g240); gapspeak.append(gpk)
    print(f"{run:12s} {p240:10.3f} {hf:12.3f} {g240:9.3f} {mv:12.3f} {hp:12.3f} {gpk:9.3f}")
print(f"\nmean/median gap @ iter240 (non-selected): {st.mean(gaps240):+.3f} / {st.median(gaps240):+.3f}")
print(f"mean/median gap @ peak (argmax-selected):  {st.mean(gapspeak):+.3f} / {st.median(gapspeak):+.3f}")
print("NOTE: iter=240 gap is measured near the metric floor (both series ~0-0.05")
print("for 7/8 runs) -- confirms no *structural* proxy-vs-honest bias near zero,")
print("but does NOT by itself confirm the same holds at mid-range true capability.")

print()
print("=" * 80)
print("(b) Does a CONSTANT true rate = honest_peak, run through the same MC,")
print("    explain the observed ceiling-clustered metric_value?")
print("=" * 80)
print(f"{'run':12s} {'honest_peak(p)':>15s} {'E[max24] pred':>14s} {'observed':>9s} {'resid':>7s}")
resids = []
for run, meta in HONEST.items():
    p = meta["honest_peak"]
    maxes = [max(sum(1 for _ in range(30) if random.random() < p) / 30.0
                 for _ in range(K)) for _ in range(N_SIM // 2)]
    pred = st.mean(maxes)
    obs = best[run]["metric_value"]
    resids.append(obs - pred)
    print(f"{run:12s} {p:15.3f} {pred:14.3f} {obs:9.3f} {obs - pred:7.3f}")
print(f"\nmean residual (observed-predicted): {st.mean(resids):+.3f}  "
      f"-> a flat true-rate-at-honest-peak model UNDER-predicts; some real,")
print("if partly transient, capability rise into the peak region is also needed.")

# ---------------------------------------------------------------- (c)
print()
print("=" * 80)
print("(c) metric_value (argmax over ~24 draws) vs honest_peak -- rank vs scale")
print("=" * 80)
xs = [best[r]["metric_value"] for r in HONEST]
ys = [HONEST[r]["honest_peak"] for r in HONEST]
n = len(xs); mx, my = st.mean(xs), st.mean(ys)
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
r = cov / (st.pstdev(xs) * st.pstdev(ys))
print(f"metric_value span: [{min(xs):.3f},{max(xs):.3f}] = {max(xs)-min(xs):.3f}")
print(f"honest_peak  span: [{min(ys):.3f},{max(ys):.3f}] = {max(ys)-min(ys):.3f}")
print(f"compression ratio (honest span / metric span): {(max(ys)-min(ys))/(max(xs)-min(xs)):.2f}x")
print(f"Pearson r = {r:.3f}, R^2 = {r*r:.3f}, N=8 (LEAD not finding at this N)")

# ---------------------------------------------------------------- (b2)
print()
print("=" * 80)
print("(b2) new-best event count per run vs i.i.d. order-statistic null")
print("=" * 80)
H24 = sum(1.0 / i for i in range(1, 25))
wcounts = defaultdict(int)
with open(IN / "winner_new_best_events.csv") as f:
    for r in csv.DictReader(f):
        wcounts[r["run"]] += 1
vals = [wcounts[run] for run in HONEST]
print(f"null E[# running maxima in 24 i.i.d. draws, no trend] = {H24:.2f}")
print("observed:", dict(wcounts), " mean =", round(st.mean(vals), 2),
      f"({st.mean(vals)/H24:.2f}x null)")

# ---------------------------------------------------------------- (d)
print()
print("=" * 80)
print("(d) Local smoothness of the proxy around each run's own peak_iter")
print("=" * 80)
for run, meta in HONEST.items():
    pk = meta["peak_iter"]
    by_iter = {int(r["iter"]): float(r["trailing_rate"]) for r in rows[run]}
    offs = [-20, -10, 0, 10, 20]
    vs = ", ".join(f"{o:+d}:{by_iter.get(pk+o, float('nan')):.2f}" for o in offs)
    neighbor_beats_peak = any(
        by_iter.get(pk + o, -1) > by_iter.get(pk, 2) for o in (-10, 10)
        if pk + o in by_iter
    )
    print(f"{run:12s} peak={pk:4d}  {vs}   neighbor>peak(proxy)? {neighbor_beats_peak}")
