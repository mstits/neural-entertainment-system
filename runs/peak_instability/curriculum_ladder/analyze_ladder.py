"""Second-pass analysis over the parsed ladder rows: per-iter deltas on the
cumulative entrance/truncated counters, regime segmentation around
(a) pre-AT-ENTRANCE ladder descent, (b) AT-ENTRANCE-to-honest-peak climb,
(c) honest-peak-to-end decay -- all three segments while tau is frozen at 0
for (b) and (c). Also checks whether the two candidate "does more time at
entrance predict worse collapse" correlations hold given the actual
observed variance in that field (spoiler: there isn't much to correlate
against, and this script says so explicitly rather than reporting a
regression on ~doubled-digit-percent range).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

OUT = Path(__file__).parent
summary = json.loads((OUT / "ladder_summary.json").read_text())

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx*vy)

runs = [s["run"] for s in summary]
frac_ae = [s["frac_iters_at_entrance"] for s in summary]
n_ae = [s["n_iters_at_entrance"] for s in summary]
lag = [s["lag_peak_minus_first_at_entrance"] for s in summary]
honest_peak = [s["honest_peak"] for s in summary]
honest_final = [s["honest_final"] for s in summary]
collapse = [hp - hf for hp, hf in zip(honest_peak, honest_final)]
collapse_ratio = [ (hf/hp if hp>0 else float('nan')) for hp,hf in zip(honest_peak,honest_final)]

print("=== Range check on candidate predictor: frac_iters_at_entrance ===")
print(f"  min={min(frac_ae):.4f} max={max(frac_ae):.4f} range={max(frac_ae)-min(frac_ae):.4f} (out of [0,1])")
print(f"  -> {'ESSENTIALLY CONSTANT across all 8 runs -- cannot explain the honest_peak spread (0.04-0.67)' if (max(frac_ae)-min(frac_ae))<0.05 else 'has real spread'}")
print()
print("=== Correlations (N=8, exploratory -- report as leads not findings) ===")
print(f"  frac_iters_at_entrance vs honest_peak:        r={pearson(frac_ae, honest_peak):+.3f}")
print(f"  frac_iters_at_entrance vs honest_final:       r={pearson(frac_ae, honest_final):+.3f}")
print(f"  frac_iters_at_entrance vs collapse(peak-final):r={pearson(frac_ae, collapse):+.3f}")
print(f"  lag(peak - ladder-finish) vs honest_peak:     r={pearson(lag, honest_peak):+.3f}")
print(f"  lag(peak - ladder-finish) vs collapse:        r={pearson(lag, collapse):+.3f}")
print()

print("=== Per-run: ladder-finish iter, peak iter, lag, % of run spent frozen at tau=0 ===")
for s in summary:
    print(f"  {s['run']:10s}  AT-ENTRANCE@{s['first_at_entrance_iter']:3d}  peak@{s['peak_iter_bestjson']:3d}  "
          f"lag={s['lag_peak_minus_first_at_entrance']:3d}  frac_frozen={s['frac_iters_at_entrance']*100:.1f}%  "
          f"honest_peak={s['honest_peak']:.3f}  honest_final={s['honest_final']:.3f}")

lags_sorted = sorted(lag)
print()
print(f"lag stats: min={min(lag)} max={max(lag)} mean={sum(lag)/len(lag):.1f} median={ (lags_sorted[3]+lags_sorted[4])/2 }")
print(f"as % of 250 total iters: min={min(lag)/250*100:.1f}% max={max(lag)/250*100:.1f}% mean={sum(lag)/len(lag)/250*100:.1f}%")

# ---- entrance cumulative-rate turning point: does the cumulative ent_frac
# curve itself peak near the honest peak, using ONLY the raw per-line
# cumulative field (not the log's trailing rate, which is a documented
# lower bound pre-force-completion)?
print()
print("=== Cumulative entrance-rate (ent_n/ent_d, cumulative since run start) turning point ===")
for name in runs:
    rows = [json.loads(l) for l in open(OUT / f"ladder_{name}.jsonl")]
    post = [r for r in rows if r["at_entrance"]]
    peak_row = max(post, key=lambda r: r["ent_frac"])
    s = next(x for x in summary if x["run"] == name)
    print(f"  {name:10s} cum-ent_frac peaks at iter {peak_row['iter']:3d} (frac={peak_row['ent_frac']:.3f}); "
          f"best.json peak_iter={s['peak_iter_bestjson']:3d}; final cum-ent_frac={rows[-1]['ent_frac']:.3f}")

# ---- truncated-counter regime check: per-iter delta, segmented into
# pre-AT-ENTRANCE / AT-ENTRANCE-to-peak / peak-to-end
print()
print("=== truncated-counter per-iter delta by regime (mean truncations logged per iter) ===")
for name in runs:
    rows = [json.loads(l) for l in open(OUT / f"ladder_{name}.jsonl")]
    s = next(x for x in summary if x["run"] == name)
    ae_iter = s["first_at_entrance_iter"]
    peak_iter = s["peak_iter_bestjson"]

    def seg_rate(lo, hi):
        seg = [r for r in rows if lo <= r["iter"] <= hi]
        if len(seg) < 2:
            return float("nan")
        return (seg[-1]["truncated"] - seg[0]["truncated"]) / (seg[-1]["iter"] - seg[0]["iter"])

    pre = seg_rate(0, ae_iter)
    ramp = seg_rate(ae_iter, peak_iter)
    post = seg_rate(peak_iter, rows[-1]["iter"])
    print(f"  {name:10s} pre-AT-ENTRANCE(0-{ae_iter})={pre:6.1f}/iter  "
          f"AT-ENTRANCE->peak({ae_iter}-{peak_iter})={ramp:6.1f}/iter  "
          f"peak->end({peak_iter}-249)={post:6.1f}/iter")
