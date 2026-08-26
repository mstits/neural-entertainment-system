"""
Death-position distribution shape: SHIFT vs COLLAPSE-TO-MODE.

Primary dataset (assigned): runs/v28_capacity/gate/*.json, 4 seeds x
{peak,final} x {eval_seed 0,1}, pooled per (seed,checkpoint) to n=100.

Adds:
  - Shannon entropy of the death-position histogram on a FIXED, shared
    binning (bin width 50 gx units, range 0..3200 -> 64 bins) so peak and
    final are compared on the same yardstick regardless of differing
    death-population size or range.
  - Levene's test (variance equality) and 2-sample KS test (distribution
    shape) between peak-deaths and final-deaths, per seed.
  - Top-5 exact-value pileups, to eyeball whether any single x accounts for
    an outsized share of deaths.
  - Cross-seed check: does the modal death bin at FINAL land in the same
    place across seeds (one shared level hazard) or a different place per
    seed (policy-specific idiosyncratic failure)?
"""
import json
import math
from collections import Counter
from scipy import stats

FLAG = 3161
BIN = 50
NBINS = 64  # covers 0..3200

raw = json.load(open('runs/peak_instability/episode_death_analysis/raw_pooled.json'))
seeds = ['seed0', 'seed1', 'seed2', 'seed3']

def pooled(seed, ckpt):
    vals = []
    for es in ['evalseed0', 'evalseed1']:
        key = f'{seed}_{ckpt}_{es}_greedy'
        vals.extend(raw[key]['max_gx_per_episode'])
    return vals

def shannon_entropy_bits(deaths):
    if not deaths:
        return None, None
    counts = Counter(min(v // BIN, NBINS - 1) for v in deaths)
    n = len(deaths)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    hmax = math.log2(len(counts)) if len(counts) > 1 else 0.0
    # normalize by max possible entropy for the number of OCCUPIED bins is
    # circular; normalize instead against log2(NBINS) so it's on a fixed
    # 0..1 yardstick across all seeds/checkpoints.
    h_norm = h / math.log2(NBINS)
    return h, h_norm

print("=" * 100)
print("ENTROPY OF DEATH-POSITION HISTOGRAM (fixed 50-unit bins, 0..3200, normalized by log2(64)=6 bits)")
print("Lower normalized entropy = more concentrated into few bins = closer to a single deterministic failure mode")
print("=" * 100)
entropy_rows = []
for seed in seeds:
    row = {}
    for ckpt in ['peak', 'final']:
        vals = pooled(seed, ckpt)
        deaths = [v for v in vals if v < FLAG]
        h, hn = shannon_entropy_bits(deaths)
        row[ckpt] = (h, hn, len(deaths))
    entropy_rows.append((seed, row))
    p_h, p_hn, p_n = row['peak']
    f_h, f_hn, f_n = row['final']
    delta = (f_hn - p_hn) if (p_hn is not None and f_hn is not None) else None
    print(f"{seed}: peak H={p_h:.3f} bits (norm {p_hn:.3f}, n={p_n})   "
          f"final H={f_h:.3f} bits (norm {f_hn:.3f}, n={f_n})   "
          f"delta_norm(final-peak)={delta:+.3f}" if delta is not None else "n/a")

deltas = [ (f_hn - p_hn) for seed,row in entropy_rows for p_h,p_hn,p_n in [row['peak']] for f_h,f_hn,f_n in [row['final']] if p_hn is not None and f_hn is not None]
print(f"\nmean delta_norm entropy (final-peak) across {len(deltas)} seeds: {sum(deltas)/len(deltas):+.3f}")
print("(negative = final MORE concentrated than peak i.e. collapse-to-mode; positive = final MORE diffuse than peak)")

print()
print("=" * 100)
print("LEVENE'S TEST (variance equality) and KS TEST (distribution shape), peak-deaths vs final-deaths, per seed")
print("=" * 100)
for seed in seeds:
    p_vals = [v for v in pooled(seed, 'peak') if v < FLAG]
    f_vals = [v for v in pooled(seed, 'final') if v < FLAG]
    if len(p_vals) < 2 or len(f_vals) < 2:
        print(f"{seed}: insufficient death samples for test (peak n={len(p_vals)}, final n={len(f_vals)})")
        continue
    lev_stat, lev_p = stats.levene(p_vals, f_vals)
    ks_stat, ks_p = stats.ks_2samp(p_vals, f_vals)
    var_p = stats.tvar(p_vals)
    var_f = stats.tvar(f_vals)
    print(f"{seed}: var(peak)={var_p:8.1f}  var(final)={var_f:8.1f}  "
          f"Levene p={lev_p:.4f} ({'variances differ' if lev_p<0.05 else 'no sig. var diff'})   "
          f"KS D={ks_stat:.3f} p={ks_p:.4f} ({'shapes differ' if ks_p<0.05 else 'no sig. shape diff'})")

print()
print("=" * 100)
print("TOP-5 EXACT-VALUE PILEUPS (where >1 episode ends at the identical gx)")
print("=" * 100)
for seed in seeds:
    for ckpt in ['peak', 'final']:
        vals = pooled(seed, ckpt)
        deaths = [v for v in vals if v < FLAG]
        if not deaths:
            print(f"{seed} {ckpt}: all episodes cleared, no deaths")
            continue
        c = Counter(deaths)
        top5 = c.most_common(5)
        n = len(deaths)
        s = ", ".join(f"gx={v}(x{cnt},{cnt/n:.0%})" for v, cnt in top5)
        print(f"{seed:6s} {ckpt:5s} (n_death={n:3d}): {s}")

print()
print("=" * 100)
print("CROSS-SEED: does the FINAL-checkpoint modal death bin land in the same place across seeds?")
print("(shared bin -> one common level hazard; scattered bins -> policy-specific idiosyncratic failure)")
print("=" * 100)
for ckpt in ['peak', 'final']:
    print(f"\n-- {ckpt} --")
    for seed in seeds:
        vals = pooled(seed, ckpt)
        deaths = [v for v in vals if v < FLAG]
        if not deaths:
            print(f"{seed}: no deaths (all clear)")
            continue
        binned = Counter((v // BIN) * BIN for v in deaths)
        top_bin, top_count = binned.most_common(1)[0]
        print(f"{seed}: modal 50-wide bin = [{top_bin},{top_bin+BIN}) "
              f"covering {top_count}/{len(deaths)} = {top_count/len(deaths):.0%} of deaths; "
              f"median={sorted(deaths)[len(deaths)//2]}")
