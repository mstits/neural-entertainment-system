"""
Where do episodes actually die, at peak vs final checkpoint?

Input: runs/v28_capacity/gate/*.json (16 honest-eval receipts: 4 seeds x
{peak,final} x {eval_seed 0,1}). Each holds max_gx_per_episode, 50 ints,
greedy/sticky-0.25/jitter-16 protocol.

Pools the two eval seeds per (run, checkpoint) into n=100 episodes, matching
the honest-eval pooling convention used elsewhere in this campaign.

Question: is the final-checkpoint collapse a SHIFT (whole distribution moves
left, still diffuse) or a COLLAPSE TO A MODE (variance drops, most episodes
end at nearly the same x -> one deterministic failure)?

Flagpole = gx 3161 (a clear). Clears are pulled out and reported separately
(clear_rate) since a clear is not a death -- pooling them into a "death
position" histogram would fabricate a fake mode at the goal line.
"""
import json
import statistics
from collections import Counter

FLAG = 3161

raw = json.load(open('runs/peak_instability/episode_death_analysis/raw_pooled.json'))

seeds = ['seed0', 'seed1', 'seed2', 'seed3']
ckpts = ['peak', 'final']

def pooled(seed, ckpt):
    vals = []
    for es in ['evalseed0', 'evalseed1']:
        key = f'{seed}_{ckpt}_{es}_greedy'
        vals.extend(raw[key]['max_gx_per_episode'])
    return vals

def death_stats(vals):
    n = len(vals)
    clears = [v for v in vals if v >= FLAG]
    deaths = [v for v in vals if v < FLAG]
    out = {
        'n': n,
        'n_clear': len(clears),
        'clear_rate': len(clears) / n,
        'n_death': len(deaths),
    }
    if deaths:
        out['death_mean'] = statistics.mean(deaths)
        out['death_median'] = statistics.median(deaths)
        out['death_stdev'] = statistics.pstdev(deaths) if len(deaths) > 1 else 0.0
        out['death_min'] = min(deaths)
        out['death_max'] = max(deaths)
        out['death_range'] = max(deaths) - min(deaths)
        # IQR
        sd = sorted(deaths)
        def pct(p):
            k = (len(sd)-1) * p
            f = int(k); c = min(f+1, len(sd)-1)
            if f == c: return sd[f]
            return sd[f] + (sd[c]-sd[f]) * (k-f)
        out['death_p25'] = pct(0.25)
        out['death_p75'] = pct(0.75)
        out['death_iqr'] = out['death_p75'] - out['death_p25']

        # modal analysis: bin at multiple resolutions
        # exact-value mode
        c = Counter(deaths)
        top_val, top_count = c.most_common(1)[0]
        out['exact_mode_value'] = top_val
        out['exact_mode_count'] = top_count
        out['exact_mode_frac'] = top_count / len(deaths)
        out['n_unique_death_values'] = len(c)

        # binned mode, bin width 50 gx units (~ a couple mario-widths)
        BIN = 50
        binned = Counter((v // BIN) * BIN for v in deaths)
        top_bin, top_bin_count = binned.most_common(1)[0]
        out['binned50_mode_bin_start'] = top_bin
        out['binned50_mode_frac'] = top_bin_count / len(deaths)
        out['n_unique_bins50'] = len(binned)

        # concentration within +-25 of the death median (tight-window test for "one wrong thing")
        med = out['death_median']
        within25 = sum(1 for v in deaths if abs(v - med) <= 25)
        within50 = sum(1 for v in deaths if abs(v - med) <= 50)
        out['frac_within25_of_median'] = within25 / len(deaths)
        out['frac_within50_of_median'] = within50 / len(deaths)
    else:
        for k in ['death_mean','death_median','death_stdev','death_min','death_max',
                   'death_range','death_p25','death_p75','death_iqr','exact_mode_value',
                   'exact_mode_count','exact_mode_frac','n_unique_death_values',
                   'binned50_mode_bin_start','binned50_mode_frac','n_unique_bins50',
                   'frac_within25_of_median','frac_within50_of_median']:
            out[k] = None
    return out

results = {}
for seed in seeds:
    results[seed] = {}
    for ckpt in ckpts:
        vals = pooled(seed, ckpt)
        results[seed][ckpt] = death_stats(vals)

json.dump(results, open('runs/peak_instability/episode_death_analysis/death_stats.json', 'w'), indent=1)

# ---- report ----
print("=" * 100)
print("PER-SEED: peak vs final, pooled over eval_seed {0,1}, n=100 each")
print("=" * 100)
for seed in seeds:
    p = results[seed]['peak']
    f = results[seed]['final']
    print(f"\n--- {seed} ---")
    print(f"  clear_rate:      peak={p['clear_rate']:.2f}   final={f['clear_rate']:.2f}")
    print(f"  n_death (of100): peak={p['n_death']:3d}       final={f['n_death']:3d}")
    if p['n_death']:
        print(f"  death mean:      peak={p['death_mean']:.1f}   final={f['death_mean']:.1f}" if f['n_death'] else f"  death mean:      peak={p['death_mean']:.1f}   final=N/A(all clear)")
        print(f"  death median:    peak={p['death_median']:.1f}   final={f['death_median']:.1f}" if f['n_death'] else "")
        print(f"  death stdev:     peak={p['death_stdev']:.1f}   final={f['death_stdev']:.1f}" if f['n_death'] else "")
        print(f"  death range:     peak={p['death_min']}-{p['death_max']} (r={p['death_range']})" +
              (f"   final={f['death_min']}-{f['death_max']} (r={f['death_range']})" if f['n_death'] else ""))
        print(f"  death IQR:       peak={p['death_iqr']:.1f}" + (f"   final={f['death_iqr']:.1f}" if f['n_death'] else ""))
        print(f"  unique values:   peak={p['n_unique_death_values']}/{p['n_death']}" +
              (f"   final={f['n_unique_death_values']}/{f['n_death']}" if f['n_death'] else ""))
        print(f"  exact mode:      peak val={p['exact_mode_value']} frac={p['exact_mode_frac']:.2f}" +
              (f"   final val={f['exact_mode_value']} frac={f['exact_mode_frac']:.2f}" if f['n_death'] else ""))
        print(f"  binned50 mode:   peak bin={p['binned50_mode_bin_start']} frac={p['binned50_mode_frac']:.2f}" +
              (f"   final bin={f['binned50_mode_bin_start']} frac={f['binned50_mode_frac']:.2f}" if f['n_death'] else ""))
        print(f"  frac w/in +-25 of median: peak={p['frac_within25_of_median']:.2f}" +
              (f"   final={f['frac_within25_of_median']:.2f}" if f['n_death'] else ""))
        print(f"  frac w/in +-50 of median: peak={p['frac_within50_of_median']:.2f}" +
              (f"   final={f['frac_within50_of_median']:.2f}" if f['n_death'] else ""))
