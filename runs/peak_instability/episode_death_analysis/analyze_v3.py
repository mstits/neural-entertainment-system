"""
Follow-ups: early-failure rate, and a speed proxy (aggregate distance/length)
to distinguish "dies fast" from "wanders/stalls without progress until
episode timeout" at peak vs final.
"""
import json
from collections import Counter

FLAG = 3161
raw = json.load(open('runs/peak_instability/episode_death_analysis/raw_pooled.json'))
seeds = ['seed0', 'seed1', 'seed2', 'seed3']

def pooled_vals_and_lengths(seed, ckpt):
    vals = []
    lengths = []
    for es in ['evalseed0', 'evalseed1']:
        key = f'{seed}_{ckpt}_{es}_greedy'
        vals.extend(raw[key]['max_gx_per_episode'])
        lengths.append(raw[key]['mean_length'])  # per-receipt aggregate, n=50 each
    return vals, lengths

print("=" * 100)
print("EARLY-FAILURE RATE (max_gx < 500, i.e. dies within roughly the first screen)")
print("=" * 100)
for seed in seeds:
    for ckpt in ['peak', 'final']:
        vals, _ = pooled_vals_and_lengths(seed, ckpt)
        deaths = [v for v in vals if v < FLAG]
        early = sum(1 for v in deaths if v < 500)
        n = len(deaths)
        frac = early / n if n else float('nan')
        print(f"{seed:6s} {ckpt:5s}: {early:3d}/{n:3d} deaths < gx=500  ({frac:.0%})")
    print()

print("=" * 100)
print("SPEED PROXY: mean_length (steps) per receipt, and mean_max_gx-over-all-100/mean_length")
print("(aggregate-level only -- no per-episode length in the receipts, so this is a population")
print(" average, not a per-episode ratio. A drop in this ratio at final = slower net progress per")
print(" step averaged over the pool (consistent with stalling/backtracking), a rise = faster net")
print(" progress per step before failing (consistent with dying quickly while still moving forward).)")
print("=" * 100)
for seed in seeds:
    for ckpt in ['peak', 'final']:
        vals, lengths = pooled_vals_and_lengths(seed, ckpt)
        mean_gx = sum(vals) / len(vals)
        mean_len = sum(lengths) / len(lengths)  # avg of the two receipts' mean_length
        print(f"{seed:6s} {ckpt:5s}: mean_max_gx(pooled)={mean_gx:7.1f}  mean_length(avg of 2 receipts)={mean_len:7.1f}  gx/step={mean_gx/mean_len:.3f}")
    print()
