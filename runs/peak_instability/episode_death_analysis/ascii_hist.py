import json
from collections import Counter

FLAG = 3161
BIN = 150
raw = json.load(open('runs/peak_instability/episode_death_analysis/raw_pooled.json'))
seeds = ['seed0', 'seed1', 'seed2', 'seed3']

def pooled(seed, ckpt):
    vals = []
    for es in ['evalseed0', 'evalseed1']:
        key = f'{seed}_{ckpt}_{es}_greedy'
        vals.extend(raw[key]['max_gx_per_episode'])
    return vals

bins = list(range(0, 3200, BIN))

for seed in seeds:
    print(f"\n=== {seed}  (bin width={BIN} gx, all 100 episodes incl. clears at right edge marked F) ===")
    p_vals = pooled(seed, 'peak')
    f_vals = pooled(seed, 'final')
    p_c = Counter(min(v // BIN, len(bins)-1) for v in p_vals)
    f_c = Counter(min(v // BIN, len(bins)-1) for v in f_vals)
    maxcount = max(max(p_c.values(), default=1), max(f_c.values(), default=1))
    scale = 40 / maxcount
    print(f"{'bin':>12} | {'PEAK':<42} | {'FINAL':<42}")
    for i, b in enumerate(bins):
        pc = p_c.get(i, 0)
        fc = f_c.get(i, 0)
        if pc == 0 and fc == 0:
            continue
        label = f"[{b:4d},{b+BIN:4d})"
        pbar = '#' * round(pc * scale)
        fbar = '#' * round(fc * scale)
        print(f"{label:>12} | {pbar:<30}{pc:>3d} | {fbar:<30}{fc:>3d}")
