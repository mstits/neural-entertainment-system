"""
Supplementary cross-check on v27_fresh_recovery/gate/*.json (NOT the assigned
dataset -- assigned dataset is v28_capacity/gate/*.json). Included only to see
whether the direction of the v28 finding (peak more concentrated, final more
diffuse) reproduces in the other completed run family.

CAVEAT, load-bearing: v27 gate receipts were captured with a different eval
harness config -- eval_workers=1, eval_rng="shared-stream" -- versus v28's
eval_workers=8, eval_rng="per-episode". That is a protocol difference, not
just a different model, so v27 numbers are NOT pooled with v28 and NOT used
to inflate N in the primary verdict. Reported side-by-side, separately.
"""
import json, math
from collections import Counter
import sys
sys.path.insert(0, 'runs/peak_instability/episode_death_analysis')
from load_json_safe import load_gate_json

FLAG = 3161
BIN = 50
NBINS = 64

seeds = ['seed0', 'seed1', 'seed2', 'seed3']

def pooled(seed, ckpt):
    tag = 'winners-best' if ckpt == 'peak' else 'vanilla_ppo_iter_00240'
    vals = []
    for es in ['es0', 'es1']:
        d = load_gate_json(f'runs/v27_fresh_recovery/gate/{seed}_{tag}_{es}.json')
        vals.extend(d['max_gx_per_episode'])
    return vals

def entropy_norm(deaths):
    if not deaths:
        return None
    counts = Counter(min(v // BIN, NBINS - 1) for v in deaths)
    n = len(deaths)
    h = -sum((c/n) * math.log2(c/n) for c in counts.values())
    return h / math.log2(NBINS)

print("v27_fresh_recovery (48k params) -- SUPPLEMENTARY, different eval protocol, not pooled with v28")
print("=" * 100)
for seed in seeds:
    p_vals = pooled(seed, 'peak')
    f_vals = pooled(seed, 'final')
    p_deaths = [v for v in p_vals if v < FLAG]
    f_deaths = [v for v in f_vals if v < FLAG]
    p_clear = sum(1 for v in p_vals if v >= FLAG) / len(p_vals)
    f_clear = sum(1 for v in f_vals if v >= FLAG) / len(f_vals)
    p_h = entropy_norm(p_deaths)
    f_h = entropy_norm(f_deaths)
    p_bin = Counter((v // BIN) * BIN for v in p_deaths).most_common(1)[0] if p_deaths else (None, 0)
    f_bin = Counter((v // BIN) * BIN for v in f_deaths).most_common(1)[0] if f_deaths else (None, 0)
    print(f"{seed}: clear_rate peak={p_clear:.2f} final={f_clear:.2f}   "
          f"H_norm peak={p_h:.3f}(n={len(p_deaths)}) final={f_h:.3f}(n={len(f_deaths)})  "
          f"delta={f_h-p_h:+.3f}   "
          f"top-bin peak=[{p_bin[0]},+50) {p_bin[1]}/{len(p_deaths)}={p_bin[1]/len(p_deaths):.0%}   "
          f"top-bin final=[{f_bin[0]},+50) {f_bin[1]}/{len(f_deaths)}={f_bin[1]/len(f_deaths):.0%}")
