import json, glob, os

out = {}
for f in sorted(glob.glob('runs/v28_capacity/gate/*.json')):
    d = json.load(open(f))
    key = os.path.basename(f).replace('.json','')
    out[key] = {
        'checkpoint': d['checkpoint'],
        'eval_seed': d['eval_seed'],
        'clear_rate': d['clear_rate'],
        'mean_return': d['mean_return'],
        'mean_length': d['mean_length'],
        'n_episodes': d['n_episodes'],
        'max_gx_per_episode': d['max_gx_per_episode'],
    }

json.dump(out, open('runs/peak_instability/episode_death_analysis/raw_pooled.json','w'), indent=1)
print("wrote", len(out), "receipts")
