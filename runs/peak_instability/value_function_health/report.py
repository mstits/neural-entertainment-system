import json
import numpy as np
from pathlib import Path

data = json.loads(Path("runs/peak_instability/value_function_health/all_series.json").read_text())

def smooth_window(vals, i, w=3):
    lo, hi = max(0, i-w), min(len(vals), i+w+1)
    xs = [v for v in vals[lo:hi] if v is not None]
    return float(np.mean(xs)) if xs else None

print(f"{'run':<10}{'peak_it':>8}{'honest':>8}  {'vloss@peak':>11}{'vloss@peak-20':>14}{'vloss@peak+20':>14}{'vloss@final':>12}  {'vloss_min_iter':>15}{'vloss_min_val':>14}")
rows_summary = []
for name, d in data.items():
    gens = d["gens"]
    vloss = d["series"]["ppo_value_loss"]
    peak_it = d["peak_iter"]
    # find index closest to peak_it, peak-20, peak+20
    def idx_at(target):
        return min(range(len(gens)), key=lambda i: abs(gens[i]-target))
    i_peak = idx_at(peak_it)
    i_pre = idx_at(peak_it-20)
    i_post = idx_at(peak_it+20)
    v_peak = smooth_window(vloss, i_peak)
    v_pre = smooth_window(vloss, i_pre)
    v_post = smooth_window(vloss, i_post)
    v_final = smooth_window(vloss, len(vloss)-1)
    # global min of vloss and its iter
    valid = [(g,v) for g,v in zip(gens, vloss) if v is not None]
    min_g, min_v = min(valid, key=lambda t: t[1])
    print(f"{name:<10}{peak_it:>8}{d['honest_peak']:>8.3f}  {v_peak:>11.4f}{v_pre:>14.4f}{v_post:>14.4f}{v_final:>12.4f}  {min_g:>15}{min_v:>14.4f}")
    rows_summary.append((name, peak_it, d['honest_peak'], v_peak, v_pre, v_post, v_final, min_g, min_v))

Path("runs/peak_instability/value_function_health/summary_table.json").write_text(
    json.dumps(rows_summary, indent=1)
)
