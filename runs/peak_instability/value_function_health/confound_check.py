import json
import numpy as np
from pathlib import Path

all_series = json.loads(Path("runs/peak_instability/value_function_health/all_series.json").read_text())
trailing = json.loads(Path("runs/peak_instability/value_function_health/trailing_series.json").read_text())

def rolling_series(arr, w=5):
    n = len(arr)
    return [float(np.mean(arr[max(0,i-w):min(n,i+w+1)])) for i in range(n)]

print(f"{'run':<10}{'honest':>8}{'vrise_onset':>13}{'entropy@vrise':>15}{'entropy_at_peak':>17}{'entropy_at_decline':>19}")
for name, d in all_series.items():
    gens = d["gens"]
    vloss = d["series"]["ppo_value_loss"]
    ent = d["series"]["ppo_entropy"]
    vloss_roll = rolling_series(vloss, w=5)
    ent_roll = rolling_series(ent, w=5)
    anchor = d["peak_iter"]
    lo, hi = max(0, anchor-40), min(len(vloss_roll), anchor+40)
    idxs = list(range(lo, hi))
    t_min = min(idxs, key=lambda i: vloss_roll[i])
    vmin = vloss_roll[t_min]
    thresh = max(1.5*vmin, vmin+5)
    vrise = None
    for i in range(t_min+1, len(vloss_roll)):
        window = vloss_roll[i:i+10]
        if len(window) < 5: break
        if np.mean(window) > thresh:
            vrise = i
            break
    ent_at_vrise = ent_roll[vrise] if vrise is not None else None
    ent_at_peak = ent_roll[anchor]
    print(f"{name:<10}{d['honest_peak']:>8.3f}{str(vrise):>13}{str(round(ent_at_vrise,4) if ent_at_vrise else None):>15}{ent_at_peak:>17.4f}", end="")
    print()
