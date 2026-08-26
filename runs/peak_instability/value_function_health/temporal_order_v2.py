import json
import numpy as np
from pathlib import Path

all_series = json.loads(Path("runs/peak_instability/value_function_health/all_series.json").read_text())
trailing = json.loads(Path("runs/peak_instability/value_function_health/trailing_series.json").read_text())

def rolling_series(arr, w=5):
    n = len(arr)
    return [float(np.mean(arr[max(0,i-w):min(n,i+w+1)])) for i in range(n)]

def find_decline_onset(tr_roll, anchor, decline_frac):
    lo, hi = max(0, anchor-5), min(len(tr_roll), anchor+6)
    peak_val = max(tr_roll[lo:hi])
    for i in range(anchor+1, len(tr_roll)):
        window = tr_roll[i:i+10]
        if len(window) < 5:
            break
        if np.mean(window) < decline_frac * peak_val:
            return i, peak_val
    return None, peak_val

def find_vloss_rise_onset(vloss_roll, anchor, rise_mult, floor_add=5):
    lo, hi = max(0, anchor-40), min(len(vloss_roll), anchor+40)
    idxs = list(range(lo, hi))
    t_min = min(idxs, key=lambda i: vloss_roll[i])
    vmin = vloss_roll[t_min]
    thresh = max(rise_mult * vmin, vmin + floor_add)
    for i in range(t_min+1, len(vloss_roll)):
        window = vloss_roll[i:i+10]
        if len(window) < 5:
            break
        if np.mean(window) > thresh:
            return i, t_min, vmin
    return None, t_min, vmin

configs = [
    (0.5, 1.5),
    (0.5, 1.3),
    (0.5, 2.0),
    (0.6, 1.5),
    (0.7, 1.5),
]

print(f"{'run':<10}{'honest':>8}", end="")
for dc, rc in configs:
    print(f"{'lag(d'+str(dc)+',r'+str(rc)+')':>16}", end="")
print()

all_lags = {name: [] for name in all_series}
for name, d in all_series.items():
    gens = d["gens"]
    vloss = d["series"]["ppo_value_loss"]
    tr_dict = trailing[name]
    tr = [float(tr_dict.get(str(g), tr_dict.get(g, 0.0))) for g in gens]
    tr_roll = rolling_series(tr, w=5)
    vloss_roll = rolling_series(vloss, w=5)
    anchor = d["peak_iter"]

    print(f"{name:<10}{d['honest_peak']:>8.3f}", end="")
    for dc, rc in configs:
        decline_onset, peak_val = find_decline_onset(tr_roll, anchor, dc)
        vloss_rise_onset, t_min, vmin = find_vloss_rise_onset(vloss_roll, anchor, rc)
        if decline_onset is not None and vloss_rise_onset is not None:
            lag = decline_onset - vloss_rise_onset
        else:
            lag = None
        all_lags[name].append(lag)
        print(f"{str(lag):>16}", end="")
    print()

print("\nsign summary per config (positive = value-loss-rise leads decline):")
for ci, (dc, rc) in enumerate(configs):
    lags = [all_lags[n][ci] for n in all_lags if all_lags[n][ci] is not None]
    pos = sum(1 for l in lags if l > 0)
    neg = sum(1 for l in lags if l < 0)
    zero = sum(1 for l in lags if l == 0)
    med = float(np.median(lags)) if lags else None
    print(f"  decline<{dc}xpeak, vloss>{rc}xmin: n={len(lags)} pos={pos} neg={neg} zero={zero} median={med}")
