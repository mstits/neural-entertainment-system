import json
import numpy as np
from pathlib import Path

all_series = json.loads(Path("runs/peak_instability/value_function_health/all_series.json").read_text())
trailing = json.loads(Path("runs/peak_instability/value_function_health/trailing_series.json").read_text())

def rolling_mean(arr, i, w=5):
    lo, hi = max(0, i-w), min(len(arr), i+w+1)
    return float(np.mean(arr[lo:hi]))

def rolling_series(arr, w=5):
    return [rolling_mean(arr, i, w) for i in range(len(arr))]

results = []
print(f"{'run':<10}{'honest':>8}{'t_peak(trail)':>15}{'t_peak(given)':>15}{'t_vmin':>9}{'vmin':>9}{'decline_onset':>15}{'vloss_rise_onset':>18}{'lag(decl-vrise)':>18}")
for name, d in all_series.items():
    gens = d["gens"]
    vloss = d["series"]["ppo_value_loss"]
    tr_dict = trailing[name]
    tr = [tr_dict.get(str(g), tr_dict.get(g, None)) for g in gens]
    # trailing_series.json keys are ints loaded from JSON -> become strings in JSON; check
    if all(v is None for v in tr):
        tr = [tr_dict[str(g)] for g in gens]
    tr = [float(x) if x is not None else 0.0 for x in tr]

    tr_roll = rolling_series(tr, w=5)
    vloss_roll = rolling_series(vloss, w=5)

    t_peak_trail = int(np.argmax(tr_roll))
    peak_val_trail = tr_roll[t_peak_trail]

    given_peak = d["peak_iter"]

    # vloss local min searched in window around given peak +-50, but also allow full-run search
    lo = max(0, given_peak - 50)
    hi = min(len(vloss_roll), given_peak + 50)
    window_idx = list(range(lo, hi))
    t_vmin = min(window_idx, key=lambda i: vloss_roll[i])
    vmin = vloss_roll[t_vmin]

    # decline onset: first i > t_peak_trail where rolling trailing < 0.5*peak and stays low for next 10
    decline_onset = None
    for i in range(t_peak_trail+1, len(tr_roll)):
        window = tr_roll[i:i+10]
        if len(window) < 3:
            break
        if np.mean(window) < 0.5 * peak_val_trail:
            decline_onset = i
            break

    # vloss rise onset: first i > t_vmin where rolling vloss > 1.5*vmin and stays elevated for next 10
    vloss_rise_onset = None
    thresh = max(1.5 * vmin, vmin + 5)
    for i in range(t_vmin+1, len(vloss_roll)):
        window = vloss_roll[i:i+10]
        if len(window) < 3:
            break
        if np.mean(window) > thresh:
            vloss_rise_onset = i
            break

    lag = None
    if decline_onset is not None and vloss_rise_onset is not None:
        lag = decline_onset - vloss_rise_onset

    print(f"{name:<10}{d['honest_peak']:>8.3f}{t_peak_trail:>15}{given_peak:>15}{t_vmin:>9}{vmin:>9.2f}"
          f"{str(decline_onset):>15}{str(vloss_rise_onset):>18}{str(lag):>18}")
    results.append(dict(name=name, honest=d['honest_peak'], t_peak_trail=t_peak_trail,
                         given_peak=given_peak, t_vmin=t_vmin, vmin=vmin,
                         decline_onset=decline_onset, vloss_rise_onset=vloss_rise_onset, lag=lag))

Path("runs/peak_instability/value_function_health/temporal_order_results.json").write_text(json.dumps(results, indent=1))
