import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = json.load(open(REPO / "runs/peak_instability/entropy_collapse_refutation/raw_extract.json"))

THRESH = [0.5, 0.3, 0.1]

def entropy_series(run):
    m = run["metrics"]
    its = sorted(int(k) for k in m.keys())
    return its, [m[str(i)]["ppo_entropy"] for i in its]

def first_crossing(its, vals, thresh):
    """first iter at which val <= thresh, requiring it stays <= thresh for
    the rest of the series (avoid a noisy transient blip false-triggering)."""
    for idx, v in enumerate(vals):
        if v <= thresh and all(x <= thresh * 1.5 for x in vals[idx:idx+3]):
            return its[idx]
    return None

def trailing_series_at_entrance(run):
    log = run["log"]
    its = sorted(int(k) for k in log.keys())
    out = []
    for i in its:
        row = log[str(i)]
        if row["at_entrance"]:
            out.append((i, row["trail_rate"], row["trail_n"], row["trail_d"]))
    return out

print(f"{'run':10} {'width':5} {'peak_it':7} {'peak_val':9} " +
      " ".join(f"cross<= {t:<4}" for t in THRESH) + "  ent@peak")
report_rows = []
for name, run in DATA.items():
    its, ent = entropy_series(run)
    peak_it = run["best"]["source_iter"]
    peak_val = run["best"]["metric_value"]
    ent_at_peak = dict(zip(its, ent)).get(peak_it)
    crossings = {t: first_crossing(its, ent, t) for t in THRESH}
    lags = {t: (crossings[t] - peak_it if crossings[t] is not None else None) for t in THRESH}
    print(f"{name:10} {run['width']:5} {peak_it:7} {peak_val:9.3f} " +
          " ".join(f"{('it'+str(crossings[t])+' lag'+str(lags[t])) if crossings[t] is not None else 'never':>14}" for t in THRESH) +
          f"  {ent_at_peak:.4f}")
    report_rows.append(dict(name=name, width=run["width"], peak_it=peak_it, peak_val=peak_val,
                             ent_at_peak=ent_at_peak, crossings=crossings, lags=lags,
                             ent_series=list(zip(its, ent))))

json.dump(report_rows, open(REPO / "runs/peak_instability/entropy_collapse_refutation/crossings.json", "w"), indent=2)
