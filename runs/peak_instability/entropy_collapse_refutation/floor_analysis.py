import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = json.load(open(REPO / "runs/peak_instability/entropy_collapse_refutation/raw_extract.json"))

def entropy_series(run):
    m = run["metrics"]
    its = sorted(int(k) for k in m.keys())
    return its, [m[str(i)]["ppo_entropy"] for i in its]

def trailing_at_entrance(run):
    log = run["log"]
    its = sorted(int(k) for k in log.keys())
    return [(i, log[str(i)]["trail_rate"], log[str(i)]["trail_n"], log[str(i)]["trail_d"],
              log[str(i)]["at_entrance"]) for i in its]

print(f"{'run':10} {'peak_it':7} {'floor_it':8} {'lag(floor-peak)':16} {'ent@floor':9} {'ent@240':8} {'further_drop':13} {'floor_run_len':13}")
rows=[]
for name, run in DATA.items():
    its, ent = entropy_series(run)
    ent_map = dict(zip(its, ent))
    peak_it = run["best"]["source_iter"]
    tr = trailing_at_entrance(run)
    # restrict to iters after peak
    tr_after = [t for t in tr if t[0] >= peak_it]
    floor_it = None
    for idx, (i, rate, n, d, at_ent) in enumerate(tr_after):
        if not at_ent:
            continue
        # "floor" = trailing rate <= 0.05 and stays <=0.10 for the rest of the run
        rest = tr_after[idx:]
        if rate <= 0.05 and all(r[1] <= 0.10 for r in rest):
            floor_it = i
            break
    if floor_it is None:
        print(f"{name:10} {peak_it:7} {'never':>8}")
        rows.append(dict(name=name, peak_it=peak_it, floor_it=None))
        continue
    ent_floor = ent_map.get(floor_it)
    ent_end = ent_map.get(max(its))
    further = ent_floor - ent_end
    run_len = max(its) - floor_it
    print(f"{name:10} {peak_it:7} {floor_it:8} {floor_it-peak_it:16} {ent_floor:9.4f} {ent_end:8.4f} {further:13.4f} {run_len:13}")
    rows.append(dict(name=name, peak_it=peak_it, floor_it=floor_it, ent_floor=ent_floor, ent_end=ent_end, further_drop=further, run_len=run_len))

json.dump(rows, open(REPO / "runs/peak_instability/entropy_collapse_refutation/floor.json", "w"), indent=2)
