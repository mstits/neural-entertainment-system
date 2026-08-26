import csv
rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in rows:
    r["iter"]=int(r["iter"]); r["is_peak"]=r["is_peak"]=="True"
    r["eff_lr_median"]=float(r["eff_lr_median"])

peak_iters={}
for r in rows:
    if r["is_peak"]: peak_iters[r["run"]]=r["iter"]

params=["fc1.weight","fc1.bias","norm1.weight","norm1.bias","fc2.weight","fc2.bias",
        "norm2.weight","norm2.bias","actor.weight","actor.bias","critic.weight","critic.bias"]

print(f"{'param':16s} " + "  ".join(f"{r:10s}" for r in ['v27_seed0','v27_seed2','v28_seed2','v28_seed3']))
for p in params:
    cells=[]
    for run in ['v27_seed0','v27_seed2','v28_seed2','v28_seed3']:
        sub = sorted([r for r in rows if r["run"]==run and r["param"]==p and r["iter"]>=20], key=lambda r:r["iter"])
        argmax_it = max(sub, key=lambda r:r["eff_lr_median"])["iter"]
        pk = peak_iters[run]
        delta = argmax_it - pk
        cells.append(f"argmax={argmax_it:3d} d={delta:+4d}")
    print(f"{p:16s} " + "  ".join(cells))
