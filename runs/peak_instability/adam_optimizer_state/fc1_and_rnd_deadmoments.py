import csv
rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in rows:
    r["iter"]=int(r["iter"])
    for k in ("eff_lr_mean","eff_lr_median","eff_lr_min","eff_lr_max","frac_v_near_zero","v_mean","v_min","v_max"):
        r[k]=float(r[k])

def series(run, param, field):
    sub = sorted([r for r in rows if r["run"]==run and r["param"]==param], key=lambda r:r["iter"])
    return [(r["iter"], r[field]) for r in sub]

for run in ["v28_seed3","v27_seed0","v27_seed2","v28_seed2"]:
    print(f"=== {run}: fc1.weight over training ===")
    for field in ("eff_lr_mean","eff_lr_median","frac_v_near_zero","v_min","v_max"):
        s = series(run, "fc1.weight", field)
        vals = ", ".join(f"{it}:{v:.3g}" for it,v in s[::2])  # every other checkpoint
        print(f"  {field:20s} {vals}")
    print()

print("=== compare fc1.weight vs rnd.predictor.net.0.weight (both take raw 712-dim tile input) ===")
for run in ["v28_seed3"]:
    for p in ["fc1.weight","rnd.predictor.net.0.weight"]:
        s = series(run,p,"frac_v_near_zero")
        print(run, p, [f"{it}:{v:.3f}" for it,v in s])
