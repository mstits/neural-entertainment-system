import csv
rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in rows:
    r["iter"]=int(r["iter"]); r["is_peak"]=r["is_peak"]=="True"
    r["eff_lr_median"]=float(r["eff_lr_median"]); r["eff_lr_mean"]=float(r["eff_lr_mean"])

policy_params = ["fc1.weight","fc1.bias","norm1.weight","norm1.bias","fc2.weight","fc2.bias",
        "norm2.weight","norm2.bias","actor.weight","actor.bias","critic.weight","critic.bias"]

peak_iters={}
for r in rows:
    if r["is_peak"]: peak_iters[r["run"]]=r["iter"]

for run in ["v27_seed0","v27_seed2","v28_seed2","v28_seed3"]:
    print(f"--- {run} (peak={peak_iters[run]}) ---")
    for it in [20, peak_iters[run], 240]:
        sub = [(r["param"], r["eff_lr_median"]) for r in rows if r["run"]==run and r["iter"]==it and r["param"] in policy_params]
        sub_sorted = sorted(sub, key=lambda x:x[1])
        lo_p, lo_v = sub_sorted[0]
        hi_p, hi_v = sub_sorted[-1]
        print(f"  iter{it:4d}: min={lo_p:14s}({lo_v:.4f})  max={hi_p:14s}({hi_v:.4f})  ratio={hi_v/lo_v:.1f}x")
    print()
