import csv, json
rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in rows:
    r["iter"]=int(r["iter"])
    r["is_peak"]=r["is_peak"]=="True"
    for k in ("eff_lr_mean","eff_lr_median","frac_v_near_zero","v_mean","m_rms"):
        r[k]=float(r[k])

peak_iters = {}
for r in rows:
    if r["is_peak"]:
        peak_iters[r["run"]] = r["iter"]

for param in ["fc1.weight","fc2.weight","norm2.weight","actor.weight","actor.bias","critic.weight","critic.bias"]:
    print(f"##### {param} — eff_lr_median across iters (peak marked with *) #####")
    for run in ["v27_seed0","v27_seed2","v28_seed2","v28_seed3"]:
        sub = sorted([r for r in rows if r["run"]==run and r["param"]==param], key=lambda r:r["iter"])
        pk = peak_iters[run]
        cells = []
        for r in sub:
            mark = "*" if r["iter"]==pk else " "
            cells.append(f"{r['iter']:3d}{mark}:{r['eff_lr_median']:.3g}")
        print(f"  {run:10s} peak={pk:3d}  " + "  ".join(cells))
    print()
