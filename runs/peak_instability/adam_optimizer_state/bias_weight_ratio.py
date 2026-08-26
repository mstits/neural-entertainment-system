import csv
rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in rows:
    r["iter"]=int(r["iter"])
    r["eff_lr_median"]=float(r["eff_lr_median"])

def val(run, it, param):
    for r in rows:
        if r["run"]==run and r["iter"]==it and r["param"]==param:
            return r["eff_lr_median"]
    return None

runs = ["v27_seed0","v27_seed2","v28_seed2","v28_seed3"]
pairs = [("fc1.weight","fc1.bias"), ("fc2.weight","fc2.bias"),
         ("actor.weight","actor.bias"), ("critic.weight","critic.bias")]

for w,b in pairs:
    print(f"--- {b}/{w} ratio, iter20 -> iter240 ---")
    for run in runs:
        r20 = val(run,20,b)/val(run,20,w)
        r240 = val(run,240,b)/val(run,240,w)
        print(f"  {run:10s} iter20={r20:.3f}  iter240={r240:.3f}  growth={r240/r20:.2f}x")
    print()
