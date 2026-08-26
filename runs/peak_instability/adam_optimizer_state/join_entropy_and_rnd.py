import json, csv

adam_rows = list(csv.DictReader(open("runs/peak_instability/adam_optimizer_state/adam_state_report.csv")))
for r in adam_rows:
    r["iter"]=int(r["iter"]); r["is_peak"]=r["is_peak"]=="True"
    r["eff_lr_median"]=float(r["eff_lr_median"]); r["v_mean"]=float(r["v_mean"])
    r["frac_v_near_zero"]=float(r["frac_v_near_zero"])

RUN_DIRS = {
    "v28_seed3": "mario_1_1_v28_capacity_seed3",
    "v27_seed0": "mario_1_1_v27_recovery_seed0",
    "v27_seed2": "mario_1_1_v27_recovery_seed2",
    "v28_seed2": "mario_1_1_v28_capacity_seed2",
}

metrics = {}
for k, d in RUN_DIRS.items():
    with open(f"checkpoints/{d}/metrics.jsonl") as f:
        lines = [json.loads(l) for l in f]
    metrics[k] = {l["generation"]: l for l in lines}

def adam_field(run, iter_, param, field):
    for r in adam_rows:
        if r["run"]==run and r["iter"]==iter_ and r["param"]==param:
            return r[field]
    return None

def metric_field(run, iter_, field):
    m = metrics[run].get(iter_)
    return m[field] if m else None

peak_iters={}
for r in adam_rows:
    if r["is_peak"]: peak_iters[r["run"]]=r["iter"]

for run in RUN_DIRS:
    print(f"=== {run} (peak={peak_iters[run]}) ===")
    print(f"{'iter':5s} {'entropy':>8s} {'rnd_loss':>9s} {'intrinsic':>10s} {'actor.bias_effLR':>17s} {'actor.w_effLR':>14s} {'rndL0_fracV0':>13s}")
    for it in range(10,241,10):
        ent = metric_field(run, it, "ppo_entropy")
        rl = metric_field(run, it, "vanilla_ppo_rnd_loss")
        intr = metric_field(run, it, "vanilla_ppo_intrinsic_mean")
        ab = adam_field(run, it, "actor.bias", "eff_lr_median")
        aw = adam_field(run, it, "actor.weight", "eff_lr_median")
        rfz = adam_field(run, it, "rnd.predictor.net.0.weight", "frac_v_near_zero")
        mark = "*" if it==peak_iters[run] else " "
        def fmt(v, w):
            return f"{v:{w}.4g}" if v is not None else f"{'NA':>{w}s}"
        print(f"{it:4d}{mark} {fmt(ent,8)} {fmt(rl,9)} {fmt(intr,10)} {fmt(ab,17)} {fmt(aw,14)} {fmt(rfz,13)}")
    print()
