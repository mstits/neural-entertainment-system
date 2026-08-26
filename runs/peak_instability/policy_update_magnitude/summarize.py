import csv, statistics as st

runs_peaks = [
    ("mario_1_1_v27_recovery_seed0", 60),
    ("mario_1_1_v27_recovery_seed1", 50),
    ("mario_1_1_v27_recovery_seed2", 90),
    ("mario_1_1_v27_recovery_seed3", 60),
    ("mario_1_1_v28_capacity_seed0", 70),
    ("mario_1_1_v28_capacity_seed1", 60),
    ("mario_1_1_v28_capacity_seed2", 120),
    ("mario_1_1_v28_capacity_seed3", 90),
]

def load(run):
    path = f"runs/peak_instability/policy_update_magnitude/series_{run}.csv"
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (float(v) if v not in ("",) else None) for k, v in row.items()})
    return rows

print(f"{'run':32s} {'peak':>5s} {'|pl| med<=peak':>14s} {'|pl| med>peak':>14s} {'|pl| max<=peak':>14s} {'|pl| max>peak':>14s} {'n |pl|>1 <=peak':>16s} {'n |pl|>1 >peak':>16s}")
for run, peak in runs_peaks:
    rows = load(run)
    pre = [r for r in rows if r["generation"] <= peak]
    post = [r for r in rows if r["generation"] > peak]
    pl_pre = [abs(r["ppo_policy_loss"]) for r in pre]
    pl_post = [abs(r["ppo_policy_loss"]) for r in post]
    n_spike_pre = sum(1 for v in pl_pre if v > 1.0)
    n_spike_post = sum(1 for v in pl_post if v > 1.0)
    print(f"{run:32s} {peak:5d} {st.median(pl_pre):14.4f} {st.median(pl_post):14.4f} {max(pl_pre):14.3f} {max(pl_post):14.3f} {n_spike_pre:16d} {n_spike_post:16d}")
