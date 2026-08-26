import csv

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
            rows.append({k: (float(v) if v != "" else None) for k, v in row.items()})
    return rows

for run, peak in runs_peaks:
    rows = load(run)
    print(f"=== {run} (peak={peak}) ===")
    for lo in range(0, 250, 25):
        hi = lo + 25
        bucket = [r for r in rows if lo <= r["generation"] < hi]
        if not bucket:
            continue
        vl = sum(r["ppo_value_loss"] for r in bucket) / len(bucket)
        pl = sum(abs(r["ppo_policy_loss"]) for r in bucket) / len(bucket)
        ent = sum(r["ppo_entropy"] for r in bucket) / len(bucket)
        sr = sum(r["success_rate"] for r in bucket) / len(bucket)
        mark = " <== peak bucket" if lo <= peak < hi else ""
        print(f"  gen[{lo:3d}-{hi:3d}) vl={vl:7.2f} |pl|={pl:6.3f} ent={ent:.3f} sr={sr:.3f}{mark}")
    print()
