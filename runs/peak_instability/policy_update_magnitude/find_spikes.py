import csv

runs = [
    "mario_1_1_v27_recovery_seed0",
    "mario_1_1_v27_recovery_seed1",
    "mario_1_1_v27_recovery_seed2",
    "mario_1_1_v27_recovery_seed3",
    "mario_1_1_v28_capacity_seed0",
    "mario_1_1_v28_capacity_seed1",
    "mario_1_1_v28_capacity_seed2",
    "mario_1_1_v28_capacity_seed3",
]

def load(run):
    path = f"runs/peak_instability/policy_update_magnitude/series_{run}.csv"
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (float(v) if v != "" else None) for k, v in row.items()})
    return rows

for run in runs:
    rows = load(run)
    print(f"=== {run} ===")
    for i, r in enumerate(rows):
        if abs(r["ppo_policy_loss"]) > 1.0:
            ctx = rows[max(0,i-1):i+2]
            print(f"  gen={int(r['generation'])} ppo_policy_loss={r['ppo_policy_loss']:.4f} ppo_loss={r['ppo_loss']:.4f} ppo_value_loss={r['ppo_value_loss']:.4f} ppo_entropy={r['ppo_entropy']:.4f} success_rate={r['success_rate']:.4f}")
            # neighbors
            for j in range(max(0,i-2), min(len(rows), i+3)):
                rr = rows[j]
                marker = "->" if j == i else "  "
                print(f"    {marker} gen={int(rr['generation']):3d} pl={rr['ppo_policy_loss']:>10.4f} loss={rr['ppo_loss']:>8.3f} vl={rr['ppo_value_loss']:>8.3f} ent={rr['ppo_entropy']:.4f} sr={rr['success_rate']:.3f}")
    print()
