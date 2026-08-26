import json, sys, csv

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
fields = ["generation", "ppo_loss", "ppo_policy_loss", "ppo_value_loss",
          "ppo_entropy", "success_rate", "episodes",
          "vanilla_ppo_clears", "vanilla_ppo_in_progress"]

for run in runs:
    path = f"checkpoints/{run}/metrics.jsonl"
    out = f"runs/peak_instability/policy_update_magnitude/series_{run}.csv"
    with open(path) as f, open(out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(fields)
        for line in f:
            row = json.loads(line)
            w.writerow([row.get(k, "") for k in fields])
    print("wrote", out)
