import json, csv, os

RUNS = [
    ("v27", "seed0", "checkpoints/mario_1_1_v27_recovery_seed0/metrics.jsonl"),
    ("v27", "seed1", "checkpoints/mario_1_1_v27_recovery_seed1/metrics.jsonl"),
    ("v27", "seed2", "checkpoints/mario_1_1_v27_recovery_seed2/metrics.jsonl"),
    ("v27", "seed3", "checkpoints/mario_1_1_v27_recovery_seed3/metrics.jsonl"),
    ("v28", "seed0", "checkpoints/mario_1_1_v28_capacity_seed0/metrics.jsonl"),
    ("v28", "seed1", "checkpoints/mario_1_1_v28_capacity_seed1/metrics.jsonl"),
    ("v28", "seed2", "checkpoints/mario_1_1_v28_capacity_seed2/metrics.jsonl"),
    ("v28", "seed3", "checkpoints/mario_1_1_v28_capacity_seed3/metrics.jsonl"),
]

PEAK_ITER = {
    ("v27", "seed0"): 60, ("v27", "seed1"): 50, ("v27", "seed2"): 90, ("v27", "seed3"): 60,
    ("v28", "seed0"): 70, ("v28", "seed1"): 60, ("v28", "seed2"): 120, ("v28", "seed3"): 90,
}

FIELDS = [
    "generation", "reward_forward", "reward_completion", "reward_time_penalty",
    "ppo_entropy", "success_rate", "vanilla_ppo_clears", "vanilla_ppo_in_progress",
    "avg_fitness", "best_fitness", "episodes",
]

out_rows = []
for run, seed, path in RUNS:
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            row = {"run": run, "seed": seed}
            for k in FIELDS:
                row[k] = d.get(k)
            f = row["reward_forward"] or 0.0
            c = row["reward_completion"] or 0.0
            t = row["reward_time_penalty"] or 0.0
            mag_total = abs(f) + abs(c) + abs(t)
            row["mag_total"] = mag_total
            row["share_forward"] = f / mag_total if mag_total > 0 else None
            row["share_completion"] = c / mag_total if mag_total > 0 else None
            row["share_time_penalty"] = abs(t) / mag_total if mag_total > 0 else None
            row["net_logged_reward"] = f + c + t
            row["peak_iter"] = PEAK_ITER[(run, seed)]
            out_rows.append(row)

out_path = "runs/peak_instability/reward_mix/reward_components_all_runs.csv"
with open(out_path, "w", newline="") as fh:
    cols = ["run", "seed"] + FIELDS + ["mag_total", "share_forward", "share_completion",
                                         "share_time_penalty", "net_logged_reward", "peak_iter"]
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in out_rows:
        w.writerow(r)

print(f"wrote {len(out_rows)} rows to {out_path}")
