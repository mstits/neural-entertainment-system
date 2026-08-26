import json
from pathlib import Path

RUNS = {
    "v27_seed0": ("checkpoints/mario_1_1_v27_recovery_seed0", 60, 0.040),
    "v27_seed1": ("checkpoints/mario_1_1_v27_recovery_seed1", 50, 0.290),
    "v27_seed2": ("checkpoints/mario_1_1_v27_recovery_seed2", 90, 0.530),
    "v27_seed3": ("checkpoints/mario_1_1_v27_recovery_seed3", 60, 0.170),
    "v28_seed0": ("checkpoints/mario_1_1_v28_capacity_seed0", 70, 0.450),
    "v28_seed1": ("checkpoints/mario_1_1_v28_capacity_seed1", 60, 0.230),
    "v28_seed2": ("checkpoints/mario_1_1_v28_capacity_seed2", 120, 0.370),
    "v28_seed3": ("checkpoints/mario_1_1_v28_capacity_seed3", 90, 0.670),
}

VALUE_FIELDS = ["ppo_value_loss", "ppo_loss", "ppo_policy_loss", "ppo_entropy",
                "success_rate", "vanilla_ppo_max_x", "avg_fitness", "best_fitness"]

out = {}
for name, (path, peak_iter, honest_peak) in RUNS.items():
    rows = [json.loads(l) for l in open(f"{path}/metrics.jsonl")]
    rows.sort(key=lambda r: r["generation"])
    series = {f: [] for f in VALUE_FIELDS}
    gens = []
    for r in rows:
        gens.append(r["generation"])
        for f in VALUE_FIELDS:
            series[f].append(r.get(f))
    out[name] = {"peak_iter": peak_iter, "honest_peak": honest_peak, "gens": gens, "series": series}

Path("runs/peak_instability/value_function_health/all_series.json").write_text(json.dumps(out, indent=1))
print("wrote all_series.json")
