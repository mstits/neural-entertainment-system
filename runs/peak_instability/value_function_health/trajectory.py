import json
import numpy as np
from pathlib import Path

data = json.loads(Path("runs/peak_instability/value_function_health/all_series.json").read_text())

for name, d in data.items():
    gens = d["gens"]
    vloss = d["series"]["ppo_value_loss"]
    ploss = d["series"]["ppo_policy_loss"]
    ent = d["series"]["ppo_entropy"]
    sr = d["series"]["success_rate"]
    peak_it = d["peak_iter"]
    print(f"\n=== {name}  (peak_iter={peak_it}, honest_peak={d['honest_peak']}) ===")
    print(f"{'iter':>5}{'vloss':>10}{'ploss':>10}{'entropy':>10}{'succ_rate':>11}")
    for i in range(0, len(gens), 10):
        marker = " <-- PEAK" if abs(gens[i]-peak_it) <= 5 else ""
        print(f"{gens[i]:>5}{vloss[i]:>10.3f}{ploss[i]:>10.4f}{ent[i]:>10.4f}{sr[i]:>11.4f}{marker}")
