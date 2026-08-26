"""Pool full 250-iter trajectories into two cohorts by outcome (honest peak
value from checkpoints/<run>/winners/best.json), not by time. High-4 vs
low-4 runs, all rows tagged with cohort + source run, written as one jsonl
for analyze.py / diverge() to consume with --split "cohort==high".
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

RUNS = [
    "mario_1_1_v27_recovery_seed0",
    "mario_1_1_v27_recovery_seed1",
    "mario_1_1_v27_recovery_seed2",
    "mario_1_1_v27_recovery_seed3",
    "mario_1_1_v28_capacity_seed0",
    "mario_1_1_v28_capacity_seed1",
    "mario_1_1_v28_capacity_seed2",
    "mario_1_1_v28_capacity_seed3",
]

# NOTE: checkpoints/<run>/winners/best.json's "metric_value" field
# (entrance_trailing_rate) is a training-time proxy the winner-selector
# uses to decide which checkpoint to keep -- it is NOT the honest
# sticky-eval number from the phenomenon table (values 0.87-1.0, barely
# discriminating). The honest numbers (0.04-0.67, cold/greedy/sticky
# 0.25/jitter+-16, pooled over 100 eps across eval seeds {0,1}) live in
# the gate/*.json receipts and were re-derived + cross-checked against
# the assignment brief's table before use here -- see verify step in
# the writeup. Hardcoded because the two run families use different
# gate filename conventions (v27: seed{s}_winners-best_es{0,1}.json;
# v28: seed{s}_peak_evalseed{0,1}_greedy.json) and re-deriving inline
# would just be this same lookup with extra steps.
peaks = {
    "mario_1_1_v27_recovery_seed0": 0.040,
    "mario_1_1_v27_recovery_seed1": 0.290,
    "mario_1_1_v27_recovery_seed2": 0.530,
    "mario_1_1_v27_recovery_seed3": 0.170,
    "mario_1_1_v28_capacity_seed0": 0.450,
    "mario_1_1_v28_capacity_seed1": 0.230,
    "mario_1_1_v28_capacity_seed2": 0.370,
    "mario_1_1_v28_capacity_seed3": 0.670,
}

ranked = sorted(RUNS, key=lambda r: peaks[r])
low4 = ranked[:4]
high4 = ranked[4:]
print("low4 (lowest honest peak):", [(r, round(peaks[r], 3)) for r in low4])
print("high4 (highest honest peak):", [(r, round(peaks[r], 3)) for r in high4])

out_path = REPO / "runs/peak_instability/hypothesis_free_sweep/outcome_split/pooled.jsonl"
n_low = n_high = 0
with open(out_path, "w") as out:
    for run in low4 + high4:
        cohort = "low" if run in low4 else "high"
        rows = []
        with open(REPO / "checkpoints" / run / "metrics.jsonl") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                row["cohort"] = cohort
                row["source_run"] = run
                rows.append(row)
                out.write(json.dumps(row) + "\n")
        if cohort == "low":
            n_low += len(rows)
        else:
            n_high += len(rows)
print(f"wrote {out_path}: low={n_low} rows, high={n_high} rows")
