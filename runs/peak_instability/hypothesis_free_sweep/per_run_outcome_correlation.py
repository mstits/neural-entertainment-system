"""Per-run whole-trajectory means for the fields that separated the outcome
cohorts, checked against honest peak value across all 8 runs (not just the
pooled 4-vs-4 split) -- does the outcome-cohort signal hold up as a
monotonic-ish relationship at N=8, or was it an artifact of which 4 runs
landed on which side of the pool?
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HONEST_PEAK = {
    "mario_1_1_v27_recovery_seed0": 0.040,
    "mario_1_1_v27_recovery_seed1": 0.290,
    "mario_1_1_v27_recovery_seed2": 0.530,
    "mario_1_1_v27_recovery_seed3": 0.170,
    "mario_1_1_v28_capacity_seed0": 0.450,
    "mario_1_1_v28_capacity_seed1": 0.230,
    "mario_1_1_v28_capacity_seed2": 0.370,
    "mario_1_1_v28_capacity_seed3": 0.670,
}

FIELDS = ["ppo_value_loss", "ppo_entropy", "reward_completion", "success_rate",
          "vanilla_ppo_samples_per_sec", "timing_rnd_intrinsic_ms", "ppo_policy_loss"]

def mean(xs):
    return sum(xs) / len(xs)

def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n**2 - 1))

rows_by_run = {}
for run in HONEST_PEAK:
    rows = [json.loads(l) for l in open(REPO / "checkpoints" / run / "metrics.jsonl") if l.strip()]
    rows_by_run[run] = rows

runs = list(HONEST_PEAK.keys())
peaks = [HONEST_PEAK[r] for r in runs]

print(f"{'field':<30}{'spearman_rho_vs_honest_peak':>28}   per-run means (ordered v27s0..v28s3)")
for field in FIELDS:
    means = []
    for run in runs:
        vals = [r[field] for r in rows_by_run[run] if field in r and r[field] is not None]
        means.append(mean(vals))
    rho = spearman(means, peaks)
    means_str = " ".join(f"{m:8.4g}" for m in means)
    print(f"{field:<30}{rho:>28.3f}   {means_str}")

print("\nrun order:", runs)
print("honest peak:", peaks)
