"""Quantify WHEN vanilla_ppo_rnd_loss collapses relative to each run's peak iter.

Companion to aggregate.py: the field x run matrix flags vanilla_ppo_rnd_loss /
vanilla_ppo_intrinsic_mean as 8/8 sign-consistent (post-peak lower than pre-peak).
This script asks the follow-up question the matrix alone can't answer: does the
RND collapse happen AT the collapse (late, alongside the honest-score crash) or
well before it (early, decoupled)? "Death" = first iter where rnd_loss stays
below THRESH for 5 consecutive logged iters (a sustained floor, not a one-off dip).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = [
    ("v27_seed0", REPO / "checkpoints/mario_1_1_v27_recovery_seed0/metrics.jsonl", 60),
    ("v27_seed1", REPO / "checkpoints/mario_1_1_v27_recovery_seed1/metrics.jsonl", 50),
    ("v27_seed2", REPO / "checkpoints/mario_1_1_v27_recovery_seed2/metrics.jsonl", 90),
    ("v27_seed3", REPO / "checkpoints/mario_1_1_v27_recovery_seed3/metrics.jsonl", 60),
    ("v28_seed0", REPO / "checkpoints/mario_1_1_v28_capacity_seed0/metrics.jsonl", 70),
    ("v28_seed1", REPO / "checkpoints/mario_1_1_v28_capacity_seed1/metrics.jsonl", 60),
    ("v28_seed2", REPO / "checkpoints/mario_1_1_v28_capacity_seed2/metrics.jsonl", 120),
    ("v28_seed3", REPO / "checkpoints/mario_1_1_v28_capacity_seed3/metrics.jsonl", 90),
]
THRESH = 0.0005

deltas = []
print(f"{'run':<12}{'peak_iter':>10}{'rnd@iter0':>12}{'rnd@peak':>12}{'death_iter':>12}{'death-peak':>12}")
for name, path, peak in RUNS:
    rows = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["generation"])
    death_iter = None
    for i in range(len(rows) - 5):
        if all(r["vanilla_ppo_rnd_loss"] < THRESH for r in rows[i:i + 5]):
            death_iter = rows[i]["generation"]
            break
    peak_row = next(r for r in rows if r["generation"] == peak)
    delta = None if death_iter is None else death_iter - peak
    if delta is not None:
        deltas.append(delta)
    print(f"{name:<12}{peak:>10}{rows[0]['vanilla_ppo_rnd_loss']:>12.5f}"
          f"{peak_row['vanilla_ppo_rnd_loss']:>12.6f}{str(death_iter):>12}{str(delta):>12}")

deltas.sort()
print(f"\ndeath_iter - peak_iter across 8 runs: {deltas}")
print(f"median = {deltas[len(deltas)//2]}, all negative (death precedes peak) = {all(d < 0 for d in deltas)}")

print("\nconfound check: any scheduled entropy_coef/rnd_intrinsic_coef override fired in these runs?")
for name, path, _ in RUNS:
    logp = None
    if "v27" in name:
        logp = REPO / f"runs/v27_fresh_recovery/train_{name.split('_')[1]}.log"
    else:
        logp = REPO / f"runs/v28_capacity/train_{name.split('_')[1]}.log"
    text = logp.read_text() if logp.exists() else ""
    hits = sum(1 for kw in ("CONSOLIDATE ARMED", "CONSOLIDATE ABORT", "backward-guard")
               if kw in text)
    print(f"  {name}: schedule/guard log lines = {hits} (0 expected -- entropy_floor=0.0, "
          f"smb_curriculum=false, no entropy_guard block in config)")
