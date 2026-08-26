#!/usr/bin/env python3
"""Does the 'best_fitness / max_x stays high' appearance require a hidden
retained-capability story, or is it fully explained as a batch-max
statistical artifact of taking max-of-N-stochastic-rollouts per iteration
with N ~ 40-60?

For each run's tail window (iter 210-240, i.e. the terminal regime),
compute:
  - observed per-iteration episode count n (from 'episodes' field) and
    success_rate p (fraction of that iteration's completed episodes that
    cleared)
  - predicted P(at least 1 clear this iteration) = 1 - (1-p)^n, i.e. what
    a null model with NO other structure (independent Bernoulli(p) draws,
    n draws/iteration) predicts for "does at least one rollout clear"
  - observed fraction of tail iterations where vanilla_ppo_max_x reached
    the near-flag threshold (proxy for 'at least one rollout basically
    cleared')

If predicted ~= observed, the persistence of best_fitness/max_x needs NO
additional mechanism -- it is arithmetic given a collapsed-but-nonzero
per-episode success probability and a multi-episode batch per iteration.

Run: .venv/bin/python runs/peak_instability/training_signal_specialization/batch_max_artifact_check.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NEAR_FLAG_THRESH = 3200

RUNS = [
    ("v27", "seed0", ROOT / "checkpoints/mario_1_1_v27_recovery_seed0"),
    ("v27", "seed1", ROOT / "checkpoints/mario_1_1_v27_recovery_seed1"),
    ("v27", "seed2", ROOT / "checkpoints/mario_1_1_v27_recovery_seed2"),
    ("v27", "seed3", ROOT / "checkpoints/mario_1_1_v27_recovery_seed3"),
    ("v28", "seed0", ROOT / "checkpoints/mario_1_1_v28_capacity_seed0"),
    ("v28", "seed1", ROOT / "checkpoints/mario_1_1_v28_capacity_seed1"),
    ("v28", "seed2", ROOT / "checkpoints/mario_1_1_v28_capacity_seed2"),
    ("v28", "seed3", ROOT / "checkpoints/mario_1_1_v28_capacity_seed3"),
]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def main() -> None:
    rows = []
    for gen, seed, ckpt_dir in RUNS:
        run = f"{gen}_{seed}"
        metrics = sorted(read_jsonl(ckpt_dir / "metrics.jsonl"), key=lambda r: r["generation"])
        tail = [r for r in metrics if 210 <= r["generation"] <= 240]

        per_iter_pred = []
        observed_hits = 0
        for r in tail:
            n = r.get("episodes") or 0
            p = r.get("success_rate") or 0.0
            pred = 1 - (1 - p) ** n if n > 0 else 0.0
            per_iter_pred.append(pred)
            if (r.get("vanilla_ppo_max_x") or 0) >= NEAR_FLAG_THRESH:
                observed_hits += 1

        mean_n = statistics.mean(r.get("episodes") or 0 for r in tail)
        mean_p = statistics.mean(r.get("success_rate") or 0.0 for r in tail)
        mean_pred_hit_rate = statistics.mean(per_iter_pred)
        observed_hit_rate = observed_hits / len(tail)

        rows.append({
            "run": run,
            "tail_n_iters": len(tail),
            "tail_mean_episodes_per_iter": round(mean_n, 1),
            "tail_mean_success_rate": round(mean_p, 4),
            "predicted_frac_iters_with_ge1_clear": round(mean_pred_hit_rate, 3),
            "observed_frac_iters_maxx_near_flag": round(observed_hit_rate, 3),
            "gap_observed_minus_predicted": round(observed_hit_rate - mean_pred_hit_rate, 3),
        })

    out = {"tail_window": "iter 210-240", "near_flag_threshold": NEAR_FLAG_THRESH, "per_run": rows}
    gaps = [r["gap_observed_minus_predicted"] for r in rows]
    out["mean_abs_gap"] = round(statistics.mean(abs(g) for g in gaps), 3)
    out["gap_range"] = [min(gaps), max(gaps)]
    (HERE / "batch_max_artifact_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
