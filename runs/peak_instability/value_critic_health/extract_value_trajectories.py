#!/usr/bin/env python3
"""Extract ppo_value_loss (and neighboring PPO fields) for all 8 from-scratch
runs, aligned to each run's authoritative peak iter (from winners/best.json).

Receipt for the value-function/critic-health dimension of the peak-instability
investigation (see runs/peak_instability/ for sibling dimensions).

Usage:
    .venv/bin/python runs/peak_instability/value_critic_health/extract_value_trajectories.py

Writes:
    runs/peak_instability/value_critic_health/value_trajectories.json  (raw per-run series)
    runs/peak_instability/value_critic_health/summary.csv              (one row per run)
Prints a human-readable summary to stdout.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUNS = {
    "v27_seed0": "mario_1_1_v27_recovery_seed0",
    "v27_seed1": "mario_1_1_v27_recovery_seed1",
    "v27_seed2": "mario_1_1_v27_recovery_seed2",
    "v27_seed3": "mario_1_1_v27_recovery_seed3",
    "v28_seed0": "mario_1_1_v28_capacity_seed0",
    "v28_seed1": "mario_1_1_v28_capacity_seed1",
    "v28_seed2": "mario_1_1_v28_capacity_seed2",
    "v28_seed3": "mario_1_1_v28_capacity_seed3",
}

# From the honest-eval table handed down with the task (peak@iter, honest@peak).
HONEST = {
    "v27_seed0": dict(peak_iter=60, honest_peak=0.040, honest_final=0.020),
    "v27_seed1": dict(peak_iter=50, honest_peak=0.290, honest_final=0.020),
    "v27_seed2": dict(peak_iter=90, honest_peak=0.530, honest_final=0.000),
    "v27_seed3": dict(peak_iter=60, honest_peak=0.170, honest_final=0.010),
    "v28_seed0": dict(peak_iter=70, honest_peak=0.450, honest_final=0.000),
    "v28_seed1": dict(peak_iter=60, honest_peak=0.230, honest_final=0.050),
    "v28_seed2": dict(peak_iter=120, honest_peak=0.370, honest_final=0.000),
    "v28_seed3": dict(peak_iter=90, honest_peak=0.670, honest_final=0.000),
}


def load_metrics(run_dir: str) -> list[dict]:
    path = ROOT / "checkpoints" / run_dir / "metrics.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_best_json(run_dir: str) -> dict:
    path = ROOT / "checkpoints" / run_dir / "winners" / "best.json"
    return json.loads(path.read_text())


def nearest_row(rows: list[dict], gen: int) -> dict | None:
    for r in rows:
        if r.get("generation") == gen:
            return r
    # fall back to nearest by absolute distance
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r.get("generation", -10**9) - gen))


def window_mean(rows: list[dict], field: str, center: int, half: int) -> float | None:
    vals = [
        r[field]
        for r in rows
        if field in r
        and r.get("generation") is not None
        and center - half <= r["generation"] <= center + half
        and r[field] is not None
    ]
    if not vals:
        return None
    return statistics.mean(vals)


def main() -> None:
    all_series: dict[str, dict] = {}
    summary_rows = []

    for label, run_dir in RUNS.items():
        rows = load_metrics(run_dir)
        best = load_best_json(run_dir)
        peak_iter = best["source_iter"]
        assert peak_iter == HONEST[label]["peak_iter"], (
            f"{label}: best.json source_iter={peak_iter} != task table "
            f"peak_iter={HONEST[label]['peak_iter']}"
        )

        series = {
            "generation": [r.get("generation") for r in rows],
            "ppo_value_loss": [r.get("ppo_value_loss") for r in rows],
            "ppo_policy_loss": [r.get("ppo_policy_loss") for r in rows],
            "ppo_loss": [r.get("ppo_loss") for r in rows],
            "ppo_entropy": [r.get("ppo_entropy") for r in rows],
            "success_rate": [r.get("success_rate") for r in rows],
            "vanilla_ppo_max_x": [r.get("vanilla_ppo_max_x") for r in rows],
        }
        all_series[label] = series

        vloss_vals = [v for v in series["ppo_value_loss"] if v is not None]
        first10 = window_mean(rows, "ppo_value_loss", center=rows[0]["generation"] + 9, half=9) if rows else None
        at_peak = window_mean(rows, "ppo_value_loss", center=peak_iter, half=2)
        last10_gen = rows[-1]["generation"] if rows else None
        at_final = window_mean(rows, "ppo_value_loss", center=last10_gen, half=4) if last10_gen else None

        # global min/max and where they occur (iter of min/max)
        vmax_row = max((r for r in rows if r.get("ppo_value_loss") is not None),
                       key=lambda r: r["ppo_value_loss"], default=None)
        vmin_row = min((r for r in rows if r.get("ppo_value_loss") is not None),
                       key=lambda r: r["ppo_value_loss"], default=None)

        # entropy at same checkpoints for cross-reference
        ent_at_peak = window_mean(rows, "ppo_entropy", center=peak_iter, half=2)
        ent_at_final = window_mean(rows, "ppo_entropy", center=last10_gen, half=4) if last10_gen else None

        summary_rows.append(dict(
            run=label,
            peak_iter=peak_iter,
            honest_peak=HONEST[label]["honest_peak"],
            honest_final=HONEST[label]["honest_final"],
            vloss_first10_mean=first10,
            vloss_at_peak_mean=at_peak,
            vloss_at_final_mean=at_final,
            vloss_global_min=vmin_row["ppo_value_loss"] if vmin_row else None,
            vloss_global_min_iter=vmin_row["generation"] if vmin_row else None,
            vloss_global_max=vmax_row["ppo_value_loss"] if vmax_row else None,
            vloss_global_max_iter=vmax_row["generation"] if vmax_row else None,
            ent_at_peak_mean=ent_at_peak,
            ent_at_final_mean=ent_at_final,
        ))

    out_dir = ROOT / "runs" / "peak_instability" / "value_critic_health"
    (out_dir / "value_trajectories.json").write_text(json.dumps(all_series, indent=2))

    csv_path = out_dir / "summary.csv"
    cols = list(summary_rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    print(f"{'run':10s} {'peak_it':>7s} {'hon@pk':>7s} {'hon@fin':>7s} "
          f"{'vl_first10':>11s} {'vl@peak':>9s} {'vl@final':>9s} "
          f"{'vl_min':>8s}@it {'vl_max':>8s}@it {'ent@pk':>7s} {'ent@fin':>7s}")
    for row in summary_rows:
        print(f"{row['run']:10s} {row['peak_iter']:>7d} "
              f"{row['honest_peak']:>7.3f} {row['honest_final']:>7.3f} "
              f"{row['vloss_first10_mean']:>11.4f} "
              f"{row['vloss_at_peak_mean']:>9.4f} "
              f"{row['vloss_at_final_mean']:>9.4f} "
              f"{row['vloss_global_min']:>8.4f}@{row['vloss_global_min_iter']:<4d} "
              f"{row['vloss_global_max']:>8.4f}@{row['vloss_global_max_iter']:<4d} "
              f"{row['ent_at_peak_mean']:>7.4f} {row['ent_at_final_mean']:>7.4f}")

    print(f"\nWrote {out_dir / 'value_trajectories.json'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
