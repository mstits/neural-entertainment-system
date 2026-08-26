#!/usr/bin/env python3
"""ASSIGNED DIMENSION: what the training signal actually rewarded.

The splitting question: does the TRAINING reward signal (avg_fitness,
best_fitness, vanilla_ppo_clears, vanilla_ppo_max_x, success_rate)
collapse across the run the same way honest performance does, or does it
stay flat/high while honest collapses to zero? If the latter, the policy
is specializing to something training rewards and honest does not
(candidates: sampled-not-greedy action selection, no sticky-action
noise, no jitter, warm mid-episode restarts vs cold start). If the
former, it is genuine degradation and the mechanism must explain BOTH
signals collapsing.

Reads all 8 runs' full 250-row metrics.jsonl (no snapshot-only view),
plus the authoritative peak (winners/best.json) and the honest datapoints
supplied in the prompt (peak/final clear_rate; mean_return/mean_length
pulled from the sibling honest_vs_telemetry/comparison.json receipt,
built independently earlier in this investigation from the same gate
JSON files under runs/<gen>/gate/).

Emits full_trajectory.csv (long format, EVERY iteration, no floor) and
prints/saves splitting_summary.json.

Run: .venv/bin/python runs/peak_instability/training_signal_specialization/full_trajectory.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TOTAL_ENV_STEPS = 1024 * 60  # rollout_steps * num_envs, identical across all 8 configs (verified in honest_vs_telemetry)

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

FINAL_ITER = 240
# From the prompt's measured table -- honest, cold/greedy/sticky-0.25/jitter16, 100 eps pooled.
HONEST = {
    "v27_seed0": {"peak_iter": 60, "honest_peak": 0.040, "honest_final": 0.020},
    "v27_seed1": {"peak_iter": 50, "honest_peak": 0.290, "honest_final": 0.020},
    "v27_seed2": {"peak_iter": 90, "honest_peak": 0.530, "honest_final": 0.000},
    "v27_seed3": {"peak_iter": 60, "honest_peak": 0.170, "honest_final": 0.010},
    "v28_seed0": {"peak_iter": 70, "honest_peak": 0.450, "honest_final": 0.000},
    "v28_seed1": {"peak_iter": 60, "honest_peak": 0.230, "honest_final": 0.050},
    "v28_seed2": {"peak_iter": 120, "honest_peak": 0.370, "honest_final": 0.000},
    "v28_seed3": {"peak_iter": 90, "honest_peak": 0.670, "honest_final": 0.000},
}

FLAG_X = 3161  # SMB 1-1 flagpole base x (RAM 0x6D:0x86); observed ceiling in this data is 3266-3267
NEAR_FLAG_THRESH = 3200  # generous: max_x reaching the flag screen at all


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def pct_change(a: float, b: float) -> float | None:
    if a in (0, None):
        return None
    return (b - a) / a * 100.0


def main() -> None:
    comparison = json.loads((ROOT / "runs/peak_instability/honest_vs_telemetry/comparison.json").read_text())
    comp_by_run = {e["run"]: e for e in comparison}

    csv_rows = []
    summary = []

    for gen, seed, ckpt_dir in RUNS:
        run = f"{gen}_{seed}"
        metrics = sorted(read_jsonl(ckpt_dir / "metrics.jsonl"), key=lambda r: r["generation"])
        peak_iter = HONEST[run]["peak_iter"]
        by_gen = {r["generation"]: r for r in metrics}

        for r in metrics:
            it = r["generation"]
            eps = r.get("episodes") or None
            approx_len = TOTAL_ENV_STEPS / eps if eps else None
            csv_rows.append({
                "run": run, "iter": it,
                "avg_fitness": r.get("avg_fitness"),
                "best_fitness": r.get("best_fitness"),
                "success_rate": r.get("success_rate"),
                "episodes": eps,
                "vanilla_ppo_clears": r.get("vanilla_ppo_clears"),
                "vanilla_ppo_in_progress": r.get("vanilla_ppo_in_progress"),
                "vanilla_ppo_max_x": r.get("vanilla_ppo_max_x"),
                "approx_mean_ep_len": approx_len,
                "ppo_entropy": r.get("ppo_entropy"),
                "reward_forward": r.get("reward_forward"),
                "reward_completion": r.get("reward_completion"),
            })

        # --- near-flag retention: does "best of num_envs stochastic rollouts
        # reaches the flag screen" persist after the honest peak, and when
        # (if ever) does it permanently stop? ---
        maxx_series = [(r["generation"], r.get("vanilla_ppo_max_x")) for r in metrics if r.get("vanilla_ppo_max_x") is not None]
        near_flag_iters = [it for it, x in maxx_series if x >= NEAR_FLAG_THRESH]
        frac_near_flag_all = len(near_flag_iters) / len(maxx_series) if maxx_series else None
        post_peak = [(it, x) for it, x in maxx_series if it > peak_iter]
        frac_near_flag_post_peak = (
            sum(1 for it, x in post_peak if x >= NEAR_FLAG_THRESH) / len(post_peak) if post_peak else None
        )
        # last iter (<=240) at which max_x was still near-flag
        near_flag_le_final = [it for it, x in maxx_series if x >= NEAR_FLAG_THRESH and it <= FINAL_ITER]
        last_near_flag_iter = max(near_flag_le_final) if near_flag_le_final else None
        # find the permanent cliff: last iter such that EVERY iter from there
        # to 240 is near-flag (i.e. the point after which it never recovers)
        cliff_iter = None
        for it, x in sorted(post_peak):
            if it > FINAL_ITER:
                break
            tail_from_here = [xx for ii, xx in post_peak if ii >= it and ii <= FINAL_ITER]
            if all(xx >= NEAR_FLAG_THRESH for xx in tail_from_here):
                cliff_iter = it  # last point before permanent below-threshold tail; updated as we scan
        # cliff_iter as computed above is the LAST iter s.t. everything after
        # it (inclusive) to 240 is >=thresh -- i.e. one past the true cliff.
        # Recompute properly: scan from 240 backward for first (highest) iter
        # that is BELOW threshold with nothing near-flag after it.
        real_cliff = None
        sorted_desc = sorted([(it, x) for it, x in maxx_series if it <= FINAL_ITER], reverse=True)
        seen_near_flag_after = False
        for it, x in sorted_desc:
            if x >= NEAR_FLAG_THRESH:
                seen_near_flag_after = True
            else:
                if not seen_near_flag_after:
                    real_cliff = it  # still inside the terminal below-threshold run
        first_iter_of_terminal_collapse = None
        # walk from FINAL_ITER backward, find first iter (highest) that's still >= thresh
        for it, x in sorted_desc:
            if x >= NEAR_FLAG_THRESH:
                first_iter_of_terminal_collapse = it + 1 if it < FINAL_ITER else None
                break

        # --- avg_fitness / reward_forward peak-vs-final (reuse comparison.json for consistency) ---
        c = comp_by_run[run]
        avgfit_pk = c["telemetry_peak"]["avg_fitness"]
        avgfit_fin = c["telemetry_final"]["avg_fitness"]
        bestfit_pk = c["telemetry_peak"]["best_fitness"]
        bestfit_fin = c["telemetry_final"]["best_fitness"]
        clears_pk = c["telemetry_peak"]["vanilla_ppo_clears"]
        clears_fin = c["telemetry_final"]["vanilla_ppo_clears"]
        succ_pk = c["telemetry_peak"]["success_rate"]
        succ_fin = c["telemetry_final"]["success_rate"]
        honest_pk = HONEST[run]["honest_peak"]
        honest_fin = HONEST[run]["honest_final"]
        honest_ret_pk = c["honest_peak"].get("mean_return")
        honest_ret_fin = c["honest_final"].get("mean_return")
        honest_len_pk = c["honest_peak"].get("mean_length")
        honest_len_fin = c["honest_final"].get("mean_length")

        # tail-30 (post-entrance-safe) window flatness check for avg_fitness:
        # is the SIGNAL still falling at iter 240, or has it plateaued?
        tail_early = [r["avg_fitness"] for r in metrics if 180 <= r["generation"] < 210]
        tail_late = [r["avg_fitness"] for r in metrics if 210 <= r["generation"] <= 240]
        tail_still_falling_pct = pct_change(statistics.mean(tail_early), statistics.mean(tail_late))

        summary.append({
            "run": run,
            "peak_iter": peak_iter,
            "final_iter": FINAL_ITER,
            "honest_clear_rate_peak": honest_pk,
            "honest_clear_rate_final": honest_fin,
            "honest_clear_rate_pct_change": pct_change(honest_pk, honest_fin) if honest_pk else None,
            "honest_mean_return_peak": honest_ret_pk,
            "honest_mean_return_final": honest_ret_fin,
            "honest_mean_return_pct_change": pct_change(honest_ret_pk, honest_ret_fin) if honest_ret_pk else None,
            "honest_mean_length_peak": honest_len_pk,
            "honest_mean_length_final": honest_len_fin,
            "honest_mean_length_pct_change": pct_change(honest_len_pk, honest_len_fin) if honest_len_pk else None,
            "train_success_rate_peak": succ_pk,
            "train_success_rate_final": succ_fin,
            "train_success_rate_pct_change": pct_change(succ_pk, succ_fin),
            "train_avg_fitness_peak": avgfit_pk,
            "train_avg_fitness_final": avgfit_fin,
            "train_avg_fitness_pct_change": pct_change(avgfit_pk, avgfit_fin),
            "train_best_fitness_peak": bestfit_pk,
            "train_best_fitness_final": bestfit_fin,
            "train_best_fitness_pct_change": pct_change(bestfit_pk, bestfit_fin),
            "train_clears_this_iter_peak": clears_pk,
            "train_clears_this_iter_final": clears_fin,
            "train_clears_pct_change": pct_change(clears_pk, clears_fin),
            "avg_fitness_still_falling_180_210_vs_210_240_pct": tail_still_falling_pct,
            "frac_iters_maxx_near_flag_ALL_250": frac_near_flag_all,
            "frac_iters_maxx_near_flag_POST_PEAK_to_240": frac_near_flag_post_peak,
            "last_iter_le240_maxx_near_flag": last_near_flag_iter,
            "first_iter_of_terminal_maxx_collapse_le240": first_iter_of_terminal_collapse,
        })

    # write csv
    with open(HERE / "full_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # aggregate stats across the 8 runs
    def col(name):
        return [s[name] for s in summary if s[name] is not None]

    agg = {
        "n_runs": len(summary),
        "train_success_rate_pct_change_median": statistics.median(col("train_success_rate_pct_change")),
        "train_success_rate_pct_change_range": [min(col("train_success_rate_pct_change")), max(col("train_success_rate_pct_change"))],
        "train_avg_fitness_pct_change_median": statistics.median(col("train_avg_fitness_pct_change")),
        "train_avg_fitness_pct_change_range": [min(col("train_avg_fitness_pct_change")), max(col("train_avg_fitness_pct_change"))],
        "train_best_fitness_pct_change_median": statistics.median(col("train_best_fitness_pct_change")),
        "train_best_fitness_pct_change_range": [min(col("train_best_fitness_pct_change")), max(col("train_best_fitness_pct_change"))],
        "train_clears_pct_change_median": statistics.median(col("train_clears_pct_change")),
        "train_clears_pct_change_range": [min(col("train_clears_pct_change")), max(col("train_clears_pct_change"))],
        "honest_mean_return_pct_change_median": statistics.median(col("honest_mean_return_pct_change")),
        "honest_mean_return_pct_change_range": [min(col("honest_mean_return_pct_change")), max(col("honest_mean_return_pct_change"))],
        "honest_mean_length_pct_change_median": statistics.median(col("honest_mean_length_pct_change")),
        "honest_mean_length_pct_change_range": [min(col("honest_mean_length_pct_change")), max(col("honest_mean_length_pct_change"))],
        "frac_iters_maxx_near_flag_POST_PEAK_to_240_median": statistics.median(col("frac_iters_maxx_near_flag_POST_PEAK_to_240")),
        "frac_iters_maxx_near_flag_POST_PEAK_to_240_range": [min(col("frac_iters_maxx_near_flag_POST_PEAK_to_240")), max(col("frac_iters_maxx_near_flag_POST_PEAK_to_240"))],
        "first_iter_of_terminal_maxx_collapse_le240_values": col("first_iter_of_terminal_maxx_collapse_le240"),
        "n_runs_with_terminal_maxx_collapse_before_240": sum(1 for s in summary if s["first_iter_of_terminal_maxx_collapse_le240"] is not None),
    }

    out = {"per_run": summary, "aggregate": agg}
    (HERE / "splitting_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
