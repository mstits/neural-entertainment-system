#!/usr/bin/env python3
"""Full-run trajectory analysis, restricted to the post-AT-ENTRANCE regime
(the backward curriculum reaches tau=0, the true cold entrance, by iter
22-29 in every one of the 8 runs -- see runs/peak_instability/
honest_vs_telemetry/at_entrance_iters.txt). From that point on, EVERY
training rollout restart is drawn from the same fixed region (the entrance
+/- window clipped at 0), so any change in the training telemetry after
iter 30 cannot be attributed to the curriculum getting harder -- the
curriculum is already maxed out and constant. This isolates genuine
policy change from curriculum-position confound.

rollout_steps=1024 * num_envs=60 = 61440 fixed env-steps/iteration
(identical across all 8 configs, verified) lets us derive an
approximate *training-time* mean episode length as 61440/episodes,
which is not otherwise logged directly.

Emits trajectory.csv (long format, one row per run x iter>=30) and a
summary printed to stdout: for each run, the iter of max avg_fitness,
max success_rate, and min episodes (== max approx episode length) within
the post-entrance window, compared against the honest-eval peak_iter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOTAL_ENV_STEPS = 1024 * 60  # rollout_steps * num_envs, verified identical across all 8 configs

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

POST_ENTRANCE_FLOOR = 30  # all 8 runs reach AT-ENTRANCE by iter 22-29; 30 is a safety margin


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


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "trajectory.csv"
    rows_out = []

    summary = []
    for gen, seed, ckpt_dir in RUNS:
        metrics = read_jsonl(ckpt_dir / "metrics.jsonl")
        best_json = json.loads((ckpt_dir / "winners" / "best.json").read_text())
        peak_iter = best_json["source_iter"]

        post = [r for r in metrics if r["generation"] >= POST_ENTRANCE_FLOOR]
        for r in post:
            eps = r.get("episodes") or None
            approx_len = TOTAL_ENV_STEPS / eps if eps else None
            rows_out.append({
                "run": f"{gen}_{seed}", "generation": r["generation"],
                "avg_fitness": r.get("avg_fitness"), "best_fitness": r.get("best_fitness"),
                "success_rate": r.get("success_rate"), "episodes": eps,
                "vanilla_ppo_clears": r.get("vanilla_ppo_clears"),
                "approx_mean_ep_len": approx_len, "ppo_entropy": r.get("ppo_entropy"),
            })

        iter_max_fitness = max(post, key=lambda r: r.get("avg_fitness", -1e9))["generation"]
        iter_max_success = max(post, key=lambda r: r.get("success_rate", -1e9))["generation"]
        iter_min_episodes = min(
            (r for r in post if r.get("episodes")), key=lambda r: r["episodes"]
        )["generation"]
        max_len = TOTAL_ENV_STEPS / min(r["episodes"] for r in post if r.get("episodes"))

        # last-30-iter window average as a smoothed "final regime" read
        tail = [r for r in post if r["generation"] >= 210]
        tail_fitness = sum(r.get("avg_fitness", 0) for r in tail) / len(tail)
        tail_success = sum(r.get("success_rate", 0) for r in tail) / len(tail)
        tail_len = TOTAL_ENV_STEPS / (sum(r["episodes"] for r in tail) / len(tail))

        summary.append({
            "run": f"{gen}_{seed}",
            "honest_peak_iter": peak_iter,
            "iter_of_max_avg_fitness": iter_max_fitness,
            "iter_of_max_success_rate": iter_max_success,
            "iter_of_max_approx_ep_len_(min_episodes)": iter_min_episodes,
            "max_approx_ep_len": round(max_len, 1),
            "tail_210_249_avg_fitness": round(tail_fitness, 1),
            "tail_210_249_success_rate": round(tail_success, 4),
            "tail_210_249_approx_ep_len": round(tail_len, 1),
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {csv_path} ({len(rows_out)} rows)")

    print()
    hdr = (
        f"{'run':10} {'honest_pk':>9} {'maxfit_it':>9} {'maxsucc_it':>10} "
        f"{'maxlen_it':>9} {'maxlen':>7} {'tailfit':>8} {'tailsucc':>9} {'taillen':>8}"
    )
    print(hdr)
    for s in summary:
        print(
            f"{s['run']:10} {s['honest_peak_iter']:>9} {s['iter_of_max_avg_fitness']:>9} "
            f"{s['iter_of_max_success_rate']:>10} {s['iter_of_max_approx_ep_len_(min_episodes)']:>9} "
            f"{s['max_approx_ep_len']:>7} {s['tail_210_249_avg_fitness']:>8} "
            f"{s['tail_210_249_success_rate']:>9} {s['tail_210_249_approx_ep_len']:>8}"
        )

    summary_path = Path(__file__).resolve().parent / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
