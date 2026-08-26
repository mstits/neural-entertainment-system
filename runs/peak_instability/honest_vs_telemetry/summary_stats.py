#!/usr/bin/env python3
"""Consolidated summary stats for the honest-vs-telemetry dimension.
Reads comparison.json (from build_comparison.py) and prints/saves:
  - training-success/honest-clear_rate ratio at peak and at final, per run
  - Spearman rank correlation (honest_peak vs training_peak success_rate)
  - relative decline (peak->final) of avg_fitness and of success_rate, per run
Writes summary_stats.json alongside this script.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rank(vals: list[float]) -> list[int]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def main() -> None:
    comp = json.loads((HERE / "comparison.json").read_text())

    rows = []
    for e in comp:
        hp = e["honest_peak"]["clear_rate"]
        hf = e["honest_final"]["clear_rate"]
        tp = e["telemetry_peak"]["success_rate"]
        tf = e["telemetry_final"]["success_rate"]
        afp = e["telemetry_peak"]["avg_fitness"]
        aff = e["telemetry_final"]["avg_fitness"]
        rfp = e["telemetry_peak"].get("reward_forward")
        rff = e["telemetry_final"].get("reward_forward")
        rcp = e["telemetry_peak"].get("reward_completion")
        # reward_completion is ABSENT from the metrics row (not 0.0) in
        # iterations where zero envs completed the level -- the reward-
        # breakdown dict only gains a "completion" key once some env
        # actually triggers it. Missing == 0 clears that iteration
        # (verified against vanilla_ppo_clears==0 for every such row).
        rcf = e["telemetry_final"].get("reward_completion")
        rcf_is_absent = rcf is None
        if rcf_is_absent:
            rcf = 0.0

        ratio_pk = (tp / hp) if hp > 0 else None
        ratio_f = (tf / hf) if hf > 0 else (0.0 if tf == 0 else None)
        avgfit_decl_pct = (aff - afp) / afp * 100 if afp else None
        succ_decl_pct = (tf - tp) / tp * 100 if tp else None

        rows.append({
            "run": e["run"],
            "honest_peak_clear_rate": hp, "train_peak_success_rate": tp,
            "ratio_peak_train_over_honest": ratio_pk,
            "honest_final_clear_rate": hf, "train_final_success_rate": tf,
            "ratio_final_train_over_honest": ratio_f,
            "avg_fitness_peak": afp, "avg_fitness_final": aff,
            "avg_fitness_pct_change_peak_to_final": avgfit_decl_pct,
            "success_rate_pct_change_peak_to_final": succ_decl_pct,
            "reward_forward_peak": rfp, "reward_forward_final": rff,
            "reward_completion_peak": rcp, "reward_completion_final": rcf,
            "reward_completion_final_field_absent_zero_clears": rcf_is_absent,
            "reward_forward_pct_change": ((rff - rfp) / rfp * 100) if rfp else None,
            "reward_completion_pct_change": ((rcf - rcp) / rcp * 100) if rcp else None,
        })

    rh = rank([r["honest_peak_clear_rate"] for r in rows])
    rt = rank([r["train_peak_success_rate"] for r in rows])
    n = len(rh)
    d2 = sum((a - b) ** 2 for a, b in zip(rh, rt))
    spearman = 1 - 6 * d2 / (n * (n**2 - 1))

    finite_ratio_pk = [r["ratio_peak_train_over_honest"] for r in rows if r["ratio_peak_train_over_honest"] is not None]
    avgfit_decls = [r["avg_fitness_pct_change_peak_to_final"] for r in rows if r["avg_fitness_pct_change_peak_to_final"] is not None]
    succ_decls = [r["success_rate_pct_change_peak_to_final"] for r in rows if r["success_rate_pct_change_peak_to_final"] is not None]
    rcomp_decls = [r["reward_completion_pct_change"] for r in rows if r["reward_completion_pct_change"] is not None]
    rfwd_decls = [r["reward_forward_pct_change"] for r in rows if r["reward_forward_pct_change"] is not None]

    out = {
        "per_run": rows,
        "spearman_honest_peak_vs_train_peak_success_rate": spearman,
        "n_runs": n,
        "ratio_peak_train_over_honest_median": statistics.median(finite_ratio_pk),
        "ratio_peak_train_over_honest_range": [min(finite_ratio_pk), max(finite_ratio_pk)],
        "avg_fitness_pct_change_median": statistics.median(avgfit_decls),
        "avg_fitness_pct_change_range": [min(avgfit_decls), max(avgfit_decls)],
        "success_rate_pct_change_median": statistics.median(succ_decls),
        "success_rate_pct_change_range": [min(succ_decls), max(succ_decls)],
        "reward_completion_pct_change_median": statistics.median(rcomp_decls) if rcomp_decls else None,
        "reward_completion_pct_change_range": [min(rcomp_decls), max(rcomp_decls)] if rcomp_decls else None,
        "reward_completion_pct_change_n_of_8": len(rcomp_decls),
        "reward_forward_pct_change_median": statistics.median(rfwd_decls) if rfwd_decls else None,
        "reward_forward_pct_change_range": [min(rfwd_decls), max(rfwd_decls)] if rfwd_decls else None,
    }

    (HERE / "summary_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
