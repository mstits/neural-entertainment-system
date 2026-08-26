#!/usr/bin/env python3
"""Honest-vs-telemetry comparison: does the TRAINING reward signal collapse
at the end of a run, or does it stay flat/high while the honest score goes
to zero?

For each of the 8 v27/v28 runs this pulls:
  - the authoritative peak iter from checkpoints/<run>/winners/best.json
  - the training-telemetry row (metrics.jsonl) at the peak iter AND at
    iter 240 (final) -- avg_fitness, best_fitness, vanilla_ppo_clears,
    success_rate, episodes, vanilla_ppo_max_x, vanilla_ppo_max_world_level,
    ppo_entropy
  - the honest gate receipt (runs/<v27|v28>/gate/*.json) at peak and final,
    pooled across eval_seed 0 and 1 -- clear_rate, mean_return, mean_length

Output: one JSON receipt (comparison.json) and a printed table.

Run: .venv/bin/python runs/peak_instability/honest_vs_telemetry/build_comparison.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUNS = [
    ("v27", "seed0", ROOT / "checkpoints/mario_1_1_v27_recovery_seed0", ROOT / "runs/v27_fresh_recovery/gate"),
    ("v27", "seed1", ROOT / "checkpoints/mario_1_1_v27_recovery_seed1", ROOT / "runs/v27_fresh_recovery/gate"),
    ("v27", "seed2", ROOT / "checkpoints/mario_1_1_v27_recovery_seed2", ROOT / "runs/v27_fresh_recovery/gate"),
    ("v27", "seed3", ROOT / "checkpoints/mario_1_1_v27_recovery_seed3", ROOT / "runs/v27_fresh_recovery/gate"),
    ("v28", "seed0", ROOT / "checkpoints/mario_1_1_v28_capacity_seed0", ROOT / "runs/v28_capacity/gate"),
    ("v28", "seed1", ROOT / "checkpoints/mario_1_1_v28_capacity_seed1", ROOT / "runs/v28_capacity/gate"),
    ("v28", "seed2", ROOT / "checkpoints/mario_1_1_v28_capacity_seed2", ROOT / "runs/v28_capacity/gate"),
    ("v28", "seed3", ROOT / "checkpoints/mario_1_1_v28_capacity_seed3", ROOT / "runs/v28_capacity/gate"),
]

FINAL_ITER = 240

TELEMETRY_FIELDS = [
    "avg_fitness",
    "best_fitness",
    "success_rate",
    "episodes",
    "vanilla_ppo_clears",
    "vanilla_ppo_in_progress",
    "vanilla_ppo_max_x",
    "vanilla_ppo_max_world_level",
    "ppo_entropy",
    "ppo_loss",
    "ppo_policy_loss",
    "ppo_value_loss",
    "vanilla_ppo_intrinsic_mean",
    "reward_forward",
    "reward_completion",
    "reward_time_penalty",
]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def read_gate_json(path: Path) -> dict:
    """Gate files sometimes have a stray nes_core stdout line before the
    JSON object (observed in v27 gate files). Strip anything before the
    first '{' and parse."""
    text = path.read_text()
    idx = text.find("{")
    if idx < 0:
        raise ValueError(f"no JSON object found in {path}")
    return json.loads(text[idx:])


def metrics_row_at(metrics: list[dict], gen: int) -> dict | None:
    for r in metrics:
        if r.get("generation") == gen:
            return r
    return None


def find_gate_files(gate_dir: Path, seed: str, which: str) -> list[Path]:
    """which is 'peak' or 'final'. Handles both v27 naming
    (seedN_winners-best_esE.json / seedN_vanilla_ppo_iter_00240_esE.json)
    and v28 naming (seedN_peak_evalseedE_greedy.json / seedN_final_evalseedE_greedy.json)."""
    if which == "peak":
        pats = [f"{seed}_winners-best_es*.json", f"{seed}_peak_evalseed*_greedy.json"]
    else:
        pats = [f"{seed}_vanilla_ppo_iter_{FINAL_ITER:05d}_es*.json", f"{seed}_final_evalseed*_greedy.json"]
    found: list[Path] = []
    for pat in pats:
        found.extend(sorted(gate_dir.glob(pat)))
    return found


def pooled_honest(gate_dir: Path, seed: str, which: str) -> dict:
    files = find_gate_files(gate_dir, seed, which)
    if not files:
        return {"n_files": 0}
    recs = [read_gate_json(f) for f in files]
    total_eps = sum(r["n_episodes"] for r in recs)
    pooled_clear = sum(r["clear_rate"] * r["n_episodes"] for r in recs) / total_eps
    pooled_return = sum(r["mean_return"] * r["n_episodes"] for r in recs) / total_eps
    pooled_length = sum(r["mean_length"] * r["n_episodes"] for r in recs) / total_eps
    return {
        "n_files": len(files),
        "n_episodes_pooled": total_eps,
        "clear_rate": pooled_clear,
        "mean_return": pooled_return,
        "mean_length": pooled_length,
        "per_eval_seed": [
            {"eval_seed": r.get("eval_seed"), "clear_rate": r["clear_rate"],
             "mean_return": r["mean_return"], "mean_length": r["mean_length"],
             "n_episodes": r["n_episodes"], "checkpoint": r.get("checkpoint")}
            for r in recs
        ],
    }


def main() -> None:
    results = []
    for gen, seed, ckpt_dir, gate_dir in RUNS:
        best_json = json.loads((ckpt_dir / "winners" / "best.json").read_text())
        peak_iter = best_json["source_iter"]
        metrics = read_jsonl(ckpt_dir / "metrics.jsonl")

        peak_row = metrics_row_at(metrics, peak_iter)
        final_row = metrics_row_at(metrics, FINAL_ITER)

        honest_peak = pooled_honest(gate_dir, seed, "peak")
        honest_final = pooled_honest(gate_dir, seed, "final")

        entry = {
            "run": f"{gen}_{seed}",
            "gen": gen,
            "seed": seed,
            "peak_iter": peak_iter,
            "final_iter": FINAL_ITER,
            "n_iters_logged": len(metrics),
            "telemetry_peak": {f: peak_row.get(f) if peak_row else None for f in TELEMETRY_FIELDS},
            "telemetry_final": {f: final_row.get(f) if final_row else None for f in TELEMETRY_FIELDS},
            "honest_peak": honest_peak,
            "honest_final": honest_final,
            "winner_selector_metric_value": best_json["metric_value"],
        }
        results.append(entry)

    out_path = Path(__file__).resolve().parent / "comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")

    # ---- printed summary table ----
    hdr = (
        f"{'run':10} {'pk_it':>5} {'success_rate':>12} {'succ_final':>10} "
        f"{'clears':>7} {'clears_f':>8} {'avg_fit':>9} {'avgfit_f':>9} "
        f"{'entropy':>8} {'entropy_f':>9} {'honest_pk':>9} {'honest_f':>9} "
        f"{'ret_pk':>8} {'ret_f':>8} {'len_pk':>7} {'len_f':>7}"
    )
    print(hdr)
    for e in results:
        tp, tf = e["telemetry_peak"], e["telemetry_final"]
        hp, hf = e["honest_peak"], e["honest_final"]

        def g(d, k, fmt="{:.3f}"):
            v = d.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "NA"

        clears_pk = tp.get("vanilla_ppo_clears")
        eps_pk = tp.get("episodes")
        clears_f = tf.get("vanilla_ppo_clears")
        eps_f = tf.get("episodes")
        clear_frac_pk = f"{clears_pk}/{eps_pk}" if clears_pk is not None else "NA"
        clear_frac_f = f"{clears_f}/{eps_f}" if clears_f is not None else "NA"

        print(
            f"{e['run']:10} {e['peak_iter']:>5} "
            f"{g(tp,'success_rate'):>12} {g(tf,'success_rate'):>10} "
            f"{clear_frac_pk:>7} {clear_frac_f:>8} "
            f"{g(tp,'avg_fitness','{:.0f}'):>9} {g(tf,'avg_fitness','{:.0f}'):>9} "
            f"{g(tp,'ppo_entropy'):>8} {g(tf,'ppo_entropy'):>9} "
            f"{g(hp,'clear_rate'):>9} {g(hf,'clear_rate'):>9} "
            f"{g(hp,'mean_return','{:.0f}'):>8} {g(hf,'mean_return','{:.0f}'):>8} "
            f"{g(hp,'mean_length','{:.0f}'):>7} {g(hf,'mean_length','{:.0f}'):>7}"
        )


if __name__ == "__main__":
    main()
