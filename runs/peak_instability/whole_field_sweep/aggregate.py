"""Aggregate scripts/analyze.py's pre-peak vs post-peak divergence across all 8 runs.

Reuses analyze.py's read_jsonl/split_rows/diverge exactly (no re-implementation of the
effect-size math) so the aggregate table is guaranteed consistent with the per-run receipts
in per_run/*.txt. Writes a field x run effect-size matrix plus a summary ranked by
sign-agreement then median |effect|.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import analyze  # noqa: E402

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

# honest score at peak vs at final, from the orchestrator brief -- used only to sort the
# outcome table, not fed into the per-run divergence math itself.
HONEST_PEAK = {
    "v27_seed0": 0.040, "v27_seed1": 0.290, "v27_seed2": 0.530, "v27_seed3": 0.170,
    "v28_seed0": 0.450, "v28_seed1": 0.230, "v28_seed2": 0.370, "v28_seed3": 0.670,
}

TRIVIAL = {"generation", "timestamp"}

field_to_run_effect: dict[str, dict[str, float]] = {}
per_run_results = {}

for name, path, peak in RUNS:
    rows = analyze.read_jsonl(path)
    cohort_a, cohort_b, excluded = analyze.split_rows(rows, "generation", ">", float(peak))
    results = analyze.diverge(cohort_a, cohort_b)
    per_run_results[name] = {r.field: r for r in results}
    for r in results:
        field_to_run_effect.setdefault(r.field, {})[name] = r.effect_size

run_names = [n for n, _, _ in RUNS]

def sign(x):
    if x != x:  # nan
        return 0
    if x == float("inf"):
        return 1
    if x == float("-inf"):
        return -1
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

summary_rows = []
for field, per_run in field_to_run_effect.items():
    effects = [per_run.get(n, float("nan")) for n in run_names]
    signs = [sign(e) for e in effects]
    n_present = sum(1 for e in effects if e == e)  # not nan
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    agree = max(pos, neg)
    finite_abs = [abs(e) for e in effects if e == e and e not in (float("inf"), float("-inf"))]
    median_abs = sorted(finite_abs)[len(finite_abs)//2] if finite_abs else float("nan")
    summary_rows.append({
        "field": field,
        "n_present": n_present,
        "agree": agree,
        "pos": pos,
        "neg": neg,
        "median_abs_effect": median_abs,
        "effects": effects,
    })

# rank: sign-agreement first (how many of 8 runs move the same direction), then median |effect|
summary_rows.sort(key=lambda r: (r["agree"], r["median_abs_effect"] if r["median_abs_effect"]==r["median_abs_effect"] else 0), reverse=True)

print("=" * 140)
print("PRE-PEAK vs POST-PEAK: field x run effect-size matrix (A=post-peak generation>peak, B=pre-peak; sign = post-peak minus pre-peak)")
print("=" * 140)
header = f"{'field':<30}{'present':>8}{'agree':>7}{'pos':>5}{'neg':>5}{'med|eff|':>10}  " + "".join(f"{n:>11}" for n in run_names)
print(header)
print("-" * len(header))
for r in summary_rows:
    trivial_tag = " [TRIVIAL]" if r["field"] in TRIVIAL else ""
    effstr = "".join(f"{analyze._fmt(e,'.3g'):>11}" for e in r["effects"])
    print(f"{r['field']:<30}{r['n_present']:>8}{r['agree']:>7}{r['pos']:>5}{r['neg']:>5}{analyze._fmt(r['median_abs_effect'],'.3g'):>10}  {effstr}{trivial_tag}")

# ---------------------------------------------------------------------------
# Outcome split: top-4 peak-honest-score runs (pooled) vs bottom-4 (pooled)
# ---------------------------------------------------------------------------
print()
print("=" * 140)
print("OUTCOME SPLIT: pooled rows from top-4-peak runs vs pooled rows from bottom-4-peak runs")
print("=" * 140)
ranked_by_honest = sorted(run_names, key=lambda n: HONEST_PEAK[n], reverse=True)
top4 = ranked_by_honest[:4]
bot4 = ranked_by_honest[4:]
print(f"top4 (by honest@peak): {[(n, HONEST_PEAK[n]) for n in top4]}")
print(f"bot4 (by honest@peak): {[(n, HONEST_PEAK[n]) for n in bot4]}")

pooled_top: list = []
pooled_bot: list = []
for name, path, peak in RUNS:
    rows = analyze.read_jsonl(path)
    if name in top4:
        pooled_top.extend(rows)
    else:
        pooled_bot.extend(rows)

outcome_results = analyze.diverge(pooled_top, pooled_bot)
outcome_ranked = sorted(outcome_results, key=lambda r: abs(r.effect_size), reverse=True)
header2 = f"{'field':<32}{'n_top/tot':>11}{'n_bot/tot':>11}{'mean_top':>12}{'mean_bot':>12}{'Δmean':>11}{'effect':>9}"
print(header2)
print("-" * len(header2))
for r in outcome_ranked:
    trivial_tag = " [TRIVIAL]" if r.field in TRIVIAL else ""
    print(f"{r.field:<32}{f'{r.n_a}/{r.total_a}':>11}{f'{r.n_b}/{r.total_b}':>11}{analyze._fmt(r.mean_a):>12}{analyze._fmt(r.mean_b):>12}{analyze._fmt(r.delta_mean):>11}{analyze._fmt(r.effect_size,'.3g'):>9}{trivial_tag}")

# dump json for downstream use
out = {
    "pre_vs_post_peak": [
        {"field": r["field"], "n_present": r["n_present"], "agree": r["agree"],
         "pos": r["pos"], "neg": r["neg"], "median_abs_effect": r["median_abs_effect"],
         "effects_by_run": dict(zip(run_names, r["effects"]))}
        for r in summary_rows
    ],
    "outcome_top4_vs_bot4": [
        {"field": r.field, "effect_size": r.effect_size, "mean_top": r.mean_a, "mean_bot": r.mean_b,
         "n_top": r.n_a, "n_bot": r.n_b}
        for r in outcome_ranked
    ],
    "top4_runs": top4,
    "bot4_runs": bot4,
}
with open(Path(__file__).parent / "aggregate_output.json", "w") as f:
    json.dump(out, f, indent=2)
