"""How robust is the value-loss -> honest-peak correlation?

The campaign's strongest non-tautological cross-run predictor was
whole-run mean ppo_value_loss vs honest peak (Spearman rho = -0.786, N=8).
Two threats: (1) N=8 leave-one-out fragility, (2) it was the best of
several fields scanned with no multiple-comparison correction.
Also checks the outlier contamination the verifier flagged in policy_loss.
"""
import json, statistics as st
from itertools import combinations

RUNS = [
    ("v27 s0", "mario_1_1_v27_recovery_seed0", 0.040),
    ("v27 s1", "mario_1_1_v27_recovery_seed1", 0.290),
    ("v27 s2", "mario_1_1_v27_recovery_seed2", 0.530),
    ("v27 s3", "mario_1_1_v27_recovery_seed3", 0.170),
    ("v28 s0", "mario_1_1_v28_capacity_seed0", 0.450),
    ("v28 s1", "mario_1_1_v28_capacity_seed1", 0.230),
    ("v28 s2", "mario_1_1_v28_capacity_seed2", 0.370),
    ("v28 s3", "mario_1_1_v28_capacity_seed3", 0.670),
]

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]: j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1): r[order[k]] = avg
        i = j + 1
    return r

def pearson(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = sum((x-ma)**2 for x in a) ** 0.5; db = sum((y-mb)**2 for y in b) ** 0.5
    return num/(da*db) if da and db else 0.0

def spearman(a, b): return pearson(rank(a), rank(b))

honest = [h for _, _, h in RUNS]
series = {}
for label, d, h in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl") if l.strip()]
    for f in ("ppo_value_loss", "ppo_policy_loss", "ppo_entropy"):
        vals = [r.get(f, 0.0) for r in rows if f in r]
        series.setdefault(f, {}).setdefault("mean", []).append(sum(vals)/len(vals))
        series[f].setdefault("median", []).append(st.median(vals))

for f in ("ppo_value_loss", "ppo_policy_loss", "ppo_entropy"):
    for agg in ("mean", "median"):
        xs = series[f][agg]
        full = spearman(xs, honest)
        loo = [spearman([x for j, x in enumerate(xs) if j != i],
                        [y for j, y in enumerate(honest) if j != i]) for i in range(8)]
        print(f"{f:18s} ({agg:6s}) rho={full:+.3f}   LOO range {min(loo):+.3f}..{max(loo):+.3f}   "
              f"|rho|<0.74 (n=8 p>.05) in {sum(1 for r in loo if abs(r) < 0.74)}/8 drops")

print()
print("outlier check on ppo_policy_loss (verifier flagged v27 s3):")
for label, d, _ in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl") if l.strip()]
    vals = [r.get("ppo_policy_loss", 0.0) for r in rows if "ppo_policy_loss" in r]
    print(f"  {label}: mean={sum(vals)/len(vals):10.3f}  median={st.median(vals):.4f}  max={max(vals):10.2f}")
