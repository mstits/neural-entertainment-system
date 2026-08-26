"""For each run and each entropy threshold in {0.5, 0.4, 0.3}: find the iter
where ppo_entropy first crosses down through that threshold (linear
interpolation between bracketing logged iters), compare to the
authoritative peak iter (winners/best.json source_iter). Purely
descriptive/quantitative context for the hypothesis-free sweep's #1
non-trivial field -- not a new causal claim, and not this dimension's
hypothesis to own.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

RUNS_PEAK_ITER = {
    "mario_1_1_v27_recovery_seed0": 60,
    "mario_1_1_v27_recovery_seed1": 50,
    "mario_1_1_v27_recovery_seed2": 90,
    "mario_1_1_v27_recovery_seed3": 60,
    "mario_1_1_v28_capacity_seed0": 70,
    "mario_1_1_v28_capacity_seed1": 60,
    "mario_1_1_v28_capacity_seed2": 120,
    "mario_1_1_v28_capacity_seed3": 90,
}

def crossing_iter(gens, ents, thresh):
    for i in range(1, len(gens)):
        if ents[i-1] >= thresh > ents[i]:
            g0, g1 = gens[i-1], gens[i]
            e0, e1 = ents[i-1], ents[i]
            frac = (e0 - thresh) / (e0 - e1)
            return g0 + frac * (g1 - g0)
    return None

data = {}
for run, peak in RUNS_PEAK_ITER.items():
    rows = [json.loads(l) for l in open(REPO / "checkpoints" / run / "metrics.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["generation"])
    gens = [r["generation"] for r in rows]
    ents = [r["ppo_entropy"] for r in rows]
    data[run] = (gens, ents, peak)

for thresh in (0.5, 0.4, 0.3):
    print(f"\n=== entropy crosses {thresh:.2f} ===")
    print(f"{'run':<32}{'peak_iter':>10}{'cross_iter':>12}{'offset':>10}")
    offsets = []
    for run, (gens, ents, peak) in data.items():
        cross = crossing_iter(gens, ents, thresh)
        offset = None if cross is None else cross - peak
        if offset is not None:
            offsets.append(offset)
        print(f"{run:<32}{peak:>10}{('%.1f' % cross) if cross is not None else 'never':>12}{('%.1f' % offset) if offset is not None else '-':>10}")
    if offsets:
        offsets_sorted = sorted(offsets)
        n = len(offsets_sorted)
        median = offsets_sorted[n//2] if n % 2 else (offsets_sorted[n//2-1]+offsets_sorted[n//2])/2
        before = sum(1 for o in offsets if o < 0)
        print(f"n={n}/8 crossed; before-peak={before}/{n}; median offset={median:+.1f}; range {min(offsets):+.1f} to {max(offsets):+.1f}")
