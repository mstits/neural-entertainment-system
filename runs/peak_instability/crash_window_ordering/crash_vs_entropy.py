"""Temporal ordering: what does entropy do ACROSS the behavioral crash?

For each run, locate the crash window: the contiguous span over which the
smoothed in-training success_rate falls from >=80% of its post-peak local
maximum down to <=15% of it. Then measure entropy at both ends.

Uses per-iteration success_rate from metrics.jsonl (a genuine fresh
per-iteration clear count, not the [backward] trailing line, which is a
known lower bound).
"""
import json, statistics as st

RUNS = [
    ("v27 s0", "mario_1_1_v27_recovery_seed0", 60),
    ("v27 s1", "mario_1_1_v27_recovery_seed1", 50),
    ("v27 s2", "mario_1_1_v27_recovery_seed2", 90),
    ("v27 s3", "mario_1_1_v27_recovery_seed3", 60),
    ("v28 s0", "mario_1_1_v28_capacity_seed0", 70),
    ("v28 s1", "mario_1_1_v28_capacity_seed1", 60),
    ("v28 s2", "mario_1_1_v28_capacity_seed2", 120),
    ("v28 s3", "mario_1_1_v28_capacity_seed3", 90),
]

def smooth(xs, w=5):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w // 2); hi = min(len(xs), i + w // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out

print(f"{'run':8s} {'peak':>5s} {'crash_start':>11s} {'crash_end':>9s} "
      f"{'sr_start':>8s} {'sr_end':>7s} {'ent_start':>9s} {'ent_end':>7s} {'ent_delta_%':>11s}")
deltas = []
for label, d, peak in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["generation"])
    gen = [r["generation"] for r in rows]
    sr = smooth([r.get("success_rate", 0.0) for r in rows])
    ent = smooth([r.get("ppo_entropy", 0.0) for r in rows])

    # local max of smoothed success at/after peak
    i_pk = max(range(len(gen)), key=lambda i: sr[i] if gen[i] >= peak else -1)
    hi = sr[i_pk]
    # crash start: last index at/after i_pk still >= 0.80*hi before the fall
    i_start = i_pk
    for i in range(i_pk, len(gen)):
        if sr[i] >= 0.80 * hi:
            i_start = i
        if sr[i] <= 0.15 * hi:
            break
    i_end = None
    for i in range(i_start, len(gen)):
        if sr[i] <= 0.15 * hi:
            i_end = i; break
    if i_end is None:
        i_end = len(gen) - 1
    dpct = 100.0 * (ent[i_end] - ent[i_start]) / ent[i_start]
    deltas.append(dpct)
    print(f"{label:8s} {peak:5d} {gen[i_start]:11d} {gen[i_end]:9d} "
          f"{sr[i_start]:8.3f} {sr[i_end]:7.3f} {ent[i_start]:9.3f} {ent[i_end]:7.3f} {dpct:+10.1f}%")

print()
print(f"entropy change across the crash window: median {st.median(deltas):+.1f}%, "
      f"range {min(deltas):+.1f}% to {max(deltas):+.1f}%")
print(f"runs where entropy FELL by more than 25% across the crash: "
      f"{sum(1 for d in deltas if d < -25)}/8")
print(f"runs where entropy was flat or rose (>= -10%): {sum(1 for d in deltas if d >= -10)}/8")
