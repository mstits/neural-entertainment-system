"""Is entropy a CLOCK or a DRIVER?

Entropy declines near-monotonically over all 250 iters, so it correlates with
anything that happens late. The discriminating test is whether entropy's decline
RATE changes at the behavioral crash.

For each run: take the crash window [start,end] from crash_vs_entropy.py, and an
equal-length control window immediately BEFORE it (the pre-crash baseline, during
which performance was flat or improving). Compare mean entropy slope (nats/iter).

Driver prediction: entropy decline ACCELERATES in the crash window (ratio >> 1).
Clock prediction:  the two rates are comparable (ratio ~1) or DECELERATE.
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
    return [sum(xs[max(0,i-w//2):min(len(xs),i+w//2+1)]) /
            len(xs[max(0,i-w//2):min(len(xs),i+w//2+1)]) for i in range(len(xs))]

def slope(ys):
    n = len(ys)
    if n < 2: return 0.0
    mx = (n - 1) / 2.0; my = sum(ys) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0

print(f"{'run':8s} {'pre_window':>12s} {'crash_window':>13s} "
      f"{'pre_slope':>10s} {'crash_slope':>12s} {'accel_ratio':>12s}")
ratios = []
for label, d, peak in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["generation"])
    gen = [r["generation"] for r in rows]
    sr = smooth([r.get("success_rate", 0.0) for r in rows])
    ent = smooth([r.get("ppo_entropy", 0.0) for r in rows])

    i_pk = max(range(len(gen)), key=lambda i: sr[i] if gen[i] >= peak else -1)
    hi = sr[i_pk]
    i_start = i_pk
    for i in range(i_pk, len(gen)):
        if sr[i] >= 0.80 * hi: i_start = i
        if sr[i] <= 0.15 * hi: break
    i_end = next((i for i in range(i_start, len(gen)) if sr[i] <= 0.15 * hi), len(gen) - 1)

    L = i_end - i_start
    j_start = max(0, i_start - L)
    pre = slope(ent[j_start:i_start + 1])
    crash = slope(ent[i_start:i_end + 1])
    ratio = crash / pre if pre != 0 else float('nan')
    ratios.append(ratio)
    print(f"{label:8s} {gen[j_start]:5d}-{gen[i_start]:<6d} {gen[i_start]:5d}-{gen[i_end]:<7d} "
          f"{pre:10.5f} {crash:12.5f} {ratio:11.2f}x")

print()
print(f"acceleration ratio (crash slope / pre-crash slope): median {st.median(ratios):.2f}x, "
      f"range {min(ratios):.2f}x to {max(ratios):.2f}x")
print(f"runs where entropy decline ACCELERATED at the crash (>1.5x): "
      f"{sum(1 for r in ratios if r > 1.5)}/8")
print(f"runs where entropy decline DECELERATED at the crash (<1.0x): "
      f"{sum(1 for r in ratios if r < 1.0)}/8")
