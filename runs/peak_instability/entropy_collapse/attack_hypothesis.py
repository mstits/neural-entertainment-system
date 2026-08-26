"""Second pass: attack the entropy-collapse hypothesis directly.

(a) stable (non-noisy) crossings, to separate a real regime change from a
    single noisy dip across the threshold.
(b) does entropy keep falling in the post-peak / near-zero-honest regime,
    or does it flatten out (decoupling entropy decline from the ongoing
    capability collapse)?
(c) 48k vs 72k entropy trajectory shape, and whether it tracks the
    (measured, real) peak difference between the widths.
(d) hunt for counterexamples: entropy already low AT the peak (peak
    performance achieved without high entropy), or entropy still
    comfortably high with performance already gone.

Run: .venv/bin/python runs/peak_instability/entropy_collapse/attack_hypothesis.py
Reads entropy_trajectories.json written by extract_entropy.py.
"""
from __future__ import annotations
import json
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = json.loads((REPO / "runs/peak_instability/entropy_collapse/entropy_trajectories.json").read_text())

RUNS = list(DATA.keys())


def value_at(gens, series, target_gen):
    for g, v in zip(gens, series):
        if g == target_gen:
            return v
    return None


def slope(gens, series, lo, hi):
    """OLS slope of series vs generation over [lo, hi] inclusive."""
    xs, ys = [], []
    for g, v in zip(gens, series):
        if lo <= g <= hi and v is not None:
            xs.append(g)
            ys.append(v)
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


print("=" * 100)
print("(a) STABLE crossings -- last generation entropy pops ABOVE threshold, +1")
print("    (None if it never stabilizes below, i.e. keeps popping back up)")
print("=" * 100)
stable_lags = {0.5: [], 0.3: [], 0.1: []}
for name in RUNS:
    d = DATA[name]
    gens = d["generations"]
    ent = d["ppo_entropy"]
    peak_it = d["peak_iter"]
    line = [name]
    for th in (0.5, 0.3, 0.1):
        above_after = [g for g, e in zip(gens, ent) if e is not None and e > th]
        if not above_after:
            stable_g = gens[0]  # started below threshold entirely (didn't happen here but guard)
        else:
            last_above = max(above_after)
            # first gen strictly after last_above
            candidates = [g for g in gens if g > last_above]
            stable_g = candidates[0] if candidates else None
        lag = (stable_g - peak_it) if stable_g is not None else None
        if lag is not None:
            stable_lags[th].append(lag)
        line.append(f"th={th}: stable_g={stable_g} lag={lag}")
    print(" | ".join(line))

print()
for th, vals in stable_lags.items():
    if vals:
        print(f"STABLE lag th={th}: n={len(vals)} vals={sorted(vals)} "
              f"median={stats.median(vals):.0f} min={min(vals)} max={max(vals)}")

print()
print("=" * 100)
print("(b) does entropy keep dropping AFTER honest capability is already gone?")
print("    honest_final is measured at iter239/240 (~0 for 6/8 runs). Compare")
print("    entropy slope in [peak,peak+60] (collapse-in-progress) vs [150,239]")
print("    (deep post-collapse, honest already near floor for most runs).")
print("=" * 100)
for name in RUNS:
    d = DATA[name]
    gens = d["generations"]
    ent = d["ppo_entropy"]
    peak_it = d["peak_iter"]
    e_peak = d["entropy_at_peak"]
    e150 = value_at(gens, ent, 150)
    e200 = value_at(gens, ent, 200)
    e_final = d["entropy_at_final"]
    s_early = slope(gens, ent, peak_it, min(peak_it + 60, 239))
    s_late = slope(gens, ent, 150, 239)
    print(f"{name:10} honest(peak->final)={d['honest_peak']:.3f}->{d['honest_final']:.3f}  "
          f"ent(peak={e_peak:.3f} -> 150={e150} -> 200={e200} -> final={e_final:.3f})  "
          f"slope[peak,+60]={s_early:.5f}/iter  slope[150,239]={s_late:.5f}/iter  "
          f"late/early ratio={abs(s_late/s_early) if s_early else float('nan'):.2f}")

print()
print("=" * 100)
print("(c) 48k (v27) vs 72k (v28): entropy trajectory shape at matched generations")
print("=" * 100)
for gen_check in (0, 30, 60, 90, 120, 150, 180, 210, 239):
    v27_vals = [value_at(DATA[n]["generations"], DATA[n]["ppo_entropy"], gen_check)
                for n in RUNS if n.startswith("v27")]
    v28_vals = [value_at(DATA[n]["generations"], DATA[n]["ppo_entropy"], gen_check)
                for n in RUNS if n.startswith("v28")]
    v27_vals = [v for v in v27_vals if v is not None]
    v28_vals = [v for v in v28_vals if v is not None]
    m27 = stats.mean(v27_vals) if v27_vals else float("nan")
    m28 = stats.mean(v28_vals) if v28_vals else float("nan")
    print(f"gen={gen_check:4d}  v27(48k) mean={m27:.4f}  v28(72k) mean={m28:.4f}  diff(72-48)={m28-m27:+.4f}")

print()
peaks27 = [DATA[n]["honest_peak"] for n in RUNS if n.startswith("v27")]
peaks28 = [DATA[n]["honest_peak"] for n in RUNS if n.startswith("v28")]
print(f"honest_peak best-of-4: v27={max(peaks27):.3f}  v28={max(peaks28):.3f}")
print(f"honest_peak mean-of-4: v27={stats.mean(peaks27):.3f}  v28={stats.mean(peaks28):.3f}")

print()
print("=" * 100)
print("(d) counterexample hunt: entropy_at_peak vs honest_peak, sorted by entropy_at_peak")
print("=" * 100)
rows = sorted(RUNS, key=lambda n: DATA[n]["entropy_at_peak"])
for n in rows:
    d = DATA[n]
    print(f"{n:10} entropy_at_peak={d['entropy_at_peak']:.4f}  honest_peak={d['honest_peak']:.3f}  "
          f"peak_iter={d['peak_iter']}  (below 0.5? {'YES' if d['entropy_at_peak']<0.5 else 'no'})")

# quick Pearson r between entropy_at_peak and honest_peak across the 8 runs
xs = [DATA[n]["entropy_at_peak"] for n in RUNS]
ys = [DATA[n]["honest_peak"] for n in RUNS]
n = len(xs)
mx, my = sum(xs)/n, sum(ys)/n
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
sx = (sum((x-mx)**2 for x in xs))**0.5
sy = (sum((y-my)**2 for y in ys))**0.5
r = cov/(sx*sy) if sx and sy else float("nan")
print(f"\nPearson r(entropy_at_peak, honest_peak) across N=8 runs = {r:.3f}")
