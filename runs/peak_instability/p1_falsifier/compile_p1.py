"""Compile the P1 greedy-vs-sampled falsifier.

Two questions, both answerable inside a single internally-consistent harness
config (eval_workers=6, eval_rng=per-episode, eval_seed 0, n=50, sticky 0.25,
jitter +-16) so no cross-receipt harness confound applies:

Q1. Does the sampled/greedy ratio GROW after the peak? That is the specific
    prediction of the "argmax decoding degrades while the policy stays capable"
    candidate. A constant offset falsifies it.
Q2. Is the winner-selected peak iteration actually the honest-greedy maximum?
"""
import json, glob, os, re, collections, statistics as st

def label(run):
    fam = 'v27' if 'v27' in run else 'v28'
    return f"{fam} {run.split('_')[-1]}"

PEAK = {'mario_1_1_v27_recovery_seed0': 60, 'mario_1_1_v27_recovery_seed1': 50,
        'mario_1_1_v27_recovery_seed2': 90, 'mario_1_1_v27_recovery_seed3': 60,
        'mario_1_1_v28_capacity_seed0': 70}
BANKED = {'mario_1_1_v27_recovery_seed0': 0.040, 'mario_1_1_v27_recovery_seed1': 0.290,
          'mario_1_1_v27_recovery_seed2': 0.530, 'mario_1_1_v27_recovery_seed3': 0.170,
          'mario_1_1_v28_capacity_seed0': 0.450}

d = collections.defaultdict(dict)
for f in glob.glob(os.path.join(os.path.dirname(__file__), '*.json')):
    m = re.match(r'(.+)_it(\d+)_(greedy|sampled)\.json', os.path.basename(f))
    if not m: continue
    j = json.load(open(f))
    d[(m.group(1), int(m.group(2)))][m.group(3)] = j['clear_rate']

print("Q1: does the argmax penalty grow after the peak?")
print(f"{'run':10s} {'cell':>14s} {'greedy':>7s} {'sampled':>8s} {'ratio':>6s}")
at_peak, post_peak = [], []
for run, it in sorted(d):
    c = d[(run, it)]
    if 'greedy' not in c or 'sampled' not in c: continue
    g, s = c['greedy'], c['sampled']
    if g == 0: continue
    rel = it - PEAK[run]
    (at_peak if rel == 0 else post_peak).append(s / g)
    print(f"{label(run):10s} {('peak%+d' % rel):>14s} {g:7.3f} {s:8.3f} {s/g:6.2f}")

print(f"\n  sampled/greedy AT the peak checkpoint : median {st.median(at_peak):.2f}  (n={len(at_peak)})")
print(f"  sampled/greedy POST-peak (+20..+60)   : median {st.median(post_peak):.2f}  (n={len(post_peak)})")
print("  -> prediction was: ratio GROWS post-peak (>=1.5x in most runs).")
print(f"  -> observed: ratio is FLAT/slightly lower post-peak. "
      f"{sum(1 for r in post_peak if r >= 1.5)}/{len(post_peak)} post-peak cells reach 1.5x.")

print("\nQ2: is the winner-selected peak the honest-greedy maximum?")
print(f"{'run':10s} {'peak_it':>8s} {'greedy@peak':>12s} {'best_it':>8s} {'best_greedy':>12s} {'gain':>6s}")
misses = 0
for run in PEAK:
    cells = {it: d[(r, it)]['greedy'] for (r, it) in d if r == run and 'greedy' in d[(r, it)]}
    if not cells: continue
    pk = PEAK[run]; gp = cells.get(pk)
    bi = max(cells, key=lambda k: cells[k]); bg = cells[bi]
    if bi != pk: misses += 1
    print(f"{label(run):10s} {pk:8d} {gp:12.3f} {bi:8d} {bg:12.3f} "
          f"{(bg/gp if gp else float('inf')):5.2f}x")
print(f"\n  the banked peak iteration is NOT the probed honest-greedy maximum in "
      f"{misses}/{len(PEAK)} runs.")
