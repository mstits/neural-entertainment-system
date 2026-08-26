"""Width comparison, local-slope-at-peak check, and counterexample dump.
Depends on raw_extract.json / crossings.json / floor.json from the
sibling scripts in this directory."""
import json, statistics as st

DATA = json.load(open("runs/peak_instability/entropy_collapse_refutation/raw_extract.json"))
crossings = json.load(open("runs/peak_instability/entropy_collapse_refutation/crossings.json"))
floor = json.load(open("runs/peak_instability/entropy_collapse_refutation/floor.json"))

def grp(rows, key):
    v27 = [r[key] for r in rows if r["name"].startswith("v27")]
    v28 = [r[key] for r in rows if r["name"].startswith("v28")]
    return v27, v28

print("=== width comparison ===")
for key in ("peak_it", "ent_at_peak"):
    v27, v28 = grp(crossings, key)
    print(key, "v27(48k)", v27, "mean", round(st.mean(v27), 4),
          "| v28(72k)", v28, "mean", round(st.mean(v28), 4))
for t in ("0.5", "0.3", "0.1"):
    v27 = [r["crossings"][t] for r in crossings if r["name"].startswith("v27")]
    v28 = [r["crossings"][t] for r in crossings if r["name"].startswith("v28")]
    print(f"cross<={t} abs-iter v27 mean {st.mean(v27):.1f} | v28 mean {st.mean(v28):.1f}")

print()
print("=== local entropy slope pre/post the authoritative peak (10-iter window) ===")

def ent_at(run, i):
    return run["metrics"][str(i)]["ppo_entropy"]

for name, run in DATA.items():
    p = run["best"]["source_iter"]
    lo, hi = max(0, p - 10), min(249, p + 10)
    slope_pre = (ent_at(run, p) - ent_at(run, lo)) / max(1, p - lo)
    slope_post = (ent_at(run, hi) - ent_at(run, p)) / max(1, hi - p)
    print(f"{name:10} peak={p:4} slope_pre10={slope_pre:+.5f} slope_post10={slope_post:+.5f} "
          f"accel={slope_post/slope_pre if slope_pre else float('nan'):.2f}x")

print()
print("=== counterexample: v28_seed2 entropy vs its own trailing-rate proxy, it75-it123 ===")
run = DATA["v28_seed2"]
for i in range(75, 124, 3):
    s = str(i)
    m, l = run["metrics"][s], run["log"][s]
    print(f"  it{i:4} entropy={m['ppo_entropy']:.4f}  trail={l['trail_n']}/{l['trail_d']}={l['trail_rate']:.3f}")
