import json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]

def load_json_maybe_prefixed(path):
    text = path.read_text()
    idx = text.find("{")
    return json.loads(text[idx:])

def pooled_clear_rate(files):
    n_total = 0
    n_clear = 0
    means = []
    for f in files:
        d = load_json_maybe_prefixed(f)
        n = d["n_episodes"]
        cr = d["clear_rate"]
        n_total += n
        n_clear += cr * n
        means.append((f.name, n, cr))
    return n_clear / n_total, means

print("=== v27 (winners-best = peak) ===")
for s in range(4):
    files = [REPO/f"runs/v27_fresh_recovery/gate/seed{s}_winners-best_es0.json",
             REPO/f"runs/v27_fresh_recovery/gate/seed{s}_winners-best_es1.json"]
    pooled, means = pooled_clear_rate(files)
    print(f"seed{s} peak pooled clear_rate={pooled:.3f}  detail={means}")

print("=== v27 (iter_00240 = final) ===")
for s in range(4):
    files = [REPO/f"runs/v27_fresh_recovery/gate/seed{s}_vanilla_ppo_iter_00240_es0.json",
             REPO/f"runs/v27_fresh_recovery/gate/seed{s}_vanilla_ppo_iter_00240_es1.json"]
    pooled, means = pooled_clear_rate(files)
    print(f"seed{s} final pooled clear_rate={pooled:.3f}  detail={means}")

print("=== v28 (peak) ===")
for s in range(4):
    files = [REPO/f"runs/v28_capacity/gate/seed{s}_peak_evalseed0_greedy.json",
             REPO/f"runs/v28_capacity/gate/seed{s}_peak_evalseed1_greedy.json"]
    pooled, means = pooled_clear_rate(files)
    print(f"seed{s} peak pooled clear_rate={pooled:.3f}  detail={means}")

print("=== v28 (final) ===")
for s in range(4):
    files = [REPO/f"runs/v28_capacity/gate/seed{s}_final_evalseed0_greedy.json",
             REPO/f"runs/v28_capacity/gate/seed{s}_final_evalseed1_greedy.json"]
    pooled, means = pooled_clear_rate(files)
    print(f"seed{s} final pooled clear_rate={pooled:.3f}  detail={means}")
