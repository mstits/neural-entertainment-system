"""Reconstructs the banked 0.767 1-1 control's provenance and compares it
against the v27/v28 recipe. Reads only files already on disk under
checkpoints/ and runs/ -- no training was run to produce this.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def entropy_trace(log_path, every=10):
    rows = []
    pat = re.compile(r"iter (\d+): completed_eps=(\d+).*?clears=(\d+).*?entropy=([\d.]+)")
    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            it, eps, clears, ent = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            rows.append((it, eps, clears, ent))
    return rows


def main():
    print("=== winners/best.json for the seed3/iter140 run (checkpoints/mario_1_1_backward) ===")
    print(json.dumps(json.load(open(REPO / "checkpoints/mario_1_1_backward/winners/best.json")), indent=2))

    print("\n=== run_manifest.json (actual num_envs used) ===")
    print(json.dumps(json.load(open(REPO / "checkpoints/mario_1_1_backward/run_manifest.json")), indent=2))

    print("\n=== eval.jsonl entries naming backward_1_1_seed3_iter140.pt ===")
    for rec in load_jsonl(REPO / "checkpoints/mario_1_1_backward/eval.jsonl"):
        if "seed3_iter140" in str(rec.get("checkpoint", "")):
            keep = {k: v for k, v in rec.items() if k != "max_gx_per_episode"}
            print(keep)

    print("\n=== control metrics.jsonl: iter/entropy/success_rate every 10 (truncated at 159) ===")
    rows = load_jsonl(REPO / "checkpoints/mario_1_1_backward/metrics.jsonl")
    for i in range(0, len(rows), 10):
        r = rows[i]
        print(f"iter {r['generation']:4d}  entropy={r['ppo_entropy']:.4f}  success_rate={r['success_rate']:.3f}  clears={r['vanilla_ppo_clears']}")
    print(f"last row: iter {rows[-1]['generation']}  entropy={rows[-1]['ppo_entropy']:.4f}")

    print("\n=== SAME CONFIG, full 250-iter run, num_envs=60, seed 0 (runs/mario_1_1_backward_seed0.log, 2026-08-08) ===")
    full = entropy_trace(REPO / "runs/mario_1_1_backward_seed0.log")
    print(f"n iters logged: {len(full)}")
    for i in range(0, len(full), 10):
        it, eps, clears, ent = full[i]
        rate = clears / eps if eps else 0.0
        print(f"iter {it:4d}  entropy={ent:.4f}  train_clear_rate={rate:.3f} ({clears}/{eps})")

    peak = max(full, key=lambda r: (r[2] / r[1] if r[1] else 0))
    print(f"\npeak train-time clear rate: iter {peak[0]}  rate={peak[2]/peak[1]:.3f}  entropy={peak[3]:.4f}")

    print("\n=== recovery_assay 60-ep collection (2026-08-24): single-seed reproduction of 0.767 ===")
    man = json.load(open(REPO / "runs/recovery_assay/manifest.json"))
    recs = man["records"]
    cleared = sum(1 for r in recs if r.get("cleared"))
    print(f"cleared {cleared}/{len(recs)} = {cleared/len(recs):.4f} (single eval_seed=0, scripts/recovery_assay.py default)")

    print("\n=== v27 gate: eval protocol actually used (per-seed, per eval-seed) ===")
    gate_path = REPO / "runs/v27_fresh_recovery/gate/seed0_winners-best_es0.json"
    text = gate_path.read_text()
    # first line is a stray engine log line before the JSON in these files
    text = text.split("\n", 1)[1] if not text.lstrip().startswith("{") else text
    d = json.loads(text)
    for k in ("n_episodes", "eval_seed", "eval_workers", "eval_rng", "sticky_prob", "start_jitter"):
        print(k, "=", d.get(k))


if __name__ == "__main__":
    main()
