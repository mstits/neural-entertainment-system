"""Parse the [vanilla_ppo] iter N: ... training log lines for all 8
v27/v28 runs into one tidy CSV, plus pull per-episode max_gx arrays out
of eval.jsonl (peak vs final checkpoint) for a return-variance proxy.

Everything here reads existing artifacts on disk. No training compute.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUNS = [
    ("v27", 0, ROOT / "runs/v27_fresh_recovery/train_seed0.log",
     ROOT / "checkpoints/mario_1_1_v27_recovery_seed0"),
    ("v27", 1, ROOT / "runs/v27_fresh_recovery/train_seed1.log",
     ROOT / "checkpoints/mario_1_1_v27_recovery_seed1"),
    ("v27", 2, ROOT / "runs/v27_fresh_recovery/train_seed2.log",
     ROOT / "checkpoints/mario_1_1_v27_recovery_seed2"),
    ("v27", 3, ROOT / "runs/v27_fresh_recovery/train_seed3.log",
     ROOT / "checkpoints/mario_1_1_v27_recovery_seed3"),
    ("v28", 0, ROOT / "runs/v28_capacity/train_seed0.log",
     ROOT / "checkpoints/mario_1_1_v28_capacity_seed0"),
    ("v28", 1, ROOT / "runs/v28_capacity/train_seed1.log",
     ROOT / "checkpoints/mario_1_1_v28_capacity_seed1"),
    ("v28", 2, ROOT / "runs/v28_capacity/train_seed2.log",
     ROOT / "checkpoints/mario_1_1_v28_capacity_seed2"),
    ("v28", 3, ROOT / "runs/v28_capacity/train_seed3.log",
     ROOT / "checkpoints/mario_1_1_v28_capacity_seed3"),
]

LINE_RE = re.compile(
    r"\[vanilla_ppo\] iter (?P<iter>\d+): completed_eps=(?P<completed_eps>\d+)"
    r"\s+mean_return=(?P<mean_return>[-\d.]+)\s+mean_len=(?P<mean_len>[-\d.]+)"
    r"\s+in_progress=(?P<in_progress>\d+)\s+ip_return=(?P<ip_return>[-\d.]+)"
    r"\s+ip_len=(?P<ip_len>[-\d.]+)\s+clears=(?P<clears>\d+)"
    r"\s+loss=(?P<loss>[-\d.]+)\s+policy=(?P<policy>[-\d.]+)"
    r"\s+value=(?P<value>[-\d.]+)\s+entropy=(?P<entropy>[-\d.]+)"
)

BACKWARD_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): tau=(?P<tau>\d+)/(?P<tau_max>\d+) "
    r"\(step (?P<step>\d+) frame (?P<frame>\d+) gx (?P<gx>\d+)\) "
    r"trailing (?P<trail_n>\d+)/(?P<trail_d>\d+)=(?P<trail_rate>[\d.]+) "
    r"\(advance at >=[\d.]+ over \d+\) advances=(?P<advances>\d+)"
    r"(?:\s+AT-ENTRANCE)?\s*\|\s*entrance (?P<ent_n>\d+)/(?P<ent_d>\d+)="
    r"(?P<ent_rate>[\d.]+)\s+\|\s+truncated (?P<truncated>\d+)"
)


def parse_run(tag, seed, log_path, ckpt_dir):
    rows = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = LINE_RE.search(line)
        if m:
            d = m.groupdict()
            it = int(d["iter"])
            row = rows.setdefault(it, {"run": tag, "seed": seed, "iter": it})
            row.update({
                "completed_eps": int(d["completed_eps"]),
                "mean_return": float(d["mean_return"]),
                "mean_len": float(d["mean_len"]),
                "in_progress": int(d["in_progress"]),
                "ip_return": float(d["ip_return"]),
                "ip_len": float(d["ip_len"]),
                "clears": int(d["clears"]),
                "loss": float(d["loss"]),
                "policy_loss": float(d["policy"]),
                "value_loss": float(d["value"]),
                "entropy": float(d["entropy"]),
                "at_entrance": "AT-ENTRANCE" in line,
            })
            continue
        m2 = BACKWARD_RE.search(line)
        if m2:
            d = m2.groupdict()
            it = int(d["iter"])
            row = rows.setdefault(it, {"run": tag, "seed": seed, "iter": it})
            row.update({
                "tau": int(d["tau"]),
                "tau_max": int(d["tau_max"]),
                "trail_rate": float(d["trail_rate"]),
                "trail_n": int(d["trail_n"]),
                "trail_d": int(d["trail_d"]),
                "ent_rate": float(d["ent_rate"]),
                "ent_n": int(d["ent_n"]),
                "ent_d": int(d["ent_d"]),
                "truncated": int(d["truncated"]),
            })
    return [rows[k] for k in sorted(rows)]


def load_eval_jsonl(ckpt_dir):
    p = ckpt_dir / "eval.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_best_json(ckpt_dir):
    p = ckpt_dir / "winners" / "best.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    out_dir = Path(__file__).resolve().parent
    all_rows = []
    eval_rows = []
    for tag, seed, log_path, ckpt_dir in RUNS:
        rows = parse_run(tag, seed, log_path, ckpt_dir)
        all_rows.extend(rows)
        best = load_best_json(ckpt_dir)
        peak_iter = best["source_iter"] if best else None
        for e in load_eval_jsonl(ckpt_dir):
            ckpt_name = Path(e["checkpoint"]).name
            which = "peak" if "best.pt" in ckpt_name else "final"
            gx = e["max_gx_per_episode"]
            n = len(gx)
            mean_gx = sum(gx) / n
            var_gx = sum((x - mean_gx) ** 2 for x in gx) / (n - 1) if n > 1 else 0.0
            std_gx = var_gx ** 0.5
            cv = std_gx / mean_gx if mean_gx else float("nan")
            eval_rows.append({
                "run": tag, "seed": seed, "which": which,
                "peak_iter": peak_iter,
                "eval_seed": e["eval_seed"],
                "n_episodes": n,
                "mean_return": e["mean_return"],
                "mean_length": e["mean_length"],
                "clear_rate": e["clear_rate"],
                "mean_gx": mean_gx,
                "std_gx": std_gx,
                "cv_gx": cv,
                "min_gx": min(gx),
                "max_gx": max(gx),
            })

    all_fields = [
        "run", "seed", "iter", "completed_eps", "mean_return", "mean_len",
        "in_progress", "ip_return", "ip_len", "clears", "loss",
        "policy_loss", "value_loss", "entropy", "at_entrance",
        "tau", "tau_max", "trail_rate", "trail_n", "trail_d",
        "ent_rate", "ent_n", "ent_d", "truncated",
    ]
    with open(out_dir / "iter_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    eval_fields = [
        "run", "seed", "which", "peak_iter", "eval_seed", "n_episodes",
        "mean_return", "mean_length", "clear_rate", "mean_gx", "std_gx",
        "cv_gx", "min_gx", "max_gx",
    ]
    with open(out_dir / "eval_variance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_fields)
        w.writeheader()
        for r in eval_rows:
            w.writerow(r)

    print(f"wrote {len(all_rows)} iter-rows -> iter_metrics.csv")
    print(f"wrote {len(eval_rows)} eval-rows -> eval_variance.csv")


if __name__ == "__main__":
    main()
