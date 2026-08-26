"""Extract per-iter curriculum-ladder state (tau, advances, entrance frac,
truncated) from [backward] log lines for the 8 v27/v28 from-scratch runs,
and relate the AT-ENTRANCE transition iter to the honest-peak iter recorded
in each run's winners/best.json.

Log line shape (two variants -- AT-ENTRANCE marker only appears once tau
hits 0, with an extra space in its place):

  [backward] iter N: tau=T/784 (step S frame F gx G) trailing A/W=R
    (advance at >=0.20 over 30) advances=K [ AT-ENTRANCE] | entrance E/D=EF
    | truncated X

All fields taken directly from the text the trainer already logs; no
re-derivation, no reads from metrics.jsonl (which does not carry these
fields -- see README for the schema check that established that).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")

RUNS = [
    ("v27_seed0", REPO / "runs/v27_fresh_recovery/train_seed0.log", REPO / "checkpoints/mario_1_1_v27_recovery_seed0"),
    ("v27_seed1", REPO / "runs/v27_fresh_recovery/train_seed1.log", REPO / "checkpoints/mario_1_1_v27_recovery_seed1"),
    ("v27_seed2", REPO / "runs/v27_fresh_recovery/train_seed2.log", REPO / "checkpoints/mario_1_1_v27_recovery_seed2"),
    ("v27_seed3", REPO / "runs/v27_fresh_recovery/train_seed3.log", REPO / "checkpoints/mario_1_1_v27_recovery_seed3"),
    ("v28_seed0", REPO / "runs/v28_capacity/train_seed0.log", REPO / "checkpoints/mario_1_1_v28_capacity_seed0"),
    ("v28_seed1", REPO / "runs/v28_capacity/train_seed1.log", REPO / "checkpoints/mario_1_1_v28_capacity_seed1"),
    ("v28_seed2", REPO / "runs/v28_capacity/train_seed2.log", REPO / "checkpoints/mario_1_1_v28_capacity_seed2"),
    ("v28_seed3", REPO / "runs/v28_capacity/train_seed3.log", REPO / "checkpoints/mario_1_1_v28_capacity_seed3"),
]

# Honest eval numbers from the task brief (peak/final honest scores), kept
# alongside the ladder data so the join to "collapse severity" doesn't
# require re-deriving them from gate/*.json.
HONEST = {
    "v27_seed0": dict(peak_iter=60, pct_through=24, honest_peak=0.040, honest_final=0.020),
    "v27_seed1": dict(peak_iter=50, pct_through=20, honest_peak=0.290, honest_final=0.020),
    "v27_seed2": dict(peak_iter=90, pct_through=36, honest_peak=0.530, honest_final=0.000),
    "v27_seed3": dict(peak_iter=60, pct_through=24, honest_peak=0.170, honest_final=0.010),
    "v28_seed0": dict(peak_iter=70, pct_through=28, honest_peak=0.450, honest_final=0.000),
    "v28_seed1": dict(peak_iter=60, pct_through=24, honest_peak=0.230, honest_final=0.050),
    "v28_seed2": dict(peak_iter=120, pct_through=48, honest_peak=0.370, honest_final=0.000),
    "v28_seed3": dict(peak_iter=90, pct_through=36, honest_peak=0.670, honest_final=0.000),
}

LINE_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): "
    r"tau=(?P<tau>\d+)/(?P<tau0>\d+) "
    r"\(step (?P<step>\d+) frame (?P<frame>\d+) gx (?P<gx>\d+)\) "
    r"trailing (?P<trail_n>\d+)/(?P<trail_d>\d+)=(?P<trail_rate>[\d.]+) "
    r"\(advance at >=0\.20 over 30\) "
    r"advances=(?P<advances>\d+)\s*"
    r"(?P<at_entrance>AT-ENTRANCE)?\s*"
    r"\| entrance (?P<ent_n>\d+)/(?P<ent_d>\d+)=(?P<ent_frac>[\d.]+) "
    r"\| truncated (?P<truncated>\d+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "iter": int(d["iter"]),
                "tau": int(d["tau"]),
                "tau0": int(d["tau0"]),
                "step": int(d["step"]),
                "frame": int(d["frame"]),
                "gx": int(d["gx"]),
                "trail_n": int(d["trail_n"]),
                "trail_d": int(d["trail_d"]),
                "trail_rate": float(d["trail_rate"]),
                "advances": int(d["advances"]),
                "at_entrance": d["at_entrance"] is not None,
                "ent_n": int(d["ent_n"]),
                "ent_d": int(d["ent_d"]),
                "ent_frac": float(d["ent_frac"]),
                "truncated": int(d["truncated"]),
            })
    return rows


def main() -> None:
    out_dir = Path(__file__).parent
    all_rows = {}
    summary = []

    for name, log_path, ckpt_dir in RUNS:
        rows = parse_log(log_path)
        if not rows:
            print(f"WARNING: no [backward] lines parsed from {log_path}")
            continue
        all_rows[name] = rows

        n_iters = len(rows)
        max_iter = max(r["iter"] for r in rows)

        # first iter where tau hits 0 (AT-ENTRANCE marker present)
        at_entrance_rows = [r for r in rows if r["at_entrance"]]
        first_at_entrance_iter = min((r["iter"] for r in at_entrance_rows), default=None)

        # sanity: once AT-ENTRANCE appears, does tau==0 hold for every
        # subsequent iter (monotone-and-stays), or does it ever reopen?
        after = [r for r in rows if r["iter"] >= (first_at_entrance_iter or 10**9)]
        reopened = any(not r["at_entrance"] for r in after)

        # best.json -- authoritative honest-peak iter (training-time
        # selection metric, not the offline honest-eval score)
        best_json = json.loads((ckpt_dir / "winners" / "best.json").read_text())
        peak_iter = best_json["source_iter"]
        peak_metric = best_json["metric_value"]

        honest = HONEST[name]
        assert honest["peak_iter"] == peak_iter, f"{name}: brief peak_iter {honest['peak_iter']} != best.json {peak_iter}"

        # iters spent AT-ENTRANCE (tau==0) out of total logged iters
        n_at_entrance = len(at_entrance_rows)
        frac_at_entrance = n_at_entrance / n_iters

        # lag between ladder-finish and honest peak: positive = peak comes
        # AFTER ladder finishes (i.e. during flat-tau=0 training);
        # negative = peak comes BEFORE ladder finishes (ladder still moving)
        if first_at_entrance_iter is not None:
            lag_peak_after_entrance = peak_iter - first_at_entrance_iter
        else:
            lag_peak_after_entrance = None

        last_row = rows[-1]

        summary.append({
            "run": name,
            "n_logged_iters": n_iters,
            "max_iter": max_iter,
            "first_at_entrance_iter": first_at_entrance_iter,
            "at_entrance_reopened_after_first": reopened,
            "n_iters_at_entrance": n_at_entrance,
            "frac_iters_at_entrance": round(frac_at_entrance, 4),
            "peak_iter_bestjson": peak_iter,
            "peak_metric_bestjson": peak_metric,
            "lag_peak_minus_first_at_entrance": lag_peak_after_entrance,
            "honest_peak": honest["honest_peak"],
            "honest_final": honest["honest_final"],
            "final_ent_frac_cumulative": last_row["ent_frac"],
            "final_truncated_cumulative": last_row["truncated"],
            "final_advances": last_row["advances"],
            "final_tau": last_row["tau"],
        })

    # dump raw parsed rows (one file per run) for anyone who wants to
    # re-derive something else from the same source lines
    for name, rows in all_rows.items():
        with open(out_dir / f"ladder_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    with open(out_dir / "ladder_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for s in summary:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
