#!/usr/bin/env python3
"""Mechanism read #4 (v28-specific): dormancy/recycle telemetry at the
new width, v27 (48k, hidden=64/trunk=32) vs v28 (72k, hidden=96/trunk=32).

Registered in docs/proposals/V28_CAPACITY_2026-08-25.md, "Mechanism
reads" item 4: "per-layer dormant fraction and recycle counts per
iteration, cumulative recycles per seed, and whether recycle events
cluster at recovery-band stalls the same way (or differently) than in
the v27 (48k) runs."

Parses the `[redo]` lines emitted by src/training/redo.py's trainer
hook (trainer.py ~L7768-7793) out of the real training logs
(runs/v27_fresh_recovery/train_seed{0..3}.log,
runs/v28_capacity/train_seed{0..3}.log) plus the two experiments'
forced-recycle tau-sweep preflight logs, and the `[backward]` ladder
telemetry (trainer.py ~L8277) so a stall-clustering check is possible
IF any recycle event exists to test against.

Output: JSON with full per-seed series + summary, written next to this
script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

ENABLED_RE = re.compile(
    r"\[redo\] ENABLED tau=(?P<tau>[\d.eE+-]+) every_iters=(?P<every>\d+) "
    r"scope=(?P<scope>\S+) sample=(?P<sample>\d+) "
    r"reset_moments=(?P<reset>\w+)"
)
ITER_RE = re.compile(
    r"\[redo\] iter (?P<it>\d+): dormant fc1 (?P<d1>\d+)/(?P<h1>\d+) "
    r"fc2 (?P<d2>\d+)/(?P<h2>\d+) recycled (?P<rec>\d+) "
    r"cum (?P<cum>\d+) agree (?P<agree>[\d.]+) "
    r"max_dlogit (?P<dlogit>[\d.eE+-]+)"
)
SKIPPED_RE = re.compile(r"\[redo\] iter (?P<it>\d+): skipped \(no gradient step\)")

BACKWARD_RE = re.compile(
    r"\[backward\] iter (?P<it>\d+): tau=(?P<tau>\d+)/(?P<taumax>\d+) "
    r"\(step \d+ frame (?P<frame>\d+) gx \d+\) "
    r"trailing (?P<succ>\d+)/(?P<att>\d+)=(?P<rate>[\d.]+) "
)


def parse_redo_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    enabled = ENABLED_RE.search(text)
    rows = []
    for m in ITER_RE.finditer(text):
        rows.append({
            "it": int(m["it"]),
            "dormant_fc1": int(m["d1"]),
            "hidden_dim": int(m["h1"]),
            "dormant_fc2": int(m["d2"]),
            "trunk_dim": int(m["h2"]),
            "recycled": int(m["rec"]),
            "cum": int(m["cum"]),
            "agree": float(m["agree"]),
            "max_dlogit": float(m["dlogit"]),
        })
    skipped = [int(m["it"]) for m in SKIPPED_RE.finditer(text)]
    return {
        "path": str(path.relative_to(REPO)),
        "enabled_line": enabled.group(0) if enabled else None,
        "tau": float(enabled["tau"]) if enabled else None,
        "check_every_iters": int(enabled["every"]) if enabled else None,
        "n_iters_logged": len(rows),
        "n_skipped_iters": len(skipped),
        "skipped_iters": skipped,
        "rows": rows,
    }


def parse_backward_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    out = []
    for m in BACKWARD_RE.finditer(text):
        out.append({
            "it": int(m["it"]),
            "tau_rung": int(m["tau"]),
            "tau_max": int(m["taumax"]),
            "frame": int(m["frame"]),
            "trailing_rate": float(m["rate"]),
        })
    return out


def summarize_seed(redo: dict, backward_rows: list[dict]) -> dict:
    rows = redo["rows"]
    n = len(rows)
    hidden_dim = rows[0]["hidden_dim"] if rows else None
    trunk_dim = rows[0]["trunk_dim"] if rows else None

    dormant_fc1_frac = [
        (r["dormant_fc1"] / r["hidden_dim"]) if r["hidden_dim"] else 0.0
        for r in rows
    ]
    dormant_fc2_frac = [
        (r["dormant_fc2"] / r["trunk_dim"]) if r["trunk_dim"] else 0.0
        for r in rows
    ]
    recycle_events = [r for r in rows if r["recycled"] > 0]
    final_cum = rows[-1]["cum"] if rows else 0
    sum_recycled = sum(r["recycled"] for r in rows)

    # Stall-clustering check: only meaningful if recycle_events is non-empty.
    # A "stall" window is defined loosely here as consecutive backward-telemetry
    # samples with trailing_rate <= 0.05 (near-zero clear rate) for context only
    # -- this read does not re-derive reads #1/#3's rung-rate definitions, it
    # only checks temporal coincidence IF there is anything to coincide with.
    stall_its = {b["it"] for b in backward_rows if b["trailing_rate"] <= 0.05}
    recycle_its_near_stall = [
        r["it"] for r in recycle_events if r["it"] in stall_its
    ]

    return {
        "hidden_dim": hidden_dim,
        "trunk_dim": trunk_dim,
        "tau": redo["tau"],
        "n_iters_logged": n,
        "n_skipped_iters": redo["n_skipped_iters"],
        "final_cumulative_recycled": final_cum,
        "sum_of_per_iter_recycled_crosscheck": sum_recycled,
        "n_recycle_events": len(recycle_events),
        "recycle_event_iters": [r["it"] for r in recycle_events],
        "recycle_event_detail": recycle_events,
        "dormant_fc1_frac_max": max(dormant_fc1_frac) if dormant_fc1_frac else None,
        "dormant_fc1_frac_mean": (
            sum(dormant_fc1_frac) / len(dormant_fc1_frac) if dormant_fc1_frac else None
        ),
        "dormant_fc2_frac_max": max(dormant_fc2_frac) if dormant_fc2_frac else None,
        "dormant_fc2_frac_mean": (
            sum(dormant_fc2_frac) / len(dormant_fc2_frac) if dormant_fc2_frac else None
        ),
        "any_nonzero_dormant_ever": any(
            r["dormant_fc1"] > 0 or r["dormant_fc2"] > 0 for r in rows
        ),
        "n_backward_telemetry_lines": len(backward_rows),
        "n_stall_iters_trailing_le_0.05": len(stall_its),
        "recycle_events_coincident_with_stall_iters": recycle_its_near_stall,
        "clustering_verdict": (
            "N/A -- zero recycle events fired during real training; no events "
            "exist to test for temporal clustering with ladder stalls"
            if not recycle_events
            else (
                f"{len(recycle_its_near_stall)}/{len(recycle_events)} recycle "
                "events coincide with a stall iter (trailing<=0.05)"
            )
        ),
    }


def load_experiment(train_dir: Path, backward_dir: Path | None = None) -> dict:
    seeds = {}
    for i in range(4):
        train_log = train_dir / f"train_seed{i}.log"
        if not train_log.exists():
            seeds[str(i)] = {"error": f"missing {train_log}"}
            continue
        redo = parse_redo_log(train_log)
        backward_rows = parse_backward_log(train_log)  # same file carries both
        seeds[str(i)] = summarize_seed(redo, backward_rows)
        seeds[str(i)]["_raw_redo"] = redo
    return seeds


def load_preflight_sweep(paths: dict[str, Path]) -> dict:
    out = {}
    for tau_label, p in paths.items():
        if not p.exists():
            out[tau_label] = {"error": f"missing {p}"}
            continue
        out[tau_label] = parse_redo_log(p)
    return out


def main() -> None:
    v27_dir = REPO / "runs" / "v27_fresh_recovery"
    v28_dir = REPO / "runs" / "v28_capacity"

    v27_seeds = load_experiment(v27_dir)
    v28_seeds = load_experiment(v28_dir)

    v27_sweep_dir = v27_dir / "preflight" / "redo_forced"
    v27_sweep = load_preflight_sweep({
        f"tau{t}": v27_sweep_dir / f"isolate_tau{t}.log"
        for t in ("0.05", "0.08", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35")
    })
    v28_preflight_dir = v28_dir / "preflight"
    v28_sweep = load_preflight_sweep({
        f"tau{t}": v28_preflight_dir / f"tau{t}.log"
        for t in ("0.15", "0.20", "0.25", "0.30")
    })

    def cross_seed_summary(seeds: dict) -> dict:
        total_recycled = sum(
            s.get("final_cumulative_recycled", 0)
            for s in seeds.values() if "error" not in s
        )
        total_events = sum(
            s.get("n_recycle_events", 0)
            for s in seeds.values() if "error" not in s
        )
        any_dormant = any(
            s.get("any_nonzero_dormant_ever", False)
            for s in seeds.values() if "error" not in s
        )
        return {
            "total_cumulative_recycled_across_seeds": total_recycled,
            "total_recycle_events_across_seeds": total_events,
            "any_seed_ever_showed_nonzero_dormant_unit": any_dormant,
        }

    result = {
        "read": "read4_dormancy_recycle_telemetry_new_width",
        "source_doc": "docs/proposals/V28_CAPACITY_2026-08-25.md",
        "v27_48k": {
            "seeds": v27_seeds,
            "cross_seed_summary": cross_seed_summary(v27_seeds),
            "forced_recycle_preflight_sweep": v27_sweep,
        },
        "v28_72k": {
            "seeds": v28_seeds,
            "cross_seed_summary": cross_seed_summary(v28_seeds),
            "forced_recycle_preflight_sweep": v28_sweep,
        },
    }

    out_path = OUT_DIR / "read4_dormancy_recycle.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")

    # Compact console summary
    for label, exp in (("v27 (48k)", result["v27_48k"]), ("v28 (72k)", result["v28_72k"])):
        print(f"\n=== {label} ===")
        for sid, s in exp["seeds"].items():
            if "error" in s:
                print(f" seed{sid}: ERROR {s['error']}")
                continue
            print(
                f" seed{sid}: hidden={s['hidden_dim']} trunk={s['trunk_dim']} "
                f"tau={s['tau']} iters_logged={s['n_iters_logged']} "
                f"cum_recycled={s['final_cumulative_recycled']} "
                f"events={s['n_recycle_events']} "
                f"any_dormant_ever={s['any_nonzero_dormant_ever']}"
            )
        print(" cross-seed:", exp["cross_seed_summary"])


if __name__ == "__main__":
    main()
