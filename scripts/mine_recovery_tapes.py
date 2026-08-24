#!/usr/bin/env python3
"""Fuel mining for the 1-1 recovery distillation (phase 0 of
docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md).

For every true-death episode in the 1-1 assay manifest, adjudicate up
to --per-episode stick states (spread over the episode's back half,
where deaths live) with a 10-minute solver each. Every solve that
finds a verified solution banks a (post-stick state, action tape)
pair into --out/tapes/. The registered VOID threshold is <15 tapes.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay-dir", default="runs/recovery_assay")
    ap.add_argument("--out", default="runs/recovery_distill/fuel")
    ap.add_argument("--profile", default="configs/mario_1_1_backward.yaml")
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-states", type=int, default=36)
    args = ap.parse_args()

    adir = REPO / args.assay_dir
    man = json.loads((adir / "manifest.json").read_text())
    out = REPO / args.out
    (out / "tapes").mkdir(parents=True, exist_ok=True)

    candidates = []
    for ep, r in enumerate(man["records"]):
        if r["cleared"] or not r["sticks"]:
            continue
        if r["length"] >= 1500:      # timeout episodes: alive, easy — skip
            continue
        sticks = [s for s in r["sticks"] if s[0] >= r["length"] * 0.4]
        # spread: take up to per-episode states evenly across the tail
        k = min(args.per_episode, len(sticks))
        idxs = [int(i * (len(sticks) - 1) / max(1, k - 1)) for i in range(k)]
        for i in sorted(set(idxs)):
            t, sp = sticks[i]
            candidates.append({"episode": ep, "t": t,
                               "death_step": r["length"], "state": sp})
    candidates = candidates[: args.max_states]
    print(f"mining {len(candidates)} states "
          f"({args.minutes} solver-min each)", flush=True)

    tapes = []
    results = []
    for i, c in enumerate(candidates):
        sdir = out / f"solve_ep{c['episode']:03d}_t{c['t']:05d}"
        cmd = [str(REPO / ".venv/bin/python"),
               str(REPO / "scripts/go_explore_solve.py"),
               "--profile", args.profile,
               "--root-state", c["state"],
               "--out", str(sdir.relative_to(REPO)),
               "--workers", str(args.workers),
               "--minutes", str(args.minutes),
               "--want-solutions", "1",
               "--seed", str(2000 + i)]
        subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=(args.minutes * 60 + 600))
        sols = sorted((sdir / "solutions").glob("sol_*actions*"))
        if sols:
            tape = out / "tapes" / (
                f"ep{c['episode']:03d}_t{c['t']:05d}_" + sols[0].name)
            shutil.copy2(sols[0], tape)
            # keep the paired state next to the tape
            shutil.copy2(REPO / c["state"],
                         tape.with_suffix(".start.state"))
            tapes.append(str(tape.relative_to(REPO)))
        results.append({**c, "recovered": bool(sols)})
        print(f"[{i+1}/{len(candidates)}] ep{c['episode']} t{c['t']} "
              f"(gap {c['death_step']-c['t']}): "
              f"{'TAPE BANKED' if sols else 'no recovery'}", flush=True)
    (out / "mining.json").write_text(json.dumps(
        {"candidates": len(candidates), "tapes": tapes,
         "results": results}, indent=2) + "\n")
    print(f"\nmined {len(tapes)} verified recovery tapes "
          f"(VOID threshold 15)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
