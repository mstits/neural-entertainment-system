#!/usr/bin/env python3
"""Score the recurrent-bottleneck A/B treatment arm.

Pre-registration: docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md
Mirrors how the banked control 0.76 was scored: per seed, select the
checkpoint at the peak TRAILING entrance rate (training signal — never
the honest eval itself), then honest-eval the selected checkpoint:
cold entrance start, greedy, sticky 0.25, jitter ±16, 50 eps x 2 eval
seeds = 100 episodes. Report per-seed and best-of-N.

    .venv/bin/python scripts/gru_ab_eval.py \
        --seeds 0 1 2 3 --episodes 50 --out runs/gru_ab/verdict.json
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_RE = re.compile(
    r"\[backward\] iter (\d+):.*entrance (\d+)/(\d+)=")
WINDOW = 30  # iters of trailing entrance window


def pick_checkpoint(seed: int) -> tuple[Path, dict]:
    log = REPO / f"runs/gru_ab/train_seed{seed}.log"
    ckdir = REPO / f"checkpoints/mario_1_1_backward_gru_seed{seed}"
    rows = []  # (iter, cum_succ, cum_att)
    for line in log.read_text().splitlines():
        m = LOG_RE.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    by_iter = {it: (s, a) for it, s, a in rows}
    cks = sorted(ckdir.glob("vanilla_ppo_iter_*.pt"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    best, best_rate, table = None, -1.0, []
    for ck in cks:
        it = int(ck.stem.split("_")[-1])
        cur = by_iter.get(it) or by_iter.get(it - 1)
        prev = by_iter.get(it - WINDOW) or by_iter.get(it - WINDOW - 1)
        if cur is None:
            continue
        s0, a0 = prev if prev else (0, 0)
        ds, da = cur[0] - s0, cur[1] - a0
        rate = ds / da if da > 0 else 0.0
        table.append({"iter": it, "trailing_entrance": round(rate, 4),
                      "window_attempts": da})
        # >= : later iter wins ties (more consolidated)
        if rate >= best_rate and da >= 10:
            best, best_rate = ck, rate
    if best is None and cks:
        best, best_rate = cks[-1], 0.0
    return best, {"selected": str(best), "trailing_entrance": best_rate,
                  "table": table}


def honest_eval(ckpt: Path, episodes: int) -> dict:
    out = {}
    for es in (0, 1):
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/eval_game.py"),
               "--game", "mario",
               "--profile", "configs/mario_1_1_backward_gru.yaml",
               "--checkpoint", str(ckpt),
               "--start-state", "runs/live_show/smb_4_4_micro/entrance_start.state",
               "--episodes", str(episodes), "--sticky-prob", "0.25",
               "--start-jitter", "16", "--eval-seed", str(es),
               "--action-select", "greedy"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=7200)
        # eval_game prints its result as indented multi-line JSON at the
        # end of stdout; parse from the LAST top-level '{' line onward.
        blob = None
        lines = r.stdout.splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.startswith("{")]
        for i in reversed(starts):
            try:
                blob = json.loads("\n".join(lines[i:]))
                break
            except json.JSONDecodeError:
                continue
        out[f"eval_seed_{es}"] = blob if blob else {
            "error": (r.stdout + r.stderr)[-2000:]}
    ok = [v for v in out.values() if "clear_rate" in (v or {})]
    if ok:
        cleared = sum(v["clear_rate"] * v.get("episodes", episodes) for v in ok)
        total = sum(v.get("episodes", episodes) for v in ok)
        out["pooled_clear_rate"] = round(cleared / total, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="runs/gru_ab/verdict.json")
    args = ap.parse_args()
    verdict = {"arms": {}, "control_banked": 0.76,
               "preregistration":
               "docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md"}
    for seed in args.seeds:
        print(f"== seed {seed}: selecting checkpoint...", flush=True)
        ck, sel = pick_checkpoint(seed)
        print(f"   {sel['selected']} (trailing entrance "
              f"{sel['trailing_entrance']:.3f})", flush=True)
        ev = honest_eval(ck, args.episodes)
        verdict["arms"][f"seed{seed}"] = {"selection": sel, "eval": ev}
        print(f"   pooled clear_rate: {ev.get('pooled_clear_rate')}",
              flush=True)
    pooled = [a["eval"].get("pooled_clear_rate")
              for a in verdict["arms"].values()
              if a["eval"].get("pooled_clear_rate") is not None]
    verdict["best_of_n"] = max(pooled) if pooled else None
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"best_of_n": verdict["best_of_n"],
                      "per_seed": pooled}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
