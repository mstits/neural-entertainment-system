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
    ckdir = REPO / f"checkpoints/mario_1_1_backward_gru_seed{seed}"
    # The pre-registration names "each seed's best preserved checkpoint"
    # as the scored artifact — that is the trainer's own preserve-on-
    # peak snapshot (winners/best.pt, selected on the training-time
    # entrance_trailing_rate, the same artifact class the control's
    # banked 0.76 came from). The 10-iter grid selection below is the
    # fallback for a run that produced no winners snapshot.
    log = REPO / f"runs/gru_ab/train_seed{seed}.log"
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
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               timeout=7200)
        except subprocess.TimeoutExpired as e:
            out[f"eval_seed_{es}"] = {"error": f"timed out after {e.timeout}s"}
            continue
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
    def _delivered(v: dict) -> int:
        """Episodes this receipt actually reports, or 0 if it does not say.

        The receipt's field is `n_episodes` (and, since the protocol-receipt
        fix, `n_episodes_delivered`) — NOT `episodes`. Reading `episodes` with
        a fallback to the CLI constant made the weighting inert: both sums
        always took the fallback, so the "weighted" pool was the unweighted
        mean of the arms. Benign while the arms are equal-n, wrong the moment
        they are not.
        """
        for k in ("n_episodes_delivered", "n_episodes"):
            try:
                n = int(v[k])
            except (KeyError, TypeError, ValueError):
                continue
            if n > 0:
                return n
        return 0

    ok = [v for v in out.values() if "clear_rate" in (v or {})]
    short = [v for v in ok if _delivered(v) != int(episodes)]
    if len(ok) < len(out):
        # Pooling fewer than both --eval-seed runs silently halves the
        # "50 eps x 2 eval seeds = 100 episodes" protocol; refuse instead
        # of reporting a partial number as if it were the full pool.
        out["pooled_clear_rate_error"] = (
            f"only {len(ok)}/{len(out)} eval-seed run(s) returned a "
            "clear_rate; refusing to pool a partial result")
    elif short:
        # ...and the same refusal when a run RETURNED but delivered fewer
        # episodes than the protocol asked for. The guard above only ever
        # checked that `clear_rate` was present, never that the promised
        # sample size arrived — which is the other half of "a partial number
        # reported as if it were the full pool".
        out["pooled_clear_rate_error"] = (
            f"{len(short)}/{len(ok)} eval-seed run(s) delivered an episode "
            f"count other than the requested {int(episodes)} "
            f"(saw {sorted(_delivered(v) for v in short)}); refusing to pool")
    elif ok:
        cleared = sum(v["clear_rate"] * _delivered(v) for v in ok)
        total = sum(_delivered(v) for v in ok)
        out["pooled_clear_rate"] = round(cleared / total, 4) if total else None
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
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        # Two artifact candidates per seed, mirroring how the control's
        # 0.76 was banked (a preserved checkpoint chosen as that seed's
        # best): the trainer's preserve-on-peak snapshot
        # (winners/best.pt) and the 10-iter grid checkpoint at peak
        # trailing entrance. Honest-eval both for the receipt, but select
        # the seed's score on the training-time trailing entrance rate
        # ONLY — never on the honest eval outcome (see module docstring).
        print(f"== seed {seed}: selecting checkpoints...", flush=True)
        cands = []
        ck, sel = pick_checkpoint(seed)
        cands.append(("grid_peak", ck, sel))
        wdir = (REPO / f"checkpoints/mario_1_1_backward_gru_seed{seed}"
                / "winners")
        if (wdir / "best.pt").exists():
            meta = json.loads((wdir / "best.json").read_text())
            cands.append(("winners_best", wdir / "best.pt",
                          {"selected": str(wdir / "best.pt"),
                           "trailing_entrance": meta.get("metric_value"),
                           "source_iter": meta.get("source_iter")}))
        arm = {"candidates": {}}
        best_train_rate, seed_score = None, None
        for name, path, meta in cands:
            print(f"   {name}: {meta['selected']}", flush=True)
            ev = honest_eval(path, args.episodes)
            arm["candidates"][name] = {"selection": meta, "eval": ev}
            r = ev.get("pooled_clear_rate")
            print(f"   {name} pooled clear_rate: {r}", flush=True)
            train_rate = meta.get("trailing_entrance")
            if train_rate is not None and (
                    best_train_rate is None or train_rate > best_train_rate):
                best_train_rate, seed_score = train_rate, r
        arm["seed_score"] = seed_score
        verdict["arms"][f"seed{seed}"] = arm
        # Persist after every seed, not only once at the end: a later
        # seed can still blow the multi-hour budget, and completed
        # seeds' results must not be discarded when it does.
        p.write_text(json.dumps(verdict, indent=2) + "\n")
    pooled = [a["seed_score"] for a in verdict["arms"].values()
              if a.get("seed_score") is not None]
    verdict["best_of_n"] = max(pooled) if pooled else None
    p.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"best_of_n": verdict["best_of_n"],
                      "per_seed": pooled}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
