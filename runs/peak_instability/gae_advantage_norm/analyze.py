"""GAE / advantage-estimation analysis for the peak-then-decay phenomenon.

Reads only artifacts already on disk (iter_metrics.csv + eval_variance.csv,
both produced by parse_logs.py from the existing train_seed*.log and
eval.jsonl files -- no new training or eval compute). Prints every table
this dimension's writeup cites, so the writeup's numbers are regenerable
by re-running this one file.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

PEAKS = {
    ("v27", "0"): 60, ("v27", "1"): 50, ("v27", "2"): 90, ("v27", "3"): 60,
    ("v28", "0"): 70, ("v28", "1"): 60, ("v28", "2"): 120, ("v28", "3"): 90,
}

GAMMA = 0.99
LAMBDA = 0.95


def load_iters():
    rows = list(csv.DictReader(open(HERE / "iter_metrics.csv")))
    for r in rows:
        r["iter"] = int(r["iter"])
        for k in ("mean_return", "mean_len", "ip_return", "ip_len", "loss",
                  "policy_loss", "value_loss", "entropy"):
            r[k] = float(r[k])
        for k in ("completed_eps", "clears"):
            r[k] = int(r[k])
    return rows


def load_evals():
    return list(csv.DictReader(open(HERE / "eval_variance.csv")))


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    rows = load_iters()
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["run"], r["seed"])].append(r)
    for v in by_run.values():
        v.sort(key=lambda r: r["iter"])

    section("0. Configured GAE hyperparameters (both v27 and v28, all 8 seeds)")
    print(f"gamma = {GAMMA}  (schema default; neither config overrides it)")
    print(f"gae_lambda = {LAMBDA}  (schema default; neither config overrides it)")
    horizon_value = 1.0 / (1.0 - GAMMA)
    horizon_gae = 1.0 / (1.0 - GAMMA * LAMBDA)
    print(f"value-bootstrap effective horizon 1/(1-gamma) = {horizon_value:.1f} env-steps")
    print(f"GAE trace effective horizon 1/(1-gamma*lambda) = {horizon_gae:.2f} env-steps")
    print("frame_skip=4 -> 1 env-step = 4 NES frames; at 60fps that is "
          f"{horizon_gae*4/60.0:.2f}s of real game time for the GAE horizon, "
          f"{horizon_value*4/60.0:.2f}s for the plain-gamma horizon.")

    section("1. mean_len (completed-episode length) vs GAE horizon, by run, "
            "iter0 / peak / peak+30 / peak+60 / final")
    hdr = (f"{'run':5}{'seed':5}{'iter':6}{'label':8}{'mean_len':>10}"
           f"{'gae_h/len':>11}{'tau':>6}{'clears':>7}{'entropy':>9}")
    print(hdr)
    for (run, seed), data in sorted(by_run.items()):
        pk = PEAKS[(run, seed)]
        for label, it in [("iter0", 0), ("peak", pk), ("+30", pk + 30),
                           ("+60", pk + 60), ("final", 239)]:
            cand = [r for r in data if r["iter"] == it]
            if not cand:
                continue
            r = cand[0]
            ratio = horizon_gae / r["mean_len"] if r["mean_len"] else float("nan")
            tau = r.get("tau", "")
            print(f"{run:5}{seed:<5}{r['iter']:<6}{label:8}{r['mean_len']:>10.1f}"
                  f"{ratio:>11.3f}{tau:>6}{r['clears']:>7}{r['entropy']:>9.4f}")
        print()

    section("2. Iteration-to-iteration volatility (first-difference stdev) in "
            "three windows relative to peak: mean_return, policy_loss, value_loss")
    print(f"{'run':5}{'seed':5}{'segment':28}{'n':4}{'mean_ret':>10}"
          f"{'d(mret)std':>12}{'pl_lvl':>9}{'d(pl)std':>10}{'vl_lvl':>8}{'d(vl)std':>10}")
    for (run, seed), data in sorted(by_run.items()):
        pk = PEAKS[(run, seed)]
        segs = [
            ("pre-peak [0,pk]", [r for r in data if r["iter"] <= pk]),
            ("post-peak-early [pk,pk+40]",
             [r for r in data if pk <= r["iter"] <= pk + 40]),
            ("post-peak-late [pk+40,239]", [r for r in data if r["iter"] >= pk + 40]),
        ]
        for label, seg in segs:
            mr = [r["mean_return"] for r in seg]
            pl = [r["policy_loss"] for r in seg]
            vl = [r["value_loss"] for r in seg]

            def seg_stats(vals):
                if len(vals) < 3:
                    return (float("nan"), float("nan"))
                diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
                return (statistics.mean(vals), statistics.pstdev(diffs))

            mr_l, mr_v = seg_stats(mr)
            pl_l, pl_v = seg_stats(pl)
            vl_l, vl_v = seg_stats(vl)
            print(f"{run:5}{seed:<5}{label:28}{len(seg):4}{mr_l:>10.1f}"
                  f"{mr_v:>12.1f}{pl_l:>9.4f}{pl_v:>10.4f}{vl_l:>8.2f}{vl_v:>10.2f}")
        print()

    section("3. Policy-loss magnitude outliers: max |policy_loss| and fraction "
            "of iters with |policy_loss| > 0.3, pre-peak vs post-peak")
    print(f"{'run':5}{'seed':5}{'pre_max':>10}{'pre_frac>0.3':>14}"
          f"{'post_max':>10}{'post_frac>0.3':>15}")
    for (run, seed), data in sorted(by_run.items()):
        pk = PEAKS[(run, seed)]
        pre = [abs(r["policy_loss"]) for r in data if r["iter"] <= pk]
        post = [abs(r["policy_loss"]) for r in data if r["iter"] > pk]
        pre_frac = sum(1 for x in pre if x > 0.3) / len(pre)
        post_frac = sum(1 for x in post if x > 0.3) / len(post)
        print(f"{run:5}{seed:<5}{max(pre):>10.4f}{pre_frac:>14.3f}"
              f"{max(post):>10.4f}{post_frac:>15.3f}")

    section("4. All individual iterations with |policy_loss| > 2.0 (explosion "
            "events), full context")
    print(f"{'run':5}{'seed':5}{'iter':6}{'dist_from_peak':>15}{'policy_loss':>13}"
          f"{'value_loss':>11}{'entropy':>9}{'clears':>7}{'completed_eps':>14}"
          f"{'trail_rate':>11}")
    for (run, seed), data in sorted(by_run.items()):
        pk = PEAKS[(run, seed)]
        for r in data:
            if abs(r["policy_loss"]) > 2.0:
                print(f"{run:5}{seed:<5}{r['iter']:<6}{r['iter']-pk:>15}"
                      f"{r['policy_loss']:>13.4f}{r['value_loss']:>11.2f}"
                      f"{r['entropy']:>9.4f}{r['clears']:>7}{r['completed_eps']:>14}"
                      f"{r.get('trail_rate',''):>11}")

    section("5. Per-episode return-distribution proxy (max_gx_per_episode from "
            "eval.jsonl honest-eval receipts): peak checkpoint vs final "
            "checkpoint, mean/std/CV")
    ev = load_evals()
    print(f"{'run':5}{'seed':5}{'which':6}{'eval_seed':>10}{'mean_gx':>10}"
          f"{'std_gx':>9}{'cv_gx':>8}")
    for r in ev:
        print(f"{r['run']:5}{r['seed']:<5}{r['which']:6}{r['eval_seed']:>10}"
              f"{float(r['mean_gx']):>10.1f}{float(r['std_gx']):>9.1f}"
              f"{float(r['cv_gx']):>8.3f}")

    # Aggregate peak vs final CV
    peak_cv = [float(r["cv_gx"]) for r in ev if r["which"] == "peak"]
    final_cv = [float(r["cv_gx"]) for r in ev if r["which"] == "final"]
    print(f"\nmean CV across the 16 peak-checkpoint eval rows:  {statistics.mean(peak_cv):.3f}")
    print(f"mean CV across the 16 final-checkpoint eval rows: {statistics.mean(final_cv):.3f}")


if __name__ == "__main__":
    main()
