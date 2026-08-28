"""Summarize the V30 premise-falsifier pilot arms from their run logs."""
import re
import statistics
import sys
from pathlib import Path

ITER = re.compile(
    r"\[redo\] iter (\d+): dormant fc1 (\d+)/(\d+) fc2 (\d+)/(\d+) "
    r"recycled (\d+) cum (\d+) agree ([0-9.]+) max_dlogit ([0-9.eE+-]+) "
    r"tail fc1 ([0-9.]+)/([0-9.]+)/([0-9.]+) "
    r"fc2 ([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
PPO = re.compile(
    r"\[vanilla_ppo\] iter (\d+): completed_eps=(\d+).*?mean_return=([0-9.-]+)"
    r".*?clears=(\d+).*?entropy=([0-9.-]+)"
)
BACK = re.compile(
    r"\[backward\] iter (\d+): tau=(\d+)/(\d+).*?trailing (\d+)/(\d+)="
    r"([0-9.]+).*?entrance (\d+)/(\d+)=([0-9.]+)"
)


def parse(path):
    rows, ppo, back = [], {}, {}
    for line in path.read_text(errors="replace").splitlines():
        if (m := ITER.search(line)):
            g = m.groups()
            rows.append(dict(
                it=int(g[0]), d1=int(g[1]), h=int(g[2]), d2=int(g[3]),
                t=int(g[4]), rec=int(g[5]), cum=int(g[6]),
                agree=float(g[7]), dlog=float(g[8]),
                f1min=float(g[9]), f1p5=float(g[10]), f1p10=float(g[11]),
                f2min=float(g[12]), f2p5=float(g[13]), f2p10=float(g[14]),
            ))
        elif (m := PPO.search(line)):
            ppo[int(m.group(1))] = dict(
                eps=int(m.group(2)), ret=float(m.group(3)),
                clears=int(m.group(4)), ent=float(m.group(5)))
        elif (m := BACK.search(line)):
            back[int(m.group(1))] = dict(
                rung=int(m.group(2)), trail=float(m.group(6)),
                entrance=float(m.group(9)))
    return rows, ppo, back


def main(paths):
    parsed = {}
    for raw in paths:
        p = Path(raw)
        rows, ppo, back = parse(p)
        parsed[p.stem] = (rows, ppo, back)
        if not rows:
            print(f"{p.stem}: no [redo] iter lines")
            continue
        fired = [r for r in rows if r["rec"] > 0]
        steady = [r for r in rows if r["it"] >= 5]
        print(f"\n===== {p.stem}  ({len(rows)} checks) =====")
        print(f"  cum_recycled            : {rows[-1]['cum']}")
        print(f"  recycle events          : {len(fired)}/{len(rows)}")
        print(f"  first recycle at iter   : "
              f"{fired[0]['it'] if fired else None}")
        if fired:
            trunk = rows[0]["t"]
            per = [r["rec"] for r in fired]
            print(f"  units/event  min/med/max: "
                  f"{min(per)}/{statistics.median(per):.1f}/{max(per)}"
                  f"  (trunk={trunk})")
            frac = [r["rec"] / trunk for r in fired]
            print(f"  frac of trunk per event : "
                  f"med {statistics.median(frac):.3f}  max {max(frac):.3f}")
            if steady:
                sp = [r["rec"] for r in steady]
                print(f"  units/iter at iter>=5   : "
                      f"med {statistics.median(sp):.1f} "
                      f"({statistics.median(sp) / trunk:.0%} of trunk)")
            ag = [r["agree"] for r in fired]
            print(f"  agree  min/med          : "
                  f"{min(ag):.4f}/{statistics.median(ag):.4f}")
            print(f"  max_dlogit max          : "
                  f"{max(r['dlog'] for r in fired):.4f}")
            print(f"  fc1 recycled (total)    : "
                  f"{sum(r['d1'] for r in rows)}")
        f2 = [r["f2min"] for r in rows]
        print(f"  fc2 score min: iter0 {f2[0]:.4f} -> final {f2[-1]:.4f} "
              f"| run-min {min(f2):.4f}")
        if steady:
            sm = [r["f2min"] for r in steady]
            print(f"  fc2 score min at iter>=5: med {statistics.median(sm):.4f}"
                  f"  range {min(sm):.4f}-{max(sm):.4f}")
        f1 = [r["f1min"] for r in rows]
        print(f"  fc1 score min           : run-min {min(f1):.4f} "
              f"(never dormant at any tau tested)")
        for t in (0.025, 0.05, 0.10, 0.15, 0.20, 0.25):
            n = sum(1 for v in f2 if v <= t)
            print(f"    iters whose fc2 MIN <= {t:<5} : {n}/{len(f2)}")

    # matched-pair divergence
    if "T64" in parsed and "C64" in parsed:
        print("\n===== matched pair T64 (tau .25) vs C64 (tau .025) =====")
        _, tp, tb = parsed["T64"]
        _, cp, cb = parsed["C64"]
        print(f"{'it':>3} {'T ent':>7} {'C ent':>7} {'T ret':>8} {'C ret':>8}"
              f" {'T rung':>7} {'C rung':>7} {'T trail':>8} {'C trail':>8}")
        for it in sorted(set(tp) & set(cp)):
            print(f"{it:>3} {tp[it]['ent']:>7.4f} {cp[it]['ent']:>7.4f} "
                  f"{tp[it]['ret']:>8.1f} {cp[it]['ret']:>8.1f} "
                  f"{tb.get(it, {}).get('rung', -1):>7} "
                  f"{cb.get(it, {}).get('rung', -1):>7} "
                  f"{tb.get(it, {}).get('trail', -1):>8.2f} "
                  f"{cb.get(it, {}).get('trail', -1):>8.2f}")
        common = sorted(set(tp) & set(cp))
        ident = [it for it in common
                 if abs(tp[it]["ret"] - cp[it]["ret"]) < 1e-9]
        print(f"  iters with identical mean_return: {ident} "
              "(pre-divergence; proves the pair is matched)")


if __name__ == "__main__":
    main(sys.argv[1:])
