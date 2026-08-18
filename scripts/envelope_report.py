"""Reactive-envelope estimator over banked per-level evaluations.

Implements the estimator from the 2026-08-17 research synthesis: for a
level of traversal length L cleared at rate C, the implied per-pixel
hazard is lambda = -ln(C)/L, and a level's predicted rate under the same
hazard density is exp(-lambda * L). The diagnostic that separates
"under-trained" from "at the reactive envelope" is the spatial spread of
deaths: concentrated deaths mean a specific unsolved obstacle, uniformly
spread deaths mean ambient execution noise.

Reads eval JSONs produced by scripts/eval_game.py. Reports per level,
then cross-predicts each level from every other level's hazard so the
outlier is visible rather than asserted.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# level -> (traversal length in gx, [eval json paths])
LEVELS = {
    "1-2": (3266, ["runs/consol2/peak_eval_seed7.json",
                   "runs/consol2/peak_eval_seed101.json"]),
    "1-3": (2515, ["runs/consol2_1_3/final_eval_seed7.json",
                   "runs/consol2_1_3/final_eval_seed101.json"]),
    "1-4": (2431, ["runs/online_1_4/final_eval_seed7.json",
                   "runs/online_1_4/final_eval_seed101.json"]),
}


def load(paths):
    gx, clears, n = [], 0, 0
    for rel in paths:
        p = REPO / rel
        if not p.exists():
            continue
        txt = p.read_text()
        d = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        gx += [int(g) for g in d.get("max_gx_per_episode", [])]
        n += int(d.get("n_episodes", 0))
        clears += round(float(d.get("clear_rate", 0.0)) * int(d.get("n_episodes", 0)))
    return gx, clears, n


def spread(deaths, length):
    """Coefficient of variation of death positions, plus the share of
    deaths inside the single worst 10%-of-level band. A concentrated
    obstacle shows a high band share; ambient noise shows a low one."""
    if len(deaths) < 3:
        return None, None
    cv = st.pstdev(deaths) / max(st.mean(deaths), 1e-9)
    band = length / 10.0
    counts = {}
    for d in deaths:
        counts[int(d // band)] = counts.get(int(d // band), 0) + 1
    return cv, max(counts.values()) / len(deaths)


def main() -> int:
    rows = []
    for name, (length, paths) in LEVELS.items():
        gx, clears, n = load(paths)
        if not n:
            print(f"{name}: no eval data found")
            continue
        rate = clears / n
        deaths = [g for g in gx if g < length - 5]
        lam = -math.log(max(rate, 1e-9)) / length
        cv, band = spread(deaths, length)
        rows.append((name, length, rate, n, lam, cv, band, len(deaths)))

    print(f"{'lvl':4} {'len':>5} {'rate':>7} {'n':>4} {'lambda(1e-4)':>12} "
          f"{'deathCV':>8} {'worstBand':>9}  verdict")
    for name, length, rate, n, lam, cv, band, nd in rows:
        if cv is None:
            verdict = "too few deaths to diagnose"
        elif band >= 0.35:
            verdict = "CONCENTRATED -> a specific obstacle remains; more training should pay"
        elif band <= 0.20:
            verdict = "SPREAD -> ambient noise; at the reactive envelope"
        else:
            verdict = "mixed"
        print(f"{name:4} {length:5d} {rate:7.2f} {n:4d} {lam*1e4:12.3f} "
              f"{cv:8.3f} {band:9.2f}  {verdict}")

    print("\ncross-prediction (each level's rate predicted from every other"
          "\nlevel's hazard density; large gaps mark the outlier):")
    header = "from\\to"
    print(f"{header:>8} " + " ".join(f"{r[0]:>8}" for r in rows))
    for src in rows:
        cells = []
        for dst in rows:
            cells.append(f"{math.exp(-src[4] * dst[1]):8.2f}")
        print(f"{src[0]:>8} " + " ".join(cells))
    print(f"{'ACTUAL':>8} " + " ".join(f"{r[2]:8.2f}" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
