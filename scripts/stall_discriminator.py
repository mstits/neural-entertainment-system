"""When a solver stalls, is it DYING or SURVIVING-AND-STUCK?

Those look identical from the outside — cells stop growing, progress
freezes — and they need opposite fixes. A search whose rollouts keep
dying at one spot needs a hazard signal. A search whose rollouts survive
and go nowhere has a broken progress metric or cell key, and no amount of
hazard modelling will help it.

Guessing wrong is expensive, and this project has already guessed wrong
once today: Ninja Gaiden's 40-minute stall looked exactly like Kung Fu's
combat wall and was actually a death byte reading 0 at the start state,
where "death = decrement" underflows to 255 and no death is ever seen.

So this measures rather than infers. It runs N independent rollouts from
the profile's own start state under a random policy and reports, per
episode: whether the death byte fell, the furthest progress reached, and
how the episode ended. Everything comes from the profile's DISCOVERED
observables — no game knowledge is used or required.

    .venv/bin/python scripts/stall_discriminator.py \\
        --profile configs/rygar.yaml --episodes 64 --steps 400
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def classify(rows: list[dict], stuck_at: int | None = None) -> dict:
    """PURE: turn per-episode outcomes into a verdict.

    DYING and STUCK are deliberately not a binary — a run can be both, and
    reporting a single label would hide that. The fractions are the
    finding; the label is a convenience.
    """
    n = len(rows)
    if not n:
        return {"verdict": "NO DATA", "n": 0}
    deaths = sum(1 for r in rows if r["died"])
    timeouts = n - deaths
    maxes = [r["max_progress"] for r in rows]
    death_pts = [r["max_progress"] for r in rows if r["died"]]
    frac_dead = deaths / n
    verdict = ("DYING" if frac_dead >= 0.7 else
               "SURVIVING-AND-STUCK" if frac_dead <= 0.3 else "MIXED")
    out = {
        "verdict": verdict, "n": n,
        "died": deaths, "died_frac": round(frac_dead, 3),
        "survived_to_cap": timeouts,
        "progress_median": statistics.median(maxes),
        "progress_max": max(maxes),
        "progress_spread": max(maxes) - min(maxes),
    }
    if death_pts:
        out["death_point_median"] = statistics.median(death_pts)
        out["death_point_spread"] = max(death_pts) - min(death_pts)
        # A tight spread means one specific hazard kills everything; a wide
        # one means death is diffuse and no single obstacle explains it.
        out["deaths_concentrated"] = (
            out["death_point_spread"] <= max(8, 0.1 * max(maxes)))
    if stuck_at is not None:
        out["reached_stall_point"] = sum(
            1 for m in maxes if m >= stuck_at)
    return out


def format_verdict(v: dict) -> str:
    L = [f"verdict: {v['verdict']}  ({v['n']} episodes)"]
    if v.get("n"):
        L.append(f"  died {v['died']}/{v['n']} ({v['died_frac']:.0%}), "
                 f"survived to cap {v['survived_to_cap']}")
        L.append(f"  progress: median {v['progress_median']}, "
                 f"max {v['progress_max']}, spread {v['progress_spread']}")
        if "death_point_median" in v:
            L.append(f"  deaths at: median {v['death_point_median']}, "
                     f"spread {v['death_point_spread']}, "
                     f"concentrated={v['deaths_concentrated']}")
    L.append("")
    if v["verdict"] == "DYING":
        L.append("  -> rollouts are being killed. A hazard signal is the "
                 "lever; a better cell key is not.")
    elif v["verdict"] == "SURVIVING-AND-STUCK":
        L.append("  -> rollouts survive and go nowhere. The progress "
                 "metric or cell key is the lever; hazard modelling "
                 "cannot help a search that is not dying.")
    else:
        L.append("  -> mixed. Both failure modes are present; neither "
                 "fix alone is sufficient.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stuck-at", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np
    import yaml
    import nes_core
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load((REPO / args.profile).read_text())
    solve = prof.get("solve") or {}
    rom = solve.get("rom") or prof.get("rom_path")
    prog_addr = (solve.get("progress") or {}).get("lo")
    lives_addr = solve.get("lives")
    if prog_addr is None or lives_addr is None:
        raise SystemExit("profile needs solve.progress.lo and solve.lives")
    bitmasks = action_space_to_bitmasks(prof["action_space"])
    n_workers = min(8, args.episodes)

    pool = nes_core.Pool(rom_path=str(REPO / rom), num_workers=n_workers,
                         frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    root = (REPO / prof["start_state_path"]).read_bytes()
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    done_eps = 0
    while done_eps < args.episodes:
        b = min(n_workers, args.episodes - done_eps)
        for i in range(n_workers):
            pool.load_worker_state(i, root)
        settled = pool.step_all(np.zeros(n_workers, dtype=np.uint8))
        start_lives = [int(settled[i][2][lives_addr]) for i in range(b)]
        best = [int(settled[i][2][prog_addr]) for i in range(b)]
        died = [False] * b
        died_at = [None] * b
        for _t in range(args.steps):
            acts = np.array([bitmasks[rng.integers(len(bitmasks))]
                             for _ in range(n_workers)], dtype=np.uint8)
            stepped = pool.step_all(acts)
            for i in range(b):
                if died[i]:
                    continue
                ram = stepped[i][2]
                p = int(ram[prog_addr])
                if p > best[i]:
                    best[i] = p
                if int(ram[lives_addr]) < start_lives[i]:
                    died[i] = True
                    died_at[i] = best[i]
        for i in range(b):
            rows.append({"died": died[i], "max_progress": best[i],
                         "died_at": died_at[i]})
        done_eps += b

    v = classify(rows, args.stuck_at)
    print(format_verdict(v))
    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"profile": args.profile, "summary": v,
                                 "episodes": rows}, indent=2) + "\n")
        print(f"\n  receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
