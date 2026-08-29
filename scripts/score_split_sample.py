"""The standing checkpoint-selection rule, as code instead of prose.

DIRECTION_2026-08-28.md §4.1 / Action 4 registered split-sample selection
as the STANDING gate rule for this line: per seed, select the checkpoint
by argmax of clear_rate on one eval seed and score it with the held-out
other's clear_rate at that checkpoint; and the mirror; the seed's score is
the mean of the two. Ties break to the LATER iteration. Best-of-N is the
max over seeds. Winner's curse is measured at 0.05 and reported as
theta_adj = theta - 0.05, never subtracted silently.

Until this file, the rule lived as prose plus inline one-off snippets —
the v27 corrected ladder was scored by an ad-hoc heredoc. A standing rule
that must be retyped each time is a standing rule only until the first
typo, and this repo's own ledger shows what a selector defect costs: the
retired argmax over `entrance_trailing_rate` under-selected by 20-40
iterations on 4 of 4 runs tested.

Input: a ladder CSV with header  run,iter,eval_seed,clear_rate,status
(the format `docs/receipts/v27_corrected_ladder/ladder.csv` banks and the
F0 driver family produces). Any row with status != "ok" poisons its whole
run: a grid with holes cannot be argmax'd honestly.

tests/test_score_split_sample.py pins the arithmetic to the banked v27
ladder: seed scores {0.110, 0.500, 0.460, 0.460}, best-of-4 0.500 — the
numbers the capacity fork was read against. If this module and that CSV
ever disagree, one of them changed and the test says which.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

CURSE = 0.05  # measured winner's-curse budget, DIRECTION §4.1


@dataclass
class SeedScore:
    run: str
    score: float
    sel_on_es0: int          # iter chosen by argmax on eval seed 0
    scored_on_es1: float     # held-out score at that iter
    sel_on_es1: int
    scored_on_es0: float
    raw_peak_es0: float
    raw_peak_es1: float


@dataclass
class SplitSampleResult:
    seeds: list[SeedScore] = field(default_factory=list)

    @property
    def best(self) -> float:
        return max(s.score for s in self.seeds)

    @property
    def best_adj(self) -> float:
        return self.best - CURSE

    @property
    def best_run(self) -> str:
        return max(self.seeds, key=lambda s: s.score).run

    def to_dict(self) -> dict:
        return {
            "estimator": "split-sample (select on one eval seed, score on "
                         "the held-out other, mirrored, mean; ties -> later "
                         "iter; best-of-N over seeds)",
            "seeds": [vars(s) for s in self.seeds],
            "best_of_n": self.best,
            "best_run": self.best_run,
            "curse_budget": CURSE,
            "best_adj": self.best_adj,
        }


def _grids(rows: list[dict]) -> dict[str, dict[tuple[int, int], float]]:
    grids: dict[str, dict[tuple[int, int], float]] = {}
    for r in rows:
        if r["status"] != "ok":
            raise ValueError(
                f"run {r['run']} iter {r['iter']} es {r['eval_seed']} has "
                f"status {r['status']!r} — a grid with holes cannot be "
                "argmax'd honestly; re-run that eval or VOID the run")
        grids.setdefault(r["run"], {})[
            (int(r["iter"]), int(r["eval_seed"]))
        ] = float(r["clear_rate"])
    return grids


def score_run(grid: dict[tuple[int, int], float], run: str) -> SeedScore:
    iters = sorted({it for it, _ in grid})
    for it in iters:
        for es in (0, 1):
            if (it, es) not in grid:
                raise ValueError(
                    f"run {run}: iter {it} missing eval seed {es} — the "
                    "grid must be complete on both eval seeds")
    # ties -> LATER iteration: max over (rate, iter) tuples.
    sel0 = max(iters, key=lambda it: (grid[(it, 0)], it))
    sel1 = max(iters, key=lambda it: (grid[(it, 1)], it))
    a = grid[(sel0, 1)]
    b = grid[(sel1, 0)]
    return SeedScore(
        run=run, score=(a + b) / 2,
        sel_on_es0=sel0, scored_on_es1=a,
        sel_on_es1=sel1, scored_on_es0=b,
        raw_peak_es0=max(grid[(it, 0)] for it in iters),
        raw_peak_es1=max(grid[(it, 1)] for it in iters),
    )


def score_ladder(csv_path: Path) -> SplitSampleResult:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: empty ladder")
    res = SplitSampleResult()
    for run, grid in sorted(_grids(rows).items()):
        res.seeds.append(score_run(grid, run))
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ladder_csv")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args(argv)
    res = score_ladder(Path(args.ladder_csv))
    for s in res.seeds:
        print(f"{s.run}: split-sample={s.score:.3f} "
              f"[es0 sel it{s.sel_on_es0}->{s.scored_on_es1:.2f} | "
              f"es1 sel it{s.sel_on_es1}->{s.scored_on_es0:.2f}] "
              f"raw peaks {s.raw_peak_es0:.2f}/{s.raw_peak_es1:.2f}")
    print(f"\nbest-of-{len(res.seeds)} = {res.best:.3f}   "
          f"theta_adj (curse {CURSE}) = {res.best_adj:.3f}   "
          f"[{res.best_run}]")
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
