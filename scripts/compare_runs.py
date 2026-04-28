"""
Merge N `metrics.jsonl` files into one comparison chart.

Usage:
    python scripts/compare_runs.py run_a/metrics.jsonl run_b/metrics.jsonl ...

Writes comparison.png beside the FIRST input file's parent. Each run is
plotted as a separate line in best-fitness and avg-fitness panels.

Intended for use after scripts/run_ablations.sh finishes a variant sweep,
but also works stand-alone to compare any pair of training runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: compare_runs.py metrics.jsonl [metrics.jsonl ...]", file=sys.stderr)
        return 1

    runs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"skip: {p} not found", file=sys.stderr)
            continue
        # Label from the enclosing directory name
        label = path.parent.name
        runs.append((label, load(path)))

    if not runs:
        print("no runs to plot", file=sys.stderr)
        return 1

    fig, (ax_best, ax_avg) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, metrics in runs:
        # Skip rows with missing keys instead of KeyError-ing the whole
        # comparison just because a run wrote a partial schema.
        gens, best, avg = [], [], []
        for m in metrics:
            if (
                "generation" in m
                and "best_fitness" in m
                and "avg_fitness" in m
            ):
                gens.append(m["generation"])
                best.append(m["best_fitness"])
                avg.append(m["avg_fitness"])
        if not gens:
            print(f"skip: {label} has no plottable rows", file=sys.stderr)
            continue
        ax_best.plot(gens, best, marker=".", label=label)
        ax_avg.plot(gens, avg, marker=".", label=label)

    ax_best.set_title("Best fitness")
    ax_best.set_ylabel("best_fitness")
    ax_best.legend()
    ax_best.grid(alpha=0.3)

    ax_avg.set_title("Average fitness")
    ax_avg.set_ylabel("avg_fitness")
    ax_avg.set_xlabel("generation")
    ax_avg.legend()
    ax_avg.grid(alpha=0.3)

    # Output beside the first run's metrics file (e.g. run_a/comparison.png),
    # NOT one level higher. The previous .parent.parent landed in CWD
    # whenever the input was `run_a/metrics.jsonl`, which made it hard
    # to associate the image with the runs that produced it.
    out = Path(paths[0]).parent / "comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
