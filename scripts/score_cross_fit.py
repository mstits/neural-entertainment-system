#!/usr/bin/env python3
"""Cross-fit split-sample reducer — the Theta/Delta estimator, as code.

Registered: docs/proposals/V31_REDO_SURGICAL_2026-08-27.md §6 (inherited
verbatim from v27/v28/v30). v27's readjudication
(runs/v27_readjudication_2026-08-27/readjudication.json) computed this
estimator by hand in a document; that is explicitly disallowed for v31 —
"Both Theta and Delta must come out of ONE committed cross-fit reducer
with tests, not arithmetic in a document, or they are not the same
estimator" (§12 item 5).

Per seed, over the 24-point checkpoint ladder (iters 10..240 step 10):

  score_A = honest clear_rate on eval_seed 1, at the checkpoint
            argmax'd (ties -> later iter) on eval_seed 0
  score_B = honest clear_rate on eval_seed 0, at the checkpoint
            argmax'd (ties -> later iter) on eval_seed 1
  seed_score = (score_A + score_B) / 2

Every one of the 100 pooled episodes (50 per eval seed) is used exactly
once as a SCORING episode and never for the selection that chose its own
checkpoint — the split-sample property. Theta = best-of-N over seeds
(max of seed_score). Delta = Theta - Theta_control, control scored with
the IDENTICAL reducer over its own receipts.

Reads eval-receipt JSON files (scripts/eval_game.py's own stdout-printed
result dict, one file per (seed, iter, eval_seed)) named
``<prefix>_seed<S>_it<III>_es<E>.json`` — the convention already in use
in runs/v27_readjudication_2026-08-27/. Each file must carry
``clear_rate`` (the honest per-eval-seed rate) and, for provenance,
``rom_sha256`` when the eval_game.py version that produced it wrote one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECEIPT_RE = re.compile(r"_seed(\d+)_it(\d+)_es(\d+)\.json$")


@dataclass
class SeedScore:
    seed: int
    selected_iter_a: int | None
    score_a: float | None
    selected_iter_b: int | None
    score_b: float | None
    seed_score: float | None
    n_candidates: int


@dataclass
class CrossFitResult:
    seed_scores: list[SeedScore] = field(default_factory=list)
    theta: float | None = None
    theta_adj: float | None = None
    winning_seed: int | None = None
    n_armed_seeds: int = 0
    verdict: str = ""


def load_receipts(
    receipts_dir: Path, *, seeds: list[int],
) -> dict[int, dict[int, dict[int, float]]]:
    """`{seed: {iter: {eval_seed: clear_rate}}}` from a receipts directory.

    Silently skips files that don't match the naming convention or don't
    belong to a requested seed — a receipts directory is expected to
    accumulate files for several campaigns over time.
    """
    out: dict[int, dict[int, dict[int, float]]] = {s: {} for s in seeds}
    for path in sorted(receipts_dir.glob("*.json")):
        m = RECEIPT_RE.search(path.name)
        if m is None:
            continue
        seed, it, es = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if seed not in out:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "clear_rate" not in data:
            continue
        out[seed].setdefault(it, {})[es] = float(data["clear_rate"])
    return out


def _argmax_iter(rates: dict[int, float], candidates: list[int]) -> int | None:
    """argmax over `candidates` present in `rates`; ties -> LATER iter."""
    available = [it for it in candidates if it in rates]
    if not available:
        return None
    best = available[0]
    for it in available[1:]:
        if rates[it] >= rates[best]:  # >= so a tie takes the LATER iter
            best = it
    return best


def cross_fit_seed_score(
    by_iter: dict[int, dict[int, float]],
    *,
    seed: int,
    candidates: list[int],
) -> SeedScore:
    """One seed's score_A/score_B/seed_score (§6), or Nones if either
    eval seed's ladder is incomplete over `candidates` (VOID-UNDERPOWERED
    territory — the caller decides what an incomplete ladder means)."""
    es0 = {it: v[0] for it, v in by_iter.items() if 0 in v and it in candidates}
    es1 = {it: v[1] for it, v in by_iter.items() if 1 in v and it in candidates}
    sel_a = _argmax_iter(es0, candidates)  # select on es0
    sel_b = _argmax_iter(es1, candidates)  # select on es1
    score_a = es1.get(sel_a) if sel_a is not None else None  # score on es1
    score_b = es0.get(sel_b) if sel_b is not None else None  # score on es0
    seed_score = None
    if score_a is not None and score_b is not None:
        seed_score = (score_a + score_b) / 2.0
    return SeedScore(
        seed=seed, selected_iter_a=sel_a, score_a=score_a,
        selected_iter_b=sel_b, score_b=score_b, seed_score=seed_score,
        n_candidates=len(set(es0) | set(es1)),
    )


def compute_theta(
    receipts: dict[int, dict[int, dict[int, float]]],
    *,
    candidates: list[int],
    required_seeds: int = 4,
    winners_curse: float = 0.05,
) -> CrossFitResult:
    """Theta = best-of-N cross-fit seed score. VOID-UNDERPOWERED (no
    Theta) unless every requested seed produced a complete seed_score —
    "fewer than 4 ARMED and scored seeds -> no Theta" (§6/§11), never a
    best-of-fewer computed against the best-of-4 bar.
    """
    result = CrossFitResult()
    for seed in sorted(receipts):
        result.seed_scores.append(
            cross_fit_seed_score(
                receipts[seed], seed=seed, candidates=candidates,
            )
        )
    complete = [s for s in result.seed_scores if s.seed_score is not None]
    result.n_armed_seeds = len(complete)
    if len(complete) < required_seeds:
        result.verdict = (
            f"VOID-UNDERPOWERED: {len(complete)}/{required_seeds} seeds "
            "produced a complete cross-fit score; no Theta is issued, "
            "per-seed numbers stand as mechanism receipts only"
        )
        return result
    best = max(complete, key=lambda s: s.seed_score)
    result.theta = best.seed_score
    result.theta_adj = best.seed_score - winners_curse
    result.winning_seed = best.seed
    if best.seed_score <= 0.767:
        result.verdict = f"FAIL (Theta={best.seed_score:.4f} <= 0.767)"
    elif best.seed_score >= 0.80:
        flag = (
            " — PASS WITHIN THE MEASURED CURSE, requires es2 confirmation"
            if best.seed_score < 0.85 else ""
        )
        result.verdict = f"PASS (Theta={best.seed_score:.4f} >= 0.80){flag}"
    else:
        result.verdict = (
            f"MARGINAL (0.767 < Theta={best.seed_score:.4f} < 0.80) — "
            "licenses no follow-on claim or campaign"
        )
    return result


def compute_delta(theta: float | None, theta_control: float | None) -> dict:
    if theta is None or theta_control is None:
        return {"delta": None, "verdict": "NOT COMPUTED"}
    delta = theta - theta_control
    if delta >= 0.15:
        verdict = "ReDo at a surgical dose is a real lever on this stack"
    elif delta <= 0.05:
        verdict = "ReDo at a surgical dose is not a lever"
    else:
        verdict = f"indeterminate at n=4 (delta={delta:.4f})"
    return {"delta": delta, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--receipts-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument(
        "--iters", type=int, nargs="+",
        default=list(range(10, 241, 10)),
        help="candidate checkpoint iters (default: 10..240 step 10)",
    )
    ap.add_argument("--control-receipts-dir", type=Path, default=None,
                     help="if given, also compute Delta against this "
                          "(e.g. v27's) receipts under the SAME reducer")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    receipts = load_receipts(args.receipts_dir, seeds=args.seeds)
    result = compute_theta(receipts, candidates=args.iters)

    payload = {
        "seed_scores": [asdict(s) for s in result.seed_scores],
        "theta": result.theta,
        "theta_adj": result.theta_adj,
        "winning_seed": result.winning_seed,
        "n_armed_seeds": result.n_armed_seeds,
        "verdict": result.verdict,
        "candidates": args.iters,
    }

    if args.control_receipts_dir is not None:
        control_receipts = load_receipts(
            args.control_receipts_dir, seeds=args.seeds,
        )
        control_result = compute_theta(control_receipts, candidates=args.iters)
        payload["control"] = {
            "theta": control_result.theta,
            "n_armed_seeds": control_result.n_armed_seeds,
            "verdict": control_result.verdict,
        }
        payload["delta"] = compute_delta(result.theta, control_result.theta)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
