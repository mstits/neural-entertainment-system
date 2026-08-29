# v33 verdict — MARGINAL at Θ = 0.780. The dose-response curve is the deliverable.

Adjudicated against `docs/proposals/V33_CAPACITY_2026-08-28.md` exactly as
registered; no numeral moved, no sub-metric added. Receipts:
`docs/receipts/v33_capacity/` (full 192-row ladder CSV, per-seed scores,
both preflights, verdict draft); training logs and grids under
`runs/v33_capacity/` and `checkpoints/mario_1_1_v33_capacity_seed{0..3}/`
(24 checkpoints each, all four seeds rc=0 at 250 iterations).

## The number, under the standing estimator

| seed | split-sample | selections | raw peaks |
|---|---|---|---|
| 0 | **0.780** | es0 it90→0.72 · es1 it90→0.84 | 0.84 / 0.72 |
| 1 | 0.470 | es0 it140→0.56 · es1 it150→0.38 | 0.50 / 0.60 |
| 2 | 0.410 | es0 it100→0.40 · es1 it150→0.42 | 0.50 / 0.56 |
| 3 | 0.560 | es0 it100→0.60 · es1 it90→0.52 | 0.60 / 0.66 |

**Θ₃₃ = 0.780 (best-of-4, seed 0). Θ_adj = 0.730** (curse budget 0.05,
reported beside, never subtracted silently).

## The verdict, item by item

**Item 3 (the bar):** 0.767 < 0.780 < 0.80 → **MARGINAL. Reported as
MARGINAL; licenses nothing** — no PASS claim, no follow-on claim, no
follow-on campaign. The bar's provenance sentence attaches: 0.767 is 46/60
at eval seed 0, shared-stream, one worker, never measured under the
two-seed per-episode protocol it gates. Note Θ_adj (0.730) sits below even
the FAIL bound: within the measured winner's curse, this result cannot be
distinguished from a sub-bar one.

**Item 6 (the registered secondary):** dose-response reads **improving** —
Θ₃₃ > 0.670 by +0.110.

**Item 4 (the power statement's consequence, primary deliverable):** the
third dose point. The curve, three points under ONE estimator for the
first time in this project's history:

| params | Θ | step |
|---|---|---|
| 48,135 | 0.500 | — |
| 72,039 | 0.670 | +0.170 |
| 95,943 | 0.780 | +0.110 |

Monotone, decelerating. Per-seed dispersion remains large ({0.78, 0.47,
0.41, 0.56}; SD ≈ 0.16) — best-of-4 is a max statistic and seed 0 carries
the headline, exactly the caveat the registration pre-stated.

## What this licenses, and what it does not

Nothing further, by the registration's own text. MARGINAL authorizes no
follow-on campaign; item 4(b)'s design-change requirement (more seeds or a
bar renegotiation, registered first) is the only path to another capacity
arm. What stands regardless of the gate: **capacity is a real, measured,
monotone lever on this recipe across a 2× parameter range** — the first
mechanism conclusion in the v27→v33 line supported by a same-estimator
curve rather than a single delta. The decelerating step (+0.170 → +0.110)
is consistent with diminishing returns and is *not* extrapolated here:
whether a fourth point continues, flattens, or reverses is precisely what
only a registered design change could measure.

Campaign integrity notes: attempt 1 VOIDed at iter 40 by the arming
deadline (config defect, ADDENDUM 1, receipted); the eval chain ran twice
concurrently by accident (a watcher believed dead had survived) — verified
harmless: 192/192 receipts, zero bad, zero missing, zero extras,
deterministic evals rewriting identical content. Both incidents are in
MISTAKES.md.
