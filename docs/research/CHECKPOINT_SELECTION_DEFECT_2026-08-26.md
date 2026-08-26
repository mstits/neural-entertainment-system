# The winner-selector under-selects: recorded peaks are not the honest peaks

**Date:** 2026-08-26
**Status:** established on 4 of 4 runs tested, at full honest protocol.
**Consequence:** every honest number this project has banked from a v27/v28-class
run was measured at a checkpoint chosen by a training proxy that does not
identify the honest-best checkpoint. The recorded numbers are systematically
LOW. This does not retract any verdict on its own — see "What this does not
say" — but it changes what the next experiment must measure.

## How it surfaced

The peak-instability campaign's registered cheap falsifier (P1) evaluated a
ladder of checkpoints spanning peak -> peak+60 on all 8 archived runs, greedy
and sampled, n=50, eval seed 0 — 64 evaluations, zero training compute. P1's
own registered question (sampled-vs-greedy ratio) returned **SPLIT**: 2 of 8
runs PROMOTE, 2 KILL, 4 neither, against a rule requiring >=6 of 8 either way.
Per its own pre-registration the v29 entropy-guard registration is therefore
**withdrawn, not run**.

The falsifier's incidental finding is larger than the question it was built to
answer: several post-peak checkpoints scored well ABOVE the run's recorded
honest@peak.

## Two verifications, run before believing it

1. **`winners/best.pt` is byte-identical to the iter checkpoint at its own
   `source_iter`.** Compared tensor-by-tensor for
   `mario_1_1_v28_capacity_seed0` (source_iter 70) and
   `mario_1_1_v27_recovery_seed3` (source_iter 60): identical=True,
   max|delta|=0.0 across all 12 tensors. So the gate and P1 measured the same
   weights, and any difference is protocol or selection, not artifact.
2. **P1 reproduces the gate exactly where the protocol matches.** v28 seed0's
   banked gate receipt records clear_rate 0.50 at eval seed 0 on its peak
   checkpoint; P1's independent run of it70/greedy/eval-seed-0 also returns
   0.50. Exact.

## The measurement

Four checkpoints that P1 flagged, re-scored at the FULL gate protocol
(cold entrance, greedy, sticky 0.25, jitter +-16, 50 eps x eval seeds {0,1} =
100 pooled) so they are directly comparable to the banked numbers:

| run | recorded honest@peak | proxy-picked iter | alt iter | s0 | s1 | pooled | delta |
|---|---|---|---|---|---|---|---|
| v28 seed0 | 0.450 | 70 | 90 | 0.70 | 0.58 | **0.640** | **+0.190** |
| v28 seed1 | 0.230 | 60 | 80 | 0.50 | 0.38 | **0.440** | **+0.210** |
| v27 seed1 | 0.290 | 50 | 70 | 0.52 | 0.48 | **0.500** | **+0.210** |
| v27 seed0 | 0.040 | 60 | 100 | 0.16 | 0.08 | **0.120** | **+0.080** |

4 of 4 improved. Every miss is in the same direction: the selector fired
**too early**, by 20-40 iterations in each case.

## Mechanism

The selector reads `entrance_trailing_rate` — a 30-episode rolling window of
in-training success — and takes its argmax across the run. Two independent
reasons that is the wrong estimator, both already documented in this repo:

- **It measures the wrong thing.** The v27 verdict recorded that
  entrance_trailing_rate overestimates the honest rate by 2-25x while only
  rank-ordering seeds correctly. A metric that preserves rank across seeds
  need not preserve rank across checkpoints WITHIN a seed, and this is direct
  evidence it does not.
- **It saturates.** The forensics campaign measured the authoritative peak
  metric at 0.867-1.000 across all 8 runs — near its ceiling, where a
  30-episode window has SE ~0.09 and cannot discriminate between checkpoints
  at all. The argmax over ~25 saturated, noisy draws is close to arbitrary.

## What this does not say

**It does not retract v27's or v28's verdicts, and no corrected best-of-4 is
claimed here.** The four checkpoints above were chosen because a noisy n=50
single-seed screen ranked them high — that is a winner's-curse selection, and
the regression from screen to full protocol is visible in the table (0.70 ->
0.640, 0.50 -> 0.440, 0.52 -> 0.500, 0.16 -> 0.120). Quoting "v28 really
scored 0.640" would repeat, in the opposite direction, exactly the selection
error this document is about.

What IS established: the proxy-selected checkpoint was not the honest-best
checkpoint in 4 of 4 runs tested, and the shortfall is +0.08 to +0.21 at full
protocol. That is a systematic instrument defect, not seed noise.

## What follows

1. **A re-scoring rule must be registered before any re-scoring is done**, or
   the correction will inflate exactly as the original deflated. The rule has
   to fix, in advance, which checkpoints get evaluated (a fixed grid, not an
   adaptive search) and how the run's number is drawn from them.
2. **The peak-instability framing needs revisiting.** "Runs peak at 20-48% and
   decay" was measured against proxy-selected peaks. The true peak is later
   and higher; the decay past it is unaffected and still real (v28 seed2 still
   runs 0.36 -> 0.04 -> 0.02 -> 0.00 across it120-it180).
3. **`winners/best.pt` remains load-bearing and correct as a floor**, just not
   as an optimum. Without it these runs report ~0.01. It should keep running.
4. The cheap-falsifier-before-big-workflow rule paid for itself here: 30
   minutes of eval returned SPLIT on its own question and surfaced this, in
   place of an 8-hour 4-seed run aimed at a target the data does not support.

Receipts: `runs/peak_instability/p1_falsifier/` (64 evaluations),
`runs/peak_instability/reselection/` (4 confirmations at eval seed 1).
