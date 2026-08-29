# v27 corrected peak ladder — the capacity delta measured two-sided for the first time

Executes Action 3 of `docs/proposals/DIRECTION_2026-08-28.md`. 192 evaluations
over `checkpoints/mario_1_1_v27_recovery_seed{0..3}/` (24-checkpoint grids,
verified complete), protocol byte-identical to the receipted v28 F0 job — the
driver differs from `runs/v29_stability/f0_ladder/run_f0.py` by exactly two
lines (run names, output path). 192/192 ok, 0 bad, 5,257 s (v28's job: 5,333 s).
Receipts: `runs/v27_corrected_ladder/` (192 JSON + `v27_ladder.log`).

## The estimator, identical to what landed v28 on 0.670

Per seed: select the checkpoint by argmax of `clear_rate` on eval seed 0, score
with eval seed 1's `clear_rate` at that checkpoint; and the mirror; seed score
= mean of the two. Ties → later iteration. Best-of-4 = max over seeds.

## Result

| seed | split-sample | sel@es0 → es1 score | sel@es1 → es0 score | raw peaks es0/es1 |
|---|---|---|---|---|
| 0 | 0.110 | it110 → 0.12 | it70 → 0.10 | 0.22 / 0.28 |
| 1 | **0.500** | it70 → 0.48 | it70 → 0.52 | 0.52 / 0.48 |
| 2 | 0.460 | it110 → 0.42 | it90 → 0.50 | 0.60 / 0.60 |
| 3 | 0.460 | it90 → 0.48 | it90 → 0.44 | 0.44 / 0.48 |

**v27 split-sample best-of-4 = 0.500** (seed 1)
**v28 split-sample best-of-4 = 0.670** (banked, same estimator)
**Corrected delta = +0.170** — *larger* than the uncorrected +0.14.

## Instrument convergence, checked before reading the fork

Both spot-corrections cited in the decision doc reproduce: seed 1's 0.290 →
0.500 spot-correction matches this ladder's 0.500 exactly; seed 0's 0.120 spot
matches 0.110 within one episode. And the old best-of-4 anchor — seed 2's
recorded 0.530 — corrects to 0.460, displaced by seed 1 as the true v27 best.
A near-arbitrary selector predicts exactly that: the recorded per-seed ranking
was itself an artifact.

## The fork, §4.2, read as registered

0.500 ≤ 0.570 → **"capacity gets its first honestly-measured data point. A
capacity campaign becomes registrable, and must satisfy §4.3 before it
launches."**

Not read into this: the +0.170 is a best-of-4 difference at n=4 per arm with
mixed per-seed deltas underneath (v27 {0.110, 0.500, 0.460, 0.460} vs v28
{0.640*, 0.440*, 0.580*, 0.670} — v28 per-seed split-sample values from the F0
ladder). The delta earns the registration, not the conclusion; §4.3 item 4's
power statement governs what the campaign itself could show.

## What this closes

The one-sided-correction objection (MISTAKES.md 2026-08-28, "a delta may only
be cited when both ends were measured under the same estimator") is resolved by
measurement: both ends now measured under the same estimator, and the effect
survived the correction it was suspected of depending on. The capacity pivot,
suspended by the direction review, is reinstated as *registrable* — with all
seven §4.3 preregistration items and all three §5 process repairs owed before
any training arm launches.
