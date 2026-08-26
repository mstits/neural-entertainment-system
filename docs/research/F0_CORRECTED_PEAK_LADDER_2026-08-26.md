# F0 — the corrected honest peak ladder

**Date:** 2026-08-26
**Cost:** 192 evaluations, zero training compute, 5,333 s. 192/192 ok, 0 failures.
**Headline:** the proxy selector really does under-select, on 3 of 4 seeds, by
+0.19 to +0.27 — **and the v28 verdict is unchanged at 0.670. FAIL stands.**

## What ran

The full honest protocol (cold entrance, greedy, sticky 0.25, jitter ±16, 50
episodes × eval seeds {0,1} = 100 pooled, max-steps 1500, `--eval-rng
per-episode`) on **every 10-iter checkpoint** of all four v28 runs — 24
checkpoints × 2 eval seeds × 4 runs. Receipts:
`runs/v29_stability/f0_ladder/<run>_it<NNN>_es<S>.json`, ladder in
`ladder.csv`.

## The selection defect is real

| seed | proxy iter | recorded honest@peak | F0 best iter | F0 best (n=100) | delta |
|---|---|---|---|---|---|
| 0 | 70 | 0.450 | 90 | 0.640 | **+0.190** |
| 1 | 60 | 0.230 | 90 | 0.590 | **+0.360** |
| 2 | 120 | 0.370 | 90 | 0.580 | **+0.210** |
| 3 | 90 | 0.670 | 80 | 0.720 | +0.050 |

4 of 4 improved. And the shape is striking: **the true honest peak clusters at
iter 80–90 on every seed**, while the proxy scattered its picks across 60–120.
`entrance_trailing_rate` does not find the honest peak. Seed 1 is the worst
case — its recorded 0.230 understates a 0.590 checkpoint by 2.6×.

## But the headline number does not move, and here is why that matters

Reading "F0 best-of-4 = 0.720" off the table above would be wrong, and would
repeat — in the opposite direction — exactly the error this document is about.
That figure is a **max over 24 noisy checkpoints**, while the v28 bar was set
against a rule that evaluated **two** checkpoints per seed. Taking the best of
24 draws is an optimistically biased estimator; the bar was not calibrated for
it.

The unbiased correction costs nothing, because both eval seeds already exist
for every checkpoint: **select the checkpoint on one eval seed, score it on the
held-out other.** Selection and scoring no longer share data, so the winner's
curse cannot inflate the result.

| seed | select es0 → score es1 | select es1 → score es0 | mean |
|---|---|---|---|
| 0 | it90 → 0.58 | it90 → 0.70 | 0.640 |
| 1 | it110 → 0.54 | it120 → 0.46 | 0.500 |
| 2 | it90 → 0.58 | it90 → 0.58 | 0.580 |
| 3 | it80 → 0.72 | it90 → 0.62 | 0.670 |

- recorded, proxy-selected, 2 ckpts/seed — **the registered rule**: **0.670**
- F0 max-over-24, optimistically biased, NOT the registered rule: 0.720
- **split-sample, selection-unbiased: 0.670**

The split-sample estimate lands exactly on the recorded number. **The v28
capacity experiment's verdict is unchanged: best-of-4 0.670 against PASS ≥0.80
/ FAIL ≤0.767. FAIL.**

## Why the headline survived a defect that hit 3 of 4 seeds

Because best-of-4 takes a maximum, and the one seed the proxy selected
*correctly* (seed 3, recorded 0.670) is also the best seed. The three
mis-measured seeds were all below it even after correction (0.640, 0.500,
0.580). The defect was real and large per-seed; it just did not reach the
statistic the gate reads.

That is luck, not robustness. Had seed 1 been the strongest run, the recorded
best-of-4 would have been 0.230 against a true 0.500, and v28 would have been
scored against a number 2.6× too low.

## What this changes

1. **Nothing about v27 or v28's verdicts.** Both stand as banked. This is the
   most trustworthy outcome available: an instrument defect was found, the
   correction was applied rigorously, and the headline did not move.
2. **Per-seed numbers in the v27/v28 verdicts are low** for every seed except
   the best one, and should not be quoted as that run's capability.
3. **The peak is at iter 80–90, not wherever the proxy said.** Training past
   ~90 in this recipe is not merely wasted, it is where the decay begins —
   consistent with the peak-instability finding, now with the peak located
   properly rather than through a scattered proxy.
4. **A future re-scoring rule must fix its estimator in advance.** The 0.720 /
   0.670 gap is the entire winner's-curse effect, measured: **0.05 of apparent
   headline is pure selection.**

## Unaffected

The decay past the peak is untouched and still total — every run still ends
near zero at iter 240, and preserve-on-peak remains the only reason these
experiments have numbers at all. F0 relocated the peak; it did not soften the
collapse.
