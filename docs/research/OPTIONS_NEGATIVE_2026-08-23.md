# Commitment options on 1-2: FAIL — overcommitment, plus a control collapse

Gate (OPTIONS_PREREG_2026-08-22): >=0.372 pooled strict honest.
**Result: control 8/100, treatment 0/100, relative -1.0. FAIL.** Both
arms verified live (pair_actor max|Δ|=0.85 from seed; fingerprints
differ; the adjudicator's frozen-arm refusal did not fire). Receipts:
runs/options/rerun2_eval_*.json, runs/options/verdict.json.

## Finding 1 — the mechanism failed by OVERCOMMITMENT

The pre-registered duration autopsy on 4,000 real 1-2 states: the final
treatment policy chooses k=4 in **93.6%** of states (greedy; 92.4%
sampled mass). Not the k=1 collapse v22 flagged as the classic
pathology — the mirror image, which v22 ALSO predicted (§4.1): under
dense per-step shaping, a k-step commitment sums ~k steps of shaped
reward into ONE decision's advantage, so long durations look better per
decision early, entrench, and starve k=1/k=2 of data. Our
duration-scaled entropy fix neutralized the entropy-farming force and
left the advantage-accumulation force unopposed.

Signature consistent everywhere: training trailing climbed to 0.40 (rung
restarts tolerate coarse control) while honest cold+sticky scored 0/100
— 1-2's pole demands single-step agility (the banked argmax-tie defect)
and a 16-frame-committed policy cannot express it.

Salvage ranking (each a NEW experiment; the no-rescue clause bars
retuning this one): (a) v22's own mitigations — per-duration advantage
normalization, tighter clip on long-k transitions, KL on the duration
marginal; (b) durations {1,2} only; (c) commitment as EVAL-time-only
smoothing on a flat-trained policy. None scheduled until the shelf
evals and any sticky-wall research land.

## Finding 2 — continued training collapses a consolidated peak (measured)

The CONTROL fell from the seed's 31/100 to **8/100** after 200
iterations of ordinary continued PPO (KL anchor active, actor live).
The campaign lore — "peaks are transient; preserve-on-peak saved 1-2,
1-3" — now has a clean number: -74% relative in 200 iters.

Design consequence for every future A/B: endpoint-vs-endpoint arms
measure mechanisms against a DEGRADING baseline. Arms should carry
preserve-on-peak (periodic honest probes, best checkpoint kept) in both
lanes, and adjudicate peak-vs-peak. This does not touch the banked
38/100 (preserved checkpoint, untouched).

## Ledger

LEARNED unchanged: 1-1 43, 1-2 38, 1-3 21, 1-4 51. The sticky wall now
carries two clean falsifications (hazard veto at eval; commitment
options trained) plus one measured design hazard (peak degradation),
which together define the next research question precisely.
