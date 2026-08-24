# Recurrent bottleneck A/B — VERDICT: FAIL (mechanism untested)

Adjudicated 2026-08-23 against the pre-registered gate in
docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md. Receipts:
runs/gru_ab/verdict_seed{0..3}.json, training logs
runs/gru_ab/train_seed{0..3}.log, checkpoints
checkpoints/mario_1_1_backward_gru_seed{0..3}/.

## Result

| seed | entrance arrival | final trailing entrance | honest sticky (100 eps) | deterministic (20 eps) |
|------|-----------------|------------------------|------------------------|-----------------------|
| 0 | iter 195 | 0.03 | 0.00 | 0.00 |
| 1 | iter ~100* | 0.33 (rising at cutoff) | **0.06** | 0.05 |
| 2 | iter ~200 | 0.10 | 0.01 | 0.00 |
| 3 | iter 182 | 0.10 | 0.01 | 0.00 |

*seed 1 stalled at rung 277 from iter ~45–100, then marched to the
entrance; seeds 2 and 3 stalled at the same rung. Honest per-seed score
= max over both preserved artifacts (winners/best.pt and the 10-iter
grid peak), the same selection form that produced the control's banked
number.

**Treatment best-of-4: 0.06. Control (banked, not re-run): 0.76.**
Gate: FAIL — 6/100 vs 76/100 is separated far beyond the one-sided 95%
binomial bound in the control's favor. Not VOID: the policy class
provably armed (`[policy] class=TileRecurrentPolicyNetwork
params=48975` per run), actors trained (entropy moving, 19 rung
advances per seed, distinct per-seed fingerprints), evals loaded
`recurrent: true`.

## The adjudication nuance that matters

The pre-registered secondary metric (deterministic↔sticky gap) shows a
signature UNLIKE the control class:

- Control class: deterministic-competent (rung gates 10/10) collapsing
  under sticky to 0.21–0.51 — a robustness failure.
- GRU class here: deterministic ≈ sticky ≈ 0 on every seed
  (0.00/0.05/0.00/0.00 vs 0.00/0.06/0.01/0.01) — a **learning**
  failure. The gap "narrowed" only vacuously.

Therefore **v25's mechanism claim (recurrence detects sticks and
restores closed-loop control) was never tested**: the treatment never
reached the deterministic competence where stickiness becomes the
binding constraint. What this experiment actually measured: the
control's exact recipe and 250-iter budget do not train
TileRecurrentPolicyNetwork to competence on 1-1 backward.

## Salvage ranking (each a candidate registered follow-up, none run)

1. **Sequence-BC.** The recipe's 30 bc_epochs + bc_replay clone through
   the GRU's stateless fallback (zero hidden). The control enters PPO
   BC-shaped; the GRU enters PPO with untrained recurrent dynamics and
   an actor whose BC gradient never saw a hidden state. This is the
   single largest known asymmetry between the arms and was accepted in
   the registration as "cannot explain a between-arm difference in the
   PPO phase" — that acceptance now looks wrong: it can explain a
   difference in where PPO STARTS.
2. **Budget.** All four seeds choked at rung 277; entrance arrivals at
   iters 182–223 left thin consolidation windows; seed 1's trailing
   rate was still rising at cutoff (0.026→0.33 over the final 80
   iters). An extension run (500 iters) is the cheapest direct probe.
3. **Hidden-reset diagnostic.** Verify hidden state actually resets at
   backward-curriculum rung restarts (the done-flag path) — a silent
   carry-across-restarts bug would poison exactly this training shape.
   Cheap: instrument one rollout, count resets vs episode boundaries.

## Standing per the two-ledger discipline

FAIL recorded; the sticky wall remains open; the banked per-level rates
(43/38/21/51) stand. Depth work returns to the research loop (v26
carries this autopsy). Breadth (odometer line: Rygar gx 5552, NG gx
5838, Zelda/Metroid nametable engine) is unaffected.
