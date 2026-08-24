# Recovery distillation on 1-1 — pre-registration

Registered 2026-08-24, before any distillation training ran. Sequel #1
from docs/research/RECOVERY_ASSAY_VERDICT_2026-08-24.md.

## Premise (measured, not assumed)

A third of the control's sticky deaths on 1-1 happen where a
recovering continuation exists (3/9 solver-adjudicated, receipts in
runs/recovery_assay/solve_ep*). The solver mints those recoveries as
replay-verified action tapes. If the policy can absorb them without
losing what it knows, honest 1-1 moves from 0.767 toward the measured
ceiling (~0.83–0.85).

## Fuel mining (phase 0)

The 3 existing recovery tapes are too thin for training. Mine more:
- Adjudicate EARLIER sticks in the same death episodes (not only the
  last one) and death-sticks from fresh collections at other eval
  seeds. Target: ≥30 recovery tapes, each a (post-stick state,
  verified action tape) pair.
- Every tape is replay-verified by the solver before banking (its
  existing discipline). No tape, no training.

## Method

Fine-tune the banked control (backward_1_1_seed3_iter140.pt) with
short-clip behavior cloning on recovery tapes, mixed with on-manifold
clips from the policy's own clearing episodes (46 banked clears in
runs/recovery_assay/manifest.json) to anchor against drift:
- Clip form: (start state obs sequence, actions) for the first ~60
  steps after the stick — the correction, not the whole level.
- Mix ratio recovery:on-manifold 1:1; low LR (1e-4); ≤10 epochs;
  honest mini-eval (30 eps) after every epoch; preserve-on-peak.

## Named risk (from our own ledger)

Pure imitation of solver tapes previously produced clone-accuracy 1.0
and honest sticky 0.00 (the "imitation ELIMINATED" verdict), and DR
v10's off-manifold drift barrier is real. This experiment is NOT that:
it starts from a sticky-competent policy and applies short local
corrections with an on-manifold anchor. If honest rate DROPS below
0.70 at any epoch, stop and record — that is the drift signature
reasserting itself, and the verdict is FAIL-by-drift.

## Gate

Honest protocol on the distilled artifact (cold entrance, greedy,
sticky 0.25, jitter ±16, 50 eps × 2 eval seeds):
- **PASS**: pooled ≥ 0.80 (the assay's trainable-slice prediction).
- **FAIL**: ≤ 0.767 with separation, or the drift stop fired.
- **VOID**: fuel mining yields <15 verified tapes (premise untestable
  at scale), or preflight refuses.

Either verdict generalizes: PASS makes recovery distillation a
standard post-solve step for every game with a solver; FAIL bounds
what solver-as-teacher can do at this parameter budget and hands v27
a precise question.
