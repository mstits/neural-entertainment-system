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

## VERDICT (2026-08-24 evening): FAIL-by-drift, at epoch 0

Executed with the loader fix in place (28dc163 — the first attempt
trained a random net and was void). With the REAL control loaded:
epoch 0 (13 Adam steps, lr 1e-4, 1,620 recovery + 1,620 anchor pairs)
took honest greedy 0.767 -> 0.033; the registered drift-stop fired
immediately. Sampled-mode eval of the same artifact reads 0.17 — the
damage is genuine policy degradation, not only argmax-tie flipping
(though knife-edge greedy margins likely amplify it; epoch-0 loss
3.35 >> ln(6) shows the recovery targets strongly contradict the
policy's distribution).

The named risk fired as written: short-clip + on-manifold anchor +
low LR was NOT enough to hold the manifold. Recovery knowledge cannot
be pushed into this artifact by naive cross-entropy at any useful
rate before the artifact's own competence erodes.

Salvage candidates (each needs its own registration):
1. KL-anchored distillation — add a KL penalty to the CONTROL's own
   logits on anchor states (the kl_anchor machinery exists in the
   trainer); directly opposes drift instead of hoping the data mix
   does.
2. LR 1e-5 with 10x epochs — test whether any usable rate exists
   below the damage threshold.
3. Advantage-filtered RL fine-tune on post-stick states (rollout from
   the mined states with the solver tape as a dense reference) —
   on-policy, no cloning pressure.

Receipts: runs/recovery_distill/{train_history.json, ckpts/},
runs/gru_ab/stick_probe_realpolicy.json (real-policy probe: divergence
0.056; stateless MLP AUC 0.76, GRU 0.74 — with the real policy,
recurrence adds NOTHING; the v26-override conclusion is confirmed on
corrected data).
