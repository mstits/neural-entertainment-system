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

## Variant A (registered 2026-08-24 evening, before running): KL anchor

Change from the failed base: the anchor term becomes
KL(net(anchor_obs) || FROZEN control(anchor_obs)) with weight 1.0 —
an explicit leash to the control's distribution — while recovery
states keep plain CE to the solver action. Two-rung LR ladder defined
in advance: lr 1e-4; if the drift-stop fires, one retry at lr 1e-5
(20 epochs). Same gate, same drift-stop, same mini-eval cadence.
FAIL only after both rungs. Nothing else changes.

## Variant A VERDICT: FAIL (both rungs)

Rung 1 (lr 1e-4 + KL leash): drift-stop at epoch 0 (0.0). Rung 2
(lr 1e-5): epoch 0 held 0.70, epoch 1 eroded to 0.60 — stop; best
artifact 0.70 < baseline 0.767 < gate 0.80. Complete adjudication of
the cloning family: cross-entropy on solver recovery actions — naive
or KL-anchored, at 1e-4 or 1e-5 — erodes the artifact faster than it
teaches. The solver's action style is off-manifold for this policy;
imitation pressure of any tested strength is net-destructive.
Receipts: runs/recovery_distill/train_history.json (both rungs'
histories in task logs).

## Variant B (registered before running): on-policy recovery PPO

Salvage 3, via existing machinery: a backward-curriculum-style run
whose restart distribution IS the mined post-stick states — the
policy learns to survive FROM those states with ordinary on-policy
PPO, no cloning target anywhere. Config = clone of the banked 1-1
recipe with: states_dir -> the 27 mined post-stick states;
kl_anchor_checkpoint -> the control (leash against peak instability,
actor NOT frozen: actor_freeze_steps 0); short run (60 iters,
checkpoints every 10); sticky-on rollouts as standard.
Gate unchanged: preserved-best honest ≥ 0.80 PASS / ≤ 0.767 FAIL;
drift semantics: any checkpoint < 0.70 honest is recorded but the run
completes (PPO variance is not BC drift; the peak selector handles
it). VOID: preflight refusal or the curriculum never restarts from
the mined states.

## Variant B VERDICT: FAIL — and the experiment's meta-conclusion

Honest scores by checkpoint: iter10 0.33, iter20 0.07, iter30 0.0,
iter40 0.07, iter50 0.0. Training telemetry matched: recovery-pool
clears 0.49 -> 0.30 and entrance rate 0.31 -> 0.06 across the run.
The KL anchor did not hold; erosion began immediately and the
recovery-state restart distribution ACCELERATED the known
peak-instability mode.

**All three salvage families are now adjudicated FAIL on this
artifact:** naive cloning (destroys at epoch 0), KL-leashed cloning
(monotone erosion at 1e-4 and 1e-5), KL-anchored on-policy PPO
(erosion from iter 1). Combined with the campaign's earlier results
(hazard veto, options, recurrence, continued-PPO collapse), the
meta-finding is:

> The consolidated 48k artifact is an ISOLATED OPTIMUM. Every
> gradient that touches it — imitation or on-policy, leashed or not —
> makes it worse. The trainable slice the assay measured exists in
> the world (the solver found the recoveries), but no post-hoc
> mechanism tested can transfer it into this artifact.

1-1's honest number remains the untouched control's 0.767. The
precise v27 question this hands the research loop: is
untrainability a property of consolidation itself (post-hoc
improvement impossible; recovery states must be in the curriculum
FROM THE START of a fresh run — config-only with this machinery), or
of the parameter budget? The fresh-run test is the registered next
candidate, sized for the engine, not run tonight.
