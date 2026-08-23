# Recurrent bottleneck A/B — the v25 policy-class falsifier

Pre-registered 2026-08-23, BEFORE any treatment training ran.

## Provenance

Deep-research v25 (responses/20260823T161759Z_v25_sticky_wall.md) was
handed every falsification we own — hazard veto 31→0/100 with 77.4%
self-veto; commitment options honest 0/100 with 93.6% k=4
overcommitment; peak instability −74%/200 iters under continued PPO
while a consolidation endpoint held 51=51; sticky-on training already
standard — and asked to argue for exactly one of (a) another mechanism,
(b) policy-class change, (c) accept-and-route-around.

Verdict: **(b)** — smallest sufficient change is a minimal recurrent
bottleneck at matched parameter budget. Mechanism claim: p=0.25 sticky
makes the env a POMDP; the executed action is unobservable, so a
feedforward policy cannot distinguish "my input was applied" from "my
input was replaced," even with a 4-frame stack (the stack shows the
motion, not the intent). A recurrent state carries intent; intent plus
observed motion detects sticks and enables closed-loop correction.

## Arms

- **Control (banked, not re-run):** `configs/mario_1_1_backward.yaml`,
  TilePolicyNetwork 64/32 (48,135 params at feature_dim 712).
  Banked seed distribution honest-greedy 0.03–0.775, best-of-N 0.76
  (`checkpoints/_preserved/backward_1_1_seed3_iter140.pt`), 250 iters,
  num_envs 60.
- **Treatment:** `configs/mario_1_1_backward_gru.yaml` —
  TileRecurrentPolicyNetwork 56/32 (48,975 params, +1.7%). The ONLY
  config diff is `recurrent: true` + the two dim knobs (verified by
  comment-stripped diff). Same iters (250), same seeds ladder
  (0,1,2,3), same launcher (`scripts/train_game.py`).

Single variable = policy class at matched budget. The dim change is
part of the treatment definition, not a confound: keeping 64/32 would
compare classes at +15% capacity.

## Protocol (immutable)

Honest eval per arm seed: cold start from the 1-1 entrance, greedy,
sticky p=0.25, jitter ±16, ≥100 episodes over 2 eval seeds, strict
predicate — `scripts/eval_game.py`, the same harness that banked the
control numbers. Preflight before scoring anything:
`scripts/experiment_preflight.py` (actor-liveness, mechanism-armed,
sentinel scan) plus the fingerprint-refusal check in
`scripts/phase3_adjudicate.py`. Preserve-on-peak applies to the
treatment exactly as it did to the control (the `_preserved/` snapshot
discipline): the scored artifact is each seed's best preserved
checkpoint, mirroring how 0.76 was banked.

## Pre-registered gate

Primary metric: best-of-4 honest strict rate, treatment vs banked 0.76.

- **PASS** — treatment best-of-4 ≥ 0.85, or ≥ 0.76 + one-sided 95%
  binomial separation on ≥100 eps. Recurrence is adopted as the policy
  class for the sticky line; next action = port to 1-2/1-3/1-4.
- **FAIL** — treatment best-of-4 ≤ 0.76 − separation, or the
  deterministic↔sticky gap (secondary, below) does not narrow.
  Verdict recorded, salvage ranked, v26 asks the next question.
- **VOID** — preflight refuses (frozen actor, dead mechanism,
  identical fingerprints), or training crashes before iter 150 on ≥2
  seeds. Nothing was tested; fix and requeue.

Secondary (mechanism signature, reported either way): per-seed
deterministic rung-gate pass rate vs honest sticky rate. v25 predicts
the RATIO narrows for the GRU even where the absolute rate ties.

Tertiary (report only): stick-recovery probe — from matched mid-air
states, force one sticky repeat and measure P(recover) per arm.

## Failure modes named in advance

- BC phases (`bc_epochs`, `bc_replay`) drive the GRU through its
  stateless fallback (zero hidden, warns once). Accepted: BC seeds
  behaviors, PPO trains recurrence. Cannot explain a between-arm
  difference in the PPO phase.
- Backward-curriculum rung restarts are episode boundaries → hidden
  resets ride the done-flag path. Verified in smoke before launch.
- Seed variance in the control arm is huge (0.03–0.775). Best-of-4 vs
  best-of-4 is the banked comparison form; per-seed deltas are noise.
