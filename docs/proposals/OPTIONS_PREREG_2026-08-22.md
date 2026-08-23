# Pre-registration: action-commitment options vs the sticky wall

v21's top-ranked intervention, registered before any code runs.

## Mechanism, and why it attacks sticky noise specifically

The honest protocol perturbs EXECUTION: each env step, with p=0.25 the
executed action is the previous executed action, not the chosen one. At
per-step decision frequency every step is a fresh chance for the noise to
diverge chosen from executed. But a sticky repeat is only harmful when it
DIFFERS from the chosen action — if the policy commits to one primitive
for k consecutive steps, then within the commitment the previous executed
action almost always equals the chosen one, and the sticky repeat becomes
a no-op. Commitment does not fight the noise; it makes the noise
harmless by construction. (This also explains a banked observation:
1-2's greedy policies already emit long runs of RIGHT+A+B; options make
that structure explicit and optimizable.)

## Design (Tier 0 — no game knowledge)

- **Option set**: all pairs (primitive action a, duration k) for the
  profile's 6 primitives and k ∈ {1, 2, 4} env steps → 18 decisions.
  Durations are in env steps (frame_skip 4), so max commitment is 16
  frames ≈ 0.27 s — long enough to bridge sticky slips, short enough to
  react.
- **CommitmentPolicy**: a stateful wrapper. At timer=0 it samples an
  (a, k) pair and starts a commitment; while timer>0 it emits the
  committed primitive. State rides the recurrent-policy plumbing that
  eval_game and the trainer already support — THE EVAL HARNESS IS
  UNTOUCHED. Sticky, jitter, episode accounting: all identical to every
  banked number. Nothing about the protocol changes; only the policy's
  internal structure does.
- **Warm start**: trunk copied from the banked 1-2 policy
  (consol2_40pct_strict_iter01120, sha 413548b9...). The 18-way actor
  head is initialized by replicating each primitive's actor row across
  its three durations, so the initial distribution over PRIMITIVES
  matches the seed policy and durations start uniform. The critic copies
  unchanged.
- **PPO over decisions (semi-MDP)**: one log-prob per commitment, reward
  summed over the commitment, GAE over decision boundaries with gamma^k.
  This is the correctness point Phase 3 made vivid: the behaviour and
  update distributions must be the same object, so the commitment
  machinery lives in the policy module used by BOTH rollout and update.

## Arms and gate

- **Control**: the Phase-3 control arm — already trained (200 iters from
  the same seed, no interventions) and already honestly evaluated:
  **31/100 pooled** (runs/phase3/eval2_control_seed{7,101}.json). Reused,
  not retrained; it is exactly the "same budget, no mechanism" arm.
- **Treatment**: options arm, same seed checkpoint, same 200-iter budget,
  same profile in all other respects.
- **GATE**: pooled strict honest clear rate over ≥100 episodes across
  seeds {7, 101} must be **≥ 0.372** (a ≥20% relative improvement over
  the control's 0.31). Adjudicated by scripts/phase3_adjudicate.py
  unchanged — same math, same UNSCORABLE rules, same no-rescue clause:
  on FAIL, no duration-set tuning, no budget extension, no level change.
- Secondary observables (reported, not gated): mean commitment length
  chosen, fraction of sticky events falling inside commitments,
  divergence of chosen-vs-executed streams.

## Kill criteria during training

KL to the warm-start prior > 0.60 sustained 2M steps, or value-loss
spike >5x baseline unrecovered — the campaign controller's standing
table, unchanged.
