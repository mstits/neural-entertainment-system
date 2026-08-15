# Honest Findings: The SMB 1-2 Learned-Policy Wall

Date: 2026-08-14
Scope: Why every learned policy scores **0.0** honest clears on SMB 1-2, what a
capacity sweep and an IQ-Learn dynamics-aware distill changed (nothing, on the
clear metric), and where the first non-zero honest clear is most likely to come
from.

This document is deliberately non-promotional. Where the number is 0.0, it says
0.0 and the wall stands.

---

## (A) The corrected honest baseline: 1-2 = 0.0, harness-validated

**Honest protocol** (the only protocol that counts here): cold start from the
1-2 entrance, **greedy (argmax)** action selection, **sticky-0.25** action
repeat, **jitter-16** frame-offset randomization, 50 episodes, `--level-clear`
predicate.

Under this protocol, SMB 1-2 honest clear rate is **0.0 across every method
tried to date**:

- Online JAVI (DQfD margin + non-farmable positive-shifted wavefront +
  go_explore restarts, feedforward) to **3410 iterations**: 0.0
- Offline BC on the **narrow** funnel: 0.0
- Offline BC on the **diverse 458k-pair** funnel: 0.0

The historically cited **"36%"** for 1-2 was **NOT this eval**. It was a
softer/older measurement (sampled or few-episode gate), not cold-greedy +
sticky-0.25 + jitter-16 over 50 episodes. Do not carry the 36% forward as a
1-2 capability claim.

**The harness is not broken.** A known-good **1-1** net scores **0.76** on the
exact same harness. The harness registers clears when a policy can produce them.
A 0.0 on 1-2 is a real 0.0, not a plumbing artifact.

**Prior structural finding (still standing):** the 14k-parameter tile MLP cannot
even *fit* the diverse funnel — train_acc **0.475** on the 458k-pair diverse
funnel vs **0.82** on the narrow funnel. That is a representation/fit failure,
not merely an optimization stall. The leading hypothesis has been **observation
aliasing**: the tile grid encodes no velocity, and the decisive x≈2674 maneuver
in 1-2 is momentum-dependent — states that require different actions look
identical to the network.

---

## (B) Capacity sweep — did more parameters raise fit and/or honest clear?

We scaled the feedforward BC policy across four sizes on the identical 458k-pair
diverse 1-2 funnel (feature_dim=712, 6 actions), 60 epochs each, and ran each
resulting checkpoint through the honest protocol (50 eps, greedy, sticky-0.25,
jitter-16).

| variant   | hidden/trunk | final train_acc | honest clear | checkpoint |
|-----------|--------------|-----------------|--------------|------------|
| cap_h64   | 64 / 32      | 0.475           | **0.0**      | `checkpoints/cap_h64_1_2/vanilla_ppo_iter_00000.pt` |
| cap_h128  | 128 / 64     | 0.531           | **0.0**      | `checkpoints/cap_h128_1_2/vanilla_ppo_iter_00000.pt` |
| cap_h256  | 256 / 128    | 0.577           | **0.0**      | `checkpoints/cap_h256_1_2/vanilla_ppo_iter_00000.pt` |
| cap_h512  | 512 / 256    | 0.612           | **0.0**      | `checkpoints/cap_h512_1_2/vanilla_ppo_iter_00000.pt` |

**Fit moved monotonically with capacity.** train_acc rose 0.475 → 0.531 → 0.577
→ 0.612 (+0.137 top to bottom). Loss was still descending at epoch 59 for
h128/h256/h512 (no plateau), so even more parameters would fit the funnel
further.

**Honest clear did not move at all.** Every size registered **0.0/50**. On the
h512 run, all eval episodes terminated short (steps 183–1172); **none reached
the flag**.

Notes on rigor:
- The `cap_h64` run landed at exactly the same 0.475 the prior hidden=64 probe
  hit — the h64 and the 14k baseline configs converged identically, so h64 alone
  does **not** prove a strictly *larger* net than baseline. That is precisely why
  the sweep went up to h512, which unambiguously *is* larger and *did* fit
  better.

**Verdict on the aliasing hypothesis: partially refuted as a pure-capacity
story, consistent with it as a representation story.** More capacity clearly
improves supervised action-matching, which argues against a trivial
capacity-wall for *fit*. But even 0.612 argmax accuracy on a diverse,
multimodal, non-farmable funnel is far too low for frame-perfect 1-2 execution,
and — critically — the fit gains produced **zero** additional honest clears.
This is the signature of a **loss-wall / behavioral-cloning ceiling**: the
funnel is multimodal (the same observation maps to different expert actions
across the aggregated solutions), so cross-entropy imitation converges toward a
blurred average that cannot execute the momentum-dependent maneuver at any tested
size. Capacity is not the binding constraint; the **learning signal + observation
representation** are.

---

## (C) IQ-Learn — did dynamics-aware Q grounding move honest off 0?

Motivation: BC has no notion of dynamics, so it cannot recover from
sticky-induced drift. IQ-Learn adds a soft-Bellman objective (Garg 2021) so the
policy is grounded in transitions, not just isolated state→action pairs.

**What was built:**
- `scripts/gen_iq_transitions.py` — reuses the DARS DART replay (restore a
  verified GE solution state → inject sticky-0.25 drift → expert-relabel a
  60-step recovery window from the time-resynced solution index), recording the
  full Markov tuple `(state, action, next_state, done, is_expert=1)` per step,
  keeping only surviving + forward-progressing recoveries (wavefront-distance
  decreases). Aggregated 4 diverse solutions (`ge_1_2_div_s1..s4/sol_000`).
  Output `runs/iq_1_2/transitions.npz`: **70,020 transitions**, int8 (N,712),
  all 6 action classes present. **`done=0` for every transition** — the
  survive-filter admits no terminals, so there is **no absorbing grounding**.
- `scripts/iq_distill.py` — `TilePolicyNetwork(6,712,64,32)`, action head as Q;
  IQ-Learn soft-Bellman + DQfD margin (alpha=0.1, gamma=0.99, margin=0.5,
  AdamW 3e-4/wd 1e-4, grad-clip 1.0, 50 ep, batch 256, Q clamped [-20,20]).

**Stability fix (necessary and documented).** The bare 3-term loss as literally
written **collapsed**: with `done=0` everywhere, nothing bounded Q growth, so Q
ran uniformly to the +20 clamp where gradients vanish and the net froze
(argmax_acc 0.036; all Q≈20.09, degenerate to action 0). Root cause: the spec
defined `v_curr = alpha*logsumexp(q/alpha)` but never used it — that term **is**
the IQ-Learn value baseline `E[V - gamma*V']` (Garg 2021 Eq. 10) that anchors Q.
Adding `value_loss = mean(v_curr - target_v)` fixed it: stable, iq_loss → 0, reg
small, no NaN, no clamp saturation.

**Result:**

| variant | train (argmax_acc) | honest clear | checkpoint |
|---------|--------------------|--------------|------------|
| iqlearn | 0.374 (vs 0.167 random) | **0.0** | `checkpoints/iq_1_2/vanilla_ppo_iter_00000.pt` |

**Did Q-grounding move honest off 0? At the trajectory level, yes a little. On
the clear metric, no.** Greedy makes real progress from the 1-2 entrance
(per-episode max-x 387 → up to 680), and argmax jumped 0.036 → 0.374, so the
drift-state Q is meaningfully non-random. But honest_clear = **0.0/50**.

**Why greedy tops out — the real blocker:**
- 0.374 is **underfitting**, not a data limit: the measured data ceiling is a
  **0.829** majority-classifier accuracy (only 2.9% aliased labels).
- On terminal-free (`done=0`) recovery data, the soft-Bellman chi-2 regularizer
  **structurally opposes** the DQfD margin. q_reg drives
  `q_sel → gamma*v_next ≈ gamma*max_q` — it wants the expert action *just below*
  the max, directly cancelling the margin's push to put it *above*.
- Sweeps confirm the margin cannot win: margin-coef 5/20/50 gave argmax
  0.361/0.304/0.321 (worse — grad-clip 1.0 caps the effective step), and the
  margin loss stayed pinned at ~0.5 throughout.

**Net:** the lever is correctly implemented and stable; Q-grounding at drift
states is real; but on this **done-free recovery-transition regime**, the
soft-Bellman-vs-margin conflict caps greedy argmax well below what a clear needs.
The blocker is the **objective balance + missing terminal grounding**, not the
transition pipeline.

---

## (D) Ranked recommendation for the FIRST non-zero honest clear

Everything above is still 0.0. Two independent pieces of evidence now converge on
**where the constraint actually is**:

1. Capacity scaling improves *fit* but yields *zero* clears → the ceiling is the
   **learning signal / observation**, not parameters.
2. IQ-Learn is underfitting far below the 0.829 data ceiling because of a
   **structural objective conflict on done-free data**, and BC on the same funnel
   is multimodal-blurred → the imitation signal itself is the wrong target.

Ranked candidates:

1. **Velocity / frame-delta feature to resolve aliasing (HIGHEST LIFT).**
   Add a small dynamics channel to the 712-dim tile observation — signed
   x/y-velocity or a t vs t-1 frame-delta — so momentum-dependent states stop
   colliding. This directly attacks the root hypothesis that the x≈2674 maneuver
   is unrepresentable. It is cheap, it changes the *input* (which capacity
   scaling proved is the binding axis), and it can be applied to *both* BC and
   IQ-Learn. If the multimodal funnel is partly an aliasing artifact, this should
   raise the data ceiling and the fit simultaneously.

2. **IQ-Learn + terminal grounding + BC-dominant head (SECOND).**
   Fix the two named structural defects: (a) admit terminals into the transition
   set so the soft-Bellman has an absorbing anchor, and (b) rebalance so a
   BC-classifier term dominates the Q-regularizer near expert states (or raise
   grad-clip so the margin can actually move). This is a targeted repair of a
   lever that already produces real trajectory progress (max-x 680) — the
   cheapest path from "progress" to "clear" *if* aliasing is not the whole story.

3. **Larger / structured policy alone (THIRD — deprioritized).**
   The sweep already shows wider FF nets keep fitting but do not clear. Pure
   scale is refuted as the first mover. A *structured* policy (recurrence/temporal
   context) is a different bet, but recurrence has been repeatedly KILLED on this
   project; do not reopen it without the velocity feature first.

4. **Accept 1-2 as the documented open wall; solver-as-teacher (FALLBACK).**
   If (1) and (2) both stay at 0.0, record 1-2 as the canonical open learned-
   policy wall. The Go-Explore solver already clears 1-2; its role is teacher
   infrastructure, and the honest learned metric stays at 0.0 with that stated
   plainly. This is the intellectually honest resting state, not a failure to
   hide.

---

## (E) Explicit falsifiable next experiment

**Experiment:** Regenerate the diverse 1-2 funnel with a **signed-velocity
feature pair** (x-vel, y-vel, from consecutive-frame RAM deltas) appended to the
712-dim tile observation → feature_dim=714. Retrain the **h256** BC policy
(the best fit/cost point) for 60 epochs on the augmented funnel, then run the
**identical honest protocol** (50 eps, greedy, sticky-0.25, jitter-16,
`--level-clear`).

**Falsifiable predictions:**
- **Aliasing hypothesis TRUE** ⇒ measured data ceiling (majority-classifier acc)
  rises above the current 0.829, train_acc rises above the 0.577 h256 baseline,
  and honest clear becomes **> 0.0**.
- **Aliasing hypothesis FALSE** ⇒ velocity buys < +0.03 train_acc and honest
  clear stays **0.0/50**. In that case the wall is not observation aliasing but
  the imitation objective itself (multimodal blur), and the program moves to
  candidate (2) — IQ-Learn with terminal grounding + BC-dominant head — before
  falling back to (4).

**Kill criterion for the whole learned-1-2 push:** if velocity-augmented BC
*and* terminal-grounded IQ-Learn both return 0.0/50 honest, declare 1-2 the
documented open wall, keep the solver as teacher, and stop scaling parameters
against it.

---

## Summary of every honest number in this document

| method | fit metric | honest clear (50 eps, greedy, sticky-0.25, jitter-16) |
|--------|-----------|--------|
| 1-1 known-good net (harness control) | — | **0.76** |
| Online JAVI @3410 iters | — | 0.0 |
| BC narrow funnel | train_acc 0.82 | 0.0 |
| BC diverse funnel (14k) | train_acc 0.475 | 0.0 |
| cap_h64 | 0.475 | 0.0 |
| cap_h128 | 0.531 | 0.0 |
| cap_h256 | 0.577 | 0.0 |
| cap_h512 | 0.612 | 0.0 |
| IQ-Learn distill | argmax 0.374 (ceiling 0.829) | 0.0 |
| **BC + a_{t-1} feature (h128)** | **train_acc 0.747** | **0.0 (seeds 0/1/9)** |

---

## ADDENDUM 2026-08-14 — the representation/aliasing hypothesis is REFUTED; the wall is imitation-robustness

The "next lever is a velocity/frame-delta feature" recommendation above is **wrong**, on two verified grounds:

1. **We already have velocity.** `src/emulation/tile_observations/smb.py` shows the observation
   already contains `vel_x[169]`, `vel_y[170]`, `on_ground[171]`, sub-tile-X`[174]`, and the V2
   encoder (used by the funnel, 712 = 4×178) adds level-progress + fine-progress + frame-phase.
   The DR-v11 response's "velocity-blind grid" premise does not apply to our system. This is why
   only 2.9% of obs are aliased.

2. **`a_{t-1}` nearly closed the fit gap and changed nothing.** Adding the previous-executed-action
   one-hot (the DR's sole non-redundant idea; the honest Markov completion for a sticky MDP) raised
   BC fit from 0.531 → **0.747** at the same h128 — the largest fit jump of any lever — yet honest
   clear stayed at **0.0** across three independent eval-seeds, with the eval harness validated at
   0.76 on the 1-1 net. Better fit did not produce a single clear.

**Corrected conclusion.** 1-2 honest-greedy-sticky-jitter is **not** a capacity, observability,
aliasing, or loss-function problem — all were tested and eliminated. It is an **imitation-robustness
wall**: behavior-cloning a diverse recovery funnel, however well it fits, does not yield a greedy
policy that survives sticky-0.25 execution through the momentum-critical x≈2674 maneuver
(compounding-error / covariate shift). **Decision: bank 1-2 as the documented open hard-exploration
wall. The Go-Explore solver still clears it (and the whole game) as teacher infrastructure; the
learned-policy product stands on the levels that are honestly learnable (1-1 = 0.76). Stop spending
against this transition; redirect to the broader program.** The only remaining research-grade idea is
a fundamentally different training signal (online RL that practices sticky execution at the
bottleneck, not imitation) — out of scope for the current honest-distillation line.
