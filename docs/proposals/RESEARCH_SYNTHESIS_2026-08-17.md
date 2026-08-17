# Research synthesis, 2026-08-17 — five rounds, one convergent answer

Five Deep Research consultations were submitted the same evening, each on
a different open front. Reports are private
(`~/Documents/research-consult/responses/`); this document records what
they settled, what they killed, and the build order that follows. Written
after the 1-2 / 1-3 / 1-4 campaigns produced measured per-level rates, so
every recommendation below was made against real numbers rather than
hypotheticals.

## The convergence

Three rounds asked different questions and returned the same artifact:

- **v18** (how to build a cumulative substrate that survives level
  changes) → a discrete-time **hazard model** trained by survival
  analysis on our own deaths.
- **v19** (why committed-action combat resists our search) → *"a hazard
  model is a vastly superior artifact for the combat class"*, framed as
  Survival Value Learning, explicitly preferred over a policy prior
  because TD bootstrapping is unstable in combat.
- **v20** (is 20-43% the ceiling) → shared substrate with per-level
  heads ranked #1 of five interventions; the substrate's content is the
  same affordance/hazard knowledge.

Independent questions, one answer. That is the strongest signal any of
these consultation rounds has produced, and it sets the build order.

## What each round settled

### v16 — room-graph / class 8-9-10 (Max tier)
A purity-compatible design for searching a **dependency graph over
discovered capabilities**: filter the 2KB RAM through an ST-DIM-style
pass to strip high-frequency noise; encode the stable remnant with a
LatPlan-style state autoencoder so a "capability" is a consistent
**bit-flip in a binary propositional latent**, never a named item;
establish edges by **counterfactual splicing** (restore a historically
blocked save-state, splice in the new RAM subset, test whether
reachability extends past the old bounding box); then plan over the
resulting latent operators with a classical PDDL planner.
**Killed, with reasons:** class 10 (menu/economy RPGs) is *"entirely out
of reach for a purely unsupervised, zero-prior system."* Classes 8 and 9
ground progression in a spatial manifold, so coordinate expansion works
as intrinsic reward without semantics; RPG progression flips invisible
narrative flags that produce no structural novelty for hours. Under our
purity line there is no signal to guide it. Zelda stays on the roadmap;
Dragon Warrior comes off it.

### v17 — outcome vocabulary for comparison-shaped wins
Four pre-registerable outcome types (`SCORE_WIN_VS_OPPONENT`,
`POSITION_TARGET_MET`, `TIME_TARGET_BEATEN`, `THRESHOLD_REACHED`), each
with an evidence standard. The key mechanism is **transfer entropy from
the controller** as the discriminator for ownership: the player's score
responds to our input, the opponent's does not, and a timer responds to
nobody — all measured, none authored.
**Decisive first experiment:** point the pipeline at R.C. Pro-Am (or
Excitebike / Mach Rider). If it auto-identifies the position byte, tags
`POSITION_TARGET_MET`, and secures a podium, detection was the sole
blocker and ~140 titles open. If it identifies the condition correctly
but cannot drive, the hypothesis is falsified and control is the wall.
Difficulty order: time-trials < continuous racing < individual sports <
team sports < board/casino.

### v18 — the cumulative substrate (build order below)
Three gated phases, each with a numeric kill criterion. Cross-game
transfer is explicitly **recipe transfer, not weight transfer** — the
asset that crosses games is the causal data-generation procedure, not
the network.

### v19 — the combat classes (Max tier)
For Castlevania's hall specifically: **remove spatial (X,Y) from the
archive cell key entirely.** Cell identity becomes a VQ-VAE latent
codebook ID over a 16-frame window of RAM + APU channels (K=512, k-means++
reinit against codebook collapse). Selection becomes hazard-weighted UCB
preferring cells with high *variance* in survival time. Rollout actions
pass through a hard hazard veto (>90% death within 30 frames → resample)
and accepted actions are **committed for N frames** so action
commitments materialize. Pre-registered gate/kill table included
(codebook discovery rate, hazard overconfidence / bang-bang trap, agency
tripwire). Also specifies an 8-hour dual-arm experiment that settles
policy-prior vs hazard-prior empirically.

### v20 — the rate ceiling
**Envelope estimator computable from our own logs:** λ = −ln(C)/L per
level; predicted rate for a new level = exp(−λ·L_new). Diagnostic for
under-trained vs saturated: **death-position variance** — spiked at one
coordinate means under-trained, uniform means at the reactive envelope.
Verdict: 20-43% is a *local* envelope set by sticky noise, not a global
ceiling, and isolated PPO runs additionally suffer plasticity loss and
representation collapse (which is why peaks do not hold).
Ranked: (1) shared substrate + per-level heads, (2) temporal abstraction
/ options — projected 38% → >60% by bypassing the per-step sticky
penalty, (3) SWA over the consolidation peak, (4) capacity — **negligible
online effect**, (5) previous-action feature — **catastrophic**.
**Explains a banked mystery:** our measured previous-action result (offline
fit 0.531→0.747, honest clears 0.0) is causal confusion — the agent
learns to copy its own last action instead of reading state, which
shatters the instant sticky noise perturbs it.

## Build order (gated; nothing later starts until the gate before it passes)

**Phase 1 — the causal data engine.** Micro-forking in the environment
wrapper: from a buffer of saved states, fork actions, step forward, log
survival or death as labeled transitions (s, a, t_death).
*Gate:* 100,000 cleanly labeled transitions in under 1 hour on the M4.
*Kill:* if branching drops throughput below 1,000 steps/s, abandon causal
forking and fall back to observational deaths from ordinary rollout logs.

**Phase 2 — the hazard substrate.** A ~100k-parameter MLP trained with a
discrete-time survival loss on the causal dataset.
*Gate:* Uno's C-index with IPCW ≥ 0.85 on a held-out trajectory set
(IPCW because successful trajectories are massively right-censored).
*Kill:* below 0.85 the 13x13 tile observation lacks the resolution to see
threats — do not integrate; fix the observation or stop.

**Phase 3 — policy integration.** Freeze the substrate; during PPO, mask
any action with predicted hazard > 0.90 via a −∞ logit penalty.
*Gate:* strict honest clear rate over ≥100 episodes improves ≥20%
relative to an unmasked control on a mid-difficulty level.
*Kill:* below that, terminate the substrate experiment and revert to
isolated per-level policies.

Only after Phase 3 passes do the dependent items become worth building:
shared trunk + per-level heads (v20 #1), the hall attack with latent
cells and hazard-weighted UCB (v19), and options/temporal abstraction
(v20 #2, the largest single projected gain).

## Immediately available, no new training required

- **SWA on 1-2's existing checkpoints** around its 12/30 peak — a 4-hour
  block, no new rollouts, tests whether the transient peak can be
  captured by averaging instead of by preserve-on-peak luck.
- **The envelope estimator on all four levels** — pure log analysis;
  tells us per level whether we are under-trained or saturated, which
  decides whether more campaigns on existing levels are worth anything.
- **Shared-substrate falsifier** — one shared base + four heads trained
  on 1-1..1-4 simultaneously, compared against the sum of the isolated
  baselines, which we already own (43 + 38 + 21 + ~20 per 100).

## Standing corrections this synthesis makes to earlier plans

- Naive multi-level pooling stays falsified (our own measurement), but
  trunk-plus-heads is *not* the same experiment and is now ranked first.
- Capacity scaling was previously assumed to help because offline fit
  scaled with width; v20 says the online effect is negligible because the
  cap is noise, not parameters. Do not spend a campaign on width.
- The previous-action feature should not be revisited; its mechanism of
  failure is now understood.
- RPGs (class 10) leave the roadmap until a non-purity-violating signal
  exists. This is a scope decision, not a deferral.
