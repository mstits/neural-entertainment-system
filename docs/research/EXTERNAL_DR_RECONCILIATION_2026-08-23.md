# Reconciliation: user-supplied DR report "Complex NES AI Research Analysis"

19-page Gemini DR survey (harnesses, Rainbow/PPO/ICM/RND/Go-Explore
primers, per-game strategies for Zelda/FF/Contra/Castlevania/Metroid).
Reviewed 2026-08-23 against v16/v19/v21/v22 and our measured results.

## Adopted signal

1. **Third convergence on the v16 room-graph design.** Its Metroid
   recipe (cite 43, "Automated Testing in Super Metroid with
   Abstraction-Guided Exploration": VAE latent embeddings + topological
   room graph with validated transition edges + archive-driven
   backtracking) matches v16's latent/capability-graph prescription and
   zeldabot's field decomposition. Three independent sources now agree
   on the class-8/9 architecture; v16 build priority rises accordingly.
2. **The options build mirrors combat-class physics.** Castlevania's
   inputs lock for multiple frames (jumps unalterable mid-air); agents
   there need committed actions and split navigation/combat policies.
   Our CommitmentPolicy is therefore not only sticky mitigation — it
   matches the native dynamics of the combat class, making Castlevania
   the natural second target if the 1-2 options gate passes.
3. **Independent sighting of the passivity trap.** Its Contra note —
   risk-averse selection "reduces suicidal actions but stalls in
   passive local optima" — is the failure our Phase-3 veto measured
   (0/100, dies standing still), seen elsewhere.

## Rejected as Tier-3

The Zelda RAM map ($00EB/$0070/$0084/$066F/$0670/$067C/$062A) and the
reward formulas built on it, and Castlevania's boss-HP reward: authored
per-game semantics from the gym-zelda-1/zelda-bot lineage. Our
observables continue to come from the 3-probe/lexicographic discovery
machinery only.

## No change

Algorithm primers (already running or superseded by v20-v22); the Final
Fantasy/MCTS section (sources are an FFXIV crafting bot and a card-game
post, not NES FF — v16's class-10 scoping stands); Contra x-scroll
reward (already self-discovered, legally).

## Addendum: second user-supplied report (Zelda-specific, 17pp)

Same genre and source pool; four net-new adoptable signals:

1. **Animation-lock gating** — Zelda ignores inputs during swing/item
   animations; good harnesses step the policy only when an action can
   alter state. Third independent frame on the commitment-options
   mechanism (after sticky robustness and Castlevania's locked-jump
   physics), and "is input live?" is discoverable by counterfactual
   fork-probe with no game knowledge.
2. **Egocentric viewport cropping** for combat generalization — legal
   with self-discovered coordinates; candidate fix for combat classes
   where our screen-fixed tile grid may itself be a handicap.
3. **OAM entity-relational features** — the hardware sprite table is a
   game-agnostic surface (like pixels), distinct from a game RAM map;
   aligns with existing probe_entity_slots work.
4. **Zelda death is NON-TERMINAL** (auto continue-screen loop): future
   Zelda predicates must not assume episode termination; death must be
   discovered (hearts-to-zero + warp signature).

Confirmed from the field: wavefront distance-gradient navigation is
their standard too — theirs from parsed layouts (authored), ours from
own solved tapes (legal). Rejected again: the info-dict RAM map and all
reward formulas over named fields.

## Addendum 2: the primary source itself — DarkAutumn/Triforce (Reddit
r/MachineLearning 1i3t4c3) and victorsevero/megai_man

Both surveys' Zelda content traces to this hobby project (PPO beats
Zelda dungeon 1; disassembly-assisted — Zelda1 speedrun community pulled
the RAM addresses, so its OBSERVABLES are Tier-3 for us; mechanisms
only). First-hand details worth keeping:

1. **Delayed-effect reward resolution by emulator time travel.** For
   bombs (seconds between action and payoff), they run the emulator
   forward to the outcome, assign the reward at the decision frame, and
   REWIND to continue. Purity-clean training mechanism in the same
   family as our micro-forking; directly relevant to Zelda bombs and
   Castlevania sub-weapons if we ever need credit assignment for slow
   projectiles.
2. **The reward debugger was their highest-leverage tool** — "75% of
   the time the model behaved incorrectly, the rewards were wrong";
   click any reward event to replay it with breakpoints and stepping.
   We have campaign logs and eval receipts but no step-level
   reward-replay tool. Candidate FORGE item.
3. **Author independently abandoned the objective vector as "too much
   like cheating"** — outside confirmation of our purity intuition,
   from someone with no ledger to defend.
4. **Compute calibration.** RTX 4090 + 10-core: ~100-185 steps/s; our
   M4 runs ~3,000 env-steps/s — 15-30x their throughput. Their scale:
   okay results ~2M, good ~10M, plateau 40-50M steps per model — the
   same regime as our per-level campaigns, so our budgets are not
   under-provisioned by hobby-project standards.
5. **megai_man** (Mega Man Cut Man stage CLEARED): pre-train the stage
   with shooting MASKED and enemies absent, then fine-tune with
   enemies; wavefront gradient for stage phase, simple damage rewards
   for boss phase; ~57M steps total. An action-curriculum data point
   for the combat classes (mask-then-unmask), and yet another
   independent wavefront user.
6. Confirmed from the primary source: egocentric viewport ("REALLY
   helped"), animation-lock gating via RAM state, tile-aligned movement
   commitment — all already in our adopted list; frame-stacking gave
   them little once entity vectors existed.
