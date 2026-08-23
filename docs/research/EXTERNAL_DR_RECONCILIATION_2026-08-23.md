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
