# Claims Policy

This project's product is an agent that **learns** to play NES games.
Every public number, stream overlay, and README claim follows this
policy. It exists because the difference between a learned policy and a
replayed trajectory is invisible in a highlight clip — and because the
field has named tools for telling them apart.

## The two ledgers

Every artifact and every number belongs to exactly one ledger:

- **LEARNED** — a policy trained by reinforcement learning (with or
  without self-generated demonstrations; see tiers below), evaluated by
  the honest protocol. Only Learned-ledger results may be described
  with the words "the AI learned/plays/beat".
- **EXHIBITION** — search output: Go-Explore/beam solutions, BC clones
  of single trajectories ("pilots"), routed replay chains. Legitimate,
  interesting, and always labeled as what it is: the *search system*
  solving the game, in the tradition of Baumgarten's A* (Mario AI
  Championship 2009) and the ALE "Brute" (Machado et al. 2018). Never
  presented as learning.

## The honest evaluation protocol (the only headline numbers)

Cold power-on start, zero state loads at test time, single-life
denominators, sticky-actions 0.25 + start-jitter 16 (Machado et al.
2018, JAIR 61), at least 50 episodes on each of two seeds, greedy and
sampled action selection both declared, action receipts recorded and
self-replay verified. A deterministic number is never shown without the
sticky number beside it. Every quoted rate names its harness (training
telemetry vs cold greedy vs sticky pair). Per-level rates are the
scoreboard; chain rates are reported as measured, alongside the honest
compounding math (e.g. twelve levels at 0.95 each ≈ 0.54 chain).

## Knowledge-injection tiers

- **Tier 0 — clean.** Live rollouts; generic distance-ladder reward
  (every-256px) + time/death penalties (the published SMB reward norm);
  novelty bonuses on layout-agnostic cells (gx bucket, y band, area
  byte); self-imitation of the agent's own training clears; the full
  controller including DOWN; sticky-on training; recurrent nets.
  Claim: "the agent learned by playing." No qualifier needed.
- **Tier 1 — accepted with disclosure** (the Nature Go-Explore class).
  Save-state exploration archives and returns (training only);
  backward-algorithm start scheduling along the agent's OWN search
  demos; demo-anchor/self-imitation losses on those demos; RAM tile
  observations; curriculum resets from own-play states. Verbatim claim
  sentence: *"The agent explores with emulator save-states it captures
  from its own play, distills its own best trajectories, and the final
  policy is evaluated from power-on under the standard
  random-perturbation protocol (Machado et al. 2018) — the same recipe
  published in Nature (Go-Explore, 2021)."* Never say "no human
  knowledge." The full injection list is disclosed here: RAM 13×13
  tiles, Δx reward ladder, save-state exploration, self-demos,
  RND/count bonuses.
- **Tier 2 — legacy, frozen, disclosed.** The five hand-calibrated
  LEVEL_* reward ladders in nes_core/src/rewards.rs predate this
  policy. They are frozen (no new hand ladders will ever be authored),
  disclosed, and retire to the generic ladder as levels are retrained
  under the honest protocol.
- **Tier 3 — banned.** Game-disassembly knowledge; human level-map or
  walkthrough route knowledge; hand-driven input segments in any
  training input; RAM edits; LLM guidance of rewards or exploration
  (LLM priors contain walkthroughs — the banned class in automated
  form); presenting deterministic replay as learning.

## Quarantine (Tier-3-contaminated artifacts)

The following artifacts were produced with banned knowledge (a
map-informed hand-driven segment for 4-2; disassembly-derived route
work for 4-4) and are quarantined in `checkpoints/QUARANTINE_tier3/`.
They must never appear in Learned-ledger training inputs:

- `demos_4_2_full.npz`, `demos_4_2_pilot.npz` (+ provenance sidecars)
- `full_4_2_solution.npy`, `full_4_2_trimmed.npy`
- `pilot_4_2.pt`

(The other castle trims — 2-4, 3-4 — cut only the input-ignored
end-of-level sequence from pure-search solutions and carry no route
knowledge; they remain Exhibition-clean. The 4-4 disassembly dossier in
the campaign doc is marked banned-knowledge and nothing derived from it
may be used.)

`runs/chain_handoffs/handoff_4-3.state` and `handoff_4-4.state` were
captured through a chain traversal that included the tainted 4-2 pilot:
usable as scaffolding for Exhibition work only, regenerated before any
Learned-ledger use.

## Enforcement

`configs/demo_allowlist.txt` is the checked-in list of demo banks
cleared for Learned-ledger training. The trainer refuses demo paths not
on the allowlist. `make provenance-check` verifies the allowlist, the
quarantine, and that no profile references quarantined artifacts.
Provenance sidecars are advisory; the allowlist is authoritative.
