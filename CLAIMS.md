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

A third ledger, **FORGE**, is defined below. It classifies machinery
rather than play; LEARNED and EXHIBITION remain the only two ledgers a
*result* can belong to.

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
  inside a run (LLM priors contain walkthroughs — the banned class in
  automated form; "Where the purity line sits" below states exactly
  what agentic work is and is not permitted); presenting
  deterministic replay as learning.

## The FORGE ledger (machinery, not play)

The other two ledgers classify *play*: what produced the frames. FORGE
classifies *machinery*: how the system acquired the capability that
produced them. It exists because "the pipeline diagnosed its own wall
and built itself a new search arm" is a distinctive claim that is true
as stated — and because the only honest way to make it is with the
word "learned" nowhere near it.

**Definition.** A result is FORGE-class when all four hold:

1. **Self-measured detection.** The need for the mechanism was found in
   the system's own telemetry — archive and cell statistics, selection
   counts, coverage histograms, progress lines, receipts from its own
   runs. Not from a human noticing, not from a walkthrough, not from
   game internals.
2. **Agentic authorship.** Design, implementation and adversarial
   review were performed by the agentic pipeline with no human
   algorithmic contribution. The owner authorizes the work, supplies
   the compute, and the commit lands under the owner's name; the claim
   is about who chose the algorithm, not about who pressed enter.
3. **The standard gates.** The mechanism ships default-off and
   byte-identical at shipped defaults — mutation-tested, not
   asserted — with the tests that prove it, and it carries a stated
   validation gate.
4. **Honest status.** The entry records whether that validation gate
   has actually been met, in those words.

**What FORGE does not claim.** It is not a learning claim and it is not
a result claim. A forged arm that later clears a level produces an
EXHIBITION-class clear like any other search output; that clear is
logged in its own ledger, on its own terms, and the two are never
merged into one sentence. FORGE entries carry no clear rate, no episode
count and no protocol number — those belong to the ledgers that define
them.

**Vocabulary, binding on every public surface** (README, stream
overlays, commit subjects, talks, posts): the words *learn*,
*learned*, *learns*, *learning* and *self-taught* are banned for
FORGE-class results. The approved verbs are **diagnosed**, **forged**,
**built**, and **extended itself**. "The system diagnosed its own wall
and forged a new search arm" is sayable. "The system learned to climb"
is not, whatever the arm goes on to do.

### Where the purity line sits

Tier 3 bans LLM guidance of rewards or exploration because LLM priors
contain walkthroughs. The T1 (agent diagnoses a stuck run) and T2
(agent forges a new arm) tiers in
`docs/proposals/TOTALITY_BASIS_2026-08-08.md` point agents at live
runs, so the boundary is stated here rather than assumed:

- **Telemetry in.** An agent may read anything the system measured
  about itself: archive and cell statistics, selection and visit
  counts, coverage and score histograms, progress lines, action
  receipts, and save-states the system captured from its own play.
- **Mechanisms out.** An agent may produce game-agnostic machinery —
  selection rules, arms, gates, knobs — and may set those knobs for a
  run when the setting is derived from that telemetry and is
  expressible without reference to game content. The setting is
  recorded with the run.
- **Never.** Routes, maps, walkthroughs, disassembly, RAM semantics
  beyond the observables the system discovers for itself, hand-authored
  input segments, per-game reward shaping, or any instruction naming
  where to go in a specific game. An agent recalling a walkthrough from
  its priors is exactly the Tier-3 injection the clause bans; the ban
  does not weaken because the recall was automated.

The same rule binds the human operator: a run whose flags were chosen
by reading a map is Tier-3 contaminated regardless of who typed them.
The test in both directions is *could this decision have been made by a
party who has never seen the game?* If not, it is banned.

### FORGE entries

**FORGE-PENDING-VALIDATION — the orthogonal-frontier arm (`--ortho`),
commit `5f8dcb7`, 2026-08-08.** Castlevania block 3's hall had stalled
under the five existing arms. The diagnosis came from the run's own
selection telemetry: with the archive score `sect * 10000 + gx` and
`sect` permanently 0, the deep arm sampled only gx buckets ≥ 71 while
every cell above y-band 8 sat at buckets 1–18 — the probability of
ever selecting a climb cell was 0, and 34 of 95 columns had never been
explored above y-band 15 across 10.6M steps. The cell key was found
*not* to be at fault (the stairs bit was already present on 40% of
cells) and was deliberately left alone, keeping banked archives
resumable. The forged mechanism is an opt-in selection arm — `--ortho
{off,up,down}` with `--ortho-pin-secs`, `--ortho-bias`, `--ortho-band`,
`--ortho-weight` and `--ortho-macro-p` — that redirects a slice of
selection pressure to per-column orthogonal extremes once the primary
frontier is pinned, with count-based decay so hammered cells yield. It
composes with the doors, barren and state-signature arms, and
`--inversion-pin-secs` lifts the previously hardcoded
heuristic-inversion trigger into a knob so the two can be sequenced
instead of fighting. The adversarial review round found and fixed an
area-restriction blocker and a default-identity coverage hole. Gate:
default-off byte-identity, covered by fifteen cases naming the arm in
`tests/test_go_explore_solve.py`, including a runtime inertness mirror
at the shipped CLI defaults.

*Status, stated plainly: no validation run has been performed.* The arm
is forged and reviewed; it has not been shown to help. The hall remains
unsolved, the standing prior is ~110M steps and 0 solutions across five
arms, and the validation is the pre-registered fidelity-corrected +
`--ortho` run scheduled in `docs/proposals/STRATEGY_2026-08-08.md` with
its stopping rule declared in advance. Until that run reports, this
entry may be cited only as *agent-forged, unvalidated*, and no clear of
any kind may be attributed to it.

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

## Documented negatives (Learned ledger)

`docs/research/RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md` records the
measured negative for World 1-2 under the honest protocol: the policy
class (compact feedforward on tile observations) was falsified by a
pre-registered, externally-reviewed protocol — SPRT-verified local
sticky-robustness at 1,900+ zones does not compose into traversal, and
the gauntlet core has a measured local sticky ceiling of ~0.03–0.05.
The companion literature audit found no published agent by any method
that clears 1-2 under this protocol. Negative results carry the same
evidentiary standard as positives and are quotable with their data.
