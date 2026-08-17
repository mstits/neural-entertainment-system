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

*Status, updated 2026-08-10 — the pre-registered A/B validation ran
(seed 303, 90 min each, both resuming the 92,785-cell stairkey
archive; receipts `runs/cv_hall_ortho_a/` vs `runs/cv_hall_ortho_ctrl/`).*
**Split verdict.** MECHANISM VALIDATED: the arm engaged (37,345
selections) and redirected exploration upward exactly as designed —
2,514 new cells above the old y-band-9 frontier vs the control's 1,471
(1.7×), heavier in every high band, at a 31% cost in total new cells
(38,776 vs 56,368). PREMISE STALE: the control ALSO reached y-band 7
without the arm — vertical starvation is no longer the hall's binding
constraint on a resumed stairkey archive; both runs pinned at gx 767,
0 solutions, and the pre-registered partial gate (ortho_cols_improved
≥ 8) failed in both (3). The wall-taxonomy discriminator
(`src/training/wall_taxonomy.py`, calibration receipted in
`docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md`, whose own
§0 status was **CONDITIONAL — the positive class is unvalidated**: both
GATED rows were the same unsolved hall read twice, not two data points)
classifies both runs GATED. Citable as: *agent-forged; validated as a
selection-pressure mechanism; not validated as a wall-cracking
mechanism; the hall's wall class has migrated from orthogonal-starvation
to gated.* No clear may be attributed to it.
The prior text follows for the record.

*Status at forging, stated plainly: no validation run had been
performed.* The arm
is forged and reviewed; it has not been shown to help. The hall remains
unsolved, the standing prior is ~110M steps and 0 solutions across five
arms, and the validation is the pre-registered fidelity-corrected +
`--ortho` run scheduled in `docs/proposals/STRATEGY_2026-08-08.md` with
its stopping rule declared in advance. Until that run reports, this
entry may be cited only as *agent-forged, unvalidated*, and no clear of
any kind may be attributed to it.

*Addendum, 2026-08-10 (later the same day) — the K-FALSIFIER struck the
GATED vocabulary.* `docs/receipts/dispatch/k_falsifier_2026-08-10.md`
scored the four registered effort-matched SOLVED archives through the
same discriminator with the RESOLVED/PROGRESSING escape branches
bypassed: all four read GATED, at 3.9x-10.8x `CONCENTRATION_GATED_MIN`
and above three of the hall's own four class-defining reads, and a false
GATED reproduces on the unmodified shipped path (`smb_4_4_micro/lvl_1-3`
seg 2, cleared 17 minutes later on a plain retry). Per that receipt's
pre-registered FAIL branch, every `GATED` / "saturated" /
visits-as-saturation citation above is **superseded**, including the
"migrated from orthogonal-starvation to gated" sentence: the hall is
relabeled **UNRESOLVED-CONCENTRATED**. The split verdict's mechanism
finding is untouched — the arm still measurably redirected exploration
upward (2,514 vs 1,471 new cells above y-band 9) — only the wall-class
WORD borrowed from the discriminator is retracted. No clear was, or is,
attributed to this arm either way.

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

`runs/smodice_1_2/` records the 2026-08-14 closure of offline imitation
for 1-2 under a pre-registered rehabilitate-or-close protocol. The prior
offline negative was contaminated (expert-only IQ-Learn degenerates to
BC; survivorship-filtered windows; no terminal grounding), so the
corrected experiment was run same-day: terminal-grounded 50/50
success/failure dataset (40,785 transitions, `transitions.npz`), a
seven-arm ablation (`abl_A*.log`) that cured the fit ceiling — argmax
0.308 → 0.669, past the 0.60 gate, best-ever for any offline method
here — and a 50-episode honest eval of the winner
(`checkpoints/smodice_1_2/abl_A7`, `honest_eval_A7.log`): 0.0 clears,
median death at x=646 of 3,266. With fit cured and clears still zero,
offline-without-rollouts is closed on sound evidence; the online pivot
is licensed. Side finding, quotable: on 100%-expert-action datasets,
DICE-style occupancy weighting acts as an adversarial subsampler
(matched pair: chi-weights 0.359 vs unweighted 0.669 on identical
data/capacity).

`runs/online_1_2_attempt_ledger.md` records the 2026-08-14/15 online
campaign that ended the 1-2 honest zero: KL-anchored warm start from
the offline winner, monotone-invariant wavefront shaping, bottleneck
restart ladder with SPDL backward walk, self-imitation on clears, and
entrance-pinned consolidation, over eight instrument-audited attempts
(each stop/re-registration documented with its evidence in the ledger
and the controller's pre-registration comments). Banked result, policy
`checkpoints/_preserved/online_v2_FINAL_consolidated.pt`
(sha256 daa34bbe…): under the canonical honest protocol (cold entrance,
greedy, sticky-0.25, jitter-16, 100 episodes over two seeds) —
**2/100 strict clears (2.0%)** (episode_success: flagpole/castle
predicate), median max-gx 2059 of 3266, 39/100 episodes past the
x≈2674 barrier, flag height reached on both seeds. Prior state of the
world: zero honest 1-2 clears across every method ever recorded here,
and a published-literature audit that found none elsewhere.

CORRECTION (2026-08-15, same day, receipts
`docs/receipts/eval_rng_regimes_2026-08-15.md`): the campaign-probe
figure of 13/90 (14.4%) initially reported as a "protocol-variant gap"
was a PREDICATE mismatch, not an RNG effect — probe rows ran
`--sequential --level-clear`, whose clear_rate fires on level-chain
advance (max-gx ≥ 3267, reaching the flag area), while the definitive
evals bank strict episode_success. On the probes' own episodes the
predicates disagree 13:1; held to a common event, every RNG regime
agrees (Fisher p 0.6–1.0, KS p 0.83–0.96; stream-correlation,
seed-lottery, and noise-distribution explanations each rejected
mechanically). Quotable numbers, each with its predicate named:
strict honest clears 3/190 pooled (~1.6%); flag-area-reach rate under
honest noise ~10–20% by probe. The `clear_rate` key must never again
be quoted without its predicate; probe tooling now reports both.

CONSOLIDATION ROUND 2 (2026-08-15/16 overnight, ledger §consol2):
30M further entrance-pinned steps produced a transient peak the probe
pair caught at both protocols simultaneously and the runner preserved
before the post-peak descent (the pre-registered collapse kill then
ended the run). Banked policy
`checkpoints/_preserved/consol2_40pct_strict_iter01120.pt`
(sha256 413548b9…), definitive eval on fresh seeds under the canonical
protocol, strict flagpole predicate throughout: **38/100 clears
(38.0%)** — seed 7: 24/50 (the same seed that measured 0/50 on this
harness two days earlier), seed 101: 14/50 — median max-gx 2095,
40/100 past the bottleneck, chain and strict predicates converged
(reach-without-grab eliminated). Honest 1-2, full arc with receipts:
0.0 in all prior history → 2.0% (first campaign) → 38.0% (round 2).
Greedy now exceeds the policy's own measured sampled ceiling (31.7%).
Receipts: runs/consol2/campaign.jsonl, peak_eval_seed{7,101}.json,
runs/online_1_2_attempt_ledger.md.

WORLD 1-3, AND THE TRANSFER CLAIM (2026-08-16/17, ledger §1-3): the
same campaign machinery, retargeted by config alone
(`--campaign-config`; the 1-2 defaults stay golden-pinned), was applied
to a level it had never trained on. Setup took one ~20-minute window
(ladder mint from the banked solver tapes, auto-derived rungs, wavefront
dmap, BC anchor, and a competence floor calibrated from the anchor's own
measured median rather than guessed). The deterministic bottleneck gate
passed **10/10 on its first probe, 35 minutes from cold start**, on
inherited thresholds with zero retuning — the milestone that cost World
1-2 eight campaign attempts and five harness fixes across two nights.
Banked policy `checkpoints/_preserved/one_three_FINAL_consol2_iter00690.pt`
(sha256 352273f9…), definitive eval on two fresh seeds under the
canonical protocol, strict flagpole predicate: **21/100 (21.0%)**
(10/50 and 11/50), chain-advance 26/100, median max-gx 1800.5 of ~2515,
flag height reached on both seeds. The learned ledger now holds three
levels — 1-1 43%, 1-2 38%, 1-3 21%, each measured under the same cold
greedy sticky-0.25 jitter-16 protocol. Also banked from the same run,
as negatives: un-anchored adversarial hardening degrades slowly on a
level without a concentrated hazard (1138→448 over three probes) rather
than collapsing as it did on 1-2, and consolidating *from* a hardened
net trips the pre-registered value-loss kill within 1M steps. First honest
clears observed at iter 380/440 mid-campaign (preserved with hashes);
adversarial-hardening phase (kernel adversary) degraded the policy and
was rolled back — its receipts and the sharp-adversary telemetry
(entropy 0.19 vs ln2, the first non-uniform adversary this project has
trained) are banked for the hardening redesign.
