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

**FORGE-CERTIFIED — the in-core PPU scroll odometer (pseudo-addresses
`0x800–0x802`), commits `c556e40` (core, v3 savestate envelope,
certification) and `074f888` (rendered-cut scene detection, v4
envelope), 2026-08-23/24.** The need was found in the system's own
gate telemetry: Rygar, Kung Fu and Ninja Gaiden ran flat on every RAM
progress scalar while their archives churned — no game-agnostic
distance signal existed for scroll-driven games. The mechanism samples
loopy_v + fine_x per scanline at dot 256, takes the modal scanline
(HUD-split immune), folds wrap-aware dx `((Xc−Xp+256) mod 512)−256`
with the attribute-table trap handled, and keeps the accumulator
inside the core savestate struct so restores are exact by
construction. Scene detection keys cells by a rendered-cut ordinal
(masked hash AND scroll-discontinuity together). Gate: certification
is five automated checks, fail-any-quarantine — **passed 5/5**
(`runs/odometer/cert_smb_2026-08-23.json`; the cert and per-game
gate JSONs are committed in `c3d7405`, and three deep-run log tails —
`runs/odometer/{ng_deep,rygar_deep,rygar_v2}.log` — in `c1f9dbe`; the
night/scene probe logs and the probe run dirs are local-only). Verdicts from its own
gate probes: Rygar SIGNAL-SOUND (117 distinct / 470 px), Ninja Gaiden
SIGNAL-SOUND (126 / 1,384 px), Contra cross-validated against the RAM
pair (162 vs 163 distinct), Kung Fu reclassified camera-static /
agent-active (OAM churn 540) — a skill wall, not an instrument fault.
Scene keying moved the Ninja Gaiden frontier from a 2.5-hour pin at
dx 6144 to 8 scenes crossed in 12 minutes (`runs/ng_odo_scene/`).
Honest status, stated plainly: the odometer-celled Go-Explore probes
(runs/ng_odo_*, runs/rygar_odo_*; committed log tails cover only the
three deep runs, the rest are local) have produced **zero solutions** on Rygar and
Ninja Gaiden — the instrument is certified, the games remain unsolved,
and no clear is attributed to it. Any future clear it enables is an
EXHIBITION result logged on its own terms.

**FORGE-VALIDATED — generic death-detection fixes: transition-blip
debounce and wrap-aware lives decrement, commits `1610093` and
`084362c`, 2026-08-23.** Diagnosed from solver telemetry, not game
knowledge: Rygar door transitions blip the lives byte through 0 for
two steps (the same signature as the Bubble Bobble 52→51→53 key blip,
2026-08-06), and Ninja Gaiden's lives byte underflows 0→255 so every
death read as a gain and froze the frontier. The forged mechanisms are
game-agnostic: death requires ≥3 consecutive dead observations, and a
life loss is detected as `(start − cur) mod 256 ∈ 1..8`. Falsified
live on the system's own runs: Rygar frontier gx 1536 → 5360 in six
minutes post-fix; Ninja Gaiden deaths detected at the ~5838 frontier.
Receipt caveat, stated plainly: the Rygar falsifier receipt
(`runs/rygar_odo_debounce/`) exists on disk uncommitted, and the NG
underflow probe was described inline in the commit with no banked
file; the committed deep-run log tails
(`runs/odometer/{ng_deep,rygar_deep,rygar_v2}.log`, commit `c1f9dbe`)
partially cover both; the night/scene log tails on disk are likewise
uncommitted.

**FORGE-SHIPPED — mechanism-liveness preflight
(`scripts/experiment_preflight.py`), trainer sentinel enforcement, and
adjudicator fingerprint-identity refusal, commit `3bb10f6`,
2026-08-23.** Forged from the week's own void ledger
(`docs/research/PROCESS_AUDIT_2026-08-23.md`): four experiment arms —
2-1 attempt 1, the 1-4 backward ladder, Phase-3 masked v1, and the
Phase-3/options arms — each passed config verification while the
mechanism under test was dead. The named root defect class: "an assay
with no positive control." The machinery makes liveness a mandatory
positive control before an arm may run, the trainer refuses the
`actor_freeze_steps` sentinel outright, and the adjudicator refuses
arms whose policy fingerprints are identical. Ships with tests.
Honest status: exercised in production by the 2026-08-23 options
re-run, whose arms were verified live (pair_actor max|Δ| = 0.85,
fingerprints differing) before adjudication.

**FORGE-PENDING-VALIDATION — the room-graph engine (T1-T4), commit
`3601c45`, 2026-08-24.** Self-measured detection: the system's own
Zelda/Metroid probe receipts (`docs/receipts/room_fp/{zelda,metroid}.md`)
found the existing scene ordinal an unreliable room-identity signal —
noisy in Metroid (spurious bumps at clamp/seam with no room actually
left) and blind by design to Zelda cave/dungeon fades and Rygar's blank
doors — which is what motivated a settled, masked blake2b-64 hash of
the physical nametable VRAM as the identity signal instead, with the
scene ordinal demoted to classifier evidence only. Agentic authorship:
synthesized and judged from three independently authored designs
(`docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md` Part I), D0 taken as
chassis with D1's detector and D2's roadmap grafted on. Forged
machinery, game-agnostic: RoomIndex (intern/lookup/directed edges,
capped with telemetry, never crashes at cap), a pan/fade/warp
transition classifier from integrated Δodometer + Δscene during the
churn window (warp-classified settles — the Zelda-death signature —
adopt the new room identity but mint no adjacency edge and are never
routable, closing the death-edge gap without any per-game death
observable), edge exemplars for sticky-replay validation, an aliasing
audit, a room-pool router arm, and restore-lockstep invariants
(append-only global structures, per-worker state re-derived on
`_assign`, never accumulated across restores). Standard gates: default
off; flags-off byte-identity verified against pre-branch HEAD on a
16,000-step SMB solve, sha256-identical RAM/archive/traces across all
8 workers; 703+ new/updated tests, full suite 3920 green at ship time.
Gate RG-0 — the offline falsifier over banked probe fixtures
(`tests/test_rg0_roomgraph.py`), mandatory before any live run per the
BINDING cheap-premise-first sequencing rule — is **9/9 PASS**
(re-run and reconfirmed 2026-08-25: `9 passed`), including all five
design-mandated assertions: Zelda east-exit pans mint exactly one node,
a Zelda death mints a warp with zero edges, idle Zelda hashes to one
value post-mask, Metroid's two doors mint exactly two pan edges, and
Metroid's scene noise mints zero extra nodes. Honest status, precise:
RG-0 is an *offline* falsifier over fixtures — it proves the classifier
and chassis behave correctly on banked probe data, nothing more. RG-1,
the live pre-registered gate (§6 of the synthesis doc — 4 unattended
90-minute Zelda runs on 12 workers, router-lift ≥1.25×, ≥30 distinct
settled rooms, edge-replay validity on 20 sampled edges, SMB regression
control, perf ≥90% baseline), is **registered but has not been run**.
No live room-graph traversal exists yet; no RG-1 numbers exist; no
Zelda/Metroid capability claim may be made from this entry alone.

**FORGE-SHIPPED — ReDo dormant-neuron recycling
(`src/training/redo.py`), commit `3bb93ef`, DR-mandated addendum to
the v27 pre-registration, 2026-08-24, agreement-bound recalibration
commit `5995272`, 2026-08-25.** Self-measured detection: DR review of
the v27 fresh-recovery pre-registration (Decision B) held that an
unmodified fresh 4×250-iteration run risked conflating a real capacity
deficit with primacy bias / dormant-unit collapse, and mandated one
config-only change — Sokar et al.'s ReDo — before any of the spend.
The addendum pins every parameter and marks which are DR-mandated vs
locally chosen: layer-mean-normalized dormancy score (correcting the
DR's own "layer max" phrasing), tau=0.025, checked every gradient
iteration after GA warmup, `fc1`/`fc2` hidden units only (policy/value
heads excluded), Kaiming-uniform reinit on incoming weights with
zeroed outgoing columns, and exact-slice-only Adam moment clears for
touched units. Single arm (ReDo-on, 4 seeds × 250 iters) per the DR's
own outcome mapping — cumulative-recycle telemetry alone proves
whether the mechanism was live or inert, so no ReDo-off control was
run. Standard gates: default off (`reinforce.redo_enabled`), 17 new
tests covering dormancy detection, exact-slice resets, optimizer-moment
handling, and OFF-arm byte-identical training; live end-to-end check
confirmed OFF prints `[redo] disabled` and samples nothing, ON at
tau=0.025 prints the armed line plus per-iteration
dormant/recycled/agreement telemetry. ADDENDUM 2 (`5995272`) root-caused
and revised the V7 agreement-bound pilot check against LayerNorm
receipts via a tau-sweep (0.05–0.35), PASS under the corrected
condition, with the V2 seed-0 pilot clean. Honest status: the mechanism
is live and instrumented inside the currently-running v27 seed0 job
(`checkpoints/mario_1_1_v27_recovery_seed0/run.log`) — as of iteration
26/250 the telemetry reads `dormant fc1 0/64 fc2 0/32 recycled 0 cum 0
agree 1.0000`, i.e. ReDo has not yet found any dormant units to recycle
this early in training. Whether it ever fires, and whether firing
changes the outcome, is unknown until the run progresses further; this
entry certifies the mechanism shipped and is instrumented correctly,
not that it mattered.

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

WORLD 1-4 (2026-08-17, ledger §1-4): the castle level, whose clear is a
world increment rather than a flagpole latch — a different branch of the
strict predicate, and therefore a test that the pipeline crosses level
TYPES rather than level instances. Same machinery retargeted by config
alone; ~15 minutes of setup. Its deterministic rung gate went 0/10 then
10/10 one probe later (the pipeline meeting real resistance and solving
it by training, not intervention), it was already clearing 6/30 from the
entrance in phase 2, and its reverse walk exhausted its budget without
reaching the entrance — so the rate below was earned WITHOUT the
backward curriculum completing. Banked policy
`checkpoints/_preserved/one_four_MEASURED60_iter00960.pt`
(sha256 c170a54d...), definitive eval on two fresh seeds under the
canonical protocol, strict predicate: **51/100 (51.0%)** — 24/50 and
27/50 — with the chain predicate identical at 51/100 and median max-gx
2428 of 2431 (the median episode reaches the axe). The learned ledger
now holds four levels: 1-1 43%, 1-2 38%, 1-3 21%, 1-4 51%.
Receipts: runs/online_1_4/final_eval_seed{7,101}.json,
runs/online_1_4_attempt1/campaign.jsonl, runs/online_1_4/campaign.jsonl.

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

VOID AND PROCESS AUDIT (2026-08-22/23, receipts
docs/research/PHASE3_HAZARD_VETO_NEGATIVE_2026-08-22.md,
docs/research/PROCESS_AUDIT_2026-08-23.md, runs/options/verdict.json):
the Phase-3 hazard-veto training arms and the first options A/B are
**VOID** — both inherited `actor_freeze_steps: 1e12` from the campaign
base profile and trained only critics; every training-side claim from
those runs is withdrawn, with the prior text preserved in the named
docs. What survives is scoped precisely: the Phase-3 eval-time finding
(veto collapse, 31/100 → 0/100 — the policy dies standing still) was
measured on a live policy and stands. Two further retractions from the
same audit, marked as such: the hazard-model KILL verdict was false —
benchmarked on a hot machine, retracted, `needs_quiet` added — and one
"read never infer" violation was caught by the fingerprint check. The
root defect class is named for the record: configs were verified,
mechanism-aliveness never was. The fixes are the preflight/sentinel/
fingerprint FORGE entry above. Claims-integrity sweep: the banked
1-1/1-2/1-3/1-4 rates are unaffected.

OPTIONS MECHANISM, RE-RUN AND FAIL (2026-08-23, prereg
docs/proposals/OPTIONS_PREREG_2026-08-22.md, receipts
runs/options/rerun2_eval_*.json, runs/options/verdict.json,
docs/research/OPTIONS_NEGATIVE_2026-08-23.md): the first real
adjudication after the void, both arms verified live. Gate: ≥ 0.372
pooled strict honest on 1-2. **Control 8/100, treatment 0/100 — FAIL**
(relative −1.0). The mechanism failed by overcommitment: the final
treatment chose duration k=4 in 93.6% of 4,000 real 1-2 states (92.4%
sampled mass) — the advantage-accumulation pathology v22 §4.1
predicted — with training trailing 0.40 against honest 0/100. The
no-rescue clause bars retuning; salvage candidates are recorded in the
doc, none scheduled (the v23 Castlevania options dependency inherits
this FAIL pending a new registered salvage). Side finding, measured
and now binding: continued PPO collapsed the consolidated 1-2 peak
31/100 → 8/100 in 200 iters (−74% relative, KL anchor active) — all
future A/Bs carry preserve-on-peak in both arms and adjudicate
peak-vs-peak. The banked 38/100 preserved checkpoint is untouched.

SHELF DISPOSITIONS, ENGINE-RUN (2026-08-23, receipts
runs/engine/logs/shelf_joint_*.log and shelf_1_4_endpoint_*.log — on
disk, uncommitted at time of writing; doc
docs/research/PROCESS_AUDIT_2026-08-23.md): two queued questions
answered autonomously by the engine. Joint-policy transfer: pooled
honest 100-ep **1-1 32/100** (vs specialist 43) and **1-2 1/100** (vs
banked 38) — the earlier 0.52 flicker does not replicate; naive
pooling stays falsified; CLOSED. 1-4 endpoint: **51/100 pooled —
exactly the banked rate**; the 0.633 probe is retracted as winner's
curse, and the corrected-ladder re-run closes the inert-ladder caveat
as "no measurable effect either way." The consolidation endpoint HELD
on 1-4 where continued PPO collapsed 1-2 (31 → 8) — the contrast is
itself a banked observation. Still queued: SWA over 1-2 peaks;
hazard→Go-Explore weighting.

RECURRENT-BOTTLENECK A/B (2026-08-23, prereg
docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md, verdict
docs/research/RECURRENT_AB_VERDICT_2026-08-23.md, receipts
runs/gru_ab/verdict_seed{0..3}.json, runs/gru_ab/train_seed{0..3}.log,
checkpoints/mario_1_1_backward_gru_seed{0..3}/): DR v25's policy-class
prescription (sticky-0.25 makes the env a POMDP; a recurrent net
should beat the feedforward control) tested against the banked
backward-1-1 control (0.76 best-of-N honest,
checkpoints/_preserved/backward_1_1_seed3_iter140.pt), single-variable
treatment (TileRecurrentPolicyNetwork, 48,975 params, +1.7%).
**Treatment best-of-4 honest sticky 0.06** (seeds 0.00/0.06/0.01/0.01,
100 eps each) vs control 0.76 — **FAIL, mechanism untested**:
deterministic ≈ sticky ≈ 0 on every seed, a learning failure, not a
robustness failure, so v25's mechanism claim never reached test. FAIL,
not VOID: the policy class was verified armed and the hidden-reset
audit ran clean (no carry-across-restart path). Retraction, marked as
such: the registration's acceptance that BC-through-stateless-fallback
"cannot explain a between-arm difference in the PPO phase" was wrong —
it can explain where PPO starts — and is retracted in the verdict doc.
Salvage was ranked; only the hidden-reset diagnostic was run.

STICK-DETECTION PROBE AND THE V26 OVERRIDE (2026-08-23, corrected
2026-08-24; receipts runs/gru_ab/stick_probe.json,
runs/gru_ab/stick_probe_nostack.json,
runs/gru_ab/stick_probe_realpolicy.json,
docs/research/V26_ADJUDICATION_2026-08-23.md): supervised
stick-detection probe on 24k steps from the banked control
(divergent-stick base rate 0.034): held-out AUC, stateless MLP 0.77
single-frame / 0.83 four-frame stack, GRUCell 0.82 / 0.87 — **PASS by
v26's own gate**, overridden on strategy: feedforward policies already
carried AUC-0.83-grade stick access and still plateaued 0.21–0.51
honest, so detection was never the binding constraint; the expensive
recurrent-RL overhaul was declined. SUPERSEDED FIGURES, marked as
such: those AUCs predate the `28dc163` loader fix; the corrected
real-policy probe (divergent-stick base rate 0.0575 per the receipt)
reads **MLP 0.76 vs GRU 0.74** —
with real weights recurrence adds nothing and the GRU edge flips sign.
The correction strengthens the override's direction; the pre-fix
"+0.04/+0.05 GRU" figures (still uncorrected in the V26 doc and the
v25 memory note) must not be quoted without this supersession.

RECOVERY ASSAY, 1-1 AND 1-2 (2026-08-24, receipts
runs/recovery_assay/{manifest,verdict}.json,
runs/recovery_assay_1_2/verdict.json,
docs/research/RECOVERY_ASSAY_VERDICT_2026-08-24.md): the sticky wall
decomposed by adjudication. 60 honest episodes of the banked 1-1
control (the collection run itself reproduced 0.767), 1,592 post-stick
snapshots via the in-harness `--dump-stick-states` hook, last
divergent stick per non-clear episode handed to a 10-minute 8-worker
Go-Explore adjudicator — EXHIBITION machinery used as a measuring
instrument; no learned claim rides the solver's clears. INTEGRITY
RETRACTION, recorded, not silent: the first pass scored **0/14 twice**
and both of those verdicts are VOID — (a) a 3-minute solver budget
that a manual 10-minute probe disproved
(runs/recovery_assay/probe_ep15_10min/), and (b) a stdout-grep success
detector that could never fire; both fixed in
scripts/recovery_assay.py, final scoring taken from the filesystem
(solutions/ = ground truth). The stray first-pass debris at
runs/recovery_assay_bad/ is that void's artifact and is named here so
it is not a silent orphan. Result, 1-1: timeout sticks **5/5
recovered** (sanity), true death-sticks **3/9 (33%)**, sticks 1–2
steps pre-death **0/3** — the fatal window is real, the honest ceiling
is strictly < 1.0 for any policy class, and perfect recovery training
would move 1-1 from 0.767 to ~0.83–0.85. Same-day 1-2 addendum
(banked consol2 from stage_03; the collection run cleared 25/60 per
its manifest, alongside the verdict doc's 0.367 baseline reproduction;
1,337 snapshots; 16/35 death-sticks adjudicated): **3/16 recovered
(19%)**, with 11/16 sticks ≤ 4 steps pre-death → implied honest
ceiling ~0.53. The 1-2 BANKED verdict stands, now with a mechanical
explanation: at sticky-0.25 that wall is mostly physics, and the
banked ~0.38 sits near its ceiling. New routing rule, binding: run
this assay before spending training effort on any level's sticky rate.

RECOVERY DISTILLATION — THREE FAMILIES, ALL FAIL (2026-08-24, prereg
and appended verdicts docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md,
receipts runs/recovery_distill/train_history.json,
runs/recovery_distill/ckpts/, runs/recovery_distill/variant_b_train.log,
runs/gru_ab/stick_probe_realpolicy.json): the attempt to train the
measured 33% recoverable slice into the banked 1-1 control (gate
≥ 0.80 honest). VOID, retracted: the first attempt trained a random
net — `build_tile_policy_from_checkpoint` silently returned an
uninitialized network for path inputs; fixed in commit `28dc163`.
RETRACTED on the same root cause, marked as such: the "0/60
unjittered argmax-tie receipt" from that day's standalone loop — every
standalone-loop anomaly of 2026-08-24 (the 0/60 collection, the t=54
deaths, the 0.0 distill epochs) had rolled uninitialized policies;
harness and solver receipts loaded correctly and are unaffected. With
the real control loaded — base method: **FAIL-by-drift at epoch 0**,
13 Adam steps at lr 1e-4 took honest greedy 0.767 → 0.033 (sampled
0.17; epoch-0 loss 3.35 ≫ ln 6). Variant A (KL-anchored cloning,
registered before running, two-rung LR ladder): **FAIL both rungs** —
lr 1e-4 drift-stopped at epoch 0; lr 1e-5 best 0.70 < baseline 0.767
< gate 0.80; cross-entropy on solver recovery actions is
net-destructive at every tested strength. Variant B (on-policy
recovery PPO from 27 mined post-stick states, KL-anchored, actor
verified unfrozen): **FAIL** — honest 0.33 at iter 10 → 0.0 by iter
30; recovery-pool clears 0.49 → 0.30; entrance rate 0.31 → 0.06.
META-FINDING: the consolidated 48k artifact is an **isolated optimum**
— every gradient that touched it made it worse; post-hoc improvement
on this artifact is closed, and 1-1's honest number remains the
untouched control's 0.767. Receipt caveat, stated plainly: the committed train_history.json
(`8bded0d`) holds the base-method history (epoch-0 0.033, loss 3.35);
the working-tree copy was later overwritten by variant A's rung-2
history (0.70/0.60, uncommitted), variant A's rung-1 history lives
only in task logs, and variant B's train log and checkpoints were on
disk uncommitted at time of writing — named here to be banked, not
lost. Registered next, not run: v27 — recovery states in the
curriculum from the start of a fresh run, to separate consolidation
from parameter budget. The banked scoreboard is unchanged: 1-1 43%,
1-2 38%, 1-3 21%, 1-4 51%; the separately banked backward-1-1 control
stands at 0.767 with a measured ceiling of ~0.83–0.85.

V27 FRESH-RECOVERY RUN — LAUNCHED, IN PROGRESS, NOT A VERDICT
(2026-08-24/25, prereg `docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md`,
configs `configs/mario_1_1_v27_seed{0,1,2,3}.yaml`, checkpoints under
`checkpoints/mario_1_1_v27_recovery_seed0/`). The registered follow-on
above is now running, not merely planned: a 785-rung interleaved
recovery ladder was minted (5/5 self-checks), pilots V1–V3 cleared, DR
review returned Decision (B) — launch blocked pending the ReDo
dormant-neuron-recycling amendment being folded into the
pre-registration first (see the ReDo FORGE entry above for that
amendment, and ADDENDUM 2 for the LayerNorm-recalibrated agreement
bound that PASSed before launch). Design: 4 seeds × 250 iterations,
60 envs, vanilla PPO on the SMB-tiles-pos encoder, gated against the
banked backward-1-1 control (0.767). As of this writing only seed0 has
started (`checkpoints/mario_1_1_v27_recovery_seed0/run.log`, iteration
26/250, `best_fitness` climbing 3633→4196 over the last five logged
iterations, `vanilla_ppo_max_world_level` 2, ReDo armed and reporting
zero dormant units recycled so far); seeds 1–3 have not yet started.
No honest-protocol number, no gate verdict, and no comparison to the
0.767 control exists yet for this run. It is named here only so the
launch is on the record as launched; the eventual result — pass, fail,
or void — gets its own entry when the run actually finishes.
