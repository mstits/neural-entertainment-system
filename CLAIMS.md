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

ADDENDUM 1 (2026-08-26) — **"zero solutions" is struck as evidence; it
was a constant, not a measurement.** The certification above is
untouched: the instrument that passed 5/5 is the ODOMETER, a progress
observable, and every gate verdict, distinct-px count and frontier
number in this entry stands. What does not stand is the sentence
citing "zero solutions" as an honest negative. `GenericGame.is_clear`
opens with `if self.level_key(ram) > tuple(start_key)`, and
`configs/rygar.yaml` and `configs/ninja_gaiden.yaml` both ship
`level_key: []` with no `clear:` and no `finale:` block. The test is
therefore `() > ()`, which is False in Python for every RAM state that
can exist. No clear predicate was wired on either profile, so no
solution could have been banked at any budget, on any seed, for any
length of run. The number was fixed before the first step was taken.

Read the sentence instead as: *no clear predicate exists for either
game, so solved/unsolved is UNMEASURED here — not measured-and-negative.
The load-bearing evidence in this entry is frontier depth only (NG area
9 / best_score 74,783; Rygar 5,680 px).* The direction of the original
error is conservative — it understated rather than overstated what the
system can do — but "the search looked and found nothing" and "nothing
ever looked" are different claims and this ledger will not merge them.

Found by the 2026-08-26 clear-detection census
(`docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md`); the algebra
independently re-verified at the top level, and confirmed against disk
— of 300 run directories containing a `solutions/` folder, every one
belongs to SMB/Lost Levels, Castlevania or Bubble Bobble (the three
profiles with a non-empty `level_key`) plus the two already-withdrawn
detector-gate false positives. Guarded going forward by
`scripts/clear_reachability.py`, which refuses a profile declaring
clear machinery that cannot fire and makes every solve run print
`[clear] NO REACHABLE CLEAR PREDICATE` at launch when its solution
count is a constant.

ADDENDUM 1a (2026-08-26, ledger audit) — **the enumeration in the
paragraph above is wrong twice, in the direction that would destroy
real negatives. The ruling on Rygar and Ninja Gaiden is untouched.**
Two corrections, both re-derived at HEAD:

*The census undercounts by ~8×.* `find runs -type d -name solutions`
returns **507 directories, 297 of them non-empty** — not 38, the
number the first sweep re-ran and reported, and not the 300 quoted
here either. The 38 came from scanning `runs/*/solutions` at depth 1.
Re-classifying all 297 finds no strays, so the family-level conclusion
survives; it survives by luck rather than by method, and the method is
what this ledger is for.

*"The three profiles with a non-empty `level_key`" implies `level_key`
is the only route to a bankable clear. It is not, and the count is
not three.* `scripts/clear_reachability.py --all` reports
**NONE=37, REACHABLE=8** across the 45 shipped solve profiles — eight
clear-capable profiles by **four** routes: `level_key`
(`bubble_bobble`, `castlevania`, `kid_icarus`), `finale`
(`excitebike`), `byte_change` (`tetris_b`), `score_jump`
(`ducktales`), and `confluence` (`contra`, `contra_blank`).
`configs/excitebike.yaml` says so in its own body (lines 214-222): a
finale profile "must NOT be assumed dead just because its `level_key`
is empty", and `runs/excitebike/excitebike_bootstrap/solutions/sol_000`
is a real banked clear on an empty-`level_key` profile. Quoting
"three profiles" or "four clear-capable profiles" forward would write
off `contra`, `contra_blank`, `ducktales` and `kid_icarus`'s
`solutions: 0` as compile-time constants when the repo's own guard says
their predicates are arithmetically fireable — the exact mirror of the
Rygar error this ADDENDUM exists to correct. Rygar and Ninja Gaiden
have none of the four routes, so nothing above them moves.

ADDENDUM 2 (2026-08-26, ledger audit) — **two of the four per-game
gate verdicts quoted in this entry are WITHDRAWN. The certification is
still untouched, and Rygar still stands.** The verdict sentence reads
"Rygar SIGNAL-SOUND (117 distinct / 470 px), Ninja Gaiden SIGNAL-SOUND
(126 / 1,384 px), Contra cross-validated against the RAM pair (162 vs
163 distinct), Kung Fu reclassified camera-static / agent-active (OAM
churn 540) — a skill wall, not an instrument fault." Every clause was
re-tested by re-running the identical gate at HEAD, same profile, same
`--steps 1200 --odometer`, same start state.

- **Rygar — STANDS.** `PASS — SIGNAL SOUND — still advancing`, 116
  distinct / range 0..467 over 138 live steps, against the banked
  117 / 470. The post-death tail contributed essentially nothing here,
  which is exactly why the other two do not survive the same test.
- **Ninja Gaiden — WITHDRAWN.** At HEAD: `FAIL — SIGNAL UNUSABLE`,
  with `[INSTRUMENT] death byte reads 0 at the start state — a
  decrement underflows and no death can be detected`, and **937 of the
  requested 1,200 steps dropped as game-over tail**. The banked
  receipt recorded `lives_at_start: 255` — an already-underflowed byte
  — and assessed "126 distinct / 1,384 px" across those 937 dead
  steps. Three *later* receipts already on disk at the time this
  entry was last quoted say the same thing and were never folded in:
  `runs/odometer/gate_ninja_gaiden_2026-08-25{,_reverify,_1200}.json`,
  all three `"passed": false, "verdict": "SIGNAL UNUSABLE"`. What
  survives is narrower and worth keeping: over the 263 live steps
  before the tail the odometer reads **192 distinct / range 0..748**,
  a live axis with *better* resolution than the banked number claimed.
  The AXIS is real; the SIGNAL-SOUND verdict and both quoted numbers
  are not.
- **Contra — WITHDRAWN, both halves.** At HEAD: `FAIL — SIGNAL
  UNUSABLE`, 19 distinct in 69 live steps, range 0..70, with
  `[INSTRUMENT] only 19 distinct values in 69 steps (< 32) — too
  coarse to be a search gradient` and **1,131 of 1,200 steps dropped**.
  So ~94% of the banked "162 distinct / 635 px" was game-over
  animation. The second half is worse: the "163" it was
  *cross-validated against* **has no receipt anywhere in the tree**.
  `find runs -iname "*contra*" -name "*.json"` returns exactly two
  files — `runs/ram_verify/contra.json`, which is not a signal gate
  and contains no 163, and the odometer receipt being corrected. A
  cross-validation whose second number cannot be produced is not a
  cross-validation, and it was the strongest-sounding evidence in the
  sentence.
- **Kung Fu — WITHDRAWN (already), with its corroboration corrected.**
  The "not an instrument fault" clause was struck in the first sweep
  against commit `bfb515b`, which deleted the vacuous
  `passed = not instrument_findings` camera-static override and names
  kungfu first among 13 profiles / 40 vacuous passes carrying the
  `distinct=1 / min=0 / max=0` signature.
  `runs/odometer/gate_kungfu_left_2026-08-23.json` carries that
  signature verbatim. The OAM churn of 540 is a real positive and
  stands. The audit adds one correction to the *supporting* clause:
  the "~719,500 steps with `current_floor` never leaving 0" figure has
  no receipt. What the banked telemetry actually shows is a frontier
  pinned at `cells: 32, max_gx_in_max_area: 244` across **both**
  Kung Fu runs — `runs/kungfu/kungfu_bootstrap` (514,239 steps) and
  `runs/live_show/kungfu/lvl_` (1,206,456 steps), ~1.72M steps
  together. That is a stronger corroboration than the one quoted, and
  it is the one on disk; but on a profile shipping `level_key: []`
  no clear could have been banked either, so "skill wall" remains an
  inference and not a measurement.

**The generalisable defect.** The first sweep validated these three
rows against ONE vacuity signature — the camera-static override it had
just watched get fixed — and never applied the *second* one, which its
own companion analysis had already documented:
`runs/onboard_wave6_d5_sweep_v3.json` finds **28 of 45 profiles
contaminated by a post-death tail**. A gate that keeps counting after
the agent is dead reports the game-over animation as progress. Checking
a receipt against the last defect found, rather than against every
defect known, is the same habit one level up.

ADDENDUM 3 (2026-08-26, Rygar R1 campaign) — **the one surviving
frontier number in this entry, "Rygar 5,680 px", is WEAKENED: the
odometer over-counts on this profile, and the gate verdict is
untouched.** ADDENDUM 1 correctly re-scoped this entry's load-bearing
evidence down to frontier depth alone. That depth is now measured to be
partly an instrument artifact.

`nes_core/src/ppu.rs::odo_fold_frame` drops the scroll anchor
(`odo_have_prev = false; return`) on any frame rendering fewer than 120
lines. For a respawn that is correct and the source says so — it
freezes rather than rewinds. But Rygar's room transitions are
**blank-type**, so the same branch fires at every door, and the
odometer **cannot see a camera reset across a transition**: each round
trip banks the forward scroll and discards the return. On the deepest
HEAD tape, 1,634 px of a 6,242 px headline is exactly this — 27 round
trips through one door, measured as 27 segments banking `dx = 0`
alternating with 27 banking +53..+64 px, with 45 of 54 post-door
segments ending on a screen already seen. The positive control that
makes this a measurement: the same test on the uninterrupted pre-door
stretch returns 51 distinct screens out of 51 pieces.

Consequences, stated narrowly. **The gate verdict in ADDENDUM 2 stands
unchanged** — 116 distinct / 0..467 over 138 live steps is a forward
hold that never reaches a door, so no ratchet can be in it. What is
weakened is the *frontier* figure: 5,680 px was measured on a
**different emulator binary** (`07f121f81fbb7d7b`; HEAD is
`54366c20d32f71cc` and refuses to resume it) and has never been
ratchet-audited. Cite it as a soft, non-HEAD-comparable baseline only.
The HEAD-verified, artifact-free frontier is **4,608 px**. Full
write-up and the replayable tape:
`docs/research/RYGAR_CAMPAIGN_2026-08-26.md`,
`docs/receipts/rygar/r1_tape_gx6242.json`.

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
Metroid's scene noise mints zero extra nodes.
[ADDENDUM RG-0a, 2026-08-26 ledger audit: **four of the five passed as
written; the fifth did not and the test says so.**
`tests/test_rg0_roomgraph.py`'s own docstring records that RG-0.5's
literal wording — "mints zero extra nodes" — is *unsatisfiable on the
measured surface*, and `test_rg0_5` asserts a reframed and weaker
claim instead: `index.n_rooms() < len(np.unique(scene))` plus lap-2
dedupe. That reframing is honest and documented where it happened; the
sentence above quotes the design wording as though it passed as
written, which is the ledger's most common failure mode — faithfully
copying a receipt's headline and dropping the receipt's own caveat.
The other four are positive-form assertions over real captured streams
(`n_rooms()==2` with one E-pan edge at d_odo 256; a warp with
`edges == []` and `warp_count == 1`; exactly one masked hash against
two unmasked; exactly two E-pan edges at d_odo 254) — positive-capable,
not vacuous, and they stand. Gate RG-0 as a whole stands; the
five-for-five phrasing does not.] Honest status, precise:
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

**FORGE-SHIPPED — engine_driver retry/abandon fix (the
unattended-operation P0) plus six further scheduler/checkpoint
hardening fixes, commit `5ce9304`, 2026-08-25.** Self-measured
detection: a second independent audit pass over surfaces the first
pass hadn't covered (`scripts/engine_driver.py`, the pyo3 bindings,
`checkpoint_manager.py`, `checkpointing.py`, `go_explore.py`) found
live evidence already sitting in the system's own state file
(`runs/engine/state.json`) that the mechanism was firing. The defect:
`_reap_one` wrote every non-recurring action's id into
`state["completed"]` on its first termination regardless of outcome —
success, verified failure, or an ambiguous crash alike — and because
`offer()` permanently excludes anything already in `completed`, this
made `MAX_ATTEMPTS_PER_ACTION`'s retry/abandon logic unreachable dead
code for every single-shot action in the system (`honest_eval`,
`config_*`, `mint_*`, `select_*` construct their `Action()` with no
done_marker): any transient crash — OOM, a disk hiccup, a
momentarily-locked ROM — permanently and silently froze that level at
its current stage with zero diagnostic trace, exactly the failure
class the "walk away for three weeks" mandate cannot tolerate. Fixed:
`completed` is now written only on a verified success; a crash or
verified failure stays retry-eligible up to the attempt cap, at which
point `tick()`'s existing (previously dead) abandon branch becomes the
real terminal-status writer. Six further generic fixes bundled in the
same pass, named precisely: a timeout-killed heavy job never arming
the `needs_quiet` settle clock (a SIGKILLed 14h campaign could be
immediately followed by a benchmark on a hot machine); three
shelf-disposition sites silently dropping skip reasons instead of
journaling them; `Pool::load_worker_state` panicking uncatchably on a
malformed 4-byte NCST-only blob and never clearing a worker's dead
flag on a successful restore (a worker panicked once could stay
short-circuited to zero-frame/done=true forever, even after a valid
reload); `save_winner`'s two-file write (`best.pt` + `best.json`) not
being one transaction, now reconciled against `best.pt`'s own embedded
metric; `save_iter`'s non-per-writer tmp path allowing a torn write
under a race between processes sharing a checkpoint_dir;
`GoExploreArchive.load()` unpickling a cells dict with zero `cell_fn`
key-schema validation. Gate: 29 new regression tests
(`tests/test_checkpoint_manager_tmp_path_race.py`,
`tests/test_engine_driver.py`, `tests/test_go_explore.py`,
`tests/test_winner_checkpoints.py`); `make test-fast`: 3942 passed, 0
failed; `cargo test --lib`: 634 passed, 0 failed. Touched
`nes_core/src/pool.rs`, `scripts/engine_driver.py`,
`src/training/checkpoint_manager.py`, `checkpointing.py`,
`go_explore.py`. Honest status, stated plainly: every fixed path is
exercised directly by its own regression test, but the fix has not yet
been exercised by a real multi-week unattended run hitting an actual
transient crash in production — no live crash-then-successful-retry
event has been observed post-fix. The system is measurably safer for
the "walk away for weeks" goal; it is not yet proven safe by a live
long-duration trial, and no unattended-operation uptime claim may be
made from this entry alone.

**FORGE-VALIDATED — signed progress-axis odometer fix
(`progress.axis` sign prefix, x/-x/y/-y), commit `9bb5122`,
2026-08-25.** Self-measured detection: League onboarding wave 3's own
solver-smoke telemetry, not a walkthrough — 1942's archive froze at 7
cells, and the deepest-cell/root diagnostics
(`diag_1942_{deepest,root}.log`) measured directly that every held
action drove the odometer's raw y-reading strictly negative from a
verified-live start state, exposing that `_xram`'s clamp-to-≥0
assumption (forward always increases the odometer — true for every
prior profile, all horizontal right-scrollers) is silently false for
1942, whose vertically-scrolling engine counts the scroll register
DOWN during forward flight. The forged fix is game-agnostic:
`progress.axis` now accepts an optional leading sign (`x`, `-x`, `y`,
`-y`) that flips the raw reading before the existing clamp is applied,
implemented additively via `getattr` so every profile that doesn't
specify a sign is unaffected. Gate/falsifier: the identical wall-clock
smoke budget went from a 7-cell freeze
(`runs/onboard_wave3/smoke_1942/`, v1) to 106 cells with an open
frontier (`smoke_1942_v2/`, v2) — receipts
`gate_1942_{right,up,A}.json`, `discover_1942_up.json`,
`diag_1942_lifecycle_noop.log`, `summary_1942.json`; the fix's
addition to the shared `go_explore_solve.py` path was verified not to
disturb any other profile via the full regression run, `tests/
test_room_fp.py` + go_explore_solve suites, 609 passed. Classified
SOUND_ADVANCING (frontier open, not frozen), one of ten games
processed this wave alongside League onboarding wave 1's twelve
(`docs/research/LEAGUE_ONBOARDING_WAVE1_2026-08-24.md`,
`LEAGUE_ONBOARDING_WAVE3_2026-08-25.md`). Honest status, stated
plainly: the fix demonstrably unfroze 1942's search — the game itself
remains unsolved (no clear, no `level_key` win predicate exercised) —
and the fix has been live-validated on exactly one game; whether other
games in the roster share the same latent negative-axis defect is
untested (Galaga was flagged in the same wave as a candidate carrying
the identical gap, not yet hit). No clear or League-inclusion claim
beyond SOUND_ADVANCING is made from this entry.

**FORGE — item/key semantic discovery engine, commit `cc0bc9e`,
2026-08-25.** Answers the original "a key can open a specific door"
capability directive with a purity-clean pipeline: `RoomIndex.cap_hist`
(an additive graft on the room-graph engine, no version bump), and
`scripts/discover_item_bits.py` running three stages — idle-prefiltered
candidate scanning, K-of-N cross-rollout confirmation with permanent
reject-on-revert, and a correlational "lead" (rarity-confound exposed
in the report) followed by behavioral verification via matched
real acquire/skip trajectory replay from a shared `save_worker_state`
snapshot, scored against a rejected-pool control battery. Validated
against IS-1a, a Zelda negative control (rupees/keys never move in any
of the 12 real archived RG-1 sequences replayed): the engine correctly
returned **FAIL, not VOID** — 51 raw false "confirmed" flags,
root-caused (not explained away) to Zelda's death→CONTINUE-menu RAM
rewrite, truncating via the already-claimed `lives` byte (no new
address, purity intact) to 13, further diagnosed as a deterministic
engine-init artifact (identical elapsed-step offsets across 8
independently-diverging lineages). Two named follow-ups, neither
requiring new game knowledge or new Rust: death/lives-based rollout
truncation ahead of `scan_rollout`; a cross-lineage step-offset
consistency check. Honest status: the engine correctly rejects a true
negative on its first live test; it has not yet been run against a
true positive (IS-1b, a real Zelda pickup, needs a new minted start
state — not run this session). No door/key semantic claim is made from
this entry — only that the machinery distinguishes signal from noise
on the one case tried.

ADDENDUM IS-1 (2026-08-26, ledger audit) — **this entry describes a
three-stage engine and validates a one-and-a-half-stage run. Stages 3a
and 3b are UNEXERCISED, and 3a is unexercised for the same reason
`is_clear` was: a compile-time constant.** Three scope corrections,
all read off the entry's own receipt
(`runs/item_semantics/is1a/is1a_stage12_3a_receipt.json`):

1. **Stage 3a returned a structural zero, not a null result.** The
   receipt records `total_leads: 0` with `leads_per_bit` all zero on
   all four RG-1 sequences, *and* `cap_hist_key_present_pregraft_run:
   false`. `discover_item_bits.py:616` reads
   `cap_hist = e.get("cap_hist") or {}`; on archives minted before the
   `EdgeStat.cap_hist` graft every edge yields `{}`, so
   `exposure_bit0 == exposure_bit1 == 0` across all 2,065 / 1,160 /
   2,086 / 1,399 edges and **no lead is generable at any budget, on any
   seed**. Zero leads was fixed before the first edge was scored.
2. **Stage 3b therefore never ran at all.** The module docstring calls
   `verify_behavioral` "the ONLY thing in this engine allowed to turn a
   Stage-3a lead into a claim", and it is the entire structural answer
   to the rarity confound. It cannot execute on zero leads. The
   sentence "running three stages" is true of the code and false of the
   run.
3. **"The engine correctly rejects a true negative" attributes the
   rejection to the wrong agent.** The engine emitted 51 raw
   "confirmed" flags, 13 after truncation — it did not reject. The
   rejection was performed afterwards by human root-causing (the
   death→CONTINUE menu rewrite; identical elapsed-step offsets 7/53/69
   across 8 independently-diverging lineages). The VERDICT process was
   correct and is worth keeping; the engine failed its own
   zero-confirmed criterion, which is precisely what the receipt says
   and precisely what this entry drops. The receipt scopes itself
   correctly — "as currently specified (§9 Task 2 scope), Stage 1-2
   does not yet achieve zero false confirmations" — and the ledger
   widened "Stage 1-2" to "the engine".

One disclosure this entry owes and does not make: the negative
control's ground truth — "rupees/keys never move" — is read from
`0x066D`/`0x066E`, **quarantined disassembly-sourced addresses**. The
receipt handles this correctly, flagging it `doctrine_crosscheck_
reporting_only` and never feeding it to the scanner, so purity is
fully intact and no result here is contaminated. But a reader of this
ledger cannot see that from the entry, and a control whose ground
truth comes from quarantined bytes must say so.

Honest form: **Stages 1-2 were exercised and FAILED their own
criterion. Stages 3a and 3b are UNEXERCISED — 3a by a structural
constant, 3b by consequence.** IS-1b remains unrun. No claim about
what this engine can distinguish survives beyond Stages 1-2.

**FORGE — fight-gate progress mechanism, commit `07f367d` design /
`af94b88` implementation, 2026-08-25.** A second progress source
alongside the PPU scroll odometer, for the CAMERA_STATIC_AGENT_ACTIVE
game class the odometer cannot serve (fixed-screen fight games — no
scroll gradient exists by definition). Purity-clean by construction:
`scripts/discover_observables.py --fight-gate` runs randomized
attack-mash/approach-retreat probes and a decrement-consensus self/foe
HP discriminator with no RAM map or game knowledge consulted, feeding
a cumulative-damage integral into `go_explore_solve.py` via
`progress: {source: fight_gate, foe_hp, foe_hp_start}` — the same
pseudo-RAM-extension pattern the odometer already uses, so every
existing consumer (cells, glitch filter, macros, progress_cap) is
unmodified. **Validated on Punch-Out: SUCCESS.** Blind discovery
(`runs/fight_gate/discover_punchout.json`) nominated addr 920
(0x0398), start 96 (0x60), attack_agree 5/5, defense_agree 0/5 — the
same byte the profile's internet-sourced `ram_mapping` label already
used, independently re-derived from hardware observables alone. Wired
live into `configs/punchout.yaml` (commit `2e2696a`) and re-verified
against the committed file, not just the design-time scratch config: a
fresh run reproduces the smoke's `max_gx_in_max_area=81` exactly.
League classification updated CAMERA_STATIC_AGENT_ACTIVE →
SOUND_ADVANCING-eligible. Two honest qualifications, named and not
fixed: the ranking heuristic does not uniquely surface 0x0398 as its
#1 pick (it is #3 of 5 candidates that pass the decrement-consensus
gates on this probe budget — the correct one was confirmed here by
cross-referencing the independently-verified `ram_mapping`, not by
discovery rank alone); `find_round_gate` returned VOID
(`insufficient_probe`, no bout boundary crossed in 24,000 probe steps)
so `round:` is omitted from the profile rather than guessed. No
League-inclusion claim beyond SOUND_ADVANCING-eligible is made.

*Extension, 2026-08-25 (later the same day) — the other three targets
tried, none validate.* Kung Fu, Ice Climber, and Galaga (the remaining
CAMERA_STATIC_AGENT_ACTIVE games) each ran the same blind discovery.
**Kung Fu**: the ranking's #1 pick (0x00B1) cleared FH1+FH2 mechanically
but was flagged suspicious by the discovery pass itself (zero cross-rep
variance across 5 independently-seeded reps — the "elapsed-step timer"
signature the ranking docstring names); a follow-up isolated-input
probe (pure NOOP, 1200 steps) CONFIRMED it tracks the game's autonomous
enemy spawn/despawn cycle, not combat damage. The candidate wiring was
committed to `configs/kungfu.yaml` as a fully receipted negative finding
but left COMMENTED OUT — the pre-existing (also non-functional, not
newly regressed) odometer-era value stays the active `progress:` —
mirroring the disclose-and-disable pattern already established for
`configs/mario_1_2_phase3_masked.yaml`: a confirmed-implausible source
does not stay live in a profile `go_explore_solve.py` can pick up by
path alone. **Ice Climber**: VOID — every candidate showed byte-
identical net deltas across all 5 reps (worse than Kung Fu's borderline
case), cross-checked against an existing wave-3 receipt that already
diagnosed this exact ROM/state/forward combination as a scripted,
input-ignoring intro window this run never escaped; genre mismatch also
noted (enemies die in one hit, no on-screen HP resource exists).
**Galaga**: VOID — all 6 candidates carried a disqualifying flag (zero
variance, or net damage exceeding the candidate's own starting value,
or a leak under the defense-only probe); wave-shooter genre mismatch
noted (no per-invader HP bar in the actual game). Neither profile was
touched. All three: `find_round_gate` returned a genuine
no_round_signal behaviour finding (not VOID) after crossing 10-46 bout
boundaries — no purity-clean round byte was found in any of them.
Honest reading: the mechanism generalizes as a DISCOVERY tool (it
correctly told all three apart from Punch-Out's true positive, using
only signal it computed itself) but has so far only produced one
usable progress source out of four attempts — 1-for-4, not evidence of
broad applicability to the CAMERA_STATIC_AGENT_ACTIVE class as a whole.

ADDENDUM FG-1 (2026-08-26, ledger audit) — **the Punch-Out result is
untouched. Three framing claims around it are not.**

*The class name was minted by a gate that could not fail.*
`CAMERA_STATIC_AGENT_ACTIVE` is the verdict the pre-`bfb515b`
`progress_signal_gate.py` printed when its inline `rx==0/ry==0` branch
deleted the "too coarse" instrument finding and forced
`passed = not instrument_findings` because OAM churn showed the agent
moving. Every verdict carrying that label is a vacuous pass on a
zero-range axis; under the shipped gate those profiles read `SIGNAL
UNUSABLE — camera static`. So "the game class the odometer cannot
serve" and "the remaining three CAMERA_STATIC_AGENT_ACTIVE games" both
inherit a label from an instrument demonstrated incapable of returning
the negative. The Punch-Out finding does not depend on the label —
blind discovery nominated addr 920 with attack_agree 5/5 /
defense_agree 0/5, and the committed profile reproduces
`max_gx_in_max_area=81` exactly. That stands. Delete the class, keep
the result.

*"It correctly told all three apart from Punch-Out's true positive" is
not what happened.* The entry's own text, plus `CLEAR_GAP_CLOSURE`
rows 13/16, records that **Ice Climber's probe never escaped a
scripted, input-ignoring intro window** and **Galaga's root state sits
inside a non-interactive attract loop** (`$0091` byte-identical under
`hold_A` vs `hold_right`; "the fix is a re-mint"). On both profiles the
instrument could not have returned a positive whether or not an HP
signal exists. Those are **VOIDs of the start state, not
discriminations by the mechanism**. Kung Fu is the one real
discrimination — its #1 pick was falsified by an isolated-input NOOP
probe, a positive counter-measurement. Corrected accounting:
**1 true positive, 1 true rejection, 2 unmeasured** — the 1-for-4
headline is right, the reasoning under it was not.

*"No purity-clean round byte was found in any of them" over-reaches.*
`round_gate_from_drives` is unusually well built for this question — it
separates `insufficient_probe` (VOID) from `no_round_signal`
(behaviour) in so many words, scans the full RAM window
(`LIVES_SCAN_TOP = RAM_SIZE`, no address blind spot), and the
precondition was genuinely met. But **the function has never returned
kind `round_gate` on any real ROM.** Its only demonstrated positive is
`tests/test_fight_gate.py::test_round_gate_nominates_the_monotone_
round_byte`, a synthetic trace built to its own shape — the exact
caveat this ledger raises about green detector suites. On the one game
with true round structure (Punch-Out) it returned
`insufficient_probe`. Its three-clause filter (`ptp<=2` within EVERY
bout, up at EVERY boundary, never back to start) is severe enough to
reject on any noisy real ROM. Honest form: *no byte satisfied the
three-clause filter under this probe* — not *no purity-clean round
byte exists in these games*.

**FORGE-SHIPPED — the six clear signals reach the vote, commit
`ee1324a`, 2026-08-26.** `entity_wipe`, `room_fp_transition`,
`input_lock`, `lock_release_novelty`, `oam_quiesce` and `scene_cut`
were built, tested and receipted earlier the same day and reached **no
production path at all**: the live vote was `tally + coord >=
min_signals` and the offline harness weighted only
audio+tally+lock+coord. Six working signals, none able to change any
verdict, while the roster closed 4 CONFIRMED / 41 VOID / 0 FAIL for
want of an instrument that could return a positive. That is the
week's pattern in its purest form — a shelf of mechanisms and a vote
that could not see them. Now wired into both votes, eligibility-gated
(an unarmed signal is `DEAD` in the quorum table with the key named,
so a profile that arms nothing is byte-identical), with Rule 3 —
one slot casts one vote — moved from documentation into the
arithmetic of `slot_ceiling()` and `_fold()`.

**Result, both halves, and the second half is not a success.**

- **Bubble Bobble FIRES.** Round 99-0 → 99-1, two witnessed tapes, was
  `n_valid: 2, hit_rate: 0.0` — its progress observable spans ONE unit
  against the ≥300 backward drop `coord` requires, so the only
  transition-evidence signal on the roster was arithmetically dead.
  With `scene_cut` armed at its own measured gate: **2/2 at +4 and +18
  frames**, independently reproduced at HEAD during this audit
  (detected 2060 vs true 2056; 2499 vs 2481). Live per-action vote
  fires 1 action after the clear and is silent before it.
- **Tetris-B DOES NOT.** It moved from *blind* to *measured-and-late*:
  `scene_cut` detects the real transition, but the screen blanks 151
  frames after the quota byte hits 0, against a 120-frame tolerance
  calibrated on SMB's fast flagpole cut. Reproduced at HEAD:
  true 8657, detected 8819, delta 162. No stride or window change
  reaches it, and widening the tolerance would manufacture the hit, so
  the row stays a real `NO_CLEAR`. Its in-tolerance signal
  (`oam_quiesce` at +28) was refused 60 times by Rule 5 — correctly:
  "the sprites went away" is what a death looks like. **Gate (b) is
  1 of 2.**

**Disclosure the first version of this entry did not make, added by
the same audit that found it (ADDENDUM CD-1).** Bubble Bobble's
headline `total_false_positive_crossings: 0` was scored on the SAME
two tapes the `scene_cut` gate was calibrated on
(`bubble_bobble_scene_cut_null_2026-08-26.json`: `blank_min: 1` set as
"the smallest integer strictly above the measured null" over those
tapes' pre-clear play). **A gate placed one unit above the null
measured on tape T cannot fire below threshold on T** — so that zero
was arithmetic, and it was reported in the same field, with the same
words, an out-of-sample zero would have used. This is the file's own
signature defect wearing its politest face.

The zero is not thrown away — an in-sample zero is a real consistency
check, and a non-zero there would be a genuine defect — but it is not
out-of-sample specificity, and the receipt now says which it is.
`clear_detect.calibration_provenance()` computes the overlap
mechanically from the banked calibration receipts and writes it beside
the count on **every** receipt, empty or not, so its absence is
visible rather than ambiguous. Bubble Bobble reads `n_in_sample: 2`;
SMB's five-trace zero reads `n_in_sample: 0` — a genuinely
out-of-sample zero, now labelled as one. Anti-vacuity:
`test_the_disclosure_is_not_a_constant` drives three inputs to three
different answers, and mutating the overlap test to `if True` reddens
it by name.

Worth stating because it bounds how much the disclosure is
conceding: Tetris-B is ALSO in-sample (`n_in_sample: 1`) and still
records **one false positive**. The calibration bound constrains the
calibrated signal on the calibrated tape; it is not a blanket
guarantee of zero, and the instrument demonstrably still fires. So the
Bubble Bobble zero is a weaker result than it read as, and a real one.

**The out-of-sample evidence for Bubble Bobble exists and is worth
citing precisely because the receipt's zero does not carry it:** the
live per-action test replays a DIFFERENT tape
(`runs/bubble_bobble/chain_day2f`, round 69, distinct from both
calibration tapes) and the detector is silent across all 14 evaluations
before the clear, then fires 1 action after it. Note 14, not 298 —
298 is the number of driven observations, the detector evaluated 14
times in that span, and the test asserts the smaller true number.

SMB control: **byte-identical, 5/5, 0 false positives**, re-run at
HEAD with zero field differences. Two scope corrections to the
wire-up's own commit message, recorded because a control's shape is
part of its result: (1) "every shipped profile is byte-identical —
none of them arms anything" is false as written; the same commit arms
three signals each on `bubble_bobble` and `tetris_b`. Blast radius was
re-checked and is nil — both use `clear` mode `byte_change`, and all
nine `mode: confluence` profiles produce identical verdicts and
ceilings either side of the commit. (2) "field-for-field identical
either side of the change" overstates it twice: three fields differ in
every row (`armed_signals`, `shelf_stats`,
`n_required_class_vetoes`, all in the comparison's ignore set), and
the BEFORE receipt already carries those keys as null — so what the
control pins is ARMED-vs-UNARMED under one code path, not
new-code-vs-old-code. That is still the control that matters, and the
narrower statement is the true one. Both corrections are recorded in
`tests/test_clear_signal_wireup.py`, where the claim is made.

Suite at ship: 5349 passed, 30 skipped, 3 xfailed, 1 known-
environmental failure; Rust 659 passed. Seven mutations were run
against the wiring, each reddening a named test.

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

**BLOCKED, not yet quarantined — `ZeldaReward` and `MetroidReward`
(`nes_core/src/rewards.rs`), surfaced 2026-08-25 during an external
research audit's review, not by a Learned-ledger run hitting it.**
Both structs declare their own compiled RAM address constants
independent of any config file (e.g. `ZeldaReward::RAM_GANON_DEFEATED
= 0x0672`, commented "aldonunez disassembly + empirically verified");
`MetroidReward`'s equivalent gap was already noted in-line at
`configs/metroid.yaml` on 2026-08-10 ("MetroidReward in rewards.rs
uses HARDCODED address constants; this ram_mapping block is
documentation only") but never carried through to this ledger. Neither
struct is named in the Tier-2 freeze above (which covers only the five
SMB `LEVEL_*` ladders) or in the Quarantine section. **Checked before
writing this entry: neither `ZeldaReward` nor `MetroidReward` appears
anywhere else in this file** — no Learned-ledger clear rate, honest
eval, or capability claim has ever been built on either, so this is a
live gap in the provenance system, not a retraction of an existing
claim. **Rule, effective now: neither struct may be used to produce a
Learned-ledger claim until this entry is resolved** — either by
re-deriving each address from this project's own discovery tools
(`discover_observables.py`, the item-semantics engine) and replacing
the disassembly-sourced constant, or by a deliberate, dated decision to
extend the Tier-2 freeze to name them explicitly, the same way the SMB
ladders were. Follow-up, not done here: the same pattern (a bespoke
bare-metal reward struct in `rewards.rs`) should be audited across
every other per-game reward for the same gap — this entry covers only
the two structs a source citation directly named, not a full sweep.
`configs/zelda.yaml`'s `ram_mapping:` block carries the same addresses
a second time as YAML and is quarantined separately, below.

### BREACHED 2026-08-26 — the quarantine was defeated by two independent paths

**Full record: `docs/research/PURITY_BREACH_2026-08-26.md`. Read it before
citing any observatory receipt.** Both paths are closed; the summary here
is the ledger entry, not the analysis.

**Path 1 — inheritance by display name.** `nes_core/src/rewards.rs`
dispatched rewards on `name.contains("zelda")`. `configs/legend_of_zelda.yaml`
— 31 lines, no `ram_mapping`, no address of any kind, no reward weights,
and passing the outside-provenance lint that `configs/zelda.yaml` trips —
therefore ran `ZeldaReward` **including its quarantined win predicate**
(`RAM_GANON_DEFEATED = 0x0672`, provenance "aldonunez disassembly + Data
Crystal", status `UNVERIFIED_EXTERNAL`). Measured on the pre-fix binary:
flipping that one byte on an otherwise-zeroed 2 KB buffer took the profile
from `(-0.001, False)` to `(19999.999, True)` with `episode_success()`
True. Two more profiles were inheriting the same way and were not in the
original finding: `configs/zelda_roomfp.yaml` (same 0x0672 jackpot,
reproduced) and `configs/metroid_roomfp.yaml` (silently acquired the
ledger-BLOCKED `MetroidReward`). The dispatch existed from the initial
commit `55e5333` (2026-04-27) to `c89a816` (2026-08-26) — **121 days**;
it was a breach *of the quarantine* only for the 1 day the quarantine had
existed, and the three inheriting profiles for 2 days. Closed
structurally: `build_reward` takes `reward_id` and no longer receives the
display name at all, so the defect is not expressible without re-plumbing
a parameter.

**Path 2 — the exclusion-set inversion, and the worse of the two.**
`configs/zelda_gui_tuned.yaml` (a fourth Zelda profile, picker-visible,
pinned bootable) carried a live 13-entry int-valued `ram_mapping` under the
comment "Win chain (disassembly + emulator-verified)", six values of which
were the exact quarantined addresses. `scripts/observatory.py` parsed
those ints and folded them into its **pre-probe exclusion set**, the gate
on candidate-predicate generation. An external RAM map was therefore
deciding which bytes this project's own discovery instrument was permitted
to nominate, and the bytes it removed were precisely the ones under
quarantine — whose documented exit condition is rediscovery *by that
instrument*. Not a weakened quarantine: the quarantine inverted, with the
contamination record used as the blindfold. Live from `c9d3d95`
(2026-07-31, when the fold was added; the addresses had been in that config
since `d0f315e`, 2026-07-12) to 2026-08-26 — **26 days**. Closed: only
solve-block coordinate bytes are excluded now, `ram_mapping` is annotation
tagged `"known": true/false`, and the caller receives a membership
*predicate* rather than a set, so `excluded |= mapped` cannot be re-added
as a one-liner.

**Path 3 — the test that could not catch either.**
`tests/test_zelda_purity_quarantine.py` hardcoded one of the five Zelda
configs, and nothing anywhere asserted that a profile declaring no reward
gets the generic one. Closed: the guard is parametrised over the same
predicate the dispatch used, plus a tree-wide sweep sourced from the
quarantine blocks themselves.

**Effect on banked results.** 13 of the 14 `observatory_v2.json` receipts
in `runs/` were produced with a folded exclusion set; their own logs record
it. They split two ways:

- **Externally steered — negatives VOID, positives stand.**
  `runs/megaman/mm2_bootstrap2`, `runs/gradius/gradius_bootstrap`,
  `runs/kid_icarus/kid_icarus_bootstrap`, `runs/kungfu/kungfu_bootstrap`
  excluded addresses that originate outside this project. Whatever those
  passes *found* was found by probing and stands. Any claim of the form
  "the instrument looked and there is nothing there" does not: for the
  excluded bytes it did not look. **Gradius's "stage byte NOT FOUND" is
  withdrawn** — that pass was pre-blinded to six candidate bytes. The same
  withdrawal applies to any "exhaustively searched" phrasing on the other
  three. Re-running under the fixed instrument is the only thing that
  restores a negative.
- **Self-derived — results stand.** `contra`, `bubble_bobble`,
  `castlevania`, `double_dragon`, `ducktales`, `excitebike`,
  `ghosts_n_goblins`, `kirby`, `metroid` folded only their own
  `[VERIFIED: ...]`-receipted addresses. Narrower than intended, but no
  outside knowledge entered.
  [ADDENDUM P-1, 2026-08-26 ledger audit: **positives stand; negatives
  from these nine carry the same caveat the external four got, and this
  bullet drops it.** `PURITY_BREACH_2026-08-26.md` says "These results
  stand. They are narrower than they should have been — a handful of
  bytes that could have been re-nominated were not." The ledger kept
  "results stand" and lost the narrowness, and — unlike the bullet
  directly above it — did not repeat the positives-stand /
  negatives-void split. Those nine passes were still pre-blinded, just
  by the project's own bytes rather than someone else's. Purity is
  fully intact and no positive is touched. But an "the instrument
  looked and found nothing" negative from any of the nine is narrowed
  by construction and must be quoted with that caveat.]

**No CONFIRMED clear is affected.** `bubble_bobble`, `castlevania`,
`excitebike` and `tetris_b` — the only four profiles that can witness their
own clear — carry no external RAM map; their sole external-provenance
mentions are negative citations.
[ADDENDUM P-2, 2026-08-26 ledger audit: **the purity conclusion
stands; the parenthetical enumeration it rests on does not.** Read
"the only four profiles that can witness their own clear" as **"the
four profiles whose CONFIRMED clears this breach could have
touched"** — `scripts/clear_reachability.py --all` reports **eight**
clear-capable profiles by four routes — the four above plus
`kid_icarus` (`level_key` 0x0130), `ducktales` (`score_jump` ≥5000),
`contra` and `contra_blank` (`confluence`); see ADDENDUM 1a on the
odometer entry. The purity finding is UNAFFECTED and was re-verified
directly against the configs: `configs/kid_icarus.yaml` (lines 180-183,
"NO external RAM maps / disassembly") and `configs/ducktales.yaml`
(lines 124-126, "no external maps") both declare self-derivation with a
receipt path, and `configs/contra.yaml` carries no quarantine marker.
What had to be corrected is the enumeration, because quoting "the only four" forward
licenses treating four other profiles' `solutions: 0` as compile-time
constants when they are arithmetically real. One scoping note this
sentence should also make explicit: it is an **`is_clear`-HOOK**
statement and is entirely separate from the confluence clear detector,
which scored 0 on the witnessed Bubble Bobble and Tetris-B clears.
Neither profile's CONFIRMED status runs through that detector, so the
detector's blindness never touched this row.]
**No Learned-ledger claim is retracted**:
the 2026-08-25 check that neither struct appears anywhere else in this file
was re-run and still holds. `zelda_roomfp`'s 443,419-cell / zero-solution
archive was already VOID for want of an admissible target signal; the
breach voids it a second time, independently, because that profile was
running the quarantined win predicate throughout.

**Standing rule, now mechanised.** A quarantined address may re-enter a
profile only as an independent rediscovery, declared in a machine-readable
`rediscovered_addresses:` entry naming role, method and a receipt path that
exists in the tree (`runs/` is gitignored and does not count). Enforced for
both `ram_mapping` and `solve:` by
`tests/test_purity_quarantine_sweep.py`, ROM-scoped via each quarantine
block's `applies_to_rom:` key so cross-game address collisions are not
reported as contamination. The three Zelda solve coordinates that needed
this were re-derived live on 2026-08-26 by held-direction differential with
no map consulted — receipt
`docs/receipts/rediscovery/zelda_coordinates_2026-08-26.json`.

**Still live, explicitly.** `ZeldaReward` and `MetroidReward` are built
almost entirely from quarantined addresses and remain reachable by profiles
that *explicitly* declare `reward_id: zelda` / `metroid`
(`configs/zelda.yaml`, `configs/zelda_gui_tuned.yaml`). Flipping 0x0672
still returns `episode_success() == True` for those. The rule above stands
unchanged: neither struct may produce a Learned-ledger claim. Making the
structs themselves inert is a real behaviour change on live training
profiles and belongs in its own dated change.

**Related class, closed the same day.** Three further name-substring→
hardcoded-address sites were found in Python and fixed:
`src/diagnostics/worker_debug.py` and `scripts/diagnose.py` both read the
quarantined 0x0070/0x0084 pair under `"zelda" in <name>`, and
`src/audio/ram_music.py` carried a whole substring→address table (deleted).
`tests/test_no_new_name_dispatch.py` holds a **shrink-only** inventory of
the eight display-name dispatch sites that remain in `src/` and `scripts/`
— it may lose entries, never gain one, and a stale entry is a hard failure.

## LEDGER AUDIT 2026-08-26 — the whole file, read against one question

Full write-up: `docs/research/LEDGER_AUDIT_2026-08-26.md`. Summary
here so no reader of this file can miss it.

**61 load-bearing claims were read and questioned. 39 warranted a
ruling: 21 STAND, 15 WEAKENED, 3 WITHDRAWN.** The remaining 22 were
checked and needed nothing recorded. Every withdrawal was made against
a receipt or a re-run at HEAD, never against a judgement; **not one
positive result in this file was withdrawn.** The four CONFIRMED
clears, all four honest-protocol rates, and every purity finding are
intact.

The three withdrawals: Kung Fu's "not an instrument fault" (against
commit `bfb515b`, which deleted the vacuous camera-static override);
Ninja Gaiden's odometer SIGNAL-SOUND verdict; Contra's odometer
SIGNAL-SOUND verdict and its unreceipted "163" cross-validation. All
three are annotated in place — ADDENDUM 2 on the odometer entry.

**The question the audit applied to every entry:** *what would this
have reported if the mechanism were absent?* Where nothing answered
it, the claim is UNMEASURED, not measured-and-negative.

**The pattern in what was corrected is a single habit, and it is
cheap to fix.** In every weakened case the underlying receipt was MORE
honest than the ledger entry summarising it. The IS-1a receipt says
"Stage 1-2"; the entry said "the engine". `test_rg0_roomgraph.py` says
RG-0.5's literal wording is unsatisfiable and asserts something
weaker; the entry said all five passed as written. `PURITY_BREACH`
says the self-derived nine are "narrower than they should have been";
the entry said "results stand". F0 says the per-seed v28 numbers are
low; the entry quoted them as capability. The failure mode is not
dishonesty and it is not sloppiness in the work — it is dropping the
receipt's own scoping caveat when quoting its headline.

**BINDING RULE, extended (2026-08-26).** The 2026-08-23 process audit
named the defect class — *an assay with no positive control* — and
built a preflight that enforces it forward on **trainer arms only**.
Nobody ever ran the rule BACKWARD over this ledger, and every finding
of the 2026-08-26 campaign was reachable from that sentence three days
earlier. The rule is therefore extended, and it is the rule this
section exists to install:

> An assay with no positive control applies to **discovery
> instruments, gates, detectors and clear predicates**, not only to
> trainer arms. **Every entry in this file that cites an absence must
> name the positive that same instrument returned on that same
> profile — or say VOID.** "The instrument looked and found nothing"
> and "nothing ever looked" are different claims and this ledger will
> not merge them.

Instruments that already meet the rule, and are the templates:
`scripts/clear_reachability.py` (poses the question verbatim in its
docstring; mutation-tested), `tests/test_purity_quarantine_sweep.py`
and `tests/test_no_new_name_dispatch.py` (both refuse to certify an
empty scan), `scripts/odometer_cert.py` (three of five checks are
positive-form), the ReDo forced-recycle sweep (the ledger's best
entry: a null banked *with* its positive control), and
`scripts/anti_vacuity_scan.py` + `tests/test_anti_vacuity_gates.py`
(commit `1580ebf`).

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

V27 FRESH-RECOVERY VERDICT: FAIL — BEST-OF-4 0.530 (2026-08-25, prereg
`docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md` VERDICT section,
commit `5b3363d`, receipts `runs/v27_fresh_recovery/gate/*.json`, 16
files — 4 seeds × 2 checkpoint classes × 2 eval seeds × 50 episodes).
Full honest-protocol scoring (cold entrance, greedy, sticky 0.25,
jitter ±16, max-steps 1500), two checkpoints per seed
(peak entrance-trailing-rate via `winners/best.pt`, and the final
iter-240 checkpoint) per the registration's fixed selection rule: seed
0 pooled 0.040 (final 0.020), seed 1 pooled 0.290 (final 0.020), seed 2
pooled 0.530 (final 0.000), seed 3 pooled 0.170 (final 0.010).
**Best-of-4 = 0.530** against a PASS bar of ≥0.80 and a FAIL bar of
≤0.767 — not close on either bound; the best individual seed (0.53)
does not even clear the control's own 0.767. Per the registration's
own verdict language: from-the-start curriculum inclusion adds nothing
at 48k parameters — the parameter-budget hypothesis takes the floor.
This closes the curriculum-shape question the isolated-optimum finding
left open (RECOVERY DISTILLATION entry above): post-hoc gradient work
on the consolidated 48k artifact and from-scratch delivery of the
identical mined recovery states (this run) have now both FAILed at the
same parameter budget — **the sticky-wall research line is CLOSED on
curriculum shape.** v28 (`docs/proposals/V28_CAPACITY_2026-08-25.md`,
pre-registered before this verdict landed) is the standing next step,
not a contingency; it tests the capacity hypothesis directly
(`tile_hidden_dim` 64→96, 48,135→72,039 params, identical treatment
and gate — its own entry follows below).

SECONDARY FINDING, LEARNED ledger / architecture-training-dynamics
class (2026-08-25, same receipts as above) — peak instability
reproduces at full strength within a single from-scratch run, not only
post-hoc: every seed's final iter-240 checkpoint scored catastrophically
below its own peak checkpoint — seed 0: 0.02 vs 0.04, seed 1: 0.02 vs
0.29, seed 2: 0.00 vs 0.53, seed 3: 0.01 vs 0.17. This is the same
continued-PPO-collapses-a-peak pattern measured two days earlier on
post-hoc training (−74% over 200 iters, OPTIONS MECHANISM entry
above), but this is the first time it has been observed WITHIN a
single run that was never post-hoc fine-tuned — the degradation is
intrinsic to training, not an artifact of resuming an
already-consolidated checkpoint. Preserve-on-peak (`winners/best.pt`)
was load-bearing, not a convenience: without it this experiment's
reported number would have been ~0.01, not 0.53. Conclusion, stated at
the strength the data supports: peak instability looks like a property
of this architecture/recipe class broadly; preserve-on-peak stays
mandatory in every future arm of this line.

SECONDARY FINDING, process/methodology class — binding, not a LEARNED,
EXHIBITION, or FORGE result (2026-08-25, same receipts as above) —
training telemetry massively overestimates the honest rate but still
ranks seeds correctly: `entrance_trailing_rate` (0.87/0.93/1.00/0.97
across seeds 0–3) predicted almost nothing about the absolute honest
rate (0.04/0.29/0.53/0.17) — a 2–25× overestimate — while still
correctly identifying seed 2 as best and seed 0 as worst. This is the
starkest confirmation yet, at the largest measured gap in this
project's history, of why the honest evaluation protocol (declared
above) must remain the sole scoring authority: training telemetry is
usable for within-run seed/checkpoint selection and unusable as a
proxy for any gate number, cold or otherwise.

ADDENDUM V-2 (2026-08-26, ledger audit) — **the binding process rule
STANDS and is strengthened. Both quantitative halves of this finding
are WEAKENED, and one clause of the v27 verdict above it is
withdrawn.** See ADDENDUM V-1 for the underlying defect.

1. **"A 2-25× overestimate" is inflated.** The comparison is not
   independent: the honest denominators were measured at checkpoints
   chosen by the very proxy under test, and those picks are
   systematically low by +0.08..+0.21. The gap is real and large; the
   multiplier is not a measured quantity.
2. **"Still ranks seeds correctly" is WITHDRAWN as stated.** On the two
   v27 seeds later re-scored, seed 1 moves 0.290 → 0.500 against seed
   2's uncorrected 0.530 — "seed 2 is best" is inside the noise.
   `CHECKPOINT_SELECTION_DEFECT_2026-08-26.md` makes the general point
   directly: *a metric that preserves rank across seeds need not
   preserve rank across checkpoints WITHIN a seed, and this is direct
   evidence it does not.* The sentence "usable for within-run
   seed/checkpoint selection" in the paragraph above is exactly the
   usage now falsified, and is withdrawn; the rest of the rule stands.
3. **The v27 verdict's "not close on either bound" is WITHDRAWN for the
   FAIL bound.** No corrected v27 ladder exists — F0 covered v28 only.
   The two v27 spot-checks both moved UP (seed 0 0.040 → 0.120, seed 1
   0.290 → 0.500). With the same systematic shortfall a corrected v27
   best-of-4 is unknown and could plausibly approach 0.7 against a
   0.767 FAIL bar. The **FAIL verdict most likely survives**; the
   margin claim does not. Downstream, "the sticky-wall research line is
   CLOSED on curriculum shape" is a strong closure resting on two FAILs
   whose per-seed numbers are now known to be systematically low, and
   should be re-stated as provisional until a registered v27 re-scoring
   grid exists.

The rule itself — **training telemetry is unusable as a gate number**
— is untouched and, if anything, better supported: the proxy is now
known to fail in a second, independent way.

V28 CAPACITY VERDICT: FAIL — BEST-OF-4 0.670 (2026-08-25, prereg
`docs/proposals/V28_CAPACITY_2026-08-25.md` VERDICT section, configs
`configs/mario_1_1_v28_seed{0,1,2,3}.yaml`, launch commit `e4e35a7`,
receipts `runs/v28_capacity/gate/*.json`, 16 files — 4 seeds × 2
checkpoint classes × 2 eval seeds × 50 episodes). Full honest-protocol
scoring, identical to the v27 gate in every respect (cold entrance from
`runs/live_show/smb_4_4_micro/entrance_start.state`, greedy, sticky
0.25, jitter ±16, 50 eps × eval seeds {0,1} = 100 pooled episodes per
checkpoint, max-steps 1500, `--eval-rng` per-episode), two checkpoints
per seed (peak entrance-trailing-rate via `winners/best.pt`, and the
final iter-240 checkpoint) per the registration's fixed
artifact-selection rule: seed 0 pooled 0.450 (final 0.000), seed 1
pooled 0.230 (final 0.050), seed 2 pooled 0.370 (final 0.000), seed 3
pooled 0.670 (final 0.000). **Best-of-4 = 0.670** against a PASS bar of
≥0.80 and a FAIL bar of ≤0.767 — FAIL, and not adjacent to the
MARGINAL band at either end; no seed's better checkpoint reached the
banked control's own 0.767. Not VOID: V1 (machine diff shows only
`name`, `tile_hidden_dim: 96`, and the explicitly-written unchanged
`tile_trunk_dim: 32` differ from the v27 configs, all four seeds), V2
(`params=72039 (hidden=96, trunk/gru=32, features=712, actions=6)`
printed verbatim), V3/V4 (per-width forced-recycle sweep ran; `[redo]
ENABLED tau=0.025` in all four training logs, `[redo] disabled` in
none), V5 (`[backward] ENABLED: 785 states ... from
checkpoints/backward_states/1-1-v27`, all four), V6 (every seed walked
the ladder to tau=0 well before iter 150), and V7 (all 16 gate receipts
record the registered start-state and flags) were each re-verified
against the artifacts at adjudication time, not carried over from the
launch commit's assertion. Registered mechanism read #2 (assay re-run)
was correctly not triggered — the registration scopes it to "if PASS or
MARGINAL."

The registered reading rule selects on the mechanism reads, not the
gate number. The falsifiable prediction, locked before any v28 compute
ran, requires that in ≥3 of 4 paired seeds BOTH read #1 (recovery-band
trailing rung-clear rate) AND read #3 (ladder-walk depth /
iters-to-entrance) move in the improving direction vs. that same seed's
v27 run. Both halves are computable — nothing in the conjunction was
lost to instrumentation — and both were recomputed from the raw
`[backward]` telemetry of all eight training logs independently of the
agents that first produced them, reproducing to the digit: seed 0
(#1 0.0286→0.0640, #3 29→25 iters), seed 1 (#1 0.0250→0.0606, #3 25→22),
seed 3 (#1 0.0271→0.0388, #3 24→23) all improve on both; seed 2 (#1
0.0222→0.0204, #3 24→29) regresses on both. **3 of 4 seeds satisfy the
conjunction**, selecting the reading rule's `FAIL/MARGINAL` ×
`improving, ≥3/4 seeds` row, whose registered verdict text is applied as
written: **capacity is a real, partial lever — 72k under-shot but the
direction supports "still capacity-constrained"; a further width step
(the pre-costed Candidate 2, `hidden_dim` 64→80 + `trunk_dim` 32→40) is
named as the natural v29 data point, "not a foregone conclusion."**
Stated plainly and at the strength the data supports: 50% more
parameters did not clear the bar and did not buy nothing either. Per
the registration's own capacity-only limitation, this FAIL is evidence
against capacity *under this exact recipe* at 72k — lr, rollout size,
and the 250-iter budget were held fixed to preserve the single-variable
claim, so a 1.50× weight space may simply be under-trained relative to
its capacity; it does not close the door on a wider net trained
differently.

Three caveats are recorded alongside that count rather than buried,
because the first would flip the selected row: (a) read #3's
entrance-rate-trajectory sub-metric — which the prediction does not
name — is ceiling-saturated in all 8 runs (peak 0.867–1.000, ±0.09 SE
off a 30-episode window) and reads flat in 3 of 4 pairs; requiring it
to move too drops the count from 3/4 to 1/4. The adjudication follows
the prediction's own wording (which names iters-to-entrance /
ladder-walk depth, 3/4 either way), and the lower-variance members of
that family agree with the named metric in every seed. (b) Reads #1 and
#3 are not independent votes — both are downstream of how readily the
policy clears rungs, so the conjunction is closer to one signal
measured twice than to two confirmations. (c) Seed 2's read #1 delta
(−0.0019 on N≈360–540 attempts) is inside the noise; "flat" is the
honest description, though flat is not improving so the count is 3/4
either way.

Seed-paired gate comparison (48k v27 → 72k v28, same protocol, same
selection rule, same ladder, same ReDo treatment, width the only
variable): seed 0 0.040→0.450 (+0.410), seed 1 0.290→0.230 (−0.060),
seed 2 0.530→0.370 (−0.160), seed 3 0.170→0.670 (+0.500); best-of-4
0.530→0.670 (+0.140). Both halves of that are stated because neither
is allowed to swallow the other: the headline moved a real +0.14 under
an identical 100-episode protocol, AND the per-seed deltas are MIXED —
two large gains, two moderate regressions. Best-of-4 takes the max over
seeds and is therefore structurally flattered by a 2-up/2-down field
that raises the max; it is the registered number (the same selection
freedom the banked 0.76 had) and it stands, but the honest description
of what 24k extra parameters bought is **higher across-seed variance
with the upside larger than the downside, not a uniform lift.** The two
seeds that gained at the gate are two of the three that improved on
both mechanism reads, and the seed that lost most at the gate is the
one that regressed on both — seed-level coherence between gate and
reads, which supports the dose-response story even under a failed gate.

ADDENDUM V-1 (2026-08-26, ledger audit) — **the FAIL headline survives,
but by a rescue this ledger never recorded, and the per-seed numbers
above are LOWER BOUNDS rather than capability.** Two documents landed
the same day as these entries and neither is cited anywhere in this
file. Both apply directly to the numbers above.

*The selector is a demonstrated-defective instrument.*
`docs/research/CHECKPOINT_SELECTION_DEFECT_2026-08-26.md` shows that
`winners/best.pt`, selected by argmax over `entrance_trailing_rate`,
**under-selects in 4 of 4 runs tested — by 20-40 iterations and
+0.08..+0.21 at full honest protocol.** The cause is mechanical:
`entrance_trailing_rate` saturates at 0.867-1.000, where a 30-episode
window carries SE ≈ 0.09, so an argmax over ~25 draws is close to
arbitrary. Every per-seed number in this entry was scored at a
checkpoint that metric chose.

*The headline was independently re-derived and survives exactly.*
`docs/research/F0_CORRECTED_PEAK_LADDER_2026-08-26.md` re-scored **192
evaluations across every 10-iteration checkpoint of all four v28 runs**
and made the estimate selection-unbiased by split-sample — select on
one eval seed, score on the held-out other. It lands on **0.670**, the
recorded number, to the digit. **FAIL stands, and for a good reason.**
The defect and the rescue both belong in this ledger; until now the
headline rested on a proxy known to be broken with its defence
undocumented.

*The per-seed field is markedly tighter than recorded.* Corrected
ladder: **0.640 / 0.590 / 0.580 / 0.720** (max-over-24) and
**0.640 / 0.500 / 0.580 / 0.670** (split-sample, unbiased), against the
recorded 0.450 / 0.230 / 0.370 / 0.670. Under-selection hit 3 of 4
seeds by +0.19..+0.36 and hit the BEST seed (3) least. Consequences,
stated at the strength the correction supports:

- The phrase **"higher across-seed variance with the upside larger than
  the downside"** is WITHDRAWN as a property of 72k weights. Most of
  that variance is selector noise; the corrected field is tight.
- The **seed-paired delta table** (+0.410 / −0.060 / −0.160 / +0.500)
  and the **"seed-level coherence between gate and reads"** argument
  both ride on seed 2 being the gate loser at 0.370. Corrected, it is
  0.580. Both are WEAKENED; neither is re-derived here, because F0's
  own rule applies — *a re-scoring grid must be registered before
  re-scoring, or the correction inflates as badly as the original
  deflated.* A corrected seed-paired table requires a registered v27
  ladder that does not exist.
- **Mechanism reads #1 and #3 are UNAFFECTED.** Both were recomputed
  from raw `[backward]` telemetry and are independent of checkpoint
  selection. The dose-response finding stands as written.
- The direction is conservative everywhere it touches a banked
  artifact: a rate measured on a fixed sha256-pinned checkpoint can
  only be *understated* by a bad selector, never inflated.

SECONDARY FINDING, LEARNED ledger / architecture-training-dynamics
class (2026-08-25, same receipts as above) — peak instability
reproduces a THIRD time and more starkly than ever, and capacity does
not mitigate it: every v28 seed's final iter-240 checkpoint scored at
or near zero against a substantially non-zero peak — 0.450→0.000,
0.230→0.050, 0.370→0.000, 0.670→0.000. Three of four collapsed to
**exactly 0.000 over 100 pooled episodes** — an erasure, not a
degradation. Prior reproductions: post-hoc continued PPO (−74% over 200
iters) and v27's from-scratch 48k runs (0.04→0.02, 0.29→0.02,
0.53→0.00, 0.17→0.01). v28 is the starkest in absolute terms because
its peaks were the highest — the wider net fell further, not less far.
**Preserve-on-peak (`winners/best.pt`) is again the only reason this
experiment has a number at all**: scored on final checkpoints alone,
v28's best-of-4 would be **0.050**, not 0.670.
[ADDENDUM V-1b, 2026-08-26 ledger audit: this finding is **STRENGTHENED,
not weakened, by ADDENDUM V-1.** The iter-240 checkpoints carry no
selection at all, so the selector defect cannot reach them; and because
the true peaks are HIGHER than recorded, the measured collapse is
LARGER than stated, not smaller. One phrase to tighten: per
`CHECKPOINT_SELECTION_DEFECT_2026-08-26.md`, preserve-on-peak is
load-bearing and correct as a **FLOOR**, not as an optimum — it is why
there is a number, but it is not the best number available.]
Demonstrated now across
two parameter budgets, two delivery mechanisms, and three campaigns;
preserve-on-peak stays mandatory in every arm of this line and any
future registration scoring only a final checkpoint is mis-specified.

SECONDARY FINDING, mechanism-attribution class (2026-08-25, receipts
`runs/v28_capacity/mechanism_reads/read4_dormancy_recycle.{py,json}`,
parsed from all 8 real training logs) — ReDo was armed, correct, and
completely inert at BOTH widths. Across all 2,000 per-iteration checks
(8 runs × 250 iters, `check_every_iters=1`, zero skipped iterations,
zero `[redo] disabled` lines) the logged dormant-unit count was `0/64`
and `0/32` on every v27 iteration and `0/96` and `0/32` on every v28
iteration; cumulative recycles 0 for all eight runs; per-layer dormant
fraction max and mean 0.0 everywhere. The mechanism is not broken — the
forced-recycle sweeps fire correctly at both widths (v28: 0 recycles at
tau 0.15/0.20, 9 at 0.25, 6 then 13 at 0.30, finite `max_dlogit`
throughout) — the registered tau=0.025 simply sits far below anything
real 1-1 rollouts produce on a SiLU + pre-activation-LayerNorm net at
either width. Consequences stated at supportable strength: (i) v27's
FAIL text quotes the DR's claim that "ReDo mathematically guarantees
that all 48k parameters were active and non-dormant" — ReDo guaranteed
nothing of the kind, it never fired; what holds is the weaker statement
AMENDMENT 1's B2/B5 registered in advance for exactly this case, that
the dormancy confound never materialized and plasticity was
*measured-intact by the dormancy statistic*, not actively maintained.
The capacity verdict stands and the registration earns credit for
writing that caveat before the telemetry returned. (ii) ReDo cannot
explain any per-seed gate delta — it was equally inert in both arms.
(iii) The per-width isolated-event boundary (~6 units at tau 0.30 for
the 96-unit net vs. ~2 for v27's 64-unit net) is itself a real finding:
dormancy dynamics ARE width-sensitive, vindicating the decision to
re-measure rather than scale v27's "15" proportionally — even though
the soft-VOID trigger was never approached, no recycle event having
occurred.

SECONDARY FINDING, process/instrumentation class — binding, not a
LEARNED, EXHIBITION, or FORGE result (2026-08-25, surfaced while
adjudicating the above) — four receipt/telemetry gaps, all fixable
before the next registration in this line. (1) The `[backward]`
telemetry line is NOT the number the winner selector reads:
`winners/best.json` for v27 seed 0 records
`entrance_trailing_rate=0.8667 @ iter 60` while that iteration's
printed line says `trailing 16/30=0.53`, because a second
`bwd_sched.record(...)` force-completion pass in `trainer.py` runs
after the line prints and before the winner-save block reads the
window. The log-derived peak is a LOWER BOUND on the selection metric;
both arms share the code path so seed-paired directions are unaffected,
but any future read treating the printed line as the selection metric
will be wrong. (2) No continuous dormancy-score trace exists — `redo.py`
logs only post-threshold counts, so "ReDo never fired" is established
but "how close did dormancy get to 0.025" is unanswerable from the
receipts, which is exactly the quantity that would say whether tau was
mis-calibrated slightly or by an order of magnitude. (3) The V1 receipt
named in the registration's receipts layout,
`runs/v28_capacity/config_diff.json`, was never written — V1 was
genuinely checked pre-launch (asserted in commit `e4e35a7`) and its
substance is re-verified above against configs git confirms unchanged
since, but a registered receipt path that is never written is a gap
even when the check itself was honest. (4) `warp_rate` has no field in
`eval_game.py`'s JSON output, though the PASS branch requires
`warp_rate 0.0`; it never bound here (gate FAILed; every clear shows
`max_gx` at the 3161 flagpole with `max_byte_seen: 0`), but a future
PASS in this family could not be scored against its own registered
condition as written.

## RYGAR R1 CAMPAIGN 2026-08-26 — EXHIBITION, verdict FAIL

Full write-up: `docs/research/RYGAR_CAMPAIGN_2026-08-26.md`. Tape and
its always-on guard: `docs/receipts/rygar/r1_tape_gx6242.json`,
`tests/test_rygar_r1_tape.py`.

**Ledger: EXHIBITION, without exception.** Every number below is
Go-Explore search output. No policy was trained for this game and no
honest-protocol evaluation (cold entrance, greedy, sticky p=0.25,
jitter ±16, 50 eps × 2 seeds = 100 pooled) was run. Nothing in this
campaign may be described with "the AI learned", "the AI plays", or
"the AI beat". Not one of the fourteen items overclaimed on this axis:
all fourteen returned `beat_prior_best: false` or had it corrected in
adjudication.

**Verdict: R1 FAIL** against a bar fixed before compute and not moved
afterwards. All four conditions were required.

| # | Condition | Bar | Measured | Verdict |
|---|---|---|---|---|
| 1 | DEPTH | ≥ 9,000 odometer x from power-on | 6,242 raw; **4,608 artifact-free** | **FAIL** |
| 2 | CLEAR PREDICATE | wired signal passing an anti-vacuity triple | none wired; R1-06 DECLINED | **FAIL** |
| 3 | REPRODUCIBILITY | 3/3 replay, ±16 px, alive at terminal | 6,242 / 6,242 / 6,242 | PASS |
| 4 | LIVENESS | no lives-0 run ≥ 3 observations | longest 2; histogram `{2: 55}` | PASS |

FAIL, not VOID: a real, live, deterministic tape was produced and
tested against every condition.

**The wall moved, and it moved before this campaign.** Verified
first-visit depth went **1,536 px → 4,608 px (3.0×)**, and 9.9× the
467 px scripted forward-hold probe. The lever was the ≥3-observation
death-blip debounce, not compute: real deaths pin the lives byte at 0
for 5,721–5,862 observations while transition blips are exactly 2 — a
~2,900× separation. Pre-debounce the search sat pinned at exactly
1,536 (the first door) for 45 minutes; post-debounce it reached 5,360
in six minutes. **This campaign itself did not move the frontier**: it
moved the raw instrument 5,893 → 6,242 and every pixel of that +349 is
ratchet. That is a plateau and is recorded as one.

**`solutions: 0` remains a compile-time constant for this profile** and
is evidence of nothing. `configs/rygar.yaml` ships `level_key: []` with
no `clear:` or `finale:`, so `is_clear` is `() > ()` — re-verified
False over 2,000 random RAM states, with every `solutions/` directory
empty. R1-06 DECLINED to arm a predicate it could not show would fail,
which is the correct call and the opposite of a vacuous gate.

**The campaign's real finding is an instrument defect** — the odometer
ratchet, recorded as ADDENDUM 3 on the odometer entry above.

**A fifth vacuous gate, found and struck.** R1-08 discriminated
loop-from-advance by asking whether `gx` ever *decreases* across a
revisit to the same room ordinal. Because the odometer re-anchors
instead of integrating at every transition, `gx` **cannot** decrease
across a revisit: the check returns REFUTED whether or not the loop
exists. Its own headline evidence — "room 219: 4672, 4736, 4791, 4855,
4910, 4974, 5029, 5089, 5148" — is the ratchet ladder read as
progress, and that conclusion propagated into R1-14's
`beat_prior_best`. Belongs in the anti-vacuity census. The working
non-vacuous replacement is in the receipts: cluster rendered frames at
fixed odometer milestones, with uninterrupted-segment milestones as the
control that proves the detector can say "different".

**The bar itself was denominated in a farmable quantity** — recorded so
it is not repeated, and NOT used to move a number retroactively. 27
door cycles bought 1,621 px (~60 px/cycle), so reaching 9,000 from
6,242 needs ~46 more cycles and no new ground. Any successor bar must
denominate depth in **first-visit territory**.

**Rooms reached: at least 3 visually distinct areas, counted by eye
from our own rendered frames — no instrument can count them.**
`odometer_scene` reads 0 cuts across a tape that provably crosses 55
blackout transitions, because `odo_fold_frame`'s blank branch returns
before the scene-cut test. Every `rooms_reached: 1` reported during the
campaign was an inference from an instrument that cannot fire.

**Negative results banked (the bulk of the campaign, all replay-
audited).** R1-01 budget sweep VOID (no arm reached its own cap, so the
variable never engaged). R1-03 cell resolution REFUTED on its
pre-registered metric — 10× the cells, `gx` flat in 19 of 24 runs.
R1-04 velocity-signed cells (the SMB 4-4 recipe) REFUTED on 2 seeds
*with mechanism*: the domination score is a pure function of `gx` for
this profile and never reads the spliced slot. R1-05 room-fingerprint
DECLINED with numbers. R1-10 `ODO_ALT`→`y` REFUTED by construction.
R1-11 self-refuted honestly: selection is not ignoring the frontier
(75.6% of selections land in the frontier band), it is blind to
remaining budget. R1-02 found the one real lever — hold-macros, +82%
px/step on a length-matched window — and it remains untested inside the
corridor that matters.

**R1-04's `--vsign-key` patch was NOT landed.** It is refuted by its
own A/B and it broke 72 tests in the shared checkout by reading
`self.vsign_key` on the progress-line path (a `SimpleNamespace` args
stub has no such attribute). The diff is preserved, tracked, at
`docs/receipts/rygar/vsign_key_REFUTED.patch` — not under gitignored
`runs/`, for the same reason the tape is not.

**Why Rygar and not the other three**, re-measured at HEAD:

- **Rygar — PASS, SIGNAL SOUND — still advancing.** 116 distinct /
  range 0..467 over 138 live steps. The "138 live steps" is a property
  of the gate's scripted forward hold, not of the game: the solver's
  own lineages run 3,865–6,018 actions in one continuous life with zero
  terminal deaths. Cite 138 only as the probe's survival.
- **Contra — SIGNAL UNUSABLE, but NOT fairly excluded.** 20 distinct in
  69 live steps, 1,131 of 1,200 dropped. **Open defect:** the gate
  computes its resolution finding on the window *after* truncation, and
  a 69-sample window cannot demonstrate a 32-distinct threshold. That
  verdict measures how fast the probe died, not the signal's
  resolution. **Contra's honest verdict is INCONCLUSIVE**, pending a
  probe that survives long enough to assess. Rygar's PASS is unaffected.
- **Kung Fu — SIGNAL UNUSABLE on both axes.** RAM byte `$0094`: 91
  distinct, 0..240, no paired high byte. Odometer: **1 distinct, range
  0..0 over 1,200 steps with OAM churn 628/1199** — the camera is
  provably static while the agent is provably moving. Fixed-screen
  fight class, same as Punch-Out; needs a fight-gate observable, not a
  scalar position repair.
- **Zelda — SIGNAL UNUSABLE and purity-blocked.** 25 distinct in 1,200
  steps, unpaired wrap, flat late; and its win chain came from a
  disassembly and is quarantined regardless of the gate.

**Kung Fu high-byte side quest — NEGATIVE, and a proof rather than a
failed search.** A paired high byte would have made Kung Fu a live
candidate; it does not exist. (i) The instrument was validated first:
on the Castlevania positive control it blind-nominated `$0041` at 14/16
wrap hits — exactly the pair shipped in `configs/castlevania.yaml`.
(ii) On Kung Fu it nominates `$006B` at 2/4; reproducing the hold, all
four "wraps" are 236→0 or 216→0 drops sitting on lives transitions —
**death-respawn resets, not wraps**. (iii) Zero wraps in live play at
three scales: 6 policies × 2 seeds (`hard_drops = 0` in all 12), 64
parallel stochastic episodes (0 crossings, max 164), and the project's
own 514,239-step receipt reaching `max_progress` 244 against
`random_baseline_max_progress` **244, identical** — search bought
nothing over random, the signature of a hard clamp. (iv) Handing
`assess()` a free *perfect* high byte still returns `passed=False`
("only 28 distinct values in 174 steps"), so **the high byte was never
the binding constraint** and `gate_flips = false` is a proof. (v) An
exhaustive 2048×2048 pair scan returns nothing real.

**Correction to the recorded reason for Kung Fu's failure.** The
"single byte wraps" premise is FALSE: `$0094` never wraps in live play.
It is a clamped screen-space coordinate that saturates at 236–244 and
resets to 0 only on death. The true instrument fault is **coarseness**
(28 distinct in a 174-step live window). Nobody should spend another
probe on the "cheap fix" of hunting its high byte.

**New defect — the gate is blind to auto-restarting ROMs.** Reported,
not fixed: changing a binding gate's truncation logic moves many
profiles' verdicts. Kung Fu's own gate run is post-death-tail
contaminated — 1,326 of 1,500 steps follow the first death — yet the
gate reports `dropped_tail_steps = 0`. **Reproduced at HEAD:** Contra
emits its `[D5]` truncation line; Kung Fu emits none and reports a full
"1200 steps". Root cause: `first_exhaustion_index` requires the
trailing quarter of the lives trace to be frozen at one value, but this
ROM **auto-restarts after GAME OVER** (3→2→1→0, then 0→3, then 3→0), so
that quarter holds `{0, 3}` and the detector returns `None`. The stasis
detector misses it independently because an auto-restarted attract game
is **busy, not frozen** (median churn 48 bytes/step). This is a **third
contamination class** beyond D5 (Arkanoid frozen-placeholder) and D1
(Ninja Gaiden blind lives byte), and it plausibly inflates the "28 of
45 profiles contaminated" count for any multi-life ROM that loops back
to attract. Cheap targeted fix: treat the FIRST death as the truncation
point whenever the lives byte later returns to a value ≥
`lives_at_start` — a restart can only be a new game, never continued
progress.

**Purity (Tier 3).** Every measurement above comes from hardware
surfaces — PPU scroll odometer, nametable VRAM, rendered frames, OAM,
render-line counts — plus each profile's own declared lives byte, on
its own start state. No disassembly, no RAM map, no walkthrough, no
recall of these titles.
