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

ADDENDUM HP-1 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`) — **the protocol as
IMPLEMENTED is sound and was measured rather than read; the paragraph
above describes three things the harness does not do.**

Verified by counting the events the protocol is defined by, not by
reading the code: sticky fires at a measured 0.2504–0.2572 against a
requested 0.25 (and at exactly 0 and exactly 1 at the endpoints); the
`step > 0` guard holds; jitter is uniform on 0..16 inclusive and produces
17/17 distinct RAM states from every banked start blob; per-episode
seeding does not collude (0 collisions in 200 derived seeds); the two RNG
modes are distributionally identical; the parallel executor agrees with
the serial loop on real emulation; `episode_success` is the strict
flagpole/castle predicate with a warp guard; the denominator is genuinely
single-life. Across 224 receipts and 11,200 episodes `clear_rate × n`
equals the independent event count EXACTLY in every receipt.

Three corrections to the text, all WEAKENING, none touching a number:

1. **"Cold power-on start, zero state loads at test time" is false for
   every per-level number.** Each honest per-level episode calls
   `pool.load_worker_state(0, blob)` on an entrance state
   (`stage_03.state` = area byte 1, x 0, lives 1, score 001595). These
   are legitimate level entrances and carry no physics advantage under a
   single-life denominator — but only 1-1's `entrance_start.state` is
   power-on equivalent (world 0, area 0, x 40, lives 2, timer 400,
   score 000000).
2. **"Greedy and sampled action selection both declared" was not honoured
   by the v27/v28 gates.** Both registrations name
   `--action-select {greedy,sample}`; `sample` is not a valid choice
   (argparse accepts `greedy`/`sampled`), and all 32 gate receipts are
   greedy. No sampled receipt exists for either campaign.
3. **"Action receipts recorded and self-replay verified" is not something
   `scripts/eval_game.py` does.** It has no `--record-actions` and no
   replay verification; only `scripts/eval_composite.py` does, and that
   is not the harness any banked learned number came from.

Also fixed rather than merely recorded: the receipt now MEASURES its own
protocol (`sticky_applied`, `sticky_eligible`, `sticky_measured`,
`jitter_hist`) and records its identity (`max_steps`, `profile`, `rom`,
`rom_sha256`, `n_episodes_delivered`). Before this, all of `sticky_prob`,
`start_jitter` and `stochastic` were computed from **argv**, so a run with
the mechanism physically deleted emitted a byte-identical receipt to a
correct one — and deleting sticky, or the Machado no-op prologue, from
both executors left 534 and 141 suite tests passing respectively.
`max_steps` — registered at 1500 for v28 and 3000 for consol2, and able to
move a 1-2 clear rate from 0.20 to 0.15 — appeared in **0 of 921** honest
receipts, as did the profile. No banked number retracts; two receipts with
identical visible fields could nonetheless have been different protocols.

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

## ENGINE PURITY 2026-08-27 — the layer the config quarantine could not reach

Full write-up: `docs/research/ENGINE_PURITY_2026-08-27.md`.

**The config quarantine retracted DOCUMENTATION claims; the executing layer
kept its own copies.** The 994-entry `configs/` sweep named this limit in
its own commit message ("Quarantining the YAML retracts the DOCUMENTATION
claim, NOT the Rust constant"). It was not a footnote — it was the whole
exposure. For **Kid Icarus (`$0130`)** and **Double Dragon (`$0030`)** the
YAML retracted a specific sentence and *that exact sentence survived
verbatim in `nes_core/src/rewards.rs`*. The documentation moved, the wiring
did not, and the two layers then disagreed in writing with the wrong one
running.

**Counts.** 134 RAM-address constants swept across `nes_core/src/` (109
`const RAM_*` in `rewards.rs`) plus 23 non-address constants carrying
semantics. 21 sweep findings ruled SEMANTIC-and-UNWITNESSED, covering **27
individual constants across 11 games — all 27 now annotated** with a
provenance tag naming what they ASSERT, that there is NO WITNESS, and what
would EARN IT. **Reward arithmetic changed: 0.** 12 Python sites corrected.
24 enforced quarantined-address sites now disclosed.

**NO BANKED CLAIM IS RETRACTED, and none was at risk.** Every one of the 27
is *unfired*: not one sits under a quoted number, all belong to games with
no witnessed clear, and no boss defeat has ever been witnessed on any game
here. That is exactly why this was cheap now and expensive after one fired.
The witnessed side was left alone and is unchanged: **SMB's block is
byte-identical** (no existing executable line in `rewards.rs` was removed or
modified anywhere in the file) and is now positively marked `PURITY:
WITNESSED`; likewise Castlevania `$0028`, Bubble Bobble `$0401`,
Excitebike's section chain, Tetris's line bytes and Punch-Out `$0398`.

**The engine came back mostly clean, and that is a good outcome, not a
failed sweep** — a third pass finding little unfired residue is evidence the
first two worked. Several blocks are already the right discipline and were
deliberately not touched: Contra's `clear_screen` 255 sentinel, Gradius's
"NO byte is trustworthy as the stage index yet", Ghosts' disabled
`stage_addr`, Bubble Bobble's `enemy_count_addr`, Kung Fu's opt-in `$04A5`.
An assertion that honestly records a null is a measurement, not a breach.

**Sharpest breach:** `DRACULA_STAGE = 0x12` justified itself by quoting the
ROM's own disassembled instruction (`cmp #$12`) — a question resolved by
knowing the game, the Tier-3 line exactly. Proof withdrawn, value kept
(removing it is behaviour), recorded as believed-not-proven.

**Measured, not argued — including against myself.** Punch-Out's null has a
built-in control: `$0001`/`$000A` held 0 across 15,000 steps while `$0398`
took 40 distinct values over the SAME window, so the bout was genuinely
being fought. Zelda's feared spurious 30,000-point payout is **REFUTED**
for this start state (`$0609` held 1 across 600 NOOP + 60,000 random steps;
0 frames == `0x10`, 0 frames with bit `0x02`), and the refutation is
reported rather than the suspicion. One inherited number did not survive
re-measurement: Castlevania `$0071` was described as "5 values in 5..14,
rising only"; it is 11 values across 0..10 — it falls too, and the
annotation declines to assert "rising only".

**Now mechanically checked**, all mutation-tested by actual revert: a
27-row provenance registry pinning each constant *at its recorded value*;
a retraction lint pinning the 8 withdrawn sentences dead; `WIN_WITNESS_LEDGER`
(17 rows, one per reward arm) with five Rust tests that DRIVE each arm
through the byte its row names; and a tree-wide scanner wired into `make
test` that derives addresses from the quarantine blocks and ownership from
the source's own dispatch table, so neither can drift from what it guards.
Full revert → 63 failures; **all 27 single-tag deletions caught**; SMB
over-withdrawal → 1 and 4.

**Three guards shipped earlier the same day were themselves defective and
were fixed before landing.** (a) The half-fix guard was largely vacuous: a
60-line lookback let a *neighbouring* constant's tag stand in, so **19 of 24
rows survived deletion of their own provenance block** — including the
Zelda, Metroid and Castlevania win chains. Tag scope is now the constant's
own comment block, with shared blocks referenced by name; **27/27 caught,
up from 5/24.** (b) The stale-`.so` guard compared only the *set* of reward
ids, so a flipped status or re-pointed predicate passed against a stale
binary — precisely the edits this sweep makes; it now compares all three
fields on all 17 rows. (The binary had not in fact drifted; the guard could
not have told.) (c) The retraction lint's 50-line marker lookback let all
8 clauses be restored silently *in the very header each was withdrawn
from*; it is now an exact quotation census.

**One behaviour change landed, outside the engine, and it is a correctness
bug rather than a purity retraction.** The ROM resolver's franchise-collision
check compared `candidate_markers - canonical_markers`, a one-sided set
difference that is empty whenever the candidate carries no installment
marker. Measured against the real canonical names in `configs/`, **five
profiles bound the wrong dump** — `lost_levels` → `Super Mario Bros.
(World).nes`, plus Castlevania III, DuckTales 2, Ninja Gaiden II and Double
Dragon II each binding the original. **`lost_levels` is on the witness
ledger.** The comparison is now symmetric, and does not overcorrect: a
glued marker (`megaman2.nes`) still matches. No banked Lost Levels result
is retracted — its receipts are replay-verified against the correct dump —
but the resolver could have bound the wrong one on a fresh run.

**The reusable finding, which generalises past this repo: a quarantine that
covers only the declarative layer is incomplete by construction.** Wherever
a claim is written in one place and executed in another — config vs code,
spec vs implementation, policy vs enforcement — retracting it in the
declarative layer feels like retracting it, produces a clean diff, and
leaves behaviour untouched. Two corollaries: a partial retraction is *not*
a partial fix, it is a new defect class (before the sweep both layers were
consistently wrong and either one could be read; after it they disagreed
and the authoritative-looking one was wrong); and the enforcement must be
**derived** from the declaration, never listed beside it, or the list is the
same defect one level up.

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

ADDENDUM N-1 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`) — **the negative is
RE-SCOPED, not withdrawn: what failed was the CGSA-PPO recipe, and the
"policy class" generalization above is refuted by this ledger's own later
receipt sixty lines down.**

The sentence held to the bar it demands of a positive:

1. **"The policy class was falsified" is WITHDRAWN.** The falsification
   checkpoint (`checkpoints/mario_1_2_cgsa_s1/vanilla_ppo_iter_05990.pt`)
   is a 2-layer LayerNorm MLP on the 712-d stacked tile observation,
   95,943 params. The banked 38/100 checkpoint
   (`checkpoints/_preserved/consol2_40pct_strict_iter01120.pt`) is **the
   same architecture family at twice the width** — 200,071 params, same
   712-d input, same `stage_03.state` entrance, same protocol family — and
   it clears **38/100**. Same class, same entrance, 0 → 38. What changed
   was the training procedure, which is precisely the variable the
   falsification sentence claimed was not responsible. Correct scope:
   *the CGSA-PPO recipe failed its own pre-registered signposts on three
   seeds.* That claim, and the SPRT machinery under it (correct Wald
   thresholds, welds accruing only at p ≥ target, 96–99% welded at full
   0.25 protocol noise, median 14–29 completed windows), **stand
   untouched.**
2. **The literature sentence is WEAKENED.** Its cited source ("external
   deep research, 2026-07-23", `RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md`
   §"The claim this documents") does not exist in this repo or in
   `research-consult/responses/`, whose earliest artifact is 2026-07-27.
   The protocol is also bespoke, which makes the claim close to
   true-by-construction. Quotable form: *"we are aware of no published
   per-level 1-2 clear rate under Machado sticky-0.25, in either
   direction."* It must never be compressed to "1-2 is unsolved by the
   field."
3. **What is genuinely strongest here is the offline-imitation closure
   below, not this paragraph.** That result is PREDICATE-INDEPENDENT — A7
   tops out at max-gx 976 of 3,266, so it reads 0.0 under the flagpole
   predicate, the level-advance predicate, and any x-threshold — and it
   was built to the standard of a positive: exact dataset verification,
   a fit gate pre-registered two hours ahead, and a matched ablation pair.

This entry is annotated rather than deleted: the underlying data is sound
and the withdrawal is of a generalization, not of a measurement. The same
correction is owed to `docs/research/README.md:21` and to the header and
2026-08-08 correction of `RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md`, both
of which propagate "formally falsified for the policy class".

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

ADDENDUM 2026-08-27 — THE 38/100 IS NOW CANONICAL, AND IT HOLDS.
The consol2 receipts backing "definitive eval ... under the canonical
protocol" were measured at `eval_rng: shared-stream`, while this
document's own LEARNED definition requires `--eval-rng per-episode`.
The word "canonical" was therefore doing work the receipts did not
support. Re-run today on the same checkpoint
(`consol2_40pct_strict_iter01120.pt`), same start state, same strict
flagpole predicate, `--eval-rng per-episode`, 50 eps x seeds {7,101}:
**31/100 (31.0%)** — seed 7: 14/50, seed 101: 17/50. Against the banked
38/100 that is z = -1.04 (SE_diff 6.7 pts), NOT significant, and the
canonical 95% Wilson interval [22.8, 40.6] contains 38. Per-seed motion
is in OPPOSITE directions (seed 7 0.48->0.28, seed 101 0.28->0.34),
i.e. RNG-regime noise rather than protocol bias. `max_byte_seen` 3 on
both, so warp_rate 0 is certified post hoc. Receipts:
`docs/receipts/consol2_canonical/perep_seed{7,101}.json`.
**Quote 31/100 when citing this result under the canonical protocol;
38/100 remains correct as the shared-stream measurement.**

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

ADDENDUM C2-1 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`) — **the 38/100
REPRODUCES bit-exactly and is the ledger's live 1-2 number; two sentences
around it are WEAKENED, and this entry now supersedes the 2/100 headline
and the policy-class falsification above (ADDENDUM N-1).**

Independently reproduced eleven days later from the committed checkpoint,
start state and harness: seed 7 → 0.48 / mean_length 676.16, seed 101 →
0.28 / 704.26, matching `peak_eval_seed{7,101}.json` to the digit.
`max_byte_seen` reads 2 and 3, retro-certifying `warp_rate 0.0`.

1. **"Definitive eval on fresh seeds under the canonical protocol" is not
   accurate.** Both receipts record `eval_rng: shared-stream` and
   `eval_workers: 1`; this ledger's LEARNED definition requires
   `--eval-rng per-episode`. The result is NOT an RNG-mode artifact — the
   campaign's own per-episode probe at this checkpoint reads 0.40 strict
   at n=30, and a 10-episode per-episode run on 2026-08-27 read 0.30 —
   but the pooled 100 should be re-run per-episode before the sentence is
   restored. **This is the single highest-value outstanding action in the
   learning track**: two evals convert the repository's strongest positive
   from nearly-canonical to canonical.
2. **"38% on 1-2" is a property of ONE PRESERVED CHECKPOINT, not of the
   recipe.** The neighbouring probes at iters 1070/1180/1230 read
   0.067/0.033/0.0 and the run was ended by its own pre-registered
   collapse kill. The 100-episode eval is an unbiased estimate of that
   fixed artifact — which is exactly what makes it citable — but the
   training recipe is not shown to reach 38% a second time. Quote it as
   *"one preserved checkpoint clears 1-2 at 38 ± 5%"*, never as a
   reproducible training outcome.
3. **"Reach-without-grab eliminated" is supported at n=30 for this
   checkpoint** (chain vs strict differ by at most 1/30 in both legs of
   `campaign.jsonl` at the peak), and should not be generalized further:
   the strict predicate is measurably conservative elsewhere — four of the
   fifty banked seed-7 episodes reach x=3266/3267, hold player_state 0x05
   for 65–95 agent-steps and advance `$075C` 1 → 2 with no life lost, a
   displayed completion the flagpole latch never sees. The predicate can
   only UNDER-count, so every banked positive is a floor and no FAIL
   comparison is biased by it.

Supersession, stated once so it stops propagating: **the 2/100 figure is
no longer the banked 1-2 state.** It is correct for its own named
predicate and reproduces exactly, and it under-counts its own artifact by
roughly 4.5× (the same 100 episodes carry ~9–11 level completions the
strict latch cannot see). The arc is 0.0 → 2.0% → **38.0%**.

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

ADDENDUM B1-1 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`) — **the 1-1 number
STANDS on re-measurement; the RECIPE that produced it is misdescribed by
twelve of its own config lines.**

Re-measured under the canonical protocol on
`checkpoints/_preserved/backward_1_1_seed3_iter140.pt`: eval seed 0 reads
**0.76**, reproducing the banked figure. (Its second seed, never
previously run on this checkpoint, reads 0.60 for a pooled 0.68 — see
ADDENDUM V-4 item 2 for what that does and does not license.)

What is corrected is the recipe text. `configs/mario_1_1_backward.yaml`
carries a **"Phase-A recipe, verbatim"** header over twelve keys that the
trainer parses, `--strict-config` accepts, the run manifest records, and
`trainer_mode: vanilla_ppo` can never act on — because `Trainer.run()`
dispatches to `_run_vanilla_ppo` and returns before the GA loop, and every
one of those keys is consumed only inside `_run_one_generation`,
`_run_bc_replay`, `_snapshot_pre_ppo_elite` or `_reinforce_update`:

```
bc_replay_enabled  bc_replay_epochs  bc_replay_every_gens
bc_replay_max_buffer  bc_replay_train_window  episodes_per_genome
warmup_gens_ga_only  preserve_elite_diversity  freeze_pre_ppo_elite
symlog_rewards  enabled  async_pipeline
```

All eight `mario_1_1_v27_seed*.yaml` / `mario_1_1_v28_seed*.yaml` configs
inherit the same twelve. **Numeric impact: none** — each key is inert
identically in the baseline and in every comparison arm, so 1-1 0.76,
v27 0.530 and v28 0.670 remain like-for-like against the 0.767 bar.

Stated plainly, because it is stronger than "the knob was ignored":
**the banked 1-1 runs had NO clear-anchoring mechanism active at all.**
Those configs declare no `sil:` block either, so it is not that a
different anchor fired instead — none did. `bc_epochs: 30` is doubly
inert: it is read only inside `_behavior_clone_seed`, gated on
`bc_demo_path`, which no config in `configs/` sets and which
`scripts/train_game.py` — the launcher every banked run used — exposes no
flag for. Setting it to 0 or 999 would produce byte-identical runs.

The trainer already carries a hand-written "this knob is a no-op, warn and
disable" guard for `preserve_elite_diversity`/`freeze_pre_ppo_elite`. It
tests `trainer_mode == 'pure_ppo'` and therefore stays silent in the only
mode any banked run used. Rather than fix that guard by hand again, the
check is now DERIVED from the AST:
`config_schema.inert_reinforce_keys_under_vanilla_ppo()` walks `run()` to
the early return, closes over `self.foo(...)` calls, and reports every
`reinforce` key whose consumption sites all lie outside that set —
20 keys, with `symlog_rewards` (found by hand on 2026-08-26) reproduced as
the calibration case and `ppo_clip_eps`/`rnd_loss_coef` correctly NOT
flagged because `ppo_updater.py` reads them off the trainer handle. It
raises rather than returning an empty set when it cannot find the dispatch
it parses. `tests/test_inert_key_reachability.py` pins both directions.

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

ADDENDUM V-3 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`, receipts
`runs/v27_readjudication_2026-08-27/`) — **item 3 above is now MEASURED
and its worry is REFUTED. The v27 FAIL margin clause is restored.**

A registration-literal re-selection was registered in full before any new
eval ran, using the selection statistic both registrations actually name —
the printed `[backward]` trailing entrance rate at at-entrance iterations,
ties → later. That statistic consumes ZERO evaluation data, so the
estimator carries no winner's curse against the honest gate.

* **v27 best-of-4 = 0.500** (per-seed 0.030 / 0.500 / 0.480 / 0.460, each
  n=100 pooled, per-episode) against a banked 0.530 and a FAIL bar of
  ≤0.767.
* **Adversarial one-sided flip test**, as registered where no full ladder
  exists: the highest pooled n=100 rate at ANY v27 checkpoint ever
  evaluated is 0.51 (seed 2 iter 110 — the winner's-curse regression the
  rule anticipated, visible in the data), one-sided 95% upper bound
  **0.592**. Nothing approaches 0.767.
* **v28 best-of-4 = 0.670**, identical to the banked headline, though 3 of
  4 per-seed numbers differ. Split-sample cross-fitting on the full
  192-receipt F0 ladder gives 0.670 or 0.720 depending on one unregistered
  tie-break.

Three independent selectors — `winners/best.pt`, the registration-literal
log peak, and split-sample cross-fitting — return one verdict: **FAIL,
both campaigns.** The "could plausibly approach 0.7" concern in item 3 is
retired, and *"not close on either bound"* is restored for v27 at 0.500
vs 0.767.

Two things move the other way. **v27's per-seed field was mostly selector
noise**: corrected it is 0.03 / 0.50 / 0.48 / 0.46, so three of four seeds
are a three-way tie and only seed 0 is genuinely bad — which finishes the
withdrawal of "telemetry ranks seeds correctly" rather than softening it.
And **"the measured winner's curse = 0.05" is WITHDRAWN as a carry-forward
constant**: it turns entirely on an unregistered tie-break (0.670 vs
0.720). `F0_CORRECTED_PEAK_LADDER_2026-08-26.md`'s headline ("under-select
on 3 of 4 seeds") also contradicts its own table three lines below (4 of 4
improved).

NEW MECHANISM, sharper than "the argmax over a saturated statistic is
near-arbitrary": the winner selector was **ceiling-LOCKED**, not merely
noisy. `save_winner` skipped on `prev_val >= metric_value` while
`entrance_trailing_rate` is successes/30 with maximum exactly 1.0 — so the
first iteration to record 1.0 made the gate mathematically unsatisfiable
and froze the winner for the rest of the run. v27 seed 2 and v28 seed 3
both froze at iter 90 of 250; across all 8 runs the last save lands at
iter 50–120 (median 65) with 3–6 saves per run. That is deterministic,
one-directional under-selection. Fixed 2026-08-27: ties now go to the
later `source_iter`, which is the rule both registrations declared and
neither implemented.

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

ADDENDUM V-4 (2026-08-27, learning-track audit
`docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md`) — **the FAIL stands at
0.670 under a third, curse-free selector (ADDENDUM V-3). Three sentences
in the paragraph above are WITHDRAWN, and both campaigns must be restated
as SINGLE-variable arms.**

1. **"Identical to the v27 gate in every respect (… `--eval-rng`
   per-episode)" is FALSE.** All 16 v27 gate receipts record
   `eval_rng: shared-stream` / `eval_workers: 1`; all 16 v28 receipts
   record `per-episode` / 8. `V28_CAPACITY`'s seed-paired "Same protocol"
   table carries the same error. Measured, it is not a bias — 53/200 vs
   57/200 on matched weights — and the registered reading rule reads only
   training-log metrics, so the adjudication itself is uncontaminated. But
   the cross-cohort GATE-DELTA table (+0.410/−0.060/−0.160/+0.500) spans a
   harness change. `PEAK_INSTABILITY_FORENSICS_2026-08-25.md` §1.5 found
   this on 2026-08-25 and it never propagated here.
2. **"No seed's better checkpoint reached the banked control's own 0.767"
   is WITHDRAWN for v28.** The 0.767 bar had never been measured under the
   protocol it gates: it is 46/60 at eval seed 0 only, shared-stream, one
   worker. `V29_STABILITY` made the two-seed re-measurement an F0
   deliverable; F0 never ran it and V29 was withdrawn. Measured
   2026-08-27 on `backward_1_1_seed3_iter140.pt` under the canonical
   protocol: es0 **0.76** (reproducing the banked single-seed figure), es1
   **0.60**, pooled **0.68**. v28's 0.670 is statistically
   indistinguishable from a same-protocol control of 0.680. **The
   registered thresholds do NOT move** — that would be the goalpost move
   this ledger treats as a fabricated result — and the FAIL stands on the
   registered bar. What does not survive is the story told underneath it.
3. **V3/V4's "ReDo ENABLED in all four training logs" certified ARMING,
   not FIRING, and ReDo never fired in either campaign.** It logged
   `recycled 0 cum 0` on every one of ~2,000 per-iteration checks across
   all 8 v27+v28 runs. This upgrades from "never fired" to **"could not
   fire, knowably, before launch"**: the dormancy statistic normalizes
   post-activation magnitudes by the layer mean while `TilePolicyNetwork`
   LayerNorms immediately BEFORE the SiLU, holding the statistic near 1.
   The repo's own pre-launch sweep recycles zero units at every tau ≤ 0.20
   and first fires at tau = 0.25 — **ten times the registered 0.025** —
   and `isolate_tau0.35.log` was written 2 min 21 s before the 8-run
   budget started, then read as a fresh-net artifact. The pre-registered
   V7 armed-evidence gate evaluated every one of its conditions at
   tau = 0.5, twenty times the experimental value, and never required the
   REGISTERED operating point to be reachable: **that is the seventh
   vacuous gate on this ledger, same shape as the previous six.**

   Honest restatement: **v27 and v28 each tested ONE variable** (merged
   ladder / capacity), not two. The FAIL verdicts are untouched — neither
   depended on ReDo doing anything — but AMENDMENT 1's two-variable
   framing is withdrawn, as is v27's un-applied pre-registered FAIL caveat
   about a ~0-recycle run.

Newly shipped so this class is detectable rather than audited:
`scripts/check_mechanism_receipt.py` returns **VOID** (not FAIL) for any
armed mechanism whose own counter never moves for a whole run, and
distinguishes that from `UNAUDITABLE` — armed with no counter at all,
which is what `[hazard-mask] ARMED` runs are. Run on these receipts it
reports `redo INERT, peak 0 over 1000 observations` for v27 and
`backward FIRED` alongside it. `tests/test_mechanism_receipt.py` pins both
directions, including a positive control on the 1-2 campaign so a checker
that returned INERT unconditionally would fail.

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
from our own rendered frames.** ~~no instrument can count them~~ —
**that clause is WITHDRAWN 2026-08-26.** `odometer_scene` reads 0 cuts
across a tape that provably crosses 55 blackout transitions, because
`odo_fold_frame`'s blank branch returns before the scene-cut test — but
the pipeline carries a *second* counter, `odo_blank`, and edge-detected
into runs it reads **55/55 transitions and 3 areas** on the same tape,
agreeing with the eye. The correct scope of the original finding is
"`odometer_scene` cannot count them", not "no instrument can". Every
`rooms_reached: 1` reported during the campaign was still an inference
from an instrument that cannot fire.

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
- **Contra — SIGNAL SOUND. The exclusion is WITHDRAWN** (corrected
  2026-08-26; defect fixed in `5a09775`, see the ODO_BLANK section
  below). The original "SIGNAL UNUSABLE — only 20 distinct in 69 steps"
  rested on a broken inference: the gate computed its resolution
  finding on the window *after* D5 truncation, and **a 69-sample window
  cannot demonstrate a 32-distinct threshold**. That verdict measured
  how fast the scripted hold died, not the signal's resolution. At HEAD
  the fixed gate returns **INCONCLUSIVE** on the same hold (69 live
  steps, 20 distinct) and **PASS — SIGNAL SOUND** under `--probe
  random` (721 live steps, **346 distinct**, range 0..1063); composed
  across both probes Contra is **SIGNAL SOUND with zero faults**. The
  pair `{lo: 0x0065, hi: 0x0064}` is a sound 16-bit progress
  observable. Rygar's PASS is unaffected.
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

---

## ODO_BLANK, THE COHORT, AND THE PROGRESS GATE 2026-08-26 — EXHIBITION

Full write-up: `docs/research/ODO_BLANK_AND_GATE_2026-08-26.md`. Rygar
refusal: `docs/receipts/rygar/clear_predicate_REFUTED.md`. Arming
policy: `scripts/scene_cut_arming.py`. Its two receipts:
`docs/receipts/clear_control/scene_cut_arming_2026-08-27.json` (the
configs as shipped) and `..._asfound_2026-08-27.json` (the 2026-08-26
arming as found — the evidence the disarm rests on).

**Ledger: EXHIBITION.** Every Rygar and Contra number in this section
is search output, scripted-hold probe output, or uniform-random rollout
output. No policy was trained for either game; no honest-protocol
evaluation was run for either. Nothing here may be described with "the
AI learned", "the AI plays", or "the AI beat".

**1. Rygar still has no clear predicate, and the blocker moved.** The
instrument objection is answered — `odo_blank`, edge-detected, reads
55/55 Rygar transitions and 3 areas on the banked R1 tape. The
predicate is still **REFUSED**, with numbers rather than a shrug: the
two predicates the instrument makes available fabricate **55** and
**28** wins on that same tape, where the honest count of new ground is
**2**. The decisive blocker is that **no Rygar clear has ever been
witnessed** — 71 of 71 `solutions/` directories empty, every one a
compile-time constant. A predicate with no witnessed positive can only
be shown *not* to fire, which is the exact shape of this week's four
vacuous gates. `configs/rygar.yaml` is unchanged (`level_key: []`, no
`clear`, no `finale`) and a test now guards that. **R1 condition 2
stays FAIL.** Step 1 of a real predicate is a witnessed positive,
blocked by the 4,608 px search wall — a *search* problem, not an
instrument problem.

**2. `odo_blank` is a transition counter for 20 of the 25 odometer-
cohort profiles.** Measured twice, independently, and **both counts are
20**: the 2026-08-26 survey at 3 × 4,000 steps per profile, and this
audit at 4 × 6,000. The membership differs by one swap and that is
reported rather than smoothed: dead in **both** are
`batman_the_video_game`, `double_dragon_ii`, `mega_man_3` and
`tetris_usa` (so 19 of 25 move it in both runs, 21 in at least one);
`ninja_gaiden` (0 → 3 runs) and `power_blade` (1 → 0) each moved in
exactly one and are marginal, not established either way. In
distinct-ROM terms, **19 of 24 games** (`metroid`/`metroid_roomfp`
share a ROM).

This matters because `clear_reachability` marks `coord` DEAD for the
whole cohort — verified for all 25 — which is what puts every one of
them at quorum UNREACHABLE. These games have never had a transition
signal that could fire.

Three usage constraints, all measured, none optional: **(a)**
edge-detect into runs — the raw counter reads 4,329 on the Rygar tape,
78× the true count of 55, because it counts blank *frames*; **(b)**
threshold on run LENGTH, calibrated per game from that game's own
rollouts — the raw counter takes the same branch for a death fade, a
boot fade and a door, so a bare "it moved" predicate fires on **every
death**. Measured over 18 rollouts, each confirmed to actually die:
death fade **14** blank frames (36/36), boot fade **9** (18/18), door
**78–79** (55/55), floor 40. That separation is **Rygar's**, not the
cohort's; **(c)** treat it as
**per-trajectory, not monotone** — `odo_blank` rides inside `OdoState`,
so a Go-Explore restore carries the saved value back in (measured 273 →
90). Under 12,036 restores the transition count held at 55 with zero
fabricated events; the error is always downward.

**3. The 2026-08-26 cohort arming (`4dd15ea`) is WITHDRAWN. All 23
profiles are disarmed.** That commit armed `scene_cut` off a survey
whose reproducer was never committed, and the arming was a judgement
rather than a measurement. Re-run as a measurement —
`scripts/scene_cut_arming.py`, 4 × 6,000 steps of each profile's own
mixed-random play, through that profile's own armed signal built from
the real YAML — **14 of the 23 armed gates fire on play that cleared
nothing**, worst `paperboy` 207, `ducktales_2` 113, `ghosts_n_goblins`
102, `gradius` 65. Every fire is a false positive and every one
survived the death veto.

**The general reason, which is the same one that refuses the Rygar
predicate.** The audit probe clears nothing, so its entire observed
`(d_scene, d_blank)` distribution is null. A gate at or below that null
fires on ordinary play — demonstrated above. A gate above it has no
evidence it can ever fire. An arm therefore needs a **witnessed
positive**: one blank run known to be a level transition, so a length
floor can be placed between it and the death population. Rygar has
exactly one (death fade 14 blank frames, door 78–79, floor 40) and is
served by its own instrument; **no other cohort profile has one**, which
the 2026-08-26 survey says itself in its own `scope_limit`. The clause
is in code (`C7_SEPARABILITY_WITNESSED`) with a register that can be
added to.

Disarming returns all 23 to quorum **UNREACHABLE / ceiling 0.75**, the
pre-`4dd15ea` state, verified profile by profile. The specific defects
review found are all real and are recorded per-profile in each config's
refusal: `kind: [fade]` armed on a blank channel measured at zero runs
(the evidence state `tetris_usa` was DECLINED for); a death veto over an
admitted `lives: 0` placeholder and over a byte this repo documents the
same day as a 0↔255 flicker artifact; gates at or below their own
measured null; and no roster-level test reading the real configs at all.

**A defect review did not find, measured here.** The death veto was not
guarding the blank channel in the first place on several profiles:
`SceneCutSignal`'s window is 240 observations while the veto's transient
is a handful, so a fade's blank movement sits in the rolling buffer long
after `dying` clears and lands in a non-vetoed window anyway. On
`megaman` the vetoed-out null is (0, 0) while the veto-independent null
is (2, 18) — its "residual 0" was a property of the veto, not the gate.

**Two receipts, deliberately.** With nothing armed, every residual
assertion iterates an empty set and passes for free — the exact vacuity
this campaign is about. So the evidence the disarm rests on is committed
too (`..._asfound_2026-08-27.json`), and a test fails rather than skips
if it goes missing.

**Scope correction to the `4dd15ea` claim.** "Eligibility only —
`is_clear()`/`solutions` byte-identical" is true of the **solver** and
false of the **offline harness**: `clear_detect.run_episode` builds the
shelf with no `mode` check, so all 23 moved UNREACHABLE → FIREABLE
there. That claim must be stated with its scope.

**4. Contra's exclusion is WITHDRAWN, and the correction is on the
record.** `progress_signal_gate.assess()` computed its resolution
finding on the post-truncation window; a 69-sample window cannot
demonstrate a 32-distinct threshold. Contra is **SIGNAL SOUND**
composed across both probes (721 live steps, 346 distinct, range
0..1063 under `--probe random`). Re-measured on the banked hold sweep,
**25 of 45** profiles have their assessed window truncated because the
probe ran off the end of live play, so this was never one game. (The
original finding put that count at 28; 25 is the number re-measured
here under the stated definition.)

**Contra's `gx 3072` wall is REAL as a position and BLIND as a
verdict**, and it is a *different code path* from the gate defect —
`go_explore_solve`'s archive-based `max_gx_in_max_area`, not `assess()`.
Evidence it is real: ~~the identical value appears in nine of ten
independently-run campaigns (2026-07-31 → 2026-08-11, materially
different strategies, millions of steps each, 280k–357k cells), 3,030
verbatim occurrences in `runs/`~~; `progress = hi<<8 | lo` verified
live, so `3072 = 12 × 256` exactly; and undirected play tops out near
1,063–1,075, so reaching it is earned. Evidence it is blind: `hi` at 12
with `lo` frozen at 0 is a **fixed-camera room**, and a scroll-derived
progress definition cannot see inside one. "Pinned at 3072" therefore
**cannot distinguish** "cannot get past the room" from "cannot see into
the room", and must never be quoted as a difficulty verdict on the
game.

> **AMENDED 2026-08-27 — the struck clause overstated the prior by
> about 2.5×, and the wall survives the deflation.** The count is
> exactly **2× double-counted** (3,030 = 1,508 `progress.jsonl` + 1,508
> tee'd `.log` twins + 14 JSON), the 1,508 lines are one-minute windows
> of a **running max** across processes rather than trials, **9 of 18
> runs are `--resume-archive` children of just 2 parents**, 8 of the 9
> fresh runs boot from the same single savestate, and **17 of 18 already
> read 3072 in their first 60-second window**. Restate as: **~8
> independent searches from 2 root states, all pinned at 3072 within 60
> seconds.** Still a real, reproduced wall — reproduced fresh at HEAD
> for the 2026-08-27 campaign — just far cheaper to have established
> than "20.7 h and 162 GB" implies. And the "blind" half is now
> **settled by measurement rather than inference**: the room is a
> screen-locked lethal arena, the agent is alive and in control inside
> it, and the camera does not move. See the CONTRA WALL section below.

**5. Contra is the recommended next campaign target over Rygar**, and
the reason is structural. Verified in code: Rygar's `_clear_mode` is
`None` and its quorum is **UNREACHABLE** (ceiling 0.75, `coord` DEAD),
so `solutions: 0` is a constant and evidence of nothing. Contra's
`_clear_mode` is `"confluence"` and its quorum is **FIREABLE** (ceiling
1.0 offline / **2.0 on the live hook the solver actually runs**,
`coord` ALIVE with transition evidence, because a 16-bit `{lo, hi}`
pair can express the backwards drop). Contra's `solutions: 0` is
therefore weak-but-real evidence rather than a constant. ~~Its gating
task is a **Finding-3-class audit of the confluence detector**~~ —
`configs/contra.yaml` documents it in-file as "UNTESTED — never
observed to fire on a genuine clear", so nine campaigns of
`solutions: 0` still cannot separate "never beaten" from "beaten and
not detected". That is a well-posed question. Rygar's is not yet
askable.

> **AMENDED 2026-08-27 — the gating task is REORDERED, and the audit
> was performed.** The audit ran (2026-08-27) and its answer is that
> **the detector cannot be audited until a clear exists**: with the
> measured `tally` null fire-rate of **1.00** (58/58 checks on
> death-terminated random play) fed to `clear_quorum`, the verdict
> flips **FIREABLE → UNREACHABLE**, ceiling 1 < required 2, `tally`
> **DEGENERATE**. The shipped 2-of-2 is a 1-of-1 `coord` vote wearing a
> corroborator's clothes, so **any Contra null from this hook is VOID,
> never a miss**. The gating task is therefore **reaching a stage
> boundary — a search problem, not a hook-tuning problem** — the same
> reordering that was the real result on Rygar. Until one clear exists,
> every "retune stride/window" instruction in `configs/contra.yaml` and
> `DETECTOR_REPAIR_2026-08-26.md` is unfalsifiable. See the CONTRA WALL
> section below.

**6. Progress gate: 7 of 45 verdicts changed, 0 of 45 `passed`
changed.** All seven `SIGNAL UNUSABLE → INCONCLUSIVE`, every one with a
live window under the 187-step floor: `batman_the_video_game` 9,
`blaster_master` 22, `bubble_bobble` 39, `ninja_gaiden_iii` 47,
`contra` 69, `contra_blank` 69, `megaman` 81. `MIN_ASSESSABLE_STEPS =
187` is **calibrated, not chosen** — the largest `steps_to_min_distinct`
over every roster profile whose signal does reach 32 distinct levels —
and documented as a lower bound. The floor gates only the failing
direction, so Rygar's 116-distinct-in-138-steps positive demonstration
still PASSES. Sensitivity shipped with the number (floor → changed):
32→2, 102→7, 187→7, 289→9, 600→12, 1200→13, with `passed` changed 0 at
every floor. Composed over both probes: 17 SOUND, 25 UNUSABLE, 1
INCONCLUSIVE, 2 INAPPLICABLE. Independently re-run at HEAD: all 45 rows
reproduce the banked receipt exactly, verdict and live-step count.

**Purity (Tier 3).** Every measurement above is a hardware surface —
PPU blank-fold counts, PPU scroll odometer, OAM, rendered frames,
render-line counts — plus each profile's own declared lives byte, on
its own start state. No disassembly, no RAM map, no walkthrough, no
recall of any title.

---

## CONTRA WALL 2026-08-27 — EXHIBITION, wall HELD, characterisation BANKED

Full write-up: `docs/research/CONTRA_WALL_2026-08-27.md`. **No tape is
preserved under `docs/receipts/` because no trajectory exceeded 3072** —
see "no tape" below.

**Ledger: EXHIBITION, without exception.** Every number in this section
is Go-Explore search output or instrument measurement. No policy was
trained for this game and no honest-protocol evaluation was run.
Nothing here may be described with "the AI learned", "the AI plays", or
"the AI beat". None of the eight attacks overclaimed on this axis: all
eight returned `beat_3072: false`.

**Emulator** `nes_core` sha256_16 `54366c20d32f71cc`. **ROM**
`roms/Contra (USA).nes` sha256 `26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519`.
**Start state** `roms/Contra (USA)_start.state.bin` sha256
`b99f9be8e0266f6dbe8ac71bc591b0deec08e66e7925707d265965a4aab922c3`.

### The verdict

**The wall HELD. Best verified gx 3072, against a prior of 3072.** Eight
attacks, each varying at least one of {cell definition, restart
distribution, action prior, maximised quantity}, ~1.3 M emulator steps
in mixed accounting units. All eight report `max_gx: 3072`,
`beat_3072: false`, alive at max, `tape_path: null`.

**No tape.** All 18 attack receipt JSONs — raw per-step traces included
— were scanned for any progress-keyed value above 3072: **all clean**.
Zero `BEAT3072_*` / `BREAKOUT_*` / `breakthrough` files exist on disk.
The extraordinary claim was never made, so no proportionate evidence is
owed and `docs/receipts/` correctly gains no Contra tape.

### What the wall physically is (the durable output)

Nine campaigns hit this number without anyone establishing what was
there. **`wall_class`: a screen-locked lethal arena behind an exact hard
camera stop.** Not a freeze, not a scripted input-dead window, not a
corpse, not a timer.

- **The camera stops exactly.** `$0064 = 12`, `$0065 = 0`, one value —
  in 300/300 sampled banked wall cells, all 14 held masks × 8 states,
  524,784 alive steps of a survival-only search, A8's 265,860 steps and
  A7's 285,672. The approach reads `$0064 = 11` with `$0065` sweeping
  178→255, so **3072 is ordinary carry from 3071**, not a wrap artifact.
- **The archive is an absorbing boundary.** Buckets 187–191 hold
  37/47/49/60/115 cells; bucket 192 holds **1,331** (8.2% of the
  archive), and nothing lies above it.
- **The agent is ALIVE and IN CONTROL inside the lock.** 300/300 cells
  restore with `lives ≥ 1`. On an alive-only window the wall's input
  reach **equals or exceeds ordinary open-field play**: 93–317
  input-dependent RAM bytes / 4–95 OAM / 4–17 sprite slots, against an
  alive open-field positive control of 185 / 88 / 9 and a dead negative
  control of 8 / 0 / 0. **20–74 bytes per state are agent-LATCHED** —
  irreversible under some mask, never under NOOP (open-field 39, dead
  control 1).
- **The two progress bytes are the exception: not input-dependent under
  any mask.**
- **Not a timer:** 486 consecutive alive steps (~32 s on one life) and
  524,784 cumulative alive steps moved nothing.
- **Survival is the co-constraint.** Random play survives a median
  26–51 steps. A long safe window exists **only via a searched action
  sequence, never a fixed hold**. **2 of 8 hand-banked wall states are
  input-DEAD** (all 14 masks die at the identical step, 10 and 12) —
  already-committed fatal windows, reproduced independently by A6 at
  steps 9 and 11. Screen wall states for input-liveness before rooting.

**The caution that is the whole point: the dead negative control —
`lives == 0`, game-over animation — moves 766 RAM bytes, MORE than the
wall does.** Byte motion is not evidence of agency and must never be
reported as such. Every liveness claim above is a differential against
that control.

**Anti-vacuity, three reverts in code.** (1) Player-control check →
dead control drops to 2–8 / 0 / 0 / 1. (2) Camera check → alive
open-field control: `$0065` sweeps 0..255, the pair visits ~1,000
states. (3) Camera check under the *same* survival search, seeded 272 px
earlier (gx 2800): identical code and budget, it reports **238 distinct
camera states while alive**, sweeping (10,243) → (11,255) → (12,0) and
stopping there — independently re-deriving the stop point from the
other side. At the wall the same call returns a single state.

### What was ruled out — the campaign's real contribution

From banked wall roots, at ~0.5–1 core-hour per arm:

- **No single held input** releases the camera (15 static masks across
  three attacks).
- **No rhythmic, mashed, lattice-swept, or iid-random pattern** (A3) —
  four action-prior families beyond the static masks.
- **No temporal action sequence found by latch-novelty search** (A8),
  including states with **73 of 74** candidate latched bytes flipped
  simultaneously while alive.
- **No single RAM byte behaves like a release counter** — nothing
  settles one-way under sustained fire against a matched fire-free
  control (A1), nothing accumulates one-way and NOOP-flat (A4), across
  all 2,046 non-progress bytes.
- **Destroying the tracked hardware does not release it** (A2, A6), and
  the "destroyed" reading is a transient multiplexing state, not a kill:
  HP refills within 1–2 steps once fire stops and 0/15 verified
  zero-hits sustain ≥20 zero steps under NOOP.
- **Enriching the cell key does not release it** — not the six
  autonomous zero-page cyclers (A5, which found 11,669 distinct
  augmented cells, **8.8× the un-augmented key**, so the axes are live),
  not entity-slot counters (A7), not raw boss HP (A2), not latch
  popcount (A8).
- **It cannot be waited out** — 524,784 cumulative alive steps.

**The limit of that claim, stated as plainly as the claim.** Five
attacks root from `solve20/archive.pkl`, three from the same session's
`head_wall_*.state`, one from both — and `solve20` is a single run from
the single shared savestate. **No attack ran a fresh from-power-on
search or reached the lock by another route.** What is established is
*"no escape from inside the lock, from banked roots of essentially one
lineage, under input-pattern / cell-key / kill-condition variation at
~0.5–1 core-hour each"*. That is **not** "the lock cannot be passed".
`attackable_by_search` is falsified for the **boundary-resident family**
only.

> **AMENDED 2026-08-27 — the one-lineage limitation is now LIFTED, and
> the wall survives.** Seven independent approaches (four fresh
> from-power-on searches with their own mints/priors/selection
> rules/seeds, two derived life variants, one independent falsifier)
> reach the identical wall: camera `{(12,0)}` across 238 input-live
> states × 14 masks, `px_max` 136, `beat_3072` false over 2,417,912
> worker-steps of fresh search. Approach demonstrably changes the
> arrival configuration (arm purity 0.920 vs chance 0.138, p = 0.0005)
> and the configuration is inert. See *CONTRA LOCK ROUTE A+B
> 2026-08-27* below.

### The clear hook cannot be validated yet, and the gating task is a stage boundary

**`detector_validatable: false`.** `clear_quorum(configs/contra.yaml)`
returns **FIREABLE, roster live, ceiling 2.0, required 2.0 — zero
slack**: ALIVE are `coord` (S_TRANSITION) and `tally` (S_CADENCE); DEAD
are `apu` (`clear.apu_weight = 0`) and all six shelf signals, wired but
unarmed because the profile declares no `solve.clear.signals` block.
`level_key(ram) == ()`, so `is_clear`'s opening test is `() > ()`,
**False always**.

**The critical amendment, measured not assumed.** `tally`'s null fire
rate is **1.00** — 58 of 58 detector checks across five random-play
episodes each terminated at the first `lives $0032` decrement, no
post-death tail (`coord` 0/58, `entity_wipe` 0/58). Fed back in,
`clear_quorum(..., null_rates={'tally': 1.0})` flips to **UNREACHABLE,
ceiling 1.0 < required 2.0, `tally` DEGENERATE** (1.00 ≥
`MAX_NULL_RATE` 0.05). **The shipped 2-of-2 is a 1-of-1 `coord` vote
wearing a corroborator's clothes**, and the hook's advertised safety
property — "only fires when BOTH agree, so an ordinary score tick or
scroll can't fake it" — is **vacuous for this profile**. **Any Contra
null from this hook is VOID, never a miss**: it must not enter a
hit-rate denominator and must never be cited as "searched and found
none".

**Correction to the recorded mechanism.**
`DETECTOR_REPAIR_2026-08-26.md` line 141 — "`tally` has no referent in
this game, so the 2-of-2 vote is unreachable at every stride and
window" — is **WRONG on its stated mechanism** and was itself an
unmeasured assertion about the title, the authored-semantics class
`MECHANISM_COVERAGE_MATRIX_2026-08-25.md` forbids in writing from the
day before. It reached the right ceiling (1) by the opposite mechanism:
`tally` does not fail to fire, it **fails to discriminate**.
`score_tally_windows` is address-free and finds any periodic
anti-correlated byte pair, so an animation counter serves; a
"timer→score conversion" was never its precondition.

**`coord` can fire, and has only ever fired falsely.** Positive control
(deliberately not death-terminated — a false-positive characterisation,
no progress claim): seeds 3/7/11, `coord` fires 12/80 checks in all
three, largest single-step drop 1061/1061/1079 against
`COORD_RESET_DROP_MIN = 300`. **Every observed joint fire is the
game-over/reset arc**, RAM-indistinguishable on these two signals from a
stage load; the lives-drop veto misses it because lives are flat at 0 by
then. That trigger is nevertheless unreachable through the live
pipeline: `Solver.start_lives` is fixed once at `seed()` (line 5561),
so `is_dead` latches permanently at the first life loss and `observe()`
returns "dead" before `is_clear`/`det.push` is reached.

**Witnessed Contra clears: ZERO, by four independent counts.** (a) 19
`solutions/` directories across 18 runs, all empty, 0 files, every tail
row `solutions: 0`; (b) no `sol_*` file matching contra anywhere in the
repo; (c) 772 rows of Contra PPO metrics — `vanilla_ppo_clears` max 0,
`success_rate` max 0, `max_screen` topping out at 10 of the wall's 12;
(d) `current_level $0030` and `boss_defeated $003B` read exactly 0
across 720 restored banked cells, 2,100 live scripted steps, four full
2-life game-over arcs and 13 screen-to-screen transitions — both
measured **non-vacuous**, neither ever witnessed to move.

> **THE GATING TASK IS REACHING A STAGE BOUNDARY, NOT TUNING A
> DETECTOR** — the same reordering that was the real result on Rygar.
> Produce ONE trajectory that survives past the fixed-camera section and
> crosses a stage transition. Until a clear exists the predicate can
> only ever be shown NOT to fire, and every "retune stride/window"
> instruction is unfalsifiable. Conditional sub-task once a clear is
> captured: derive `level_key` from `$0030` / `$003B`.

**`solutions: 0` for Contra is weak-but-real evidence** — the hook
*could* fire, unlike Rygar's UNREACHABLE-by-construction quorum — but
weak-but-real still **cannot separate "never beaten" from "beaten and
not detected"**, because the predicate has never returned a true
positive on any game.

### The wall is not an artifact of the cell definition — four ways

**No collapse** (1,331 distinct cells at the wall bucket in `solve20`,
6,190 in `r1_ortho`, 47,429 in `stage1_v4_localkk`). **Invariant to a 2×
key-resolution change** (`gx_bucket 8 / y_band 16` vs the defaults 16 /
32 — same 3072). **Invariant across four materially different key
compositions.** **Internal falsifier:** the one run whose key genuinely
did collapse — `stage1_baseline_collapsed_cells`, 13 cells over 1.3 M
records — walled at **2816, not 3072**. When the cell definition really
collapses in this codebase it produces a different and lower number.

**The missing axis was tested and is DEAD.** `player_x $0334` spans
25..136 with 99 distinct values across 400 banked wall cells; 40 roots ×
11 actions × 400 held steps, death-terminated, give `px_max = 136`
**exactly, every action, every root** (and 136 again without the
death-tail). In the scrolling region px caps at 128 — the scroll-lock
signature, self-validating the byte with no external map. The camera
stops **and the player stops**; there is no hidden forward gradient the
key was blind to.

### Defects found (secondary yield, all independent of the verdict)

1. **`max_gx_in_max_area` is a binary inside the lock, not a gradient.**
   `solve.area` is unset so `area()` returns literal 0, `max_area` pins
   at 0, and the metric degenerates to the global running max of
   `progress` — frozen at 3072 by construction. Stop grading this wall
   with a running max.
   **AMENDED 2026-08-27 — right defect, wrong mechanism.** Selection
   never reads `max_gx_in_max_area`; it is a *frontier tracker*. The
   count arm reads `best_score` and the deep-frontier arm (40% of picks
   at the default bias) reads only `key[-1]` and picks **uniformly**. At
   the wall `best_score = 3072 + (16 − hp)·2000` with `hp` itself a
   cell-key slot, so it carries zero information beyond the key and its
   one live axis is the transient typed-HP artifact. FIXED under
   `--lock-objective`; see *CONTRA LOCK ROUTE A+B 2026-08-27*.
2. **Five of eleven cell-key fields are structurally constant.** `area`,
   `sect`, `psig` (`room_id` is the constant `(0,)` with `level_key: []`
   and no `area`/`room_sig`, so `_transit` can never fire), `tb`, `kk`.
   That also kills the score's leading term — `score = sect*10000 + gx +
   score_bonus` — leaving gx plus the HP bonus as the only gradient.
   Anti-vacuity: 500 random RAM snapshots give contra 1 distinct
   `room_id`/`area`/`level_key`, against castlevania's 215 and kirby's
   220.
3. **A stage clear would be unrepresentable in the key and invisible in
   the headline.** `progress` hi is `$0064`, the profile's own
   `ram_mapping` calls it a within-stage screen counter that **resets
   each stage**, so an advance collapses gx toward 0, trips the loop-back
   rule, and archives at a low bucket indistinguishable from ordinary
   early-stage cells — while the running max simply does not move.
   **Fix before any post-wall run.**
4. **`score_bonus`'s kill incentive is erased mid-fight.** `_typed_hp`
   returns `sum(live) if live else self._bt_start`, so when no tracked
   type is currently multiplexed in, the bonus reads **exactly 0 —
   identical to "never engaged"**, which traces show happens routinely
   mid-fight. Fix: track a monotone kill count separately from the
   instantaneous HP sum.
5. **A lives-equality liveness gate is measurably insufficient here.**
   A7's first pass reported the camera apparently leaving the lock
   (`{(12,0), (0,0)}`); it built a tripwire, replayed the exact
   trajectory, and traced `(0,0)` to an archive-resumed cell whose
   `entry_lives` was already 0 — a game-over artifact invisible to
   `lives == entry_lives`, because the lives byte holds **flat** across
   the game-over screen. Every `(0,0)` sample had gx exactly 0, so
   `beat_3072` was never at risk, but the diversity metrics were
   contaminated. Pass 2 required `gx == 3072` at entry and every step,
   rejected 91 contaminated entries, and never saw `(0,0)` again. The
   contaminated pass was kept on disk. **Adopt the pattern repo-wide.**

### Corrections landed

- **WITHDRAWN — "NOOP/A/B/up+B/down/down+B all survive the full 400
  steps at the boundary" at `px == 136`.** It does not reproduce:
  **0 of 39** px==136 roots survive 400 NOOP steps (mean 31.7, median
  31, max 78), independently confirmed by A1 (0/183 reach even 120
  steps, forcing a mid-run control pivot NOOP → DOWN), A4 (0/173 survive
  300) and A7 (median 34). The long window lives at **px 55–63** —
  4 of 22 survive the full 400, mean 103.7, and A4's 6 usable roots all
  landed there independently. A real observation attached to the wrong
  sub-population; same class as the `px_max 249` figure below.
- **WITHDRAWN — `runs/play_one_well/contra/wave2_geometry.json`'s
  unattributed `upright_fire px_max: 249`.** Not reproducible with or
  without a post-death tail; its producing script is not in the repo.
- **WITHDRAWN — Contra's banked "162 distinct" odometer figure**, ~94%
  game-over animation.
- **Corrected in place:**
  `runs/contra_wall/A4/probe_release_report.json` carried two stale
  summary strings ("50 diverse banked wall roots"; anti-vacuity "at
  gx~2800") contradicting its own data block (`n_roots_usable: 6`,
  `prewall_gx: 1297`). The submitted narrative was accurate on both; the
  strings now match the data and the correction is recorded in the file.
- **Compute breach, self-reported by A2**: NW=8 pool lanes for ~8 min
  against a `--workers 3` instruction. Does not affect validity.
- **Do not trust cross-build trace replay for this game.** The
  2026-08-01 `traces.pkl` action traces do **not** replay on today's
  core — 16 of 16 wall traces max out at raw progress 1807–1815. Every
  state used in this campaign was minted fresh.

### Next

> **AMENDED 2026-08-27 — items (1) and the objective defect were both
> RUN, and both are CLOSED.** A fresh from-power-on search reaches the
> same wall from seven independent approaches, and a non-flat objective
> inside the lock makes selection genuinely discriminate among the wall
> states without moving the camera. See the *CONTRA LOCK ROUTE A+B
> 2026-08-27* section below; items (2)–(5) here are unchanged and item
> (2) is still outstanding.

Do not spend another probe on input-pattern variation at this boundary.
Ranked: (1) **reach the lock by another route** — a fresh from-power-on
search, since the single biggest weakness in the evidence is that every
attack rooted in one lineage from one savestate; (2) fix defects 3 and 4
above *before* any post-wall run, so a clear would be visible if one
happens; (3) adopt A7's liveness pattern repo-wide; (4) if revisited on
the release hypothesis, the untested residue is narrow and has no
candidate — a sequence-gated release outside A8's latch space, an
address outside the 2 KB CPU RAM window, or an enemy-type-specific
condition; (5) otherwise **re-shelve Contra with this receipt**. Saying
the wall is not attackable by the boundary-resident family is the
correct outcome, and it is recorded so nobody pays for it twice.

**Purity (Tier 3).** No disassembly, no RAM maps, no walkthroughs, no
recall of this title. Every byte's role — death-tail marker, animation
counter, projectile slot, vertical-state byte, input-active flag — was
inferred solely from its own measured time-series shape under our own
inputs, on this profile's own declared addresses and start state. The
screen description in the write-up is read off our own rendered frames.

---

## CONTRA LOCK ROUTE A+B 2026-08-27 — EXHIBITION, wall HELD, both open branches CLOSED

Full write-up: `docs/research/CONTRA_LOCK2_2026-08-27.md`. Predecessor:
`docs/research/CONTRA_WALL_2026-08-27.md` (commit `1954037`). **No tape
is preserved under `docs/receipts/` because no trajectory exceeded
3072.**

**Ledger: EXHIBITION, without exception.** Go-Explore search output and
instrument measurement only. No policy was trained for this game and no
honest-protocol evaluation was run. Nothing here may be described with
"the AI learned", "the AI plays", or "the AI beat".

**Emulator** `nes_core` sha256_16 `54366c20d32f71cc`. **ROM**
`roms/Contra (USA).nes` sha256 `26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519`.
**Start state** `roms/Contra (USA)_start.state.bin` sha256
`b99f9be8e0266f6dbe8ac71bc591b0deec08e66e7925707d265965a4aab922c3`.

### The verdict

**The wall HELD. Best verified gx 3072, against a prior of 3072.** Six
archives were scanned cell by cell — the ancestor plus all five this
campaign produced — and **none holds a cell above gx bucket 192**
(`runs/contra_lock2/_LANDING/land_verify.py`). `max_gx_in_max_area` reads
3072 on **all 87** progress lines the campaign emitted — counted from
`progress.jsonl` alone, because a naive grep returns 174: every run tees
each line to a `.log` twin, which is exactly the 2x double-count this
campaign's own predecessor had to deflate out of the prior "3,030
occurrences" figure. `progress_glitches: 0` everywhere, so the hampel
torn-read screen never dropped a sample and cannot be hiding an advance.

**Both branches the predecessor left explicitly open are now CLOSED.**

- **ROUTE A (approach) — CLOSED, by a third outcome its own decision rule
  did not anticipate.** Approach *does* determine the arrival
  configuration, reproducibly and strongly; the configuration is
  **inert**.
- **ROUTE B (the objective) — CLOSED as a diagnosis.** The named defect
  was real, is fixed, and **selection now demonstrably discriminates
  among the wall states**. The camera still did not move.

### Route A — the state changes, and it changes nothing

Six approaches (four fresh from-power-on searches with their own mints,
priors, selection rules and seeds; two derived life-count variants) plus
an independently designed 7th falsifier, against two prior-lineage
reference populations. **429 wall states collected.**

**The axis is live:** `$00AA` is constant within each arm and differs
between arms (16/19/16/0) at values present in **zero** of the 69
reference states all eight prior attacks used; steps-to-arrive bands do
not overlap (772–816 / 848–862 / 941–1050 / 1371–1520); latch capacity
ranges 122–255 across arms; 8 RAM bytes are within-arm constant and
across-arm different. NN arm purity **0.920** against chance 0.138,
permutation +133.6, **p = 0.0005**, with a sham half/half split of one arm
at **−3.2, p = 0.86** and an open-field-vs-wall positive control at
**+257.3**. Mints verified independent for the landing: six distinct RAM
images differing pairwise by **165–407 bytes**, distinct hashes, seeds,
burst lengths, priors and selection rules.

**And it buys nothing.** Camera `{(12,0)}` across all **238 input-live
states × 14 held masks**, every arm, no exception — while the same probe
gives 20–63 camera pairs on open-field controls. `px_max = 136` exactly
in all 8 populations. A matched pilot varying **only** the root
population gave the prior campaign's own single-lineage roots the
**largest** distinct-cell count of the four (LEGACY 20,436 vs C 19,988,
A 15,160, B 12,475) — fresh configurations open *less* search surface.
`beat_3072` false across **2,417,912 worker-steps** of fresh
from-power-on search and ~1.25 M alive steps at the lock. The 7th
approach reproduces it independently: fresh mint, UCB1-flavoured
selection, burst lengths {80,96,112}, **588,600 worker-steps, 573,950
alive steps, `max_gx_alive` exactly 3072 on every worker.**

**Route A's durable by-product: 45% of wall arrivals are input-DEAD**
(238/429 LIVE), five times the rate the prior 2-of-8 hand sample implied.
The screen is validated three ways on one code path — a dead control at
2 input-dependent bytes / 0 OAM, an open-field control at 6/8 LIVE
(median 442/120), and, strongest because the labels predate it, it flags
exactly `head_wall_0` and `head_wall_1`, the two the characterisation
found by hand, plus one the prior labelling did not separate. LIVE-vs-DEAD
separation is two orders of magnitude (475–588 bytes vs 2–5).
**Screen every root before use, and never gate liveness on
`lives == entry_lives` here** — it failed twice in this campaign alone.

**Anomaly resolved, not published as a result:** `B_coverage` roots read
px 137–144, above the supposedly universal 136, replicating across four
seeds and cold-restoring at progress 3072 with lives 2. **156 of 156 such
witnesses screen input-DEAD** (140 unresponsive, 16 fatal-window;
input-dependent RAM 0–4, OAM 0 — the dead control's own signature), with
median-alive counting down monotonically: one already-committed
commitment sampled a step further along each time. **Corrected statement:
`px_max = 136` holds for every input-LIVE state in every approach.**

### Route B — the defect was real, and the premise needed correcting

**CORRECTION to the predecessor's own framing.** Score at the wall is
**not flat**: it takes exactly **14 values, 3072 → 35072**. But
`best_score = 3072 + (16 − hp)·2000` where `hp` is itself a cell-key slot
— re-derived here and holding for **100% of wall cells in all six
archives** — so it carries **zero information beyond the key**, and the
one axis it expresses is the transient typed-HP multiplex artifact the
characterisation had already ruled out (and `best_score` ratchets, so a
cell that once caught a flicker keeps the elevated score forever).
Worse: **the deep-frontier arm, which does 40% of the picking at the
default `--deep-bias 0.4`, reads no score at all** and sampled the
2,079-cell top band **uniformly**. A score-only edit would therefore have
moved the count-arm weight by 3×10⁻⁵, been invisible to the deep arm, and
**tripped the abort clause while the actual defect went untested.** Route
B was implemented as two coupled edits under one flag: a merit comparator
in the domination test and a merit read in **both** arms.

**Why key enrichment had failed but objective change might not.** The key
controls RESOLUTION (how the budget is partitioned); the score controls
PREFERENCE (which partition gets it). Attack 6 raised resolution 8.8× and
proved the axes live, but under an indifferent preference more resolution
can only **dilute** — every new cell landed in the same uniform top band,
so the per-cell budget fell ~8.8× exactly where concentration was needed.
Key enrichment is structurally a spreading operator; the score is the only
mechanism here that can concentrate, and it had never been changed.

**Four objectives, all lexicographic `(gx, then merit)` with merit in
[0,1) so no merit can reorder two states of different gx:** LEX-YIELD
(B-O4, cell generativity as a burst root, free — it reads bookkeeping
`_assign()` already computed and threw away), LEX-SURVIVAL (B-O1,
squashed alive-in-lock steps), LEX-LATCH (B-O2, control-differenced latch
count against a **paired NOOP continuation on a private probe emulator**,
so a corpse cannot score), LEX-NOVELTY (B-O3, descriptor novelty held
**outside** the cell key so wall cardinality stays A/B-comparable with all
nine prior campaigns). All four: `max_gx` 3072 on every progress line.

### Selection DID discriminate — three independent ways

1. **Offline, on the real banked archive, shuffle-controlled.**
   20,000–30,000 real `select()` calls over the real 1,331 wall cells
   against two pre-registered thresholds (TV ≥ 0.20; top/bottom merit
   decile ≥ 3×). Measured TV 0.15–0.25, decile ratios **2.0×–8.4×**.
   The paired shuffle — same merit values, reassigned across the same
   keys, judged against the **fixed real-merit ordering** — collapses the
   decile ratio to **0.79–1.0×** every time. A TV-only check would have
   missed this: shuffling preserves the weight multiset, so `tv_shuffled`
   lands within a hair of `tv_real` (0.232 vs 0.245). **The decile ratio
   against a fixed ordering is the statistic that separates preference
   from re-weighted noise.**
2. **Distributionally, at the live operating point** (200k real `select()`
   calls, count arm, `--lock-weight 4.0`): inside the lock,
   corr(picks, merit) **0.688**, decile ratio 2.65×, lock share of picks
   0.097 → 0.181. **Outside** the lock, TV **0.0123** against a **0.0111**
   seed-to-seed noise floor — indistinguishable from noise.
3. **Receipt-level, the strongest, re-derived independently at landing**
   (`runs/contra_lock2/_LANDING/dom_check.py`). B-O1 and B-O2 resumed
   `solve20`, where the pre-change rule can only ratchet equal-score
   cells toward FEWER steps. Over the 16,298 cells each shares with the
   ancestor, their archives hold **1,176 and 1,118 equal-score
   replacements carrying MORE steps** — and **100% (1,176/1,176,
   1,118/1,118) sit at gx bucket 192, ZERO outside it.** The comparator
   provably ran live and provably touched nothing outside the lock. The
   statistic is **only defined for a resumed archive** — a fresh run
   shares keys by coincidence and was never domination-compared against
   the ancestor at all; the receipt says so in its own docstring so the
   number is not re-derived later and misread as a leak.

**The abort criterion did NOT fire.** The diagnosed defect was real and
is fixed.

**Inertness was mutated, not inspected** (six vacuous gates have shipped
here): forcing `in_lock_key` true fails 3 tests; dropping the `_in_lock`
check in **both** selection arms — the exact "not inert outside the lock"
defect — fails 4 including the leak probe; removing `observe()`'s mode
scoping fails 1; on B-O2's branch the same mutation fails 6. Forcing
`lock_armed` true fails only its own detector, **correctly**, because
off-safety rests on the `lock_mode == "off"` compare — disclosed by the
work that found it, not papered over. **One exactness caveat:** an armed
run at merit 0.0 is not byte-identical to off, because exact rejection
sampling needs a data-independent ceiling `Wmax = 1 + lock_weight`; what
is asserted and tested is that the **outcome distribution** stays uniform
(TV < 0.08). With the arm off, both arms are byte-identical to the
pre-change path over 400–500 picks including final RNG state.

### Is the lock a game gate or a search artifact?

**GAME GATE — strongly indicated, not proven.** A search artifact is a
failure of the searcher, and all three of its forms have now been tested
and failed: *cannot represent* (falsified four ways by the predecessor —
the one run whose key genuinely collapsed walled **lower**, at 2816),
*cannot reach* (falsified by Route A — seven approaches, provably
different configurations, same wall), and *cannot prefer* (falsified by
Route B — four merit axes, discrimination proved under control, camera
unmoved). **What it does NOT establish:** the runs were short
(~212 s / ~810 s / ~1,290 s genuinely armed), so "the right lever needing
an order of magnitude more armed compute" is unfunded rather than
excluded; four merits is not all merits; and "game gate" means *the
transition is not offered to this search under these conditions* — it is
**not** a claim about what the game contains, which would breach Tier-3
purity.

### Defects found and fixed in this landing

1. **`--lock-objective latch` was a dead choice that read as armed.** It
   parsed, printed `lock_mode: latch` with a non-zero `lock_cells`, and
   changed **not one draw** — 3,000 selections byte-identical to `off`.
   An operator would have read a null as "the objective did not help"
   when nothing ran. **The seventh vacuity, shipped inside the campaign
   whose brief holds the previous six.** Cause: B-O2's implementation is
   on unmerged branch `contra-lock-b-o2` while the shared flag family
   that landed on `main` carried only its *name*. Fixed: `latch` removed
   from the choices, plus `tests/test_lock_objective_roster.py`, a
   **behavioural** roster guard — every declared mode must change the
   pick stream AND leave a merit footprint `off` does not — whose
   non-vacuity direction is a test that runs both probes against a
   deliberately fabricated name and **requires them to come back inert**.
   Mutation-verified three ways, each run and restored: re-adding `latch`
   fails it with the exact diagnosis; a **half-wired** mode (present in
   `select()`, disabled in `observe()`) fails on the observation half
   alone — the case a selection-only probe would have passed; and
   reverting the lock guard itself fails 3 tests across the two sibling
   inertness files, independently reproducing the adjudicated result
   rather than citing it.
2. **`lock_armed_secs` was not armed seconds.** It was
   `round(now − _pin_time)` — time since the *frontier* moved, which
   accrues `--lock-pin-secs` before the objective steers anything. Two
   reports read it as armed time and overstated by **2.4–2.5×** (B-O3's
   "26.5 min armed" is ~21.5; B-O4's "512 s" is ~212). Fixed with a pure
   `lock_clocks()` returning both `lock_pinned_secs` and a true
   `lock_armed_secs`, cross-checked in test against `lock_armed()`, the
   predicate the arms actually gate on. Mutation-verified.
3. **`--lock-weight`'s help text was wrong** — it claimed the deep arm
   ignores the value. That arm accepts with probability
   `(1 + W·merit)/(1 + W)`, so `W` sets its preference strength too.

### Corrections to the four attack reports

- **"Alive by construction because `observe()` resolves death first" is
  overstated.** Contra's `is_dead` is lives-only and that byte was
  measured holding flat through committed fatal windows. The alive claim
  rests on **tape replay and the differential screen**, not code
  ordering: **12 of 12** wall tapes (the three longest from each of the
  four runs, 1,254–1,507 actions) replay from the declared start state on
  a fresh 1-worker pool to gx **exactly 3072 with zero life losses**
  (`runs/contra_lock2/_LANDING/replay_check.py`), and a 14-mask × 32-step
  screen of 30 restored wall states gave 20/30 LIVE at 124–368
  input-dependent bytes against a same-lineage dead control at 2.
- **B-O3's "matched off control" is not matched** — the armed run also
  carried `--gate-opener enumerate --gate-pin-secs -1`. The gate never
  armed (`gate_armed: false` on all 28 lines, 0 injections), so the
  confound is small, but the pair must not be cited as matched.
- **The shipped `--lock-weight` default is 4.0**, not the 1.0 two reports
  state.
- **B-O2's resume failure was its own missing flag** — `solve20` *was*
  banked with `--kill-key` (`key_config.kk = 1`); the lineage guard fired
  correctly on the run's own omission.
- **B-O2's disclosed calibration caveat stands:** `--lock-latch-ceiling 96`
  saturates within 1–2 min of arming, so most of its window ran with less
  merit headroom than its offline test showed the mechanism has.
- **The design's trace-inflation risk materialised, mildly.** Under
  LEX-SURVIVAL the wall's max `best_steps` rose 1,297 → **1,507** (+16%),
  the ratchet inversion the design predicted. B-O2 1,326, B-O3 1,276,
  B-O4 1,314.

### REFUTED — the "everything is checkpoint continuation" side-finding

The 7th approach reported that *"every reached-gx-3072 claim in this whole
body of work is an archive/checkpoint-continuation result, not a single
unbroken life."* **Tested both halves; the generalisation does not hold.**
Its *own* concatenated tapes do die on replay (first life lost at step
73–116, gx capped 486–1266, its own receipt recording
`baseline_alive_at_wall: false`). The **solver's** do not: 24 of 24
adjudicated wall tapes, and 12 of 12 re-verified independently for this
landing, replay end to end from the declared start state to gx exactly
3072 with **zero life losses**. And it is not a property of the sample:
**all 7,954 wall traces across the four archives carry
`root_id: "entrance"`** — full single-session tapes, not chained
fragments. **The defect is in that
approach's own tape bookkeeping, not in the campaign's receipts.** Its
recommendation to screen roots before use is unaffected and stands.

### Unchanged from the predecessor

`solutions: 0` remains **VOID, never a miss**, on this profile —
`level_key: []` makes `is_clear` reduce to `() > ()`, and the shipped
2-of-2 clear vote is a 1-of-1 `coord` vote at a measured `tally` null fire
rate of 1.00 (58/58). It is recorded and never interpreted, and never
enters a denominator. The gating task is still **reaching a stage
boundary, not tuning a detector**.

### Next

1. **Do not fan out on approach** (Route A is closed) and **do not enrich
   the cell key** — it is a spreading operator and re-running it recreates
   attack 6's 8.8× dilution confound, which would make any future negative
   uninterpretable.
2. **The one funded-but-untried variable is armed compute, not another
   merit.** One long, pre-registered run of the objective with the best
   measured live merit dispersion (LEX-YIELD held 0.125–0.857 across 1,743
   wall cells for its whole armed window) at an order of magnitude more
   armed time and `--lock-band > 0`, with its abort stated in advance.
3. **Fix the metric before, not after** (still outstanding): wire
   `current_level $0030` / `boss_defeated $003B` into `solve:` and split a
   monotone kill count out of `score_bonus`.
4. **Reconcile or retire branch `contra-lock-b-o2`.** The roster test now
   prevents its *name* shipping without it.
5. **Otherwise, re-shelve Contra with this receipt.** Two named open
   branches closed for ~2 core-hours of search plus the landing.

**Purity (Tier 3).** No disassembly, no RAM maps, no walkthroughs, no
recall of this title. The lock predicate contains **no game constant**:
`in_lock_key` is `key[0] == max_sect and key[-5] == max_area and
key[-1] >= topgx − band` — every term a property of this run's own search
state, no address, no bucket number, no `3072` — so it means the same
thing on a different game, core build or session, and refuses to mean
anything on a run that has not reached a frontier. It self-disarms the
instant the frontier advances, because that is what resets `_pin_time`.

## RYGAR TRANSITION AXIS 2026-08-27 — EXHIBITION, wall HELD, mechanism LANDED

Full write-up: `docs/research/RYGAR_TRANSITIONS_2026-08-27.md`. Predecessors:
`docs/research/RYGAR_CAMPAIGN_2026-08-26.md`,
`docs/research/ODO_BLANK_AND_GATE_2026-08-26.md`,
`docs/receipts/rygar/clear_predicate_REFUTED.md`. **No tape is preserved
under `docs/receipts/` because no trajectory crossed a stage boundary.**

**Ledger: EXHIBITION, without exception.** Go-Explore search output and
instrument measurement only. No policy was trained for this game and no
honest-protocol evaluation was run. Nothing here may be described with
"the AI learned", "the AI plays", or "the AI beat".

**Emulator** `nes_core` sha256_16 `54366c20d32f71cc`. **ROM**
`roms/Rygar (USA).nes` sha256
`d87a7b3250eb8d6af3b725169a02dab492a10b92dcd03eb44ddce34e1124bbbf`.
**Start state** `roms/Rygar (USA)_start.state.bin` sha256
`9befb1cefd597130b1b3d80fbd77e9363cf8336b53da1542c2e6846b628f0971`.

### The verdict

**The wall HELD. Best verified 4,608 px, against a prior of 4,608 px.** No
stage boundary was crossed. The `($0014,$001C)` area key took **exactly
three values** — `(0,14)`, `(3,16)`, `(0,29)` — across every cell of every
archive this campaign produced, which are exactly the three the prior
room-graph corpus already held. **A fourth value was never observed** in
1,711,525 steps across seven runs and four root classes (~70 min, 2 workers
throughout, contended by a concurrent Contra workflow; 224–458 sps).

**The mechanism is the result.** Before this landing `room_id()` was the
**constant function `(0,)`** on this profile — re-verified at HEAD on three
disjoint RAM images — so `sect` was identically 0, and with it the leading
cell-key slot, the leading score term and the deep-arm filter. Every one of
the failed campaign's 14 hypotheses ranked candidates by `gx` alone. That is
now fixed: `solve.transit_source: blank_run` routes the transit test through
the calibrated blank-fold witness, and `sect` is a live variable.

### The axis is live, and correctly signed

- **It stratifies the archive.** Two independent cold runs put their
  `sect = 1` band at **exactly 1536–4608 px** — the first door and the
  artifact-free ceiling, reproduced to the pixel by an instrument never told
  either number. `best_score` reads 25,002 / 25,081 = `2 × 10000 + gx`.
- **It survives a real revert**, done in an isolated worktree at HEAD:
  8/8 ROM-free wiring tests fail with meaningful `AttributeError` /
  `SystemExit` / structural assertions; the score-site revert fails its 2
  structural tests. **Named limit:** at the default weight the reverted
  score literal computes an identical score, so the real-emulator score
  replay does **not** catch it — only the AST checks do.
- **Its absence is measured on the same steps.** On the banked R1 tape the
  blank-run witness banks **2** novel arrivals where the `room_id()`
  inequality it replaces banks **0**.
- **The value in `key[0]` is `novel`, never a raw count.** On that tape the
  three candidate numbers read **2 / 55 / 4,329**; only the first may enter
  a cell key. The score weight (10,000) is sized so one novel arrival
  outranks the verified frontier **plus** the entire measured ratchet.

### 4,608 is the number — decomposed, not asserted

All four deepest tapes were replayed on fresh pools with the solver's own
replay recipe and **reproduce their filed terminals exactly**, every one
**ALIVE** (longest debounced dead run **2** observations on all four — the
door-blip signature, never the `≥3` debounce, nowhere near a real death's
5,721+ pin). Cut at every area-key change, each tape is: **one** `+1,536` px
ridge traverse, **one** `+3,063..3,072` px corridor traverse (real ground
tops out at **4,608**), then an alternation of **zero-gain** visits to
`(0,29)` and **+55..+64 px** re-anchor blips in `(3,16)`, forever.
**`(0,29)` banked exactly zero pixels in 83 of 83 visits.** The ratchet
totals 4,964 px across the four tapes.

The smoking gun: `far_arrival1_obs2682.state` and `far_arrival2_obs2820.state`
stand **in the same room** and their odometers read **4,608** and **9,280**.
Above 4,608, raw x is not a position on this profile. **Cite 4,608. Never
6,242, never 7,029, never 9,344.**

### The wall

**Routing blindness is REFUTED.** The search now has a transition
dimension, uses it in key, score and selection, and it works — it re-derived
both known doors from cold, to the pixel, in under two minutes, twice. The
frontier still did not move. "The solver could not tell a door from walking"
is no longer available as an explanation.

**The room-graph work's pre-registered falsifier FIRED.** Its rule: a
bounded, death-terminated search from inside `(0,29)` scoring on anything
except odometer x that still finds no fourth area makes `(0,29)` a dead end.
R1/R2/X1 are the strict form — three independently written harnesses, three
non-x selection rules, **359,829 steps, zero fourth areas**, with 149
debounced deaths in R1 alone proving the death path live. D3/run2 is the
loose form (transition-led solver, `gx` still the tiebreak): 274,454 further
steps, also zero. **By its own rule, `(0,29)` is a dead end**; the next
target is a mid-corridor exit.

**Two things are NOT excluded and may not be dropped.** (1) The area key may
be too coarse: `$0014/$001C` take three values in 38,650 cells plus 1.7M
fresh steps, and X1's nametable-hash probe **saturated its 64-hash cap in
both areas within ~20 s**, so it neither confirms nor refutes the concern.
(2) Budget: 1.71M steps is small beside the 232,548 restarts already spent
inside `(0,29)`. Subject to those, the wall is what the campaign originally
classed it — **skill and survival on a route the search can now see**.

**The one live cost of the axis working:** `deep` is a hard `key[0] ==
max_sect` filter, so the instant a cold run reaches `sect = 2` the whole
deep pool becomes cells inside `(0,29)`, where `dx = 0` by construction.
Both cold runs spent ~90% of their budget pinned there.
`--transit-deep-relax` exists for exactly this and **was never exercised on
a clean cold run** — D1 ran at the default 0, and D2, the only run that
passed `--transit-deep-relax 2`, was a contaminated resume whose `max_sect`
never left 1.

### Defects found

- **The `seen` carry is not optional on resume — FIXED.** D2 resumed a
  pre-axis archive: **493 of 1,647 trace records are 7-tuples** with no
  occupied-area set, so restored lineages started empty and re-banked rooms
  they had already occupied — **9 cells at `sect ≥ 1` whose arriving area
  key is the START area**. Error direction is **fabrication**, and `psig`
  cannot backfill it (legacy Rygar records carry `psig == ()`), so
  `check_transit_resume` now refuses such a resume outright, no override.
  Anti-vacuity verified by revert; one test replays the real D2 archive and
  asserts `493 of 1647`.
- **A bare attribute broke seven pre-existing tests — FIXED.** The deep-arm
  filter's `self.transit_deep_relax` raised `AttributeError` against the
  duck-typed Solver stand-ins in **four** test files; only one had been
  updated. Moved to `getattr(..., 0)`, the form the score site already uses.
- **`_sel_lowl_band24` is dead code — RECORDED, not fixed.** `deep` is
  filtered to `key[0] == max_sect`, so `lowl` is `deep` identically and the
  70%-of-the-time branch in `select()` samples the same list as its
  else-branch. **Any claim that the solver "already prefers lower-transit
  cells" is false** and must not be relied on as a safety valve.

### The predicate question is still not answerable

**No predicate was minted and none should be.** `configs/rygar.yaml` still
carries `level_key: []` with no clear and no finale, and its guarding test
is green. Both naive candidates still fabricate wins on Rygar's own deepest
tape — "a transition happened" fires 55 times, a level_key on the
blind-discovered area byte fires 28 and latches TRUE on 83% of observations.
`solutions: 0` on this profile is a compile-time constant, **evidence of
nothing**, never a miss, never in a denominator. The gating task is
unchanged: **one trajectory that crosses a stage boundary.**

### Corrections

- The failed campaign's "no instrument in the pipeline can count rooms" is
  **scoped, not withdrawn** — one could, and the pipeline now does. Its
  frontier conclusion (4,608 px) **stands**, and now stands against a
  room-aware search too.
- **D2's filed `novel_transitions: 0` understates its own instrument** — its
  tape banks 2. The conclusion ("no corpus-novel fourth area") was right;
  the field was wrong.
- **X1's prose calls its 4,608 root reading "an independent reproduction of
  the ceiling".** It is not — the sibling root in the same room reads 9,280.
  X1's JSON records both honestly; the narrative inference does not hold.
- **R1, R2 and X1 banked no replayable tape.** Recorded as a receipt gap.
- `docs/research/ROOM_GRAPH_LEVERAGE_2026-08-27.md` never existed; four
  source files referenced it and now point at this campaign's write-up.

### Next

1. **Run the relax arm** — one cold run at `--transit-deep-relax 1..2`,
   pre-registered. The one knob the design specified and the campaign never
   got a clean run of; ~15 minutes.
2. **Adjudicate the area key** before spending more compute on the null —
   an unsaturating masked nametable hash beside the area key. Do not gate on
   it until calibrated.
3. **Move the target to a mid-corridor exit**, which needs a loadable
   mid-corridor savestate that does not exist yet.
4. **Bank a tape from every falsifier search**, including the nulls.
5. Otherwise **re-shelve Rygar with this receipt**. The expensive question —
   is the solver blind to rooms? — is closed.

**Purity (Tier 3).** Every observable is a hardware surface (2 KB CPU RAM as
an opaque array, the PPU scroll odometer, the PPU blank-fold counter) or a
byte this profile already declared by blind statistical search over its own
rollouts. No disassembly, no RAM map, no walkthrough, no recall of this
title. `solve.area_key` is a config field **distinct from** `solve.level_key`
so no code path can promote a search-derived byte into a clear predicate.
