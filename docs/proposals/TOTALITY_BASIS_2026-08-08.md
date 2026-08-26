# The Totality Basis — which games span "any game"

*2026-08-08. The question: what is the minimal set of games such that
beating them certifies the system can beat (nearly) any NES game in
show mode, and how do we measure true progress toward that?*

*Companion doc: `STRATEGY_2026-08-08.md` sequences this basis into the
30/90-day plan and is the source of truth on current allocations and
figures (e.g. the Castlevania status below).*

## Games are not the unit — mechanism classes are

"Beat N famous games" is the wrong target; games are bundles of
mechanism classes, and totality = covering the classes. The NES action
library decomposes into roughly ten:

| # | Mechanism class | Purest exemplar | Status |
|---|---|---|---|
| 1 | Linear momentum platforming (dense progress axis, frame precision) | SMB1 | **CERTIFIED** (beaten live, 32/32) |
| 2 | Coverage/maze (momentum-vs-coverage taxonomy) | SMB 4-4 / 8-4 | **CERTIFIED** (coverage recipes, live) |
| 3 | Committed-action combat platforming (knockback punish, holds) | Castlevania | blocks 0-2 of ~18 clear; hall (block 3) is the active wall — the old "90%" figure is retired, see `STRATEGY_2026-08-08.md` |
| 4 | Vertical/orthogonal progress (scoring axis ⊥ progress) | CV block-3 hall; Kid Icarus | IN FLIGHT (--ortho arm) |
| 5 | Boss state machines + projectile pressure | Contra; Mega Man | open (adapter partly built) |
| 6 | Pure reactive pattern timing (zero navigation) | Punch-Out!! | open (ASM path exists) |
| 7 | Non-spatial planning (no gx-shaped state at all) | Tetris Type-B | open — cheap, high-signal |
| 8 | Room-graph / item-gated open world (no progress byte) | Metroid | open — the big research lift |
| 9 | Interaction-discovery top-down (triggers, inventory) | Zelda | open (win-predicate wired) |
| 10 | Menu/economy/text (RPGs) | Dragon Warrior | OUT OF SCOPE v1 (declared) |

**The basis set (8 games):** SMB1 ✅, Castlevania, Contra, Mega Man,
Punch-Out, Tetris-B, Metroid, Zelda. Neither {Mario, Metroid, Zelda}
(misses 5/6/7) nor {Contra, SMB3, Tetris} (misses 8/9 — the biggest
class) spans the space. SMB3 adds little over SMB1+coverage; it is
content, not a new axiom.

## The totality instrument — how "true progress" is measured

1. **Capability matrix** (above): a class is CERTIFIED only by a
   receipted show-mode clear of its exemplar. Basis progress = 2/8.
2. **The League**: every cycle, sample K games *stratified-randomly*
   from the 793-boot library, point show mode at them UNATTENDED with
   a fixed budget, and score tiers: T1 = verified level-1 clear,
   T2 = deep run (≥half the game), T3 = credits roll. The trend of
   unattended tier rates over random samples — not hand-picked games —
   is the honest totality number.
3. **"Point at any game" gate** (pre-registered): ≥80% of sampled
   games reach T1 within 1 h unattended; ≥⅓ reach T3 within a
   show-night budget; zero fabricated clears (detector-verified).

   **⚠ See ADDENDUM B (2026-08-26) below before running a League
   cycle against this text — the T1 rate is currently UNDEFINED for
   40 of 45 solve-profiles and the fabrication clause is vacuous on
   those same profiles.**

   *Amendment A1, 2026-08-10 — PROPOSED, not in force.* The three
   lines above are the registered text and stay the text the ledger is
   scored against until the owner signs this amendment into
   `CLAIMS.md`. Three changes, each labelled by direction:

   - **LOOSENS — denominator.** "sampled games" → "scorable sampled
     games", on both the ≥80% T1 clause and the ≥⅓ T3 clause. No human
     adjudicates "scorable" per game and no adjudication happens
     mid-campaign: a game is scorable iff a pre-campaign run of
     `scripts/discover_observables.py` yields a progress or
     room-counter observable that clears that probe's own
     NOOP-flatness and saturation gates, and UNSCORABLE otherwise.
     The owner commits the probe's findings JSON under `docs/receipts/`
     as the freeze receipt — one file, every sampled game and its
     verdict — before the cycle's first League run; it is not editable
     for that cycle. No receipt, or a sampled game absent from it, and
     the original undifferentiated denominator applies for that cycle.
   - **LOOSENS — fabrication clause.** "zero fabricated clears
     (detector-verified)" → "zero unretracted fabricated clears".
     Fabrications are COUNTED, one per claim as it is gated, and that
     running count is what gets reported — never a frequency. A
     retraction counts only if it lands BEFORE the claim is published
     on any surface the CLAIMS.md vocabulary rule binds (README,
     stream overlay, post, talk, commit subject). A fabrication after
     publication is a gate violation, not a retraction, and voids the
     tier rate that carried it for that cycle. "At ledger close" is
     explicitly NOT the deadline: that reading lets any fabrication be
     retracted before it counts, which makes the clause unfalsifiable.
   - **TIGHTENS — publication form.** A tier rate is never printed
     alone. It is worded as a floor rather than an estimate ("T1 was
     reached in no fewer than X of the Y games we pointed at"), and it
     travels with four numbers: how many games the cycle sampled, how
     many of those the freeze receipt marked scorable, how many were
     dropped, and — the number this amendment adds — how many clears
     the per-claim detector gate actually checked, and under how many
     independent detector modalities. Each dropped game is listed with
     its reason. All of it sits next to the rate, not in a footnote,
     because the excluded games are the part a reader most needs to
     audit. The second detector modality this leans on shipped in
     `e5e0957`.

   ### ADDENDUM B, 2026-08-26 — the T1 denominator, and a vacuous half of the fabrication clause

   *Unlike Amendment A1 this is not a proposed loosening; it is a
   correction of fact about what the registered text currently
   measures. The registered text is not edited — it stays the text the
   ledger is scored against — but a cycle run against it today would
   produce a number that is about the instrument, not the agent.*

   **T1 is unreachable by construction for 40 of the 45 profiles that
   carry a `solve:` block.** T1 is defined as a *verified level-1
   clear*, and verification runs through `GenericGame.is_clear`, which
   opens with `if self.level_key(ram) > tuple(start_key)`. Forty of the
   45 profiles ship `level_key: []` (two more omit the key entirely),
   which makes that test `() > ()` — False in Python for every RAM
   state that can exist. Exactly three profiles have a non-empty key
   that can advance (Castlevania, Bubble Bobble, Kid Icarus) and five
   more have any other reachable outcome path at all. A
   stratified-random League sample drawn today therefore scores near
   zero on T1 **for instrument reasons**, and the gate measures the
   detector rather than the agent.

   **A1's scorability criterion does not cover this, and must not be
   read as if it did.** A1 defines scorable as *"a progress or
   room-counter observable that clears the probe's NOOP-flatness and
   saturation gates"* — a claim about a PROGRESS observable. A win
   predicate is a different instrument. A game can be fully scorable
   under A1, with a certified odometer and a healthy frontier, and
   still be structurally unable to report a tier. Ninja Gaiden and
   Rygar are precisely that case today.

   **The fabrication clause is simultaneously a vacuous pass on those
   same profiles.** "Zero fabricated clears (detector-verified)" cannot
   fail where no clear can be banked: it reports PASS identically
   whether the detector is sound or entirely absent. The 2026-08-10
   Zelda receipt shipped exactly this reading — "fabrication tripwire
   CLEAN" on a profile with no reachable predicate — and has been
   corrected.

   **Required before this gate is run:** a game with no witnessed clear
   is UNSCORABLE for T1 and must be excluded from the T1 denominator,
   with the exclusion listed per the A1 publication form; if it is not
   excluded, the cycle's T1 rate is reported VOID rather than as a
   number. The mechanical test is
   `scripts/clear_reachability.py --all`, which classifies every
   profile as REACHABLE (a hook that can fire), NONE (no predicate — 37
   today) or UNFIREABLE (a declared hook that provably cannot fire, and
   a hard refusal at solver launch). Only REACHABLE profiles belong in
   the T1 denominator.

   **Credit where it is due:**
   `docs/research/MECHANISM_COVERAGE_MATRIX_2026-08-25.md` §3b published
   an independent and correct version of this one day before the census
   found it — *"the roster can currently verify a T1 on 5 of 43
   games… A game can be scorable under A1 and still be unable to report
   a tier."* This Basis text was simply never reconciled against it.
   That reconciliation is this addendum. Source:
   `docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md`.

## Dependency-ordered path (each unlocks the next)

1. **CV hall** (class 4) — in flight; a receipted clear certifies 4
   (the hall is that class's exemplar). Class 3 is not certified with
   it: its exemplar is Castlevania entire, and the blocks past the hall
   still have to fall.
2. **Confluence clear-detector v2** (combat-blip + room-transition
   modes still unfixed) — THE league prerequisite: unattended totality
   is unmeasurable without trustworthy generic clear detection. This,
   plus game-over/menu auto-recovery (launcher backlog), makes
   "unattended" real.
3. **Contra** (5) → **Mega Man** (5 hardened + light gating).
4. **Tetris-B** (7) — cheap test that the archive abstraction isn't
   secretly gx-shaped; excellent show content.
5. **Punch-Out** (6) — pure timing axiom; showstopper.
6. **Metroid** (8) — the research lift: solver must build its own
   room graph. **Zelda** (9) rides on it.
7. **League mode**: random-game show nights, tier scoreboard public.

Classes 1-2 took ~6 weeks including all infrastructure. 3-7 reuse that
infrastructure; 8 is the one genuine research unknown remaining.

## The dispatch architecture (the "tree")

The operational form of totality is a diagnosis-dispatch loop, and the
key design decision is WHAT gets classified: not the game, the WALL.
Game-level classification is a lookup table that ends at 793 rows;
wall-level classification generalizes to games nobody profiled.

The system already half-implements this, self-measured (purity-safe —
it diagnoses its own telemetry, never the game's internals):

  telemetry signal            -> wall class          -> mechanism armed
  frontier saturation window  -> momentum wall       -> heuristic-inversion
  cell-key churn/self-similar -> coverage wall       -> coverage recipes
  gx pinned, y-bands starved  -> orthogonal wall     -> --ortho arm
  deep tips die at fixed +N   -> doomed-tip drain    -> barren filter
  room byte + no gradient     -> discrete transition -> derived hold-macros
  coverage saturated at a     -> gated wall          -> CALIBRATING
    boundary, entropy high       (item/key gate)

The gated-wall row ships marked CALIBRATING, and that marker is load
bearing: no solver loop dispatches on it, and this document fixes no
thresholds for it. The distinction it has to draw is against the barren
counter already in the solver — barren increments when a burst off a
cell returns nothing new and resets the instant novelty appears, so it
names a region that never accumulated coverage in the first place. The
gated case is the opposite failure: a region that accumulated all the
coverage there is to accumulate, keeps spending selections on the same
exit, and still never moves the room count. Whether those two separate
cleanly in the telemetry the fleet actually records is an empirical
question, and it is being settled offline against banked archives
rather than asserted here. The numbers belong in that calibration's
receipt; until one exists and is cited, the row stays inert.

The wall library also now carries the Bubble Bobble receipts, which
contributed two *observable*-side classes rather than search-side ones:
the **observable-noise wall** (the y-scratch class — a byte that passes
every naive movement test yet carries no progress; BB's `$0021` does not
enter the learnfun ranking at all over a 30-round chain, `2adb17d`) and
**saturated-counter detector-blindness** (a screen-bound coordinate that
rises then sits dead flat, indistinguishable from real progress until
the saturation gate runs). Both are T1-diagnosis targets, not new arms.

What is missing for totality: (1) SELF-ARMING — several arms are still
opt-in flags a human sets per run; the dispatcher should arm them from
the same telemetry that motivated them, and log the arming as part of
the show ("the AI noticed it was stuck sideways and started climbing").
(2) The NO-ARM branch: when telemetry matches no known wall class,
that IS the research queue — the hall was exactly this. Each new arm
joins the library permanently; capability compounds. The League
measures how complete the arm library is; the basis games are how new
arms get forged.

The learning track is the same pattern one level up: search AI (solver)
manufactures demonstrations and state ladders -> curriculum/distillation
AI turns them into policies -> PPO consolidates -> honest eval gates.
Different AI per phase, dispatched by where the pipeline is stuck.

## The three tiers of unsticking

  T0 REFLEX (in-engine, ms):    telemetry -> known wall class -> arm.
     Hardcoded dispatch, runs inside the solver loop. Exists today.
  T1 DIAGNOSIS (agentic, min):  reflexes failed -> an agent reads the
     run's telemetry/receipts, classifies the wall (or declares it
     novel), tunes/combines existing arms, relaunches. This is what
     the operator + workflows do manually today; productizing it means
     a stuck run auto-emits a diagnosis bundle and invokes the agent
     pipeline — narratable live ("consulting the strategist").
  T2 FORGE (agentic research, hrs): novel wall -> the full workflow
     pattern (recon -> design -> implement -> adversarial review ->
     gated validation) builds a NEW arm that joins T0 permanently.
     The hall/--ortho campaign was T2 run by agents end-to-end.

Two invariants keep it honest at every tier: agents consume only
SELF-MEASURED telemetry (never game internals/disassembly — purity),
and no arm joins T0 without default-off byte-identity + its validation
gate (the same discipline every arm shipped under so far).

## Capability classes (added 2026-08-10)

Two capabilities landed this week that widen the basis without adding
mechanism classes to it. Both are default-inert.

**Controller 2** (`env.set_buttons_p2`, `pool.step_all_2p`; `e5e0957`,
with `step_all` left byte-identical) opens three distinct things:
**cooperation** — the material difference that lets Contra re-enter as
a class-5 lane under elimination-ledger rules (prior stated up front,
what changed named in writing, stopping rule registered before launch,
gate = telemetry not a clear); **competition / self-play**, which is
*post-D1* and subject to the eighth-family rule (no new training family
without a genuinely new, externally-sourced idea); and plainly
**2P-scorable League games**, which cannot be played to completion
one-handed and today would score UNSCORABLE for the wrong reason.

**The learnfun shortlisting instrument** (`2adb17d`) is the League's
auto-profiling path: it ranks all 2048 RAM locations by lexicographic
progress weight over our own banked tapes, so a new game's
scoring-vocabulary freeze starts from a short list instead of 2048
bytes. Two adjudicated limits travel with it. It is **chains-only** — a
single tape does not surface the round counter, only a chain does. And
its **free-running-timer trap** (cadence bytes outrank true progress on
short traces, and the instrument is structurally blind to it) is killed
not by the instrument but by the existing NOOP-flatness gate its
candidates are now routed through. It shortlists; it is never a scorer.
