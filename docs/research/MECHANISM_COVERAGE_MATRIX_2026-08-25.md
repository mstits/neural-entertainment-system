# The Mechanism Coverage Matrix

*2026-08-25/26. The first sweep of every game-agnostic discovery engine
this project has built across the whole onboarded roster. Each mechanism
was validated on one to three games; none had ever been pointed at all of
them. This is that sweep.*

*Companion: `docs/proposals/TOTALITY_BASIS_2026-08-08.md`, which argues
that games are not the unit — mechanism classes are — and that totality
means covering the classes. This document is the empirical half of that
argument: it measures how far each mechanism actually reaches, and what
game property predicts the reach.*

---

## 0. What this answers for the Totality Basis

The Basis names ten mechanism classes and eight exemplar games, and
measures progress by counting CERTIFIED exemplars (2/8). That is the
right ledger for capability. It is silent on a different question that
turns out to matter more right now:

**Of the games we have already onboarded, how many can the system
measure at all?**

The Basis's own League gate assumes this away. Amendment A1 makes the
denominator "scorable sampled games", where scorable means a
`discover_observables.py` probe yields an observable that clears its
gates. This sweep is the first time that denominator has been computed
over the standing roster, and the answer is worse than the framing
implies — not because the games are hard, but because two instruments
are wrong in ways that make sound games look unscorable and unsolvable.

Three numbers carry the finding:

| | |
|---|---|
| Games with a usable progress signal from some mechanism | **38 of 43** |
| Games whose declared lives byte reads 0 at their own root state | **11 of 43** |
| Games that can play but **cannot recognize their own win** | **38 of 43** |

The first number is good news and mostly closed. The second is a single
unfixed defect in one function. The third is the structural problem: a
League of games that cannot score themselves cannot produce a tier rate,
and the Basis's dependency-ordered path already puts "confluence
clear-detector v2" at step 2 and calls it *the* League prerequisite.
This sweep quantifies exactly how large that prerequisite is.

---

## 1. THE MATRIX

Rows are games; columns are the eight mechanisms in the inventory.

**Legend**

| Code | Meaning |
|---|---|
| `A` | **APPLIES** — the mechanism yields a signal the solver can use on this game *today*, with evidence on disk. |
| `N` | **NOT_APPLICABLE** — a *measured* game property falsifies the mechanism's precondition. Never a guess from genre. |
| `X` | **BROKEN** — the mechanism ran and returned a wrong or unusable answer. An instrument fault, not a game finding. |
| `?` | **NOT_TESTED** — no evidence either way. Deliberately common. An incomplete matrix labelled honestly beats a complete one with guesses in it. |

**Columns**

`M1` PPU scroll odometer · `M2` room-graph (masked nametable fingerprint)
· `M3` fight-gate (self/foe HP discriminator + round gate) · `M4`
item-semantics · `M5` death detection · `M6` clear detection · `M7`
deepest-cell diagnostic · `M8` scene-cut detector.

| Game                        | M1 odo   | M2 room  | M3 fight | M4 item  | M5 death | M6 clear | M7 deep  | M8 cut   |
|-----------------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| 1942                        | A        | N        | N        | ?        | X        | ?        | A        | ?        |
| Arkanoid                    | N        | N        | N        | ?        | A        | ?        | A        | ?        |
| Bad Dudes                   | A        | N        | ?        | ?        | X        | ?        | ?        | ?        |
| Batman - The Video Game     | A        | N        | N        | ?        | A        | ?        | ?        | ?        |
| Bionic Commando             | A        | ?        | N        | ?        | A        | ?        | A        | ?        |
| Blaster Master              | A        | ?        | N        | ?        | A        | ?        | ?        | A        |
| Bubble Bobble               | N        | N        | N        | ?        | A        | A        | ?        | ?        |
| Castlevania                 | A        | N        | N        | ?        | A        | A        | ?        | ?        |
| Castlevania III             | A        | ?        | N        | ?        | A        | ?        | ?        | ?        |
| Chip 'n Dale Rescue Rangers | N        | ?        | ?        | ?        | X        | ?        | A        | A        |
| Contra                      | A        | N        | ?        | ?        | A        | ?        | ?        | ?        |
| Darkwing Duck               | A        | N        | N        | ?        | X        | ?        | ?        | ?        |
| Double Dragon               | N        | N        | A        | ?        | A        | X        | ?        | ?        |
| Double Dragon II            | N        | ?        | A        | ?        | A        | ?        | ?        | ?        |
| DuckTales                   | A        | N        | N        | ?        | A        | ?        | ?        | ?        |
| DuckTales 2                 | A        | N        | N        | ?        | X        | ?        | ?        | ?        |
| Excitebike                  | A        | N        | N        | ?        | A        | A        | ?        | ?        |
| Galaga                      | N        | N        | A        | ?        | X        | ?        | A        | ?        |
| Ghosts'n Goblins            | A        | N        | N        | ?        | A        | ?        | A        | A        |
| Gradius                     | A        | N        | N        | ?        | A        | ?        | ?        | ?        |
| Ice Climber                 | N        | N        | X        | ?        | A        | ?        | A        | A        |
| Journey to Silius           | A        | ?        | N        | ?        | X        | ?        | ?        | ?        |
| Kid Icarus                  | N        | ?        | A        | ?        | A        | ?        | ?        | ?        |
| Kirby's Adventure           | N        | A        | N        | ?        | A        | X        | A        | A        |
| Kung Fu                     | N        | N        | X        | ?        | A        | ?        | ?        | ?        |
| Mega Man (USA)              | A        | ?        | N        | ?        | A        | ?        | ?        | ?        |
| Mega Man 2                  | A        | ?        | N        | ?        | A        | ?        | A        | ?        |
| Mega Man 3                  | A        | ?        | ?        | ?        | X        | ?        | ?        | ?        |
| Metroid                     | A        | A        | ?        | ?        | A        | ?        | A        | A        |
| Ninja Gaiden                | A        | ?        | ?        | ?        | X        | ?        | ?        | A        |
| Ninja Gaiden II             | A        | ?        | ?        | ?        | A        | ?        | ?        | ?        |
| Ninja Gaiden III            | A        | ?        | ?        | ?        | A        | ?        | ?        | ?        |
| Paperboy                    | A        | N        | ?        | ?        | X        | ?        | ?        | ?        |
| Power Blade                 | N        | ?        | ?        | ?        | A        | ?        | ?        | ?        |
| Punch-Out!!                 | N        | N        | A        | ?        | A        | ?        | ?        | ?        |
| Rygar                       | A        | ?        | ?        | ?        | A        | ?        | ?        | ?        |
| Shatterhand                 | A        | ?        | ?        | ?        | X        | ?        | ?        | ?        |
| Super C                     | A        | N        | ?        | ?        | A        | ?        | ?        | ?        |
| Tetris (Type-A)             | N        | N        | ?        | ?        | ?        | ?        | ?        | ?        |
| Tetris (Type-B)             | N        | N        | ?        | ?        | A        | ?        | A        | ?        |
| The Legend of Zelda         | N        | A        | ?        | ?        | X        | ?        | A        | ?        |
| Super Mario Bros.           | ?        | ?        | ?        | ?        | A        | A        | A        | ?        |
| Lost Levels                 | ?        | ?        | ?        | ?        | A        | A        | ?        | ?        |

**Cells that need their footnote to be read correctly**

- **Arkanoid M1 = N.** The odometer signal is *real* — paddle position
  leaks into `$2005` — but it is screen-bounded and saturates rather
  than tracking level advance. Kept as a certified demoted fallback; the
  RAM pair `$0016/$0010` is primary. Not an instrument fault.
- **Kirby M1 = N.** Fails at the root state only, where the entrance
  camera clamps at 10 px. The same odometer moves 100-222 px one room
  further in. RAM world-X is wired as primary, correctly.
- **Power Blade M1 = N.** Real, symmetric, bidirectional scroll that
  caps hard at 86 px / 23 distinct values — below the 32-distinct
  gradient threshold. Three independent methods converge on the same
  ~86 gx ceiling with the agent still animating. Signal sound, wall
  early.
- **Zelda M1 = N.** Flat under right/up/down; `left` produces a single
  256 px screen warp at 18 distinct values. That is the discrete
  room-transition signature, not a gradient — which is exactly why
  Zelda is the M2 case.
- **Galaga / Kid Icarus M3 = A.** The HP half only. Both surfaced a
  clean foe-HP byte through all three gates; both had the round-gate
  half return no usable boundary signal, so neither is wired.
- **Kid Icarus M5 = A** despite its declared lives byte reading 0. That
  byte is a deliberately inert monotone stage sentinel, and `is_dead`
  ORs it with an independently verified `player_state` (`$00A6`, reads
  7, death state 0). It is the only one of the eleven zero-reading games
  that has such a fallback.
- **Mega Man (USA) M5 = A**, but only since today: the original
  nomination `$0410` reads 0 at root (verified here by direct RAM peek);
  the hand-corrected `$00A6` reads 2.
- **Power Blade M5 = A** for the same reason and by the same route: the
  tool's own top pick `$0530` reads 0 (verified here), was caught by
  hand during onboarding, and replaced with `$07CA`, which reads 1.
- **Zelda M5 = X** for a *different* failure mode than the other
  eleven. Its lives byte is a health proxy reading 253, and the profile
  states plainly that Zelda's real death — the continue/save menu — was
  never resolved, so `is_dead` fires on first contact and terminates
  roughly one third of 700-step rollouts spuriously. Over-triggering,
  not under-triggering.
- **Bubble Bobble M6 = A** for round-to-round clears only (two observed
  transitions, two banked solutions). Its whole-game finale detector is
  an explicitly unverified tripwire.
- **Double Dragon / Kirby M6 = X.** Not absent — *tried and withdrawn*.
  The generic confluence detector was replayed against real traces on
  both and fired on a combat RAM blip (Double Dragon) and an ordinary
  door load (Kirby). Left unwired deliberately.
- **SMB / Lost Levels.** These run on the bespoke certified SMB adapter,
  not the generic profile shelf, so the generic-mechanism columns are
  honestly `?` for them. Their death and clear detection are certified
  (32/32 levels beaten live).

---

## 2. COVERAGE SUMMARY

| Mechanism | APPLIES | NOT_APPLICABLE | BROKEN | NOT_TESTED | What predicts applicability |
|---|---|---|---|---|---|
| **M1** PPU scroll odometer | **26** | 15 | 0 | 2 | The camera pans continuously under a single held direction. |
| **M2** Room-graph | **3** | 22 | 0 | **18** | Discrete screen-to-screen transitions with a settling camera; a *calibrated* mask exists. |
| **M3** Fight-gate | **5** | 18 | 2 | **18** | Camera static **and** OAM churning — a fixed arena with an active agent. |
| **M4** Item-semantics | **0** | 0 | 0 | **43** | Unknown. Never established. |
| **M5** Death detection | **30** | 0 | **12** | 1 | Whether the declared lives byte reads non-zero at the root state. |
| **M6** Clear detection | **5** | 0 | 2 | **36** | Whether a real clear was ever *observed*, so a predicate could be derived from it. |
| **M7** Deepest-cell diagnostic | 13 | 0 | 0 | 30 | Universal in principle; run ad hoc, on demand. |
| **M8** Scene-cut detector | 7 | 0 | 0 | 36 | Rendering-gated frame discontinuity; universal in principle, swept nowhere. |

### M1 — the odometer is the workhorse, and its dead class is sharp

26 of 41 measured games (63%). The predictor is clean and it is a
*camera* property, not a genre label: does `$2005` move under a held
direction. Every `N` in this column falls into one of four measured
shapes:

- **Fixed single screen** — Bubble Bobble, Galaga, Tetris (both modes),
  Punch-Out (28-46 px of screen shake in both directions), Ice Climber,
  Kung Fu, Double Dragon and Double Dragon II (combat-gated camera lock).
- **Per-room camera clamp** — Chip 'n Dale (~80 px), Kirby (10 px at the
  entrance).
- **Discrete screen warp** — Zelda (one 256 px jump, 18 distinct values).
- **Screen-bounded non-spatial** — Arkanoid, Power Blade's early cap.

Note what is *not* in that list: no game produced a `BROKEN` odometer
cell. The instrument itself is sound everywhere it was pointed. Where
the gate reported FAIL, the cause was either a real game property or —
in four cases — the death byte, discussed in §4.

Signed-axis support matters and is not cosmetic: 1942 counts *down* on
its forward axis, and until the axis was made `-y` the unsigned clamp
pinned progress at 0 (7 cells, best score 0). With the sign, 106 cells
and best score 764.

### M2 — validated twice, demonstrated once, untested eighteen times

Only two games carry a calibrated, receipted room fingerprint
(`docs/receipts/room_fp/zelda.md`, `metroid.md`) and only those two have
a `room_fp:` block in a config. Kirby is the third `A`: its core
primitives were driven directly this sweep — a same-room jiggle and an
archived post-door cell each collapse to exactly one stable masked hash,
with zero overlap between the two rooms — but nothing is wired.

The 18 `?` cells are the real content of this column, and they are not
laziness. M2 is not a flag; it is a per-game calibration pipeline
(capture → mask → mint → replay) with hand-tuned settle and warp
thresholds, and the Zelda receipt records that a mistuned settle mints
garbage room counts. The plausible untested candidates, ranked by how
room-shaped their measured behaviour already looks: **Castlevania III**
(camera-clamped discrete screens with a rebasing position pair, the most
Metroid-like shape on the roster), **Chip 'n Dale** (9 room transitions
in one short probe), **Ninja Gaiden I/II/III** (their profiles already
order cells by `(scene, x)` off an area byte), **Blaster Master**
(unreached on-foot sections), **Bionic Commando** (the overworld
area-select screen), **Rygar**, **Double Dragon II**.

### M3 — the record is better than "1-for-4", and the failure is localized

The inventory records fight-gate as 1-for-4. Across the eight games it
has now been run on, the two halves of the mechanism have very different
records:

| Half | Record | Games |
|---|---|---|
| Self/foe HP discriminator | **5 of 8** | Punch-Out, Double Dragon, Double Dragon II, Galaga, Kid Icarus |
| Round / bout-boundary gate | **0 of 8** | — |

The HP discriminator generalizes better than believed — it cleanly
separated foe HP from self HP on Double Dragon (`$03D6`, surviving a
Gate FH0 veto that rejected 148 other candidates) and on Double Dragon
II (`$0420`, zero vetoes needed), and it even produced a corroborated
candidate on Kid Icarus, which is not an arena game at all. The round
gate has *never* succeeded: it returns `insufficient_probe` (no bout
boundary observed in 24,000 steps) or `no_round_signal` (boundaries
crossed, no byte both stable within a bout and monotone across
boundaries). Only Punch-Out is wired as a progress source, because only
there does foe HP alone carry the whole game.

The two `X` cells are genuine instrument faults and share one shape —
**a candidate that moves on its own is ranked before it is vetoed**:

- **Kung Fu** `$00B1`: attack nets byte-identical (`-1` × 5) across five
  independently seeded reps. A pure-NOOP probe run this sweep shows the
  byte flipping 0 → 135 with zero buttons pressed — an enemy spawn timer.
- **Ice Climber** `$0650`: every candidate's net delta identical across
  all five "randomized" reps, because the probe never escapes a scripted,
  input-ignoring intro window. Voided.

Gate FH0's NOOP-oscillation veto exists and works (it rejected 148
candidates on Double Dragon, 33 on Kid Icarus) but sits *after* the
mirror-sibling bonus in the sort, so a flagged candidate can still rank
first. That ordering is the bug.

### M4 — zero coverage

The item-semantics engine has been run exactly once, on Zelda, where it
correctly failed a negative control. It has never been run on a true
positive, and never on a second game. Nothing in this sweep changes
that. Reported as `?` on all 43 rows because that is what it is.

### M5 and M6 — see §3 and §4, which are the substance of this document.

### M7 / M8 — universal in principle, ad hoc in practice

Both are diagnostics rather than progress engines, and both are
essentially unswept: M7 has receipts on 13 games, M8 on 7. Neither has a
per-game entry point that a League cycle could call. They are the two
cheapest columns to fill and would cost almost nothing to fill, but
filling them unblocks no game by itself.

---

## 3. THE GAPS

### 3a. Games where no mechanism gives a usable progress signal — 5 of 43

These five are *blind*: M1 is dead, no other mechanism is wired, and the
`progress:` key in the profile points at something its own comments
admit is not a progress signal.

| Game | Why blind |
|---|---|
| **Ice Climber** | Odometer flat under five held drivers with the agent active; M3 voided (scripted intro window, plus one-hit enemies with no HP resource); no room structure. The profile's declared source is the dead odometer. Genuinely 0-for-8. |
| **Galaga** | Fixed screen, 599/599 OAM churn, zero scroll under every direction. M3 found a foe-HP byte but the round gate gave it no backing, so it is unwired. `progress: {lo: 0x91}` is a provisional activity counter. |
| **Kung Fu** | Flat both directions with the agent active. M3's candidate is a confirmed false positive. `progress: {lo: 0x94}` is described in its own profile as "the active (if still unhelpful) value" — present only because `GenericGame` requires the key. |
| **Power Blade** | Real scroll that caps at 86 px / 23 distinct within ~90 world-units; three independent methods agree. Below the gradient threshold. |
| **Tetris (Type-A)** | Camera flat *and* agent inert under right and down. Profile is marked BLOCKED: `GenericGame` hard-requires `y` / `level_key` / `lives`, all platformer-shaped, none of which Tetris has. |

Two observations worth carrying forward. First, four of the five are
**class-6/7 shapes from the Totality Basis** — pure reactive timing and
non-spatial planning — which is exactly the region the Basis flags as
open. The blindness is not random; it clusters on the classes we have
not certified. Second, Kung Fu and Tetris-A are blocked partly by
`GenericGame` demanding a platformer-shaped observable set. That is an
adapter constraint, not a game property, and it is cheap to relax.

### 3b. Games that can play but cannot recognize their own win — 38 of 43

This is the structural problem, and it deserves stating plainly.

**Only 5 of 43 games have a win predicate that has ever been shown to
fire on a real clear**: Super Mario Bros. and Lost Levels (bespoke
certified adapter), Castlevania (`level_key: [0x0028]`, three
replay-verified solutions banked), Bubble Bobble (`level_key:
[0x0401, 0x0462]`, two solutions, round-level only), and Excitebike
(`finale: ready_flag == 2`, observed only at and after the finish line).

The other 38 break down as:

| Class | Count | Games |
|---|---|---|
| A predicate is wired but has **never once fired** | 5 | Contra (confluence, never fired across 10+ campaigns and 95M+ steps), Gradius (confluence, proven only against false-fires on deaths, never on a true positive), DuckTales (`score_jump` $500k, non-fakeable, never observed), Kid Icarus (`level_key: [0x0130]`, `max_area` pinned at 0 for a whole smoke run), Tetris Type-B (`byte_change $0050 → 0`) |
| The generic detector was **tried and withdrawn as unsafe** | 2 | Double Dragon (fires on a combat RAM blip: progress 72 → 846 → 88 in five steps, no death, no advance), Kirby (fired 3× in 24 s, every one an ordinary room transition) |
| **Nothing at all** — `level_key: []`, no clear block, no receipt | 31 | everything else |

The independent check is starker than the config audit. Across *every*
onboarding smoke run banked in this repository, exactly **two games ever
emitted a single solution file**: Castlevania (3) and Bubble Bobble (2).
Every other game's `solutions/` directory does not exist.

> **UPDATE 2026-08-26 — the Gradius row was too generous.** It is
> classed above as "a predicate is wired but has never once fired".
> Verified since: Gradius's hook could not have fired. On 2026-08-24
> the League onboarding wave (commit `3e9a502`) swapped that profile
> from `progress: {lo: 0x003E, hi: 0x003F}` to
> `progress: {source: odometer, axis: x}`, and the confluence vote
> needs `coord`, which requires the progress readout to fall by ≥300.
> The odometer cannot fall: `nes_core/src/ppu.rs odo_fold_frame` drops
> its anchor on a mostly-blank frame and re-anchors rather than
> integrating across the discontinuity, so a stage wipe FREEZES the
> integral; `Solver._xram` clamps backward-of-origin to 0; and
> `Solver._assign` rebuilds the detector's rolling window on every
> state restore. Ceiling: 1 of the 2 votes needed. So the class for
> Gradius was **"wired and structurally unfireable"**, and a progress
> change made in an unrelated wave silently disarmed a win condition
> with nothing to notice. The dead declaration has been withdrawn from
> `configs/gradius.yaml` (it is now honestly in the "nothing at all"
> class, count 31 → 32 and the "never fired" class 5 → 4), and
> `scripts/clear_reachability.py` now refuses that combination at
> solver launch so it cannot recur silently.
>
> Contra's row is **unchanged and correct as written**: its progress is
> a 16-bit `{lo, hi}` pair, so `coord`'s required drop is arithmetically
> possible there and "wired but never fired" is the honest class. Ruling
> it structurally dead would require asserting that the game has no
> timer→score tally, which is a fact about the title obtainable only by
> measuring it — the authored-semantics class the purity line forbids.

Two compounding facts make this worse than a missing feature:

1. **`clear_detect.py` has no per-game entry point.** Its CLI exposes
   `--test`, `--runs`, `--discover`, all scoped to curated Super Mario
   Bros. solution traces. There is no `--profile` path, so the
   confluence detector cannot be pointed at an arbitrary onboarded game
   without new plumbing. Several probes in this sweep wanted to run it
   and could not.
2. **It is chicken-and-egg by construction.** A predicate is derived
   from an observed clear; a clear is only *recognized* if a predicate
   exists. `level_key: []` is the honest output of a purity line that
   forbids guessing a stage byte — and it is also a permanent trap
   unless something else can witness the first clear.

Against the Basis's pre-registered gate — "≥80% of sampled games reach
T1 (verified level-1 clear) within 1 h unattended" — the roster can
currently *verify* a T1 on 5 of 43 games. A stratified-random League
sample drawn today would score near zero for instrument reasons, and
Amendment A1's "scorable" denominator would not save it, because A1
scopes scorability to *progress* observables and says nothing about
clear observables. **A game can be scorable under A1 and still be
unable to report a tier.** That is a gap in the amendment, not just in
the code.

---

## 4. THE FALSE-DEATH SWEEP

### The number

**11 of 43 games have a declared lives byte that reads 0 at their own
root state.** Measured first-hand this sweep, one profile at a time,
using the same load order the solver and the progress gate use
(`reset_all()` → `load_worker_state` → one NOOP step):

| Game | Declared lives | Reads at root | Independent death signal? |
|---|---|---|---|
| 1942 | `$00C9` | 0 | none |
| Bad Dudes | `$034F` | 0 | none |
| Chip 'n Dale Rescue Rangers | `$0052` | 0 | none |
| DuckTales 2 | `$000B` | 0 | none |
| Galaga | `$0073` | 0 | none |
| Journey to Silius | `$00B9` | 0 | none |
| Kid Icarus | `$0130` | 0 | **yes** — `player_state $00A6` |
| Mega Man 3 | `$001A` | 0 | none |
| Ninja Gaiden | `$001F` | 0 | none |
| Paperboy | `$000B` | 0 | none |
| Shatterhand | `$0051` | 0 | none |

**10 of the 11 have no `player_state` fallback at all.** For those ten,
death detection is not degraded — it is inverted. The delta computation
folds wraps, so the first 0 → 255 the byte makes reads as a clean
one-step decrement, i.e. a death, at whatever frame it happens to occur.
On 1942 a 900-step trace shows `$00C9` flickering 0 → 255 → 0 → 255 at
steps ~300 and ~510-600 while the plane is demonstrably still flying,
long before the real terminal freeze at step ~750.

### The blast radius is larger than eleven

Four more games are touched by the same defect and are only clean
because a person caught it by hand, one profile at a time:

- **Mega Man (USA)** — the confirmed case. Original nomination `$0410`
  reads 0 (verified here). It froze the frontier at 10 cells; moving to
  `$00A6` (reads 2) took it to 16 cells and progress from 16 px to 24 px.
- **Power Blade** — the tool's own top pick `$0530` (5/5 agreement)
  reads 0 (verified here); caught during onboarding and replaced with
  `$07CA` (reads 1).
- **Darkwing Duck** — ships `lives: 0` as an explicit *unset placeholder*
  (no death detection at all). Running the discovery tool on it fresh
  nominated `$0090`, which reads 0 (verified here). Wiring the tool's own
  recommendation would reproduce the Mega Man failure exactly.
- **Ice Climber** — the wired byte `$002C` reads 1 and is healthy, but
  the same discovery run also nominated `$0091`, which reads 0. The
  landmine is in the candidate pool; only the ranking kept it out.

**Total blast radius: 15 of 43 games.** Eleven live, four defused by
hand.

And the origin case is still live. `progress_signal_gate.py`'s own
module docstring cites Ninja Gaiden's `$001F` as the canonical example
of this trap. **`configs/ninja_gaiden.yaml` still declares
`lives: 0x001F`, and it still reads 0.** The check exists, the
documentation exists, and the profile that inspired both was never
fixed.

### It also silently fails the progress gate

This is the part that connects the defect to the League. Four games —
**Mega Man 3, Ninja Gaiden, Paperboy, Shatterhand** — FAIL
`progress_signal_gate.py` on the lives byte *alone*. Their odometer
signals are excellent:

| Game | Distinct values / 600 steps | Range | Gate verdict | Sole instrument finding |
|---|---|---|---|---|
| Paperboy | 510 | 0..1012 | FAIL | death byte reads 0 |
| Ninja Gaiden | 353 | 0..1335 | FAIL | death byte reads 0 |
| Mega Man 3 | 95 | 0..367 | FAIL | death byte reads 0 |
| Shatterhand | 61 | 0..245 | FAIL | death byte reads 0 |

Under Amendment A1's freeze-receipt procedure, a game is scorable iff a
pre-campaign probe yields an observable clearing its own gates. These
four would be marked **UNSCORABLE**, excluded from the League
denominator, and listed as dropped — on the strength of a bug in a
different observable entirely. The amendment's own tightening clause
requires each dropped game to be listed with its reason; the reason
would be wrong.

### The tool-level cause, confirmed unfixed and worse than described

In `scripts/discover_observables.py`, `find_lives`:

```python
floor = floor and bool((log[:end + 1, i] == 0).any())
```

`log[0, i]` *is* the candidate's start value. When a candidate starts at
0, `reaches_empty` is trivially true regardless of whether the byte ever
legitimately emptied. That is the known half.

The unknown half is that the *second* guard fails the same way. The
stock check is:

```python
stock = stock and (start <= spent <= start - first)
```

With `start = 0`, a single wrap gives `first = -1` and `spent = 1`, so
the test reads `0 <= 1 <= 1` — also trivially true. **Both quality gates
that were supposed to separate a life counter from noise are
simultaneously vacuous at `start == 0`**, and the sort key ranks
`reaches_empty and spends_its_stock` first. A byte sitting at zero that
wraps once is scored as a *perfect* life counter.

There is no `start > 0` rejection anywhere in the function. The fix is
one guard.

### Verdict

This was filed as a per-game annoyance. At 11 live games, 15 touched,
10 with no fallback whatsoever, 4 additionally mis-marked as having an
unusable progress signal, and both quality gates provably vacuous at the
defect's trigger condition — **it is the highest-value single fix
available in this codebase**, and it is a guard clause.

---

## 5. RANKED NEXT MECHANISMS

Ordered by how many currently-blocked games each would unblock, with the
cost and the confidence stated, because a large headcount at low
confidence is not a better buy than a small one at high confidence.

### 1. The zero-start guard in `lives_from_death_drives` — 11 games, hours

Reject any lives candidate whose start value is 0 (or require
`reaches_empty` to be witnessed *after* index 0), then re-nominate
across the 11 affected profiles and re-run the progress gate. Highest
confidence of anything on this list: the mechanism is proven (Mega Man
USA, 10 → 16 cells, 16 px → 24 px on the same day), the cause is read
directly from source, and the affected set is enumerated above with
first-hand root-state readings.

Do it as a tool fix, not eleven config edits. Eleven config edits is what
has happened four times already, and it is why the origin case is still
live.

Secondary, same pass: add the same start-value sanity check to the
`self_hp` nomination (Ice Climber's already collides with its lives
byte), and move Gate FH0's NOOP-oscillation veto *ahead* of the
mirror-sibling bonus in the fight-gate sort — that is the Kung Fu bug,
and it is another ordering fix.

### 2. Generic clear detection with a per-game entry point — 38 games, research

The largest headcount and the Basis's own step-2 prerequisite, but it
cannot be bought outright. Three separable pieces, in dependency order:

1. **A `--profile` entry point for `clear_detect.py`** — days, no
   research. Today the confluence detector is unreachable for any game
   outside the curated SMB traces. Several probes in this sweep wanted
   to run it and could not. This unblocks *measurement*, not detection.
2. **Fix the two known false-positive modes** — combat-blip (Double
   Dragon) and room-transition (Kirby). Both have banked reproducing
   traces, so both are testable today without new runs. This is what
   turns the 2 withdrawn `X` cells back into candidates and makes the 5
   wired-but-unfired predicates trustworthy when they do fire.
3. **A first-clear witness that does not need a predicate** — the actual
   research lift, and the thing that breaks the chicken-and-egg. The
   most promising purity-clean candidates already exist in this tree and
   are unswept: M8's scene-cut detector (a stage clear is a rendering
   discontinuity), the learnfun lexicographic shortlister run over
   *chains* rather than single tapes, and audio-side signals. Note the
   ordering constraint: **on the 11 false-death games the agent dies at
   frame one, so it cannot reach a clear to witness.** Item 1 above is a
   hard prerequisite for this one.

### 3. Room-graph calibration as a repeatable procedure — up to 18 games, weeks

M2 is validated but not *deployable*: it takes a hand-tuned per-game
calibration, which is why 18 cells are `?`. Turning `room_fp_calibrate.py`
into something that mints a mask from a scripted capture without human
threshold-tuning would let the column be swept. Start with Castlevania
III — its measured behaviour (camera-clamped discrete screens, position
pair rebasing at transitions) is the most Metroid-like shape on the
roster and Metroid is already receipted. This is also the Basis's
class-8 lift, the one it names as the single genuine research unknown.

### 4. The fight-gate's round gate — completes 5 games, weeks

The HP half works 5-for-8; the round half is 0-for-8. Both failure
modes are diagnosed: `insufficient_probe` means the randomized
attack-mash probe never spans a real bout boundary (a probe-budget and
probe-shape problem), and `no_round_signal` means no byte was both
stable within a bout and monotone across boundaries (possibly a real
absence). A round gate would let Galaga and Kid Icarus's already-found
foe-HP bytes be wired, and would give Double Dragon and Double Dragon II
a bout-level progress axis. It does not unblock a *blind* game by
itself.

### 5. Relax `GenericGame`'s platformer-shaped observable requirement — 2 games, days

Tetris Type-A is BLOCKED and Kung Fu carries a knowingly-useless
`progress:` key purely because the adapter hard-requires `y`,
`level_key` and `lives`. Making those optional is small, and it is the
only thing standing between Tetris — the Basis's class-7 exemplar,
described there as "cheap, high-signal" — and being onboarded at all.

### 6. Sweep M7 and M8 across the roster — 0 games directly, cheap

Neither unblocks a game on its own, which is why they rank last despite
being the cheapest. They rank at all because M8 is a candidate ingredient
for item 2.3, and a swept M8 column is how we would find out.

---

## Method, receipts, and limits

**What was measured first-hand for this document**

- The complete false-death sweep: every profile's declared lives byte
  read at its own root state, plus `player_state` where declared, in one
  pass. Receipt: `runs/mechanism_matrix/false_death_sweep.json`.
- Corroboration reads on the four hand-defused / near-miss cases
  (Mega Man USA `$0410` vs `$00A6`, Power Blade `$0530` vs `$07CA`,
  Darkwing Duck `$0090`, Ice Climber `$002C` vs `$0091`) and on
  Ninja Gaiden `$001F`. Receipt:
  `runs/mechanism_matrix/false_death_corroboration.json`.
- 22 fresh 600-step odometer gate runs covering the games whose
  mechanism probes were unavailable, including opposite-direction
  retests on every flat axis (Zelda left/up/down, Tetris A/B down,
  Punch-Out left). Receipts: `runs/mechanism_matrix/m1_*.json` / `.log`.
- The clear-detection audit: `level_key` / `clear` / `finale` read from
  all 45 profile files (43 distinct games), cross-checked against every `solutions/`
  directory under `runs/onboard_wave*/`.
- The tool-level cause, read from `scripts/discover_observables.py`
  source rather than inferred from behaviour.

**Counting rule.** 43 rows. The campaign counted 44 profiles;
deduplicating variants that point at the same ROM (`zelda` /
`zelda_roomfp` / `legend_of_zelda`; `metroid` / `metroid_roomfp`) and
splitting Tetris into its two genuinely different modes yields 43
distinct games. Super Mario Bros. and Lost Levels are included because
they are the certified class-1/2 line, and their `?` cells in the
generic columns are themselves a finding: the two games we have actually
beaten do not run on the generic mechanism shelf at all.

**Limits, stated plainly.** 184 of 344 cells are `NOT_TESTED` — 53%.
More than half this matrix is blank, and it is concentrated exactly
where the inventory said each mechanism had only ever been validated on
one to three games: M4 (43/43 untested), M6 (36/43), M8 (36/43), M7
(30/43), M2 (18/43) and M3 (18/43). The two columns that are nearly
complete — M1 (2/43 untested) and M5 (1/43) — are the two that a
profile cannot be onboarded without. Several `?` cells are deprioritizations against a shared
CPU budget rather than genuine unknowns, and where a mechanism was
skipped because its precondition was falsified by measurement, the cell
says `NOT_APPLICABLE` and not `NOT_TESTED`. Where the distinction was
unclear, the cell says `NOT_TESTED`. No cell in this matrix was filled
by reasoning from genre alone.
