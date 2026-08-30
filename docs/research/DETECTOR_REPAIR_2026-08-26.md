# Detector Repair — the verdict on the 0-of-38, 2026-08-26

**Falsifier verdict: `INSTRUMENT_BROKEN`, upheld.** The prior round's
*"0 predicates confirmed, 0 of 38 games unblocked"* was **VOID, not FAIL** — every one of
those nulls came from a check that could not have returned a positive. The repaired
instrument was then pointed at the roster, and it moved **one** game into the confirmed
column: **Excitebike**, whose predicate was already shipped and had simply never been
watched firing. **Zero new predicates were written.** The roster still stands at **4 of 45
solver profiles — 5 of 46 counting SMB — able to witness their own clear.**

That last number is the finding. It is not a shortfall against a target we missed; it is a
measurement of what the League can currently score, and §6 says what follows from it.

---

## 1. The falsifier, and what it actually rested on

The falsifier returned `INSTRUMENT_BROKEN` with `null_is_void: true`. It is upheld, and the
reason it is upheld is worth separating from the reason it was *proposed*, because only one
of the two is decisive.

**The weak leg — two controls that did not fire.** Bubble Bobble and Tetris-B are clears
witnessed on a rendered screen (round 69 → 70 with a new stage layout and fresh enemies;
`LINES-000` with the game's own SUCCESS banner) on which the live confluence detector fired
0/6 and 0/1. Two misses on two witnessed clears is the textbook `INSTRUMENT_BROKEN`
condition. On its own it would still be arguable — two games, small n, maybe tuning.

**The decisive leg — a controlled swap.** `runs/clear_control_2026-08-26/cv_odometer_swap.json`
holds everything fixed on Castlevania — same three tapes, same detector, ground truth pinned
so the clear frame is identical in both arms — and varies **only the progress source**:

| Progress source | Tapes hit | Max single-step backward step | `coord` fires |
|---|---:|---:|---:|
| `ram_pair` (shipped) | **3 / 3** | 592 px | 4 per tape |
| `odometer` (the profile's own certified fallback) | **0 / 3** | **4 px** | 0 |

`coord` requires a backward jump of ≥300. Swapping a *working* profile onto the odometer
breaks its clear detection outright. That is not a tuning gap, and it is not stochastic —
`nes_core/src/ppu.rs` `odo_fold_frame` re-anchors **without integrating** across a rendered
scene cut, with a unit test named `rendered_cut_bumps_scene_and_does_not_integrate` asserting
the integral is unchanged across the cut. A stage wipe therefore produces a backward step of
exactly zero, **by design**.

**And the baseline that makes this diagnostic rather than ambiguous:** on SMB the same
instrument scores 5/5 with 0 false positives. The detector is not inert. It works on the one
game it was built for, and its two live signals are both artifacts of that game:

- `coord` — needs a ≥300 backward jump landing ≤200. Counted directly off the 45 solve
  blocks this pass: **25 are `progress: {source: odometer}`**, where the re-anchor makes the
  drop structurally zero, and **10 read a single RAM byte**, which maxes at 255 and cannot
  represent a drop of 300 arithmetically. **That is 35 of 45 on which `coord` cannot fire,
  by design or by arithmetic.** Only the 8 profiles on a 16-bit `{lo, hi}` pair can
  represent the drop at all (the remaining 2 are Ducktales' HUD-tile readout and Punch-Out's
  fight gate). Bubble Bobble's progress spans 98..99. Tetris-B's spans 0..32.
  *(This corrects `CLEAR_GAP_CLOSURE_2026-08-26.md` §3, which gives 26 odometer profiles and
  does not separate the single-byte set; the re-count is 25 + 10, and the conclusion is
  stronger, not weaker.)*
- `tally` — needs a byte decrementing on a cadence while another increments: an
  end-of-level timer-to-score conversion. Contra, Metroid, Kirby, Zelda and Gradius have no
  such tally at any point in play.

With `min_signals=2` the vote needs both. On 35 of 45 profiles the `coord` arm is dead, so
the ceiling is 1 of 2 and the vote is unreachable by construction — not merely untuned. The
detector could not have said yes.

**The instrument's production record outside SMB, stated without decoration:** replayed
against real clears on three games — 1 hit, 2 misses, both misses arithmetic impossibilities
rather than near-misses. Fired in production on non-SMB games exactly twice — Double Dragon
(a combat RAM blip) and Kirby (three fires in 24 seconds, every one an ordinary door load).
Both false, both withdrawn. Fired 9-for-9 falsely on Ghosts 'n Goblins under a weakened
`min_signals=1` rig. **True positives banked in production on a non-SMB game: zero. False
positives banked: two.**

### 1a. The gate that never ran

The census's own spec required three positive controls — Castlevania, Bubble Bobble,
Tetris-B — to gate the fan-out, and **they never ran**. Zero receipt directories existed for
any of the three. By the campaign's own VOID clause the fan-out was ungated. The census said
this about itself, and it was right: *"a 29-for-29 null is exactly what a broken instrument
also produces."* The controls have now run (`runs/clear_control_2026-08-26/`), and they are
what turned a suspicion into the verdict above.

---

## 2. FAIL or VOID — and why the distinction is the whole product

| Bucket | Definition | Answer to "can this game tell us it won?" |
|---|---|---|
| **CONFIRMED** | The hook has been watched firing at a real clear, with a witness independent of the hook. | Yes. |
| **FAIL** | Measured with an instrument *demonstrated capable of returning a positive on that profile*, and no clear was reachable. | No, and we checked properly. |
| **VOID** | Never validly measured — no predicate, or one never shown to fire on anything, or the search never presented the phenomenon. | Unknown; the check could not have said yes. |

The line between FAIL and VOID is one question, and it is the question this week has been
paid for in real defects: **what would this check have reported if the mechanism were
absent?** Where the answer is "exactly the same thing", the result is VOID. A null from an
instrument that cannot return a positive carries zero bits.

**Applied to the prior round: 0 of 38 was VOID, on two independent layers.**

1. **The predicate layer.** `scripts/go_explore_solve.py` `GenericGame.is_clear` opens with
   `if self.level_key(ram) > tuple(start_key)`. With `level_key: []` the key is `()`, and
   `() > ()` is `False` in Python for every RAM array that can exist. Re-counted directly
   from the configs this pass: **45 profiles carry a `solve:` block, 40 ship `level_key: []`
   and 2 omit the key entirely** — 42 for which that branch is an algebraic identity. Only
   three carry a key that can advance (`castlevania`, `bubble_bobble`, `kid_icarus`).
2. **The detector layer.** **Zero** of the census's 29 profiles were wired
   `clear: {mode: confluence}`, so for them the detector was never even constructed. On the
   single profile where an agent did reach it — `ghosts_n_goblins`, under a deliberately
   weakened rig — it fired 9 times out of 9 falsely.

Neither layer produced a measurement about any of those games.

**The number of roster games where a working instrument looked for a clear and honestly
failed to find one is zero.** There is no measured negative result about clear detection
anywhere in this repo outside the confirmed games. Reporting the gap as a failure rate was
reporting the shape of the YAML.

**What VOID does not say.** It is not "nothing was learned". The census's non-detector work —
odometer plateaus, death-discriminator probes, input-invariance tests, start-state mint
defects — is a different instrument and survives this verdict intact; it is catalogued in
`CLEAR_GAP_CLOSURE_2026-08-26.md` §8. What does not survive is any reading of those nulls as
information about whether the games clear.

---

## 3. What the `() > ()` identity cost

Not search time. **Credibility of the record.** The banked `solutions: 0` was a compile-time
constant, and ten documents cited it as a search outcome — several multiplying it against
compute totals so the constant read as corroboration. The corrections landed in `423ef9a`,
annotated and dated, never deleted; 23 files touched. The pattern, by severity:

| Where | What it claimed | What was true |
|---|---|---|
| `CLAIMS.md` (odometer FORGE entry) | "zero solutions on Rygar and Ninja Gaiden" as an **honest negative** | Neither profile has a clear predicate at all. Solved/unsolved was **unmeasured**, not measured-and-negative. The odometer's own 5/5 certification is untouched — that is a different instrument. |
| `contra_reentry_2026-08-10.md` §0/§1 | Bolded prior: "162 GB, 95,161,110 emulator steps, 20.7 h, **zero solutions**" | Reads as "95M steps against a live win test that never fired." No win test was live. The shelving stands on the `gx 3072` frontier pin, reproduced in 9 of 10 runs. |
| `RECEIPTS_INDEX`, `CAPABILITY_REPORT_2026-08-24` | "0 solutions in all 15 odometer-celled probe runs" | One constant reported fifteen times. The "gate-SOUND" half stands; the parenthetical was struck. |
| `zelda_onboarding_2026-08-10.md` | "fabrication tripwire **CLEAN**" | Vacuous. With no path that could bank a solution it reports PASS identically whether the detector is sound or entirely absent. No fabrication check ran. |
| `TOTALITY_BASIS_2026-08-08.md` | League gate: "≥80% of sampled games reach T1 (verified level-1 clear)… zero fabricated clears (detector-verified)" | T1 is unreachable by construction for 40 of 45 profiles, so a cycle run today measures the detector, not the agent — and "zero fabricated clears" is a vacuous pass on the same profiles. |
| `configs/gradius.yaml` | Win hook "wired and hardened" | Wired, but `progress` had been swapped to the odometer on 2026-08-24 by an unrelated onboarding commit, silently disarming `coord` and dropping the vote ceiling to 1 of 2. The declaration was **withdrawn** rather than left dead. |
| `configs/contra.yaml` | "if it never fires, raise stride/window" | Stride and window are not the binding constraint. `tally` has no referent in this game, so the 2-of-2 vote is unreachable at **every** stride and window. |
| 5 further configs | "0 solutions" quoted beside real coverage numbers | Annotated "(expected; empty coverage baseline)", matching four siblings that already did it correctly. |

Two of these — the Zelda tripwire and the TOTALITY_BASIS clause — are **vacuous gates**: they
report PASS whether the mechanism is sound or absent. That is the third and fourth instance
of that exact failure shape in one week, alongside two vacuous gates that shipped and a green
test suite that certified a regression by encoding an object lifetime production never
produces.

**Direction of error is uniformly conservative.** Every distortion is of the form *"we said
the search tried and failed, when we never asked the question."* No banked win is fabricated
by this bug. The only fabricated clears in the repo came through the confluence path and were
already caught and withdrawn on 2026-08-06.

### 3a. What was NOT distorted — checked, not assumed

- **Onboarding.** `SOUND_ADVANCING` / `SOUND_GAME_STOPS` / `CAMERA_STATIC_AGENT_ACTIVE` are
  gate-and-frontier verdicts sourced from the odometer gate and coverage growth.
  `wall_taxonomy._evidence` computes `topo_delta` and `map_delta` from `max_area`/`max_sect`/
  `max_room`/`max_gx` — none routes through `is_clear`. All five wave documents cite zero
  `solutions` figures as evidence. **None of the ~15 onboarding verdicts needs retraction.**
- **RL training and eval.** `trainer.py` and `eval_game.py` take clears from
  `reward_fns[i].episode_success()`; `wave_shaping`'s `is_clear` comes from a completion-total
  delta. Nothing in the learning track touches `GenericGame.is_clear`, so the honest
  sticky-eval numbers (1-1 at 0.76, the 0.65 pooled flagship, the v25/v27/v28 verdicts, the
  recovery-assay ceiling) are untouched. **One exception, found this pass — see §5b.**

---

## 4. What the repaired instrument found

The repair (`e1b9a68`…`c27a3a8`) removed the `SmbGame()` hardcode from
`run_ground_truth_test`, threaded `ctx` into `is_clear` so stateful modes are not
short-circuited to False by construction, called `note_start` at the root so `byte_change`
is not inert, and added game-agnostic signal primitives (`entity_wipe`, `oam_quiesce`,
`scene_cut`, `room_fp_transition`, `input_lock`, `lock_release_novelty`). It was then pointed
at the roster, profile by profile, with anti-vacuity controls demanded of every verdict.

**Roster arithmetic, re-derived from the configs this pass rather than inherited:**

| Bucket | Profiles | Detection |
|---|---:|---|
| **CONFIRMED** — hook watched firing on a real, witnessed clear | **4** | `bubble_bobble`, `castlevania`, `excitebike`, `tetris_b` |
| **VOID** — hook declared, never seen to fire on anything | 4 | `contra`, `contra_blank`, `ducktales`, `kid_icarus` |
| **VOID** — no clear predicate declared at all (`is_clear` is `() > ()`) | 37 | `clear_reachability.py` → `NONE` |
| **FAIL** | **0** | — |
| Total with a `solve:` block | 45 | |

The lint's independent counts — `NONE = 37`, `REACHABLE = 8` — partition exactly across these
buckets (`8 = 4 confirmed + 4 declared-but-unwitnessed`). The table and the tool agree without
being fitted to each other. Adding SMB: **5 of 46 solver-capable configurations can witness
their own clear.**

**What the new signals did when actually run.** This is the part worth carrying forward,
because it is a negative result about the proposed fix. The strategy phase argued that
`coord` is two signals welded together and that splitting out its game-agnostic entity-wipe
half would give ~20 odometer games a vote they cannot currently have. Measured on real
traces, the split helps less than hoped and hurts in a specific way:

- **Where it works.** On Bubble Bobble the entity-wipe half alone fires 4 windows on the
  round-69 clear and 4 in the entire 299-action trace — every one at the clear, zero
  elsewhere. Same shape on Castlevania.
- **Where it does not.** Tetris-B has **0** entity-wipe windows: the split helps
  entity-bearing games and leaves abstract ones needing a different vote entirely.
- **Where it is actively unsafe.** On Bionic Commando `entity_wipe` fired **5 times during
  ordinary forward walking** and only 3 across the flagged anomaly — more on plain play than
  on the event. On Chip 'n Dale both `entity_wipe` and `oam_quiesce` fire on the terminal
  **GAME OVER** screen. On Kid Icarus `oam_quiesce`, `scene_cut(fade)`, `apu_activity` and
  `input_lock` **all** fire on the death → Game-Over transition. On Gradius the blank-fold
  half of `scene_cut` fires on 6 of 6 real deaths at 8–12× background rate. On Metroid
  `room_fp_transition`, run at the profile's own **shipped** calibration, produced 9 false
  novel-room votes against 2 correct door votes. On Castlevania III, uncalibrated, it fired
  84 times against 3 real transitions.

The consistent shape: **every game-agnostic signal that fires on a stage clear also fires on
a death**, because a death and a clear look the same to a RAM-population or scene-cut
observer. That is why the roster's broken death discriminators (7 games) are not a side
issue — they are the blocker underneath the blocker. No clear predicate built on these
signals can be *certified* on a game whose death cannot be detected, no matter how well the
clear half behaves.

---

## 5. The honest count of games that moved

**One. And its predicate was already in the config.**

| | |
|---|---|
| New predicates written into configs | **0** |
| Games moved VOID → CONFIRMED | **1** (`excitebike`) |
| Games moved by acquiring a predicate they did not have | **0** |
| Games moved CONFIRMED → VOID | 0 |
| Hooks **withdrawn** as structurally unfireable | 1 (`gradius`, in `423ef9a`) |

Excitebike moved because it had never been adjudicated — it was not among the census's 29
profiles at all (`grep -ci excitebike` on the census returns **0**), so `predicate: null` was
never its measured status. It shipped a hand-verified `finale:` hook on 2026-08-03 with a
banked solution, and nobody had watched it fire under the repaired instrument or given it
negative controls. Both have now been done, independently at the top level:

- **Positive.** Replaying `runs/excitebike/excitebike_bootstrap/solutions/sol_000`,
  `is_finale` fires at raw frame **4574** of 6176. Rendered witness: FINISH line, checkered
  flag, cleared TIME. A **second, independent** positive path that uses no banked solution at
  all: hold-A-only reaches the same predicate at agent step **1195**.
- **Negative controls — the predicate can fail, and does.** 3000 agent steps of pure NOOP
  idle: 0 fires, `ready_flag` never exceeds 1. 3000 steps of uniform-random mashing
  containing 64 tumble, 21 crash-onset and 1848 overheat-remount frames: 0 fires. All three
  section boundaries inside the winning tape (raw frames 21, 2440, 4461): `ready_flag == 0`,
  so it does not merely shadow the section counter.

### 5a. Why Excitebike survives the `() > ()` bug — and the cheap sweep this implies

`GenericGame` reaches a win through two methods that open with **different comparisons on the
same empty key**:

```
is_clear    level_key(ram) > tuple(start_key)          ->  () > ()   = False, always
is_finale   tuple(start_key) == tuple(f["level_key"])  ->  () == ()  = True, then a live byte test
```

Strict-greater is an algebraic identity on an empty key. Equality is not. So Excitebike's
`is_clear` arm is inert and always has been — the win has never come through it — while its
`finale` arm is a live read of `ram[0x000E]`.

**The general rule, which is the transferable part: any profile carrying a `solve.finale`
block is NOT covered by the census's null and must be re-adjudicated rather than assumed
dead.** Excitebike is the only one today. That makes the sweep free now and valuable the
moment a second finale hook is written.

### 5b. Two defects found by asking what the check reports if the mechanism is absent

**(i) `scripts/replay_sweep.py` asked only half the win test — FIXED HERE.** Line 497 built
its predicate as `lambda ram: bool(game.is_clear(start_key, ram))`, with no `or is_finale`.
Any `finale:`-hooked profile's tape was therefore scored `FAIL: never satisfied is_clear` —
the same null whether the predicate works or does not exist. Excitebike is the only such
profile and has no tape in the sweep corpus, so nothing was mis-scored in practice, but the
defect was live. Now routed through `clear_predicate(game, start_key)`, which ORs both arms
and degrades safely for a duck-typed adapter lacking the second. Covered by five tests,
including an anti-vacuity case that fails when the finale arm is removed (verified by
deleting the mechanism and watching the test go red) and a real-config anchor asserting
Excitebike's `is_clear` arm is the dead identity. `scripts/clear_detect.py` already got this
right; the sweep is now in line with it.

**(ii) Excitebike's RL win metric measures the wrong event — NAMED, NOT FIXED.**
`nes_core/src/rewards.rs` `ExcitebikeReward` pays `completion_bonus`, sets `finished` and
`done`, and returns `episode_success() == true` on:

```rust
if !self.finished && section >= Self::FINAL_SECTION && section > self.start_section {
```

`section >= 3` is **entering** the final section, not crossing the line. Measured on two
independent trajectories:

| Trajectory | Reward fires | `is_finale` fires | Gap |
|---|---:|---:|---:|
| banked `sol_000` | raw frame 4461 | raw frame 4574 | **113 raw frames / 28 agent steps** |
| hold-A-only | agent step 1155 | agent step 1195 | **40 agent steps** |

The episode is terminated at the earlier point, so the policy is never asked to cross the
line. The solver predicate and `episode_success()` are reporting **different events**, and
every historical Excitebike `episode_success` figure is the earlier one. This is a fabricated
win in the learning-track metric — the one exception to §3a's otherwise-clean RL finding.
**Not fixed here:** `rewards.rs` is mid-edit by a concurrent lane (789 lines changed in the
working tree) and a Rust change needs a rebuild; the defect was re-read at `HEAD` to confirm
it is not an artifact of that lane, and is recorded in `configs/excitebike.yaml` beside the
hook it contradicts.

### 5c. One thing to temper

Excitebike is a game where **holding one button wins**. The confirmation is sound and the
negative controls are real, but the banked solution is not evidence that a hard search was
solved and must not be cited as one. Raising the witnessable count from 4 to 5 is a
bookkeeping correction, not a capability result.

---

## 6. WHAT THIS MEANS FOR THE LEAGUE

**Stated plainly: the repaired detector still cannot give most of the roster a clear
predicate, and it is not going to. 41 of 45 profiles have no witnessable clear today, and
for most of them the blocker is not the detector.** Scoring the League on clears means
scoring 41 games at zero for reasons that have nothing to do with how well an agent plays
them.

This is a legitimate finding and it is worth considerably more than a forced predicate would
be. **VOID is not FAIL, and neither is a failure of nerve.** The evidence supports VOID, and
the honest move is to report it and change what the League measures — not to manufacture
predicates that would fire on deaths and door loads to make a gate go green.

### 6a. Why "just build more predicates" does not close this

The 41 VOID profiles do not share one cause, and only one of the causes is a detector
problem:

| Blocker | Games | Would a better detector help? |
|---|---:|---|
| Search/skill wall — the agent never reaches a level end | ~15 | **No.** The predicate is never asked the question. Contra is pinned at the byte-identical `gx 3072` across six campaigns and ~280k–357k cells. |
| Death discriminator broken or absent | 7 | **No, and it blocks certification.** Every game-agnostic clear signal also fires on death (§4); with no death veto, no predicate can be certified even if built. |
| Start-state mint defect — root sits in an attract loop or an input-insensitive state | 3 | **No.** A re-mint job. |
| Purity-blocked — the only known win address is quarantined | 3 | **No.** `zelda`, `zelda_roomfp`, `metroid`. Metroid's win state lives in cartridge PRG-RAM outside `get_ram`'s `$07FF` reach — a hardware-reach problem, not a detection one. |
| Cannot construct — `make_game()` raises before step one | 2 | **No.** `legend_of_zelda`, `tetris_usa`. |
| Spurious declared key — the YAML names the wrong byte | 2 | **No.** Both refuted by direct measurement against their own shipped comments. |
| Healthy search, detection gap only | **5** | **Yes.** `bionic_commando` (30 declared areas), `castlevania_iii` (`max_sect` 16), `metroid_roomfp` (68 room identities, elasticity still rising), `kirby` (real doors), `double_dragon_ii` (frontier still climbing). |

**Five.** That is the honest size of the population a better clear detector could unblock.
Even winning all five takes the roster to 10 of 46. A clear-scored League is not one
detector-repair away from being informative; it is structurally unable to score ~80% of its
own roster, and most of that is capability and start-state work, not detection work.

### 6b. What the League should be scored on instead

The good news is that the substrate already exists and is separately certified. **The
progress observables work where the clear predicates do not.** The PPU scroll odometer is
certified 5/5, and it produces a monotone game-agnostic scalar on games that have no clear
predicate and never will — Ninja Gaiden reached area 9 / best_score 74,783; Rygar reached
5,680 px; Metroid reached 68 distinct room identities with real pan edges. Those are real,
purity-clean capability measurements taken on games scored as zero by any clear-based tier.

Concretely, three changes, in dependency order:

1. **Make the denominator explicit and honest.** A game with no witnessed clear must be
   **excluded from the T1 denominator**, or the tier reported `VOID`. It must never be scored
   as a failed T1 — that is the same error as the "zero solutions" citations in §3, one
   abstraction up. `TOTALITY_BASIS_2026-08-08.md` carries ADDENDUM B saying this; the
   registered gate text itself is still unamended and is what the ledger is scored against.
   **That reconciliation is the actionable item.**
2. **Score depth, not binary clears.** Frontier depth on a certified progress observable is
   available for ~26 games today, is monotone, is game-agnostic, and — critically — is
   already gated by an instrument with a real pass/fail record (`SIGNAL SOUND`,
   `CAMERA_STATIC_AGENT_ACTIVE`, and genuine failures like Kung Fu's skill wall). A
   depth-scored League can rank an agent on Rygar. A clear-scored one cannot say anything
   about Rygar at all.
3. **Keep clears as a bonus tier, and keep the bar high.** The 4 (now 5) confirmed predicates
   stay exactly as they are: witnessed, with an independent rendered witness and negative
   controls. Do not lower that bar to grow the count. The two false clears already banked in
   `runs/detector_gate_20260810/` — Kirby and Double Dragon, both with
   `start_wd == clear_wd == []` — are what a lowered bar produces.

### 6c. The pre-registration that has to change

`TOTALITY_BASIS_2026-08-08.md` registers: *"≥80% of sampled games reach T1 within 1 h
unattended; … zero fabricated clears (detector-verified)."*

Run today, that gate scores near zero for instrument reasons, and its second clause is a
vacuous pass on the same profiles that make the first fail. **It measures the detector, not
the agent, and it cannot fail in the direction it was written to catch.** It should be
reported VOID and re-registered against a progress-depth tier before the next League cycle.
Running it as written would produce exactly the kind of number this document exists to
retract.

---

## 7. Receipts and provenance

- Positive controls, witness frames, replay harnesses: `runs/clear_control_2026-08-26/`
  (including `cv_odometer_swap.json`, the controlled progress-source swap).
- Per-game re-census receipts: `runs/clear_recensus/<game>/`.
- Roster adjudication by name, with cause for all 45: `docs/research/CLEAR_GAP_CLOSURE_2026-08-26.md`.
- Prior round and its self-reported integrity gap: `docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md`.
- Distorted-claim corrections: `423ef9a` (23 files, annotated and dated, nothing deleted).

**Scope of this document's own claims.** The excitebike replay, both negative controls, the
hold-A path, the reward-divergence measurement, the `() > ()` and `() == ()` algebra, and the
45/40/2/3 config counts were all re-derived at the top level rather than inherited from the
census or the adjudication. The signal-behaviour results in §4 are taken from the per-game
receipts under `runs/clear_recensus/` and are attributed there, not re-run here.

**Concurrency note.** This work was done in a shared tree carrying a concurrent lane's
uncommitted reward-dispatch migration (`nes_core/src/rewards.rs`, `src/utils/reward_functions/`,
and a `reward_id:` key added to ~120 configs). Nothing from that lane was staged. The
`rewards.rs` defect in §5b was confirmed against `HEAD`, not against the modified working
copy.
