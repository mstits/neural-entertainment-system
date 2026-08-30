# The Clear Census — 2026-08-26

**Bottom line: 29 profiles adjudicated, 0 clear predicates certified, 0 configs edited,
0 of the 38-game gap closed.**

> **SUPERSEDED IN PART, 2026-08-26 — see `docs/research/CLEAR_GAP_CLOSURE_2026-08-26.md`.**
> Everything below stands as evidence. The headline number does not: "0 of 38" pooled three
> different results under one zero. The closure pass separates them across the whole
> 45-profile roster — **4 CONFIRMED, 41 VOID, 0 FAIL** — and gives every one of the 41 a
> named cause, including the 16 profiles this census never touched. It also reports the
> positive controls §5 correctly flags as never having run: they have now run, and the
> confluence detector hits 1 of 3 non-SMB games, missing the other two for arithmetic
> reasons that apply to 26 profiles on the roster.

The campaign spec said plainly that "a wired clear predicate is a POSSIBLE output of this
campaign, never a required one." That was the right framing, and it is now the result. The
League's `witnessable` denominator is unchanged at **5 of 43**. Every game this census
touched is on the exclusion list, by name and with cause, and that enumerated list — not a
predicate — is the deliverable.

Tonight's other lesson holds here too. Earlier a projection of "11 games unblocked"
delivered 3, and the honest 3 was worth more than the projected 11. This campaign projected
"a possible predicate" and delivered zero. That zero is worth more than a speculative
predicate wired on the theory it would be right later, because the census also found the
reason zero was the only available answer, and that reason is fixable.

---

## 1. What the census found, in one paragraph

Not one of the 29 profiles failed for the reason the campaign was built to address. The
campaign hypothesised that the missing piece was *the rule that decides what a witness saw*,
and built the one-way-door test (S4) to supply it. S4 ran to a scoring decision on exactly
one profile (Metroid, on two hand-picked doors) out of 29. The other 28 never produced a
candidate transition for it to score. The blocker upstream of the decision rule turned out
to be measurable in the configs themselves, without an emulator:

| Structural fact, measured this pass over all 29 campaign configs | Count |
|---|---|
| `solve.level_key: []` — so `GenericGame.is_clear` reduces to `() > ()` | **28 / 29** |
| No `solve.area` declared — so `GenericGame.area()` returns the literal `0` | **22 / 29** |
| Any `solve.clear` block at all | **1 / 29** (ducktales) |
| Real declared level-key byte that has simply never advanced | **1 / 29** (kid_icarus, `$0130`) |

Both reductions are code facts, not inferences: `scripts/go_explore_solve.py:2572`
(`if self.level_key(ram) > tuple(start_key)`) with `self._clear_mode` unset returns `False`
unconditionally, and `scripts/go_explore_solve.py:2447` returns `0` when `self._area is None`.

Two consequences follow, and they matter more than any individual game's verdict:

1. **`solutions: 0` in the banked archives was never evidence.** Across the runs this census
   inspected — millions of banked steps, up to 443,419 cells on a single profile — the zero
   was an algebraic identity, not a search outcome. Every prior reading of those numbers as
   "the search could not find a clear" was reading a guaranteed constant.
2. **`n_area == 1` was, in 22 of 29 cases, a property of the YAML and not of the game.** Agents
   were instructed to distinguish the two and did; the split is 22 profile-artifact vs. 7 where
   an area/level byte exists and stayed put.

---

## 2. Disposition table (29 profiles, 28 unique ROMs)

Zelda appears twice — `configs/zelda.yaml` and `configs/zelda_roomfp.yaml` are two profiles
over one ROM, adjudicated by separate lanes. Kung Fu and Rygar have mirrored evidence under both
`runs/clear_detection/` and `runs/clear_census/`; each is one verdict.

| # | Game | Disposition | Predicate | Cause, in one line |
|---|---|---|---|---|
| 1 | arkanoid | NOT_WITNESSABLE_FROM_HERE | — | No horizontal position observable exists in the profile, so no closed-loop paddle controller is buildable; 22/22 policies incl. NOOP die in the same 528–656 step band |
| 2 | batman_the_video_game | NOT_WITNESSABLE_FROM_HERE | — | Odometer hard-stalls at 2612px; 7 of 8 held inputs move it zero, `left` costs 6 of 8 lives; reproduced by an independently-seeded prior run at 2611px |
| 3 | ducktales | NOT_WITNESSABLE_FROM_HERE | — | Wired `score_jump` threshold ($500,000) exceeds every measurement this repo holds by 50×; max single-step delta over 111,236 fresh steps is $10,000 |
| 4 | ghosts_n_goblins | NOT_WITNESSABLE_FROM_HERE | — | Odometer plateaus at 3326px across 4 independent runs; the one reachable confluence setting fires 9/9 on non-clearing traces |
| 5 | ice_climber | NOT_WITNESSABLE_FROM_HERE | — | `odo_x` dead; `odo_y` moves only as a fixed −48 step ~774 frames after each death — a death detector, not a climb axis |
| 6 | kid_icarus | NOT_WITNESSABLE_FROM_HERE | — | Real level byte `$0130` never advanced in 113 banked cells or ~21,500 fresh steps; progress *regresses* under every tested strategy incl. NOOP |
| 7 | kungfu | NOT_WITNESSABLE_FROM_HERE | — | `current_floor` never left 0 across ~719,500 combined steps; the one alternative candidate was disqualified by an absence test (jump-only input climbs it identically to attack-only) |
| 8 | mega_man_usa | NOT_WITNESSABLE_FROM_HERE | — | Odometer ceiling x=24, triangulated three ways; frontier cell revisited 17,054 times, barren |
| 9 | megaman | NOT_WITNESSABLE_FROM_HERE | — | `is_clear` structurally unfireable; empirical plateau at gx 4864 corroborates but is not load-bearing |
| 10 | metroid | NOT_WITNESSABLE_FROM_HERE | — | 68 distinct rooms reached with real edges, but win state lives in cartridge PRG-RAM outside `get_ram`'s $07FF reach; the only in-reach addresses are quarantined |
| 11 | ninja_gaiden_ii | NOT_WITNESSABLE_FROM_HERE | — | Full instrument budget reached two confirmed GAME OVER walls (on-screen text), input-invariant across 4 tested actions each |
| 12 | power_blade | NOT_WITNESSABLE_FROM_HERE | — | Odometer input-invariant at x=88 across three independent methods; all four strategies incl. no-directional-input die at the same ~1144-frame mark |
| 13 | rygar | NOT_WITNESSABLE_FROM_HERE | — | No area, no room_sig, no non-empty level_key — no axis attempted at all; 0 clears in 26,702,638 recorded step-observations across 5 archives |
| 14 | shatterhand | NOT_WITNESSABLE_FROM_HERE | — | Reproducible x=1023 wall across two independent seeds; 5 replayed lineages all converge there with no scene cut |
| 15 | super_c | NOT_WITNESSABLE_FROM_HERE | — | Progress byte `$00A4` wraps mod 256 (found behaviourally); with an unwrap-aware score, 25 seeds never exceed 640 |
| 16 | bad_dudes | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | Declared lives reads 0 at root; replacement `$00CD` is a combat/hit-stun oscillator firing ~24–25×/700 death-free steps |
| 17 | darkwing_duck | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | Best candidate `$05E6` *refills* at every respawn — inverted, not merely noisy; 5 re-derived alternatives all oscillate |
| 18 | ducktales_2 | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | `$000B` free-runs: 7–15 "decrements" per 700-step window containing zero deaths; flips 0→1 on the first forward-hold step |
| 19 | journey_to_silius | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | `$0135` fires on movement onset, not death; wiring it collapsed a 774-cell archive to 2 and the frontier from 1269px to 7px |
| 20 | ninja_gaiden | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | Four candidates tried and rejected; `$0386` empties while the odometer keeps climbing (+49px/60 steps) |
| 21 | paperboy | INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | — | Declared byte and best alternative both free-run under pure NOOP on a ~117-step cadence — a shared background clock, not a life stock |
| 22 | galaga | INSTRUMENT_BLOCKED_NO_PROGRESS_SIGNAL | — | Lives, progress and PPU vertical scroll are all byte-identical under `hold_right` vs `hold_A`; scroll jumps exactly +224 at each lives "refill" — a non-interactive presentation loop |
| 23 | double_dragon_ii | NO_TRANSITION_IN_BANK | — | 195/195 cells at `level_key==()`, `area==0`; no scoreable clear-observable exists for an instrument to aim at |
| 24 | mega_man_3 | NO_TRANSITION_IN_BANK | — | 2,969 cells across 4 independent archives, level_key/area constant in 100% |
| 25 | punchout | PURITY_BLOCKED | — | Only knockdown-vs-match-end discriminator is the quarantined `$0001`; the purity-clean candidate self-falsified (see §4) |
| 26 | zelda | PURITY_BLOCKED | — | Ganon-defeated address is disassembly-derived and BLOCKED; room-graph engine cannot tell a dungeon-entry fade from a death fade |
| 27 | zelda_roomfp | PURITY_BLOCKED | — | Same block; 443,419 cells over 4× 90-min/12-worker runs, zero `solutions` increments |
| 28 | double_dragon | SPURIOUS_KEY_DIMENSION | — | The profile's own frontier axis `$00B2` is input-independent deterministic churn (see §4) |
| 29 | 1942 | ONEWAY_TEST_ABSTAIN | — | All 14 declared actions produce strictly positive scroll delta (min +13, max +249) — a forced auto-scroller with no reversibility to test |

**Rollup:**

| Disposition | Count |
|---|---|
| NOT_WITNESSABLE_FROM_HERE | 15 |
| INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR | 6 |
| PURITY_BLOCKED | 3 |
| NO_TRANSITION_IN_BANK | 2 |
| INSTRUMENT_BLOCKED_NO_PROGRESS_SIGNAL | 1 |
| SPURIOUS_KEY_DIMENSION | 1 |
| ONEWAY_TEST_ABSTAIN | 1 |
| **PREDICATE_CERTIFIED** | **0** |

---

## 3. How much of the 38-game gap this closed

**Few. Specifically: none.**

| Metric | Before | After |
|---|---|---|
| Roster games that can detect their own clear | 5 / 43 | 5 / 43 |
| Games given a CONFIRMED predicate by this campaign | — | **0** |
| Configs edited | — | **0** |
| Gap closed | — | **0 of 38 (0.0%)** |

What did change is the *quality of the denominator*. Before tonight, 38 games were an
undifferentiated block labelled `level_key: []`. Now 29 of them carry an adjudicated cause,
banked evidence, and a named next lever, and the block splits into four repair classes
that need four different fixes:

| Repair class | Games | What it actually needs |
|---|---|---|
| No clear axis was ever attempted | 22 (no `solve.area` at all) | Discover an area/room byte behaviourally, then wire it. Prerequisite for everything else. |
| Death discriminator is broken or absent | 6 | A verified life-stock byte. Until then the solver retires healthy lineages and banks corpses. |
| Purity-blocked at the only known address | 3 | Either a blind re-discovery, or an explicit decision to stay blocked. |
| Agent cannot physically reach a level end | ~15 (overlapping) | Search capability, not detection. No predicate helps. |

The classes overlap; a game can need two or three. That overlap is why "wire 38 predicates"
was never a coherent plan, and this census is the receipt for that.

---

## 4. Every false positive the census found

These are the campaign's real yield: fourteen measured, reproduced defects, each caught
behaviourally, several of them in mechanisms that had already passed a static gate or
shipped.

| # | Where | The false positive | How it was caught |
|---|---|---|---|
| 1 | double_dragon `$00B2` (the profile's Go-Explore frontier axis) | Reads as a room counter; is input-independent deterministic churn. The identical sub-sequence `[1,3,1,3]` appears under two completely different input sequences. Fires "advance" on 19/34 steps of an ordinary trace with zero real transitions. | Replayed the archive's own best lineage; it never reproduces its own archived score. Determinism double-replay ruled out replay noise. Archive depth claims (793 cells, "room 15") are inflated by this. |
| 2 | ghosts_n_goblins confluence at `min_signals=1` | Fires 9/9 on non-clearing traces, always at the first possible check (step 19–20), on this ROM's deterministic opening RAM churn. | 9 independently-rooted traces, 14,356 steps, 3 archives. Also established that the shipped default `min_signals=2` is *unreachable by construction* here, not safe — `coord` is STRUCTURALLY_UNAVAILABLE on an odometer-sourced profile, capping the vote at 1. |
| 3 | ninja_gaiden `$0386` | Nominated as lives; empties while the PPU odometer keeps climbing 94→426px. | Wiring it collapsed a Go-Explore smoke from 1096 cells to 24. |
| 4 | journey_to_silius `$0135` | Drops 1→0 within 4 steps of *any* bare RIGHT hold — movement onset, not death. | A/B smoke: 774 cells → 2, frontier 1269px → 7px. |
| 5 | darkwing_duck `$05E6` | Passed every static gate and a 5/5 death-drive agreement, then *refills to its start value at every respawn* — an inversion: it fires on chip damage and misses the actual death. | Extended drives 4–9× past the tool's own 700-step window, 11 trials, >20,000 steps. Five re-derived alternatives all oscillate too. |
| 6 | bad_dudes `$00CD` | Reads 2→0→2 in a single frame-pair mid-attack; ~24–25 fires per 700 death-free steps. | Independently authored 700-step attack-mash schedule, distinct from the banked one. |
| 7 | paperboy `$000B` and `$00B2` | Both free-run under pure NOOP; `$00B2` cycles 4→3→2→1→0→4 with ~117-step spacing, four full cycles per 3000 steps. | 3000-step zero-input probe. Eight ranked candidates share tick boundaries — one background clock, not eight life counters. |
| 8 | ducktales_2 `$000B` | 7–15 "decrements" per 700-step window containing zero deaths; flips 0→1 on the first forward-hold step from a fresh load. | Three fresh 700-step driven windows, each independently confirmed to contain no reload-sized RAM churn. |
| 9 | galaga `$0464` (re-nominated as *fixed* earlier tonight) | Still free-runs: 14 change events in 6000 NOOP steps, and byte-identical under `hold_right` vs `hold_A`. | The extended-idle absence test. This **flipped a same-night "behaviourally verified" conclusion.** |
| 10 | ninja_gaiden_ii `$004C` | The opposite failure: stays flat across a real terminal death, so `is_dead`'s modular window (1 ≤ d ≤ 8) can never fire. | Rendered frames showing GAME OVER while the byte held; explains 15,579 fruitless frontier revisits. |
| 11 | ice_climber `axis: y` (a candidate this census generated and pre-registered) | Proposed as a climb witness; actually moves only as a fixed −48 step ~774 frames after every death. A naive gate fires 2/741 on ordinary play and both fires are death-respawns. | Pre-registered before the confirming runs, then rejected by its own absence test (removing the restart loop collapsed the observed range 0..−1248 → 0..−96). |
| 12 | punchout `opp_health==0 && opp_down==1` | The only purity-clean bout-win candidate. Flips back to false on its own: opponent health refills to 96 about 8 seconds later under pure NOOP, round unchanged. | 300-step NOOP continuation from the exact banked knockdown state. |
| 13 | ducktales `solve.clear.score_jump` threshold (**shipped, live**) | Threshold is 50× the largest value ever measured. Justification ($1,000,000 boss treasure vs $50,000 gem) is untraceable to any measurement in this repo. | 8,030 archived cells + 111,236 fresh steps: max single-step delta $10,000, max per life $28,000. Downgraded to **T2_WIRED_UNCERTIFIED**. |
| 14 | shatterhand — *the census's own harness* | Reusing one `Pool` across replays leaked the odometer accumulator, fabricating `x_end` values that were exact multiples (1023, 2046, 3069, 4092, 5115) of one true value. | Caught by the agent before reporting, root-caused, fixed to fresh-Pool-per-replay, re-verified with a byte-identical determinism double-replay. |

Two instrumentation traps worth propagating to any future replay work:

- `traces.pkl` stores **action-space indices, not button bitmasks**. Feeding indices straight
  into `Pool.step_all` produces a near-inert trajectory that looks like a real null result.
  Two independent agents (punchout, double_dragon) hit this; both caught it themselves.
- `sps` is not a contention indicator on this machine. One run held 375–391 sps while load
  average swung ~14×.

---

## 5. Instrument integrity — the part that must not be buried

The campaign spec made this non-negotiable: three positive controls (Castlevania `$0028`,
Bubble Bobble `$0401`, Tetris-B `$0050`) are adjudicated **first**, the fan-out is **gated**
on them, and if the procedure does not return PREDICATE_CERTIFIED on all three, *every other
result in the campaign is void*.

**The controls were never run.** There is no `runs/clear_detection/castlevania`,
`.../bubble_bobble`, or `.../tetris_b`; no control receipt exists anywhere in the tree; and
the 29 census verdicts all carry timestamps of 07:28–07:58 with no control lane preceding
them. The fan-out ran ungated.

The honest reading, stated without softening:

- **The instrument's ability to return a positive is unverified by this campaign.** The census
  produced a 29-for-29 null, which is exactly the outcome a broken instrument produces, and
  the one control designed to tell those apart did not run.
- **What rescues most of the result is that most of it is not an instrument measurement.** For
  28 of 29 profiles the non-witness is established by a *code fact* — `is_clear` reduces to
  `() > ()` — read from source and independently reproducible without running anything. That
  finding does not depend on the detector working. Similarly, the 6 death-discriminator
  blocks and the 3 purity blocks are established by direct RAM drives and by CLAIMS.md, not
  by the confluence detector.
- **What does not survive is any claim that the detector was exercised and found nothing.** It
  was reached on one profile (ghosts_n_goblins, where it fired 9/9 false) and scored a
  reversibility decision on one other (metroid). Everywhere else it was never reached.

The controls remain the correct first action for whoever picks this up.

### Preflight lane: 0 of 3 fixes landed

The fan-out was also supposed to be gated on three preflight fixes. None of them are in the
tree. `scripts/clear_detect.py`'s most recent commit is `222817c` (01:49, hardening wave 5 —
a different lane), hours before the census ran:

| Preflight item | Status | Evidence |
|---|---|---|
| `clear_detect.py --profile <config>` entry point | **not landed** | `run_ground_truth_test` still hardcodes `game = SmbGame()` (line 975). No non-SMB profile can reach the detector through it. |
| Offline env-handle plumbing for signal 3 | **not landed** | No `--profile` path exists to plumb it through. |
| `trailing_median` leading back-fill | **not landed** | Line 370 still reads `np.full(k - 1, x[0], ...)` — it preserves the impulse it exists to remove. |
| `persist_checks` | pre-existing | Shipped earlier in `20671eb` / `b8f9e4a`, not by this campaign. |
| `room_veto` refusing configuration without a room observable | **not landed** | No `room_veto` symbol exists in `clear_detect.py`. |

The Double Dragon edge-aligned fabrication at `progress_median=5` is therefore still live,
and — independently — Double Dragon's frontier axis is now known to be spurious anyway (§4 #1).

---

## 6. Anti-vacuity

Of the 25 verdicts that report the flag explicitly, 23 reported the seven-clause contract as
satisfied and 2 (mega_man_usa, rygar) reported `false`. Both `false`s are correct: neither
proposed a predicate, so there was nothing for the clauses to bind to, and both said so
rather than filling the template with numbers. Several others reached the same conclusion and
recorded per-clause `N/A_BY_CONSTRUCTION` with reasons instead of defaulting to `false`. The
remaining 4 profiles (kid_icarus, metroid, zelda, zelda_roomfp) carry an
`anti_vacuity_contract` block in the verdict file rather than a top-level boolean. That
refusal to manufacture a number is the contract working — the same instinct that had a
coin-flip gate held back earlier tonight rather than shipped.

Where the discipline paid for itself concretely: galaga (#9 above) flipped a same-night
"verified" conclusion, ice_climber (#11) rejected a candidate this campaign itself generated
and had pre-registered, and shatterhand (#14) caught a fabrication in its own harness before
it reached a verdict file. Several agents also wrote `s7_prereg.json` before `s7_verdict.json`
with the mtime gap on disk (1942: 34s; punchout: 40s; ghosts_n_goblins: 42s).

---

## 7. Proposed config changes — none applied

No agent edited a config; `git status` confirms zero diffs under `configs/`. Two proposals
were emitted as text and are recorded here for a human, not applied:

1. **ducktales** — annotate `solve.clear` as purity-unverified, and stop treating
   `episode_success` as load-bearing for League tier reporting until a boss-treasure pickup is
   actually witnessed. The census attempted to re-derive a data-driven threshold and found it
   impossible: the measured data holds exactly one pickup population ($2k–$10k), with no
   second population to set a boundary against. Declining to invent one was correct.
2. **ice_climber** — a comment recording that `axis: y` was tested and rejected, so a future
   pass does not re-spend the probe.

Neither is a clear predicate. Nothing was wired.

---

## 8. What would actually close the gap

Ordered by evidence, cheapest first. None of these is "wire more predicates."

1. **Run the three positive controls.** Until Castlevania, Bubble Bobble and Tetris-B return
   PREDICATE_CERTIFIED through this procedure, the census's 29-for-29 null and a broken
   instrument are the same observation. This is one agent-hour and it gates everything below.
2. **Land the three preflight fixes.** The detector is unreachable from any non-SMB profile.
   Every future clear-detection result is capped at "we could not run it" until `--profile`
   exists.
3. **Discover area/room bytes for the 22 profiles that never declared one.** This is an
   *observable-discovery* problem, not a detection problem, and it is strictly upstream of
   S2–S5. `discover_observables.py` already does the analogous job for progress/lives; the
   census found no case where the missing piece was the decision rule S4 was built to supply.
4. **Fix the 6 broken death discriminators.** They are not merely inert — journey_to_silius
   and ninja_gaiden demonstrate they actively destroy search (774→2 cells, 1096→24 cells).
   Tonight's oscillation-rejection work in `discover_observables.py` (`9514723`) is aimed at
   exactly this family; darkwing_duck's refill-at-respawn and ninja_gaiden_ii's flat-through-death
   are two new shapes it should be tested against.
5. **Only then** re-run a census. The one-way-door test remains the right mechanism for the
   room-vs-stage discrimination. It has not yet been given a transition to judge.

---

## 9. Gates run before commit

```
.venv/bin/pytest tests/test_profile_configs.py tests/test_clear_detect_ground_truth.py tests/test_confluence_v2.py -q
352 passed, 21 skipped in 4.89s
```

All 29 campaign configs YAML-parsed clean: **29 parsed OK, 0 missing, 0 parse failures.**
`git status configs/` is empty — no config was touched by the census or by this write-up.

Evidence lives under `runs/clear_detection/<game>/` and `runs/clear_census/<game>/`
(31 verdict files, 29 profiles: Kung Fu and Rygar are mirrored across both trees;
`runs/clear_census/1942/` is an empty stub, that verdict lives under `runs/clear_detection/1942/`).

---

## 10. Follow-up landed: the distorted claims are corrected, and a guard now refuses

*Added 2026-08-26, after this census. Every code fact §1–§8 rests on was
re-verified independently at the top level before any document was touched.*

### 10.1 Corrections applied

Each is an annotation, not a deletion — the original text stays, struck or
marked, with the correction dated beside it.

| Document | What was struck | What it actually meant |
|---|---|---|
| `CLAIMS.md` (odometer FORGE entry) | "have produced **zero solutions** on Rygar and Ninja Gaiden … the games remain unsolved" | No clear predicate exists on either profile, so solved/unsolved is UNMEASURED. The 5/5 odometer certification is untouched — that instrument is a *progress* observable. |
| `docs/receipts/games/contra_reentry_2026-08-10.md` §0, §1, r1_ortho verdict | "ten campaigns produced zero solutions"; the bolded "…95,161,110 emulator steps … zero solutions" prior; "52,539 cells, 0 solutions" | The shelving stands on the **gx 3072 frontier pin**, reproduced in nine of ten runs. The solution count was fixed before the first step and is struck from the prior. |
| `docs/research/RECEIPTS_INDEX_2026-08-24.md` | "NG and Rygar are gate-SOUND but **UNSOLVED** … 0 solutions in all 15 probe runs" | One constant reported fifteen times. "gate-SOUND" stands (that gate is the odometer); "UNSOLVED" does not. |
| `docs/research/CAPABILITY_REPORT_2026-08-24.md` | "both UNSOLVED (0 solutions)" | Depth-only results; not scorable either way. |
| `docs/receipts/games/zelda_onboarding_2026-08-10.md` (orchestrator verdict) | "fabrication tripwire CLEAN", twice | A vacuous gate: it reports PASS identically whether the detector is sound or absent. The doc's own §Read-outs already said this correctly and needed no change. |
| `docs/proposals/TOTALITY_BASIS_2026-08-08.md` | — (ADDENDUM B added; registered text deliberately unedited) | T1 is unreachable by construction for 40 of 45 profiles, so a League cycle run today measures the detector, not the agent. A1's "scorable" covers *progress* observables and does not save it. Games with no witnessed clear leave the T1 denominator, or the rate is VOID. |
| `docs/research/MECHANISM_COVERAGE_MATRIX_2026-08-25.md` | Gradius's class, "wired but never fired" | "Wired and structurally unfireable" — see §10.2. The rest of §3b was already correct one day before this census and is the document the others should have been reconciled against. |
| `configs/gradius.yaml` | `clear: {mode: confluence}` | Withdrawn; see §10.2. |
| `configs/contra.yaml` | "if it never fires, raise stride/window" | Struck as unfounded. Status is UNTESTED — see §10.3 for why it is *not* struck as impossible. |
| `configs/{galaga,chip_n_dale_rescue_rangers,double_dragon,blaster_master,ice_climber}.yaml` | bare "0 solutions" in smoke receipts | Annotated "(expected; level_key is the empty coverage baseline)", matching the four sibling configs that already got this right. |

**Two flagged claims were inspected and deliberately left alone**, because an
over-correction is its own defect: `configs/ducktales.yaml` (a live
`score_jump` predicate — its 0 is a real negative) and
`configs/bubble_bobble.yaml` (a non-empty `level_key`; the config's own
diagnosis, that the one-byte key could not see a transition the search kept
making, is exactly right).

### 10.2 A new finding this census did not have: Gradius was disarmed by an unrelated commit

`configs/gradius.yaml`'s win-condition comment described a hook that was
wired and then hardened against a measured false fire — and that comment was
**true when it was written**. On 2026-08-06 the profile's progress was the
RAM pair `{lo: 0x003E, hi: 0x003F}`, which really did reset to 0 on a death;
that is how the false fire at 205 actions was produced, and the lives-drop
veto really did close it.

On **2026-08-24 the League onboarding wave (`3e9a502`) swapped the profile to
`progress: {source: odometer, axis: x}`** for better frontier coverage. That
swap silently disarmed the win condition, and nothing noticed, because a hook
that *never* fires is indistinguishable from one that *has not fired yet*:

* the live streaming detector votes on `tally` + `coord` at `min_signals=2`
  (`audio` and `lock` do not survive into the solver hook);
* `coord` requires the progress readout to fall by ≥ `COORD_RESET_DROP_MIN`
  (300) and land ≤ 200;
* the odometer cannot fall. `nes_core/src/ppu.rs odo_fold_frame` drops its
  anchor on a mostly-blank frame (< 120 rendered lines) and **re-anchors**
  rather than integrating across the discontinuity, so a stage wipe freezes
  the integral instead of rewinding it; `Solver._xram` clamps
  backward-of-origin to 0; and `Solver._assign` rebuilds the detector's ctx —
  and with it its rolling window — on every state restore, so a restore
  cannot smuggle a drop in either.

Ceiling: **1 of the 2 votes needed.** So "54+ min of unattended search that
could never bank a win" was not fixed; it was fixed for eighteen days and
then silently unfixed by a commit about something else. The dead declaration
is withdrawn rather than left in place — advertising a win condition the code
cannot reach is what made the blindness invisible.

### 10.3 Where this census over-claimed, and the correction is narrower than it prescribed

§Strategy asserted that Contra's confluence vote is unreachable because
"`tally` has no referent in this game" — and likewise for Metroid, Kirby,
Zelda. **That is a fact about a title, and it can only be established by
measuring.** Asserting it from recalled knowledge is the authored-semantics
class CLAIMS.md Tier 3 forbids, and the fact that it points in the pessimistic
direction does not exempt it.

The purity-clean statement is narrower and is what shipped: Contra's
`progress` is a 16-bit `{lo, hi}` pair, so `coord`'s required drop is
arithmetically **possible** there, unlike on an odometer-sourced profile. Its
honest status is **UNTESTED**, not impossible. `configs/contra.yaml` now says
that, and the guard below deliberately passes Contra rather than refusing it.

### 10.4 The guard

`scripts/clear_reachability.py` (+ `tests/test_clear_reachability.py`,
`make clear-lint`, wired into `make test`). It answers one question per
profile, with no ROM, no emulator, and no game knowledge:

* **REACHABLE** — some declared hook can fire (8 profiles today).
* **NONE** — no clear machinery declared (37 today). Legal and common, but
  `solutions` is a compile-time 0 and `--want-solutions` is inert.
* **UNFIREABLE / DEGENERATE** — machinery IS declared and provably cannot
  fire, or is trivially true. **Hard refusal**, raised from `Solver.__init__`
  before the pool is built, so a bad profile costs zero emulator seconds.

Rules cover: `level_key: []` with no hook; confluence whose `min_signals`
exceeds the votes that can be cast (odometer/fight_gate progress, or a
single RAM byte that tops out at 255 < 300); `byte_change` with no `addr`;
`score_jump` with a non-positive threshold; an unknown mode; and a `finale`
whose `level_key` arity can never match. `COORD_RESET_DROP_MIN` is read out
of `clear_detect.py` rather than copied, and a rename raises rather than
defaulting.

**The line that would have prevented all ten distorted claims** now prints at
every launch of a predicate-less run, into the log every receipt is written
from:

```
[clear] [configs/rygar.yaml] NO REACHABLE CLEAR PREDICATE — … This run
CANNOT bank a solution: `solutions: 0` is a compile-time constant, not a
search result, and --want-solutions is inert. … Cite it only as "no clear
predicate wired", never as "searched and found none".
```

Every test is written against *what would this report if the mechanism were
absent?* — including a mutation suite that breaks one rule at a time and
requires the verdict to **move**, and `test_the_guard_is_not_a_constant`,
which fails any guard that answers the same thing for every input. That is
the property `adjudicate_soak._check_budget` and the Zelda "fabrication
tripwire" both lacked.

### 10.5 A live defect found while tracing consumers, fixed

`scripts/replay_sweep.py` computed `start_key = tuple(spec.start_wd) if
spec.start_wd else None`. `start_wd` is `Optional[list]`, and its two falsy
values mean opposite things: `None` = "no key was recorded", `[]` = "a key
was recorded and it is empty" — which is what **every `level_key: []`
profile banks**. The sweep therefore reported `ERROR: "start_wd not
recorded"` for tapes that had recorded it; `evaluate_gate` counts ERROR as a
failure and never as a skip; and the handler written for exactly this shape
(`clear_wd == start_wd` → UNSCORABLE) was unreachable, because the ERROR
branch `continue`s first. Reproduced on both
`runs/detector_gate_20260810/{kirby,double_dragon}` tapes. Fixed with `is
not None`, extracted to a named `resolve_start_key` so it is testable, and
covered by five tests including one anchored to the two real tapes.

### 10.6 Two test files were pinned to the roster instead of the property

`test_the_shipped_confluence_profiles_stay_on_the_default_path` (in both
`tests/test_confluence_v2.py` and `tests/test_detector_v3.py`) and
`test_the_shipped_profiles_warm_up_inside_a_single_burst` hardcoded
`("contra.yaml", "gradius.yaml")` and asserted each still **declared** the
hook — never that it could **fire**. They stayed green throughout the
eighteen days Gradius's hook was inert, and would have gone red for the
*correct* withdrawal of it. All three now discover confluence profiles by
globbing, assert the set is non-empty so they cannot pass vacuously, and
additionally assert reachability.
