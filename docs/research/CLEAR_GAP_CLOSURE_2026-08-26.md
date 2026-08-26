# The Clear Gap — Closure Pass, 2026-08-26

**45 solver profiles carry a `solve:` block. Four of them have a clear predicate that has
been watched firing on a real, rendered clear. The other 41 do not — and every one of those
41 nulls is VOID. Not one is a FAIL.**

The previous round's headline was *"0 predicates, 0 of 38 closed."* That number was not
false. It was meaningless, because it pooled three different things under one zero: games
with a working predicate, games measured by an instrument that could have said yes and
didn't, and games where nothing was ever measured at all. This pass separates them and
counts each by name.

The short version of the correction: **the third bucket is the whole roster.** After the
census, the stale triage, the eight-profile sweep and three independent adjudication
rounds, the number of roster games where a working instrument looked for a clear and
honestly failed to find one is **zero**. There is no measured negative result about clear
detection anywhere in this repo outside the four confirmed games. Reporting the gap as a
failure rate was reporting the shape of the configs.

---

## 0. The three buckets, defined before anything is counted

Definitions first, so the counting cannot be argued backwards from the numbers.

| Bucket | Definition | The question it answers |
|---|---|---|
| **CONFIRMED** | The profile declares a clear hook, and that hook has been observed firing at a real clear — with an independent witness that does not depend on the hook itself (a rendered frame, or a replay-verified chain across independent runs). | "Can this game tell us it won?" — **yes**. |
| **FAIL** | The profile was measured with an instrument *demonstrated capable of returning a positive on that profile*, and no clear was reachable. | "Can this game tell us it won?" — **no, and we checked properly**. |
| **VOID** | Never validly measured. Either no predicate exists to fire, or one exists but has never been shown to fire on anything, or the search never presented the detector with the phenomenon. | "Can this game tell us it won?" — **unknown; the check could not have said yes**. |

The line between FAIL and VOID is one question, and it is the question this whole week has
been paid for in real defects: **what would this check have reported if the mechanism had
been present?** Where the answer is "exactly the same thing", the result is VOID. A null
from an instrument that cannot return a positive carries zero bits.

Two vacuous gates shipped this week because nothing asked that question. A behavioural gate
that caught its target 56% of the time was held back rather than shipped. Green tests
certified a regression by encoding an object lifetime production never produces. The same
discipline applied here retires 41 of 41 nulls.

---

## 1. The roster, by name, with cause

45 profiles over 40 unique ROMs (Contra ×2, Metroid ×2, Tetris ×2, Zelda ×3). SMB is not in
this table: it has no `solve:` block because it is served by `SmbGame`, the one hardcoded
adapter. Counting SMB, **5 of 46 solver-capable configurations can witness their own clear.**

`Hook` is what `scripts/clear_reachability.py` reports the profile declares. Every row's
bucket is derived from the hook plus the evidence in the `Cause` column — nothing here is
inherited from a prior verdict without being re-derived.

### 1a. CONFIRMED — 4

| Profile | Hook | Cause — why this bucket |
|---|---|---|
| `bubble_bobble` | `level_key: [0x0401, 0x0462]` | 103 solution-bearing archives; the chain walks rounds 22 → 99 across independently-rooted runs, each entrance minted from the previous round's clear. Rendered witness (`bb0_*`): at the firing step the round readout is 69; 45 actions later it is 70, in a visibly different room layout. |
| `castlevania` | `level_key: [0x0028]` | Banked, replay-verified block clears in five independent run trees (`cv_smoke`, `cv_chain_a`, `cv_chain_hw`, `cv_chain_hw2`, `cv_chain_poweron`), including a power-on chain that walks 0 → 1 → 2. Rendered witness is honest but subtle: the block counter increments without a scene wipe, so the frame pair shows the same locale one boundary apart. The chain across independent entrances is what carries this row, not the single frame. |
| `excitebike` | `finale: {addr: 0x000E, value: 2}` | **Newly witnessed in this pass.** I replayed `runs/excitebike/excitebike_bootstrap/solutions/sol_000` deterministically: `is_finale` fires at action 1143 of 1144, the section byte reaches its top value 3, and progress sits at the profile's documented full-track maximum 773. Rendered witness (`eb0_at_predicate`) shows the FINISH line, the checkered flag and a cleared TIME field. |
| `tetris_b` | `clear: {mode: byte_change, addr: 0x0050, direction: down, target: 0}` | Fires at action 4328 of a hand-solved B-TYPE win. Rendered witness (`tet0_after_45`) shows `LINES-000` and the game's own **SUCCESS** banner on the playfield — the least ambiguous witness on the roster. |

### 1b. VOID — a hook is declared but has never been seen to fire — 4

These four are `REACHABLE` to the lint. That is a statement about arithmetic, not about
evidence. None has ever fired on a real clear, on this ROM or any other.

| Profile | Hook | Cause — why VOID and not FAIL |
|---|---|---|
| `contra` | `clear: {mode: confluence}` | Arithmetically reachable (progress is a 16-bit `{lo, hi}` pair, so `coord`'s ≥300 drop is representable — unlike every odometer profile). Never fired on a real clear anywhere. Search is pinned at the byte-identical frontier `gx = 3072` across six chained campaigns (~280k–357k cells); no lineage has ever reached a stage boundary, so the predicate has never been asked the question. A **coverage** void. |
| `contra_blank` | `clear: {mode: confluence}` | Same ROM, same solve block (verified identical by parse). Sweep confirmed reachability live and confirmed the same wall. One additional finding: a false fire *is* constructible on a two-death game-over arc, but is unreachable through the real `Solver.observe()` order, because `is_dead` is checked before `is_clear` and a persistent dead state retires the lineage in three observations. |
| `ducktales` | `clear: {mode: score_jump, threshold: 5000}` | The threshold is **50× the largest value ever measured in this repo**: over 8,030 archived cells plus 111,236 fresh steps the maximum single-step delta is 100 progress units, against a threshold of 5000. If the game had no boss treasure at all, this hook would report exactly what it reports now. Downgraded to `T2_WIRED_UNCERTIFIED` by the census; that stands. |
| `kid_icarus` | `level_key: [0x0130]` | The one profile with a real, declared, monotone stage byte. It has **never been observed to change** — not in 113 banked cells, not in ~21,500 fresh steps, not on death. An unfired predicate that has never been witnessed firing on anything is indistinguishable from an inert one; the counterfactual test returns the same null either way. The honest reading is that the search cannot reach a stage end, which is a capability finding, not a detection result. |

### 1c. VOID — no clear predicate is declared at all — 37

For every row here, `GenericGame.is_clear` reduces to `level_key(ram) > start_key` with
`level_key: []`, i.e. `() > ()`, which is `False` for every RAM state that exists. The
banked `solutions: 0` is a compile-time constant. `scripts/clear_reachability.py` returns
`NONE` for all 37, and its message is the correct one to cite: *"`solutions: 0` is a
compile-time constant, not a search result, and `--want-solutions` is inert."*

The `Cause` column records what *else* is true about the profile — the finding that
survives the void, and the thing that would have to be fixed first.

| # | Profile | Cause — what is actually blocking, beyond the missing predicate | Prior verdict |
|---|---|---|---|
| 1 | `1942` | Forced auto-scroller: all 14 declared actions produce strictly positive progress delta (min +13, max +249, zero reversals), so the one-way-door test has no discriminating power here. The ABSTAIN is about the test, and never claimed the game has no reachable clear. | census · triage SURVIVED |
| 2 | `arkanoid` | No paddle or ball X observable is declared anywhere in the solve block, so no closed-loop controller is buildable. 22/22 policies including pure NOOP die in the same 528–656 step band, with a control check ruling out a dead input pipe. The only profile in its family whose `n_area == 1` is a real read (`area: 0x0010` is declared) rather than a YAML artifact. | census · triage SURVIVED |
| 3 | `bad_dudes` | Death discriminator broken: declared lives reads 0 at root; the replacement `$00CD` reads 2→0→2 in a single frame-pair mid-attack and fires ~24–25× per 700 death-free steps. | census · triage SURVIVED |
| 4 | `batman_the_video_game` | Odometer hard-stalls at 2612 px; 7 of 8 held inputs move it zero, `left` costs 6 of 8 lives. Reproduced at 2611 px by an independently-seeded prior run. Search wall. | census |
| 5 | `bionic_commando` | Never adjudicated individually — but the best re-onboarding candidate on the roster: 1126 cells over **30 distinct declared-area values**, verdict `SOUND_ADVANCING`, frontier growing at cutoff. Real stage structure exists and is already keyed; only the clear hook is missing. | not measured |
| 6 | `blaster_master` | Declares `area: 0x0020`, so `n_area == 256` is not the usual YAML artifact — but the **declared byte is the wrong one**. Reproduced live: holding right ramps it 32 → 136 → 244 and then saturates flat at 244; holding left pins it at 32. It is a clamped position readout, not a room id, directly contradicting the shipped YAML comment calling it a "non-saturating room/screen counter". The declared lives byte `$00BE` is also unstable (0/1 with 17 transitions on a right-hold, constant 3 on a left-hold). | sweep (never measured before) |
| 7 | `castlevania_iii` | Never adjudicated individually. Smoke is healthy: 1 → 52 → 173 cells, `max_sect` 6 → 16, verdict `SOUND_ADVANCING`. Declares a real `area: 0x0057`. Second-best re-onboarding candidate after Bionic Commando. | not measured |
| 8 | `chip_n_dale_rescue_rangers` | Never adjudicated individually. Declares `area: 0x0023`, which stayed frozen at its root value 36 across 88,765 steps — classified `SOUND_GAME_STOPS` / skill wall by its own smoke. | not measured |
| 9 | `darkwing_duck` | Death discriminator broken, and inverted rather than merely noisy: the best candidate `$05E6` **refills to its start value at every respawn**, so it fires on chip damage and misses the actual death. Five re-derived alternatives all oscillate. | census |
| 10 | `double_dragon` | The profile's own Go-Explore frontier axis `$00B2` is input-independent deterministic churn: the identical sub-sequence `[1,3,1,3]` appears under two completely different input sequences, and it "advances" on 19 of 34 steps of an ordinary trace with zero real transitions. Archive depth claims are inflated by this. Also one of the two profiles where the confluence detector once fired in production — on a combat RAM blip (progress 72 → 846 → 88 in five steps), withdrawn as unsafe. | census |
| 11 | `double_dragon_ii` | 195/195 banked cells at `level_key == ()` and `area == 0`. Not stalled: the frontier grew 1 → 133 → 195 over two minutes and was still climbing. This verdict independently discovered both root defects before the campaign did and named them verbatim. | census · triage SURVIVED |
| 12 | `ducktales_2` | Death discriminator broken: `$000B` goes 0→1 within **one** forced forward-hold step from a fresh load (which is why the discovery tool's guard missed it — it samples its reference *after* that step), and produces 7–15 "decrements" per 700-step window containing zero deaths. | census · triage SURVIVED |
| 13 | `galaga` | Start-state mint defect, not a detection problem: `$0091` is byte-for-byte identical under `hold_A` and `hold_right`; `$0464` free-runs under 6000 pure-NOOP steps; the scroll odometer jumps a constant +224 (one NTSC visible-frame height) 2–5 steps after every lives-byte refill. The root state sits inside a non-interactive attract loop. The fix is a re-mint. | census · triage SURVIVED |
| 14 | `ghosts_n_goblins` | Odometer plateaus at 3326 px across four independent runs. Separately, the **only** profile in the census that reached a confluence configuration at all — under a deliberately weakened `min_signals=1` rig, where it fired **9 times out of 9** on non-clearing traces, always at the first possible check. | census |
| 15 | `gradius` | Adjudicated in the census follow-up: the hook was wired and hardened on 2026-08-06 against a measured false fire, and then **silently disarmed on 2026-08-24** by commit `09299fa`, an unrelated League onboarding change that swapped `progress` from a RAM pair to the odometer. `coord` then became unreachable and the vote ceiling fell to 1 of 2. The declaration was withdrawn rather than left dead. | census follow-up |
| 16 | `ice_climber` | Sent back by triage, re-measured, still VOID. `odo_x` never leaves 0 under any policy across ~186k combined steps; `odo_y` moves only as a fixed −48 step that co-occurs with death/respawn. The A1/A2 control is the best anti-vacuity work in the batch: removing the restart loop collapsed the apparent `odo_y` range 0..−1248 → {0, −48, −96}, falsifying the campaign's own pre-registered candidate. Eight archived cells whose keys are identical except the animation-phase digit — one reachable state in 83,812 steps. | census · triage **VOID** · sweep re-measured → VOID |
| 17 | `journey_to_silius` | Death discriminator broken: the declared byte reads 0 at root; the only nonzero-starting alternative `$0135` drops 1→0 within 4 steps of a bare RIGHT hold but stays flat under NOOP — keyed to movement onset, not death. Wiring it collapsed a 774-cell archive to 2 and the frontier from 1269 px to 7 px. | census · triage SURVIVED |
| 18 | `kirby` | Never adjudicated individually. The second profile where the confluence detector fired in production — three times in a 24-second smoke, every one an ordinary door load. Withdrawn as unsafe. Declares a real `area: 0x004F` and has genuine door/room structure; per-room camera clamp (~10 px) is the search-side blocker. | not measured |
| 19 | `kungfu` | `current_floor` never left 0 across ~719,500 combined steps and two campaign dates. The alternative score candidate was disqualified by its own absence test: substituting the jump button for the attack button gives an *identical* 22 events / 800 steps. Odometer gate reads `CAMERA_STATIC_AGENT_ACTIVE`. Skill wall — the search never gets off the first floor. | census · triage SURVIVED |
| 20 | `legend_of_zelda` | **Cannot construct.** `make_game()` raises `KeyError: 'y'` — the solve block is a three-line stub (`rom`, `progress`) with no `y`, `level_key`, `lives` or `area`. One of only two profiles on the roster that cannot take a single search step. Separately dirty at runtime — see §7. | sweep (never measured before) |
| 21 | `mega_man_3` | 2,969 cells across four independent archives; `level_key` and `area` constant in 100% of them. | census |
| 22 | `mega_man_usa` | Odometer ceiling `x = 24`, triangulated three ways, one of which uses no search machinery at all (3 held strategies × 300 steps, lives unchanged throughout — the character is physically blocked, not dying). With area/hp/sig all constant the cell key collapses to `(y//32, x//16)`, so the "17,054 revisits, barren" figure is partly a degenerate-key artifact; the deterministic probe reproduces the ceiling without cells. Suspected start-state mint defect. | census · triage SURVIVED |
| 23 | `megaman` | `is_clear` structurally unfireable; the empirical plateau at `gx 4864` corroborates but was never load-bearing. | census |
| 24 | `metroid` | 68 distinct rooms reached with real edges — the search works. The win state lives in cartridge PRG-RAM outside `get_ram`'s `$07FF` reach, and the only in-reach addresses are quarantined. A genuine game-structure blocker layered on top of the code-level one. | census |
| 25 | `metroid_roomfp` | Declares a real `area: 0x804` (room-fingerprint pseudo-RAM) and a fresh 3-minute burst reached 2,299 cells / 68 distinct room identities with elasticity still rising at cutoff — **this profile's search demonstrably works**, so it must not be filed alongside the no-structure plateaus. Only the clear predicate is missing. A one-way return-search on two hand-scripted doors was correctly refused promotion: neither door sits within two rooms of the frontier. | sweep (never measured before) |
| 26 | `ninja_gaiden` | Death discriminator broken: four candidates tried and rejected; `$0386` empties while the odometer keeps climbing 94 → 143 px over the following 60 steps. This receipt independently discovered and wrote down the `() > ()` identity itself, then correctly ruled it non-binding because the S0 blocker is prior. | census · triage SURVIVED |
| 27 | `ninja_gaiden_ii` | Sent back by triage, re-measured, still VOID — and reclassified. Its disposition rested on "0 solutions and no area/level transition of any kind", both algebraic identities. Its own S6 finding shows `is_dead` never fires either: the declared lives byte `$004C` reads a flat 1 through a rendered GAME OVER. Re-measured this pass against the campaign's own calibrated oscillation gate: **REJECT**, 6 cycles against a tolerance of 3, first cycle at step 55 — and it still rejects (5 cycles) when the log is truncated strictly before the death, so the oscillation is not death-driven. Belongs in the death-discriminator family. What survives: two visually-confirmed GAME OVER walls at x≈2043 and x≈2557, input-invariant across four actions each. | census · triage **VOID** · sweep re-measured → VOID |
| 28 | `ninja_gaiden_iii` | Never adjudicated individually. Odometer gate PASS / SIGNAL SOUND; declares a real `area: 0x00DC`. | not measured |
| 29 | `paperboy` | Death discriminator broken, with the cleanest absence test in the set: under pure NOOP — no player action, no death opportunity — `$000B` flickers 0↔255 thirty times in 3000 steps and `$00B2` cycles 4→3→2→1→0→4 four full times on a ~117-step cadence. Eight ranked candidates share tick boundaries: one background clock, not eight life counters. | census · triage SURVIVED |
| 30 | `power_blade` | Start-state mint defect: `hold_A_only`, which presses no directional button at all, produces the identical `x = 88..88` trace as `hold_right` across 300 steps, and every tested strategy ends in a lives decrement on a fixed ~1144-frame schedule independent of the action taken. The degenerate control arm was included and behaved identically to the real policies — which is itself the finding. | census · triage SURVIVED |
| 31 | `punchout` | Sent back by triage as UNCERTAIN, re-measured, still VOID — and the prior PURITY_BLOCKED over-claimed. Leg 1 stands and was reproduced: the purity-clean candidate `opp_hp==0 AND opp_down==1` is falsified by a generic pure-NOOP continuation (opponent HP refills 0 → 96 at step 121; `opp_down` never reverts across 900 steps, so after the first knockdown the conjunction degenerates to bare `opp_hp==0`). Leg 2 does **not** stand: the claim that the quarantined `$0001` is the only byte that discriminates a knockdown from a bout end was asserted from the quarantined documentation, not measured. A real terminal event was witnessed this pass at step 667. **But the follow-up candidate `$000A` must not be promoted**: the adjudicator re-ran the blind full-RAM diff and found 37 bytes change at that transition, **ten** of which survive both anti-vacuity filters (one-way latched, and flat across both knockdowns), four with the identical 0→1 shape. Only the quarantined block picks `$000A` out of that tie. Purity-clean evidence supports "a one-way terminal event occurred", and nothing narrower. | census · triage **UNCERTAIN** · sweep re-measured → VOID |
| 32 | `rygar` | No area, no `room_sig`, no non-empty `level_key` — no clear axis was attempted at all. The oft-cited "0 clears in 26,702,638 recorded step-observations across 5 archives" is one constant reported five times. | census |
| 33 | `shatterhand` | Reproducible `x = 1023` wall across two independent seeds; five replayed lineages all converge there with no scene cut. Notable for catching a fabrication in the census's own harness first: reusing one `Pool` across replays leaked the odometer accumulator and produced `x_end` values that were exact multiples of one true value. | census |
| 34 | `super_c` | Progress byte `$00A4` wraps mod 256 (found behaviourally, not from a map); with an unwrap-aware score, 25 seeds never exceed 640. | census |
| 35 | `tetris_usa` | **Cannot construct.** `make_game()` raises `KeyError: 'y'` — no `solve.y`, no `solve.lives`, no `solve.level_key`, no `solve.area`. No bank was ever produced, so there is no `n_area` to be an artifact of. The odometer's flatness here is a genuine genre finding (60,000 idle NOOP frames, `x` stays exactly 0), *not* an instrument fault. The real blocker is upstream: no HUD-tile discovery tool exists for a decimal-counter progress axis, and no `GenericGame` variant exists that does not hard-require `y`. | sweep (never measured before) |
| 36 | `zelda` | PURITY_BLOCKED, and the block is real: `ZeldaReward::RAM_GANON_DEFEATED` is a compiled-in Rust constant whose own comment cites a disassembly, quarantined under CLAIMS.md and defended by a test. Three independent measured negatives back the "no clean substitute" leg: the room-graph engine cannot separate a dungeon-entry fade from a death fade (identical classifier signature), S2 replay fails to restore to the recorded source room 47.5% of the time on a random 40-edge sample, and a blind rediscovery of a substitute win byte was attempted, receipted and failed on all three methods. This profile's `solutions: 0` is one of the few honest ones on the roster — it was **pre-registered** as a coverage baseline, because on a grid map a `level_key` advance would fabricate a clear every time a worker walks east. | census · triage SURVIVED |
| 37 | `zelda_roomfp` | Same quarantine. Its headline number — 443,419 archive cells over 4 × 90-minute 12-worker runs, zero solutions increments — is the single largest void on the roster and should be struck from every future citation. The disposition nonetheless stands on a leg upstream of any instrument: there is no admissible target signal to build against, so the campaign could not have told a real Ganon defeat from an arbitrary rare fingerprint. Its 1024 measured rooms are a real, purity-clean *progress* signal and stay in the scored denominator. | census · triage SURVIVED |

---

## 2. Rollup — the number, corrected

| Bucket | Profiles | Unique ROMs |
|---|---:|---:|
| **CONFIRMED** | **4** | 4 |
| **VOID** | **41** | 37 |
| **FAIL** | **0** | **0** |
| Total with a `solve:` block | 45 | 40 |

The ROM columns do not sum, and that is not an error: **Tetris (USA)** appears in both
buckets, because `tetris_b` is CONFIRMED and `tetris_usa` — a second profile over the same
ROM — is VOID. It is the only overlap. Profiles are the unit that matters here; a config is
what the lint runs on and what a solver run loads.

Adding SMB (no `solve:` block; served by `SmbGame`): **5 of 46 solver-capable
configurations can witness their own clear**, which reproduces the League's standing
`witnessable` figure from an independent direction.

VOID breaks down as:

| VOID sub-cause | Count | Detection |
|---|---:|---|
| No clear predicate declared — `is_clear` is `() > ()` | 37 | `clear_reachability.py` → `NONE` |
| Predicate declared, never witnessed firing on anything | 4 | `contra`, `contra_blank`, `ducktales`, `kid_icarus` |

And the two counts the lint produces independently — `NONE = 37`, `REACHABLE = 8` — partition
exactly across these buckets (`8 = 4 CONFIRMED + 4 declared-but-unwitnessed`). The table and
the tool agree without being fitted to each other.

**Coverage of the adjudication.** 34 of the 41 VOID profiles were adjudicated individually
this round (29 census + 5 sweep, with 3 of the 29 re-measured). `gradius` and `contra` were
adjudicated in the census follow-up. The remaining 5 — `bionic_commando`, `castlevania_iii`,
`chip_n_dale_rescue_rangers`, `kirby`, `ninja_gaiden_iii` — had never been named in any
clear-detection pass; they are adjudicated here from the lint plus their own shipped smoke
receipts, and they land in the same bucket for the same reason. **41 of 41 are now named with
a cause.**

---

## 3. Why FAIL is exactly zero

Three structural facts, each read from source and independently reproducible without running
anything:

1. **`scripts/go_explore_solve.py:2575`** — `is_clear` opens with
   `if self.level_key(ram) > tuple(start_key)`. With `level_key: []` this is `() > ()`,
   which is `False` in Python for every possible RAM array. 41 of 45 profiles carry
   `level_key: []`; 37 of them declare no other hook. **This line was not repaired and is
   live verbatim at HEAD.** The lint reports it; it does not fix it.
2. **`scripts/go_explore_solve.py:2450`** — `area()` returns the literal `0` when no
   `solve.area` is declared. 30 of 45 profiles declare none, so `n_area == 1` in those banks
   is a property of the YAML. Every conclusion of the form "this game shows no stage
   structure" that rests on such an `n_area` measured the config, not the ROM.
3. **`scripts/clear_detect.py:2180`** — the live streaming vote is still exactly
   `tally + coord >= min_signals` (default 2, `apu_weight` default 0). `coord` requires the
   progress readout to fall by ≥300 and land ≤200. **26 of the 45 profiles are
   `progress: {source: odometer}`**, and `nes_core/src/ppu.rs` `odo_fold_frame` re-anchors
   rather than integrating across a scene cut, so the integral freezes instead of rewinding.
   For all 26, `coord` is arithmetically dead and the vote ceiling is 1 of 2 — unreachable by
   construction, not merely untuned.

Given (1) and (3), the only way a profile could produce a FAIL is if it were one of the four
in §1b, and each of those fails the counterfactual test for its own reason (§1b's Cause
column). Hence zero.

**What this does *not* say.** VOID on the clear question is not "nothing was learned". Many
of these profiles carry real, detector-independent findings that survive intact — search
walls, mint defects, broken death discriminators, purity blocks. Those are catalogued in §8.
The point is narrower and sharper: none of them is evidence about whether a clear predicate
would work.

---

## 4. The instrument's own record — the positive controls finally ran

The census flagged this as the thing that must not be buried: three positive controls were
specified, the fan-out was supposed to be gated on them, and **they never ran**. They have
now run (`runs/clear_control_2026-08-26/`), and the result is more interesting than a pass
or a fail.

Each control replays a tape that straddles a real clear at the solver's own observation
cadence. Ground truth is the profile's own declared predicate firing; a rendered frame is
the human-checkable witness that does not depend on it.

| Control | Tape | Live hook (`tally`+`coord`, min 2) | Offline harness (4 signals, ≥0.75) | Why |
|---|---|---|---|---|
| **SMB** | 5 banked level clears | — | **5/5 hit, 0 false positives** | Its home game. `coord` fires on all 5. |
| **Castlevania** | `cv_smoke/sol_000` + 2 chain tapes | **HIT** — 3/3 in the source-swap experiment; fires 12 actions late | **2/6 (33%)**, gate 0.8 → fail | Progress is a 16-bit RAM pair spanning 14..679, so a 592-unit drop is representable. |
| **Bubble Bobble** | `r99_fixed/sol_000` + 6 chain tapes | **MISS** — `coord` 0/30 checks | **1/7 (14%)**, plus 1 false positive firing 1124 frames early | Progress spans **98..99**. Max possible drop is **1**, against a required 300. Arithmetically impossible. |
| **Tetris B** | hand-solved 4329-action win | **MISS** — `coord` 0/220 checks | **0/1** | Progress spans **0..32**. Max possible drop **32**, required 300. Same arithmetic. |

And the decisive controlled experiment, `cv_odometer_swap.json` — same three Castlevania
tapes, same detector, **only the progress source swapped**:

| Progress source | Hits | Max single-step drop | `coord` checks fired |
|---|---:|---:|---:|
| `ram_pair` (shipped) | **3 / 3** | 592 | 4 per tape |
| `odometer` | **0 / 3** | **4** | 0 |

That is the arithmetic dead-end of §3 item 3, demonstrated on a game where the detector
otherwise works. It is not a tuning gap. Swapping a working profile onto the odometer breaks
its clear detection, which is exactly what happened to Gradius on 2026-08-24 and went
unnoticed for eighteen days.

**The honest summary of the confluence detector's record outside SMB:**

- Replayed against real clears on three games: **hits 1, misses 2** — and both misses are
  arithmetic impossibilities, not near-misses.
- Fired in production on non-SMB games exactly twice: **Double Dragon** (a combat RAM blip,
  progress 72 → 846 → 88 in five steps) and **Kirby** (three fires in 24 seconds, every one
  an ordinary door load). Both false. Both withdrawn. Both still sitting in
  `runs/detector_gate_20260810/` as solution tapes with `start_wd == clear_wd == []`.
- Fired 9 times out of 9 falsely on Ghosts 'n Goblins under a weakened `min_signals=1` rig.

**True positives banked in production on a non-SMB game: zero. False positives banked: two.**

### 4a. A harness gap found while adding a fifth control

The control harness (`runs/clear_control_2026-08-26/{live_control,witness}.py`) establishes
ground truth by calling `game.is_clear(...)` and nothing else. `GenericGame` exposes a
**separate** `is_finale()` method, and Excitebike's predicate is a `finale:` hook. Running
the harness on Excitebike therefore returned `truth_action: null` — "the predicate never
fired" — on a tape where the predicate fires at action 1143 of 1144. That is the failure
mode this whole campaign exists to catch, in the campaign's own instrument: it reports the
same null whether the predicate works or does not exist.

I re-ran it against `is_finale` and got the FINISH-line witness in §1a. Two related notes:

- `scripts/clear_detect.py:2273` **does** check `is_clear(...) or is_finale(...)`. The
  shipped harness is correct; only the ad-hoc control script was not.
- `scripts/replay_sweep.py:497` checks `game.is_clear(...)` only. Any `finale:`-hooked
  profile's tape will be reported by the sweep as "never satisfied `is_clear`". Excitebike
  is the only such profile today, so nothing is currently mis-scored — but the defect is
  live and should be fixed with the same `or game.is_finale(...)`. Left untouched here
  because that file is currently owned by a sibling workflow.

---

## 5. The 17 stale verdicts — what survived, and what "survived" means

Seventeen verdicts were carried into this round from the broken instrument. Triage returned
**14 SURVIVE, 2 VOID, 1 UNCERTAIN**; the three that did not survive were re-measured from
scratch by the sweep and all three came back VOID.

| Outcome | Games |
|---|---|
| **SURVIVED** (14) | `1942`, `arkanoid`, `bad_dudes`, `double_dragon_ii`, `ducktales_2`, `galaga`, `journey_to_silius`, `kungfu`, `mega_man_usa`, `ninja_gaiden`, `paperboy`, `power_blade`, `zelda`, `zelda_roomfp` |
| **VOID → re-measured** (2) | `ice_climber`, `ninja_gaiden_ii` — both re-measured under a different seed and harness; both still VOID |
| **UNCERTAIN → re-measured** (1) | `punchout` — leg 1 upheld and reproduced, leg 2 falsified, disposition still VOID |

**"Survived" means the verdict's own sentence stands as written. It does not promote any of
them out of VOID.** Every one of the 14 is scoped to *"from here"* — from these observables
and this controller — and not one of them ever claimed the game has no reachable clear. And
**none of the 17 ever reached the confluence detector in the first place**: only three have
an `s3_signals.json` at all, and all three record a non-run (`1942`
"NOT_APPLICABLE_NO_S2_WINDOW", `kungfu` "ROUTED_AROUND_PER_S1", `zelda` "NOT_REACHED"). What
carries the 14 is RAM peeks under held inputs, odometer gates and absence tests. That is why
they survive; it is also exactly why they say nothing about clear detection.

Two shared caveats the triage attached to the death-discriminator exits in this set
(`bad_dudes`, `ducktales_2`, `journey_to_silius`, `ninja_gaiden`, `paperboy` — `darkwing_duck`
is the sixth of that family but was not among the 17), worth carrying forward: (a) they prove that **no death veto can be built**, so no predicate can be
*certified* — they are not evidence the game lacks a reachable clear, and must never be
counted that way in a denominator; (b) four of the five (`ducktales_2`, `journey_to_silius`,
`ninja_gaiden`, `paperboy`) exited one step more conservatively than the procedure required
and skipped the offline bank census, which costs under a second and is entirely independent
of the lives byte. That left free information on the table without making any verdict wrong.

Three label corrections for the census ledger, found by triage: `galaga` is
`INSTRUMENT_BLOCKED_NO_PROGRESS_SIGNAL` (not `NOT_WITNESSABLE_FROM_HERE`),
`double_dragon_ii` is `NO_TRANSITION_IN_BANK` (not `ONEWAY_TEST_ABSTAIN` /
`SPURIOUS_KEY_DIMENSION`), and `ninja_gaiden_ii` belongs in the
`INSTRUMENT_BLOCKED_NO_DEATH_DISCRIMINATOR` family on its own evidence.

---

## 6. The 5 that had never been measured

| Profile | Resolution |
|---|---|
| `blaster_master` | VOID — instrument. No `clear:` key, so `is_clear` is the `() > ()` identity. Bonus finding: its declared `area: 0x0020` is refuted by direct measurement (saturating position readout, not a room id), so its `n_area = 256` bank shape is a mischosen address, and its declared lives byte is unstable. |
| `contra_blank` | VOID — **coverage**, not instrument. The detector is genuinely reachable and non-vacuous here; the search has simply never reached a stage boundary. Categorically different from the other four and must not be pooled with them. |
| `legend_of_zelda` | VOID — **construction**. `KeyError: 'y'` before step one. Strictly worse than the other four. See §7. |
| `metroid_roomfp` | VOID — instrument. But its search demonstrably works (2,299 cells / 68 room identities, elasticity still rising at cutoff), so it is a detection gap on a healthy profile, not a plateau. |
| `tetris_usa` | VOID — construction, plus no death discriminator. `KeyError: 'y'`; no bank ever existed. Its flat odometer is a real genre finding, not an instrument fault. |

Two of the five cannot construct a solver adapter at all. I ran that check across the whole
roster: **43 of 45 profiles construct; the two failures are exactly `legend_of_zelda.yaml`
and `tetris_usa.yaml`**, both `KeyError: 'y'`. `clear_reachability.py` reports `NONE` for
both — the same string it reports for a perfectly runnable coverage baseline like
`blaster_master`. A caller reading only that verdict cannot tell "legitimate coverage
baseline" from "cannot run at all". The lint should also validate that `y`, `level_key` and
`lives` are present when a profile intends to use `GenericGame`.

---

## 7. The Zelda ruling, stated plainly

**`configs/legend_of_zelda.yaml` is not a clean path to Zelda. It is clean as text and dirty
at runtime, and it cannot run at all.** Every claim below I reproduced myself in the current
tree.

1. **Clean as text.** 31 lines, no `ram_mapping`, no quarantine block, no address of any
   kind. It passes the in-tree outside-provenance lint that `configs/zelda.yaml` trips.
2. **Dirty at runtime.** Dispatch in this codebase is by profile *name substring*
   (`nes_core/src/rewards.rs:4443`, `nes_core/src/depth_tracker.rs:28`, both
   `contains("zelda")`), and this profile's name is `"The Legend of Zelda"`. Built straight
   from the YAML: flipping RAM byte `0x0672` on an otherwise-zeroed 2 KB buffer turns
   `(-0.001, False)` into `(19999.999, True)` with `episode_success() == True`. That byte is
   the quarantined `q_ganon_defeated`. `DepthTracker("The Legend of Zelda")` returns the
   Zelda arm's caption on zeroed RAM, where `DepthTracker("Blaster Master")` returns the
   generic one. The profile declares no reward weights, so it silently inherits the
   disassembly-sourced win predicate as a default.
3. **Non-functional as a solver profile.** `make_game()` raises `KeyError: 'y'`.
4. **Checkpoint collision.** `derive_checkpoint_dir` slugs both this profile and
   `configs/zelda.yaml` to `checkpoints/the_legend_of_zelda`. Artifacts from the blocked
   profile and the supposedly-clean one land in one indistinguishable subtree.
5. **Its declared progress axis is measured dead.** `runs/onboard_wave2/gate_legend_of_zelda.json`
   records `distinct=1, min=0, max=0` over 1200 steps — the camera never moved, because the
   overworld flips screens rather than scrolling — and the gate still returned
   `passed: true` because `progress_signal_gate.py` computes `passed = not instrument_findings`
   and files "camera never moved" as a *behaviour* finding. A constant-zero progress column is
   the same class of vacuous instrument as `() > ()`.

**The quarantine did not cost us Zelda. The substring dispatch did.** Renaming the file is a
one-line band-aid that also moves the checkpoint slug and leaves the trap armed for the next
profile whose title contains "zelda" or "mario". The fix is to route reward and depth by an
explicit profile key defaulting to `Reward::Generic` / `GameKind::Generic`, plus the
falsifier that is missing today: a test asserting a profile which declares no reward gets the
generic one. Nothing anywhere would currently fail if that dispatch were wrong.

### 7a. A fourth Zelda config the adjudication missed — verified

`configs/zelda_gui_tuned.yaml` is picker-visible, pinned bootable by
`tests/test_profile_configs.py`, and still carries a **live, unquarantined 13-entry
`ram_mapping:`** under the comment *"Win chain (disassembly + emulator-verified)"*. Verified
by parse: it contains `ganon_defeated 0x672`, `triforce_pieces 0x671`, `song 0x609`,
`current_hearts`/`max_hearts 0x66f`, `dungeon_level 0x10`, `rupees`, `bombs`, `keys`,
`link_x/y`, `world_map_x/y` — the whole table `configs/zelda.yaml` quarantines.

The values are int-parseable, which matters: `scripts/observatory.py:489` computes
`mapped = {int(a) for a in profile["ram_mapping"].values()}` and folds them into the
**pre-probe exclusion set**. So an external RAM map would steer this project's own discovery
instrument *away from* precisely the bytes under quarantine. The string-valued design of
`zelda.yaml`'s quarantine block exists specifically to make that `int()` raise; this file
defeats it. `tests/test_zelda_purity_quarantine.py:29` hardcodes
`PROFILE = configs/zelda.yaml`, so nothing catches it. (A fifth config,
`configs/zelda_multidemo_overrides.yaml`, was checked and is clean: no name, no
`ram_mapping`, no action space.)

Recommended, not applied here: quarantine `zelda_gui_tuned.yaml` the same way, re-parametrize
the quarantine test over every `configs/*zelda*.yaml`, and add one check that could actually
fail — assert no `ram_mapping` value in **any** config equals a quarantined address. The
existing suite passes today with a full unquarantined copy of the table sitting one file over.

### 7b. What Zelda's own numbers do and do not mean

`configs/zelda.yaml`'s `solutions: 0` is **not** a void-suspect null. It was pre-registered:
`level_key` empty is a correctness requirement on a grid map, because a `level_key` advance
would fabricate a clear every time a worker walks east and execute every worker that walks
west. That is one of the few honest zeros on the roster.

`zelda_roomfp`'s 443,419 cells with zero solution increments is the opposite — the single
largest void figure in the repo. Strike it from every future citation.

Finally, Zelda is a **ready-made falsifier for the repair still in flight**: a dungeon-entry
fade and a death fade share an identical classifier signature (Δscene ≥ 2, odometer flat).
Any new room-fingerprint transition signal needs a death discriminant beneath it before it is
trusted on any game, and Zelda is the negative control that already exists.

---

## 8. What survives the void

The 41 VOID nulls carry no information about clear detection. They carry a great deal about
everything else, and none of it is retracted:

| Surviving class | Games | Status |
|---|---|---|
| **Death discriminator broken or absent** | `bad_dudes`, `darkwing_duck`, `ducktales_2`, `journey_to_silius`, `ninja_gaiden`, `paperboy`, + `ninja_gaiden_ii` (reclassified this pass) | 7. Not merely inert — two of them actively destroy search (774 → 2 cells; 1096 → 24 cells). |
| **Start-state mint defect** | `galaga`, `power_blade`, `mega_man_usa` (suspected) | 3. Input-insensitive fixed-schedule behaviour. A re-mint job, not a detector job. |
| **Spurious declared key dimension** | `double_dragon` (`$00B2`), `blaster_master` (`$0020`) | 2. Both refuted by direct measurement against the shipped YAML comment. |
| **Purity-blocked at the only known address** | `zelda`, `zelda_roomfp`, `metroid` | 3. Genuine, defended by tests. `punchout` was downgraded out of this class this pass. |
| **Cannot construct** | `legend_of_zelda`, `tetris_usa` | 2. Roster-wide check: 43/45 construct. |
| **Search / skill wall, detector irrelevant** | `batman_the_video_game`, `contra`, `contra_blank`, `chip_n_dale_rescue_rangers`, `ghosts_n_goblins`, `ice_climber`, `kungfu`, `mega_man_3`, `megaman`, `mega_man_usa`, `rygar`, `shatterhand`, `super_c`, `arkanoid`, `kid_icarus` | ~15, overlapping. If the agent cannot reach a level end, no predicate helps. |
| **Healthy search, detection gap only** | `bionic_commando` (30 areas), `castlevania_iii` (`max_sect` 16), `metroid_roomfp` (68 rooms), `kirby` (real doors), `double_dragon_ii` (frontier still climbing) | 5. The best re-onboarding candidates on the roster. |

A shared-engine defect surfaced by `ninja_gaiden_ii` and worth filing separately:
`is_dead`'s modular check `1 <= (start_lives - lives) % 256 <= 8` misses a terminal death
that leaves the lives byte **unchanged** (d = 0) — a distinct failure mode from the 0→255
wrap it was built to catch. Relatedly, and correcting a reasoning error that would mislead
elsewhere: `is_dead` does **not** latch permanently after a life loss. Because the window is
modular, a game-over reset that restores lives to the baseline makes it read `False` again.
The `contra_blank` false positive is unreachable via the 3-observation death debounce, not
via monotonicity.

---

## 9. New defects found in this pass

1. **The purity gate was red at HEAD and nobody knew.** `tests/test_zelda_purity_quarantine.py::test_no_source_file_reads_the_quarantine_key`
   failed at `f618eef` on `runs/clear_census/zelda_roomfp/s0_peek_zelda_roomfp.py`, a census
   probe script (created 07:58 on 2026-08-26) that copied the three blocked win-signal
   addresses and the literal quarantine key name into executable code. **This is a gate
   firing on a real hit**, and it went unreported because the census ran a narrower gate set
   that did not include the purity test. Fixed by redacting the enumeration in that probe and
   replacing it with a pointer to the two designated homes; the script's measurement
   behaviour is byte-identical (the redacted dict was only printed). The file is under
   `runs/`, which is gitignored, so **the fix does not appear in this commit** — anyone
   re-cloning will not see it, and the gate will not go red for them either, because the
   artifact is not in version control. The lasting fix is procedural: run the purity test in
   every campaign's gate set.
2. **The control harness could not see a `finale:` predicate** (§4a) — it checks `is_clear`
   only, and returned a clean-looking null on a game whose predicate fires. Fixed for the
   Excitebike control by calling `is_finale`; `scripts/replay_sweep.py:497` carries the same
   gap and is left for its owning workflow.
3. **`blaster_master`'s declared `area: 0x0020` is refuted by measurement** (§1c row 6). Its
   own YAML comment cites `discover_observables` as recommending it as a "non-saturating
   room/screen counter"; a live replay shows it saturating at 244 and tracking directional
   input smoothly in both directions. Re-run discovery on this profile.
4. **`ninja_gaiden_ii`'s declared `lives: 0x004C` fails the campaign's own calibrated
   oscillation gate** — REJECT at 6 cycles vs tolerance 3, and still REJECT at 5 cycles when
   truncated strictly before the death, so it is not a death/respawn artifact.
5. **`punchout`'s `$000A` is a ten-way tie, not a discriminator** (§1c row 31). The blind
   full-RAM diff narrows to ten candidates that pass both anti-vacuity filters; only the
   quarantined block picks one. Recorded as unresolved.
6. **Onboarding probes may have certified progress/area bytes on post-death data.** While
   verifying `arkanoid`, triage found `runs/onboard_wave3/discover_arkanoid_right.json`
   recording `sat_n_transitions = 2` on the declared `$0010` pair, contradicting
   `s6_instrument.json` where all 22 policies report `area_ever_nonzero: false`. The likely
   reconciliation is that the 1200-step hold kept stepping ~600 frames past ball-loss. The
   same "1200-step hold without death termination" recipe was used across onboarding, so
   other profiles' certifications may carry the same contamination. Worth an independent
   check; it does not flip Arkanoid's disposition.

---

## 10. What would actually close the gap

Ordered cheapest first. None of these is "wire more predicates."

1. **Decide whether the new signals are ever going to reach the live vote.** The repair
   landed `entity_wipe_windows`, `RoomFpTransitionSignal`, `InputLockSignal`,
   `LockReleaseNoveltyTrack`, `OamQuiesceSignal` and `SceneCutSignal` — six new signals, all
   of them reachable **only** from the offline replay harness. The live streaming vote is
   still `tally + coord`, and the offline harness still weights only `audio + tally + lock +
   coord`. Until that changes, the 26 odometer profiles remain arithmetically incapable of
   firing `coord`, and §4's control result is the ceiling.
2. **Fix `is_clear`'s `() > ()` identity, or make it a hard error.** A lint that reports the
   defect while the defect stays live will be read as the defect being fixed. The lint is
   good and should stay; it is not a repair.
3. **Extend `clear_reachability.py` to validate `y` / `level_key` / `lives` presence** so it
   stops returning the same `NONE` for a runnable coverage baseline and a profile that cannot
   construct.
4. **Onboard clear predicates for the five healthy-search profiles** — `bionic_commando`,
   `castlevania_iii`, `metroid_roomfp`, `kirby`, `double_dragon_ii`. These are the only
   profiles where a predicate is the *only* missing piece; everywhere else something upstream
   blocks first. Bionic Commando is the strongest: 30 distinct declared-area values already
   keyed into the archive.
5. **Fix the 7 broken death discriminators.** They are not inert; they destroy search.
6. **Cut the name-substring dispatch** (§7) and quarantine `zelda_gui_tuned.yaml` (§7a).
7. **Re-mint the three start states** — `galaga`, `power_blade`, and probably
   `mega_man_usa`. Cheaper than any more search.

---

## 11. Gates

```
.venv/bin/pytest tests/test_profile_configs.py tests/test_clear_detect_ground_truth.py \
                 tests/test_confluence_v2.py tests/test_zelda_purity_quarantine.py -q
363 passed, 21 skipped
```

Before the §9.1 redaction this set was **1 failed, 362 passed, 21 skipped** — a pre-existing
red at `f618eef`, the HEAD this pass started from, not a regression introduced here. The
offending file is untracked and predates this session.

```
.venv/bin/pytest tests/ -q --timeout=120
1 failed, 4730 passed, 27 skipped, 1 xfailed in 374.66s   (at f618eef)
1 failed, 4730 passed, 27 skipped, 1 xfailed in 371.78s   (re-run at b8ebded)
```

The sibling lane landed three further commits mid-pass (`ef87556`, `698f142`, `b8ebded`); the
suite was re-run after them and returns the identical counts, so nothing in this document's
numbers moved under it. Every load-bearing code fact was re-derived at the later HEAD as well:
`NONE=37 / REACHABLE=8` from the lint, `passed = (tally + coord) >= min_signals` at
`clear_detect.py:2180`, and `if self.level_key(ram) > tuple(start_key)` at
`go_explore_solve.py:2575`.

The stated baseline was 4477 passed / 23 skipped / 1 xfailed; the suite has grown by 253
tests since, all from the sibling detector work in the tree (`clear_reachability`,
`oam_quiesce`, the profile-entry harness). **No regression.** The single failure is
`tests/test_night2_runner.py::test_dry_run_passes_live`, and it is environmental, not code:
the dry run refuses to plan a seeding step because `checkpoints/mario_1_2_consol2` already
holds 32 checkpoints past the pinned iter 911. That directory was last written on 2026-08-15,
eleven days before this session, and nothing in this pass touches checkpoints. Deliberately
not "fixed" — the only fix would be deleting a real training tree.

Note also that a bare `.venv/bin/pytest -q` from the repo root does not collect at HEAD: it
picks up `scripts/test_ncst_pool.py` (dated 2026-04-19) which imports a module that no longer
exists, and dies during collection. The suite the Makefile runs, and the one measured above,
is `pytest tests/ -q`.

All 45 `solve:`-bearing configs YAML-parse clean and are re-parsed by this document's own
enumeration. **No config was edited by this pass** — there were no CONFIRMED predicates to
commit, and a predicate is the only thing this pass was authorised to write into a config.
The four already-confirmed predicates were verified where they stand and left untouched.
(The config annotations that landed today — the struck "0 solutions" citations and the
Gradius withdrawal — are `ef87556`, the sibling detector lane, not this pass.)

Evidence: `runs/clear_control_2026-08-26/` (controls, including the Excitebike frames added
here), `runs/clear_detection/<game>/`, `runs/clear_census/<game>/`, `runs/clear_gap/<game>/`.
`runs/` is gitignored, so those artifacts are local receipts, not committed ones.
