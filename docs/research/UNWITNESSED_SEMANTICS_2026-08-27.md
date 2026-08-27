# Unwitnessed semantics in `ram_mapping` — tree-wide sweep, 2026-08-27

## The discriminant

**An annotation asserting semantics tied to an event this project has never
witnessed could not have been measured here.**

That is the whole rule, and it is mechanical rather than a matter of taste. It
does not ask whether an address is right. It asks whether the *claim attached
to it* names something anybody here ever saw. "Increments on a real stage
clear" is a statement about what a byte does at a stage clear; on a ROM where
no stage clear has ever occurred in this repo, that statement came from
somewhere else, whatever the entry says about itself. A `VERIFIED WIN LATCH`
tag on a bout that has never ended is still imported knowledge — the tag is
part of the claim, not evidence for it.

Two corollaries kept the sweep from eating good work:

* **An honest null is not a breach.** `kid_icarus.yaml`'s "Held 0 across 300k+
  frames under every policy" is a measurement of absence, correctly reported.
  It stays. What does not stay is a *false-positive rate* asserted on top of
  that null — "never a false positive" cannot be known from zero observed
  increments.
* **Over-withdrawal is its own defect.** Stripping a receipted entry to look
  rigorous destroys real work and is the same error wearing the opposite mask.
  The default was to leave an entry alone unless its semantic claim could be
  shown to reference an event with no witness in this repo.

## What the sweep found

| | count |
|---|---|
| configs carrying a live, int-parseable `ram_mapping` | 101 |
| int-parseable entries across them | 994 |
| entries carrying any annotation at all | 139 |
| annotations asserting **event-tied** semantics (clear / win / boss / completion / level transition) | 23 |
| of those, referencing an event with **no witness in this repo** | 17 |
| further unwitnessed-event claims found outside the inline-comment scan | 2 |
| **entries quarantined** | **7** |
| **entries downgraded** (annotation rewritten to what was measured) | **24** |
| **entries left alone** | **963** |
| configs touched | 10 |

Method: a keyword scan over each entry's attached comment flagged 31
candidates; hand review discarded 8 as false hits (a comment matching "floor
~176" or "after kill runs" is not an event claim) leaving 23 real event-tied
annotations. Two more unwitnessed-event claims sit outside that scan's reach —
`megaman.yaml`'s boss-HP claim lives in the file header rather than an inline
comment, and `ducktales.yaml`'s `level_index` names levels 1-4 without using
any flagged keyword. The downgrade set is larger than the unwitnessed-claim set
because it also absorbed five entries rewritten for a different defect
(addresses this repo measured dead, and an unreceipted weapon enum).

## The audit came back mostly clean, and that is the headline

**963 of 994 entries were left exactly as they were.** 84 of the 101 configs
are the SMB family — 839 entries, every one grounded in a game this project has
completely solved, with a single cold-boot tape, 31,202 steps, all 32 levels,
and an ending frame anybody can open. `mario.yaml`'s "0x03 = climbing flagpole
(level clear)" is a level clear this repo has watched 32 times. Not one SMB
entry was touched.

The confirmed-clear profiles held up too, and they held up *because* their
annotations describe what was seen:

* `tetris_b.yaml` — 4 event-tied annotations, all KEEP. `clear_counter` reads
  "[MEASURED: flat at 0 across 6,000 samples of no-clear random play; monotone
  non-decreasing 0x01 -> 0x23 across the 24-clear run]". That is the shape a
  witnessed claim has.
* `excitebike.yaml` — "reaching 3 == FINISH LINE crossed. THE win signal",
  backed by a rendered finish-line frame and a 1144-action power-on tape. KEEP.
  A preliminary pass had proposed withdrawing `surface_status` on the grounds
  that it was steering the discovery instrument; that justification was false
  (see below) and the entry was left alone.
* `bubble_bobble.yaml`, `ghosts_n_goblins.yaml`, `metroid.yaml` — untouched.
  `metroid.yaml` is the model: it already quarantined its external material and
  labels `area_number` "the 'region index' LABEL is not yet re-derived".
  `ghosts_n_goblins.yaml` records its missing stage byte as an explicit
  NOT FOUND rather than guessing one.

A sweep that finds little is a good outcome. The breach here is real but it is
narrow, and it is concentrated exactly where the witness ledger says it would
be: on profiles with no witnessed clear that nonetheless annotate clear
semantics.

## Quarantined (7 entries, 5 configs)

Each moved into `quarantined_external_knowledge:` on the `configs/zelda.yaml`
pattern — `q_*` **string** values so `int()` raises, plus `provenance`,
`status: UNVERIFIED_EXTERNAL`, and a `rediscovery_rule` naming the event that
has to be witnessed before the byte can come back.

| config | key | addr | the event with no witness |
|---|---|---|---|
| `contra.yaml` | `q_current_level` | `0x0030` | a Contra stage clear |
| `contra.yaml` | `q_boss_defeated` | `0x003B` | a Contra stage-boss death |
| `contra_blank.yaml` | `q_current_level` | `0x0030` | (same block, byte-identical) |
| `contra_blank.yaml` | `q_boss_defeated` | `0x003B` | (same block, byte-identical) |
| `punchout.yaml` | `q_match_id` | `0x0001` | a Punch-Out bout win |
| `megaman.yaml` | `q_boss_health` | `0x06C1` | reaching a Mega Man 2 boss room |
| `castlevania.yaml` | `q_boss_health` | `0x04A0` | a Castlevania boss fight |

Why these seven and not others: each asserted an unwitnessed event's semantics
**as established fact, with no disclosure that the event had never been seen**.
`boss_defeated: 0->1 when the current stage boss dies` states a mechanism;
`match_id: VERIFIED WIN LATCH ... THE win` states it and certifies it. Contrast
the entries that were downgraded instead — `kungfu.yaml`'s `current_floor`
already said "CROSS-SOURCED (design brief live pass + datacrystal)",
`double_dragon.yaml`'s `mission` already said "Could NOT be driven to increment
... so its INCREMENT is unverified", `kid_icarus.yaml` already said "transition
not reached under blind play". Those disclose. The defect there is a single
unmeasurable clause bolted onto an honest null, and the fix is to delete the
clause, not the entry.

Three of the seven are strongly falsified rather than merely unwitnessed:

* **`contra` `boss_defeated`** — `docs/research/CONTRA_WALL_2026-08-27.md` row
  A2: 81,244 alive steps, 830 zero-hits, 15 replay-verified, **0/15 sustaining
  ≥20 zero steps**; HP refills within 1-2 steps once fire stops. Joint-zero is a
  live-suppression multiplexing artifact, not a kill.
* **`punchout` `match_id`** — `runs/clear_gap/punchout/s8_remeasure_verdict.json`
  reproduced the falsifier byte-for-byte: from the archived knockdown state
  (`opp_hp=0, opp_down=1`) a pure-NOOP continuation refills `opp_hp` 0→96 at
  step 120 while `match_id` stays constant 0. That receipt calls the claim "a
  purity leak dressed as an empirical find". `runs/fight_gate/smoke/solutions/`
  is empty.
* **`castlevania` `boss_health`** — graded `churn 0.0/1k` in
  `docs/receipts/ram_verify/castlevania.json`; commit ad01b4f already recorded
  it reading 0 through live fights and left it unconfigured.

## Downgraded (24 entries, 8 configs)

Annotations rewritten to what was actually observed, entries kept. The
substantive ones:

* **`contra` / `contra_blank` / `contra_screen9`** — the block header claimed
  its addresses were "emulator-verified during the win-verification pass"; no
  Contra win exists to have verified them. `player_x`'s aside that `$0031` "is
  the game-completion COUNT" is an identity claim for whole-game completion
  resting on a byte all we know about is that it reads constant 0 — which is
  consistent with anything. `screen_number`'s "RESETS to 0 each stage"
  describes a stage boundary this repo has never crossed. `player_y` and (on
  `contra_screen9`) `player_x` are measured dead by
  `docs/receipts/ram_verify/contra.json`. `player_state` is contested: the
  receipt and a later purpose-built real-death test disagree and neither was
  re-run, so the comment now says so instead of picking a winner. The weapon
  enum has no receipt and `docs/research/CONTRA_ROUTE_A_2026-08-27.md`
  deliberately declines to say what any `$00AA` value means.
* **`double_dragon`** — `scroll_coarse` was **rejected by this profile's own
  discovery receipt** ("camera/parallax jitter that only LOOKS monotone",
  raw series `16,16,32,32,96,48,160,176,96,32,240,48,...`) while the annotation
  still advertised it as opt-in progress. `mission` keeps its honest null and
  loses "never a false positive". `hearts` keeps the measured coupling to
  `0x0042` and loses the Game-Genie-sourced "move unlock".
* **`kungfu`** — `current_floor`'s null is exact and stands (0x0058 == 0 in
  32/32 banked cells, ~719,500 steps). "It can only rise on a real floor clear"
  is retracted: unfalsified is not verified.
* **`kid_icarus`** — null stands; "an unambiguous stage clear (never a false
  positive)" retracted. The comment now flags that the win predicate keys on
  this byte and that every clear it reports is UNCONFIRMED.
* **`castlevania` `hearts`** — not withdrawn, because the case against it does
  not hold up. `nes_core/src/rewards.rs` claims `0x0045` "is actually Simon's
  real HP" while declaring `RAM_PLAYER_HEALTH = 0x0044` three lines above, and
  its replacement `0x0071` has no receipt anywhere under `docs/` or `runs/`.
  Two in-repo readings conflict; the annotation now records the conflict.
* **`ducktales` `level_index`** — the value-to-level-name table for levels 1-4
  was cross-sourced; no level past the start level has been reached here.

## Did any banked claim depend on a quarantined entry?

**No banked result is retracted by this change, and no solver gate breaks.**
Verified rather than assumed:

* `solve.level_key` is **empty** on `contra`, `contra_blank`, `punchout` and
  `megaman`, so none of the four quarantined-from-solve addresses was ever a
  cell key. `castlevania`'s `level_key` is `[0x0028]` (`stage_number`), which is
  untouched — its clear chain across five run trees is unaffected.
* No quarantined address appears in any `solve:` block or in
  `reward_weights.*_addr` on any profile.
* `scripts/observatory.py` does not read these values into anything that gates
  search (see below).

**But two of the seven are still live in the engine, and that must be said
plainly.** `nes_core/src/rewards.rs` hardcodes its own copies:
`RAM_MATCH_ID = 0x0001` drives Punch-Out's `won` / `done` / `episode_success()`,
and `RAM_BOSS_HEALTH = 0x06C1` drives Mega Man's boss term. Quarantining the
YAML entry retracts the *documentation* claim; it does **not** disarm the Rust
constant. Neither has ever fired — Punch-Out has never won a bout and Mega Man
has never reached a boss room — so nothing banked rests on them today. The
honest statement is that this sweep covered `configs/`, and the same class of
claim lives in the executing layer uncovered.

**Named follow-up.** `nes_core/src/rewards.rs` carries unwitnessed semantics of
its own, including a Kung Fu header block describing floor-by-floor boss
progression as "RAM verified live on this ROM" for a byte measured at 0 in 32/32
banked cells, and a Castlevania `RAM_BOSS_HEALTH = 0x01A9` sourced to
"Data Crystal + tasvideos ... not yet Dracula-fight-verified". A configs-only
sweep certifies nothing about that layer. It should get the same discriminant.

## Correcting the premise this sweep was launched on

The brief opened by stating that `scripts/observatory.py` folds every
int-parseable `ram_mapping` value into its pre-probe exclusion set
(`mapped = {int(a) ...}` then `excluded |= known`), steering the discovery
instrument away from exactly the bytes quarantine exists to force a
rediscovery of.

**That is no longer true, and it was already fixed before this sweep began.**
`_mapping_bytes()` is private and reaches `main()` only behind an
`is_known(addr)` predicate, which exists to tag receipt rows `"known": True/False`
and cannot be unioned into anything. The exclusion set is `coord_bytes | stack |
OAM | mirrors` and nothing else; the receipt's own exclusion log records the
`ram_mapping` region with `"excludes": False`. The fix landed 2026-08-26 in
`eb8174b` / `2e6014f` and is guarded by `tests/test_observatory_exclusions.py`.

So **no config in this tree was steering the discovery instrument**, and none of
the seven quarantined entries ever did. This does not change any verdict — the
discriminant is about whether a claim could have been measured here, not about
which mechanism carries the harm — but it changes the urgency, and a report that
repeated a stale premise as live would be making the same mistake it is auditing.

## The guard

`tests/test_purity_quarantine_sweep.py` gains two parametrized tests over the
seven retractions:

* `test_unwitnessed_semantic_stays_quarantined` — the block still exists, is
  scoped to the right ROM, still carries `status`, `provenance` and
  `rediscovery_rule`, and holds the address **as a string on which `int()`
  raises**.
* `test_unwitnessed_semantic_is_not_live_on_any_profile_for_that_rom` — no
  profile on that ROM carries the address in a live `ram_mapping`.

The generic sweep already in this file derives its address set from whatever
quarantine blocks happen to exist, so deleting a block makes it pass
vacuously — there is nothing left to compare against. Naming the seven
explicitly closes that.

**Anti-vacuity, verified by actually reverting** (six vacuous gates have shipped
in this repo; a claim of non-vacuity is worthless without the receipt):

* Restoring all five quarantine configs to `HEAD`: **14 of 14 new assertions
  fail**, 320 pass.
* Restoring only `contra_blank.yaml` while leaving `contra.yaml` quarantined:
  **7 fail** — including the `contra.yaml` row, because the address is still
  live on a sibling profile for the same ROM. The two files carried
  byte-identical `ram_mapping` blocks (verified with `diff`), so a half-fix
  retracts nothing, and the guard says so.
* With the change in place: 334 passed.
