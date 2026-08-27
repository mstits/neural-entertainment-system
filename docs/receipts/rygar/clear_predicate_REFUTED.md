# Rygar clear predicate — **REFUSED**, with the measurement that refuses it

**Date:** 2026-08-26
**Ledger:** **EXHIBITION.** Everything below is Go-Explore search output plus
scripted and uniform-random rollouts. No policy was trained for this game and
no honest-protocol evaluation was run. Nothing here may be described with "the
AI learned", "the AI plays", or "the AI beat" (`CLAIMS.md`).
**Purity (Tier 3):** every observable below was derived by blind statistical
search over this profile's own rollouts — the PPU blank-fold counter, and a
RAM address found by asking which bytes are constant inside every
inter-blackout segment. No disassembly, no RAM map, no walkthrough, no recall
of this title. Litmus: a party who had never seen this game runs
`python scripts/transition_witness.py discover` and gets the same list.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc`.

---

## 1. The task, and the answer

> Mint Rygar a clear predicate. Rygar can search to 4,608 px and cannot
> recognise winning.

**Refused.** Not for want of an instrument — the instrument now exists and
works — but because every predicate the instrument makes available **fabricates
wins on Rygar's own deepest banked tape**, and because no Rygar clear has ever
been witnessed, so there is no positive to calibrate against.

Two things changed today, and they point opposite ways:

| | Before | Now |
|---|---|---|
| Can any instrument count Rygar's rooms? | No — `RYGAR_CAMPAIGN_2026-08-26.md` §6: *"No instrument in the pipeline can count them"* | **Yes.** 55/55 transitions, 3 areas, agreeing with §6's hand count |
| Can a clear predicate be built on it? | Unknown (R1-06 **DECLINED**, no numbers) | **No — REFUTED with numbers.** §4 below |

---

## 2. The instrument: `odo_blank`, edge-detected

`ppu.rs::odo_fold_frame` carries two counters and they count different events.
`odometer_scene` bumps on a *rendered* scroll cut. `odo_blank` bumps on a
mostly-blank frame — and the blank branch `return`s **before** the scene-cut
test ever runs. Rygar's transitions are blackouts, so `odometer_scene` reads
**0 cuts across the entire 6,018-action tape** and the campaign concluded the
pipeline was blind. It was blind in one eye.

Measured on the banked R1 tape (`r1_tape_gx6242.json`, replayed at HEAD):

```
terminal odo_blank        4329        <- BLANK FRAMES, not transitions
edge-detected runs          56        = 1 boot fade + 55 doors
run lengths            {9: 1, 78: 28, 79: 27}
odometer_scene               0        <- the counter this replaces
```

**Never read the raw counter as a count.** 4,329 is 78x the true transition
count. The run count is the transition count.

### The two guards, and what each is worth

Measured over 18 rollouts (scripted `right` / `noop` / `left` plus 15 seeded
uniform-random), every one of which died and sustained the death for 500+
observations:

```
death fade         14 blank frames, 36 of 36 (the fade + the game-over reload)
door transition    78-79 blank frames, 55 of 55
boot fade           9 blank frames, 18 of 18
```

The raw counter **does not** discriminate: it takes the same branch for all
three. Two independent guards do.

1. **Run-length floor, 40 blank frames.** 2.9x above the death fade, 2.0x
   below the door. This is a per-game number derived from this game's own
   rollouts. It is not a property of the counter and it must be re-earned per
   game.
2. **Lives debounce, ≥ 3 observations.** Rygar's lives byte blips to 0 for
   exactly **2** observations at every door, 55 of 55, and pins for 500–1,258
   at a real death.

Both are load-bearing, each proven by removal on real streams
(`tests/test_transition_witness.py::TestGuardsAreLoadBearing`):

| Guard removed | Misfire |
|---|---|
| floor → 1 | 18 boot fades banked as transitions (was 0) |
| floor → 1 **and** debounce off | 54 death/boot fades banked as transitions (was 0) |
| debounce → 2 | **all 55 real transitions destroyed**, misfiled as deaths |
| novelty memory cleared | 55 "new areas" instead of 2 — the naive count |

### It survives the solver

`odo_blank` rides **inside** `OdoState`, so it is a per-trajectory count, not a
monotone global: a Go-Explore restore carries the saved value back in.
Measured on a real save/restore round trip mid-tape, the counter jumps
273 → 90. The witness reports SPLICE and banks nothing. Replayed under the
clear detector's own differential probe shape at **602 restores**, the tape
still reads 55 transitions / 2 novel / 3 areas, with `odometer_x` unperturbed
at 6,242.

### The area key, found blind

Which RAM bytes are constant inside every one of the 56 inter-blackout
segments and vary across them? Exactly **8**, out of 2,048. Ranked by distinct
values:

```
$001C  3 values   14, 16, 29, 16, 29, 16, 29, ...
$0014  2 values    0,  3,  0,  3,  0,  3,  0, ...
$014B, $0128, $0130, $00B0, $0048, $0782
```

`($0014, $001C)` reads **3 distinct areas** across the whole tape. §6 of the
campaign counted "at least 3 visually distinct areas" **by eye** and recorded
that no instrument could count them. The instrument and the eye agree.

The pixels agree too. A RAM key that partitions segments is only an *area* key
if the screen says so, so `discover` cross-checks it: cluster each segment's
midpoint frame by the key, compare within-cluster spread to between-cluster
distance (`python scripts/transition_witness.py discover`, last block).

```
area (0,14)    1 segment   intra   0.00
area (0,29)   27 segments  intra   3.96
area (3,16)   28 segments  intra  28.52     <- the long scrolling corridor
inter (0,14) vs (0,29)  72.28
inter (0,14) vs (3,16)  53.36
inter (0,29) vs (3,16)  78.98
max intra 28.52 < min inter 53.36
```

`(3,16)`'s intra of 28.52 is not noise: it is a single continuously scrolling
area spanning x 1,536 → 6,242, so its own frames legitimately differ end to
end. The bar that matters is that the **largest** intra still sits below the
**smallest** inter, and it does, with a 1.9x margin.

---

## 3. What the instrument says about the tape

```
observations   6019      transitions   55      areas    3
novel           2        revisit       53      deaths   0
short runs      1        splices        0
```

**Two novel areas over 6,018 actions.** That is the honest count of new ground
on the deepest Rygar trajectory this project has produced.

---

## 4. Why no predicate is minted — the refutation

### 4a. "A transition happened" fires 55 times on a corridor

The tape's 55 transitions are **27 round trips through one door**. The area
key alternates perfectly — `(3,16), (0,29), (3,16), (0,29), …` — for all 55,
and the receipt's independently-measured ratchet accounting agrees exactly:
27 zero-gain post-door segments, 27 gaining ones. The trajectory's
artifact-free frontier was 4,608 px before those 54 segments and 4,608 px
after.

A predicate of the form *"a transition happened"* would have banked **55
solutions on a tape that never left the corridor**.

### 4b. A `level_key` on the discovered area byte fires 28 times

`GenericGame.is_clear` fires on `level_key(ram) > tuple(start_key)`. The
discovered area byte is the only `level_key` candidate Rygar has. Wire it and:

```
key-change events on the tape                55
of which lexicographically forward           28   <- 28 fabricated wins
observations on which the predicate is TRUE  4978 of 6019 (83%)
same, for a one-byte `level_key: [0x14]`      3725 of 6019 (62%)
```

It is not merely over-firing; it **latches**. A solver checking `is_clear`
every observation banks on 62-83% of the tape depending on the key width.

### 4c. There is no positive to calibrate against

This is the decisive one, and no amount of instrument work fixes it.
**No Rygar clear has ever been witnessed** — not on this tape, not in any
banked archive, not by eye. Every `solutions: 0` in every Rygar archive is a
compile-time constant (`is_clear` reduces to `() > ()`), re-verified over 2,000
random RAM states. Re-counted today: **71 of 71** `solutions/` directories
under `runs/` for this profile are empty, and not one of them is evidence of
anything.

A predicate with no witnessed positive cannot be shown to fire when it should.
It can only be shown not to fire — which is what four vacuous gates did this
week. Minting one here would be the fifth, and it would be the worst kind,
because a false clear does not merely fail to detect: it **fabricates a win**,
which this project bans outright.

### 4d. Anti-vacuity: the tests were verified to fail with the mechanism gone

Four vacuous gates shipped this week, so "34 tests pass" is not evidence. Each
guard was deleted from `scripts/transition_witness.py` in turn and the suite
re-run. Every mutant is killed, each by a **different** test:

| Guard deleted | First test that fails |
|---|---|
| splice detection (`delta < 0 or delta > frames_per_step`) | `TestSplices::test_a_decrease_is_a_splice_not_a_transition` |
| run-length floor (`frames >= min_blank_frames`) | `TestTheWitnessOnRealRygar::test_it_counts_every_one_of_the_55_transitions` |
| lives debounce (`max_dead >= death_debounce`) | `TestTheWitnessOnRealRygar::test_it_does_not_fire_on_a_real_death` |
| novelty memory (`key in self.seen`) | `TestTheRatchet::test_53_of_the_55_transitions_arrive_somewhere_already_seen` |
| UNINSTRUMENTED reporting | `TestMechanismAbsent::test_a_frozen_counter_reports_uninstrumented` |
| UNAVAILABLE short-circuit | `TestMechanismAbsent::test_a_disabled_odometer_reports_unavailable_not_zero` |

The source was restored byte-identical afterwards (verified by `diff`).

---

## 5. What a Rygar clear predicate would actually require

In order. Each step's exit criterion is falsifiable.

1. **A witnessed positive.** A trajectory that reaches an area outside
   `{(0,14), (3,16), (0,29)}`, i.e. one `novel` event past the third area,
   with the rendered frames banked. Until then there is nothing to calibrate.
   *This is currently blocked:* a 14-hypothesis campaign produced zero forward
   progress past 4,608 px, so the blocker is the search wall, not the
   predicate.
2. **A repeat of that positive**, from a cold start, with the tape replaying
   deterministically — otherwise the "new area" may be a glitch state.
3. **A discriminant that fires on it and not on the 55 known negatives.** The
   witness is already the harness for this: the banked streams in
   `transition_streams.json` are the negative corpus, and any candidate must
   score 0 on them before it scores 1 on the positive.
4. **Only then** a `clear:` block in `configs/rygar.yaml` — and never a
   `level_key` on the area byte, for the reason in §4b.

Until step 1 lands, the honest state of `configs/rygar.yaml` is exactly what it
is today: `level_key: []`, no `clear:`, no `finale:`, and R1 condition 2 still
**FAIL**. That FAIL is now backed by measurement rather than by an absence.

### The standing risk on the instrument

**Death-with-respawn is untested.** Rygar's lives byte reads 1 at this start
state and never exceeds 1 anywhere on the tape, so the only death class
reachable without injecting game knowledge is death → game over, whose fade is
14 frames. A respawn that reloads the level could plausibly blank for a
transition-length interval, which is exactly the case the 14-vs-78 margin does
not cover. The lives debounce is the guard that would have to hold there, and
it has not been exercised on that shape. Named, not hidden.

**And the length separation is Rygar's, not the cohort's.** Whether 14-vs-78
survives on any other game is untested. The floor must be re-derived per
profile from that profile's own rollouts, or the instrument reports nothing.

---

## 6. Receipts

| What | Where |
|---|---|
| The mechanism | `scripts/transition_witness.py` |
| Its tests, incl. every guard-removal | `tests/test_transition_witness.py` |
| Banked observation streams (tape + 18 death rollouts, RLE) | `docs/receipts/rygar/transition_streams.json` — **tracked** |
| The tape being replayed | `docs/receipts/rygar/r1_tape_gx6242.json` |
| Blind area-key discovery | `python scripts/transition_witness.py discover` |
| Re-bank the streams | `python scripts/transition_witness.py bank` |
| The campaign this corrects | `docs/research/RYGAR_CAMPAIGN_2026-08-26.md` §6, §11.2 |
