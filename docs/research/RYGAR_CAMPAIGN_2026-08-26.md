# Rygar R1 campaign — verdict, receipts, and the instrument defect it found

**Date:** 2026-08-26
**Ledger:** **EXHIBITION.** Everything below is Go-Explore search output. No
policy was trained and no honest-protocol evaluation was run for this game.
Nothing here may be described with "the AI learned", "the AI plays", or "the AI
beat" — see `CLAIMS.md`.
**Verdict:** **R1 FAIL** against the bar pre-registered before compute.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc` (HEAD, and the installed
`.venv` `.so` — verified identical).

---

## 1. The question the campaign was asked

> Play one game really well. Not measure it, not onboard it. Play it.

**Can we play Rygar really well? No — not yet.** We can search it far, we can
replay what we find deterministically, and we can prove the agent is alive the
whole way. We cannot yet recognise winning, and roughly a quarter of the
headline distance turned out to be an instrument artifact rather than travel.

**Did the wall move? Yes, substantially — and not today.** The wall moved
earlier this week and this campaign's job turned out to be finding out where it
actually is. See §4.

---

## 2. The pre-registered bar, and the result on each condition

The bar was fixed on 2026-08-26 *before* further compute. It was not moved
afterwards. All four conditions were required.

| # | Condition | Bar | Measured | Verdict |
|---|---|---|---|---|
| 1 | **DEPTH** | ≥ 9,000 odometer x from power-on | 6,242 raw instrument; **4,608 artifact-free** | **FAIL** |
| 2 | **CLEAR PREDICATE** | wired signal passing an anti-vacuity triple | none wired; R1-06 **DECLINED** | **FAIL** |
| 3 | **REPRODUCIBILITY** | replays 3/3, ±16 px, alive at terminal | 6,242 / 6,242 / 6,242, terminal alive | **PASS** |
| 4 | **LIVENESS** | no lives-0 run ≥ 3 observations | longest run = 2; histogram `{2: 55}` | **PASS** |

**Overall: FAIL.** Not VOID — a real, live, deterministic tape was produced and
tested against every condition. It simply falls short on two of four.

Condition (2) failing is the structurally important one. `configs/rygar.yaml`
carries `level_key: []` with no `clear:` or `finale:` hook, so `is_clear` is
`() > ()` — **False for every state**. I confirmed this is a compile-time
constant, not an empty search result: `is_clear` and `is_finale` return False
over 2,000 random RAM states each, and every `solutions/` directory is empty.
**`solutions: 0` in any Rygar archive is evidence of nothing** and must never be
cited as a search result.

---

## 3. What I verified myself before landing this

I replayed the candidate tape from power-on with a harness written fresh for
this write-up, importing nothing from any campaign item's own verifier
(`runs/rygar_campaign/_LANDING/`).

```
n_actions                6018        terminal_x            6242
max_x                    6242        lives at start           1
dead_run_histogram   {2: 55}        longest_dead_run          2
terminal_alive           True        scene_cuts_total          0
blackout_runs              56        run lengths     {20: 55, 2: 1}
first step reaching 4608 2632        px gained after 4608   1634
x non-decreasing frac  0.9004
```

Every number in the campaign's own receipts reproduced exactly. **No item in
this campaign fabricated or inflated a number.**

Three notes that matter more than the totals:

- **55 blackouts of exactly 20 fs4-steps each, and 55 lives-0 blips of exactly
  length 2 — one per blackout.** The blips *are* the transitions. Several items
  reported "blips occur on a flat, uncut camera"; that is wrong.
- **`scene_cuts_total = 0`** across the whole tape. The odometer's scene counter
  is structurally blind here (§5), so every `rooms_reached: 1, scene_cuts=0`
  claim in the campaign is an inference from an instrument that cannot fire.
- **Contamination checks came back clean.** Of 8,444 px of total positive
  odometer gain, only **14 px accrued while the death predicate was true**
  (0.17%) — against Contra's ~94%. The death discriminator is demonstrably able
  to fire *and stay fired*: from power-on, holding `right` dies at step 138 and
  pins the lives byte at 0 for **5,862** consecutive observations, and a noop
  hold dies at step 279 and pins for **5,721**; forked from inside this tape,
  `right` and `left` pin for 444 and 391. Against a transition blip of exactly
  **2**, that is a separation of roughly three orders of magnitude. There is no
  frozen-GAME-OVER inflation on any tape here.

---

## 4. Did the wall move?

**Yes — from 1,536 px to 4,608 px of verified first-visit territory, a 3.0×
move, and 9.9× the forward-hold probe.** That is real progress and should be
reported as such.

| Milestone | Depth | What it is |
|---|---|---|
| Forward-hold probe | **467 px** | scripted `right` hold; walks into a hazard and dies at step 138 |
| Pre-debounce solver wall | **1,536 px** | pinned here for 45 min across two run links — this is the **first door** |
| Post-debounce, HEAD, artifact-free | **4,608 px** | screen-verified first-visit territory |
| Raw instrument headline | 6,242 px | 1,634 px of which is not travel (§5) |

**The lever that moved it was the death-blip debounce, not compute.** Real
deaths pin the lives byte at 0 for 5,721–5,862 observations; transition blips
are exactly 2 observations. That is roughly a 2,900× separation, with the
shipped `≥ 3` threshold sitting on the boundary with a full step of margin.
Before the debounce, every door read as a death and the search could not leave
room one — pinned at exactly 1,536 for 45 minutes. After it, the same search
reached 5,360 in **six minutes**. The remaining four hours of that banked chain
bought +320 px and ended flat.

**But this campaign did not move the frontier.** It moved the raw instrument
number from 5,893 to 6,242 (+5.9%) — and every pixel of that +349 is ratchet.
The artifact-free frontier was already 4,608 before the fourteen items ran, and
it is 4,608 after. Twenty-five minutes of compute bought zero new ground. That
is a plateau, and it is recorded here as a plateau.

---

## 5. The campaign's actual finding: the odometer ratchet

`nes_core/src/ppu.rs::odo_fold_frame` sets `odo_have_prev = false` and returns
on any frame rendering fewer than 120 lines. The source comment states the
intent plainly — *"a respawn therefore FREEZES the odometer rather than
rewinding it"* — and for a respawn that is correct. But Rygar's room
transitions are **blank-type**, so the same branch fires at every door, and it
means the odometer **cannot see a camera reset across a transition**. Every
round trip through a door banks the forward scroll and silently discards the
return.

The search found this ratchet and has been riding it.

### The measurement, with its positive control

I cut the tape at its 55 blackouts and asked, per segment: how much odometer did
it bank, and what was on screen at its end?
(`runs/rygar_campaign/_LANDING/segment_test.json`)

```
post-door segments                    54
  banking dx = 0                      27   ← the far room, always zero
  banking dx > 0                      27   ← the near room, +53..+64 px (mode 64)
  total banked                     1,621 px
  distinct end-screen clusters           9   across all 54 segments
  segments reusing an earlier cluster   45   of 54

POSITIVE CONTROL — the uninterrupted pre-door segment, x 1,536 → 4,608,
cut into 51 pieces of the same length:
  distinct end-screen clusters          51   of 51
  repeats                                0
```

**Exactly 27 zero-gain segments and exactly 27 gaining segments** — a perfect
alternation, i.e. 27 round trips through one door.

The positive control is the part that makes this a finding rather than an
assertion: the same clustering test, run on genuinely new ground, says
"different" **51 times out of 51**. A second cut at fixed 64-px milestones
agrees — 71 milestones before the door give 70 distinct screens with a median
45.8% of pixels differing, while 7 of 26 milestones after the door are
pixel-identical to an earlier one. The test can fail, and before the door it
always does.

### The bar was denominated in a farmable quantity

This is a defect in the bar itself, recorded here so it is not repeated. Twenty-
seven door cycles bought 1,621 px, about 60 px per cycle. Reaching the
pre-registered 9,000 from 6,242 needs 2,758 px — roughly **46 more door
cycles**, no new ground required. Any successor bar must denominate depth in
**first-visit territory**, not in a global odometer integral.

### The fifth vacuous gate of the week

R1-08 discriminated loop-from-advance by asking whether `gx` ever *decreases*
across a revisit to the same room ordinal. Because the odometer re-anchors
instead of integrating at every transition, **`gx` cannot decrease across a
revisit** — the check returns REFUTED whether or not the loop exists. Its own
headline evidence, "room 219: 4672, 4736, 4791, 4855, 4910, 4974, 5029, 5089,
5148", *is* the ratchet ladder, transcribed and read as progress. That
conclusion then propagated into R1-14's `beat_prior_best: true`.

This belongs in the anti-vacuity census. The working non-vacuous replacement is
cheap and is in the receipts: cluster rendered frames at fixed milestones, with
uninterrupted-segment milestones as the control that proves the detector can say
"different".

---

## 6. Rooms reached

**At least 3 visually distinct areas from power-on**, counted by hand from our
own rendered frames: the start ridge, an orange/grey cavern, and a green stone
hall. **No instrument in the pipeline can count them.** `odometer_scene` reads
0 cuts across the entire tape because `odo_fold_frame`'s blank branch returns
before the scene-cut test ever runs, and `room_fp`'s settle mechanism was
declined on measurement (§7). The honest statement is "at least 3, counted by
eye from our own rollout, not by any instrument."

---

## 7. The fourteen items: what was tested and what came back

Negative results are the bulk of this campaign and are the most useful part of
it. Every one below was replay-audited.

| Item | Lever | Result |
|---|---|---|
| R1-01 | step-budget sweep | **VOID** — no arm reached its own cap, so the variable never engaged |
| R1-02 | step efficiency | **Real lever.** hold-macros +82% px/step on a length-matched window; `burst 256` 17% *worse* than default |
| R1-03 | cell resolution | **REFUTED** — 10× the cells (129 → 1,335), `gx` flat at 2,400–2,560 in 19 of 24 runs |
| R1-04 | velocity-signed cells (SMB 4-4 recipe) | **REFUTED**, 2 seeds, with mechanism — the domination score is a pure function of `gx` and never reads the spliced slot |
| R1-05 | room identity via nametable fingerprint | **DECLINED with numbers** — 649 distinct hashes / 3,490 frames; mask calibration found 0 volatile bytes; no settle in [3,60] yields a stable transition-aligned room set |
| R1-06 | clear predicate | **DECLINED** — correctly refused to arm a predicate that could not be shown to fail |
| R1-07 | blackout tax on the step budget | **Prototype, not landed** — 2,451 → 5,250 under identical contention for equal compute |
| R1-08 | loop-or-terrain | **OVERTURNED** — its refutation was the vacuous check above |
| R1-10 | orthogonal arm / inversion pin | **REFUTED** — `ODO_ALT` into the `y` slot inherits that axis's measured flatness by construction; shortening the inversion pin made the one real wall *worse* |
| R1-11 | selection policy | **Self-refuted honestly** — selection is *not* ignoring the frontier (75.6% of selections land in the frontier band); it is blind to *remaining budget* |
| R1-12 | action granularity | **Weak** — 4 of 7 configs plateau at exactly 1,536, the first door, so they cannot be ranked |
| R1-13 | liveness audit | **Confirmed** — debounce is sound; 2 vs 5,721–5,862 observation separation |
| R1-14 | R1 receipt | **FAIL**, correctly — but its `beat_prior_best` was ratchet |

### R1-07's prototype carries a warning

Its currency is precisely blackout steps, and blackout steps are the carousel's
mechanism. Not charging them structurally **subsidises door cycling**. It should
not be landed on the strength of its 5,250 headline; audited, that tape's
first-visit depth is ~4,883, and its gain is partly a more efficient ratchet.

---

## 8. Why not Contra, Kung Fu, or Zelda

Measured at HEAD today by re-running `scripts/progress_signal_gate.py` on each
profile against its own declared progress signal. All four reproduced exactly.

### Rygar — **PASS, SIGNAL SOUND — still advancing** (the reason it was chosen)

```
138 live steps, 116 distinct, range 0..467, high byte=True, lives@start=1
[D5] 1062 of 1200 steps dropped — the hold ran off the end of live play
signal is usable: enough resolution, no unpaired wrap, still moving late,
death detectable
```

The "138 live steps" is a property of the **gate's scripted forward-hold probe**,
not of the game. It holds `right` from a `lives=1` start and walks into a
hazard. Under any non-scripted policy the window is far longer: uniform-random
survives a median 677 steps, and the solver's own deepest lineages run
3,865–6,018 actions in **one continuous life with zero terminal deaths**. Cite
the 138 only as "the forward-hold probe's survival", never as the profile's live
window.

### Contra — SIGNAL UNUSABLE, **but not fairly excluded**

```
69 live steps, 20 distinct, range 0..70, high byte=True, lives@start=2
[INSTRUMENT] only 20 distinct values in 69 steps (< 32)
[BEHAVIOUR]  flat for the last quarter of the roll
[D5] 1131 trailing steps of the requested 1200 were dropped
```

**Open defect, reported not fixed:** the gate computes its resolution
INSTRUMENT finding on the window *after* D5 truncation. A 69-sample window
cannot demonstrate a 32-distinct threshold — that verdict measures how fast the
forward-hold probe died, not the signal's resolution. **Contra's honest verdict
is INCONCLUSIVE**, pending a probe that survives long enough to assess. Rygar's
PASS is unaffected (116 distinct in 138 live steps clears 32 with room), so the
game choice stands on Rygar's own merits — but Contra deserves a re-run.

Also withdrawn earlier this week (commit `c146769`): Contra's banked "odometer
162 distinct, cross-validated 162 vs 163". About 94% of that 162 was game-over
animation, and the "163" has no receipt anywhere in the tree. Contra's
documented wall "pinned at gx 3072 across six campaigns" was measured on that
contaminated signal.

### Kung Fu — SIGNAL UNUSABLE, on **both** axes

```
RAM byte $0094:  1200 steps, 91 distinct, range 0..240, high byte=False
[INSTRUMENT] reaches >=200 with no paired high byte

Odometer axis:   1200 steps, 1 distinct, range 0..0, OAM churn 628/1199
[INSTRUMENT] camera never moved over 1200 steps (agent active (OAM moving))
```

The odometer result is the decisive one: the camera is provably static while the
agent is provably moving. Kung Fu's wall is the **fixed-screen fight class**,
the same as Punch-Out, and it needs a fight-gate-style observable — not a scalar
position repair. Also withdrawn this week: the "skill wall, not an instrument
fault" verdict, which came from a vacuous gate branch (deleted in `bfb515b`)
that forced `passed=true` whenever OAM churn showed the agent moving.

### Zelda — SIGNAL UNUSABLE **and** purity-blocked

```
1200 steps, 25 distinct, range 86..208, high byte=False, lives@start=253
[INSTRUMENT] only 25 distinct values in 1200 steps (< 32)
[INSTRUMENT] reaches >=200 with no paired high byte
[BEHAVIOUR]  flat for the last quarter of the roll
```

Independently of the instrument, Zelda's win chain came from a disassembly and
is quarantined under the Tier-3 purity rule. It is not a candidate regardless of
what the gate says.

---

## 9. Kung Fu high-byte side quest — **NEGATIVE, and provably so**

This was run because a paired high byte would have made Kung Fu a live
candidate. It would have been a real deliverable. **It does not exist**, and the
following is a proof rather than a failed search.

1. **The instrument was validated first.** `scripts/find_wrap_pair.py` on the
   Castlevania positive control found 256 distinct values and 16 wrap events in
   `$0040`, and blind-nominated `$0041` as its *only* structural candidate at
   14/16 wrap hits — exactly the pair shipped in `configs/castlevania.yaml`.
   Wrap detection and high-byte scoring both work.

2. **On Kung Fu the same instrument nominates a corpse.** It reports `$0094`:
   91 distinct, 4 wrap events, and nominates `$006B` at 2/4 → reject.
   Reproducing the hold exactly, the 4 "wraps" land at steps 180, 390, 854,
   1343, and each is a 236→0 or 216→0 drop sitting on a lives transition
   (deaths at 174, 384, 595, 851, 1189, 1340). They are **death-respawn resets,
   not wraps**.

3. **Zero wraps in live play, at three sample scales.** Six scripted policies ×
   two seeds: `hard_drops = 0` in all 12. Sixty-four parallel stochastic
   episodes: 0 non-death boundary crossings, max `$0094` = 164. And the
   project's own prior receipt — 514,239 steps of Go-Explore reaching
   `max_progress` 244, against `random_baseline_max_progress` **244, identical**.
   Search bought nothing over random: the signature of a hard clamp. With no
   wrap event there is definitionally nothing for a high byte to pair against.

4. **The high byte was never the binding constraint.** Handing `assess()` a
   free, *perfect* high byte still returns `passed=False`, SIGNAL UNUSABLE —
   *"only 28 distinct values in 174 steps (< 32)"*. So `gate_flips = false` is a
   proof, not an absence of evidence.

5. **Exhaustive 2048 × 2048 pair scan** over death-terminated forward / noop /
   reverse holds, requiring net gain ≥ 260, quiet under noop, monotone ≥ 0.95,
   ≥ 32 distinct, and gaining more forward than reverse: **nothing real**. The
   2,511 raw hits before the distinct filter all have `distinct = 3`, artifacts
   of `$000D` flipping 0→145 once and being multiplied by 256.

**Conclusion: Kung Fu does not become a second candidate.** Rygar remains the
correct and only choice of the four.

**And the recorded reason for Kung Fu's failure has a false premise.** `$0094`
never wraps in live play — it is a clamped screen-space coordinate that
saturates at 236–244 and resets to 0 only on death. The true instrument fault is
**coarseness** (28 distinct in a 174-step live window), not an unpaired wrapping
byte. The capability ledger's recorded reason should be corrected so nobody
spends another probe on this "cheap fix".

---

## 10. New defect found: the gate is blind to auto-restarting ROMs

**Reported, not fixed** — changing a binding gate's truncation logic moves many
profiles' verdicts and needs its own review.

Kung Fu's own gate run is post-death-tail contaminated: 1,326 of 1,500 steps are
after the first death, yet the gate reports `dropped_tail_steps = 0` and
`exhaustion.lives_index = null`. **This reproduces at HEAD** — in the runs above,
Contra correctly emits its `[D5]` truncation line and Kung Fu emits none at all,
while reporting a full "1200 steps".

Root cause: `first_exhaustion_index` requires the trailing quarter of the lives
trace to be frozen at exactly one value, but this ROM **auto-restarts after GAME
OVER** — lives run 3→2→1→0 (game over at step 595), then 0→3 at step 851 and
3→0 at 1189. The trailing quarter therefore holds `{0, 3}`, `len(set) == 2`, so
the detector returns `None` and all 1,200 steps are kept. The stasis detector
misses it independently, because an auto-restarted attract/demo game is **busy,
not frozen** (median churn 48 bytes/step, longest frozen tail 0 of 1,200).

Both of the gate's exhaustion detectors are blind to this shape. It is a **third
contamination class** beyond the two documented in the file (D5 Arkanoid
frozen-placeholder, D1 Ninja Gaiden blind-lives-byte), and it plausibly inflates
the "28 of 45 profiles contaminated" count for any multi-life profile whose ROM
loops back to attract.

Cheap targeted fix: treat the **first** death as the truncation point whenever
the lives byte later returns to a value ≥ `lives_at_start` — a restart can only
be a new game, never continued progress.

---

## 11. What to do before any more Rygar compute

1. **Treat 4,608 as the frontier** — not 5,680, not 5,893, not 6,242. Re-audit
   `runs/rygar_odo_night/` (the banked 5,680, a *different emulator binary*,
   never ratchet-audited) the same way before it is cited again.
2. **Make the progress signal ratchet-proof.** Score progress per room-identity
   rather than as one global integral, so re-entering a visited room cannot bank
   new pixels. The two-value room tuple recovered by measurement alone during
   adjudication shows this is mechanisable without importing game knowledge.
3. **Retire R1-08's monotone-`gx` loop check** or gate it, and add its failure
   mode to the anti-vacuity census.
4. **Adopt the frame-clustering loop detector** with its uninterrupted-segment
   positive control (§5) as the standard test for any continuous-scroll profile.
5. **Re-run Contra's gate** with a probe that survives, and fix the
   post-truncation resolution finding. Contra was not fairly excluded.
6. **Fix the auto-restart exhaustion blindness** (§10) and re-audit the
   contamination census.
7. Combine R1-02's hold-macros with a relieved step budget, measured
   specifically through x ≈ 4,600–5,300 — the only untested combination of the
   one lever that actually moved the efficiency metric. Measure it in
   first-visit pixels.

---

## 12. Receipts

| What | Where |
|---|---|
| **The tape** (6,018 actions, full provenance) | `docs/receipts/rygar/r1_tape_gx6242.json` — **tracked**, not under `runs/` |
| Its always-on guard | `tests/test_rygar_r1_tape.py` |
| Independent landing replay | `runs/rygar_campaign/_LANDING/landing_replay.npz` |
| Ratchet segment test + positive control | `runs/rygar_campaign/_LANDING/segment_test.json` |
| Milestone clustering | `runs/rygar_campaign/_LANDING/ratchet_test.json` |
| Gate re-runs at HEAD | `runs/rygar_campaign/_LANDING/gate_rygar_HEAD.json`, `gate_kungfu_odo_HEAD.json` |
| Room-fingerprint decline | `docs/receipts/room_fp/rygar.md`, and the note in `configs/rygar.yaml` |
| Per-item run dirs | `runs/rygar_campaign/R1-*/` |
| Skeptic audit + adjudication | `runs/rygar_campaign/_skeptic_audit/`, `runs/rygar_campaign/ADJUDICATION/` |

`runs/` is gitignored. The tape and its guard are deliberately **not** — a
receipt that lives only under `runs/` disappears on a fresh checkout, which is
exactly how a control receipt hid earlier this week and left its only test
silently skipping.

**Provenance of the tape:**

```
rom_sha256          d87a7b3250eb8d6af3b725169a02dab492a10b92dcd03eb44ddce34e1124bbbf
start_state_sha256  9befb1cefd597130b1b3d80fbd77e9363cf8336b53da1542c2e6846b628f0971
nes_core sha256_16  54366c20d32f71cc
frame_skip 4        hw_flags []        14-action generic controller
```

**Purity (Tier 3).** Every measurement above comes from hardware surfaces — PPU
scroll odometer, nametable VRAM, rendered frames, OAM, PPU render-line counts —
plus the profile's own declared lives byte, all on this profile's own start
state. No disassembly, no RAM map, no walkthrough, no recall of this title.
