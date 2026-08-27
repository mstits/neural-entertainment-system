# Rygar gets a room axis — **the axis is live, the frontier did not move**

**Date:** 2026-08-27
**Verdict:** best verified **4,608 px**, against a prior of **4,608 px**. **Not
beaten. No stage boundary crossed.** The transition dimension is **live,
correctly signed, and used** — and that is the durable result.

**Ledger: EXHIBITION, without exception.** Everything below is Go-Explore
search output plus scripted and uniform-random rollouts. No policy was
trained for this game and no honest-protocol evaluation was run. Nothing
here may be described with "the AI learned", "the AI plays", or "the AI
beat" (`CLAIMS.md`).

**Purity (Tier 3).** Every observable is a hardware surface — 2 KB CPU RAM
read as an opaque array, the PPU scroll odometer, the PPU blank-fold
counter — or a byte this profile already declared by blind statistical
search over its own rollouts. No disassembly, no RAM map, no walkthrough,
no recall of this title.

**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc`.
**ROM:** `roms/Rygar (USA).nes` sha256
`d87a7b3250eb8d6af3b725169a02dab492a10b92dcd03eb44ddce34e1124bbbf`.
**Start state:** `roms/Rygar (USA)_start.state.bin` sha256
`9befb1cefd597130b1b3d80fbd77e9363cf8336b53da1542c2e6846b628f0971`.

**No tape is preserved under `docs/receipts/` because no trajectory crossed
a stage boundary.** Predecessors: `docs/research/RYGAR_CAMPAIGN_2026-08-26.md`,
`docs/research/ODO_BLANK_AND_GATE_2026-08-26.md`,
`docs/receipts/rygar/clear_predicate_REFUTED.md`.

---

## 1. The two questions, answered plainly

> **Did giving the search a room axis move Rygar?**

**No.** Best verified artifact-free depth is **4,608 px**, exactly the prior.
Seven searches from four different root classes spent **1,711,525 steps** and
**~70 minutes** of wall clock, and not one of them reached a pixel of ground
that had not already been reached.

> **Was a stage boundary crossed?**

**No.** The `($0014,$001C)` area key took **exactly three values** —
`(0,14)`, `(3,16)`, `(0,29)` — across every cell of every archive this
campaign produced, which are exactly the three values the prior room-graph
corpus (285 tapes, 514,078 steps, 38,650 cells) already held. **A fourth
value was never observed.** `solutions: 0` is still a compile-time constant
on this profile and is cited here as evidence of nothing.

> **Is the dimension live, or inert?**

**Live, and load-bearing.** This is the result that survives the null.
Before this landing, `sect` — the leading slot of the cell key, the leading
term of the archive score, and the hard filter on the deep selection arm —
was **identically 0** on Rygar for every search ever run against it. It is
now a real variable that stratifies the archive, feeds the score, and
reconfigures selection. Section 3 proves that, and proves what its absence
produces on the same inputs.

---

## 2. What was wrong, and what changed

`GenericGame.room_id(ram)` is the **constant function `(0,)`** on this
profile. Re-verified at HEAD by constructing the profile and evaluating
`room_id` on three disjoint RAM images (all-zero, all-`0xFF`, random):

```
room_id on 3 disjoint RAM images: (0,) (0,) (0,)
level_key: () ()
```

`configs/rygar.yaml` carries `level_key: []`, no `solve.room_sig` and no
`solve.area`, so the transit test `_rid != c["p0750"]` could never fire.
Therefore `c["sect"] ≡ 0`, `key[0] ≡ 0`, `max_sect ≡ 0`, and
`score ≡ gx`. **The failed 14-hypothesis campaign did not search a space
with a weak transition dimension; it searched a space with none.**

The landing routes the transit test through the already-certified
blank-fold witness instead, as a **mode switch at the existing site**, not
a new key slot:

| `solve:` key | Rygar | default (all 70 other profiles) |
|---|---|---|
| `transit_source` | `blank_run` | `room_id` — byte-identical to before |
| `area_key` | `[0x0014, 0x001C]` | absent |
| `min_blank_frames` | `40` | absent (no default; a missing floor is a missing calibration) |

Arity is frozen at 6 leading + 5 from `cell_fn`, so `key[-5]`/`key[-1]`
indexing, `key_schema_from_keys`, archive resume and A/B comparability are
untouched. The value entering `key[0]` is **`novel`** — distinct areas this
lineage has occupied, minus one — never the raw transition count and never
the raw blank-frame counter. On the banked R1 tape those three numbers read
**2, 55 and 4,329** respectively, and only the first may enter a cell key:
the 55 is 27 round trips through one door, and paying for it would mint 55
stacked copies of one corridor.

Score: the literal `sect * 10000` became
`lex_score(sect, gx + bonus, weight=self.transit_weight)`, defaulting to
`TRANSIT_SCORE_WEIGHT = 10000` — the exact prior constant. The weight is
sized against Rygar's own numbers: one novel arrival must outrank the
verified frontier (4,608) **plus** the entire measured re-anchor ratchet
(1,634), i.e. the whole discounted 6,242 headline, so that no amount of
walking or door-cycling can outbid arriving somewhere new.

Selection: `--transit-deep-relax N` widens the deep arm's filter from
`key[0] == max_sect` to `key[0] >= max_sect - N`, default `N=0`
(algebraically identical, since `key[0]` can never exceed `max_sect`).

---

## 3. The axis is live — positive evidence, and the revert

### 3.1 It stratifies the archive

Read directly out of the banked `archive.pkl` / `traces.pkl` of each run
(`key[0]` = `sect`, `key[-1] * 16` = gx bucket, `key[3]` = `psig`):

| run | root | cells | `sect` histogram | gx band per `sect` |
|---|---|---|---|---|
| D1 | power-on | 717 | `{0: 380, 1: 230, 2: 107}` | 0–1536 / **1536–4608** / 4384–4992 |
| D3/run1 | power-on | 752 | `{0: 378, 1: 284, 2: 90}` | 0–1536 / **1536–4608** / 4512–5072 |
| D2/run1 | resumed | 1647 | `{0: 1056, 1: 591}` | 0–6240 / 1536–7024 |
| D3/run2 | inside `(0,29)` | 472 | `{0: 3, 1: 469}` | 4608 / 4352–6256 |

Two independent cold runs, with no game-internals knowledge of any kind,
put their `sect = 1` band at **exactly 1536–4608** — the first door
position and the artifact-free ceiling, **reproduced to the pixel by an
instrument that has never been told either number**. `psig` carries real
area keys (`(3,16)`, `(0,29)`), and `best_score` reads 25,002 and 25,081
= `2 × 10000 + gx`, which is the score formula demonstrably consuming
`sect` on live search output rather than in a unit test.

### 3.2 It survives an actual revert

Verified in an isolated `git worktree` at HEAD rather than by touching the
shared tree (a sibling workflow was committing throughout; it landed
`7e1b7ed` mid-session).

* **Solver and config reverted to the pre-landing commit `7e1b7ed`, library
  and tests present** — `tests/test_transit_wiring.py` fails **15 of 15**
  with the ROM available, and **14 of 15 without it** (only the
  real-emulator replay skips). Meaningful `AttributeError` / `SystemExit` /
  structural assertions, not a bare collection error, and the ROM-free
  majority is the split the receipt pattern requires.
* **Score site reverted to the bare literal, everything else present** —
  `tests/test_transit_score_wiring.py` fails exactly its 2 structural
  tests; the other 28 pass. **Named honestly: at the default weight the
  reverted literal computes the identical score, so only the two AST
  checks catch it. The real-emulator score replay does not.** That is a
  real, if narrow, coverage limit and it is recorded rather than papered
  over.
* **Mechanism present** — **91/91** across the four transit suites, the six
  new resume-guard tests (§6.1) included.

### 3.3 What its absence produces, on the same steps

`tests/test_transit_wiring.py::TestRealReplay` drives the banked R1 tape
(6,018 actions) through the real emulator and through **both** transit
tests in lockstep: the blank-run witness banks **2** novel arrivals, the
`room_id()` inequality it replaces banks **0**. Same tape, same steps, same
frame.

---

## 4. The search, and the null

Seven runs, four root classes, two workers throughout (a sibling Contra
workflow held the other cores; every sps figure below reads as contended,
not idle).

| run | root | steps | wall s | sps | cells | max `sect` | terminal gx | novel areas |
|---|---|---|---|---|---|---|---|---|
| D1 | power-on | 310,918 | 723 | 430 | 717 | **2** | 5,002 | 2 |
| D3/run1 | power-on (frozen profile copy) | 386,536 | 901 | 429 | 752 | **2** | 5,081 | 2 |
| D2/run1 | resume `R1-14/extend` | 379,788 | 900 | 422 | 1,647 | 1 † | 7,029 | 2 |
| D3/run2 | `far_arrival1` inside `(0,29)` | 274,454 | 600 | 457 | 472 | 1 | 6,266 | 1 |
| R1 | both `(0,29)` arrivals, non-x key | 45,920 | 100 | 458 | 922 | — | — | **0** |
| R2 | both `(0,29)` arrivals, count-weighted | 206,308 | 480 | 430 | 400 | — | — | **0** |
| X1 | both `(0,29)` arrivals, inverse-visit | 107,601 | 480 | 224 | 428 + 403 | — | — | **0** |

**Total 1,711,525 steps / 4,184 s.** † D2's `sect` stream is not
comparable — see §6.1.

**Both cold runs found both known doors inside their first two minutes and
then went flat for the rest of their budget.** D1 held `max_sect = 2` and
`gx = 5,002` for its last 208,404 steps, with `c_local` frozen at 388.
D3/run1 held `max_sect = 2` and `gx = 5,081` for its last 180,000, with
`c_local` frozen at 379. The axis did not fail to fire; it fired
immediately, twice, and then there was nothing left for it to find.

---

## 5. Why 7,029 px is not progress — decomposed, not asserted

The four deepest tapes were replayed on fresh single-worker pools using the
solver's own replay recipe (`apply_hw_flags` → `set_headless` →
`set_skip_preprocess` → `set_odometer_enabled` → `reset_all` →
`load_worker_state` → one rooting NOOP). **All four reproduce their filed
terminals exactly**, and every one ends **ALIVE**: longest debounced dead
run is **2 observations** on all four — the door-blip signature, never the
`≥3` debounce, and nowhere near the 5,721+ observation pin of a real death.

Each tape was then cut at every area-key change and the odometer delta
banked per segment:

| tape | acts | terminal x | door crossings | `(0,14)` | `(3,16)` | `(0,29)` |
|---|---|---|---|---|---|---|
| D1 | 3,924 | 5,002 | 15 | 1 seg, **+1,536** | 8 seg: one **+3,063**, seven +55..+64 | **7 seg, all dx = 0** |
| D3/run1 | 3,989 | 5,081 | 17 | 1 seg, **+1,536** | 9 seg: one **+3,063**, eight +55..+64 | **8 seg, all dx = 0** |
| D2/run1 | 7,599 | 7,029 | 81 | 1 seg, **+1,536** | 41 seg: one **+3,072**, forty +55..+64 | **40 seg, all dx = 0** |
| D3/run2 | 3,938 | 6,266 | 55 | — (root is inside `(0,29)`) | 28 seg, +55..+64, sum 1,658 | **28 seg, all dx = 0** |

Every tape decomposes the same way and there is nothing else in it:

* **one** traverse of the start ridge, `+1,536` px, ending at door 1;
* **one** traverse of the corridor, `+3,063..3,072` px, ending at door 2 —
  so real ground tops out at `1,536 + 3,072 = 4,608`;
* then an alternation of **zero-gain visits to `(0,29)`** and
  **+55..+64 px re-anchor blips in `(3,16)`**, forever.

**`(0,29)` banked exactly zero pixels in 83 of 83 visits across the four
tapes.** The ratchet totals 403 / 482 / 2,421 / 1,658 px respectively —
4,964 px of "gain" that is a bookkeeping artifact of the odometer
re-anchoring at each door, not distance travelled.

The smoking gun is simpler still. `far_arrival1_obs2682.state` and
`far_arrival2_obs2820.state` are two savestates **standing in the same
room**, and their odometers read **4,608** and **9,280** — a 4,672 px
spread for one location. Above 4,608, raw x is not a position on this
profile. **Cite 4,608. Never 6,242, never 7,029, never 9,344.**

---

## 6. Defects found, and fixed

### 6.1 The `seen` carry is not optional on resume — **fixed here**

Found by reading D2's banked output, not by reasoning. D2 resumed
`runs/rygar_campaign/R1-14/extend`, an archive written before this axis
existed: **493 of its 1,647 trace records are 7-tuples**, carrying no
occupied-area set. Every lineage restored from one therefore started with
an **empty `seen`** and re-banked the first area it arrived in. The
fingerprint is in the archive: **9 cells at `sect = 1` whose arriving area
key is `(0,14)`, the room the run starts in.** That is the design's own
"treadmill one restore deep" hazard, live — and its error direction is
**fabrication**, the one direction a novelty gate may never fail in. It
also depressed D2's `max_sect` to 1 and makes its `sect` stream
non-comparable to the cold runs.

`psig` cannot backfill it: a legacy Rygar record carries `psig == ()`
because `room_id`-mode transit never fired on this profile at all. So
`scripts/go_explore_solve.check_transit_resume` now **refuses the resume
outright**, with no override — the same reasoning `_resume_room_index`
uses, that the resumed records are not a weaker lineage but a lineage whose
history is unknown. Guarded by
`tests/test_transit_wiring.py::TestLegacyArchiveResumeIsRefused`
(6 tests), whose anti-vacuity was verified by actual revert: neutering the
helper fails 2 behavioural tests, removing the call site fails the
structural one, and one test replays the real D2 archive and asserts the
message reads `493 of 1647` (skips off this machine — `runs/` is
gitignored — while the synthetic cases do not).

### 6.2 A bare attribute broke seven pre-existing tests — **fixed here**

The deep-arm filter read `self.transit_deep_relax` directly.
**Four** test files carry duck-typed `SimpleNamespace` Solver stand-ins
that predate this axis, and only one of them had been updated; the full
suite came back with **7 new `AttributeError` failures** in
`test_room_router.py`, `test_terminal_stasis.py` and
`test_gate_k0_reforge.py`. Changed to
`getattr(self, "transit_deep_relax", 0)` — the same form, for the same
measured reason, the score site already uses — and the structural test that
pinned the bare-attribute form updated to accept either. This is the
lesson: **a full-suite gate is not a formality on a change that touches a
hot shared method.**

### 6.3 `_sel_lowl_band24` is dead code — **recorded, not fixed**

`deep` is filtered to `key[0] == max_sect`, so
`minl = min(c.key[0] for c in deep) == max_sect` and
`lowl = [c for c in deep if c.key[0] <= minl + 1]` is `deep` identically.
The 70%-of-the-time `pool_band = self._sel_lowl_band24` branch in
`select()` has been sampling the same list as its else-branch ever since
the leading key slot became `sect`. **Any claim that the solver "already
prefers lower-transit cells" is false**, and it must not be relied on as a
safety valve for §7.2. Left unfixed deliberately: changing a selection
branch is not a landing-day edit.

---

## 7. What the wall is

### 7.1 Routing blindness is REFUTED

This is the question the landing was built to answer, and it has an answer.
The search now has a transition dimension. It **uses** it — the dimension
enters the cell key, the score and the deep-arm filter, all three
demonstrably. It is **correctly signed** — the objective used to pay
+55..+64 px for every door crossing and nothing for a new room, and now
pays 10,000 for a room and cannot be farmed by re-crossing. It **works** —
it re-derived both of Rygar's known doors from a cold start, to the pixel,
in under two minutes, twice.

**And the frontier did not move.** "The solver could not tell a door from
walking" is no longer available as an explanation for Rygar's wall.

### 7.2 The pre-registered falsifier fired

The room-graph work pre-registered this: *"if a bounded, death-terminated
search from inside `(0,29)` scoring on anything except odometer x still
produces no fourth `($0014,$001C)` value, then `(0,29)` is a dead end and
the next target becomes a mid-corridor exit."*

**R1, R2 and X1 are the strict form** — three independently written
standalone harnesses, three different non-x selection rules (least-visited
over an OAM/nametable-hash key; count-weighted with five rotating action
biases; inverse-visit over `(area, x//8, y//8)`), all death-terminated,
all rooted at the two banked `(0,29)` arrivals. **359,829 steps, zero
fourth areas.** R1 alone terminated 149 debounced deaths, so the death path
was demonstrably live rather than silently inert. **D3/run2 is the loose
form**, and is labelled as such: the solver from the same root with the
transition dimension leading the objective but `gx` still the tiebreak —
274,454 further steps, also zero. **634,283 steps from inside that room,
and the key never took a fourth value.**

**By its own rule, `(0,29)` is a dead end.** The next target is a
mid-corridor exit: the corridor's 3,072 px have carried 218,925
observations over 919 traversals with transitions observed at exactly two x
positions — its two ends — and nowhere between.

### 7.3 The honest residual

Two things are **not** excluded, and neither may be quietly dropped.

1. **The area key may be too coarse to see the thing we are hunting.**
   `$0014/$001C` take three values in a corpus of 38,650 cells plus
   1.7M fresh steps. If a stage boundary lands in an area whose key aliases
   one already seen, the dimension is blind to precisely the event it
   exists to catch. X1 ran the named probe — a masked nametable hash beside
   the area key — and it **saturated its 64-hash cap in both areas within
   ~20 seconds**, so it neither confirms nor refutes the concern. X1
   correctly declined to gate on it.

2. **Budget.** 1.71M steps is small beside the 2,382,973 observations and
   232,548 restarts this project has already spent inside `(0,29)` alone.
   A null at this scale is evidence, not proof.

Subject to those two, the wall is what the campaign originally classed it:
**skill and survival on a route the search can now see** — uniform-random
survives a median 677 steps here, and the corridor is where the deaths are.
Two prior live runs converged at x 5,272 and 5,336 and flattened; these
runs flattened at 5,002 and 5,081, which is the same place once the ratchet
is deflated out of all four figures.

### 7.4 The one live cost of the axis working

`deep` is a **hard** filter on `key[0]`. The instant a cold run reaches
`sect = 2`, the entire deep selection pool becomes the 107 (D1) / 90
(D3/run1) cells inside `(0,29)` — a static screen where `dx = 0` by
construction, in 83 of 83 measured visits. The dimension did exactly what
it was designed to do, and on this particular game the newest room **is**
the dead end. Both cold runs therefore spent roughly 90% of their budget
with the deep arm pinned somewhere x cannot move.

`--transit-deep-relax` exists for precisely this and **was never exercised
on a clean cold run**: D1 ran at the default 0, and D2 — the only run that
passed `--transit-deep-relax 2` — was the contaminated resume, so its
`max_sect` never left 1 and the relax had nothing to relax. **That is the
cheapest untried arm and it should be next.**

---

## 8. Corrections to the record

* **The 14-hypothesis campaign's conclusion is now scoped, not withdrawn.**
  It said no instrument in the pipeline could count rooms. One could; the
  pipeline now does. Its *frontier* conclusion stands untouched: 4,608 px,
  and the wall held against a room-aware search too.
* **D2's filed `novel_transitions: 0` understates its own instrument.** Its
  tape banks 2 novel arrivals. It meant "no corpus-novel fourth area",
  which is the right conclusion; the field is wrong.
* **X1's prose claims its 4,608 root reading is "an independent
  reproduction of the ceiling".** It is not — it is an artifact of which
  root it loaded, since the sibling root in the same room reads 9,280.
  X1's JSON records both honestly (`x0: [4608, 9280]`); the inference in
  its narrative does not hold and must not be repeated.
* **R1, R2 and X1 banked no replayable tape.** Their conclusions rest on
  summary JSON that cannot be re-verified by replay. Recorded as a receipt
  gap, not excused — see `MISTAKES.md`.
* **`docs/research/ROOM_GRAPH_LEVERAGE_2026-08-27.md` never existed.** Four
  source files referenced it; all four now point here.

---

## 9. The predicate question

**It is not answerable, and this landing does not make it answerable.**

A predicate with no witnessed positive can only be shown *not* to fire. No
Rygar clear has ever been witnessed — 71 of 71 `solutions/` directories are
empty, every one a compile-time constant. `configs/rygar.yaml` still
carries **`level_key: []`** with no clear and no finale, and the test that
guards that is still green. **No predicate was minted here and none should
be.** Both naive candidates still fabricate wins on Rygar's own deepest
tape: "a transition happened" fires 55 times, and a level_key on the
blind-discovered area byte fires 28 times and latches TRUE on 83% of
observations.

The gating task is unchanged: **one trajectory that crosses a stage
boundary.** Get that, and the predicate question becomes answerable for the
first time. Until then, `solutions: 0` on this profile is evidence of
nothing and is never cited as a miss.

---

## 10. Next

1. **Run the relax arm.** One cold run at `--transit-deep-relax 1` or `2`,
   pre-registered, so the deep arm keeps the corridor band reachable after
   `sect` advances into the dead end. This is the one knob the design
   specified, the one the campaign never got a clean run of, and it costs
   ~15 minutes.
2. **Adjudicate the area key before spending more compute on the null.**
   Sample a masked nametable hash on the first rendered frame after a run
   closes, with a cap large enough not to saturate, and record it beside
   the area key. A new hash under a seen area key is the instrument saying
   the key is too coarse. **Do not gate on it until it is calibrated.**
3. **Move the target to a mid-corridor exit**, per §7.2's own
   pre-registration. This needs a loadable mid-corridor savestate, which
   does not exist yet — only a 500-frame RAM/OAM capture. Minting one is
   the concrete task.
4. **Bank a tape from any falsifier search that reports a null.** Three of
   this campaign's seven runs cannot be replayed.
5. **Otherwise re-shelve Rygar with this receipt.** The expensive question
   — "is the solver blind to rooms?" — is now closed, and closing it is
   worth more than the pixels it did not buy.

---

## 11. Files

**Landed:** `scripts/transition_witness.py` (`TRANSIT_SCORE_WEIGHT`,
`transit_dimension`, `lex_score` — additive), `scripts/go_explore_solve.py`
(the mode switch, the witness lifetime, the `seen` carry, the score
parameter, `--transit-weight`, `--transit-deep-relax`,
`check_transit_resume`), `configs/rygar.yaml` (three `solve:` keys),
`tests/test_transit_dimension.py`, `tests/test_transit_wiring.py`,
`tests/test_transit_score_wiring.py`, `tests/test_go_explore_solve.py`
(stand-in defaults).

**Receipts (gitignored `runs/`, on this machine only):**
`runs/rygar_transitions/{D1,D2,D3,R1,R2,X1}/`.
