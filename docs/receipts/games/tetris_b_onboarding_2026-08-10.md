# Tetris B-TYPE — basis class 7 onboarding (non-spatial planning)

**Date:** 2026-08-10
**Profile shipped:** `configs/tetris_b.yaml` (NEW; `configs/tetris.yaml`
untouched)
**Dump:** `roms/Tetris (USA).nes`, md5 `5b0e571558c8c796937b96af469561c6`,
mapper 1 (MMC1), iNES — read from `nes_core.rom_info`
**Core:** `nes_core` 0.1.0, `nes_core.abi3.so` sha256[:16]
`e09e8191b8d40490`, no `set_hw_*` flags anywhere in this work
**Purity:** every address below was produced by differential probing of
our own scripted rollouts on this dump. No RAM map, no disassembly, no
walkthrough was consulted at any point, including for the menu route —
the menu was navigated by looking at the frames the PPU drew.

---

## 0. Which Tetris

`roms/` contains three cartridges matching "Tetris":

| file | md5 | mapper | used |
|---|---|---|---|
| `Tetris (USA).nes` | `5b0e571558c8c796937b96af469561c6` | 1 | **yes** — the Nintendo 1P game, A-TYPE/B-TYPE |
| `Tetris 2 (USA).nes` | `3b697df25030ec1375d26c6d3985eee3` | 4 | no — a different game, no B-TYPE |
| `Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes` | — | — | no — multicart carrying a European Tetris |

There is **no Tengen Tetris dump in the library** (the only Tengen
cartridge present is `Pac-Man (USA) (Tengen).nes`), so the "note both if
both exist" branch of the spec resolves to: only the Nintendo dump
exists, and it is the one used.

---

## 1. Menu structure — measured, not assumed

Cold boot renders a legal/copyright screen; the title screen's attract
demo ignores input. Screens were identified by rendering the frame and
reading it, and each transition is confirmed in RAM before the next
input is sent.

```
360 f NOOP                -> legal / copyright
START 5 on / 20 off       -> title  ("PUSH START")
START 5 on / 20 off       -> GAME TYPE   (A-TYPE | B-TYPE, MUSIC TYPE)
RIGHT 4 on / 20 off       -> B-TYPE selected      [assert $00C1 == 1]
START 5 on / 30 off       -> B-TYPE setup (LEVEL 0-9 grid, HEIGHT 0-5 grid)
START 5 on / 30 off       -> play, LEVEL 0 / HEIGHT 0
120 f NOOP                -> settle
```

Confirmations at the play frame: `$00C1 == 1`, `$0050 == 0x25`, all 200
playfield cells `0xEF`, HUD reads `LINES-025 / LEVEL 00 / HEIGHT 0`.

**Menu bytes found by two-branch differential** (identical rollouts, one
input differs):

* `$00C1` — game type. idle `0`, one RIGHT tap `1`, second RIGHT still
  `1` (clamped), RIGHT-then-LEFT back to `0`. The on-screen `>A-TYPE<` /
  `>B-TYPE<` cursor tracks it exactly.
* `$0047` (mirror `$0067`) — LEVEL grid index. RIGHT `+1` (0→1, 0→3 over
  three taps), DOWN `+5` (0→5): the 2×5 grid the screen draws.

---

## 2. Play observables

Method, per byte: drive one input from a saved state, drive a different
input from the same state, diff RAM; then re-drive to confirm direction,
reversibility and saturation.

| byte | meaning | measurement |
|---|---|---|
| `$0040` (`$0060`) | falling-piece **column** | 5 at spawn → 2 under 3× LEFT → 8 under 3× RIGHT; a flat piece dropped hard-left fills board columns 0-3, hard-right 6-9 |
| `$0041` (`$0061`) | falling-piece **row** (0-19) | climbs 1→8 over 300 frames of gravity alone, resets at every lock |
| `$0042` (`$0062`) | piece + **orientation** id | 18 at spawn, 17 after one A, 18 after two (an I-piece's two orientations); 19 distinct values across play |
| `$0050` (`$0070`) | **B-TYPE quota**, packed BCD | `0x25` at the B-TYPE opening vs `0x00` at the A-TYPE opening, captured at the same elapsed frame so timers cancel (HUD: `LINES-025` / `LINES-000`) |
| `$0090` | B-TYPE **starting quota**, constant | `0x25` at the opening, through 24 clears, through a top-out and at the win — never observed to move |
| `$0053/54/55` | **score**, packed BCD | `$0054` monotone 0→23 over the clear run; `$0053` wraps; `$0055` never left 0 |
| `$0048` (`$0068`) | **play state** | `{1,2,3,5,6,7,8}` across 6,000 samples of live play, never 10; `10` for 4,300 consecutive samples after a top-out and never recovers |
| `$0058` | game-over **curtain ramp** | `0` throughout play; after a top-out it ramps and parks at 20 |
| `$00D8` | **clear-locked counter**, BCD | flat at 0 across 6,000 no-clear samples; monotone `0x01`→`0x23` across 24 real clears |
| `$0400-$04C7` | **playfield**, 20×10 row-major | all `0xEF` at the opening; hard-left drop fills `$04BE-$04C1` (row 19, cols 0-3), hard-right `$04C4-$04C7` (cols 6-9), a second piece stacks into row 18. Occupied values measured: `0x7B/0x7C/0x7D` in play, `0x4F` in the top-out curtain |
| `$0044` | **UNCONFIRMED** | read 0 in every sample we have (opening, 24 clears, top-out, victory). At LEVEL 0 a level counter and a dead byte are indistinguishable, so it is left unlabelled and unwired |

Mirror structure worth knowing: the game shadows its `$0040-$005F`
block at `+$20`, and part of it again at `+$40`. Only primaries are
wired.

---

## 3. The win — pre-registered, then actually measured

**Pre-registration** (written before any win was reached): *the B-TYPE
win is "lines-remaining reaches 0", i.e. `$0050 == 0`, implemented as
the stateless `clear: {mode: byte_change, addr: 0x0050, direction:
down, target: 0}` hook. The detector's confluence mode is NOT used.*

The spec anticipated that a win could not be reached by probing. It
could. A **probe instrument** — a 1-ply placement search over the board
array we had just located (4 rotations × 10 columns per piece, four pool
workers evaluating candidates in parallel, scored on measured holes /
aggregate height / bumpiness) — drove the quota to zero:

* **70 pieces, 24 clear events, 25 lines, 202.9 s wall, 4 workers.**
* Quota timeline (raw bytes): `0x25 0x24 0x23 0x22 0x21 0x20 0x19 …
  0x16 → 0x14 (a double) … 0x02 0x01 0x00`. The `0x20 → 0x19` and
  `0x16 → 0x14` steps are what prove the encoding is **packed BCD** and
  the direction is **down, one unit per cleared line**.
* Re-running the (deterministic) instrument reproduced the identical
  70-piece trajectory and the identical 24 events.

**The terminal state, read directly.** Resuming the *winning* lineage
for 1,800 frames: `$0050 = 0`, `$0048 = 6`, `$0058 = 0`, `$00D8 = 0x23`,
`$0090 = 0x25`, playfield frozen — none of them move for the whole 30 s.
The rendered screen is the B-TYPE victory screen (`SCORE / LEVEL /
HEIGHT / TOTAL 001723` over the cathedral). So the pre-registered
definition is not a guess: it is the measured win.

> Process note, recorded because it nearly produced a false reading: the
> first pass stepped whichever candidate branch happened to be resident
> in the pool after the win rather than the winning branch, and reported
> `$0050` going `0 → 1` "after the win". That was a *different game*,
> not a quota reset. Re-run with the winning blob reloaded, the state is
> frozen. Resume the lineage you banked, not the worker you last used.

### Why not confluence

The spec offered "quota reaches 0 **or** the detector's confluence
fires"; **the quota test is what shipped, and confluence is not wired.**
A single measured byte states this win exactly, the test is stateless
(so `clear_verify_margin` and `clear_observation_budget` are both 0 and
a replay sees exactly what the live hook saw), and the confluence path
is the one that needs the counterfactual gate to police it and has been
confirmed unsafe on three other secondary games. Nothing here needs it.

### Why `target: 0` cannot fabricate a win

The failure mode is the quota byte reading 0 somewhere that is not a
win. Measured, it does not:

* `$0050` held `0x25` for all **6,000** samples of random play;
* it held `0x25` for all **4,300** samples *after* a top-out (~72 s past
  it), i.e. a loss never walks the byte to 0 within any plausible
  episode;
* the adapter resolves `is_dead` **before** `is_clear`, and
  `player_state: 0x0048 / death_states: [10]` marks a top-out at the
  first observation after it, permanently;
* at the win `$0048` reads 6, so the death hook cannot beat the clear
  hook to a real victory.

---

## 4. The archive axis — what replaces `gx`

Answer: **`$00D8`, the clear-locked counter**, as `progress: {lo:
0x00D8}` with `progress_cap: 64`.

The reasoning is all rejection:

* **`$0050` (the quota) is disqualified.** It counts *down*, and the
  adapter's progress is a monotone-increasing frontier; wiring it would
  make "25 lines still to go" the deepest cell in the search.
* **Score is disqualified as farmable.** `$0054` is monotone, but
  soft-drop points alone move the odometer, so banking higher score pays
  the search for stalling. This is the same flatness gate that rejects a
  free-running frame counter (and it did reject `$001A`, which drifts
  under no-clear play).
* **`$00D8` passes both halves**: flat at 0 across 6,000 samples of
  random play in which no line was ever cleared, and monotone
  non-decreasing across all 24 forced clears. It is the only byte in the
  2 KB that did both. No byte in RAM equals lines-cleared exactly.

**Unit caveat, stated because it is not pinned.** `$00D8` is BCD and
rose by 1 on each of the 23 single clears but did **not** step on the
one double clear, ending at BCD 23 after 25 lines. It is "a counter that
advances with clear events"; whether the unit is lines or events is
undetermined. Monotonicity and clear-locking are what the axis needs and
both are measured. *Experiment that would settle it:* re-run the
placement instrument with the objective changed to prefer multi-line
clears, and trace `$00D8` per frame across a triple — if it steps by 3
it counts lines, if by 1 it counts events, if by 0 the update is
skipped for multis and the byte is lines-mod-something.

**Board signature (`state_sig`).** 20 bits, each "this cell is occupied"
against the measured occupied set `{0x4F, 0x7B, 0x7C, 0x7D}`. Which rows
was decided by measurement: occupancy rate and 10-bit pattern diversity
were computed per row over two independent corpora.

| row | random-play occupancy / patterns | clear-run occupancy / patterns |
|---|---|---|
| 15 | 0.14 / 19 | 0.02 / **2** |
| 16 | 0.13 / 19 | 0.02 / 2 |
| 17 | 0.22 / 25 | 0.09 / 12 |
| 18 | 0.30 / 23 | 0.39 / 21 |
| 19 | 0.43 / 23 | 0.70 / 18 |

Rows 17/18/19 top both. Row 15 — the first cut of this block — looked
fine under random play and is near-dead on the trajectories that
actually clear lines, so it was dropped.

* bits 0-9 — row 19 (`$04BE-$04C7`), all ten columns: the row a clear
  consumes, so its fill pattern is the measured distance to the next
  unit of progress;
* bits 10-14 — row 18 (`$04B4`, every other column);
* bits 15-19 — row 17 (`$04AA`, every other column) — the coarse "how
  tall is the stack here" profile, so a tall board and a flat board with
  the same bottom row are not the same cell.

If the search saturates, widen rows 17 and 18 to all ten columns before
adding a fourth row.

**`y: 0x0041`** (piece row) is wired but **inert at the default
`--y-band 32`**, because the row only spans 0-19. Deliberate: banking a
cell per row of a still-falling piece multiplies the archive 20× for a
transient. `--y-band 4` activates it.

**`area` is deliberately unset.** `$0042` (piece identity) is the
natural occupant and is measured, but `area` also seeds
`Solver.max_area` — read **once** from the root — and several heuristics
filter cells on `key[-5] == max_area`. A value that changes every piece
would silently narrow those to whichever piece was falling at the root.

**`lives: 0x0090`.** Tetris has no lives. The adapter requires the key,
so it points at the byte measured constant in every B-TYPE sample, which
makes `lives(ram) < start_lives` inert *by construction* rather than by
luck. Explicitly **not** `$0050`: the quota decrements on every clear, so
wiring it there would report a death on each success.

---

## 5. Defect found in `nes_core`'s TetrisReward (spec, not shipped)

`rewards.rs::TetrisReward` reads lines as the 2-byte value at
`[$0050, $0051]` (`$0051` is 0 in every sample we took) and scores
`lines - prev_lines`, a **count-up** assumption. Under B-TYPE's
countdown:

1. the line-clear channel is structurally dead — the saturating
   subtraction is 0 on every real clear; and
2. **`episode_success` is fabricated on the first step of every B-TYPE
   episode.** `max_lines` is seeded from the first frame (`0x25` read as
   binary = 37) and success is `max_lines >= line_goal`; at the shipped
   `line_goal: 10`, `37 >= 10` fires immediately, with zero lines
   cleared.

`configs/tetris_b.yaml` neutralises (2) from the profile side with
`line_goal: 99` (the counter only ever falls from 37) and disarms
`line_clear_bonus`. The real fix is a code change in `rewards.rs` —
detect the B-TYPE countdown (`$00C1 == 1`) and score
`prev_lines - lines`, seeding `max_lines` from the quota rather than
from the raw first reading. Not done here: this task's edit surface is
the profile and this receipt.

---

## 6. Honest blockers

* **HEIGHT selector not derived.** The B-TYPE setup screen draws a
  `HEIGHT 0-5` grid next to `LEVEL`. Plain LEFT/RIGHT/UP/DOWN drive the
  LEVEL index (`$0047`) only; `A+RIGHT`, `A+DOWN` and `A` held across a
  RIGHT edge (12 f settle, 3 × 4 f pulses) all left the highlighted
  HEIGHT cell on `0` and moved no byte we could tie to it. So the
  profile is **LEVEL 0 / HEIGHT 0 only** — which is the clean case
  (empty board) and does not change the 25-line quota. Unblocking it
  needs a wider input sweep on that screen (SELECT, B+direction, longer
  holds, and a per-frame RAM trace rather than an end-state diff).
* **`$0044` unresolved** (see §2) — needs a LEVEL > 0 entry, which needs
  the LEVEL selector driven from the root, which is easy; it just was
  not needed for this attempt.
* **`$00D8` unit unresolved** (see §4) — experiment named above.
* **The clear rate is unmeasured under search.** The 202.9 s to 25 lines
  is the *placement instrument's* number, not Go-Explore's. It bounds
  nothing about the bounded attempt; it only proves the win is
  reachable and what it looks like. In 4 × 3 minutes of actual search,
  **zero lines were cleared** — the archive never left progress tier 0.
* **The action-sampling weights are hardcoded in the solver**
  (`action_weights`: `right` +2, `A` +2, `B` +1 by button name), which
  is a spatial-platformer prior with nothing to say about Tetris. The
  profile can only steer it by which actions exist, which is what the
  two soft-drop diagonals do (DOWN-bearing share 9% → 33%). A proper fix
  — per-profile sampling weights — is a change to
  `scripts/go_explore_solve.py` and is therefore filed here as a spec,
  not made.

---

## 7. Validation performed

* `pytest tests/test_profile_configs.py` — 133 passed, 4 skipped (the
  new profile is picked up by both the schema and the GUI/trainer
  boot-contract parametrisations).
* `src.training.config_schema.validate_profile` — one warning, `unknown
  top-level profile key 'solve'`, which every solver profile in the repo
  emits (`contra.yaml`, `kirby.yaml`, `metroid.yaml`).
* `GenericGame` constructed straight from the profile: adapter resolves,
  ROM path exists, `clear_verify_margin() == 0`,
  `clear_observation_budget() == 0`, `needs_apu() == False`,
  `derive_transition_macros` returns `[]` (no UP action, gate inert).
  Synthetic RAM checks: no clear at the opening, **clear at `$0050 ==
  0`**, **dead at `$0048 == 10`**.
* Root state minted by the documented route and confirmed to load into a
  `Pool` (`$00C1 == 1`, `$0050 == 0x25`, 200 empty cells after the load).
* Four 3-minute, 4-worker `go_explore_solve.py` runs against the
  profile, all seed 0, all clean starts and clean exits:

| lane | signature | action space | burst / sticky | cells @180 s | frontier | improvements |
|---|---|---|---|---|---|---|
| 1 | 15-bit (rows 19+15) | 6 | 64 / 0.5 | 50 @120 s | 1 | 46 |
| 2 | 15-bit (rows 19+15) | 6 | 512 / 0.6 | 93 | 22 | 63 |
| 3 | 20-bit (rows 19+18+17) | 6 | 512 / 0.6 | 70 | 13 | 25 |
| 4 | **20-bit (shipped)** | **8 (shipped)** | 512 / 0.6 | **95** | 23 | 18 |

Throughput was flat across all four at ~1,870-1,970 solver steps/s
(4 workers, frame_skip 2).

Read honestly: **n = 1 per lane, and lanes 2 and 4 are a wash on cells.**
The two changes that survive are the ones with an argument behind them
rather than a number: the burst had to grow because a piece needs ~480
solver steps to land under LEVEL 0 gravity (measured) and a 64-step
burst cannot land one; the signature moved to rows 17/18/19 because row
15 is near-dead on clear-run trajectories (measured, table above); and
the two soft-drop diagonals took the 20-bit key from 70 back to 95, i.e.
they paid for the finer key. None of this says the archive is *good* —
95 cells from 337k steps is thin, and no lane cleared a line. That is
the honest state of a first attempt on this basis class.

---

## 8. The counterfactual gate must NOT be run on this profile

**Status: blocker CONFIRMED and fixed.** The first cut of this work
shipped launch commands carrying `--counterfactual-gate`, and §"SAFETY
OF THE `target: 0` TEST" in the profile argued for it. Both were wrong.
The profile's comment block has been rewritten; the corrected launch
commands are in §9.

Artifacts: `docs/receipts/games/tetris_b_cf_gate_2026-08-10/`
— `place_instrument.py` (the 1-ply placement instrument),
`run_cf_gate.py` (binds the **unmodified**
`Solver.counterfactual_probe` to a shim; `scripts/go_explore_solve.py`
was not touched), `win_trace_4329.npy`, `win_timeline.json`,
`cf_gate_verdict.json`.

### 8.1 What the earlier argument got wrong

It said: the hook is stateless, so `clear_verify_margin` and
`clear_observation_budget` are both 0 and "`--verify-bank` /
`--counterfactual-gate` see exactly what the live hook saw." That is a
claim about **warm-up and margin**. The gate does not decide on warm-up
and margin; it decides on **branch agreement**, and it decides with this
polarity:

* high agreement across perturbed branches → `commits` → **banked**;
* low agreement → `contingent` → `ok=False` → `_dump_solution` returns
  False (line 2196) → `observe()` returns `"dead"` (lines 1523-1524).

So a clear that *depended on the player's inputs* is precisely what the
gate refuses — and it also **kills the lineage** that produced it.

The gate's own premise, from its docstring: *"A real stage clear is
committed — a flag slide, an exit pipe animation, a tally cutscene all
run themselves out whatever the pad does."* Tetris B-TYPE has no such
animation to ride out. The 25th line only clears if the last piece is
steered into it.

### 8.2 The cheap experiment nobody ran, run

`place_instrument.py` drove the quota `0x25 → 0x00`: **93 pieces, 23
clear events, 25 lines, 102.7 s, 4 pool workers**, on the shipped root
state, on the same machine `counterfactual_probe` replays on (Pool,
`frame_skip 2`, no hw flags, root blob + one rooting NOOP). Terminal
state at the win: `$0050 = 0`, `$0048 = 6`, `$00D8 = 0x32`. The action
trace — 4,329 indices into the profile's own action space, ending on the
step where `$0050` reads 0 — is banked as `win_trace_4329.npy`.

**Lock → quota-decrement latency, all 23 clears** (solver steps at
`frame_skip 2`; 1 step = 2 frames):

| quantity | min | max | median |
|---|---|---|---|
| piece lock → `$0050` decrement | 11 | 13 | 12 |
| last steerable step → `$0050` decrement | 13 | 16 | 14 |
| clearing piece's whole life (spawn → decrement) | 24 | 43 | — |

**Zero of 23** reach the gate's 32-step pivot. The pivot therefore lands
**19-21 steps before the deciding piece has even locked**, in 23 of 23
clears. On the win itself: pivot 4297, piece spawn 4305, lock 4318,
quota 0 at 4329 — the snapshot is taken **8 steps before the winning
piece exists**, so the perturbed tail is the entire decisive placement
plus a bit of the previous piece's settle.

### 8.3 The gate's actual verdict on the actual win

Unmodified probe, shipped profile, default knobs (`k=8`, `p=0.25`,
`agree=0.5`, `pre=32`):

```
control : {"verdict": "clear", "at": 32}      <- probe HAD discriminating power
branches: 1 clear / 8                          (perturbed 5-15 actions each)
agreement: 0.125   threshold: 0.5
VERDICT: contingent   ok=False   -> CLEAR REJECTED, lineage killed
```

Not a seed artifact — `cf_seed` 0/1/2/3/4 give `agree` 1,0,0,0,0 of 8,
all `contingent`.

The same probe on the **first** line clear (230-action trace, variant
profile identical except `target: 0x24`): control `clear`, **0/8**
branches, agreement 0.0, `contingent`.

### 8.4 `--cf-pre-steps` does not rescue it

| `--cf-pre-steps` | win trace (4329) | first-clear trace (230) |
|---|---|---|
| 4 | — | commits 1.00 |
| 8 | commits 1.00 | commits 1.00 |
| 12 | commits 1.00 | commits 1.00 |
| 16 | commits 0.625 | **contingent 0.25** |
| 32 (default) | **contingent 0.125** | **contingent 0.00** |

The verdict tracks the 11-13 step lock latency and nothing else. Set the
pivot below it and every branch trivially agrees because the board is
already written and the collapse animation is running; set it above and
the deciding placement is inside the perturbed tail. Either way the
number being measured is the pivot depth, not the candidate — and note
the boundary is not even stable (`16` passes on one trace and fails on
the other). **This is a knob to leave alone, not to tune.**

### 8.5 What to use instead

`--verify-bank` (default ON). Run against the same winning trace:

```
replay_verify: {"ok": true, "verdict": "clear", "at": 4329,
                "n_actions": 4329, "margin": 0, "elapsed_s": 8.165}
```

It re-fires on the exact recorded action, which is all a stateless
single-byte hook needs and all the fabrication classes here require.
The rolling-detector-state artifact class the gate exists for cannot
arise on this profile by construction: `clear_verify_margin() == 0`,
`clear_observation_budget() == 0`, and the hook reads one byte of the
current frame.

### 8.6 Honest scope

* Measured at LEVEL 0 / HEIGHT 0 only — the only entry this profile has
  (see §6). Gravity at higher levels shortens a piece's *fall*, not the
  lock → decrement latency, which is the quantity that decides this; but
  it is not measured at LEVEL > 0 and is not claimed.
* The instrument is a **probe, not a solving path**. It uses only the
  board array and piece bytes measured in §2; none of its strategy is
  written into the profile or consumed by the solver. Only byte
  semantics and timing crossed over.
* The instrument's 93-piece win is not a search result. Go-Explore's
  clear rate on this profile remains **unmeasured and plausibly zero**
  (§7: four 3-minute lanes, no line ever cleared). What §8 establishes
  is what happens to a solution *if* the search finds one.

---

## 9. Corrected launch commands

Command #1 (mint the root state) is unchanged. Command #2 loses
`--counterfactual-gate`:

```
.venv/bin/python -u scripts/go_explore_solve.py \
  --out runs/tetris_b/tetb_attempt1_2026-08-10 \
  --root-state "roms/Tetris (USA)_btype_start.state.bin" \
  --profile configs/tetris_b.yaml \
  --workers 8 --minutes 25 --seed 0 \
  --gx-bucket 1 --burst 512 --sticky 0.6 --max-steps 16000 \
  --no-counterfactual-gate 2>&1 | tee /tmp/tetb_attempt1.log
```

`--no-counterfactual-gate` is the default; it is passed explicitly so
the receipt records the choice rather than an omission. `--verify-bank`
is also the default and is the check that carries the banking decision.
