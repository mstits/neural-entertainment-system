# The Legend of Zelda — solver onboarding (basis class 9: room graph)

**Date:** 2026-08-10
**ROM:** `roms/Legend of Zelda, The (USA) (Rev A).nes` (md5
`f4095791987351be68674a9355b266bc`)
**Profile touched:** `configs/zelda.yaml` — one new `solve:` block appended.
Nothing else in the file changed, and the trainer never reads `solve:`, so
every training path is byte-identical.
**Emulator:** this core, `Pool(frame_skip=4)`, headless except where a
rendered frame is the measurement.
**Probe scripts:** written under `/tmp/agent_zeldablocks/` (scratch, not
committed); every number below is reproducible from the protocol described
here, which is the part that matters.

## 0. Purity statement

No RAM map, disassembly, walkthrough or recalled fact about Zelda entered
any decision here. The only ground truth used is (a) RAM read back from our
own rollouts and (b) pixels this emulator rendered. The profile's
pre-existing `ram_mapping:` block descends from a disassembly; it was **not**
consulted as evidence and is not the source of any address in `solve:`.
Where a probe landed on an address that block also names, that coincidence
is recorded below as an observation *after* the fact — the probe is the
evidence, and the profile comment says so.

## 1. Which start state (measured, not assumed)

Ten `roms/zelda*.state.bin` files exist; four load on the current
serializer (`zelda.ines1_start`, `zelda_journey_attempt`, `zelda_start_419`,
`zelda_start_eagle`, `zelda_start_the eagle`, `zelda_west_start` all fail
deserialization — recorded here so the next agent does not re-discover it).

Rendering the HUD of the survivors (3× zoom of rows 0-63) shows:

| state | screen index `$00EB` | HUD hearts | verdict |
|---|---|---|---|
| `zelda_start.state.bin` | 119 | **3 full** | usable |
| `zelda_start2.state.bin` | 119 | 3 full | duplicate (profile notes it is frozen) |
| `zelda_sword_start.state.bin` | 119 | 3 full | duplicate |
| `zelda_start_ctrl.state.bin` (profile's `start_state_path`) | 118 | **1 full + 2 empty** | lethal |

Driving the same explorer from each: from `zelda_start_ctrl`, 283 attempts
produced **78 deaths and exactly 2 distinct screens**; from `zelda_start`,
319 attempts produced **2 deaths, 114 verified transitions and 11 distinct
screens**. The solver therefore roots on `roms/zelda_start.state.bin`
(`--root-state`). `start_state_path:` is left untouched on purpose —
changing it would move the training path, which is out of scope for this
attempt.

## 2. Protocol: what counts as a screen transition

The naive detectors both fail on this game, and both failures were observed
before the protocol was fixed:

* **Mass-RAM-churn reset detection fails.** Zelda's death is a slow
  animation into a CONTINUE/SAVE/RETRY menu. Peak per-step churn over a
  700-step lethal hold is ~300 changed bytes — *under* the 350-byte
  threshold `discover_observables.Discoverer` uses — so no reset is ever
  detected and death-spanning deltas silently pollute every statistic.
* **Frame-difference alone fails.** A first pass counted any run of ≥4
  steps with a large play-area pixel delta as a transition. That bans
  nothing: the death spin is exactly such a run. It banked 143 "transitions"
  of which the up/down half were deaths, and produced the false reading that
  the room byte steps ±1 vertically.

The protocol that survived:

1. Settle 5 steps, capture a **block fingerprint** of the play area (rows
   64-239 reduced to 11×16 blocks of mean RGB, quantized /40).
2. Optional lateral jiggle to find an opening, then re-settle and re-capture;
   **discard the attempt if the jiggle itself crossed** (fingerprint moved
   >25%). This is the fix for the direction-mislabelling that produced the
   first, wrong vertical reading.
3. Hold the target direction (with a periodic `A`) until a sustained scroll
   (≥4 steps of large frame delta), then settle 6 steps.
4. Accept the transition **only if** the settled fingerprint differs from the
   pre-push fingerprint by >25% of blocks and mean brightness stayed >20
   (i.e. we are not in the game-over menu).
5. Bank a savestate whenever the fingerprint is new, and explore outward from
   the bank — a mini Go-Explore over screens, which is what gets past the
   two-screen radius a from-start-only driver is stuck in.

**114 accepted transitions / 53 banked states / 11 screens** in 141 s on one
worker.

## 3. The bytes, each with its receipt

### `$00EB` — overworld screen index → `solve.area` and `solve.room_advance.addr`

Per-direction delta over the 114 accepted transitions (dominant delta and
its share):

| held direction | delta | purity |
|---|---|---|
| right | **+1** | 36/36 = 1.00 |
| left | **−1** | 49/49 = 1.00 |
| up | **−16** | 8/12 = 0.67 |
| down | **+16** | 11/17 = 0.65 |

The vertical impurity is the driver, not the byte: every residual sample
carries a ±1 delta, i.e. Link slid along a wall and left through an
east/west edge while the button said UP. So `$00EB` is a **16-wide grid
index, `row*16 + column`** — measured, not assumed.

Two independent cross-checks:

* **Bijection.** Over the 53 banked states, a background fingerprint
  (per-block median of 9 noop samples, so moving enemies drop out) never
  mapped to two different `$00EB` values (0 violations). The reverse
  direction has 8 apparent violations, all explained by residual enemy
  motion in the fingerprint, not by the byte.
* **Contiguity.** The 11 values reached — 87, 88, 89, 90, 102, 103, 104,
  118, 119, 120, 121 — decode as rows 5-7 × columns 6-10: a contiguous
  block, which is what a grid index does and a hash does not.

`$00EC` steps identically but read a stale `120` once mid-load (1/125
transitions), so the clean twin is wired and `$00EC` is not.

### `$0070` — Link's on-screen X → `solve.progress.lo`

45-step directional holds from `zelda_start.state.bin`, frame_skip 4:

```
noop  : 84 84 84 84 84 84 84 84          (flat)
right : 89 121 153 185 217 240 240 156   (climbs, then RE-BASES past the east edge)
left  : 78  45  13   0  16 112 208 234   (falls, then re-bases past the west edge)
up    : 88 88 88 88 88 88 88 88          (flat)
down  : 88 88 88 88 88 88 88 88          (flat)
```

Monotone under the matching axis, flat under the orthogonal one and under
noop, and it **re-bases at every screen edge** — the room-graph signature.
That is why it is wired as the within-room axis only, with the room byte
leading the cell key as `area` instead of riding as a `<<8` page: pairing
them would make one column step east (+256) outrank the entire width of a
screen and bias the frontier east/south for a reason nothing measured.

### `$0084` — Link's on-screen Y → `solve.y`

Same probe: `133 → 85` under UP (clamps at the screen's walkable top),
`133 → 189` under DOWN, flat under LEFT/RIGHT/NOOP. `$0248` and `$024C`
track it exactly and are OAM-shadow mirrors of the same sprite; the
zero-page byte is the source and the mirrors are not wired.

### `$0670` — fine HP → `solve.lives` (damage proxy)

Health was read from the **HUD heart box** located by rendering the HUD at
3× zoom: rows 44-58, columns 170-208; one full heart = 41 red pixels
(`R>150, G<90, B<90`), a half heart ≈ 20-21.

> **Corrected 2026-08-10** after review. The first version of this section
> claimed `$0670` is "non-increasing in 3/3 lethal rollouts". **That claim
> was wrong** and is retracted — the raw ladder below increases at
> 127 → 254. A reviewer then read the same artifact the other way and
> concluded the byte "fabricates deaths at full health"; that is also wrong,
> and §3.5a is the measurement that settles both. Repro for everything
> here: `docs/receipts/games/zelda_hp_ladder_probe.py` (~3 min, 4 workers).

Health was read from the **HUD heart box** located by rendering the HUD at
3× zoom: rows 44-58, columns 170-208; one full heart = 41 red pixels
(`R>150, G<90, B<90`), a half heart ≈ 20-21.

**The complete ladder**, driving into a mob from the full-health root down
to the death fade, HUD sampled one step late (see §3.5a for why):

| `$0670` | 255 | 127 | 254 | 126 | 253 | 125 | 0 |
|---|---|---|---|---|---|---|---|
| `$066F` | 34 | 34 | 33 | 33 | 32 | 32 | 32 |
| HUD red px | 123 | 103 | 82 | 62 | 41 | 21 | 0 |
| hearts | 3.0 | 2.5 | 2.0 | 1.5 | 1.0 | 0.5 | dead |

So `$0670` is a **half-heart-resolution HP register**: bit 7 is "the current
heart is full (1) / half (0)" and the low 7 bits are a full-hearts counter
(127 → 126 → 125), with 0 = dead. All six visible HP levels are
distinguished — it is the finest damage signal measured on the bus. It also
sat perfectly **constant through undamaged play** (18 of 24 macro rollouts
of ≤600 steps never moved it), so it is not a churny combat byte.

`$066F` tracks only the low-7 half of that ladder (34/33/32) and **saturates
at 32 through 1.0, 0.5 *and* 0.0 hearts** — strictly coarser, blind to the
last two HP levels and to death itself. Not wired.

### 3.5a Why the raw byte is wired unmasked — and the desync that fooled two passes

**`$0670` LEADS the rendered HUD by exactly one `frame_skip=4` step.** At the
step the byte moves, the heart box still shows the *old* value; it repaints
on the next step. Sampling byte and pixels on the same frame therefore makes
every real hit look like "the damage byte moved while the HUD showed full
health" — a one-step desync artifact, the same class of error as the 4-frame
replay desync logged during B4.

Discriminator: the instant `$0670` leaves its root value, **freeze the
controller to no-op for 20 steps**. A no-op cannot inflict new damage, so if
the HUD then drops and stays down, the hit was already committed in RAM.

| firings over 40 rollouts (10 trials × 4 workers) | n |
|---|---|
| damage already visible on the same frame | 5 |
| HUD lagged one step — **real hit** | 9 |
| genuinely full health after 20 no-op steps — real false death | **0** |

14/14 were true positives and the HUD never returned to 123 px. Scored
on-frame the same run calls 9 of those 14 "false". Re-running the reviewer's
own 16-rollout harness at its own seed 1234 with the single-line correction
(read the HUD one step later) flips it from **3 false / 2 true** to
**0 false / 5 true**.

**On monotonicity.** As an integer the ladder is genuinely *not* monotone:
127 → 254 is an increase while HP falls from 2.5 to 2.0 hearts. This does
not reach `GenericGame.is_dead`, which is the one-shot
`lives(ram) < start_lives`: the root reads 255 and every damaged value
(254, 127, 126, 125, 253, 0) is below it, so the test is exact and fires on
— and only on — the first half-heart lost.

**Precondition, load-bearing:** the root state must read `$0670 == 255`.
Rooted at partial health the non-monotonicity does bite — from 127 the next
hit reads 254, which is not `< 127`, so that half-heart is missed.
`roms/zelda_start.state.bin` reads 255 and is what the launch command roots
at; `zelda_start_ctrl.state.bin` reads 253, whose ladder (253 → 125 → 0)
happens to stay monotone. **Any new root must be re-checked** — including a
reverse-curriculum dungeon-entrance state, which this profile invites.

To make the wiring root-independent instead, `GenericGame` would need an
optional `lives_mask` (`int(ram[self._lives]) & self._lives_mask`, default
`0xFF` = byte-identical for every existing profile) and this profile would
set `0x7F`, whose ladder 127/127/126/126/125/125/0 *is* monotone. That is a
**sensitivity change, not a bug fix**: masked, the first half-heart is
invisible (127 == 127) and `is_dead` fires one full heart late. Costed here,
not requested — no solver-side change is needed for the shipped root.

**Known limitation.** `lives` semantics are "a decrease means dead", so this
terminates the lineage on the **first hit**, exactly like Metroid's
`0x0107` energy proxy — measured, ~1/3 of 700-step macro rollouts end that
way. Zelda's real death is the CONTINUE/SAVE/RETRY menu. No menu-state byte
survived probing (§5), so the proxy anticipates the death rather than
detecting it.

## 4. What is deliberately NOT wired

* **`level_key: []`** — and here that is a correctness requirement, not a
  default. In `GenericGame`, a `level_key` *advance* IS `is_clear`, and a
  `level_key` change that is not an advance kills the lineage as a warp.
  Wiring the measured room byte as `level_key` — which is what the task
  spec asked for, and the one deviation taken — would **fabricate a "clear"
  every time a worker walks east and execute every worker that walks west**,
  on a map whose index decreases westward by construction. The sect/room
  machinery the spec wanted still engages: `room_id() = level_key + area +
  room_sig`, and `area` alone changes at every crossing, so `max_sect`
  counts room transitions exactly as intended (verified live in §6).
* **`entity_slots`** — no clean kill-flag array was isolated. A strided scan
  (strides 1/2/4/8/16/32, ≥4 members, each with ≥2 nonzero→0 events) over a
  400-step sword-spam drive returns only the OAM shadow `$0200-$02FF` at
  stride 8 — sprite attribute data, not per-enemy life state. The kill-key
  dimension stays unavailable until a better probe exists.
* **`rupees` / `keys`** — no pickup ever occurred in any probe rollout (every
  HUD counter read `x0` throughout), so there was nothing to calibrate
  against. Claiming an address here would be inventing one.
* **`clear:`** — no measured win/clear signal of any kind exists for this
  game yet. Absent rather than borrowed from another profile.

## 5. Rejected candidates (so they are not re-proposed)

| candidate | why rejected |
|---|---|
| `$00E8` as room counter | what `discover_observables.find_room_counter` returns. It churns *during* the scroll (189 changes in 2400 steps, 188 of them inside scroll windows) and returns — an animation/scroll byte, not stable-within-room. |
| `$00FD \| $00E8<<8` as progress | what `find_progress` recommends. Same animation byte as the page term. |
| `$000A` as HP | what `find_hp_lives` recommends (`hp_death_proxy`, value 8). Uncorrelated with the HUD heart box. |
| `$00EC` as area | correct behaviour but read a stale `120` mid-load once in 125 transitions; the twin `$00EB` did not. |
| `$0248` / `$024C` as y | exact mirrors of `$0084` in the OAM shadow; mirrors, not sources. |
| `$0604`/`$0605`/`$060E` etc. as transition bytes | moved on the *death jingle*, which the first (broken) transition detector counted as an up/down transition. |
| menu/game-over state byte | searched three ways (constant-before/constant-after across the menu boundary; bytes shared by 3 lethal runs and absent from 2 live runs; brightness-gated switch test). Nothing survived. Open item. |

## 6. Validation run (bounded, 4 workers, 4 minutes)

```
.venv/bin/python scripts/go_explore_solve.py \
  --out /tmp/agent_zeldablocks/solve_probe \
  --root-state roms/zelda_start.state.bin \
  --profile configs/zelda.yaml --workers 4 --minutes 4 --seed 1
```

```
[seed] rooted at roms/zelda_start.state.bin wd=() lives=255 area=119; archive={"cells": 1, ...}
{"elapsed_s":  60, "cells": 1434, "max_area": 121, "max_sect": 16, "solutions": 0, "sps": 953, "door_macros_injected":  415, "max_room": 121}
{"elapsed_s": 120, "cells": 1875, "max_area": 126, "max_sect": 16, "solutions": 0, "sps": 939, "door_macros_injected":  774, "max_room": 126}
{"elapsed_s": 180, "cells": 2352, "max_area": 127, "max_sect": 16, "solutions": 0, "sps": 939, "door_macros_injected": 1148, "max_room": 127}
{"elapsed_s": 240, "cells": 2735, "max_area": 127, "max_sect": 16, "solutions": 0, "sps": 940, "door_macros_injected": 1496, "max_room": 127}
done: {"cells": 2735, "frontier": 2533, "best_score": 160240, ...}, 0 solutions
```

**Rooms visited** (the requested telemetry) is the count of distinct `area`
values in the archive — `key[-5]`, the same projection `wall_taxonomy`
uses:

```
rooms visited: 15   (4 min, 4 workers)
as (row, col): (5,15) (6,6) (6,7) (6,8) (6,15) (7,6) (7,7) (7,8) (7,9)
               (7,10) (7,11) (7,12) (7,13) (7,14) (7,15)
cells/room   : 118:379  119:845  120:774  121:207  122:32  123:48 ... 127:48
```

Read-outs:

* **Zero fabricated clears.** `solutions: 0` for the whole run, which is the
  point of `level_key: []` — a room-graph game with no measured clear must
  bank coverage, not wins.
* **The transition gate fires and works.** 1,496 door macros injected;
  `max_room` climbs 119 → 127.
* **`max_sect` pinned at 16 from the first minute** — the default
  `--sect-cap`. That looks like the Lost Levels failure mode (a cap sized
  for SMB's ~5-room levels saturating instantly on a game whose room id
  changes at every crossing), so it was tested rather than assumed — see
  §6b, which **refutes** raising it here.
* **Frontier bias is east/south.** Rooms 122-127 hold exactly 48 cells each:
  the deep-frontier bias follows the numerically largest `area`, so it runs
  the row-7 corridor east and thins out. Honest consequence of using a grid
  index as the depth axis; nothing external says east is "forward", and
  nothing here claims it is.

### 6b. `--sect-cap` A/B — raising it makes coverage WORSE

Same profile, same root, same budget (4 workers × 4 min), `--sect-cap 64`
(`--seed 2`) against the default 16 (`--seed 1`):

| | cells | rooms visited | deepest `area` |
|---|---|---|---|
| `--sect-cap 16` (default) | 2,735 | **15** | **127** |
| `--sect-cap 64` | 5,395 | 8 | 124 |

Twice the cells and **half the rooms**. The transit counter is part of the
cell key, so a bigger cap mostly mints duplicate cells for the *same* screen
(rooms 119/120 alone hold 2,108 and 2,493 cells at cap 64 versus 845 and 774
at cap 16) and dilutes selection away from the frontier. Cell count is not
coverage — this is a clean local instance of that. **Keep the default cap.**
Caveat: different seeds, one run each; the direction is large but the
magnitude is not calibrated.

## 7. Wall classification

`gated_wall_verdict` on this run returns **INSUFFICIENT** — `4 progress
records < MIN_RECORDS=12`. That is the correct answer for a 4-minute probe
and not a finding: the classifier needs ≥12 one-minute records plus a
`WINDOW_RECORDS=10` window, i.e. **≥13 minutes of run** before it can rule.

So the question the spec poses — *does the taxonomy read Zelda as gated
(item-gated) as D2 predicts?* — is **not answered by this bounded attempt
and must not be reported as answered.** What can be said now is that the
4-minute segment is plainly still expanding (cells 1434 → 2735, +18%/min at
the end; rooms 11 → 15), which is `COVERAGE_LIMITED` territory, not a wall.
The 30-minute run is what produces a rulable verdict; re-run:

```
.venv/bin/python -c "
from src.training.wall_taxonomy import telemetry_from_paths, gated_wall_verdict
t = telemetry_from_paths('runs/<out>/progress.jsonl',
                         archive_path='runs/<out>/archive.pkl', label='zelda')
v = gated_wall_verdict(t); print(v.wall_class, v.evidence)"
```

## 7b. Reading the telemetry the spec asked for

Rooms visited is not printed by the solver; it is one line over the archive
(no new script, and the same `key[-5]` projection the taxonomy uses):

```
.venv/bin/python -c "
import collections
from src.training.wall_taxonomy import _StateDroppingUnpickler
cells = _StateDroppingUnpickler(open('runs/<out>/archive.pkl','rb')).load()
a = collections.Counter(k[-5] for k in cells)
print('rooms visited:', len(a))
print('as (row,col):', sorted(((v>>4, v&15) for v in a)))
print('cells/room:', dict(sorted(a.items())))"
```

Growth of that number over successive flushes is the bounded attempt's
success criterion; `max_room` / `max_sect` in `progress.jsonl` are the
per-minute proxies.

## 8. Open items for the next attempt

1. **A real death signal.** Find the CONTINUE/SAVE/RETRY menu byte so
   `player_state` + `death_states` can carry death and `lives` can stop
   terminating lineages on chip damage.
2. **Item/rupee/key counters**, once a probe can actually cause a pickup —
   correlate the HUD digit boxes (rupees ≈ rows 23-32 × cols 88-110; keys
   ≈ rows 38-47; bombs ≈ rows 48-55) against RAM.
3. **The D2 question proper.** Cave/dungeon entry is a *fade*, not a scroll,
   so the §2 transition detector does not see it. A dungeon-entry probe needs
   a fade-aware event track before "item-gated" can be tested at all.
4. **Frontier depth axis.** If east/south bias becomes the limiting factor,
   the honest alternative is a count-based / novelty selection over `area`
   rather than `max(area)`, which needs no new addresses.
