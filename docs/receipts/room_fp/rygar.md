# room_fp calibration receipt — Rygar (2026-08-26, R1-05)

Profile: `configs/rygar.yaml` at HEAD `2459e5e` (nes_core sha256_16 `54366c20d32f71cc`)
Tool: `scripts/room_fp_calibrate.py` (capture / mask / replay) +
`/private/tmp/.../scratchpad/rygar_roomfp_capture.py` (replays a real banked
trace through the same capture format — see "Fixture" below)
ROM: `roms/Rygar (USA).nes` (sha256 `d87a7b3250eb8d6af3b725169a02dab4…`)
Start state: `roms/Rygar (USA)_start.state.bin` (sha256 `9befb1cefd597130b1b3d80fbd77e936…`)

**VERDICT: DECLINED.** The mechanism hypothesis is CONFIRMED — Rygar's
transitions are blank-type and the odometer's scene-cut test is provably
blind to them (code citation below) — but the shipped room_fp settle
mechanism does not turn that into a usable room identity on this profile at
any tested settle value. `solve.room_fp` is NOT added to
`configs/rygar.yaml`. This receipt records the numbers so the next attempt
starts from a measured limit, not a guess.

Purity: every input is a hardware surface (2 KB physical nametable VRAM via
`Pool.peek_nametables`, PPU scroll odometer, scene ordinal, `odo_debug`
rendered-line count) — the same class room_fp already uses for Metroid and
Zelda. The `lives: 0x0303` byte appears only as a cross-check against the
already-onboarded death observable (banked in `configs/rygar.yaml`,
independently re-derived by the discovery pipeline, not a new RAM-map read).
No disassembly, no walkthrough, no recall of this title.

## Mechanism check (why the hypothesis is right)

`nes_core/src/ppu.rs::odo_fold_frame` (line 3122):

```rust
if n < 120 {
    // Mostly-blank frame ... nothing trustworthy to integrate.
    self.odo_blank = self.odo_blank.wrapping_add(1);
    self.odo_have_prev = false;
    return;              // <-- returns BEFORE the dx/dy scene-cut test below
}
```

The scene-cut increment (`self.odometer_scene += 1` on a >64px modal jump)
lives in the branch this `return` never reaches when a frame renders fewer
than 120 lines. Confirmed live: the deepest banked trace below scrolls
0 → 5,336 px with **zero** scene-ordinal bumps (`scene 0->0`) while
containing 26 blank runs. The odometer's own scene mechanism cannot see a
blank-type cut by construction — not a bug, a documented design choice
(`odo_blank` is the counter that IS supposed to catch this) that simply has
no room-identity consumer wired to it yet. room_fp is the intended
consumer.

## Mask (auto volatility, §4) — from this profile's own start state

```
capture --frame-skip 1 --script "noop*300"                          -> rygar_idle_fs1.npz
capture --frame-skip 1 --script "right*30,left*30,right*20,left*20" -> rygar_walk_fs1.npz
mask rygar_idle_fs1.npz rygar_walk_fs1.npz --game rygar
```

Both scripts were pre-checked to stay camera-still: a step-by-step fs1
census (`right`-hold from boot) shows the odometer x pinned at 0 for 90
consecutive fs1 steps before scrolling begins (character walks to a
screen-anchor position first) — the walk script's 30/30/20/20 oscillation
stays entirely inside that dead zone (confirmed: `odo (0,0)->(0,0)`, 0
scene cuts, for both captures).

Result: **zero volatile bytes → `mask: []`.** Idle: 300 fs1 steps, 287
rendered (13 blank at the pre-title/boot fade) = **1** hash pre- and
post-mask. Walk: 100 fs1 steps, 87 rendered = **1** hash pre- and
post-mask. Same result as Metroid (0 volatile bytes) — Rygar's HUD (energy
bar / score) does not repaint inside this idle/walk window, so unlike
Zelda there is nothing to mask. This part of the protocol PASSES cleanly.

## Fixture: a real banked trace, not a hand-authored script

Rather than script a probe walk toward a guessed transition, the capture
replays `runs/rygar_ceiling_2026-08-26`'s deepest cell with a trace
(root state + 3,992 action indices, banked score 5,336 — the same run
the campaign brief cites as "CURRENT CEILING / config A") through a
capture recorder built to the exact `room_fp_calibrate` `.npz` schema
(nt / odo / scene / lines / meta), so the shipped `hash_stats` and
`replay_room_stream` routines run on it unmodified. Replay reproduced
x 0 → 5,336 to the pixel, confirming the trace and root state are exactly
what they claim to be.

`runs/rygar_campaign/R1-05/rygar_deeptrace_fs4.npz` — 3,993 steps @ fs4,
`lines` strictly bimodal (0 or 240, 2 distinct values — no partial-render
states on this profile), scene 0→0 (0 cuts) across the whole run.

## Anti-vacuity numbers: churn, not identity

Pre/post-mask distinct hashes over the fixture (mask is empty, so these are
identical — reported both per the gate's required format):

| capture | rendered frames | distinct hashes pre-mask | post-mask |
|---|---|---|---|
| rygar_deeptrace_fs4 (3,993 steps) | 3,490 | **649** | **649** |

This reproduces the phenomenon the campaign brief pre-registered from a
neighboring snapshot of the same live archive (868 distinct / 3,962 steps —
the archive was being written by the shared campaign between the two
measurements; same order of magnitude, same finding): raw per-step hashing
is dominated by continuous-scroll churn, not room identity. That much was
already known going in; the mask cannot fix it because there is no
per-byte noise to mask (0 volatile bytes, above) — the churn is legitimate
content change from scrolling, and the shipped design's answer to that is
the settle mechanism, not the mask. So the real test is: does settle turn
649 churning hashes into a small stable room set?

## Blank-run census (the actual transitions)

26 blank runs (`odo_debug` lines < 120) in the 3,993-step fixture: one
3-step run at boot (steps 0-2, pre-title fade), then **25 runs of exactly
20 fs4-steps (80 frames) each** — a much longer blackout than the
previously-cited "2-step" figure, which measures only the `lives`-byte
glitch at the *leading edge* of the blackout, not the blackout's own
duration. Every one of the 25 leaves the odometer x unchanged across it
(re-anchor, no integration — as designed) and every one shows `lives`
dip to 0 for exactly its first 2 samples before recovering to 1 for the
remaining 18 *while the screen is still blank* — confirming the debounce
boundary and the blank-duration are two different measurements of the
same event, not two events.

**Structural finding, unplanned but load-bearing for R1-06 or whoever
picks up the frontier-stall question next:** from x≈4,608 to x≈5,272 (the
last 664 px of this trace — the exact region the campaign's live solver
runs also report flattening at), the 25 non-boot blanks stop being
isolated and become **12 tight back-to-back pairs**, each pair landing at
the *same* x (±2 px) roughly 50 fs4-steps apart, with net progress of only
~55-64 px between one pair and the next:

```
1536                                    (isolated — the one clean transition)
4608, 4608   4672, 4672   4736, 4736   4789/4791, 4791
4855, 4855   4908/4910, 4910   4974, 4974   5028/5029, 5029
5093, 5093   5146/5148, 5148   5208, 5208   5272, 5272
```

24 of the fixture's final ~1,300 steps (480 steps, 37%) are spent inside
this double-blackout ladder for 664 px of net gain — reported here as a
measured pattern only (no interpretation of what triggers the pairing;
purity forbids guessing at game content), but it is a plausible mechanical
reason a fixed step budget stalls exactly in this band.

## Settle sweep — no value gives a small stable room set

`replay_room_stream` over the fixture, `mask: []`, `pan_odo: [128,384]`,
`warp_scene_min: 2` (Metroid/Zelda's own constants — nothing new invented
for this game), sweeping `settle`:

| settle | rooms minted | events fired | events adjacent to a real blank (±5 steps) |
|---|---|---|---|
| 3  | 243 | 301 | — |
| 8  | 104 | 121 | — |
| 14 (Zelda/Metroid's value) | **41** | 43 | **6 / 26** |
| 18 | 23 | 24 | 5 / 26 |
| 20 | 19 | 20 | 5 / 26 |
| 24 | 11 | 12 | 5 / 26 |
| 30 | 8 | 8 | 4 / 26 |
| 40 | 3 | 3 | 2 / 26 |
| 60 | 1 | 0 | 0 / 26 |

Reading both ends: low settle (3-14) mints a room on almost every ordinary
in-run pause — 37 of the 43 events at settle=14 fire with **no** blank
anywhere near their onset (`mid-scroll-stall`: the agent stops moving
mid-corridor — blocked by terrain, fighting, jumping — for ≥14 consecutive
fs4 samples, which this profile's continuous side-scroll produces
constantly and Metroid/Zelda's discrete static-camera rooms do not).
High settle (30-60) suppresses the false positives but suppresses the real
transitions right along with them — at settle=60 nothing fires at all,
because the trace never holds one screen still for 240 frames (4 s)
even where a real transition sits. **No settle value in [3, 60] gets
both a small room count and coverage of the real transitions**: the
best real-transition recall (6/26, at the design's own settle=14) still
misses 77% of the measured blackouts, because the double-blackout pairs
above are closer together than the settle window can survive — a second
blank interrupts the pend counter before the first can reach 14 stable
samples, so no adoption event fires for it at all and the room graph
silently carries the *old* identity across a real transition.

This is a structural mismatch, not a tuning gap: `fp_settle` mints an
identity on "camera held one view for N samples," which is the right
primitive for Metroid/Zelda's static-camera rooms but the wrong primitive
for Rygar's continuously-scrolling ones, where a genuine held-still moment
(blocked by a wall, fighting) is indistinguishable from a genuine new
room by that test alone. The blank-run gate itself (`lines < 120` for
~20 steps) is a clean, high-precision transition signal completely
independent of any of this — the census above found it with zero false
positives — but the shipped engine routes room identity through the
settle/fingerprint path, not through the blank-run gate directly, and
extending it to do so is a room_fp-engine change (shared with Metroid/
Zelda), out of scope for a single-profile calibration pass.

## Decision

`solve.room_fp` is **not** added to `configs/rygar.yaml`. Declining with
the number the anti-vacuity gate asks for: unmasked churn is 649 distinct
hashes over 3,490 rendered frames of one continuous run, and no settle
value between 3 and 60 turns that into a small, stable, transition-aligned
room set — the best case (settle=14) still mints 41 rooms while resolving
only 6 of 26 measured real transitions individually. Arming it as shipped
would silently under-count real transitions by ~77% while over-counting
"rooms" by treating ordinary combat/jump pauses as new locations — the
false-split failure mode room_fp's own design docs warn about, several
times worse here than on either precedent game.

**What would need to change, for the record:** a Rygar-shaped room
identity would key off the blank-run gate itself (a run of ≥K low-line
samples IS a transition, full stop — no fingerprint-settle needed to
detect it) and use the nametable fingerprint only to name the room on the
*other* side, with a short settle (a handful of samples) applied only to
post-blank content rather than to the whole stream. That is a change to
the shared T1 engine (`fp_settle`/`replay_room_stream`), not this
profile's config, and is left as a scoped, named follow-up rather than
attempted here mid-campaign against a shared file three other profiles
depend on.
