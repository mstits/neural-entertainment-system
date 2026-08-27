# The Contra gx-3072 wall — what it physically is, and whether we moved it

**Date:** 2026-08-27
**Ledger: EXHIBITION, without exception.** Every number below is Go-Explore
search output or instrument measurement. No policy was trained for this game
and no honest-protocol evaluation was run. Nothing here may be described with
"the AI learned", "the AI plays", or "the AI beat" — see `CLAIMS.md`.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc`.
**ROM** `roms/Contra (USA).nes` sha256
`26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519`.
**Start state** `roms/Contra (USA)_start.state.bin` sha256
`b99f9be8e0266f6dbe8ac71bc591b0deec08e66e7925707d265965a4aab922c3`.

---

## 1. The two answers, up front

**What is the gx-3072 wall?** A **screen-locked lethal arena behind an exact
hard camera stop**. The screen byte `$0064` reaches 12, the fine-scroll byte
`$0065` goes to 0, and both stop dead. The player is **alive and fully under
control inside the lock** — it responds to input at or above the rate it does in
open field, and can make 20–74 *irreversible* byte changes per state — but
nothing it does moves the camera. It is not a freeze, not a scripted input-dead
window, not a corpse, and not a timer.

**Did we move it? No.** Best verified gx is **3072**, against a prior of
**3072**. No trajectory in this campaign exceeded it, so **there is no tape to
preserve** and `docs/receipts/` gains no Contra tape. What the campaign produced
instead is (a) the first physical characterisation of the wall, and (b) a
falsified class of attack, stated precisely enough to stop the next person
repeating it.

> **The Rygar lesson, applied.** That campaign threw fourteen genuinely
> different search hypotheses at a wall nobody had characterised and produced
> zero forward progress. Its most useful output was establishing *what kind of
> problem the blocker was*. This campaign characterised first and fanned out
> second, and the fan-out was budgeted at roughly half a core-hour per arm
> rather than twenty hours. Ruling out a class is the deliverable.

---

## 2. What is physically happening at 3072

Nine campaigns hit this number without anyone establishing what was there. This
section is the durable output of the work and stands whether or not the wall
ever moves.

### The camera stops, exactly

`progress = $0064 << 8 | $0065`, so `3072 = 12 × 256` exactly: the screen byte
sits at 12 with fine scroll frozen at 0. The stop is not a range, an
oscillation, or a slow crawl — it is one value.

| Measurement | Camera states observed while alive |
|---|---|
| 300 sampled banked wall cells on restore (my own measurement, §7) | `{(12, 0)}` |
| 14 held button masks × 8 states × 40 steps | `{(12, 0)}` |
| 87,464 surviving 6-step bursts, 524,784 alive steps (survival-only search) | `{(12, 0)}` |
| A8's 265,860-step latch-novelty search | `{(12, 0)}` |
| A7's 285,672 worker-steps, gx-gated liveness | `{(12, 0)}` |

The approach immediately before it reads `$0064 = 11` with `$0065` sweeping
178 → 255, so 3072 is reached by **ordinary carry from 3071**, not by a wrap or
an aliasing artifact. The archive's gx buckets run contiguously 0..192 with
nothing above.

### The archive shape is an absorbing boundary

Measured directly on `runs/play_one_well/contra/solve20/archive.pkl` (16,298
cells):

```
bucket 187:   37 cells
bucket 188:   47
bucket 189:   49
bucket 190:   60
bucket 191:  115
bucket 192: 1331   <- gx 3072, 8.2% of the whole archive
```

An 11.6× jump into the terminal bucket and nothing beyond it.

### The agent is alive in there, and in control

This is the part that had never been established, and it is what makes the wall
interesting rather than merely hard.

- **Alive:** 300/300 sampled gx-3072 cells restore with `lives $0032 ≥ 1`,
  camera exactly `(12, 0)`, gx exactly 3072. Every input measurement in this
  campaign is truncated at the first step its lineage loses the life it was
  banked with; no post-death tail is scored anywhere.
- **In control:** on an alive-only window, 6 of 8 sampled wall states show
  93–317 RAM bytes, 4–95 OAM bytes, and 4–17 visible sprite slots taking
  different values across 14 held masks. Calibration: an **alive open-field
  positive control** gives 185 / 88 / 9; a **dead negative control** (`lives == 0`,
  game-over animation) gives 8 / 0 / 0. The wall's input reach **equals or
  exceeds ordinary open-field play**.
- **Irreversibly in control:** 20–74 bytes per state are *agent-latched* — they
  leave their entry value under some mask, never return, and do not do so under
  NOOP. Open-field control 39, dead control 1.
- **Not input-dependent, under any mask:** the two progress bytes. `$0065` takes
  exactly one value (0) and `$0064` exactly one value (12) in every arm.

### It is not a timer

486 consecutive alive steps (1,944 frames, ~32 s on one life) and 524,784
cumulative alive steps at the wall produced no camera motion and no progress
above 3072.

### What is on screen, and what moves

Described from our own rendered frames only. A fixed frame that never scrolls:
a foliage-topped rock face with stepped ledges over the left two-thirds, and on
the right a tall grey mechanical assembly in front of a flat blue vertical-barred
panel, with a grey box at ground level carrying a raised X-plate over a recessed
dark circle. Under held NOOP the terrain, the structure, and the blue panel are
**pixel-identical frame to frame**; per-8×8-tile temporal variance puts every
changing tile inside x 32–184, y 56–232, and the right ~28% of the screen is
bit-identical across the whole window. Only figures, small travelling spheres,
brief starburst puffs, a rotating nub and the X-plate centre change.

Underneath, 419–566 of the 2048 RAM bytes change during a NOOP hold (alive
window only), and 260–433 of those are **not** input-dependent — the wall runs
on its own. They cluster in the sprite shadow page `0x0200–0x02FF` (stride-4
groups), a band of ~40 counters/phases in `0x0100–0x017F` with tiny cycling
alphabets, and scattered zero page.

> **The caution that is the whole point.** The dead negative control — `lives ==
> 0`, game-over animation — moves **766 bytes, more than the wall does**. Byte
> motion is not evidence of agency and must never be reported as such. Every
> liveness claim above rests on a *differential* against that control, not on a
> raw churn count.

### Anti-vacuity: three reverts, in code

Each check was shown to move when its mechanism was removed.

1. **Player-control check** → dead control (`lives == 0`): 2–8 input-dependent
   RAM bytes, 0 OAM, 0 sprite slots, 1 latched byte, against the wall's
   93–317 / 4–95 / 4–17 / 20–74.
2. **Camera check** → alive open-field control: `$0065` sweeps 0..255 and the
   pair visits ~1,000 states.
3. **Camera check under the same survival search**, reverted by seeding it 272 px
   earlier (gx 2800): identical code, identical budget, it reports **238 distinct
   camera states while alive**, sweeping (10,243) → (11,255) → (12,0), and stops
   there. At the wall the same call returns a single state. That run also
   independently re-derives the stop point from the other side.

### Survival, and the fatal-window subset

Survival at the wall is hard but not impossible: random play survives a median
26–51 steps (~2–3 s); a survival-only Go-Explore reached 486 consecutive alive
steps on one life; A2's search reached 390. **A genuinely long safe window
exists only via a searched action sequence, never a fixed hold.**

**2 of 8 hand-banked wall states are input-DEAD** — all 14 masks die at the
identical step (10 and 12) with only 2 input-dependent bytes. These are
already-committed fatal windows, not evidence about the wall, and any future
work must screen wall states for input-liveness before using them as roots.
A6 independently reproduced both under a different action prior (death at steps
9 and 11).

---

## 3. Did the wall move? No — and here is exactly what was ruled out

Eight attacks were fanned out, each at roughly half a core-hour, each varying at
least one of {cell definition, restart distribution, action prior, maximised
quantity}. **All eight report `beat_3072: false`, `max_gx: 3072`, alive at max,
`tape_path: null`.**

| # | Angle | Scale | Result |
|---|---|---|---|
| **A1** | Boundary-resident differential probe; settle + eligibility gated; fire-free control pivoted NOOP → DOWN mid-run | 146 roots × 6 actions, 40,604 alive steps | **0 candidate bytes.** Naive gate's 8–10 hits hand-inspected and **withdrawn** as truncated cycles and onset transients |
| **A2** | Kill-condition: boss-typed HP array into the cell key, hunt joint-zero-while-alive, verify by NOOP continuation | 81,244 alive steps, 830 zero-hits, 15 replay-verified | **Falsified.** Joint-zero is a live-suppression mux reading, not a kill: hp refills within 1–2 steps once fire stops, 0/15 sustain ≥20 zero steps |
| **A3** | Rhythmic / mashed / full-lattice-swept / iid-random action priors | 30 roots × 10 conditions, 25,948 alive steps | **Negative.** Every gate hit resolves to an already-characterised category (input-active flag, vertical-state byte, projectile slot, jump height) |
| **A4** | Differential scored for **accumulating** one-way movement, with a pre-wall positive control | 6 screened-survivable roots × 11 actions, 18,449 steps | **0 bytes pass.** Positive control recovers real progress (lo net +26..98, monotone 0.99–1.0) — so the null is meaningful |
| **A5** | Cell-key augmentation with the six autonomous zero-page cyclers + real go-explore churn across 10 route lineages | 23 roots, 168,768 steps, 4,026 restarts | **Negative.** 11,669 distinct augmented cells (8.8× the un-augmented key) — the axes are live — and none co-occurs with camera motion |
| **A6** | Typed-HP kill-through: maximise cumulative damage, not gx, under a survival-selected beam | 20 roots × 800 gens, ~384,000 steps | **Falsified.** HP driven to floor in 6/20, damage plateaus by gen 25–50 and sits flat for 750 more; camera `(12,0)` in 100% of samples |
| **A7** | Entity-slot + zero-page-cycler key, least-visited-weighted resume, brief-tap risky action prior | 285,672 worker-steps, 5,309 bursts | **Negative** — and it caught a contamination artifact others would have published (§5) |
| **A8** | Latch-signature novelty search over temporal action *sequences* | 265,860 steps, 90 cells, 102 flip signatures | **Negative.** Up to 73 of 74 candidate latched bytes flipped simultaneously while alive; camera never left `(12,0)` |

**The class that is ruled out, stated precisely.** From banked wall roots, over
roughly 1.3 million emulator steps in mixed accounting units:

- **No single held input** releases the camera — 14 masks in the characterisation,
  11 in A4, 6 in A1.
- **No rhythmic, mashed, lattice-swept, or iid-random input pattern** releases it
  (A3). That is four action-prior families beyond the fifteen static masks.
- **No temporal action sequence found by novelty search over the agent's own
  latch space** releases it (A8), including states with 73 of 74 latchable bytes
  flipped.
- **No single RAM byte behaves like a release counter** — nothing settles one-way
  under sustained fire against a matched fire-free control (A1), and nothing
  accumulates one-way and NOOP-flat (A4), across all 2,046 non-progress bytes.
- **Destroying the tracked hardware does not release it** (A2, A6), and the
  "destroyed" reading is itself a transient multiplexing state rather than a
  persistent kill.
- **Enriching the cell key does not release it** — not with the six autonomous
  zero-page cyclers (A5), not with entity-slot counters (A7), not with raw
  boss-HP (A2), not with latch popcount (A8).
- **It cannot be waited out** — 524,784 cumulative alive steps at the wall.

**The limit of that claim, stated as plainly.** The eight attacks are less
independent than they look: five root from `solve20/archive.pkl`, three from the
same session's `head_wall_*.state` files, one from both. `solve20` is a single
run booted from the single shared savestate. **No attack ran a fresh
from-power-on search or reached the lock by another route.** What is jointly
established is *"no escape from inside the lock, from banked roots of
essentially one lineage, under input-pattern / cell-key / kill-condition
variation at ~0.5–1 core-hour each."* That is real and it is worth having. It is
**not** "the lock cannot be passed". `attackable_by_search` is falsified for the
**boundary-resident family**, not in general.

---

## 4. The prior was weaker than it looked — and the wall is still real

The brief that opened this campaign cited "3,030 verbatim occurrences across 9
of 10 independently-run campaigns". Both halves needed deflating, and the
deflation is arithmetically exact.

**The count is exactly 2× double-counted.** 3,030 grep hits = 1,508 in
`progress.jsonl` + 1,508 in the tee'd `.log` twins + 14 in a JSON receipt. Every
run writes each telemetry line to both paths.

**The 1,508 lines are not 1,508 trials.** `max_gx_in_max_area` is a *running
max*, and **17 of 18 runs already read 3072 in their first 60-second telemetry
window**. A running max can only ever re-report itself, so no later line is an
independent test of "can the search pass 3072".

**Nine of eighteen runs are resumes of two parents.** Chain A is
`stage1_v5 → v6 → v7 → v7b → v8`; chain B is
`contra_reentry/r1_ortho → {gate_opener k0/rev0, k0/rev1, k0v2/rev0, k0v2/rev1,
k0v21/rev0}`, all five with `--seed 0` and identical `root_sha
b99f9be8e0266f6d` — which is the start state's own sha256 prefix. A resumed run
does not re-derive 3072; it *prints its parent's number on the seed line before
taking a step*. Eight of the nine fresh-archive runs boot from that same single
savestate, and one of the nine is the broken-observable control.

**Restate the prior as: ~8 independent searches from 2 root states, all pinned
at 3072 within 60 seconds.**

**That is still a real, reproduced wall** — it is simply much cheaper to have
established than 20.7 h and 162 GB implies. And it was reproduced fresh at HEAD
for this campaign: a new 12-worker, 7-minute, 856,548-step run hit 3072 inside
60 s and never exceeded it.

### The wall is not an artifact of the cell definition — four ways

- **No cell collapse.** 1,331 distinct cells at the wall bucket in `solve20`
  (8.2% of cells, 32.3% of visits), 6,190 in `r1_ortho`, 47,429 in
  `stage1_v4_localkk`. The search was never starved of return targets.
- **Invariant to a 2× key-resolution change.** v2–v5 ran at `gx_bucket 8 /
  y_band 16`; the later runs at the defaults 16 / 32. Halving both spatial
  resolutions did not move the wall one pixel.
- **Invariant across four materially different key compositions.** `hp`/`state_sig`
  live in some runs and constant-dead in others; `kk` live in one and dead in the
  rest. Different keys, same 3072.
- **Internal falsifier.** The one run whose key genuinely *did* collapse —
  `stage1_baseline_collapsed_cells`, 13 cells over 1.3M records — walled at
  **2816, not 3072**. When the cell definition really collapses in this codebase
  it produces a different and lower number.

### The missing axis was tested, and it is dead

The key has no player-screen-x term. Measured: across 400 banked wall cells,
`player_x $0334` spans 25..136 with 99 distinct values at gx 3072. Then 40 wall
roots × all 11 profile actions × 400 held steps, terminated at lives-drop:
**`px_max = 136` exactly, every action, every root**; `gx_max = 3072`, every
action, every root. Re-run *without* death termination: still 136. In the
scrolling region px is capped at 128 — the scroll-lock signature, self-validating
the byte's meaning with no external map.

So the camera stops **and the player stops**, 8 px past the scroll-lock position.
There is no hidden forward gradient the key was blind to.

---

## 5. Defects found (the campaign's secondary yield)

### 5.1 `max_gx_in_max_area` is a binary inside the lock, not a gradient

`max_gx_in_max_area = max_gx_in_area.get(max_area, 0)` (`go_explore_solve.py`
line 7604), and `solve.area` is unset for this profile so `area()` returns
literal 0 for every RAM state and `max_area` is pinned at 0. The metric therefore
degenerates to the global running max of `progress`, which is **frozen by
construction at 3072** inside the lock. Deep-bias scoring carries no signal there
even though the archive still discriminates 1,331 distinct wall cells on its
other key dimensions. That is the specific, fixable defect that shaped the whole
fan-out: **stop grading this wall with a running max**.

### 5.2 Five of the eleven cell-key fields are structurally constant

The key is `(sect, tb, kk, psig, loops, route_sig) + cell_fn(ram)` (line 4717).
For `configs/contra.yaml`, `area` is 0 (no `solve.area`); `sect` and `psig` can
never move because `_transit` requires `room_id` to change and `room_id =
level_key + (area,) + room_sig` is the constant `(0,)` with `level_key: []`, no
`area`, no `room_sig`; `tb` is 0 without `--time-bins`; `kk` is 0 without
`--kill-key`, which was off in 7 of the runs whose profile carefully builds
`entity_slots` machinery. That also kills the score's leading term: `score =
sect * 10000 + gx + score_bonus` (line 4747), so the entire 10,000-per-transit
cross-room axis is unreachable and the only gradient is gx plus the HP bonus.

Anti-vacuity on that claim: 500 random RAM snapshots per profile give
`contra.yaml` and `contra_blank.yaml` **1 distinct** `room_id`/`area`/`level_key`
each, while `castlevania.yaml` gives `room_id` 215 and `kirby.yaml` gives `area`
220. The constancy is a property of the profile, not a broken probe.

### 5.3 A stage clear would be unrepresentable in the key and invisible in the headline

`progress` hi is `$0064`, which `contra.yaml`'s own `ram_mapping` calls a
within-stage **screen counter that resets to 0 each stage**. A genuine stage
advance therefore collapses gx toward 0, trips the per-trajectory loop-back rule
(`_lgx < prev_gx - 100`, line 7500), scores near zero, and is archived at a low
gx bucket **indistinguishable from ordinary early-stage cells** — while
`max_gx_in_max_area`, being a max, simply does not move. `current_level $0030`
and `boss_defeated $003B` are named in `ram_mapping` and wired into neither.
**This must be fixed before any post-wall run**, but it cannot explain a wall
the search never crossed.

### 5.4 `score_bonus`'s kill incentive is erased mid-fight

`_typed_hp` returns `sum(live) if live else self._bt_start`, and `score_bonus`
is `max(0, _bt_start - _typed_hp) * 2000`. When no tracked type is currently
multiplexed into the four tracked addresses, `_typed_hp` returns the sentinel and
the bonus reads **exactly 0 — identical to "never engaged"**. A6's traces show
that happens routinely mid-fight, not only at a true clear. A gx/score-following
search therefore has its kill incentive erased far more often than intended.
Independent of this campaign's verdict; a legitimate small fix is to track a
monotone kill count separately from the instantaneous HP sum.

### 5.5 A lives-equality liveness gate is measurably insufficient here

**A7's methodology is the most valuable single item in the fan-out.** Its first
pass reported `camera_values = {(12,0), (0,0)}` — apparently the camera leaving
the lock. Rather than publish it, it built a tripwire that replays the exact
trajectory and freezes on the reading. The `(0,0)` traced to an archive-resumed
cell whose `entry_lives` was **already 0** at the priming step: a
post-death/game-over artifact invisible to a lives-equality gate, because the
lives byte holds *flat* across the game-over screen so "lives never changed this
burst" is trivially true there. Every `(0,0)` sample had gx exactly 0, so
`beat_3072` was never at risk — but the cell and camera diversity metrics were
contaminated. Pass 2 additionally required `gx == 3072` at burst entry and every
in-burst step, rejected 91 contaminated entries, and never again saw any camera
value but `(12,0)`. The contaminated pass was kept on disk rather than deleted.

**The lesson generalises past Contra: `lives == entry_lives` is not a liveness
gate on a ROM whose game-over screen holds the lives byte flat.** Pair it with a
positional invariant.

---

## 6. Corrections landed here

- **WITHDRAWN — "NOOP/A/B/up+B/down/down+B all survive the full 400 steps at the
  boundary" at `px == 136`.** It does not reproduce. My own measurement: **0 of
  39** px==136 roots survive 400 NOOP steps (mean 31.7, median 31, max 78).
  Three attacks hit this independently: A1 found 0/183 reaching even 120 steps
  and pivoted its fire-free control from NOOP to DOWN mid-run; A4 found 0/173
  surviving 300 and screened px==136 out entirely; A7 measured a median of 34.
  A real observation was attached to the wrong sub-population.
- **The long window exists at a different position.** It lives around px 55–63.
  My sample: 4 of 22 survive the full 400 NOOP steps, mean 103.7 — against 0 of
  39 at px==136. A4's 6 usable roots all landed at px 55–63 independently. (A
  parallel independent sample measured 25/60 at px 55–63; the rate is
  sample-dependent, the *split* is not, and only the px 55–63 group produces any
  full-400 survivor at all.)
- **WITHDRAWN — `runs/play_one_well/contra/wave2_geometry.json`'s unattributed
  `upright_fire px_max: 249`.** Not reproducible with or without a post-death
  tail, and its producing script is not in the repo. Treat as withdrawn pending a
  named harness.
- **WITHDRAWN — Contra's banked "162 distinct" odometer figure**, which was ~94%
  game-over animation.
- **Corrected in place:** `runs/contra_wall/A4/probe_release_report.json` carried
  two stale summary strings ("50 diverse banked wall roots", anti-vacuity "at
  gx~2800") contradicting its own data block (`n_roots_usable: 6`, `prewall_gx:
  1297`). The submitted narrative was accurate on both points; the strings are
  now corrected and the correction is recorded in the file.
- **Corrected — the mechanism behind Contra's clear-hook ceiling.**
  `DETECTOR_REPAIR_2026-08-26.md` line 141 says "`tally` has no referent in this
  game, so the 2-of-2 vote is unreachable". That is **wrong on its stated
  mechanism** and was itself an unmeasured assertion about the title. `tally`
  does not fail to fire — it fires on **58 of 58** checks. It fails to
  *discriminate*. See §7.

---

## 7. Can Contra's clear hook ever be validated? Not yet — and the gating task is a stage boundary

**Answer: `detector_validatable: false`.** The hook is structurally sound and
empirically non-vacuous, and it has still never been shown a clear to test
against.

### The quorum, from code, no ROM

`scripts/clear_reachability.py::clear_quorum(configs/contra.yaml)` returns
**FIREABLE, roster live, ceiling 2.0, required 2.0**. ALIVE: `coord`
(S_TRANSITION, w=1 — "progress is a 16-bit {lo, hi} pair") and `tally`
(S_CADENCE, w=1). DEAD: `apu` (`clear.apu_weight = 0`) and all six shelf signals
(`scene_cut`, `room_fp_transition`, `input_lock`, `oam_quiesce`, `entity_wipe`,
`lock_release_novelty`) — wired but unarmed, because `configs/contra.yaml`
declares no `solve.clear.signals` block, so no signal object is constructed for
any of them.

**Live-roster ceiling is exactly 2.0 against a required 2.0: zero slack.**

`level_key(ram) == ()`, so `is_clear`'s opening test is `() > ()`, **False
always** — the same compile-time constant that holds on 152 of 155 profiles.

### The critical amendment: measured, the quorum flips

`tally`'s null fire-rate was **measured**, not assumed. Five random-play episodes
from the profile's own start state, each terminated at the first `lives $0032`
decrement with no post-death tail: **58 detector checks, `tally` fired on 58/58 —
null fire-rate 1.00.** `coord` fired 0/58; `entity_wipe` 0/58.

Feeding that measurement back in (verified at HEAD for this write-up):

```
clear_quorum(contra.yaml, null_rates={'tally': 1.0})
  -> UNREACHABLE, ceiling 1.0, required 2.0
     tally = DEGENERATE (measured null fire-rate 1.00 >= MAX_NULL_RATE 0.05:
     it fires on ordinary play, so it carries no bits and cannot corroborate)
```

**So the shipped 2-of-2 is a 1-of-1 `coord` vote wearing a corroborator's
clothes.** The hook's advertised safety property — *"only fires when BOTH agree,
so an ordinary score tick or scroll can't fake it"* — is **vacuous for this
profile**. `score_tally_windows` is address-free and finds any periodic
anti-correlated byte pair, so an animation or frame counter serves; a
"timer→score conversion" was never its actual precondition.

**Any Contra null from this hook is VOID, never a miss.** It must not enter a
hit-rate denominator and must never be cited as "searched and found none".

### `coord` can fire, and has only ever fired falsely

Positive control on real Contra RAM (deliberately *not* death-terminated — this
is a false-positive characterisation, not a progress claim): seeds 3/7/11, 1600
steps, lives 2→0. `coord` fired 12/80 checks in all three; largest single-step
progress drop 1061/1061/1079, well past `COORD_RESET_DROP_MIN = 300` and landing
at 0 ≤ `COORD_RESET_ABS_MAX = 200`. The joint `tally`+`coord` latch lands at the
**same step** in all three, because `tally` is already 1 everywhere.

**Every observed joint fire is the game-over/reset arc** — RAM-indistinguishable
on these two signals from a stage load. The lives-drop veto does not catch it
(lives are flat at 0 by then, not decrementing that step).

That specific trigger is nevertheless unreachable through the live pipeline:
`Solver.start_lives` is fixed **once** at `seed()` (line 5561), never
per-lineage, so `is_dead`'s `(start_lives - lives) % 256 in 1..8` latches
permanently at the first life loss, and `observe()` (line 4524) returns "dead"
before `is_clear`/`det.push` is reached. A live worker cannot process a second
death in one detector instance.

### Witnessed clears: ZERO, by four independent counts

1. **19 `solutions/` directories across 18 Contra solver runs — all empty, 0
   files.** Every tail row reads `solutions: 0` with `max_gx_in_max_area: 3072`
   (one 13-cell baseline at 2816).
2. **No `sol_*` file matching contra exists anywhere in the repo.**
3. **772 rows of Contra PPO metrics** across `checkpoints/contra*`:
   `vanilla_ppo_clears` max 0, `success_rate` max 0, `max_screen` tops out at 10
   of the wall's 12.
4. `runs/clear_recensus/contra/SUMMARY.json`: `current_level $0030` and
   `boss_defeated $003B` read **exactly 0** across 720 restored banked cells
   spanning screens 0..12, 2,100 live scripted steps, four full 2-life game-over
   arcs, and 13 screen-to-screen transitions. Both are measured **non-vacuous**
   and neither has ever been witnessed to move.

### The reordering — the same one that was the real result on Rygar

> **The gating task is reaching a stage boundary, not tuning a detector.**

Produce **one** Contra trajectory that survives past the mid-stage-1 fixed-camera
section and crosses a stage transition. This is a search / skill-coverage
problem. Until a clear exists, the predicate can only ever be shown *not* to
fire, and **every "retune stride/window" instruction — in `configs/contra.yaml`
and in `DETECTOR_REPAIR_2026-08-26.md` — is unfalsifiable.**

The concretely orderable sub-task, *conditional on a clear ever being captured*:
derive `level_key` from `current_level $0030` / `boss_defeated $003B`, both
already measured non-vacuous.

**`solutions: 0` for Contra is weak-but-real evidence** — the hook *could* fire,
unlike Rygar's, whose `_clear_mode` is `None` and whose quorum is UNREACHABLE by
construction. But weak-but-real still **cannot separate "never beaten" from
"beaten and not detected"**, because the predicate has never returned a true
positive on any game.

---

## 8. What I verified myself before landing this

Written fresh for this write-up, importing nothing from any attack item's own
harness: `runs/contra_wall/_LANDING/land_verify.py`, output
`runs/contra_wall/_LANDING/land_verify.json`.

```
[arch]     16,298 cells; max gx_bucket 192 -> gx 3072; 1,331 cells at max (8.2%)
           buckets contiguous 0..192, none above; 191:115 -> 192:1331
[restore]  300 sampled wall cells: alive 300/300
           camera states {(12, 0)}   gx values {3072}
           88 distinct player_x spanning 0..136
[noop400]  px == 136 : 0/39 survive 400   mean 31.7   median 31   max  78
           px 55..63 : 4/22 survive 400   mean 103.7  median 41   max 400
```

Separately verified at HEAD: `clear_quorum` returns FIREABLE/2.0/2.0 as shipped
and UNREACHABLE/1.0 with the measured `tally` null of 1.00; `_typed_hp`'s
sentinel fallback is present exactly as §5.4 describes; `area()` returns 0 when
`_area is None`; `contra.yaml` ships `level_key: []` with no `area` and no
`room_sig`.

**No tape exceeds 3072.** I scanned all 18 attack receipt JSONs — including raw
per-step traces — for any progress-keyed value above 3072: **all clean.** Zero
`BEAT3072_*` / `BREAKOUT_*` / `breakthrough` files exist on disk. The
extraordinary claim was never made, so no proportionate evidence is owed, and
`docs/receipts/` correctly gains no Contra tape.

---

## 9. What to do next

**Do not spend another probe on input-pattern variation at this boundary.**
Four action-prior families and fifteen static masks are now cleared with zero gx
movement and zero out-of-known-region byte candidates.

Ranked, cheapest first:

1. **Reach the lock by another route before attacking it again.** The single
   biggest weakness in the evidence is that every attack rooted in one lineage
   from one savestate. A fresh from-power-on search — or entry from a different
   approach lineage — costs little and tests whether the lock is a property of
   *this* entry or of the position.
2. **Fix the metric before, not after.** Wire `current_level $0030` and
   `boss_defeated $003B` into `solve:` so a stage advance is representable in the
   key and visible in the headline (§5.3), and split a monotone kill count out of
   `score_bonus` (§5.4). Both are small, both are independent of the wall's
   verdict, and both must precede any post-wall run — otherwise a clear, if one
   happens, is archived as an ordinary low-gx cell and nobody sees it.
3. **Adopt A7's liveness pattern repo-wide** (§5.5): never gate liveness on
   `lives == entry_lives` alone on a ROM whose game-over screen holds the byte
   flat.
4. **If Contra is revisited on the release hypothesis**, the untested residue is
   narrow and should be named honestly: a release gated on a specific *sequence*
   rather than any constant hold and outside A8's latch space; a condition on an
   address outside the 2 KB CPU RAM window every probe read; or a condition tied
   to a specific enemy type that a constant-hold protocol never targeted. None of
   these is cheap, and none has a candidate.
5. **Otherwise, re-shelve Contra with this receipt.** Saying the wall is not
   attackable by the boundary-resident family is the correct outcome, and it is
   recorded here so nobody pays for it twice.

---

## 10. Purity and ledger

**Purity (Tier 3), held throughout.** No disassembly, no RAM maps, no
walkthroughs, and no recall of anything about this title. Every byte's role
above — death-tail marker, animation counter, projectile slot, vertical-state
byte, input-active flag — was inferred solely from its own measured time-series
shape under our own inputs, on this profile's own declared addresses and start
state. The screen description in §2 is read off our own rendered frames. The
litmus was applied: a party who had never seen this game could have derived every
statement here.

**Ledger: EXHIBITION.** Solver and instrument measurement only. No learned-
capability claim is made or supported anywhere in this document.

---

## Receipts

Tracked, in-repo:

- `docs/research/CONTRA_WALL_2026-08-27.md` — this document
- `CLAIMS.md` § *CONTRA WALL 2026-08-27* — the ledger entry
- `runs/contra_wall/A4/probe_release_report.json` — corrected in place (§6)

Under gitignored `runs/` (the eight attacks' own receipts and producing scripts):

- `runs/contra_wall/A1/` — `boundary_probe.{py,json,log}`, `inspect_candidates.py`
- `runs/contra_wall/A2/` — `A2_RECEIPT.json`, `hpkey_survive.*`, `verify_zero_hits.*`, `screen_survival.*`
- `runs/contra_wall/A3/` — `A3_report.json`, `rhythmic_probe.py`
- `runs/contra_wall/A4/` — `probe_release.{py,json}`, `run.log`, `extend_candidate.py`
- `runs/contra_wall/A5/` — `SUMMARY.json`, `run_a5.py`, `run_v2_forced_restart.log`
- `runs/contra_wall/A6/` — `A6_RECEIPT.json`, `phase1_trace.json`, `phase2_beam.json`
- `runs/contra_wall/A7/` — `attack_a7_results.json`, the superseded
  `..._PASS1_CONTAMINATED.json`, `anti_contamination_tripwire_repro.txt`,
  `root_manifest.json`
- `runs/contra_wall/A8/` — `A8_result.json`, `A8_latch_novelty.py`,
  `A8_verify_tape.py`, `latch_candidates.json`, `roots_used.json`
- `runs/contra_wall/_LANDING/` — `land_verify.py` / `land_verify.json`, the
  independent landing verification of §8

Characterisation receipts (session scratchpad): `CONTRA_WALL_RECEIPT.json`
(consolidated), `alive_input_{wall,ctrl}.json`, `what_moves_{wall,ctrl}.json`,
`latch_{wall,ctrl}.json`, `survive_archive.json`, `PREWALL_survive.json`
(the anti-vacuity revert), `wall_dynamics.json`, `approach.json`, and frames
`wall2_entry.png`, `wall_grid.png`, `wall7_noop_series.png`,
`deepest_survivor.png`, `noop_change_heatmap.png`.

**Do not trust cross-build trace replay for this game.** The 2026-08-01
`traces.pkl` action traces do **not** replay on today's core — 16 of 16 wall
traces max out at raw progress 1807–1815. Every state used in this campaign was
minted fresh.
