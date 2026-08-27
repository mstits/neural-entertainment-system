# `odo_blank`, the progress gate, and what each is actually worth

**Date:** 2026-08-26 (landed 2026-08-27)
**Ledger:** **EXHIBITION**, without exception. Every Rygar and Contra number in
this document is Go-Explore search output, scripted-hold probe output, or
uniform-random rollout output. No policy was trained for either game and no
honest-protocol evaluation (cold entrance, greedy, sticky p=0.25, jitter ±16,
50 episodes × 2 eval seeds = 100 pooled, `--eval-rng` per episode) was run for
either. Nothing here may be described with "the AI learned", "the AI plays", or
"the AI beat" — see `CLAIMS.md`.
**Purity (Tier 3):** every observable below is a hardware surface — PPU
blank-fold counts, PPU scroll odometer, OAM, rendered frames, render-line
counts — plus each profile's own declared lives byte, on its own start state.
No disassembly, no RAM map, no walkthrough, no recall of any title. Litmus: a
party who had never seen these games runs `scripts/transition_witness.py
discover` and `scripts/scene_cut_arming.py --audit` and gets the same numbers.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc` — re-built from HEAD with
`make build` before every measurement below and hashed byte-identical to the
installed `.venv` `.so`.

---

## The three questions, answered first

| # | Question | Answer |
|---|---|---|
| 1 | Does Rygar now have a defensible clear predicate? | **No.** The instrument exists and works; the *positive* does not. §1 |
| 2 | Is `odo_blank` a transition counter for the odometer cohort? | **Yes as an instrument** — it moves for **20 of the 25** cohort profiles (19 of 24 distinct ROMs), measured, not estimated. **But not yet as an armed gate on any of them**: driven as a measurement, 14 of the 23 gates armed on 2026-08-26 fire on play that cleared nothing, and all 23 are disarmed here. §2 |
| 3 | Was Contra excluded unfairly, and is its `gx 3072` wall real? | **Excluded unfairly — exclusion withdrawn.** The wall is **real as a position and blind as a verdict**. §3 |

**Verdicts changed under the fixed progress gate: 7 of 45 profiles.** All seven
move `SIGNAL UNUSABLE → INCONCLUSIVE`; `passed` changed for **zero** of the 45.
§4.

**Recommendation: Contra is the better next campaign target than Rygar**, and
the reason is structural rather than a matter of taste. §5.

---

## 1. Rygar has no defensible clear predicate, and the blocker is not the instrument

**Answer: no — and minting one now would be the fifth vacuous gate of the
week.**

What changed today is real and points the *other* way. `RYGAR_CAMPAIGN_2026-08-26.md`
§6 asserted that **"No instrument in the pipeline can count them"** about
Rygar's room transitions. That sentence is **withdrawn**. The pipeline had two
counters and the campaign only ever read one of them:

- `odometer_scene` bumps on a **rendered** scroll cut.
- `odo_blank` bumps on a **mostly-blank** frame (`n < 120` rendered lines), and
  the blank branch `return`s *before* the rendered scene-cut test ever runs
  (`nes_core/src/ppu.rs::odo_fold_frame`).

Rygar's transitions are blackouts. So `odometer_scene` read **0 cuts across the
entire 6,018-action tape** while `odo_blank`, edge-detected into runs, reads
**55 of 55** — agreeing exactly with the campaign's own hand count, and with an
independent instrument (`odo_debug` rendered-lines) that recorded
`blackout_run_lengths {"20": 55, "2": 1}`.

`scripts/transition_witness.py` is the committed instrument, and
`docs/receipts/rygar/clear_predicate_REFUTED.md` is the refusal.

Re-derived at HEAD for this write-up, driving the banked per-observation stream
(`docs/receipts/rygar/transition_streams.json`, 6,019 observations) through the
shipped class with a harness written fresh:

```
transitions 55   novel 2   revisit 53   deaths 0   splices 0   short_runs 1
areas [(0,14), (0,29), (3,16)]
```

Every number reproduces.

### Why the predicate is still refused

The instrument makes exactly two predicates available on Rygar, and **both
fabricate wins on Rygar's own deepest banked tape**:

| Candidate predicate | Fires on the tape | What actually happened |
|---|---|---|
| "a transition happened" | **55** | 27 round trips through one door |
| `level_key` on the blind-discovered area byte | **28** forward, TRUE on 4,978 / 6,019 observations (83%) | same |
| the witness's own `novel` | **2** | the honest count of new ground |

All three counts above were re-derived at HEAD from the banked stream, not
cited: 55 key-change events of which 28 are lexicographically forward; the
two-byte key reads TRUE on 4,978 of 6,019 observations and the one-byte variant
on 3,725 (62%). **It does not merely over-fire, it latches** — a solver
checking `is_clear` every observation banks on 62–83% of the tape.

The discovered area key alternates *perfectly* — `(3,16), (0,29), (3,16), …` —
for all 55 transitions, independently re-deriving the receipt's 27 zero-gain
segments from a different instrument. Artifact-free frontier before those 54
segments: **4,608 px**. After: **4,608 px**.

**The decisive blocker is that no Rygar clear has ever been witnessed.**
Re-counted: **71 of 71** `solutions/` directories for this profile are empty,
and every one of them is a compile-time constant — `configs/rygar.yaml` ships
`level_key: []` with no `clear:` and no `finale:`, so `is_clear` reduces to
`() > ()`, False for every state, re-verified over 2,000 random RAM states.
`solutions: 0` in any Rygar archive is **evidence of nothing**.

A predicate with no witnessed positive can only ever be shown *not* to fire.
That is precisely the shape of the four vacuous gates that shipped this week,
and it is the worst version of it: a false clear does not merely fail to
detect, it **fabricates a win**.

### What it would take, in order

1. **A witnessed positive.** One Rygar level end, on tape, replayable. Blocked
   by the **4,608 px search wall** — a 14-hypothesis campaign produced zero
   forward progress past it, so the wall is real and not a budget problem.
2. **Calibrate the run-length floor against that positive.** The current floor
   of 40 blank frames separates Rygar's death fade (14 frames, 36/36) from its
   door (78–79 frames, 55/55) with margins of 2.9× and 2.0×. A level *end*
   blackout has never been observed and may not sit in that band.
3. **Then** mint the predicate, and prove it fails with the mechanism removed.

Step 1 is the gating task, and it is a **search** problem, not an instrument
problem. That reordering is the campaign's real result.

`configs/rygar.yaml` is unchanged: `level_key: []`, no `clear`, no `finale`.
`tests/test_transition_witness.py` carries a guard asserting exactly that, so
the refusal cannot be quietly undone.

---

## 2. `odo_blank` is a transition counter — for 20 of 25 cohort profiles

**Answer: yes as an instrument, and it is the most valuable thing in this
document — but not yet as an armed gate on any of these games.** Those are two
different claims and the 2026-08-26 arming ran them together.

It matters because the odometer cohort has never had a transition signal that
could fire at all:
`clear_reachability` marks `coord` **DEAD** for every one of them (the camera
integral freezes and re-anchors across a cut and clamps backward-of-origin to
0, so the ≥300 px backwards drop `coord` requires never appears), which is what
puts the whole cohort at quorum **UNREACHABLE**.

### The measured count, not an estimate

Cohort **derived, not listed**: every `configs/*.yaml` whose
`solve.progress.source == "odometer"`. That is **25 profiles / 24 distinct
ROMs** (`metroid` and `metroid_roomfp` share one ROM). The brief's "26" is one
too many.

`odo_blank` moves — produces at least one edge-detected run — for **20 of the
25** profiles. **Counted twice, independently, and both counts are 20:**

| measurement | budget/profile | moves | does not move |
|---|---|---:|---|
| 2026-08-26 survey | 3 × 4,000 | 20 | `batman_the_video_game`, `double_dragon_ii`, `mega_man_3`, `ninja_gaiden`, `tetris_usa` |
| this audit, 2026-08-27 | 4 × 6,000 | 20 | `batman_the_video_game`, `double_dragon_ii`, `mega_man_3`, `power_blade`, `tetris_usa` |

The **count** is stable; the **membership** differs by one swap, and that is
reported rather than smoothed over. Dead in **both** measurements:
`batman_the_video_game`, `double_dragon_ii`, `mega_man_3`, `tetris_usa` — four
profiles. `ninja_gaiden` (0 → 3 runs) and `power_blade` (1 → 0) each moved in
exactly one and are marginal, established neither way. So the bounds are:

- **20 of 25 profiles** in each measurement taken on its own — the headline;
- **19 of 25** if you require it to move in *both*;
- **21 of 25** if *either* suffices.

Deduplicating the shared `metroid` ROM, the headline in distinct-game terms is
**19 of 24 games** — the same number under either measurement.

That `odo_blank`-versus-`odometer_scene` distinction was blurred in the
2026-08-26 arming, whose headline "the counter is alive for 24 of 25" was true
of `odo_blank` **or** `odometer_scene`, not of `odo_blank` — the counter
actually being armed. Four profiles were armed `kind: [fade]`, which votes on
the blank half, over a blank channel measured at zero.

### Required usage shape (all three, or it is not a transition counter)

1. **Edge-detect into runs. Never read the raw value as a count.** On the Rygar
   tape the raw counter terminates at **4,329** — **78×** the true transition
   count of 55 — because it counts blank *frames*, and one `frame_skip=4` step
   is four folds.
2. **Threshold on run LENGTH, calibrated per game from that game's own
   rollouts.** The raw counter takes the *same branch* for a death fade, a boot
   fade and a door, so a predicate reading "`odo_blank` moved" fires on **every
   death**. Measured over 18 rollouts, each confirmed to actually die and
   sustain it for 500+ observations: death fade **14** blank frames (36 of 36),
   boot fade **9** (18 of 18), door **78–79** (55 of 55). The floor of 40 sits
   2.9× above the death fade and 2.0× below the door, with zero overlap.
   **That number is Rygar's and must be re-earned per profile.**
3. **Treat it as per-trajectory, not monotone, across savestate restores.**
   `odo_blank` rides *inside* `OdoState` (`get_odo_state` emits it,
   `apply_odo_state` restores it), so a Go-Explore cell restore carries the
   saved value back in — measured jumping 273 → 90 on a real round trip. Any
   consumer assuming monotonicity across a restore is wrong. The witness
   reports `SPLICE` and banks nothing.

The counter never counts its own observer: under **12,036** restores — harsher
than any real solver — the transition count stayed at 55 with **zero**
fabricated events, and `odometer_x` was unperturbed at 6,242. The error is
always downward.

### The 2026-08-26 arming is withdrawn

Commit `4dd15ea` armed `solve.clear.signals.scene_cut` on 23 profiles off a
survey whose reproducer was never committed. Review found four ways the arming
could not survive contact with its own recorded evidence: profiles armed
`kind: [fade]` on a blank channel measured at **zero** runs; a death veto armed
over an admitted `lives: 0` placeholder and over a byte this repo documents the
same day as a 0↔255 flicker artifact; gates set at or below their own measured
null with the veto as sole guard; and no roster-level test reading the real
configs at all.

**I re-ran the arming as a measurement rather than a judgement, and it does not
survive.**

### What the measurement says, per profile

Driven at **4 × 6,000 steps** of each profile's own mixed-random play (2× the 2026-08-26 survey budget), through that profile's **own** armed `SceneCutSignal`, built from the real YAML by `clear_detect.build_shelf_signals`. Nothing was cleared in any of it, so **every fire is a false positive** and the whole observed `(d_scene, d_blank)` distribution is null.

| profile | `odo_blank` runs | scene runs | null `d_scene`/`d_blank` | gate as armed | **fires** |
|---|---:|---:|---:|---|---:|
| `1942` | 147 | 0 | 0 / 108 | `s1 b1 pan/warp/fade` | **42** |
| `bad_dudes` | 8 | 102 | 268 / 11 | `s14 b1 fade` | **19** |
| `batman_the_video_game` | 0 | 70 | 3 / 0 | `s6 b1 fade` | 0 |
| `bionic_commando` | 82 | 71 | 4 / 96 | `s6 b1 fade` | **52** |
| `blaster_master` | 20 | 12 | 4 / 29 | `s1 b1 pan/warp/fade` | **14** |
| `darkwing_duck` | 7 | 0 | 0 / 11 | `s1 b1 pan/warp/fade` | 0 |
| `double_dragon_ii` | 0 | 8 | 1 / 0 | `s5 b1 fade` | 0 |
| `ducktales_2` | 45 | 0 | 0 / 60 | `s1 b1 pan/warp/fade` | **113** |
| `ghosts_n_goblins` | 108 | 0 | 0 / 285 | `s1 b1 pan/warp/fade` | **102** |
| `gradius` | 68 | 0 | 0 / 96 | `s1 b1 pan/warp/fade` | **65** |
| `ice_climber` | 8 | 0 | 0 / 9 | `s1 b1 pan/warp/fade` | **20** |
| `journey_to_silius` | 24 | 4 | 1 / 28 | `s1 b1 pan/warp/fade` | **39** |
| `mega_man_3` | 0 | 43 | 8 / 0 | `s5 b1 fade` | 0 |
| `mega_man_usa` | 12 | 0 | 0 / 12 | `s1 b1 pan/warp/fade` | **12** |
| `megaman` | 12 | 8 | 2 / 18 | `s5 b1 fade` | 0 |
| `metroid` | 8 | 10 | 4 / 16 | `s5 b1 fade` | 0 |
| `metroid_roomfp` | 8 | 10 | 4 / 16 | `s5 b1 fade` | 0 |
| `ninja_gaiden` | 3 | 17 | 4 / 1 | `s6 b1 fade` | **5** |
| `ninja_gaiden_ii` | 3 | 11 | 2 / 1 | `s5 b1 fade` | 0 |
| `ninja_gaiden_iii` | 12 | 73 | 4 / 1 | `s6 b1 fade` | **15** |
| `paperboy` | 123 | 92 | 2 / 26 | `s5 b1 fade` | **207** |
| `power_blade` | 0 | 0 | 0 / 0 | `s1 b1 pan/warp/fade` | 0 |
| `shatterhand` | 10 | 10 | 1 / 4 | `s5 b1 fade` | **13** |

**14 of the 23 armed gates fire on play that cleared nothing**, worst first: `paperboy` 207, `ducktales_2` 113, `ghosts_n_goblins` 102, `gradius` 65, `bionic_commando` 52, `1942` 42, `journey_to_silius` 39, `ice_climber` 20, `bad_dudes` 19, `ninja_gaiden_iii` 15, `blaster_master` 14, `shatterhand` 13, `mega_man_usa` 12, `ninja_gaiden` 5.

**4 arm a channel that never moved**: `batman_the_video_game`, `double_dragon_ii`, `mega_man_3`, `power_blade`.

### The verdict: all 23 disarmed

There is no gate to recalibrate to, and the reason is not a numbers problem —
it is the **same refusal as §1**, arrived at from the other end.

The audit probe **clears nothing**. It is undirected play on a roster whose
real level transitions gate behind exactly the search this project runs
Go-Explore for. So every blank run it observes is a non-transition, and its
whole `(d_scene, d_blank)` distribution is null. Two exhaustive cases follow:

- **gate ≤ null** — the gate fires on ordinary play. Demonstrated above, on
  14 of 23, up to 207 times in 24,000 steps.
- **gate > null** — the gate is above every blank run the profile has ever
  been seen to produce, and nothing establishes that a real transition would
  clear it. Setting `ghosts_n_goblins` to `blank_min: 286` would silence it and
  buy nothing: for scale, the only level transition this project has ever
  *witnessed* — Rygar's door — is **78–79** blank frames.

An arm therefore requires a **witnessed positive**: one blank run known to be a
level transition, so a length floor can be placed between it and the death
population. Rygar has one and is served by its own instrument. **No other
cohort profile has one** — the 2026-08-26 survey says so itself, in its own
`scope_limit`: *"'ARM' records that the counter is alive … NOT that a real
transition has been witnessed on tape."*

So all 23 are disarmed with `enabled: false` and a per-profile reason carrying
that profile's own numbers, and the cohort returns to quorum **UNREACHABLE /
ceiling 0.75** — the pre-`4dd15ea` state, verified profile by profile. The rule
lives in code as `C7_SEPARABILITY_WITNESSED`, with a register
(`WITNESSED_TRANSITIONS`) that is how a profile earns its way back.

### Three things the review did not find, which the measurement did

1. **The gates were not merely armed on thin evidence — they fire.** 14 of 23,
   after the death veto, on play that cleared nothing.
2. **The death veto was not guarding the blank channel.** `SceneCutSignal`'s
   window is 240 observations while the veto's transient is a handful, so a
   fade's blank movement sits in the rolling buffer long after `dying` clears
   and lands in a non-vetoed window anyway. On `megaman` the vetoed-out null is
   `(0, 0)` while the veto-independent null is `(2, 18)` — its `residual: 0` was
   a property of the veto, not the gate. That profile is why the calibration
   clause reads the **veto-independent** null, and there is a test demanding at
   least one such case exist on the real roster.
3. **`power_blade` is a fifth dead channel, not a live one.** The survey saw
   1 blank run in 12,000 steps; at 24,000 steps with different seeds it saw
   **0**. Symmetrically, `ninja_gaiden` read 0 in the survey and **3** here.
   Both are marginal, neither is evidence of a working channel.

### Two receipts, deliberately

With nothing armed, every residual assertion iterates an empty set and passes
for free — the exact vacuity this campaign is about. So both measurements ship:

| receipt | describes |
|---|---|
| `scene_cut_arming_asfound_2026-08-27.json` | the 2026-08-26 arming, as found — the evidence the disarm rests on |
| `scene_cut_arming_2026-08-27.json` | the configs as they now ship |

`test_the_disarm_had_evidence_and_the_evidence_is_committed` **fails rather
than skips** if the as-found receipt goes missing.

---

## 3. Contra was excluded on a broken inference, and the exclusion is withdrawn

### 3a. The correction, on the record

`scripts/progress_signal_gate.py::assess()` computed its **resolution** finding
on the window that survived D5 truncation. Contra was condemned with:

> only 20 distinct values in 69 steps (< 32) — too coarse to be a search gradient

Those 69 steps are all that survived truncation of a requested 1,200. **A
69-sample window cannot demonstrate a 32-distinct threshold.** That verdict
measured how fast the scripted forward-hold died, not the signal's resolution.
Reproduced at HEAD under the fixed gate:

```
--probe hold     INCONCLUSIVE — probe died too early to assess
                 69 live steps, 20 distinct, range 0..70, 1131 steps dropped
--probe random   PASS — SIGNAL SOUND — still advancing
                 721 live steps, 346 distinct, range 0..1063
composed         SIGNAL SOUND, zero faults
```

**Contra's signal is sound.** The pair `{lo: 0x0065, hi: 0x0064}` gives 346
distinct values over a 721-step live window — 10.8× the 32-distinct floor, in a
window 3.9× `MIN_ASSESSABLE_STEPS`. No unpaired wrap, no axis-sign fault under
the directed hold, death cleanly detectable via `lives` (`0x0032`). **The
exclusion is withdrawn.**

The random probe is *not* uniformly better and says so in code: an undirected
policy that shows few levels may simply not have gone anywhere
(`bionic_commando`: 122 distinct held forward, 21 under uniform-random over the
same 1,200 steps). So under `--probe random` the axis-sign check, the
resolution *fault*, and the camera-static *fault* are all disarmed — an
undirected probe may only **add** a positive demonstration or a longer window.
`compose()` unions faults and takes the best resolution evidence, which is why
Kung Fu stays UNUSABLE (its unpaired-wrap fault is visible only to the directed
hold) while Contra becomes SOUND.

This was **not** one game. Measured on the banked hold sweep — profiles whose
assessed `live_steps` fall short of the requested 1,200 because the probe ran
off the end of live play — **25 of 45** are truncated, so 25 profiles' findings
are computed on a window shorter than the one that was asked for. (The
2026-08-26 finding put this at 28; the number here is the one I re-measured,
under the stated definition.)

### 3b. Is the `gx 3072` wall real, or an artifact of the contaminated signal?

**Real as a position. Blind as a verdict. And it is a different code path from
the gate bug** — the two must not be conflated.

*Real*, on four independent grounds:

1. **Different code path.** The gate's defect was in `assess()`'s threshold
   logic over a single scripted hold that dies once and stops. `3072` comes
   from `go_explore_solve.py`'s own `max_gx_in_max_area` telemetry — an
   archive-based search with lives- and wrap-aware death handling that resets to
   promising cells across millions of steps. A single early death cannot cap it
   the way it capped the hold.
2. **Convergence across independent algorithms.** The identical value 3072 is
   logged verbatim in **nine of ten** campaigns spanning 2026-07-31 → 2026-08-11
   (`stage1_v2`…`v8`, `poweron_solve`, three `gate_opener_k0*`), using
   materially different strategies (coverage-only, cumulative-kk, local-kk,
   boss-typed HP, resume+time-bins, doctrine, gate-opener), each running
   millions of steps and 280k–357k cells. `max_gx_in_max_area": 3072` appears
   **3,030 times** in `runs/`, more than twice any other value.
3. **Arithmetic.** Verified against the live profile: `progress = hi<<8 | lo`,
   so `3072 = 12 × 256` **exactly** — `hi` (`0x0064`, the screen byte) at 12
   with `lo` (`0x0065`, fine scroll) frozen at 0.
4. **Depth.** Undirected random play tops out around 1,063–1,075. Reaching 3,072
   at all takes the sustained coordinated play only the multi-million-step
   campaigns produced. It is an earned position, not an early-death artifact.

*Blind*, and this is the part that must travel with the number: `hi` at 12 with
`lo` pinned at 0 is a **fixed-camera room**. A scroll-derived progress
definition is, by construction, unable to see anything happening inside one.
"Pinned at 3072" therefore **cannot distinguish** "the search cannot get past
this room" from "the instrument cannot see inside this room". It is a wall in
*this metric*, not a proof about the game.

### 3c. The Finding-3-class audit Contra needs — and how it differs from Rygar's

Contra carries `solve.clear: {mode: confluence}` and `level_key: []`. Verified
in code against the live profile:

|  | Rygar | Contra |
|---|---|---|
| `_clear_mode` | `None` | `"confluence"` |
| `is_clear` with `level_key: []` | `() > ()` → **compile-time constant False** | falls through to a **live** confluence path |
| `clear_reachability` quorum | **UNREACHABLE**, ceiling 0.75 | **FIREABLE**, ceiling 1.0 |
| `coord` signal | **DEAD** (odometer re-anchors; the ≥300 px drop never appears) | **ALIVE**, transition evidence (16-bit `{lo, hi}` pair, so the drop is representable) |
| Is `solutions: 0` evidence? | **No** — a constant | **Yes, weakly** — a live detector that never fired |

So Contra is *not* in Rygar's position. Its detector can fire. But
`configs/contra.yaml` documents the confluence path in-file as **"UNTESTED —
never observed to fire on a genuine clear"**, so `solutions: 0` across nine
campaigns still cannot separate "the boss was never beaten" from "it was beaten
and the detector stayed silent". **Verifying or replacing that confluence
detector is Contra's gating task**, exactly as minting a predicate is Rygar's —
but Contra starts from FIREABLE rather than UNREACHABLE.

---

## 4. The progress gate: 7 of 45 verdicts changed, 0 of 45 `passed` changed

`assess()` now takes `min_window` and a third finding class,
`inconclusive_findings`. The shortfall finding splits:

```
if distinct < MIN_DISTINCT:
    if n >= min_window and directed:  -> instrument fault (SIGNAL UNUSABLE, unchanged)
    else                              -> inconclusive     (INCONCLUSIVE — probe died too early)
```

`verdict_label()` is the single precedence point: demonstrated fault >
unsupportable window > behaviour > sound. `passed = not (instrument +
inconclusive)` — INCONCLUSIVE **blocks but does not condemn**. `exit_code()`
gives 0/1/2 so the VOID-vs-FAIL distinction reaches a caller. `min_window=0`
reproduces the pre-fix assessor bit for bit.

**The floor is calibrated, not chosen.** `MIN_ASSESSABLE_STEPS = 187` is the
largest `steps_to_min_distinct` over every roster profile whose signal *does*
reach 32 distinct levels (Kung Fu: 187). Below 187, a signal we have measured
as resolving had not itself cleared the bar. The 18 measured values:
`[32,32,32,33,34,39,40,42,45,46,46,47,53,53,55,57,102,187]`. Documented as a
**lower** bound — signals slower than their own probe survival are censored out.

**The asymmetry is the point.** The floor gates only the failing direction.
Rygar's 116 distinct in 138 live steps is a *positive demonstration* and still
PASSES.

**Roster re-run** (`scripts/progress_gate_sweep.py`, receipt
`docs/receipts/progress_gate_window_sweep_2026-08-26.json`), 45 profiles with a
`solve:` block:

| | count |
|---|---|
| verdicts changed | **7** |
| `passed` changed | **0** |
| all changes | `SIGNAL UNUSABLE → INCONCLUSIVE` |

The seven, with their live windows, every one under 187:
`batman_the_video_game` 9, `blaster_master` 22, `bubble_bobble` 39,
`ninja_gaiden_iii` 47, `contra` 69, `contra_blank` 69, `megaman` 81.
Two profiles (`ducktales`, `punchout`) are `GATE INAPPLICABLE` — no scalar
progress declared — and are reported as out of domain, not as failures.

**Sensitivity shipped with the number** (floor → verdicts changed): 32→2,
102→7, **187→7**, 289→9, 600→12, 1200→13. `passed` changed is **0 at every
floor**.

**Reproducibility.** The `before` column matches all 43 rows of the banked
`docs/receipts/progress_gate_stasis_sweep_2026-08-26.json` exactly — verdict
*and* live-step count — so the diff has one moving part.

**Composed over both probes**, 45 profiles: **17 SOUND, 25 UNUSABLE, 1
INCONCLUSIVE (`blaster_master`), 2 INAPPLICABLE.**

**Also fixed:** `note_camera_static` asserted "camera never moved over 1200
steps" about a hold that died at step 257 (`double_dragon_ii`) because it was
handed the *requested* step count. It now cites the live window and is subject
to the same floor and directedness gate.

---

## 5. Recommendation

**Point the next campaign at Contra, not Rygar.**

The two games fail on different axes, and only one of them fails on the axis
that is cheap to move:

- **Rygar** is blocked on a **search wall at 4,608 px** that fourteen
  hypotheses failed to move, and its clear predicate cannot even be *designed*
  until that wall falls, because there is no positive to calibrate against. The
  ordering is forced: search first, predicate second.
- **Contra** has a **sound progress signal** (now demonstrated, exclusion
  withdrawn), a **FIREABLE** clear quorum with a live `coord` term, and a
  documented, reproducible frontier at a *named* place — a fixed-camera boss
  room at screen 12. Its blocker is an **instrument question with a definite
  answer**: does the confluence detector fire on a genuine Contra clear? That is
  answerable by a targeted probe, and it does not require the wall to fall
  first.

The honest framing of the wall belongs in the recommendation too: Contra is not
"nearly solved". It is a game where the next question is well-posed. Rygar is a
game where the next question cannot be asked yet.

**Both remain EXHIBITION.** Nothing in either line has been evaluated under the
honest protocol, and no result from either may be described with "the AI
learned/plays/beat".

---

## 6. Anti-vacuity — what each check reports when the mechanism is absent

Four vacuous gates shipped this week, the fourth written by the agent fixing
the previous three, so "the tests pass" is not evidence here.

| Mechanism | Absent-mechanism answer, in code | Killed by |
|---|---|---|
| odometer disabled | `summary()["verdict"] == "UNAVAILABLE"` — never a silent zero | `test_a_disabled_odometer_reports_unavailable_not_zero` |
| counter never moves | `"UNINSTRUMENTED"` — never a silent zero | `test_a_frozen_counter_reports_uninstrumented` |
| splice detection | a decrease is a SPLICE, banks nothing | `TestSplices::test_a_decrease_is_a_splice_not_a_transition` |
| run-length floor | 18 boot fades banked as transitions | `test_it_counts_every_one_of_the_55_transitions` |
| lives debounce | all 55 real transitions destroyed as "deaths" | `test_it_does_not_fire_on_a_real_death` |
| novelty memory | 55 "new areas" instead of 2 | `TestTheRatchet::test_53_of_the_55_transitions_arrive_somewhere_already_seen` |
| lives baseline (rise) | one 255 tick pins `dying` for the episode, `n_triggers` 3 → 0 | `test_one_spurious_high_lives_tick_does_not_disable_the_signal` |
| lives baseline (fall) | a spent life latches the veto ON for the episode | `test_a_genuinely_spent_life_re_baselines_once_the_world_renders` |
| game-over wrap | 0 → 255 releases the veto mid-blackout | `test_a_rise_is_not_evidence_of_life_during_a_game_over_wrap` |
| arming policy | every clause returns `ok`, roster passes vacuously | `test_the_policy_can_still_fail` |
| C7 (can it ever say yes?) | "everything declined" would be indistinguishable from a policy that can never arm | `test_c7_is_what_refuses_the_roster_and_it_can_be_satisfied` |
| a blanket disarm | one boilerplate reason across 23 profiles reads like giving up | `test_the_whole_cohort_is_declined_and_every_refusal_names_its_numbers` |
| an unmeasurable profile | `residual: None` would read as clean | `test_adjudicate_declines_on_a_nonzero_residual` (`C4_UNMEASURABLE`) |
| C6 vs the older null rule | if the two rules agreed everywhere, C6 would be decoration | `test_c6_catches_at_least_one_gate_the_older_null_rule_would_pass` (`megaman`) |
| the disarm itself | with nothing armed, every residual assertion iterates an empty set | `test_the_disarm_had_evidence_and_the_evidence_is_committed` |

Each guard in `scripts/transition_witness.py` and `SceneCutSignal` was deleted
in turn and the suite re-run: every mutant was killed, each by a *different*
named test, and the source was restored `diff`-verified byte-identical.

The arming decision itself — the gap `4dd15ea` shipped through — now has
roster-level tests that read the **real** `configs/*.yaml`, not synthetic
profiles, and refuse on measured evidence rather than on judgement.

---

## 7. Receipts

| What | Where |
|---|---|
| Rygar clear predicate, refused with numbers | `docs/receipts/rygar/clear_predicate_REFUTED.md` |
| Rygar transition streams | `docs/receipts/rygar/transition_streams.json` |
| Rygar R1 tape (6,018 actions, replays 3/3 exact) | `docs/receipts/rygar/r1_tape_gx6242.json` |
| Transition instrument | `scripts/transition_witness.py`, `tests/test_transition_witness.py` |
| Cohort arming policy + audit | `scripts/scene_cut_arming.py`, `tests/test_scene_cut_arming.py` |
| Cohort arming receipt | `docs/receipts/clear_control/scene_cut_arming_2026-08-27.json` |
| Cohort arming, AS FOUND (evidence for the disarm) | `docs/receipts/clear_control/scene_cut_arming_asfound_2026-08-27.json` |
| 2026-08-26 cohort survey (superseded on the arming decision) | `docs/receipts/clear_control/odometer_cohort_scene_cut_survey_2026-08-26.json` |
| Progress-gate window sweep + composed roster | `docs/receipts/progress_gate_window_sweep_2026-08-26.json` |
| Progress-gate random probe | `docs/receipts/progress_gate_random_probe_2026-08-26.json` |
| Contra re-test | `runs/contra_retest/*.json` (gitignored) |
