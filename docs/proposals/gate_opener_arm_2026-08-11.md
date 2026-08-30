# The GATE-OPENER arm — design for the ~~GATED~~ UNRESOLVED-CONCENTRATED wall class

> **VOCABULARY ADDENDUM (2026-08-11) — the class this document is named
> for does not exist.** The K-FALSIFIER pre-registered in
> `docs/proposals/GATE_OPENER_CAMPAIGN_2026-08-11.md` §7 RAN and **FAILED**
> (`docs/receipts/dispatch/k_falsifier_2026-08-10.md` §0), and the
> follow-up search for a size-decoupled replacement returned nothing
> (`docs/receipts/dispatch/size_decoupled_statistic_2026-08-11.md` §0:
> 22 candidates, both directions, 0 separate). `CONCENTRATION_GATED_MIN`
> cannot tell a gated search from a solved one: four registered SOLVED
> archives score 3.9×–10.8× the constant, and *solved Castlevania blocks
> on this same profile and grid* read more stuck than the hall does.
> **Every `GATED` / `saturated` / visits-as-saturation usage below is
> STRUCK and the class label reads `UNRESOLVED-CONCENTRATED`** —
> descriptive (five runs, ~10.7 h, ~77M steps, pinned best score, zero
> crossings, zero solutions), not a verdict about cause. Load-bearing
> usages carry an inline `[struck 2026-08-11: see §12 / falsifier
> receipt]` strike-line — **"§12" is the campaign doc's §12** (D3
> lineage-check verdict), this document having no §12 of its own.
> Nothing is rewritten: the original wording stays legible so the
> reasoning that produced it can be audited.
>
> **What survives, and it is most of the document.** §2.2's measurement —
> the archive holds one live game-state axis at the boundary and could
> not have RECORDED an interaction — is untouched, and `k_falsifier` §9
> independently confirms it: `live_state_axis_count` is the one statistic
> that separates the hall from the comparators, it separates in the
> OPPOSITE direction from a wall statistic, and it therefore supports the
> interaction-blind thesis and the `KEY_BLIND` family rather than
> saturation. §2.5's sampler-reachability arithmetic, §3's arm, §7.6's
> `entity_slots` refutation and every probe receipt stand unaffected. What
> falls is only the *label* the remedy was dispatched under.
>
> **Two later verdicts also bind this document** (recorded here so it is
> not read as live): the campaign doc's §12 struck the **gx-767 pin as a
> borrow-glitch PHANTOM** — the true frontier is gx 751, bucket 93, and
> band-95 is struck from every root pool, which supersedes §2.3 and
> §5's `pin_gxb = 95` — and its §13 records **K0 FAIL, campaign
> DISARMED**.

**Status: DESIGN ONLY. Nothing here is implemented.**
`scripts/go_explore_solve.py` is owned by another lane; this document is
the specification a post-fix implementer works from. The only code that
ships alongside it is one additive, gate-nothing diagnostic in
`src/training/wall_taxonomy.py` (`boundary_axis_profile`), which is the
measurement the arm needs to self-arm and which produced every number in
§2.

**Companion documents.** The discriminator this arm is dispatched by:
`src/training/wall_taxonomy.py` + `docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md`.
The A/B that produced the ~~GATED~~ reading: `CLAIMS.md` FORGE entry for
`--ortho`, commit `9cd6de5` [struck 2026-08-11: see §12 / falsifier
receipt. The A/B itself stands; only the class it was read into is gone.
The matching relabel in `CLAIMS.md` and the HUD is OWED and not done here
— k_falsifier §12.4].

---

## 1. What class this is, and why the previous arm was the wrong shape

The calibrated discriminator separates three ways a search can stop:

| class | meaning | remedy |
|---|---|---|
| COVERAGE_LIMITED | still expanding; the wall is wall-clock | wait |
| BARREN / KEY_BLIND | the archive is frozen or spatially degenerate | fix the key / the reset |
| ~~**GATED**~~ | ~~local coverage saturated, boundary frozen~~ | **orthogonal mechanism** |

[struck 2026-08-11: see §12 / falsifier receipt — the third row's *class*
is struck. There is no statistic that separates "local coverage
saturated" from "solved"; `concentration` is `cells` wearing a hat
(Spearman +0.929 against `cells`, −0.041 against `distinct_spatial`), and
21 further candidates were built and all failed. The first two rows are
untouched: `COVERAGE_LIMITED` and `KEY_BLIND` rest on other evidence and
the falsifier explicitly leaves them standing.]

The Castlevania hall now reads ~~GATED~~ UNRESOLVED-CONCENTRATED, and so
does its arm-free control [struck 2026-08-11: see §12 / falsifier
receipt].
The `--ortho` arm was the first thing dispatched at it, and its premise —
"the frontier is pinned because the search never climbs" — was measured
STALE: the control reached the same vertical extent without the arm.

The `--ortho` arm searched for a different **position**. This document's
claim is that a gated wall, by construction, is not opened by finding a
position. It is opened by causing a **state change** — a kill, a pickup,
a switch, an elapsed timer, a door — after which positions that were
previously unreachable become reachable. The arm therefore searches over
*interactions*, and its progress signal is not "did we reach a new cell"
but "did the boundary move once we did X".

BB 99-1 is the same family, and the manual version of this arm has
already been run there by hand (§3.2).

---

## 2. What the run-A archive actually contains at the pin

Everything below was measured today, read-only, from the banked
archives. `runs/` is gitignored, so the figures are transcribed here.

### 2.1 The verdicts

`gated_wall_verdict` over both 90-minute A/B runs (full 90 progress
records + the flushed archive):

| | `cv_hall_ortho_a` (arm) | `cv_hall_ortho_ctrl` (control) |
|---|---|---|
| verdict | ~~**GATED**~~ (degraded) ᵈ | ~~**GATED**~~ (degraded) ᵈ |
| cells | 131,561 | 149,153 |
| `distinct_spatial` | 1,096 | 1,094 |
| `spatial_span` | 95 | 95 |
| **concentration** | **120.04** (4.8× the threshold) | **136.34** (5.5×) |
| `boundary_cells` | 18 | 14 |
| `boundary_visit_entropy` | 0.9848 | 0.9849 |
| `window_steps` | 946,690 | 1,009,720 |
| records since the map moved | 79 | 83 |
| solutions | 0 | 0 |

ᵈ [struck 2026-08-11: see §12 / falsifier receipt — the verdict row is
struck as a CLASS reading. Every other number in this table is a
measurement and stands.]

~~Both far above `CONCENTRATION_GATED_MIN = 25.0`. The label is not
marginal; the *positive class* is still unvalidated (§7.4).~~

[struck 2026-08-11: see §12 / falsifier receipt. "Far above the
threshold" turned out to carry no information: `smb_4_4_micro/lvl_3-3`
scores **270.03** and `lvl_1-3` **240.19** — both SOLVED, both above the
control's 136.34. Effort-normalized the ordering inverts outright (hall
8.0–15.1 concentration per M archive records, solved archives 70.4–342.4:
the hall is the *least* concentrated search in the set). And at full grid
parity the hall's re-keyed `distinct_spatial` is 213, the **smallest** in
the comparison set. The class was not merely "unvalidated" — the
statement "the label is not marginal" was the reverse of true, because
the constant's separating band (20.58, 31.04] was fitted inside a
~20k-cell cap created by a pickle read error that does not reproduce.]

### 2.2 The cell key at the pin — the finding

The fleet's key is
`(sect, tb, kk, psig, loops, route_sig) + game.cell_fn(ram)`, and for
Castlevania `cell_fn` returns `(area, boss_hp, state_sig, y//Y_BAND,
progress//GX_BUCKET)`. Arity 11, uniform across all 131,561 cells.

`boundary_axis_profile(cells, band=24, bookkeeping=(4, 5))` over the
pinned band — the same 24-gx-bucket window `_sel_band24` samples from:

| position | axis | distinct values in the band (arm / control) |
|---|---|---|
| 0 | `sect` (room-transit count) | **1 / 1** |
| 1 | `tb` (time bin) | **1 / 1** |
| 2 | `kk` (kill count) | **1 / 1** |
| 3 | `psig` (room signature) | **1 / 1** |
| 4 | `loops` (maze loop counter) | 3 / 5 |
| 5 | `route_sig` (own gx→y-band history) | 31 / 31 |
| 6 | `area` | **1 / 1** |
| 7 | `boss_hp` | **1 / 1** |
| 8 | `state_sig` (the on-stairs bit) | 2 / 2 |
| 9 | `y//Y_BAND` | 17 / 17 |
| 10 | `progress//GX_BUCKET` | 24 / 24 |

```
band cells        41,639 (arm)    49,564 (control)
distinct positions   308              308
alias ratio       135.19           160.92
constant axes     {0, 1, 2, 3, 6, 7}   (six of eleven)
live GAME-state axes  {8}          — exactly one, and it is one bit
live bookkeeping axes {4, 5}       — both derived from our own trajectory
interaction_blind  True             True
```

**Read that literally.** At the boundary the archive distinguishes 308
positions, and it multiplies each of them ~135× using: the maze loop
counter (3 values), the route signature (31 values) and one player-mode
bit (2 values). `3 × 31 × 2 = 186` possible aliases per position; 135 of
them are realised, i.e. **73% of the nuisance-alias space is already
full** (control: 161/310 = 52%). That is exactly the mechanism
`CONCENTRATION_GATED_MIN`'s docstring names — "the key keeps
manufacturing nuisance novelty" — measured on the actual key. [2026-08-11:
the MECHANISM survives the falsifier and is independently reconfirmed —
size_decoupled §8 finds the hall does NOT flatten (its tail growth
exponent is the corpus's *highest*) precisely because the key
manufactures novelty forever at a fixed location. What is struck is the
CONSTANT the docstring belongs to, not this measurement. See §12 /
falsifier receipt.]

And of the four things a *game* could be withholding, the archive can
represent **none**:

* **kills** — `kk ≡ 0` in every one of the 131,561 cells. `--kill-key`
  was off. It is off by default, and `scripts/go_explore_chain.py` does
  not forward it at all, so **the CV hall has never once run with the
  kill axis on**, despite `entity_slots: {lo: 0x0450, hi: 0x0458}` being
  in the profile since 2026-07-29. *Update 2026-08-10:* that range has
  now been probed for the first time (§7.6). The eight bytes are a real,
  measured, agent-coupled occupancy array — but their `1→0` edge is
  driven by **scrolling, not by attacking**, so turning the axis on
  would have keyed cells on a positional alias. The archive's blindness
  to interactions is confirmed; the cheap fix for it is not.
* **boss / entity HP** — `boss_hp ≡ 0`; `configs/castlevania.yaml` has no
  `boss:` block, so the slot is a constant.
* **possession / items** — no axis exists. There is no BB-style
  `$0030`-class bit in the CV profile.
* **timers / elapsed state** — `tb ≡ 0`; `--time-bins` was off.

So the honest statement of what the 131k-cell archive proves is:

> The search has tried every POSITION it can reach in the hall, ~135
> times over per position, and has no memory in which an interaction
> could have been recorded. "It has tried everything" is not supported
> by this archive; "it has tried every position" is.

This is why the arm's first move is discovery, not more search.

### 2.3 The pin itself

* Frontier gx bucket **95** in both runs. With `GX_BUCKET = 8` (inferred:
  `767 // 8 = 95`, and `Y_BAND = 8` from the y-band ceiling of 25 against
  a 1-byte y) that is raw `progress` 760–767, the banked `best_score` 767.
* Bucket 95 holds **18 cells** (arm) / **14** (control). All at
  `state_sig = 0` (never on stairs), y-bands {19:1, 20:1, 21:2, 22:3,
  24:11} — the floor. `explored_fraction = 1.0`, `visits = 21`,
  `times_chosen = 101`: every frontier cell has been selected and
  returned nothing.
* **Bucket 94 (raw gx 752–759) is EMPTY in both archives**, while bucket
  93 holds 2,443 / 2,805. The density does not taper into the pin; there
  is a hole and then an 18-cell beachhead. Either `progress` jumps there
  (a camera/scroll snap) or the last stretch is entered exactly one way.
  Either reading makes gx the wrong coordinate at the pin, and makes
  buckets 93–95 the region the arm should probe first.

### 2.4 What the `--ortho` arm's own gate figures did at the pin

From `runs/cv_hall_ortho_a/progress.jsonl`:

| | line 1 (t = 60 s) | line 90 (t = 5,401 s) |
|---|---|---|
| `ortho_best_yband` | 7 | 7 |
| `ortho_deep_yband` | 9 | 9 |
| `ortho_selections` | 0 | 37,345 |
| `ortho_cols_improved` | 0 | 3 (gate wanted ≥ 8) |
| `max_gx_in_max_area` | 760 | 767 |

`ortho_deep_yband` — the pre-registered figure, min y-band inside
`gxb ≥ frontier − 24` — reproduces exactly from my own read of the
archive (min y-band in the band = 9). **It was 9 at minute one and 9 at
minute ninety.** And the vertical high ground the arm was chasing does
exist in the archive — 3,255 cells at y-band ≤ 8 — but **every one of
them is at gx bucket 1–18**, the entrance region. At `gxb ≥ 92` there
are **zero** cells above y-band 9.

So the honest post-mortem is narrower than "the premise was stale": the
arm concentrated selection pressure upward, and the ceiling *at the pin*
never moved a single band in 90 minutes and 37,345 selections. That is a
gate signature, not a starvation signature.

### 2.5 The interaction the sampler structurally cannot emit

CV's action space is 16 entries; `action_weights` sums to 36, so
`P(NOOP) = 1/36`. With `--sticky 0.5` the per-step continuation
probability of any held action is `0.5 + 0.5/36 = 0.5139`. Expected
number of *naturally sampled* sustained NOOP runs across run A's
8,972,160 worker-steps:

| hold H (steps) | ≈ frames @ fs 4 | P(run of H) | expected in the run |
|---|---|---|---|
| 8 | 32 | 9.5e-03 | 2.4e+03 |
| 24 | 96 | 2.2e-07 | **5.6e-02** |
| 64 | 256 | 6.1e-19 | 1.5e-13 |
| 180 | 720 | 1.8e-52 | 4.4e-47 |
| 512 | 2048 | 1.8e-148 | 4.5e-143 |

The profile's `hold_macros` do fire — `p = 0.02 + 0.01` per free slot,
≈ 11,215 injections over the run — but they are **`up` and `down` only,
24 steps only**. There is no NOOP macro, no macro for any other action,
and nothing longer than 24 steps. A wall whose key is "stand still for
two seconds", "hold `up` for 500 frames", or "wait for a cycle" is not
merely unlikely under the shipped sampler; it is unreachable. This is
the cheapest sub-arm to build and it is the one with the largest gap
between "we searched" and "we searched that".

---

## 3. The arm

Three parts, in dependency order: **(a)** make state changes happen and
be representable, **(b)** test whether any of them opens the boundary,
**(c)** emit the telemetry that lets the discriminator dispatch it.

### 3.1 (a) Interaction enumeration in the pinned region

All four sub-arms are rooted at pinned-band archive cells — the solver
already holds a save-state for each, so every probe starts *at the wall*
rather than replaying the level.

#### (a0) Turn on the axes that already exist — ~~free, do this first~~ **withdrawn 2026-08-10**

*Superseded before launch by the probe that was supposed to justify it.*

The original text proposed one bounded `--kill-key` run "whose addresses
are already receipted". They were not: `entity_slots` was the single
byte-range in the CV solve block with no artifact behind it, and §5's
anti-gaming clause cannot require a probe receipt for new axes while the
arm's own foundation rests on a prose comment. So the range was probed
first (§7.6, `scripts/probe_entity_slots.py`, two sites, whole-page
sweep, every action in the profile's action space).

**Result: (a0)'s question is already answered, and the answer is no.**
The eight bytes are a genuine occupancy array — `lo`/`hi` re-derived
from measurement, not assumed — and they do respond to our inputs. But
every arm that elevates the `1→0` rate above the NOOP baseline contains
`right`, at both sites; the two attack actions sit exactly at baseline;
and with locomotion held constant, adding continuous attack moves the
pooled rate by +2.0/1k and +0.0/1k, indistinguishable from a non-attack
negative control. `kk` would leave 0 in the pinned band all right — and
where it did move, it would be counting distance, not kills.

Running (a0) anyway would burn 20 minutes to re-learn this and would
seed the archive with a positional alias in the very slot (a1) wants.
**Phase 0 is therefore deleted from the validation order** (§5) and its
budget returned. What (a0) was really for — "cheap disconfirmation
before a campaign" — is preserved: the probe did it in 4 minutes instead
of 20, without a solver run.

`--time-bins` is untouched by this and remains available, but it keys on
elapsed time rather than on anything the game withholds, so it is not
promoted.

#### (a1) Kill-pattern search — **demoted to conditional**

**Precondition, binding:** (a1) may not launch until a contact-
conditioned differential (P1–P4, §3.1(a2)) identifies at least one
address whose `1→0` edge survives the locomotion control — i.e. a
candidate that moves when the agent *acts on something* and not merely
when the camera advances. Absent that, `kill_mask` is a positional
alias and (a1) is a re-run of the ortho arm's mistake with a new key
slot.

The measured farmability figure is already fatal to the *count* form:
317 `1→0` edges over 8 slots at each of two sites means slots respawn
~40× over, so the cumulative count `--kill-key` maintains is farmable by
walking. The mask form below survives that specific objection — it is
set-valued and high-water-reset — but only if the underlying edge means
something, which is what the precondition tests.

Spec, unchanged in shape, with the measured corrections folded in:

`--kill-key` keys on a capped cumulative *count*. A prerequisite gate is
usually a *set* ("all four candles", "the guard, not the bat"), which a
count aliases away. Spec:

* `kill_mask` = bitmask over `entity_slots` of slots observed
  non-zero → zero **since the pin was crossed into**, reset when the
  lineage sets a new progress high-water mark (the existing
  `kill_key_local` semantics, which exist precisely because Contra's
  cumulative count saturated its cap).
* Bounded cardinality: ~~CV has 8 slots → 256 masks~~ — **measured, the
  reachable space is smaller.** At the hall root `0x0451` and `0x0452`
  are byte-identical columns over 32,000 steps (7 distinct signatures
  across 8 slots) and `0x0453` produced zero spawns, so the effective
  mask alphabet is well under 256. `--gate-cap` (default 64) on distinct
  observed values is comfortable rather than tight.
* **Farmability test, mandatory** (the BB lesson: bubble pops are a
  farmable 10-point event, so a `kill_key` on them would be a fake
  gradient). Over a pre-registered random-rollout budget from pinned
  cells, if the number of 1→0 transitions exceeds the slot count — i.e.
  slots respawn — the *count* is farmable. Key on the mask of
  first-clear only, or refuse the axis and say so. **Already run,
  already failed:** 317 transitions over 8 slots at each of two sites
  (`docs/receipts/probes/cv_entity_slots_attribution_2026-08-10.json`).
  The count form is refused. The mask form stands or falls on the
  precondition above.

#### (a2) Pickup / possession search — the generic contact probe

This is BB's `$0030` protocol, generalised and made cheap. Four
protocols, run by a **new** script `scripts/probe_interactions.py` (not
in the solver), emitting a receipt to `docs/receipts/probes/` and a
`state_sig:` YAML fragment for the profile.

* **P1 — onset localisation.** From K pinned-band save-states, run N
  arms × L steps over a fixed policy family set (each single action
  held; random; random + macros; pure NOOP). Log the full 2 KB RAM and
  the profile's already-verified `progress`/`y`. For each byte: modal
  value over the log; an *onset* is a departure from it persisting ≥ 2
  observations. Keep bytes that are (i) low base-rate — this is what
  discards frame counters, animation phases and timers — and (ii)
  spatially concentrated, i.e. the (progress, y) at onset falls inside a
  box that is a small fraction of the visited box. BB's signature was
  **104/104 onsets in one box**; that is the shape being matched.
* **P2 — persistence class.** Bank a save-state at each onset edge, step
  512 NOOP observations, and classify: **TIMED** (reverts — BB's
  possession reverted in 22/24 within 420 frames), **LATCHED** (never
  reverts — the strongest gate candidate: a key, a switch, an opened
  door), **COUNTER** (monotone drift — usually nuisance).
* **P3 — controllability.** From the pre-onset save-state, confirm the
  onset is reproducible by a policy rather than an artefact of where the
  probe happened to be. Non-reproducible ⇒ discard.
* **P4 — nuisance / farmability.** Distinct values the byte takes over a
  long random rollout. High or unbounded cardinality ⇒ refuse. This is
  the `route_sig` lesson stated as an admission rule: 31 values per
  position is what made the concentration 120.

Only LATCHED and TIMED candidates that survive P3 and P4 become
`state_sig` bits.

#### (a3) Sustained-hold and wait probes

Given §2.5, replace the *probabilistic* macro roll with a
**deterministic enumeration** for bursts rooted in the gate pool: each
such burst is assigned the next `(action, hold)` pair from the grid
`action_space × --gate-holds` by a round-robin counter, so grid coverage
is guaranteed rather than sampled. Default holds `8,24,64,180,512`
(32–2048 frames at `frame_skip 4`). With 16 CV actions that is an 80-cell
grid; at `--burst 64` one full sweep from one cell costs ~5,120
worker-steps, i.e. ~0.06% of one 90-minute run. The NOOP row of that
grid is the wait probe.

#### (a4) Registered but out of scope: the second controller

BB's untested lead — the pool drives P1 only, and controller-2 capability
landed in `873776c`. Named here so it is not re-discovered as novel; it
is **not** part of the CV validation.

### 3.2 (b) The gate TEST — does the boundary open?

This is the part that makes the arm falsifiable, and it is cheap because
the solver already records every cell.

**Definitions, frozen when the arm arms:**

* `pin_gxb` = frontier gx bucket at arming time (95 for the hall).
* A burst `b` is **active for candidate c** if `c` held at `b`'s root or
  became true at any observation inside `b`.
* `cross(b)` = `b` recorded at least one archive cell at
  `key[-1] > pin_gxb`.
* `lift(b)` = new cells recorded at `key[-1] >= pin_gxb - 1`.

**Primary gate — first crossing, with counterfactual attribution.**
The pin has stood through ~110M steps and five arms with zero crossings,
so a crossing is a ~zero-rate event and one of them is decisive *if* it
is attributable. Sequence:

1. `cross(b)` fires under candidate `c`.
2. The crossing trajectory is **replay-verified** from its root
   (`--verify-bank`, already default-on).
3. **Matched control:** M paired replays from the same root with `c`
   suppressed (the kill segment skipped / the pickup box avoided / the
   hold shortened). If the controls also cross, the crossing was
   coverage — record it as such and do **not** attribute it to `c`.
   This reuses the reasoning already shipped as
   `--counterfactual-gate`, which exists because a windowed detector can
   fire on evidence corresponding to nothing the game did.

**Secondary, graded — for ranking when nothing crosses.**
Per candidate, `E[lift | active]` vs `E[lift | ¬active]`, on matched
burst counts. This is a pure archive delta: two counters per candidate,
incremented at the existing `record()` site. It is what lets the arm say
"`c` is warmer" without claiming a crack, and it is what BB's
possession campaign needed and did not have — the honest negative there
("possession is real; it does not damage the boss") cost ~3.1M frames
and 24 policy families to establish. With this statistic the same
verdict is a bounded number of bursts.

**What the test must be able to say out loud:**

| outcome | meaning |
|---|---|
| crossing, attributed | the gate is `c`. Bank it, add it to the profile, re-chain. |
| crossing, control also crossed | coverage finished the wall. **The taxonomy's own falsifier fired** (§7.4). |
| no crossing, some `c` with lift > control | ranked leads; extend or escalate |
| no crossing, all lifts flat | the arm is a no-op on this wall; report the null |
| no candidates survived P1–P4 | the discovery half is null here; report it (BB-style) |

### 3.3 (c) Telemetry for self-arming

Tied to the calibrated discriminator. The receipt's §8 already ranks
`c_local` first and `boundary_action_entropy` second; the gate-opener
produces both as by-products — it enumerates actions systematically, so
the action histogram at the boundary is free.

Fields the solver must add to `progress_line()` (armed only):

| field | why |
|---|---|
| `c_local` | receipt §8 #1. `len({(area, y_band, gx_bucket)})`. Converts concentration from one number into a curve. |
| `boundary_action_entropy` | receipt §8 #2. The adopted form's second term, currently unmeasured everywhere. |
| `boundary_state_axes` | live non-spatial GAME-state axes in the band — `boundary_axis_profile(...).live_state_axis_count`. **The new one.** |
| `boundary_alias_ratio` | band cells / distinct positions (135.19 for run A) — concentration measured locally at the pin. |
| `gate_pin_gx`, `gate_pinned_secs` | what the gate test froze, and when |
| `gate_pool`, `gate_selections` | is the arm engaging at all (the `--ortho` receipt's most useful line) |
| `gate_grid_covered` | fraction of the `action × hold` grid enumerated |
| `gate_candidates`, `gate_tested`, `gate_refuted` | discovery/test bookkeeping |
| `gate_crossings` | the primary gate, as a counter |

**The dispatch rule this enables** (specified, deliberately NOT
auto-armed — self-arming is a separate decision per the D2 verdict, and
`wall_taxonomy` is runtime-inert by design):

```
verdict == GATED  AND  boundary_state_axes < BOUNDARY_STATE_AXES_MIN
    -> GATED-BLIND: run the DISCOVERY half first (a0, a2). The remedy
       "switch to an orthogonal arm" is premature — there is no axis
       for the orthogonal mechanism's result to land on.

verdict == GATED  AND  boundary_state_axes >= BOUNDARY_STATE_AXES_MIN
    -> GATED-TRUE: run the SEARCH half (a1, a3) + the gate test.
```

[struck 2026-08-11: see §12 / falsifier receipt — **the `verdict ==
GATED` conjunct is struck from both branches.** No verdict is GATED; the
size-decoupled receipt's §12 recommends deleting the branch outright and
returning `INDETERMINATE`. The `boundary_state_axes` conjunct — the half
this document contributed — is unaffected and is the half that carries
the dispatch, so with the dead conjunct removed the rule reduces to
`boundary_state_axes < BOUNDARY_STATE_AXES_MIN -> run the DISCOVERY half
first`, which is what the campaign doc's §4b actually dispatches on.
Also struck by the same receipt: the labels `GATED-BLIND` / `GATED-TRUE`
themselves, which name a class that has no member.]

Run A reads ~~**GATED-BLIND**~~ interaction-blind (`boundary_state_axes =
1`) [struck 2026-08-11: see §12 / falsifier receipt]. That is the single
most actionable sentence this analysis produces, and no statistic in the
shipped module could have said it — a claim the falsifier independently
*strengthened*: sweeping ~20 statistics against the corpus,
`live_state_axis_count` was one of only three that ordered the hall apart
from the comparators, and the only one whose separation was not a
geometry artifact (k_falsifier §9).

---

## 4. Implementation spec

Against `scripts/go_explore_solve.py` as of `48c645b`. Every item is
**default-off and byte-identical when off**, matching the discipline the
`--ortho` arm was reviewed under.

### 4.1 CLI

| flag | default | meaning |
|---|---|---|
| `--gate-opener {off,probe,search,both}` | `off` | master switch |
| `--gate-pin-secs SECS` | `300` | arming delay; **reuse `self._pin_time`**, the same self-measured clock `--inversion-pin-secs` and `--ortho-pin-secs` use. Negative disables. |
| `--gate-band N` | `24` | band width in gx buckets; same window as `_sel_band24` |
| `--gate-bias P` | `0.25` | share of selections routed to the arm |
| `--gate-holds L,...` | `8,24,64,180,512` | the hold grid |
| `--gate-axes A,...` | `kill_mask,hold` | which sub-arms; `possession` only once a probe receipt exists |
| `--gate-cap N` | `64` | max distinct values a new axis may take before it is declared nuisance and evicted |
| `--gate-probe-bursts N` | `0` | discovery budget for `probe` mode |

### 4.2 The key axis — reuse slot 2, do **not** grow the key

`_spatial_key`, `_refresh_sel_cache`, `ortho_pool`, `column_extremes` and
`wall_taxonomy` all index `key[0]`, `key[-5]`, `key[-2]`, `key[-1]`.
Growing the prefix keeps those correct, **but it changes arity**, and a
resumed 11-tuple archive would then alias with nothing: all 149k cells
get re-discovered as new. Do not do that.

Instead: **`gk` occupies `key[2]`, the existing `kk` slot**, which is
constant `0` in every banked CV archive.

```
gk = popcount(kill_mask) capped 15          # bits 0-3
   | (possession_bits & 0x3) << 4           # bits 4-5, from state_sig-class candidates
```

Cardinality ≤ 64, arity unchanged, and a resumed cell (`gk = 0`) aliases
exactly with a new "no state change yet" cell — resume is preserved.

**`--kill-key` and `--gate-opener` both write slot 2.** Make that an
`argparse` error, not a silent precedence rule.

### 4.3 Selection arm

Insert in `select()` **between the ortho arm and the count arm**, so
`--deep-bias` semantics are unchanged — the same insertion discipline
the ortho arm used and the same review finding applies (an empty pool
must fall through, never return `None`).

`_gate_pool` = pinned-band cells whose `gk` is the least-represented
value at their own `(y_band, gx_bucket)`. This is a count prior over the
**state** axis rather than over position — the arm's whole point. Same
`1/sqrt(times_chosen + 1)` weight, same O(1) rejection sampling bounded
at 64 draws, same `barren` skip, same `_refresh_sel_cache` rebuild
alongside `_ortho_pool`.

Heed the shipped fix in `_refresh_sel_cache`: scope the deep-area filter
to the arm's own subset. Overwriting `self._sel_cells` emptied the pool
and reset every worker to the entrance root.

### 4.4 Action-shaping arm

In `explore()`, in the shared macro slot, ordered **after** the
transition-macro check and **before** the ordinary macro roll. When the
burst is rooted in `_gate_pool` and the arm is armed, do not roll:
assign `self._gate_grid[self._gate_cursor % len(grid)]` and advance the
cursor. Deterministic enumeration is the departure from stochastic
sampling that §2.5 justifies. Record macro steps verbatim in the trace,
as the existing macros already do.

### 4.5 Gate-test bookkeeping

In `observe()`, at the existing `dom` / `archive.record()` site:

```
if key[-1] > self._gate_pin_gxb and recorded_new:
    self._gate_cross[ctx["gate_c"]] += 1
elif key[-1] >= self._gate_pin_gxb - 1 and recorded_new:
    self._gate_lift[ctx["gate_c"]] += 1
```

with the per-candidate burst denominator debited in `_assign` next to
the existing `barren` credit/debit. All O(1); no new scans.

### 4.6 Chain plumbing

`scripts/go_explore_chain.py` forwards a fixed subset of solver flags and
**does not forward `--kill-key`, `--ortho*`, `--time-bins`,
`--frontier-throttle`, `--door-weight` or `--resume-archive`**. That is
why the kill axis has never run on this wall. Add `--gate-opener` and its
knobs to the forwarded list, or the arm is unreachable from the chain.

**Also:** `--resume-archive` rebuilds `max_gx` as `key[-1] * GX_BUCKET`.
Resuming an archive under a different `--gx-bucket` silently corrupts the
frontier tracker. The validation run must use `--gx-bucket 8 --y-band 8`,
matching the archive it resumes.

### 4.7 Tests the implementer owes

Mirroring the fifteen the ortho arm shipped, in
`tests/test_go_explore_solve.py`:

1. default-off byte-identity at the shipped CLI defaults (runtime
   inertness mirror);
2. key arity unchanged when armed, and `gk == 0` for a state-change-free
   burst so resumed cells still alias;
3. `--kill-key` + `--gate-opener` raises;
4. the hold grid is covered exactly once per sweep by the round-robin;
5. an empty `_gate_pool` falls through rather than returning `None`;
6. gate credit fires only strictly above `pin_gxb`, and `lift` only in
   `[pin_gxb - 1, pin_gxb]`;
7. an axis exceeding `--gate-cap` is evicted and reported;
8. `--gate-pin-secs` negative ⇒ arm never arms.

And in `tests/test_wall_taxonomy.py`, for the helper shipped with this
document (it currently has **none** — that file was outside this lane's
edit scope):

9. `boundary_axis_profile` on a synthetic position-only archive reports
   `live_state_axes == ()` and `interaction_blind is True`;
10. one live game-state axis + two bookkeeping axes ⇒
    `live_state_axes == (8,)`, `live_bookkeeping_axes == (4, 5)` when
    `bookkeeping=(4, 5)` is passed, and `live_state_axes == (4, 5, 8)`
    when it is not;
11. mixed-arity archive ⇒ profiled at the modal arity, ties break long,
    `band_cells > profiled_cells`;
12. empty archive is safe;
13. `alias_ratio` equals `profiled_cells / distinct_positions`;
14. no verdict in `CORPUS` or `LIVE` changes — the helper gates nothing.

---

## 5. Pre-registered validation

**One bounded run + one matched control, declared before launching.**

**Wall:** Castlevania block 3, the hall. Root
`runs/cv_chain_hw2/entrances/entrance_after_2.state`, `wd = (3,)`,
`lives = 4`, hw flags
`reset_alignment,mmio_read_timing,dmc_stall_timing,nmi_poll_timing`,
`frame_skip 4`, `--gx-bucket 8 --y-band 8`.

**Resume:** `runs/cv_hall_ortho_ctrl` — 149,153 cells, the deepest
**arm-free** coverage state banked. Deliberate: resuming the state that
coverage alone produced means any crossing cannot be explained by "we
gave coverage more time". Both the arm run and its control resume the
same directory.

**Budget:** 90 minutes each, matching the banked A/B. Same `--seed 303`,
same `--workers` as the control (the A/B's worker count is not recorded
in any log — read it from the orchestrator's launch record and put it in
the receipt).

**Order:**

| phase | mode | budget | purpose | status |
|---|---|---|---|---|
| ~~0~~ | ~~`--kill-key` only~~ | ~~20 min~~ | ~~is `kk` ever non-zero at the pin?~~ | **deleted 2026-08-10** — answered off-solver by `scripts/probe_entity_slots.py` in 4 min; the axis is locomotion-coupled (§7.6), so seeding it would inject a positional alias |
| 0′ | `scripts/probe_entity_slots.py`, both sites | 4 min | **done.** Receipts the `entity_slots` range and refutes its kill reading | complete |
| 1 | `--gate-opener probe` | 30 min | P1–P4 discovery; emits candidates + receipt. **Now the first phase that needs the solver, and the gate on (a1).** | needs new code |
| 2 | `--gate-opener search` | 90 min | the arm + gate test | needs new code |
| 2c | control: identical, `--gate-opener off` | 90 min | matched | runnable |

Phase 0′ is reproducible today and costs nothing:

```
.venv/bin/python scripts/probe_entity_slots.py \
  --out docs/receipts/probes/cv_entity_slots_2026-08-10.json
.venv/bin/python scripts/probe_entity_slots.py \
  --root-state "roms/Castlevania (USA)_start.state.bin" \
  --site block1_profile_start_state --hold-steps 800 --random-steps 4000 \
  --seed 909 --out docs/receipts/probes/cv_entity_slots_site2_2026-08-10.json
.venv/bin/python scripts/probe_entity_slot_attribution.py
```

**Any profile axis this arm proposes to key cells on must pass 0′ first**
— that is the generalisation of the `entity_slots` lesson, and it is
cheap enough that there is no excuse for the exception.

**GATE (binary, pre-registered).** PASS on either:

* **G1 — the wall moves.** **Any** of the following, replay-verified
  from its root and NOT reproduced by the matched control:
  * **G1a — frontier.** At least one archive cell at raw
    `progress > 767` (gx bucket > 95). This is the spec's "frontier past
    gx 767".
  * **G1b — exit.** A banked solution, i.e. a clear the counterfactual
    gate accepted (`runs/<out>/solutions/sol_*`).
  * **G1c — transition.** An observed `level_key` (`0x0028`) value the
    banked archives do not contain, or `area`/`sect` advance if either
    is ever configured.

  **G1a alone is not sufficient as a gate, and the original single-clause
  form could have read FAIL on the wall being beaten.** Three verified
  constraints combine to produce that: (i) CV's `solve` block has no
  `area:` key, so `GenericGame.area` returns literal `0`
  (`scripts/go_explore_solve.py:609` sets `self._area = None`, `:921-922`
  returns `0` for it) and `max_area` is `0` in both
  banked logs; (ii) no `room_advance:` is configured, so `sect` never
  leaves 0 (`max_sect 0`); (iii) `767 = 0x02FF` is exactly the page
  boundary of the 16-bit LE `progress` pair. If the hall's exit is a
  `level_key` change rather than continued rightward scroll, the clear
  hook fires, `progress` resets low, and `keep_exploring(1, 1)` returns
  `False` at `scripts/go_explore_solve.py:2783` — the run **breaks on
  the first banked solution**. The outcome is a replay-verified solution
  and *zero* cells above bucket 95. Under the old wording, §5's FAIL
  clause ("neither") would then have scored a cracked wall as a failure.
  G1b/G1c close that hole; the phrase "deepest area" is dropped because
  for this profile it is a constant and reads as a filter that does not
  exist.

  **Corollary for the read-out.** `--want-solutions 1` means G1b and G1a
  are partly exclusive: the run cannot keep extending the frontier after
  it banks a clear. Do not read "no cells above 95" as "no crossing"
  without first checking `runs/<out>/solutions/` and the run's exit
  reason.
* **G2 — the wall is re-classified with evidence.** The discovery half
  returns ≥ 1 candidate surviving P1–P4; adding it raises
  `boundary_state_axes` from **1** to **≥ 2** at the pin; and with that
  axis live the frontier still does not move. ~~The hall then moves from
  GATED-BLIND to GATED-TRUE — a strictly narrower, falsifiable claim
  ("we now have an interaction axis and the wall still holds") and the
  **first non-circular evidence** that the GATED positive class is real.~~

  [struck 2026-08-11: see §12 / falsifier receipt. G2 is doubly dead. It
  was already DELETED by the campaign doc (rev-2 A1, restated §2: a
  reclass "can never satisfy STRATEGY G4" and is a reported FINDING
  only). It is now also incoherent on its own terms: a reclass between
  two labels of a class with no separating statistic is not evidence
  that the class is real, it is a rename. The falsifier reached G2's
  target — "the first non-circular evidence about the positive class" —
  by the opposite route and cheaper: the class has no member and no
  statistic, so there was nothing to confirm.]

FAIL: neither. `boundary_state_axes` stays 1, **no G1a/G1b/G1c
crossing**, all per-candidate lifts flat. A FAIL may only be declared
after all three G1 clauses have been checked and the check recorded —
"we looked at the frontier bucket" is not a FAIL read-out.

**Anti-gaming clauses, binding:**

* `boundary_state_axes` may only be raised by an axis with a probe
  receipt. Adding a high-cardinality byte to inflate the count is the
  `route_sig` failure mode and is refused by P4. **This applies to
  incumbent axes too** — the clause was written exempting
  `entity_slots`, the one axis the arm was actually built on and the one
  axis with no receipt. That exemption is withdrawn; the range has been
  probed (§7.6) and the "cheap first move" it justified has been
  deleted from the validation order. Any future axis, incumbent or new,
  passes Phase 0′ or does not enter the key.
* A crossing that the matched control also produces is **not** a pass.
  It is the taxonomy's falsifier (§7.4) and must be reported as such.
* Cite the harness by name in every claim. "7/24" once meant training
  telemetry and cost a day.

---

## 6. Kill criteria

Stop and report, rather than iterating:

1. **Throughput.** The arm costs > 40% of the control's steps/s. The
   ortho arm cost 31% of total new cells and that already made the A/B
   hard to read; beyond 40% the comparison is meaningless. Re-tune
   before validating, do not "just run it longer".
2. **Discovery null.** P1–P4 return zero candidates within
   `--gate-probe-bursts`. The possession sub-arm is dead for CV. Publish
   the null the way BB's was published.
3. **Farmable axis.** A candidate's cardinality grows without bound
   across bursts. Refuse it; it is nuisance novelty by definition.
4. **Archive blow-up.** Adding an axis multiplies cells > 3× at equal
   steps with no movement above the pin. Evict it. (The v4 full-RAM-hash
   lesson: 585k one-visit cells.)
5. **Two seeds, no crossing.** Two bounded runs, different seeds, G1 and
   G2 both fail ⇒ the hall is not gated on any axis this arm can
   express. Escalate to (a4) or a different mechanism class, and record
   that the ~~GATED~~ positive class is **still empty** [struck
   2026-08-11: see §12 / falsifier receipt — the class is not merely
   empty of members, it is empty of a *statistic*; there is no longer a
   class for a wall to be recorded into].
6. **Scope discipline.** A lockstep/fidelity receipt only covers the
   trajectory it covers — the bat-wake campaign's closing lesson. If the
   arm's evidence comes from bursts that never reach the pinned band, it
   says nothing about the pin.

---

## 7. Standing caveats

**7.1 Purity and provenance.** Every observable this arm may use comes
from our own differential probing, and every one now has a citable
artifact:

| observable | addresses | receipt |
|---|---|---|
| `progress` | `0x0040`/`0x0041` | `docs/receipts/ram_verify/castlevania.json` |
| `y` | `0x003F` | `docs/receipts/ram_verify/castlevania.json` |
| `lives` | `0x002A` | `docs/receipts/ram_verify/castlevania.json` |
| `level_key` | `0x0028` | `docs/receipts/ram_verify/castlevania.json` |
| `state_sig` | `0x0004 == 8` | `docs/receipts/observatory/cv_lvl02_acceptance.json` (run `runs/cv_chain_hw/lvl_02`, rank 0 of 1888, cond_MI 21.411) |
| `entity_slots` | `0x0450`–`0x0457` | `docs/receipts/probes/cv_entity_slots_2026-08-10.json` + `…_site2_…` + `…_attribution_…` |

**The `entity_slots` row is new, and it was a real gap.** Until
2026-08-10 the *only* provenance for `{lo: 0x0450, hi: 0x0458}` was a
prose comment in `configs/castlevania.yaml` asserting "block-3
differential probes 2026-07-29/30". No such artifact existed — a
repo-wide search for the address returned this document, the profile
comment, and two unrelated files. That is precisely the standard
`docs/receipts/games/gradius_receipt.json` sets when it says entity-slot
candidates "still need the verify-RAM receipt process before entering
the production profile", and §5's anti-gaming clause cannot demand a
probe receipt for every new axis while exempting the one the arm is
built on. The gap is now closed by measurement, not by grandfathering —
see §7.6 for what the measurement actually returned, because it does not
say what the profile comment says.

P1–P4 are differential probes of our own rollouts. No map, no
walkthrough, no disassembly. The test in both directions: *could this
decision have been made by a party who has never seen the game?*

**7.2 Discovery must precede search.** §2.2's measurement is the reason
the arm has a `probe` mode at all. Dispatching a search arm at a wall
whose archive has one state bit repeats the ortho arm's mistake one
level up.

**7.3 The helper gates nothing.** `boundary_axis_profile` is additive and
reported. No banked verdict moves. Promoting `boundary_state_axes` into
`gated_wall_verdict` follows the receipt §8 promotion sequence: (a) emit
the field, (b) bank runs with labelled outcomes, (c) derive the band,
(d) flip the switch — in that order, in the commit that measures it.

**7.4 The positive class is still empty.** The receipt's falsifier was
"if run A finishes the hall by ordinary coverage, the GATED class has
nothing to separate." ~~It did not fire~~ — run A ended with 0 solutions and
the pin unmoved — but *not firing is not confirmation*. [struck
2026-08-11: see §12 / falsifier receipt — it fired on 2026-08-10 by a
route this paragraph did not anticipate. The falsifier was written to
require run A to FINISH the hall; scoring already-banked SOLVED archives
through the same statistic turned out to be sufficient and cost no
compute at all.] The hall remains
one unsolved candidate scored four times by the same statistic. ~~**G2 is
the first design in this program that could confirm the class without
solving the wall**, and G1 confirms it by solving it. Until one of them
lands, `CONCENTRATION_GATED_MIN` still rests on a 1.51×-wide band whose
upper bracket is an unvalidated wall, and the honest summary stays: *we
can say reliably when a wall is NOT gated; whether we can say when one
IS remains untested.*~~

**[struck 2026-08-11: see §12 / falsifier receipt — §7.4's falsifier
fired, by a cheaper route than either G1 or G2.** It did not need the arm
to finish the hall. Scoring four banked SOLVED archives through the same
statistic with the RESOLVED/PROGRESSING escape branches bypassed was
enough: all four read GATED, three outrank every hall read, and no
threshold can exclude them without emptying the class of its own
definers. So the honest summary inverts — **we cannot reliably say a wall
is NOT gated either**, because `smb_4_4_micro/lvl_1-3` seg 2 read GATED
on the *unmodified shipped path* and then cleared 17 minutes later on a
plain retry with no orthogonal mechanism of any kind. That is a false
GATED with no bypass at all, and it falsifies the calibration's strongest
surviving claim (`test_no_false_gated_anywhere_in_the_corpus`). The
1.51×-wide band was fitted between a 19,958-cell resolved archive and a
28,929-cell gated one because a pickle read error capped the corpus's
resolved class at ~20k cells — a read error that does not reproduce. What
this paragraph got right and should be kept: the positive class was one
unsolved wall scored repeatedly by the same statistic, and that was
always the defect. The remaining honest sentence is the one the
size-decoupled receipt lands on — *the classes overlap, and the missing
input is `c_local` emitted per progress line, not a 23rd surrogate
statistic.*]

**7.5 Not covered here.** The gate-opener is a solver arm. It produces
teacher data; it is not the learned policy that is the product. Every
level it opens is curriculum fuel, and that hand-off is out of scope for
this document.

**7.6 What the `entity_slots` probe returned — the range is real, the
"kill" reading is refuted.** Two sites, whole-page sweep of
`0x0400`–`0x04FF`, every action in the profile's own action space held
in isolation, plus sticky-random; 32,000 steps at the block-3 hall root
and 16,800 at the profile start state. Generator:
`scripts/probe_entity_slots.py` and
`scripts/probe_entity_slot_attribution.py`.

* **The bounds are confirmed, and derived rather than assumed.** The
  probe does not look at `0x0450`; it computes every maximal contiguous
  run of binary-`{0,1}` bytes in the whole page and asks whether the
  longest one coincides with the claim. At the hall root the longest run
  is **exactly `0x0450`–`0x0458`, length 8**; the runner-up is length 3
  (`0x04AE`–`0x04B1`). At the start state the run is
  `0x0450`–`0x0457` (length 7) because the top slot briefly held `0xFF`
  there. So `lo` is exact at both sites and `hi` is exact at the site
  the arm runs from. Twenty-four bytes in the page are binary; only
  these eight form a contiguous block. **The claimed range is a measured
  structure.**
* **Occupancy is agent-coupled.** Pooled `1→0` edges over the eight
  slots: 0.00 per 1k under the NOOP arm at the hall root, 24.00 per 1k
  under the best arm. Slots genuinely spawn and despawn under our
  inputs.
* **But the despawn edge is locomotion, not combat.** Every arm above
  the NOOP baseline at *both* sites contains `right`; every arm without
  it sits *exactly* at baseline — including `B` and `down+B`, the two
  attack actions. Holding locomotion constant, `right+B` vs `right`
  moves the pooled rate by **+2.0 / 1k** at the hall root and **+0.0 /
  1k** at the start state — the same order as the non-attack negative
  control `right+A` vs `right` (**+0.0 / 1k** at both). The slots track
  what has scrolled into and out of the camera.
* **Therefore the profile's comment is wrong where it matters.**
  `configs/castlevania.yaml` says "`--kill-key` counts 1→0 transitions
  as kills". Measured, those transitions count *scroll-outs*. A
  cumulative count over them is a monotone function of distance
  travelled — a positional alias wearing an interaction axis's clothes,
  which is the exact failure mode (a1)'s farmability rule exists to
  refuse. And it fails that rule outright: 317 `1→0` edges over 8 slots
  at each site, i.e. slots respawn ~40× over.
* **Scope caveat, stated at the strength the data supports.** Held
  single actions from one root per site. A whip only connects when
  something is in range, so a null attack contrast from a fixed root is
  evidence that *held* attack does not drive the edge — not proof that
  no contact ever clears a slot. The instrument that could settle it is
  a contact-conditioned differential, i.e. P1–P4, not `--kill-key`.
* **Residual defects in the range, recorded rather than smoothed.** At
  the hall root `0x0451` and `0x0452` are byte-identical columns (7
  distinct signatures across 8 slots) and `0x0453` showed 5 despawns and
  **zero** spawns in 32,000 steps. So (a1)'s "8 slots → 256 masks" is an
  overestimate: the reachable mask space is smaller, and `--gate-cap 64`
  is comfortable rather than tight.
