# Contra re-entry — registration under the elimination-ledger rule

**Date:** 2026-08-10
**Status:** REGISTERED, NOT LAUNCHED. This document is the pre-commitment.
No solver campaign was run for it. The only emulator work below is a
read-only observable re-probe and a 2-minute / 4-worker command smoke test,
both scoped and receipted in §4 and §7.
**Profile touched:** `configs/contra.yaml` — comments only, parse-identical
(§6).
**Gate artifact:** `docs/receipts/games/contra_reentry_gate_2026-08-10.py`,
frozen and hashed in §5. The stopping rule is executable and self-testing,
not prose plus a shell snippet — the first draft was the latter and could
not emit its own predicted outcome (§5, §7).
**Rule being satisfied:** a shelved game may be re-entered only if the prior
is stated, a material difference is named, and a stopping rule is registered
*before* the attempt.

---

## 0. The one-paragraph version

Contra was shelved after ten campaigns left the frontier pinned at exactly
the same value every time.

> **CORRECTION 2026-08-26.** This sentence read "…after ten campaigns
> produced zero solutions with the frontier pinned at exactly the same
> value every time." The "zero solutions" half is struck. `configs/contra.yaml`
> ships `level_key: []`, so `GenericGame.is_clear`'s opening test is
> `() > ()` — False for every RAM state — and the `clear: {mode: confluence}`
> fallback has never been shown to fire on a real clear of any game. The
> solution count across those ten campaigns was therefore not a search
> outcome; it was fixed before the first step ran. **The shelving decision
> is unaffected and stands on the frontier pin at gx 3072 alone**, which is
> a genuine measurement. See the ADDENDUM at the end of §1. Re-entering it is
justified only if something is different. Three things are, and they are
dated: replay-verified banking plus the counterfactual gate (2026-08-08 /
2026-08-10), the wall discriminator (2026-08-10), and a fresh observable
re-probe (run today, §4). Applying the discriminator to the ten banked
campaigns — free, no compute — **names the wall GATED and prescribes an
orthogonal arm**, and the orthogonal arm shipped 2026-08-08, seven days
after the last Contra campaign ended. That, and not "try harder", is the
material difference. The registered attempt is 30 bounded minutes of that
arm, and it is graded on **telemetry, not on a clear**. §8 also records
why the honest reading of the GATED label is weaker than it looks.

---

## 1. The prior, stated

Ten campaigns, `runs/breadth_contra/`, 2026-07-31 → 2026-08-01. **162 GB on
disk, 95,161,110 emulator steps, 74,636 s (20.7 h) of solver wall clock, and
a frontier that never left gx 3072 in nine of the ten.** ~~zero solutions~~
(struck 2026-08-26 — see the ADDENDUM below). In nine of the ten the frontier
(`max_gx_in_max_area`) reached **3072** — screen 12, the fixed-camera base
wall — within the first minute and never moved again. The tenth was a
control with deliberately broken observables.

| run | recs | frontier | cells | distinct spatial | concentration | verdict (2026-08-10) |
|---|---|---|---|---|---|---|
| `poweron_to_wall` | 24 | 3072 | 212,367 | 7,427 | 28.59 | **gated** |
| `stage1_baseline_collapsed_cells` | 8 | 2816 | 13 | — | — | insufficient |
| `stage1_v2` (clean coverage control) | 120 | 3072 | 313,666 | 7,983 | 39.29 | **gated** |
| `stage1_v3_kk_saturated` | 42 | 3072 | 154,896 | 7,654 | 20.24 | coverage_limited |
| `stage1_v4_localkk` | 120 | 3072 | 432,521 | 7,962 | 54.32 | **gated** |
| `stage1_v5_bosstyped` | 90 | 3072 | 289,945 | 8,046 | 36.04 | **gated** |
| `stage1_v6_resume` | 120 | 3072 | 652,059 | 8,182 | 79.69 | **gated** |
| `stage1_v7_doctrine` | 120 | 3072 | 1,443,275 | 8,281 | 174.29 | **gated** |
| `stage1_v7b_doctrine` | 420 | 3072 | 2,087,424 | 8,444 | 247.21 | **gated** |
| `stage1_v8_strategy` | 180 | 3072 | 2,248,480 | 8,477 | 265.24 | **gated** |

Read-only, from banked telemetry; the command is in §7. The wall itself is
described in `runs/breadth_contra/WALL_DOSSIER.md`.

**Five of these rows were re-measured from the archives on 2026-08-10** —
`poweron_to_wall`, `stage1_v2`, `stage1_v3_kk_saturated`,
`stage1_v4_localkk`, `stage1_v5_bosstyped` — and every `cells`,
`distinct spatial`, `spatial_span` and `concentration` figure reproduced
this table exactly. That matters because §5's gate is now built out of these
numbers: `FOOTPRINT_FLOOR` is `poweron_to_wall`'s 7,427 and `SPAN_BASE` is
the 385 confirmed across all five. The four largest archives (13-46 GB) were
not re-read; their rows stand as originally recorded.

### ADDENDUM (2026-08-26) — the prior is restated without the solution count

The registered prior above bolded *"…95,161,110 emulator steps, 74,636 s
(20.7 h) of solver wall clock, zero solutions."* Compute multiplied against
a zero reads as **"we ran 95M steps against a live win test and it never
fired."** No win test was live.

`configs/contra.yaml` ships `level_key: []`, so the opening branch of
`GenericGame.is_clear` is `level_key(ram) > tuple(start_key)` evaluated as
`() > ()` — False for every RAM state. The `clear: {mode: confluence}`
fallback is a 2-of-2 vote between `tally` and `coord` (the offline
detector's `audio` and `lock` signals do not survive into the live solver
hook, which sees only a RAM snapshot), and it has never been observed to
fire on a genuine clear of any game — the only times it ever fired in
production were the two false positives withdrawn on 2026-08-06. So across
all ten campaigns the solution counter was a constant, and citing it
alongside the compute total implies a corroboration that was never
available.

**What actually carries the shelving, and it is enough:** the frontier pin.
Nine of ten runs reached gx 3072 inside the first minute and never moved
again, across 2.2M-cell archives and eight distinct doctrine variants. That
is a real, reproduced, load-bearing measurement, and the GATED taxonomy
label and the orthogonal-arm prescription rest on it, not on the solution
count. Nothing in §5's gate, §7's commands or §8's honest reading changes.

**What is now UNKNOWN that this document implied was known:** whether
Contra's confluence hook could fire at all. Contra's `progress` is a 16-bit
`{lo: 0x65, hi: 0x64}` pair, so `coord`'s required ≥300 backwards drop is
arithmetically *possible* here — unlike an odometer-sourced profile, where
it is not. Whether `tally` has any referent in this game is a fact about the
game that can only be settled by measuring it, and the purity line (CLAIMS.md
Tier 3) forbids asserting it from recalled knowledge in either direction. The
honest status is UNTESTED, and `scripts/clear_reachability.py` deliberately
passes this profile for exactly that reason rather than refusing it on a
hunch.

Source: `docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md`. The same
correction applies to the r1_ortho orchestrator verdict later in this
document ("52,539 cells, 0 solutions"), annotated in place.

The fresh campaigns' own progress series also fix the budget arithmetic §5
depends on. At **t = 1740 s** — the registered budget — the five *fresh*
(non-resumed) campaigns stood at **122,603 / 137,855 / 151,495 / 169,070**
cells, and `poweron_to_wall` reached 205,826 by its 1441 s finish. So a
fresh 30-minute run should be expected in the **~120k-210k cell** band.

The single most important row is not a verdict, it is **`distinct spatial`**.
Across a 13x growth in cells (154,896 → 2,248,480) the map footprint moved
7,654 → 8,477, about +10%, and `spatial_span` is **385 in every campaign
measured**. The archive was not exploring. It was re-describing one frozen
footprint in ever finer detail.

---

## 2. Material differences NOW

Each one is dated against the campaign window (2026-07-31 → 08-01), so the
claim "this is new" is checkable rather than asserted.

### (a) Trustworthy termination — replay-verified banking + the counterfactual gate

- `55c1996`, **2026-08-08**: confluence detector v2 — death-first ordering,
  blip persistence, transition veto, replay-verified banking.
- `f45dda1`, **2026-08-10**: detector v3 (APU channel-activity vote) and the
  counterfactual candidate gate.
- `ce519e6`, **2026-08-10**: the counterfactual gate refuses state artifacts.

All three postdate every Contra campaign by 7-10 days.

This matters specifically for this profile. `configs/contra.yaml` carries
`clear: {mode: confluence}` — a **windowed** hook, the one class of clear
signal that can fire on evidence accumulated over a whole trajectory and be
reproduced perfectly by `--verify-bank` while corresponding to nothing the
game did. The solver's own docstring says such a profile should be
configured expecting the counterfactual gate to be on. The registered
command therefore passes `--counterfactual-gate` explicitly. It is
default-OFF, costs nothing unless a candidate actually fires, and touches
only the banking path — never the search.

**Stated honestly:** zero clears ever fired in the ten campaigns, so no
Contra receipt was ever banked untrustworthily. This difference is
**prospective**, not a repair. It is what makes a fire *this* time
bankable; it retracts nothing.

### (b) The wall can now be named — and the named remedy is a mechanism we did not have

`src/training/wall_taxonomy.py` (`f45dda1`, **2026-08-10**) classifies a
stalled search. Applied to the ten campaigns it returns **GATED** on eight
of them, with the remedy: *"switch to an orthogonal arm: a different action
axis, or a mechanic the current cell key cannot express."*

The orthogonal-frontier arm — `--ortho {up,down}` and its knobs — shipped in
`5f8dcb7`, **2026-08-08**. It did not exist during any Contra campaign and
has never been pointed at this game. The v4/v5/v6 arms (local kill-key,
typed boss HP, time bins) all added *dimensions to the cell key*; none of
them changed *which cells selection restarts from*, which is what the ortho
arm does.

Contra is **not** in the discriminator's calibration corpus
(`docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md`), so this is
an out-of-sample read, not a fit. See §8 for how much weight the label can
actually carry.

### (c) Verified observables — the 3-probe protocol, re-run today

Full results in §4. Summary: the two bytes the solver actually depends on
re-derive cleanly from scratch on today's core, and the two disputed
player-state bytes are both refuted as death signals by direct measurement.

### (d) 2P — registered as a FUTURE material difference, explicitly NOT this attempt

Two-player Contra is a genuine mechanical difference and is *not* being
claimed here. Registering it now so a later re-entry cannot present it as a
fresh idea:

- **Core support exists and is default-inert.** `nes_core/src/pool.rs`
  exposes `step_all_2p(actions, actions_p2)`, with tests pinning that
  all-zero P2 masks are byte-identical to `step_all`
  (`zero_p2_masks_are_byte_identical_to_single_player_step`).
- **No Python caller uses it.** `grep step_all_2p` over `scripts/`, `src/`
  and `tests/` returns nothing. `scripts/go_explore_solve.py` is
  single-controller throughout. A 2P campaign is therefore solver work, not
  a flag.
- **Why not now.** A 2P line needs a 2P start state, a second action space,
  a doubled action-product per step, and its own chain lineage — a second
  from-scratch chain opened while the first one's wall is still unnamed in
  the frontier sense. The Bubble Bobble receipts are the precedent against
  it: chains cost 25-45 min per level and the BB chain is still unfinished
  at round 60. Opening a second unfinished chain to avoid finishing the
  first is the failure mode this ledger exists to prevent.
- **Preconditions for 2P to become a live option:** (1) this registration's
  attempt reads out; (2) `step_all_2p` has a solver-side caller with its own
  tests; (3) the 1P base wall is either passed or eliminated on the record.

---

## 3. What is deliberately NOT claimed as a difference

- **"More compute."** 20.7 h has been spent. More of the same is the
  definition of what the ledger forbids.
- **"A better cell key."** v4/v5/v6 already tried three. All read GATED.
- **A cleared stage.** No clear has ever fired on Contra. Nothing here
  predicts one, and §5's gate does not reward one.
- **That the 07-31 wall receipts are wrong.** They are not being revisited.
  `WALL_DOSSIER.md` stands as written.

---

## 4. The fresh 3-probe protocol (run 2026-08-10)

`scripts/discover_observables.py` from this profile's own verified start
state. Every number is measured by this core under our own scripted inputs:
forward-hold, NOOP-hold, reverse-hold, plus the advance probe (macro drive
with reload-on-mass-reset) for the death events. No external RAM map, no
disassembly, no walkthrough. Raw findings JSON:
`/tmp/agent_contra_reentry/probe_findings.json` (scratch, not committed).

```
ROM   roms/Contra (USA).nes                 md5 7bdad8b4a7a56a634c9649d20bd3011b
state roms/Contra (USA)_start.state.bin     sha256[:16] b99f9be8e0266f6d
frame_skip 4 (from the profile), forward='right', seed 1
```

### PROGRESS — re-derived identical to what is wired

| candidate | Gate 1 forward | Gate 1 NOOP | wrap-coupled | Gate 2 saturation |
|---|---|---|---|---|
| `$0065\|$0064<<8` (**recommended**) | net +635, monotone 1.00 | net 0 (flat) | 3 | no clamp; rebases 0, within-room cap false |
| `$00FD\|$0064<<8` | net +635, monotone 1.00 | net 0 (flat) | 3 | no clamp |

The recommendation is byte-for-byte the pair already in `solve:`. `$00FD`
moves identically to `$0065` and is a co-equal candidate the probe cannot
separate; it is recorded, not adopted, because switching between two
indistinguishable bytes buys nothing and costs comparability.

### ROOM COUNTER — `$0064`, which is already the progress page byte

`role: progress_page_as_screen_counter`, 3 distinct values (0, 1, 2) over
the probe, 0 rebase hits.

### LIVES — `$0032`, re-derived, top of 4 candidates

start 2, 2 clean decrements, churn 1.11. Runners-up `$003A`, `$0050`,
`$0051` all show a single decrement. Matches what is wired and what memory
already recorded as independently confirmed.

### PLAYER STATE — both candidate bytes REFUTED as death signals

The profile wires `player_state: 0x002C, death_states: [1, 2]`; the
`ram_mapping` block claims `0x0090` is the "corrected" byte. Measured:

| byte | fraction in {1,2}, forward | NOOP | reverse | advance |
|---|---|---|---|---|
| `$002C` (wired in `solve:`) | 0.000 | 0.000 | 0.000 | 0.000 |
| `$0090` (the June "correction") | 0.895 | 0.967 | 0.991 | 0.198 |

- **`$002C` is measurably inert.** Observed values are 3, 4 and 10; it never
  once entered its own declared death set across 3,265 probe steps. Over the
  advance probe, `is_dead` with it wired fires on 0.893 of steps — exactly
  the lives-only rate, to three decimals. Its clause adds **+0.000**. It is
  harmless (zero false positives) and detects nothing.
- **`$0090` is refuted, hard.** Wiring it with the same death set would make
  `is_dead` true on **0.996** of advance-probe steps against 0.893 for lives
  alone. It sits in {1,2} during ordinary living play. It would truncate
  essentially every trajectory.
- **Neither is patched.** `$002C` stays exactly as wired (changing it is a
  behaviour change with no measured benefit); `$0090` stays out. The
  standing UNRESOLVED status is correct and is now backed by numbers rather
  than by recollection.
- **Consequence to state plainly:** Contra death detection is, today,
  **lives-only**. That is sound — `$0032` re-derives cleanly — but the
  adapter has no player-state signal, so a death is seen only when the life
  counter moves, not at the instant of the fatal hit.

### Y — the wired byte is rank 3 of 6, not rank 1

| addr | base | mode | max dev | score |
|---|---|---|---|---|
| `$0010` | 0 | 0 | 115 | **90.8** |
| `$0015` | 66 | 116 | 66 | 70.7 |
| `$031A` (**wired**) | 52 | 100 | 52 | 57.9 |

Both pass the ballistic signature. Not adopted — see §6.

---

## 5. The stopping rule, registered in advance

### Shape

**30 minutes, bounded, one seed.** Graded on **telemetry, not on a clear.**
A clear is not required to pass and would not by itself pass — a fired
clear must additionally survive `--verify-bank` and `--counterfactual-gate`
before it means anything, and that is a separate receipt.

### The arm

The orthogonal arm named by the discriminator's own remedy (§2b), at the
receipted base configuration otherwise. `--ortho down`: the wall's core sits
low, which is our own pixel measurement from banked wall states
(`/tmp/contra_wall_*.png`, `runs/breadth_contra/pixel_phase_receipt.json`,
2026-07-31) and not outside knowledge; the profile's action space already
carries the `["down","B"]` prone move added for exactly that geometry.

The arm engages only after `--ortho-pin-secs` of a *pinned* frontier.
Measured in the §7 smoke test: at 120 s elapsed the run reported
`pinned_secs: 91`, so the pin clock starts about 29 s in and the arm engages
at roughly **150 s elapsed** — about 8% of a 30-minute budget spent
unarmed. `--ortho-pin-secs` is left at its default 120 deliberately: tuning
a knob down on a single smoke observation, in the same run that is supposed
to test the arm, is how a registration turns into a fishing expedition.

### Gate

The gate is **an executable file, frozen at registration**:

```
docs/receipts/games/contra_reentry_gate_2026-08-10.py
sha256 90ddb5b2b57a4447c5e15c1f6700779c5ff7d998d9c41d0cb79aca15587f10b4
```

Reading the attempt with a modified copy is not the registered gate. Its
`decide()` is pure and its `--self-test` runs the branch table below as
assertions — 27 of them, all passing (§7). The prose here and that table
are the same commitment stated twice; if they ever disagree, this section
wins and the file is wrong.

#### Why this is not keyed on the taxonomy verdict

The natural way to write "the wall is still there" is "`gated_wall_verdict`
still returns GATED". **That would have been a broken rule, and the first
draft of this registration made exactly that mistake.** `gated_wall_verdict`
reaches GATED on this profile only through `CONCENTRATION_GATED_MIN = 25.0`,
and per §8 caveat 2 concentration here is very nearly `cells / 8000` — a
monotone function of run length. Measured from the five fresh campaigns'
own progress series at t=1740 s, a fresh 30-minute run lands at
**122k-206k cells, i.e. concentration ~15-26**. It straddles the threshold.
So an ELIMINATE branch conditioned on the word GATED would decide a
re-shelving on a coin flip about throughput, and the single most likely
read — a `coverage_limited` verdict like `stage1_v3`'s, whose remedy is
literally *"give it more wall-clock before changing anything"* — would have
been scored as a partial success.

The statistics that carry the load are the ones that **do not move**:
`distinct_spatial` and `spatial_span`. They saturate fast. `poweron_to_wall`
reached 7,427 buckets in 24 fresh minutes, and the remaining 20.7 hours of
prior search bought +14% and **zero** change in `spatial_span`, which is 385
in all ten campaigns and was re-measured at 385 in five of them on
2026-08-10. That is a quantity a 30-minute run can actually move, and
therefore a claim a 30-minute run can actually falsify.

**`CONCENTRATION_GATED_MIN` appears in no branch condition below.**
Concentration and `wall_class` are still printed, tagged `REPORTED_ONLY`.

#### Constants, all measured in §1

| constant | value | what it is |
|---|---|---|
| `BASE_FRONTIER` | 3072 | frontier receipted in nine of ten campaigns |
| `SPAN_BASE` | 385 | `spatial_span` in every campaign measured |
| `FOOTPRINT_FLOOR` | 7,427 | smallest footprint any campaign reached (`poweron_to_wall`, fresh, 24 min) |
| `FOOTPRINT_CURVE` | 9 points | `(cells, distinct_spatial)` for the nine frontier-3072 campaigns |
| `FOOTPRINT_MARGIN` | 0.10 | how far past the matched-coverage prior counts as a shape change |

`footprint_envelope(cells)` is the **upper envelope** of `FOOTPRINT_CURVE` —
the best `distinct_spatial` any prior campaign reached at or below that cell
count — clamped, not extrapolated, outside the measured range. The raw curve
is non-monotone (`poweron_to_wall` sits *below* `stage1_v3` despite 37% more
cells), so the running maximum both restores monotonicity and picks the
conservative direction: the attempt is measured against the best the prior
ever did at that coverage, never its average. Clamping below 154,896 cells
compares against 7,654, which is a *higher* bar than extrapolating down
would set, so a thin run cannot earn PARTIAL against a conveniently low
expectation.

`FOOTPRINT_MARGIN = 0.10` is derived, not chosen: the envelope runs
7,654 → 8,477, a factor of 1.108, so +10% at matched coverage means the
30-minute arm found as much additional footprint as **the entire 20.7 hours
of prior search**. The curve's residual scatter at comparable cell counts is
~3% (7,427 at 212k vs 7,654 at 155k), so the margin is ~3x noise.

#### The three inputs

- **G-0 (validity — the arm actually ran).** `ortho_selections > 0` in the
  run's own last progress segment. The solver emits `ortho_selections`,
  `ortho_pool`, `ortho_cols_improved`, `ortho_best_yband` and `pinned_secs`
  per record. Read from the *last segment* specifically, so a resumed file
  cannot credit this attempt with a previous one's arm.
- **G-A (frontier).** `max_gx_in_max_area > BASE` somewhere in the run's own
  `progress.jsonl`. A pure high-water mark; needs no archive.
- **G-B (footprint).** From the run's own `archive.pkl`:
  `spatial_span > SPAN_BASE` **or**
  `distinct_spatial >= 1.10 x footprint_envelope(cells)`.
  `gated_wall_verdict` is consulted **only** as a guard — `barren` or
  `key_blind` means the arm broke the search rather than testing it.

#### The four branches, in evaluation order

| # | condition | outcome |
|---|---|---|
| 1 | `ortho_selections == 0` | **VOID** `arm_never_engaged` |
| 2 | `max_gx_in_max_area > 3072` | **PASS** |
| 3 | no `archive.pkl` | **VOID** `no_archive` |
| 4 | `wall_class` in {`barren`, `key_blind`} | **VOID** `search_broken` |
| 5 | `wall_class` in {`insufficient`, `indeterminate`} | **VOID** `unclassifiable` |
| 6 | `distinct_spatial < 7,427` | **VOID** `under_covered` |
| 7 | `spatial_span > 385` or `distinct_spatial >= 1.10 x envelope` | **PARTIAL** |
| 8 | otherwise | **ELIMINATE** |

Four outcomes — VOID, PASS, PARTIAL, ELIMINATE. VOID carries a
machine-readable `reason` because there are four distinct ways for an
attempt to test nothing and they call for four different repairs. G-A is
checked before the archive-dependent guards on purpose: passing a frontier
that stood for 20.7 hours is the headline whatever the classifier thinks.

**Row 6 is the ELIMINATE precondition, and it is the point.** A run that
does not re-cover the smallest footprint any prior campaign reached has not
earned an elimination: failing to *exceed* a prior you did not *reproduce*
is not evidence the prior is immovable. That reads VOID and buys more
budget, not a re-shelving. `poweron_to_wall` cleared this floor in 24 fresh
minutes from power-on, so a 30-minute run at the same base configuration is
expected to clear it — but the gate checks rather than assuming.

**G-B alone cannot pass this gate, deliberately.** §1 already discharges
"the wall can be named" for free, from disk. A gate satisfiable by
re-deriving something already known would be rigged.

#### Retrospective check on the rule itself

Applied to the nine campaigns it is built from, the rule returns
**ELIMINATE on every one** (asserted in `--self-test`). That is the property
that makes a PARTIAL mean something: if any prior campaign scored PARTIAL,
the rule would be calling the prior's own null result a shape change.

### Pre-registered prediction (so this is falsifiable)

**The frontier stays at 3072, the footprint stays frozen, and the gate
prints `ELIMINATE`.** Ten campaigns say so. Note that this is now the
literal string the gate emits — verified by running it against
`poweron_to_wall` and `stage1_v3` with the arm's telemetry injected, both of
which print `ELIMINATE / footprint_frozen` (§7). The earlier draft of this
gate printed `PARTIAL` for both, which is why this section was rewritten.

If the frontier moves, that is new information and worth the 30 minutes; if
it does not, ELIMINATE fires and Contra goes back on the shelf **with the
orthogonal arm crossed off**, which is the actual product of this attempt.

### Budget

30 min wall clock, 8 workers, one seed, ~1.5-2.5 GB of archive. No resume,
no second seed, no extension.

The gate has no ambiguous outcome by construction — every input combination
lands in exactly one of the four branches, which is what `--self-test`
checks. The one re-run this registration permits is a **VOID repair**: a
VOID means the attempt tested nothing (the arm never armed, the search
broke, or the run under-covered), so it may be re-run **once**, with the
named defect fixed, and that repair is bounded by the same 30 minutes. PASS,
PARTIAL and ELIMINATE are all terminal for this registration. "The result
was ELIMINATE but let us try once more" is exactly what the ledger forbids.

---

## 6. What was amended in `configs/contra.yaml`

**Comments only. Zero functional change.** Verified by parsing `HEAD`'s copy
and the working copy with `yaml.safe_load` and comparing: `parsed-equal:
True`, and the `solve:` block is JSON-identical under sorted keys.
`make_game(profile)` builds the same `GenericGame` with the same addresses
(progress `0x65|0x64`, y `0x31a`, lives `0x32`, area `None`, clear
`confluence`, pstate `0x2c`, death states `(1,2)`).

Recorded in the profile: the re-verification of progress and lives with
today's numbers; the measured inertness of `$002C` and the refutation of
`$0090`, with a rediscovery rule; and the two probe-surfaced candidates
below, marked NOT ADOPTED.

### Deliberately not adopted

- **`area: 0x0064`.** The probe names it the room/screen counter, and it is
  correct — but it is already this profile's progress *page* byte, so
  wiring it as `area` adds no information to the cell key while re-basing
  `max_area` / `max_gx_in_max_area`. The re-entry gate is measured against a
  frontier of 3072 recorded under `area = 0`. Changing the denominator in
  the same breath as running the experiment would destroy the comparison.
- **`y: 0x0010`.** Outranks the wired `$031A` on the probe's heuristic score
  (90.8 vs 57.9). Both pass the ballistic gate. Swapping y re-bands every
  cell in the archive and, again, breaks comparability with §1. Owed a
  controlled A/B, not a silent swap on the strength of a ranking heuristic.

Both are behaviour changes, and this task's constraint is
opt-in / default-identical.

---

## 7. Commands

All paths relative to the repo root; `.venv/bin/python` throughout.

**The registered bounded attempt** (~31 min: 30 min budget plus startup and
final flush):

```sh
mkdir -p runs/contra_reentry_2026-08-10
.venv/bin/python scripts/go_explore_solve.py \
  --profile configs/contra.yaml \
  --root-state "roms/Contra (USA)_start.state.bin" \
  --out runs/contra_reentry_2026-08-10/r1_ortho \
  --minutes 30 --workers 8 --burst 64 --seed 0 \
  --want-solutions 1 \
  --sel-mode count --ortho down --ortho-pin-secs 120 \
  --verify-bank --counterfactual-gate \
  2>&1 | tee runs/contra_reentry_2026-08-10/r1_ortho.log
```

`--sel-mode count` is required for `--ortho-weight` to have any effect (the
flag's own help says so). Everything else is the receipted base
configuration: no hw flags, no resume, default `--gx-bucket 16`,
`--y-band 32`, `--burst 64`. The clear hook's observation budget is 20,
comfortably inside a burst of 64, so the hook does not starve.

**The gate readout** (seconds of CPU; peak RSS scales with the archive —
39 GB on the 41 GB `v7b` archive — so run it alone):

```sh
.venv/bin/python docs/receipts/games/contra_reentry_gate_2026-08-10.py \
  runs/contra_reentry_2026-08-10/r1_ortho
```

Verify first that the frozen gate is the registered one:

```sh
shasum -a 256 docs/receipts/games/contra_reentry_gate_2026-08-10.py
# 90ddb5b2b57a4447c5e15c1f6700779c5ff7d998d9c41d0cb79aca15587f10b4
.venv/bin/python docs/receipts/games/contra_reentry_gate_2026-08-10.py --self-test
# all 27 branch assertions pass
```

The gate lives under `docs/` rather than `scripts/` for a mechanical
reason, not a stylistic one: `tests/test_wall_taxonomy.py::
test_module_is_not_wired_into_any_runtime_dispatch` fails if any `.py`
under `src/`, `scripts/`, `nes_core/` or `configs/` so much as mentions
`wall_taxonomy`. It is read-only and out-of-band on purpose. **Do not move
it into `scripts/`, and do not wire it into the solver.**

**Gate validation already performed** (2026-08-10, read-only, against the
banked prior — the gate was exercised end to end before being registered):

| fixture | expected | printed |
|---|---|---|
| `--self-test` (18 branch rows + 9 retrospective) | all pass | **27/27 pass** |
| `stage1_v3` untouched (no arm field) | VOID | `VOID / arm_never_engaged` |
| `stage1_v3` + arm injected — *the coverage_limited boundary case* | ELIMINATE | `ELIMINATE / footprint_frozen` |
| `poweron_to_wall` + arm injected — *the predicted-failure picture* | ELIMINATE | `ELIMINATE / footprint_frozen` |
| two-segment file, arm engaged only in the **first** segment | VOID | `VOID / arm_never_engaged` |

The middle two rows are the whole repair. Both campaigns printed **PARTIAL**
under the earlier draft of this gate — one of them a `coverage_limited` read
whose own remedy is "give it more wall-clock", the other the archetypal
nothing-happened campaign. Scoring either as "the arm changed the shape of
the search" would have been false. Fixtures were built in `/tmp` by copying
each campaign's `progress.jsonl` with `ortho_selections` injected and
symlinking its real `archive.pkl`, so the footprint statistics are the real
measured ones, not synthetic.

The `--self-test` retrospective block additionally asserts that the rule
returns ELIMINATE on **all nine** frontier-3072 campaigns, so no prior
campaign's null result is scored as a shape change.

**Command smoke tests already performed** (2026-08-10, 2 min, 4 workers,
out to `/tmp`, both exit 0):

- *base configuration* — archive and `progress.jsonl` written,
  `hw_flags: []`, `nes_core sha256[:16] e09e8191b8d40490`.
- *the registered ortho arm, verbatim* — same result plus the arm's own
  telemetry: `ortho_pool 2535`, `ortho_cols_improved 101`,
  `ortho_best_yband 7`, `pinned_secs 91`, `ortho_selections 0`. The gate
  readout above, run verbatim against it, returned **`GATE: VOID`** — which
  is the correct answer for a 2-minute run whose arm never armed, and is
  the check that G-0 exists to make.

The base run reproduced the base wall on
today's core — `max_gx_in_max_area = 3072` inside the first 60 s, and
`best_score 35072`, identical to `stage1_v8_strategy`'s final banked score.
That matters because the core has changed since 2026-08-01 (the NMI
sub-cycle work) and the ten campaigns predate the hw-provenance plumbing:
none of their `archive.stats.json` files carry a `hw_provenance` block, so
their core lineage is unrecorded. The base wall reproducing bit-for-bit on
today's default core is what licenses comparing a new run against 3072 at
all.

---

## 8. Caveats — how much the GATED label can actually carry

Recorded here rather than discovered later.

1. **The discriminator's positive class is unvalidated.** Its own
   calibration receipt opens with a CONDITIONAL status: the corpus contains
   **no confirmed gated wall**, both GATED rows are the same Castlevania
   hall from two hardware lineages, and every band is conditional on
   `runs/cv_hall_ortho_a` reading out. `CONCENTRATION_GATED_MIN = 25.0` is
   therefore a provisional threshold. Contra's verdicts inherit that
   condition in full.

2. **Concentration is partly a clock.** This is the sharpest caveat and it
   is visible directly in §1. Across the campaigns the spatial footprint is
   effectively frozen (7,654 → 8,477 buckets, `spatial_span` 385 throughout)
   while cells grow 13x, so `concentration = cells / buckets` is very nearly
   `cells / 8000` — a monotone function of how long the run went. The
   shortest run (`v3`, 42 records) sits at 20.24 and reads
   COVERAGE_LIMITED; every longer run crosses 25 and reads GATED. **On a
   pinned frontier this statistic will eventually cross the threshold no
   matter what the wall is made of.** So "eight of ten read GATED" is
   substantially a restatement of "eight of ten ran long", and the load is
   carried by the frozen footprint (`distinct spatial`, `spatial_span`),
   not by the verdict word.

   **This caveat is now structural, not just noted.** §5's gate reads
   `distinct_spatial` and `spatial_span` and does not reference
   `CONCENTRATION_GATED_MIN` in any branch condition; `wall_class` survives
   only as a guard against `barren`/`key_blind`, and concentration is
   printed tagged `REPORTED_ONLY`. The arithmetic that forced this: a fresh
   30-minute run lands at 122k-206k cells (measured at t=1740 s in the five
   fresh campaigns), i.e. concentration ~15-26 — straddling 25. A gate keyed
   on the word GATED could not reliably reach its own ELIMINATE branch
   inside the registered budget, and its most likely single output would
   have been "the arm changed the shape of the search" on a run that did
   nothing of the kind. Worth carrying back to the discriminator's owner: a
   threshold on `cells / distinct_spatial` is, on any pinned frontier, a
   threshold on elapsed time.

3. **Every verdict here is `degraded: true`.** The solver does not emit
   `c_local`, so the classifier ran without the distinct-spatial-bucket
   series it wants and fell back to the cross-sectional archive statistic.
   `boundary_action_entropy` is likewise absent.

4. **The 3072 frontier is one number from one adapter.** It is the progress
   pair's high-water mark. It is not a claim about what is on the screen,
   and this document makes no claim about what the wall *is* beyond
   `WALL_DOSSIER.md`.

5. **This registration predicts failure.** §5 pre-commits to the frontier
   staying at 3072. The value of the attempt is a crossed-off mechanism,
   not an expected win. If that is not worth 30 minutes of an M4, the
   correct action is to leave Contra shelved and record *that* — which is
   also a valid outcome of a registration.

6. **The ELIMINATE precondition is expected to be met, but not proven to
   be.** `FOOTPRINT_FLOOR = 7,427` is reachable — `poweron_to_wall` cleared
   it in 24 fresh minutes, and `stage1_v3` stood at 7,654 with only 154,896
   cells — but no banked archive records a *mid-run* footprint, so the
   footprint-versus-time curve inside 30 minutes is inferred from final
   snapshots rather than measured. If the attempt lands at the bottom of the
   projected band (~122k cells) it could finish just under the floor.

   That case is handled rather than hidden: it reads **VOID
   `under_covered`**, not ELIMINATE and not PARTIAL, and VOID is the one
   outcome this registration allows to be repaired — re-run once with more
   budget. The failure mode being avoided is the expensive one, a false
   elimination or a false shape-change; the cost of the conservative
   handling is one repeated run. If the orchestrator would rather not risk
   the repeat, the correct pre-launch amendment is to raise `--minutes`,
   which is a budget change to this section, **not** a change to
   `FOOTPRINT_FLOOR` — lowering the floor after seeing the result would be
   tuning the gate to the outcome.

---

## 9. Findings outside this task's edit scope

Reported, not fixed — both fall outside the `solve:` block.

1. **ROM hash mismatch in `configs/contra.yaml`.** The profile declares
   `rom_hashes: ["a70547ac1a4d0f86bdba6b31bf25c8aa"]`. The actual
   `roms/Contra (USA).nes` is **`7bdad8b4a7a56a634c9649d20bd3011b`**
   (whole-file MD5, which `scripts/train_game.py:rom_md5` documents as the
   convention, header included). `scripts/train_game.py` would log
   `ROM MD5 MISMATCH`; `go_explore_solve.py` does not check hashes at all,
   so every one of the ten campaigns ran against a dump the profile does
   not identify. Nothing about the results is thereby wrong — but the ten
   campaigns are not pinned to a dump, and neither would the new one be.
   Resolve before the attempt is receipted, by re-deriving the hash from the
   file in hand rather than by trusting either string.

2. **`ram_mapping.player_state: 0x0090`** carries an in-profile claim that
   `0x002C` was "the game-mode byte, not the player state". §4 refutes
   `0x0090` as a death byte on this ROM and shows `0x002C` never enters its
   death set. The `ram_mapping` block is the trainer's map, not the solver's,
   so it was left alone; it should be reconciled with §4 by whoever owns the
   training profile.

## Orchestrator verdict — Registered-attempt verdict, r1_ortho (2026-08-10, orchestrator)

30 min, seed 0: G-0 PASS (ortho engaged: 12,455 selections, 197
cols_improved), G-A FAIL (frontier ended AT gx 3072 — the 10-campaign
wall to the pixel — not past it), 52,539 cells, 0 solutions.
<!-- CORRECTION 2026-08-26: "0 solutions" here is a constant, not a
     result — configs/contra.yaml has level_key: [] so is_clear()'s
     opening test is `() > ()`, and the confluence fallback has never
     been shown to fire. The verdict rests on the gx 3072 frontier pin
     and the 52,539-cell footprint, both real. See the §1 ADDENDUM. -->
(Solution count struck 2026-08-26 — see the §1 ADDENDUM; the verdict
is unchanged, and rests on the frontier pin and the cell footprint.)
Taxonomy: **GATED** — Contra's wall is NAMED for the first time, and
it joins the CV hall and the BB 99-1 boss room in the same class.
Contra remains demoted per the registration (gate not passed); the
registered next material difference = the gate-opener arm
(docs/proposals/gate_opener_arm_2026-08-11.md), which now has three
targets in its class.

## Correction (2026-08-10): taxonomy label struck

The r1_ortho "GATED" classification is retracted per the K-FALSIFIER
(docs/receipts/dispatch/k_falsifier_2026-08-10.md — the statistic
tracks archive size). What stands, receipted: frontier AT gx 3072
(the 10-campaign wall), arm engaged (12,455 sel / 197 cols), 0
solutions. Contra's wall class: UNTYPED (as rev-4 already
re-registered it). The gate-opener discovery sweep remains the
registered next material difference on its own merits.
