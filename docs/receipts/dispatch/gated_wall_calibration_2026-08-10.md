# Gated-wall discriminator — offline calibration

**Date:** 2026-08-10 (revised 11:40 — see §0)
**Module:** `src/training/wall_taxonomy.py`
**Tests:** `tests/test_wall_taxonomy.py` (58 passing)
**Constant tag:** `CALIBRATED-OFFLINE-2026-08-10`
**Runtime status:** INERT — nothing imports the module; no dispatch is armed.
A test (`test_module_is_not_wired_into_any_runtime_dispatch`) enforces that.

No new solver runs were launched for this work. Every number below comes
from telemetry already on disk.

---

## 0. Status: CONDITIONAL — the positive class is unvalidated

> **The corpus contains no confirmed gated wall.**
>
> Both GATED rows are `lvl_03_trace` — the *same* Castlevania hall, from
> two hardware-flag lineages. One wall seen twice is not two data points,
> and that wall **has never been solved**. Nothing in `docs/` receipts it
> as gated. The only thing pointing that way is that an orthogonal arm
> (`runs/cv_hall_ortho_a`) was launched at it, which is the *conclusion*
> being tested, not evidence for it. Calibrating a discriminator on a
> label produced by the belief the discriminator is supposed to check is
> circular, and this receipt previously did exactly that when it wrote
> "one CONFIRMED gated wall" and "INDEPENDENTLY, from two separate
> campaigns."
>
> **The hall is PENDING-VALIDATION, and every band below is conditional
> on `runs/cv_hall_ortho_a` reading out.**
>
> The falsifier is named and cheap: if that arm finishes the hall by
> ordinary coverage rather than by its orthogonal mechanism — the SMB 8-4
> outcome this module's own opening paragraph cites, stuck for 44 minutes
> and then simply done — then the positive class is **empty**,
> `CONCENTRATION_GATED_MIN` has nothing to separate, and every band in §5
> collapses. In that case the surviving results are the negative ones:
> the refutations in §3, KEY_BLIND, and the no-false-positive property.
>
> Two revisions were forced by this, both recorded rather than
> retro-fitted:
>
> 1. **The live arm's numbers in this document were stale.** §2, §6 and
>    §9 asserted it had flushed no archive. It had — see §0.1.
> 2. **A calibration bracket was taken from a live run.** §5 bracketed
>    `EFFORT_MIN_STEPS` at `cv_hall_ortho_a`'s 1,010,590-step window; an
>    hour later that window read 968,490. The bracket is now SMB 8-4's
>    932,340, which is banked and cannot move.

### 0.1 The live arm HAS an archive, and it reads GATED

Read-only, single-threaded, 0.6 s per read. The arm flushed twice during
this revision, so both readings are given:

| | flush @ 11:26 | flush @ 11:41 |
|---|---|---|
| `archive.pkl` | **exists**, 2,436,606,838 B | 2,571,077,274 B |
| cells | 114,699 | 121,029 |
| `distinct_spatial` | 1,095 | — |
| `spatial_span` | 95 | 95 |
| **concentration** | **104.75** (4.2× `CONCENTRATION_GATED_MIN`) | **110.53** (4.4×) |
| verdict, with archive | **GATED** | **GATED** |
| verdict, archive stripped | indeterminate | indeterminate |

So the claim "it reads INDETERMINATE, for want of an archive" was a
transient artifact of *when* the probe ran, and "intended conservative
degradation, not a miss" was describing a state that no longer existed.
Corrected in §2, §6 and §9 below.

The verdict is stable across both flushes; the *numbers* are not, which
is the second reason this run is unfit as a calibration fixture (the
first being §0's circularity). It is excluded from `CORPUS` and from the
`LIVE` regression list, and covered instead by a single documenting test,
`test_the_live_ortho_arm_now_reads_gated_and_is_excluded`, which pins the
11:26 snapshot rather than re-reading a moving directory.

This does **not** promote the hall to confirmed. It is a *third* read of
the same wall by the same statistic, and it is the pending-validation arm
scoring itself.

---

## 1. What was being calibrated

The D2-adopted discriminator separates three walls that look identical
from the outside and demand opposite responses:

| verdict | meaning | correct response |
|---|---|---|
| **GATED** | local coverage saturated, boundary frozen | switch to an orthogonal arm |
| **BARREN** | local coverage never accumulated | fix the cell key / reset / determinism |
| **COVERAGE-LIMITED** | still expanding productively | give it more wall-clock |

Adopted form:

```
GATED  <=>  C_local SATURATED
            AND high action-entropy at the boundary
            AND zero topological transition
            AND zero permanent-map delta

BARREN <=>  C_local STAGNANT
```

---

## 2. The corpus

Every run below was already banked. `runs/` is gitignored, so the frozen
statistics are additionally replayed as synthetic telemetry inside the
test file (`CORPUS`), and the live directories are re-scored by a
skip-if-absent regression (`LIVE`).

| run | ground truth | source |
|---|---|---|
| `runs/cv_chain_hw2/lvl_03_trace` | **gated — PENDING-VALIDATION** (unsolved hall) | 89 progress records + 1.97 GB archive |
| `runs/cv_chain_hw/lvl_03_trace` | **gated — PENDING-VALIDATION** (*same* hall, earlier hw lineage) | 14 records + archive |
| `runs/cv_hall_ortho_a` | **PENDING-VALIDATION** — the arm testing the label above; not evidence for it | 56 records + 2.44 GB archive flushed 11:26 (§0.1) |
| `runs/bubble_bobble/r68_retry_ortho` | **orthogonal** — fell once the key saw x | 30 records + archive |
| `runs/bubble_bobble/r69_retry_ortho` | orthogonal | 30 records + archive |
| `runs/bubble_bobble/r68_retry_xsig` | **resolved** (the x-signature run) | 30 records + archive |
| `runs/bubble_bobble/r99_retry`, `r99_retry2` | orthogonal | 30 records + archive |
| `runs/bubble_bobble/r99_1_boss_retry` | **unknown / mechanic-gated** | 30 records + 434 MB archive |
| `runs/bubble_bobble/chain_day2h_item/lvl_00_99-1{,_s1,_s2}` | unknown / mechanic-gated | 45 records + archive each |
| `runs/live_show/smb_4_4_micro/lvl_4-4` | **momentum/coverage wall that RESOLVED** | 189 records in 5 appended segments, **no archive** |
| `runs/live_show/smb_4_4_micro/lvl_8-4` | **coverage wall that RESOLVED** | 56 records, **no archive** |
| `runs/ge_chain/lvl_11_4-4` | coverage (same level, resolved campaign) | 21 records + archive — the archive-path stand-in for 4-4 |
| `runs/ge_chain_w8/lvl_02_8-3` | coverage (resolved) | archive only — cross-sectional control nearest 8-4 |

### Fields that actually exist

`scripts/go_explore_solve.py:progress_line()` writes, per 60 s:

```
t elapsed_s cells max_area max_gx_in_max_area max_sect solutions
best_sol_actions steps sps [stall_flat_windows] [ortho_*] [pinned_secs]
[doors edges] [door_macros_injected max_room] [verify_checks verify_rejections]
```

`archive.pkl` is `dict[cell_key, Cell]` with `Cell(key, state, best_score,
best_steps, visits, times_chosen, explored)`. The key is
`(sect, tb, kk, psig, loops, route_sig) + game.cell_fn(ram)`, and every
adapter's `cell_fn` ends `(..., y // Y_BAND, progress // GX_BUCKET)` with
`area` five from the end — the layout `go_explore_solve.py` already
relies on for its own selection caches. Archives banked before the
`sect/psig` prefix carry a bare 4-tuple; the adapter handles both.

Two adapter hazards worth naming, both found here:

* **Appended segments.** A retried level appends to the *same*
  `progress.jsonl`. `lvl_4-4/progress.jsonl` holds five attempts; read as
  one series its `max_gx` is non-monotone and the cell curve saw-tooths,
  fabricating "map advanced" and "coverage collapsed" events.
  `load_progress_segments()` splits on `elapsed_s` going backwards.
* **Show runs persist no archive.** `runs/live_show/*` sets
  `flush_secs` to ~forever, so the two ground-truth coverage walls have
  progress lines only. Three SMB archives that *do* exist under
  `smb_4_4_micro` (`lvl_1-3`, `lvl_3-3`, `lvl_4-3`, 3-4.5 GB each) are
  **truncated** — `pickle.load` raises `EOFError: Ran out of input` — so
  they contributed nothing.

---

## 3. Headline result: the obvious statistic is wrong

The natural reading of "C_local plateau" is *archive cell count stops
growing*. **It does not separate the corpus.** Raw-cell saturation
(1 − trailing yield / best sustained yield, both per emulator step),
trailing 10 records:

| run | ground truth | raw-cell saturation |
|---|---|---|
| `cv_chain_hw/lvl_03_trace` | **gated** | **0.179** |
| `smb 8-4` | coverage, resolved | 0.189 |
| `cv_chain_hw2/lvl_03_trace` | **gated** | **0.343** |
| `smb 4-4` seg4 | coverage, resolved | 0.352 |

The two gated runs *straddle* the two resolved ones in both directions.
No threshold works, in either sense.

**Why:** the cell key carries nuisance dimensions (`phase`, `vsign`,
`route_sig`, `loops`, boss HP, state-signature bits) that manufacture
novelty forever at a fixed location. The Castlevania hall grew from
3,354 to 91,995 cells over 89 minutes while `max_gx_in_max_area` sat at
exactly 767 for 84 straight records. Cell count is not C_local.

Ships as `RAW_COVERAGE_SATURATION_IS_SEPARATING = False`, reported in
every verdict's `evidence` so nobody re-derives it and believes it.

Two more candidates refuted the same way:

* **Per-window churn** (new cells / archive size). Inverted against
  ground truth: BB 99-1 day-2h (0.00024, a live 2,989-cell archive)
  churns an order of magnitude *less* than BB r69 ortho (0.01667, a
  frozen 48-cell archive that ticked once). `CHURN_IS_SEPARATING = False`.
* **Boundary visit entropy** — normalized Shannon entropy of visit mass
  across cells in the deepest bucket, the only offline stand-in for the
  adopted form's action-entropy term. Every class scores ≥ 0.77 (gated CV
  hall 0.9837, barren BB r68 ortho 0.7775, resolved SMB 1-4 0.9999). It
  measures how evenly *returns* were spread, not how varied the *actions*
  were. `BOUNDARY_ENTROPY_IS_SEPARATING = False`.
* **Map-stall length** (consecutive records at zero `max_gx` delta) does
  separate on this corpus — gated hall 84, SMB 8-4 44, SMB 4-4 25 — but
  it is a pure function of how long the run was left alive: a longer 8-4
  crosses any fixed threshold. Deliberately not shipped as a gate;
  reported so a reader can see the horizon a verdict was taken over.
  `MAP_STALL_WINDOWS_IS_SEPARATING = False`.

---

## 4. What does separate: spatial C_local

C_local is **coverage of the map footprint**, not of the archive. From
the cell key's spatial projection `(area, y_band, gx_bucket)`:

```
distinct_spatial = |{(area, y_band, gx_bucket)}|
concentration    = cells / distinct_spatial
spatial_span     = distinct gx buckets inside the deepest area
```

`concentration` is the cross-sectional stand-in for "C_local has
plateaued": an archive that keeps multiplying inside a map footprint that
stopped growing is saturated locally by definition.

| archive | ground truth | cells | distinct_spatial | **concentration** | spatial_span | boundary cells | boundary entropy |
|---|---|---|---|---|---|---|---|
| `cv_chain_hw2/lvl_03_trace` | **gated** | 92,785 | 1,089 | **85.20** | 95 | 13 | 0.9837 |
| `cv_chain_hw/lvl_03_trace` | **gated** | 28,929 | 932 | **31.04** | 94 | 5 | 0.9697 |
| `ge_chain_w8/lvl_02_8-3` | resolved | 19,958 | 970 | 20.58 | 223 | 16 | 0.9629 |
| `ge_chain/lvl_11_4-4` | resolved | 5,885 | 750 | 7.85 | 130 | 8 | 0.9728 |
| `ge_1_4_solve` | resolved | 4,815 | 638 | 7.55 | 152 | 8 | 0.9999 |
| `ge_1_2_solve` | resolved | 8,180 | 1,112 | 7.36 | 204 | 8 | 1.0000 |
| `ge_chain/lvl_02_2-3` | resolved | 5,662 | 1,178 | 4.81 | 230 | 8 | 0.9997 |
| `cv_chain_hw/lvl_02` | solved CV block | 1,202 | 902 | 1.33 | 80 | 1 | 0.0 |
| `bubble_bobble/r99_1_boss_retry` | unknown | 9,475 | 32 | 296.09 | **1** | 9,475 | 0.8991 |
| `bubble_bobble/…/lvl_00_99-1` | unknown | 2,989 | 16 | 186.81 | **1** | 2,989 | 0.8761 |
| `bubble_bobble/r99_retry2` | orthogonal | 691 | 16 | 43.19 | **1** | 691 | 0.8583 |
| `bubble_bobble/r68_retry_xsig` | resolved | 605 | 16 | 37.81 | **1** | 605 | 0.8253 |
| `bubble_bobble/r68_retry_ortho` | orthogonal | 96 | 16 | 6.00 | **1** | 96 | 0.7775 |

Two things fall out.

**(a) Concentration separates gated from coverage-limited** — but only
for spatially resolved games, and by the module's thinnest margin:
20.58 (SMB 8-3, resolved) .. 31.04 (CV hall hw, gated). Total spread
1.51×; ~1.2× either side of the shipped 25.0.

**(b) Bubble Bobble pins `spatial_span` at 1 for every run.** The game is
one screen: its progress byte never moves, so the "zero permanent-map
delta" term of the adopted form carries no information there, and any
coverage statistic over the key's spatial projection is meaningless. That
is not a nuisance — it *is* the r68 finding. The wall fell the moment x
entered the cell key. So a degenerate spatial projection gets its own
verdict, **KEY_BLIND**, which names the missing axis instead of guessing
at gated-vs-barren over a blind key.

---

## 5. Shipped constants and their separating bands

Every constant sits inside a band measured by re-scoring the corpus while
sweeping it. `test_shipped_constants_sit_inside_their_measured_separating_bands`
asserts the bands, so moving one out fails loudly.

| constant | value | separating band (sweep-verified) | brackets |
|---|---|---|---|
| `CONCENTRATION_GATED_MIN` | **25.0** | (20.58, 31.04] | SMB 8-3 resolved ↔ CV hall hw *(unvalidated)* |
| `SPATIAL_SPAN_MIN` | **8** | [2, 94] | every BB run (1) ↔ CV hall hw (94) |
| `C_LOCAL_FLOOR_BUCKETS` | **64** | (32, 638] | BB 99-1 boss retry (32) ↔ `ge_1_4_solve` (638) |
| `EFFORT_MIN_STEPS` | **250,000** | (88,680, 932,340] | starved SMB 4-4 seg0 ↔ SMB 8-4's window |
| `COVERAGE_FLOOR_CELLS` | **256** | [97, 606] | BB r68 ortho (96) ↔ BB r68 xsig (605) |
| `FROZEN_WINDOWS_MAX` | **12** | [8, 19] | BB r99 retry alive (7) ↔ BB r69 ortho frozen (19) |
| `WINDOW_RECORDS` | **10** | [5, 20] | flat across the whole sweep |
| `MIN_RECORDS` | **12** | [11, 14] | window+1 ↔ CV hall hw's 14 records |
| `C_LOCAL_SATURATION_MIN` | 0.85 | **PROVISIONAL** — no banked run emits a C_local series | — |
| `C_LOCAL_SERIES_MAY_CERTIFY_GATED` | **False** | structural consequence of the row above | — |

`C_LOCAL_FLOOR_BUCKETS` is the series-path twin of `SPATIAL_SPAN_MIN`,
and it is measured on the same column: `c_local` is *defined* as
`|{(area, y_band, gx_bucket)}|`, which is the `distinct_spatial` column
of the §4 table. The band there is wide — every degenerate run sits at
8–32, every spatially resolved one at 638–1,178 — so 64 is deliberately
placed 2× above the worst degenerate case rather than at the midpoint.

`C_LOCAL_SERIES_MAY_CERTIFY_GATED = False` is not a tuning knob but the
honest consequence of the line above it: a threshold that has never been
measured against a labelled run must not be able to produce, on its own,
the one verdict that costs an orthogonal campaign. See §6.1.

Sweep excerpts (a row is OK only if **all 15** scored cases keep their
ground-truth verdict):

```
CONCENTRATION_GATED_MIN  8.0 OK | 20.0 OK | 25.0 OK | 31.0 OK | 31.1 no (CV hall hw -> coverage_limited) | 90.0 no (both halls lost)
SPATIAL_SPAN_MIN           1 no (BB r99/99-1 -> GATED, false positives) | 2..94 OK | 95 no (CV hall hw -> key_blind)
FROZEN_WINDOWS_MAX         7 no (BB r99 retry -> false barren) | 8..19 OK | 20 no (BB r69 ortho missed)
COVERAGE_FLOOR_CELLS      96 no (BB r68 ortho missed) | 97..606 OK | 700 no (BB r99 retry/retry2 -> false barren)
MIN_RECORDS               11..14 OK | 15 no (CV hall hw -> insufficient)
EFFORT_MIN_STEPS          50k..933k OK | 1.10M no (cv_hall_ortho_a, BB r69, BB 99-1 -> insufficient)
```

`EFFORT_MIN_STEPS`'s *upper* bracket is the smallest trailing window in a
**banked** case that must still be classified: SMB 8-4 at 932,340 steps.
Its *lower* bracket is a design choice — SMB 4-4 seg0's 88,680-step
window is a throughput-starved show segment and must not be classified at
all. The effort test sits after the RESOLVED/PROGRESSING tests, so a short
run that is plainly still working is never blocked by it.

> **Correction.** This bracket previously read 1,010,590, taken from
> `cv_hall_ortho_a` while that run was still going. Re-measured an hour
> later the same window read **968,490**. A live run cannot bracket a
> frozen constant — the number it contributes depends on when you looked.
> Every bracket in the table above now comes from a finished run.
> `test_shipped_constants_sit_inside_their_measured_separating_bands`
> asserts the corrected bound and records why.

Note the `SPATIAL_SPAN_MIN = 1` row: without the key-blind gate, four
Bubble Bobble runs read **GATED** on concentration alone. The gate is
what keeps the module from certifying a wall over a blind key.

The concentration band from *scored* cases is wider, (7.85, 31.04], since
SMB 8-3 has only one progress record and enters the calibration as a
cross-sectional control. 25.0 is chosen to hold against 8-3 as well —
roughly the geometric midpoint of 20.58 and 31.04 (25.3).

---

## 6. Decision order and the full corpus result

```
1. records < MIN_RECORDS                       -> INSUFFICIENT
2. a solution landed                           -> RESOLVED
3. topo_delta > 0 or map_delta > 0             -> PROGRESSING
4. window_steps < EFFORT_MIN_STEPS             -> INSUFFICIENT
5. spatial_span < SPATIAL_SPAN_MIN
     or c_local < C_LOCAL_FLOOR_BUCKETS        -> KEY_BLIND
6. frozen_windows >= MAX, or cells < FLOOR,
     or C_local peak yield == 0 (STAGNANT)     -> BARREN
7. C_local saturation < MIN (still expanding)  -> COVERAGE_LIMITED
8. C_local plateau AND concentration >= MIN    -> GATED
   C_local plateau, uncorroborated             -> INDETERMINATE
9. no C_local series and no archive            -> INDETERMINATE
10. concentration >= MIN / otherwise           -> GATED / COVERAGE_LIMITED
```

KEY_BLIND precedes BARREN deliberately: both point at the cell key, but a
degenerate spatial projection names *which* axis is missing, and a frozen
archive in a spatially blind game is frozen *because* of that blindness.

### 6.1 The C_local series path, and why it is symmetric with the archive

Steps 5–7 apply the *same* tests to the C_local series and to the archive
snapshot. That symmetry is load-bearing: **a run must not change verdict
merely because the solver started emitting a field.** Three properties
make that concrete, and all three were violated by the first
implementation:

* **STAGNANT ≠ plateaued.** `saturation()` returned `1.0` when the peak
  yield was zero, i.e. when the series *never grew*. That made a series
  pinned at a constant score as maximally saturated and read **GATED** —
  for the exact case the module's own adopted form defines as BARREN
  (`BARREN <=> C_local STAGNANT`). `saturation()` now returns `None`
  there, `series_yields()` exposes the peak so the two states stay
  distinguishable, and step 6 names it BARREN.
* **The key-blindness guard lived only on the archive.** It required
  `archive is not None`, so a span-degenerate run — *every* Bubble Bobble
  profile — that emitted `c_local` before flushing an archive read
  **GATED** where the identical run with an archive read KEY_BLIND.
  Measured, both directions, before the fix. `C_LOCAL_FLOOR_BUCKETS`
  closes it. Below `SPATIAL_SPAN_MIN` the series does more than agree: it
  *proves* the archive guard, since `spatial_span <= c_local` by
  construction.
* **A PROVISIONAL threshold may not manufacture a positive.** The series
  can move a verdict *away* from GATED on its own (step 7 — a still
  expanding footprint overrides even a gated concentration, and that
  direction cannot cost a campaign). To move one *toward* GATED it needs
  the cross-sectional concentration to agree; otherwise the answer is
  INDETERMINATE, whose remedy — "collect the missing telemetry" — is
  precisely right for a threshold nobody has measured. Flipping
  `C_LOCAL_SERIES_MAY_CERTIFY_GATED` to `True` is the one-line promotion,
  and it belongs in the same commit as the receipt that calibrates
  `C_LOCAL_SATURATION_MIN`.

`tests/test_wall_taxonomy.py` §2b covers this world end to end, including
a parity property (`test_emitting_c_local_never_upgrades_a_banked_run_to_gated`)
that re-scores every corpus run under all three C_local shapes, with and
without its archive, and fails if any of them invents a GATED verdict the
archive path did not already give.

Full run (`concentration`/`span` blank where no archive exists):

Re-derived 2026-08-10 11:45 from the real run directories, post-fix:

```
case                     ground truth         verdict            deg      cells    conc  span  frz  rawsat mapstall
CV hall lvl_03 (hw2)     gated PENDING-VAL    gated              True      91995  85.202    95    0   0.343       84
CV hall lvl_03 (hw)      gated PENDING-VAL    gated              True      27619  31.040    94    0   0.179       11
CV hall ortho_a live     PENDING-VALIDATION   gated              True     123216 110.529    95    0   0.421       55
BB r68 ortho             ORTHOGONAL/key       key_blind          False        96   6.000     1   29       -       29
BB r69 ortho             ORTHOGONAL/key       key_blind          False        48   6.000     1   19   0.010       29
BB r68 xsig              RESOLVED             resolved           False       605  37.812     1    4   0.834       29
BB r99 retry             ORTHOGONAL/key       key_blind          False       639  39.938     1    7   0.950       29
BB r99 retry2            ORTHOGONAL/key       key_blind          False       691  43.188     1    4   0.933       29
BB 99-1 boss retry       UNKNOWN/mechanic     key_blind          False      9475 296.094     1    0   0.941       29
BB 99-1 d2h              UNKNOWN/mechanic     key_blind          False      2989 186.812     1    1   0.987       44
BB 99-1 d2h_s1           UNKNOWN/mechanic     key_blind          False      2985 186.562     1    1   0.993       44
BB 99-1 d2h_s2           UNKNOWN/mechanic     key_blind          False      2967 185.438     1    3   0.994       44
SMB 4-4 seg0             COVERAGE (resolved)  insufficient       True     719840       -     -    0   0.263       41
SMB 4-4 seg1             COVERAGE (resolved)  indeterminate      True    1020500       -     -    0   0.377       41
SMB 4-4 seg2             COVERAGE (resolved)  indeterminate      True    1058555       -     -    0   0.364       40
SMB 4-4 seg4             COVERAGE (resolved)  progressing        False   1164599       -     -    0   0.352        0
SMB 8-4                  COVERAGE (resolved)  progressing        False   1190873       -     -    0   0.189        0
SMB 4-4 (ge_chain)       COVERAGE (resolved)  coverage_limited   True       5885   7.847   130    0   0.736       12
```

Two cells changed against the pre-fix printing of this table, and both
are corrections rather than drift:

* **`CV hall ortho_a live` was `indeterminate`; it is `gated`.** The
  earlier row was taken before the arm had flushed an archive (§0.1).
* **`BB r68 ortho`'s `rawsat` was `1.000`; it is now `-`.** Its cell
  count never moved at all, so there is no peak yield to normalize
  against. `1.000` read as "maximally saturated" for an archive that
  never accumulated a thing — the same conflation that produced the
  GATED-for-STAGNANT bug in §6.1.

**No false GATED anywhere**, in the full path or the degraded one
(`test_no_false_gated_anywhere_in_the_corpus` re-scores every case twice,
with and without its archive, and asserts the GATED set is exactly the
two Castlevania hall runs).

This is the strongest claim the corpus supports, and it is a *negative*
one: the module does not cry wolf on any run whose ground truth is known.
It is not evidence that the module can find a wolf. Every run it has ever
called GATED is the same unsolved wall.

Reading the rows:

* **The two hall runs are the only positives, and they are not
  independent.** Both are `lvl_03_trace`: the *same* Castlevania hall,
  from two hardware-flag lineages, at concentrations 31.04 and 85.20.
  The second is that wall left running 6× longer; concentration climbs
  with time while `distinct_spatial` grows only 932 → 1,089 (+17%) as
  cells go 28,929 → 92,785 (3.2×). That ratio is the *candidate* gated
  signature — a shape measured twice on one unsolved wall, not a class
  with two members. An earlier draft of this bullet said
  "INDEPENDENTLY, from two separate campaigns"; separate campaigns are
  not separate walls. See §0.
* **`cv_hall_ortho_a` — the running orthogonal arm — reads GATED**, at
  concentration 104.75–110.53 across two flushes (§0.1). An earlier
  draft claimed it read INDETERMINATE "because it has not flushed an
  archive yet" and called that intended conservative degradation; the
  archive existed when that was written. The corrected reading is *not*
  a third positive: it is the arm under test scoring the wall it was
  launched at, and it is excluded from the calibration for that reason.
* **SMB 4-4 seg1/seg2 read INDETERMINATE**, which is the whole point:
  mid-run they are statistically indistinguishable from the hall on
  progress lines alone, and the module abstains instead of firing an
  orthogonal arm at a wall that was going to fall on its own.
* **SMB 4-4 seg4 and 8-4 read PROGRESSING** — both broke through inside
  the final window (`map_delta` 520 and 446). The clear itself lands
  between progress lines, so `solutions` is still 0 on the last record;
  the map delta is what catches it.
* **SMB 4-4 seg0 reads INSUFFICIENT** — 88,680 steps in its trailing
  window. A paced show segment starved of throughput is not evidence
  about a wall.
* **BB 99-1 (four runs, ground truth unknown) reads KEY_BLIND.** Honest:
  the key sees no spatial axis, so the module declines to call it gated
  or barren and names the fix instead. Reported, not resolved.

---

## 7. Degraded path

Stripping the archive from every case:

```
CV hall lvl_03 (hw2)  -> indeterminate      BB r99 retry      -> indeterminate
CV hall lvl_03 (hw)   -> indeterminate      BB r99 retry2     -> indeterminate
CV hall ortho_a live  -> indeterminate      BB 99-1 boss      -> indeterminate
BB r68 ortho          -> barren             BB 99-1 d2h       -> indeterminate
BB r69 ortho          -> barren             SMB 4-4 seg1/2    -> indeterminate
BB r68 xsig           -> resolved           SMB 4-4 seg4, 8-4 -> progressing
SMB 4-4 seg0          -> insufficient       ge_chain 4-4      -> indeterminate
```

Progress-only telemetry can still catch RESOLVED, PROGRESSING and BARREN
(the two frozen archives fall out via `FROZEN_WINDOWS_MAX` *and*
independently via `COVERAGE_FLOOR_CELLS` — two unrelated confirmations),
but it **never certifies GATED**.

That property is now enforced as a property. It previously rested on
`test_progress_only_telemetry_never_certifies_gated` exercising a single
hand-picked telemetry — one that happened to carry no `c_local`, so the
test could not have caught the c_local gate that *did* certify GATED
without an archive (§6.1). The test now crosses **every** corpus run with
**every** C_local shape — 13 × 4 = 52 combinations — and asserts that
none of them reaches GATED without an archive.

---

## 8. What a runtime version needs the solver to emit

Ranked by lift. All four are additions to `progress_line()`.

1. **`c_local`** — `len({(area, y_band, gx_bucket)})` over the archive,
   per progress line. This is the single missing field. It converts the
   cross-sectional concentration proxy (one number, thinnest margin in
   the module) into the actual adopted statistic (a curve, on which a
   plateau is visible), and it removes the dependency on a flushed
   multi-GB archive — which the show runs never write. Cost: one set
   maintained alongside the archive, or one pass over keys at flush
   cadence.

   **Landing the field is not the whole job, and the order matters.**
   `C_LOCAL_SATURATION_MIN` is PROVISIONAL, so until it is re-derived
   against a corpus that actually contains a validated positive, the
   series may not certify GATED on its own —
   `C_LOCAL_SERIES_MAY_CERTIFY_GATED` stays `False` and an
   uncorroborated plateau degrades to INDETERMINATE (§6.1). Emitting
   `c_local` into the earlier build would have been actively harmful:
   the field alone flipped span-degenerate runs from INDETERMINATE to
   GATED, and pinned series from BARREN to GATED. The promotion
   sequence is (a) emit the field, (b) bank runs with labelled
   outcomes, (c) re-derive the threshold and publish the band,
   (d) flip the switch — in that order, in the commit that measures it.
2. **`boundary_action_entropy`** — Shannon entropy of the action
   distribution actually sampled from cells in the frontier bucket. The
   adopted form's second term is *currently unmeasured*: nothing banked
   records actions per cell, and the visit-mass surrogate was refuted
   (§3). Until this exists every GATED verdict carries `degraded=True`
   and lists it in `missing`.
3. **`frontier_bucket_cells`** — cells in the deepest reachable bucket.
   Available only from an archive snapshot today, and the two ground-truth
   coverage walls persisted no archive at all.
4. **A game-neutral permanent-map counter.** `max_gx_in_max_area` is the
   current stand-in and it is game-shaped: every Bubble Bobble run pins
   it at a constant, so the "zero permanent-map delta" term is vacuous
   there and KEY_BLIND has to carry the whole decision.

Operational fix worth pairing with (1): **let show runs flush an archive.**
`runs/live_show/*` sets `flush_secs` to ~forever, and the three SMB show
archives that do exist are truncated. The two best coverage-wall
ground truths in the corpus could only be scored on the degraded path.

---

## 9. Scope limits

* **18 scored runs (15 in the threshold sweeps), ZERO confirmed gated
  walls.** The positive class is one *candidate* — the Castlevania hall
  — scored three times (`cv_chain_hw`, `cv_chain_hw2`, and the live
  `cv_hall_ortho_a` arm). Same level, same game, three reads; that is
  one unvalidated data point, not three. An earlier draft of this bullet
  said "one CONFIRMED gated wall"; nothing confirms it. The hall has
  never been solved, and no receipt in `docs/` labels it gated — the
  label's entire provenance is that an orthogonal arm was launched at
  it, which is the hypothesis, not a test of it.
  `CONCENTRATION_GATED_MIN` rests on a 1.51×-wide band whose upper
  bracket is that unvalidated wall.
* **4-4 and 8-4 have no archives**, so the archive path was calibrated
  against `ge_chain/lvl_11_4-4` (same level, resolved campaign) and
  `ge_chain_w8/lvl_02_8-3` (nearest neighbour to 8-4). Both substitutions
  are marked in the tables; neither is 4-4 or 8-4 itself.
* **BB 99-1's ground truth is unknown.** It is reported, never used to
  fit a threshold.
* **`cv_hall_ortho_a` was live while this was written, and the whole
  calibration is conditional on how it reads out.** Its numbers move
  between reads — cells 105,913 → 112,148 → 123,216, archive
  concentration 104.75 → 110.53 within fifteen minutes — and it reads
  **GATED**, not the INDETERMINATE an earlier draft recorded.

  **The falsifier.** If that arm finishes the hall by ordinary coverage
  rather than by its orthogonal mechanism — the SMB 8-4 outcome cited in
  the module's opening paragraph, stuck for 44 minutes and then simply
  done — then the hall was never gated, the positive class is empty,
  `CONCENTRATION_GATED_MIN` has nothing to separate, and §5's bands
  collapse. What survives that outcome is the negative half of this
  work: the three refuted statistics in §3, the KEY_BLIND class (which
  rests on Bubble Bobble ground truth, not on the hall), the adapter
  hazards in §2, and the no-false-positive property in §6. What does not
  survive is any claim that gated walls are detectable at all.

  A "PENDING-VALIDATION" label is therefore not a formality here. Until
  Run A reads out, this receipt calibrates a discriminator against a
  hypothesis, and the honest summary is: *we can reliably say when a
  wall is NOT gated; whether we can say when one IS remains untested.*
* **`concentration` is geometry-dependent.** It compares meaningfully
  within a spatial-resolution class, which is exactly why
  `SPATIAL_SPAN_MIN` gates it. A game with a coarse `GX_BUCKET` or an
  unusually verbose key will shift the scale; re-derive per profile
  family before trusting the number on a new title.
* **Nothing here is armed.** Self-arming dispatch is a separate decision
  per the D2 verdict. This module is a statistic and a receipt.
