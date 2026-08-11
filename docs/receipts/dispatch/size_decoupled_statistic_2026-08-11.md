# Size-decoupled wall classification — can anything replace `concentration`?

**Date:** 2026-08-11
**Answers:** the open question left by `docs/receipts/dispatch/k_falsifier_2026-08-10.md`
(§0: the wall class is relabelled `UNRESOLVED-CONCENTRATED` *"pending a size-decoupled
statistic"*).
**Module under test:** `src/training/wall_taxonomy.py` — **not modified.** Integration
spec in §12 instead.
**Prototype:** `/tmp/wall_stat_proto.py`
**Compute:** offline, read-only. 19 archives unpickled (34.8 GiB, 7.1 s), 276
`archive.stats.json` sidecars, 276 `progress.jsonl`. No solver run, no build, no commit,
nothing written to `runs/`.

---

## 0. VERDICT — **NONE SURVIVES. Classification stays struck.**

> 22 candidate statistics were built and scored against the K-falsifier's own corpus and
> ground truth, in **both** directions. **Every one fails.** 15 of the 22 fail in the
> strongest possible way — *straddling*, with resolved archives on both sides of the
> hall's range, so no threshold works in either direction. The best cut any candidate
> admits still condemns **3 of 13** resolved archives as walls; the shipped statistic
> condemns 4. That is not a replacement.
>
> The `GATED` verdict has no statistic behind it, and this receipt found none. Per the
> K-falsifier's FAIL branch the vocabulary stays struck; §12 recommends **removing** the
> `GATED` branch rather than re-thresholding it.

Two findings are positive and worth banking anyway:

1. **The size confound is fixable, and fixing it does not help.** Candidates (a) and (f)
   drive Spearman(statistic, `cells`) from the shipped **+0.940** down to **+0.074** and
   **−0.019**. They are genuinely size-decoupled — and they still do not separate. So
   §5.1 of the K-falsifier diagnosed a *real* defect that was not the *binding* one. The
   binding constraint is that the hall and a resolved search are not distinguishable by
   any cross-sectional archive statistic available offline.
2. **A same-game control kills the best candidate outright.** Novelty-per-record — the
   strongest new construction, and the only one that passes the same-chain sibling test —
   is refuted by **solved Castlevania blocks**, in the same game, on the same grid, at the
   same scale as the hall (§10).

---

## 1. What was asked

Prototype replacements for the struck `concentration` statistic and test them against the
same corpus and ground truth. A candidate passes only if it:

* separates **all** resolved-class archives from **all** five CV-hall reads,
* with a reported margin and no threshold gerrymandering,
* survives the same-chain `ge_chain_w8` sibling test (15.66 / 20.58 / 98.30 — one chain,
  one grid, one campaign, all three solved), and
* is computable from telemetry that already exists on disk.

Registered candidate families: (a) size-partialed concentration, (b) per-step normalized
boundary statistics *(including: understand why `explored_fraction` and
`boundary_visit_entropy` failed the K-falsifier's admissible check, and design around it)*,
(c) effort-matched percentile, (d) growth-curve shape, (e) the `boundary_axis_profile`
interaction-blindness profile — plus anything else devised along the way. Two were
devised: **(f)** the shape of the archive's own spatial mass distribution, and the
**novelty-per-record** family in (b).

---

## 2. Method, and one methodological correction to the falsifier

Everything is computed through the shipped module's own API —
`_StateDroppingUnpickler`, `summarize_archive_cells`, `_spatial_key`,
`boundary_axis_profile`, `load_progress_segments`, `WINDOW_RECORDS` — so a candidate that
had passed would have been implementable without new plumbing.

**Reproduction check first.** All 19 archives were re-read from scratch. Every
`concentration` reproduces the K-falsifier's tables to the digit: 508.54 / 136.34 / 120.04
/ 85.20 / 31.04 (hall), 270.03 / 240.19 / 149.63 / 98.30 (registered four), 15.66 / 20.58
(siblings), 7.85 / 7.55 / 7.36 / 4.81 / 4.31 / 1.33 / 1.06, 43.60 (Contra). Read time
7.1 s for 34.8 GiB, matching the falsifier's 7.8 s / 37.7 GiB.

### 2.1 The falsifier's PASS clause is direction-locked

§7 of the campaign doc defines PASS as *"some statistic ranks the CV hall **ABOVE** all
four"*. A statistic that puts the hall uniformly **below** the resolved class separates
exactly as well — the sign is a labelling convention, not evidence. The K-falsifier's §9
scan correctly noted several statistics were *"inverted"* and scored them `no`, which
under a direction-free reading would have been a candidate PASS.

**This gate tests both directions.** For every candidate it computes
`min(hall) − max(resolved)` *and* `min(resolved) − max(hall)` and reports whichever is
positive. **Neither is ever positive.** The K-falsifier's FAIL verdict therefore survives
the repair of its own PASS clause — a stronger result than it claimed.

### 2.2 `archive.stats.json` — the effort denominator is free

Every banked archive (276 of them) carries a ~130-byte sidecar:

```json
{"cells": 28929, "frontier": 10184, "best_score": 767,
 "records": 2554439, "new_cells": 28929, "improvements": 44153}
```

Verified against direct archive reads on three runs, exactly, no rounding:

| field | equals | checked on |
|---|---|---|
| `records` | `Σ visits` over every cell (`GoExploreArchive.total_records`) | `ge_1_2_solve` 762,290 · `ge_1_4_solve` 771,291 · `cv_chain_hw/lvl_02` 140,596 |
| `frontier` | `cells × (1 − explored_fraction)` — the unexplored count | same three |
| `new_cells` | `cells` for every banked archive | all 276 |

`Cell.visits` starts at 1 and increments on every re-record (`go_explore.py:88,167`), so
`Σ visits ≡ total_records` by construction. This is the same column the K-falsifier's §5
effort table calls **"archive records"** — it was reading this file. Consequence: **every
effort-normalized statistic in this receipt is computable without opening a multi-GB
pickle**, which is why the held-out sweep in §10 could cover 84 archives instead of 4.

It also makes the falsifier's §7 segment-pairing hazard automatic, and confirms its
finding: `ge_chain_w8/lvl_00_8-1`'s 8,269,310 "total steps" is a cross-segment figure
while the archive belongs to the 4-record retry that recorded 731,234. The
archive-intrinsic denominator has no such ambiguity. (`/tmp/wall_stat_proto.py` also
automates the §7 audit for progress-derived candidates: pair the archive with the segment
whose peak cell count is nearest the archive's own.)

---

## 3. Corpus and ground truth

**Positive class — 5 reads, one wall.** `cv_chain_hw/lvl_03_trace`,
`cv_chain_hw2/lvl_03_trace`, `cv_chain_hw/lvl_03_overnight`, `cv_hall_ortho_a`,
`cv_hall_ortho_ctrl`. All five bank zero solutions. Unchanged from the falsifier.

**Resolved class — 13 archives**, the falsifier's registered four plus the same-chain
siblings plus every spatially-resolved resolved row from the calibration corpus.

**Excluded, legitimately:** the Bubble Bobble family. Every BB profile pins
`spatial_span` at 1, and `SPATIAL_SPAN_MIN` / `KEY_BLIND` fires at decision step 5 —
before any concentration test. That gate rests on Bubble Bobble ground truth (the r68
x-signature finding), not on the hall, and the K-falsifier explicitly left it standing.

**The exclusion is not doing work for the candidates.** All 13 resolved archives clear
every gate that precedes the concentration branch — `COVERAGE_FLOOR_CELLS = 256`,
`SPATIAL_SPAN_MIN = 8`, `C_LOCAL_FLOOR_BUCKETS = 64` — verified per row. Nothing in §4-§11
is rescued or condemned by a pre-gate.

**One label caveat, stated rather than smoothed.** `runs/ge_chain/lvl_11_4-4/solutions/`
is **empty**. The calibration receipt labels it *"coverage (same level, resolved
campaign)"* — the label is inherited from a campaign, not from a solution receipt in that
directory. It is scored here as the calibration scores it, and every table that turns on
it says so. Dropping it improves the best candidate from 3/13 false GATED to 2/12; it
changes no verdict.

---

## 4. TABLE A — the primary corpus, every candidate value

`ds` = `distinct_spatial`, `spn` = `spatial_span`, `conc` = shipped concentration,
`ν` = `cells / records`, `spcR` = size-partialed residual (a), `evenV` = visit-mass
evenness (f), `n/sel` = new cells per return-selection (b), `expR` = explored-fraction
residual against the coupon null (b), `axes` = `live_state_axis_count` (e).

| run | class | cells | ds | spn | records | conc | ν | spcR | evenV | n/sel | expR | axes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cv_chain_hw/lvl_03_overnight` | **HALL** | 560,410 | 1102 | 95 | 45,640,527 | 508.54 | 0.01228 | +0.120 | 0.133 | 0.737 | −0.0665 | 3 |
| `cv_hall_ortho_ctrl` | **HALL** | 149,153 | 1094 | 95 | 9,052,551 | 136.34 | 0.01648 | +0.025 | 0.193 | 0.420 | −0.0519 | 1 |
| `cv_hall_ortho_a` | **HALL** | 131,561 | 1096 | 95 | 8,902,418 | 120.04 | 0.01478 | +0.013 | 0.211 | 0.381 | −0.0643 | 1 |
| `cv_chain_hw2/lvl_03_trace` | **HALL** | 92,785 | 1089 | 95 | 10,656,920 | 85.20 | 0.00871 | −0.008 | 0.167 | 0.484 | −0.0398 | 1 |
| `cv_chain_hw/lvl_03_trace` | **HALL** | 28,929 | 932 | 94 | 2,554,439 | 31.04 | 0.01132 | +0.058 | 0.115 | 0.778 | −0.0756 | 1 |
| `ge_chain_w8/lvl_00_8-1` | resolved | 218,218 | 2220 | 383 | 731,234 | 98.30 | 0.29842 | −0.654 | 0.412 | 27.714 | −0.0253 | 2 |
| `live_show/smb_4_4_micro/lvl_1-3` | resolved | 210,648 | 877 | 115 | 1,742,758 | 240.19 | 0.12087 | +0.272 | 0.628 | 5.030 | −0.0957 | 3 |
| `lost_levels/ll_1_1_transfer` | resolved | 166,683 | 1114 | 195 | 2,126,825 | 149.63 | 0.07837 | +0.015 | 0.043 | 14.200 | −0.0436 | 2 |
| `live_show/smb_4_4_micro/lvl_3-3` | resolved | 141,764 | 525 | 73 | 788,717 | 270.03 | 0.17974 | +0.755 | 0.628 | 7.467 | −0.0322 | 4 |
| `contra…/r1_ortho` | *UNTYPED* | 52,539 | 1205 | 193 | 3,143,695 | 43.60 | 0.01671 | −0.153 | 0.154 | 0.701 | −0.2232 | 2 |
| `ge_chain_w8/lvl_02_8-3` | resolved | 19,958 | 970 | 223 | 43,347 | 20.58 | 0.46042 | −0.011 | 0.451 | 48.208 | −0.0059 | 2 |
| `ge_chain_w8/lvl_01_8-2` | resolved | 19,330 | 1234 | 223 | 45,286 | 15.66 | 0.42684 | −0.254 | 0.507 | 40.104 | −0.0082 | 2 |
| `ge_1_2_solve` | resolved | 8,180 | 1112 | 204 | 762,290 | 7.36 | **0.01073** | −0.216 | 0.032 | 0.916 | −0.0371 | 1 |
| `cv_chain_hw/lvl_01` | resolved | 5,936 | 1376 | 207 | 125,991 | 4.31 | 0.04711 | −0.454 | 0.280 | 2.937 | −0.0426 | 1 |
| `ge_chain/lvl_11_4-4` ¹ | resolved | 5,885 | 750 | 130 | 3,319,567 | 7.85 | **0.00177** | +0.152 | 0.556 | 0.098 | −0.0013 | 1 |
| `ge_chain/lvl_02_2-3` | resolved | 5,662 | 1178 | 230 | 46,418 | 4.81 | 0.12198 | −0.302 | 0.440 | 6.114 | −0.0066 | 1 |
| `ge_1_4_solve` | resolved | 4,815 | 638 | 152 | 771,291 | 7.55 | **0.00624** | +0.298 | 0.054 | 0.415 | +0.0225 | 1 |
| `cv_chain_hw/lvl_02` | resolved | 1,202 | 902 | 80 | 140,596 | 1.33 | **0.00855** | −0.155 | 0.204 | 0.568 | −0.0750 | 1 |
| `cv_chain_hw/lvl_00` | resolved | 460 | 435 | 85 | 7,550 | 1.06 | 0.06093 | +0.500 | 0.311 | 6.765 | −0.0113 | 0 |

¹ no solution file in its own directory (§3).
**Bold ν** = resolved archives that land at or below the hall's ν band.

---

## 5. TABLE B — the gate, all 22 candidates

`ρ` = Spearman(statistic, `cells`) — the size-decoupling half of the job.
`margin` = the multiplicative gap between the classes; **0.000 means the ranges overlap.**
`STRADDLE` = resolved archives sit on **both** sides of the hall's range, so no threshold
works in either direction.

| statistic | verdict | ρ | margin | hall range | resolved range | offenders |
|---|---|---:|---:|---|---|---|
| `concentration` *(shipped)* | FAILS | **+0.940** | 0.000 | 31.04 .. 508.5 | 1.057 .. 270 | 4 in-band |
| **(a)** size-partialed residual | FAILS | **+0.074** | 0.000 | −0.0076 .. 0.1196 | −0.654 .. 0.755 | STRADDLE, 1 in-band |
| **(b)** ν = cells/records | FAILS | +0.300 | 0.000 | 0.00871 .. 0.01648 | 0.00177 .. 0.4604 | STRADDLE, 1 in-band |
| **(b)** records per cell | FAILS | −0.300 | 0.000 | 60.69 .. 114.9 | 2.172 .. 564.1 | STRADDLE, 1 in-band |
| **(b)** improvements/record | FAILS | +0.156 | 0.000 | 0.01069 .. 0.01814 | 0.00894 .. 0.1321 | STRADDLE, 2 in-band |
| **(b)** improvements/cell | FAILS | −0.363 | 0.000 | 0.7234 .. 1.729 | 0.2294 .. 5.043 | STRADDLE, 4 in-band |
| **(b)** new cells / selection | FAILS | +0.147 | 0.000 | 0.3808 .. 0.7778 | 0.0976 .. 48.21 | STRADDLE, 2 in-band |
| **(b)** burst length | FAILS | −0.263 | 0.000 | 25.5 .. 68.68 | 41.54 .. 181.2 | 7 in-band |
| **(b)** explored-fraction residual | FAILS | −0.456 | 0.000 | −0.0756 .. −0.0398 | −0.0957 .. 0.0225 | STRADDLE, 3 in-band |
| **(b)** selections per cell | FAILS | −0.147 | 0.000 | 1.286 .. 2.626 | 0.0207 .. 10.25 | STRADDLE, 2 in-band |
| **(c)** size-band percentile | FAILS | +0.185 | 0.000 | 0.143 .. 1 | 0 .. 1 | 10 in-band |
| **(d)** log-log growth exponent (tail) | FAILS | +0.632 | 0.000 | 0.1995 .. 0.899 | 0.0230 .. 0.8889 | 4 in-band |
| **(d)** exponent ratio tail/head | FAILS | +0.308 | 0.000 | 1.055 .. 5.437 | 0.674 .. 1.227 | 2 in-band |
| **(d)** pre-solution exponent | FAILS | +0.406 | 0.000 | 0.0970 .. 0.7486 | 0.0218 .. 0.8836 | STRADDLE |
| **(d)** marginal novelty, 10-rec tail | FAILS | +0.503 | 0.000 | 0.00363 .. 0.00957 | 4.9e−05 .. 0.1244 | STRADDLE |
| **(d)** marginal novelty, pre-solution | FAILS | +0.503 | 0.000 | 0.00363 .. 0.00957 | 4.9e−05 .. 0.1244 | STRADDLE |
| **(d)** pre-solution cells/step | FAILS | +0.366 | 0.000 | 0.00864 .. 0.01633 | 0.00176 .. 0.4531 | STRADDLE, 1 in-band |
| **(e)** `live_state_axis_count` | FAILS | +0.701 | 0.000 | 1 .. 3 | 0 .. 4 | STRADDLE, 11 in-band |
| **(e)** `alias_ratio` | FAILS | +0.875 | 0.000 | 45.98 .. 715.2 | 1.057 .. 387.5 | 3 in-band |
| **(f)** evenness, cell mass | FAILS | −0.760 | 0.000 | 0.5747 .. 0.8661 | 0.5417 .. 0.9914 | STRADDLE, 5 in-band |
| **(f)** evenness, visit mass | FAILS | **−0.019** | 0.000 | 0.1148 .. 0.2112 | 0.0321 .. 0.6284 | STRADDLE, 1 in-band |
| **(f)** effective-support concentration | FAILS | +0.963 | 0.000 | 53.8 .. 884.8 | 1.078 .. 443.3 | 4 in-band |

**22 of 22 FAIL. 0 of 22 report a positive margin in either direction.**

---

## 6. (a) Size-partialed concentration — dead by algebra, confirmed by measurement

Regress `log concentration` on `log cells` over the scored corpus and keep the residual.

```
OLS:  log conc = -6.1008 + 0.9227 * log cells        (Spearman conc~cells = +0.940)
```

Because `conc ≡ cells / distinct_spatial` by definition, the residual is

```
resid = log conc - b0 - b1 log cells
      = (1 - b1) log cells  -  log ds  -  b0
      =  0.0773 log cells   -  log ds  -  b0
```

The `log cells` term spans 0.47 – 1.02 across the corpus; the `−log ds` term spans a range
2.9× wider. **The residual is `−log(distinct_spatial)` wearing a hat.** Measured:
Pearson(residual, `−log ds`) = **+0.9066**, Pearson(residual, `log cells`) = −0.0000
(zero by construction).

And `distinct_spatial` is exactly the quantity the K-falsifier already showed cannot
separate — it is bounded by map geometry, effectively a per-level constant:

```
hall     ds:  932, 1089, 1094, 1096, 1102
resolved ds:  435, 525, 638, 750, 877, 902, 970, 1112, 1114, 1178, 1205, 1234, 1376, 2220
```

The hall sits in the **middle** of the resolved range. `lost_levels/ll_1_1_transfer`
(ds 1114) is one bucket away from `cv_chain_hw/lvl_03_overnight` (ds 1102) and lands
inside the hall's residual band; `ge_chain/lvl_11_4-4`, `ge_1_4_solve` and
`cv_chain_hw/lvl_00` land above it. STRADDLE.

**This is the receipt's cleanest result: (a) succeeds completely at its stated job and
that job turns out to be worthless.** ρ falls +0.940 → +0.074 and the hall's spread
collapses from 16.4× to a residual band 0.13 wide — and the classes still overlap. Any
future attempt to "fix concentration by controlling for size" is fixing a confound that
was never the binding one.

---

## 7. (b) Why the boundary statistics failed — the coupon-collector null

The K-falsifier's §9 recorded that `explored_fraction` and `boundary_visit_entropy` *do*
rank the hall above the registered four and are nevertheless inadmissible, refuted by
`ge_chain/lvl_11_4-4` (0.9986 explored) and `ge_1_2_solve` (1.0000 entropy). It did not
say **why**. Here is why, and it is not a coincidence.

`Cell.explored` flips the first time a cell is **chosen** as a return target
(`go_explore_solve.py:2797,2821,2862`). Under a null in which `T` selections fall over
`N` cells with no structure at all,

```
E[explored_fraction] = 1 - (1 - 1/N)^T  ~=  1 - exp(-T/N)
```

`T = Σ times_chosen` and `N = cells` are both read straight off the archive. Measured
across all 19:

| run | cells | selections | sel/cell | observed | coupon null | residual |
|---|---:|---:|---:|---:|---:|---:|
| `ge_chain_w8/lvl_02_8-3` | 19,958 | 414 | 0.021 | 0.0146 | 0.0205 | −0.0059 |
| `ge_chain_w8/lvl_01_8-2` | 19,330 | 482 | 0.025 | 0.0165 | 0.0246 | −0.0082 |
| `ge_chain_w8/lvl_00_8-1` | 218,218 | 7,874 | 0.036 | 0.0101 | 0.0354 | −0.0253 |
| `lost_levels/ll_1_1_transfer` | 166,683 | 11,738 | 0.070 | 0.0244 | 0.0680 | −0.0436 |
| `smb_4_4_micro/lvl_3-3` | 141,764 | 18,985 | 0.134 | 0.0931 | 0.1253 | −0.0322 |
| `cv_chain_hw/lvl_00` | 460 | 68 | 0.148 | 0.1261 | 0.1374 | −0.0113 |
| `ge_chain/lvl_02_2-3` | 5,662 | 926 | 0.164 | 0.1443 | 0.1509 | −0.0066 |
| `smb_4_4_micro/lvl_1-3` | 210,648 | 41,878 | 0.199 | 0.0846 | 0.1803 | −0.0957 |
| `cv_chain_hw/lvl_01` | 5,936 | 2,021 | 0.340 | 0.2460 | 0.2886 | −0.0426 |
| `ge_1_2_solve` | 8,180 | 8,934 | 1.092 | 0.6274 | 0.6645 | −0.0371 |
| **`cv_chain_hw/lvl_03_trace`** | 28,929 | 37,195 | 1.286 | 0.6480 | 0.7236 | −0.0756 |
| **`cv_chain_hw/lvl_03_overnight`** | 560,410 | 760,899 | 1.358 | 0.6763 | 0.7428 | −0.0665 |
| `contra…/r1_ortho` | 52,539 | 74,987 | 1.427 | 0.5369 | 0.7600 | −0.2232 |
| `cv_chain_hw/lvl_02` | 1,202 | 2,115 | 1.760 | 0.7529 | 0.8279 | −0.0750 |
| **`cv_chain_hw2/lvl_03_trace`** | 92,785 | 191,652 | 2.066 | 0.8335 | 0.8733 | −0.0398 |
| **`cv_hall_ortho_ctrl`** | 149,153 | 355,065 | 2.381 | 0.8556 | 0.9075 | −0.0519 |
| `ge_1_4_solve` | 4,815 | 11,603 | 2.410 | 0.9327 | 0.9102 | +0.0225 |
| **`cv_hall_ortho_a`** | 131,561 | 345,470 | 2.626 | 0.8633 | 0.9276 | −0.0643 |
| `ge_chain/lvl_11_4-4` | 5,885 | 60,316 | 10.249 | 0.9986 | 1.0000 | −0.0013 |

**Pearson(observed, coupon null) = +0.9906.**

`explored_fraction` is a saturating function of **selections per cell** and essentially
nothing else. It "ranked the hall above the four" because the hall ran 1.3 – 2.6
selections per cell while the four ran 0.02 – 0.20 — a pure effort/size ordering, the
identical defect §5.1 diagnosed in `concentration`, wearing a different hat.
`boundary_visit_entropy` fails for the adjacent reason already in the calibration: it
measures how evenly *returns* were spread, not how varied the *actions* were, and every
class scores ≥ 0.77.

**Designing around it does not rescue it.** The residual `observed − null` (the
principled correction) is column `expR` in Table A. It STRADDLES: `ge_1_4_solve` (+0.0225)
sits above every hall read, while `smb_4_4_micro/lvl_1-3` (−0.0957),
`cv_chain_hw/lvl_02` (−0.0750) and `cv_chain_hw/lvl_01` (−0.0426) sit at or inside the
hall's band. The residual is small and noisy for everything: no archive deviates from the
uniform null by more than 0.10 except the UNTYPED Contra arm (−0.223).

So does the effort measure itself: `selections per cell`, which is what the whole family
reduces to, puts the hall at 1.29 – 2.63 — squarely between `ge_1_2_solve` (1.09) and
`ge_1_4_solve` (2.41), with `ge_chain/lvl_11_4-4` at 10.25 far above.

---

## 8. (c) Effort-matched percentile, (d) growth-curve shape, (e) interaction-blindness

**(c) size-band percentile** — rank of `concentration` among archives within ±0.5 dex of
log `cells`. **10 of 11** scorable resolved archives land inside the hall's band. This is
the worst-performing candidate after (e), and the reason is mechanical: inside a narrow
size band, `concentration` has almost no variance left to rank on — the K-falsifier's
+0.929 *is* the statement that `concentration` is a monotone function of `cells`, so
conditioning on `cells` conditions away the statistic. Percentile-within-band converts a
size clock into a coin flip, not into a wall detector.

**(d) growth-curve shape** — four forms tested: log-log exponent of `cells` vs `steps`
over the trailing half, the tail/head exponent ratio (a curvature measure), the exponent
over the pre-solution prefix only, and the marginal `Δcells/Δsteps` over the last
`WINDOW_RECORDS = 10` records before the decision moment.

The registered intuition — *"resolved archives keep growing to solution, walls flatten"* —
**is not what the telemetry shows.** The hall does not flatten. `cv_chain_hw2/lvl_03_trace`
has the **highest** tail exponent in the entire corpus (**0.899**), above
`ge_chain_w8/lvl_00_8-1` (0.889) and `smb_4_4_micro/lvl_1-3` (0.841), while
`ge_chain/lvl_11_4-4` (0.023) and `ge_1_4_solve` (0.068) — both resolved — are the
flattest things measured. This is the calibration receipt's §3 finding restated in
scale-free form: the cell key carries nuisance dimensions (`phase`, `vsign`, `route_sig`,
`loops`, state-signature bits) that **manufacture novelty forever at a fixed location**.
The hall grew 3,354 → 91,995 cells with `max_gx_in_max_area` pinned at 767 for 84
consecutive records. A curve that keeps climbing while the map is frozen cannot be caught
by any statistic on that curve.

Two coverage notes, so the row is not over-read: the 10-record marginal forms are scorable
on only 3–4 resolved archives (most banked runs hold fewer than 11 progress records), and
they STRADDLE even there. The wide-coverage `pre-solution cells/step` variant reaches 12
of 13 and also STRADDLEs, killed by `ge_1_2_solve` (0.00176 vs the hall's 0.00864 –
0.01633).

**(e) interaction-blindness profile** — `live_state_axis_count` is the worst candidate in
the set: hall `[3, 1, 1, 1, 1]`, resolved `[0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4]`.
**11 of 13** resolved archives fall inside the hall's range and both tails are populated.
The K-falsifier already recorded that `lvl_03_overnight` reads 3 live axes and so the
signal does not hold across the hall; scored against a resolved class larger than four, it
does not hold on the other side either. `alias_ratio` — band-local concentration — behaves
exactly like `concentration` (ρ +0.875) and is killed by the same three SMB archives.

**(e) is still worth keeping for what the K-falsifier said it was worth keeping for.** It
supports the interaction-blind thesis and the `KEY_BLIND` family; it is not, and this
receipt does not make it, a wall statistic.

---

## 9. (f) Two devised candidates, the same-chain sibling test, and the near-misses

The two constructions devised here are the only ones that beat the shipped statistic on
*any* axis, so they get the fullest accounting.

**`ν = cells / records`** — the fraction of recorded observations that created a new cell.
Archive-intrinsic (no progress file, no pacing confound, no segment-pairing ambiguity),
free from the sidecar, and interpretable: *how much of what the search saw was new?*

**Visit-mass evenness** — `exp(H) / D`, where `H` is the Shannon entropy of visit mass
across the archive's `D` occupied spatial buckets. A normalized effective-support ratio in
`(0, 1]`; multiplying every count by a constant leaves it unchanged, so it is size-free by
construction (ρ = **−0.019**, the best decoupling in the set).

### 9.1 The same-chain sibling test — both PASS

`runs/ge_chain_w8` is one chain, one profile, one `--gx-bucket 16` grid, all three levels
solved. It is where `concentration` embarrassed itself: a 6.3× intra-chain spread that
straddles the shipped 25.0 and puts a solved archive 3.2× over the "gated" upper bracket.

| | conc | ν | size-partialed | evenness (visit) | new/selection |
|---|---:|---:|---:|---:|---:|
| `lvl_01_8-2` | 15.66 | 0.42684 | −0.254 | 0.507 | 40.10 |
| `lvl_02_8-3` | 20.58 | 0.46042 | −0.011 | 0.451 | 48.21 |
| `lvl_00_8-1` | 98.30 | 0.29842 | −0.654 | 0.412 | 27.71 |
| **intra-chain spread** | **6.3×** | **1.5×** | — | **1.2×** | **1.7×** |
| hall band | 31.0 – 508.5 | 0.0087 – 0.0165 | −0.008 – +0.120 | 0.115 – 0.211 | 0.38 – 0.78 |
| hall intra-class spread | **16.4×** | **1.9×** | — | **1.8×** | **2.0×** |

`ν` and visit-mass evenness both **pass the sibling test cleanly**: all three siblings sit
far on the resolved side (ν 18.1× above the hall's max; evenness 1.95× above), and both
collapse the 6.3× intra-chain spread to 1.2 – 1.5× while holding the hall's own five reads
inside 1.8 – 1.9× across a 19× range in `cells`. That is real, and it is the first time
any statistic in this line has held the hall stable across the overnight/trace size gap.

### 9.2 …and both fail the gate anyway

* **`ν`** is killed by four resolved archives at or below the hall's 0.016476:
  `ge_1_2_solve` **0.01073** (inside the band), `cv_chain_hw/lvl_02` 0.00855,
  `ge_1_4_solve` 0.00624, `ge_chain/lvl_11_4-4` 0.00177. STRADDLE.
* **visit-mass evenness** is killed by `cv_chain_hw/lvl_02` (0.204, inside the band) with
  `ge_1_2_solve` 0.032, `lost_levels/ll_1_1_transfer` 0.043 and `ge_1_4_solve` 0.054
  below, and everything else above. STRADDLE.

The mechanism is the same for both and it is worth naming, because it is the trap for the
next attempt: **a run that solves does not stop recording.** `EXPLORE_AFTER_FIRST_CLEAR`
keeps replacing elites with shorter trajectories (`go_explore.py:173-179`), so a resolved
archive buries its discovery phase under post-clear re-treading and reads as stuck.
`ge_1_4_solve` spent 771,291 records on 4,815 cells; `ge_chain/lvl_11_4-4` spent 3,319,567
on 5,885. Truncating at the solution is the obvious repair, and it is exactly what
candidate (d)'s pre-solution forms do — they STRADDLE too (§8), because the *other* half
of the offenders (`ge_1_2_solve`) is genuinely, measurably a low-novelty search that
solved anyway.

### 9.3 Cost of the best possible threshold

No candidate separates, so the only remaining question is what the least-bad cut costs.
For each candidate: the threshold that captures all five hall reads, and how many of the
13 resolved archives it also condemns as walls.

| candidate | dir | cut | false GATED | condemned resolved archives |
|---|---|---:|---:|---|
| `concentration` *(shipped, struck)* | hi | 31.04 | **4 / 13** | `lvl_3-3`, `lvl_1-3`, `ll_1_1_transfer`, `lvl_00_8-1` |
| **(b) new cells / selection** | lo | 0.7778 | **3 / 13** | `lvl_11_4-4`¹, `ge_1_4_solve`, `cv/lvl_02` |
| (b) ν = cells/records | lo | 0.01648 | 4 / 13 | `lvl_11_4-4`¹, `ge_1_2_solve`, `ge_1_4_solve`, `cv/lvl_02` |
| (b) explored residual | lo | −0.0398 | 4 / 13 | `lvl_1-3`, `ll_1_1_transfer`, `cv/lvl_01`, `cv/lvl_02` |
| (f) evenness, visit mass | lo | 0.2112 | 4 / 13 | `ll_1_1_transfer`, `ge_1_2_solve`, `ge_1_4_solve`, `cv/lvl_02` |
| (a) size-partialed residual | hi | −0.0076 | 6 / 13 | + `lvl_11_4-4`¹, `ge_1_4_solve`, `cv/lvl_00` |
| (f) evenness, cell mass | lo | 0.8661 | 8 / 13 | — |
| (e) live state axes | hi | 1 | 12 / 13 | — |

¹ the row with no solution file of its own (§3). Dropping it: best becomes **2 / 12**.

**The best available replacement misclassifies 15 – 23% of the resolved class.** The
calibration receipt's strongest surviving claim was *"No false GATED anywhere"* — already
falsified by the K-falsifier's §7 on one archive. No candidate here restores it; the best
one breaks it three times.

---

## 10. Held-out generalization — the same-game control

`archive.stats.json` (§2.2) makes `ν`, improvements-per-record and `explored_fraction`
computable for every banked archive. So the sidecar-derived candidates get a real
held-out test, which the K-falsifier could not run: **84 solved archives outside the
primary corpus**, excluding the `KEY_BLIND` Bubble Bobble family — 32 SMB levels from
`ge_chain_final_a`/`_b` (the full game-complete chain), plus `ge_chain_w5plus*`,
`lost_levels`, `cv_chain_a`, `cv_chain_poweron`, `cv_chain_hw2`, `detector_gate_20260810`,
`smb_chain_regress`, `excitebike` and others.

**`ν`: 8 of 84 held-out solved archives fall at or below the hall's maximum.** Each was
read to verify it is genuinely admissible rather than something a pre-existing gate would
have caught first:

| ν | cells | records | span | ds | run | admissible? |
|---:|---:|---:|---:|---:|---|---|
| 0.000128 | 197 | 1,542,302 | 41 | 190 | `cv_chain_a/lvl_00_2` | no — `cells < COVERAGE_FLOOR_CELLS` |
| 0.000184 | 220 | 1,194,447 | 41 | 181 | `cv_chain_a/reenter_b2_s1001` | no — same |
| 0.000327 | 99 | 302,809 | **1** | 99 | `excitebike/excitebike_bootstrap` | no — `KEY_BLIND` |
| **0.006895** | **1,481** | 214,800 | 80 | 932 | **`cv_chain_poweron/lvl_02_2_kit`** | **YES** |
| **0.007409** | **1,346** | 181,676 | 80 | 850 | **`cv_chain_hw2/lvl_02`** | **YES** |
| 0.011181 | 107 | 9,570 | 44 | 88 | `cv_chain_poweron/lvl_00_0` | no — `cells < 256` |
| 0.015453 | 84 | 5,436 | 46 | 75 | `cv_smoke` | no — `cells < 256` |
| 0.016014 | 103 | 6,432 | 44 | 84 | `cv_chain_a/lvl_00_0` | no — `cells < 256` |

Two survive every pre-gate, and **both are Castlevania**. `cv_chain_hw2/lvl_02`
(ds 850, span 80) and `cv_chain_poweron/lvl_02_2_kit` (ds 932, span 80) are the *same
game*, the *same grid*, the *same discretization* and the *same order of magnitude* as
`cv_chain_hw/lvl_03_trace` (ds 932, span 94) — and they **solved**, at ν 0.0069 and 0.0074
against the hall's 0.0087 – 0.0165.

This is the cleanest kill in the receipt. Every other offender can be argued about as a
cross-game scale artifact. These cannot: the solved Castlevania block immediately before
the hall, on the identical profile, reads *more* stuck than the hall does.

Improvements-per-record and `explored_fraction` fail held-out the same way — the same two
CV blocks land inside the hall's band on both, and 79 of 84 held-out archives fall below
the hall's `explored_fraction`, which is §7 restated at scale.

---

## 11. Why this is a null result and not a search that stopped too early

Four structural reasons, in increasing order of how much they should discourage a
follow-up.

1. **Both halves of the problem were solved separately and the conjunction is still
   empty.** Size-decoupling: achieved (ρ +0.940 → +0.074 / −0.019). Within-class
   stability: achieved (hall spread 16.4× → 1.8×). Same-chain coherence: achieved (§9.1).
   The remaining failure is not a modelling failure — it is the classes overlapping.
2. **15 of 22 candidates STRADDLE.** Overlap on one side invites "try a better transform".
   Resolved archives on *both* sides of the hall means the hall is interior to the
   resolved distribution on that axis, and no monotone transform of that axis can fix it.
3. **The ground truth is a prediction, not a measurement.** The K-falsifier's §7 found
   `smb_4_4_micro/lvl_1-3`'s scored archive was a *genuinely* stuck search — zero
   topological transition, zero map delta, 17 records since the map moved — that cleared
   17 minutes later on a plain retry with no orthogonal mechanism. A statistic asked to
   separate "resolved" from "gated" archives is being asked to read the future off a
   snapshot of a search that had not yet resolved. `ge_1_2_solve` and the two held-out CV
   blocks are the same shape: low-novelty searches that solved anyway.
4. **The positive class is still one wall seen five times, with no held-out positive.**
   This receipt scanned 22 candidates × 2 directions against it. Had one passed, the
   K-falsifier's §9 objection would have applied verbatim — a separator selected by
   sweeping against a single unreplicated positive has no standing, and would need its own
   pre-registration, its own band, and a second wall before it could gate anything. That
   none passed is the one outcome this design *can* report cleanly, and it is the outcome.

**What would actually move this.** Not another cross-sectional statistic. The calibration
receipt's §8 already named the missing input and it is still missing: `c_local` —
`len({(area, y_band, gx_bucket)})` **emitted per progress line**. The hall's real
signature is visible in Table A and is not a level, it is a *derivative*: `distinct_spatial`
pinned at 932 → 1,102 (+18%) while `cells` goes 28,929 → 560,410 (19×). Footprint
elasticity `d log(ds) / d log(cells)` is scale-free, is the actual adopted form, and is
**not reconstructible from anything on disk** — a banked archive holds exactly one
`distinct_spatial` reading, at the flush. Every candidate in §4-§10 is a surrogate for a
derivative that nobody recorded. The correct next step is to emit the field and bank
labelled runs, in the promotion order §8 of the calibration already specifies — not to
build a 23rd surrogate.

---

## 12. Integration spec (`src/training/wall_taxonomy.py` — NOT modified here)

Nothing in this receipt promotes a statistic, so nothing here adds a gate. Four edits are
owed, all of them subtractive or reportorial, and they compose with the four the
K-falsifier's §12 already owes.

1. **Remove the `GATED` branch rather than re-thresholding it.** The K-falsifier's §12.2
   left the choice open ("must carry a `REFUTED-OFFLINE-2026-08-10` tag **or** the `GATED`
   branch must be removed"). This receipt closes it: a search for a replacement over 22
   candidates and 103 archives returned nothing, so there is no constant to re-tag into
   correctness. Delete branches 8 and 10's `GATED` outcomes; a plateau with no admissible
   corroboration is `INDETERMINATE`, whose remedy — *"collect the missing telemetry"* — is
   now literally correct (§11, `c_local`). `WallClass.GATED` itself should stay in the
   enum only if something still reads it; nothing does
   (`test_module_is_not_wired_into_any_runtime_dispatch`).

2. **Record the refutations in the module's own idiom.** It already ships
   `RAW_COVERAGE_SATURATION_IS_SEPARATING`, `CHURN_IS_SEPARATING`,
   `BOUNDARY_ENTROPY_IS_SEPARATING`, `MAP_STALL_WINDOWS_IS_SEPARATING`, all `False`, all so
   nobody re-derives them and believes it. Add, tagged `REFUTED-OFFLINE-2026-08-11` and
   pointing at this file:

   ```
   SIZE_PARTIALED_CONCENTRATION_IS_SEPARATING = False   # = -log(ds); §6
   NOVELTY_PER_RECORD_IS_SEPARATING           = False   # §9.2, §10
   EXPLORED_FRACTION_IS_SEPARATING            = False   # coupon-collector null, §7
   SPATIAL_EVENNESS_IS_SEPARATING             = False   # §9.2
   GROWTH_EXPONENT_IS_SEPARATING              = False   # the hall does NOT flatten; §8
   EFFORT_MATCHED_PERCENTILE_IS_SEPARATING    = False   # §8
   ```

   Each should carry the offender that kills it, the way §3 of the calibration does, and
   §7's Pearson +0.9906 belongs in the `EXPLORED_FRACTION` docstring — it is the general
   argument, not one corpus's accident, and it pre-empts the whole family.

3. **Adopt the sidecar as a first-class adapter (reporting only, gates nothing).**
   `archive.stats.json` is ~130 bytes and carries `records` (≡ `Σ visits`), `new_cells`,
   `improvements` and `frontier` (≡ unexplored cells) — verified exactly, §2.2. Proposed:

   ```python
   @dataclass(frozen=True)
   class ArchiveCounters:
       cells: int; records: int; new_cells: int
       improvements: int; frontier: int; best_score: float

   def read_archive_counters(path) -> Optional[ArchiveCounters]: ...
   ```

   Value: it is the archive-intrinsic effort denominator, so it removes the segment-pairing
   hazard the K-falsifier had to audit by hand in its §7 (`ge_chain_w8/lvl_00_8-1`'s
   8.27 M "total steps" belong to two attempts; its archive recorded 731,234). Surface
   `records` and `explored_fraction` in `_evidence()` so every verdict shows the effort it
   was taken over — the same reason `map_stall_windows` is reported without gating.
   **Do not** let any of it reach a branch.

4. **Automate the §7 segment audit in `telemetry_from_paths`.** Add
   `segment="auto"`, pairing the archive with the segment whose peak `cells` is nearest
   `archive.cells`. Default must stay `segment=-1` so no banked verdict moves; `"auto"`
   is opt-in and would have caught `lvl_1-3` (archive is segment 2 of four) without a
   human reading mtimes. Implemented and exercised in `/tmp/wall_stat_proto.py`.

**Tests.** `test_no_false_gated_anywhere_in_the_corpus` is already scheduled to fail under
the K-falsifier's §12.2; under edit 1 it becomes vacuous and should be replaced by
`test_no_verdict_is_gated` plus a property asserting each new `*_IS_SEPARATING` constant is
`False` and cited. No new corpus fixtures are needed — every number here is recomputable
from the sidecars.

---

## 13. Reproduction

```
prototype     : /tmp/wall_stat_proto.py            (scan | read | evaluate)
python        : .venv/bin/python
module        : src/training/wall_taxonomy.py @ CALIBRATED-OFFLINE-2026-08-10 (unmodified)
archives read : 19 distinct, 34.8 GiB of archive.pkl, 7.1 s unpickle
sidecars      : 276 archive.stats.json + 276 progress.jsonl  (a few hundred KB total)
archive read  : wall_taxonomy._StateDroppingUnpickler
spatial proj. : wall_taxonomy._spatial_key / summarize_archive_cells
band profile  : wall_taxonomy.boundary_axis_profile(cells, band=24, bookkeeping=(4,5))
progress read : wall_taxonomy.load_progress_segments, paired by nearest peak cell count
counters      : Sigma visits == archive.stats.json["records"]  (verified 3/3 by direct read)
coupon null   : p_null = 1 - exp(-Sigma times_chosen / cells)
gate          : max(0, min(hall) - max(resolved), min(resolved) - max(hall)) > 0
                — both directions, unlike the registration's PASS clause (§2.1)
```

Cached intermediates live in `/tmp/wall_stat_proto_cache/`; deleting it re-derives
everything from `runs/` in about 10 s.

---

## 14. Scope — what this does not say

* **It does not say the Castlevania hall is coverage-limited.** It says that after a
  second, wider search there is still no statistic that tells a stuck search from one that
  is about to resolve. The hall's facts are unchanged: five runs, ~10.7 h, ~77 M steps,
  best score pinned at 767, zero crossings, zero solutions. `UNRESOLVED-CONCENTRATED`
  remains the honest label, and it is now the label after a genuine attempt to improve on
  it rather than a placeholder.
* **It does not invalidate the module further than the K-falsifier already did.**
  `KEY_BLIND`, the three earlier `REFUTED-OFFLINE` statistics, the segment-splitting
  adapter, the `STAGNANT ≠ plateaued` fix, `SPATIAL_SPAN_MIN`, `C_LOCAL_FLOOR_BUCKETS`,
  `COVERAGE_FLOOR_CELLS`, `FROZEN_WINDOWS_MAX` and `EFFORT_MIN_STEPS` are untouched, and
  §3 verifies all 13 resolved archives clear every one of them, so none of them is
  propping up a result here.
* **It does not license Contra.** `contra_reentry_2026-08-10/r1_ortho` was carried through
  every candidate as an unlabelled row and never used to fit anything. It reads
  hall-adjacent on ν (0.01671 against the hall's 0.01648) and far off it on the explored
  residual (−0.2232, the largest deviation in the corpus). Two candidates disagreeing about
  an unlabelled arm is exactly why it stays UNTYPED.
* **`ge_chain/lvl_11_4-4` banks no solution of its own** (§3) and is the offender in three
  of the eight rows of §9.3. Every table that leans on it says so; dropping it moves the
  best candidate from 3/13 to 2/12 and changes no verdict.
* **One wall, five reads, no held-out positive.** Unchanged, and now with 84 held-out
  *negatives* on the other side, which is the asymmetry that makes a PASS structurally
  hard to earn and a FAIL easy to trust.
* **Nothing is armed and nothing was changed.** `src/training/wall_taxonomy.py` is
  byte-identical to its state at the start of this work; §12 is a spec, not a diff.
