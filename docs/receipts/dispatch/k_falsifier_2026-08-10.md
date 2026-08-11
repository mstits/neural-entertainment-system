# K-FALSIFIER — does the concentration statistic separate GATED from SOLVED?

**Date:** 2026-08-10
**Registered in:** `docs/proposals/GATE_OPENER_CAMPAIGN_2026-08-11.md` §7 (K-FALSIFIER),
scheduled §5 D2, motivated §10-O1.
**Module under test:** `src/training/wall_taxonomy.py` (`CALIBRATED-OFFLINE-2026-08-10`).
**Calibration receipt:** `docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md`.
**Compute:** offline, read-only. No solver run, no build, no commit. 20 distinct
archives read, 37.7 GiB of `archive.pkl`, 7.8 s total read time.

---

## 0. VERDICT — **FAIL**

> All four registered SOLVED archives read **GATED** under the bypassed statistic,
> at 3.9× – 10.8× `CONCENTRATION_GATED_MIN` and 3.2× – 8.7× the *upper* bracket of
> the constant's own separating band. Three of the four outrank **every** Castlevania
> hall read that defines the positive class. No admissible statistic ranks the hall
> above all four.

Per §7's pre-registered consequence, the FAIL branch fires:

* **`GATED`, `saturated`, `tried ~135×` and every visits-as-saturation phrasing is
  struck** from the campaign doc, the HUD, and FORGE#2.
* **Castlevania block 3 is relabelled `UNRESOLVED-CONCENTRATED`.**
* **The gate-opener arm is NOT killed.** §0's thesis was already reworded to stand
  on its own ("the archive shows every position VISITED heavily … and shows NOTHING
  about which INTERACTIONS were tried there"); the interaction sweep proceeds
  unchanged as a discovery instrument, and its §2 primary endpoint
  (G1-DISCOVERY: surviving candidate + clean K1 sham null + shadow_yield > 0)
  never referenced the label.

The registration's first required output — reproduce the red-team scoring — is
reproduced **exactly**: 270.03 / 240.19 / 149.63 / 98.30 against the hall's 136.34.

---

## 1. What was registered

§7, verbatim:

> K-FALSIFIER (NEW, D2, offline, BLOCKING for vocabulary — §10-O1): score
> `runs/live_show/smb_4_4_micro/lvl_3-3`, `lvl_1-3`, `runs/ge_chain_w8/lvl_00_8-1`,
> `runs/lost_levels/ll_1_1_transfer` (effort-matched RESOLVED archives, all on disk)
> through wall_taxonomy with the solutions/progressing branches bypassed. PASS = some
> statistic ranks the CV hall ABOVE all four (concentration alone does not: red-team
> scoring found 270.03/240.19/149.63/98.30 vs the hall's 136.34 — to be reproduced as
> this gate's first output). FAIL = strike GATED / "saturated" / any
> visits-as-saturation phrasing from this doc, the HUD, and FORGE#2; relabel CV
> UNRESOLVED-CONCENTRATED; the sweep proceeds unchanged as a pure discovery instrument
> (its value never depended on the label).

The defect it targets (§10-O1): `gated_wall_verdict` tests RESOLVED at branch 2 and
PROGRESSING at branch 3, and only reaches the concentration test at branches 7/9.
A solved archive therefore **cannot** reach the statistic, so the calibration corpus's
separating band — 20.58 (SMB 8-3, resolved) .. 31.04 (CV hall, gated) — was never
required to hold against a large solved archive. Verified in source:
`wall_taxonomy.py:911` (`solutions > 0` → RESOLVED), `:916` (`topo_delta > 0 or
map_delta > 0` → PROGRESSING), `:994` and `:1022` (the two concentration branches).

### 1.1 Substitutions

**None.** All four registered paths exist on disk with a readable `archive.pkl` and a
banked solution receipt. The registration's escape hatches ("whatever
`ge_chain_w8/lvl_00_8-1` path exists", "substitute another solved archive if absent")
were not needed.

| registered path | on disk | `archive.pkl` | solutions/ |
|---|---|---|---|
| `runs/live_show/smb_4_4_micro/lvl_3-3` | yes | 3,011,004,094 B | `sol_000` (`clear_wd [2,3]`) |
| `runs/live_show/smb_4_4_micro/lvl_1-3` | yes | 4,473,984,913 B | `sol_000` (`clear_wd [0,3]`) |
| `runs/ge_chain_w8/lvl_00_8-1` | yes | 4,633,665,475 B | `sol_000` (`clear_wd [7,1]`) |
| `runs/lost_levels/ll_1_1_transfer` | yes | 2,177,870,985 B | `sol_000`..`sol_004` |

---

## 2. Method

The statistic was called through the shipped code path, not re-implemented. A copy of
`gated_wall_verdict` was made with **exactly two branches deleted** — the `solutions > 0`
RESOLVED test and the `topo_delta > 0 or map_delta > 0` PROGRESSING test — and nothing
else changed: same `_evidence()`, same `summarize_archive_cells()`, same
`_spatial_key()`, same constants, same branch order for tests 1 and 4-9. Archives were
read with the module's own `_StateDroppingUnpickler`; progress files with the module's
own `load_progress_segments()`.

Two quantities are reported per run and they must not be conflated:

* **`concentration`** — `cells / distinct_spatial` over the banked `archive.pkl`.
  Cross-sectional, segment-independent, and the quantity the FAIL/PASS turns on.
  This is what `CONCENTRATION_GATED_MIN = 25.0` gates.
* **the bypassed verdict** — the full classifier, which additionally needs a progress
  segment. Segment pairing is audited in §7 because three of the four registered
  directories hold multiple appended attempts.

---

## 3. TABLE A — the four registered SOLVED archives, branches bypassed

`CONCENTRATION_GATED_MIN = 25.0`; the constant's measured band is (20.58, 31.04].

| archive | cells | distinct_spatial | **concentration** | × 25.0 | × 31.04 | span | boundary cells | boundary entropy | shipped verdict | **bypassed verdict** |
|---|---|---|---|---|---|---|---|---|---|---|
| `smb_4_4_micro/lvl_3-3` | 141,764 | 525 | **270.03** | 10.80× | 8.70× | 73 | 192 | 0.9336 | progressing | **GATED** |
| `smb_4_4_micro/lvl_1-3` | 210,648 | 877 | **240.19** | 9.61× | 7.74× | 115 | 89 | 0.9504 | **gated** | **GATED** |
| `lost_levels/ll_1_1_transfer` | 166,683 | 1,114 | **149.63** | 5.99× | 4.82× | 195 | 240 | 0.8204 | resolved | **GATED** |
| `ge_chain_w8/lvl_00_8-1` | 218,218 | 2,220 | **98.30** | 3.93× | 3.17× | 383 | 15 | 0.9384 | insufficient¹ | **GATED** |

¹ the segment that owns this archive holds 4 progress records < `MIN_RECORDS = 12`; see §7.

**4 of 4 read GATED.** §0's "banked SOLVED archives score 4–11× `CONCENTRATION_GATED_MIN`"
is reproduced as **3.9× – 10.8×**.

---

## 4. TABLE B — the GATED-positive receipts, same statistic

| archive | role | cells | distinct_spatial | **concentration** | span | shipped verdict | bypassed verdict |
|---|---|---|---|---|---|---|---|
| `cv_chain_hw/lvl_03_overnight` | hall, deepest banked attack (§4b resume archive) | 560,410 | 1,102 | **508.54** | 95 | **progressing**² | gated |
| `cv_hall_ortho_ctrl` | hall, §0 headline "≈136" | 149,153 | 1,094 | **136.34** | 95 | gated | gated |
| `cv_hall_ortho_a` | hall, class-definer #3 | 131,561 | 1,096 | **120.04** | 95 | gated | gated |
| `cv_chain_hw2/lvl_03_trace` | hall, class-definer #2 | 92,785 | 1,089 | **85.20** | 95 | gated | gated |
| `cv_chain_hw/lvl_03_trace` | hall, class-definer #1 — **sets the band's upper bracket** | 28,929 | 932 | **31.04** | 94 | gated | gated |
| `contra_reentry_2026-08-10/r1_ortho` | UNTYPED (§4d, decline-to-arm) | 52,539 | 1,205 | **43.60** | 193 | gated | gated |

² **the deepest hall archive does not read GATED on the shipped path at all.** It reads
PROGRESSING on `topo_delta = 186`, which comes entirely from the `doors` counter, and
`doors` is not monotone: its trailing eleven values are
`12694, 12707, 12756, 12838, 12901, 12950, 12981, 12987, 12925, 12874, 12880`.
`max_area`, `max_sect` and `max_gx` are all pinned. A fluctuating edge counter is
carrying a "topology moved" verdict on the archive the campaign promotes to THE resume
archive for all T1 arms.

### 4.1 The merged ranking — the falsifier in one column

| rank | archive | class | concentration |
|---|---|---|---|
| 1 | `cv_chain_hw/lvl_03_overnight` | hall | 508.54 |
| 2 | `smb_4_4_micro/lvl_3-3` | **SOLVED** | 270.03 |
| 3 | `smb_4_4_micro/lvl_1-3` | **SOLVED** | 240.19 |
| 4 | `smb_4_4_micro/lvl_4-3` | **SOLVED** (partial read, §8) | 210.45 |
| 5 | `lost_levels/ll_1_1_transfer` | **SOLVED** | 149.63 |
| 6 | `cv_hall_ortho_ctrl` | hall | 136.34 |
| 7 | `cv_hall_ortho_a` | hall (class-definer) | 120.04 |
| 8 | `ge_chain_w8/lvl_00_8-1` | **SOLVED** | 98.30 |
| 9 | `cv_chain_hw2/lvl_03_trace` | hall (class-definer) | 85.20 |
| 10 | `contra/r1_ortho` | UNTYPED | 43.60 |
| 11 | `cv_chain_hw/lvl_03_trace` | hall (class-definer) | 31.04 |

* All four registered SOLVED archives outrank **two of the three class-definers**.
* Three of four outrank **all three class-definers** and the ctrl arm as well.
* The only hall read that outranks all four is `lvl_03_overnight`, which spent
  **45.9 M steps / 45.6 M archive records** — 5.0× – 62× the archive records of the four
  (§5) — and which the shipped classifier does not call GATED anyway (note ²).
* **No threshold rescues it.** To exclude all four solved archives the constant must
  exceed 270.03. At any such value, four of the five hall reads — including all three
  class-definers and the 31.04 read that sets the band's upper bracket — fall to
  COVERAGE_LIMITED, and the positive class is emptied of its own definition.

---

## 5. Effort control — the solved archives are effort-DISADVANTAGED

The registration calls the four "effort-matched". They are not matched; they are
cheaper, which strengthens the falsifier rather than weakening it.

| archive | class | total steps | elapsed s | archive records | concentration | **conc / M archive records** |
|---|---|---|---|---|---|---|
| `smb_4_4_micro/lvl_3-3` | SOLVED | 970,224 | 3,256 | 788,717 | 270.03 | **342.4** |
| `smb_4_4_micro/lvl_1-3` | SOLVED | 2,617,896 | 8,481 | 1,742,758 | 240.19 | **137.8** |
| `ge_chain_w8/lvl_00_8-1` | SOLVED | 8,269,310 | 2,915 | 731,234 | 98.30 | **134.4** |
| `lost_levels/ll_1_1_transfer` | SOLVED | 2,145,350 | 1,081 | 2,126,825 | 149.63 | **70.4** |
| `cv_hall_ortho_ctrl` | hall | 9,135,920 | 5,401 | 9,052,551 | 136.34 | 15.1 |
| `cv_hall_ortho_a` | hall | 8,972,160 | 5,401 | 8,902,418 | 120.04 | 13.5 |
| `cv_chain_hw/lvl_03_trace` | hall | 2,404,020 | 840 | 2,554,439 | 31.04 | 12.2 |
| `cv_chain_hw/lvl_03_overnight` | hall | 45,898,118 | 21,545 | 45,640,527 | 508.54 | 11.1 |
| `cv_chain_hw2/lvl_03_trace` | hall | 10,643,480 | 5,340 | 10,656,920 | 85.20 | 8.0 |
| `contra/r1_ortho` | UNTYPED | 3,202,552 | 1,800 | 3,143,695 | 43.60 | 13.9 |

Two effort-matched pairs make the point without any normalization:

* `smb_1-3` **2.62 M steps → 240.19** vs `cv_chain_hw/lvl_03_trace` **2.40 M steps → 31.04**.
  The solved archive scores **7.7×** the archive that sets the GATED bracket, at 9% more
  compute.
* `smb_8-1` **8.27 M steps → 98.30** vs `cv_hall_ortho_a` **8.97 M steps → 120.04**.
  Nearly identical compute, same order of magnitude, both far above 25.0 — and one of
  them cleared the level.

Effort-normalized, the ordering **inverts completely**: every hall read (8.0 – 15.1
concentration per M archive records) sits below every solved archive (70.4 – 342.4).
On the only effort-controlled form of the statistic, the hall is the *least* concentrated
search in the set.

### 5.1 Why: concentration is archive size wearing a hat

Across the 20 distinct archives scored here (Spearman ρ on ranks):

| relation | ρ |
|---|---|
| `concentration` ~ `cells` | **+0.929** |
| `concentration` ~ `archive_records` | +0.632 |
| `concentration` ~ `distinct_spatial` | **−0.041** |
| `cells` ~ `distinct_spatial` | +0.233 |

`distinct_spatial` is bounded by map geometry and varies only 435 – 2,220 across
everything from a 460-cell CV block to a 560,410-cell hall attack. The denominator is
effectively a per-level constant, so `concentration` is a monotone re-expression of
`cells`, and `cells` is a runtime clock. That is §10-O1's claim, measured.

The calibration corpus could not see this because its resolved class contained **no large
archives**: `ge_1_2` 8,180 cells, `ge_1_4` 4,815, `ge_chain/lvl_11_4-4` 5,885,
`ge_chain/lvl_02_2-3` 5,662, `ge_chain_w8/lvl_02_8-3` 19,958. The band
(20.58, 31.04] was measured between a 19,958-cell resolved archive and a 28,929-cell
gated one. The four archives registered here are 141,764 – 218,218 cells — the first
resolved archives in the hall's size class ever scored — and the band does not survive
contact with them.

---

## 6. Same-family, same-grid control — the band fails without leaving one chain

The strongest objection to §3-§5 is that `concentration` is geometry-dependent
(calibration §9: "re-derive per profile family before trusting the number on a new
title"). It fails inside a single family too.

`runs/ge_chain_w8` is **one chain, one profile, one campaign, one `--gx-bucket 16`
discretization**, and every level in it was **SOLVED**:

| archive | cells | distinct_spatial | concentration | verdict at 25.0 |
|---|---|---|---|---|
| `ge_chain_w8/lvl_01_8-2` | 19,330 | 1,234 | **15.66** | below threshold |
| `ge_chain_w8/lvl_02_8-3` | 19,958 | 970 | **20.58** | below threshold — *this is the band's lower bracket* |
| `ge_chain_w8/lvl_00_8-1` | 218,218 | 2,220 | **98.30** | **GATED** |

Within one chain, resolved concentration spans **6.3×**, straddles the shipped 25.0, and
its top member exceeds the "gated" upper bracket by **3.2×**. The lower bracket of the
shipped band and a 3.2×-over-the-upper-bracket false positive are **siblings in the same
directory**.

The same holds for Castlevania itself: the solved CV blocks `lvl_00` / `lvl_01` / `lvl_02`
score 1.06 / 4.31 / 1.33 — but at 460 / 5,936 / 1,202 cells they are 30× – 1,200× smaller
than the hall attacks, so they confirm §5.1 rather than the constant.

### 6.1 Grid parity — the one real confound, measured

The hall ran at `gx_bucket 8`; all four registered comparators ran at `gx_bucket 16`
(derived exactly, per run, by dividing the progress line's `max_gx_in_max_area` at the
flush that produced the archive by the archive's own maximum `key[-1]`: 16.13, 16.05,
16.01, 16.02 for the four; 8.07 for both ortho arms). A finer grid inflates
`distinct_spatial` and therefore **deflates** the hall's concentration. Correcting it is
the hall's best case, so it is measured rather than waved at.

Each hall archive was re-projected onto the comparators' grid two ways:

* **conc_UB** — coarsen the denominator only (`gx//2`, `y//4`). Not a discretization any
  archive could produce, since it counts cells that exist *only because of* the finer
  grid against a coarser footprint. Reported as a strict upper bound.
* **conc_rekeyed** — coarsen the whole cell key, numerator and denominator together
  (`key[:-2] + (key[-2]//4, key[-1]//2)`). The faithful estimate of what the archive would
  look like at `gx_bucket 16 / y_band 32`.

| hall archive | native conc (gx8) | ds native | ds @ gx16/y32 | conc_UB | **conc_rekeyed** |
|---|---|---|---|---|---|
| `cv_chain_hw/lvl_03_trace` | 31.04 | 932 | 195 | 148.35 | **41.23** |
| `cv_chain_hw2/lvl_03_trace` | 85.20 | 1,089 | 213 | 435.61 | **110.77** |
| `cv_hall_ortho_a` | 120.04 | 1,096 | 214 | 614.77 | **147.37** |
| `cv_hall_ortho_ctrl` | 136.34 | 1,094 | 213 | 700.25 | **173.47** |
| `cv_chain_hw/lvl_03_overnight` | 508.54 | 1,102 | 213 | 2631.03 | **789.80** |

At full grid parity the falsifier still fires: `smb_3-3` (270.03) and `smb_1-3` (240.19)
outrank **every** hall read except the overnight; `ll_1_1_transfer` (149.63) outranks two
of the three class-definers; `smb_8-1` (98.30) outranks one. Grid parity moves the numbers
and does not move the verdict.

Note also what grid parity does to the hall's *footprint*: re-keyed, the hall's
`distinct_spatial` is **213**, the **smallest** in the entire comparison set (the four
comparators sit at 525 – 2,220). Any statistic that ranked the hall highly on
footprint-derived grounds inverts here (§9).

---

## 7. Segment audit — and one false GATED with no bypass at all

Three of the four registered directories hold several attempts appended to one
`progress.jsonl`. The archive on disk belongs to exactly one of them, so the pairing was
audited before any verdict was read.

| directory | segments (records) | which attempt owns `archive.pkl` | pairing |
|---|---|---|---|
| `lvl_3-3` | [38] | the single attempt; archive flushed at 141,764 cells, run ended at 160,843 | **sound** |
| `lvl_1-3` | [10, 2, 70, 10] | seg 2 (70 records, 1.99 h, ended 222,321 cells, 0 solutions); archive 210,648 | **sound** (flush ≈5% before segment end; the clear came in seg 3) |
| `lvl_00_8-1` | [45, 4] | **seg 1** — the 4-record retry that ended at 218,218 cells with `solutions = 1` | sound for the archive; seg 1 is too short for a verdict |
| `ll_1_1_transfer` | [18] | the single attempt; archive 166,683 = last record exactly, 5 solutions in-window | **sound, unimpeachable** |

Consequences:

* `smb_8-1`'s **seg 0** (45 records, 7.52 M steps, `max_gx` pinned at 3,900 for 44
  consecutive records, 0 solutions) reads **GATED** on the shipped path — but its own
  archive was overwritten by the retry, so that pairing is **unsound and is excluded**
  from any false-positive claim. It is still an interesting negative: a segment that
  looked identically stuck for 44 minutes and then fell to a plain retry, the module
  docstring's own SMB 8-4 scenario.
* `smb_1-3`'s **seg 2** pairing IS sound, and it reads **`GATED` on the unmodified
  shipped path** — reasons: *"zero topological transition and zero permanent-map delta
  over 10 records (17 records since the map last moved)"* + *"coverage concentration
  240.19 >= CONCENTRATION_GATED_MIN=25.0"*. Its remedy string is *"switch to an
  orthogonal arm."* The level was cleared **17 minutes after the scored archive was
  flushed**, on the very next attempt — a plain 10-minute retry (seg 3), with no
  orthogonal mechanism of any kind. Mtimes on disk: `archive.pkl` 11:09 (end of seg 2),
  `progress.jsonl` 11:25 (end of seg 3), `solutions/sol_000.json` 11:26.

That contradicts the calibration receipt's strongest surviving claim — §6's
*"No false GATED anywhere"* and `test_no_false_gated_anywhere_in_the_corpus`. The
property was never false-*ified* by that test because these archives were never in the
corpus (§8). It is false now.

The three remaining registered runs behave as the corpus predicts on the shipped path
(`resolved`, `progressing`, `insufficient`) — which is precisely §10-O1's point: the
earlier branches are what hide the statistic's behaviour, not the statistic being right.

---

## 8. Correction to the calibration receipt: the truncation exclusion does not reproduce

`gated_wall_calibration_2026-08-10.md` §2 records:

> Three SMB archives that *do* exist under `smb_4_4_micro` (`lvl_1-3`, `lvl_3-3`,
> `lvl_4-3`, 3-4.5 GB each) are **truncated** — `pickle.load` raises `EOFError: Ran out
> of input` — so they contributed nothing.

Re-read today with the module's own `read_archive_summary`:

| archive | today's read | cells recovered |
|---|---|---|
| `lvl_3-3` | **complete**, 0.7 s | 141,764 (= `archive.stats.json`) |
| `lvl_1-3` | **complete**, 1.0 s | 210,648 (= `archive.stats.json`) |
| `lvl_4-3` | **truncated**, `EOFError` confirmed | 145,000 of 139,072 stated¹ |

¹ recovered by a pure-Python partial unpickler that keeps the prefix; the count lands on
a round `SETITEMS` batch boundary and exceeds the stats file's stale figure, so it is a
prefix of a later, larger flush. Reported at 210.45 concentration as a control only, never
as a registered row.

So two of the three exclusions were wrong, and those two are the two largest resolved SMB
archives on disk — the exact counterexamples the corpus needed. This is the mechanical
cause of the whole finding: the corpus's resolved class was capped at ~20 k cells by a read
error that does not reproduce, and `CONCENTRATION_GATED_MIN` was fitted inside that cap.

---

## 9. The PASS clause: is there *any* statistic that ranks the hall above all four?

§7 allows a PASS if "some statistic ranks the CV hall ABOVE all four". Every statistic the
module computes, plus every derived quantity available from a banked archive, was scanned.
"Hall" is read as the class (all five reads), and a candidate must also survive the wider
resolved set — the four registered archives **plus** the calibration corpus's own resolved
rows, since a statistic that only beats the four while losing to `ge_1_4_solve` has
separated nothing.

| statistic | hall min | registered-four max | beats the four? | beats **all** resolved? | killed by |
|---|---|---|---|---|---|
| `concentration` (**shipped**) | 31.04 | 270.03 | **no** | no | `smb_3-3` 270.03 |
| `alias_ratio` (band-24 local concentration) | 45.98 | 387.49 | no | no | `smb_3-3` 387.49 |
| `explored_fraction` | 0.6480 | 0.0931 | *yes* | **no** | `ge_chain/lvl_11_4-4` 0.9986, `ge_1_4_solve` 0.9327, `cv_chain_hw/lvl_02` 0.7529 |
| `boundary_visit_entropy` | 0.9697 | 0.9504 | *yes* | **no** | `ge_1_2_solve` 1.0000, `ge_1_4_solve` 0.9999 — and already `BOUNDARY_ENTROPY_IS_SEPARATING = False` |
| `distinct_positions` in band | 297 | 200 | *yes* | *yes* | **grid artifact** — see below |
| `raw_coverage_saturation` | 0.179 | 0.453 | no | no | already `REFUTED-OFFLINE` |
| `map_stall_windows` | 11 | 17 | no | no | already `REFUTED-OFFLINE` |
| `churn_per_window` | 0.0019 | 0.0456 | no | no | already `REFUTED-OFFLINE` |
| `spatial_span` | 94 | 383 | no | no | `smb_8-1` 383 |
| `boundary_cells` (either sign) | 5 / −21 | 240 / −15 | no | no | both directions |
| `distinct_spatial` | 932 | 2,220 | no | no | `smb_8-1` 2,220 |
| `live_state_axis_count` | 1 | 4 | no | no | inverted — see below |
| `concentration` / M archive records | 8.0 | 342.4 | no | no | fully inverted (§5) |
| `cells` / M archive records | 8,707 | 298,424 | no | no | inverted |
| `concentration` × span | 2,918 | 37,648 | no | no | `smb_3-3` |
| `concentration` / span | 0.330 | 3.699 | no | no | `smb_3-3` |
| `alias_ratio` / `concentration` | 1.126 | 1.435 | no | no | `smb_3-3` |
| `distinct_spatial` / span | 9.91 | 7.63 | *yes* | **no** | `cv_chain_hw/lvl_02` 11.28 |
| `band_cells` | 13,655 | 77,497 | no | no | `smb_3-3` |
| `cells` | 28,929 | 218,218 | no | no | `smb_8-1` |

Three statistics order the hall above the registered four. All three are inadmissible:

* **`explored_fraction`** and **`boundary_visit_entropy`** are refuted the moment the
  calibration corpus's own resolved runs are re-included — by `ge_chain/lvl_11_4-4`
  (0.9986 explored, a *resolved* SMB 4-4 archive) and `ge_1_2_solve` (1.0000 entropy).
  `boundary_visit_entropy` is additionally already shipped as `REFUTED-OFFLINE`.
* **`distinct_positions` in the pinned band** is the only statistic that orders the hall
  above everything, and it is pure discretization geometry: the hall's band contains
  24 gx buckets × ~13 y-bands ≈ 308 triples **because the hall ran at `gx_bucket 8` and a
  finer y band**. Re-projected onto the comparators' grid (§6.1) the hall's whole-archive
  footprint is 213 — the *smallest* in the set — and the ordering inverts. It measures room
  shape and bucket size, not the search, and would score a tall unsolved room and a tall
  solved room identically.
* **`live_state_axis_count`** separates in the *opposite* direction (hall 1, comparators
  2 – 4, with `interaction_blind = True` for three of the five hall reads and `False` for
  all four comparators). That is real and it is worth keeping — but it says *"our cell key
  could not have recorded an interaction here"*, i.e. it supports the §0 interaction-blind
  thesis and the KEY_BLIND family, not saturation. It is not a GATED statistic and cannot
  be cited as one. (`lvl_03_overnight` reads 3 live state axes, so it does not hold across
  the hall either.)

There is a further, structural reason not to accept a PASS from this scan even if one
candidate had survived: selecting a separator by sweeping ~20 statistics against a positive
class of **one wall seen five times**, with no held-out positive, is the same circularity
§0-repair-2 built the tuner/grader wall to prevent. A statistic promoted this way would
need its own pre-registration, its own separating band, and a second wall — none of which
exist.

**PASS clause: not satisfied.**

---

## 10. Scope and what this does not say

* **It does not say the Castlevania hall is coverage-limited.** It says the evidence
  offered for "gated" is a statistic that cannot tell gated from solved. The hall's real,
  unchanged facts remain: five runs, ~10.7 h, ~77 M steps, best score pinned at 767, zero
  crossings, zero solutions. `UNRESOLVED-CONCENTRATED` is the honest label for that.
* **It does not invalidate the module.** The negative half of the calibration survives
  intact and is untouched here: the three `REFUTED-OFFLINE` statistics (§3 of the
  calibration), `KEY_BLIND` (which rests on Bubble Bobble ground truth, not on the hall),
  the segment-splitting adapter, and the `STAGNANT ≠ plateaued` fix. `SPATIAL_SPAN_MIN`,
  `C_LOCAL_FLOOR_BUCKETS`, `COVERAGE_FLOOR_CELLS`, `FROZEN_WINDOWS_MAX` and
  `EFFORT_MIN_STEPS` are not implicated. What falls is `CONCENTRATION_GATED_MIN` and the
  `GATED` verdict that only it can produce.
* **It does not license Contra.** `r1_ortho` at 43.60 reads GATED under the same broken
  statistic, which is one more reason §4d's decline-to-arm stands. Contra remains UNTYPED.
* **`lvl_1-3`'s ground truth is "resolved by retry", not "resolved in this segment".**
  The archive scored is attempt 3's, which banked no solution; the clear came on attempt 4.
  That is what makes it the sharpest false positive (§7) and it is stated rather than
  smoothed: on the corpus's own labelling convention this row is a COVERAGE wall that
  resolved, the class the module is supposed to send back to "give it more wall-clock".
* **`lvl_4-3` (210.45) is a partial read** and is a control row only, never a registered
  one. Excluding it changes nothing.
* **One wall, five reads.** The positive class is still `lvl_03_trace` seen five times.
  This receipt does not add a positive; it removes the statistic that was standing in for
  one.

---

## 11. Reproduction

```
archives read : 20 distinct (37.7 GiB of archive.pkl)   total read time 7.8 s
python        : .venv/bin/python
module        : src/training/wall_taxonomy.py @ CALIBRATED-OFFLINE-2026-08-10
bypass        : gated_wall_verdict() with the `solutions > 0` branch (:911) and the
                `topo_delta > 0 or map_delta > 0` branch (:916) deleted; every other
                branch, constant and helper unchanged
archive read  : wall_taxonomy._StateDroppingUnpickler
progress read : wall_taxonomy.load_progress_segments (segment named per run in §7)
concentration : wall_taxonomy.summarize_archive_cells(...).concentration
band profile  : wall_taxonomy.boundary_axis_profile(cells, band=24, bookkeeping=(4,5))
grid parity   : coarse_key = key[:-2] + (key[-2]//4, key[-1]//2)   # gx8/y8 -> gx16/y32
bucket derive : max_gx_in_max_area at the flush record / max(key[-1]) over the archive
```

Nothing was written to `runs/`. The scoring driver was ephemeral (scratchpad); every
number above is recomputable from the six lines of `wall_taxonomy` API listed here.

---

## 12. Downstream edits this verdict requires

Owed, not done in this receipt (read-only scope):

1. `docs/proposals/GATE_OPENER_CAMPAIGN_2026-08-11.md` — strike `GATED-candidate` from the
   header wall class; relabel `UNRESOLVED-CONCENTRATED`; drop "saturated" and
   "tried ~135×" wherever they survive; §8-(vi)'s conditional becomes unconditional.
2. `src/training/wall_taxonomy.py` — `CONCENTRATION_GATED_MIN` must carry a
   `REFUTED-OFFLINE-2026-08-10` tag or the `GATED` branch must be removed. Leaving a
   `CALIBRATED-OFFLINE` tag on it is now a false provenance claim.
   `test_no_false_gated_anywhere_in_the_corpus` must be extended with `lvl_1-3` seg 2 and
   will fail; that failure is the finding, not a regression to paper over.
3. `docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md` — §2's truncation
   exclusion is wrong for `lvl_1-3` and `lvl_3-3` (§8); §5's band and §6's
   no-false-GATED claim are superseded; §9's falsifier ("if that arm finishes the hall by
   ordinary coverage … the positive class is empty") has been reached by a cheaper route
   than waiting for the arm.
4. HUD / FORGE#2 — no `GATED` vocabulary for any target; Castlevania reads
   `UNRESOLVED-CONCENTRATED`, Contra stays UNTYPED, Bubble Bobble stays
   instrument-control.
5. The gate-opener arm proceeds unchanged. Its §2 primary endpoint is discovery-grade
   (surviving candidate + clean K1 sham null + shadow_yield > 0), which never depended on
   the label, and §0's thesis is already written to stand without it.
