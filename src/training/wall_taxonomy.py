"""Wall taxonomy over solver telemetry — offline, runtime-inert.

WHAT THIS IS
------------
Pure statistics over solver telemetry, plus a classifier that says what
— if anything — they license about a search that has stopped making
progress.

WHAT IT NO LONGER IS
--------------------
It no longer answers "is this wall GATED?", i.e. "has the search
saturated everything it can reach, so that something in the game must be
withholding the next transition?". That question had a shipped answer —
coverage concentration over a calibrated threshold — and two receipts
killed it:

  * docs/receipts/dispatch/k_falsifier_2026-08-10.md — the shipped
    `concentration` band does not separate the unsolved Castlevania hall
    from archives that were SOLVED. `ge_chain_w8/lvl_00_8-1` solved at
    concentration 98.30, 3.2x the "gated" upper bracket, inside one
    chain whose two solved siblings read 15.66 and 20.58.
  * docs/receipts/dispatch/size_decoupled_statistic_2026-08-11.md — a
    search for a replacement: 22 candidates over 103 archives, including
    two devised here that pass a same-chain sibling test cleanly. ALL
    STRADDLE. The least-bad cut available anywhere in the set (new cells
    per selection at 0.7778) still condemns 3 of 13 resolved archives as
    walls; the shipped statistic's own best cut condemns 4.

So `WallClass.GATED` is GONE — REMOVED, per §12.1 of the second receipt,
rather than re-thresholded, because after that search there is no
constant left to re-tag into correctness. Where it used to fire the
module now ABSTAINS (`INDETERMINATE`, whose remedy "collect the missing
telemetry" is now literally correct) and attaches the descriptive,
non-verdict label `UNRESOLVED_CONCENTRATED`: a statement about what was
OBSERVED — the archive piled cells into a map footprint that stopped
growing, and the search has not resolved — never about why, and never a
licence to spend an orthogonal campaign. `UNRESOLVED-CONCENTRATED` is
also the vocabulary the K-falsifier's §12.4 requires downstream: no
GATED wording for any target.

Note what this does NOT say. It does not say the Castlevania hall is
coverage-limited. The hall's facts are unchanged — five runs, ~10.7 h,
~77 M steps, best score pinned, zero crossings, zero solutions. It says
only that no statistic yet found tells a search that is stuck from one
that is about to finish, which is exactly the 8-4 lesson:
`runs/live_show/smb_4_4_micro/lvl_8-4` looked identically stuck for 44
straight minutes and then simply finished.

WHAT STILL STANDS
-----------------
The subtractive and representational half of the form, which neither
receipt disturbed (§14 enumerates what it leaves untouched):

    BARREN    <=>  local coverage STAGNANT (C_local never accumulated)
    KEY_BLIND <=>  the cell key's spatial projection is degenerate, so
                   no coverage statistic computed over it means anything

together with `SPATIAL_SPAN_MIN`, `C_LOCAL_FLOOR_BUCKETS`,
`COVERAGE_FLOOR_CELLS`, `FROZEN_WINDOWS_MAX`, `EFFORT_MIN_STEPS`, the
segment-splitting progress adapter, and the STAGNANT-is-not-plateaued
fix. All 13 resolved archives in the second receipt's §3 clear every one
of them, so none is propping up a result.

This module is explicit — in `MISSING_TELEMETRY` and in every constant's
provenance tag — about which terms are measured, which are substituted,
and which were refuted outright.

RUNTIME STATUS: INERT for classification. Nothing imports the verdict
path; the one tolerated runtime reader takes the pure
`boundary_axis_profile` and nothing else. Self-arming dispatch is a
later, separate decision, and there is now nothing to arm.

RECEIPTS
    docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md
        the original corpus and bands — SUPERSEDED in its §5/§6/§9
    docs/receipts/dispatch/k_falsifier_2026-08-10.md
        the falsifier that struck `concentration`
    docs/receipts/dispatch/size_decoupled_statistic_2026-08-11.md
        the replacement search that returned nothing; §12 is the spec
        this module's current shape implements
    docs/proposals/GATE_OPENER_CAMPAIGN_2026-08-11.md
        §12, the D3 lineage verdict that struck `doors` from topo_delta
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Stamped into every verdict so a receipt can never be read as if it came
# from a differently-tuned build. BUMPED 2026-08-11: verdicts moved when
# the GATED branch was removed, so a JSON produced before that date must
# not be mistaken for one produced after it.
CALIBRATION_TAG = "CLASSIFICATION-STRUCK-2026-08-11"

#: Descriptive, NON-VERDICT label that replaces the removed `GATED`
#: vocabulary. It names what was observed — an unresolved search whose
#: archive is piling cells into a map footprint that stopped growing —
#: and asserts nothing about why, because after 22 candidates over 103
#: archives nothing separates that observation from a resolved coverage
#: wall (size_decoupled_statistic_2026-08-11.md §0, §9.3, §14). It rides
#: on `WallVerdict.descriptor`, never on `wall_class`, so no caller can
#: switch on it as if it were a diagnosis.
UNRESOLVED_CONCENTRATED = "UNRESOLVED-CONCENTRATED"

#: Pointer stamped into the reasons of every verdict that carries the
#: label above, so a pasted verdict always arrives with its refutation.
STRUCK_CLASSIFICATION_RECEIPT = (
    "docs/receipts/dispatch/size_decoupled_statistic_2026-08-11.md"
)


# --------------------------------------------------------------------------
# Constants.
#
# Every constant below carries its provenance:
#   CALIBRATED-OFFLINE  -- a separating band was measured on the banked
#                          corpus; the comment names the band and the two
#                          runs that bracket it.
#   PROVISIONAL         -- the term is in the adopted form but no banked
#                          run can measure it; the value is a placeholder
#                          and the verdict says so via `degraded`.
#   REFUTED-OFFLINE     -- a candidate statistic that does NOT separate
#                          the corpus. Kept as a reported diagnostic so
#                          nobody re-derives it and believes it.
#   DESCRIPTIVE         -- decides only the wording of a reported label.
#                          Gates nothing; moving it moves no `wall_class`
#                          (asserted by test).
# --------------------------------------------------------------------------

#: Trailing window, in progress records. The fleet's progress cadence is
#: 60 s (scripts/go_explore_solve.py:progress_line), so 10 records is a
#: 10-minute window — long enough that one lucky burst cannot move it,
#: short enough that an 8-4-style breakthrough is not averaged away.
WINDOW_RECORDS = 10  # CALIBRATED-OFFLINE-2026-08-10

#: A verdict needs the window plus a couple of records of context.
MIN_RECORDS = 12  # CALIBRATED-OFFLINE-2026-08-10

#: Minimum emulator steps inside the window before any wall verdict is
#: allowed. Separating band: 88,680 (SMB 4-4 segment 0, a throughput-
#: starved show segment that must NOT be classified) .. 932,340 (SMB 8-4,
#: the least-effort run that must be).
EFFORT_MIN_STEPS = 250_000  # CALIBRATED-OFFLINE-2026-08-10

#: Below this many archive cells the search never accumulated coverage at
#: all. Separating band: 96 (BB r68 ortho, frozen) .. 605 (BB r68 xsig,
#: the smallest live archive in the corpus).
COVERAGE_FLOOR_CELLS = 256  # CALIBRATED-OFFLINE-2026-08-10

#: Consecutive 60 s windows with zero new cells, as counted by the
#: solver's own stall watchdog (`stall_flat_windows`). Separating band:
#: 7 (BB r99 retry, alive but slow) .. 19 (BB r69 ortho, frozen).
FROZEN_WINDOWS_MAX = 12  # CALIBRATED-OFFLINE-2026-08-10

#: Distinct gx buckets discovered inside the deepest area. Under this the
#: cell key's spatial projection is degenerate: the key cannot see the
#: axis the problem lives on, so no coverage statistic computed over it
#: means anything. Separating band: 1 (every Bubble Bobble run — single
#: screen) .. 94 (Castlevania hall). SMB levels sit at 130-230.
SPATIAL_SPAN_MIN = 8  # CALIBRATED-OFFLINE-2026-08-10

#: Coverage concentration = cells / distinct spatial buckets. Shipped
#: 2026-08-10 as the cross-sectional stand-in for "C_local has
#: plateaued", inside a 1.51x separating band (20.58 SMB 8-3 .. 31.04 CV
#: hall) that was the thinnest margin in the module. STRUCK.
#:
#: The band was an artifact of a four-archive negative class. Widened,
#: `ge_chain_w8/lvl_00_8-1` SOLVED at 98.30 — 3.2x the upper bracket —
#: inside one chain, one profile, one `--gx-bucket 16` grid whose other
#: two solved levels read 15.66 and 20.58, a 6.3x intra-chain spread that
#: straddles the threshold on its own (k_falsifier §6, §9.1). The
#: mechanism is that concentration is archive SIZE wearing a hat:
#: Spearman(conc, cells) = +0.940, and `distinct_spatial` is bounded by
#: map geometry, so the ratio mostly counts how long the run was left
#: alive. The best cut that captures all five hall reads (31.04) condemns
#: 4 of 13 resolved archives as walls (§9.3).
CONCENTRATION_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: Above this the word "concentrated" is honest in the DESCRIPTIVE label
#: `UNRESOLVED_CONCENTRATED`. Inherited from the struck calibration, kept
#: only so the label's English is not a lie; it partitions no verdict.
#: `gated_wall_verdict` returns the same `wall_class` for any value of
#: this constant — the abstention above it and the abstention below it
#: are the SAME abstention — which is the whole difference between a
#: description and the gate that used to live here.
CONCENTRATION_DESCRIPTIVE_MIN = 25.0  # DESCRIPTIVE-2026-08-11 (gates nothing)

#: Saturation of a true C_local time series, when one exists. Defined
#: exactly like the coverage-saturation statistic but on the count of
#: distinct spatial buckets rather than on raw cells. No banked run emits
#: that series, so this value is NOT calibrated; it is the value the
#: runtime version should re-derive first.
#:
#: It survives the 2026-08-11 strike because of the DIRECTION it is used
#: in. It can only REFUTE a plateau (-> COVERAGE_LIMITED, "the footprint
#: is still expanding"), name STAGNANT (-> BARREN) and name a blind key
#: (-> KEY_BLIND). It can no longer promote anything, because there is
#: nothing left to promote to: the branch a corroborated plateau used to
#: certify is gone.
C_LOCAL_SATURATION_MIN = 0.85  # PROVISIONAL-2026-08-10 (no offline series)

#: Absolute floor on a reported `c_local` before ANY C_local statistic
#: over it is allowed to gate. `c_local` is defined as
#: |{(area, y_band, gx_bucket)}| — the same column the §4 archive table
#: calls `distinct_spatial` — so the corpus measures it directly:
#: separating band 32 (BB 99-1 boss retry, the largest degenerate
#: projection) .. 638 (`ge_1_4_solve`, the smallest spatially resolved
#: archive). This is the series-path twin of SPATIAL_SPAN_MIN. Without
#: it, a single-screen profile that emits `c_local` and flushes no
#: archive escaped the KEY_BLIND verdict the identical run WITH an
#: archive got — a divergence measured before this floor existed, and
#: still worth closing now that the escape lands in an abstention.
C_LOCAL_FLOOR_BUCKETS = 64  # CALIBRATED-OFFLINE-2026-08-10

#: Saturation computed on RAW archive cells. REFUTED: on the corpus the
#: Castlevania hall (0.343) sits BETWEEN two resolved coverage
#: walls, SMB 8-4 (0.190) and SMB 4-4 (0.352). No threshold separates it
#: in either direction. Raw cell count is not C_local, because nuisance
#: dimensions in the cell key manufacture novelty forever at a fixed
#: location. Reported, never gated on.
RAW_COVERAGE_SATURATION_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

#: Per-window churn (new cells / archive size). REFUTED: BB 99-1 day-2h
#: (0.00024, a live archive) churns an order of magnitude LESS than BB
#: r69 ortho (0.01667, a frozen 48-cell archive that ticked once), so the
#: ordering is inverted against the barren ground truth. Reported only.
CHURN_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

#: Normalized Shannon entropy of visit mass across boundary cells — the
#: only offline stand-in available for "high action-entropy at the
#: boundary". REFUTED as a gate: every class in the corpus scores >= 0.77
#: (CV hall 0.984, barren BB r68 ortho 0.778, resolved SMB 1-4 0.9999).
#: It measures how evenly returns were spread, not how varied the actions
#: were. Reported only.
BOUNDARY_ENTROPY_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

#: Consecutive windows at zero permanent-map delta. On the 2026-08-10
#: corpus the hall (81 windows) did separate from resolved SMB 8-4 (44)
#: and SMB 4-4 (25) — but the statistic is a pure function of how long
#: the run was left alive, so a longer 8-4 would have crossed any fixed
#: threshold. Deliberately NOT shipped as a gate; reported so a human can
#: see the horizon a verdict was taken over.
MAP_STALL_WINDOWS_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

# ---- The 2026-08-11 replacement search. Six candidates, all dead. -------
#
# size_decoupled_statistic_2026-08-11.md scored 22 candidate statistics
# over 103 archives looking for anything that could take the struck
# `concentration` branch's place. Nothing did. Each constant below names
# the offender that killed it, the way the block above does, so that the
# next attempt starts after this search rather than inside it.
#
# Definitions of the two classes used throughout: POSITIVE = the five
# reads of the unsolved Castlevania hall; NEGATIVE = 13 archives with a
# banked solution. A candidate STRADDLEs when at least one negative
# falls inside the positives' range.

#: `concentration` residualized on `log cells` — the principled fix for
#: "the statistic is size wearing a hat". REFUTED, and by algebra before
#: measurement: because conc == cells / distinct_spatial by definition,
#: the residual reduces to -log(distinct_spatial) plus a vanishing
#: 0.0773 * log cells term (measured Pearson vs -log ds = +0.9066). And
#: distinct_spatial is bounded by map geometry, so the hall's 932..1102
#: sits in the MIDDLE of the resolved 435..2220 — `ll_1_1_transfer`
#: (1114) is one bucket from `cv_chain_hw/lvl_03_overnight` (1102). The
#: candidate succeeds completely at decoupling size (rho +0.940 ->
#: +0.074) and the classes still overlap; best cut condemns 6/13. §6.
SIZE_PARTIALED_CONCENTRATION_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: nu = cells / records — the fraction of recorded observations that
#: created a new cell, read free from `archive.stats.json`. The best
#: candidate found: archive-intrinsic, no pacing or segment-pairing
#: confound, and it passes the same-chain sibling test cleanly (holds the
#: hall inside 1.9x across a 19x range in cells while collapsing
#: `ge_chain_w8`'s 6.3x intra-chain concentration spread to 1.5x).
#: REFUTED anyway: `ge_1_2_solve` SOLVED at 0.01073, inside the hall's
#: 0.0087..0.016476 band, with `cv_chain_hw/lvl_02` 0.00855,
#: `ge_1_4_solve` 0.00624 and `ge_chain/lvl_11_4-4` 0.00177 below it.
#: The mechanism is the trap for the next attempt: a run that solves does
#: not stop recording — EXPLORE_AFTER_FIRST_CLEAR keeps re-treading, so a
#: resolved archive buries its discovery phase and reads as stuck.
#: Truncating at the solution does not rescue it (§8). §9.2, §10.
NOVELTY_PER_RECORD_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: Fraction of cells ever chosen as a return target. REFUTED, and this
#: one pre-empts its whole family: `Cell.explored` flips on selection, so
#: under a structureless null in which T selections fall over N cells,
#: E[explored_fraction] = 1 - exp(-T/N). Measured across all 19 scored
#: archives, Pearson(observed, that null) = +0.9906. It is
#: selections-per-cell and essentially nothing else — the identical
#: size/effort defect that killed `concentration`, wearing a different
#: hat. It "ranked the hall above the registered four" only because the
#: hall ran 1.3-2.6 selections per cell against their 0.02-0.20.
#: `ge_chain/lvl_11_4-4` resolved at 0.9986, above every hall read; the
#: principled residual (observed - null) straddles too, `ge_1_4_solve`
#: (+0.0225) above the hall and `smb_4_4_micro/lvl_1-3` (-0.0957) below.
#: Reported by `ArchiveSummary.explored_fraction`, never gated on. §7.
EXPLORED_FRACTION_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: exp(H) / D over the archive's D occupied spatial buckets — normalized
#: effective support of the visit-mass distribution. Size-free by
#: construction and measurably so (rho = -0.019, the best decoupling in
#: the whole set), and it passes the same-chain sibling test. REFUTED
#: anyway: `cv_chain_hw/lvl_02` SOLVED at 0.204, inside the hall's
#: 0.115..0.211 band, with `ge_1_2_solve` 0.032, `ll_1_1_transfer` 0.043
#: and `ge_1_4_solve` 0.054 below it. Same post-clear re-treading
#: mechanism as nu. Best cut condemns 4/13. §9.2.
SPATIAL_EVENNESS_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: Log-log exponent of cells vs steps over the trailing half, and its
#: three curvature variants. REFUTED, and the registered intuition —
#: "resolved archives keep growing to solution, walls flatten" — is
#: BACKWARDS. The hall does not flatten: `cv_chain_hw2/lvl_03_trace` has
#: the HIGHEST tail exponent in the corpus (0.899), above
#: `ge_chain_w8/lvl_00_8-1` (0.889), while the two flattest curves
#: measured, `ge_chain/lvl_11_4-4` (0.023) and `ge_1_4_solve` (0.068),
#: both RESOLVED. Nuisance dimensions in the cell key manufacture novelty
#: forever at a fixed location: the hall grew 3,354 -> 91,995 cells with
#: max_gx_in_max_area pinned for 84 consecutive records. No statistic
#: computed on that curve can catch a map that is frozen underneath it.
#: §8(d).
GROWTH_EXPONENT_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: Rank of `concentration` among archives within +-0.5 dex of log cells —
#: "compare it only to runs of its own size". REFUTED, worst-performing
#: candidate but one: 10 of 11 scorable resolved archives land inside the
#: hall's band. Mechanical, not accidental: the falsifier's +0.929 IS the
#: statement that concentration is a monotone function of cells, so
#: conditioning on cells conditions away the statistic. Percentile-
#: within-band converts a size clock into a coin flip, not into a wall
#: detector. §8(c).
EFFORT_MATCHED_PERCENTILE_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-11

#: Is the solver's `doors` counter a monotone progress counter? NO.
#: `doors` counts ARTICULATION POINTS in the discovered room graph, which
#: is a connectivity property of a graph that keeps being rewritten, not
#: a ratchet: a newly discovered edge can turn an articulation point into
#: an ordinary node, and re-exploration churns the count in both
#: directions. Adding it to `topo_delta` therefore reads "topology moved"
#: off pure churn. Measured on `runs/cv_chain_hw/lvl_03_overnight`, the
#: frozen Castlevania hall: over its trailing 10 records `doors` moved
#: 12,694 -> 12,880 while max_area, max_sect, max_room and max_gx were
#: all flat, and the module returned PROGRESSING for a search that had
#: not moved the map in 355 consecutive records. DROPPED from topo_delta
#: 2026-08-11; still reported as `doors_delta`.
#: GATE_OPENER_CAMPAIGN_2026-08-11.md §12 (D3 adjudication).
DOORS_IS_MONOTONE = False  # REFUTED-OFFLINE-2026-08-11

#: Cell-key positions that `_spatial_key` projects onto: (area, y_band,
#: gx_bucket). Negative, so they stay correct if a future arm grows the
#: key's PREFIX — every adapter's `cell_fn` output is the last five
#: elements and `scripts/go_explore_solve.py` indexes `key[-5]`/`key[-1]`
#: on exactly that promise. Read by `boundary_axis_profile` to decide
#: which axes are positional and which are not.
SPATIAL_KEY_POSITIONS = (-5, -2, -1)

#: Non-spatial axes that must carry at least TWO distinct values at the
#: boundary before "the search has tried every interaction" is even a
#: sayable claim, as opposed to "our cell key cannot represent an
#: interaction, so we cannot know". OBSERVED, NEVER GATED ON: nothing in
#: `gated_wall_verdict` reads this, and it is not calibrated against a
#: labelled corpus. It exists because the measurement that motivated it
#: is stark — `runs/cv_hall_ortho_a`, 131,561 cells at concentration
#: 120.04, the highest in the corpus, has SIX of eleven key positions
#: CONSTANT in its pinned band (sect, time-bin, kill-count, room-sig,
#: area, boss-HP all identically 0/empty), leaving exactly ONE game-state
#: axis (the on-stairs bit) against two trajectory-bookkeeping axes (loop
#: count, route signature). A search whose memory has one state bit
#: cannot have "tried every interaction"; it can only have tried every
#: position. See docs/proposals/gate_opener_arm_2026-08-11.md.
#:
#: NOT a wall statistic, and now measured not to be one: scored across 13
#: resolved archives, `live_state_axis_count` is the WORST candidate in
#: the 2026-08-11 set — hall [3, 1, 1, 1, 1] against resolved [0, 1, 1,
#: 1, 1, 1, 1, 2, 2, 2, 2, 3, 4], 11 of 13 inside the hall's range with
#: both tails populated. `alias_ratio` behaves like `concentration`
#: (rho +0.875) and dies to the same three SMB archives. It supports the
#: interaction-blind thesis and the KEY_BLIND family; that is all.
#: size_decoupled_statistic_2026-08-11.md §8(e).
BOUNDARY_STATE_AXES_MIN = 2  # OBSERVED-2026-08-10 (reported, not a gate)


#: Telemetry the form wants and the fleet does not emit. This is the
#: shopping list a runtime version needs the solver to add to
#: `progress_line()`; each entry names the field and why it matters.
#: These are SPECS, not work items filed against this module — nothing
#: here wires new solver telemetry, and the solver is not modified.
MISSING_TELEMETRY: dict[str, str] = {
    "c_local": (
        "count of DISTINCT spatial buckets in the archive — |{(area, "
        "y_band, gx_bucket)}| — per progress line. Without it, C_local "
        "saturation can only be approximated cross-sectionally from a "
        "final archive snapshot, which is one number instead of a curve "
        "and cannot see a plateau form. It is also the field that makes "
        "INDETERMINATE's remedy actionable: since 2026-08-11 the "
        "abstention is where every uncorroborated plateau lands."
    ),
    "boundary_action_entropy": (
        "Shannon entropy of the ACTION distribution actually sampled "
        "from cells at the frontier bucket, per progress line. Nothing "
        "banked records actions-per-cell, so the adopted form's "
        "entropy term is currently unmeasured; the visit-mass entropy "
        "substituted offline was refuted as a discriminator."
    ),
    "frontier_bucket_cells": (
        "cells sitting in the deepest reachable bucket, per progress "
        "line. Available only from an archive snapshot today, and the "
        "two SMB show runs in the corpus (4-4, 8-4) persisted no "
        "archive at all."
    ),
    "permanent_map_delta": (
        "an explicit monotone map-progress counter. `max_gx_in_max_area` "
        "is the current stand-in and it is game-shaped: every Bubble "
        "Bobble run pins it at a constant because the game is one "
        "screen, so the 'zero permanent-map delta' term carries no "
        "information there."
    ),
    "doors_cumulative": (
        "the monotone replacement for the `doors` counter dropped from "
        "topo_delta on 2026-08-11 — |{articulation points EVER seen}| "
        "rather than |{articulation points in the CURRENT room graph}|. "
        "A cumulative set is a ratchet and would be admissible where the "
        "live count is not (DOORS_IS_MONOTONE), and it is trivial where "
        "the graph already lives: union the articulation set into a "
        "run-scoped set instead of recomputing a size. SPEC ONLY. It is "
        "deliberately not wired here: this module reads telemetry and "
        "does not commission it, and the counter as emitted today stays "
        "reported-as-`doors_delta` until the solver emits the ratchet."
    ),
    "archive_snapshot_on_show_runs": (
        "runs/live_show/* writes progress.jsonl but no archive.pkl "
        "(flush_secs is set to ~forever), so the two ground-truth "
        "coverage walls can only be scored on the degraded path."
    ),
}


class WallClass(str, Enum):
    """Verdict labels, ordered from 'nothing to do' to 'act now'.

    There is NO `GATED` member and re-adding one is the mutation this
    module's tests exist to catch. It was removed on 2026-08-11 because
    no statistic — the shipped one or any of 22 candidates scored over
    103 archives — separates a search that is stuck from one that is
    about to finish. See the module docstring for the two receipts, and
    `UNRESOLVED_CONCENTRATED` for the descriptive label that replaced the
    vocabulary. What used to be certified here is now abstained on.
    """

    #: The search already produced a solution inside the window.
    RESOLVED = "resolved"
    #: Topology or the permanent map moved inside the window.
    PROGRESSING = "progressing"
    #: The map footprint is measurably STILL EXPANDING, so whatever else
    #: is true the search has not run out of reachable ground. Reachable
    #: only from a C_local series that refutes a plateau — never from a
    #: cross-sectional statistic, which cannot see a direction.
    COVERAGE_LIMITED = "coverage_limited"
    #: Coverage never accumulated: the archive is frozen or trivially
    #: small. The cell key, the reset, or determinism is broken.
    BARREN = "barren"
    #: The cell key's spatial projection is degenerate, so no coverage
    #: statistic over it is meaningful. Enrich the key first.
    KEY_BLIND = "key_blind"
    #: The search is stalled and nothing in this telemetry licenses
    #: saying why. The terminal class for every stalled run that is not
    #: barren and not key-blind, including every archive that would once
    #: have read GATED; `descriptor` then carries
    #: `UNRESOLVED_CONCENTRATED`.
    INDETERMINATE = "indeterminate"
    #: Not enough records or not enough compute spent to say anything.
    INSUFFICIENT = "insufficient"


#: What a human should do about each verdict. Advisory text only — this
#: module dispatches nothing.
REMEDY: dict[WallClass, str] = {
    WallClass.RESOLVED: "nothing — harvest the solution and move on",
    WallClass.PROGRESSING: "nothing — the frontier is still moving",
    WallClass.COVERAGE_LIMITED: "give it more wall-clock before changing anything",
    WallClass.BARREN: "fix the search, not the game: check the cell key, the "
                      "reset path, and determinism — the archive is frozen",
    WallClass.KEY_BLIND: "add the missing state axis to the cell key; the "
                         "current key cannot see where progress happens",
    WallClass.INDETERMINATE: "collect the missing telemetry (see "
                             "MISSING_TELEMETRY) before deciding — no "
                             "statistic here distinguishes a stalled search "
                             "from one about to finish, so the operator "
                             "decides on grounds this module does not supply",
    WallClass.INSUFFICIENT: "keep running; too little evidence to classify",
}


@dataclass(frozen=True)
class ProgressRecord:
    """One line of a solver `progress.jsonl`, normalized.

    Only fields the discriminator reads are kept. `c_local` is the
    forward-compatible slot for the distinct-spatial-bucket count the
    runtime version should emit; it is `None` for every banked run.
    """

    elapsed_s: int
    cells: int
    steps: int
    solutions: int = 0
    max_area: int = 0
    max_sect: int = 0
    max_gx: int = 0
    max_room: int = 0
    doors: int = 0
    stall_flat_windows: int = 0
    c_local: Optional[int] = None


@dataclass(frozen=True)
class ArchiveSummary:
    """Cross-sectional statistics over a final `archive.pkl` snapshot.

    Produced by `summarize_archive_cells` (pure, testable without a
    pickle) or `read_archive_summary` (reads the real file, dropping the
    multi-KB emulator-state blobs as it goes).
    """

    cells: int
    #: |{(area, y_band, gx_bucket)}| over the archive's keys.
    distinct_spatial: int
    #: Distinct gx buckets inside the deepest area — the discovered
    #: horizontal extent of the map, and the applicability test for every
    #: spatial statistic here.
    spatial_span: int
    #: Cells sitting in the deepest bucket of the deepest area.
    boundary_cells: int
    #: Normalized Shannon entropy of visit mass across those cells.
    #: Reported only (BOUNDARY_ENTROPY_IS_SEPARATING is False).
    boundary_visit_entropy: float
    #: Fraction of cells that have ever been chosen as a return target.
    #: Reported only, and pre-emptively so: it is a saturating function of
    #: selections-per-cell and nothing else (Pearson +0.9906 against the
    #: coupon-collector null), which is why the whole boundary family died
    #: — EXPLORED_FRACTION_IS_SEPARATING.
    explored_fraction: float = 0.0

    @property
    def concentration(self) -> float:
        """cells per distinct spatial bucket.

        Reported only since 2026-08-11: it is archive SIZE wearing a hat
        (Spearman +0.940 against `cells`) and it does not separate a wall
        from a search that solved — CONCENTRATION_IS_SEPARATING. It still
        decides the WORDING of the `UNRESOLVED_CONCENTRATED` descriptor,
        which is a description of the archive and not a claim about it.
        """
        return self.cells / max(1, self.distinct_spatial)


@dataclass(frozen=True)
class ArchiveCounters:
    """The `archive.stats.json` sidecar every banked archive carries.

    ~130 bytes next to a multi-GB `archive.pkl`, and it is the free
    effort denominator: `records` is exactly `Sigma visits` over every
    cell (`GoExploreArchive.total_records`; `Cell.visits` starts at 1 and
    increments on every re-record), verified to the unit against direct
    archive reads on three runs. `frontier` is the UNEXPLORED count, so
    `explored_fraction` falls out without unpickling anything.

    Why it matters even though it gates nothing: an archive-intrinsic
    denominator has no segment-pairing ambiguity. `ge_chain_w8/
    lvl_00_8-1`'s 8,269,310 "total steps" is a cross-segment figure
    spanning two attempts while the archive itself recorded 731,234 —
    the hazard the K-falsifier's §7 had to audit by hand.

    REPORTING ONLY. `cells / records` is the `nu` statistic and
    `explored_fraction` is the boundary statistic; both were scored and
    both STRADDLE (NOVELTY_PER_RECORD_IS_SEPARATING,
    EXPLORED_FRACTION_IS_SEPARATING). Nothing here may reach a branch.
    size_decoupled_statistic_2026-08-11.md §2.2, §12.3.
    """

    cells: int
    records: int
    new_cells: int
    improvements: int
    frontier: int
    best_score: float
    #: Present on newer sidecars only (72 of 279 banked): hw flags, frame
    #: skip and the nes_core build hash. Carried through untouched
    #: because a lineage mismatch is what D3 caught by hand.
    hw_provenance: Optional[Mapping[str, Any]] = None

    @property
    def novelty_per_record(self) -> float:
        """`nu` — the fraction of recorded observations that were new."""
        return self.cells / max(1, self.records)

    @property
    def explored_fraction(self) -> float:
        """1 - frontier/cells, the sidecar's view of the same column
        `ArchiveSummary.explored_fraction` computes from the pickle."""
        return 1.0 - (self.frontier / max(1, self.cells))


@dataclass(frozen=True)
class WallTelemetry:
    """Everything the classifier is allowed to look at."""

    records: tuple[ProgressRecord, ...]
    archive: Optional[ArchiveSummary] = None
    label: str = ""
    #: The sidecar, when it was read. Surfaced in `evidence` so a verdict
    #: shows the effort it was taken over; never read by a branch.
    counters: Optional[ArchiveCounters] = None


@dataclass(frozen=True)
class WallVerdict:
    wall_class: WallClass
    label: str
    #: True when the verdict was reached without the evidence the adopted
    #: form actually calls for. Always True on the progress-only path.
    degraded: bool
    #: Telemetry the adopted form wanted and this call did not have.
    missing: tuple[str, ...]
    #: Human-readable trace of which tests fired, in order.
    reasons: tuple[str, ...]
    #: Every statistic computed, whether or not it gated anything.
    evidence: Mapping[str, Any] = field(default_factory=dict)
    remedy: str = ""
    #: A DESCRIPTIVE, non-verdict annotation — today only
    #: `UNRESOLVED_CONCENTRATED`, and "" whenever there is nothing to
    #: describe. It rides beside `wall_class` rather than inside it
    #: precisely so that no caller can dispatch on it: it says what the
    #: archive LOOKS like, and the receipt it cites says that what an
    #: archive looks like does not predict whether the wall will fall.
    descriptor: str = ""
    calibration: str = CALIBRATION_TAG

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_class": self.wall_class.value,
            "label": self.label,
            "degraded": self.degraded,
            "missing": list(self.missing),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "remedy": self.remedy,
            "descriptor": self.descriptor,
            "calibration": self.calibration,
        }


# --------------------------------------------------------------------------
# Telemetry adapters
# --------------------------------------------------------------------------

def record_from_json(obj: Mapping[str, Any]) -> ProgressRecord:
    """Normalize one `progress.jsonl` object.

    Tolerant by design: the fleet's progress line has grown fields over
    time (`stall_flat_windows` arrived 2026-08-06; `max_room`/`doors`
    only appear when the relevant arms are on), and older banked runs
    predate all of them.

    `doors` is still parsed. It no longer counts toward `topo_delta`
    (DOORS_IS_MONOTONE) — it is reported as `doors_delta` instead.
    """
    return ProgressRecord(
        elapsed_s=int(obj.get("elapsed_s", 0)),
        cells=int(obj.get("cells", 0)),
        steps=int(obj.get("steps", 0)),
        solutions=int(obj.get("solutions") or 0),
        max_area=int(obj.get("max_area", 0)),
        max_sect=int(obj.get("max_sect", 0)),
        max_gx=int(obj.get("max_gx_in_max_area", 0)),
        max_room=int(obj.get("max_room", 0)),
        doors=int(obj.get("doors", 0)),
        stall_flat_windows=int(obj.get("stall_flat_windows", 0)),
        c_local=(int(obj["c_local"]) if obj.get("c_local") is not None
                 else (int(obj["distinct_spatial"])
                       if obj.get("distinct_spatial") is not None else None)),
    )


def load_progress_segments(path: str | Path) -> tuple[tuple[ProgressRecord, ...], ...]:
    """Read a `progress.jsonl` and split it into per-attempt segments.

    A single progress.jsonl can hold SEVERAL runs appended end to end —
    the file is opened in append mode and re-used when a level is
    retried. `runs/live_show/smb_4_4_micro/lvl_4-4/progress.jsonl` holds
    five such attempts, and reading it as one series makes `max_gx`
    non-monotone and the cell curve saw-toothed. Split wherever
    `elapsed_s` goes backwards; the LAST segment is the live one.
    """
    segments: list[list[ProgressRecord]] = []
    prev = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = record_from_json(json.loads(line))
        if prev is None or rec.elapsed_s < prev:
            segments.append([])
        segments[-1].append(rec)
        prev = rec.elapsed_s
    return tuple(tuple(s) for s in segments)


def _spatial_key(key: Any) -> tuple[int, int, int]:
    """Project an archive cell key onto (area, y_band, gx_bucket).

    The fleet's key is `(sect, tb, kk, psig, loops, route_sig) +
    game.cell_fn(ram)` and every adapter's `cell_fn` ends with
    `(..., y // Y_BAND, progress // GX_BUCKET)` with `area` five from the
    end — scripts/go_explore_solve.py relies on exactly that layout for
    its own selection caches (`key[-5]` / `key[-1]`). Archives banked
    before the sect/psig prefix landed carry a bare 4-tuple, so `area`
    falls back to the first element there.
    """
    area = key[-5] if len(key) >= 5 else key[0]
    return int(area), int(key[-2]), int(key[-1])


def summarize_archive_cells(cells: Mapping[Any, Any]) -> ArchiveSummary:
    """Cross-sectional archive statistics. Pure: any mapping of cell key
    to an object exposing `.visits` / `.explored` will do, so tests need
    no pickle and no emulator."""
    n = len(cells)
    if n == 0:
        return ArchiveSummary(0, 0, 0, 0, 0.0, 0.0)

    spatial: set[tuple[int, int, int]] = set()
    max_area = None
    for key in cells:
        area, y_band, gx = _spatial_key(key)
        spatial.add((area, y_band, gx))
        if max_area is None or area > max_area:
            max_area = area

    deep_gx = {gx for area, _y, gx in spatial if area == max_area}
    span = len(deep_gx)
    frontier_gx = max(deep_gx) if deep_gx else 0

    boundary = [c for k, c in cells.items()
                if _spatial_key(k)[0] == max_area and _spatial_key(k)[2] == frontier_gx]
    visits = [max(1, int(getattr(c, "visits", 1) or 1)) for c in boundary]
    entropy = _normalized_entropy(visits)
    explored = sum(1 for c in cells.values() if getattr(c, "explored", False))

    return ArchiveSummary(
        cells=n,
        distinct_spatial=len(spatial),
        spatial_span=span,
        boundary_cells=len(boundary),
        boundary_visit_entropy=entropy,
        explored_fraction=explored / n,
    )


def _normalized_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy of `counts` as a distribution, divided by log(n).

    1.0 = perfectly even, 0.0 = one bin takes everything (and by
    convention for a single bin, which carries no information)."""
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return h / math.log(len(counts))


class _StateDroppingUnpickler(pickle.Unpickler):
    """Unpickle an archive without retaining its emulator-state blobs.

    A banked archive is mostly `Cell.state` — 21-45 KB of opaque save
    state per cell, which is 1.97 GB for the Castlevania hall. Only the
    keys and a few scalars matter here, so `Cell` is swapped for a slot
    class that drops `state` on `__setstate__`: the blob is still
    materialized transiently by the pickle machinery but freed
    immediately, keeping peak RSS proportional to the cell COUNT rather
    than to the file size.
    """

    def find_class(self, module: str, name: str):  # noqa: D102
        if name == "Cell":
            return _LiteCell
        return super().find_class(module, name)


class _LiteCell:
    __slots__ = ("best_score", "best_steps", "visits", "times_chosen", "explored")

    def __init__(self, *_a, **_k) -> None:
        for slot in self.__slots__:
            setattr(self, slot, None)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            state = state[0] or {}
        for slot in self.__slots__:
            setattr(self, slot, state.get(slot))


def read_archive_summary(path: str | Path) -> ArchiveSummary:
    """`summarize_archive_cells` over a real `archive.pkl` on disk."""
    with open(path, "rb") as fh:
        cells = _StateDroppingUnpickler(fh).load()
    return summarize_archive_cells(cells)


#: Sidecar filename the solver writes beside every flushed `archive.pkl`.
ARCHIVE_STATS_FILENAME = "archive.stats.json"


def _stats_path(path: str | Path) -> Path:
    """Resolve anything that identifies a run to its sidecar.

    Accepts the sidecar itself, the run directory, or the `archive.pkl`
    beside it, because callers hold whichever of the three they happened
    to be given and the point of this adapter is that reading counters
    must never cost more than reading a filename.
    """
    p = Path(path)
    if p.is_dir():
        return p / ARCHIVE_STATS_FILENAME
    if p.name == ARCHIVE_STATS_FILENAME:
        return p
    if p.suffix == ".pkl":
        return p.with_suffix(".stats.json")
    return p


def read_archive_counters(path: str | Path) -> Optional[ArchiveCounters]:
    """Read an `archive.stats.json` sidecar. REPORTING ONLY.

    Returns `None` — never raises — when the sidecar is absent,
    unreadable, not an object, or missing a core field. That is
    deliberate: this is a free diagnostic hung off the side of a verdict,
    and a malformed 130-byte file must not be able to take down a
    classification that never depended on it. 279 sidecars were surveyed;
    all 279 carry the six core fields, 72 also carry `hw_provenance`.
    """
    p = _stats_path(path)
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, Mapping):
        return None
    try:
        return ArchiveCounters(
            cells=int(raw["cells"]),
            records=int(raw["records"]),
            new_cells=int(raw["new_cells"]),
            improvements=int(raw["improvements"]),
            frontier=int(raw["frontier"]),
            best_score=float(raw["best_score"]),
            hw_provenance=(raw["hw_provenance"]
                           if isinstance(raw.get("hw_provenance"), Mapping)
                           else None),
        )
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Boundary axis profile — reported diagnostic, gates nothing.
#
# ADDITIVE (2026-08-11). `gated_wall_verdict` does not read any of this;
# every banked verdict is byte-identical with or without this section.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryAxisProfile:
    """Per-axis cardinality of the cell key inside the pinned band.

    A GATED verdict asserts that local coverage saturated and the
    boundary froze, and its remedy says "switch to an orthogonal arm".
    That reading only holds if the archive could have REMEMBERED an
    interaction in the first place. This profile answers the prior
    question — how many axes of the cell key are actually varying at the
    boundary, and how many of them are anything other than position.

    When every non-spatial axis is constant, GATED and KEY_BLIND stop
    being distinguishable in the direction that matters: the search has
    tried every POSITION, which is not the same claim as having tried
    every interaction, and no statistic over a position-only key can
    tell the two apart.
    """

    #: Deepest gx bucket in the deepest area — the pin.
    frontier_bucket: int
    #: Width of the profiled band, in gx buckets behind the pin.
    band: int
    #: Cells in the band, at any key arity.
    band_cells: int
    #: Cells in the band at the modal arity (the ones actually profiled).
    #: Differs from `band_cells` only for an archive whose key layout
    #: changed mid-campaign, which is itself worth seeing.
    profiled_cells: int
    key_arity: int
    #: Distinct values per key position, position 0 first.
    axis_cardinality: tuple[int, ...]
    #: Positions holding one value for every profiled cell.
    constant_axes: tuple[int, ...]
    #: Non-spatial positions with >= 2 values, minus `bookkeeping`.
    live_state_axes: tuple[int, ...]
    #: Non-spatial positions with >= 2 values that the caller named as
    #: trajectory bookkeeping (loop counters, route signatures): real
    #: variation, but variation the AGENT manufactured, not state the
    #: GAME changed.
    live_bookkeeping_axes: tuple[int, ...]
    #: |{(area, y_band, gx_bucket)}| inside the band.
    distinct_positions: int

    @property
    def live_state_axis_count(self) -> int:
        return len(self.live_state_axes)

    @property
    def alias_ratio(self) -> float:
        """Cells per distinct position inside the band — `concentration`
        computed locally, at the pin, rather than over the whole map."""
        return self.profiled_cells / max(1, self.distinct_positions)

    @property
    def interaction_blind(self) -> bool:
        """Fewer live game-state axes than `BOUNDARY_STATE_AXES_MIN`.

        Reported, never gated on. True means a GATED verdict on this
        archive should be read as "and we could not have seen an
        interaction anyway", which is a different remedy: give the key an
        axis a state change can land on BEFORE spending an orthogonal
        campaign on it.
        """
        return self.live_state_axis_count < BOUNDARY_STATE_AXES_MIN

    def as_dict(self) -> dict[str, Any]:
        return {
            "frontier_bucket": self.frontier_bucket,
            "band": self.band,
            "band_cells": self.band_cells,
            "profiled_cells": self.profiled_cells,
            "key_arity": self.key_arity,
            "axis_cardinality": list(self.axis_cardinality),
            "constant_axes": list(self.constant_axes),
            "live_state_axes": list(self.live_state_axes),
            "live_bookkeeping_axes": list(self.live_bookkeeping_axes),
            "distinct_positions": self.distinct_positions,
            "alias_ratio": round(self.alias_ratio, 3),
            "interaction_blind": self.interaction_blind,
            "calibration": CALIBRATION_TAG,
        }


def boundary_axis_profile(
    cells: Mapping[Any, Any],
    *,
    band: int = 24,
    bookkeeping: Sequence[int] = (),
) -> BoundaryAxisProfile:
    """Per-axis cardinality of the cell key inside the pinned band. Pure.

    `band` is the same 24-gx-bucket window `scripts/go_explore_solve.py`
    calls `_sel_band24` and samples from with `--deep-bias`, so the
    profile describes the region the primary arm is actually spending its
    budget on rather than an arbitrary slice.

    `bookkeeping` names key positions that vary because of the AGENT's
    own trajectory rather than the game's state; they are counted, but
    reported separately and excluded from `live_state_axes`. For the
    fleet's layout — `(sect, tb, kk, psig, loops, route_sig) +
    cell_fn(ram)` — that is positions 4 and 5, the maze loop counter and
    the route signature, both derived from the rollout's own gx history.
    Default `()` counts nothing as bookkeeping, so a caller that has not
    thought about its key layout gets the conservative (higher) reading.

    Mixed-arity archives (a key layout that changed mid-campaign) are
    profiled at the MODAL arity; `band_cells` vs `profiled_cells` shows
    how much was set aside.
    """
    empty = BoundaryAxisProfile(0, band, 0, 0, 0, (), (), (), (), 0)
    if not cells:
        return empty

    # Same projection and same "deepest area, deepest bucket" frontier
    # definition `summarize_archive_cells` uses, so `frontier_bucket` here
    # and `spatial_span`/`boundary_cells` there describe one region.
    spatial_of = {k: _spatial_key(k) for k in cells}
    max_area = max(s[0] for s in spatial_of.values())
    deep = [k for k, s in spatial_of.items() if s[0] == max_area]
    frontier = max(spatial_of[k][2] for k in deep)
    in_band = [k for k in deep if spatial_of[k][2] >= frontier - band]
    if not in_band:                     # only reachable for a negative band
        return empty

    arities: dict[int, int] = {}
    for k in in_band:
        arities[len(k)] = arities.get(len(k), 0) + 1
    # Ties break toward the LONGER key: a key that grew gained an axis,
    # and profiling the shorter half would report that axis as absent.
    modal = max(arities, key=lambda a: (arities[a], a))
    profiled = [k for k in in_band if len(k) == modal]

    seen: list[set] = [set() for _ in range(modal)]
    for k in profiled:
        for i, v in enumerate(k):
            seen[i].add(v)
    card = tuple(len(s) for s in seen)

    spatial = {p % modal for p in SPATIAL_KEY_POSITIONS if -modal <= p < modal}
    book = {p % modal for p in bookkeeping if -modal <= p < modal}
    constant = tuple(i for i, c in enumerate(card) if c <= 1)
    live_state = tuple(i for i, c in enumerate(card)
                       if c >= 2 and i not in spatial and i not in book)
    live_book = tuple(i for i, c in enumerate(card)
                      if c >= 2 and i not in spatial and i in book)

    return BoundaryAxisProfile(
        frontier_bucket=frontier,
        band=band,
        band_cells=len(in_band),
        profiled_cells=len(profiled),
        key_arity=modal,
        axis_cardinality=card,
        constant_axes=constant,
        live_state_axes=live_state,
        live_bookkeeping_axes=live_book,
        distinct_positions=len({spatial_of[k] for k in profiled}),
    )


def read_boundary_axis_profile(path: str | Path, *, band: int = 24,
                               bookkeeping: Sequence[int] = ()
                               ) -> BoundaryAxisProfile:
    """`boundary_axis_profile` over a real `archive.pkl` on disk."""
    with open(path, "rb") as fh:
        cells = _StateDroppingUnpickler(fh).load()
    return boundary_axis_profile(cells, band=band, bookkeeping=bookkeeping)


def telemetry_from_paths(
    progress_path: str | Path,
    *,
    archive_path: Optional[str | Path] = None,
    segment: int = -1,
    label: str = "",
    counters: bool = True,
) -> WallTelemetry:
    """Convenience adapter: one run directory's telemetry, ready to score.

    The `archive.stats.json` sidecar is picked up automatically from the
    run directory — it costs one 130-byte read and it is the only effort
    denominator that is intrinsic to the archive rather than to the
    progress file. It changes no verdict; pass `counters=False` to prove
    that, which is what the test does.
    """
    segments = load_progress_segments(progress_path)
    records = segments[segment] if segments else ()
    archive = read_archive_summary(archive_path) if archive_path else None
    stats = (read_archive_counters(archive_path or Path(progress_path).parent)
             if counters else None)
    return WallTelemetry(records=records, archive=archive, counters=stats,
                         label=label or str(progress_path))


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------

def _window_yield(records: Sequence[ProgressRecord], attr: str,
                  start: int, end: int) -> Optional[float]:
    """New units of `attr` per emulator step between two indices."""
    a, b = records[start], records[end]
    d_steps = b.steps - a.steps
    if d_steps <= 0:
        return None
    va, vb = getattr(a, attr), getattr(b, attr)
    if va is None or vb is None:
        return None
    return (vb - va) / d_steps


def series_yields(records: Sequence[ProgressRecord], attr: str = "cells",
                  window: int = WINDOW_RECORDS
                  ) -> tuple[Optional[float], Optional[float]]:
    """(trailing yield, best yield this run ever sustained), per step.

    `(None, None)` when the series cannot be measured at all: too few
    records, no usable step delta, or the field missing at either end of
    the trailing window. A returned peak of exactly 0.0 means the series
    was measurable and NEVER GREW — a distinct state from "grew, then
    stopped", and the caller must not conflate them.
    """
    n = len(records)
    if n < window + 1:
        return None, None
    recent = _window_yield(records, attr, n - window - 1, n - 1)
    if recent is None:
        return None, None
    peak = 0.0
    for i in range(window, n):
        y = _window_yield(records, attr, i - window, i)
        if y is not None and y > peak:
            peak = y
    return recent, peak


def saturation(records: Sequence[ProgressRecord], attr: str = "cells",
               window: int = WINDOW_RECORDS) -> Optional[float]:
    """1 - (trailing yield / best yield this run ever sustained).

    Scale-free in both directions: normalizing by STEPS rather than by
    seconds keeps a throughput change (worker count, sps drift, a paced
    show run) out of the number, and normalizing by the run's own peak
    keeps game-to-game cell-key verbosity out of it.

    Returns None when the window has no usable step delta AND when the
    series never grew (peak yield 0). The second case matters: a series
    that never accumulated is STAGNANT, which this module's adopted form
    defines as BARREN — the OPPOSITE verdict from a plateau. Reporting it
    as saturation 1.0 makes the two indistinguishable at the call site
    and reads "maximally saturated" for a search that produced nothing.
    Callers that need to tell them apart use `series_yields`.
    """
    recent, peak = series_yields(records, attr, window)
    if recent is None or peak is None or peak <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - recent / peak))


def map_stall_windows(records: Sequence[ProgressRecord]) -> int:
    """Consecutive trailing records with zero permanent-map delta.

    Reported, never gated on: it grows with how long the run was left
    alive, so any fixed threshold eventually fires on a healthy run
    (MAP_STALL_WINDOWS_IS_SEPARATING is False).
    """
    n = len(records)
    stall = 0
    for i in range(n - 1, 0, -1):
        if records[i].max_gx != records[i - 1].max_gx:
            break
        stall += 1
    return stall


def _evidence(tel: WallTelemetry, window: int) -> dict[str, Any]:
    recs = tel.records
    n = len(recs)
    a, b = recs[n - window - 1], recs[n - 1]
    d_steps = b.steps - a.steps
    d_cells = b.cells - a.cells
    # `doors` is NOT summed in. It counts articulation points in a room
    # graph that keeps being rewritten, so it churns in both directions
    # and is not a ratchet; summing it made the frozen Castlevania hall
    # read PROGRESSING off +186 doors with every genuine counter flat.
    # DOORS_IS_MONOTONE; reported below as `doors_delta`.
    topo = ((b.max_area - a.max_area) + (b.max_sect - a.max_sect)
            + (b.max_room - a.max_room))
    ev: dict[str, Any] = {
        "records": n,
        "window": window,
        "elapsed_s": b.elapsed_s,
        "cells": b.cells,
        "window_steps": d_steps,
        "window_new_cells": d_cells,
        "topo_delta": topo,
        "map_delta": b.max_gx - a.max_gx,
        "solutions": b.solutions,
        "frozen_windows": b.stall_flat_windows,
        # Reported-only diagnostics (every one REFUTED as a gate).
        "doors_delta": b.doors - a.doors,
        "raw_coverage_saturation": saturation(recs, "cells", window),
        "churn_per_window": (d_cells / max(1, b.cells)) / window,
        "map_stall_windows": map_stall_windows(recs),
    }
    # The C_local series, when the solver emits one. Peak yield is kept
    # alongside saturation because saturation() collapses "never grew"
    # into None, and never-grown is BARREN's own definition.
    c_local_peak = series_yields(recs, "c_local", window)[1]
    ev["c_local"] = recs[-1].c_local
    ev["c_local_records"] = sum(1 for r in recs if r.c_local is not None)
    ev["c_local_peak_yield"] = c_local_peak
    ev["c_local_saturation"] = saturation(recs, "c_local", window)
    if tel.archive is not None:
        ev["archive_cells"] = tel.archive.cells
        ev["distinct_spatial"] = tel.archive.distinct_spatial
        ev["spatial_span"] = tel.archive.spatial_span
        ev["concentration"] = round(tel.archive.concentration, 3)
        ev["boundary_cells"] = tel.archive.boundary_cells
        ev["boundary_visit_entropy"] = round(tel.archive.boundary_visit_entropy, 4)
        ev["explored_fraction"] = round(tel.archive.explored_fraction, 4)
    # The sidecar. Reported for the same reason `map_stall_windows` is:
    # so a verdict always shows the effort it was taken over, and shows
    # it on the archive's own clock rather than on a progress file whose
    # step count may span two attempts.
    if tel.counters is not None:
        ev["archive_records"] = tel.counters.records
        ev["archive_improvements"] = tel.counters.improvements
        ev["archive_frontier"] = tel.counters.frontier
        ev["archive_best_score"] = tel.counters.best_score
        ev["archive_novelty_per_record"] = round(
            tel.counters.novelty_per_record, 6)
        ev["archive_explored_fraction"] = round(
            tel.counters.explored_fraction, 4)
    return ev


def _concentration_descriptor(arc: Optional[ArchiveSummary]) -> str:
    """The descriptive label, or "" when there is nothing to describe.

    A pure function of the ARCHIVE — never of the progress series — so
    that a run cannot acquire the label by the solver starting to emit a
    field. It annotates; it does not classify.
    """
    if arc is None:
        return ""
    if arc.concentration < CONCENTRATION_DESCRIPTIVE_MIN:
        return ""
    return UNRESOLVED_CONCENTRATED


def gated_wall_verdict(telemetry: WallTelemetry,
                       *, window: int = WINDOW_RECORDS) -> WallVerdict:
    """Classify a stalled search. Pure; reads nothing but `telemetry`.

    Test order is deliberate — cheapest and most certain first, so a run
    that is plainly still working never reaches a wall test:

      1. too few records / too little compute  -> INSUFFICIENT
      2. a solution landed                     -> RESOLVED
      3. topology or the map moved             -> PROGRESSING
      4. spatially degenerate cell key         -> KEY_BLIND
      5. archive frozen, or C_local stagnant   -> BARREN
      6. C_local still climbing                -> COVERAGE_LIMITED
      7. anything else                         -> INDETERMINATE

    KEY_BLIND precedes BARREN on purpose. Both point at the cell key, but
    a degenerate spatial projection names WHICH axis is missing, and a
    frozen archive in a spatially blind game (every Bubble Bobble run in
    the corpus) is frozen BECAUSE of that blindness.

    Steps 4-6 apply to the C_local series and to the archive snapshot
    SYMMETRICALLY, which is the point: the same run must not change
    verdict just because the solver started emitting `c_local`.

    Step 7 is where the module used to fork, and the fork is gone. A
    plateau — corroborated, contradicted, or seen from one side only —
    now lands in the SAME abstention as no evidence at all, because
    that is what the evidence supports: 22 candidate statistics scored
    over 103 archives and none of them tells a stalled search from one
    about to finish. Concentration survives as a DESCRIPTION, on
    `WallVerdict.descriptor`, and every remaining test is subtractive
    (BARREN, KEY_BLIND, INSUFFICIENT) or affirmative about something
    directly observed (RESOLVED, PROGRESSING, COVERAGE_LIMITED).

    Note the asymmetry that keeps COVERAGE_LIMITED honest: only the
    C_local SERIES can reach it, because only a series can show a
    direction. A cross-sectional number is one point on a curve and the
    hall's curve is the steepest in the corpus while its map is frozen.
    """
    recs = telemetry.records
    #: Fields the form wanted and THIS call could not use. Set properly
    #: once the evidence exists — a telemetry that carries a usable
    #: `c_local` must not be told `c_local` is missing.
    missing: tuple[str, ...] = ("c_local", "boundary_action_entropy")

    def out(cls: WallClass, reasons: Sequence[str], ev: Mapping[str, Any],
            degraded: bool, miss: Sequence[str] = (),
            descriptor: str = "") -> WallVerdict:
        return WallVerdict(
            wall_class=cls, label=telemetry.label, degraded=degraded,
            missing=tuple(miss), reasons=tuple(reasons), evidence=dict(ev),
            remedy=REMEDY[cls], descriptor=descriptor,
        )

    def abstain(reasons: list[str], ev: Mapping[str, Any],
                miss: Sequence[str]) -> WallVerdict:
        """The terminal class, plus the description and its receipt."""
        # `missing` is what THIS call lacked, so an archive snapshot —
        # which is where `boundary_cells` comes from — settles the
        # frontier field rather than leaving it on the shopping list.
        if telemetry.archive is None:
            miss = tuple(miss) + ("frontier_bucket_cells",)
        desc = _concentration_descriptor(telemetry.archive)
        if desc:
            reasons = reasons + [
                f"{desc}: concentration {telemetry.archive.concentration:.2f} "
                f"(>= CONCENTRATION_DESCRIPTIVE_MIN="
                f"{CONCENTRATION_DESCRIPTIVE_MIN}) DESCRIBES this archive and "
                f"classifies nothing — the statistic that used to certify a "
                f"wall here was struck after 22 candidates over 103 archives "
                f"all straddled the resolved class; see "
                f"{STRUCK_CLASSIFICATION_RECEIPT}"]
        return out(WallClass.INDETERMINATE, reasons, ev, degraded=True,
                   miss=miss, descriptor=desc)

    if len(recs) < max(MIN_RECORDS, window + 1):
        return out(WallClass.INSUFFICIENT,
                   [f"{len(recs)} progress records < MIN_RECORDS={MIN_RECORDS}"],
                   {"records": len(recs)}, degraded=True, miss=missing)

    ev = _evidence(telemetry, window)
    # A `c_local` that exists but is too short to yield a series is still
    # a gap, so usability — not presence — decides.
    if ev["c_local_peak_yield"] is not None:
        missing = ("boundary_action_entropy",)

    if ev["solutions"] > 0:
        return out(WallClass.RESOLVED,
                   [f"{ev['solutions']} solution(s) banked"], ev,
                   degraded=False)

    if ev["topo_delta"] > 0 or ev["map_delta"] > 0:
        return out(WallClass.PROGRESSING,
                   [f"topo_delta={ev['topo_delta']}, map_delta={ev['map_delta']} "
                    f"over the trailing {window} records"], ev, degraded=False)

    if ev["window_steps"] < EFFORT_MIN_STEPS:
        return out(WallClass.INSUFFICIENT,
                   [f"window_steps={ev['window_steps']} < "
                    f"EFFORT_MIN_STEPS={EFFORT_MIN_STEPS}"], ev,
                   degraded=True, miss=missing)

    reasons = [f"zero topological transition and zero permanent-map delta "
               f"over {window} records ({ev['map_stall_windows']} records "
               f"since the map last moved)"]

    arc = telemetry.archive
    c_local = ev["c_local"]

    # ---- 4. the cell key cannot see the axis the problem lives on ----
    # Both branches are the same test on the same quantity: `c_local`
    # counts (area, y_band, gx_bucket) triples and `spatial_span` counts
    # gx buckets inside the deepest area, so spatial_span <= c_local
    # always. Applying it to the series as well as to the archive is what
    # keeps a run's verdict from flipping when `c_local` starts being
    # emitted.
    if arc is not None and arc.spatial_span < SPATIAL_SPAN_MIN:
        reasons.append(f"spatial_span={arc.spatial_span} < "
                       f"SPATIAL_SPAN_MIN={SPATIAL_SPAN_MIN}: the cell key's "
                       f"spatial projection is degenerate")
        return out(WallClass.KEY_BLIND, reasons, ev, degraded=False)
    if c_local is not None and c_local < C_LOCAL_FLOOR_BUCKETS:
        proof = (f"; c_local < SPATIAL_SPAN_MIN={SPATIAL_SPAN_MIN} also "
                 f"PROVES spatial_span is degenerate, since "
                 f"spatial_span <= c_local by construction"
                 if c_local < SPATIAL_SPAN_MIN else "")
        reasons.append(f"c_local={c_local} < "
                       f"C_LOCAL_FLOOR_BUCKETS={C_LOCAL_FLOOR_BUCKETS}: the "
                       f"cell key's spatial projection is degenerate, so no "
                       f"C_local statistic over it means anything{proof}")
        return out(WallClass.KEY_BLIND, reasons, ev, degraded=False)

    # ---- 5. coverage never accumulated -------------------------------
    if ev["frozen_windows"] >= FROZEN_WINDOWS_MAX:
        reasons.append(f"frozen_windows={ev['frozen_windows']} >= "
                       f"FROZEN_WINDOWS_MAX={FROZEN_WINDOWS_MAX}")
        return out(WallClass.BARREN, reasons, ev, degraded=False)
    if ev["cells"] < COVERAGE_FLOOR_CELLS:
        reasons.append(f"cells={ev['cells']} < "
                       f"COVERAGE_FLOOR_CELLS={COVERAGE_FLOOR_CELLS}")
        return out(WallClass.BARREN, reasons, ev, degraded=False)
    if ev["c_local_peak_yield"] == 0:
        # Measurable series, zero growth anywhere in the run: this is the
        # form's "BARREN <=> C_local STAGNANT" verbatim. It is
        # emphatically NOT a plateau — nothing ever accumulated to
        # plateau — and the two must not share a class, whatever that
        # class is called.
        reasons.append(f"c_local pinned at {c_local} for the whole run "
                       f"(peak yield 0.0 new buckets/step): C_local is "
                       f"STAGNANT, not plateaued")
        return out(WallClass.BARREN, reasons, ev, degraded=False)

    # ---- 6/7. the C_local series, when the solver emits one ----------
    c_local_sat = ev["c_local_saturation"]
    if c_local_sat is not None:
        if c_local_sat < C_LOCAL_SATURATION_MIN:
            # The one affirmative reading a series still licenses, and
            # the safe direction: a footprint measurably still expanding
            # has not run out of reachable ground. It overrides a high
            # cross-sectional concentration because a direction beats a
            # point — and because concentration no longer decides
            # anything to override.
            reasons.append(f"C_local saturation {c_local_sat:.3f} < "
                           f"C_LOCAL_SATURATION_MIN={C_LOCAL_SATURATION_MIN}: "
                           f"the map footprint is still expanding")
            return out(WallClass.COVERAGE_LIMITED, reasons, ev, degraded=True,
                       miss=("boundary_action_entropy",))
        # A plateau. This is where the GATED branch was, on 2026-08-10
        # requiring cross-sectional corroboration and on 2026-08-11
        # requiring nothing at all, because there is nothing left that
        # could corroborate it: the corroborating statistic was struck.
        reasons.append(f"C_local saturation {c_local_sat:.3f} >= "
                       f"C_LOCAL_SATURATION_MIN={C_LOCAL_SATURATION_MIN} "
                       f"(PROVISIONAL, never measured against a labelled run): "
                       f"a plateau, which is not by itself evidence of a wall "
                       f"— the hall's own cell curve is the steepest in the "
                       f"corpus with its map frozen underneath it")
        return abstain(reasons, ev, missing)

    # ---- 7. the terminal abstention ----------------------------------
    #
    # Everything that survives to here is a stalled search that is not
    # barren and not key-blind, and there is nothing admissible left to
    # ask of it. Until 2026-08-11 this was a fork on `concentration`:
    # over the threshold GATED, under it COVERAGE_LIMITED. Both arms are
    # gone, not just the expensive one, because a single cross-sectional
    # number cannot support EITHER claim — the same statistic that read
    # 31.04 on the unsolved hall read 98.30 on `ge_chain_w8/lvl_00_8-1`,
    # which was SOLVED, and 15.66 on one of its own chain siblings.
    if arc is None:
        reasons.append("no C_local series and no archive snapshot: nothing "
                       "measurable is left to ask")
    else:
        reasons.append(
            f"coverage concentration {arc.concentration:.2f} ({arc.cells} "
            f"cells over {arc.distinct_spatial} spatial buckets) is REPORTED "
            f"and gates nothing — CONCENTRATION_IS_SEPARATING is False, and "
            f"no replacement was found for it")
    return abstain(reasons, ev, missing)
