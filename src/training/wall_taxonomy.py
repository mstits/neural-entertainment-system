"""Gated-wall discriminator — offline-calibrated, runtime-inert.

WHAT THIS IS
------------
A pure statistic over solver telemetry that answers one question about a
search that has stopped making progress:

    is this wall GATED (the search has saturated everything it can reach
    and something in the game is withholding the next transition), or is
    it BARREN (the search is not generating local novelty at all), or is
    it merely COVERAGE-LIMITED (still productively expanding — it just
    has not arrived yet)?

The three demand opposite responses. A gated wall wants an ORTHOGONAL
mechanism (a different action axis, a mechanic the current key cannot
express); a barren wall wants the CELL KEY or the reset fixed; a
coverage-limited wall wants nothing but more wall-clock. Getting it
backwards costs hours in either direction: `runs/live_show/
smb_4_4_micro/lvl_8-4` looked identically stuck for 44 straight minutes
and then simply finished, while `runs/cv_hall_ortho_a` is an orthogonal
arm launched at the Castlevania hall on the belief that it is gated.

THE CALIBRATION IS CONDITIONAL ON A READ-OUT THAT HAS NOT HAPPENED.
That belief is UNVALIDATED. The hall has never been solved; the corpus'
only two positives are the SAME level (`lvl_03_trace`) from two
hardware-flag lineages, which is one wall seen twice, not two
independent ones; and the arm testing it (`runs/cv_hall_ortho_a`) is
still running. If that arm reads out COVERAGE — the 8-4 outcome above —
the positive class is empty and every band below collapses. Treat the
hall as PENDING-VALIDATION, and read §9 of the receipt before citing
any number here as evidence that gated walls are detectable.

The discriminator's adopted form is:

    GATED  <=>  local coverage SATURATED (C_local plateau)
                AND high action-entropy at the boundary
                AND zero topological transition
                AND zero permanent-map delta

    BARREN <=>  local coverage STAGNANT (C_local never accumulated)

This module implements that form against the telemetry the fleet
ACTUALLY records, and is explicit — in `MISSING_TELEMETRY` and in every
constant's calibration tag — about which terms are measured, which are
substituted, and which were refuted outright by the banked corpus.

RUNTIME STATUS: INERT. Nothing imports this module. Thresholds are
frozen constants tagged with how they were derived; self-arming dispatch
is a later, separate decision. Import it from an analysis script or a
test, not from a solver loop.

CALIBRATION: see docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md
for the corpus, the per-run statistic tables, and the separating bands
each shipped constant sits in.
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
# from a differently-tuned build.
CALIBRATION_TAG = "CALIBRATED-OFFLINE-2026-08-10"


# --------------------------------------------------------------------------
# Calibrated constants.
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

#: Coverage concentration = cells / distinct spatial buckets. The
#: cross-sectional stand-in for "C_local has plateaued": a search whose
#: archive keeps multiplying inside a map footprint that stopped growing
#: is, by definition, saturated locally. Separating band: 20.58 (SMB 8-3,
#: resolved) .. 31.04 (Castlevania hall, gated). This is the THINNEST
#: shipped margin in the module (1.51x total, ~1.2x either side) and the
#: receipt says so.
CONCENTRATION_GATED_MIN = 25.0  # CALIBRATED-OFFLINE-2026-08-10

#: Saturation of a true C_local time series, when one exists. Defined
#: exactly like the coverage-saturation statistic but on the count of
#: distinct spatial buckets rather than on raw cells. No banked run emits
#: that series, so this value is NOT calibrated; it is the value the
#: runtime version should re-derive first.
C_LOCAL_SATURATION_MIN = 0.85  # PROVISIONAL-2026-08-10 (no offline series)

#: Absolute floor on a reported `c_local` before ANY C_local statistic
#: over it is allowed to gate. `c_local` is defined as
#: |{(area, y_band, gx_bucket)}| — the same column the §4 archive table
#: calls `distinct_spatial` — so the corpus measures it directly:
#: separating band 32 (BB 99-1 boss retry, the largest degenerate
#: projection) .. 638 (`ge_1_4_solve`, the smallest spatially resolved
#: archive). This is the series-path twin of SPATIAL_SPAN_MIN. Without
#: it, a single-screen profile that emits `c_local` and flushes no
#: archive reads GATED where the identical run WITH an archive reads
#: KEY_BLIND — a divergence measured before this floor existed.
C_LOCAL_FLOOR_BUCKETS = 64  # CALIBRATED-OFFLINE-2026-08-10

#: May a C_local series certify GATED with nothing to corroborate it?
#: NO, while C_LOCAL_SATURATION_MIN is PROVISIONAL. A threshold that has
#: never been measured against a labelled run must not be able to
#: manufacture the one verdict that costs an orthogonal campaign, so a
#: plateau seen only in the series degrades to INDETERMINATE unless the
#: cross-sectional concentration agrees. The series path is still fully
#: load-bearing in the safe directions: it can REFUTE a plateau
#: (-> COVERAGE_LIMITED, overriding concentration), name STAGNANT
#: (-> BARREN) and name a blind key (-> KEY_BLIND). Flip this to True
#: only together with a calibrated C_LOCAL_SATURATION_MIN, in the same
#: commit as the receipt that measures it.
C_LOCAL_SERIES_MAY_CERTIFY_GATED = False  # PROVISIONAL-2026-08-10

#: Saturation computed on RAW archive cells. REFUTED: on the corpus the
#: gated Castlevania hall (0.343) sits BETWEEN two resolved coverage
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
#: (gated CV hall 0.984, barren BB r68 ortho 0.778, resolved SMB 1-4
#: 0.9999). It measures how evenly returns were spread, not how varied
#: the actions were. Reported only.
BOUNDARY_ENTROPY_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

#: Consecutive windows at zero permanent-map delta. On the corpus the
#: gated hall (81 windows) does separate from resolved SMB 8-4 (44) and
#: SMB 4-4 (25) — but the statistic is a pure function of how long the
#: run was left alive, so a longer 8-4 would have crossed any fixed
#: threshold. Deliberately NOT shipped as a gate; reported so a human can
#: see the horizon a verdict was taken over.
MAP_STALL_WINDOWS_IS_SEPARATING = False  # REFUTED-OFFLINE-2026-08-10

#: Cell-key positions that `_spatial_key` projects onto: (area, y_band,
#: gx_bucket). Negative, so they stay correct if a future arm grows the
#: key's PREFIX — every adapter's `cell_fn` output is the last five
#: elements and `scripts/go_explore_solve.py` indexes `key[-5]`/`key[-1]`
#: on exactly that promise. Read by `boundary_axis_profile` to decide
#: which axes are positional and which are not.
SPATIAL_KEY_POSITIONS = (-5, -2, -1)

#: Non-spatial axes that must carry at least TWO distinct values at the
#: boundary before a GATED verdict can be read as "the game is
#: withholding a transition" rather than "our cell key cannot represent
#: an interaction, so we cannot know". OBSERVED, NEVER GATED ON: nothing
#: in `gated_wall_verdict` reads this, and it is not calibrated against
#: a labelled corpus. It exists because the measurement that motivated
#: it is stark — `runs/cv_hall_ortho_a`, 131,561 cells, reads GATED at
#: concentration 120.04, and in its pinned band SIX of eleven key
#: positions are CONSTANT (sect, time-bin, kill-count, room-sig, area,
#: boss-HP all identically 0/empty), leaving exactly ONE game-state axis
#: (the on-stairs bit) against two trajectory-bookkeeping axes (loop
#: count, route signature). A search whose memory has one state bit
#: cannot have "tried every interaction"; it can only have tried every
#: position. See docs/proposals/gate_opener_arm_2026-08-11.md.
BOUNDARY_STATE_AXES_MIN = 2  # OBSERVED-2026-08-10 (reported, not a gate)


#: Telemetry the adopted form wants and the fleet does not emit. This is
#: the shopping list a runtime version needs the solver to add to
#: `progress_line()`; each entry names the field and why it matters.
MISSING_TELEMETRY: dict[str, str] = {
    "c_local": (
        "count of DISTINCT spatial buckets in the archive — |{(area, "
        "y_band, gx_bucket)}| — per progress line. Without it, C_local "
        "saturation can only be approximated cross-sectionally from a "
        "final archive snapshot (see CONCENTRATION_GATED_MIN), which is "
        "one number instead of a curve and cannot see a plateau form."
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
    "archive_snapshot_on_show_runs": (
        "runs/live_show/* writes progress.jsonl but no archive.pkl "
        "(flush_secs is set to ~forever), so the two ground-truth "
        "coverage walls can only be scored on the degraded path."
    ),
}


class WallClass(str, Enum):
    """Verdict labels, ordered from 'nothing to do' to 'act now'."""

    #: The search already produced a solution inside the window.
    RESOLVED = "resolved"
    #: Topology or the permanent map moved inside the window.
    PROGRESSING = "progressing"
    #: Still expanding productively; the wall is wall-clock, not structure.
    COVERAGE_LIMITED = "coverage_limited"
    #: Local coverage saturated, boundary frozen: needs an orthogonal
    #: mechanism, not more of the same search.
    GATED = "gated"
    #: Coverage never accumulated: the archive is frozen or trivially
    #: small. The cell key, the reset, or determinism is broken.
    BARREN = "barren"
    #: The cell key's spatial projection is degenerate, so no coverage
    #: statistic over it is meaningful. Enrich the key first.
    KEY_BLIND = "key_blind"
    #: Preconditions met, but the evidence needed to separate GATED from
    #: COVERAGE_LIMITED is not present in this telemetry.
    INDETERMINATE = "indeterminate"
    #: Not enough records or not enough compute spent to say anything.
    INSUFFICIENT = "insufficient"


#: What a human should do about each verdict. Advisory text only — this
#: module dispatches nothing.
REMEDY: dict[WallClass, str] = {
    WallClass.RESOLVED: "nothing — harvest the solution and move on",
    WallClass.PROGRESSING: "nothing — the frontier is still moving",
    WallClass.COVERAGE_LIMITED: "give it more wall-clock before changing anything",
    WallClass.GATED: "switch to an orthogonal arm: a different action axis, "
                     "or a mechanic the current cell key cannot express",
    WallClass.BARREN: "fix the search, not the game: check the cell key, the "
                      "reset path, and determinism — the archive is frozen",
    WallClass.KEY_BLIND: "add the missing state axis to the cell key; the "
                         "current key cannot see where progress happens",
    WallClass.INDETERMINATE: "collect the missing telemetry (see "
                             "MISSING_TELEMETRY) before deciding",
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
    explored_fraction: float = 0.0

    @property
    def concentration(self) -> float:
        """cells per distinct spatial bucket — the C_local stand-in."""
        return self.cells / max(1, self.distinct_spatial)


@dataclass(frozen=True)
class WallTelemetry:
    """Everything the discriminator is allowed to look at."""

    records: tuple[ProgressRecord, ...]
    archive: Optional[ArchiveSummary] = None
    label: str = ""


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
) -> WallTelemetry:
    """Convenience adapter: one run directory's telemetry, ready to score."""
    segments = load_progress_segments(progress_path)
    records = segments[segment] if segments else ()
    archive = read_archive_summary(archive_path) if archive_path else None
    return WallTelemetry(records=records, archive=archive,
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
    topo = ((b.max_area - a.max_area) + (b.max_sect - a.max_sect)
            + (b.max_room - a.max_room) + (b.doors - a.doors))
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
        "frozen_windows": max(r.stall_flat_windows for r in recs[n - window - 1:]),
        # Reported-only diagnostics (all three REFUTED as gates).
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
    return ev


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
      7. C_local plateau, concentration agrees -> GATED
      8. no C_local evidence available         -> INDETERMINATE
      9. otherwise, on concentration alone     -> GATED / COVERAGE_LIMITED

    KEY_BLIND precedes BARREN on purpose. Both point at the cell key, but
    a degenerate spatial projection names WHICH axis is missing, and a
    frozen archive in a spatially blind game (every Bubble Bobble run in
    the corpus) is frozen BECAUSE of that blindness.

    Steps 4-6 apply to the C_local series and to the archive snapshot
    SYMMETRICALLY, which is the point: the same run must not change
    verdict just because the solver started emitting `c_local`. The
    series can only move a verdict toward GATED when the cross-sectional
    concentration says the same thing (see
    C_LOCAL_SERIES_MAY_CERTIFY_GATED); it can move one away from GATED
    on its own, because that direction cannot cost a campaign.
    """
    recs = telemetry.records
    #: Fields the adopted form wanted and THIS call could not use. Set
    #: properly once the evidence exists — a telemetry that carries a
    #: usable `c_local` must not be told `c_local` is missing.
    missing: tuple[str, ...] = ("c_local", "boundary_action_entropy")

    def out(cls: WallClass, reasons: Sequence[str], ev: Mapping[str, Any],
            degraded: bool, miss: Sequence[str] = ()) -> WallVerdict:
        return WallVerdict(
            wall_class=cls, label=telemetry.label, degraded=degraded,
            missing=tuple(miss), reasons=tuple(reasons), evidence=dict(ev),
            remedy=REMEDY[cls],
        )

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
        # adopted form's "BARREN <=> C_local STAGNANT" verbatim. It is
        # emphatically NOT a plateau — nothing ever accumulated to
        # plateau — and calling it GATED here would fire an orthogonal
        # campaign at a search that is not searching.
        reasons.append(f"c_local pinned at {c_local} for the whole run "
                       f"(peak yield 0.0 new buckets/step): C_local is "
                       f"STAGNANT, not plateaued")
        return out(WallClass.BARREN, reasons, ev, degraded=False)

    # ---- 6/7. the C_local series, when the solver emits one ----------
    conc = arc.concentration if arc is not None else None
    c_local_sat = ev["c_local_saturation"]
    if c_local_sat is not None:
        if c_local_sat < C_LOCAL_SATURATION_MIN:
            # Refutation is always allowed on the series alone: it can
            # only move a verdict AWAY from GATED, and the adopted form
            # makes C_local the primary term, so it also overrides a high
            # cross-sectional concentration.
            reasons.append(f"C_local saturation {c_local_sat:.3f} < "
                           f"C_LOCAL_SATURATION_MIN={C_LOCAL_SATURATION_MIN}: "
                           f"the map footprint is still expanding")
            return out(WallClass.COVERAGE_LIMITED, reasons, ev, degraded=True,
                       miss=("boundary_action_entropy",))
        reasons.append(f"C_local saturation {c_local_sat:.3f} >= "
                       f"C_LOCAL_SATURATION_MIN={C_LOCAL_SATURATION_MIN} "
                       f"(PROVISIONAL, never measured against a labelled run)")
        if conc is not None and conc >= CONCENTRATION_GATED_MIN:
            reasons.append(f"corroborated by concentration {conc:.2f} >= "
                           f"CONCENTRATION_GATED_MIN={CONCENTRATION_GATED_MIN}")
            return out(WallClass.GATED, reasons, ev, degraded=True,
                       miss=("boundary_action_entropy",))
        if C_LOCAL_SERIES_MAY_CERTIFY_GATED:
            return out(WallClass.GATED, reasons, ev, degraded=True,
                       miss=("boundary_action_entropy",))
        reasons.append(
            "abstaining: " + (
                f"concentration {conc:.2f} < "
                f"CONCENTRATION_GATED_MIN={CONCENTRATION_GATED_MIN} "
                f"CONTRADICTS the series"
                if conc is not None else
                "no archive snapshot to corroborate it")
            + ", and C_LOCAL_SERIES_MAY_CERTIFY_GATED is False while the "
              "threshold is PROVISIONAL")
        return out(WallClass.INDETERMINATE, reasons, ev, degraded=True,
                   miss=missing + ("frontier_bucket_cells",))

    # ---- 8. nothing left to measure C_local with ---------------------
    if arc is None:
        reasons.append("no C_local series and no archive snapshot: cannot "
                       "separate a saturated wall from a slow one")
        return out(WallClass.INDETERMINATE, reasons, ev, degraded=True,
                   miss=missing + ("frontier_bucket_cells",))

    # ---- 9. cross-sectional concentration alone ----------------------
    if conc >= CONCENTRATION_GATED_MIN:
        reasons.append(f"coverage concentration {conc:.2f} >= "
                       f"CONCENTRATION_GATED_MIN={CONCENTRATION_GATED_MIN} "
                       f"({arc.cells} cells over {arc.distinct_spatial} "
                       f"spatial buckets): C_local has plateaued while the "
                       f"key keeps manufacturing nuisance novelty")
        return out(WallClass.GATED, reasons, ev, degraded=True,
                   miss=missing)

    reasons.append(f"coverage concentration {conc:.2f} < "
                   f"CONCENTRATION_GATED_MIN={CONCENTRATION_GATED_MIN}: the "
                   f"map footprint is still absorbing new cells")
    return out(WallClass.COVERAGE_LIMITED, reasons, ev, degraded=True,
               miss=missing)
