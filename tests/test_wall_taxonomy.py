"""Tests for src/training/wall_taxonomy.py — the gated-wall discriminator.

Four layers:

1. Unit tests over the pure helpers (segmentation, key projection, the
   saturation statistic, the archive summary).
2. Verdict ordering, plus (2b) the C_local SERIES path — the world the
   receipt's §8 asks the fleet to move to. No banked run emits
   `c_local`, so that world is reachable only by construction, which is
   exactly why it is pinned down here rather than after the field
   lands.
3. A frozen CALIBRATION FIXTURE (`CORPUS`) holding the statistics each
   banked run in the 2026-08-10 calibration actually produced, replayed
   through synthetic telemetry. These lock the shipped thresholds: move
   a constant out of its measured separating band and a corpus row
   flips. `runs/` is gitignored, so this layer carries the calibration
   without needing the multi-GB archives on disk.
4. A live regression that re-derives the same verdicts from the real
   run directories, skipped when they are absent.

READ THE CORPUS HEALTH WARNING before treating a GATED expectation here
as ground truth: the corpus has no validated positive.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.training import wall_taxonomy as wt
from src.training.wall_taxonomy import (
    ArchiveSummary,
    ProgressRecord,
    WallClass,
    WallTelemetry,
    gated_wall_verdict,
    load_progress_segments,
    record_from_json,
    saturation,
    summarize_archive_cells,
)

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _FakeCell:
    def __init__(self, visits: int = 1, explored: bool = True) -> None:
        self.visits = visits
        self.explored = explored


def make_records(
    *,
    n: int = 20,
    cells_start: int = 10_000,
    cells_end: int = 20_000,
    steps_start: int = 0,
    steps_end: int = 5_000_000,
    solutions: int = 0,
    map_gx: int = 500,
    map_gx_end: int | None = None,
    max_area: int = 0,
    max_area_end: int | None = None,
    frozen: int = 0,
    c_local: tuple[int, int] | None = None,
    c_local_series: Sequence[int] | None = None,
) -> tuple[ProgressRecord, ...]:
    """Linear-ramp telemetry with exactly the aggregate the tests need.

    `c_local` ramps linearly between two endpoints; `c_local_series`
    overrides it with an explicit curve, which is what the plateau tests
    need — a linear ramp can only ever express "climbing" or "pinned",
    and pinned is STAGNANT, not plateaued.
    """
    if c_local_series is not None:
        assert len(c_local_series) == n, "c_local_series must be one per record"
    recs = []
    for i in range(n):
        f = i / max(1, n - 1)
        if c_local_series is not None:
            cl: int | None = int(c_local_series[i])
        elif c_local is not None:
            cl = round(c_local[0] + f * (c_local[1] - c_local[0]))
        else:
            cl = None
        recs.append(ProgressRecord(
            elapsed_s=60 * (i + 1),
            cells=round(cells_start + f * (cells_end - cells_start)),
            steps=round(steps_start + f * (steps_end - steps_start)),
            solutions=solutions if i == n - 1 else 0,
            max_area=(max_area if max_area_end is None or i < n - 1 else max_area_end),
            max_gx=(map_gx if map_gx_end is None or i < n - 1 else map_gx_end),
            stall_flat_windows=frozen,
            c_local=cl,
        ))
    return tuple(recs)


def c_local_shapes(n: int, peak: int) -> dict[str, list[int]]:
    """The three shapes a C_local series can have, by name.

    The distinction the module turns on: `stagnant` never grew (the
    adopted form calls that BARREN), `plateaued` grew and then stopped
    (that is the gated signature), and they are NOT the same curve even
    though both are flat in the trailing window.
    """
    return {
        "stagnant": [peak] * n,
        "plateaued": [min(peak, round(peak * 3 * i / max(1, n - 1)))
                      for i in range(n)],
        "climbing": [round(peak * (i + 1) / n) for i in range(n)],
    }


def archive(*, cells: int, distinct_spatial: int, spatial_span: int,
            boundary_cells: int = 8, entropy: float = 0.95) -> ArchiveSummary:
    return ArchiveSummary(
        cells=cells, distinct_spatial=distinct_spatial,
        spatial_span=spatial_span, boundary_cells=boundary_cells,
        boundary_visit_entropy=entropy, explored_fraction=0.9,
    )


# --------------------------------------------------------------------------
# 1. pure helpers
# --------------------------------------------------------------------------

def test_load_progress_segments_splits_on_elapsed_reset(tmp_path):
    """A retried level appends to the SAME progress.jsonl.

    runs/live_show/smb_4_4_micro/lvl_4-4/progress.jsonl holds five
    attempts back to back; read as one series its max_gx is non-monotone
    and the cell curve saw-tooths, which fabricates 'map advanced' and
    'coverage collapsed' events that never happened.
    """
    p = tmp_path / "progress.jsonl"
    rows = [
        {"elapsed_s": 60, "cells": 10, "steps": 100, "max_gx_in_max_area": 5},
        {"elapsed_s": 120, "cells": 20, "steps": 200, "max_gx_in_max_area": 9},
        {"elapsed_s": 60, "cells": 3, "steps": 90, "max_gx_in_max_area": 4},
        {"elapsed_s": 120, "cells": 8, "steps": 180, "max_gx_in_max_area": 7},
        {"elapsed_s": 180, "cells": 11, "steps": 260, "max_gx_in_max_area": 7},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    segs = load_progress_segments(p)
    assert [len(s) for s in segs] == [2, 3]
    assert segs[-1][-1].cells == 11
    # The naive single-series read would show max_gx dropping 9 -> 4.
    for seg in segs:
        gx = [r.max_gx for r in seg]
        assert gx == sorted(gx)


def test_load_progress_segments_ignores_blank_lines(tmp_path):
    p = tmp_path / "progress.jsonl"
    p.write_text('{"elapsed_s": 60, "cells": 1, "steps": 2}\n\n\n')
    assert [len(s) for s in load_progress_segments(p)] == [1]


def test_record_from_json_tolerates_fields_added_over_time():
    """stall_flat_windows landed 2026-08-06; max_room/doors only appear
    when those arms are on; best_sol_actions is null before a solve."""
    old = record_from_json({"elapsed_s": 60, "cells": 3354, "steps": 134310,
                            "max_area": 0, "max_gx_in_max_area": 751,
                            "max_sect": 0, "solutions": 0,
                            "best_sol_actions": None, "sps": 2238})
    assert old.stall_flat_windows == 0
    assert old.max_room == 0 and old.doors == 0
    assert old.c_local is None
    assert old.solutions == 0


def test_record_from_json_reads_the_forward_compatible_c_local_slot():
    assert record_from_json({"c_local": 1089}).c_local == 1089
    assert record_from_json({"distinct_spatial": 750}).c_local == 750


def test_spatial_key_handles_legacy_and_modern_key_arities():
    """Modern keys are (sect, tb, kk, psig, loops, route_sig) + cell_fn;
    archives banked before that prefix carry a bare 4-tuple."""
    modern = (0, 0, 0, (), 0, (), 3, 7, 0, 12, 95)   # area=3, y=12, gx=95
    legacy = (4, 5, 2, 2)                            # area=4, y=2,  gx=2
    assert wt._spatial_key(modern) == (3, 12, 95)
    assert wt._spatial_key(legacy) == (4, 2, 2)


def test_summarize_archive_cells_computes_concentration_and_span():
    # Two areas; area 1 spans gx buckets 0..2, with 3 cells piled in the
    # deepest bucket (nuisance variation at one location).
    cells = {
        (0, 0, 0, (), 0, (), 0, 0, 0, 1, 0): _FakeCell(5),
        (0, 0, 0, (), 0, (), 1, 0, 0, 1, 0): _FakeCell(5),
        (0, 0, 0, (), 0, (), 1, 0, 0, 1, 1): _FakeCell(5),
        (0, 0, 0, (), 0, (), 1, 0, 0, 1, 2): _FakeCell(4),
        (0, 0, 0, (), 0, (), 1, 0, 1, 1, 2): _FakeCell(4),
        (0, 0, 0, (), 0, (), 1, 0, 2, 1, 2): _FakeCell(4),
    }
    s = summarize_archive_cells(cells)
    assert s.cells == 6
    # (1,1,2) collapses three keys that differ only in a nuisance slot.
    assert s.distinct_spatial == 4
    assert s.spatial_span == 3          # gx buckets 0,1,2 inside area 1
    assert s.boundary_cells == 3        # deepest bucket of the deepest area
    assert s.concentration == pytest.approx(1.5)
    assert s.boundary_visit_entropy == pytest.approx(1.0)  # 4/4/4, even


def test_summarize_archive_cells_empty_is_safe():
    s = summarize_archive_cells({})
    assert s.cells == 0 and s.concentration == 0.0


def test_normalized_entropy_bounds():
    assert wt._normalized_entropy([]) == 0.0
    assert wt._normalized_entropy([7]) == 0.0            # one bin: no information
    assert wt._normalized_entropy([5, 5, 5]) == pytest.approx(1.0)
    assert 0.0 < wt._normalized_entropy([100, 1, 1]) < 0.5


def test_saturation_is_step_normalized_not_time_normalized():
    """The corpus mixes 300 sps paced show runs with 2800 sps headless
    ones. Normalizing coverage yield by STEPS keeps throughput out of the
    statistic: halving sps must not move saturation."""
    fast = make_records(n=30, steps_end=6_000_000)
    slow = make_records(n=30, steps_end=3_000_000)
    # Tolerance covers only the integer rounding in the ramp fixture; the
    # statistic itself is exactly throughput-invariant.
    assert saturation(fast) == pytest.approx(saturation(slow), abs=1e-5)


def test_saturation_detects_a_plateau_and_ignores_a_steady_climb():
    steady = make_records(n=30, cells_start=0, cells_end=30_000)
    assert saturation(steady) == pytest.approx(0.0, abs=1e-5)

    # Front-loaded: all the coverage arrives in the first third.
    recs = []
    for i in range(30):
        recs.append(ProgressRecord(elapsed_s=60 * i,
                                   cells=min(10_000, i * 1000),
                                   steps=i * 100_000))
    assert saturation(tuple(recs)) == pytest.approx(1.0)


def test_saturation_returns_none_when_the_window_has_no_evidence():
    assert saturation(make_records(n=5)) is None
    flat = tuple(ProgressRecord(elapsed_s=60 * i, cells=5, steps=0)
                 for i in range(30))
    assert saturation(flat) is None


def test_saturation_separates_never_grew_from_grew_then_stopped():
    """Both series are flat in the trailing window and they mean OPPOSITE
    things: one never accumulated (STAGNANT -> BARREN), the other
    accumulated and stopped (plateau -> the gated signature).

    Returning 1.0 for the never-grown case made them identical at the
    call site and read as 'maximally saturated' for a search that
    produced nothing at all.
    """
    never_grew = make_records(n=30, cells_start=1089, cells_end=1089,
                              steps_end=10_000_000)
    assert saturation(never_grew) is None
    recent, peak = wt.series_yields(never_grew)
    assert recent == 0.0 and peak == 0.0      # measurable, and zero

    grew_then_stopped = make_records(
        n=30, steps_end=10_000_000,
        c_local_series=c_local_shapes(30, 1089)["plateaued"])
    assert saturation(grew_then_stopped, "c_local") == pytest.approx(1.0)
    assert wt.series_yields(grew_then_stopped, "c_local")[1] > 0

    # And an unmeasurable series stays unmeasurable, not zero.
    assert wt.series_yields(make_records(n=30), "c_local") == (None, None)


def test_map_stall_windows_counts_the_trailing_frozen_run():
    recs = make_records(n=20, map_gx=500)
    assert wt.map_stall_windows(recs) == 19
    moved = make_records(n=20, map_gx=500, map_gx_end=900)
    assert wt.map_stall_windows(moved) == 0


# --------------------------------------------------------------------------
# 2. verdict ordering
# --------------------------------------------------------------------------

def test_too_few_records_is_insufficient():
    v = gated_wall_verdict(WallTelemetry(records=make_records(n=4)))
    assert v.wall_class is WallClass.INSUFFICIENT
    assert v.degraded is True


def test_a_banked_solution_short_circuits_everything():
    tel = WallTelemetry(records=make_records(n=30, solutions=6),
                        archive=archive(cells=605, distinct_spatial=16,
                                        spatial_span=1))
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.RESOLVED
    assert v.degraded is False


def test_map_or_topology_movement_beats_every_wall_test():
    moved_map = WallTelemetry(records=make_records(n=30, map_gx=3844,
                                                   map_gx_end=4290))
    assert gated_wall_verdict(moved_map).wall_class is WallClass.PROGRESSING

    moved_topo = WallTelemetry(records=make_records(n=30, max_area=3,
                                                    max_area_end=4))
    assert gated_wall_verdict(moved_topo).wall_class is WallClass.PROGRESSING


def test_low_effort_window_is_insufficient_not_a_wall():
    """SMB 4-4 segment 0 spent 88,680 steps in its trailing window — a
    throughput-starved show segment, not a wall."""
    tel = WallTelemetry(records=make_records(n=43, steps_end=400_000))
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.INSUFFICIENT
    assert "EFFORT_MIN_STEPS" in v.reasons[0]


def test_degenerate_spatial_projection_is_key_blind_before_barren():
    """Bubble Bobble is one screen: every run pins spatial_span at 1, so
    no coverage statistic over the key's spatial projection means
    anything. KEY_BLIND outranks BARREN because it names WHICH axis is
    missing — the r68 wall fell the moment x entered the cell key."""
    tel = WallTelemetry(
        records=make_records(n=30, cells_start=96, cells_end=96, frozen=29),
        archive=archive(cells=96, distinct_spatial=16, spatial_span=1),
    )
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.KEY_BLIND
    assert "spatial_span" in v.reasons[-1]


def test_frozen_archive_without_an_archive_snapshot_is_barren():
    tel = WallTelemetry(records=make_records(
        n=30, cells_start=96, cells_end=96, frozen=29))
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.BARREN
    assert "frozen_windows" in v.reasons[-1]


def test_trivially_small_archive_is_barren_independently_of_the_stall_flag():
    tel = WallTelemetry(records=make_records(
        n=30, cells_start=40, cells_end=48, frozen=0))
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.BARREN
    assert "COVERAGE_FLOOR_CELLS" in v.reasons[-1]


def test_progress_only_telemetry_never_certifies_gated():
    """The headline safety property, enforced as a property.

    Without an archive snapshot the cross-sectional term is unmeasured
    and the series term rests on a threshold that has never been
    measured against a labelled run, so the module must abstain rather
    than guess — the Castlevania hall and a mid-run SMB 8-4 look
    IDENTICAL on progress.jsonl alone.

    This used to exercise exactly ONE hand-picked telemetry, one that
    happened to carry no `c_local`; the property it claimed was false
    for any telemetry that did carry one. It is now asserted over every
    corpus run crossed with every C_local shape, which is the state §8
    of the receipt is asking the fleet to move to.
    """
    hall = WallTelemetry(records=make_records(
        n=89, cells_start=3354, cells_end=91_995, steps_end=10_643_480))
    v = gated_wall_verdict(hall)
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.degraded is True
    assert "c_local" in v.missing

    offenders = []
    for label, _truth, _expected, kw, arc in CORPUS:
        peak = arc.distinct_spatial if arc is not None else 1000
        shapes = dict(c_local_shapes(kw["n"], peak), none=None)
        for name, series in shapes.items():
            tel = WallTelemetry(
                records=make_records(**kw, c_local_series=series),
                archive=None, label=label)
            if gated_wall_verdict(tel).wall_class is WallClass.GATED:
                offenders.append(f"{label} [c_local={name}]")
    assert offenders == [], (
        f"progress-only telemetry certified GATED for: {offenders}")


def test_saturated_local_coverage_at_a_frozen_boundary_is_gated():
    tel = WallTelemetry(
        records=make_records(n=89, cells_start=3354, cells_end=91_995,
                             steps_end=10_643_480),
        archive=archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                        boundary_cells=13, entropy=0.9837),
    )
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.GATED
    assert v.evidence["concentration"] >= wt.CONCENTRATION_GATED_MIN
    assert v.evidence["topo_delta"] == 0 and v.evidence["map_delta"] == 0
    # Still degraded: the entropy term of the adopted form is unmeasured.
    assert v.degraded is True
    assert "boundary_action_entropy" in v.missing


def test_a_still_expanding_map_footprint_is_coverage_limited():
    tel = WallTelemetry(
        records=make_records(n=21, cells_start=5100, cells_end=5885,
                             steps_end=3_335_420),
        archive=archive(cells=5885, distinct_spatial=750, spatial_span=130),
    )
    assert gated_wall_verdict(tel).wall_class is WallClass.COVERAGE_LIMITED


# --------------------------------------------------------------------------
# 2b. the C_local series path — the state §8 asks the fleet to move to
#
# `c_local` does not exist in any banked run, so this whole world is
# reachable only by construction. It is also the world the receipt ranks
# as its #1 runtime requirement, which makes it the one that must be
# nailed down BEFORE the field lands rather than after.
# --------------------------------------------------------------------------

def _series_tel(shape: str, *, peak: int, arc=None, n: int = 30,
                cells_end: int = 90_000, **kw) -> WallTelemetry:
    return WallTelemetry(
        records=make_records(n=n, cells_start=1000, cells_end=cells_end,
                             c_local_series=c_local_shapes(n, peak)[shape],
                             **kw),
        archive=arc)


def test_a_pinned_c_local_series_is_barren_not_gated():
    """`BARREN <=> C_local STAGNANT` is the module's own adopted form.

    This case used to read GATED — and it is the worst possible verdict
    to get wrong here, because GATED spends an orthogonal campaign on a
    search that never accumulated anything. The mechanism was
    `saturation()` reporting 1.0 for a series with zero peak yield: a
    curve that NEVER GREW scored as maximally plateaued.
    """
    v = gated_wall_verdict(_series_tel(
        "stagnant", peak=1089,
        arc=archive(cells=90_000, distinct_spatial=1089, spatial_span=95)))
    assert v.wall_class is WallClass.BARREN
    assert "STAGNANT, not plateaued" in v.reasons[-1]
    assert v.evidence["c_local_peak_yield"] == 0
    assert v.evidence["c_local_saturation"] is None

    # Same series, no archive: still BARREN, never GATED.
    assert gated_wall_verdict(
        _series_tel("stagnant", peak=1089)).wall_class is WallClass.BARREN


def test_a_c_local_series_below_the_floor_is_key_blind_either_way():
    """The divergence the floor closes.

    Every Bubble Bobble profile is span-degenerate: one screen, 16
    distinct spatial buckets, progress byte constant. WITH an archive
    the module has always said KEY_BLIND. WITHOUT one — the moment
    `c_local` starts being emitted and the archive has not flushed — the
    identical run used to say GATED, because the archive's
    `spatial_span` guard was the only place that check lived.
    """
    bb_arc = archive(cells=90_000, distinct_spatial=16, spatial_span=1,
                     boundary_cells=90_000, entropy=0.87)
    for shape in ("stagnant", "plateaued", "climbing"):
        with_arc = gated_wall_verdict(_series_tel(shape, peak=16, arc=bb_arc))
        without = gated_wall_verdict(_series_tel(shape, peak=16))
        assert with_arc.wall_class is WallClass.KEY_BLIND, shape
        assert without.wall_class is WallClass.KEY_BLIND, shape
    assert "C_LOCAL_FLOOR_BUCKETS" in gated_wall_verdict(
        _series_tel("plateaued", peak=16)).reasons[-1]


def test_a_tiny_c_local_proves_the_span_guard_would_have_fired():
    """`c_local` counts (area, y_band, gx_bucket) triples and
    `spatial_span` counts gx buckets inside the deepest area, so
    spatial_span <= c_local always. Below SPATIAL_SPAN_MIN the series
    alone therefore PROVES the archive path's guard, rather than
    merely agreeing with a separately calibrated floor."""
    v = gated_wall_verdict(_series_tel("plateaued", peak=4))
    assert v.wall_class is WallClass.KEY_BLIND
    assert "PROVES spatial_span is degenerate" in v.reasons[-1]


def test_a_climbing_c_local_series_overrides_a_gated_concentration():
    """The one direction the series is trusted alone: refutation.

    A still-expanding map footprint cannot be a saturated wall whatever
    the cross-section says, and this direction can only ever move a
    verdict AWAY from GATED, so it costs nothing if the PROVISIONAL
    threshold is off."""
    hall_arc = archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                       boundary_cells=13, entropy=0.9837)
    assert gated_wall_verdict(
        WallTelemetry(records=make_records(n=30, cells_start=3354,
                                           cells_end=91_995),
                      archive=hall_arc)).wall_class is WallClass.GATED
    v = gated_wall_verdict(_series_tel("climbing", peak=5000, arc=hall_arc))
    assert v.wall_class is WallClass.COVERAGE_LIMITED
    assert "still expanding" in v.reasons[-1]


def test_a_c_local_plateau_certifies_gated_only_when_corroborated():
    """The promotion direction, which the PROVISIONAL threshold does not
    yet license on its own.

    Corroborated by concentration -> GATED (two statistics agreeing is
    strictly stronger than today's archive-only verdict). Contradicted,
    or with nothing to corroborate it -> INDETERMINATE, whose remedy is
    'collect the missing telemetry', which is exactly right for a
    threshold nobody has measured."""
    corroborating = archive(cells=92_785, distinct_spatial=1089,
                            spatial_span=95, boundary_cells=13, entropy=0.98)
    v = gated_wall_verdict(_series_tel("plateaued", peak=1089,
                                       arc=corroborating))
    assert v.wall_class is WallClass.GATED
    assert "corroborated by concentration" in v.reasons[-1]
    assert v.degraded is True and "boundary_action_entropy" in v.missing

    # ge_chain/lvl_11_4-4's geometry: a resolved coverage wall, 7.85.
    contradicting = archive(cells=5885, distinct_spatial=750, spatial_span=130)
    v = gated_wall_verdict(_series_tel("plateaued", peak=750, cells_end=5885,
                                       arc=contradicting))
    assert v.wall_class is WallClass.INDETERMINATE
    assert "CONTRADICTS the series" in v.reasons[-1]

    v = gated_wall_verdict(_series_tel("plateaued", peak=1089))
    assert v.wall_class is WallClass.INDETERMINATE
    assert "no archive snapshot to corroborate it" in v.reasons[-1]


def test_the_series_path_is_promoted_by_calibrating_its_threshold(monkeypatch):
    """What flipping the switch buys, so the promotion step is a tested
    one-line change rather than a rewrite. Ship state stays False."""
    assert wt.C_LOCAL_SERIES_MAY_CERTIFY_GATED is False
    monkeypatch.setattr(wt, "C_LOCAL_SERIES_MAY_CERTIFY_GATED", True)
    assert gated_wall_verdict(
        _series_tel("plateaued", peak=1089)).wall_class is WallClass.GATED
    # The guards in front of it still hold after promotion.
    assert gated_wall_verdict(
        _series_tel("stagnant", peak=1089)).wall_class is WallClass.BARREN
    assert gated_wall_verdict(
        _series_tel("plateaued", peak=16)).wall_class is WallClass.KEY_BLIND


def test_missing_reports_what_this_call_lacked_not_what_the_fleet_lacks():
    """`missing` is per-verdict evidence, not a static shopping list.

    Telling a caller that supplied a usable C_local series that
    `c_local` is missing sends it collecting a field it already has, and
    hides the one term that really is unmeasured everywhere.
    """
    no_series = gated_wall_verdict(WallTelemetry(
        records=make_records(n=89, cells_start=3354, cells_end=91_995,
                             steps_end=10_643_480)))
    assert "c_local" in no_series.missing

    with_series = gated_wall_verdict(_series_tel("plateaued", peak=1089))
    assert "c_local" not in with_series.missing
    assert "boundary_action_entropy" in with_series.missing

    # Present but too short to yield a series still counts as missing.
    short = make_records(n=30, steps_end=10_000_000)
    patched = short[:-1] + (
        ProgressRecord(**{**short[-1].__dict__, "c_local": 900}),)
    v = gated_wall_verdict(WallTelemetry(records=patched))
    assert v.evidence["c_local"] == 900
    assert v.evidence["c_local_peak_yield"] is None
    assert "c_local" in v.missing


def test_emitting_c_local_never_upgrades_a_banked_run_to_gated():
    """PARITY, as a property over the whole corpus.

    A run's verdict must not become GATED merely because the solver
    started emitting a field. Only the GATED direction is asserted:
    other movements are legitimate (a hypothetically pinned C_local on a
    coverage-limited run IS barren by the adopted form), but inventing a
    positive is the failure this lane exists to prevent.
    """
    offenders = []
    for label, _truth, _expected, kw, arc in CORPUS:
        peak = arc.distinct_spatial if arc is not None else 1000
        baseline = gated_wall_verdict(
            WallTelemetry(records=make_records(**kw), archive=arc)).wall_class
        for name, series in c_local_shapes(kw["n"], peak).items():
            for a in (arc, None):
                tel = WallTelemetry(
                    records=make_records(**kw, c_local_series=series),
                    archive=a, label=label)
                got = gated_wall_verdict(tel).wall_class
                if got is WallClass.GATED and baseline is not WallClass.GATED:
                    offenders.append(
                        f"{label} [c_local={name}, archive={a is not None}] "
                        f"{baseline.value} -> gated")
    assert offenders == [], f"c_local invented a GATED verdict: {offenders}"


def test_every_verdict_carries_a_remedy_and_serializes():
    tel = WallTelemetry(records=make_records(n=30), label="unit")
    v = gated_wall_verdict(tel)
    assert v.remedy == wt.REMEDY[v.wall_class]
    assert v.calibration == wt.CALIBRATION_TAG
    json.dumps(v.as_dict())  # must not raise


# --------------------------------------------------------------------------
# 3. frozen calibration corpus
# --------------------------------------------------------------------------

#: Statistics measured on the banked runs during the 2026-08-10 offline
#: calibration, replayed as synthetic telemetry. Columns:
#:   label, ground truth, expected verdict, telemetry kwargs, archive
#: See docs/receipts/dispatch/gated_wall_calibration_2026-08-10.md.
#:
#: GROUND-TRUTH HEALTH WARNING. The negatives here are solid — every
#: `resolved` row was actually solved. The POSITIVES are not: both are
#: `lvl_03_trace`, i.e. the same Castlevania hall from two hardware-flag
#: lineages, and that hall has never been solved. Nothing receipts it as
#: gated; the only thing pointing that way is that an orthogonal arm was
#: launched at it, which is the conclusion, not the evidence. Every row
#: marked PENDING-VALIDATION is conditional on `runs/cv_hall_ortho_a`
#: reading out — see §9 of the receipt for what happens to these bands
#: if it reads out COVERAGE instead.
CORPUS = [
    ("cv_chain_hw2/lvl_03_trace", "gated (PENDING-VALIDATION)", WallClass.GATED,
     dict(n=89, cells_start=3354, cells_end=91_995, steps_end=10_643_480,
          map_gx=767),
     archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
             boundary_cells=13, entropy=0.9837)),
    ("cv_chain_hw/lvl_03_trace", "gated (PENDING-VALIDATION, same hall)",
     WallClass.GATED,
     dict(n=14, cells_start=17_000, cells_end=27_619, steps_end=2_404_020,
          map_gx=767),
     archive(cells=28_929, distinct_spatial=932, spatial_span=94,
             boundary_cells=5, entropy=0.9697)),
    # FROZEN at the 24-record, pre-flush snapshot the original probe saw.
    # The live directory has since flushed an archive and its verdict has
    # moved (see test_the_live_ortho_arm_now_reads_gated_and_is_excluded).
    # This row is kept as a fixture of the DEGRADED path, not as evidence:
    # it is the arm whose read-out the calibration is conditional on.
    ("cv_hall_ortho_a @24 records (pre-flush snapshot)", "PENDING-VALIDATION",
     WallClass.INDETERMINATE,
     dict(n=24, cells_start=93_720, cells_end=107_890, steps_end=2_440_150,
          map_gx=767),
     None),
    ("bubble_bobble/r68_retry_ortho", "orthogonal/key", WallClass.KEY_BLIND,
     dict(n=30, cells_start=96, cells_end=96, steps_end=3_675_730, frozen=29),
     archive(cells=96, distinct_spatial=16, spatial_span=1,
             boundary_cells=96, entropy=0.7775)),
    ("bubble_bobble/r69_retry_ortho", "orthogonal/key", WallClass.KEY_BLIND,
     dict(n=30, cells_start=40, cells_end=48, steps_end=3_153_184, frozen=19),
     archive(cells=48, distinct_spatial=8, spatial_span=1,
             boundary_cells=48, entropy=0.78)),
    ("bubble_bobble/r68_retry_xsig", "resolved", WallClass.RESOLVED,
     dict(n=30, cells_start=224, cells_end=605, steps_end=3_570_080,
          solutions=6, frozen=4),
     archive(cells=605, distinct_spatial=16, spatial_span=1,
             boundary_cells=605, entropy=0.8253)),
    ("bubble_bobble/r99_retry2", "orthogonal/key", WallClass.KEY_BLIND,
     dict(n=30, cells_start=300, cells_end=691, steps_end=3_544_480, frozen=4),
     archive(cells=691, distinct_spatial=16, spatial_span=1,
             boundary_cells=691, entropy=0.8583)),
    ("bubble_bobble/r99_1_boss_retry", "unknown/mechanic", WallClass.KEY_BLIND,
     dict(n=30, cells_start=2931, cells_end=9475, steps_end=3_249_620),
     archive(cells=9475, distinct_spatial=32, spatial_span=1,
             boundary_cells=9475, entropy=0.8991)),
    ("bubble_bobble/chain_day2h_item/lvl_00_99-1", "unknown/mechanic",
     WallClass.KEY_BLIND,
     dict(n=45, cells_start=1462, cells_end=2989, steps_end=5_188_710),
     archive(cells=2989, distinct_spatial=16, spatial_span=1,
             boundary_cells=2989, entropy=0.8761)),
    ("live_show/smb_4_4_micro/lvl_4-4 seg1", "coverage (resolved)",
     WallClass.INDETERMINATE,
     dict(n=44, cells_start=50_366, cells_end=1_020_500, steps_end=6_800_000,
          map_gx=2059),
     None),
    ("live_show/smb_4_4_micro/lvl_4-4 seg4", "coverage (resolved)",
     WallClass.PROGRESSING,
     dict(n=50, cells_start=51_853, cells_end=1_164_599, steps_end=7_286_076,
          map_gx=2055, map_gx_end=2575),
     None),
    ("live_show/smb_4_4_micro/lvl_8-4", "coverage (resolved)",
     WallClass.PROGRESSING,
     dict(n=56, cells_start=45_035, cells_end=1_190_873, steps_end=7_203_648,
          map_gx=3844, map_gx_end=4290),
     None),
    ("ge_chain/lvl_11_4-4", "coverage (resolved)", WallClass.COVERAGE_LIMITED,
     dict(n=21, cells_start=5100, cells_end=5885, steps_end=3_335_420,
          map_gx=2068),
     archive(cells=5885, distinct_spatial=750, spatial_span=130,
             boundary_cells=8, entropy=0.9728)),
]


@pytest.mark.parametrize("label,truth,expected,kw,arc", CORPUS,
                         ids=[c[0] for c in CORPUS])
def test_corpus_verdicts(label, truth, expected, kw, arc):
    tel = WallTelemetry(records=make_records(**kw), archive=arc, label=label)
    v = gated_wall_verdict(tel)
    assert v.wall_class is expected, (
        f"{label} (ground truth {truth}) -> {v.wall_class.value}; "
        f"reasons={v.reasons}")


def test_no_false_gated_anywhere_in_the_corpus():
    """No FALSE positives: every run whose ground truth is known — every
    resolved coverage wall, every representation-limited Bubble Bobble
    run — must land somewhere other than GATED, in both the full and the
    degraded path.

    The converse is NOT asserted, because it cannot be: the only two
    rows that do read GATED are the same unsolved Castlevania hall seen
    twice, so this test proves the module does not cry wolf, not that it
    can find a wolf. See the CORPUS health warning.
    """
    gated = set()
    for label, _truth, _expected, kw, arc in CORPUS:
        for a in (arc, None):
            tel = WallTelemetry(records=make_records(**kw), archive=a)
            if gated_wall_verdict(tel).wall_class is WallClass.GATED:
                gated.add(label)
    assert gated == {"cv_chain_hw2/lvl_03_trace",
                     "cv_chain_hw/lvl_03_trace"}, (
        "the GATED set must stay exactly the (unvalidated) hall pair")


def test_the_live_ortho_arm_now_reads_gated_and_is_excluded():
    """`runs/cv_hall_ortho_a` flushed an archive; its verdict moved.

    Measured 2026-08-10 11:26 from the real 2,436,606,838-byte
    `archive.pkl`: 114,699 cells over 1,095 distinct spatial buckets,
    span 95 -> concentration 104.75, which is 4.2x
    CONCENTRATION_GATED_MIN. The run reads GATED, not the INDETERMINATE
    an earlier read of the same directory recorded when no archive had
    been flushed yet.

    It is deliberately NOT in CORPUS, and the exclusion is the point.
    It is a THIRD read of the same Castlevania hall, so it adds no
    independent evidence; and it is the pending-validation arm itself,
    so scoring it and counting the result as confirmation would be
    grading the experiment with the instrument under test.
    """
    measured = archive(cells=114_699, distinct_spatial=1095, spatial_span=95,
                       boundary_cells=16, entropy=0.983)
    assert measured.concentration == pytest.approx(104.748, abs=0.01)
    assert measured.concentration > 4 * wt.CONCENTRATION_GATED_MIN
    tel = WallTelemetry(
        records=make_records(n=56, cells_start=93_720, cells_end=119_535,
                             steps_end=5_621_170, map_gx=767),
        archive=measured, label="cv_hall_ortho_a @56 records")
    assert gated_wall_verdict(tel).wall_class is WallClass.GATED

    corpus_labels = {c[0] for c in CORPUS}
    assert not any(lab.startswith("cv_hall_ortho_a @56") for lab in corpus_labels)
    assert not any(d[0].endswith("cv_hall_ortho_a") for d in LIVE), (
        "the live, still-mutating ortho directory must not be a regression "
        "fixture: its statistics move between reads")


def test_shipped_constants_sit_inside_their_measured_separating_bands():
    """Each band is (nearest counter-example, nearest positive]. Moving a
    constant outside its band silently reclassifies a banked run, so the
    bands are asserted here rather than living only in the receipt.

    Every bracket below must come from a BANKED run. The earlier
    EFFORT_MIN_STEPS bracket was taken from `cv_hall_ortho_a` while it
    was still running, and it drifted (1,010,590 -> 968,490 over the
    following hour) — a live run cannot bracket a frozen constant.
    """
    # (20.58 = SMB 8-3, resolved) .. (31.04 = CV hall hw, gated)
    assert 20.58 < wt.CONCENTRATION_GATED_MIN <= 31.04
    # (1 = every Bubble Bobble run) .. (94 = CV hall hw)
    assert 1 < wt.SPATIAL_SPAN_MIN <= 94
    # (32 = BB 99-1 boss retry, the largest degenerate spatial projection)
    # .. (638 = ge_1_4_solve, the smallest spatially resolved archive).
    # Same column as SPATIAL_SPAN_MIN's band, one level up the projection.
    assert 32 < wt.C_LOCAL_FLOOR_BUCKETS <= 638
    # (88,680 = starved SMB 4-4 seg0) .. (932,340 = SMB 8-4's trailing
    # window, the smallest BANKED case that must still be classified).
    assert 88_680 < wt.EFFORT_MIN_STEPS <= 932_340
    # (96 = BB r68 ortho, frozen) .. (605 = BB r68 xsig, the smallest live one)
    assert 96 < wt.COVERAGE_FLOOR_CELLS <= 605
    # (7 = BB r99 retry, alive) .. (19 = BB r69 ortho, frozen)
    assert 7 < wt.FROZEN_WINDOWS_MAX <= 19
    # 14 records is the shortest corpus run that must still be classified.
    assert wt.WINDOW_RECORDS < wt.MIN_RECORDS <= 14
    # Uncalibrated thresholds may not certify the expensive verdict.
    assert wt.C_LOCAL_SERIES_MAY_CERTIFY_GATED is False


def test_refuted_statistics_are_reported_but_never_gated_on():
    """Raw-cell saturation is the obvious candidate for 'C_local
    plateau' and it is wrong: on the corpus the gated hall straddles two
    resolved coverage walls in BOTH directions."""
    assert wt.RAW_COVERAGE_SATURATION_IS_SEPARATING is False
    assert wt.CHURN_IS_SEPARATING is False
    assert wt.BOUNDARY_ENTROPY_IS_SEPARATING is False
    assert wt.MAP_STALL_WINDOWS_IS_SEPARATING is False

    measured = {                     # raw-cell saturation, trailing 10 records
        "cv_hall_hw (gated)": 0.179,
        "smb_8-4 (resolved)": 0.189,
        "cv_hall_hw2 (gated)": 0.343,
        "smb_4-4 seg4 (resolved)": 0.352,
    }
    gated_vals = [v for k, v in measured.items() if "gated" in k]
    resolved = [v for k, v in measured.items() if "resolved" in k]
    assert min(gated_vals) < min(resolved) < max(gated_vals) < max(resolved), (
        "the corpus interleaves; no raw-saturation threshold separates it")

    # And the statistic is still reported, so a human can see it.
    tel = WallTelemetry(records=make_records(n=30))
    assert "raw_coverage_saturation" in gated_wall_verdict(tel).evidence
    assert "churn_per_window" in gated_wall_verdict(tel).evidence


def test_missing_telemetry_names_the_runtime_gaps():
    for key in ("c_local", "boundary_action_entropy", "frontier_bucket_cells"):
        assert key in wt.MISSING_TELEMETRY
        assert len(wt.MISSING_TELEMETRY[key]) > 40


#: The ONE runtime reference this module tolerates: the gate-opener arm
#: reads the PURE `boundary_axis_profile` off a flush snapshot for its
#: `boundary_state_axes` / `alias_ratio` telemetry. Everything that
#: CLASSIFIES — `gated_wall_verdict`, `WallClass`, the thresholds — stays
#: operator-read between sessions.
_TAXONOMY_PURE_READER = "scripts/go_explore_solve.py"


def test_module_is_not_wired_into_any_runtime_dispatch():
    """Self-arming is a later decision (D2 verdict). Nothing outside
    tests and docs may import this module, with exactly one exception:
    the gate-opener arm's pure `boundary_axis_profile` read (taxonomy
    KEYED, never taxonomy-WIRED — the companion test below pins how
    narrow that exception is).
    """
    importers = []
    for root in ("src", "scripts", "nes_core", "configs"):
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if path.name == "wall_taxonomy.py":
                continue
            rel = str(path.relative_to(REPO))
            if rel == _TAXONOMY_PURE_READER:
                continue
            if "wall_taxonomy" in path.read_text(errors="ignore"):
                importers.append(rel)
    assert importers == [], f"wall_taxonomy is wired into: {importers}"


def test_the_one_tolerated_reader_imports_only_the_pure_profile():
    """And the exception is exactly as narrow as it claims: the solver
    imports `boundary_axis_profile` and nothing else, lazily, inside the
    background flush — never at module scope and never in the hot loop.
    """
    import ast

    src = (REPO / _TAXONOMY_PURE_READER).read_text()
    tree = ast.parse(src)
    imported, at_module_scope = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "src.training.wall_taxonomy":
            continue
        imported += [a.name for a in node.names]
    for node in tree.body:                       # module-scope statements
        if isinstance(node, ast.ImportFrom) and node.module and \
                "wall_taxonomy" in node.module:
            at_module_scope.append(node.module)
    assert imported == ["boundary_axis_profile"], imported
    assert at_module_scope == [], at_module_scope
    # ...and no VERDICT function is reachable from the runtime at all.
    assert "gated_wall_verdict(" not in src
    assert "WallClass" not in src


# --------------------------------------------------------------------------
# 4. live regression against the banked runs (skipped without them)
#
# Only FINISHED run directories belong here. `runs/cv_hall_ortho_a` is
# excluded on purpose: it is still being written, so its statistics move
# between reads, and it is the arm the calibration is conditional on.
# The two GATED expectations below are the unvalidated hall pair — they
# assert reproducibility of the statistic, not correctness of the label.
# --------------------------------------------------------------------------

LIVE = [
    ("runs/cv_chain_hw2/lvl_03_trace", True, -1, WallClass.GATED),
    ("runs/cv_chain_hw/lvl_03_trace", True, -1, WallClass.GATED),
    ("runs/bubble_bobble/r68_retry_ortho", True, -1, WallClass.KEY_BLIND),
    ("runs/bubble_bobble/r68_retry_xsig", True, -1, WallClass.RESOLVED),
    ("runs/bubble_bobble/r99_1_boss_retry", True, -1, WallClass.KEY_BLIND),
    ("runs/live_show/smb_4_4_micro/lvl_8-4", False, -1, WallClass.PROGRESSING),
    ("runs/live_show/smb_4_4_micro/lvl_4-4", False, 1, WallClass.INDETERMINATE),
    ("runs/ge_chain/lvl_11_4-4", True, -1, WallClass.COVERAGE_LIMITED),
]


@pytest.mark.parametrize("rundir,with_archive,segment,expected", LIVE,
                         ids=[c[0] for c in LIVE])
def test_live_corpus_reproduces_the_calibrated_verdicts(
        rundir, with_archive, segment, expected):
    base = REPO / rundir
    progress = base / "progress.jsonl"
    arch = base / "archive.pkl"
    if not progress.exists() or (with_archive and not arch.exists()):
        pytest.skip(f"{rundir} not present (runs/ is gitignored)")
    tel = wt.telemetry_from_paths(
        progress, archive_path=(arch if with_archive else None),
        segment=segment, label=rundir)
    assert gated_wall_verdict(tel).wall_class is expected
