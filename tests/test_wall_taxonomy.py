"""Tests for src/training/wall_taxonomy.py — the wall taxonomy.

Five layers:

1. Unit tests over the pure helpers (segmentation, key projection, the
   saturation statistic, the archive summary, the sidecar adapter).
2. Verdict ordering, plus (2b) the C_local SERIES path — the world the
   receipts ask the fleet to move to. No banked run emits `c_local`, so
   that world is reachable only by construction, which is exactly why
   it is pinned down here rather than after the field lands.
3. A frozen CORPUS fixture holding the statistics each banked run in
   the 2026-08-10 calibration actually produced, replayed through
   synthetic telemetry. `runs/` is gitignored, so this layer carries
   the corpus without needing the multi-GB archives on disk.
4. The STRIKE, and its mutation guards. `WallClass.GATED` was removed
   on 2026-08-11; re-adding it — the branch, the enum member, or a
   verdict that turns on `concentration` by any other name — must fail
   a test here. That is the point of `test_re_adding_a_gated_class_...`
   and of `test_the_descriptive_threshold_moves_no_verdict`.
5. A live regression that re-derives the same verdicts from the real
   run directories, skipped when they are absent.

WHY THE POSITIVE CLASS IS GONE. This file used to assert that two rows
read GATED and that nothing else did. Both receipts below killed that:

  docs/receipts/dispatch/k_falsifier_2026-08-10.md
      `ge_chain_w8/lvl_00_8-1` SOLVED at concentration 98.30 — 3.2x the
      "gated" upper bracket — inside one chain whose siblings read
      15.66 and 20.58.
  docs/receipts/dispatch/size_decoupled_statistic_2026-08-11.md
      22 candidate replacements over 103 archives. All straddle. The
      best cut in the whole set still condemns 3 of 13 solved archives.

So there is nothing here that asserts a wall can be detected, because
nothing measured supports it. What is asserted is the half that
survived: the subtractive classes (BARREN, KEY_BLIND, INSUFFICIENT),
the two directly-observed ones (RESOLVED, PROGRESSING), the one
direction a C_local series still licenses (COVERAGE_LIMITED), and the
abstention everything else falls into.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.training import wall_taxonomy as wt
from src.training.wall_taxonomy import (
    ArchiveCounters,
    ArchiveSummary,
    ProgressRecord,
    WallClass,
    WallTelemetry,
    gated_wall_verdict,
    load_progress_segments,
    read_archive_counters,
    record_from_json,
    saturation,
    summarize_archive_cells,
)

REPO = Path(__file__).resolve().parent.parent

#: Every class `gated_wall_verdict` is allowed to return. Enumerated so
#: that ADDING one is a deliberate, visible act rather than a diff nobody
#: reads — the GATED branch got shipped exactly once, and quietly.
LICENSED_CLASSES = {
    WallClass.RESOLVED,
    WallClass.PROGRESSING,
    WallClass.COVERAGE_LIMITED,
    WallClass.BARREN,
    WallClass.KEY_BLIND,
    WallClass.INDETERMINATE,
    WallClass.INSUFFICIENT,
}

#: The `*_IS_SEPARATING`-family constants and the receipt section that
#: kills each. A candidate statistic that has been scored and refuted
#: lives here so nobody re-derives it and believes it.
REFUTED_CONSTANTS = {
    # struck 2026-08-10, by the original calibration
    "RAW_COVERAGE_SATURATION_IS_SEPARATING": "2026-08-10",
    "CHURN_IS_SEPARATING": "2026-08-10",
    "BOUNDARY_ENTROPY_IS_SEPARATING": "2026-08-10",
    "MAP_STALL_WINDOWS_IS_SEPARATING": "2026-08-10",
    # struck 2026-08-11, by the falsifier and the replacement search
    "CONCENTRATION_IS_SEPARATING": "2026-08-11",
    "SIZE_PARTIALED_CONCENTRATION_IS_SEPARATING": "2026-08-11",
    "NOVELTY_PER_RECORD_IS_SEPARATING": "2026-08-11",
    "EXPLORED_FRACTION_IS_SEPARATING": "2026-08-11",
    "SPATIAL_EVENNESS_IS_SEPARATING": "2026-08-11",
    "GROWTH_EXPONENT_IS_SEPARATING": "2026-08-11",
    "EFFORT_MATCHED_PERCENTILE_IS_SEPARATING": "2026-08-11",
    "DOORS_IS_MONOTONE": "2026-08-11",
}


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
    max_room: int = 0,
    max_room_end: int | None = None,
    doors: int = 0,
    doors_end: int | None = None,
    frozen: int = 0,
    c_local: tuple[int, int] | None = None,
    c_local_series: Sequence[int] | None = None,
) -> tuple[ProgressRecord, ...]:
    """Linear-ramp telemetry with exactly the aggregate the tests need.

    `c_local` ramps linearly between two endpoints; `c_local_series`
    overrides it with an explicit curve, which is what the plateau tests
    need — a linear ramp can only ever express "climbing" or "pinned",
    and pinned is STAGNANT, not plateaued.

    The `*_end` parameters (`map_gx_end`, `max_area_end`, `max_room_end`,
    `doors_end`) move a counter on the LAST record only, which makes the
    trailing-window delta exactly `end - start` rather than a fraction of
    a ramp. `doors` uses that to reproduce a measured window delta to the
    unit — see `test_doors_churn_no_longer_reads_as_topological_progress`.
    """
    if c_local_series is not None:
        assert len(c_local_series) == n, "c_local_series must be one per record"
    recs = []
    for i in range(n):
        f = i / max(1, n - 1)
        last = i == n - 1
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
            solutions=solutions if last else 0,
            max_area=(max_area if max_area_end is None or not last else max_area_end),
            max_room=(max_room if max_room_end is None or not last else max_room_end),
            max_gx=(map_gx if map_gx_end is None or not last else map_gx_end),
            doors=(doors if doors_end is None or not last else doors_end),
            stall_flat_windows=frozen,
            c_local=cl,
        ))
    return tuple(recs)


def c_local_shapes(n: int, peak: int) -> dict[str, list[int]]:
    """The three shapes a C_local series can have, by name.

    The distinction the module turns on: `stagnant` never grew (the
    adopted form calls that BARREN), `plateaued` grew and then stopped,
    and they are NOT the same curve even though both are flat in the
    trailing window.
    """
    return {
        "stagnant": [peak] * n,
        "plateaued": [min(peak, round(peak * 3 * i / max(1, n - 1)))
                      for i in range(n)],
        "climbing": [round(peak * (i + 1) / n) for i in range(n)],
    }


def archive(*, cells: int, distinct_spatial: int, spatial_span: int,
            boundary_cells: int = 8, entropy: float = 0.95,
            explored: float = 0.9) -> ArchiveSummary:
    return ArchiveSummary(
        cells=cells, distinct_spatial=distinct_spatial,
        spatial_span=spatial_span, boundary_cells=boundary_cells,
        boundary_visit_entropy=entropy, explored_fraction=explored,
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


def test_record_from_json_still_parses_doors_after_it_stopped_counting():
    """`doors` left `topo_delta` on 2026-08-11 but not the record.

    Dropping the FIELD would have been the wrong repair: the counter is
    real telemetry and a cumulative variant of it is admissible (see
    MISSING_TELEMETRY['doors_cumulative']). What was wrong was summing a
    non-monotone quantity into a monotone one.
    """
    rec = record_from_json({"elapsed_s": 60, "cells": 10_492, "steps": 159_152,
                            "doors": 1621, "edges": 16_286})
    assert rec.doors == 1621


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
    accumulated and stopped.

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
# 1b. the archive.stats.json sidecar adapter (reporting only)
# --------------------------------------------------------------------------

#: The real sidecar of `runs/cv_chain_hw/lvl_03_overnight`, byte for
#: byte. 279 sidecars were surveyed: all 279 carry these six fields, 72
#: also carry `hw_provenance`.
OVERNIGHT_SIDECAR = {
    "cells": 560_410, "frontier": 181_425, "best_score": 767,
    "records": 45_640_527, "new_cells": 560_410, "improvements": 827_786,
}


def _write_sidecar(dirpath: Path, payload) -> Path:
    p = dirpath / wt.ARCHIVE_STATS_FILENAME
    p.write_text(json.dumps(payload))
    return p


def test_read_archive_counters_parses_the_sidecar(tmp_path):
    _write_sidecar(tmp_path, OVERNIGHT_SIDECAR)
    c = read_archive_counters(tmp_path)
    assert c == ArchiveCounters(cells=560_410, records=45_640_527,
                                new_cells=560_410, improvements=827_786,
                                frontier=181_425, best_score=767.0)


def test_the_sidecar_reproduces_the_receipt_s_own_columns(tmp_path):
    """The two derived columns, checked against the numbers the receipt
    published for this exact run.

    `explored_fraction` is `1 - frontier/cells` and the receipt's §7
    coupon-collector table lists 0.6763 for `cv_chain_hw/lvl_03_overnight`;
    `nu = cells/records` is the §9 candidate and the hall's band is
    0.0087 - 0.0165. Both are computed here from the sidecar alone, which
    is the whole point of §2.2: no multi-GB unpickle is needed to get the
    effort denominator.
    """
    _write_sidecar(tmp_path, OVERNIGHT_SIDECAR)
    c = read_archive_counters(tmp_path)
    assert c.explored_fraction == pytest.approx(0.6763, abs=5e-5)
    assert c.novelty_per_record == pytest.approx(0.012278, abs=5e-6)
    assert 0.0087 <= c.novelty_per_record <= 0.0165     # inside the hall band


def test_read_archive_counters_resolves_a_dir_a_pkl_or_the_sidecar(tmp_path):
    """Callers hold whichever of the three they were given."""
    sidecar = _write_sidecar(tmp_path, OVERNIGHT_SIDECAR)
    expected = read_archive_counters(sidecar)
    assert expected is not None
    assert read_archive_counters(tmp_path) == expected
    assert read_archive_counters(tmp_path / "archive.pkl") == expected


def test_read_archive_counters_carries_hw_provenance_when_present(tmp_path):
    """72 of 279 sidecars carry it, and a lineage mismatch is exactly
    what the D3 adjudication had to catch by hand."""
    prov = {"hw_flags": [], "frame_skip": 4,
            "nes_core": {"sha256_16": "e09e8191b8d40490"}}
    _write_sidecar(tmp_path, dict(OVERNIGHT_SIDECAR, hw_provenance=prov))
    assert read_archive_counters(tmp_path).hw_provenance == prov
    # ...and its absence is None, not a KeyError.
    _write_sidecar(tmp_path, OVERNIGHT_SIDECAR)
    assert read_archive_counters(tmp_path).hw_provenance is None


@pytest.mark.parametrize("payload", [
    pytest.param(None, id="absent"),
    pytest.param("{not json", id="malformed"),
    pytest.param("[1, 2, 3]", id="not-an-object"),
    pytest.param('{"cells": 10}', id="missing-fields"),
    pytest.param('{"cells": "many", "records": 1, "new_cells": 1, '
                 '"improvements": 1, "frontier": 0, "best_score": 1}',
                 id="unparseable-field"),
])
def test_read_archive_counters_returns_none_and_never_raises(tmp_path, payload):
    """A free 130-byte diagnostic must not be able to take down a
    classification that never depended on it."""
    if payload is not None:
        (tmp_path / wt.ARCHIVE_STATS_FILENAME).write_text(payload)
    assert read_archive_counters(tmp_path) is None
    assert read_archive_counters(tmp_path / "nope") is None


def test_counters_are_reported_and_cannot_move_a_verdict():
    """The adapter's contract in one property: it adds evidence columns
    and changes nothing else. Asserted over the whole corpus, because
    'reporting only' is the kind of claim that decays quietly."""
    counters = ArchiveCounters(cells=560_410, records=45_640_527,
                               new_cells=560_410, improvements=827_786,
                               frontier=181_425, best_score=767.0)
    for label, _truth, expected, _desc, kw, arc in CORPUS:
        recs = make_records(**kw)
        bare = gated_wall_verdict(WallTelemetry(records=recs, archive=arc))
        with_counters = gated_wall_verdict(
            WallTelemetry(records=recs, archive=arc, counters=counters))
        assert with_counters.wall_class is bare.wall_class, label
        assert with_counters.descriptor == bare.descriptor, label
        assert with_counters.reasons == bare.reasons, label
        added = set(with_counters.evidence) - set(bare.evidence)
        assert added == {"archive_records", "archive_improvements",
                         "archive_frontier", "archive_best_score",
                         "archive_novelty_per_record",
                         "archive_explored_fraction"}, label


def test_explored_fraction_is_reported_from_the_archive_too():
    """Reported from both sources and gated on from neither — it is a
    saturating function of selections-per-cell (Pearson +0.9906 against
    the coupon-collector null) and nothing else."""
    assert wt.EXPLORED_FRACTION_IS_SEPARATING is False
    tel = WallTelemetry(
        records=make_records(n=30, cells_start=3354, cells_end=91_995),
        archive=archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                        explored=0.8335))
    assert gated_wall_verdict(tel).evidence["explored_fraction"] == \
        pytest.approx(0.8335)


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

    # And a monotone room counter still counts, so dropping `doors` did
    # not disarm the branch — it removed one non-monotone addend.
    moved_room = WallTelemetry(records=make_records(n=30, max_room=11,
                                                    max_room_end=12))
    assert gated_wall_verdict(moved_room).wall_class is WallClass.PROGRESSING


def test_doors_churn_no_longer_reads_as_topological_progress():
    """The D3 defect, reproduced from the real telemetry that exposed it.

    `runs/cv_chain_hw/lvl_03_overnight` is the frozen Castlevania hall:
    355 consecutive records without the map moving, zero solutions in
    ~6 h. Over its trailing 10 records `max_area`, `max_sect`, `max_room`
    and `max_gx` were all flat and `doors` moved 12,694 -> 12,880, and
    the module returned PROGRESSING — "the frontier is still moving" —
    for a search that had not moved anything.

    `doors` counts ARTICULATION POINTS in the discovered room graph. That
    is a connectivity property of a graph that keeps being rewritten, not
    a ratchet: a newly discovered edge can demote an articulation point,
    and re-exploration churns the count in both directions. It is not
    summable into a monotone delta.
    GATE_OPENER_CAMPAIGN_2026-08-11.md §12.
    """
    kw = dict(n=30, cells_start=548_583, cells_end=559_310,
              steps_start=44_777_250, steps_end=45_898_118,
              map_gx=767, doors=12_694, doors_end=12_880)
    v = gated_wall_verdict(WallTelemetry(records=make_records(**kw),
                                         label="lvl_03_overnight"))

    assert v.evidence["doors_delta"] == 186          # the churn is still real
    assert v.evidence["topo_delta"] == 0             # ...and no longer counted
    assert v.evidence["map_delta"] == 0
    assert v.wall_class is not WallClass.PROGRESSING
    assert v.wall_class is WallClass.INDETERMINATE

    # The repair is exactly the removal of one addend, and the fixture
    # must actually exercise it: the pre-repair sum was positive, which
    # is the whole reason the run read PROGRESSING.
    pre_repair_topo = v.evidence["topo_delta"] + v.evidence["doors_delta"]
    assert pre_repair_topo > 0
    # ...and the map had been frozen for the whole run while it said so.
    assert v.evidence["map_stall_windows"] >= 29


def test_doors_keeps_a_documented_monotone_successor():
    """Dropping a term without saying what would replace it invites the
    next person to re-add the same one."""
    assert wt.DOORS_IS_MONOTONE is False
    spec = wt.MISSING_TELEMETRY["doors_cumulative"]
    assert "articulation points EVER seen" in spec
    assert "SPEC ONLY" in spec


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


def test_a_saturated_boundary_abstains_and_only_describes_itself():
    """The row this module was built to certify, and no longer does.

    The Castlevania hall's own statistics: 92,785 cells over 1,089
    spatial buckets, concentration 85.2, boundary entropy 0.984, map
    frozen. Under the 2026-08-10 build this returned GATED with the
    remedy "switch to an orthogonal arm" — a several-hour commitment.
    It now abstains and reports what it saw.
    """
    tel = WallTelemetry(
        records=make_records(n=89, cells_start=3354, cells_end=91_995,
                             steps_end=10_643_480),
        archive=archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                        boundary_cells=13, entropy=0.9837),
    )
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == wt.UNRESOLVED_CONCENTRATED
    assert v.evidence["concentration"] >= wt.CONCENTRATION_DESCRIPTIVE_MIN
    assert v.evidence["topo_delta"] == 0 and v.evidence["map_delta"] == 0
    assert v.degraded is True
    assert "boundary_action_entropy" in v.missing
    # The label never travels without the receipt that struck the branch.
    assert any(wt.STRUCK_CLASSIFICATION_RECEIPT in r for r in v.reasons)
    # ...and the remedy is an instruction to gather evidence, not to act.
    assert v.remedy == wt.REMEDY[WallClass.INDETERMINATE]
    assert "collect the missing telemetry" in v.remedy


def test_a_low_concentration_archive_abstains_without_the_label():
    """The other arm of the removed fork, and it went too.

    `ge_chain/lvl_11_4-4`'s geometry — 5,885 cells over 750 buckets,
    concentration 7.85 — used to return COVERAGE_LIMITED, i.e. "give it
    more wall-clock". A single cross-sectional number cannot support
    that claim either: the same statistic read 15.66 on a solved archive
    and 98.30 on another one in the same chain. Same abstention, minus
    the descriptive label, because 7.85 is not concentrated.
    """
    tel = WallTelemetry(
        records=make_records(n=21, cells_start=5100, cells_end=5885,
                             steps_end=3_335_420),
        archive=archive(cells=5885, distinct_spatial=750, spatial_span=130),
    )
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == ""
    assert "gates nothing" in v.reasons[-1]


# --------------------------------------------------------------------------
# 2b. the C_local series path — the state the receipts ask the fleet for
#
# `c_local` does not exist in any banked run, so this whole world is
# reachable only by construction. It is also the field that makes
# INDETERMINATE's remedy actionable, which makes it the one that must be
# nailed down BEFORE it lands rather than after.
# --------------------------------------------------------------------------

def _series_tel(shape: str, *, peak: int, arc=None, n: int = 30,
                cells_end: int = 90_000, **kw) -> WallTelemetry:
    return WallTelemetry(
        records=make_records(n=n, cells_start=1000, cells_end=cells_end,
                             c_local_series=c_local_shapes(n, peak)[shape],
                             **kw),
        archive=arc)


def test_a_pinned_c_local_series_is_barren_not_a_plateau():
    """`BARREN <=> C_local STAGNANT` is the module's own adopted form.

    The mechanism this guards is `saturation()` reporting 1.0 for a
    series with zero peak yield: a curve that NEVER GREW scoring as
    maximally plateaued, which put a search that is not searching into
    the same class as one that has run out of ground.
    """
    v = gated_wall_verdict(_series_tel(
        "stagnant", peak=1089,
        arc=archive(cells=90_000, distinct_spatial=1089, spatial_span=95)))
    assert v.wall_class is WallClass.BARREN
    assert "STAGNANT, not plateaued" in v.reasons[-1]
    assert v.evidence["c_local_peak_yield"] == 0
    assert v.evidence["c_local_saturation"] is None
    # BARREN is a diagnosis of the SEARCH, so it carries no description
    # of the archive even though this one is concentrated.
    assert v.descriptor == ""

    # Same series, no archive: still BARREN.
    assert gated_wall_verdict(
        _series_tel("stagnant", peak=1089)).wall_class is WallClass.BARREN


def test_a_c_local_series_below_the_floor_is_key_blind_either_way():
    """The divergence the floor closes.

    Every Bubble Bobble profile is span-degenerate: one screen, 16
    distinct spatial buckets, progress byte constant. WITH an archive
    the module has always said KEY_BLIND. WITHOUT one — the moment
    `c_local` starts being emitted and the archive has not flushed — the
    identical run escaped that verdict, because the archive's
    `spatial_span` guard was the only place the check lived.
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


def test_a_climbing_c_local_series_is_the_one_affirmative_reading_left():
    """COVERAGE_LIMITED survives, and only from a SERIES.

    A map footprint measurably still expanding has not run out of
    reachable ground — that is a direction, and a direction is the one
    thing a cross-sectional number cannot supply. The hall's own cell
    curve has the highest tail exponent in the corpus (0.899) while its
    map is frozen, which is precisely why the point statistic went and
    the series statistic stayed.
    """
    hall_arc = archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                       boundary_cells=13, entropy=0.9837)
    # The same archive without a series only gets to describe itself.
    baseline = gated_wall_verdict(
        WallTelemetry(records=make_records(n=30, cells_start=3354,
                                           cells_end=91_995),
                      archive=hall_arc))
    assert baseline.wall_class is WallClass.INDETERMINATE
    assert baseline.descriptor == wt.UNRESOLVED_CONCENTRATED

    v = gated_wall_verdict(_series_tel("climbing", peak=5000, arc=hall_arc))
    assert v.wall_class is WallClass.COVERAGE_LIMITED
    assert "still expanding" in v.reasons[-1]
    # An affirmative reading of its own, so it does not also carry the
    # description of a stalled archive.
    assert v.descriptor == ""


def test_a_c_local_plateau_certifies_nothing_however_it_is_corroborated():
    """The promotion direction, removed rather than re-thresholded.

    Under the 2026-08-10 build a plateau plus a high concentration was
    the strongest verdict the module could reach: two statistics
    agreeing. They agree about the same confound. `concentration` is
    archive SIZE wearing a hat and the plateau is measured on the very
    curve size accumulates along, so corroboration between them is not
    independent evidence — and the corroborating statistic has since
    been shown not to separate at all.

    All three corroboration states now land in the same abstention. Only
    the DESCRIPTION differs, and it differs on the archive, which is what
    a description is allowed to do.
    """
    corroborating = archive(cells=92_785, distinct_spatial=1089,
                            spatial_span=95, boundary_cells=13, entropy=0.98)
    v = gated_wall_verdict(_series_tel("plateaued", peak=1089,
                                       arc=corroborating))
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == wt.UNRESOLVED_CONCENTRATED
    assert v.degraded is True and "boundary_action_entropy" in v.missing
    assert "not by itself evidence of a wall" in v.reasons[-2]

    # ge_chain/lvl_11_4-4's geometry: 7.85, once read as a contradiction.
    contradicting = archive(cells=5885, distinct_spatial=750, spatial_span=130)
    v = gated_wall_verdict(_series_tel("plateaued", peak=750, cells_end=5885,
                                       arc=contradicting))
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == ""

    v = gated_wall_verdict(_series_tel("plateaued", peak=1089))
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == ""


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

    # And an archive settles `frontier_bucket_cells`, which is derived
    # from `boundary_cells`: it is on the shopping list only for a call
    # that could not have had it.
    assert "frontier_bucket_cells" in no_series.missing
    with_archive = gated_wall_verdict(WallTelemetry(
        records=make_records(n=89, cells_start=3354, cells_end=91_995,
                             steps_end=10_643_480),
        archive=archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
                        boundary_cells=13)))
    assert "frontier_bucket_cells" not in with_archive.missing
    assert with_archive.evidence["boundary_cells"] == 13

    # Present but too short to yield a series still counts as missing.
    short = make_records(n=30, steps_end=10_000_000)
    patched = short[:-1] + (
        ProgressRecord(**{**short[-1].__dict__, "c_local": 900}),)
    v = gated_wall_verdict(WallTelemetry(records=patched))
    assert v.evidence["c_local"] == 900
    assert v.evidence["c_local_peak_yield"] is None
    assert "c_local" in v.missing


def test_the_descriptor_is_a_function_of_the_archive_alone():
    """PARITY, as a property over the whole corpus.

    The label that replaced GATED must not be reachable by the solver
    starting to emit a field. Under the old build this same sweep
    guarded a VERDICT from being invented by `c_local`; the verdict is
    gone, so it now guards the label, which is the only thing left that
    a reader could mistake for one.
    """
    offenders = []
    for label, _truth, _expected, _desc, kw, arc in CORPUS:
        peak = arc.distinct_spatial if arc is not None else 1000
        shapes = dict(c_local_shapes(kw["n"], peak), none=None)
        for a in (arc, None):
            expected = (wt.UNRESOLVED_CONCENTRATED
                        if a is not None
                        and a.concentration >= wt.CONCENTRATION_DESCRIPTIVE_MIN
                        else "")
            for name, series in shapes.items():
                tel = WallTelemetry(
                    records=make_records(**kw, c_local_series=series),
                    archive=a, label=label)
                v = gated_wall_verdict(tel)
                # A verdict that diagnoses the search (BARREN/KEY_BLIND)
                # or observes movement describes nothing; everything else
                # describes exactly what its archive shows.
                if v.wall_class is not WallClass.INDETERMINATE:
                    if v.descriptor != "":
                        offenders.append(f"{label} [{name}] {v.wall_class.value}"
                                         f" carried {v.descriptor!r}")
                elif v.descriptor != expected:
                    offenders.append(f"{label} [c_local={name}, "
                                     f"archive={a is not None}] "
                                     f"{v.descriptor!r} != {expected!r}")
    assert offenders == [], f"the descriptor moved with the series: {offenders}"


def test_every_verdict_carries_a_remedy_and_serializes():
    tel = WallTelemetry(records=make_records(n=30), label="unit")
    v = gated_wall_verdict(tel)
    assert v.remedy == wt.REMEDY[v.wall_class]
    assert v.calibration == wt.CALIBRATION_TAG
    blob = json.loads(json.dumps(v.as_dict()))   # must not raise
    assert blob["descriptor"] == v.descriptor
    assert blob["wall_class"] == v.wall_class.value


# --------------------------------------------------------------------------
# 3. frozen corpus
# --------------------------------------------------------------------------

#: Statistics measured on the banked runs during the 2026-08-10 offline
#: calibration, replayed as synthetic telemetry. Columns:
#:   label, ground truth, expected verdict, expected descriptor,
#:   telemetry kwargs, archive
#:
#: WHAT THIS FIXTURE IS FOR, AFTER THE STRIKE. It used to lock a set of
#: separating bands: move a constant out of its measured band and a row
#: flips. Two of those bands are gone — `concentration` does not separate
#: this corpus or any other — so what the rows lock now is narrower and
#: more honest:
#:
#:   * the SUBTRACTIVE classes still fire where they always did
#:     (KEY_BLIND on every span-degenerate Bubble Bobble profile, BARREN
#:     on the frozen ones, INSUFFICIENT on the starved show segment);
#:   * the two directly-observed classes still fire (RESOLVED where a
#:     solution was banked, PROGRESSING where a monotone counter moved);
#:   * and everything the module can no longer tell apart lands in ONE
#:     class, which is how you can see at a glance how much was struck.
#:
#: The five hall reads and the resolved coverage walls now share the
#: INDETERMINATE row. That is not a loss of resolution the fixture is
#: papering over — it IS the finding.
CORPUS = [
    ("cv_chain_hw2/lvl_03_trace", "unresolved hall",
     WallClass.INDETERMINATE, wt.UNRESOLVED_CONCENTRATED,
     dict(n=89, cells_start=3354, cells_end=91_995, steps_end=10_643_480,
          map_gx=767),
     archive(cells=92_785, distinct_spatial=1089, spatial_span=95,
             boundary_cells=13, entropy=0.9837)),
    ("cv_chain_hw/lvl_03_trace", "unresolved hall, second lineage",
     WallClass.INDETERMINATE, wt.UNRESOLVED_CONCENTRATED,
     dict(n=14, cells_start=17_000, cells_end=27_619, steps_end=2_404_020,
          map_gx=767),
     archive(cells=28_929, distinct_spatial=932, spatial_span=94,
             boundary_cells=5, entropy=0.9697)),
    # The doors fixture. Progress-only ON PURPOSE: D3 ruled this run's
    # archive INCOMPATIBLE as evidence (1-flag lineage against the 4-flag
    # arms, backfilled provenance, a disjoint tb/kk key subspace, and a
    # "3.8x largest archive" that was key-inflation — 560,410 collapsing
    # to 88,212). The row is here for the progress series, which is where
    # the taxonomy defect lived, not to contribute archive statistics.
    ("cv_chain_hw/lvl_03_overnight (D3: INCOMPATIBLE lineage)",
     "unresolved hall; read PROGRESSING off doors churn",
     WallClass.INDETERMINATE, "",
     dict(n=30, cells_start=548_583, cells_end=559_310,
          steps_start=44_777_250, steps_end=45_898_118, map_gx=767,
          doors=12_694, doors_end=12_880),
     None),
    ("cv_hall_ortho_a @24 records (pre-flush snapshot)", "unresolved hall",
     WallClass.INDETERMINATE, "",
     dict(n=24, cells_start=93_720, cells_end=107_890, steps_end=2_440_150,
          map_gx=767),
     None),
    ("bubble_bobble/r68_retry_ortho", "orthogonal/key",
     WallClass.KEY_BLIND, "",
     dict(n=30, cells_start=96, cells_end=96, steps_end=3_675_730, frozen=29),
     archive(cells=96, distinct_spatial=16, spatial_span=1,
             boundary_cells=96, entropy=0.7775)),
    ("bubble_bobble/r69_retry_ortho", "orthogonal/key",
     WallClass.KEY_BLIND, "",
     dict(n=30, cells_start=40, cells_end=48, steps_end=3_153_184, frozen=19),
     archive(cells=48, distinct_spatial=8, spatial_span=1,
             boundary_cells=48, entropy=0.78)),
    ("bubble_bobble/r68_retry_xsig", "resolved", WallClass.RESOLVED, "",
     dict(n=30, cells_start=224, cells_end=605, steps_end=3_570_080,
          solutions=6, frozen=4),
     archive(cells=605, distinct_spatial=16, spatial_span=1,
             boundary_cells=605, entropy=0.8253)),
    ("bubble_bobble/r99_retry2", "orthogonal/key", WallClass.KEY_BLIND, "",
     dict(n=30, cells_start=300, cells_end=691, steps_end=3_544_480, frozen=4),
     archive(cells=691, distinct_spatial=16, spatial_span=1,
             boundary_cells=691, entropy=0.8583)),
    ("bubble_bobble/r99_1_boss_retry", "unknown/mechanic",
     WallClass.KEY_BLIND, "",
     dict(n=30, cells_start=2931, cells_end=9475, steps_end=3_249_620),
     archive(cells=9475, distinct_spatial=32, spatial_span=1,
             boundary_cells=9475, entropy=0.8991)),
    ("bubble_bobble/chain_day2h_item/lvl_00_99-1", "unknown/mechanic",
     WallClass.KEY_BLIND, "",
     dict(n=45, cells_start=1462, cells_end=2989, steps_end=5_188_710),
     archive(cells=2989, distinct_spatial=16, spatial_span=1,
             boundary_cells=2989, entropy=0.8761)),
    ("live_show/smb_4_4_micro/lvl_4-4 seg1", "coverage (resolved)",
     WallClass.INDETERMINATE, "",
     dict(n=44, cells_start=50_366, cells_end=1_020_500, steps_end=6_800_000,
          map_gx=2059),
     None),
    ("live_show/smb_4_4_micro/lvl_4-4 seg4", "coverage (resolved)",
     WallClass.PROGRESSING, "",
     dict(n=50, cells_start=51_853, cells_end=1_164_599, steps_end=7_286_076,
          map_gx=2055, map_gx_end=2575),
     None),
    ("live_show/smb_4_4_micro/lvl_8-4", "coverage (resolved)",
     WallClass.PROGRESSING, "",
     dict(n=56, cells_start=45_035, cells_end=1_190_873, steps_end=7_203_648,
          map_gx=3844, map_gx_end=4290),
     None),
    ("ge_chain/lvl_11_4-4", "coverage (resolved)",
     WallClass.INDETERMINATE, "",
     dict(n=21, cells_start=5100, cells_end=5885, steps_end=3_335_420,
          map_gx=2068),
     archive(cells=5885, distinct_spatial=750, spatial_span=130,
             boundary_cells=8, entropy=0.9728)),
]


@pytest.mark.parametrize("label,truth,expected,descriptor,kw,arc", CORPUS,
                         ids=[c[0] for c in CORPUS])
def test_corpus_verdicts(label, truth, expected, descriptor, kw, arc):
    tel = WallTelemetry(records=make_records(**kw), archive=arc, label=label)
    v = gated_wall_verdict(tel)
    assert v.wall_class is expected, (
        f"{label} (ground truth {truth}) -> {v.wall_class.value}; "
        f"reasons={v.reasons}")
    assert v.descriptor == descriptor, (
        f"{label} -> descriptor {v.descriptor!r}, expected {descriptor!r}")


def test_the_solved_archives_that_broke_the_band_are_not_condemned():
    """The three counter-examples that ended the classifier, scored.

    All three were SOLVED. Under the shipped 25.0 threshold the first two
    read GATED — "switch to an orthogonal arm" — and the third read
    COVERAGE_LIMITED from one chain over, which is how a 6.3x intra-chain
    spread hides inside a 1.51x "separating band". None of them may be
    condemned now, and the mechanism by which none of them is is that
    concentration reaches no branch at all.
    """
    solved = [
        # ge_chain_w8, one chain, one profile, one --gx-bucket 16 grid.
        ("lvl_00_8-1", 218_218, 2220, 98.30),
        ("lvl_02_8-3", 19_958, 970, 20.58),
        ("lvl_01_8-2", 19_330, 1234, 15.66),
    ]
    for name, cells, ds, conc in solved:
        arc = archive(cells=cells, distinct_spatial=ds, spatial_span=120)
        assert arc.concentration == pytest.approx(conc, abs=0.6), name
        tel = WallTelemetry(
            records=make_records(n=30, cells_start=cells // 2, cells_end=cells,
                                 steps_end=8_000_000, map_gx=2000),
            archive=arc, label=name)
        v = gated_wall_verdict(tel)
        assert v.wall_class in LICENSED_CLASSES
        assert v.wall_class is WallClass.INDETERMINATE, name
        assert v.remedy == wt.REMEDY[WallClass.INDETERMINATE]
    # The spread that did the damage, asserted so the number is on record.
    assert 98.30 / 15.66 > 6.0


def test_the_live_ortho_arm_gets_a_description_and_stays_excluded():
    """`runs/cv_hall_ortho_a` flushed an archive; its statistics are the
    most extreme in the corpus and they still classify nothing.

    Measured 2026-08-10 11:26 from the real 2,436,606,838-byte
    `archive.pkl`: 114,699 cells over 1,095 distinct spatial buckets,
    span 95 -> concentration 104.75, 4.2x the descriptive floor. It is
    deliberately NOT in CORPUS, and the exclusion is the point: it is a
    THIRD read of the same Castlevania hall, so it adds no independent
    evidence, and it was the pending-validation arm itself.

    It also settles what the arm was launched to settle, in the only
    direction the evidence allows: the arm ran, and no statistic computed
    on any of its five hall reads separates them from an archive that
    solved.
    """
    measured = archive(cells=114_699, distinct_spatial=1095, spatial_span=95,
                       boundary_cells=16, entropy=0.983)
    assert measured.concentration == pytest.approx(104.748, abs=0.01)
    assert measured.concentration > 4 * wt.CONCENTRATION_DESCRIPTIVE_MIN
    tel = WallTelemetry(
        records=make_records(n=56, cells_start=93_720, cells_end=119_535,
                             steps_end=5_621_170, map_gx=767),
        archive=measured, label="cv_hall_ortho_a @56 records")
    v = gated_wall_verdict(tel)
    assert v.wall_class is WallClass.INDETERMINATE
    assert v.descriptor == wt.UNRESOLVED_CONCENTRATED

    corpus_labels = {c[0] for c in CORPUS}
    assert not any(lab.startswith("cv_hall_ortho_a @56") for lab in corpus_labels)
    assert not any(d[0].endswith("cv_hall_ortho_a") for d in LIVE), (
        "the live, still-mutating ortho directory must not be a regression "
        "fixture: its statistics move between reads")


def test_surviving_constants_sit_inside_their_measured_separating_bands():
    """Each band is (nearest counter-example, nearest positive]. Moving a
    constant outside its band silently reclassifies a banked run, so the
    bands are asserted here rather than living only in the receipt.

    The two bands that used to head this list are gone. `concentration`
    has no band — it was measured not to have one — and the flag that
    could have promoted the C_local series into the same branch went with
    the branch. What remains is the set §14 of the second receipt
    enumerates as untouched, and all 13 resolved archives clear every one
    of them.
    """
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
    # No band survives for the struck statistic, under any name.
    assert not hasattr(wt, "CONCENTRATION_GATED_MIN")
    assert not hasattr(wt, "C_LOCAL_SERIES_MAY_CERTIFY_GATED")


# --------------------------------------------------------------------------
# 4. the strike, and its mutation guards
# --------------------------------------------------------------------------

def test_re_adding_a_gated_class_fails_here():
    """THE MUTATION GUARD for the removal.

    Re-adding `GATED` — as an enum member, as a remedy, or as a verdict
    string — fails this test. It is spelled out at three levels because
    the shipped branch was one `if` and one enum line, and either alone
    would have been enough to put "switch to an orthogonal arm" back in
    front of an operator on evidence that does not support it.
    """
    names = {c.name for c in WallClass}
    values = {c.value for c in WallClass}
    assert "GATED" not in names, (
        "WallClass.GATED was removed on 2026-08-11 after 22 candidate "
        "statistics over 103 archives failed to separate a wall from a "
        f"search that solved. See {wt.STRUCK_CLASSIFICATION_RECEIPT} §12.1.")
    assert "gated" not in values
    assert set(WallClass) == LICENSED_CLASSES
    assert set(wt.REMEDY) == LICENSED_CLASSES
    assert not any("gated" in r.lower() for r in wt.REMEDY.values())

    # ...and no CODE path can name one. Checked on the parse tree rather
    # than on the text so that the module is free to keep explaining, at
    # whatever length, why the class is not there.
    import ast

    tree = ast.parse((REPO / "src/training/wall_taxonomy.py").read_text())
    named = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "GATED"
             and isinstance(n.value, ast.Name) and n.value.id == "WallClass"]
    assert named == [], f"WallClass.GATED referenced on {len(named)} line(s)"


def test_no_telemetry_anywhere_in_the_corpus_certifies_a_wall():
    """The headline safety property, now unconditional.

    It used to be conditional — progress-only telemetry may not certify
    GATED — and that condition was the concession that made the shipped
    branch look safe: with an archive it certified freely, and four of
    the thirteen archives it would have certified had already solved.

    Swept over every corpus run, every C_local shape, and both archive
    states. Nothing may leave the licensed set, and nothing may return a
    class this module no longer has.
    """
    seen = set()
    for label, _truth, _expected, _desc, kw, arc in CORPUS:
        peak = arc.distinct_spatial if arc is not None else 1000
        shapes = dict(c_local_shapes(kw["n"], peak), none=None)
        for name, series in shapes.items():
            for a in (arc, None):
                tel = WallTelemetry(
                    records=make_records(**kw, c_local_series=series),
                    archive=a, label=label)
                v = gated_wall_verdict(tel)
                assert v.wall_class in LICENSED_CLASSES, (
                    f"{label} [{name}] -> {v.wall_class}")
                assert v.descriptor in ("", wt.UNRESOLVED_CONCENTRATED)
                seen.add(v.wall_class)
    # The sweep must actually exercise the interesting classes, or the
    # assertion above is vacuous.
    assert {WallClass.INDETERMINATE, WallClass.KEY_BLIND,
            WallClass.BARREN, WallClass.RESOLVED} <= seen


def test_the_descriptive_threshold_moves_no_verdict(monkeypatch):
    """The other mutation guard, and the sharper of the two.

    `CONCENTRATION_DESCRIPTIVE_MIN` is the last place the struck
    statistic appears. If anyone re-wires it into a branch — under any
    name, in either direction, whether it returns GATED or something
    invented to avoid the word — driving it to both extremes will move a
    `wall_class`, and this fails. Only the DESCRIPTOR is allowed to
    respond to it, which is the operational difference between a
    description and a gate.
    """
    def sweep():
        return [(gated_wall_verdict(WallTelemetry(records=make_records(**kw),
                                                  archive=arc, label=label))
                 ) for label, _t, _e, _d, kw, arc in CORPUS]

    baseline = [(v.wall_class, v.descriptor) for v in sweep()]

    monkeypatch.setattr(wt, "CONCENTRATION_DESCRIPTIVE_MIN", 0.0)
    everything_concentrated = sweep()
    monkeypatch.setattr(wt, "CONCENTRATION_DESCRIPTIVE_MIN", 1e9)
    nothing_concentrated = sweep()

    for (cls, _desc), lo, hi in zip(baseline, everything_concentrated,
                                    nothing_concentrated):
        assert lo.wall_class is cls
        assert hi.wall_class is cls
    # And the descriptor DOES respond, or the sweep proved nothing.
    assert any(v.descriptor for v in everything_concentrated)
    assert not any(v.descriptor for v in nothing_concentrated)


def test_refuted_statistics_are_reported_but_never_gated_on():
    """Raw-cell saturation was the obvious candidate for 'C_local
    plateau' and it is wrong: on the corpus the hall straddles two
    resolved coverage walls in BOTH directions."""
    measured = {                     # raw-cell saturation, trailing 10 records
        "cv_hall_hw (unresolved)": 0.179,
        "smb_8-4 (resolved)": 0.189,
        "cv_hall_hw2 (unresolved)": 0.343,
        "smb_4-4 seg4 (resolved)": 0.352,
    }
    hall = [v for k, v in measured.items() if "unresolved" in k]
    resolved = [v for k, v in measured.items() if "(resolved)" in k]
    assert min(hall) < min(resolved) < max(hall) < max(resolved), (
        "the corpus interleaves; no raw-saturation threshold separates it")

    # And the statistics are still reported, so a human can see them.
    tel = WallTelemetry(records=make_records(n=30))
    ev = gated_wall_verdict(tel).evidence
    for key in ("raw_coverage_saturation", "churn_per_window",
                "map_stall_windows", "doors_delta"):
        assert key in ev


@pytest.mark.parametrize("name,when", sorted(REFUTED_CONSTANTS.items()))
def test_every_refuted_constant_is_false_and_cites_its_kill(name, when):
    """A refutation nobody can read is a refutation somebody re-derives.

    Each constant must be `False`, must carry a `REFUTED-OFFLINE-<date>`
    provenance tag on its own line, and must be documented in a comment
    block that names the OFFENDER that killed it — a run and the value it
    scored — in the idiom §3 of the calibration set. The 2026-08-11
    cohort must additionally point at the receipt section it came from,
    because that receipt scored 22 candidates and a bare assertion is not
    findable inside it.
    """
    assert getattr(wt, name) is False

    src = (REPO / "src/training/wall_taxonomy.py").read_text().splitlines()
    idx = next(i for i, line in enumerate(src) if line.startswith(f"{name} ="))
    assert f"REFUTED-OFFLINE-{when}" in src[idx], src[idx]

    block = []
    for line in reversed(src[:idx]):
        if not line.startswith("#:"):
            break
        block.append(line)
    doc = " ".join(reversed(block))
    assert len(doc) > 120, f"{name} has no documented kill"
    assert any(ch.isdigit() for ch in doc), (
        f"{name} names no measured offender")
    if when == "2026-08-11":
        assert "§" in doc or ".md" in doc, f"{name} cites no receipt section"


def test_the_module_names_the_receipts_that_struck_it():
    src = (REPO / "src/training/wall_taxonomy.py").read_text()
    for receipt in ("k_falsifier_2026-08-10.md",
                    "size_decoupled_statistic_2026-08-11.md",
                    "GATE_OPENER_CAMPAIGN_2026-08-11.md"):
        assert receipt in src
    assert wt.STRUCK_CLASSIFICATION_RECEIPT in src
    assert (REPO / "docs/receipts/dispatch"
            / "size_decoupled_statistic_2026-08-11.md").exists()
    assert wt.CALIBRATION_TAG == "CLASSIFICATION-STRUCK-2026-08-11", (
        "the tag stamped into every verdict must change when the verdicts "
        "do, or a pre-strike receipt reads as a post-strike one")


def test_missing_telemetry_names_the_runtime_gaps():
    for key in ("c_local", "boundary_action_entropy", "frontier_bucket_cells",
                "doors_cumulative"):
        assert key in wt.MISSING_TELEMETRY
        assert len(wt.MISSING_TELEMETRY[key]) > 40


#: The ONE runtime reference this module tolerates: the gate-opener arm
#: reads the PURE `boundary_axis_profile` off a flush snapshot for its
#: `boundary_state_axes` / `alias_ratio` telemetry. Everything that
#: CLASSIFIES — `gated_wall_verdict`, `WallClass`, the thresholds — stays
#: operator-read between sessions.
_TAXONOMY_PURE_READER = "scripts/go_explore_solve.py"


def test_module_is_not_wired_into_any_runtime_dispatch():
    """Self-arming is a later decision (D2 verdict), and after the strike
    there is nothing to arm. Nothing outside tests and docs may import
    this module, with exactly one exception: the gate-opener arm's pure
    `boundary_axis_profile` read (taxonomy KEYED, never taxonomy-WIRED —
    the companion test below pins how narrow that exception is).
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
# 5. live regression against the banked runs (skipped without them)
#
# Only FINISHED run directories belong here. `runs/cv_hall_ortho_a` is
# excluded on purpose: it is still being written, so its statistics move
# between reads, and it was the arm the old calibration was conditional
# on.
# --------------------------------------------------------------------------

LIVE = [
    ("runs/cv_chain_hw2/lvl_03_trace", True, -1, WallClass.INDETERMINATE,
     wt.UNRESOLVED_CONCENTRATED),
    ("runs/cv_chain_hw/lvl_03_trace", True, -1, WallClass.INDETERMINATE,
     wt.UNRESOLVED_CONCENTRATED),
    # Progress-only: the archive is 11.9 GB and D3 ruled it inadmissible
    # as evidence anyway. This row exists to hold the doors repair against
    # the real series that exposed it.
    ("runs/cv_chain_hw/lvl_03_overnight", False, -1, WallClass.INDETERMINATE,
     ""),
    ("runs/bubble_bobble/r68_retry_ortho", True, -1, WallClass.KEY_BLIND, ""),
    ("runs/bubble_bobble/r68_retry_xsig", True, -1, WallClass.RESOLVED, ""),
    ("runs/bubble_bobble/r99_1_boss_retry", True, -1, WallClass.KEY_BLIND, ""),
    ("runs/live_show/smb_4_4_micro/lvl_8-4", False, -1,
     WallClass.PROGRESSING, ""),
    ("runs/live_show/smb_4_4_micro/lvl_4-4", False, 1,
     WallClass.INDETERMINATE, ""),
    ("runs/ge_chain/lvl_11_4-4", True, -1, WallClass.INDETERMINATE, ""),
]


@pytest.mark.parametrize("rundir,with_archive,segment,expected,descriptor",
                         LIVE, ids=[c[0] for c in LIVE])
def test_live_corpus_reproduces_the_calibrated_verdicts(
        rundir, with_archive, segment, expected, descriptor):
    base = REPO / rundir
    progress = base / "progress.jsonl"
    arch = base / "archive.pkl"
    if not progress.exists() or (with_archive and not arch.exists()):
        pytest.skip(f"{rundir} not present (runs/ is gitignored)")
    tel = wt.telemetry_from_paths(
        progress, archive_path=(arch if with_archive else None),
        segment=segment, label=rundir)
    v = gated_wall_verdict(tel)
    assert v.wall_class is expected
    assert v.descriptor == descriptor


def test_the_live_doors_defect_is_fixed_on_the_real_file():
    """The exact read that produced the misclassification, off disk.

    Synthetic fixtures reproduce the window deltas; this asserts the
    verdict on the 359-record file itself, so the repair cannot be true
    only of the reconstruction.
    """
    progress = REPO / "runs/cv_chain_hw/lvl_03_overnight/progress.jsonl"
    if not progress.exists():
        pytest.skip("runs/cv_chain_hw/lvl_03_overnight not present")
    tel = wt.telemetry_from_paths(progress, label="lvl_03_overnight")
    v = gated_wall_verdict(tel)
    assert v.evidence["doors_delta"] == 186
    assert v.evidence["topo_delta"] == 0
    assert v.evidence["map_stall_windows"] == 355
    assert v.wall_class is WallClass.INDETERMINATE

    # The sidecar came along for free and reports the effort the verdict
    # was taken over, on the archive's own clock.
    assert tel.counters is not None
    assert v.evidence["archive_records"] == 45_640_527
    assert v.evidence["archive_explored_fraction"] == pytest.approx(0.6763,
                                                                    abs=5e-5)
