"""Tests for scripts/death_cluster_audit.py.

Pure accounting over eval records — no emulator, no rollouts. Exists because
hand-tallied death-cluster/clear claims got two serious defects into a receipt:
(1) "zero deaths at exactly X" contradicted by the records, and (2) episodes
reaching flag-x conflated with registered clears (clear_rate). Every number
this tool emits is computed, named by predicate, and cross-checked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.death_cluster_audit import (  # noqa: E402
    aggregate_audit,
    audit_record,
    main,
)


def _rec(gx, clear_rate=None, seq_clear_rate=None, **extra):
    r = {"max_gx_per_episode": list(gx), "n_episodes": len(gx)}
    if clear_rate is not None:
        r["clear_rate"] = clear_rate
    if seq_clear_rate is not None:
        r["seq_clear_rate"] = seq_clear_rate
    r.update(extra)
    return r


# ---------------------------------------------------------------- per-record

def test_deaths_exclude_flag_reachers():
    rec = _rec([100, 2675, 3266, 3267], clear_rate=0.5)
    a = audit_record(rec, flag_x=3266)
    assert a["n"] == 4
    assert sorted(a["deaths"]) == [100, 2675]
    assert a["flag_reached"] == 2


def test_clear_counts_by_predicate_and_discrepancy():
    # 30 eps, clear_rate 19/30 but 21 episodes at/past flag-x:
    # the two predicates must be reported separately, discrepancy = 2.
    gx = [3266] * 21 + [2675] * 9
    rec = _rec(gx, clear_rate=19 / 30, seq_clear_rate=21 / 30)
    a = audit_record(rec, flag_x=3266)
    assert a["clears_episode_success"] == 19
    assert a["clears_seq"] == 21
    assert a["flag_reached"] == 21
    assert a["flag_without_episode_success"] == 2


def test_clear_count_rounding_is_exact_for_thirds():
    # 0.06666...*30 must count 2, not 1 (float-floor hazard).
    rec = _rec([3266] * 2 + [2675] * 28, clear_rate=2 / 30)
    a = audit_record(rec, flag_x=3266)
    assert a["clears_episode_success"] == 2


def test_missing_predicates_reported_as_none():
    a = audit_record(_rec([100, 200]), flag_x=3266)
    assert a["clears_episode_success"] is None
    assert a["clears_seq"] is None
    assert a["flag_without_episode_success"] is None


# ---------------------------------------------------------------- aggregate

def test_aggregate_exact_histogram_and_bands():
    recs = [
        _rec([2675, 2675, 2979, 3266], clear_rate=0.25),
        _rec([2979, 2980, 2995, 3267], clear_rate=0.25),
    ]
    agg = aggregate_audit(recs, flag_x=3266, bands=[(2950, 3010)])
    assert agg["n_episodes"] == 8
    assert agg["death_histogram"]["2675"] == 2
    assert agg["death_histogram"]["2979"] == 2
    assert agg["bands"]["2950-3010"]["count"] == 4
    assert agg["bands"]["2950-3010"]["histogram"] == {
        "2979": 2, "2980": 1, "2995": 1}
    assert agg["flag_reached"] == 2
    assert agg["clears_episode_success"] == 2
    assert agg["deaths_total"] == 6


def test_aggregate_band_bounds_inclusive():
    recs = [_rec([2950, 3010, 2949, 3011])]
    agg = aggregate_audit(recs, flag_x=3266, bands=[(2950, 3010)])
    assert agg["bands"]["2950-3010"]["count"] == 2


def test_aggregate_flag_vs_success_discrepancy_rolls_up():
    recs = [
        _rec([3266] * 21 + [2675] * 9, clear_rate=19 / 30),
        _rec([3267] * 3 + [2675] * 27, clear_rate=0.0),
    ]
    agg = aggregate_audit(recs, flag_x=3266, bands=[])
    assert agg["flag_reached"] == 24
    assert agg["clears_episode_success"] == 19
    assert agg["flag_without_episode_success"] == 5


# ---------------------------------------------------------------- CLI

def test_cli_end_to_end(tmp_path):
    jl = tmp_path / "eval.jsonl"
    rows = [
        _rec([100, 3266], clear_rate=0.5),        # row 0 (excluded)
        _rec([2675, 2979, 3266], clear_rate=1 / 3),  # row 1
        _rec([2979, 3267], clear_rate=0.0),          # row 2
    ]
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    extra = tmp_path / "final.json"
    extra.write_text(json.dumps(_rec([2980, 3266], clear_rate=0.5)))
    out = tmp_path / "audit.json"
    rc = main(["--eval-jsonl", str(jl), "--rows", "1,2",
               "--extra-json", str(extra), "--flag-x", "3266",
               "--band", "2950:3010", "--exact", "2979,2980",
               "--out", str(out)])
    assert rc == 0
    agg = json.loads(out.read_text())
    assert agg["n_episodes"] == 7   # rows 1,2 (5) + extra (2); row 0 excluded
    assert agg["death_histogram"]["2979"] == 2
    assert agg["exact"]["2979"] == 2
    assert agg["exact"]["2980"] == 1
    assert agg["bands"]["2950-3010"]["count"] == 3
    assert agg["flag_reached"] == 3
    assert agg["clears_episode_success"] == 2
    assert agg["flag_without_episode_success"] == 1


def test_cli_requires_rows_or_all(tmp_path):
    jl = tmp_path / "eval.jsonl"
    jl.write_text(json.dumps(_rec([100])) + "\n")
    out = tmp_path / "a.json"
    rc = main(["--eval-jsonl", str(jl), "--flag-x", "3266",
               "--out", str(out)])
    assert rc == 0  # default = all rows
    assert json.loads(out.read_text())["n_episodes"] == 1


def test_cli_rejects_row_out_of_range(tmp_path):
    jl = tmp_path / "eval.jsonl"
    jl.write_text(json.dumps(_rec([100])) + "\n")
    with pytest.raises(SystemExit):
        main(["--eval-jsonl", str(jl), "--rows", "5", "--flag-x", "3266"])
