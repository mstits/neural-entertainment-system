"""Tests for analyze.py's cohort split and divergence-ranking logic."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import (  # noqa: E402
    diverge,
    numeric_values,
    parse_split,
    rank,
    read_jsonl,
    split_rows,
)


# ---------------------------------------------------------------------------
# parse_split
# ---------------------------------------------------------------------------


def test_parse_split_gt_numeric():
    field, op, value = parse_split("generation>100")
    assert field == "generation"
    assert op == ">"
    assert value == 100.0


def test_parse_split_lt_float():
    field, op, value = parse_split("clear_rate<0.1")
    assert field == "clear_rate"
    assert op == "<"
    assert value == 0.1


def test_parse_split_tolerates_spaces():
    field, op, value = parse_split("generation >= 50")
    assert (field, op, value) == ("generation", ">=", 50.0)


def test_parse_split_ge_le_eq_ne():
    assert parse_split("x>=1")[1] == ">="
    assert parse_split("x<=1")[1] == "<="
    assert parse_split("x==1")[1] == "=="
    assert parse_split("x!=1")[1] == "!="


def test_parse_split_non_numeric_value_kept_as_string():
    field, op, value = parse_split("stage==default")
    assert field == "stage"
    assert op == "=="
    assert value == "default"


def test_parse_split_rejects_garbage():
    with pytest.raises(ValueError, match="can't parse"):
        parse_split("not a predicate")


def test_parse_split_rejects_bad_field_name():
    with pytest.raises(ValueError):
        parse_split("1abc>5")


# ---------------------------------------------------------------------------
# split_rows
# ---------------------------------------------------------------------------


def test_split_rows_partitions_by_predicate():
    rows = [{"generation": g} for g in range(5)]
    a, b, excluded = split_rows(rows, "generation", ">", 2.0)
    assert [r["generation"] for r in a] == [3, 4]
    assert [r["generation"] for r in b] == [0, 1, 2]
    assert excluded == 0


def test_split_rows_excludes_rows_missing_field():
    rows = [{"generation": 1}, {"other": 2}, {"generation": 3}]
    a, b, excluded = split_rows(rows, "generation", ">", 2.0)
    assert len(a) == 1
    assert len(b) == 1
    assert excluded == 1


def test_split_rows_excludes_rows_with_null_field():
    rows = [{"generation": None}, {"generation": 5}]
    a, b, excluded = split_rows(rows, "generation", ">", 2.0)
    assert len(a) == 1
    assert excluded == 1


def test_split_rows_string_equality():
    rows = [{"stage": "default"}, {"stage": "hard"}, {"stage": "default"}]
    a, b, excluded = split_rows(rows, "stage", "==", "default")
    assert len(a) == 2
    assert len(b) == 1
    assert excluded == 0


def test_split_rows_type_mismatch_excluded_not_crashed():
    # value is numeric but the field holds a string that can't compare -> excluded.
    rows = [{"stage": "n/a"}, {"stage": 5}]
    a, b, excluded = split_rows(rows, "stage", ">", 3.0)
    # "n/a" can't be float()'d against a numeric threshold -> excluded.
    assert excluded == 1
    assert len(a) == 1  # stage=5 > 3.0


# ---------------------------------------------------------------------------
# numeric_values / diverge / rank
# ---------------------------------------------------------------------------


def test_numeric_values_skips_non_numeric_and_missing():
    rows = [
        {"x": 1.0},
        {"x": "nope"},
        {"x": True},  # bool excluded even though isinstance(bool, int)
        {"y": 2.0},
        {"x": float("nan")},
        {"x": float("inf")},
    ]
    assert numeric_values(rows, "x") == [1.0]


def test_diverge_identical_cohorts_zero_effect():
    a = [{"x": 5.0} for _ in range(10)]
    b = [{"x": 5.0} for _ in range(10)]
    results = {r.field: r for r in diverge(a, b)}
    assert results["x"].effect_size == 0.0
    assert results["x"].delta_mean == 0.0


def test_diverge_step_function_is_infinite_effect():
    # Zero variance in both cohorts but different constant values: the
    # strongest possible finite-data signal, must not divide-by-zero.
    a = [{"x": 10.0} for _ in range(5)]
    b = [{"x": 1.0} for _ in range(5)]
    results = {r.field: r for r in diverge(a, b)}
    assert math.isinf(results["x"].effect_size)
    assert results["x"].effect_size > 0


def test_diverge_field_present_in_one_cohort_only_ranks_infinite():
    a = [{"schema_version": 1, "x": 1.0} for _ in range(5)]
    b = [{"x": 1.0} for _ in range(5)]  # no schema_version at all
    results = {r.field: r for r in diverge(a, b)}
    sv = results["schema_version"]
    assert math.isinf(sv.effect_size)
    assert sv.note == "A-only"  # present in cohort A (n_a=5), absent from B (n_b=0)
    assert sv.n_a == 5 and sv.n_b == 0
    assert sv.total_a == 5 and sv.total_b == 5
    # x is identical in both -> should NOT be flagged one-sided.
    assert results["x"].note == ""


def test_diverge_field_numeric_nowhere_is_skipped():
    a = [{"stage": "default"}]
    b = [{"stage": "hard"}]
    results = {r.field: r for r in diverge(a, b)}
    assert "stage" not in results  # never numeric -- not ours to rank


def test_diverge_partial_coverage_reported_not_crashed():
    # Half of cohort A never recorded "reward_completion" -- must not
    # crash, and coverage must reflect the partial n.
    a = [{"reward_completion": 1.0}, {"reward_completion": 2.0}, {"other": 1}]
    b = [{"reward_completion": 10.0}]
    results = {r.field: r for r in diverge(a, b)}
    rc = results["reward_completion"]
    assert rc.n_a == 2
    assert rc.total_a == 3
    assert rc.n_b == 1
    assert rc.total_b == 1


def test_diverge_known_effect_size_sign_and_magnitude():
    # mean_a=10, mean_b=0, std both 0 within group except spread of 2 ->
    # pooled_std computed explicitly so the effect size is checkable by hand.
    a = [{"x": 8.0}, {"x": 12.0}]  # mean 10, std 2
    b = [{"x": -2.0}, {"x": 2.0}]  # mean 0, std 2
    results = {r.field: r for r in diverge(a, b)}
    r = results["x"]
    assert r.mean_a == 10.0
    assert r.mean_b == 0.0
    assert r.delta_mean == 10.0
    pooled = math.sqrt((2.0**2 + 2.0**2) / 2.0)
    assert r.effect_size == pytest.approx(10.0 / pooled)


def test_rank_orders_by_absolute_effect_size_descending():
    a = [{"small": 1.01, "big": 100.0}]
    b = [{"small": 1.0, "big": 0.0}]
    # two single-row cohorts -> zero variance everywhere -> both infinite;
    # use non-degenerate variance instead so ranking is meaningfully ordered.
    a = [{"small": 1.05, "big": 20.0}, {"small": 0.95, "big": -20.0}]
    b = [{"small": 1.0, "big": 0.0}, {"small": 1.0, "big": 0.0}]
    results = diverge(a, b)
    ranked = rank(results, top=10)
    fields_in_order = [r.field for r in ranked]
    assert fields_in_order.index("big") < fields_in_order.index("small")


def test_rank_respects_top_n():
    a = [{f"f{i}": float(i) for i in range(10)}]
    b = [{f"f{i}": float(i) + 5.0 for i in range(10)}]
    results = diverge(a, b)
    assert len(rank(results, top=3)) == 3
    assert len(rank(results, top=100)) == len(results)


# ---------------------------------------------------------------------------
# read_jsonl: tolerant of corrupt / mixed-schema real-world files
# ---------------------------------------------------------------------------


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_skips_corrupt_lines(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"generation": 1}\n{ not json\n{"generation": 2}\n\n')
    rows = read_jsonl(p)
    assert [r["generation"] for r in rows] == [1, 2]


def test_read_jsonl_mixed_field_sets_across_rows(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        json.dumps({"generation": 1, "x": 1.0}) + "\n"
        + json.dumps({"generation": 2, "x": 2.0, "schema_version": 1}) + "\n"
    )
    rows = read_jsonl(p)
    assert "schema_version" not in rows[0]
    assert rows[1]["schema_version"] == 1


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess, catches argparse-level breakage)
# ---------------------------------------------------------------------------


def _write_metrics(path: Path) -> None:
    rows = []
    for g in range(20):
        row = {"generation": g, "success_rate": 1.0 if g < 10 else 0.0,
               "ppo_entropy": 1.5 if g < 10 else 0.05}
        if g >= 10:
            row["schema_version"] = 1
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_cli_subprocess_smoke(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path)
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--metrics", str(metrics_path),
         "--split", "generation>9", "--top", "5"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "A (match)=10 rows, B (rest)=10 rows" in result.stdout
    assert "ppo_entropy" in result.stdout


def test_cli_subprocess_run_dir_form(tmp_path):
    run_dir = tmp_path / "some_ckpt"
    run_dir.mkdir()
    _write_metrics(run_dir / "metrics.jsonl")
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run", str(run_dir),
         "--split", "generation>9"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "schema_version" in result.stdout  # B-only field surfaced


def test_cli_bad_split_exits_nonzero(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path)
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--metrics", str(metrics_path),
         "--split", "garbage"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "can't parse" in result.stderr


def test_cli_missing_file_exits_nonzero(tmp_path):
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--metrics", str(tmp_path / "nope.jsonl"),
         "--split", "generation>1"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "no such file" in result.stderr


def test_cli_requires_metrics_or_run(tmp_path):
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--split", "generation>1"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


def test_cli_empty_cohort_does_not_crash(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path)
    script = ROOT / "scripts" / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--metrics", str(metrics_path),
         "--split", "clear_rate<0.1"],  # field doesn't exist in this file
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "nothing to rank" in result.stdout
