"""Tests for the per-generation timing accumulator."""

from __future__ import annotations

import time

from src.training.timing import GenTimer


def test_section_accumulates_across_calls() -> None:
    t = GenTimer()
    with t.section("foo"):
        time.sleep(0.02)
    with t.section("foo"):
        time.sleep(0.02)
    snap = t.snapshot()
    assert "timing_foo_ms" in snap
    # Slack of a few ms for scheduler jitter.
    assert 30 < snap["timing_foo_ms"] < 200


def test_reset_clears_accumulators() -> None:
    t = GenTimer()
    with t.section("a"):
        time.sleep(0.005)
    t.reset()
    assert t.snapshot() == {}


def test_add_records_manual_duration() -> None:
    t = GenTimer()
    t.add("raw", 5_000_000)  # 5 ms in ns
    snap = t.snapshot()
    assert "timing_raw_ms" in snap
    assert abs(snap["timing_raw_ms"] - 5.0) < 0.01


def test_multiple_sections_independent() -> None:
    t = GenTimer()
    with t.section("x"):
        time.sleep(0.01)
    with t.section("y"):
        time.sleep(0.02)
    snap = t.snapshot()
    assert snap["timing_y_ms"] > snap["timing_x_ms"]
