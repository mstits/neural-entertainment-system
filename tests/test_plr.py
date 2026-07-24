"""Unit tests for the isolated PLR context (src/training/plr.py)."""

import random
from pathlib import Path

import pytest

from src.training.plr import build_plr_context, PLRContext


def _write_state(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(b"\x00" * 32)  # blob content is opaque to PLR
    return str(p)


def _profile(tmp_path: Path, *, enabled=True, with_holdout=True) -> dict:
    lv = [
        {"level": "1-1", "state": None},
        {"level": "1-2", "state": _write_state(tmp_path, "s12.state")},
        {"level": "1-4", "state": _write_state(tmp_path, "s14.state")},
    ]
    if with_holdout:
        lv.append({"level": "1-3", "state": _write_state(tmp_path, "s13.state"),
                   "holdout": True})
    return {"reinforce": {"plr_enabled": enabled}, "plr_levels": lv}


def test_disabled_returns_none(tmp_path):
    prof = _profile(tmp_path, enabled=False)
    assert build_plr_context(prof, 8) is None


def test_no_levels_returns_none():
    assert build_plr_context({"reinforce": {"plr_enabled": True}}, 8) is None


def test_index_zero_is_cold_boot(tmp_path):
    ctx = build_plr_context(_profile(tmp_path), 8)
    assert isinstance(ctx, PLRContext)
    assert ctx.idx_to_level[0] == "1-1"      # cold-boot level pinned to index 0
    assert ctx.states[0] is None             # index 0 has no blob (reset_all covers it)
    assert ctx.states[1] is not None and ctx.states[2] is not None
    assert ctx.level_to_idx == {"1-1": 0, "1-2": 1, "1-4": 2}


def test_holdout_excluded_from_training(tmp_path):
    ctx = build_plr_context(_profile(tmp_path), 8)
    assert "1-3" not in ctx.train_labels
    assert "1-3" in ctx.holdout
    # sampler never returns the holdout level
    for _ in range(200):
        assert ctx.sample() != "1-3"


def test_requires_exactly_one_cold_boot(tmp_path):
    # two null states -> ambiguous cold-boot index
    prof = {"reinforce": {"plr_enabled": True}, "plr_levels": [
        {"level": "1-1", "state": None},
        {"level": "1-2", "state": None},
    ]}
    with pytest.raises(ValueError):
        build_plr_context(prof, 8)


def test_missing_state_file_raises(tmp_path):
    prof = {"reinforce": {"plr_enabled": True}, "plr_levels": [
        {"level": "1-1", "state": None},
        {"level": "1-2", "state": str(tmp_path / "does_not_exist.state")},
    ]}
    with pytest.raises(ValueError):
        build_plr_context(prof, 8)


def test_env_level_initialized_to_cold_boot(tmp_path):
    ctx = build_plr_context(_profile(tmp_path), 5)
    assert ctx.env_level == ["1-1"] * 5
    assert ctx.distribution() == {"1-1": 5}


def test_inverse_success_upweights_hard_level(tmp_path):
    random.seed(0)
    ctx = build_plr_context(_profile(tmp_path, with_holdout=False), 8)
    # Make 1-1 easy (always clears) and 1-4 hard (never clears); 1-2 neutral.
    for _ in range(60):
        ctx.record("1-1", True)
        ctx.record("1-4", False)
        ctx.record("1-2", True)
        ctx.record("1-2", False)
    counts = {"1-1": 0, "1-2": 0, "1-4": 0}
    for _ in range(6000):
        counts[ctx.sample()] += 1
    # The hard level (1-4) must be sampled strictly more than the easy one (1-1).
    assert counts["1-4"] > counts["1-1"], counts
    # And 1-1 (always-success) should be the least sampled.
    assert counts["1-1"] == min(counts.values()), counts


def test_success_rates_reported(tmp_path):
    ctx = build_plr_context(_profile(tmp_path, with_holdout=False), 8)
    for _ in range(20):
        ctx.record("1-1", True)
    for _ in range(20):
        ctx.record("1-4", False)
    rates = ctx.success_rates()
    assert rates["1-1"] == pytest.approx(1.0)
    assert rates["1-4"] == pytest.approx(0.0)
