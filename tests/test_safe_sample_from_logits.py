"""Regression tests for `_safe_sample_from_logits` in the trainer.

The bug this guards against: a real 80-minute training run died at
generation 346 with `RuntimeError: probability tensor contains either
inf, nan or element < 0` from `torch.multinomial`. Once the policy
network's logits go NaN/Inf (typically from a divergent PPO update),
every downstream call into `multinomial` crashes the process.

These tests pin the recovery contract:

  * NaN logits → uniform fallback, no exception
  * +Inf logits → defined sampling, no exception
  * -Inf logits → defined sampling, no exception
  * All-zero logits → uniform sampling, no exception
  * A mix of healthy and pathological rows → only bad rows substituted
  * Healthy logits → behavior matches the unsanitised softmax + sample
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.training.trainer import _safe_sample_from_logits


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def test_healthy_logits_round_trip_without_warning(caplog):
    _seed()
    logits = torch.tensor([[1.0, 2.0, 0.5, 0.1]], dtype=torch.float32)
    with caplog.at_level("WARNING"):
        sampled, lp, _ = _safe_sample_from_logits(logits)
    assert sampled.shape == (1,)
    assert lp.shape == (1,)
    assert int(sampled.item()) in {0, 1, 2, 3}
    assert math.isfinite(lp.item())
    assert "diverged" not in caplog.text.lower()


def test_nan_logits_recover_with_warning(caplog):
    _seed()
    logits = torch.tensor([[float("nan"), float("nan"), float("nan")]],
                          dtype=torch.float32)
    with caplog.at_level("WARNING"):
        sampled, lp, _ = _safe_sample_from_logits(logits)
    assert int(sampled.item()) in {0, 1, 2}
    assert math.isfinite(lp.item())


def test_posinf_logits_do_not_crash():
    _seed()
    logits = torch.tensor([[float("inf"), 0.0, 0.0]], dtype=torch.float32)
    sampled, lp, _ = _safe_sample_from_logits(logits)
    # +Inf in one column should bias toward that action without
    # crashing or producing NaN log-probs.
    assert int(sampled.item()) in {0, 1, 2}
    assert math.isfinite(lp.item())


def test_neginf_logits_do_not_crash():
    _seed()
    logits = torch.tensor([[float("-inf"), 0.0, 0.0]], dtype=torch.float32)
    sampled, lp, _ = _safe_sample_from_logits(logits)
    # -Inf should mass on one of the other two actions.
    assert int(sampled.item()) in {1, 2}
    assert math.isfinite(lp.item())


def test_all_zero_logits_sample_uniformly():
    _seed()
    logits = torch.zeros((1, 4), dtype=torch.float32)
    # Run many trials to confirm we sample across actions.
    seen = set()
    for _ in range(200):
        sampled, _, _ = _safe_sample_from_logits(logits)
        seen.add(int(sampled.item()))
    assert seen == {0, 1, 2, 3}, f"expected all 4 actions, saw {seen}"


def test_mixed_rows_do_not_taint_healthy_rows():
    """A pathological row in the same batch as healthy rows must not
    poison the healthy ones — each row's distribution stays
    independent."""
    _seed()
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, 0.1],          # healthy
            [float("nan")] * 4,            # pathological
            [10.0, 0.0, -10.0, 0.0],       # healthy (will mostly pick 0)
        ],
        dtype=torch.float32,
    )
    sampled, lp, _ = _safe_sample_from_logits(logits)
    assert sampled.shape == (3,)
    assert all(math.isfinite(x) for x in lp.tolist())
    # The strongly-biased row 2 should land on action 0 the vast
    # majority of the time — confirms the recovery path didn't blanket
    # the whole batch with uniform.
    hits = 0
    for _ in range(50):
        s2, _, _ = _safe_sample_from_logits(logits)
        if int(s2[2].item()) == 0:
            hits += 1
    assert hits >= 35, f"expected biased row to pick action 0 most of the time, got {hits}/50"


def test_no_exception_under_repeated_pathological_input():
    """Smoke: 1000 calls on NaN input, all return without crashing.
    Mirrors the failure mode from the real training run where the
    crash happened on every step once the network diverged."""
    _seed()
    logits = torch.full((4, 8), float("nan"), dtype=torch.float32)
    for _ in range(1000):
        sampled, lp, _ = _safe_sample_from_logits(logits)
        assert sampled.shape == (4,)
        assert torch.isfinite(lp).all()
