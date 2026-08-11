"""The offline instrument sweep (`scripts/score_banked_iterates.py`).

Everything the sweep computes is a pure function of arrays, so it is tested
the way arithmetic should be: against hand-constructed inputs whose answer is
known independently, with no checkpoints, no emulator and no torch. The one
orchestration test injects a duck-typed `forward` so the whole per-iterate
path runs without a policy network existing.

Context: `configs/mario_1_1_backward_v4.yaml` registered "score the late
`vanilla_ppo_iter_*.pt` checkpoints too" and exactly one iterate was ever
scored (iter 260 -> greedy 0.00). These instruments are what makes scoring
the other forty-two free.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.score_banked_iterates import (
    dormant_fraction,
    effective_rank,
    eval_cost_estimate,
    iterate_number,
    margin_summary,
    param_drift,
    score_one_iterate,
    sort_iterates,
    top_two_margin,
)

_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================================
# Top-two logit margin — "is the argmax a tie?"
# ==========================================================================

def test_top_two_margin_is_the_gap_between_the_best_two_logits():
    logits = np.array([
        [0.0, 5.0, 1.0],      # 5.0 - 1.0
        [3.0, 3.0, -9.0],     # an exact tie
        [-1.0, -1.25, -8.0],  # a near-tie, negative logits
    ])
    assert top_two_margin(logits) == pytest.approx([4.0, 0.0, 0.25])


def test_top_two_margin_accepts_a_single_row_and_ignores_order():
    assert top_two_margin([2.0, 7.0, 7.5]) == pytest.approx([0.5])
    # The margin depends on the two largest values only, not their positions.
    a = top_two_margin([[1.0, 9.0, 4.0]])
    b = top_two_margin([[9.0, 4.0, 1.0]])
    assert a == pytest.approx(b)


def test_top_two_margin_of_a_one_action_space_cannot_tie():
    assert np.isinf(top_two_margin([[3.0]])).all()


def test_margin_summary_reports_the_threshold_it_used():
    logits = np.array([[0.0, 0.05], [0.0, 1.0], [0.0, 2.0], [0.0, 0.1]])
    s = margin_summary(logits, tie_threshold=0.15)
    assert s["n_states"] == 4
    assert s["min"] == pytest.approx(0.05)
    assert s["median"] == pytest.approx(0.55)
    # 0.05 and 0.10 are under 0.15 -> half the states read as tied.
    assert s["tie_fraction"] == pytest.approx(0.5)
    assert s["tie_threshold"] == pytest.approx(0.15)
    # The threshold is reported, never baked in: a different one moves only
    # the fraction.
    assert margin_summary(logits, tie_threshold=1.5)["tie_fraction"] == \
        pytest.approx(0.75)


# ==========================================================================
# Dormant units (Sokar et al., ICML 2023)
# ==========================================================================

def test_dormant_fraction_counts_exactly_dead_units_at_tau_zero():
    # Four units, two of which never activate.
    acts = np.array([[1.0, 0.0, 3.0, 0.0],
                     [2.0, 0.0, 1.0, 0.0]])
    assert dormant_fraction(acts) == pytest.approx(0.5)


def test_dormant_fraction_is_normalised_not_absolute():
    """The score is a unit's share of the LAYER's mean activation, so scaling
    every unit by the same constant cannot change the answer — that is what
    makes it comparable across iterates with different activation scales."""
    acts = np.array([[1.0, 0.0, 3.0]])
    assert dormant_fraction(acts * 1e-6) == pytest.approx(dormant_fraction(acts))
    assert dormant_fraction(acts * 1e6) == pytest.approx(dormant_fraction(acts))


def test_dormant_fraction_tau_catches_nearly_dead_units():
    # Unit 1 carries 1/101 of the mass; at tau=0 it is alive, at tau=0.1 it
    # is dormant.
    acts = np.array([[100.0, 1.0]])
    assert dormant_fraction(acts, tau=0.0) == pytest.approx(0.0)
    assert dormant_fraction(acts, tau=0.1) == pytest.approx(0.5)


def test_dormant_fraction_of_a_fully_dead_layer_is_one_not_nan():
    assert dormant_fraction(np.zeros((4, 8))) == pytest.approx(1.0)


def test_dormant_fraction_rejects_a_shapeless_input():
    with pytest.raises(ValueError):
        dormant_fraction(np.zeros((2, 2, 2)))


# ==========================================================================
# Effective rank
# ==========================================================================

def test_effective_rank_of_a_rank_one_matrix_is_one():
    m = np.outer(np.arange(1.0, 6.0), np.ones(4))
    r = effective_rank(m)
    assert r["srank"] == 1
    assert r["erank"] == pytest.approx(1.0)


def test_effective_rank_of_an_identity_is_full_and_maximal_entropy():
    r = effective_rank(np.eye(6))
    assert r["srank"] == 6
    assert r["erank"] == pytest.approx(6.0)


def test_effective_rank_collapses_as_the_spectrum_concentrates():
    """A representation that loses directions loses effective rank — the
    reading the plasticity story actually rests on."""
    full = effective_rank(np.diag([1.0, 1.0, 1.0, 1.0]))
    partial = effective_rank(np.diag([1.0, 1.0, 0.001, 0.001]))
    collapsed = effective_rank(np.diag([1.0, 1e-9, 1e-9, 1e-9]))
    assert full["erank"] > partial["erank"] > collapsed["erank"]
    assert full["srank"] >= partial["srank"] >= collapsed["srank"] == 1


def test_effective_rank_of_a_zero_matrix_is_zero_not_nan():
    r = effective_rank(np.zeros((3, 3)))
    assert r == {"srank": 0, "erank": 0.0, "n_singular_values": 3}


# ==========================================================================
# Parameter drift  ||theta_t - theta_0||
# ==========================================================================

class _FakeTensor:
    """Duck-typed stand-in for a torch tensor (detach/cpu/numpy chain)."""

    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def test_param_drift_is_the_global_l2_over_shared_tensors():
    t0 = {"a": np.array([0.0, 0.0]), "b": np.array([[3.0]])}
    tt = {"a": np.array([3.0, 4.0]), "b": np.array([[3.0]])}
    d = param_drift(tt, t0)
    assert d["l2"] == pytest.approx(5.0)
    assert d["n_tensors"] == 2
    assert d["largest_tensor"] == "a"
    assert d["identical"] is False


def test_param_drift_flags_identical_weights():
    """The 'was the scored network the trained network?' discriminator: zero
    drift means no eval difference between the two can be real."""
    sd = {"w": _FakeTensor([1.0, -2.0, 0.5])}
    d = param_drift(sd, {"w": _FakeTensor([1.0, -2.0, 0.5])})
    assert d["l2"] == 0.0 and d["identical"] is True
    assert d["relative"] == pytest.approx(0.0)


def test_param_drift_is_relative_to_the_baseline_norm():
    d = param_drift({"w": np.array([2.0])}, {"w": np.array([1.0])})
    assert d["relative"] == pytest.approx(1.0)


def test_param_drift_skips_mismatched_and_missing_tensors():
    d = param_drift(
        {"w": np.array([1.0, 1.0]), "extra": np.array([9.0])},
        {"w": np.array([0.0, 0.0, 0.0]), "other": np.array([1.0])},
    )
    assert d["n_tensors"] == 0 and d["l2"] == 0.0
    # No shared, comparable tensors -> not a claim of identity.
    assert d["identical"] is False


# ==========================================================================
# Iterate discovery + the stub cost model
# ==========================================================================

def test_iterate_number_parses_the_trainer_naming_and_orders_by_it():
    assert iterate_number("checkpoints/x/vanilla_ppo_iter_00080.pt") == 80
    assert iterate_number("best_1-1.pt") is None
    ordered = sort_iterates([
        "d/vanilla_ppo_iter_00260.pt", "d/vanilla_ppo_iter_00080.pt",
        "d/best_1-1.pt", "d/vanilla_ppo_iter_00090.pt",
    ])
    assert [iterate_number(p) for p in ordered] == [80, 90, 260, None]


def test_eval_cost_estimate_is_arithmetic_and_says_it_is_an_estimate():
    """The paired curve is the one instrument that costs emulation, so the
    sweep prices it instead of silently spending an hour."""
    c = eval_cost_estimate(43, 30, seconds_per_episode=4.5, workers=8)
    assert c["total_episodes"] == 43 * 30 * 2
    assert c["serial_seconds"] == pytest.approx(43 * 30 * 2 * 4.5)
    assert c["wall_seconds_estimate"] == pytest.approx(c["serial_seconds"] / 8)


# ==========================================================================
# The per-iterate orchestration, with an injected forward
# ==========================================================================

def test_score_one_iterate_runs_every_free_instrument_without_torch():
    states = np.zeros((5, 3), dtype=np.float32)

    def fake_forward(_s):
        logits = np.tile(np.array([1.0, 1.05, -4.0]), (5, 1))
        trunk = np.tile(np.array([1.0, 0.0, 2.0, 0.0]), (5, 1))
        hidden = np.tile(np.array([1.0, 1.0, 1.0]), (5, 1))
        return logits, trunk, hidden

    row = score_one_iterate(
        "checkpoints/run/vanilla_ppo_iter_00130.pt", states=states,
        theta_0=None, tie_threshold=0.15, dormant_tau=0.0, srank_delta=0.01,
        forward=fake_forward,
    )
    assert row["iter"] == 130
    assert row["margin"]["tie_fraction"] == pytest.approx(1.0)   # 0.05 < 0.15
    assert row["dormant"]["trunk"] == pytest.approx(0.5)
    assert row["dormant"]["hidden"] == pytest.approx(0.0)
    assert row["effective_rank"]["trunk"]["srank"] == 1          # 5 copies
    assert "drift" not in row                                    # no baseline


def test_score_one_iterate_without_states_is_drift_only():
    row = score_one_iterate(
        "d/vanilla_ppo_iter_00010.pt", states=None, theta_0=None,
        tie_threshold=0.15, dormant_tau=0.0, srank_delta=0.01,
        forward=lambda s: (None, None, None),
    )
    assert set(row) == {"path", "iter"}


# ==========================================================================
# CLI contract — no emulation by default
# ==========================================================================

def test_cli_stub_costs_the_curve_and_never_emulates_by_default(tmp_path):
    """`--eval` is opt-in. Without it the sweep must produce the free
    instruments plus a labelled COST ESTIMATE — never a number that looks
    like a measured clear rate."""
    import torch

    for it in (10, 20):
        torch.save(
            {"net_state_dict": {"w": torch.full((2, 2), float(it))}},
            tmp_path / f"vanilla_ppo_iter_{it:05d}.pt",
        )
    out = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "score_banked_iterates.py"),
         "--iterates", str(tmp_path / "vanilla_ppo_iter_*.pt"),
         "--episodes", "30", "--eval-workers", "8"],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(out.stdout)
    assert doc["eval_ran"] is False
    assert doc["n_iterates"] == 2
    assert "estimate" in doc["curve_stub_note"]
    assert doc["curve_stub_cost"]["total_episodes"] == 2 * 30 * 2
    # Drift is measured against the lowest-numbered iterate by default.
    assert [r["iter"] for r in doc["iterates"]] == [10, 20]
    assert doc["iterates"][0]["drift"]["l2"] == pytest.approx(0.0)
    assert doc["iterates"][1]["drift"]["l2"] == pytest.approx(20.0)
    assert all("curve" not in r for r in doc["iterates"])
