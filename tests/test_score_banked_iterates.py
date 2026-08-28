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


def test_param_drift_discloses_skipped_tensors_and_never_calls_a_resized_net_identical():
    """A resized trunk (e.g. hidden_dim 64 vs 96) must not (a) vanish from the
    report with no trace, or (b) let a bit-identical head make the whole pair
    read as 'identical' while the trunk went uncompared."""
    t0 = {"same": np.array([1.0, 2.0]), "trunk": np.zeros((64, 10))}
    tt = {"same": np.array([1.0, 2.0]), "trunk": np.ones((96, 10))}
    d = param_drift(tt, t0)
    assert d["skipped_tensors"] == ["trunk"]
    assert d["n_skipped"] == 1
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


def test_cli_survives_one_unloadable_iterate_and_keeps_the_rest(tmp_path):
    """A later iterate that is truncated / has no recognisable state_dict must
    not cost the whole sweep: the run must still exit 0, still print the
    iterates already scored, and still honour --out — never a bare traceback
    with nothing on stdout and no file on disk."""
    import torch

    torch.save(
        {"net_state_dict": {"w": torch.full((2, 2), 10.0)}},
        tmp_path / "vanilla_ppo_iter_00010.pt",
    )
    # No net_state_dict/state_dict/model_state_dict key, and not every value
    # is tensor-shaped -> load_iterate() raises ValueError for this one.
    torch.save({"not_a_policy": "garbage"}, tmp_path / "vanilla_ppo_iter_00020.pt")

    out_path = tmp_path / "sweep.json"
    out = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "score_banked_iterates.py"),
         "--iterates", str(tmp_path / "vanilla_ppo_iter_*.pt"),
         "--out", str(out_path)],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(out.stdout)
    assert doc["n_iterates"] == 2
    rows = {r["iter"]: r for r in doc["iterates"]}
    assert rows[10]["drift"]["l2"] == pytest.approx(0.0)
    assert "error" in rows[20] and rows[20]["error"]
    # --out must carry the same, not be left unwritten by the failure.
    assert json.loads(out_path.read_text()) == doc


# ==========================================================================
# INSTRUMENT 6 — V_adv, the B5 discriminator.
#
# Registration: docs/proposals/VADV_PREREG_2026-08-27.md. The whole point of
# this instrument is that it could come back COLLAPSED or LIVE, so every test
# below pins a DIRECTION, and the two anti-vacuity tests are paired with a
# revert check that proves they fail when the mechanism is stubbed out.
# ==========================================================================

from scripts.score_banked_iterates import (            # noqa: E402
    advantage_variance,
    bootstrap_eta2,
    classify_vadv,
    critic_advantage,
    decode_gx,
    injected_power,
    load_transition_bank,
    normalise_vadv,
    parse_bands,
    permutation_null,
    qualifying_rows,
    score_vadv,
    state_cell_keys,
)

_F = 178  # SMBTileObservationV2 features per frame; 4 stacked = 712


def _obs(gx_values, *, frames=4, jitter=None):
    """Minimal stacked v2 tile observations carrying a chosen gx."""
    gx_values = np.asarray(gx_values, dtype=np.int64)
    out = np.zeros((gx_values.size, frames * _F), dtype=np.int8)
    for f in range(frames):
        out[:, f * _F + 175] = (gx_values // 256).astype(np.int8)
        out[:, f * _F + 176] = ((gx_values % 256) // 2).astype(np.int8)
    if jitter is not None:
        out[:, 0] = np.asarray(jitter, dtype=np.int8)
    return out


# --- the anti-vacuity pair -------------------------------------------------

def _flat_batch():
    """Every action carries the SAME advantage: no action effect at all.

    Advantage varies by STATE and by within-cell noise — which is what real
    data looks like — but not by action. Anything that reports this as LIVE
    is measuring state heterogeneity, not action discrimination.
    """
    cells = np.repeat(np.arange(40), 6)
    actions = np.tile(np.array([0, 0, 0, 1, 1, 1]), 40)
    noise = np.random.default_rng(11).normal(0.0, 0.3, cells.size)
    return np.repeat(np.linspace(-1.0, 1.0, 40), 6) + noise, cells, actions


def _separated_batch():
    """Actions are cleanly separated within every cell: a live action effect."""
    adv, cells, actions = _flat_batch()
    return adv + 5.0 * actions, cells, actions


def test_vadv_reports_collapsed_when_all_actions_share_an_advantage():
    adv, cells, actions = _flat_batch()
    obs = advantage_variance(adv, cells, actions)
    null = permutation_null(adv, cells, actions, n_perm=200)
    assert obs["eta2"] < 0.25
    assert obs["eta2"] <= null["q975"]
    assert classify_vadv(obs, null) == "COLLAPSED"


def test_vadv_of_an_utterly_variance_free_batch_is_zero_not_none():
    """The identity carve-out: no variance anywhere is COLLAPSED by
    arithmetic, and 0/0 must not leak out as a None that reads VOID."""
    cells = np.repeat(np.arange(10), 6)
    adv = np.repeat(np.linspace(-1.0, 1.0, 10), 6)   # constant within a cell
    actions = np.tile(np.array([0, 0, 0, 1, 1, 1]), 10)
    obs = advantage_variance(adv, cells, actions)
    assert obs["raw"] == pytest.approx(0.0)
    assert obs["eta2"] == pytest.approx(0.0)
    assert classify_vadv(obs, permutation_null(adv, cells, actions,
                                               n_perm=50)) == "COLLAPSED"


def test_vadv_reports_live_when_actions_carry_different_advantages():
    adv, cells, actions = _separated_batch()
    obs = advantage_variance(adv, cells, actions)
    assert obs["raw"] > 0.0
    assert obs["eta2"] > 0.9
    null = permutation_null(adv, cells, actions, n_perm=200)
    assert obs["eta2"] > null["q975"]
    assert classify_vadv(obs, null) == "LIVE"


def test_the_anti_vacuity_pair_fails_when_the_mechanism_is_reverted():
    """REVERT CHECK. A metric that returned the same shape everywhere would
    'confirm' whatever it was pointed at. Stub V_adv to a constant — the
    shape a vacuous instrument has — and BOTH assertions above must break."""
    def reverted(_adv, _cells, _actions):
        return {"eta2": 0.5, "raw": 1.0, "n_cells": 1,
                "eta2_null_analytic": 0.5}

    flat_in, live_in = _flat_batch(), _separated_batch()
    flat, live = reverted(*flat_in), reverted(*live_in)
    assert flat["eta2"] == live["eta2"]          # no discrimination survives

    # The real instrument SEPARATES these two; the reverted one cannot.
    real_flat = advantage_variance(*flat_in)
    real_live = advantage_variance(*live_in)
    assert real_live["eta2"] > real_flat["eta2"] + 0.5
    assert live["eta2"] == flat["eta2"]

    # The reverted stub hands back ONE verdict for both batches. The real
    # instrument hands back two different ones. That difference is the entire
    # claim this instrument makes, and it is what the revert removes.
    live_null = permutation_null(*live_in, n_perm=100)
    flat_null = permutation_null(*flat_in, n_perm=100)
    assert classify_vadv(live, live_null) == classify_vadv(flat, flat_null)
    assert classify_vadv(real_live, live_null) == "LIVE"
    assert classify_vadv(real_flat, flat_null) == "COLLAPSED"


# --- the quantity itself ---------------------------------------------------

def test_advantage_variance_raw_is_the_variance_of_per_action_means():
    # one cell, two actions, means 0 and 4 -> population variance 4.0
    adv = np.array([0.0, 0.0, 4.0, 4.0])
    obs = advantage_variance(adv, np.zeros(4), np.array([0, 0, 1, 1]))
    assert obs["raw"] == pytest.approx(4.0)
    assert obs["n_cells"] == 1


def test_eta2_is_invariant_to_affine_rescaling_of_the_critic():
    """An UNNORMALISED V_adv would collapse or inflate purely from critic
    scale drift — the fifth-vacuous-instrument failure. eta2 must not."""
    adv, cells, actions = _separated_batch()
    base = advantage_variance(adv, cells, actions)
    scaled = advantage_variance(37.0 * adv + 11.0, cells, actions)
    assert scaled["eta2"] == pytest.approx(base["eta2"])
    assert scaled["raw"] != pytest.approx(base["raw"])   # raw is NOT invariant


def test_eta2_null_is_reported_because_it_is_not_zero():
    """The ReDo lesson: a threshold outside its acting range never fires.
    eta2's null expectation for a 2-action, 3-row cell is 0.5, not 0."""
    obs = advantage_variance(np.array([1.0, 2.0, 3.0]), np.zeros(3),
                             np.array([0, 0, 1]))
    assert obs["eta2_null_analytic"] == pytest.approx(0.5)


def test_critic_advantage_zeroes_the_bootstrap_on_absorbing_rows():
    a = critic_advantage([1.0, 1.0], [10.0, 10.0], done=[0, 1], gamma=0.5)
    assert a[0] == pytest.approx(4.0)     # 0.5*10 - 1
    assert a[1] == pytest.approx(-1.0)    # 0.5*10*0 - 1


def test_normalise_vadv_calls_a_constant_critic_degenerate_not_collapsed():
    """A dead critic must read VOID. Returning 0.0 would manufacture the
    mis-specification signature out of a broken checkpoint."""
    out = normalise_vadv(0.0, 0.0)
    assert out["raw_norm"] is None
    assert out["degenerate_critic"] is True
    assert normalise_vadv(2.0, 4.0)["raw_norm"] == pytest.approx(0.5)


def test_classify_vadv_voids_on_a_degenerate_null_instead_of_collapsing():
    obs = {"eta2": 0.0}
    assert classify_vadv(obs, {"median": None, "q975": None}) == "VOID"
    assert classify_vadv({"eta2": None},
                         {"median": 0.1, "q975": 0.2}) == "VOID"


# --- cells, bands, coverage ------------------------------------------------

def test_qualifying_rows_drops_singleton_actions_and_thin_cells():
    cells = np.array([0, 0, 0, 0, 1, 1, 1])
    actions = np.array([0, 0, 1, 1, 0, 0, 1])     # cell 1 has a singleton
    keep = qualifying_rows(cells, actions, min_rows_per_action=2,
                           min_actions=2, min_rows=4)
    assert keep.tolist() == [True, True, True, True, False, False, False]


def test_decode_gx_reads_the_encoders_own_progress_scalars():
    gx = decode_gx(_obs([0, 160, 2674, 2872, 3266]))
    assert gx.tolist() == [0, 160, 2674, 2872, 3266]


def test_decode_gx_refuses_a_width_that_is_not_a_stacked_v2_observation():
    with pytest.raises(ValueError, match="not a multiple"):
        decode_gx(np.zeros((3, 175), dtype=np.int8))


def test_state_cell_keys_ignore_the_animation_phase_but_not_the_tiles():
    same = _obs([100, 100])
    same[1, 3 * _F + 177] = 42                       # phase byte only
    assert len(set(state_cell_keys(same).tolist())) == 1
    differ = _obs([100, 100])
    differ[1, 3 * _F + 5] = 1                        # a tile
    assert len(set(state_cell_keys(differ).tolist())) == 2


def test_parse_bands_reads_the_registered_band_spec():
    assert parse_bands("WALL=2674:2872,PC_B5=2872:3267") == {
        "WALL": (2674, 2872), "PC_B5": (2872, 3267)}


def test_injected_power_is_high_for_a_real_effect_and_low_for_none():
    adv, cells, actions = _flat_batch()
    strong = injected_power(adv, cells, actions, effect=5.0, n_trials=8,
                            n_perm=60)
    none = injected_power(adv, cells, actions, effect=0.0, n_trials=8,
                          n_perm=60)
    assert strong["power"] >= 0.95
    assert none["power"] <= 0.25


def test_bootstrap_ci_brackets_a_live_reading():
    adv, cells, actions = _separated_batch()
    ci = bootstrap_eta2(adv, cells, actions, n_boot=60)
    assert ci["lo"] > 0.5 and ci["hi"] <= 1.0


# --- the offline bank shell ------------------------------------------------

def test_load_transition_bank_refuses_to_splice_across_unknown_episodes():
    """obs_0 without traj_len: consecutive rows are not transitions."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.npz"
        np.savez(p, obs_0=np.zeros((4, 712), dtype=np.int8),
                 act_0=np.zeros(4, dtype=np.int64))
        with pytest.raises(ValueError, match="traj_len"):
            load_transition_bank([str(p)])


def test_load_transition_bank_never_crosses_a_trajectory_boundary():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.npz"
        obs = _obs([0, 1 * 2, 2 * 2, 100, 101 * 2])
        np.savez(p, obs_0=obs, act_0=np.arange(5),
                 traj_len=np.array([3, 2]))
        bank = load_transition_bank([str(p)])
        # 3-step traj -> 2 transitions, 2-step traj -> 1. Never 4.
        assert bank["state"].shape[0] == 3


def test_score_vadv_separates_a_live_band_from_a_flat_band():
    """End-to-end through the band machinery with an injected critic: the
    positive band must read LIVE and the flat band COLLAPSED, or nothing
    downstream of this instrument is admissible."""
    rng = np.random.default_rng(0)
    n_cells, per = 30, 6
    gx_live = np.repeat(np.arange(2900, 2900 + n_cells * 2, 2), per)
    gx_flat = np.repeat(np.arange(2674, 2674 + n_cells * 2, 2), per)
    gx = np.concatenate([gx_live, gx_flat])
    actions = np.tile(np.array([0, 0, 0, 1, 1, 1]), 2 * n_cells)
    # Rows inside one cell differ in the HISTORY frames (not in the cell key),
    # so V(s) varies within a cell exactly as it does under the real loose
    # key — that heterogeneity has to land in SS_within, not SS_between.
    state = _obs(gx, jitter=np.arange(gx.size) % 120)
    # The successor must actually DEPEND on the action, or the bank encodes
    # no action axis for the critic to discriminate over.
    nxt = state.copy()
    nxt[:, 1] = (actions + 1).astype(np.int8)
    nxt[:, 2] = (np.arange(gx.size) % 120).astype(np.int8)

    live_mask = np.arange(gx.size) < gx_live.size
    v_state = rng.normal(size=gx.size)
    # Successors differ by action ONLY in the live band.
    v_next = np.where(live_mask, 9.0 * actions, 0.0) + 0.05 * rng.normal(
        size=gx.size)
    values = {}
    for i in range(gx.size):
        values[state[i].tobytes()] = float(v_state[i])
        values[nxt[i].tobytes()] = float(v_next[i])

    def vfn(batch):
        return np.array([values[r.tobytes()] for r in batch])

    bank = {"state": state, "action": actions, "next_state": nxt,
            "done": np.zeros(gx.size, dtype=np.int64), "sources": ["synthetic"]}
    out = score_vadv(bank, vfn,
                     bands={"PC": (2872, 3267), "WALL": (2674, 2872)},
                     n_perm=120, n_boot=60)
    assert out["regions"]["PC"]["verdict"] == "LIVE"
    assert out["regions"]["WALL"]["verdict"] == "COLLAPSED"
    assert out["regions"]["PC"]["observed"]["eta2"] > \
        out["regions"]["WALL"]["observed"]["eta2"]
    assert out["degenerate_critic"] is False


def test_score_vadv_with_a_zeroed_critic_is_degenerate_not_collapsed():
    """NC-a. A provably constant critic carries no gradient anywhere; the
    instrument must say so rather than emit a comfortable 0.0."""
    gx = np.repeat(np.arange(2900, 2960, 2), 6)
    state = _obs(gx, jitter=np.repeat(np.arange(30), 6) % 120)
    bank = {"state": state, "action": np.tile([0, 0, 0, 1, 1, 1], 30),
            "next_state": state.copy(),
            "done": np.zeros(gx.size, dtype=np.int64), "sources": ["synthetic"]}
    out = score_vadv(bank, lambda b: np.full(len(b), 3.0),
                     bands={"PC": (2872, 3267)}, n_perm=40, n_boot=20)
    assert out["degenerate_critic"] is True
    assert out["regions"]["PC"]["raw_norm"] is None
    # Caught live by the A3 control on a real checkpoint: a zeroed critic
    # makes every advantage exactly 0, and the arithmetic carve-out reported
    # a confident COLLAPSED — the mis-specification signature, manufactured
    # out of a broken checkpoint. The critic IS the instrument; a dead one
    # reads VOID.
    assert out["regions"]["PC"]["verdict"] == "VOID"
    assert out["regions"]["PC"]["void_reason"] == "degenerate_critic"
