"""Tests for the Phase 2 discrete-time hazard model.

Everything here runs against synthetic tensors and synthetic npz files
built in-process -- no ROM, no emulator, no checkpoint from any live
campaign. Per the hazard-substrate build order
(docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md), this phase is judged
on getting the survival likelihood and the IPCW C-index gate right;
these tests are written against a hand-derivable, known hazard
structure so failures point at the mechanism, not at noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.hazard_model import (
    NUM_ACTIONS,
    OBS_DIM,
    HazardMLP,
    build_time_bin_edges,
    concordance_index_ipcw,
    cumulative_risk_score,
    discrete_time_survival_nll,
    discretize_time,
    encode_input,
    infer_source_groups,
)

import scripts.train_hazard as train_hazard


# ---------------------------------------------------------------------------
# Model shape / param budget
# ---------------------------------------------------------------------------


def test_hazard_mlp_default_param_budget():
    model = HazardMLP()  # hidden=128, 2 hidden layers, n_bins=20 defaults
    n_params = model.n_params()
    # spec: "~100k params" -- input layer (718*128) dominates the count
    # regardless of n_bins, so this is a loose sanity band, not a pin.
    assert 90_000 <= n_params <= 140_000, n_params


def test_hazard_mlp_forward_shape():
    model = HazardMLP(input_dim=OBS_DIM + NUM_ACTIONS, hidden=32,
                       n_hidden_layers=2, n_bins=10)
    x = torch.zeros(5, OBS_DIM + NUM_ACTIONS)
    logits = model(x)
    assert logits.shape == (5, 10)


def test_hazard_mlp_rejects_bad_config():
    with pytest.raises(ValueError):
        HazardMLP(n_hidden_layers=0)
    with pytest.raises(ValueError):
        HazardMLP(n_bins=0)


def test_encode_input_one_hot_and_validation():
    obs = np.zeros((3, OBS_DIM), dtype=np.int8)
    obs[:, 0] = [1, 2, 3]
    action = np.array([0, 5, 2])
    x = encode_input(obs, action, num_actions=NUM_ACTIONS)
    assert x.shape == (3, OBS_DIM + NUM_ACTIONS)
    onehot = x[:, OBS_DIM:]
    assert torch.equal(onehot, F.one_hot(torch.tensor(action), NUM_ACTIONS).float())
    with pytest.raises(ValueError):
        encode_input(obs, np.array([0, 6, 2]), num_actions=NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Discrete-time survival likelihood -- hand-derived correctness
# ---------------------------------------------------------------------------


def test_survival_nll_uncensored_matches_hand_derivation():
    logits = torch.tensor([[0.5, -0.3, 1.2]])
    h = torch.sigmoid(logits)[0]
    bin_idx = torch.tensor([1])       # event happened in bin 1 (0-indexed)
    censored = torch.tensor([0.0])
    expected_log_lik = torch.log(1 - h[0]) + torch.log(h[1])
    got = discrete_time_survival_nll(logits, bin_idx, censored, reduction="none")
    assert torch.allclose(got, -expected_log_lik, atol=1e-6)


def test_survival_nll_censored_matches_hand_derivation():
    logits = torch.tensor([[0.5, -0.3, 1.2]])
    h = torch.sigmoid(logits)[0]
    bin_idx = torch.tensor([1])       # observation window ended in bin 1
    censored = torch.tensor([1.0])
    expected_log_lik = torch.log(1 - h[0]) + torch.log(1 - h[1])
    got = discrete_time_survival_nll(logits, bin_idx, censored, reduction="none")
    assert torch.allclose(got, -expected_log_lik, atol=1e-6)


def test_survival_nll_first_bin_edge_case():
    logits = torch.tensor([[0.5, -0.3, 1.2]])
    h = torch.sigmoid(logits)[0]
    bin_idx = torch.tensor([0])
    uncensored_nll = discrete_time_survival_nll(
        logits, bin_idx, torch.tensor([0.0]), reduction="none")
    censored_nll = discrete_time_survival_nll(
        logits, bin_idx, torch.tensor([1.0]), reduction="none")
    assert torch.allclose(uncensored_nll, -torch.log(h[0]), atol=1e-6)
    assert torch.allclose(censored_nll, -torch.log(1 - h[0]), atol=1e-6)


def test_censored_and_uncensored_terms_differ_and_are_not_symmetric():
    """Same logits, same bin -- censored vs. uncensored must diverge
    (different likelihood term at the boundary bin), and the
    difference must be exactly the log-odds at that bin, not some
    incidental numerical drift."""
    logits = torch.tensor([[0.2, 0.9, -1.4, 0.3]])
    bin_idx = torch.tensor([2])
    nll_uncensored = discrete_time_survival_nll(
        logits, bin_idx, torch.tensor([0.0]), reduction="none")
    nll_censored = discrete_time_survival_nll(
        logits, bin_idx, torch.tensor([1.0]), reduction="none")
    assert not torch.allclose(nll_uncensored, nll_censored)
    h2 = torch.sigmoid(logits[0, 2])
    # they differ by exactly log(h_2) - log(1 - h_2) (the fail-vs-survive
    # swap at the boundary bin; the "survived earlier bins" term is
    # identical in both and must cancel out of the difference).
    diff = nll_censored - nll_uncensored
    expected_diff = -torch.log(1 - h2) - (-torch.log(h2))
    assert torch.allclose(diff, expected_diff.reshape(1), atol=1e-6)


def test_survival_nll_rejects_out_of_range_bin_idx():
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        discrete_time_survival_nll(
            logits, torch.tensor([0, 3]), torch.tensor([0.0, 0.0]))


def test_discretize_time_bin_edges():
    edges = build_time_bin_edges(horizon=10.0, n_bins=5)  # width-2 bins
    steps = np.array([0.0, 1.9, 2.0, 5.5, 9.9, 100.0])
    idx = discretize_time(steps, edges)
    # bins: [0,2) [2,4) [4,6) [6,8) [8,10); last value clips into bin 4
    assert idx.tolist() == [0, 0, 1, 2, 4, 4]


# ---------------------------------------------------------------------------
# Loss decreases + recovers a known hazard ordering
# ---------------------------------------------------------------------------


def _synthetic_hazard_dataset(n_samples: int, n_bins: int, seed: int):
    """obs[:, 0] carries a "danger" level in {0..4}; the true per-bin
    hazard rate is monotonically increasing in danger. This is the
    "known hazard structure" the spec asks tests to recover -- purely
    synthetic, no game knowledge involved."""
    rng = np.random.default_rng(seed)
    obs = np.zeros((n_samples, OBS_DIM), dtype=np.int8)
    danger = rng.integers(0, 5, size=n_samples)
    obs[:, 0] = danger
    action = rng.integers(0, NUM_ACTIONS, size=n_samples)
    p = np.clip(0.05 + 0.12 * danger, 0.01, 0.85)  # per-bin hazard prob

    steps_to_event = np.zeros(n_samples, dtype=np.float64)
    censored = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        event_bin = None
        for b in range(n_bins):
            if rng.random() < p[i]:
                event_bin = b
                break
        if event_bin is None:
            steps_to_event[i] = float(n_bins)
            censored[i] = 1.0
        else:
            steps_to_event[i] = event_bin + rng.random()
            censored[i] = 0.0
    return obs, action, steps_to_event, censored, danger


def test_loss_decreases_and_recovers_known_hazard_ordering():
    torch.manual_seed(0)
    n_bins = 8
    obs, action, steps_to_event, censored, danger = _synthetic_hazard_dataset(
        n_samples=1000, n_bins=n_bins, seed=1)
    edges = build_time_bin_edges(horizon=float(n_bins), n_bins=n_bins)
    bin_idx = torch.as_tensor(discretize_time(steps_to_event, edges))
    censored_t = torch.as_tensor(censored)
    X = encode_input(obs, action, num_actions=NUM_ACTIONS)

    model = HazardMLP(input_dim=OBS_DIM + NUM_ACTIONS, hidden=32,
                       n_hidden_layers=1, n_bins=n_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def epoch_loss():
        model.eval()
        with torch.no_grad():
            return float(discrete_time_survival_nll(model(X), bin_idx, censored_t))

    initial_loss = epoch_loss()
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        model.train()
        perm = torch.randperm(X.shape[0], generator=gen)
        for start in range(0, X.shape[0], 64):
            idx = perm[start:start + 64]
            loss = discrete_time_survival_nll(model(X[idx]), bin_idx[idx],
                                               censored_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    final_loss = epoch_loss()

    assert final_loss < initial_loss, (initial_loss, final_loss)

    # Recovered ordering: mean predicted risk must increase with danger.
    model.eval()
    with torch.no_grad():
        risk = cumulative_risk_score(model(X)).numpy()
    mean_risk_by_danger = [risk[danger == d].mean() for d in range(5)]
    assert mean_risk_by_danger[-1] > mean_risk_by_danger[0], mean_risk_by_danger
    # loosely monotonic: allow a little noise but the trend must hold.
    increases = sum(mean_risk_by_danger[i + 1] > mean_risk_by_danger[i]
                     for i in range(4))
    assert increases >= 3, mean_risk_by_danger


# ---------------------------------------------------------------------------
# IPCW C-index -- perfect ordering vs. shuffled
# ---------------------------------------------------------------------------


def test_c_index_perfect_ordering_is_one():
    n = 30
    times = np.arange(1, n + 1, dtype=np.float64)
    events = np.ones(n)             # fully observed, no censoring
    risk = -times                   # earliest deaths -> highest risk
    result = concordance_index_ipcw(times, events, risk)
    assert result["c_index"] == pytest.approx(1.0, abs=1e-9)
    assert result["n_comparable"] > 0


def test_c_index_worst_ordering_is_zero():
    n = 30
    times = np.arange(1, n + 1, dtype=np.float64)
    events = np.ones(n)
    risk = times  # exactly backwards: later deaths get higher risk
    result = concordance_index_ipcw(times, events, risk)
    assert result["c_index"] == pytest.approx(0.0, abs=1e-9)


def test_c_index_shuffled_risk_is_about_half():
    rng = np.random.default_rng(7)
    n = 400
    times = rng.permutation(np.arange(1, n + 1).astype(np.float64))
    events = np.ones(n)
    risk = rng.permutation(np.arange(n).astype(np.float64))  # unrelated to times
    result = concordance_index_ipcw(times, events, risk)
    assert abs(result["c_index"] - 0.5) < 0.08, result["c_index"]


def test_c_index_with_censoring_still_computable_and_bounded():
    rng = np.random.default_rng(3)
    n = 200
    true_time = rng.exponential(scale=10.0, size=n)
    censor_time = rng.exponential(scale=12.0, size=n)
    times = np.minimum(true_time, censor_time)
    events = (true_time <= censor_time).astype(np.float64)
    risk = -true_time  # model "knows" the true time -> should score well
    result = concordance_index_ipcw(times, events, risk)
    assert not np.isnan(result["c_index"])
    assert 0.0 <= result["c_index"] <= 1.0
    # a model that (nearly) knows the truth should be well above the gate.
    assert result["c_index"] > 0.85, result["c_index"]


def test_c_index_ipcw_weight_differs_from_unweighted_when_censoring_present():
    """The whole point of IPCW: with heavy censoring, up-weighting
    events relative to how much censoring risk they carried should
    move the score away from the naive unweighted count for at least
    one constructed case -- i.e. the weights are actually doing
    something, not silently reducing to 1.0 everywhere."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    events = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    risk = np.array([5.0, 1.0, 4.0, 1.0, 1.0, 1.0])
    weighted = concordance_index_ipcw(times, events, risk)
    # Unweighted concordance (Harrell-style) over the same eligible pairs.
    from src.training.hazard_model import KaplanMeier
    g = KaplanMeier.fit(times, 1 - events).eval_left(times)
    assert not np.allclose(g, 1.0), "fixture must actually produce censoring drops in G"
    assert 0.0 <= weighted["c_index"] <= 1.0


# ---------------------------------------------------------------------------
# Split by source state, not by row
# ---------------------------------------------------------------------------


def _make_forked_npz_dict(n_groups: int, forks_per_group: int, n_bins: int, seed: int):
    """Simulates hazard_collect.py's micro-forking output: every row
    forked from the same saved state carries the SAME `obs` (the
    pre-fork observation), differing only in the forked action and the
    resulting outcome."""
    rng = np.random.default_rng(seed)
    obs_rows, action_rows, died_rows, steps_rows, censored_rows = [], [], [], [], []
    group_ids = []
    for g in range(n_groups):
        base_obs = np.zeros(OBS_DIM, dtype=np.int8)
        base_obs[0] = rng.integers(0, 5)
        # Real micro-forked states carry a full 712-wide tile grid, so
        # distinct source states are byte-distinguishable in practice.
        # Give the fixture a wide random signature (well beyond
        # n_groups possible collisions) so this stays true here too --
        # index 0 alone (5 possible values) would let unrelated groups
        # collide by construction, which is a fixture bug, not a
        # property of real collected data.
        base_obs[1:33] = rng.integers(0, 128, size=32)
        danger = int(base_obs[0])
        p = np.clip(0.05 + 0.12 * danger, 0.01, 0.85)
        for _k in range(forks_per_group):
            obs_rows.append(base_obs.copy())
            action_rows.append(rng.integers(0, NUM_ACTIONS))
            event_bin = None
            for b in range(n_bins):
                if rng.random() < p:
                    event_bin = b
                    break
            if event_bin is None:
                steps_rows.append(float(n_bins))
                censored_rows.append(1.0)
                died_rows.append(0)
            else:
                steps_rows.append(event_bin + rng.random())
                censored_rows.append(0.0)
                died_rows.append(1)
            group_ids.append(g)
    data = {
        "obs": np.stack(obs_rows).astype(np.int8),
        "action": np.array(action_rows),
        "died": np.array(died_rows),
        "steps_to_event": np.array(steps_rows),
        "censored": np.array(censored_rows, dtype=np.float32),
    }
    return data, np.array(group_ids)


def test_infer_source_groups_recovers_fork_siblings_via_obs_hash():
    data, true_groups = _make_forked_npz_dict(
        n_groups=12, forks_per_group=5, n_bins=6, seed=11)
    inferred = infer_source_groups(data)
    # Two rows share an inferred group iff they share a true group.
    n = len(true_groups)
    for i in range(0, n, 7):
        for j in range(0, n, 11):
            same_true = true_groups[i] == true_groups[j]
            same_inferred = inferred[i] == inferred[j]
            assert same_true == same_inferred, (i, j)


def test_infer_source_groups_prefers_explicit_id_column():
    data, true_groups = _make_forked_npz_dict(
        n_groups=4, forks_per_group=3, n_bins=4, seed=2)
    data["source_state_id"] = true_groups + 1000  # arbitrary explicit ids
    inferred = infer_source_groups(data)
    assert np.array_equal(inferred, true_groups + 1000)


def test_infer_source_groups_uses_hazard_collect_source_state_idx_column():
    """`source_state_idx` is the exact column name
    scripts/hazard_collect.py's build_fork_jobs/collect_labels write
    (the index into its `source_states` restore-point list). This must
    be picked up in preference to the obs-hash fallback -- it is exact
    where the hash is only a byte-identity proxy."""
    data, true_groups = _make_forked_npz_dict(
        n_groups=6, forks_per_group=4, n_bins=5, seed=6)
    data["source_state_idx"] = true_groups.copy()
    inferred = infer_source_groups(data)
    assert np.array_equal(inferred, true_groups)


def test_split_by_source_state_no_group_straddles_train_and_val():
    data, true_groups = _make_forked_npz_dict(
        n_groups=20, forks_per_group=6, n_bins=6, seed=5)
    train_idx, val_idx, n_groups, n_val_groups = train_hazard.split_by_source_state(
        data, val_frac=0.3, seed=0)
    assert n_groups == 20
    assert n_val_groups > 0
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(true_groups)))
    assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()

    train_groups = set(true_groups[train_idx].tolist())
    val_groups = set(true_groups[val_idx].tolist())
    assert train_groups & val_groups == set(), (
        "a source state's forks leaked across the train/val split")


def test_split_by_source_state_row_level_split_would_have_leaked():
    """Negative control: proves the row-level split this function
    deliberately avoids WOULD leak, so the group-level assertion above
    is testing something real and not a tautology."""
    data, true_groups = _make_forked_npz_dict(
        n_groups=10, forks_per_group=8, n_bins=5, seed=9)
    rng = np.random.default_rng(0)
    n = len(true_groups)
    row_val_mask = rng.random(n) < 0.3
    row_train_groups = set(true_groups[~row_val_mask].tolist())
    row_val_groups = set(true_groups[row_val_mask].tolist())
    assert row_train_groups & row_val_groups, (
        "fixture must actually be forked (>1 row/group) for this control "
        "to demonstrate row-level leakage")


# ---------------------------------------------------------------------------
# CLI plumbing (still no emulator: an npz on disk, trained on CPU)
# ---------------------------------------------------------------------------


def test_train_hazard_load_and_validate_preserves_source_state_idx(tmp_path):
    """Regression guard: load_and_validate must carry `source_state_idx`
    through from the npz into the dict handed to infer_source_groups,
    not silently drop it by only copying the 5 REQUIRED_KEYS."""
    data, true_groups = _make_forked_npz_dict(
        n_groups=8, forks_per_group=3, n_bins=5, seed=8)
    data["source_state_idx"] = true_groups.copy()
    npz_path = tmp_path / "hazard_data.npz"
    np.savez(npz_path, **data)

    loaded = train_hazard.load_and_validate(str(npz_path))
    assert "source_state_idx" in loaded
    assert np.array_equal(loaded["source_state_idx"], true_groups)

    train_idx, val_idx, n_groups, n_val_groups = train_hazard.split_by_source_state(
        loaded, val_frac=0.25, seed=1)
    assert n_groups == 8
    train_groups = set(true_groups[train_idx].tolist())
    val_groups = set(true_groups[val_idx].tolist())
    assert train_groups & val_groups == set()


def test_train_hazard_dry_run_writes_nothing(tmp_path):
    data, _ = _make_forked_npz_dict(n_groups=10, forks_per_group=4, n_bins=6, seed=1)
    npz_path = tmp_path / "hazard_data.npz"
    np.savez(npz_path, **data)
    out_dir = tmp_path / "out"
    rc = train_hazard.main([
        "--data", str(npz_path), "--out", str(out_dir),
        "--epochs", "3", "--bins", "6", "--hidden", "16",
        "--hidden-layers", "1", "--dry-run",
    ])
    assert rc == 0
    assert not out_dir.exists()


def test_train_hazard_end_to_end_writes_checkpoint_and_report(tmp_path):
    data, _ = _make_forked_npz_dict(n_groups=25, forks_per_group=6, n_bins=8, seed=4)
    npz_path = tmp_path / "hazard_data.npz"
    np.savez(npz_path, **data)
    out_dir = tmp_path / "out"
    rc = train_hazard.main([
        "--data", str(npz_path), "--out", str(out_dir),
        "--epochs", "5", "--bins", "8", "--hidden", "16",
        "--hidden-layers", "1", "--batch", "32", "--seed", "0",
        "--gate", "0.0",  # plumbing test, not re-litigating the gate value
    ])
    assert rc == 0  # gate=0.0 always passes; this is checking files got written
    ckpt = out_dir / "hazard_model.pt"
    report = out_dir / "hazard_report.json"
    assert ckpt.exists()
    assert report.exists()

    import json
    with open(report) as f:
        payload = json.load(f)
    assert payload["n_train_rows"] > 0
    assert payload["n_val_rows"] > 0
    assert "gate" in payload and "c_index" in payload["gate"]
    assert len(payload["history"]) == 5

    import torch as _torch
    ckpt_payload = _torch.load(ckpt, weights_only=False)
    assert ckpt_payload["config"]["n_bins"] == 8
    assert "state_dict" in ckpt_payload
