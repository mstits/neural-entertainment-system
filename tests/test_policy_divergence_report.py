"""Tests for scripts/policy_divergence_report.py.

Pure-CPU, no emulator: synthetic checkpoints + synthetic observation npz.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.policy_divergence_report import (  # noqa: E402
    compare_policies,
    compute_state_metrics,
    infer_dims,
    load_policy_from_checkpoint,
    main,
    metrics_from_logits,
    near_tie_summary,
    obs_provenance,
)
from src.models.tile_policy import TilePolicyNetwork  # noqa: E402


# ---------------------------------------------------------------- fixtures

FEAT, HID, TRUNK, ACTS = 24, 16, 8, 6


def _make_net(seed: int = 0) -> TilePolicyNetwork:
    torch.manual_seed(seed)
    return TilePolicyNetwork(num_actions=ACTS, feature_dim=FEAT,
                             hidden_dim=HID, trunk_dim=TRUNK)


@pytest.fixture()
def trainer_ckpt(tmp_path):
    """Trainer-style checkpoint: {'iter': N, 'net_state_dict': ...}."""
    net = _make_net(seed=1)
    p = tmp_path / "trainer_style.pt"
    torch.save({"iter": 123, "net_state_dict": net.state_dict()}, p)
    return p, net


@pytest.fixture()
def obs_npz(tmp_path):
    rng = np.random.default_rng(7)
    obs = rng.integers(-4, 5, size=(40, FEAT), dtype=np.int8)
    p = tmp_path / "obs.npz"
    np.savez(p, state=obs)
    return p, obs


# ---------------------------------------------------------------- dims / load

def test_infer_dims_from_state_dict():
    net = _make_net()
    dims = infer_dims(net.state_dict())
    assert dims == {"feature_dim": FEAT, "hidden_dim": HID,
                    "trunk_dim": TRUNK, "num_actions": ACTS}


def test_load_trainer_style_checkpoint(trainer_ckpt):
    path, ref = trainer_ckpt
    net, meta = load_policy_from_checkpoint(path)
    assert meta["iter"] == 123
    assert net.feature_dim == FEAT and net.num_actions == ACTS
    x = torch.zeros(3, FEAT)
    with torch.no_grad():
        got, _ = net.forward_ac(x)
        want, _ = ref.forward_ac(x)
    assert torch.allclose(got, want)


def test_load_native_save_format(tmp_path):
    net = _make_net(seed=2)
    p = tmp_path / "native.pt"
    net.save(p)
    loaded, meta = load_policy_from_checkpoint(p)
    x = torch.ones(2, FEAT)
    with torch.no_grad():
        assert torch.allclose(loaded.forward_ac(x)[0], net.forward_ac(x)[0])


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "junk.pt"
    torch.save({"hello": 1}, p)
    with pytest.raises(ValueError):
        load_policy_from_checkpoint(p)


# ---------------------------------------------------------------- metric math

def test_metrics_from_logits_known_values():
    logits = torch.tensor([
        [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],   # gap 1.0, argmax 0
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # uniform: gap 0, ent ln6
    ])
    m = metrics_from_logits(logits)
    assert m["argmax"].tolist() == [0, 0]
    assert m["action_gap"][0] == pytest.approx(1.0, abs=1e-6)
    assert m["action_gap"][1] == pytest.approx(0.0, abs=1e-6)
    assert m["entropy"][1] == pytest.approx(math.log(6.0), abs=1e-5)
    assert m["entropy"][0] < m["entropy"][1]
    # p_top1 of the uniform row is exactly 1/6.
    assert m["p_top1"][1] == pytest.approx(1.0 / 6.0, abs=1e-6)
    # sampled-vs-argmax disagreement mass = 1 - p_top1.
    assert m["p_disagree"][1] == pytest.approx(5.0 / 6.0, abs=1e-6)


def test_near_tie_summary_counts():
    gaps = np.array([0.05, 0.2, 0.4, 3.0])
    ents = np.array([1.0, 1.0, 1.0, 0.01])
    summary = near_tie_summary({"action_gap": gaps, "entropy": ents,
                                "p_top1": np.array([.3, .4, .5, .99]),
                                "p_disagree": np.array([.7, .6, .5, .01])},
                               thresholds=(0.1, 0.25, 0.5))
    assert summary["n_states"] == 4
    assert summary["near_tie_counts"]["0.1"] == 1
    assert summary["near_tie_counts"]["0.25"] == 2
    assert summary["near_tie_counts"]["0.5"] == 3
    assert summary["near_tie_frac"]["0.5"] == pytest.approx(0.75)
    assert summary["entropy"]["mean"] == pytest.approx(np.mean(ents))
    assert summary["action_gap"]["p50"] == pytest.approx(np.median(gaps))


def test_compute_state_metrics_batching_matches_single_pass(obs_npz):
    _, obs = obs_npz
    net = _make_net(seed=3)
    whole = compute_state_metrics(net, obs, batch_size=1024)
    batched = compute_state_metrics(net, obs, batch_size=7)
    for k in ("entropy", "action_gap", "p_top1"):
        np.testing.assert_allclose(whole[k], batched[k], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(whole["argmax"], batched["argmax"])


# ---------------------------------------------------------------- compare

def test_compare_identical_policy_no_divergence(obs_npz):
    _, obs = obs_npz
    net = _make_net(seed=4)
    rep = compare_policies(net, net, obs, top_k=5)
    assert rep["argmax_disagree_frac"] == 0.0
    assert rep["mean_kl_ab"] == pytest.approx(0.0, abs=1e-7)
    assert rep["top_divergent_states"] == [] or all(
        row["kl"] <= 1e-7 for row in rep["top_divergent_states"])


def test_compare_different_policies_reports_divergence(obs_npz):
    _, obs = obs_npz
    rep = compare_policies(_make_net(seed=5), _make_net(seed=6), obs, top_k=3)
    assert 0.0 <= rep["argmax_disagree_frac"] <= 1.0
    assert rep["mean_kl_ab"] > 0.0
    assert len(rep["top_divergent_states"]) == 3
    kls = [r["kl"] for r in rep["top_divergent_states"]]
    assert kls == sorted(kls, reverse=True)
    assert all(0 <= r["index"] < len(obs) for r in rep["top_divergent_states"])


# ---------------------------------------------------------------- provenance

def test_provenance_reads_meta_and_expert_window(tmp_path):
    rng = np.random.default_rng(11)
    obs = rng.integers(-4, 5, size=(10, FEAT), dtype=np.int8)
    meta = {"provenance": {"source": "unit-test"}}
    p = tmp_path / "expert.npz"
    np.savez(p, state=obs, meta_json=json.dumps(meta),
             is_expert_window=np.ones(10, dtype=np.int8))
    prov = obs_provenance(p)
    assert prov["meta_json"] == meta
    assert prov["is_expert_window_frac"] == pytest.approx(1.0)
    assert prov["has_position_data"] is False
    # The scope caveat must say metrics only cover the supplied distribution
    # and that no per-state position data exists to slice by x.
    note = prov["distribution_scope"]
    assert "supplied" in note and "position" in note


def test_provenance_detects_position_keys(tmp_path):
    obs = np.zeros((4, FEAT), dtype=np.int8)
    p = tmp_path / "pos.npz"
    np.savez(p, state=obs, x_pos=np.array([1, 2, 3, 4]))
    prov = obs_provenance(p)
    assert prov["has_position_data"] is True
    assert "x_pos" in prov["position_keys"]


def test_provenance_plain_npz_still_carries_scope_note(tmp_path):
    p = tmp_path / "plain.npz"
    np.savez(p, state=np.zeros((3, FEAT), dtype=np.int8))
    prov = obs_provenance(p)
    assert prov["meta_json"] is None
    assert prov["is_expert_window_frac"] is None
    assert prov["has_position_data"] is False
    assert isinstance(prov["distribution_scope"], str)
    assert prov["distribution_scope"]


def test_cli_report_embeds_provenance(tmp_path, trainer_ckpt):
    ckpt_path, _ = trainer_ckpt
    rng = np.random.default_rng(13)
    obs = rng.integers(-4, 5, size=(8, FEAT), dtype=np.int8)
    npz = tmp_path / "obs_meta.npz"
    np.savez(npz, state=obs, meta_json=json.dumps({"k": 1}),
             is_expert_window=np.ones(8, dtype=np.int8))
    out = tmp_path / "rep.json"
    rc = main(["--checkpoint", str(ckpt_path), "--obs", str(npz),
               "--obs-key", "state", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    prov = rep["obs"]["provenance"]
    assert prov["meta_json"] == {"k": 1}
    assert prov["is_expert_window_frac"] == pytest.approx(1.0)
    assert prov["has_position_data"] is False
    assert "distribution_scope" in prov


# ---------------------------------------------------------------- CLI

def test_cli_single_checkpoint_end_to_end(tmp_path, trainer_ckpt, obs_npz):
    ckpt_path, _ = trainer_ckpt
    npz_path, obs = obs_npz
    out = tmp_path / "report.json"
    rc = main(["--checkpoint", str(ckpt_path), "--obs", str(npz_path),
               "--obs-key", "state", "--out", str(out), "--top-k", "4"])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["n_states"] == len(obs)
    assert rep["checkpoint"]["iter"] == 123
    assert "near_tie_counts" in rep and "entropy" in rep
    assert len(rep["worst_near_ties"]) == 4
    gaps = [r["action_gap"] for r in rep["worst_near_ties"]]
    assert gaps == sorted(gaps)


def test_cli_two_checkpoints_adds_comparison(tmp_path, obs_npz):
    npz_path, _ = obs_npz
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"iter": 1, "net_state_dict": _make_net(seed=8).state_dict()}, a)
    torch.save({"iter": 2, "net_state_dict": _make_net(seed=9).state_dict()}, b)
    out = tmp_path / "cmp.json"
    rc = main(["--checkpoint", str(a), "--checkpoint-b", str(b),
               "--obs", str(npz_path), "--obs-key", "state",
               "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert "comparison" in rep
    assert "argmax_disagree_frac" in rep["comparison"]
