"""scripts/train_shared_substrate.py — shared-trunk / per-level-head
offline BC trainer.

Everything here runs on SYNTHETIC npz fixtures and small in-memory
tensors — no emulator, no rollout, no eval subprocess. The one
subprocess test is the --dry-run path, which parses CLI args, reads the
synthetic npz fixtures, and constructs a model; it performs no
rollouts, no training, no writes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_shared_substrate import (  # noqa: E402
    _STANDARD_HEAD_KEYS,
    balance_pairwise,
    build_arg_parser,
    build_model,
    export_checkpoints,
    export_multihead_checkpoint,
    format_balance_report,
    infer_feature_dim,
    infer_num_actions,
    load_level_pairs,
    main,
    parse_data_args,
    parse_warm_heads_args,
    train_joint_bc,
    validate_data_files,
    warm_start_head,
    warm_start_trunk,
)

from src.models.tile_policy import (  # noqa: E402
    TilePolicyNetwork,
    build_tile_policy_from_checkpoint,
)

try:
    from src.training.multihead_policy import MultiHeadTilePolicy
    _HAVE_MULTIHEAD = True
except ImportError:
    MultiHeadTilePolicy = None  # type: ignore[assignment]
    _HAVE_MULTIHEAD = False

_needs_multihead = pytest.mark.skipif(
    not _HAVE_MULTIHEAD,
    reason="src/training/multihead_policy.MultiHeadTilePolicy not on disk",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _pairs(n, feature_dim=6, num_actions=4, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, feature_dim)).astype(np.float32)
    act = rng.integers(0, num_actions, size=(n,)).astype(np.int64)
    return obs, act


def _write_level_npz(path: Path, obs, act, *, convention="obs_0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if convention == "obs_0":
        np.savez(path, n=1, obs_0=obs, act_0=act)
    else:
        np.savez(path, obs=obs, act=act)


class _FakeMultiHead(nn.Module):
    """A minimal stand-in built to the SAME interface contract this
    script depends on (forward_ac(x, level_ids) with per-sample
    routing; export_head(level) in TilePolicyNetwork layout), so the
    orchestration logic in train_shared_substrate.py (balancing,
    per-epoch reporting, gradient isolation, export-shape validation)
    is exercised end-to-end regardless of whether the real
    MultiHeadTilePolicy has landed yet."""

    def __init__(self, levels, num_actions, feature_dim,
                 hidden_dim=16, trunk_dim=8):
        super().__init__()
        self.levels = list(levels)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, trunk_dim)
        self.norm2 = nn.LayerNorm(trunk_dim)
        self.heads = nn.ModuleDict({
            lvl: nn.ModuleDict({
                "actor": nn.Linear(trunk_dim, num_actions),
                "critic": nn.Linear(trunk_dim, 1),
            })
            for lvl in self.levels
        })

    def _trunk(self, x):
        h = F.silu(self.norm1(self.fc1(x)))
        h = F.silu(self.norm2(self.fc2(h)))
        return h

    def forward_ac(self, x, level_ids):
        h = self._trunk(x)
        num_actions = self.heads[self.levels[0]]["actor"].out_features
        logits = torch.zeros(x.shape[0], num_actions)
        value = torch.zeros(x.shape[0])
        for i, lvl in enumerate(self.levels):
            mask = level_ids == i
            if bool(mask.any()):
                logits[mask] = self.heads[lvl]["actor"](h[mask])
                value[mask] = self.heads[lvl]["critic"](h[mask]).squeeze(-1)
        return logits, value

    def export_head(self, level):
        head = self.heads[level]
        return {
            "fc1.weight": self.fc1.weight.detach().cpu().clone(),
            "fc1.bias": self.fc1.bias.detach().cpu().clone(),
            "norm1.weight": self.norm1.weight.detach().cpu().clone(),
            "norm1.bias": self.norm1.bias.detach().cpu().clone(),
            "fc2.weight": self.fc2.weight.detach().cpu().clone(),
            "fc2.bias": self.fc2.bias.detach().cpu().clone(),
            "norm2.weight": self.norm2.weight.detach().cpu().clone(),
            "norm2.bias": self.norm2.bias.detach().cpu().clone(),
            "actor.weight": head["actor"].weight.detach().cpu().clone(),
            "actor.bias": head["actor"].bias.detach().cpu().clone(),
            "critic.weight": head["critic"].weight.detach().cpu().clone(),
            "critic.bias": head["critic"].bias.detach().cpu().clone(),
        }


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


def test_parse_data_args_basic():
    got = parse_data_args(["1-1=a.npz", "1-2=b.npz"])
    assert got == {"1-1": Path("a.npz"), "1-2": Path("b.npz")}
    assert list(got.keys()) == ["1-1", "1-2"]  # order preserved


def test_parse_data_args_missing_equals_raises():
    with pytest.raises(ValueError, match="LEVEL=PATH"):
        parse_data_args(["1-1_a.npz"])


def test_parse_data_args_duplicate_level_raises():
    with pytest.raises(ValueError, match="more than once"):
        parse_data_args(["1-1=a.npz", "1-1=b.npz"])


def test_parse_data_args_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        parse_data_args([])


def test_parse_data_args_empty_path_raises():
    with pytest.raises(ValueError, match="empty path"):
        parse_data_args(["1-1="])


def test_parse_warm_heads_args_allows_empty():
    assert parse_warm_heads_args([]) == {}


def test_validate_data_files_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_data_files({"1-1": tmp_path / "nope.npz"})


def test_validate_data_files_present_ok(tmp_path):
    p = tmp_path / "a.npz"
    p.write_bytes(b"")
    validate_data_files({"1-1": p})  # no raise


# ---------------------------------------------------------------------------
# npz loading
# ---------------------------------------------------------------------------


def test_load_level_pairs_house_demo_convention(tmp_path):
    obs, act = _pairs(10)
    path = tmp_path / "lvl.npz"
    _write_level_npz(path, obs, act, convention="obs_0")
    got_obs, got_act = load_level_pairs(path)
    assert got_obs.shape == (10, 6)
    assert got_act.shape == (10,)
    assert got_obs.dtype == np.float32
    assert got_act.dtype == np.int64
    np.testing.assert_allclose(got_obs, obs.astype(np.float32))
    np.testing.assert_array_equal(got_act, act)


def test_load_level_pairs_pooled_convention(tmp_path):
    obs, act = _pairs(7)
    path = tmp_path / "lvl.npz"
    _write_level_npz(path, obs, act, convention="obs")
    got_obs, got_act = load_level_pairs(path)
    assert got_obs.shape == (7, 6)


def test_load_level_pairs_missing_keys_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="expected obs_0/act_0"):
        load_level_pairs(path)


def test_load_level_pairs_mismatched_lengths_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, obs_0=np.zeros((5, 4)), act_0=np.zeros((3,)))
    with pytest.raises(ValueError, match="must be 1-D matching"):
        load_level_pairs(path)


# ---------------------------------------------------------------------------
# feature_dim / num_actions inference
# ---------------------------------------------------------------------------


def test_infer_feature_dim_agrees():
    by_level = {
        "1-1": _pairs(5, feature_dim=6),
        "1-2": _pairs(5, feature_dim=6),
    }
    assert infer_feature_dim(by_level) == 6


def test_infer_feature_dim_disagreement_raises():
    by_level = {
        "1-1": _pairs(5, feature_dim=6),
        "1-2": _pairs(5, feature_dim=9),
    }
    with pytest.raises(ValueError, match="disagree on feature_dim"):
        infer_feature_dim(by_level)


def test_infer_num_actions():
    by_level = {
        "1-1": _pairs(20, num_actions=4, seed=1),
        "1-2": _pairs(20, num_actions=7, seed=2),
    }
    # max action index across both + 1 (num_actions is a lower bound;
    # random sampling may not hit the top index, so just check sane range)
    n = infer_num_actions(by_level)
    assert 1 <= n <= 7


def test_infer_num_actions_empty_raises():
    with pytest.raises(ValueError, match="no pairs"):
        infer_num_actions({"1-1": (np.zeros((0, 3), dtype=np.float32),
                                    np.zeros((0,), dtype=np.int64))})


# ---------------------------------------------------------------------------
# pairwise balancing
# ---------------------------------------------------------------------------


def test_balance_pairwise_trims_to_smallest_and_preserves_leading_order():
    by_level = {
        "1-1": _pairs(50, seed=1),
        "1-2": _pairs(120, seed=2),
        "1-3": _pairs(30, seed=3),
    }
    levels = ["1-1", "1-2", "1-3"]
    pooled = balance_pairwise(by_level, levels)
    assert pooled["counts_before_balance"] == {"1-1": 50, "1-2": 120, "1-3": 30}
    assert pooled["counts"] == {"1-1": 30, "1-2": 30, "1-3": 30}
    assert pooled["obs"].shape == (90, 6)
    assert pooled["act"].shape == (90,)
    assert pooled["level_ids"].shape == (90,)
    # level_ids in block order 0,0,...,1,1,...,2,2,... matching `levels`
    assert list(pooled["level_ids"][:30]) == [0] * 30
    assert list(pooled["level_ids"][30:60]) == [1] * 30
    assert list(pooled["level_ids"][60:]) == [2] * 30
    # leading-order trim: the kept pairs for 1-2 are its FIRST 30, not a
    # random subset.
    full_obs_1_2, full_act_1_2 = by_level["1-2"]
    np.testing.assert_allclose(pooled["obs"][30:60], full_obs_1_2[:30])
    np.testing.assert_array_equal(pooled["act"][30:60], full_act_1_2[:30])


def test_balance_pairwise_empty_level_raises():
    by_level = {
        "1-1": _pairs(10),
        "1-2": (np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
    }
    with pytest.raises(RuntimeError, match="no pairs for level"):
        balance_pairwise(by_level, ["1-1", "1-2"])


def test_format_balance_report_mentions_every_level():
    pooled = balance_pairwise(
        {"1-1": _pairs(10, seed=1), "1-2": _pairs(20, seed=2)},
        ["1-1", "1-2"])
    report = format_balance_report(pooled)
    assert "1-1: 10 -> 10" in report
    assert "1-2: 20 -> 10" in report


# ---------------------------------------------------------------------------
# gradient isolation (the core correctness property of the routing) —
# runs against the fake to be independent of the concurrent module's
# arrival, AND against the real MultiHeadTilePolicy when it's present.
# ---------------------------------------------------------------------------


def _assert_gradient_isolation(net_factory):
    levels = ["1-1", "1-2", "1-3"]
    net = net_factory(levels)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    # A minibatch that touches ONLY levels 0 ("1-1") and 2 ("1-3") —
    # "1-2" (index 1) never appears.
    x = torch.randn(8, 6)
    y = torch.randint(0, 4, (8,))
    level_ids = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])

    opt.zero_grad()
    logits, _value = net.forward_ac(x, level_ids)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    def head_params(level):
        if hasattr(net, "heads"):  # fake
            head = net.heads[level]
            return list(head["actor"].parameters()) + list(head["critic"].parameters())
        idx = net.levels.index(level)  # real MultiHeadTilePolicy
        return (list(net.actor_heads[idx].parameters())
                + list(net.critic_heads[idx].parameters()))

    for p in head_params("1-2"):
        assert p.grad is None or bool((p.grad == 0).all()), (
            "level absent from the minibatch received nonzero gradient "
            "at its head — routing leaked")

    for level in ("1-1", "1-3"):
        touched = any(p.grad is not None and bool((p.grad != 0).any())
                      for p in head_params(level))
        assert touched, f"level {level} was IN the minibatch but its " \
                         "head received no gradient"

    trunk_params = [net.fc1.weight, net.fc2.weight]
    assert any(p.grad is not None and bool((p.grad != 0).any())
              for p in trunk_params), "trunk received no gradient at all"


def test_gradient_isolation_fake_model():
    _assert_gradient_isolation(
        lambda levels: _FakeMultiHead(levels, num_actions=4, feature_dim=6))


@_needs_multihead
def test_gradient_isolation_real_model():
    _assert_gradient_isolation(
        lambda levels: MultiHeadTilePolicy(
            num_actions=4, num_levels=len(levels), feature_dim=6,
            hidden_dim=16, trunk_dim=8, levels=levels))


# ---------------------------------------------------------------------------
# training loop — per-level accuracy reporting
# ---------------------------------------------------------------------------


def _pooled_two_level(n_per_level=40, feature_dim=6, num_actions=4):
    by_level = {
        "1-1": _pairs(n_per_level, feature_dim=feature_dim,
                      num_actions=num_actions, seed=11),
        "1-2": _pairs(n_per_level, feature_dim=feature_dim,
                      num_actions=num_actions, seed=22),
    }
    return balance_pairwise(by_level, ["1-1", "1-2"])


def test_train_joint_bc_reports_per_level_accuracy_every_epoch():
    pooled = _pooled_two_level()
    net = _FakeMultiHead(["1-1", "1-2"], num_actions=4, feature_dim=6)
    logs = []
    history = train_joint_bc(net, pooled, epochs=3, lr=1e-2, batch_size=16,
                              seed=0, val_frac=0.1, log=logs.append)
    assert len(history) == 3
    assert len(logs) == 3
    for row in history:
        assert set(row["per_level_train_acc"].keys()) == {"1-1", "1-2"}
        assert set(row["per_level_val_acc"].keys()) == {"1-1", "1-2"}
        for acc in row["per_level_train_acc"].values():
            assert 0.0 <= acc <= 1.0
        for acc in row["per_level_val_acc"].values():
            assert 0.0 <= acc <= 1.0  # val_frac=0.1 on n=40/level -> non-empty
        assert 0.0 <= row["overall_train_acc"] <= 1.0
        assert 0.0 <= row["overall_val_acc"] <= 1.0


def test_train_joint_bc_val_frac_zero_reports_nan_and_trains_on_everything():
    """val_frac<=0 is the explicit opt-out: no split happens, every
    val-side number is nan (never silently 0.0 or a copy of train), and
    all pairs are available to the gradient — the pre-fix behavior,
    preserved for callers who ask for it by name."""
    import math
    pooled = _pooled_two_level(n_per_level=20)
    net = _FakeMultiHead(["1-1", "1-2"], num_actions=4, feature_dim=6)
    history = train_joint_bc(net, pooled, epochs=2, lr=1e-2, batch_size=8,
                              seed=0, val_frac=0.0, log=lambda *_: None)
    for row in history:
        assert math.isnan(row["overall_val_acc"])
        assert all(math.isnan(a) for a in row["per_level_val_acc"].values())
        assert not math.isnan(row["overall_train_acc"])


def test_train_joint_bc_val_never_touched_by_gradient():
    """The defect this guards against: reporting fitting-set accuracy
    as if it were evidence of generalization. Labels are pure noise
    (independent of obs), so a high-capacity net can memorize the
    TRAIN split's specific (x, y) pairs (train_acc -> high) while a
    held-out split it never trained on stays near chance (val_acc close
    to 1/num_actions) — proving per_level_val_acc is actually computed
    on samples excluded from every gradient step, not a relabeled copy
    of the train-set number."""
    rng = np.random.default_rng(3)
    n, num_actions, feature_dim = 200, 4, 6
    obs = rng.normal(size=(n, feature_dim)).astype(np.float32)
    act = rng.integers(0, num_actions, size=(n,)).astype(np.int64)  # noise
    by_level = {"1-1": (obs, act)}
    pooled = balance_pairwise(by_level, ["1-1"])
    net = _FakeMultiHead(["1-1"], num_actions=num_actions,
                         feature_dim=feature_dim, hidden_dim=128,
                         trunk_dim=64)
    history = train_joint_bc(net, pooled, epochs=60, lr=1e-2, batch_size=16,
                              seed=0, val_frac=0.25, log=lambda *_: None)
    final = history[-1]
    assert final["overall_train_acc"] > 0.9, (
        "expected the high-capacity net to memorize the noise-label "
        "train split")
    assert final["overall_val_acc"] < 0.6, (
        "held-out accuracy on noise labels should stay near chance "
        f"(1/{num_actions}); {final['overall_val_acc']} suggests the "
        "val split leaked into training")


def test_train_joint_bc_learns_something_on_a_toy_problem():
    """Not a tight bound — just proves the optimization loop actually
    moves the loss down on trivially-learnable synthetic data (obs sign
    determines the action), catching a broken wiring (e.g. detached
    graph, wrong loss target) that a shape-only test would miss."""
    rng = np.random.default_rng(0)
    n = 64
    obs = rng.normal(size=(n, 6)).astype(np.float32)
    act = (obs[:, 0] > 0).astype(np.int64)  # 2 classes, trivially separable
    by_level = {"1-1": (obs, act), "1-2": (obs.copy(), act.copy())}
    pooled = balance_pairwise(by_level, ["1-1", "1-2"])
    net = _FakeMultiHead(["1-1", "1-2"], num_actions=2, feature_dim=6,
                         hidden_dim=32, trunk_dim=16)
    history = train_joint_bc(net, pooled, epochs=40, lr=5e-2, batch_size=32,
                              seed=0, val_frac=0.0, log=lambda *_: None)
    assert history[-1]["overall_train_acc"] > history[0]["overall_train_acc"]
    assert history[-1]["overall_train_acc"] >= 0.9


def test_train_joint_bc_rejects_empty_pool():
    pooled = {"obs": np.zeros((0, 6), dtype=np.float32),
              "act": np.zeros((0,), dtype=np.int64),
              "level_ids": np.zeros((0,), dtype=np.int64),
              "levels": ["1-1"]}
    net = _FakeMultiHead(["1-1"], num_actions=4, feature_dim=6)
    with pytest.raises(ValueError, match="zero pairs"):
        train_joint_bc(net, pooled, epochs=1, lr=1e-3, batch_size=8, seed=0)


# ---------------------------------------------------------------------------
# export — must round-trip through the REAL TilePolicyNetwork /
# eval-harness shape inference, using the fake (interface-conformance
# check independent of the concurrent module) and the real model when
# present.
# ---------------------------------------------------------------------------


def _assert_export_roundtrips(net, levels, out_dir):
    written = export_checkpoints(net, levels, out_dir, epoch=7,
                                  extra_provenance={"note": "test"})
    assert set(written.keys()) == set(levels)
    for level in levels:
        path = written[level]
        assert path.exists()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        assert payload["iter"] == 7
        sd = payload["net_state_dict"]
        assert set(sd.keys()) == set(_STANDARD_HEAD_KEYS)
        # The eval harness's own shape-inference + loader must accept it
        # with zero missing/unexpected keys.
        built, is_recurrent = build_tile_policy_from_checkpoint(
            payload, num_actions=sd["actor.weight"].shape[0],
            feature_dim=sd["fc1.weight"].shape[1])
        assert is_recurrent is False
        assert isinstance(built, TilePolicyNetwork)
        load_res = built.load_state_dict(sd, strict=False)
        assert load_res.missing_keys == []
        assert load_res.unexpected_keys == []


def test_export_checkpoints_fake_model(tmp_path):
    net = _FakeMultiHead(["1-1", "1-2"], num_actions=4, feature_dim=6,
                         hidden_dim=16, trunk_dim=8)
    _assert_export_roundtrips(net, ["1-1", "1-2"], tmp_path / "out")


@_needs_multihead
def test_export_checkpoints_real_model(tmp_path):
    levels = ["1-1", "1-2"]
    net = MultiHeadTilePolicy(num_actions=4, num_levels=2, feature_dim=6,
                              hidden_dim=16, trunk_dim=8, levels=levels)
    _assert_export_roundtrips(net, levels, tmp_path / "out")


def test_export_checkpoints_rejects_bad_export_head(tmp_path):
    class _Broken(_FakeMultiHead):
        def export_head(self, level):
            sd = super().export_head(level)
            del sd["critic.bias"]  # simulate an interface mismatch
            return sd

    net = _Broken(["1-1"], num_actions=4, feature_dim=6)
    with pytest.raises(RuntimeError, match="missing standard"):
        export_checkpoints(net, ["1-1"], tmp_path / "out")


def test_export_multihead_checkpoint_writes_full_state_dict(tmp_path):
    net = _FakeMultiHead(["1-1", "1-2"], num_actions=4, feature_dim=6)
    path = export_multihead_checkpoint(net, tmp_path / "out", epoch=5,
                                       extra_provenance={"x": 1})
    assert path.name == "multihead.pt"
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert payload["iter"] == 5
    assert set(payload["net_state_dict"].keys()) == set(net.state_dict().keys())
    assert payload["shared_substrate"]["x"] == 1


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------


@_needs_multihead
def test_warm_start_trunk_copies_weights_exactly(tmp_path):
    src = TilePolicyNetwork(num_actions=4, feature_dim=6, hidden_dim=16,
                            trunk_dim=8)
    ckpt = tmp_path / "specialist.pt"
    torch.save({"net_state_dict": src.state_dict(), "iter": 0}, str(ckpt))

    net = build_model(["1-1", "1-2"], num_actions=4, feature_dim=6,
                       hidden_dim=16, trunk_dim=8)
    warm_start_trunk(net, ckpt)
    assert torch.allclose(net.fc1.weight, src.fc1.weight)
    assert torch.allclose(net.fc1.bias, src.fc1.bias)
    assert torch.allclose(net.fc2.weight, src.fc2.weight)
    assert torch.allclose(net.norm1.weight, src.norm1.weight)
    assert torch.allclose(net.norm2.weight, src.norm2.weight)


@_needs_multihead
def test_warm_start_head_only_touches_its_own_level(tmp_path):
    src = TilePolicyNetwork(num_actions=4, feature_dim=6, hidden_dim=16,
                            trunk_dim=8)
    ckpt = tmp_path / "specialist.pt"
    torch.save({"net_state_dict": src.state_dict(), "iter": 0}, str(ckpt))

    net = build_model(["1-1", "1-2"], num_actions=4, feature_dim=6,
                       hidden_dim=16, trunk_dim=8)
    other_actor_before = net.actor_heads[1].weight.clone()

    warm_start_head(net, "1-1", ckpt)

    assert torch.allclose(net.actor_heads[0].weight, src.actor.weight)
    assert torch.allclose(net.actor_heads[0].bias, src.actor.bias)
    assert torch.allclose(net.critic_heads[0].weight, src.critic.weight)
    # the OTHER level's head must be untouched
    assert torch.allclose(net.actor_heads[1].weight, other_actor_before)


@_needs_multihead
def test_warm_start_trunk_shape_mismatch_raises(tmp_path):
    src = TilePolicyNetwork(num_actions=4, feature_dim=6, hidden_dim=99,
                            trunk_dim=8)
    ckpt = tmp_path / "specialist.pt"
    torch.save({"net_state_dict": src.state_dict(), "iter": 0}, str(ckpt))
    net = build_model(["1-1"], num_actions=4, feature_dim=6,
                       hidden_dim=16, trunk_dim=8)
    with pytest.raises(ValueError):
        warm_start_trunk(net, ckpt)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


@_needs_multihead
def test_build_model_constructs_with_expected_shapes():
    net = build_model(["1-1", "1-2", "1-3"], num_actions=5, feature_dim=12,
                       hidden_dim=32, trunk_dim=16)
    x = torch.randn(3, 12)
    level_ids = torch.tensor([0, 1, 2])
    logits, value = net.forward_ac(x, level_ids)
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_build_model_without_module_raises_actionable_error(monkeypatch):
    import scripts.train_shared_substrate as mod
    monkeypatch.setattr(mod, "MultiHeadTilePolicy", None)
    monkeypatch.setattr(mod, "_IMPORT_ERROR", ImportError("simulated"))
    with pytest.raises(RuntimeError, match="multihead_policy.py is not importable"):
        mod.build_model(["1-1"], num_actions=4, feature_dim=6,
                        hidden_dim=16, trunk_dim=8)


# ---------------------------------------------------------------------------
# end-to-end main() on synthetic fixtures
# ---------------------------------------------------------------------------


def _write_fixture_pool(tmp_path, levels=("1-1", "1-2"), n=(24, 24),
                        feature_dim=6, num_actions=4):
    paths = {}
    for i, lvl in enumerate(levels):
        obs, act = _pairs(n[i], feature_dim=feature_dim,
                          num_actions=num_actions, seed=100 + i)
        p = tmp_path / f"{lvl}.npz"
        _write_level_npz(p, obs, act)
        paths[lvl] = p
    return paths


def test_dry_run_returns_zero_and_writes_nothing(tmp_path, capsys):
    paths = _write_fixture_pool(tmp_path)
    out_dir = tmp_path / "out"
    argv = ["--out", str(out_dir), "--dry-run"]
    for lvl, p in paths.items():
        argv += ["--data", f"{lvl}={p}"]
    rc = main(argv)
    captured = capsys.readouterr()
    if _HAVE_MULTIHEAD:
        assert rc == 0, captured.err
        assert "dry-run" in captured.out
        assert not out_dir.exists() or list(out_dir.iterdir()) == []
    else:
        assert rc == 3
        assert "multihead_policy.py is not importable" in captured.err
    # the balance/export plan prints regardless of model availability
    assert "balance" in captured.out.lower()
    assert "export plan" in captured.out


def test_main_missing_data_file_exits_2(tmp_path, capsys):
    rc = main(["--data", f"1-1={tmp_path / 'nope.npz'}",
              "--out", str(tmp_path / "out"), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not found" in captured.err


def test_main_malformed_data_arg_exits_2(tmp_path):
    rc = main(["--data", "not-a-pair", "--out", str(tmp_path / "out")])
    assert rc == 2


@_needs_multihead
def test_main_full_run_trains_and_exports(tmp_path, capsys):
    paths = _write_fixture_pool(tmp_path, n=(30, 30))
    out_dir = tmp_path / "out"
    argv = ["--out", str(out_dir), "--epochs", "2", "--batch", "8",
           "--hidden-dim", "16", "--trunk-dim", "8", "--seed", "0"]
    for lvl, p in paths.items():
        argv += ["--data", f"{lvl}={p}"]
    rc = main(argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (out_dir / "multihead.pt").exists()
    assert (out_dir / "1-1.pt").exists()
    assert (out_dir / "1-2.pt").exists()
    # per-level train AND held-out accuracy visible in the training log
    assert "1-1(train=" in captured.out and "1-2(train=" in captured.out
    assert ",val=" in captured.out
    # exported single-head checkpoint loads through the real eval-side
    # shape inference with no missing/unexpected keys.
    payload = torch.load(str(out_dir / "1-1.pt"), map_location="cpu",
                         weights_only=False)
    built, _ = build_tile_policy_from_checkpoint(
        payload, num_actions=4, feature_dim=6)
    load_res = built.load_state_dict(payload["net_state_dict"], strict=False)
    assert load_res.missing_keys == []


# ---------------------------------------------------------------------------
# dry-run subprocess — behaves correctly whether or not the concurrent
# module has landed on disk (see the DEPENDENCY note in the script).
# ---------------------------------------------------------------------------


def test_dry_run_subprocess(tmp_path):
    paths = _write_fixture_pool(tmp_path)
    out_dir = tmp_path / "out"
    argv = [sys.executable, str(ROOT / "scripts" / "train_shared_substrate.py"),
           "--out", str(out_dir), "--dry-run"]
    for lvl, p in paths.items():
        argv += ["--data", f"{lvl}={p}"]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=90)
    if _HAVE_MULTIHEAD:
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "constructed MultiHeadTilePolicy" in proc.stdout
    else:
        assert proc.returncode == 3
        assert "multihead_policy.py is not importable" in proc.stderr
    assert not out_dir.exists() or list(out_dir.iterdir()) == []
