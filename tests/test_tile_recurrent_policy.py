"""Tests for the recurrent tile policy network. Class-only — trainer
integration (PPO replay through GRU) is a separate task documented
in TaskList #36."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from src.models.tile_policy import (
    TilePolicyNetwork,
    TileRecurrentPolicyNetwork,
    build_tile_policy_from_checkpoint,
    checkpoint_is_recurrent,
    state_dict_is_recurrent,
)


def test_forward_ac_recurrent_shapes() -> None:
    net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
    x = torch.randn(4, 175)
    h = net.initial_hidden(batch_size=4, device=torch.device("cpu"))
    logits, value, h_next = net.forward_ac_recurrent(x, h)
    assert logits.shape == (4, 8)
    assert value.shape == (4,)
    assert h_next.shape == (4, net.gru_dim)


def test_initial_hidden_is_zeros() -> None:
    net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
    h = net.initial_hidden(batch_size=8, device=torch.device("cpu"))
    assert torch.all(h == 0)
    assert h.shape == (8, net.gru_dim)


def test_hidden_evolves_over_steps() -> None:
    """Hidden state should change across steps with the same input,
    confirming the GRU actually carries information forward."""
    net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
    x = torch.randn(2, 175)
    h0 = net.initial_hidden(batch_size=2, device=torch.device("cpu"))
    _, _, h1 = net.forward_ac_recurrent(x, h0)
    _, _, h2 = net.forward_ac_recurrent(x, h1)
    assert not torch.allclose(h0, h1)
    assert not torch.allclose(h1, h2)


def test_stateless_forward_is_one_step_recurrence() -> None:
    """`forward_ac` is the stateless-shaped fallback — calling it
    should produce the same result as one recurrent step starting
    from zero hidden."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    x = torch.randn(3, 175)
    logits_a, value_a = net.forward_ac(x)
    h = net.initial_hidden(3, torch.device("cpu"))
    logits_b, value_b, _ = net.forward_ac_recurrent(x, h)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(value_a, value_b)


def test_param_count_in_expected_range() -> None:
    """At default widths (hidden=64, gru=32, input=175, 8 actions):
    roughly 14k params total. Same order of magnitude as the
    stateless TilePolicyNetwork — keeps the small-network advantage."""
    net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
    assert 10_000 < net.num_params < 30_000


def test_orthogonal_init_zeros_actor_critic_biases() -> None:
    net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
    assert net.actor.bias.abs().sum().item() < 1e-6
    assert net.critic.bias.abs().sum().item() < 1e-6


def test_gradient_flows_through_recurrence() -> None:
    """Backward through the recurrent step should produce gradients on
    GRU weights — sanity check on the BPTT path."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    x = torch.randn(2, 175)
    h = net.initial_hidden(2, torch.device("cpu"))
    logits, value, h_next = net.forward_ac_recurrent(x, h)
    (logits.sum() + value.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name


def test_two_step_bptt_matches_unrolled() -> None:
    """Sanity check that hidden propagation works end-to-end:
    two recurrent steps with carrying hidden produce a different
    output than two independent stateless calls."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    x1 = torch.randn(2, 175)
    x2 = torch.randn(2, 175)

    # With recurrence
    h = net.initial_hidden(2, torch.device("cpu"))
    _, v1_rec, h1 = net.forward_ac_recurrent(x1, h)
    _, v2_rec, _ = net.forward_ac_recurrent(x2, h1)

    # Stateless (each step gets a fresh zero hidden)
    h0 = net.initial_hidden(2, torch.device("cpu"))
    _, v1_sl, _ = net.forward_ac_recurrent(x1, h0)
    _, v2_sl, _ = net.forward_ac_recurrent(x2, h0)

    # First step should match (both started from zero hidden).
    assert torch.allclose(v1_rec, v1_sl)
    # Second step diverges: recurrent carries info from step 1,
    # stateless doesn't.
    assert not torch.allclose(v2_rec, v2_sl)


# ----- save / load round-trip (F14) -----------------------------------


def test_save_writes_kind_and_arch_version(tmp_path: Path) -> None:
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)
    blob = torch.load(str(ckpt), weights_only=False)
    assert blob["kind"] == "tile_gru"
    assert blob["arch_version"] == TileRecurrentPolicyNetwork.ARCH_VERSION


def test_save_load_roundtrip_identical_forward(tmp_path: Path) -> None:
    """Save a recurrent policy, reload it, and confirm the reloaded net
    produces bit-identical logits/value/hidden for the same input +
    hidden — the core F14 guarantee (a trained recurrent agent can be
    reloaded for eval/demo)."""
    net = TileRecurrentPolicyNetwork(num_actions=6, feature_dim=175)
    net.eval()
    x = torch.randn(2, 175)
    h = net.initial_hidden(2, torch.device("cpu"))
    with torch.no_grad():
        logits_before, value_before, h_before = net.forward_ac_recurrent(x, h)

    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)

    net2 = TileRecurrentPolicyNetwork.load(ckpt)
    net2.eval()
    with torch.no_grad():
        logits_after, value_after, h_after = net2.forward_ac_recurrent(x, h)

    assert torch.allclose(logits_before, logits_after, atol=1e-6)
    assert torch.allclose(value_before, value_after, atol=1e-6)
    assert torch.allclose(h_before, h_after, atol=1e-6)


def test_load_preserves_arch_hparams(tmp_path: Path) -> None:
    """Loading reconstructs the architecture from checkpoint metadata,
    even when the caller's defaults differ."""
    net = TileRecurrentPolicyNetwork(
        num_actions=5, feature_dim=200, hidden_dim=48, gru_dim=24
    )
    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)
    loaded = TileRecurrentPolicyNetwork.load(ckpt)
    assert loaded.num_actions == 5
    assert loaded.feature_dim == 200
    assert loaded.hidden_dim == 48
    assert loaded.gru_dim == 24


def test_load_rejects_arch_version_mismatch(tmp_path: Path) -> None:
    """F62 guard: an incompatible arch_version must fail loud, not
    silently misload."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)
    blob = torch.load(str(ckpt), weights_only=False)
    blob["arch_version"] = TileRecurrentPolicyNetwork.ARCH_VERSION + 99
    torch.save(blob, str(ckpt))
    with pytest.raises(ValueError, match="arch_version"):
        TileRecurrentPolicyNetwork.load(ckpt)


def test_load_warns_on_missing_arch_version(tmp_path: Path, caplog) -> None:
    """A legacy (arch_version-less) checkpoint loads but warns."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)
    blob = torch.load(str(ckpt), weights_only=False)
    blob.pop("arch_version")
    torch.save(blob, str(ckpt))
    with caplog.at_level(logging.WARNING):
        TileRecurrentPolicyNetwork.load(ckpt)
    assert any("arch_version" in r.getMessage() for r in caplog.records)


def test_load_raises_on_missing_keys(tmp_path: Path) -> None:
    """Dropping a required GRU weight must be a hard error, never a
    partial (half-random) load."""
    net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    ckpt = tmp_path / "tile_gru.pt"
    net.save(ckpt)
    blob = torch.load(str(ckpt), weights_only=False)
    del blob["state_dict"]["gru.weight_ih"]
    torch.save(blob, str(ckpt))
    with pytest.raises(ValueError, match="missing"):
        TileRecurrentPolicyNetwork.load(ckpt)


def test_load_rejects_stateless_kind(tmp_path: Path) -> None:
    """Loading a stateless (tile_mlp) checkpoint via the recurrent loader
    must fail loud rather than misload a shape-incompatible policy."""
    mlp = TilePolicyNetwork(num_actions=4, feature_dim=175)
    ckpt = tmp_path / "tile_mlp.pt"
    mlp.save(ckpt)  # kind == "tile_mlp"
    with pytest.raises(ValueError, match="tile_gru"):
        TileRecurrentPolicyNetwork.load(ckpt)


# ----- checkpoint dispatch (eval/demo path) ---------------------------


def test_eval_dispatch_picks_recurrent_for_tile_gru() -> None:
    """The dispatch eval_game.py / demo_game.py share must build the
    recurrent policy for a recurrent checkpoint. Trainer/winner
    checkpoints store a raw `net_state_dict` with no kind tag, so the
    GRU weights are the discriminator."""
    rec = TileRecurrentPolicyNetwork(num_actions=7, feature_dim=175)
    ckpt = {"net_state_dict": rec.state_dict(), "iter": 10}

    assert checkpoint_is_recurrent(ckpt)

    net, is_recurrent = build_tile_policy_from_checkpoint(
        ckpt, num_actions=7, feature_dim=175
    )
    assert is_recurrent
    assert isinstance(net, TileRecurrentPolicyNetwork)

    # The dispatched net loads the recurrent weights with no gaps.
    res = net.load_state_dict(ckpt["net_state_dict"], strict=False)
    assert not res.missing_keys
    assert not res.unexpected_keys


def test_dispatch_picks_stateless_for_tile_mlp() -> None:
    mlp = TilePolicyNetwork(num_actions=7, feature_dim=175)
    ckpt = {"net_state_dict": mlp.state_dict(), "iter": 10}

    assert not checkpoint_is_recurrent(ckpt)

    net, is_recurrent = build_tile_policy_from_checkpoint(
        ckpt, num_actions=7, feature_dim=175
    )
    assert not is_recurrent
    assert isinstance(net, TilePolicyNetwork)
    res = net.load_state_dict(ckpt["net_state_dict"], strict=False)
    assert not res.missing_keys
    assert not res.unexpected_keys


def test_dispatch_honors_explicit_kind_tag() -> None:
    """An explicit `kind:"tile_gru"` tag routes to recurrent even when the
    state_dict keys aren't inspectable."""
    assert checkpoint_is_recurrent({"kind": "tile_gru", "state_dict": {}})


def test_state_dict_is_recurrent_detects_gru_keys() -> None:
    rec = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
    mlp = TilePolicyNetwork(num_actions=4, feature_dim=175)
    assert state_dict_is_recurrent(rec.state_dict())
    assert not state_dict_is_recurrent(mlp.state_dict())
    # Works on a bare-dict checkpoint too.
    assert checkpoint_is_recurrent(rec.state_dict())


def test_dispatch_raises_on_unresolvable_state_dict() -> None:
    """A checkpoint whose `net_state_dict` entry is present but empty
    resolves no usable state_dict — this must fail loud, not silently
    hand back a freshly-initialized (random) policy (the same loader
    footgun class 38f2358 patched for the str/Path input case)."""
    with pytest.raises(ValueError, match="state_dict"):
        build_tile_policy_from_checkpoint(
            {"net_state_dict": {}}, num_actions=6, feature_dim=175
        )


# ----- stateless-fallback warning -------------------------------------


def test_stateless_forward_warns_once(caplog) -> None:
    """Calling the stateless `forward_ac` on a recurrent policy silently
    degrades it — it must emit a one-time WARNING (not per-step spam)."""
    TileRecurrentPolicyNetwork._stateless_fallback_warned = False
    try:
        net = TileRecurrentPolicyNetwork(num_actions=4, feature_dim=175)
        x = torch.randn(2, 175)
        with caplog.at_level(logging.WARNING):
            net.forward_ac(x)
            net.forward_ac(x)
            net.forward_ac(x)
        warns = [
            r for r in caplog.records if "forward_ac" in r.getMessage()
        ]
        assert len(warns) == 1
    finally:
        TileRecurrentPolicyNetwork._stateless_fallback_warned = False
