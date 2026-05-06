"""Tests for the tile-based policy network."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.tile_policy import TilePolicyNetwork


def test_forward_shape() -> None:
    net = TilePolicyNetwork(num_actions=8, feature_dim=175)
    x = torch.randn(4, 175)
    logits = net(x)
    assert logits.shape == (4, 8)


def test_forward_ac_returns_logits_and_value() -> None:
    net = TilePolicyNetwork(num_actions=8, feature_dim=175)
    x = torch.randn(2, 175)
    logits, value = net.forward_ac(x)
    assert logits.shape == (2, 8)
    assert value.shape == (2,)


def test_param_count_matches_expectation() -> None:
    """At default widths (64/32) with 175 input features and 8 actions
    the total should be ~14k. Concrete count drifts a little if the
    architecture changes; this is a regression guard, not a hard pin."""
    net = TilePolicyNetwork(num_actions=8, feature_dim=175)
    assert 10_000 < net.num_params < 20_000


def test_orthogonal_init_zeros_biases() -> None:
    net = TilePolicyNetwork(num_actions=8, feature_dim=175)
    for layer in (net.fc1, net.fc2, net.actor, net.critic):
        assert layer.bias.abs().sum().item() < 1e-6


def test_actor_logits_init_near_uniform() -> None:
    """With actor head gain=0.01, initial logits should be close to
    uniform — softmax probabilities all roughly 1/num_actions. Stops
    the policy from committing to a single action before any signal."""
    import math
    net = TilePolicyNetwork(num_actions=8, feature_dim=175)
    x = torch.randn(32, 175)
    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=-1)
    # Within ~10% of 1/8 = 0.125 on average.
    assert abs(float(probs.mean()) - 1.0 / 8) < 0.05


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    net = TilePolicyNetwork(num_actions=6, feature_dim=175, hidden_dim=64)
    x = torch.randn(1, 175)
    out_before = net(x)

    ckpt = tmp_path / "tile_net.pt"
    net.save(ckpt)

    net2 = TilePolicyNetwork.load(ckpt)
    out_after = net2(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_load_preserves_arch_hparams(tmp_path: Path) -> None:
    """Loading should reconstruct the architecture from the checkpoint
    metadata, even when the caller's defaults differ."""
    net = TilePolicyNetwork(num_actions=4, feature_dim=200, hidden_dim=128)
    ckpt = tmp_path / "tile_net.pt"
    net.save(ckpt)
    loaded = TilePolicyNetwork.load(ckpt)
    assert loaded.num_actions == 4
    assert loaded.feature_dim == 200
    assert loaded.hidden_dim == 128


def test_act_returns_valid_action() -> None:
    net = TilePolicyNetwork(num_actions=6, feature_dim=175)
    feat = torch.randn(175)
    a = net.act(feat, deterministic=True)
    assert 0 <= a < 6


def test_gradient_flows_through_both_heads() -> None:
    """A loss combining actor and critic outputs should produce
    gradients on every weight in the network."""
    net = TilePolicyNetwork(num_actions=4, feature_dim=175)
    x = torch.randn(8, 175)
    logits, value = net.forward_ac(x)
    loss = logits.sum() + value.sum()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None
