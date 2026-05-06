"""Tests for the recurrent tile policy network. Class-only — trainer
integration (PPO replay through GRU) is a separate task documented
in TaskList #36."""

from __future__ import annotations

import torch

from src.models.tile_policy import TileRecurrentPolicyNetwork


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
