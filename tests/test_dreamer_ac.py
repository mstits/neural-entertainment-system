"""Dreamer actor + critic + lambda-returns tests."""

from __future__ import annotations

import torch

from src.models.dreamer_ac import DreamerActor, DreamerCritic, lambda_returns


# ---------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------


def test_actor_logits_shape() -> None:
    actor = DreamerActor(latent_dim=128, num_actions=8)
    x = torch.randn(4, 128)
    logits = actor(x)
    assert logits.shape == (4, 8)


def test_actor_sample_returns_one_hot() -> None:
    actor = DreamerActor(latent_dim=128, num_actions=8)
    x = torch.randn(4, 128)
    action, log_prob, entropy = actor.sample(x)
    assert action.shape == (4, 8)
    # Forward pass sees discrete one-hot (sum of any row ≈ 1 from the
    # one_hot + (probs - probs.detach()) trick — the residual is ~0
    # at forward time but provides gradient through `probs`).
    row_sums = action.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    # Entropy is non-negative for any categorical distribution.
    assert (entropy >= 0).all()
    # Upper bound is log(num_actions) for uniform.
    import math
    assert (entropy <= math.log(8) + 1e-5).all()


def test_actor_entropy_max_at_uniform() -> None:
    """A near-uniform logits vector should yield entropy close to
    log(num_actions); a peaked distribution should yield less."""
    import math
    actor = DreamerActor(latent_dim=4, num_actions=8)
    # Force the network to output near-uniform logits by zeroing the
    # final linear's bias and weight.
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
    _, _, entropy = actor.sample(torch.randn(2, 4))
    assert torch.allclose(entropy, torch.full_like(entropy, math.log(8)), atol=1e-4)


def test_actor_sample_gradient_flows() -> None:
    """Straight-through means gradients should reach the actor's params
    even though forward uses discrete bits."""
    actor = DreamerActor(latent_dim=128, num_actions=8)
    x = torch.randn(4, 128, requires_grad=False)
    action, log_prob, entropy = actor.sample(x)
    # Use all three return values so the test exercises every gradient
    # path through the straight-through sampler.
    (action.sum() + log_prob.sum() + entropy.sum()).backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in actor.parameters()
    )
    assert has_grad


# ---------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------


def test_critic_value_shape() -> None:
    critic = DreamerCritic(latent_dim=128)
    x = torch.randn(4, 128)
    v = critic(x)
    assert v.shape == (4,)


def test_target_network_starts_identical_to_online() -> None:
    critic = DreamerCritic(latent_dim=128)
    x = torch.randn(4, 128)
    online_v = critic(x)
    target_v = critic.target_value(x)
    assert torch.allclose(online_v, target_v)


def test_target_polyak_update_blends_correctly() -> None:
    critic = DreamerCritic(latent_dim=128)
    # Snapshot the initial weight, mutate the online net, then update.
    p_online = next(iter(critic.online.parameters()))
    p_target_before = next(iter(critic.target.parameters())).detach().clone()

    with torch.no_grad():
        p_online.add_(1.0)  # change the online by 1.0 everywhere
    expected_after = p_target_before * (1.0 - 0.1) + (p_target_before + 1.0) * 0.1

    critic.update_target(tau=0.1)
    p_target_after = next(iter(critic.target.parameters()))
    assert torch.allclose(p_target_after, expected_after, atol=1e-5)


def test_target_grad_disabled() -> None:
    critic = DreamerCritic(latent_dim=128)
    for p in critic.target.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------
# Lambda returns
# ---------------------------------------------------------------------


def test_lambda_returns_shape() -> None:
    rewards = torch.zeros(2, 5)
    values = torch.zeros(2, 5)
    bootstrap = torch.zeros(2)
    continues = torch.ones(2, 5)
    out = lambda_returns(rewards, values, bootstrap, continues)
    assert out.shape == (2, 5)


def test_lambda_returns_terminates_on_zero_continue() -> None:
    """When `continues[t] == 0`, the recurrence should collapse to
    just `rewards[t]` (no discounting beyond)."""
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.tensor([[10.0, 10.0, 10.0]])
    bootstrap = torch.tensor([10.0])
    continues = torch.tensor([[1.0, 0.0, 1.0]])
    out = lambda_returns(rewards, values, bootstrap, continues, discount=0.99, lambd=0.95)
    # At t=1, continue=0: G_1 should equal exactly rewards[1] = 1.0.
    assert abs(float(out[0, 1]) - 1.0) < 1e-5


def test_lambda_returns_matches_undiscounted_sum_on_simple_case() -> None:
    """With γ=1.0, λ=1.0, V=0, bootstrap=0, the λ-returns recurrence
    reduces to a plain reverse cumulative sum of rewards."""
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    values = torch.zeros(1, 4)
    bootstrap = torch.zeros(1)
    continues = torch.ones(1, 4)
    out = lambda_returns(rewards, values, bootstrap, continues, discount=1.0, lambd=1.0)
    # Reverse cumulative sums: [10, 9, 7, 4]
    expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_lambda_returns_decreases_along_time_when_undiscounted_positive() -> None:
    """With positive constant rewards and γ<1, the t=0 return should
    exceed the t=H-1 return (more reward ahead at t=0)."""
    rewards = torch.ones(1, 5)
    values = torch.zeros(1, 5)
    bootstrap = torch.zeros(1)
    continues = torch.ones(1, 5)
    out = lambda_returns(rewards, values, bootstrap, continues, discount=0.9, lambd=0.95)
    assert out[0, 0] > out[0, -1]
