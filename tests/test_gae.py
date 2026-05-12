"""Tests for the GAE-λ recurrence.

Two branches matter for correctness:
  1. Natural termination (traj_len_full < max_episode_steps) — final
     V(s_{T+1}) must be 0 so the death/clear signal isn't muted by
     the critic bootstrap.
  2. Truncation (traj_len_full == max_episode_steps) — bootstrap
     with V(s_T) so the value estimate carries through the cutoff.

The 2026-04-21 PPO/GAE fix found that the prior unconditional
bootstrap muted death signals ~100× at γ=0.99. Lock both branches
down.
"""

from __future__ import annotations

import numpy as np
import torch

from src.training.gae import gae


def test_gae_natural_termination_zero_bootstrap() -> None:
    """When the trajectory terminated naturally, the last step's
    advantage must equal r_T - V(s_T) (no future-value bootstrap)."""
    rewards = torch.tensor([0.0, 0.0, -10.0])  # the death step
    values = torch.tensor([5.0, 5.0, 5.0])
    advantages, value_targets = gae(
        rewards=rewards,
        values_pred=values,
        traj_len_full=3,                 # ended at step 3
        max_episode_steps=100,           # well below the cap
        gamma=0.99,
        gae_lambda=0.95,
        normalize=False,
    )
    # Last advantage with V_next=0: delta_T = -10 + 0.99*0 - 5 = -15.0,
    # and since it's the last step the GAE running advantage = delta_T.
    assert advantages[-1].item() == -15.0
    # value_target_T = advantages_T + V(s_T) = -15 + 5 = -10.
    assert value_targets[-1].item() == -10.0


def test_gae_truncation_bootstraps_with_v_T() -> None:
    """When the trajectory was truncated at max_episode_steps, the
    final V(s_{T+1}) is V(s_T) so the value estimate carries through."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([5.0, 5.0, 5.0])
    advantages, _ = gae(
        rewards=rewards,
        values_pred=values,
        traj_len_full=3,
        max_episode_steps=3,             # truncated at the cap
        gamma=0.99,
        gae_lambda=0.95,
        normalize=False,
    )
    # Last advantage with V_next=V(s_T)=5: delta_T = 1 + 0.99*5 - 5 = 0.95.
    np.testing.assert_allclose(advantages[-1].item(), 0.95, rtol=1e-5)


def test_gae_recurrence_matches_hand_computation() -> None:
    """Three-step hand-rolled GAE to pin the recurrence direction
    (backward sweep, λ-weighted decay of future deltas)."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    gamma, lam = 0.9, 0.5
    advantages, _ = gae(
        rewards=rewards,
        values_pred=values,
        traj_len_full=3,
        max_episode_steps=10,            # natural termination → V_next[-1]=0
        gamma=gamma,
        gae_lambda=lam,
        normalize=False,
    )
    # Hand roll:
    #   delta_0 = 1 + 0.9*0.5 - 0.5 = 0.95
    #   delta_1 = 2 + 0.9*0.5 - 0.5 = 1.95
    #   delta_2 = 3 + 0.9*0    - 0.5 = 2.5     (V_next=0 since natural)
    # Backward sweep:
    #   adv_2 = 2.5
    #   adv_1 = 1.95 + 0.9*0.5*2.5  = 1.95 + 1.125 = 3.075
    #   adv_0 = 0.95 + 0.9*0.5*3.075 = 0.95 + 1.38375 = 2.33375
    expected = torch.tensor([2.33375, 3.075, 2.5])
    torch.testing.assert_close(advantages, expected, rtol=1e-5, atol=1e-5)


def test_gae_normalize_flag_zero_means_unity_var() -> None:
    """normalize=True must produce ~zero-mean / ~unit-std advantages."""
    rewards = torch.linspace(-2.0, 2.0, 16)
    values = torch.zeros(16)
    advantages, _ = gae(
        rewards=rewards,
        values_pred=values,
        traj_len_full=16,
        max_episode_steps=100,
        gamma=0.99,
        normalize=True,
    )
    assert abs(advantages.mean().item()) < 1e-5
    # Allow some slack — small sample, but should be close to 1.
    np.testing.assert_allclose(advantages.std().item(), 1.0, rtol=1e-4)


def test_gae_value_targets_equal_advantages_plus_values() -> None:
    """value_targets = advantages + V(s_T) before normalization (the
    critic learns toward the discounted-return estimate, not the
    normalized advantage)."""
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    values = torch.tensor([1.0, 1.0, 1.0, 1.0])
    advantages_unnorm, value_targets = gae(
        rewards=rewards,
        values_pred=values,
        traj_len_full=4,
        max_episode_steps=100,
        gamma=0.99,
        normalize=False,
    )
    torch.testing.assert_close(
        value_targets, advantages_unnorm + values, rtol=1e-5, atol=1e-5
    )
