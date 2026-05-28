"""Unit tests for the pure vanilla_ppo building blocks in src.training.ppo.

These were extracted verbatim from the ~600-line `_run_vanilla_ppo`
method; the tests pin the exact semantics so the extraction can't
silently change training behavior, and so the Phase 1 RND wiring has
a regression net under the loss assembly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.training.ppo import (
    batched_gae,
    fold_intrinsic_into_rewards,
    ppo_losses,
)


def _reference_gae(reward_buf, value_buf, done_buf, final_values, gamma, lam):
    """Independent textbook implementation for cross-checking."""
    T, N = reward_buf.shape
    adv = np.zeros((T, N), dtype=np.float32)
    running = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = final_values if t == T - 1 else value_buf[t + 1]
        mask = (~done_buf[t]).astype(np.float32)
        delta = reward_buf[t] + gamma * next_v * mask - value_buf[t]
        running = delta + gamma * lam * running * mask
        adv[t] = running
    return adv, adv + value_buf


def test_batched_gae_matches_reference() -> None:
    rng = np.random.default_rng(0)
    T, N = 12, 5
    reward_buf = rng.standard_normal((T, N)).astype(np.float32)
    value_buf = rng.standard_normal((T, N)).astype(np.float32)
    done_buf = rng.random((T, N)) < 0.15
    final_values = rng.standard_normal(N).astype(np.float32)
    gamma, lam = 0.99, 0.95

    adv, vt = batched_gae(reward_buf, value_buf, done_buf, final_values, gamma, lam)
    ref_adv, ref_vt = _reference_gae(
        reward_buf, value_buf, done_buf, final_values, gamma, lam
    )
    np.testing.assert_allclose(adv, ref_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vt, ref_vt, rtol=1e-5, atol=1e-5)


def test_batched_gae_value_targets_identity() -> None:
    # value_targets must equal advantages + value_buf exactly (the
    # critic regression target lives on the reward scale, pre-norm).
    T, N = 6, 3
    reward_buf = np.ones((T, N), dtype=np.float32)
    value_buf = np.full((T, N), 0.5, dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=bool)
    final_values = np.zeros(N, dtype=np.float32)
    adv, vt = batched_gae(reward_buf, value_buf, done_buf, final_values, 0.99, 0.95)
    np.testing.assert_allclose(vt, adv + value_buf)


def test_batched_gae_done_breaks_boundary() -> None:
    # A done at step t must zero the GAE recurrence carried back from
    # t+1 — the advantage at t must not see any reward after the done.
    T, N = 4, 1
    reward_buf = np.array([[0.0], [0.0], [0.0], [100.0]], dtype=np.float32)
    value_buf = np.zeros((T, N), dtype=np.float32)
    final_values = np.zeros(N, dtype=np.float32)

    # No done: the big terminal reward propagates back to step 0.
    no_done = np.zeros((T, N), dtype=bool)
    adv_open, _ = batched_gae(reward_buf, value_buf, no_done, final_values, 0.99, 0.95)
    assert adv_open[0, 0] > 1.0  # reward leaked all the way back

    # Done at step 2: step 0/1 advantages must be exactly 0 (no reward
    # before the boundary, and the post-done reward can't cross it).
    done = np.zeros((T, N), dtype=bool)
    done[2, 0] = True
    adv_cut, _ = batched_gae(reward_buf, value_buf, done, final_values, 0.99, 0.95)
    assert adv_cut[0, 0] == 0.0
    assert adv_cut[1, 0] == 0.0
    assert adv_cut[2, 0] == 0.0  # reward at t=3 is after the t=2 boundary


def test_fold_intrinsic_adds_on_live_steps() -> None:
    reward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    intrinsic = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    done = np.zeros((2, 2), dtype=bool)
    out = fold_intrinsic_into_rewards(reward, intrinsic, done)
    np.testing.assert_allclose(out, reward + 0.5)
    # Must be a new array, not a mutation of reward_buf.
    assert out is not reward
    np.testing.assert_allclose(reward, [[1.0, 2.0], [3.0, 4.0]])


def test_fold_intrinsic_zeroed_on_done_steps() -> None:
    reward = np.zeros((2, 2), dtype=np.float32)
    intrinsic = np.ones((2, 2), dtype=np.float32)
    done = np.array([[False, True], [True, False]], dtype=bool)
    out = fold_intrinsic_into_rewards(reward, intrinsic, done)
    # done steps get no intrinsic; live steps get it.
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.0, 1.0]])


def _toy_inputs(batch=8, num_actions=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, num_actions, generator=g)
    values_pred = torch.randn(batch, generator=g)
    actions = torch.randint(0, num_actions, (batch,), generator=g)
    log_probs_old = torch.randn(batch, generator=g)
    advantages = torch.randn(batch, generator=g)
    value_targets = torch.randn(batch, generator=g)
    return logits, values_pred, actions, log_probs_old, advantages, value_targets


def test_ppo_losses_returns_scalars_with_grad() -> None:
    logits, values_pred, actions, lpo, adv, vt = _toy_inputs()
    logits = logits.requires_grad_(True)
    values_pred = values_pred.requires_grad_(True)
    loss, pl, vl, ent = ppo_losses(
        logits, values_pred, actions, lpo, adv, vt,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, value_loss_kind="huber",
    )
    for t in (loss, pl, vl, ent):
        assert t.ndim == 0
    loss.backward()  # must be differentiable end-to-end
    assert logits.grad is not None


def test_ppo_losses_entropy_of_uniform_logits() -> None:
    # Uniform logits → entropy = log(num_actions).
    num_actions = 5
    logits = torch.zeros(3, num_actions)
    values_pred = torch.zeros(3)
    actions = torch.zeros(3, dtype=torch.int64)
    zeros = torch.zeros(3)
    _, _, _, ent = ppo_losses(
        logits, values_pred, actions, zeros, zeros, zeros,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, value_loss_kind="huber",
    )
    assert math.isclose(float(ent), math.log(num_actions), rel_tol=1e-5)


def test_ppo_losses_policy_loss_when_ratio_unity() -> None:
    # When new log-probs == old (ratio == 1), the clipped surrogate is
    # just advantages, so policy_loss == -mean(advantages).
    num_actions = 3
    logits = torch.full((4, num_actions), -math.log(num_actions))  # log(1/3) each
    # log_prob of any action under uniform = log(1/3); set old to match.
    log_probs_old = torch.full((4,), math.log(1.0 / num_actions))
    actions = torch.tensor([0, 1, 2, 0])
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0])
    values_pred = torch.zeros(4)
    value_targets = torch.zeros(4)
    _, policy_loss, _, _ = ppo_losses(
        logits, values_pred, actions, log_probs_old, advantages, value_targets,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.0, value_loss_kind="huber",
    )
    assert math.isclose(float(policy_loss), -float(advantages.mean()), rel_tol=1e-5)


def test_ppo_losses_mse_vs_huber_differ_on_large_residual() -> None:
    # Large value residual: MSE penalizes quadratically, Huber linearly.
    logits = torch.zeros(2, 2)
    actions = torch.zeros(2, dtype=torch.int64)
    zeros = torch.zeros(2)
    values_pred = torch.zeros(2)
    value_targets = torch.full((2,), 10.0)  # residual of 10
    _, _, vl_mse, _ = ppo_losses(
        logits, values_pred, actions, zeros, zeros, value_targets,
        clip_eps=0.2, value_coef=1.0, entropy_coef=0.0, value_loss_kind="mse",
    )
    _, _, vl_huber, _ = ppo_losses(
        logits, values_pred, actions, zeros, zeros, value_targets,
        clip_eps=0.2, value_coef=1.0, entropy_coef=0.0, value_loss_kind="huber",
    )
    assert float(vl_mse) > float(vl_huber)
    assert math.isclose(float(vl_mse), 100.0, rel_tol=1e-5)      # 10^2
    assert math.isclose(float(vl_huber), 9.5, rel_tol=1e-5)      # |10| - 0.5
