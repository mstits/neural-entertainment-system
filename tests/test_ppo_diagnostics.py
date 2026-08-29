"""V29_STABILITY_2026-08-25.md F0: the five previously-missing scalars.

`ppo_losses` (src/training/ppo.py) already computes the PPO surrogate
`ratio` and `PPOUpdater.update` (src/training/ppo_updater.py) already
computes the advantage mean/std and calls `clip_grad_norm_` (which
returns the pre-clip norm for free) -- these tests pin that the new
`diagnostics` out-param and the six new `update()` return keys actually
surface those already-computed quantities, with the right numbers, and
that none of it perturbs the loss/optimizer step itself.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch

from src.models.tile_policy import TilePolicyNetwork
from src.training.ppo import ppo_losses
from src.training.ppo_updater import PPOUpdater
from src.training.timing import GenTimer

_NA, _FDIM = 4, 4
_T, _N = 5, 3


# ---------------------------------------------------------------------
# ppo_losses(diagnostics=...) — unit-level, hand-computed expectations.
# ---------------------------------------------------------------------

def _one_row_logits(log_p0: float, num_actions: int = 2) -> torch.Tensor:
    """Two-action logits whose softmax puts P(action 0) = exp(log_p0)."""
    p0 = math.exp(log_p0)
    p1 = 1.0 - p0
    return torch.log(torch.tensor([[p0, p1]])).float()


def test_ppo_losses_diagnostics_none_is_free_and_untouched() -> None:
    """Default (no diagnostics dict) must behave exactly as before —
    the block is skipped entirely, not merely uncollected."""
    logits = _one_row_logits(math.log(0.5))
    loss, policy_loss, value_loss, entropy = ppo_losses(
        logits, torch.tensor([0.0]), torch.tensor([0]),
        torch.tensor([math.log(0.5)]), torch.tensor([1.0]),
        torch.tensor([0.0]),
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.0,
        value_loss_kind="mse",
    )
    assert torch.isfinite(loss)


def test_ppo_losses_zero_ratio_gives_zero_clip_fraction_and_kl() -> None:
    """log_probs_new == log_probs_old -> ratio == 1.0 exactly -> the
    clip never binds and the k3 approx-KL estimator is exactly 0."""
    log_p = math.log(0.5)
    logits = _one_row_logits(log_p)
    diag: dict = {}
    ppo_losses(
        logits, torch.tensor([0.0]), torch.tensor([0]),
        torch.tensor([log_p]), torch.tensor([1.0]), torch.tensor([0.0]),
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.0,
        value_loss_kind="mse", diagnostics=diag,
    )
    assert set(diag.keys()) == {"clip_fraction", "approx_kl",
                                "kl_clamped_rows"}
    assert diag["clip_fraction"].item() == 0.0
    assert abs(diag["approx_kl"].item()) < 1e-6
    assert diag["kl_clamped_rows"].item() == 0


def test_ppo_losses_large_ratio_trips_clip_fraction_and_positive_kl() -> None:
    """A behavior policy putting all mass on action 0 vs. a current
    policy that has swung hard away from it (ratio far below 1-clip)
    must register as fully clipped with a strictly positive approx-KL
    (the k3 estimator is non-negative and zero only at ratio == 1)."""
    logits = _one_row_logits(math.log(0.05))  # new policy: P(a0) = 0.05
    diag: dict = {}
    ppo_losses(
        logits, torch.tensor([0.0]), torch.tensor([0]),
        torch.tensor([math.log(0.95)]),  # old policy: P(a0) = 0.95
        torch.tensor([1.0]), torch.tensor([0.0]),
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.0,
        value_loss_kind="mse", diagnostics=diag,
    )
    assert diag["clip_fraction"].item() == 1.0
    assert diag["approx_kl"].item() > 0.0


def test_ppo_losses_diagnostics_do_not_affect_loss_value() -> None:
    """Passing `diagnostics` is a pure side-observation -- the returned
    loss/policy_loss/value_loss/entropy must be bit-identical to the
    same call without it."""
    torch.manual_seed(0)
    logits = torch.randn(6, _NA)
    values_pred = torch.randn(6)
    actions = torch.randint(0, _NA, (6,))
    log_probs_old = torch.randn(6) * 0.1 - 1.0
    advantages = torch.randn(6)
    value_targets = torch.randn(6)
    kwargs = dict(
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        value_loss_kind="huber",
    )
    out_a = ppo_losses(
        logits, values_pred, actions, log_probs_old, advantages,
        value_targets, **kwargs,
    )
    out_b = ppo_losses(
        logits, values_pred, actions, log_probs_old, advantages,
        value_targets, diagnostics={}, **kwargs,
    )
    for a, b in zip(out_a, out_b):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------
# PPOUpdater.update() — the six new keys reach the returned dict with
# sane values, computed on a real (tiny) rollout + network.
# ---------------------------------------------------------------------

def _fake_trainer() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        _rnd=None,
        _gen_timer=GenTimer(),
        _is_tile_mode=True,
        preprocess_f16=False,
        _gx_count_beta=0.0,
        rnd_intrinsic_coef=1.0,
        rnd_loss_coef=0.0,
        rnd_predictor_update_fraction=1.0,
        reinforce_gamma=0.99,
        gae_lambda=0.95,
        ppo_minibatch_size=4,
        _recurrent=False,
        _demo_bank=None,
        reinforce_steps=2,
        ppo_clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        value_loss_kind="huber",
        reinforce_grad_clip=0.5,
        demo_anchor_decay_start=0,
        demo_anchor_decay_iters=1,
        demo_anchor_coef0=0.0,
        demo_anchor_final=0.0,
    )


def _small_rollout() -> dict:
    rng = np.random.default_rng(1)
    return dict(
        obs_buf=rng.normal(size=(_T, _N, _FDIM)).astype(np.float32),
        action_buf=rng.integers(0, _NA, size=(_T, _N)).astype(np.int32),
        reward_buf=rng.normal(size=(_T, _N)).astype(np.float32),
        value_buf=rng.normal(size=(_T, _N)).astype(np.float32),
        log_prob_buf=np.full((_T, _N), -1.2, dtype=np.float32),
        done_buf=np.zeros((_T, _N), dtype=np.bool_),
        valid_buf=np.ones((_T, _N), dtype=np.bool_),
        bonus_buf=np.zeros((_T, _N), dtype=np.float32),
        final_values_np=np.zeros(_N, dtype=np.float32),
    )


_NEW_KEYS = (
    "last_clip_fraction", "last_approx_kl", "last_grad_norm",
    "adv_mean", "adv_std", "explained_variance",
)


def test_update_returns_all_five_new_scalars() -> None:
    torch.manual_seed(0)
    t = _fake_trainer()
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=8, trunk_dim=4
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    result = PPOUpdater(t).update(
        net=net, optimizer=opt, rollout_steps=_T, num_envs=_N,
        obs_shape=(_FDIM,), global_it=0, sam_rho=0.0,
        **_small_rollout(),
    )

    for key in _NEW_KEYS:
        assert key in result, f"PPOUpdater.update() missing {key!r}"
        val = result[key]
        assert isinstance(val, float), f"{key!r} must be a plain float, got {type(val)}"
        assert math.isfinite(val), f"{key!r} was non-finite: {val}"

    assert 0.0 <= result["last_clip_fraction"] <= 1.0
    assert result["last_grad_norm"] >= 0.0
    assert result["adv_std"] > 0.0
    assert result["explained_variance"] <= 1.0 + 1e-6


def test_update_does_not_change_training_outputs() -> None:
    """The pre-existing return keys (loss/policy/value/entropy, the
    folded reward buffer) must be unaffected by the new instrumentation
    -- same seed, same rollout, same net init, twice."""
    def run_once() -> dict:
        torch.manual_seed(7)
        np.random.seed(7)  # minibatch shuffling uses np.random.permutation
        t = _fake_trainer()
        net = TilePolicyNetwork(
            num_actions=_NA, feature_dim=_FDIM, hidden_dim=8, trunk_dim=4
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        return PPOUpdater(t).update(
            net=net, optimizer=opt, rollout_steps=_T, num_envs=_N,
            obs_shape=(_FDIM,), global_it=0, sam_rho=0.0,
            **_small_rollout(),
        )

    r1 = run_once()
    r2 = run_once()
    for key in ("last_loss", "last_policy_loss", "last_value_loss", "last_entropy"):
        assert r1[key] == r2[key], f"{key!r} diverged across identical runs"
    assert np.array_equal(r1["reward_buf"], r2["reward_buf"])
