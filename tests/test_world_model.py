"""DreamerV3 world-model tests — shape invariants, KL math, latent
kind switching, and full training-step loss propagation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.world_model import (
    RSSM,
    ObservationDecoder,
    ObservationEncoder,
    RSSMState,
    WorldModel,
    WorldModelConfig,
)


# ---------------------------------------------------------------------
# Encoder / Decoder shapes
# ---------------------------------------------------------------------


def test_encoder_output_shape() -> None:
    cfg = WorldModelConfig(num_actions=4)
    enc = ObservationEncoder(cfg)
    x = torch.zeros(3, cfg.obs_channels, cfg.obs_size, cfg.obs_size)
    out = enc(x)
    assert out.shape == (3, cfg.encoder_feat_dim)


def test_decoder_output_shape_matches_obs() -> None:
    cfg = WorldModelConfig(num_actions=4)
    dec = ObservationDecoder(cfg)
    latent = torch.zeros(2, cfg.deter_dim + cfg.stoch_flat)
    out = dec(latent)
    # Round-trip back to the original obs spatial shape — decoder
    # convolution math is fragile so this is the canary test.
    assert out.shape == (2, cfg.obs_channels, cfg.obs_size, cfg.obs_size)
    # Sigmoid output → values in [0, 1].
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------
# Latent kind switching
# ---------------------------------------------------------------------


def test_categorical_latent_dim() -> None:
    cfg = WorldModelConfig(latent_kind="categorical", stoch_dim=32, stoch_classes=32)
    assert cfg.stoch_flat == 32 * 32
    rssm = RSSM(cfg)
    s = rssm.initial_state(2, torch.device("cpu"))
    assert s.stoch.shape == (2, 32 * 32)


def test_gaussian_latent_dim() -> None:
    cfg = WorldModelConfig(latent_kind="gaussian", stoch_dim=32)
    assert cfg.stoch_flat == 32
    rssm = RSSM(cfg)
    s = rssm.initial_state(2, torch.device("cpu"))
    assert s.stoch.shape == (2, 32)


def test_unknown_latent_kind_raises() -> None:
    import pytest
    cfg = WorldModelConfig(latent_kind="garbage")
    # Construction picks up the bad name when _StochHead is built.
    with pytest.raises(ValueError):
        RSSM(cfg)


# ---------------------------------------------------------------------
# RSSM observe / imagine
# ---------------------------------------------------------------------


def _seq_inputs(B: int = 2, L: int = 4, cfg: WorldModelConfig | None = None):
    cfg = cfg or WorldModelConfig(num_actions=4)
    embed = torch.randn(B, L, cfg.encoder_feat_dim)
    actions = F.one_hot(
        torch.randint(0, cfg.num_actions, (B, L)),
        num_classes=cfg.num_actions,
    ).float()
    return cfg, embed, actions


def test_observe_returns_one_state_per_step() -> None:
    cfg, embed, actions = _seq_inputs(B=3, L=5)
    rssm = RSSM(cfg)
    states = rssm.observe(embed, actions)
    assert len(states) == 5
    for s in states:
        assert s.deter.shape == (3, cfg.deter_dim)
        assert s.stoch.shape == (3, cfg.stoch_flat)
        # observe() populates BOTH posterior and prior — KL needs both.
        assert s.post_params is not None
        assert s.prior_params is not None


def test_imagine_returns_horizon_states() -> None:
    cfg = WorldModelConfig(num_actions=4)
    rssm = RSSM(cfg)
    init = rssm.initial_state(2, torch.device("cpu"))
    actions = F.one_hot(
        torch.randint(0, cfg.num_actions, (2, 6)), num_classes=cfg.num_actions
    ).float()
    states = rssm.imagine(init, actions)
    assert len(states) == 6
    for s in states:
        # Imagine has prior only, no posterior (no observations to condition on).
        assert s.prior_params is not None
        assert s.post_params is None


# ---------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------


def test_kl_loss_nonnegative_categorical() -> None:
    cfg = WorldModelConfig(latent_kind="categorical")
    cfg, embed, actions = _seq_inputs(B=2, L=4, cfg=cfg)
    rssm = RSSM(cfg)
    states = rssm.observe(embed, actions)
    # Free-nats=0 lets us see the raw KL; clamp(min=0) - 0 → still >= 0.
    kl = rssm.kl_loss(states, free_nats=0.0)
    assert kl.item() >= 0.0


def test_kl_loss_zero_when_post_equals_prior() -> None:
    """With identical post and prior params, KL should be exactly zero."""
    cfg = WorldModelConfig(latent_kind="categorical")
    rssm = RSSM(cfg)
    fake_params = torch.randn(2, cfg.stoch_dim * cfg.stoch_classes)
    state = RSSMState(
        deter=torch.zeros(2, cfg.deter_dim),
        stoch=torch.zeros(2, cfg.stoch_flat),
        prior_params=fake_params.clone(),
        post_params=fake_params.clone(),
    )
    kl = rssm.kl_loss([state], free_nats=0.0)
    assert kl.item() < 1e-5


def test_kl_loss_gaussian_path_runs() -> None:
    cfg = WorldModelConfig(latent_kind="gaussian")
    cfg, embed, actions = _seq_inputs(B=2, L=3, cfg=cfg)
    rssm = RSSM(cfg)
    states = rssm.observe(embed, actions)
    kl = rssm.kl_loss(states, free_nats=0.0)
    assert kl.item() >= 0.0


# ---------------------------------------------------------------------
# Full training-step gradient flow
# ---------------------------------------------------------------------


def test_world_model_loss_backprops() -> None:
    cfg = WorldModelConfig(num_actions=4, latent_kind="categorical")
    wm = WorldModel(cfg)
    B, L = 2, 5
    obs = torch.rand(B, L, cfg.obs_channels, cfg.obs_size, cfg.obs_size)
    actions = F.one_hot(
        torch.randint(0, cfg.num_actions, (B, L)), num_classes=cfg.num_actions
    ).float()
    rewards = torch.randn(B, L)
    continues = torch.ones(B, L)
    loss, stats = wm.loss(obs, actions, rewards, continues)
    # Backward must not throw — verifies the entire computation graph
    # is valid for both categorical sampling and KL paths.
    loss.backward()
    # Check that at least one parameter received a gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in wm.parameters()
    )
    assert has_grad
    # Loss components surface correctly in stats.
    for key in ("wm_recon", "wm_reward", "wm_continue", "wm_kl"):
        assert key in stats


def test_world_model_loss_gaussian_backprops() -> None:
    cfg = WorldModelConfig(num_actions=4, latent_kind="gaussian")
    wm = WorldModel(cfg)
    obs = torch.rand(2, 3, cfg.obs_channels, cfg.obs_size, cfg.obs_size)
    actions = F.one_hot(
        torch.randint(0, cfg.num_actions, (2, 3)), num_classes=cfg.num_actions
    ).float()
    rewards = torch.randn(2, 3)
    continues = torch.ones(2, 3)
    loss, _ = wm.loss(obs, actions, rewards, continues)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in wm.parameters()
    )
    assert has_grad
