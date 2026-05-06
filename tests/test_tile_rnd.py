"""Tests for the tile-mode RND module."""

from __future__ import annotations

import torch

from src.models.tile_rnd import TileRND, _TileRNDEncoder


def test_encoder_output_shape() -> None:
    enc = _TileRNDEncoder(in_features=175, feat_dim=64)
    x = torch.randn(8, 175)
    out = enc(x)
    assert out.shape == (8, 64)


def test_per_sample_error_shape() -> None:
    rnd = TileRND(feature_dim=175)
    x = torch.randn(8, 175)
    err = rnd(x)
    assert err.shape == (8,)


def test_target_is_frozen() -> None:
    rnd = TileRND(feature_dim=175)
    for p in rnd.target.parameters():
        assert not p.requires_grad


def test_target_eval_mode_persists() -> None:
    """Even when the outer module is set to train(), the target stays
    in eval — RND requires a fixed reference."""
    rnd = TileRND(feature_dim=175)
    rnd.train(True)
    assert rnd.target.training is False
    assert rnd.predictor.training is True


def test_predictor_grads_flow_target_does_not() -> None:
    rnd = TileRND(feature_dim=175)
    x = torch.randn(4, 175)
    err = rnd(x)
    err.mean().backward()
    for p in rnd.predictor.parameters():
        assert p.grad is not None
    for p in rnd.target.parameters():
        assert p.grad is None


def test_normalization_stabilizes_error_scale() -> None:
    rnd = TileRND(feature_dim=175)
    errs = []
    for _ in range(8):
        x = torch.randn(4, 175)
        e = rnd(x)
        rnd.update_normalization(x, e.detach())
        errs.append(float(e.mean().detach()))
    # First call has uninitialized stats so the value is large; subsequent
    # calls should converge to O(1)-scale once running stats catch up.
    assert all(0.0 <= v < 100.0 for v in errs)


def test_state_dict_roundtrip() -> None:
    rnd = TileRND(feature_dim=175)
    x = torch.randn(4, 175)
    rnd.update_normalization(x, rnd(x).detach())
    state = rnd.state_dict()
    rnd2 = TileRND(feature_dim=175)
    rnd2.load_state_dict(state)
    # Running-stats buffers must match.
    assert torch.allclose(rnd.obs_rms.mean, rnd2.obs_rms.mean)
    assert torch.allclose(rnd.reward_rms.mean, rnd2.reward_rms.mean)


def test_num_params_in_expected_range() -> None:
    """Sanity guard on encoder size — ~30k predictor params at default
    widths. Helps catch accidental architecture changes."""
    rnd = TileRND(feature_dim=175)
    assert 10_000 < rnd.num_params < 100_000


def test_handles_int8_input() -> None:
    """The trainer feeds tile features as int8; TileRND must accept
    them after a `.float()` cast (as the trainer does for PolicyNetwork
    forward)."""
    rnd = TileRND(feature_dim=175)
    x = torch.randint(-128, 127, (4, 175), dtype=torch.int8).float()
    err = rnd(x)
    assert err.shape == (4,)
    assert torch.isfinite(err).all()
