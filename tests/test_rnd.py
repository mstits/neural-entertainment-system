"""RND tests — frozen target invariant, normalization stats math,
prediction-error gradient flow, and state_dict roundtrip."""

from __future__ import annotations

import torch

from src.models.rnd import RND, _RunningMeanStd


# ---------------------------------------------------------------------
# Running mean/std
# ---------------------------------------------------------------------


def test_running_mean_std_recovers_known_values() -> None:
    rms = _RunningMeanStd(shape=())
    xs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rms.update(xs)
    assert abs(float(rms.mean) - 3.0) < 1e-3
    # population std of [1..5] = sqrt(2.0) ≈ 1.4142
    assert abs(float(rms.std) - 1.4142135) < 1e-3


def test_running_mean_std_incremental_matches_batch() -> None:
    """Two-batch incremental update should match a single-batch
    update over the concatenated data — Welford correctness check."""
    a = torch.randn(50)
    b = torch.randn(50)
    rms_inc = _RunningMeanStd(shape=())
    rms_inc.update(a)
    rms_inc.update(b)
    rms_full = _RunningMeanStd(shape=())
    rms_full.update(torch.cat([a, b]))
    # Allow a little numeric slack for the count-1e-4 prior.
    assert abs(float(rms_inc.mean) - float(rms_full.mean)) < 1e-3
    assert abs(float(rms_inc.std) - float(rms_full.std)) < 1e-3


def test_running_mean_std_shaped() -> None:
    """Per-element stats for tensor shapes."""
    rms = _RunningMeanStd(shape=(2, 3))
    x = torch.randn(20, 2, 3)
    rms.update(x)
    assert rms.mean.shape == (2, 3)
    # First-element mean ≈ data mean for that element.
    assert abs(float(rms.mean[0, 0]) - float(x[:, 0, 0].mean())) < 1e-3


# ---------------------------------------------------------------------
# RND mechanics
# ---------------------------------------------------------------------


def test_target_is_frozen() -> None:
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    for p in rnd.target.parameters():
        assert not p.requires_grad


def test_target_eval_mode_persists() -> None:
    """Even when the outer module is set to train(), the target net
    must stay in eval mode (no buffer updates)."""
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    rnd.train(True)
    assert rnd.target.training is False


def test_per_sample_error_shape() -> None:
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    x = torch.rand(8, 4, 84, 84)
    err = rnd(x)
    assert err.shape == (8,)


def test_predictor_grads_flow_target_does_not() -> None:
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    x = torch.rand(4, 4, 84, 84)
    err = rnd(x)
    err.mean().backward()
    # Predictor params get gradients.
    for p in rnd.predictor.parameters():
        assert p.grad is not None
    # Target params have no grad.
    for p in rnd.target.parameters():
        assert p.grad is None


def test_normalization_stabilizes_error_scale() -> None:
    """After several update_normalization calls, the per-sample error
    scale should stay in a bounded range — without normalization it
    would either explode (initial network weights) or shrink toward
    zero as the predictor improves."""
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    errs = []
    for _ in range(8):
        x = torch.rand(4, 4, 84, 84)
        e = rnd(x)
        rnd.update_normalization(x, e.detach())
        errs.append(float(e.mean().detach()))
    # First call has uninitialized stats so the value is large; later
    # calls should be O(1). Ensure we don't drift outside a sane range.
    assert all(0.0 <= v < 100.0 for v in errs)


def test_state_dict_roundtrip() -> None:
    rnd = RND(in_channels=4, feat_dim=64, obs_size=84)
    x = torch.rand(4, 4, 84, 84)
    rnd.update_normalization(x, rnd(x).detach())
    state = rnd.state_dict()
    rnd2 = RND(in_channels=4, feat_dim=64, obs_size=84)
    rnd2.load_state_dict(state)
    # The running stats buffers should match.
    assert torch.allclose(rnd.obs_rms.mean, rnd2.obs_rms.mean)
    assert torch.allclose(rnd.reward_rms.mean, rnd2.reward_rms.mean)
