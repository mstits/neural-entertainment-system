"""Unit guards for the ReDo mechanism (src/training/redo.py).

Registered: docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md AMENDMENT 1
(B3 mechanism, B4 knobs, B6 armed-evidence lines). Pins:

  1. Dormancy detection is Sokar et al. Definition 1 — mean |post-
     activation| normalized by the LAYER MEAN, dormant iff s_i <= tau
     (inclusive), all-silent layer counts as fully dormant.
  2. Recycle touches exactly the registered slices: incoming row + bias
     re-sampled, LayerNorm affine reset to 1/0, outgoing columns
     exactly 0.0 (actor AND critic for fc2 units), everything else
     bit-untouched — including the intersection entry when an fc1 unit
     and an fc2 unit are recycled together.
  3. Adam moments are zeroed for exactly the touched slices; per-tensor
     `step` counters survive; `reset_optimizer_moments=False` leaves
     the moments alone.
  4. OFF is byte-identical: the disabled gate consumes no numpy/torch
     RNG and mutates nothing, and the trainer-side hook is guarded so a
     `redo_enabled: false` run cannot drift (source pins follow the
     `test_trainer_mechanism_guards.py` precedent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.tile_policy import TilePolicyNetwork
from src.training.redo import (
    check_and_recycle,
    dormant_indices,
    hidden_activations,
    maybe_check_and_recycle,
    recycle,
)

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_PATH = ROOT / "src" / "training" / "trainer.py"

_NUM_ACTIONS = 6
_FEATURE_DIM = 12
_HIDDEN = 8
_TRUNK = 5


def _small_net(seed: int = 0) -> TilePolicyNetwork:
    torch.manual_seed(seed)
    return TilePolicyNetwork(
        num_actions=_NUM_ACTIONS, feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN, trunk_dim=_TRUNK,
    )


def _clone_state(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


# ---------------------------------------------------------------------------
# 1. Dormancy detection (Definition 1: layer-MEAN normalization)
# ---------------------------------------------------------------------------

def test_dormancy_layer_mean_normalization_and_threshold():
    # Column mean |h|: [2.0, 2.0, 0.05, 0.0] -> layer mean 1.0125
    # -> scores [1.9753, 1.9753, 0.0494, 0.0].
    h = torch.zeros(4, 4)
    h[:, 0] = 2.0
    h[:, 1] = torch.tensor([2.0, -2.0, 2.0, -2.0])  # abs() must be used
    h[:, 2] = 0.05
    h[:, 3] = 0.0
    assert dormant_indices(h, tau=0.025).tolist() == [3]
    # tau is inclusive and calibrated to the MEAN normalizer: at 0.05
    # unit 2 (score ~0.0494 <= 0.05) joins unit 3. Under the DR's
    # (uncorrected) max-normalization its score would be 0.025 and the
    # 0.025 threshold would already grab it — this pins the registered
    # correction.
    assert dormant_indices(h, tau=0.05).tolist() == [2, 3]
    assert dormant_indices(h, tau=0.025).dtype == torch.int64


def test_dormancy_threshold_is_inclusive():
    # Scores are exactly [1.5, 0.5]: tau=0.5 must include unit 1.
    h = torch.zeros(2, 2)
    h[:, 0] = 3.0
    h[:, 1] = 1.0
    assert dormant_indices(h, tau=0.5).tolist() == [1]
    assert dormant_indices(h, tau=0.49).tolist() == []


def test_dormancy_all_silent_layer_is_fully_dormant():
    h = torch.zeros(16, 5)
    assert dormant_indices(h, tau=0.025).tolist() == [0, 1, 2, 3, 4]


def test_dormancy_rejects_non_2d():
    with pytest.raises(ValueError):
        dormant_indices(torch.zeros(3, 4, 5), tau=0.025)


def test_hidden_activations_match_forward_ac():
    net = _small_net()
    x = torch.randn(32, _FEATURE_DIM)
    h1, h2, logits = hidden_activations(net, x)
    ref_logits, ref_value = net.forward_ac(x)
    assert torch.equal(logits, ref_logits)
    assert h1.shape == (32, _HIDDEN) and h2.shape == (32, _TRUNK)
    # SiLU output floor: silu(x) >= silu(-1.278...) ~ -0.2785
    assert float(h1.min()) >= -0.279 and float(h2.min()) >= -0.279


# ---------------------------------------------------------------------------
# 2. Recycle resets exactly the registered slices
# ---------------------------------------------------------------------------

def _paint_distinctive(net) -> None:
    """Give every parameter a value a reset cannot coincide with."""
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(7.0)


def test_recycle_resets_right_slices_and_nothing_else():
    net = _small_net()
    _paint_distinctive(net)
    before = _clone_state(net)
    fc1_idx = torch.tensor([2, 5])
    fc2_idx = torch.tensor([3])
    torch.manual_seed(123)
    recycle(net, None, fc1_idx, fc2_idx, reset_optimizer_moments=False)
    net.requires_grad_(False)  # scalar reads below without autograd warnings

    fc1_bound = 1.0 / np.sqrt(_FEATURE_DIM)
    fc2_bound = 1.0 / np.sqrt(_HIDDEN)

    # fc1 dormant rows: re-sampled Kaiming-uniform at fan_in=12.
    for i in (2, 5):
        row = net.fc1.weight[i].detach()
        assert not torch.equal(row, before["fc1.weight"][i])
        assert float(row.abs().max()) <= fc1_bound
        assert abs(float(net.fc1.bias[i].detach())) <= fc1_bound
        assert float(net.norm1.weight[i]) == 1.0
        assert float(net.norm1.bias[i]) == 0.0
        # outgoing column in fc2 exactly 0.0 — including the row that
        # was itself re-initialized this event (fc2.weight[3, i]).
        assert torch.equal(net.fc2.weight[:, i],
                           torch.zeros(_TRUNK))
    # fc2 dormant row 3: re-sampled at fan_in=8, except the zeroed
    # dormant-fc1 columns.
    keep_cols = [c for c in range(_HIDDEN) if c not in (2, 5)]
    assert float(net.fc2.weight[3, keep_cols].detach().abs().max()) <= fc2_bound
    assert not torch.equal(net.fc2.weight[3], before["fc2.weight"][3])
    assert abs(float(net.fc2.bias[3])) <= fc2_bound
    assert float(net.norm2.weight[3]) == 1.0
    assert float(net.norm2.bias[3]) == 0.0
    # outgoing columns of the fc2 unit: actor AND critic, exactly 0.0.
    assert torch.equal(net.actor.weight[:, 3], torch.zeros(_NUM_ACTIONS))
    assert torch.equal(net.critic.weight[:, 3], torch.zeros(1))

    # Everything else is bit-untouched (still the painted 7.0).
    untouched_fc1_rows = [i for i in range(_HIDDEN) if i not in (2, 5)]
    assert torch.equal(net.fc1.weight[untouched_fc1_rows],
                       before["fc1.weight"][untouched_fc1_rows])
    assert torch.equal(net.fc1.bias[untouched_fc1_rows],
                       before["fc1.bias"][untouched_fc1_rows])
    assert torch.equal(net.norm1.weight[untouched_fc1_rows],
                       before["norm1.weight"][untouched_fc1_rows])
    for r in range(_TRUNK):
        if r == 3:
            continue
        assert torch.equal(net.fc2.weight[r, keep_cols],
                           before["fc2.weight"][r, keep_cols])
    untouched_fc2 = [r for r in range(_TRUNK) if r != 3]
    assert torch.equal(net.fc2.bias[untouched_fc2],
                       before["fc2.bias"][untouched_fc2])
    assert torch.equal(net.norm2.weight[untouched_fc2],
                       before["norm2.weight"][untouched_fc2])
    keep_trunk = [c for c in range(_TRUNK) if c != 3]
    assert torch.equal(net.actor.weight[:, keep_trunk],
                       before["actor.weight"][:, keep_trunk])
    assert torch.equal(net.critic.weight[:, keep_trunk],
                       before["critic.weight"][:, keep_trunk])
    # Head rows / biases are never recycled.
    assert torch.equal(net.actor.bias, before["actor.bias"])
    assert torch.equal(net.critic.bias, before["critic.bias"])


def test_check_and_recycle_detects_forced_dormant_unit():
    net = _small_net(seed=1)
    # Force fc1 unit 4 exactly silent through the REAL forward path:
    # zero its LayerNorm affine -> normalized output 0 -> silu(0) = 0.
    with torch.no_grad():
        net.norm1.weight[4] = 0.0
        net.norm1.bias[4] = 0.0
    x = torch.randn(64, _FEATURE_DIM)
    stats = check_and_recycle(
        net=net, optimizer=None, sample_obs=x, tau=0.025,
        reset_optimizer_moments=False,
    )
    net.requires_grad_(False)  # scalar reads below without autograd warnings
    assert stats.fc1_indices == [4]
    assert stats.dormant_fc1 == 1
    assert stats.recycled == 1 + stats.dormant_fc2
    assert stats.hidden_dim == _HIDDEN and stats.trunk_dim == _TRUNK
    # Outgoing column zeroed; LN affine re-armed so the unit can wake.
    assert torch.equal(net.fc2.weight[:, 4], torch.zeros(_TRUNK))
    assert float(net.norm1.weight[4]) == 1.0
    # B3.7 diagnostics measured and finite.
    assert 0.0 <= stats.agree <= 1.0
    assert np.isfinite(stats.max_dlogit)


def test_check_and_recycle_zero_dormant_is_exact_noop():
    net = _small_net(seed=2)
    before = _clone_state(net)
    x = torch.randn(64, _FEATURE_DIM)
    rng_before = torch.get_rng_state()
    stats = check_and_recycle(
        net=net, optimizer=None, sample_obs=x, tau=0.0,
    )
    assert stats.recycled == 0
    assert stats.agree == 1.0 and stats.max_dlogit == 0.0
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), k
    # A zero-recycle check draws nothing from the torch RNG.
    assert torch.equal(torch.get_rng_state(), rng_before)


# ---------------------------------------------------------------------------
# 3. Optimizer-moment handling
# ---------------------------------------------------------------------------

def _populated_adam(net, steps: int = 3):
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(16, _FEATURE_DIM)
    for _ in range(steps):
        logits, value = net.forward_ac(x)
        loss = logits.sum() + value.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return opt


def _moments(opt, param):
    st = opt.state[param]
    return st["exp_avg"], st["exp_avg_sq"], st["step"]


def test_recycle_zeroes_adam_moment_slices_and_keeps_step():
    net = _small_net(seed=3)
    opt = _populated_adam(net)
    fc1_idx = torch.tensor([1])
    fc2_idx = torch.tensor([0])
    pre = {
        name: tuple(t.clone() for t in _moments(opt, p))
        for name, p in net.named_parameters()
    }
    # Every moment is genuinely non-zero before the recycle.
    assert float(pre["fc1.weight"][0].abs().min()) > 0.0
    recycle(net, opt, fc1_idx, fc2_idx, reset_optimizer_moments=True)

    for key in (0, 1):  # exp_avg, exp_avg_sq
        assert torch.equal(_moments(opt, net.fc1.weight)[key][1],
                           torch.zeros(_FEATURE_DIM))
        assert float(_moments(opt, net.fc1.bias)[key][1]) == 0.0
        assert float(_moments(opt, net.norm1.weight)[key][1]) == 0.0
        assert float(_moments(opt, net.norm1.bias)[key][1]) == 0.0
        # fc2: recycled row 0 AND the outgoing column of fc1 unit 1.
        assert torch.equal(_moments(opt, net.fc2.weight)[key][0],
                           torch.zeros(_HIDDEN))
        assert torch.equal(_moments(opt, net.fc2.weight)[key][:, 1],
                           torch.zeros(_TRUNK))
        assert float(_moments(opt, net.fc2.bias)[key][0]) == 0.0
        assert float(_moments(opt, net.norm2.weight)[key][0]) == 0.0
        assert float(_moments(opt, net.norm2.bias)[key][0]) == 0.0
        assert torch.equal(_moments(opt, net.actor.weight)[key][:, 0],
                           torch.zeros(_NUM_ACTIONS))
        assert torch.equal(_moments(opt, net.critic.weight)[key][:, 0],
                           torch.zeros(1))
        # Untouched slices keep their momentum bit-for-bit.
        assert torch.equal(_moments(opt, net.fc1.weight)[key][0],
                           pre["fc1.weight"][key][0])
        assert torch.equal(_moments(opt, net.actor.weight)[key][:, 2],
                           pre["actor.weight"][key][:, 2])
        assert torch.equal(_moments(opt, net.actor.bias)[key],
                           pre["actor.bias"][key])
    # Per-tensor step counters untouched (B3.6).
    for name, p in net.named_parameters():
        assert torch.equal(_moments(opt, p)[2], pre[name][2]), name


def test_reset_optimizer_moments_false_leaves_moments_alone():
    net = _small_net(seed=4)
    opt = _populated_adam(net)
    pre = {
        name: tuple(t.clone() for t in _moments(opt, p))
        for name, p in net.named_parameters()
    }
    recycle(net, opt, torch.tensor([1]), torch.tensor([0]),
            reset_optimizer_moments=False)
    for name, p in net.named_parameters():
        for key in (0, 1, 2):
            assert torch.equal(_moments(opt, p)[key], pre[name][key]), name


def test_recycle_with_unstepped_optimizer_is_safe():
    net = _small_net(seed=5)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)  # no state yet
    recycle(net, opt, torch.tensor([0]), torch.tensor([1]),
            reset_optimizer_moments=True)
    assert torch.equal(net.fc2.weight[:, 0], torch.zeros(_TRUNK))


# ---------------------------------------------------------------------------
# 4. OFF = byte-identical (the gate consumes nothing when disabled)
# ---------------------------------------------------------------------------

def _one_training_step(net, opt, x):
    logits, value = net.forward_ac(x)
    loss = logits.square().mean() + value.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_disabled_gate_is_byte_identical_training():
    """Two identical seeded training runs, one calling the disabled
    hook between steps exactly as the trainer does: final params,
    optimizer moments, and both RNG streams must be byte-equal."""
    results = []
    for with_disabled_hook in (False, True):
        torch.manual_seed(77)
        np.random.seed(77)
        net = _small_net(seed=77)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        x = torch.randn(32, _FEATURE_DIM)
        valid_indices = np.arange(32)
        for it in range(3):
            _one_training_step(net, opt, x)
            if with_disabled_hook:
                out = maybe_check_and_recycle(
                    enabled=False, net=net, optimizer=opt,
                    obs_all=None, valid_indices=valid_indices,
                    tau=0.025, sample_batch=4096,
                    check_every_iters=1, global_it=it,
                )
                assert out is None
        results.append((
            {k: v.clone() for k, v in net.state_dict().items()},
            torch.get_rng_state().clone(),
            np.random.get_state(),
        ))
    (sd_a, trng_a, nrng_a), (sd_b, trng_b, nrng_b) = results
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k
    assert torch.equal(trng_a, trng_b)
    assert nrng_a[0] == nrng_b[0] and (nrng_a[1] == nrng_b[1]).all()


def test_enabled_gate_requires_obs_and_respects_cadence():
    net = _small_net(seed=6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(64, _FEATURE_DIM)
    valid = np.arange(64)
    # Off-cadence: no sampling, no stats.
    assert maybe_check_and_recycle(
        enabled=True, net=net, optimizer=opt, obs_all=x,
        valid_indices=valid, tau=0.025, sample_batch=32,
        check_every_iters=5, global_it=3,
    ) is None
    # On-cadence: a real check runs on min(sample_batch, valid) rows.
    stats = maybe_check_and_recycle(
        enabled=True, net=net, optimizer=opt, obs_all=x,
        valid_indices=valid, tau=0.025, sample_batch=32,
        check_every_iters=5, global_it=5,
    )
    assert stats is not None and stats.hidden_dim == _HIDDEN
    # Enabled without a rollout obs tensor is a LOUD failure, not a
    # silent skip (the mechanism-armed-but-inert class).
    with pytest.raises(ValueError):
        maybe_check_and_recycle(
            enabled=True, net=net, optimizer=opt, obs_all=None,
            valid_indices=valid, tau=0.025, sample_batch=32,
            check_every_iters=1, global_it=0,
        )


def test_forced_high_tau_recycles_and_preserves_identity_diagnostics():
    """The V7 forced-recycle shape: a throwaway high tau must recycle
    >= 1 unit with finite diagnostics. One unit is pinned quiet-but-
    nonzero (tiny LN gain) so a recycle is guaranteed at tau=0.5 while
    the registered tau=0.025 would not fire — deterministic, unlike a
    fresh net whose scores hover near 1."""
    net = _small_net(seed=8)
    with torch.no_grad():
        net.norm1.weight[4] = 1e-3
    opt = _populated_adam(net)
    x = torch.randn(256, _FEATURE_DIM)
    stats = check_and_recycle(
        net=net, optimizer=opt, sample_obs=x, tau=0.5,
    )
    assert stats.recycled >= 1
    assert np.isfinite(stats.max_dlogit)
    assert 0.0 <= stats.agree <= 1.0
    for i in stats.fc1_indices:
        assert torch.equal(net.fc2.weight[:, i], torch.zeros(_TRUNK))
    for j in stats.fc2_indices:
        assert torch.equal(net.actor.weight[:, j],
                           torch.zeros(_NUM_ACTIONS))
        assert torch.equal(net.critic.weight[:, j], torch.zeros(1))


# ---------------------------------------------------------------------------
# 5. Trainer-side wiring pins (source-structure assertions, following the
#    test_trainer_mechanism_guards.py precedent)
# ---------------------------------------------------------------------------

def _trainer_source() -> str:
    return _TRAINER_PATH.read_text()


def test_trainer_redo_default_off_and_registered_knobs():
    src = _trainer_source()
    assert 'rl_cfg.get("redo_enabled", False)' in src
    assert 'rl_cfg.get("redo_tau", 0.025)' in src
    assert 'rl_cfg.get("redo_check_every_iters", 1)' in src
    assert 'rl_cfg.get("redo_sample_batch", 4096)' in src
    assert 'rl_cfg.get("redo_reset_optimizer_moments", True)' in src


def test_trainer_emits_registered_armed_evidence_lines():
    """B6 grep targets, exact prefixes — the V2/V7 preflights grep for
    them; rewording voids real runs."""
    src = _trainer_source()
    assert '"[redo] ENABLED tau=%g every_iters=%d scope=fc1,fc2 "' in src
    assert '"[redo] disabled"' in src
    assert '"[redo] iter %d: dormant fc1 %d/%d fc2 %d/%d "' in src
    assert '"recycled %d cum %d agree %.4f max_dlogit %.6f"' in src
    assert '"[redo] iter %d: skipped (no gradient step)"' in src


def test_trainer_hook_sits_after_update_and_is_gated():
    """The hook must (a) run after the PPO update result unpack and
    before the PR-MDP block, (b) pass `enabled=redo_on` so the disabled
    path never samples."""
    src = _trainer_source()
    upd = src.index('mb_size = _upd["mb_size"]')
    hook = src.index("_redo_maybe_check(")
    prmdp = src.index("PR-MDP ADVERSARY UPDATE")
    assert upd < hook < prmdp
    assert "enabled=redo_on" in src
