"""Mechanism 3 — self-imitation buffer (`reinforce.sil`).

Full trajectories of episodes that CLEAR the level (the vanilla loop's
completion-diff detector) are stored in a small ring buffer; each PPO
iteration adds a BC cross-entropy term on minibatches sampled from the
stored clears, coefficient `bc_coef`. No prioritized replay. Default-off:
absent config leaves the update byte-identical (also pinned by the C0
golden).
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.tile_policy import TilePolicyNetwork
from src.training.sil import SelfImitationBuffer
from src.training.timing import GenTimer

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Buffer mechanics
# ---------------------------------------------------------------------------

def _traj(n, fdim=6, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(-4, 4, size=(n, fdim)).astype(np.int8),
        rng.integers(0, 5, size=n).astype(np.int64),
    )


def test_clears_enter_buffer_and_counters_track() -> None:
    buf = SelfImitationBuffer(capacity=3)
    assert len(buf) == 0
    assert buf.n_steps == 0
    o, a = _traj(10)
    buf.add(o, a)
    assert len(buf) == 1
    assert buf.n_steps == 10
    assert buf.total_clears == 1
    buf.add(*_traj(4, seed=1))
    assert len(buf) == 2
    assert buf.n_steps == 14


def test_capacity_evicts_oldest() -> None:
    buf = SelfImitationBuffer(capacity=2)
    buf.add(*_traj(3, seed=0))
    buf.add(*_traj(5, seed=1))
    buf.add(*_traj(7, seed=2))
    assert len(buf) == 2
    assert buf.n_steps == 12  # 5 + 7 — the 3-step one was evicted
    assert buf.total_clears == 3  # lifetime counter keeps counting


def test_empty_trajectory_is_ignored() -> None:
    buf = SelfImitationBuffer(capacity=2)
    buf.add(np.zeros((0, 6), dtype=np.int8), np.zeros(0, dtype=np.int64))
    assert len(buf) == 0
    assert buf.total_clears == 0


def test_sample_shapes_and_membership() -> None:
    buf = SelfImitationBuffer(capacity=4)
    o1, a1 = _traj(6, seed=3)
    buf.add(o1, a1)
    obs, act = buf.sample(32)
    assert obs.shape == (32, 6)
    assert act.shape == (32,)
    assert act.dtype == np.int64
    # Every sampled row must be one of the stored (obs, action) pairs.
    for row, lab in zip(obs, act):
        matches = np.all(o1 == row, axis=1)
        assert matches.any()
        assert lab in a1[matches]


# ---------------------------------------------------------------------------
# BC loss wiring in the PPO update (fake-trainer harness on the real updater)
# ---------------------------------------------------------------------------

_NA, _FDIM = 4, 8
_T, _N = 6, 2


def _fake_trainer(sil_buffer, bc_coef, *, set_attr=True):
    t = types.SimpleNamespace(
        device=torch.device("cpu"),
        _rnd=None,
        _gen_timer=GenTimer(),
        _is_tile_mode=True,
        preprocess_f16=False,
        _gx_count_beta=0.0,
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
        rnd_predictor_update_fraction=1.0,
        demo_anchor_decay_start=0,
        demo_anchor_decay_iters=1,
        demo_anchor_coef0=0.0,
        demo_anchor_final=0.0,
    )
    if set_attr:
        t._sil_buffer = sil_buffer
        t._sil_bc_coef = bc_coef
    return t


def _run_update(t, seed=555):
    from src.training.ppo_updater import PPOUpdater

    np.random.seed(seed)
    torch.manual_seed(seed)
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=8, trunk_dim=4
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    obs_buf = rng.integers(-4, 4, size=(_T, _N, _FDIM)).astype(np.int8)
    action_buf = rng.integers(0, _NA, size=(_T, _N)).astype(np.int32)
    reward_buf = rng.normal(size=(_T, _N)).astype(np.float32)
    value_buf = rng.normal(size=(_T, _N)).astype(np.float32)
    log_prob_buf = np.full((_T, _N), -1.2, dtype=np.float32)
    done_buf = np.zeros((_T, _N), dtype=np.bool_)
    valid_buf = np.ones((_T, _N), dtype=np.bool_)
    bonus_buf = np.zeros((_T, _N), dtype=np.float32)
    final_values = np.zeros(_N, dtype=np.float32)
    out = PPOUpdater(t).update(
        net=net, optimizer=opt, obs_buf=obs_buf, action_buf=action_buf,
        reward_buf=reward_buf, value_buf=value_buf, log_prob_buf=log_prob_buf,
        done_buf=done_buf, valid_buf=valid_buf, bonus_buf=bonus_buf,
        final_values_np=final_values, rollout_steps=_T, num_envs=_N,
        obs_shape=(_FDIM,), global_it=0, sam_rho=0.0,
    )
    return out, net


def test_bc_term_only_when_buffer_nonempty() -> None:
    # Empty buffer: no SIL loss reported, update proceeds normally.
    empty = SelfImitationBuffer(capacity=4)
    out_empty, net_empty = _run_update(_fake_trainer(empty, 0.5))
    assert out_empty["sil_loss_n"] == 0

    # Non-empty buffer: SIL loss appears and shifts the update.
    filled = SelfImitationBuffer(capacity=4)
    rng = np.random.default_rng(9)
    filled.add(
        rng.integers(-4, 4, size=(12, _FDIM)).astype(np.int8),
        rng.integers(0, _NA, size=12).astype(np.int64),
    )
    out_full, net_full = _run_update(_fake_trainer(filled, 0.5))
    assert out_full["sil_loss_n"] > 0
    assert np.isfinite(out_full["sil_loss_accum"])
    diff = any(
        not torch.equal(a, b)
        for (_, a), (_, b) in zip(
            net_empty.state_dict().items(), net_full.state_dict().items()
        )
    )
    assert diff, "BC term had no effect on the update"


def test_default_off_is_untouched() -> None:
    """No `_sil_buffer` attr (mechanism never configured) == buffer None ==
    zero coefficient: all three produce bit-identical updates."""
    out_none, net_none = _run_update(_fake_trainer(None, 0.0))
    out_absent, net_absent = _run_update(_fake_trainer(None, 0.0, set_attr=False))
    filled = SelfImitationBuffer(capacity=4)
    filled.add(*_traj(8, fdim=_FDIM, seed=2))
    out_zero, net_zero = _run_update(_fake_trainer(filled, 0.0))
    for (k, a) in net_none.state_dict().items():
        assert torch.equal(a, net_absent.state_dict()[k]), k
        assert torch.equal(a, net_zero.state_dict()[k]), k
    assert out_none["sil_loss_n"] == out_absent["sil_loss_n"] == 0
    assert out_zero["sil_loss_n"] == 0


# ---------------------------------------------------------------------------
# Trainer wiring + schema
# ---------------------------------------------------------------------------

def test_trainer_flushes_clears_into_the_buffer() -> None:
    """The flush sits inside the completion-diff clear detector (the same
    trusted signal the clear counter and PLR use), and accumulators reset
    on done and at the iter boundary."""
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    i_clear = src.find("n_clears_this_iter += 1")
    assert i_clear > 0
    flush = src.find("_sil_flush_clear(i)")
    assert 0 < flush, "no SIL flush call in trainer"
    assert abs(flush - i_clear) < 2000, (
        "SIL flush is not adjacent to the completion-diff clear detector"
    )
    assert "_sil_drop_episode(i)" in src, "no accumulator drop on done"


def test_config_schema_sil_block() -> None:
    from src.training.config_schema import (
        KNOWN_REINFORCE_KEYS, validate_profile,
    )
    assert "sil" in KNOWN_REINFORCE_KEYS
    clean = {"name": "x", "reinforce": {"sil": {
        "enabled": True, "buffer_size": 64, "bc_coef": 0.5,
    }}}
    assert validate_profile(clean) == []
    typo = {"name": "x", "reinforce": {"sil": {"buffer_sise": 64}}}
    assert any("buffer_sise" in w for w in validate_profile(typo))
