"""RND obs-rms/reward-rms update-normalization must see only valid rows.

`PPOUpdater.update`'s RND intrinsic block runs the predictor over the FULL
`(rollout_steps * num_envs)` obs buffer (needed so `fold_intrinsic_into_rewards`
can zero the bonus on done/padded steps itself), but the running-stats push
into `t._rnd.obs_rms` / `reward_rms` must be restricted to the valid rows --
same as the GAE-normalization pass and the K-epoch minibatch loop later in the
same function, both of which index by `valid_buf`/`valid_indices`. When most
envs in a rollout are frozen after death (`trainer.py`'s `active_in_iter`
freeze), `obs_buf[t, i]` for every remaining step of that env is a
byte-identical copy of the death frame while `valid_buf[t, i]` stays False --
folding those duplicate rows into the Welford stats mis-scales the intrinsic
bonus's divisor and every future `_normalize_obs` call for the rest of the run.
"""
from __future__ import annotations

import types

import numpy as np
import torch

from src.models.rnd import _RunningMeanStd
from src.models.tile_policy import TilePolicyNetwork
from src.models.tile_rnd import TileRND
from src.training.ppo_updater import PPOUpdater
from src.training.timing import GenTimer

_NA, _FDIM = 4, 4
_T, _N = 6, 2


def _fake_trainer(rnd: TileRND) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        _rnd=rnd,
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


def _frozen_padding_rollout() -> dict:
    """Env 0 dies at step 2 and is frozen for steps 3-5 -- its obs stays a
    byte-identical copy of the death frame, mirroring `trainer.py`'s
    `active_in_iter[i] = False` freeze. Env 1 stays alive/valid throughout."""
    rng = np.random.default_rng(0)
    obs_buf = rng.integers(-2, 2, size=(_T, _N, _FDIM)).astype(np.int8)
    death_frame = np.full(_FDIM, 50, dtype=np.int8)
    obs_buf[2, 0] = death_frame
    obs_buf[3, 0] = death_frame
    obs_buf[4, 0] = death_frame
    obs_buf[5, 0] = death_frame

    valid_buf = np.ones((_T, _N), dtype=np.bool_)
    valid_buf[3:, 0] = False  # frozen padding rows for env 0

    done_buf = np.zeros((_T, _N), dtype=np.bool_)
    done_buf[2, 0] = True

    return dict(
        obs_buf=obs_buf,
        action_buf=rng.integers(0, _NA, size=(_T, _N)).astype(np.int32),
        reward_buf=rng.normal(size=(_T, _N)).astype(np.float32),
        value_buf=rng.normal(size=(_T, _N)).astype(np.float32),
        log_prob_buf=np.full((_T, _N), -1.2, dtype=np.float32),
        done_buf=done_buf,
        valid_buf=valid_buf,
        bonus_buf=np.zeros((_T, _N), dtype=np.float32),
        final_values_np=np.zeros(_N, dtype=np.float32),
    )


def test_rnd_obs_rms_update_excludes_frozen_padding_rows() -> None:
    torch.manual_seed(0)
    rnd = TileRND(feature_dim=_FDIM, feat_dim=8)
    t = _fake_trainer(rnd)
    rollout = _frozen_padding_rollout()

    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=8, trunk_dim=4
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Independently compute the expected obs-rms stats from a FRESH
    # `_RunningMeanStd`, fed only the valid rows in the same flat row order
    # the real update builds `rnd_obs_t` in.
    obs_flat = torch.from_numpy(
        rollout["obs_buf"].reshape(_T * _N, _FDIM)
    ).float()
    valid_flat = torch.from_numpy(rollout["valid_buf"].reshape(-1))
    expected_rms = _RunningMeanStd(shape=(_FDIM,))
    expected_rms.update(obs_flat[valid_flat])

    # Sanity check the fixture actually distinguishes masked from unmasked:
    # folding in the frozen padding rows must move the stats measurably.
    unmasked_rms = _RunningMeanStd(shape=(_FDIM,))
    unmasked_rms.update(obs_flat)
    assert not torch.allclose(expected_rms.mean, unmasked_rms.mean)

    PPOUpdater(t).update(
        net=net, optimizer=opt, rollout_steps=_T, num_envs=_N,
        obs_shape=(_FDIM,), global_it=0, sam_rho=0.0, **rollout,
    )

    assert torch.allclose(rnd.obs_rms.mean, expected_rms.mean), (
        "obs_rms folded in rows outside valid_buf -- the RND running-stats "
        "update must be masked like every other consumer of the rollout."
    )
    assert torch.allclose(rnd.obs_rms.var, expected_rms.var)
    assert torch.allclose(rnd.obs_rms.count, expected_rms.count)
