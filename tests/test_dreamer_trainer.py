"""DreamerTrainer integration tests — checkpoint roundtrip + train-step
sanity. We don't spin up a real nes_core pool here because the existing
test infrastructure doesn't sandbox ROM access; instead we exercise the
internal train loops directly with a synthetic replay buffer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.training.dreamer import DreamerTrainer


def _profile() -> dict:
    return {
        "name": "test",
        "action_space": ["noop", "right", "a", "b"],
        "frame_skip": 16,
        "reward_weights": {},
        "curriculum": {"stages": []},
        "dreamer": {
            "batch_size": 2,
            "seq_len": 8,
            "imag_horizon": 4,
            "replay_capacity": 200,
            "warmup_steps": 0,
        },
    }


def _populate(trainer: DreamerTrainer, n: int = 50) -> None:
    for _ in range(n):
        trainer.replay.add(
            obs=np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            action=int(np.random.randint(0, trainer.num_actions)),
            reward=float(np.random.randn()),
            done=False,
        )


def test_world_model_train_step_runs() -> None:
    t = DreamerTrainer(rom_path="/dev/null", game_profile=_profile(), num_instances=2)
    _populate(t)
    stats = t._train_world_model()
    for k in ("wm_recon", "wm_reward", "wm_continue", "wm_kl", "wm_total"):
        assert k in stats
        assert np.isfinite(stats[k])


def test_actor_critic_train_step_runs() -> None:
    t = DreamerTrainer(rom_path="/dev/null", game_profile=_profile(), num_instances=2)
    _populate(t)
    stats = t._train_actor_critic()
    for k in ("actor_loss", "critic_loss", "imag_return_mean", "imag_value_mean"):
        assert k in stats
        assert np.isfinite(stats[k])


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Save model + restore — params should match exactly afterwards."""
    t = DreamerTrainer(
        rom_path="/dev/null",
        game_profile=_profile(),
        num_instances=2,
        checkpoint_dir=str(tmp_path),
    )
    _populate(t)
    t._train_world_model()
    t._train_count = 7
    t._step_count = 1234
    ref_weight = t.wm.encoder.proj.weight.detach().clone()
    ckpt = t._save_checkpoint()
    assert ckpt.exists()

    # Mutate state, then load — should restore to snapshot.
    with torch.no_grad():
        t.wm.encoder.proj.weight.zero_()
    t._train_count = 0
    t._step_count = 0
    t._load_checkpoint(ckpt)
    assert torch.allclose(t.wm.encoder.proj.weight, ref_weight)
    assert t._train_count == 7
    assert t._step_count == 1234


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    t = DreamerTrainer(
        rom_path="/dev/null",
        game_profile=_profile(),
        num_instances=2,
        checkpoint_dir=str(tmp_path),
    )
    _populate(t)
    assert t._find_latest_checkpoint() is None
    t._train_count = 1
    t._save_checkpoint()
    t._train_count = 2
    p2 = t._save_checkpoint()
    found = t._find_latest_checkpoint()
    assert found == p2


def test_checkpoint_pruning(tmp_path: Path) -> None:
    """Only the last 5 checkpoints should remain on disk."""
    t = DreamerTrainer(
        rom_path="/dev/null",
        game_profile=_profile(),
        num_instances=2,
        checkpoint_dir=str(tmp_path),
    )
    _populate(t)
    for i in range(10):
        t._train_count = i + 1
        t._save_checkpoint()
    surviving = sorted(t.dreamer_ckpt_dir.glob("dreamer_*.pt"))
    assert len(surviving) == 5
    # The last 5 should be 6..10.
    last_train_counts = [int(p.stem.split("_")[1]) for p in surviving]
    assert last_train_counts == [6, 7, 8, 9, 10]


def test_categorical_latent_dim_propagates_to_actor() -> None:
    """When the profile picks categorical, the actor's first FC layer
    width should match `deter_dim + stoch_dim*stoch_classes`."""
    profile = _profile()
    profile["dreamer"]["latent_kind"] = "categorical"
    t = DreamerTrainer(rom_path="/dev/null", game_profile=profile, num_instances=2)
    expected = t.cfg.deter_dim + t.cfg.stoch_dim * t.cfg.stoch_classes
    assert t.actor.net[0].in_features == expected


def test_gaussian_latent_dim_propagates_to_actor() -> None:
    profile = _profile()
    profile["dreamer"]["latent_kind"] = "gaussian"
    t = DreamerTrainer(rom_path="/dev/null", game_profile=profile, num_instances=2)
    expected = t.cfg.deter_dim + t.cfg.stoch_dim
    assert t.actor.net[0].in_features == expected
