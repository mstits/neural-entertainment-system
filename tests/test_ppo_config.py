"""Parse tests for PPOConfig.from_profile — pinned against the exact
expressions in Trainer.__init__ (src/training/trainer.py, the
rl_cfg = game_profile.get("reinforce", {}) block).

Two cases: a real shipped profile (values must come from the profile,
not the defaults) and an empty profile (every field must fall back to
__init__'s exact default).
"""
from pathlib import Path

import yaml

from src.training.ppo_config import PPOConfig

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text()) or {}


def test_mario_tiles_vanilla_profile_matches_trainer_init_expressions():
    profile = _load("mario_tiles_vanilla.yaml")
    cfg = PPOConfig.from_profile(profile)

    # Sanity: the fields this test is pinning are actually set in the
    # profile's reinforce block (not silently falling back to defaults).
    rl_cfg = profile["reinforce"]
    assert "rollout_steps" in rl_cfg
    assert "ppo_minibatch_size" in rl_cfg
    assert "steps" in rl_cfg
    assert "lr" in rl_cfg
    assert "value_loss" in rl_cfg
    assert "value_coef" in rl_cfg
    # These four are deliberately absent from the profile, so they must
    # come from __init__'s literal defaults, not from the yaml.
    assert "gamma" not in rl_cfg
    assert "gae_lambda" not in rl_cfg
    assert "ppo_clip_eps" not in rl_cfg
    assert "grad_clip" not in rl_cfg

    assert cfg.rollout_steps == 1024
    assert cfg.reinforce_steps == 10
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_eps == 0.2
    assert cfg.value_coef == 0.25
    assert cfg.value_loss_kind == "huber"
    assert cfg.ppo_minibatch_size == 256
    assert cfg.grad_clip == 1.0
    assert cfg.lr == 3.0e-4


def test_empty_profile_falls_back_to_trainer_init_defaults():
    cfg = PPOConfig.from_profile({})

    assert cfg.rollout_steps == 512
    assert cfg.reinforce_steps == 3
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_eps == 0.2
    assert cfg.value_coef == 0.5
    assert cfg.value_loss_kind == "mse"
    assert cfg.ppo_minibatch_size == 256
    assert cfg.grad_clip == 1.0
    assert cfg.lr == 1e-4


def test_missing_reinforce_block_behaves_like_empty_dict():
    """`game_profile.get("reinforce", {})` — a profile with no `reinforce`
    key at all must parse identically to an explicit empty one."""
    assert PPOConfig.from_profile({"name": "no reinforce block here"}) == (
        PPOConfig.from_profile({})
    )


def test_value_loss_kind_is_lowercased_and_validated():
    cfg = PPOConfig.from_profile({"reinforce": {"value_loss": "HUBER"}})
    assert cfg.value_loss_kind == "huber"

    try:
        PPOConfig.from_profile({"reinforce": {"value_loss": "bogus"}})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown value_loss")


def test_types_are_ints_not_floats_where_init_casts_to_int():
    """__init__ wraps rollout_steps and ppo_minibatch_size in int(), so a
    yaml float/string must coerce the same way here."""
    cfg = PPOConfig.from_profile(
        {"reinforce": {"rollout_steps": "128", "ppo_minibatch_size": 64.0}}
    )
    assert cfg.rollout_steps == 128 and isinstance(cfg.rollout_steps, int)
    assert cfg.ppo_minibatch_size == 64 and isinstance(
        cfg.ppo_minibatch_size, int
    )
