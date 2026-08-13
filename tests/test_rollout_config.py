"""Parse tests for RolloutConfig.from_profile — pinned against the exact
expressions in Trainer.__init__ (src/training/trainer.py, the
rl_cfg = game_profile.get("reinforce", {}) block plus the top-level
frame_skip/max_episode_steps reads and the constructor's own
num_instances/max_episode_steps defaults).

Two cases: a real shipped profile (values must come from the profile,
not the defaults) and an empty profile (every field must fall back to
__init__'s exact default).
"""
from pathlib import Path

import yaml

from src.training.rollout_config import RolloutConfig

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text()) or {}


def test_mario_tiles_vanilla_profile_matches_trainer_init_expressions():
    profile = _load("mario_tiles_vanilla.yaml")
    cfg = RolloutConfig.from_profile(profile)

    # Sanity: the fields this test is pinning are actually set in the
    # profile (not silently falling back to defaults).
    assert "frame_skip" in profile
    assert "max_episode_steps" in profile
    rl_cfg = profile["reinforce"]
    assert "sticky_action_prob" in rl_cfg
    assert "encoder" in rl_cfg
    assert "tile_frame_stack" in rl_cfg
    # These are deliberately absent from the profile, so they must come
    # from __init__'s literal defaults, not from the yaml.
    assert "preprocess_f16" not in rl_cfg
    assert "tile_hidden_dim" not in rl_cfg
    assert "tile_trunk_dim" not in rl_cfg
    assert "num_instances" not in profile

    assert cfg.sticky_action_prob == 0.0
    assert cfg.frame_skip == 4
    assert cfg.preprocess_f16 is False
    assert cfg.encoder_kind == "smb_tiles"
    assert cfg.is_tile_mode is True
    assert cfg.tile_frame_stack == 4
    assert cfg.tile_hidden_dim == 64
    assert cfg.tile_trunk_dim == 32
    # SMB tile feature_dim (13x13 grid + 6 scalars = 175) x frame_stack 4.
    assert cfg.tile_feature_dim == 700
    assert cfg.num_instances == 16
    assert cfg.max_episode_steps == 2400


def test_empty_profile_falls_back_to_trainer_init_defaults():
    cfg = RolloutConfig.from_profile({})

    assert cfg.sticky_action_prob == 0.0
    assert cfg.frame_skip == 16
    assert cfg.preprocess_f16 is False
    assert cfg.encoder_kind == "nature_dqn"
    assert cfg.is_tile_mode is False
    # Not tile mode -> default is 1, not the tile-mode default of 4.
    assert cfg.tile_frame_stack == 1
    assert cfg.tile_hidden_dim == 64
    assert cfg.tile_trunk_dim == 32
    # tile_feature_dim stays 0 unless is_tile_mode.
    assert cfg.tile_feature_dim == 0
    assert cfg.num_instances == 16
    assert cfg.max_episode_steps == 1000


def test_missing_reinforce_block_behaves_like_empty_dict():
    """`game_profile.get("reinforce", {})` — a profile with no `reinforce`
    key at all must parse identically to an explicit empty one."""
    assert RolloutConfig.from_profile({"name": "no reinforce block here"}) == (
        RolloutConfig.from_profile({})
    )


def test_tile_frame_stack_default_flips_with_tile_mode():
    """__init__'s default for tile_frame_stack is conditional on
    is_tile_mode (4 if tile mode else 1) — not a flat constant."""
    tile_cfg = RolloutConfig.from_profile(
        {"reinforce": {"encoder": "smb_tiles_pos"}}
    )
    assert tile_cfg.is_tile_mode is True
    assert tile_cfg.tile_frame_stack == 4

    pixel_cfg = RolloutConfig.from_profile(
        {"reinforce": {"encoder": "nature_dqn"}}
    )
    assert pixel_cfg.is_tile_mode is False
    assert pixel_cfg.tile_frame_stack == 1


def test_frame_skip_is_top_level_not_under_reinforce():
    """frame_skip is read from the top-level profile, not rl_cfg — a
    reinforce.frame_skip key must be ignored."""
    cfg = RolloutConfig.from_profile(
        {"frame_skip": 8, "reinforce": {"frame_skip": 2}}
    )
    assert cfg.frame_skip == 8


def test_preprocess_f16_has_no_bool_cast():
    """__init__ assigns `rl_cfg.get("preprocess_f16", False)` with no
    bool() wrapper — a non-bool truthy value passes through verbatim."""
    cfg = RolloutConfig.from_profile({"reinforce": {"preprocess_f16": 1}})
    assert cfg.preprocess_f16 == 1
    assert cfg.preprocess_f16 is not False


def test_types_are_ints_where_init_casts_to_int():
    cfg = RolloutConfig.from_profile(
        {
            "frame_skip": "8",
            "max_episode_steps": "500",
            "reinforce": {
                "tile_frame_stack": "2",
                "tile_hidden_dim": 128.0,
                "tile_trunk_dim": 16.0,
            },
        }
    )
    assert cfg.frame_skip == 8 and isinstance(cfg.frame_skip, int)
    assert cfg.max_episode_steps == 500 and isinstance(
        cfg.max_episode_steps, int
    )
    assert cfg.tile_frame_stack == 2 and isinstance(cfg.tile_frame_stack, int)
    assert cfg.tile_hidden_dim == 128 and isinstance(cfg.tile_hidden_dim, int)
    assert cfg.tile_trunk_dim == 16 and isinstance(cfg.tile_trunk_dim, int)
