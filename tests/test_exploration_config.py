"""ExplorationConfig.from_profile -- pins it against `trainer.py`.

Task 0.5 of docs/proposals/trainer_decomposition_plan.md (§5): a pure,
additive extraction of the RND / count-bonus / Go-Explore knobs the
trainer parses (`rl_cfg = profile.get("reinforce", {})` in `__init__`,
plus the `_ge_cfg` / `_geb_cfg` reads in `_run_vanilla_ppo`). Not wired
into the trainer yet.

Checks, each meant to BITE on drift:

  1. `configs/mario_tiles_vanilla.yaml` parses to the exact literal
     values the trainer would compute from that profile's `reinforce`
     block (verified by reading the actual parsing expressions, not
     guessed).
  2. An empty profile (`{}`) falls back to the trainer's exact defaults
     for every field, including the nested Go-Explore / burst configs.
  3. A profile that sets every go_explore / go_explore_fallback key
     threads each value through untouched (the real fixture profile
     doesn't exercise these, so this closes that gap).
  4. The out-of-range `rnd_predictor_update_fraction` guard raises,
     matching `__init__`'s ValueError.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from src.training.exploration_config import (
    ExplorationConfig,
    GoExploreConfig,
    GoExploreFallbackConfig,
)

ROOT = Path(__file__).resolve().parent.parent
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"


def test_parses_mario_tiles_vanilla_profile() -> None:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)

    # Sanity: the fixture profile must actually exercise the fields
    # we're pinning, or this test would pass vacuously against a stale
    # default.
    assert profile["reinforce"]["rnd_intrinsic_coef"] == 0.1
    assert profile["reinforce"]["rnd_loss_coef"] == 1.0
    assert "rnd_predictor_update_fraction" not in profile["reinforce"]
    assert "gx_count_bonus_coef" not in profile["reinforce"]
    assert "go_explore" not in profile["reinforce"]
    assert "go_explore_fallback" not in profile["reinforce"]

    cfg = ExplorationConfig.from_profile(profile)

    # float(rl_cfg.get("rnd_intrinsic_coef", 0.0))
    assert cfg.rnd_intrinsic_coef == 0.1
    # float(rl_cfg.get("rnd_loss_coef", 1.0))
    assert cfg.rnd_loss_coef == 1.0
    # float(rl_cfg.get("rnd_predictor_update_fraction", 1.0)) -- absent
    assert cfg.rnd_predictor_update_fraction == 1.0
    # float(rl_cfg.get("gx_count_bonus_coef", 0.0)) -- absent
    assert cfg.gx_count_beta == 0.0
    # go_explore / go_explore_fallback blocks absent -> nested defaults
    assert cfg.go_explore == GoExploreConfig()
    assert cfg.go_explore_fallback == GoExploreFallbackConfig()

    assert isinstance(cfg.rnd_intrinsic_coef, float)
    assert isinstance(cfg.rnd_loss_coef, float)
    assert isinstance(cfg.rnd_predictor_update_fraction, float)
    assert isinstance(cfg.gx_count_beta, float)


def test_empty_profile_falls_back_to_init_defaults() -> None:
    cfg = ExplorationConfig.from_profile({})

    assert cfg.rnd_intrinsic_coef == 0.0
    assert cfg.rnd_loss_coef == 1.0
    assert cfg.rnd_predictor_update_fraction == 1.0
    assert cfg.gx_count_beta == 0.0

    assert cfg.go_explore == GoExploreConfig(
        enabled=False,
        save_every=10,
        inline_return_prob=0.0,
        seed=None,
        score="auto",
        cell={},
    )
    assert cfg.go_explore_fallback == GoExploreFallbackConfig(
        enabled=False,
        stall_patience=60,
        burst_iters=30,
        burst_env_frac=0.25,
        burst_env_cap=8,
    )

    assert isinstance(cfg.rnd_intrinsic_coef, float)
    assert isinstance(cfg.rnd_loss_coef, float)
    assert isinstance(cfg.rnd_predictor_update_fraction, float)
    assert isinstance(cfg.gx_count_beta, float)


def test_missing_reinforce_block_behaves_like_empty_dict() -> None:
    # game_profile.get("reinforce", {}) -- a profile with no `reinforce`
    # key at all must parse identically to reinforce: {}.
    cfg_no_block = ExplorationConfig.from_profile({"name": "no reinforce"})
    cfg_empty_block = ExplorationConfig.from_profile({"reinforce": {}})

    assert (
        cfg_no_block
        == cfg_empty_block
        == ExplorationConfig.from_profile({})
    )


def test_go_explore_overrides_are_honored() -> None:
    # _ge_cfg = dict((profile.get("reinforce", {}) or {}).get("go_explore", {}) or {})
    profile = {
        "reinforce": {
            "go_explore": {
                "enabled": True,
                "save_every": 25,
                "inline_return_prob": 0.35,
                "seed": 777,
                "score": "max_x",
                "cell": {"type": "ram_bytes", "addresses": [1, 2, 3], "bucket": 2},
            },
        }
    }
    cfg = ExplorationConfig.from_profile(profile)

    assert cfg.go_explore.enabled is True
    assert cfg.go_explore.save_every == 25
    assert cfg.go_explore.inline_return_prob == 0.35
    assert cfg.go_explore.seed == 777
    assert cfg.go_explore.score == "max_x"
    assert cfg.go_explore.cell == {
        "type": "ram_bytes", "addresses": [1, 2, 3], "bucket": 2,
    }


def test_go_explore_save_every_clamped_to_at_least_one() -> None:
    # max(1, int(_ge_cfg.get("save_every", 10)))
    profile = {"reinforce": {"go_explore": {"save_every": 0}}}
    cfg = ExplorationConfig.from_profile(profile)
    assert cfg.go_explore.save_every == 1


def test_go_explore_seed_absent_is_none_not_zero() -> None:
    # `int(_ge_cfg.get("seed", self.seed) or 0)` in the trainer falls
    # back to the constructor's `seed` param, which from_profile cannot
    # see -- the pure-profile mirror is None (see module docstring).
    cfg = ExplorationConfig.from_profile({})
    assert cfg.go_explore.seed is None


def test_go_explore_fallback_overrides_are_honored() -> None:
    # _rl_cfg = profile.get("reinforce", {}) or {}
    # _geb_cfg = dict(_rl_cfg.get("go_explore_fallback", {}) or {})
    profile = {
        "reinforce": {
            "go_explore_fallback": {
                "enabled": True,
                "stall_patience": 12,
                "burst_iters": 5,
                "burst_env_frac": 0.5,
                "burst_env_cap": 3,
            },
        }
    }
    cfg = ExplorationConfig.from_profile(profile)

    assert cfg.go_explore_fallback.enabled is True
    assert cfg.go_explore_fallback.stall_patience == 12
    assert cfg.go_explore_fallback.burst_iters == 5
    assert cfg.go_explore_fallback.burst_env_frac == 0.5
    assert cfg.go_explore_fallback.burst_env_cap == 3


def test_go_explore_fallback_clamps_stall_patience_and_burst_iters() -> None:
    # max(1, int(_geb_cfg.get("stall_patience", 60)))
    # max(1, int(_geb_cfg.get("burst_iters", 30)))
    profile = {
        "reinforce": {
            "go_explore_fallback": {"stall_patience": 0, "burst_iters": -5},
        }
    }
    cfg = ExplorationConfig.from_profile(profile)
    assert cfg.go_explore_fallback.stall_patience == 1
    assert cfg.go_explore_fallback.burst_iters == 1


def test_rnd_predictor_update_fraction_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        ExplorationConfig.from_profile(
            {"reinforce": {"rnd_predictor_update_fraction": 0.0}}
        )
    with pytest.raises(ValueError):
        ExplorationConfig.from_profile(
            {"reinforce": {"rnd_predictor_update_fraction": 1.5}}
        )


def test_rnd_predictor_update_fraction_boundary_one_is_valid() -> None:
    cfg = ExplorationConfig.from_profile(
        {"reinforce": {"rnd_predictor_update_fraction": 1.0}}
    )
    assert cfg.rnd_predictor_update_fraction == 1.0


def test_frozen_dataclass_is_immutable() -> None:
    cfg = ExplorationConfig.from_profile({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rnd_intrinsic_coef = 0.5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.go_explore.enabled = True  # type: ignore[misc]
