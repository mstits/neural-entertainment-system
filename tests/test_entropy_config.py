"""EntropyControlConfig.from_profile — pins it against `trainer.__init__`.

Task 0.5 of docs/proposals/trainer_decomposition_plan.md (§5): a pure,
additive extraction of the entropy-bonus knobs `NESTrainer.__init__`
parses (trainer.py ~928-943). Not wired into the trainer yet.

Two checks, each meant to BITE on drift:

  1. `configs/mario_tiles_vanilla.yaml` parses to the exact literal
     values `__init__` would compute from that profile's `reinforce`
     block (verified by reading the actual `__init__` expressions, not
     guessed).
  2. An empty profile (`{}`) falls back to `__init__`'s exact defaults
     for every field, including the `_entropy_coef_base` mirror.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from src.training.entropy_config import EntropyControlConfig

ROOT = Path(__file__).resolve().parent.parent
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"


def test_parses_mario_tiles_vanilla_profile() -> None:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)

    # Sanity: the fixture profile must actually exercise the field we're
    # pinning, or this test would pass vacuously against a stale default.
    assert profile["reinforce"]["entropy_coef"] == 0.01
    assert "entropy_floor" not in profile["reinforce"]
    assert "entropy_coef_max" not in profile["reinforce"]

    cfg = EntropyControlConfig.from_profile(profile)

    # rl_cfg.get("entropy_coef", 0.01) with reinforce.entropy_coef: 0.01
    assert cfg.entropy_coef == 0.01
    # float(rl_cfg.get("entropy_floor", 0.0)) -- key absent from this
    # profile's reinforce block, so the __init__ default applies.
    assert cfg.entropy_floor == 0.0
    # float(rl_cfg.get("entropy_coef_max", 0.05)) -- likewise absent.
    assert cfg.entropy_coef_max == 0.05
    # self._entropy_coef_base = self.entropy_coef (post-parse mirror).
    assert cfg.entropy_coef_base == 0.01

    assert isinstance(cfg.entropy_floor, float)
    assert isinstance(cfg.entropy_coef_max, float)


def test_empty_profile_falls_back_to_init_defaults() -> None:
    cfg = EntropyControlConfig.from_profile({})

    assert cfg.entropy_coef == 0.01
    assert cfg.entropy_floor == 0.0
    assert cfg.entropy_coef_max == 0.05
    assert cfg.entropy_coef_base == 0.01

    assert isinstance(cfg.entropy_floor, float)
    assert isinstance(cfg.entropy_coef_max, float)


def test_missing_reinforce_block_behaves_like_empty_dict() -> None:
    # game_profile.get("reinforce", {}) -- a profile with no `reinforce`
    # key at all must parse identically to reinforce: {}.
    cfg_no_block = EntropyControlConfig.from_profile({"name": "no reinforce"})
    cfg_empty_block = EntropyControlConfig.from_profile({"reinforce": {}})

    assert cfg_no_block == cfg_empty_block == EntropyControlConfig.from_profile({})


def test_overrides_are_honored_and_base_tracks_parsed_coef() -> None:
    # A profile that sets all three keys (plus a floor above zero, the
    # opt-in adaptive-controller path) must thread every value through
    # untouched, and entropy_coef_base must mirror the PARSED
    # entropy_coef, not the hardcoded 0.01 default.
    profile = {
        "reinforce": {
            "entropy_coef": 0.02,
            "entropy_floor": 0.15,
            "entropy_coef_max": 0.08,
        }
    }
    cfg = EntropyControlConfig.from_profile(profile)

    assert cfg.entropy_coef == 0.02
    assert cfg.entropy_floor == 0.15
    assert cfg.entropy_coef_max == 0.08
    assert cfg.entropy_coef_base == 0.02


def test_frozen_dataclass_is_immutable() -> None:
    cfg = EntropyControlConfig.from_profile({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.entropy_coef = 0.5  # type: ignore[misc]
