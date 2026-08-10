"""Guard the profile schema validator + keep its registry honest.

- A typo'd reinforce key is flagged (the silent-default bug class).
- A clean profile passes.
- The registry stays in sync with what the trainer actually consumes:
  every key read via `<cfg>.get("...")` in trainer.py must be
  registered, or this test fails (so a new knob can't be added without
  registering it — closing the parsed-but-inert loophole at the source).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config_schema import (
    KNOWN_BACKWARD_CURRICULUM_KEYS,
    KNOWN_REINFORCE_KEYS,
    ConfigSchemaError,
    check_profile,
    consumed_backward_curriculum_keys_from_source,
    consumed_reinforce_keys_from_source,
    validate_profile,
)

REPO = Path(__file__).resolve().parent.parent


def test_typo_reinforce_key_flagged():
    prof = {"name": "x", "reinforce": {"entropy_flooor": 0.5, "lr": 1e-4}}
    warnings = validate_profile(prof)
    assert any("entropy_flooor" in w for w in warnings)


def test_clean_profile_passes():
    prof = {
        "name": "x", "frame_skip": 4,
        "reinforce": {"lr": 1e-4, "gamma": 0.99, "entropy_floor": 0.5},
    }
    assert validate_profile(prof) == []


def test_strict_mode_raises():
    prof = {"reinforce": {"bogus_knob": 1}}
    with pytest.raises(ConfigSchemaError):
        check_profile(prof, strict=True)


def test_typo_in_the_backward_curriculum_block_is_flagged():
    """The one sub-block validated a level deeper: it is how pre-
    registered runs are configured, so a silently-ignored knob there
    costs an attended window."""
    prof = {"name": "x", "reinforce": {"backward_curriculum": {
        "states_dir": "d", "count_truncation": True,  # missing the 's'
    }}}
    warnings = validate_profile(prof)
    assert any("count_truncation'" in w for w in warnings), warnings


def test_clean_backward_curriculum_block_passes():
    prof = {"name": "x", "reinforce": {"backward_curriculum": {
        "states_dir": "d", "count_truncations": True,
        "rung_step_budget": {"base": 600, "per_entry": 2.0},
    }}}
    assert validate_profile(prof) == []


def test_carrying_both_truncation_names_is_flagged():
    """Only `count_truncations` is read, so the alias is inert — the
    parsed-but-does-nothing class, one level down."""
    prof = {"name": "x", "reinforce": {"backward_curriculum": {
        "count_truncations": True, "truncation_is_failure": False,
    }}}
    assert any("truncation_is_failure" in w for w in validate_profile(prof))


def test_registry_covers_everything_trainer_consumes():
    """Every reinforce key the trainer reads must be registered — a new
    knob added without registration fails here, at the source of the
    parsed-but-inert bug class."""
    consumed = consumed_reinforce_keys_from_source(
        REPO / "src/training/trainer.py"
    )
    missing = consumed - KNOWN_REINFORCE_KEYS
    assert not missing, (
        f"trainer consumes reinforce keys not in the schema registry "
        f"(register them in config_schema.KNOWN_REINFORCE_KEYS): "
        f"{sorted(missing)}"
    )


def test_registry_covers_the_backward_curriculum_sub_block():
    """Same guard, one level deeper: a new curriculum knob cannot ship
    unregistered (and therefore un-typo-checked)."""
    consumed = consumed_backward_curriculum_keys_from_source(
        REPO / "src/training/trainer.py"
    )
    assert consumed, "the sub-block scanner found nothing — regex rotted?"
    missing = consumed - KNOWN_BACKWARD_CURRICULUM_KEYS
    assert not missing, (
        f"trainer consumes backward_curriculum keys not in the schema "
        f"registry (register them in "
        f"config_schema.KNOWN_BACKWARD_CURRICULUM_KEYS): {sorted(missing)}"
    )


def test_shipped_profiles_pass_strict_validation():
    """Every backward profile must survive `--strict-config`, which is
    mandatory for the pre-registered runs."""
    import yaml

    for path in sorted((REPO / "configs").glob("*backward*.yaml")):
        prof = yaml.safe_load(path.read_text()) or {}
        check_profile(prof, strict=True)
