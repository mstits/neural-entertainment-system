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
    KNOWN_REINFORCE_KEYS,
    ConfigSchemaError,
    check_profile,
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
