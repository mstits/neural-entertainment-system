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
    KNOWN_CONSOLIDATE_LEVEL_KEYS,
    KNOWN_CONSOLIDATE_PROBE_KEYS,
    KNOWN_REINFORCE_KEYS,
    ConfigSchemaError,
    check_profile,
    consumed_backward_curriculum_keys_from_source,
    consumed_consolidate_level_keys_from_source,
    consumed_consolidate_probe_keys_from_source,
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


def test_episode_metrics_key_is_registered():
    """episode_metrics gates the per-episode metrics sidecar (trainer.py
    ``_rl_cfg.get("episode_metrics", False)``) — registering it here is
    what test_registry_covers_everything_trainer_consumes enforces."""
    prof = {"name": "x", "reinforce": {"episode_metrics": True}}
    assert validate_profile(prof) == []


def test_tensorboard_top_level_key_passes_strict():
    """`tensorboard` is a real top-level knob (trainer.py reads it via
    `game_profile.get("tensorboard", True)` to gate MetricsSink's
    TensorBoard writer) — it must not be flagged as an unknown key, or
    `--strict-config` aborts a launch over a knob that is neither a typo
    nor ignored."""
    prof = {"name": "x", "tensorboard": False}
    assert validate_profile(prof) == []
    check_profile(prof, strict=True)


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


def test_typo_in_the_consolidate_level_block_is_flagged():
    """The gate block is the other one a pre-registered run is configured
    through, and B6 makes the cost of a silent typo concrete: a `sticky_probb`
    would leave the probe deterministic — the exact v4 defect — while the
    profile looked configured."""
    prof = {"name": "x", "reinforce": {"consolidate_level": {
        "target": "1-1", "accept_barr": 0.5,   # missing the second 'a'
    }}}
    warnings = validate_profile(prof)
    assert any("accept_barr" in w for w in warnings), warnings


def test_typo_in_the_consolidate_probe_sub_block_is_flagged():
    prof = {"name": "x", "reinforce": {"consolidate_level": {
        "target": "1-1", "probe": {"episodes": 30, "sticky_probb": 0.25},
    }}}
    warnings = validate_profile(prof)
    assert any("sticky_probb" in w for w in warnings), warnings
    assert any("consolidate_level.probe" in w for w in warnings), warnings


def test_clean_consolidate_level_block_passes():
    prof = {"name": "x", "reinforce": {"consolidate_level": {
        "enabled": True, "target": "1-1", "protect": [], "accept_bar": 0.5,
        "accept_probes": 3, "use_wilson_bound": True,
        "wilson_confidence": 0.95, "accept_rule": "wilson_lb_gt_best_point",
        "schedule": {"entropy": {"from": 0.01, "to": 0.002, "iters": 80}},
        "probe": {"every": 12, "episodes": 30, "max_steps": 1500,
                  "sticky_prob": 0.25, "start_jitter": 16, "eval_seed": 7,
                  "eval_workers": 8},
    }}}
    assert validate_profile(prof) == []


def test_unknown_accept_rule_is_flagged():
    prof = {"name": "x", "reinforce": {"consolidate_level": {
        "target": "1-1", "accept_rule": "wilson",
    }}}
    assert any("accept_rule" in w for w in validate_profile(prof))


def test_registry_covers_the_consolidate_level_sub_block():
    consumed = consumed_consolidate_level_keys_from_source(
        REPO / "src/training/trainer.py"
    )
    assert consumed, "the consolidate_level scanner found nothing — regex rotted?"
    missing = consumed - KNOWN_CONSOLIDATE_LEVEL_KEYS
    assert not missing, (
        f"trainer consumes consolidate_level keys not in the schema registry "
        f"(register them in config_schema.KNOWN_CONSOLIDATE_LEVEL_KEYS): "
        f"{sorted(missing)}"
    )


def test_registry_covers_the_consolidate_probe_sub_block():
    """The probe block is resolved in ONE place (`resolve_probe_settings`)
    precisely so the gate's baseline probe and its per-cycle probe cannot
    drift apart the way v4's did — so that is where the registry is checked."""
    consumed = consumed_consolidate_probe_keys_from_source(
        REPO / "src/training/oneshot_curriculum.py"
    )
    assert consumed, "the probe scanner found nothing — regex rotted?"
    missing = consumed - KNOWN_CONSOLIDATE_PROBE_KEYS
    assert not missing, (
        f"the probe resolver reads keys not in the schema registry "
        f"(register them in config_schema.KNOWN_CONSOLIDATE_PROBE_KEYS): "
        f"{sorted(missing)}"
    )


def test_shipped_profiles_pass_strict_validation():
    """Every backward profile must survive `--strict-config`, which is
    mandatory for the pre-registered runs."""
    import yaml

    for path in sorted((REPO / "configs").glob("*backward*.yaml")):
        prof = yaml.safe_load(path.read_text()) or {}
        check_profile(prof, strict=True)


def test_solve_profile_known_keys():
    """Every shipped `solve:` block validates clean: `solve` itself is a
    known top-level key and every key inside it is registered.

    9 of the 45 solve configs also carry an unrelated pre-existing
    unregistered top-level key (purity-quarantine machinery, out of
    scope for this registry) - excluded generically by pattern rather
    than by name, since naming it here would itself trip the source
    sweep that key's own guard runs. `solve` is exempted from that
    exclusion so a regression in its own registration still fails this
    test, and every other kind of warning still does too.
    """
    import re

    import yaml

    top_level = re.compile(r"^unknown top-level profile key '([^']+)'")
    for f in Path("configs").glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        if isinstance(d, dict) and isinstance(d.get("solve"), dict):
            warnings = []
            for w in validate_profile(d):
                m = top_level.match(w)
                if m and m.group(1) != "solve":
                    continue
                warnings.append(w)
            assert warnings == [], f"{f}: {warnings}"


def test_solve_typo_is_flagged():
    prof = {"name": "x", "solve": {"rom": "x.nes", "no_clear_predicat": True}}
    assert any("no_clear_predicat" in w for w in validate_profile(prof))
