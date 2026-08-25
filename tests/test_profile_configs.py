"""Schema smoke-test for the per-game profile YAMLs.

Every top-level config in `configs/` is loaded with PyYAML and any
`action_space` block is verified to use the list-of-lists shape the
trainer expects. The trainer iterates each entry character-by-
character if it's a bare string, so an `- right+A` typo silently
explodes downstream with `KeyError: 'r'`. Catching it here at
collection time turns a runtime crash into a unit-test failure.

Auto-generated stubs under `configs/auto/` and partial overlays
under `configs/overrides/` are excluded — they're not standalone
profiles and many lack `action_space` by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _top_level_yaml_profiles() -> list[Path]:
    """Return every profile YAML at the top of `configs/` (no recursion).

    Only top-level files are full standalone profiles; the `auto/`
    subdirectory is auto-generated stubs and `overrides/` holds
    overlays that get merged onto a base profile at load time.
    """
    return sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("profile_path", _top_level_yaml_profiles(), ids=lambda p: p.name)
def test_profile_yaml_loads_and_action_space_is_list_of_lists(profile_path: Path) -> None:
    """Every top-level profile must parse + (if it declares an
    action_space) have each entry be a list of button-name strings,
    not a bare string."""
    with profile_path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{profile_path.name}: root is not a mapping"

    action_space = data.get("action_space")
    if action_space is None:
        # Overlay/partial profiles legitimately omit action_space.
        pytest.skip(f"{profile_path.name} has no action_space block")

    assert isinstance(action_space, list), (
        f"{profile_path.name}: action_space must be a list, "
        f"got {type(action_space).__name__}"
    )
    for idx, entry in enumerate(action_space):
        assert not isinstance(entry, str), (
            f"{profile_path.name}: action_space[{idx}] is a bare string "
            f"({entry!r}); the trainer iterates it char-by-char and "
            f"crashes with KeyError. Wrap it as a list: [\"{entry}\"]."
        )
        assert isinstance(entry, list), (
            f"{profile_path.name}: action_space[{idx}] must be a list "
            f"(or [] for NOOP), got {type(entry).__name__}"
        )
        for j, btn in enumerate(entry):
            assert isinstance(btn, str), (
                f"{profile_path.name}: action_space[{idx}][{j}] "
                f"must be a button-name string, got {btn!r}"
            )


def _gui_offered_profiles() -> list[Path]:
    """Every top-level `configs/*.yaml` the GUI profile dropdown offers.

    Mirrors the discovery rule in `main_window._populate_profiles`
    exactly: a top-level YAML that parses to a mapping with a *truthy*
    `action_space`. Override-only overlays (no action_space) are hidden
    by the GUI so a user can't select one and crash the trainer with an
    empty action_space — they're excluded here too.
    """
    offered: list[Path] = []
    for path in _top_level_yaml_profiles():
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and data.get("action_space"):
            offered.append(path)
    return offered


def _trainer_boot_problems(profile: dict) -> list[str]:
    """Return the reasons `Trainer.__init__` would reject `profile`, empty
    if it would construct.

    The cheapest possible check of the load-bearing construction contract
    (trainer.py ~840-870) without building a Trainer or touching a ROM:
      * `num_actions = len(action_space)`; an absent/empty action_space
        raises ``ValueError("Game profile must define a non-empty
        action_space")``;
      * `_build_bitmask_table()` folds each entry via
        `action_space_to_bitmasks`, which raises on a bare-string entry
        or an unknown button name.
    """
    from src.training.profile_utils import action_space_to_bitmasks

    action_space = profile.get("action_space", [])
    if not action_space:
        return ["action_space missing or empty (Trainer requires a "
                "non-empty action_space)"]
    try:
        action_space_to_bitmasks(action_space)
    except (ValueError, TypeError) as exc:
        return [f"action_space does not fold to NES bitmasks: {exc}"]
    return []


@pytest.mark.parametrize(
    "profile_path", _gui_offered_profiles(), ids=lambda p: p.name
)
def test_gui_offered_profile_meets_trainer_boot_contract(profile_path: Path) -> None:
    """Every profile the GUI dropdown offers must actually construct a
    Trainer.

    ISSUE-1 guard: `zelda_gui_tuned.yaml` shipped with only reward/GA
    overlay blocks and no action_space, so selecting it and pressing
    Start raised `ValueError: Game profile must define a non-empty
    action_space` from `Trainer.__init__` — it had plausibly never
    booted. If a profile is offered in the picker, it must clear the
    minimum construction contract.
    """
    data = yaml.safe_load(profile_path.read_text())
    problems = _trainer_boot_problems(data)
    assert not problems, f"{profile_path.name}: {'; '.join(problems)}"


def test_zelda_gui_tuned_is_offered_and_boots() -> None:
    """The spectator profile must now be picker-visible AND bootable, with
    its own checkpoint subtree (no collision with zelda.yaml).

    Fails before the ISSUE-1 fix: the old file had no action_space, so
    `_gui_offered_profiles()` never lists it (the GUI hides it).
    """
    from src.training.profile_utils import derive_checkpoint_dir

    path = CONFIG_DIR / "zelda_gui_tuned.yaml"
    assert path in _gui_offered_profiles(), (
        "zelda_gui_tuned.yaml is not offered by the GUI dropdown — it "
        "still lacks a truthy action_space."
    )
    data = yaml.safe_load(path.read_text())
    assert not _trainer_boot_problems(data), _trainer_boot_problems(data)

    base = yaml.safe_load((CONFIG_DIR / "zelda.yaml").read_text())
    base_dir = derive_checkpoint_dir("./checkpoints", base.get("name"))
    tuned_dir = derive_checkpoint_dir("./checkpoints", data.get("name"))
    assert tuned_dir != base_dir, (
        f"spectator profile checkpoint dir {tuned_dir} collides with "
        f"zelda.yaml's {base_dir}; give it a distinct name."
    )


def test_override_only_profile_fails_boot_contract() -> None:
    """Negative case: a profile shaped like the old broken
    zelda_gui_tuned (reward/GA/reinforce overlay, no action_space) must
    be rejected by the boot-contract check, proving the lint above has
    teeth."""
    broken = {
        "reward_weights": {"exploration_bonus": 50.0},
        "ga_params": {"mutation_std": 0.008},
        "reinforce": {"pace_multiplier": 2.0},
    }
    assert _trainer_boot_problems(broken), (
        "a profile with no action_space must fail the boot contract."
    )


def test_vanilla_ppo_profile_declares_existing_start_state() -> None:
    """Regression guard: the vanilla_ppo SMB profile MUST declare a
    start_state_path pointing to a real file.

    Without it the emulator cold-boots to the title screen, where the
    attract-mode demo auto-plays and IGNORES controller input — every
    env runs the identical scripted sequence, the agent never controls
    Mario, and the policy gets zero learning signal. This silently
    wasted entire training runs (it presented as "PPO won't learn /
    entropy pinned at max"). Pin the start state so it can't regress.
    """
    profile_path = CONFIG_DIR / "mario_vanilla_ppo.yaml"
    with profile_path.open() as fh:
        data = yaml.safe_load(fh)
    ssp = data.get("start_state_path")
    assert ssp, (
        "mario_vanilla_ppo.yaml must declare start_state_path — without "
        "it training cold-boots to the title-screen demo (inputs ignored)."
    )
    assert (Path(__file__).resolve().parents[1] / ssp).exists(), (
        f"start_state_path {ssp!r} does not exist; training would fall "
        f"back to the title-screen demo."
    )


def test_mario_1_2_phase3_masked_hazard_veto_stays_disabled() -> None:
    """Regression guard: the falsified hazard-veto must stay disarmed.

    docs/research/PHASE3_HAZARD_VETO_NEGATIVE_2026-08-22.md is a
    pre-registered, two-seed, >=100-episode negative: this exact
    checkpoint (runs/engine/hazard/hazard_model.pt, gate c_index
    0.9170 PASS) vetoed 77.4% of the control's chosen actions in
    veto-active states and collapsed the honest clear rate from 31/100
    to 0/100. The standing instruction is "no hazard-veto revival"
    against this checkpoint absent a new, freshly pre-registered
    experiment. Nothing in config_schema.py blocks enabled: true at
    load time, so this config file is the only guard — this test
    fails loudly if anyone flips the flag back on.
    """
    profile_path = CONFIG_DIR / "mario_1_2_phase3_masked.yaml"
    with profile_path.open() as fh:
        data = yaml.safe_load(fh)
    hazard_mask = ((data.get("reinforce") or {}).get("hazard_mask")) or {}
    assert hazard_mask.get("enabled") is not True, (
        "mario_1_2_phase3_masked.yaml re-enables reinforce.hazard_mask "
        "against the falsified checkpoint (c_index 0.9170, cited in "
        "PHASE3_HAZARD_VETO_NEGATIVE_2026-08-22.md as the model that "
        "collapsed the honest clear rate 31/100 -> 0/100). This is a "
        "standing-instruction violation, not a config choice — revert."
    )


def test_mario_1_2_phase3_masked_profile_still_boots_with_veto_disabled() -> None:
    """The hazard-veto fix must not regress the profile's ability to
    construct a Trainer (hazard_mask is disabled, not deleted, so the
    key must still be tolerated by the boot-contract check)."""
    profile_path = CONFIG_DIR / "mario_1_2_phase3_masked.yaml"
    data = yaml.safe_load(profile_path.read_text())
    assert not _trainer_boot_problems(data), _trainer_boot_problems(data)
