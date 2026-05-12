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
