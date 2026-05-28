"""Unit tests for `src.training.profile_utils`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.profile_utils import (
    BUTTON_NAME_TO_BIT,
    action_space_to_bitmasks,
    derive_checkpoint_dir,
    profile_slug,
    resolve_encoder,
    validate_profile,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Super Mario Bros.", "super_mario_bros"),
        ("The Legend of Zelda", "the_legend_of_zelda"),
        ("Mega Man", "mega_man"),
        ("Contra", "contra"),
        ("Castlevania", "castlevania"),
        ("Metroid", "metroid"),
        ("Mega Man 2", "mega_man_2"),
        ("  spaces  ", "spaces"),
        ("UPPERCASE", "uppercase"),
        ("with_underscores", "with_underscores"),
        ("with-dashes", "with_dashes"),
        ("hyphen-then-space and dot.", "hyphen_then_space_and_dot"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_profile_slug(name: str | None, expected: str) -> None:
    assert profile_slug(name) == expected


def test_derive_checkpoint_dir_default_appends_slug() -> None:
    # Both the legacy `./checkpoints` and the bare `checkpoints` form
    # should trigger the per-game subdir behavior.
    assert derive_checkpoint_dir("./checkpoints", "Super Mario Bros.") == Path(
        "checkpoints/super_mario_bros"
    )
    assert derive_checkpoint_dir("checkpoints", "Zelda") == Path(
        "checkpoints/zelda"
    )


def test_derive_checkpoint_dir_explicit_override_preserved() -> None:
    # An explicit non-default path is honored verbatim — tests and
    # debug runs can still target arbitrary directories.
    p = derive_checkpoint_dir("/tmp/my_test_run", "Super Mario Bros.")
    assert p == Path("/tmp/my_test_run")

    p = derive_checkpoint_dir("./custom_dir", "Zelda")
    assert p == Path("./custom_dir")


def test_derive_checkpoint_dir_handles_missing_profile_name() -> None:
    p = derive_checkpoint_dir("./checkpoints", None)
    assert p == Path("checkpoints/unknown")

    p = derive_checkpoint_dir("./checkpoints", "")
    assert p == Path("checkpoints/unknown")


def test_button_name_to_bit_matches_frame_utils() -> None:
    # The profile_utils map duplicates frame_utils' bit layout for
    # import-lightness. Guard that the two never drift — a mismatch
    # would silently send the wrong controller bits during eval.
    from src.emulation import frame_utils as fu

    assert BUTTON_NAME_TO_BIT["A"] == fu.BUTTON_A
    assert BUTTON_NAME_TO_BIT["B"] == fu.BUTTON_B
    assert BUTTON_NAME_TO_BIT["up"] == fu.BUTTON_UP
    assert BUTTON_NAME_TO_BIT["down"] == fu.BUTTON_DOWN
    assert BUTTON_NAME_TO_BIT["left"] == fu.BUTTON_LEFT
    assert BUTTON_NAME_TO_BIT["right"] == fu.BUTTON_RIGHT
    assert BUTTON_NAME_TO_BIT["start"] == fu.BUTTON_START
    assert BUTTON_NAME_TO_BIT["select"] == fu.BUTTON_SELECT
    assert BUTTON_NAME_TO_BIT["NOOP"] == fu.BUTTON_NOOP


def test_action_space_to_bitmasks_smb() -> None:
    # The canonical SMB action space → the bitmask ints the old
    # eval script hardcoded as SMB_ACTIONS.
    action_space = [
        [],                        # NOOP   0x00
        ["right"],                 #        0x80
        ["right", "A"],            #        0x81
        ["right", "B"],            #        0x82
        ["right", "A", "B"],       #        0x83
        ["A"],                     #        0x01
        ["left"],                  #        0x40
    ]
    assert action_space_to_bitmasks(action_space) == (
        0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40,
    )


def test_action_space_to_bitmasks_matches_trainer_table() -> None:
    # The trainer's _build_bitmask_table now delegates here; verify
    # the delegation produces an identical table for a representative
    # action space (this is the contract the hot path depends on).
    action_space = [[], ["right", "A"], ["left", "B"], ["up"], ["down"]]
    assert action_space_to_bitmasks(action_space) == (
        0x00, 0x81, 0x42, 0x10, 0x20,
    )


def test_action_space_to_bitmasks_rejects_bare_string() -> None:
    with pytest.raises(ValueError, match="is a string"):
        action_space_to_bitmasks([[], "right+A"])


def test_action_space_to_bitmasks_rejects_unknown_button() -> None:
    with pytest.raises(ValueError, match="unknown button"):
        action_space_to_bitmasks([[], ["jump"]])


def test_resolve_encoder_smb_tiles() -> None:
    profile = {"reinforce": {"encoder": "smb_tiles", "tile_frame_stack": 4}}
    extractor, feature_dim, stacked = resolve_encoder(profile)
    assert hasattr(extractor, "extract")
    assert feature_dim == 175
    assert stacked == 700


def test_resolve_encoder_default_stack_size() -> None:
    # Missing tile_frame_stack defaults to 4.
    profile = {"reinforce": {"encoder": "smb_tiles"}}
    _, feature_dim, stacked = resolve_encoder(profile)
    assert stacked == feature_dim * 4


def test_resolve_encoder_missing_encoder_raises() -> None:
    with pytest.raises(ValueError, match="no reinforce.encoder"):
        resolve_encoder({"reinforce": {}})


def test_validate_profile_accepts_well_formed() -> None:
    good = {
        "name": "Super Mario Bros.",
        "action_space": [[], ["right"], ["right", "A"]],
        "reward_weights": {"forward_progress": 1.0},
        "reinforce": {"encoder": "smb_tiles", "lr": 3.0e-4, "gamma": 0.9},
    }
    assert validate_profile(good) == []


def test_validate_profile_flags_each_problem() -> None:
    bad = {
        "name": "",                       # empty name
        "action_space": [[], "right+A"],  # bare-string entry
        "reward_weights": [1, 2],         # not a mapping
        "reinforce": {
            "encoder": 123,               # not a string
            "lr": "3e-4",                 # quoted number
        },
    }
    problems = validate_profile(bad)
    joined = " | ".join(problems)
    assert "name" in joined
    assert "action_space" in joined
    assert "reward_weights" in joined
    assert "reinforce.encoder" in joined
    assert "reinforce.lr" in joined


def test_validate_profile_allows_extra_keys() -> None:
    # Profiles carry game-specific extensions; unknown keys are fine.
    p = {"name": "X", "curriculum": {"stages": []}, "some_game_specific": 7}
    assert validate_profile(p) == []


def test_real_profiles_validate() -> None:
    # Every shipped STANDALONE profile must pass validation. Overlay
    # profiles (merged onto a base at load time, e.g. *_overrides /
    # *_tuned) legitimately omit `name` and are out of scope — they're
    # identified by the absence of a `name` key.
    import yaml
    cfg = Path(__file__).resolve().parents[1] / "configs"
    for prof in cfg.glob("*.yaml"):
        with prof.open() as fh:
            data = yaml.safe_load(fh)
        if not (isinstance(data, dict) and data.get("name")):
            continue  # overlay/partial, not a standalone profile
        assert validate_profile(data) == [], f"{prof.name}: {validate_profile(data)}"
