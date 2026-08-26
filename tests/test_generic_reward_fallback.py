"""A profile that declares `reward_id: generic` — or declares nothing —
must construct a working axis-free reward instead of raising.

Originally this file guarded a different fix: `build_reward` returned
None for any game without a hand-authored reward and the Rust factory
turned that into a hard ValueError at trainer build. The fallback is now
reached by DECLARATION (`reward_id: generic`, or omitting the key)
rather than by falling off the end of a substring match chain, so the
fixtures below name the id.

`test_known_id_still_dispatches_specifically` is a rewrite. It used to
build a profile named "super mario bros" and assert only
`isinstance(reward, float)` / `isinstance(done, bool)` — which passes
identically under GenericReward, so it certified nothing about dispatch
from the day it was written. It now asserts the arm.
"""

from __future__ import annotations

import pytest

from src.utils.reward_functions import build_reward_function


def test_profile_with_no_reward_id_builds_generic_reward() -> None:
    rf = build_reward_function({"name": "galaga", "reward_weights": {}})
    assert rf.kind == "generic"
    ram = bytes(2048)
    reward, done, _level_id = rf.compute(ram, action=0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_axis_free_games_build_bespoke_rewards() -> None:
    # Tetris + Bubble Bobble have their own axis-free rewards (with real
    # win predicates), not the meaningless generic fallback.
    for reward_id in ("tetris", "bubble_bobble"):
        rf = build_reward_function({"name": "n", "reward_id": reward_id})
        assert rf.kind == reward_id
        reward, done, _level_id = rf.compute(bytes(2048), action=0)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


def test_generic_reward_weights_are_configurable() -> None:
    # Weights from the profile flow through to the generic reward.
    rf = build_reward_function(
        {
            "name": "some new game",
            "reward_id": "generic",
            "reward_weights": {"time_penalty": -0.5},
        }
    )
    assert rf.kind == "generic"
    ram = bytes(2048)
    # Warmup steps suppress signal early; just assert it computes finitely.
    for _ in range(4):
        reward, _done, _lvl = rf.compute(ram, action=0)
        assert isinstance(reward, float)


def test_known_id_still_dispatches_specifically() -> None:
    """The default must not shadow a declared hand-authored arm.

    Asserting `rf.kind` rather than the type of the returned reward:
    every arm returns a float, so a type check here could never tell
    MarioReward from GenericReward and never did.
    """
    rf = build_reward_function({"name": "anything", "reward_id": "mario"})
    assert rf.kind == "mario"
    reward, done, level_id = rf.compute(bytes(2048), action=0)
    assert isinstance(reward, float)
    # Only MarioReward reports a world-level id in this "W-L" shape;
    # GenericReward returns "stage_N".
    assert level_id == "1-1"


def test_a_display_name_alone_never_reaches_a_hand_authored_arm() -> None:
    rf = build_reward_function({"name": "super mario bros", "reward_weights": {}})
    assert rf.kind == "generic"
    assert rf.compute(bytes(2048), action=0)[2] != "1-1"


def test_a_typo_in_reward_id_raises_instead_of_downgrading() -> None:
    with pytest.raises(ValueError):
        build_reward_function({"name": "n", "reward_id": "generci"})
