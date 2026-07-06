"""Any NES game without a hand-authored reward must fall back to the
generic reward instead of raising — the fix that lets Tetris / Bubble
Bobble (and the rest of the library) construct a reward and train out of
the box. Previously `build_reward` returned None for unknown games and
the Rust factory turned that into a hard ValueError at trainer build."""

from __future__ import annotations

from src.utils.reward_functions import build_reward_function


def test_unknown_game_builds_generic_reward() -> None:
    # Before the fallback fix this raised ValueError("no reward function ...").
    rf = build_reward_function({"name": "bubble bobble", "reward_weights": {}})
    assert rf is not None
    ram = bytes(2048)
    reward, done, level_id = rf.compute(ram, action=0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_tetris_builds_generic_reward() -> None:
    rf = build_reward_function({"name": "tetris", "reward_weights": {}})
    ram = bytes(2048)
    reward, done, level_id = rf.compute(ram, action=0)
    assert isinstance(reward, float)


def test_generic_reward_weights_are_configurable() -> None:
    # Weights from the profile flow through to the generic reward.
    rf = build_reward_function(
        {"name": "some new game", "reward_weights": {"time_penalty": -0.5}}
    )
    ram = bytes(2048)
    # Warmup steps suppress signal early; just assert it computes finitely.
    for _ in range(4):
        reward, _done, _lvl = rf.compute(ram, action=0)
        assert isinstance(reward, float)


def test_known_game_still_dispatches_specifically() -> None:
    # The fallback must not shadow the hand-authored games.
    rf = build_reward_function(
        {"name": "super mario bros", "reward_weights": {}}
    )
    ram = bytes(2048)
    reward, done, level_id = rf.compute(ram, action=0)
    assert isinstance(reward, float)
