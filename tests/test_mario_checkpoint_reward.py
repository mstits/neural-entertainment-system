"""Tests for the Mario reward function's dense progress checkpoints."""

from __future__ import annotations

import nes_core


def _profile(checkpoint_scale: float = 1.0) -> dict:
    return {
        "name": "mario",
        "reward_weights": {
            "forward_progress": 1.0,
            "checkpoint_scale": checkpoint_scale,
            "death_penalty": 0.0,
            "time_penalty": 0.0,
            "completion_bonus": 2000.0,
            "score_delta": 0.0,
        },
    }


def _ram_with_x(world_x: int) -> bytes:
    """Build minimal RAM with Mario at the given world X."""
    buf = bytearray(2048)
    buf[0x006D] = (world_x >> 8) & 0xFF
    buf[0x0086] = world_x & 0xFF
    return bytes(buf)


def test_no_checkpoint_below_first_threshold() -> None:
    r = nes_core.build_reward_function(_profile())
    r.reset()
    # Mario at x=300, before the first checkpoint at x=350.
    reward, _, _ = r.compute(_ram_with_x(300))
    assert "checkpoint" not in dict(r.breakdown)


def test_first_checkpoint_fires_at_x350() -> None:
    r = nes_core.build_reward_function(_profile())
    r.reset()
    # First step initializes prev_x. Step Mario to past x=350.
    r.compute(_ram_with_x(0))
    reward, _, _ = r.compute(_ram_with_x(400))
    bd = dict(r.breakdown)
    assert bd.get("checkpoint", 0.0) == 50.0


def test_each_checkpoint_fires_at_most_once_per_episode() -> None:
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(_ram_with_x(0))
    r.compute(_ram_with_x(400))  # fires checkpoint at 350 (+50)
    bd_after_first = dict(r.breakdown)
    r.compute(_ram_with_x(500))  # still past 350, should NOT re-fire
    bd_after_second = dict(r.breakdown)
    assert bd_after_first.get("checkpoint") == bd_after_second.get("checkpoint")


def test_multiple_checkpoints_accumulate() -> None:
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(_ram_with_x(0))
    # Jump straight past first three checkpoints (x=350, 720, 1100).
    r.compute(_ram_with_x(1200))
    bd = dict(r.breakdown)
    # Bonuses 50 + 100 + 150 = 300.
    assert bd["checkpoint"] == 300.0


def test_checkpoint_scale_zero_disables() -> None:
    r = nes_core.build_reward_function(_profile(checkpoint_scale=0.0))
    r.reset()
    r.compute(_ram_with_x(0))
    r.compute(_ram_with_x(3000))  # would fire all checkpoints if enabled
    bd = dict(r.breakdown)
    assert "checkpoint" not in bd or bd["checkpoint"] == 0.0


def test_reset_re_arms_checkpoints() -> None:
    """`reset()` should re-arm all checkpoints so a new episode can
    earn the bonuses again."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(_ram_with_x(0))
    r.compute(_ram_with_x(400))  # fires first checkpoint
    r.reset()
    r.compute(_ram_with_x(0))
    r.compute(_ram_with_x(400))  # should fire again
    bd = dict(r.breakdown)
    assert bd["checkpoint"] == 50.0
