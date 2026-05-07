"""Tests for cause-of-death differentiation in MarioReward.

Each death now contributes a base `death` penalty plus optionally
one or both of:
  * `pit_fall` — Mario was airborne with Y > 200 at death (fell)
  * `enemy_death` — an active enemy slot was within 16px of Mario at death

These give PPO a sharper gradient than a single undifferentiated
death penalty. Tests pin each branch fires correctly and the
breakdown includes the right components.
"""

from __future__ import annotations

import nes_core


PLAYER_STATE_DYING = 0x0B


def _profile() -> dict:
    return {
        "name": "mario",
        "reward_weights": {
            "forward_progress": 0.0,
            "score_delta": 0.0,
            "checkpoint_scale": 0.0,
            "completion_bonus": 0.0,
            "time_penalty": 0.0,
            "death_penalty": -15.0,
            "pit_fall_extra": -20.0,
            "enemy_death_extra": -10.0,
        },
    }


def _ram_alive(x: int = 100, y: int = 100, on_ground: bool = True) -> bytearray:
    """Minimal RAM for a living Mario at the given (X, Y) screen position."""
    buf = bytearray(2048)
    buf[0x006D] = (x >> 8) & 0xFF
    buf[0x0086] = x & 0xFF
    buf[0x00CE] = y
    buf[0x001D] = 0 if on_ground else 1  # 0 = on ground
    buf[0x000E] = 0  # alive
    buf[0x075A] = 3  # 3 lives remaining
    return buf


def _trigger_death(buf: bytearray, *, mario_y: int = 100, on_ground: bool = True) -> bytearray:
    """Flip Mario into the dying player state at the given Y."""
    buf[0x000E] = PLAYER_STATE_DYING
    buf[0x00CE] = mario_y
    buf[0x001D] = 0 if on_ground else 1
    return buf


def test_baseline_death_only_no_extras() -> None:
    """Mario dies on solid ground at low Y with no enemies nearby —
    only the base death penalty fires; neither extra triggers."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    # Step 1: Mario alive — establishes prev_lives = 3.
    r.compute(bytes(_ram_alive(x=100, y=100, on_ground=True)))
    # Step 2: Mario dies on ground at low Y.
    r.compute(bytes(_trigger_death(_ram_alive(x=100), mario_y=100, on_ground=True)))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert "pit_fall" not in bd
    assert "enemy_death" not in bd


def test_pit_fall_fires_when_dying_airborne_at_high_y() -> None:
    """Mario dying with Y > 200 AND on_ground=False is a pit fall.
    Both `death` and `pit_fall` contribute to the breakdown."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive()))
    r.compute(bytes(_trigger_death(_ram_alive(), mario_y=220, on_ground=False)))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert bd.get("pit_fall", 0) == -20.0
    # No enemies nearby in this fixture — enemy_death must NOT fire.
    assert "enemy_death" not in bd


def test_pit_fall_does_not_fire_on_ground_even_at_high_y() -> None:
    """On_ground=True at the moment of death suppresses pit_fall —
    Mario can't have fallen into a pit if his feet are touching ground.
    (Edge case: Mario dies the frame he LANDS at the bottom of a pit;
    the gate keeps things conservative — better false negatives than
    over-firing on Goomba contacts at low Y.)"""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive()))
    r.compute(bytes(_trigger_death(_ram_alive(), mario_y=220, on_ground=True)))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert "pit_fall" not in bd


def test_enemy_death_fires_when_active_enemy_within_one_tile() -> None:
    """An active enemy slot within ±16 px of Mario in both X and Y
    triggers the enemy_death extra. Mario at (100, 100); enemy at
    (108, 108) — well within the 16px radius."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive(x=100, y=100)))
    buf = _ram_alive(x=100, y=100)
    # Activate enemy slot 0 at (108, 108)
    buf[0x000F] = 1
    buf[0x006E] = 0
    buf[0x0087] = 108
    buf[0x00CF] = 108
    _trigger_death(buf, mario_y=100, on_ground=True)
    r.compute(bytes(buf))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert bd.get("enemy_death", 0) == -10.0
    # On_ground at low Y — pit_fall must NOT fire.
    assert "pit_fall" not in bd


def test_enemy_death_skips_inactive_slots() -> None:
    """Enemy slot bytes set but flag=0 means inactive — should not
    trigger enemy_death."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive()))
    buf = _ram_alive()
    # Slot 0 inactive (flag=0) but position bytes set
    buf[0x000F] = 0
    buf[0x006E] = 0
    buf[0x0087] = 100
    buf[0x00CF] = 100
    _trigger_death(buf, mario_y=100, on_ground=True)
    r.compute(bytes(buf))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert "enemy_death" not in bd


def test_pit_and_enemy_can_stack() -> None:
    """If Mario falls into a pit AND an enemy is within range
    (rare but possible — falling onto a flying Bullet Bill etc),
    both extras fire."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive(x=100, y=100)))
    buf = _ram_alive(x=100, y=220)
    buf[0x001D] = 1  # airborne
    # Enemy adjacent
    buf[0x000F] = 1
    buf[0x006E] = 0
    buf[0x0087] = 108
    buf[0x00CF] = 220
    _trigger_death(buf, mario_y=220, on_ground=False)
    r.compute(bytes(buf))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert bd.get("pit_fall", 0) == -20.0
    assert bd.get("enemy_death", 0) == -10.0


def test_death_only_fires_once_across_subsequent_steps() -> None:
    """The dying state can persist for multiple frames as the death
    animation plays; the reward should NOT keep adding -15 each frame.
    `self.died` flag gates the entire death-detection block."""
    r = nes_core.build_reward_function(_profile())
    r.reset()
    r.compute(bytes(_ram_alive()))
    death_buf = bytes(_trigger_death(_ram_alive(), mario_y=100, on_ground=True))
    r.compute(death_buf)
    r.compute(death_buf)  # same dying state — should NOT re-fire
    r.compute(death_buf)
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0  # still just one


def test_extras_default_to_zero_for_backward_compat() -> None:
    """A profile that doesn't specify the new keys should not emit
    pit_fall or enemy_death components — preserves old behavior for
    every game profile that hasn't been migrated."""
    profile = {
        "name": "mario",
        "reward_weights": {
            "forward_progress": 0.0,
            "score_delta": 0.0,
            "checkpoint_scale": 0.0,
            "completion_bonus": 0.0,
            "time_penalty": 0.0,
            "death_penalty": -15.0,
            # Note: pit_fall_extra / enemy_death_extra NOT set
        },
    }
    r = nes_core.build_reward_function(profile)
    r.reset()
    r.compute(bytes(_ram_alive()))
    buf = _ram_alive(x=100, y=220)
    buf[0x001D] = 1  # airborne
    buf[0x000F] = 1
    buf[0x006E] = 0
    buf[0x0087] = 108
    buf[0x00CF] = 220
    _trigger_death(buf, mario_y=220, on_ground=False)
    r.compute(bytes(buf))
    bd = dict(r.breakdown)
    assert bd.get("death", 0) == -15.0
    assert "pit_fall" not in bd
    assert "enemy_death" not in bd
