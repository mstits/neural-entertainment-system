"""
Reward-function factory. Thin dispatcher to `nes_core.build_reward_function`.

Per-game logic lives in `nes_core/src/rewards.rs` — one single file
with a Rust struct per game, exposing a byte-identical contract to the
historical Python classes that used to live here:

    reset()                                  -> None
    compute(ram: bytes, *, action: int = 0)  -> (reward: float, done: bool, level_id: str)
    episode_success()                        -> bool
    .breakdown                               -> dict[str, float]  (cumulative by signal)

Parity verified via `scripts/test_rewards_parity.py` on 500-step random-
RAM sequences per game. The Rust hot path is ~8.6x faster than the
Python it replaced.

Games with hand-authored rewards: Zelda, Mario, Contra, Mega Man,
Castlevania, Metroid (substring match on `profile["name"]`). Any other
game falls back to a generic axis-free reward (RAM-churn motion,
survival, auto-detected score bytes) so it can train out of the box.
"""

from __future__ import annotations

from typing import Protocol


class RewardFunction(Protocol):
    """Structural type for a reward function. Implemented in Rust by
    `nes_core::rewards::Reward` via the PyO3 `nes_core.RewardFunction`
    class."""

    def reset(self) -> None: ...

    def compute(self, ram: bytes, *, action: int = 0) -> tuple[float, bool, str]: ...

    def episode_success(self) -> bool: ...


__all__ = ["RewardFunction", "build_reward_function"]


def build_reward_function(game_profile: dict) -> RewardFunction:
    """Factory — maps profile["name"] (case-insensitive substring) to
    a concrete `nes_core.RewardFunction`, falling back to the generic
    reward for any game without a hand-authored one. Raises `ImportError`
    if the Rust wheel isn't available."""
    import nes_core

    return nes_core.build_reward_function(game_profile)
