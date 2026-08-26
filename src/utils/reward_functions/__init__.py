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

Which reward a profile gets is decided by ONE thing: its top-level
`reward_id` key. The valid ids are `nes_core.reward_ids()`. A profile
that declares none gets `generic` — an axis-free reward (RAM-churn
motion, survival, auto-detected score bytes) that reads no
hand-authored address and carries no win predicate, so a profile can
never inherit a clear it did not ask for. An id outside the table is a
`ValueError`, not a silent downgrade.

The display name selects nothing. It used to, by case-insensitive
substring, which handed configs/legend_of_zelda.yaml — 31 lines, no
reward weights, no addresses — Zelda's quarantined win predicate purely
because its title contains "Zelda", while withholding Mario's reward
from configs/smb_4_4_micro.yaml because its title does not contain
"mario".
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
    """Factory — maps profile["reward_id"] to a concrete
    `nes_core.RewardFunction`. A missing / empty id resolves to
    `generic`; an id outside `nes_core.reward_ids()` raises `ValueError`
    naming the valid set. Raises `ImportError` if the Rust wheel isn't
    available."""
    import nes_core

    return nes_core.build_reward_function(game_profile)
