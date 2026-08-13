"""Entropy-control config — a pure, typed slice of trainer.__init__.

Mirrors the exact `reinforce.*` reads `NESTrainer.__init__` performs for
the entropy-bonus knobs (trainer.py, roughly lines 928-943). This module
is additive only: it does not wire into the trainer yet (see
docs/proposals/trainer_decomposition_plan.md Task 0.5). `__init__` keeps
computing these fields itself for now; a later, separate step swaps its
assignments for `EntropyControlConfig.from_profile(game_profile)`.

Field-for-field mapping to `__init__`:
    entropy_coef      <- rl_cfg.get("entropy_coef", 0.01)
    entropy_floor     <- float(rl_cfg.get("entropy_floor", 0.0))
    entropy_coef_max  <- float(rl_cfg.get("entropy_coef_max", 0.05))
    entropy_coef_base <- self.entropy_coef  (i.e. the freshly-parsed
                          entropy_coef, BEFORE any per-iteration schedule
                          mutates it — trainer.py's private
                          `self._entropy_coef_base`, the floor the
                          entropy-floor controller decays back toward)

where `rl_cfg = game_profile.get("reinforce", {})`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntropyControlConfig:
    """Entropy-bonus knobs consumed by the PPO update and the adaptive
    entropy-floor controller (see trainer.py `_entropy_floor_step`)."""

    entropy_coef: float
    entropy_floor: float
    entropy_coef_max: float
    entropy_coef_base: float

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "EntropyControlConfig":
        rl_cfg = profile.get("reinforce", {})
        entropy_coef = rl_cfg.get("entropy_coef", 0.01)
        entropy_floor = float(rl_cfg.get("entropy_floor", 0.0))
        entropy_coef_max = float(rl_cfg.get("entropy_coef_max", 0.05))
        return cls(
            entropy_coef=entropy_coef,
            entropy_floor=entropy_floor,
            entropy_coef_max=entropy_coef_max,
            entropy_coef_base=entropy_coef,
        )
