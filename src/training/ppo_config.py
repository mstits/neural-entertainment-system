"""Typed PPO/REINFORCE hyperparameter config, extracted from the config
parsing block of ``Trainer.__init__`` (src/training/trainer.py).

This is a pure, additive extraction (trainer_decomposition_plan.md, Task
0.5 / §5): ``PPOConfig.from_profile`` reproduces exactly what
``Trainer.__init__`` currently computes for these ten fields from a game
profile's ``reinforce`` block, so it can be unit-tested standalone before
any wiring happens. It is not yet consumed by the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int
    reinforce_steps: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    value_coef: float
    value_loss_kind: str
    ppo_minibatch_size: int
    grad_clip: float
    lr: float

    def __post_init__(self) -> None:
        if self.value_loss_kind not in ("mse", "huber", "smooth_l1"):
            raise ValueError(
                f"reinforce.value_loss must be 'mse' or 'huber'/'smooth_l1', "
                f"got {self.value_loss_kind!r}"
            )

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "PPOConfig":
        rl_cfg = profile.get("reinforce", {})
        return cls(
            rollout_steps=int(rl_cfg.get("rollout_steps", 512)),
            reinforce_steps=rl_cfg.get("steps", 3),
            gamma=rl_cfg.get("gamma", 0.99),
            gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
            clip_eps=rl_cfg.get("ppo_clip_eps", 0.2),
            value_coef=float(rl_cfg.get("value_coef", 0.5)),
            value_loss_kind=str(rl_cfg.get("value_loss", "mse")).lower(),
            ppo_minibatch_size=int(rl_cfg.get("ppo_minibatch_size", 256)),
            grad_clip=rl_cfg.get("grad_clip", 1.0),
            lr=rl_cfg.get("lr", 1e-4),
        )
