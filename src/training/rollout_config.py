"""Typed rollout/observation-pipeline config, extracted from the config
parsing block of ``Trainer.__init__`` (src/training/trainer.py).

This is a pure, additive extraction (trainer_decomposition_plan.md, Task
0.5 / §5): ``RolloutConfig.from_profile`` reproduces exactly what
``Trainer.__init__`` currently computes for these fields from a game
profile, so it can be unit-tested standalone before any wiring happens.
It is not yet consumed by the trainer.

Field-for-field mapping to `__init__` (line numbers as of the decomposition
plan's baseline):
    sticky_action_prob <- float(rl_cfg.get("sticky_action_prob", 0.0))
    frame_skip         <- int(game_profile.get("frame_skip", 16))
                           (top-level key, NOT under `reinforce` — the
                           constructor has no `frame_skip` parameter, only
                           the profile default of 16)
    preprocess_f16      <- rl_cfg.get("preprocess_f16", False)
                            (no bool() cast in __init__ — a truthy
                            non-bool profile value passes through as-is)
    encoder_kind        <- str(rl_cfg.get("encoder", "nature_dqn"))
    is_tile_mode        <- encoder_kind in ("smb_tiles", "smb_tiles_pos")
    tile_frame_stack    <- int(rl_cfg.get(
                                "tile_frame_stack",
                                4 if is_tile_mode else 1))
    tile_hidden_dim     <- int(rl_cfg.get("tile_hidden_dim", 64))
    tile_trunk_dim      <- int(rl_cfg.get("tile_trunk_dim", 32))
    tile_feature_dim    <- 0 unless is_tile_mode, in which case
                            `get_extractor(encoder_kind).feature_dim *
                            max(1, tile_frame_stack)`
    num_instances       <- NOT read from the profile at all — __init__
                            takes it as a constructor parameter
                            (`num_instances: int = 16`) and assigns it
                            verbatim (`self.num_instances = num_instances`).
                            `from_profile` takes only a profile dict, so it
                            reproduces the constructor's own default (16),
                            matching what __init__ computes when no caller
                            overrides it.
    max_episode_steps   <- int(game_profile.get(
                                "max_episode_steps", max_episode_steps))
                            where the constructor parameter
                            `max_episode_steps` itself defaults to 1000.
                            (Note: __init__ additionally clamps this value
                            downward as a memory-safety guard tied to
                            `num_instances` and the encoder's per-step byte
                            size — that clamp is derived safety logic, not
                            profile parsing, and is out of scope here; it
                            never fires for either case this module's test
                            exercises.)

where `rl_cfg = game_profile.get("reinforce", {})`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout/observation-pipeline knobs: sticky actions, frame skip,
    observation dtype, tile-mode dispatch + dims, and the per-env/episode
    budget (see trainer.py's `rl_cfg = game_profile.get("reinforce", {})`
    block)."""

    sticky_action_prob: float = 0.0
    frame_skip: int = 16
    preprocess_f16: bool = False
    encoder_kind: str = "nature_dqn"
    is_tile_mode: bool = False
    tile_frame_stack: int = 1
    tile_hidden_dim: int = 64
    tile_trunk_dim: int = 32
    tile_feature_dim: int = 0
    num_instances: int = 16
    max_episode_steps: int = 1000

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "RolloutConfig":
        rl_cfg = profile.get("reinforce", {})

        sticky_action_prob = float(rl_cfg.get("sticky_action_prob", 0.0))
        frame_skip = int(profile.get("frame_skip", 16))
        preprocess_f16 = rl_cfg.get("preprocess_f16", False)

        encoder_kind = str(rl_cfg.get("encoder", "nature_dqn"))
        is_tile_mode = encoder_kind in ("smb_tiles", "smb_tiles_pos")

        tile_frame_stack = int(
            rl_cfg.get("tile_frame_stack", 4 if is_tile_mode else 1)
        )
        tile_hidden_dim = int(rl_cfg.get("tile_hidden_dim", 64))
        tile_trunk_dim = int(rl_cfg.get("tile_trunk_dim", 32))

        tile_feature_dim = 0
        if is_tile_mode:
            from src.emulation.tile_observations import get_extractor

            extractor = get_extractor(encoder_kind)
            base_dim = extractor.feature_dim
            tile_feature_dim = base_dim * max(1, tile_frame_stack)

        # Not sourced from the profile — see the module docstring.
        num_instances = 16

        max_episode_steps = int(profile.get("max_episode_steps", 1000))

        return cls(
            sticky_action_prob=sticky_action_prob,
            frame_skip=frame_skip,
            preprocess_f16=preprocess_f16,
            encoder_kind=encoder_kind,
            is_tile_mode=is_tile_mode,
            tile_frame_stack=tile_frame_stack,
            tile_hidden_dim=tile_hidden_dim,
            tile_trunk_dim=tile_trunk_dim,
            tile_feature_dim=tile_feature_dim,
            num_instances=num_instances,
            max_episode_steps=max_episode_steps,
        )
