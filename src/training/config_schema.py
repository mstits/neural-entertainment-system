"""Profile config validation — catch the silent-typo / dead-knob class.

A misspelled reinforce key (``entropy_flooor: 0.5``) is silently
ignored today: ``rl_cfg.get("entropy_floor", default)`` returns the
default and the intended setting never takes effect — a whole class of
"why isn't my knob doing anything" bugs (the review counted ~193
unregistered/dead knobs; sticky_action_prob and eval --start-state were
each parsed-but-inert for weeks).

This validator holds the registry of reinforce keys the trainer
actually consumes and flags any key in a profile's ``reinforce`` block
that is not registered. It is intentionally a WARN-by-default (a
genuinely new knob should not hard-block a run) with a ``strict`` mode
that raises — used by the test suite and available via CLI so CI-less
local runs can opt into hard rejection.

Keeping this registry in sync is enforced by test_config_schema.py,
which re-derives the consumed set from the trainer source and asserts
it is a subset of the registry (a newly-consumed key must be
registered, or the test fails).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Reinforce keys the trainer consumes. Kept in sync with trainer.py by
# test_config_schema.py (which parses rl_cfg.get(...) sites). Sub-block
# dict keys (consolidate_level.*, cold_eval.*, etc.) are validated only
# at the top level here; their internals are owned by their own readers.
KNOWN_REINFORCE_KEYS: frozenset[str] = frozenset({
    "advance", "asm_bulk_cycles", "async_pipeline", "autocast_fp16",
    "batched_render", "bc_demo_path", "bc_epochs", "bc_replay_enabled",
    "bc_replay_epochs", "bc_replay_every_gens", "bc_replay_max_buffer",
    "bc_replay_train_window", "cold_eval", "consolidate",
    "consolidate_level", "demo_anchor_coef", "demo_anchor_coef_final",
    "demo_anchor_decay_iters", "demo_anchor_decay_start",
    "demo_anchor_enabled", "demo_anchor_margin", "demo_anchor_minibatch",
    "demo_anchor_paths", "device", "drq_aug", "drq_pad", "enabled",
    "encoder", "entropy_coef", "entropy_coef_max", "entropy_floor",
    "cgsa", "episodes_per_genome", "freeze_pre_ppo_elite", "gae_lambda", "gamma",
    "go_explore", "go_explore_fallback", "grad_clip",
    "gx_count_bonus_coef", "inherit_curriculum_on_fresh", "layernorm",
    "lr", "max_steps_per_traj", "num_envs", "num_instances",
    "pace_multiplier", "panic_isolation", "plr_enabled", "ppo_clip_eps",
    "ppo_minibatch_size", "preprocess_f16", "preserve_elite_diversity",
    "prmdp", "recurrent", "recurrent_env_minibatch", "rnd_intrinsic_coef",
    "rnd_loss_coef", "rnd_predictor_update_fraction", "rollout_steps",
    "sam_rho", "smb_curriculum", "steps", "sticky_action_prob", "substage_ladder",
    "symlog_rewards", "tile_frame_stack", "tile_hidden_dim", "tile_trunk_dim",
    "top_k", "torch_compile",
    "trainer_mode", "value_coef", "value_loss", "vmap_forward",
    "warm_start", "warmup_gens_ga_only", "wavefront_reward",
})

# Top-level profile keys read outside `reinforce` — by the launcher,
# the GA path (`ga_params`), the Dreamer path (`dreamer`), and the
# composite-eval manifest family (`levels`, `hysteresis_k`).
KNOWN_TOP_KEYS: frozenset[str] = frozenset({
    "name", "game", "rom_path", "rom", "rom_hashes", "expected_md5",
    "rom_md5", "description", "frame_skip", "max_episode_steps",
    "start_state_path", "start_state", "ram_mapping", "reinforce",
    "reward_weights", "action_space", "curriculum", "env_spec", "seed",
    "ga_params", "dreamer", "levels", "hysteresis_k", "stop_after_worlds",
    "plr_levels",
})


class ConfigSchemaError(ValueError):
    """Raised in strict mode when a profile has unregistered keys."""


def validate_profile(profile: dict[str, Any]) -> list[str]:
    """Return a list of human-readable warnings for unregistered keys.

    Empty list == clean. Does not mutate the profile.
    """
    warnings: list[str] = []
    for k in profile:
        if k not in KNOWN_TOP_KEYS:
            warnings.append(
                f"unknown top-level profile key {k!r} "
                f"(typo? it will be silently ignored)"
            )
    rl = profile.get("reinforce")
    if isinstance(rl, dict):
        for k in rl:
            if k not in KNOWN_REINFORCE_KEYS:
                warnings.append(
                    f"unknown reinforce key {k!r} — NOT consumed by the "
                    f"trainer (typo? renamed? it will be silently ignored)"
                )
    return warnings


def check_profile(profile: dict[str, Any], *, strict: bool = False,
                  logger=None) -> list[str]:
    """Validate and surface warnings (log.warning), or raise in strict.

    Returns the warning list. The trainer launcher calls this at startup
    so a misspelled knob is loud instead of silent.
    """
    warnings = validate_profile(profile)
    if warnings:
        msg = "profile schema: " + "; ".join(warnings)
        if strict:
            raise ConfigSchemaError(msg)
        if logger is not None:
            logger.warning("[config] %s", msg)
        else:
            print(f"[config] WARNING: {msg}")
    return warnings


def consumed_reinforce_keys_from_source(trainer_src: Path) -> set[str]:
    """Re-derive the keys the trainer reads from `reinforce` config, by
    parsing `<cfg>.get("key"...)` sites. Used by the schema test to keep
    KNOWN_REINFORCE_KEYS honest — a newly-consumed key that is not
    registered fails the test."""
    txt = trainer_src.read_text()
    pat = re.compile(
        r'(?:rl_cfg|_rl_cfg|reinforce_cfg|reinforce)\.get\('
        r'\s*["\']([a-z0-9_]+)["\']'
    )
    return set(pat.findall(txt))
