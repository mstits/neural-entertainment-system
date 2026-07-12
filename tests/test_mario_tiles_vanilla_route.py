"""Guard the SMB tile-mode "route onto the proven advancing path" fix.

mario_tiles.yaml pins `trainer_mode: ga_ppo`, whose curriculum advance
gate averages EVERY genome's episodes against a 0.80 threshold and so can
never fire for a population (see configs/mario_tiles_vanilla.yaml header).
The fix is configs/mario_tiles_vanilla.yaml: the same tile encoder /
reward / action space, flipped onto vanilla_ppo, which has the working
single-policy save-state advance gate.

These tests lock in that route:
  * mario_tiles_vanilla.yaml resolves to the vanilla_ppo + smb_tiles
    combination with the vanilla-correctness knobs (start state present,
    sticky OFF, rollout geometry set).
  * mario_tiles.yaml stays on ga_ppo (the task forbids repointing the
    legacy GA config).
  * the A/B invariant holds: the two configs share IDENTICAL reward
    weights and action space, so the only variable between them is the
    trainer mode.

Pure-YAML checks (no ROM / Rust pool needed), mirroring
tests/test_profile_configs.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.training.profile_utils import validate_profile

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TILES = CONFIG_DIR / "mario_tiles.yaml"
_TILES_VANILLA = CONFIG_DIR / "mario_tiles_vanilla.yaml"


def _load(path: Path) -> dict:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path.name}: root is not a mapping"
    return data


def test_tiles_vanilla_routes_to_the_proven_advancing_path() -> None:
    """mario_tiles_vanilla.yaml must resolve to vanilla_ppo + smb_tiles
    with the SMB curriculum active and the vanilla-correctness knobs set."""
    prof = _load(_TILES_VANILLA)
    rl = prof.get("reinforce", {})

    # The whole point: vanilla_ppo trainer mode (the working advance gate).
    assert str(rl.get("trainer_mode", "")).lower() == "vanilla_ppo", (
        "mario_tiles_vanilla.yaml must pin trainer_mode: vanilla_ppo — that "
        "is the routing change that makes the SMB tile curriculum advance."
    )
    # Tile encoder — the whole reason mario_tiles exists.
    assert rl.get("encoder") == "smb_tiles"
    # Name must keep the substring 'mario' so Trainer sets
    # _smb_curriculum_active (the vanilla save-state advance gate is
    # SMB-name-gated).
    assert "mario" in str(prof.get("name", "")).lower()

    # CRITICAL: a start state, or vanilla_ppo trains on the title-screen
    # attract demo (inputs ignored) and learns nothing.
    assert prof.get("start_state_path"), (
        "mario_tiles_vanilla.yaml needs start_state_path or vanilla_ppo "
        "cold-boots to the title-screen demo and learns nothing."
    )

    # Sticky actions MUST be off under vanilla PPO — sticky overwrites the
    # executed action and invalidates PPO's importance ratio.
    assert float(rl.get("sticky_action_prob", 0.0)) == 0.0, (
        "sticky_action_prob must be 0.0 under vanilla_ppo (it corrupts the "
        "PPO importance ratio)."
    )
    # Vanilla PPO needs an explicit rollout geometry; ga_ppo ignores it.
    assert int(rl.get("rollout_steps", 0)) > 0, (
        "vanilla_ppo requires rollout_steps (ga_ppo uses whole episodes)."
    )
    assert int(rl.get("ppo_minibatch_size", 0)) > 0
    # GA/BC population machinery must be off in vanilla mode.
    assert rl.get("bc_replay_enabled") is False
    # Vanilla's only exploration knob is the entropy bonus; the GA value
    # (0.001) starves a single policy. Expect the canonical vanilla value.
    assert float(rl.get("entropy_coef", 0.0)) >= 0.005, (
        "entropy_coef too low for single-policy exploration under vanilla_ppo."
    )

    # It must still be a valid profile per the repo's own validator.
    assert validate_profile(prof) == []


def test_legacy_ga_tiles_config_is_not_repointed() -> None:
    """The existing ga_ppo config must remain ga_ppo — the vanilla route is
    an ADDITIONAL config, it does not repoint or replace the GA one."""
    prof = _load(_TILES)
    assert str(prof.get("reinforce", {}).get("trainer_mode", "")).lower() == "ga_ppo"


def test_ab_invariant_reward_and_actions_are_identical() -> None:
    """The vanilla route holds mario_tiles' EXACT reward + action space so
    the only variable in the A/B is the trainer mode. If someone edits one
    config's reward, this fails loudly so the comparison stays honest."""
    tiles = _load(_TILES)
    vanilla = _load(_TILES_VANILLA)
    assert vanilla["reward_weights"] == tiles["reward_weights"], (
        "mario_tiles_vanilla.yaml reward_weights drifted from mario_tiles.yaml; "
        "the A/B is no longer a clean trainer-mode-only comparison."
    )
    assert vanilla["action_space"] == tiles["action_space"]
    # Same episode budget + frame skip too (game-time-per-episode parity).
    assert vanilla["frame_skip"] == tiles["frame_skip"]
    assert vanilla["max_episode_steps"] == tiles["max_episode_steps"]
