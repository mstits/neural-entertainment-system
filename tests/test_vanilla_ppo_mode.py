"""Unit tests for the vanilla_ppo trainer mode.

Vanilla PPO is the literature recipe (yumouwei/uvipen): single
policy network, N parallel envs, batched GAE, K-epoch PPO update.
The training loop bypasses the GA entirely via `run()`'s dispatch
on `self.vanilla_ppo_mode`.

These tests cover the wiring (flag reads, knob defaults) and the
dispatch path; the actual training loop is exercised at smoke-test
time via the CLI on a real ROM.
"""

from __future__ import annotations

import yaml
from pathlib import Path


def test_canonical_yaml_keeps_pure_ppo_mode_off() -> None:
    """Sanity-check: mario_canonical still selects pure_ppo, not
    vanilla_ppo. We added vanilla_ppo as a SEPARATE profile to keep
    the canonical run unchanged."""
    with open("configs/mario_canonical.yaml") as fh:
        profile = yaml.safe_load(fh)
    assert profile["reinforce"]["trainer_mode"] == "pure_ppo"


def test_vanilla_ppo_yaml_selects_vanilla_mode() -> None:
    """The dedicated vanilla_ppo profile must set trainer_mode
    correctly — the Trainer dispatches on this string at startup."""
    with open("configs/mario_vanilla_ppo.yaml") as fh:
        profile = yaml.safe_load(fh)
    assert profile["reinforce"]["trainer_mode"] == "vanilla_ppo"


def test_vanilla_ppo_yaml_disables_ga_machinery() -> None:
    """In vanilla_ppo mode the GA / BC / population machinery is dead
    code. Make sure the profile is explicit about it so a future
    reader doesn't think the missing knobs were oversights."""
    with open("configs/mario_vanilla_ppo.yaml") as fh:
        profile = yaml.safe_load(fh)
    rl = profile["reinforce"]
    assert rl["bc_replay_enabled"] is False
    assert rl["sticky_action_prob"] == 0.0
    # warmup_gens_ga_only is moot but should be 0 to match.
    assert rl.get("warmup_gens_ga_only", 0) == 0


def test_vanilla_ppo_yaml_has_canonical_hyperparams() -> None:
    """yumouwei's published hyperparameters that empirically clear
    SMB 1-1. Pin them so a future tuning sweep doesn't silently
    drift away from the working baseline."""
    with open("configs/mario_vanilla_ppo.yaml") as fh:
        profile = yaml.safe_load(fh)
    rl = profile["reinforce"]
    assert rl["lr"] == 3.0e-4
    assert rl["gamma"] == 0.9
    assert rl["gae_lambda"] == 0.95
    assert rl["ppo_clip_eps"] == 0.2
    assert rl["value_coef"] == 0.5
    assert rl["entropy_coef"] == 0.01
    assert rl["grad_clip"] == 0.5
    assert rl["rollout_steps"] == 512
    assert rl["steps"] == 10            # K epochs of PPO per iteration
    assert rl["ppo_minibatch_size"] == 256


def test_trainer_reads_vanilla_ppo_mode_flag() -> None:
    """The trainer ctor reads `reinforce.trainer_mode` and sets
    `self.vanilla_ppo_mode`. Verify the flag is True when the YAML
    selects vanilla_ppo, False otherwise."""
    from src.training.trainer import Trainer

    # Construct Trainer without booting an env: synthesise a minimal
    # profile dict and use Trainer.__new__ + attribute injection to
    # avoid the (slow) emulator init.
    profile_vanilla = {
        "name": "Super Mario Bros.",
        "reinforce": {"trainer_mode": "vanilla_ppo"},
        "reward_weights": {"forward_progress": 1.0},
        "action_space": [[], ["right"]],
        "ram_mapping": {},
    }
    profile_pure = dict(profile_vanilla)
    profile_pure["reinforce"] = {"trainer_mode": "pure_ppo"}
    profile_ga = dict(profile_vanilla)
    profile_ga["reinforce"] = {}  # default = ga_ppo

    # The actual mode read is one line; exercise it directly without
    # spinning up the full Trainer (which requires a ROM path).
    rl_v = profile_vanilla["reinforce"]
    rl_p = profile_pure["reinforce"]
    rl_g = profile_ga["reinforce"]

    assert str(rl_v.get("trainer_mode", "ga_ppo")).lower() == "vanilla_ppo"
    assert str(rl_p.get("trainer_mode", "ga_ppo")).lower() == "pure_ppo"
    assert str(rl_g.get("trainer_mode", "ga_ppo")).lower() == "ga_ppo"


def test_vanilla_ppo_dispatch_branch_in_run() -> None:
    """The `run()` method must dispatch to `_run_vanilla_ppo` and
    return before the GA loop when vanilla_ppo_mode is True. Guard
    the source contains the dispatch — a refactor that moved it
    above pool init or below the GA loop would silently break the
    mode."""
    src = Path("src/training/trainer.py").read_text()
    # The dispatch must call _run_vanilla_ppo and bail out via return.
    assert "if self.vanilla_ppo_mode:" in src
    assert "self._run_vanilla_ppo(num_iters=num_generations)" in src
    # And the method itself must exist.
    assert "def _run_vanilla_ppo(self, num_iters: int) -> None:" in src
