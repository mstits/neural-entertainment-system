"""End-to-end smoke test for the vanilla_ppo training loop.

The production training path (`Trainer._run_vanilla_ppo`) had ZERO
automated coverage — it was exercised only by manual CLI runs. This
test runs the real loop for a couple of tiny iterations on the actual
SMB ROM, into an isolated temp checkpoint dir with resume disabled, so
it never touches real artifacts. It exercises, end to end:

    rollout collection -> reward fn -> auto-reset/freeze handling ->
    batched GAE -> K-epoch PPO update -> metrics emission -> iter-
    boundary reset + stacker re-seed -> checkpoint save.

It is deliberately a "does the wiring hold together" guard, not a
learning test — 2 iters can't learn anything. It catches crashes and
regressions in the loop's plumbing (e.g. the stage-seed / auto-reset
re-seed bookkeeping) that unit tests on the extracted pure functions
(test_ppo.py) cannot see.
"""

from __future__ import annotations

import queue as _queue
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_vanilla_ppo.yaml"

pytestmark = pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / vanilla_ppo profile not present.",
)


def test_vanilla_ppo_two_iters_end_to_end() -> None:
    from src.training.trainer import Trainer

    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    # Shrink everything so two iters run in a few seconds.
    profile["reinforce"]["rollout_steps"] = 24
    profile["reinforce"]["steps"] = 2
    profile["reinforce"]["ppo_minibatch_size"] = 16

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="vppo_smoke_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=4,
            population_size=4,
            checkpoint_dir=tmp,            # explicit -> isolated, not per-game
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
        )
        # Explicit non-default dir must be honored verbatim (no slug).
        assert str(trainer.checkpoint_dir) == tmp

        trainer.run(num_generations=2, resume_from=None)

        # If the profile enables RND (rnd_intrinsic_coef > 0), the module
        # must have been built and exercised on the rollout path.
        if profile["reinforce"].get("rnd_intrinsic_coef", 0.0) > 0.0:
            assert trainer._rnd is not None, (
                "rnd_intrinsic_coef > 0 but RND module was never built"
            )

    # The full rollout -> GAE -> PPO update -> metrics path must have
    # emitted at least one iteration's metrics with the PPO loss keys.
    emitted = []
    while not metrics_q.empty():
        emitted.append(metrics_q.get_nowait())
    assert emitted, "no metrics emitted — the loop didn't complete an iter"
    ppo_rows = [m for m in emitted if "ppo_loss" in m]
    assert ppo_rows, "no ppo_loss metric emitted — PPO update didn't run"
    last = ppo_rows[-1]
    # Entropy is finite and within [0, ln(num_actions)] — a sanity gate
    # that the loss math produced real numbers, not NaN/inf.
    import math
    assert math.isfinite(last["ppo_entropy"])
    assert 0.0 <= last["ppo_entropy"] <= math.log(len(profile["action_space"])) + 1e-3
