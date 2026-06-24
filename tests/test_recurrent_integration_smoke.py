"""End-to-end smoke for the recurrent (GRU) vanilla_ppo path.

The recurrent integration (rollout hidden-state threading + the
sequence-BPTT _recurrent_ppo_update) is the most complex code added for
the SMB-past-1-2 lever. It had no committed CI guard — only a scratchpad
smoke. This runs the real loop with reinforce.recurrent=true for two
tiny iters on the actual SMB ROM and asserts:

  - the recurrent net (TileRecurrentPolicyNetwork) is actually used;
  - the full recurrent path (rollout threads hidden state -> done-masked
    -> bootstrap -> _recurrent_ppo_update sequence BPTT) runs without
    crashing and emits FINITE losses (the common recurrent-PPO failure
    is BPTT divergence to NaN/inf).

It guards the wiring, not learning (2 iters can't learn). Pairs with
test_recurrent_policy_learns.py (which guards the GRU mechanics).
"""

from __future__ import annotations

import math
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


def test_recurrent_vanilla_ppo_two_iters_end_to_end() -> None:
    from src.training.trainer import Trainer
    from src.models.tile_policy import TileRecurrentPolicyNetwork

    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    r = profile["reinforce"]
    r["recurrent"] = True                 # <-- opt-in recurrent path
    r["recurrent_env_minibatch"] = 4
    r["rollout_steps"] = 24               # short -> fast BPTT
    r["steps"] = 2
    r["ppo_minibatch_size"] = 16

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="recurrent_smoke_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=6,
            population_size=6,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
        )
        # The recurrent flag must be live and the network must be the GRU.
        assert trainer._recurrent is True
        assert isinstance(trainer._make_network(), TileRecurrentPolicyNetwork)

        trainer.run(num_generations=2, resume_from=None)

    emitted = []
    while not metrics_q.empty():
        emitted.append(metrics_q.get_nowait())
    assert emitted, "recurrent loop emitted no metrics"
    ppo_rows = [m for m in emitted if "ppo_loss" in m]
    assert ppo_rows, "no ppo_loss — the recurrent _recurrent_ppo_update didn't run"
    # BPTT must produce finite losses (no NaN/inf divergence).
    for m in ppo_rows:
        for k in ("ppo_loss", "ppo_value_loss", "ppo_policy_loss", "ppo_entropy"):
            assert math.isfinite(m[k]), f"recurrent BPTT produced non-finite {k}={m[k]}"
