"""End-to-end smoke for Go-Explore wired into the vanilla_ppo loop.

The archive's pure logic (insertion, domination, frontier-weighted
selection, save/load) is unit-tested in tests/test_go_explore.py. These
smokes cover the TRAINER WIRING the pure tests cannot see: the RECORD hook
grows the archive, the iter-boundary RETURN hook selects frontier cells,
and the mutual-exclusivity gate keeps go_explore off whenever the SMB
save-state curriculum is active.
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


def _tiny_profile() -> dict:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 24
    profile["reinforce"]["steps"] = 2
    profile["reinforce"]["ppo_minibatch_size"] = 16
    return profile


def test_go_explore_grows_archive_and_drives_returns() -> None:
    from src.training.trainer import Trainer

    profile = _tiny_profile()
    # Go-Explore is mutually exclusive with the SMB curriculum.
    profile["reinforce"]["smb_curriculum"] = False
    profile["reinforce"]["go_explore"] = {
        "enabled": True,
        "cell": {
            "type": "ram_bytes",
            "addresses": [0x075F, 0x0760, 0x006D],
            "bucket": 1,
        },
        "score": "max_x",
        "save_every": 2,
    }
    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="ge_smoke_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=4,
            population_size=4,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
        )
        # >= 2 iters so an iter-boundary return runs against a non-empty archive.
        trainer.run(num_generations=4, resume_from=None)

        arc = trainer._go_explore
        assert arc is not None, "go_explore enabled but archive not built"
        # RECORD hook fired: cells + records grew.
        assert len(arc) > 0, "archive recorded no cells"
        assert arc.total_records > 0
        # RETURN hook fired: at least one cell was selected as a return target.
        assert any(c.times_chosen > 0 for c in arc.cells.values()), (
            "no cell was ever chosen as a return target"
        )

    emitted = []
    while not metrics_q.empty():
        emitted.append(metrics_q.get_nowait())
    ge_rows = [m for m in emitted if "go_explore_cells" in m]
    assert ge_rows, "no go_explore_cells metric emitted"
    assert ge_rows[-1]["go_explore_cells"] >= 1


def test_go_explore_stays_off_when_smb_curriculum_active() -> None:
    """Mutual exclusivity: with the SMB curriculum ON (Mario default),
    enabling go_explore must NOT build the archive."""
    from src.training.trainer import Trainer

    profile = _tiny_profile()
    profile["reinforce"]["smb_curriculum"] = True  # curriculum ON
    profile["reinforce"]["go_explore"] = {"enabled": True}  # requested but must lose
    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="ge_excl_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=4,
            population_size=4,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
        )
        trainer.run(num_generations=1, resume_from=None)
        assert trainer._go_explore is None, (
            "go_explore built despite an active SMB curriculum (not exclusive)"
        )
