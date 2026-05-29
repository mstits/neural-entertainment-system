"""Unit tests for the reproducibility run manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.run_manifest import write_run_manifest


def _profile() -> dict:
    return {
        "name": "Super Mario Bros.",
        "reinforce": {
            "trainer_mode": "vanilla_ppo",
            "encoder": "smb_tiles",
            "device": "cpu",
            "lr": 3.0e-4,
            "gamma": 0.9,
            "rollout_steps": 1024,
            "rnd_intrinsic_coef": 0.5,
            "bc_replay_enabled": False,   # not a pinned hyperparam -> excluded
        },
    }


def test_write_run_manifest_captures_provenance(tmp_path: Path) -> None:
    p = write_run_manifest(
        tmp_path,
        game="Super Mario Bros.",
        rom_path="roms/Super Mario Bros. (World).nes",
        start_state_path="roms/smb_start.state.bin",
        seed=123,
        profile=_profile(),
        num_envs=12,
        frame_skip=4,
        created_at=1000.0,
    )
    assert p == tmp_path / "run_manifest.json"
    m = json.loads(p.read_text())

    # Provenance fields needed to reproduce.
    assert m["game"] == "Super Mario Bros."
    assert m["rom_path"].endswith("Super Mario Bros. (World).nes")
    assert m["start_state_path"].endswith("smb_start.state.bin")
    assert m["seed"] == 123
    assert m["num_envs"] == 12
    assert m["frame_skip"] == 4
    assert m["created_at"] == 1000.0
    assert m["trainer_mode"] == "vanilla_ppo"
    assert m["encoder"] == "smb_tiles"
    assert "git_commit" in m  # may be None outside a git checkout

    # Pinned hyperparameters captured; non-pinned excluded.
    hp = m["hyperparams"]
    assert hp["lr"] == 3.0e-4
    assert hp["gamma"] == 0.9
    assert hp["rollout_steps"] == 1024
    assert hp["rnd_intrinsic_coef"] == 0.5
    assert "bc_replay_enabled" not in hp


def test_write_run_manifest_handles_missing_start_state(tmp_path: Path) -> None:
    m = json.loads(
        write_run_manifest(
            tmp_path, game="Contra", rom_path="roms/Contra (USA).nes",
            start_state_path=None, seed=None, profile={"reinforce": {}},
            num_envs=8, frame_skip=4,
        ).read_text()
    )
    assert m["start_state_path"] is None
    assert m["seed"] is None
    assert m["hyperparams"] == {}


def test_write_run_manifest_is_atomic_overwrite(tmp_path: Path) -> None:
    # Writing twice overwrites cleanly (no .tmp left behind).
    for ss in ("a.state", "b.state"):
        write_run_manifest(
            tmp_path, game="X", rom_path="r.nes", start_state_path=ss,
            seed=1, profile={"reinforce": {}}, num_envs=1, frame_skip=4,
        )
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["start_state_path"] == "b.state"
    assert not (tmp_path / "run_manifest.json.tmp").exists()
