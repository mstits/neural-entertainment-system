"""Unit tests for the reproducibility run manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.run_manifest import (
    _PINNED_PACKAGES,
    dependency_snapshot,
    update_run_manifest_redo,
    write_run_manifest,
)


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

    # Dependency snapshot: the packages that can move a banked number
    # even with commit+seed fixed, plus the nes_core extension's own
    # build identity (dist version is static across local rebuilds, so
    # the .so digest is what actually pins the binary).
    deps = m["dependencies"]
    assert set(_PINNED_PACKAGES) <= set(deps)
    for name in _PINNED_PACKAGES:
        assert isinstance(deps[name], str) and deps[name]
    assert set(deps["nes_core"]) == {"dist_version", "module", "sha256_16"}


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


def test_write_run_manifest_preserves_rom_md5_on_second_write_without_it(
    tmp_path: Path,
) -> None:
    """train_game.py writes the correct rom_md5 first; vanilla_ppo's own
    CheckpointManager.write_manifest has no rom_md5 kwarg and calls back
    into write_run_manifest for the same checkpoint dir + ROM moments
    later, defaulting rom_md5 to None. That second write must not
    clobber the already-recorded MD5 with null."""
    rom_path = "roms/Super Mario Bros. (World).nes"
    write_run_manifest(
        tmp_path, game="Super Mario Bros.", rom_path=rom_path,
        start_state_path=None, seed=1, profile={"reinforce": {}},
        num_envs=8, frame_skip=4,
        rom_md5="deadbeefcafebabe1234567890abcde",
    )
    write_run_manifest(
        tmp_path, game="Super Mario Bros.", rom_path=rom_path,
        start_state_path=None, seed=1, profile={"reinforce": {}},
        num_envs=8, frame_skip=4,
    )
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["rom_md5"] == "deadbeefcafebabe1234567890abcde"


def test_dependency_snapshot_records_unknown_on_lookup_failure(monkeypatch) -> None:
    """A `pip`-metadata lookup failure must degrade to "unknown" per
    field, not raise out of a function called at the start of every
    training run."""
    import importlib.metadata as md

    def _boom(name):
        raise RuntimeError("simulated metadata lookup failure")

    monkeypatch.setattr(md, "version", _boom)
    snap = dependency_snapshot()

    for name in _PINNED_PACKAGES:
        assert snap[name] == "unknown"
    assert snap["nes_core"]["dist_version"] == "unknown"


def test_dependency_snapshot_handles_missing_nes_core(monkeypatch) -> None:
    """A bare checkout before `make build` (no nes_core installed yet)
    must still produce a snapshot with "unknown" fields for nes_core,
    not an ImportError that takes the manifest write down with it."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "nes_core":
            raise ImportError("simulated missing nes_core")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    snap = dependency_snapshot()

    assert snap["nes_core"]["module"] == "unknown"
    assert snap["nes_core"]["sha256_16"] == "unknown"


def test_update_run_manifest_redo_patches_an_existing_manifest(tmp_path: Path) -> None:
    """V31_REDO_SURGICAL_2026-08-27.md §12 item 6: the pre-run manifest
    gets ReDo's summary telemetry patched in after the run ends, without
    disturbing the provenance fields written before training started.
    """
    write_run_manifest(
        tmp_path, game="Super Mario Bros.",
        rom_path="roms/x.nes", start_state_path=None, seed=0,
        profile=_profile(), num_envs=60, frame_skip=4, created_at=1000.0,
    )
    p = update_run_manifest_redo(
        tmp_path, redo_tau=0.10, cum_recycled=460, recycle_events=230,
        first_recycle_iter=16, median_agree=0.93, median_dose_frac=0.09375,
        distinct_fc2_indices=24,
    )
    m = json.loads(p.read_text())
    assert m["seed"] == 0  # pre-run field untouched
    assert m["redo_tau"] == 0.10
    assert m["redo_cum_recycled"] == 460
    assert m["redo_recycle_events"] == 230
    assert m["redo_first_recycle_iter"] == 16
    assert m["redo_median_agree"] == 0.93
    assert m["redo_median_dose_frac"] == 0.09375
    assert m["redo_distinct_fc2_indices"] == 24


def test_update_run_manifest_redo_never_raises_on_a_missing_file(
    tmp_path: Path,
) -> None:
    """Called on a checkpoint dir where the pre-run manifest write
    itself failed (best-effort, never blocks training) — must still
    produce a manifest rather than raising and masking a VOID abort's
    real exception.
    """
    p = update_run_manifest_redo(
        tmp_path, redo_tau=0.25, cum_recycled=0, recycle_events=0,
        first_recycle_iter=None, median_agree=None, median_dose_frac=None,
        distinct_fc2_indices=0,
    )
    m = json.loads(p.read_text())
    assert m["redo_cum_recycled"] == 0
    assert m["redo_first_recycle_iter"] is None
