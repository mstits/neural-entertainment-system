"""Guard: a fresh vanilla_ppo run's Go-Explore archive starts empty
instead of silently inheriting a prior run's visited cells.

Background (2026-08-25): `ExplorationController.build_go_explore` had no
`fresh_start` awareness and always loaded `checkpoint_dir/go_explore/
archive.pkl` when present. An explicit from-scratch relaunch (GUI
"Resume" unticked / headless `--no-resume`) correctly relocates old
`vanilla_ppo_iter_*.pt` checkpoints and skips the saved SMB curriculum
stage states, but the Go-Explore archive was never included in that
reset — a brand-new, randomly initialized policy could be teleported by
`select_return_states` into a state (e.g. deep into a later level) it
never earned, while count/novelty bookkeeping treated those stale cells
as already visited.

`build_go_explore(ge_cfg, fresh_start=...)`:
  * fresh_start=True  -> IGNORE the on-disk archive, start empty;
  * fresh_start=False / default -> load the on-disk archive (unchanged
    resume behavior).

Fresh runs NEVER delete or move the archive file — a paused experiment
must stay resumable — so these tests also assert the file on disk stays
byte-for-byte untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.training.exploration_controller import ExplorationController
from src.training.go_explore import GoExploreArchive, ram_downsample_cell


def _make_controller(tmp_path):
    trainer = SimpleNamespace(
        seed=0,
        checkpoint_dir=tmp_path,
        game_profile={"name": "smb"},
        _go_explore=None,
    )
    return ExplorationController(trainer), trainer


def _seed_archive(tmp_path):
    """Write a small on-disk archive with one visited cell + a fake
    saved-state blob, mimicking a prior run's `save_go_explore`."""
    arc_path = tmp_path / "go_explore" / "archive.pkl"
    archive = GoExploreArchive(ram_downsample_cell(stride=64, bucket=16), seed=0)
    archive.record(
        b"\x00" * 2048, b"stale-worker-state-blob", score=1234.0, steps=10,
    )
    archive.save(arc_path)
    return arc_path


def test_fresh_start_ignores_saved_archive_and_starts_empty(tmp_path):
    arc_path = _seed_archive(tmp_path)
    before = arc_path.read_bytes()

    controller, trainer = _make_controller(tmp_path)
    archive, _use_max_x = controller.build_go_explore({}, fresh_start=True)

    # Fresh: no cells inherited from the prior run.
    assert len(archive) == 0
    assert trainer._go_explore is archive

    # File on disk is byte-for-byte untouched — never deleted/rewritten.
    assert arc_path.read_bytes() == before


def test_non_fresh_resumes_saved_archive(tmp_path):
    arc_path = _seed_archive(tmp_path)
    before = arc_path.read_bytes()

    controller, _trainer = _make_controller(tmp_path)
    archive, _use_max_x = controller.build_go_explore({}, fresh_start=False)

    # Resume behavior is unchanged: inherit the saved archive's cells.
    assert len(archive) == 1
    assert arc_path.read_bytes() == before


def test_default_call_resumes_like_headless(tmp_path):
    # Default arg (no fresh_start passed) preserves the historical
    # auto-resume that headless scripts rely on.
    _seed_archive(tmp_path)
    controller, _trainer = _make_controller(tmp_path)
    archive, _use_max_x = controller.build_go_explore({})
    assert len(archive) == 1


def test_fresh_start_with_no_archive_on_disk_is_a_noop(tmp_path):
    controller, _trainer = _make_controller(tmp_path)
    archive, _use_max_x = controller.build_go_explore({}, fresh_start=True)
    assert len(archive) == 0
