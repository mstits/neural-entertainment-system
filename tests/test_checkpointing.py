"""Tests for the checkpoint save / archive / rotate / find module.

Atomicity, mtime-ordering, and the rotation invariant all have
real data-loss histories — each is locked down here without
spinning up a full Trainer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.training.checkpointing import (
    archive_previous_run,
    find_latest_checkpoint,
    rotate_old_checkpoints,
    save_checkpoint_atomic,
)


class _FakeGA:
    """Tiny stand-in for the GA's save_checkpoint side."""
    def __init__(self, payload: bytes = b"ga-state") -> None:
        self.payload = payload
        self.calls: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        Path(path).write_bytes(self.payload)
        self.calls.append(path)


class _FakeCurriculum:
    def __init__(self, state: dict | None = None) -> None:
        self._state = state or {"stage": "stage_1", "episodes": 0}

    def state_dict(self) -> dict:
        return self._state


def test_save_checkpoint_atomic_uses_tmp_rename(tmp_path: Path) -> None:
    """Final file exists, .tmp does not — atomicity invariant."""
    ga = _FakeGA(b"genome-payload")
    save_checkpoint_atomic(
        path=tmp_path / "gen_00007.pt",
        ga=ga,
        curriculum=_FakeCurriculum(),
        checkpoint_dir=tmp_path,
    )
    assert (tmp_path / "gen_00007.pt").read_bytes() == b"genome-payload"
    assert not (tmp_path / "gen_00007.pt.tmp").exists()
    assert not (tmp_path / "curriculum.json.tmp").exists()
    # Verify GA was called against the .tmp path, not the final path.
    assert ga.calls[0].endswith(".tmp")


def test_save_checkpoint_atomic_writes_curriculum_json(tmp_path: Path) -> None:
    curr = _FakeCurriculum({"stage": "stage_3", "ep": 42})
    save_checkpoint_atomic(
        path=tmp_path / "gen_00001.pt",
        ga=_FakeGA(),
        curriculum=curr,
        checkpoint_dir=tmp_path,
    )
    data = json.loads((tmp_path / "curriculum.json").read_text())
    assert data == {"stage": "stage_3", "ep": 42}


def _make_ckpt(p: Path, payload: bytes = b"x") -> Path:
    p.write_bytes(payload)
    return p


def test_find_latest_checkpoint_orders_by_mtime(tmp_path: Path) -> None:
    """A higher-named older file must not outrank a lower-named newer
    one. This is the gen_01094-vs-gen_00010 data-loss case."""
    older_higher_name = _make_ckpt(tmp_path / "gen_01094.pt")
    time.sleep(0.05)  # nudge mtime past the resolution boundary
    newer_lower_name = _make_ckpt(tmp_path / "gen_00010.pt")
    assert find_latest_checkpoint(tmp_path) == newer_lower_name


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_rotate_old_checkpoints_keeps_newest_by_mtime(tmp_path: Path) -> None:
    """Rotate must use mtime, not lex order — same data-loss case."""
    paths = []
    for i, name in enumerate(["gen_05000.pt", "gen_05010.pt", "gen_00100.pt"]):
        p = _make_ckpt(tmp_path / name)
        # Force monotonic mtimes regardless of lex order: older first.
        ts = time.time() - (3 - i) * 10
        import os
        os.utime(p, (ts, ts))
        paths.append(p)
    # Keep last 2 by mtime: the newest two (00100, 05010), drop 05000.
    rotate_old_checkpoints(tmp_path, keep_last=2)
    assert paths[0].exists() is False  # gen_05000 (oldest mtime) deleted
    assert paths[1].exists()            # gen_05010 (newer mtime) kept
    assert paths[2].exists()            # gen_00100 (newest mtime) kept


def test_rotate_old_checkpoints_removes_paired_mlpackage(tmp_path: Path) -> None:
    """When a checkpoint is rotated out, its .mlpackage sibling must
    follow — otherwise checkpoint_dir grows .mlpackage clutter."""
    import os
    p1 = _make_ckpt(tmp_path / "gen_00001.pt")
    ml1 = tmp_path / "gen_00001.mlpackage"
    ml1.mkdir()  # mlpackages are directories
    (ml1 / "model.mlmodel").write_bytes(b"")
    p2 = _make_ckpt(tmp_path / "gen_00002.pt")

    # Make p1 older.
    ts1 = time.time() - 10
    os.utime(p1, (ts1, ts1))

    rotate_old_checkpoints(tmp_path, keep_last=1)
    assert not p1.exists()
    assert not ml1.exists()
    assert p2.exists()


def test_rotate_old_checkpoints_noop_when_below_keep_last(tmp_path: Path) -> None:
    p = _make_ckpt(tmp_path / "gen_00001.pt")
    rotate_old_checkpoints(tmp_path, keep_last=5)
    assert p.exists()


def test_rotate_old_checkpoints_keep_last_zero_is_noop(tmp_path: Path) -> None:
    """keep_last <= 0 disables rotation (used by tests + manual ops)."""
    p = _make_ckpt(tmp_path / "gen_00001.pt")
    rotate_old_checkpoints(tmp_path, keep_last=0)
    assert p.exists()


def test_archive_previous_run_noop_on_empty_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")  # empty file
    result = archive_previous_run(tmp_path, metrics, game_profile={"name": "smb"})
    assert result is None
    # Empty file should still be present (no archive happened).
    assert metrics.exists()


def test_archive_previous_run_noop_on_missing_metrics(tmp_path: Path) -> None:
    result = archive_previous_run(
        tmp_path, tmp_path / "nope.jsonl", game_profile={"name": "smb"}
    )
    assert result is None


def test_archive_previous_run_moves_metrics_and_copies_companions(tmp_path: Path) -> None:
    """The metrics file is moved (not copied) so the next run starts
    with an empty slate; companions are copied so they remain
    available for resume."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"a":1}\n')
    (tmp_path / "curriculum.json").write_text('{"stage":"x"}')
    (tmp_path / "bc_success_cache.npz").write_bytes(b"\x00")
    (tmp_path / "run.log").write_text("hello\n")
    _make_ckpt(tmp_path / "gen_00007.pt", b"latest")

    archive = archive_previous_run(
        tmp_path, metrics, game_profile={"name": "zelda", "frame_skip": 4}
    )
    assert archive is not None
    assert (archive / "metrics.jsonl").read_text() == '{"a":1}\n'
    # Metrics file is gone from the source (moved, not copied).
    assert not metrics.exists()
    # Companions copied — still exist in source.
    assert (tmp_path / "curriculum.json").exists()
    assert (archive / "curriculum.json").exists()
    assert (archive / "bc_success_cache.npz").exists()
    assert (archive / "run.log").exists()
    # Latest gen ckpt copied.
    assert (archive / "gen_00007.pt").read_bytes() == b"latest"
    # Profile serialized as self-describing YAML.
    import yaml
    saved = yaml.safe_load((archive / "game_profile.yaml").read_text())
    assert saved == {"name": "zelda", "frame_skip": 4}


def test_archive_previous_run_swallows_errors(tmp_path: Path) -> None:
    """An archive failure must NEVER raise — startup paths depend on
    this. Pass an unserializable profile; the YAML dump fails but the
    rest still completes (or at worst, returns None)."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"a":1}\n')

    class _Unrepr:
        pass

    bad_profile: Any = {"thing": _Unrepr()}  # yaml.safe_dump will refuse
    # Must not raise.
    archive = archive_previous_run(tmp_path, metrics, game_profile=bad_profile)
    # Either succeeded with archive (most fields OK) or returned None — both fine.
    assert archive is None or archive.is_dir()


# ---------------------------------------------------------------------------
# Vanilla-PPO rotation (external audit 2026-08-28): the helper had one
# call site (the GA path, gen_*.pt) and the vanilla path never rotated —
# checkpoints/ reached 98 GB. Rotation is now pattern-aware and OPT-IN
# for the vanilla family, defaulting to keep-all so a registered
# campaign's full grid (v32 cross-fit, the corrected ladders) can never
# be pruned by a rotation it did not ask for.
# ---------------------------------------------------------------------------

def _mk(p: Path, name: str, ts: float) -> Path:
    f = p / name
    f.write_bytes(b"x")
    import os
    os.utime(f, (ts, ts))
    return f


def test_rotate_vanilla_pattern_prunes_only_its_own_family(tmp_path: Path) -> None:
    t = time.time()
    old1 = _mk(tmp_path, "vanilla_ppo_iter_00010.pt", t - 300)
    old2 = _mk(tmp_path, "vanilla_ppo_iter_00020.pt", t - 200)
    new1 = _mk(tmp_path, "vanilla_ppo_iter_00030.pt", t - 100)
    new2 = _mk(tmp_path, "vanilla_ppo_iter_00040.pt", t - 10)
    ga = _mk(tmp_path, "gen_00001.pt", t - 500)          # other family
    state = _mk(tmp_path, "smb.state", t - 500)           # not a checkpoint
    winners = tmp_path / "winners"
    winners.mkdir()
    best = _mk(winners, "best.pt", t - 500)

    rotate_old_checkpoints(tmp_path, keep_last=2,
                           pattern="vanilla_ppo_iter_*.pt")

    assert not old1.exists() and not old2.exists()
    assert new1.exists() and new2.exists()
    # Everything outside the pattern is untouchable, including the GA
    # family, raw states, and winners/best.pt.
    assert ga.exists() and state.exists() and best.exists()


def test_rotate_default_pattern_ignores_vanilla_family(tmp_path: Path) -> None:
    t = time.time()
    v = [_mk(tmp_path, f"vanilla_ppo_iter_{i:05d}.pt", t - i) for i in
         (10, 20, 30, 40, 50)]
    rotate_old_checkpoints(tmp_path, keep_last=1)  # gen_*.pt default
    assert all(f.exists() for f in v), (
        "the GA-pattern call must never prune the vanilla family"
    )


def test_vanilla_save_path_wires_the_rotation_key() -> None:
    """Anti-dead-knob: `checkpoint_keep_last` must reach the vanilla-PPO
    save path with the vanilla pattern, and the default must be the
    keep-all no-op. Source-level, same style as the trainer mechanism
    guards — a config key registered in the schema but consumed nowhere
    is this repo's single most-shipped defect class (20 found in one
    audit)."""
    src = (Path(__file__).resolve().parent.parent
           / "src" / "training" / "checkpoint_manager.py").read_text()
    assert '"checkpoint_keep_last", 0' in src, (
        "the key must be read with a keep-all default of 0"
    )
    assert 'pattern="vanilla_ppo_iter_*.pt"' in src, (
        "the vanilla save path must rotate ONLY its own family"
    )
    idx_save = src.index('saved checkpoint')
    idx_rot = src.index('rotate_old_checkpoints(', idx_save)
    assert idx_rot > idx_save, (
        "rotation must run after the atomic save completes, never before"
    )
    from src.training.config_schema import KNOWN_REINFORCE_KEYS
    assert "checkpoint_keep_last" in KNOWN_REINFORCE_KEYS
