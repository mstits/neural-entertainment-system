"""Guard: a fresh vanilla_ppo run starts the SMB save-state curriculum
at stage 0 instead of silently inheriting a prior run's mid-game stage.

Background (2026-07-14): `fresh_start=True` (GUI "Resume" unticked /
headless `--no-resume`) resets the model weights but historically let the
curriculum loader inherit whatever `stage_NN.state` files sat in the
checkpoint dir — a validated confusion where a supposedly-fresh run
resumed at stage 2 ("[vanilla_ppo] curriculum resumed at stage 2").

The loader now lives in
`Trainer._load_smb_curriculum_from_disk(fresh_start=...)`:
  * fresh_start=True  -> IGNORE saved stage states, return stage 0;
  * fresh_start=True + reinforce.inherit_curriculum_on_fresh=True
    (mirrored onto `_inherit_curriculum_on_fresh`) -> inherit as before;
  * fresh_start=False / default -> inherit (unchanged resume behavior).

Fresh runs NEVER delete or move the saved states — a paused experiment
must stay resumable — so these tests also assert the files on disk are
byte-for-byte untouched.

Like test_vanilla_ppo_resume_gate, these exercise the loader directly
(Trainer.__new__ + a temp checkpoint dir with sidecar meta so no
emulator/pool is booted).
"""

from __future__ import annotations

import json

from src.training.trainer import Trainer


def _make_trainer(tmp_path, *, active=True, inherit_on_fresh=False):
    t = Trainer.__new__(Trainer)
    t.checkpoint_dir = tmp_path
    t._smb_curriculum_active = active
    t._inherit_curriculum_on_fresh = inherit_on_fresh
    # `pool` is only touched on the legacy (no-sidecar) path, which these
    # tests avoid by always writing a sidecar; set None to catch any
    # accidental reliance on it.
    t.pool = None
    return t


def _write_stage(curr_dir, n, anchor, payload):
    curr_dir.mkdir(parents=True, exist_ok=True)
    (curr_dir / f"stage_{n:02d}.state").write_bytes(payload)
    (curr_dir / f"stage_{n:02d}.meta.json").write_text(
        json.dumps({"anchor": anchor})
    )


def _seed_two_stages(tmp_path):
    """Two saved stages: stage 1 (anchor 2 = 1-2), stage 2 (anchor 5 = 1-3)."""
    curr_dir = tmp_path / "smb_curriculum"
    _write_stage(curr_dir, 1, 2, b"stage-one-blob")
    _write_stage(curr_dir, 2, 5, b"stage-two-blob")
    return curr_dir


def _snapshot(curr_dir):
    """Map file name -> (bytes, mtime_ns) for the whole curriculum dir."""
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(curr_dir.iterdir())
    }


def test_fresh_start_ignores_saved_curriculum_and_starts_stage_0(tmp_path):
    curr_dir = _seed_two_stages(tmp_path)
    before = _snapshot(curr_dir)

    t = _make_trainer(tmp_path)
    states, anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=True)

    # Fresh: start at stage 0 with only the cold-boot seed entry.
    assert stage == 0
    assert states == [None]
    assert anchors == [0]

    # Files on disk are byte-for-byte untouched — never deleted/moved/rewritten.
    after = _snapshot(curr_dir)
    assert before == after


def test_fresh_start_opt_in_inherits_saved_curriculum(tmp_path):
    curr_dir = _seed_two_stages(tmp_path)
    before = _snapshot(curr_dir)

    t = _make_trainer(tmp_path, inherit_on_fresh=True)
    states, anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=True)

    # Opt-in restores the historical inherit-on-fresh behavior.
    assert stage == 2
    assert states == [None, b"stage-one-blob", b"stage-two-blob"]
    assert anchors == [0, 2, 5]

    # Reads only — files still untouched.
    assert _snapshot(curr_dir) == before


def test_non_fresh_resumes_saved_curriculum(tmp_path):
    curr_dir = _seed_two_stages(tmp_path)
    before = _snapshot(curr_dir)

    t = _make_trainer(tmp_path)
    states, anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=False)

    # Resume behavior is unchanged: inherit the saved curriculum.
    assert stage == 2
    assert states == [None, b"stage-one-blob", b"stage-two-blob"]
    assert anchors == [0, 2, 5]
    assert _snapshot(curr_dir) == before


def test_default_call_resumes_like_headless(tmp_path):
    # Default arg (no fresh_start passed) preserves the historical
    # auto-resume that headless scripts rely on.
    _seed_two_stages(tmp_path)
    t = _make_trainer(tmp_path)
    _states, _anchors, stage = t._load_smb_curriculum_from_disk()
    assert stage == 2


def test_fresh_start_empty_dir_is_stage_0(tmp_path):
    t = _make_trainer(tmp_path)
    states, anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=True)
    assert stage == 0
    assert states == [None]
    assert anchors == [0]


def test_non_mario_never_loads_curriculum(tmp_path):
    # Curriculum inactive (non-Mario): even a resume ignores stage files
    # so the area-byte gate never reads garbage from another game's RAM.
    _seed_two_stages(tmp_path)
    t = _make_trainer(tmp_path, active=False)
    _states, _anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=False)
    assert stage == 0


def test_partial_curriculum_stops_at_gap(tmp_path):
    # Only stage 1 present (stage 2 missing) -> resume stops at stage 1.
    curr_dir = tmp_path / "smb_curriculum"
    _write_stage(curr_dir, 1, 2, b"stage-one-blob")
    t = _make_trainer(tmp_path)
    _states, anchors, stage = t._load_smb_curriculum_from_disk(fresh_start=False)
    assert stage == 1
    assert anchors == [0, 2]
