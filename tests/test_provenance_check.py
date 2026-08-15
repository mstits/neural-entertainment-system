"""demo_anchor_paths sweep in scripts/provenance_check.py.

demo_anchor_paths entries live in tracked configs/*.yaml but usually
point at runs/ — git-ignored, so the SEEDS rglob (checkpoints/harvested_
seeds/) never sees them. A config referencing a runs/ npz that was never
committed (or got cleaned up) must fail the gate, not pass silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import provenance_check
from scripts.provenance_check import check_demo_anchor_paths, collect_demo_anchor_refs


def _write_config(configs_dir, name, demo_anchor_paths):
    configs_dir.mkdir(parents=True, exist_ok=True)
    lines = ["demo_anchor_paths:"] + [f'- "{p}"' for p in demo_anchor_paths]
    (configs_dir / name).write_text("\n".join(lines) + "\n")


def test_collect_demo_anchor_refs_reads_paths_from_tracked_config(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    _write_config(configs, "mario_1_2.yaml", ["runs/ge_1_2_solve/demos/demo_000.npz"])

    refs = collect_demo_anchor_refs(configs, repo)

    assert refs == {
        "runs/ge_1_2_solve/demos/demo_000.npz": ["configs/mario_1_2.yaml"],
    }


def test_check_demo_anchor_paths_fails_on_missing_runs_file(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    _write_config(configs, "mario_1_2.yaml", ["runs/ge_1_2_solve/demos/demo_000.npz"])
    seeds = repo / "checkpoints/harvested_seeds"

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert hashes == {}
    assert len(errors) == 1
    assert "runs/ge_1_2_solve/demos/demo_000.npz" in errors[0]
    assert "configs/mario_1_2.yaml" in errors[0]


def test_check_demo_anchor_paths_hashes_existing_runs_file(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    demo_rel = "runs/ge_1_2_solve/demos/demo_000.npz"
    _write_config(configs, "mario_1_2.yaml", [demo_rel])
    demo_path = repo / demo_rel
    demo_path.parent.mkdir(parents=True)
    np.savez(str(demo_path), obs_0=np.zeros((3, 4), dtype=np.int8),
              act_0=np.zeros((3,), dtype=np.int64))
    seeds = repo / "checkpoints/harvested_seeds"

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert errors == []
    assert demo_rel in hashes
    assert len(hashes[demo_rel]) == 64


def test_check_demo_anchor_paths_skips_files_already_under_seeds(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    seeds = repo / "checkpoints/harvested_seeds"
    demo_rel = "checkpoints/harvested_seeds/demos_x.npz"
    _write_config(configs, "mario_1_2.yaml", [demo_rel])
    demo_path = repo / demo_rel
    demo_path.parent.mkdir(parents=True)
    np.savez(str(demo_path), obs_0=np.zeros((3, 4), dtype=np.int8),
              act_0=np.zeros((3,), dtype=np.int64))

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert errors == []
    assert hashes == {}


# ---- soak receipt trails (gate extension approved 2026-08-15) ----------

FAKE_HARNESS_OK = """
def verify_receipt_trail(soak_dir):
    from pathlib import Path
    if (Path(soak_dir) / "TAMPERED").exists():
        return [f"receipt chain broken at seg_0001"]
    return []
"""


def _soak_repo(tmp_path, *, harness=True, dirs=()):
    repo = tmp_path
    (repo / "scripts").mkdir(exist_ok=True)
    if harness:
        (repo / "scripts" / "soak_harness.py").write_text(FAKE_HARNESS_OK)
    for name, final in dirs:
        d = repo / "runs" / "soak" / name
        d.mkdir(parents=True)
        if final is not None:
            import json as _json
            (d / "final_receipt.json").write_text(_json.dumps(final))
    return repo


def test_soak_absent_runs_dir_is_clean(tmp_path):
    errors, verified, scoreable = provenance_check.check_soak_trails(tmp_path)
    assert errors == [] and verified == 0 and scoreable == 0


def test_soak_dirs_without_harness_fail_loud(tmp_path):
    repo = _soak_repo(tmp_path, harness=False,
                      dirs=[("soak_a", None)])
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert len(errors) == 1 and "verifier" in errors[0].replace(
        "chain verifier", "verifier")
    assert verified == 0


def test_soak_valid_chain_counts_scoreable_and_selfcheck(tmp_path):
    repo = _soak_repo(tmp_path, dirs=[
        ("soak_real", {"backend_scoreable": True, "selfcheck": False}),
        ("soak_stub", {"backend_scoreable": True, "selfcheck": True}),
        ("soak_unscoreable", {"backend_scoreable": False,
                              "selfcheck": False}),
    ])
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert errors == []
    assert verified == 3
    assert scoreable == 1  # stub + unscoreable chains verify but never score


def test_soak_tampered_chain_fails(tmp_path):
    repo = _soak_repo(tmp_path, dirs=[
        ("soak_bad", {"backend_scoreable": True, "selfcheck": False})])
    (repo / "runs" / "soak" / "soak_bad" / "TAMPERED").touch()
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert any("chain broken" in e for e in errors)
    assert verified == 0 and scoreable == 0
