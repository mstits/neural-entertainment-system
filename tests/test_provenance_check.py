"""demo_anchor_paths sweep in scripts/provenance_check.py.

demo_anchor_paths entries live in tracked configs/*.yaml but usually
point at runs/ — git-ignored, so the SEEDS rglob (checkpoints/harvested_
seeds/) never sees them. A config referencing a runs/ npz that was never
committed (or got cleaned up) must fail the gate, not pass silently.
"""
from __future__ import annotations

import numpy as np
import pytest

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
