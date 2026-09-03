"""Tests for scripts/check_learned_lineage.py.

Every test builds a whole fixture tree in tmp_path -- manifest, profile,
phase config, metrics -- so nothing here reads `runs/` or `checkpoints/`,
both of which are gitignored and absent from a clean checkout.

The shape of each test is the same: build the clean tree, assert the
guard is silent on it (the negative control, which is what keeps the
positive cases from being satisfied by an always-red guard), corrupt one
thing, assert the guard names that one thing, restore, assert silent
again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_learned_lineage as guard  # noqa: E402


CLEAN_PROFILE = """\
reward_weights:
  forward_progress: 1.0
  death_penalty: -15.0
  completion_bonus: 50.0
  checkpoint_scale: 0.0
"""

LADDER_PROFILE = """\
reward_weights:
  forward_progress: 1.0
  death_penalty: -15.0
  completion_bonus: 50.0
  checkpoint_scale: 1.0
"""


def _spec(name="fixture 99/100", **over):
    spec = {
        "name": name,
        "manifest": "runs/fixture/manifest.json",
        "profile_key": ("config", "base_profile"),
        "expect_profile": "configs/fixture_profile.yaml",
        "phase_configs_min": 1,
        "metrics": "checkpoints/fixture/metrics.jsonl",
        "metrics_rows_min": 3,
    }
    spec.update(over)
    return spec


def build_tree(root: Path, spec=None, rows=3):
    """A tree that the guard passes: manifest -> clean profile, one clean
    phase config, a metrics log with reward_forward and no
    reward_checkpoint."""
    spec = spec or _spec()
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "fixture_profile.yaml").write_text(CLEAN_PROFILE)
    (root / "configs" / "ladder_profile.yaml").write_text(LADDER_PROFILE)

    run_dir = root / "runs" / "fixture"
    (run_dir / "phase_configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"campaign": "fixture", "config": {
            "run_dir": "runs/fixture",
            "base_profile": "configs/fixture_profile.yaml"}}))
    (run_dir / "phase_configs" / "phase_5.yaml").write_text(CLEAN_PROFILE)

    ck = root / "checkpoints" / "fixture"
    ck.mkdir(parents=True, exist_ok=True)
    ck.joinpath("metrics.jsonl").write_text("".join(
        json.dumps({"iter": i, "reward_forward": 1.5,
                    "reward_time_penalty": -0.01}) + "\n"
        for i in range(rows)))
    return spec


def test_clean_fixture_passes(tmp_path):
    """Negative control. Without this every assertion below is satisfied
    by a guard that always reports a violation."""
    spec = build_tree(tmp_path)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_flipped_manifest(tmp_path):
    """The item's named revert-verify: point one manifest at a profile
    that sets checkpoint_scale: 1.0 and the guard goes red."""
    spec = build_tree(tmp_path)
    manifest = tmp_path / "runs" / "fixture" / "manifest.json"
    original = manifest.read_text()

    doc = json.loads(original)
    doc["config"]["base_profile"] = "configs/ladder_profile.yaml"
    manifest.write_text(json.dumps(doc))

    hits = guard.check_lineage(spec, tmp_path)
    assert hits, "flipped manifest passed the lineage guard"
    assert any("configs/ladder_profile.yaml" in h and "checkpoint_scale: 1.0" in h
               for h in hits), hits
    assert any("banked against configs/fixture_profile.yaml" in h
               for h in hits), hits

    manifest.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_ladder_in_recorded_profile(tmp_path):
    """The manifest pointer is right and the profile itself was edited."""
    spec = build_tree(tmp_path)
    profile = tmp_path / "configs" / "fixture_profile.yaml"
    original = profile.read_text()

    profile.write_text(LADDER_PROFILE)
    hits = guard.check_lineage(spec, tmp_path)
    assert any("configs/fixture_profile.yaml" in h and "checkpoint_scale: 1.0" in h
               for h in hits), hits

    profile.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_ladder_in_phase_config(tmp_path):
    """A base profile at 0.0 with a phase config that turns it back on:
    the case reading the base profile alone would miss."""
    spec = build_tree(tmp_path)
    phase = tmp_path / "runs" / "fixture" / "phase_configs" / "phase_5.yaml"
    original = phase.read_text()

    phase.write_text(LADDER_PROFILE)
    hits = guard.check_lineage(spec, tmp_path)
    assert any("phase_5.yaml" in h and "checkpoint_scale: 1.0" in h
               for h in hits), hits

    phase.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_reward_checkpoint_row(tmp_path):
    """One metrics row logging reward_checkpoint fails the lineage even
    when every config on the paper trail reads 0.0."""
    spec = build_tree(tmp_path)
    metrics = tmp_path / "checkpoints" / "fixture" / "metrics.jsonl"
    original = metrics.read_text()

    rows = original.splitlines()
    row = json.loads(rows[1])
    row["reward_checkpoint"] = 34.96
    rows[1] = json.dumps(row)
    metrics.write_text("\n".join(rows) + "\n")

    hits = guard.check_lineage(spec, tmp_path)
    assert any("reward_checkpoint" in h and "row(s): 2" in h for h in hits), hits

    metrics.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_nested_reward_checkpoint_row(tmp_path):
    """The trainer nests its reward breakdown under a sub-object in some
    schemas; a top-level key scan would call that row clean."""
    spec = build_tree(tmp_path)
    metrics = tmp_path / "checkpoints" / "fixture" / "metrics.jsonl"
    original = metrics.read_text()

    rows = original.splitlines()
    row = json.loads(rows[0])
    row["reward_breakdown"] = {"reward_checkpoint": 34.96}
    rows[0] = json.dumps(row)
    metrics.write_text("\n".join(rows) + "\n")

    hits = guard.check_lineage(spec, tmp_path)
    assert any("reward_checkpoint" in h for h in hits), hits

    metrics.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_emptied_phase_config_dir(tmp_path):
    """Deleting the recorded phase configs must not turn the phase-config
    check into a vacuous pass."""
    spec = build_tree(tmp_path)
    phase = tmp_path / "runs" / "fixture" / "phase_configs" / "phase_5.yaml"
    original = phase.read_text()

    phase.unlink()
    hits = guard.check_lineage(spec, tmp_path)
    assert any("holds 0 phase configs" in h for h in hits), hits

    phase.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_truncated_metrics(tmp_path):
    """A metrics log short of its banked row count cannot clear the
    lineage by having fewer rows to disagree with."""
    spec = build_tree(tmp_path)
    metrics = tmp_path / "checkpoints" / "fixture" / "metrics.jsonl"
    original = metrics.read_text()

    metrics.write_text(original.splitlines()[0] + "\n")
    hits = guard.check_lineage(spec, tmp_path)
    assert any("has 1 rows, fewer than the 3 banked" in h for h in hits), hits

    metrics.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_metrics_with_no_reward_categories(tmp_path):
    """Positive control on the metrics scan: a log that records no reward
    category at all is silent about reward_checkpoint for the wrong
    reason."""
    spec = build_tree(tmp_path)
    metrics = tmp_path / "checkpoints" / "fixture" / "metrics.jsonl"
    original = metrics.read_text()

    metrics.write_text("".join(
        json.dumps({"iter": i, "loss": 0.1}) + "\n" for i in range(3)))
    hits = guard.check_lineage(spec, tmp_path)
    assert any("logs no reward_forward in any row" in h for h in hits), hits

    metrics.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_missing_manifest(tmp_path):
    spec = build_tree(tmp_path)
    manifest = tmp_path / "runs" / "fixture" / "manifest.json"
    original = manifest.read_text()

    manifest.unlink()
    hits = guard.check_lineage(spec, tmp_path)
    assert any("manifest runs/fixture/manifest.json is missing" in h
               for h in hits), hits

    manifest.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_lineage_guard_flags_missing_profile_pointer(tmp_path):
    """The key-path half of the attribution rule: a manifest that does
    not record a profile is a failure, not an absence to be filled in by
    guessing a config name."""
    spec = build_tree(tmp_path)
    manifest = tmp_path / "runs" / "fixture" / "manifest.json"
    original = manifest.read_text()

    doc = json.loads(original)
    del doc["config"]["base_profile"]
    manifest.write_text(json.dumps(doc))

    hits = guard.check_lineage(spec, tmp_path)
    assert any("no profile pointer at config.base_profile" in h
               for h in hits), hits

    manifest.write_text(original)
    assert guard.check_lineage(spec, tmp_path) == []


def test_specialist_key_path_is_read_as_recorded(tmp_path):
    """runs/interference records 1-1 under config.specialists["1-1"].profile,
    not config.base_profile. This asserts the guard follows a nested key
    path, and that the wrong path is reported rather than skipped."""
    spec = build_tree(tmp_path, _spec(
        profile_key=("config", "specialists", "1-1", "profile")))
    manifest = tmp_path / "runs" / "fixture" / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["config"]["specialists"] = {
        "1-1": {"profile": "configs/fixture_profile.yaml"}}
    manifest.write_text(json.dumps(doc))
    assert guard.check_lineage(spec, tmp_path) == []

    doc["config"]["specialists"]["1-1"]["profile"] = "configs/ladder_profile.yaml"
    manifest.write_text(json.dumps(doc))
    hits = guard.check_lineage(spec, tmp_path)
    assert any("checkpoint_scale: 1.0" in h for h in hits), hits


def test_real_lineages_are_pinned_to_their_manifests():
    """The registry itself: five banked lineages, each naming the key
    path its manifest actually uses. Runs on a clean checkout (no
    filesystem reads)."""
    assert len(guard.LINEAGES) == 5
    by_manifest = {s["manifest"]: s for s in guard.LINEAGES}
    assert by_manifest["runs/interference/manifest.json"]["profile_key"] == (
        "config", "specialists", "1-1", "profile")
    for m in ("runs/consol2/manifest.json", "runs/consol2_1_3/manifest.json",
              "runs/consol2_1_3_round2/manifest.json",
              "runs/online_1_4/manifest.json"):
        assert by_manifest[m]["profile_key"] == ("config", "base_profile")
    assert {s["expect_profile"] for s in guard.LINEAGES} == {
        "configs/mario_1_1_backward.yaml",
        "configs/mario_1_2_consol2.yaml",
        "configs/mario_1_3_online_v1.yaml",
        "configs/mario_1_4_online_v1.yaml",
    }
    assert {s["metrics"] for s in guard.LINEAGES} == {
        "checkpoints/mario_1_1_consolidate_exp/metrics.jsonl",
        "checkpoints/mario_1_2_consol2/metrics.jsonl",
        "checkpoints/mario_1_3_online_v1/metrics.jsonl",
        "checkpoints/mario_1_4_online_v1/metrics.jsonl",
    }
    # The names are what ADDENDUM RL-1 quotes; dropping or renaming one
    # silently narrows what `make lineage-check` covers.
    assert [s["name"] for s in guard.LINEAGES] == [
        "1-1 43/100 (metrics: manifest reference-receipt run)",
        "1-2 38/100 shared-stream, 31/100 canonical",
        "1-3 21/100",
        "1-3 21/100 (round 2)",
        "1-4 51/100",
    ]
    # The 1-1 row's metrics file is the interference manifest's
    # reference_role: prior_band_only receipt run, not the 43/100
    # policy's own training log, which no manifest in this repo records.
    # The name has to keep saying so: a rename back to a bare "1-1
    # 43/100" would put an unreceipted provenance claim behind
    # `make lineage-check` and behind ADDENDUM RL-1, which is the exact
    # failure this guard exists to close.
    one_one = by_manifest["runs/interference/manifest.json"]
    assert "reference-receipt" in one_one["name"], one_one["name"]
    assert one_one["metrics"] == (
        "checkpoints/mario_1_1_consolidate_exp/metrics.jsonl")
    # Anti-vacuity floors, pinned from the artifacts as read 2026-09-01.
    assert [s["metrics_rows_min"] for s in guard.LINEAGES] == [120, 326, 30, 30, 109]
    assert [s["phase_configs_min"] for s in guard.LINEAGES] == [0, 0, 1, 1, 1]


@pytest.mark.parametrize("profile", [
    "configs/mario_1_1_backward.yaml",
    "configs/mario_1_2_consol2.yaml",
    "configs/mario_1_3_online_v1.yaml",
    "configs/mario_1_4_online_v1.yaml",
])
def test_banked_profiles_set_the_ladder_off(profile):
    """The four committed profiles, read directly. These are tracked
    files, so this runs on a clean checkout; the manifests and metrics
    that complete the chain are gitignored and live in
    `make lineage-check`."""
    import yaml
    path = ROOT / profile
    if not path.is_file():
        pytest.skip(f"{profile} absent from this checkout")
    doc = yaml.safe_load(path.read_text())
    assert doc["reward_weights"]["checkpoint_scale"] == 0.0
