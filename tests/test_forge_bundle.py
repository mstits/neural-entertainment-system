"""src/forge/bundle.py -- the diagnosis bundle, both wall shapes.

FORGE_SPEC_2026-09-01.md §2b. Each test is revert-verified against a
named corruption in its own docstring. Fixtures are built fresh under
pytest's `tmp_path` -- never real `runs/` data -- so these tests run in
milliseconds and never depend on the multi-GB archives on disk.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge.bundle import (  # noqa: E402
    CERTAINTIES, MECHANISM_CLASSES, _axis_profile, build_bundle,
)

_PATH_RE = re.compile(r"^[A-Za-z0-9_./\-]+(:\d+(-\d+)?)?$")


def _is_path_like(s: str) -> bool:
    return "/" in s and bool(_PATH_RE.match(s))


# --------------------------------------------------------------- fixtures

def _write_manifest(repo: Path, wall_id: str, manifest: dict) -> None:
    walls_dir = repo / "runs" / "forge" / "walls"
    walls_dir.mkdir(parents=True, exist_ok=True)
    (walls_dir / f"{wall_id}.json").write_text(json.dumps(manifest))


def _build_key_blind_archive() -> dict:
    """Cell-key fixture with six constant axes (positions 0,1,2,3,6,7)
    and exactly one live, 1-bit axis (position 8) -- the same shape
    docs/proposals/gate_opener_arm_2026-08-11.md:139-215 measured on the
    real Castlevania hall archive."""
    cells: dict[tuple, None] = {}
    for i in range(50):
        key = (0, 0, 0, 0, i % 3, i % 4, 0, 0, i % 2, (i // 10) % 3, i % 20)
        cells[key] = None
    return cells


def _build_axis_blind_low_count_archive() -> dict:
    """Cell-key fixture (arity 8) with five constant axes (positions
    0,1,2,3,6) and zero live non-spatial, non-bookkeeping axes --
    `interaction_blind` is True but the constant-axis count is one
    short of `CONSTANT_AXES_MIN` (6)."""
    cells: dict[tuple, None] = {}
    for i in range(50):
        key = (0, 0, 0, 0, i % 3, i % 4, 0, i % 20)
        cells[key] = None
    return cells


def _build_axis_not_blind_archive() -> dict:
    """Cell-key fixture (arity 11) with six constant axes (positions
    0,1,2,3,6,9) -- clearing `CONSTANT_AXES_MIN` -- but two live,
    non-spatial, non-bookkeeping axes (7,8), so `interaction_blind`
    reads False."""
    cells: dict[tuple, None] = {}
    for i in range(50):
        key = (0, 0, 0, 0, i % 3, i % 4, 0, i % 2, (i // 10) % 2, 0, i % 20)
        cells[key] = None
    return cells


def _axis_blind_low_count_fixture(repo: Path) -> None:
    member_dir = repo / "member_low_count"
    member_dir.mkdir(parents=True, exist_ok=True)
    with open(member_dir / "archive.pkl", "wb") as f:
        pickle.dump(_build_axis_blind_low_count_archive(), f)
    (member_dir / "progress.jsonl").write_text(
        json.dumps({"elapsed_s": 60, "cells": 50, "steps": 1000,
                    "stall_flat_windows": 0}) + "\n")
    _write_manifest(repo, "axis_blind_low_count_fixture", {
        "wall_id": "axis_blind_low_count_fixture", "prior_best": 100,
        "prior_best_replay_verified": False,
        "missing": [],
        "members": [{"dir": "member_low_count", "shape": "progress"}],
    })


def _axis_not_blind_fixture(repo: Path) -> None:
    member_dir = repo / "member_not_blind"
    member_dir.mkdir(parents=True, exist_ok=True)
    with open(member_dir / "archive.pkl", "wb") as f:
        pickle.dump(_build_axis_not_blind_archive(), f)
    (member_dir / "progress.jsonl").write_text(
        json.dumps({"elapsed_s": 60, "cells": 50, "steps": 1000,
                    "stall_flat_windows": 0}) + "\n")
    _write_manifest(repo, "axis_not_blind_fixture", {
        "wall_id": "axis_not_blind_fixture", "prior_best": 100,
        "prior_best_replay_verified": False,
        "missing": [],
        "members": [{"dir": "member_not_blind", "shape": "progress"}],
    })


def _cv_hall_fixture(repo: Path) -> None:
    member_dir = repo / "member_a"
    member_dir.mkdir(parents=True, exist_ok=True)
    with open(member_dir / "archive.pkl", "wb") as f:
        pickle.dump(_build_key_blind_archive(), f)
    (member_dir / "progress.jsonl").write_text(
        json.dumps({"elapsed_s": 60, "cells": 50, "steps": 1000,
                    "stall_flat_windows": 0}) + "\n")
    _write_manifest(repo, "cv_fixture", {
        "wall_id": "cv_fixture", "prior_best": 767,
        "prior_best_replay_verified": False,
        "missing": ["replay_verified_frontier: fixture gap"],
        "members": [{"dir": "member_a", "shape": "progress"}],
    })


def _contra_fixture(repo: Path) -> None:
    a8_dir = repo / "A8"
    a8_dir.mkdir(parents=True, exist_ok=True)
    (a8_dir / "A8_result.json").write_text(json.dumps({
        "camera_ever_moved": False, "breakthrough_found": False,
        "max_progress_seen": 3072,
    }))
    a6_dir = repo / "A6"
    a6_dir.mkdir(parents=True, exist_ok=True)
    (a6_dir / "A6_RECEIPT.json").write_text(json.dumps({
        "beat_3072": False, "max_gx": 3072,
        "objective": {
            "what_is_maximised": "cumulative damage",
            "why_not_score_bonus": ("this is a genuine scoring-cliff "
                                    "defect in the shipped adapter"),
        },
    }))
    _write_manifest(repo, "contra_fixture", {
        "wall_id": "contra_fixture", "prior_best": 3072,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": "A8", "shape": "receipt", "receipt": "A8_result.json",
             "terminal_field": "breakthrough_found",
             "best_field": "max_progress_seen", "root_family": "head_wall"},
            {"dir": "A6", "shape": "receipt", "receipt": "A6_RECEIPT.json",
             "terminal_field": "beat_3072", "best_field": "max_gx",
             "root_family": "solve20"},
        ],
    })


# ------------------------------------------------------------------ tests

def test_axis_profile_matches_the_real_boundary_axis_profile():
    """`_axis_profile` is a DELIBERATE DUPLICATE of `boundary_axis_profile`
    in `src/training/` (module docstring), not an import of it -- so
    nothing at import time proves the duplicate still computes what the
    real function computes. This module lives outside that guard's
    scanned roots (`src`, `scripts`, `nes_core`, `configs`) and is free
    to import the real function directly to check, field for field, on
    a non-trivial key set: exact dict equality against
    `boundary_axis_profile(cells, ...).as_dict()`, including
    `calibration`, so a caller genuinely cannot tell which one produced
    a given profile.

    Revert-verify: in `bundle.py`'s `_axis_profile`, change
    `constant = [i for i, c in enumerate(card) if c <= 1]` to `c < 1`
    (silently redefining "constant" as "always zero" instead of "one
    distinct value" -- a plausible drift, not a typo). `constant_axes`
    and therefore `live_state_axes` diverge from the real function's on
    this fixture and this test fails on the dict-equality assertion.
    """
    from src.training.wall_taxonomy import boundary_axis_profile

    cells = _build_key_blind_archive()
    ours = _axis_profile(cells, bookkeeping=(4, 5))
    real = boundary_axis_profile(cells, bookkeeping=(4, 5)).as_dict()
    assert ours == real


def test_bundle_cv_hall_reports_key_blind_from_receipt(tmp_path):
    """Six constant axes + one live 1-bit axis -> KEY_BLIND,
    confirmed_by_receipt, citing the gate-opener receipt.

    Revert-verify: change `_classify_key_blind` to drop the
    `len(constant_axes) >= CONSTANT_AXES_MIN` clause (check
    `interaction_blind` alone) -- still passes on this fixture, so this
    alone would not catch it; the actual corruption this test guards is
    dropping the constant-axis COUNT: replace `CONSTANT_AXES_MIN` with a
    value the fixture's six constant axes cannot clear (e.g. 7). Then
    mechanism_class reads [{"class": "UNKNOWN", ...}] instead of
    KEY_BLIND, and this test fails on the class assertion below.
    """
    _cv_hall_fixture(tmp_path)
    verdict = {"verdict": "STALLED", "kind": "campaign", "wall_id": "cv_fixture",
              "missing": ["replay_verified_frontier: fixture gap"]}
    bundle = build_bundle("cv_fixture", verdict, repo=tmp_path)

    assert bundle["wall_id"] == "cv_fixture"
    assert bundle["frontier_shape"]["certainty"] == "confirmed_by_receipt"
    assert bundle["frontier_shape"]["data"]["constant_axes"] == [0, 1, 2, 3, 6, 7]
    assert bundle["frontier_shape"]["data"]["interaction_blind"] is True
    assert bundle["mechanism_class"] == [
        {"class": "KEY_BLIND", "certainty": "confirmed_by_receipt",
         "receipt": "docs/proposals/gate_opener_arm_2026-08-11.md:139-215"}]
    assert bundle["ram_observables"] == {"certainty": "not_probed", "data": None}
    assert "replay_verified_frontier: fixture gap" in bundle["missing"]


def test_bundle_low_constant_count_yields_unknown_not_key_blind(tmp_path):
    """Five constant axes (one short of `CONSTANT_AXES_MIN`) with
    `interaction_blind` True must NOT read KEY_BLIND -- mechanism_class
    reads `[UNKNOWN candidate]` citing the wall's own manifest path.
    Proves the `len(constant_axes) >= CONSTANT_AXES_MIN` conjunct in
    `_classify_key_blind` is load-bearing: the existing cv_hall test's
    own docstring already notes that dropping this conjunct while
    leaving `interaction_blind` alone still passes on ITS fixture; this
    fixture is built so that is no longer true anywhere in the suite.

    Revert-verify: in `_classify_key_blind`, drop the
    `len(constant_axes) >= CONSTANT_AXES_MIN` clause so only
    `interaction_blind` gates the class -- mechanism_class flips to
    KEY_BLIND on this fixture and the assertion below fails.
    """
    _axis_blind_low_count_fixture(tmp_path)
    verdict = {"verdict": "STALLED", "kind": "campaign",
               "wall_id": "axis_blind_low_count_fixture", "missing": []}
    bundle = build_bundle("axis_blind_low_count_fixture", verdict, repo=tmp_path)

    assert bundle["frontier_shape"]["certainty"] == "confirmed_by_receipt"
    assert bundle["frontier_shape"]["data"]["constant_axes"] == [0, 1, 2, 3, 6]
    assert bundle["frontier_shape"]["data"]["interaction_blind"] is True
    assert bundle["mechanism_class"] == [
        {"class": "UNKNOWN", "certainty": "candidate",
         "receipt": "runs/forge/walls/axis_blind_low_count_fixture.json"}]


def test_bundle_not_interaction_blind_yields_unknown_not_key_blind(tmp_path):
    """Six constant axes clears `CONSTANT_AXES_MIN`, but two live
    non-spatial, non-bookkeeping axes make `interaction_blind` False --
    mechanism_class must read `[UNKNOWN candidate]` citing the wall's
    own manifest path, never KEY_BLIND. Proves the `interaction_blind`
    conjunct in `_classify_key_blind` is independently load-bearing
    (the axis count alone is not sufficient).

    Revert-verify: in `_classify_key_blind`, drop the
    `interaction_blind` conjunct so only the axis count gates the class
    -- mechanism_class flips to KEY_BLIND on this fixture and the
    assertion below fails.
    """
    _axis_not_blind_fixture(tmp_path)
    verdict = {"verdict": "STALLED", "kind": "campaign",
               "wall_id": "axis_not_blind_fixture", "missing": []}
    bundle = build_bundle("axis_not_blind_fixture", verdict, repo=tmp_path)

    assert bundle["frontier_shape"]["certainty"] == "confirmed_by_receipt"
    assert bundle["frontier_shape"]["data"]["constant_axes"] == [0, 1, 2, 3, 6, 9]
    assert bundle["frontier_shape"]["data"]["interaction_blind"] is False
    assert bundle["mechanism_class"] == [
        {"class": "UNKNOWN", "certainty": "candidate",
         "receipt": "runs/forge/walls/axis_not_blind_fixture.json"}]


def test_bundle_contra_from_receipts_only_yields_two_classes(tmp_path):
    """A receipt-only wall with an A8-shaped member (`camera_ever_moved:
    false`) and an A6-shaped member (an `objective` field whose own text
    names a defect) yields exactly [SCRIPTED_RELEASE candidate,
    OBSERVABLE_DEFECT confirmed_by_receipt]; `ram_observables.certainty`
    stays `not_probed`.

    Revert-verify: change `build_bundle`'s `ram_observables` literal from
    `{"certainty": "not_probed", ...}` to `{"certainty": "confirmed_by_receipt", ...}`
    -- this test's `ram_observables` assertion fails.
    """
    _contra_fixture(tmp_path)
    verdict = {"verdict": "STALLED", "kind": "campaign", "wall_id": "contra_fixture"}
    bundle = build_bundle("contra_fixture", verdict, repo=tmp_path)

    assert bundle["frontier_shape"] == {"certainty": "not_probed", "data": None}
    classes = {c["class"]: c for c in bundle["mechanism_class"]}
    assert set(classes) == {"SCRIPTED_RELEASE", "OBSERVABLE_DEFECT"}
    assert classes["SCRIPTED_RELEASE"]["certainty"] == "candidate"
    assert classes["SCRIPTED_RELEASE"]["receipt"] == "A8/A8_result.json"
    assert classes["OBSERVABLE_DEFECT"]["certainty"] == "confirmed_by_receipt"
    assert classes["OBSERVABLE_DEFECT"]["receipt"] == "A6/A6_RECEIPT.json"
    assert bundle["ram_observables"]["certainty"] == "not_probed"


def test_bundle_never_contains_gated_vocabulary(tmp_path):
    """No struck-classification wording or "saturated" survives into a
    serialized bundle, for either wall shape.

    Revert-verify: temporarily append `" (GATED)"` to one of
    `build_bundle`'s hardcoded `missing` strings -- this test fails on
    the substring check.
    """
    _cv_hall_fixture(tmp_path)
    _contra_fixture(tmp_path)
    cv_bundle = build_bundle("cv_fixture", {"missing": []}, repo=tmp_path)
    contra_bundle = build_bundle("contra_fixture", {"missing": []}, repo=tmp_path)

    for bundle in (cv_bundle, contra_bundle):
        text = json.dumps(bundle).lower()
        assert "gated" not in text
        assert "saturated" not in text


def test_bundle_string_fields_are_controlled_vocabulary_or_paths(tmp_path):
    """Every string leaf under `frontier_shape`/`cell_rate_history`/
    `ram_observables`/`mechanism_class` is a controlled-vocabulary member
    (a `certainty` or `class` value) or a repo path / `path:line` receipt
    -- never free text -- and no field outside the fixed schema exists.

    Revert-verify: add `bundle["mechanism_class"][0]["notes"] = "looks
    like a wire fault"` inside `build_bundle` before returning -- this
    test fails on the per-entry key-set assertion.
    """
    _contra_fixture(tmp_path)
    bundle = build_bundle("contra_fixture", {"missing": []}, repo=tmp_path)

    assert set(bundle.keys()) == {
        "wall_id", "verdict", "frontier_shape", "cell_rate_history",
        "ram_observables", "mechanism_class", "arms_tried", "missing",
    }
    assert isinstance(bundle["wall_id"], str)

    for section in ("frontier_shape", "cell_rate_history", "ram_observables"):
        assert set(bundle[section].keys()) == {"certainty", "data"}
        assert bundle[section]["certainty"] in CERTAINTIES

    for entry in bundle["mechanism_class"]:
        assert set(entry.keys()) == {"class", "certainty", "receipt"}
        assert entry["class"] in MECHANISM_CLASSES
        assert entry["certainty"] in CERTAINTIES
        assert _is_path_like(entry["receipt"]), entry["receipt"]

    assert bundle["arms_tried"] == []
    assert isinstance(bundle["missing"], list)
    assert all(isinstance(m, str) for m in bundle["missing"])
