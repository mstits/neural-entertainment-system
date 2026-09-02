"""src/forge/stall.py -- the STALLED verdict, both kinds.

FORGE_SPEC_2026-09-01.md §2a. Each test is revert-verified against a
named corruption (in the test's own docstring) and, where the real
cv_hall/contra_wall data on disk is the fixture, checked against
numbers pulled straight from the repo's own runs/ directory rather than
hand-picked constants -- so a future edit to those receipts is a
red test, not a silently-stale one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge.stall import (  # noqa: E402
    EFFORT_MIN_STEPS, FROZEN_WINDOWS_MAX, archive_verdict, campaign_verdict,
)
from tests.skip_gates import requires  # noqa: E402

# The two tests below read run data off the real repo root. runs/ is gitignored
# (.gitignore:86), so a clean clone carries no member run directories. Each
# test gates on the files it reads that a clean clone actually lacks, not on
# the repo root, which is always present and therefore gates nothing.
CV_HALL_TAIL = "runs/cv_hall_ortho_ctrl/progress.jsonl"
# One member's data per wall. The two wall manifests those members belong to,
# runs/forge/walls/{cv_hall,contra_wall}.json, are TRACKED despite the runs/
# ignore (git ls-files runs/forge lists both), so every clone carries them and
# gating on them would gate nothing. What a clean clone lacks is the member
# dirs the manifests point at: campaign_verdict then measures nothing and
# returns UNMEASURED, which is the STALLED != UNMEASURED failure this gate
# turns into a skip.
WALL_MEMBER_DATA = (
    "runs/cv_hall_ortho_a/roots.json",
    "runs/contra_wall/A1/boundary_probe.json",
)


def test_frozen_windows_max_and_effort_min_steps_match_wall_taxonomy():
    """src/forge/stall.py deliberately DUPLICATES these two constants
    rather than importing src.training.wall_taxonomy (that module is
    kept runtime-inert on purpose; importing it from src/forge/ would
    fail tests/test_wall_taxonomy.py::test_module_is_not_wired_into_any_
    runtime_dispatch by name). This is the tripwire that keeps the
    duplicate honest: tests/ is exempt from that isolation guard, so
    this file may import wall_taxonomy directly to compare values.

    Revert-verify: change stall.py's FROZEN_WINDOWS_MAX to 11 (or
    EFFORT_MIN_STEPS to any other value) and this fails.
    """
    from src.training import wall_taxonomy as wt

    assert FROZEN_WINDOWS_MAX == wt.FROZEN_WINDOWS_MAX
    assert EFFORT_MIN_STEPS == wt.EFFORT_MIN_STEPS


# --------------------------------------------------------------- archive

def test_archive_verdict_stalled_at_threshold():
    """One baseline row (primes the watchdog) plus 12 flat rows at the
    same cell count, steps clearing EFFORT_MIN_STEPS -> STALLED with
    evidence.flat_windows == 12.

    Revert-verify: change the `>=` on flat_windows in archive_verdict to
    `>` and this fails -- 12 flat rows would then read WATCHING, not
    STALLED, at exactly the threshold this test pins.
    """
    tail = [{"cells": 500, "steps": 100}]
    for i in range(12):
        tail.append({"cells": 500, "steps": 100 + (i + 1) * 25_000})
    v = archive_verdict(tail, wall_id="w")
    assert v["verdict"] == "STALLED"
    assert v["kind"] == "archive"
    assert v["evidence"]["flat_windows"] == 12
    assert v["evidence"]["steps"] >= 250_000


@requires(CV_HALL_TAIL)
def test_archive_verdict_advancing_on_real_cv_hall_tail():
    """The real last 5 rows of runs/cv_hall_ortho_ctrl/progress.jsonl --
    cells strictly increasing every row -- must replay to ADVANCING.

    Revert-verify: hardcode the verdict to "STALLED" inside
    archive_verdict and this fails against real, still-growing data.
    """
    lines = (ROOT / "runs" / "cv_hall_ortho_ctrl" / "progress.jsonl").read_text().splitlines()
    tail = [json.loads(line) for line in lines[-5:]]
    assert tail == sorted(tail, key=lambda r: r["cells"]), (
        "fixture assumption broken: the real tail is no longer monotone "
        "increasing in cells -- re-pick the tail rows for this test")
    v = archive_verdict(tail, wall_id="cv_hall_ortho_ctrl")
    assert v["verdict"] == "ADVANCING"
    assert v["evidence"]["flat_windows"] == 0


def test_archive_verdict_unmeasured_on_empty_tail():
    """Zero rows -> UNMEASURED, never ADVANCING.

    Revert-verify: change the empty-tail branch's default verdict to
    "ADVANCING" and this fails.
    """
    v = archive_verdict([], wall_id="w")
    assert v["verdict"] == "UNMEASURED"
    assert v["verdict"] != "ADVANCING"


# -------------------------------------------------------------- campaign

def _write_progress_member(d: Path, *, cells: int, solutions: int,
                            steps: int, best_score: int,
                            roots_bytes: bytes) -> None:
    d.mkdir(parents=True, exist_ok=True)
    row = {"t": "2026-01-01T00:00:00", "elapsed_s": 100, "cells": cells,
           "max_area": 0, "max_gx_in_max_area": best_score, "max_sect": 0,
           "solutions": solutions, "best_sol_actions": None,
           "steps": steps, "sps": 1000, "stall_flat_windows": 0}
    (d / "progress.jsonl").write_text(json.dumps(row) + "\n")
    (d / "archive.stats.json").write_text(json.dumps({"best_score": best_score}))
    (d / "roots.json").write_bytes(roots_bytes)


def _write_receipt_member(d: Path, name: str, fields: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(fields))


def test_campaign_stalled_on_progress_shaped_family(tmp_path):
    """Three fixture dirs, byte-identical roots.json, best_score/
    solutions matching the real cv_hall shape -> STALLED,
    terminal_runs:3, distinct_roots:1.

    Revert-verify: change the root-id computation from
    sha256(roots.json bytes) to counting directories instead (so three
    distinct dirs always read distinct_roots:3) -- the assertion below
    on distinct_roots==1 then fails even though terminal_runs stays 3.
    """
    roots = b'{"root": "entrance_after_2.state"}'
    for name in ("m1", "m2", "m3"):
        _write_progress_member(tmp_path / name, cells=131561, solutions=0,
                                steps=8_000_000, best_score=767,
                                roots_bytes=roots)
    manifest = {
        "wall_id": "fixture_cv", "prior_best": 767,
        "prior_best_replay_verified": False,
        "members": [{"dir": n, "shape": "progress"} for n in ("m1", "m2", "m3")],
    }
    v = campaign_verdict(manifest, repo=tmp_path)
    assert v["verdict"] == "STALLED"
    assert v["evidence"]["terminal_runs"] == 3
    assert v["evidence"]["advances"] == 0
    assert v["evidence"]["distinct_roots"] == 1
    assert v["degraded"] is True


def test_campaign_stalled_on_receipt_family_with_prose_only_member(tmp_path):
    """Three falsy-boolean receipts (beat_3072: false) plus one receipt
    whose only field is a prose "verdict" string (no boolean terminal
    field) -> STALLED, terminal_runs:3, members_unmeasured:["A4"].

    Revert-verify: in _member_receipt, treat a non-boolean/missing
    terminal_field value as falsy (parse the prose as terminal_no_advance
    anyway) instead of returning unmeasured -- terminal_runs becomes 4
    and members_unmeasured / missing go empty, and both assertions below
    fail.
    """
    for name in ("A1", "A2", "A3"):
        _write_receipt_member(tmp_path / name, "r.json",
                               {"beat_3072": False, "global_max_gx": 3072})
    _write_receipt_member(tmp_path / "A4", "r.json",
                           {"verdict": "FALSIFIED at this budget"})
    manifest = {
        "wall_id": "fixture_contra", "prior_best": 3072,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": "A1", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat_3072", "best_field": "global_max_gx",
             "root_family": "solve20"},
            {"dir": "A2", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat_3072", "best_field": "global_max_gx",
             "root_family": "solve20"},
            {"dir": "A3", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat_3072", "best_field": "global_max_gx",
             "root_family": "head_wall"},
            {"dir": "A4", "shape": "receipt", "receipt": "r.json",
             "terminal_field": None, "best_field": None,
             "root_family": "solve20"},
        ],
    }
    v = campaign_verdict(manifest, repo=tmp_path)
    assert v["verdict"] == "STALLED"
    assert v["evidence"]["terminal_runs"] == 3
    assert v["evidence"]["members_unmeasured"] == ["A4"]
    assert v["missing"], "A4's gap must be named in `missing`, not omitted"


def test_campaign_advancing_when_one_member_beats_prior(tmp_path):
    """One receipt member's best_field beats prior_best (800 > 767) ->
    ADVANCING, regardless of how many other members are terminal.

    Revert-verify: drop the `best_seen > prior_best` half of the advance
    test in campaign_verdict (only "solutions > 0" left) and this fails,
    since none of these fixture members ever report a solutions field.
    """
    _write_receipt_member(tmp_path / "beats", "r.json", {"score": 800})
    _write_receipt_member(tmp_path / "flat1", "r.json", {"score": 767})
    _write_receipt_member(tmp_path / "flat2", "r.json", {"score": 767})
    manifest = {
        "wall_id": "fixture_advance", "prior_best": 767,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": "beats", "shape": "receipt", "receipt": "r.json",
             "terminal_field": None, "best_field": "score",
             "root_family": "x"},
            {"dir": "flat1", "shape": "receipt", "receipt": "r.json",
             "terminal_field": None, "best_field": "score",
             "root_family": "x"},
            {"dir": "flat2", "shape": "receipt", "receipt": "r.json",
             "terminal_field": None, "best_field": "score",
             "root_family": "x"},
        ],
    }
    v = campaign_verdict(manifest, repo=tmp_path)
    assert v["verdict"] == "ADVANCING"
    assert v["evidence"]["advances"] == 1


@requires(*WALL_MEMBER_DATA)
def test_campaign_verdict_matches_real_cv_hall_and_contra_wall_manifests():
    """The two manifests the build lands (runs/forge/walls/{cv_hall,
    contra_wall}.json) against the real repo data, pinned to the exact
    numbers FORGE_SPEC_2026-09-01.md §4 row (a) pre-registers.

    Revert-verify: change _member_progress's root-id computation from
    sha256(roots.json bytes) to str(dir_path) (same corruption as
    test_campaign_stalled_on_progress_shaped_family, exercised here
    against the real on-disk cv_hall manifest instead of a fixture) --
    cv_hall's three real member dirs are of course all distinct paths,
    so distinct_roots becomes 3 and the assertion below fails.
    """
    from src.forge.stall import load_wall_manifest

    cv = campaign_verdict(load_wall_manifest("cv_hall"), repo=ROOT)
    assert cv["verdict"] == "STALLED"
    assert cv["evidence"]["terminal_runs"] == 3
    assert cv["evidence"]["advances"] == 0
    assert cv["evidence"]["distinct_roots"] == 1
    assert cv["degraded"] is True

    contra = campaign_verdict(load_wall_manifest("contra_wall"), repo=ROOT)
    assert contra["verdict"] == "STALLED"
    assert contra["evidence"]["terminal_runs"] == 7
    assert contra["evidence"]["distinct_roots"] == 2
    assert contra["evidence"]["members_unmeasured"] == ["A4"]


# ------------------------------------------------------------------ wiring

def test_engine_journal_byte_identical_without_forge_flag(tmp_path, monkeypatch):
    """tick(dry=True) with a STALLED wall manifest sitting on disk, the
    Forge module fully importable, but `--forge` (forge=False, the
    default) absent: the journal is exactly the one "idle" row a
    pre-Forge engine would have written -- byte-identical.

    Revert-verify: in tick(), change `if forge: stall_check(state, repo)`
    to call stall_check(state, repo) unconditionally -- the STALLED
    fixture wall then journals an extra "stall_verdict" row before the
    idle row and this fails.
    """
    import scripts.engine_driver as ed

    engine_dir = tmp_path / "engine"
    monkeypatch.setattr(ed, "ENGINE_DIR", engine_dir)
    monkeypatch.setattr(ed, "JOURNAL", engine_dir / "journal.jsonl")
    monkeypatch.setattr(ed, "STATE_PATH", engine_dir / "state.json")
    monkeypatch.setattr(ed, "STALL_RECEIPTS", engine_dir / "stall_receipts.jsonl")
    monkeypatch.setattr(ed, "plan", lambda state, repo=ed.REPO, emulator_only=None: None)
    # Freeze journal()'s timestamp so the "byte-identical" claim below is
    # a literal byte comparison, not a same-shape-minus-clock one.
    monkeypatch.setattr(ed.time, "time", lambda: 1735689600.0)

    walls_dir = tmp_path / "runs" / "forge" / "walls"
    walls_dir.mkdir(parents=True)
    stalled_manifest = {
        "wall_id": "fixture_wall", "prior_best": 767,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": "m1", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat", "best_field": "score",
             "root_family": "x"},
            {"dir": "m2", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat", "best_field": "score",
             "root_family": "x"},
            {"dir": "m3", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat", "best_field": "score",
             "root_family": "x"},
        ],
    }
    (walls_dir / "fixture_wall.json").write_text(json.dumps(stalled_manifest))
    for name in ("m1", "m2", "m3"):
        _write_receipt_member(tmp_path / name, "r.json",
                               {"beat": False, "score": 700})

    state = {"attempts": {}, "consecutive_failures": 0, "halted": None,
             "running": None, "completed": {}}
    rec = ed.tick(state, repo=tmp_path, dry=True, forge=False)

    assert rec["decision"] == "idle"
    journal_bytes = ed.JOURNAL.read_bytes()
    expected = (json.dumps({"type": "tick", "decision": "idle",
                             "reason": "nothing left in the computed plan",
                             "t": 1735689600.0})
                + "\n").encode()
    assert journal_bytes == expected
    assert not ed.STALL_RECEIPTS.exists()
    assert "stalled_notified" not in state
