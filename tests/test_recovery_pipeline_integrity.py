"""Integrity guards on the recovery assay/distill pipeline scripts.

The pipeline adjudicates registered verdicts, and its historical
failure mode is the silent zero: a crashed solver scored as "no
recovery", an alive timeout scored as a death-preceding stick, an eval
whose output cannot be parsed silently disabling the drift stop.
Every test here pins either a loud failure or a correct partition for
one of those paths.  Token-bound — no emulator, no ROM, no solver.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.distill_recovery as distill_recovery  # noqa: E402
from scripts.distill_recovery import parse_clear_rate  # noqa: E402
from scripts.mine_recovery_tapes import select_candidates  # noqa: E402
from scripts.recovery_assay import (  # noqa: E402
    partition_suspects, solver_solutions,
)


def _rec(cleared, length, sticks):
    return {"cleared": cleared, "length": length, "sticks": sticks}


# ---------------------------------------------------------------- assay


def test_partition_excludes_alive_timeouts():
    records = [
        _rec(True, 900, [[10, "s0"]]),            # cleared: not a suspect
        _rec(False, 800, [[100, "a"], [700, "b"]]),  # true death
        _rec(False, 1500, [[300, "c"]]),          # alive timeout: skip
        _rec(False, 1600, [[300, "d"]]),          # over-cap timeout: skip
        _rec(False, 500, []),                     # no sticks
    ]
    suspects, timeouts = partition_suspects(records, max_steps=1500)
    assert [s["episode"] for s in suspects] == [1]
    assert timeouts == [2, 3]
    # the suspect carries the LAST stick before death
    assert suspects[0] == {"episode": 1, "t": 700,
                           "death_step": 800, "state": "b"}


def test_partition_respects_max_steps_argument():
    records = [_rec(False, 1000, [[400, "a"]])]
    suspects, timeouts = partition_suspects(records, max_steps=1000)
    assert suspects == [] and timeouts == [0]


def test_solver_crash_is_loud_not_scored(tmp_path):
    r = SimpleNamespace(returncode=1, stdout="", stderr="Traceback: boom")
    with pytest.raises(RuntimeError, match="exited 1"):
        solver_solutions(r, tmp_path / "solve_x")


def test_missing_solutions_dir_is_loud(tmp_path):
    out_dir = tmp_path / "solve_x"
    out_dir.mkdir()
    r = SimpleNamespace(returncode=0, stdout="done", stderr="")
    with pytest.raises(RuntimeError, match="no solutions dir"):
        solver_solutions(r, out_dir)


def test_solution_count_not_doubled_by_paired_json(tmp_path):
    sol_dir = tmp_path / "solve_x" / "solutions"
    sol_dir.mkdir(parents=True)
    (sol_dir / "sol_000.actions.npy").write_bytes(b"x")
    (sol_dir / "sol_000.json").write_text("{}")
    r = SimpleNamespace(returncode=0, stdout="done", stderr="")
    sols = solver_solutions(r, tmp_path / "solve_x")
    assert len(sols) == 1


def test_collect_refuses_to_clobber_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"records": []}')
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/recovery_assay.py"),
         "collect", "--dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


# ----------------------------------------------------------------- mine


def test_mining_candidates_skip_timeouts_and_cap():
    records = [
        _rec(False, 1000, [[500, "a"], [700, "b"], [900, "c"]]),
        _rec(False, 1500, [[600, "t"]]),           # alive timeout: skip
        _rec(False, 1000, [[100, "early"]]),       # front-half stick only
        _rec(True, 1000, [[500, "clr"]]),          # cleared
    ]
    cands = select_candidates(records, per_episode=2, max_states=10)
    assert all(c["episode"] == 0 for c in cands)
    # spread over the tail: first and last back-half sticks
    assert [c["state"] for c in cands] == ["a", "c"]
    assert select_candidates(records, per_episode=2, max_states=1) == cands[:1]


# -------------------------------------------------------------- distill


def test_parse_clear_rate_reads_result_blob():
    out = "ep 1: clear\n" + json.dumps(
        {"clear_rate": 0.767, "n_episodes": 30})
    assert parse_clear_rate(out) == 0.767
    # eval_game prints its result with indent=2 (multi-line)
    out = "progress line\n" + json.dumps(
        {"clear_rate": 0.5, "n_episodes": 30}, indent=2)
    assert parse_clear_rate(out) == 0.5


def test_parse_clear_rate_rejects_status_blob():
    # early-exit blobs parse as valid JSON but carry no clear_rate;
    # treating them as a rate of None must be surfaced, not swallowed
    assert parse_clear_rate('{"status": "no_checkpoint"}') is None


def test_parse_clear_rate_rejects_empty_and_garbage():
    assert parse_clear_rate("") is None
    assert parse_clear_rate("{not json\nplain line") is None


def test_build_demos_empty_tape_dir_is_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(distill_recovery, "FUEL", tmp_path / "fuel")
    monkeypatch.setattr(distill_recovery, "OUT", tmp_path / "out")
    args = SimpleNamespace(clip=60, profile="unused")
    with pytest.raises(SystemExit, match="no tapes"):
        distill_recovery.build_demos(args)


def test_build_demos_tape_count_must_match_mining_ledger(
        tmp_path, monkeypatch):
    fuel = tmp_path / "fuel"
    (fuel / "tapes").mkdir(parents=True)
    (fuel / "tapes" / "ep000_t00001_sol_000.actions.npy").write_bytes(b"x")
    (fuel / "mining.json").write_text(json.dumps({"tapes": ["a", "b"]}))
    monkeypatch.setattr(distill_recovery, "FUEL", fuel)
    monkeypatch.setattr(distill_recovery, "OUT", tmp_path / "out")
    args = SimpleNamespace(clip=60, profile="unused")
    with pytest.raises(SystemExit, match="mining.json"):
        distill_recovery.build_demos(args)
