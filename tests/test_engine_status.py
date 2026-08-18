"""scripts/engine_status.py — the board must READ, never infer.

Each test pins a behaviour that a real failure in this project taught:
terminal campaign state comes from the log (not from whether a process
exists), roll-up numbers are cross-checked against receipted entries,
and quarantined runs never appear beside live ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.engine_status as es  # noqa: E402


def _log(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    p = d / "campaign.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_completed_campaign_is_read_from_the_log_not_the_process_table(tmp_path):
    """The 2026-08-17 error: death inferred from a missing OS process.

    A log ending in campaign_complete must summarize as complete even
    though no process is running.
    """
    p = _log(tmp_path, "r", [
        {"type": "phase_start", "name": "consolidation"},
        {"type": "probe", "env_steps": 3e7, "median_max_x": 1000,
         "clear_rate_strict": 0.1},
        {"type": "campaign_complete"},
    ])
    out = es.summarize_campaign(p)
    assert "campaign_complete" in out
    assert "in progress" not in out


def test_abort_reason_is_surfaced(tmp_path):
    p = _log(tmp_path, "r", [
        {"type": "phase_start", "name": "sticky_local"},
        {"type": "abort", "reason": "trainer subprocess exited -9 in phase 2"},
    ])
    out = es.summarize_campaign(p)
    assert "abort" in out and "exited -9" in out


def test_no_terminal_row_is_reported_as_unknown_not_as_success(tmp_path):
    p = _log(tmp_path, "r", [{"type": "phase_start", "name": "local_clear"}])
    out = es.summarize_campaign(p)
    assert "in progress (no terminal row)" in out


def test_gate_and_honest_probes_are_reported_separately(tmp_path):
    """Rung-local deterministic gates are NOT honest clears; never merge them."""
    p = _log(tmp_path, "r", [
        {"type": "gate_probe", "median_max_x": 3299, "clear_rate_strict": 1.0},
        {"type": "probe", "env_steps": 1e7, "median_max_x": 716,
         "clear_rate_strict": 0.0},
    ])
    out = es.summarize_campaign(p)
    assert "gate(rung,det)" in out and "honest(entrance)" in out
    assert "strict=1.0" in out and "strict=0.0" in out


def test_unreadable_log_is_reported_not_raised(tmp_path):
    missing = tmp_path / "runs" / "gone" / "campaign.jsonl"
    out = es.summarize_campaign(missing)
    assert "unreadable" in out


def test_quarantined_runs_are_excluded_from_live_campaigns(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "REPO", tmp_path)
    _log(tmp_path, "online_2_1", [{"type": "campaign_complete"}])
    _log(tmp_path, "online_2_1_INVALID_1_3_rungs", [{"type": "abort"}])
    names = [p.parent.name for p in es.campaign_logs()]
    assert "online_2_1" in names
    assert not any("INVALID" in n for n in names)


def test_rollup_mismatch_is_flagged(tmp_path, monkeypatch):
    """1-1's 43% lived only in prose. That must never pass silently."""
    monkeypatch.setattr(es, "REPO", tmp_path)
    (tmp_path / "CLAIMS.md").write_text(
        "now holds four levels: 1-1 43%, 1-2 38%\n"
        "1-2 scored **38/100 clears** with receipts.\n")
    c = es.check_ledger_receipts()
    assert not c.ok
    assert "43" in c.detail


def test_rollup_all_receipted_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "REPO", tmp_path)
    (tmp_path / "CLAIMS.md").write_text(
        "now holds two levels: 1-2 38%, 1-3 21%\n"
        "**38/100 clears** ... **21/100 (21.0%)**\n")
    c = es.check_ledger_receipts()
    assert c.ok, c.detail


def test_short_episode_counts_are_not_treated_as_honest_claims(tmp_path,
                                                               monkeypatch):
    """30-episode probes overstate (the winner's curse); >=100 only."""
    monkeypatch.setattr(es, "REPO", tmp_path)
    (tmp_path / "CLAIMS.md").write_text(
        "now holds one level: 1-2 40%\n**12/30** on a peak probe\n")
    c = es.check_ledger_receipts()
    assert not c.ok, "a 30-episode probe must not satisfy a roll-up entry"


def test_check_flag_returns_nonzero_when_a_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(es, "RIGOR", (
        lambda: es.Check("always fails", False, "d", "f"),))
    assert es.main(["--check"]) == 1


def test_check_flag_returns_zero_when_all_pass(monkeypatch):
    monkeypatch.setattr(es, "RIGOR", (
        lambda: es.Check("always passes", True, "d", "f"),))
    assert es.main(["--check"]) == 0


def test_a_raising_check_is_a_failing_check(monkeypatch):
    def boom() -> es.Check:
        raise RuntimeError("kaboom")
    monkeypatch.setattr(es, "RIGOR", (boom,))
    board = es.build_board()
    assert board.checks and not board.checks[0].ok
    assert "kaboom" in board.checks[0].detail


def test_json_output_is_valid(monkeypatch, capsys):
    monkeypatch.setattr(es, "RIGOR", (
        lambda: es.Check("c", True, "d", "f"),))
    es.main(["--json"])
    json.loads(capsys.readouterr().out)


def test_a_terminal_row_before_a_resume_is_superseded(tmp_path):
    """The tool's own version of the failure it guards.

    A resume appends a fresh campaign_start. The previous attempt's abort
    is still the last terminal-typed row in the file, so a naive
    backwards scan reports a live, progressing run as aborted.
    """
    p = _log(tmp_path, "r", [
        {"type": "phase_start", "name": "sticky_local"},
        {"type": "abort", "reason": "trainer subprocess exited -9"},
        {"type": "campaign_start", "start_phase": 2},
        {"type": "phase_start", "name": "reverse_walk"},
        {"type": "probe", "env_steps": 6.7e7, "median_max_x": 1016,
         "clear_rate_strict": 0.0},
    ])
    out = es.summarize_campaign(p)
    assert "abort" not in out, out
    assert "resumed" in out
    assert "reverse_walk" in out


def test_an_abort_after_the_last_resume_is_still_reported(tmp_path):
    p = _log(tmp_path, "r", [
        {"type": "campaign_start"},
        {"type": "phase_start", "name": "local_clear"},
        {"type": "abort", "reason": "real failure"},
    ])
    assert "abort" in es.summarize_campaign(p)
