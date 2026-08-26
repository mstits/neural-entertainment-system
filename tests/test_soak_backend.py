"""Regression test for scripts/soak_backend.py's hang-vs-budget outcome.

Covers the confirmed defect: a roster entry whose budget_s is at or
under SolverBackend.hang_timeout_s (210s) can never let the hang
watchdog's `now - last_progress_t > hang_timeout_s` check fire before
the segment's own deadline does, so a solver silently wedged since
segment start (alive, but never appending to progress.jsonl) used to
be misreported as a clean OUTCOME_BUDGET exit instead of OUTCOME_CRASH.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from soak_backend import SolverBackend  # noqa: E402
from soak_harness import OUTCOME_CRASH, RosterEntry  # noqa: E402


class _NeverExitingProcess:
    """Stand-in for subprocess.Popen: alive for the whole segment."""

    pid = -1
    returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return None


class _FakeClock:
    """Deterministic time.monotonic() replacement.

    Advances by `step` on every call so the poll loop's deadline check
    fires on its first pass, regardless of real wall-clock speed.
    """

    def __init__(self, step: float = 1000.0):
        self._n = 0
        self._step = step

    def __call__(self) -> float:
        val = self._n * self._step
        self._n += 1
        return val


def test_short_budget_hang_reports_crash_not_budget(tmp_path, monkeypatch):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("{}\n")
    root_state = tmp_path / "root.state"
    root_state.write_bytes(b"fake")

    entry = RosterEntry(
        name="smoke",
        config=str(profile_path),
        budget_s=120.0,
        root_state=str(root_state),
    )
    assert entry.budget_s <= SolverBackend.hang_timeout_s

    backend = SolverBackend()
    monkeypatch.setattr(backend, "_stop", lambda proc: None)
    monkeypatch.setattr(
        backend, "_read_new_progress", lambda path, pos: ([], pos))
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: _NeverExitingProcess())
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(time, "monotonic", _FakeClock())

    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    result = backend.run_segment(entry, seg_dir)

    assert result.outcome == OUTCOME_CRASH
    assert result.detail["end_reason"] == "hang"
    assert "diagnosis" in result.detail
