"""Tests for scripts/live_solve_show.py's phase-clock/clear-ledger state.

Show.__init__ is cheap to construct for an SMB-engine profile (no ROM
open, no Pool) — audio is forced off so these don't touch real hardware.
Covers the audience-legibility mechanisms added 2026-08-06: the
independent phase-elapsed clock (_set_mode/phase_started_t) and the
per-campaign clear ledger (clear_log), both read by _render_impl on its
own QTimer tick rather than from inside the solver hook.
"""

from __future__ import annotations

import time

from scripts.live_solve_show import Show, default_args


def _show() -> Show:
    return Show(default_args(profile="configs/mario.yaml", audio="off"))


def test_show_starts_in_boot_mode_with_a_fresh_phase_clock() -> None:
    sh = _show()
    assert sh.mode == "boot"
    assert sh.clear_log == []
    assert time.time() - sh.phase_started_t < 5.0


def test_set_mode_updates_mode_and_restarts_the_phase_clock() -> None:
    sh = _show()
    stale_t = sh.phase_started_t = time.time() - 999.0
    sh._set_mode("search")
    assert sh.mode == "search"
    assert sh.phase_started_t > stale_t


def test_clear_log_entries_are_appended_not_replaced() -> None:
    sh = _show()
    sh.clear_log.append({"level": "1-1", "arm_name": "coverage",
                         "attempt": 0, "elapsed_s": 42, "t": time.time()})
    sh.clear_log.append({"level": "1-2", "arm_name": "micro",
                         "attempt": 1, "elapsed_s": 900, "t": time.time()})
    assert len(sh.clear_log) == 2
    assert [e["level"] for e in sh.clear_log] == ["1-1", "1-2"]


def test_every_extract_next_entrance_call_threads_hw_flags() -> None:
    """Every extract_next_entrance() call site must resolve and forward
    hw_flags — otherwise a profile's solve.hw_flags: configures the
    search pool but the post-clear entrance snapshot is minted on the
    stock machine (hw_flags=()), and the next level's Solver.seed()
    trips *** HW-FLAG LINEAGE MISMATCH *** against a root that was
    never actually produced under the configured hardware."""
    import inspect

    for method in (Show._reenter, Show.run):
        src = inspect.getsource(method)
        calls = src.count("extract_next_entrance(")
        threaded = src.count("hw_flags=resolve_hw_flags(self.profile)")
        assert calls > 0
        assert threaded == calls, (
            f"{method.__qualname__}: {calls} extract_next_entrance() "
            f"call(s) but only {threaded} pass hw_flags")
