"""
Regression coverage for `Watchdog._dump_stacks` cooldown handling.

The stack dump is meant to be a post-mortem safety net: a growth-triggered
dump must not suppress a subsequently more urgent threshold-triggered dump
(or vice versa) inside the shared 30s cooldown window, since that leaves
the on-disk stack trace stale right when a crash or freeze is imminent.
"""

from __future__ import annotations

import subprocess

from src.gui.watchdog import Watchdog


def test_threshold_dump_not_suppressed_by_recent_growth_dump(tmp_path, monkeypatch):
    wd = Watchdog(log_dir=tmp_path)
    wd._stack_path = tmp_path / "watchdog-test.stacks.txt"

    now = [1_000_000.0]
    monkeypatch.setattr("src.gui.watchdog.time.time", lambda: now[0])

    wd._dump_stacks(reason="growth", rss=8_000, growth=4_500)

    now[0] += 5.0
    wd._dump_stacks(reason="threshold", rss=61_000, growth=100)

    contents = wd._stack_path.read_text()
    assert "reason=growth" in contents
    assert "reason=threshold" in contents


def test_sample_omits_threads_when_ps_fails(tmp_path, monkeypatch):
    wd = Watchdog(log_dir=tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["ps", "-M"]:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.gui.watchdog.subprocess.run", fake_run)

    sample = wd._sample(start_ts=0.0)

    assert "threads" not in sample
