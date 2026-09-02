"""The shared fingerprinted run-lock (src/utils/run_lock.py).

Extracted from train_game.py after two dated concurrent-writer
incidents (duplicate chain watchers on one ladder 2026-08-29; the
update-ref race before that). The contract under test: one live holder
per lockfile, stale locks (dead PID / reused PID / unparseable) are
reclaimed, and the CLI wrapper refuses a second instance with exit 75.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.utils.run_lock as run_lock  # noqa: E402
from src.utils.run_lock import (  # noqa: E402
    EXIT_HELD,
    acquire,
    acquire_resource,
    lock_pid_is_live,
    pid_start_time,
    read_lock,
    release,
    release_resource,
)


def test_acquire_fresh_then_release(tmp_path):
    lock = tmp_path / ".run.lock"
    assert acquire(lock, extra="roms/x.nes") is None
    holder = read_lock(lock)
    assert holder.pid == os.getpid()
    assert "roms/x.nes" in holder.raw
    release(lock)
    assert not lock.exists()


def test_second_acquire_refuses_while_holder_lives(tmp_path):
    lock = tmp_path / ".run.lock"
    assert acquire(lock) is None
    # Same process is trivially alive: the second acquire must refuse
    # and name the holder.
    holder = acquire(lock)
    assert holder is not None and holder.pid == os.getpid()


def test_stale_dead_pid_is_reclaimed(tmp_path):
    lock = tmp_path / ".run.lock"
    # A child that has exited: its PID is (very likely) unclaimed, and
    # even if recycled the recorded fingerprint won't match.
    proc = subprocess.run([sys.executable, "-c", "print('x')"],
                          capture_output=True)
    assert proc.returncode == 0
    dead_pid = 999999  # above macOS PID_MAX (99998) — guaranteed dead
    lock.write_text(f"{dead_pid}\nWed Jan  1 00:00:00 2020\n")
    assert acquire(lock) is None, "a dead holder must be reclaimed"
    assert read_lock(lock).pid == os.getpid()


def test_pid_reuse_is_not_mistaken_for_liveness(tmp_path):
    lock = tmp_path / ".run.lock"
    # A LIVE pid (ours) but a fingerprint from a different boot era:
    # that's PID reuse, and the lock is stale.
    lock.write_text(f"{os.getpid()}\nWed Jan  1 00:00:00 2020\n")
    assert not lock_pid_is_live(os.getpid(), "Wed Jan  1 00:00:00 2020")
    assert acquire(lock) is None


def test_unparseable_lock_is_stale(tmp_path):
    lock = tmp_path / ".run.lock"
    lock.write_text("not-a-pid\n")
    assert read_lock(lock) is None
    assert acquire(lock) is None


def test_missing_fingerprint_stays_conservative():
    # No recorded start-time + a live PID = live (never "confirmed
    # different process" on missing data).
    assert lock_pid_is_live(os.getpid(), "")
    my_start = pid_start_time(os.getpid())
    assert my_start  # ps must resolve our own PID on macOS
    assert lock_pid_is_live(os.getpid(), my_start)


def test_cli_runs_command_and_releases(tmp_path):
    lock = tmp_path / ".watch.lock"
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.run_lock", str(lock),
         "--", sys.executable, "-c", "import sys; sys.exit(7)"],
        cwd=REPO, capture_output=True,
    )
    assert r.returncode == 7, "wrapper must pass through the exit code"
    assert not lock.exists(), "wrapper must release on exit"


def test_cli_refuses_second_instance_with_exit_75(tmp_path):
    lock = tmp_path / ".watch.lock"
    # First instance: hold the lock while a second tries to start.
    first = subprocess.Popen(
        [sys.executable, "-m", "src.utils.run_lock", str(lock),
         "--", sys.executable, "-c",
         "import time,sys; print('up',flush=True); time.sleep(30)"],
        cwd=REPO, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert first.stdout.readline().strip() == "up"
        second = subprocess.run(
            [sys.executable, "-m", "src.utils.run_lock", str(lock),
             "--", sys.executable, "-c", "print('should not run')"],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        assert second.returncode == EXIT_HELD
        assert "should not run" not in second.stdout
    finally:
        first.kill()
        first.wait()


def test_cli_usage_error_without_separator():
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.run_lock", "some.lock", "cmd"],
        cwd=REPO, capture_output=True,
    )
    assert r.returncode == 2


# ---------------------------------------------------------------------------
# Named-resource locks (acquire_resource/release_resource): a path lock
# stops two instances of the SAME script colliding on the SAME --out; it
# says nothing about a DIFFERENT script stepping the same physical
# resource (the emulator pool) from an unrelated --out. `_RESOURCE_LOCK_DIR`
# is monkeypatched to tmp_path in every test below so these never touch
# the real repo's runs/.locks/ directory.
# ---------------------------------------------------------------------------

def test_acquire_resource_fresh_then_release(tmp_path, monkeypatch):
    monkeypatch.setattr(run_lock, "_RESOURCE_LOCK_DIR", tmp_path / "locks")
    assert acquire_resource("emulator-pool", extra="pytest") is None
    lock = tmp_path / "locks" / "emulator-pool.lock"
    assert lock.exists()
    holder = read_lock(lock)
    assert holder.pid == os.getpid()
    assert "pytest" in holder.raw
    release_resource("emulator-pool")
    assert not lock.exists()


def test_second_acquire_resource_refuses_while_holder_lives(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(run_lock, "_RESOURCE_LOCK_DIR", tmp_path / "locks")
    assert acquire_resource("emulator-pool") is None
    # Same process is trivially alive: a second acquire on the SAME
    # resource name — regardless of caller, --out, or script — must
    # refuse and name the live holder.
    holder = acquire_resource("emulator-pool")
    assert holder is not None and holder.pid == os.getpid()
    release_resource("emulator-pool")
    assert acquire_resource("emulator-pool") is None, (
        "acquire must succeed again once the first holder released"
    )
    release_resource("emulator-pool")


def test_resource_lock_is_keyed_by_name_not_a_shared_singleton(tmp_path,
                                                                monkeypatch):
    """Two DIFFERENT resource names must not contend with each other —
    only same-name callers do."""
    monkeypatch.setattr(run_lock, "_RESOURCE_LOCK_DIR", tmp_path / "locks")
    assert acquire_resource("emulator-pool") is None
    assert acquire_resource("some-other-resource") is None
    release_resource("emulator-pool")
    release_resource("some-other-resource")
