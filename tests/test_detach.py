"""scripts/detach.py — the child must lead its OWN process group.

The property under test is not "the process started" but "our group's
signals cannot reach it", which is what nohup failed to provide when the
2-1 campaign was killed at 12.07M steps. So the central test signals our
whole group and asserts the child lives through it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.detach import launch, main, split_argv  # noqa: E402


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass


def test_split_argv_separates_our_flags_from_the_command():
    ours, cmd = split_argv(["--log", "a.log", "--", "python", "-c", "--log"])
    assert ours == ["--log", "a.log"]
    assert cmd == ["python", "-c", "--log"]


def test_split_argv_requires_the_separator():
    with pytest.raises(SystemExit):
        split_argv(["--log", "a.log", "python"])


def test_launch_rejects_empty_command(tmp_path):
    with pytest.raises(ValueError):
        launch([], tmp_path / "x.log")


def test_child_leads_its_own_process_group(tmp_path):
    """pgid == pid is the structural proof of detachment."""
    pid = launch([sys.executable, "-c", "import time; time.sleep(30)"],
                 tmp_path / "x.log")
    try:
        out = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)],
                             capture_output=True, text=True)
        pgid = int(out.stdout.strip())
        assert pgid == pid, f"pgid {pgid} != pid {pid} — not detached"
        assert pgid != os.getpgrp(), "child shares our group"
    finally:
        _reap(pid)


def test_child_survives_a_signal_that_kills_its_launcher_group(tmp_path):
    """The actual regression: the 2-1 killer was a group-wide signal.

    Signalling THIS process's group would take down the test runner and
    its shell, so the blast radius is isolated one level down instead:

      helper  -> its own session/group H (start_new_session)
      child   -> launched by helper via detach.launch, group G != H

    SIGINT then goes to group H. The helper dies; if detachment works the
    child outlives it. A child that merely inherited nohup semantics
    would share H and die with it, which is precisely what happened to
    the campaign trainer.
    """
    child_pidfile = tmp_path / "child.pid"
    helper_src = f"""
import sys, time
sys.path.insert(0, {str(ROOT)!r})
from scripts.detach import launch
pid = launch([sys.executable, "-c", "import time; time.sleep(30)"],
             {str(tmp_path / "child.log")!r})
open({str(child_pidfile)!r}, "w").write(str(pid))
time.sleep(30)
"""
    helper = subprocess.Popen([sys.executable, "-c", helper_src],
                              start_new_session=True)
    try:
        for _ in range(100):
            if child_pidfile.exists() and child_pidfile.read_text().strip():
                break
            time.sleep(0.1)
        child_pid = int(child_pidfile.read_text().strip())
        assert _alive(child_pid)

        os.killpg(os.getpgid(helper.pid), signal.SIGINT)
        helper.wait(timeout=10)
        time.sleep(0.5)

        assert not _alive(helper.pid), "helper survived; test is not valid"
        assert _alive(child_pid), (
            "child died with its launcher's process group — not detached")
    finally:
        _reap(helper.pid)
        if child_pidfile.exists() and child_pidfile.read_text().strip():
            _reap(int(child_pidfile.read_text().strip()))


def test_output_is_captured_and_pid_file_written(tmp_path, capsys):
    log = tmp_path / "run.log"
    rc = main(["--log", str(log), "--",
               sys.executable, "-c", "print('hello from child')"])
    assert rc == 0
    pid = int((tmp_path / "run.log.pid").read_text().strip())
    for _ in range(50):
        if log.exists() and "hello from child" in log.read_text():
            break
        time.sleep(0.1)
    assert "hello from child" in log.read_text()
    _reap(pid)


def test_log_is_appended_not_truncated(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("earlier run\n")
    pid = launch([sys.executable, "-c", "print('later run')"], log)
    for _ in range(50):
        if "later run" in log.read_text():
            break
        time.sleep(0.1)
    assert "earlier run" in log.read_text(), "clobbered prior output"
    _reap(pid)


def test_launch_records_the_failure_in_the_log_before_reraising(tmp_path):
    """A launch that never starts still leaves a legible trace in <log>.

    Popen failing (missing interpreter mid-rebuild, bad cwd, ...) used to
    raise with the log left empty -- the one place a human checks after
    an unattended caller dies has no clue which command failed or why.
    """
    log = tmp_path / "run.log"
    missing = tmp_path / "no-such-interpreter"
    with pytest.raises(OSError):
        launch([str(missing), "-c", "pass"], log)
    assert log.read_text().strip(), "log left empty on a failed launch"


def test_stdin_is_devnull_so_a_read_cannot_hang(tmp_path):
    """A detached process blocking on stdin has no terminal to free it."""
    log = tmp_path / "run.log"
    pid = launch([sys.executable, "-c",
                  "import sys; print(repr(sys.stdin.read()))"], log)
    for _ in range(50):
        if "''" in log.read_text():
            break
        time.sleep(0.1)
    assert "''" in log.read_text(), "stdin was not /dev/null"
    _reap(pid)
