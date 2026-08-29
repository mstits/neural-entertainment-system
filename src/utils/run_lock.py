"""Fingerprinted run-locks for processes that write shared paths.

One process per lockfile: a trainer per checkpoint dir, a chain watcher
per receipt tree, an eval pipeline per ladder dir. The lock is a small
text file:

    <pid> <optional free text>
    <ps -o lstart= fingerprint of that pid>
    <extra context lines, e.g. the ROM path>

The PID alone is not identity: after an unclean shutdown (OOM-kill,
power loss) macOS can reissue a dead process's PID to an unrelated
process within seconds of boot, which would make a stale lock look live
forever and permanently block every later unattended restart. The
start-time fingerprint (line 2) catches PID reuse; when either
fingerprint can't be determined the check stays conservative and
reports live.

History: this file extracts `scripts/train_game.py`'s `.run.lock`
implementation so other writers can share it. Two dated incidents made
the case — two chain watchers running the same 192-eval ladder against
shared receipt paths (2026-08-29, harmless only because eval rewrites
are deterministic), and a `git update-ref` race on `refs/heads/main`
between concurrent sessions. Both are the same defect: concurrent
writers on shared state with no mutual exclusion. Shell users get the
same protection via the CLI:

    python -m src.utils.run_lock runs/foo/.watch.lock -- ./chain.sh ...

which acquires (or exits 75 if a live holder exists), runs the command,
and releases on exit.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# EX_TEMPFAIL from sysexits.h: "try again later" — the conventional exit
# for "another instance is already running".
EXIT_HELD = 75


def pid_start_time(pid: int) -> Optional[str]:
    """Best-effort start-time fingerprint for `pid` (via `ps -o lstart=`).

    Returns None if `ps` is unavailable or the PID can't be resolved —
    callers must treat that as "unknown", never as "confirmed different
    process".
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def lock_pid_is_live(pid: int, recorded_start: str) -> bool:
    """True iff `pid` is still the process that recorded `recorded_start`.

    A bare `os.kill(pid, 0)` only proves *some* process currently holds
    that PID (success), or that one exists but isn't ours
    (PermissionError) — not that it's the same process that wrote the
    lock. Comparing start-time fingerprints catches PID reuse; when
    either fingerprint can't be determined this stays conservative and
    reports live.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # some process holds this PID, not us — check the fingerprint
    if not recorded_start:
        return True
    current_start = pid_start_time(pid)
    return current_start is None or current_start == recorded_start


@dataclass
class LockHolder:
    pid: int
    start: str
    raw: str  # full lock text, for error messages


def read_lock(path: Path) -> Optional[LockHolder]:
    """Parse an existing lockfile. None = missing or unparseable (an
    unparseable lock is stale by definition — nothing can release it)."""
    try:
        raw = Path(path).read_text()
        lines = raw.splitlines()
        pid = int(lines[0].split()[0])
    except (OSError, ValueError, IndexError):
        return None
    start = lines[1] if len(lines) > 1 else ""
    return LockHolder(pid=pid, start=start, raw=raw)


def acquire(path: Path, extra: str = "") -> Optional[LockHolder]:
    """Try to take the lock. Returns None on success, or the live
    holder that refused us.

    Creation uses O_CREAT|O_EXCL so two racing acquirers can't both
    conclude "no lock" from a stat and then overwrite each other (the
    read-check-write pattern this replaces had exactly that window). A
    stale lock (dead PID, PID-reuse mismatch, or unparseable content)
    is unlinked and the create retried once.

    Raises OSError only for filesystem failures on the create itself —
    callers decide whether that's fatal (a watcher should refuse; a
    trainer historically warns and continues).
    """
    path = Path(path)
    body = f"{os.getpid()}\n{pid_start_time(os.getpid()) or ''}\n{extra}"
    if extra and not body.endswith("\n"):
        body += "\n"
    for attempt in (1, 2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                         0o644)
            with os.fdopen(fd, "w") as f:
                f.write(body)
            return None
        except FileExistsError:
            holder = read_lock(path)
            if holder is not None and lock_pid_is_live(holder.pid,
                                                       holder.start):
                return holder
            # Stale (dead, reused-PID, or unparseable): reclaim.
            if attempt == 2:
                # Two stale-reclaim rounds means something else keeps
                # recreating it faster than we can claim it; treat the
                # last-read holder as live rather than loop.
                return holder or LockHolder(pid=-1, start="",
                                            raw="<unreadable>")
            try:
                path.unlink()
            except OSError:
                pass
    return None  # unreachable; loop always returns


def release(path: Path) -> None:
    """Remove the lockfile if it exists. Never raises."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[1] != "--":
        print("usage: python -m src.utils.run_lock <lockfile> -- "
              "<command> [args...]", file=sys.stderr)
        return 2
    lock_path = Path(argv[0])
    cmd = argv[2:]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = acquire(lock_path, extra=" ".join(cmd))
    if holder is not None:
        print(f"[run_lock] {lock_path} held by live PID {holder.pid} — "
              f"refusing to run a second instance.", file=sys.stderr)
        return EXIT_HELD
    try:
        return subprocess.call(cmd)
    finally:
        release(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
