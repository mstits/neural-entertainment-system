"""Launch a long-running command in its OWN session, so it survives ours.

`nohup cmd &` is not enough, and the difference is not academic: it cost
the 2-1 campaign attempt 2 at 12.07M steps, eleven minutes into phase 2,
with phase 1's deterministic gate already earned (10/10 strict, median
max-x 3299 against a flag at 3298).

nohup only ignores SIGHUP. The child stays in the launching shell's
PROCESS GROUP, so anything that signals that group — a Ctrl-C in the
controlling terminal, an agent session tearing down its tool's children —
reaches it anyway. The campaign log recorded `trainer subprocess exited
-9`, with no jetsam event, no swap in use and 73 GB free, i.e. no
resource-exhaustion signature at all; the kill came from outside.

The fix is a new SESSION, not merely an ignored signal. `os.setsid()` in
the child (via Popen's start_new_session) detaches it from the terminal
and puts it in a fresh process group that our group's signals cannot
reach. macOS ships no setsid(1) binary, so the guidance to "use
launchd/nohup" has no portable one-liner behind it here — hence this
script.

What it guarantees, and what it does not:
  * survives Ctrl-C, session teardown and terminal close;
  * does NOT survive `sudo shutdown`, a panic, or an explicit
    `kill <pid>` — for reboot survival use launchd, which this script
    deliberately does not wrap (a LaunchAgent that relaunches a training
    run after a crash would silently restart a corrupted campaign).

Usage:
    .venv/bin/python scripts/detach.py --log runs/x.log -- \\
        .venv/bin/python scripts/run_online_campaign.py ...

Writes <log> for output and <log>.pid for the pid, prints both, and
exits immediately. Verify with:  ps -o pid,pgid,sess,command -p <pid>
— pgid should equal pid, proving the child leads its own group.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Everything before the first bare `--` is ours; the rest is the command.

    Required because the command being launched has its own flags, and
    argparse would otherwise claim them.
    """
    if "--" not in argv:
        raise SystemExit("detach.py: expected `-- <command>` (no -- found)")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def launch(cmd: list[str], log_path: str | Path,
           cwd: str | Path | None = None) -> int:
    """Start `cmd` in a brand-new session. Returns its pid.

    Output is redirected at the file-descriptor level rather than piped,
    so nothing depends on this process staying alive to drain a pipe.
    stdin is /dev/null: a detached run that blocks on input is a hang
    with no terminal to unblock it.
    """
    if not cmd:
        raise ValueError("empty command")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as log, \
            open(os.devnull, "rb") as devnull:
        try:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, stdin=devnull,
                cwd=str(cwd) if cwd else None,
                start_new_session=True,   # os.setsid() in the child
            )
        except OSError as exc:
            # Popen can fail before the child ever runs (missing
            # interpreter mid-rebuild, bad cwd, ...). Unguarded, that
            # raises with the log left empty -- the one place a human
            # checks after an unattended caller dies has no clue which
            # command failed or why. Record it here, then still raise:
            # this script does not silently swallow a launch failure.
            log.write(f"detach.py: launch failed for {cmd!r}: {exc}\n"
                      .encode())
            raise
    return proc.pid


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ours, cmd = split_argv(raw)
    ap = argparse.ArgumentParser(prog="detach.py")
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", default=None)
    args = ap.parse_args(ours)

    log_path = Path(args.log)
    pid = launch(cmd, log_path,
                 Path(args.cwd) if args.cwd else None)
    pid_path = log_path.with_suffix(log_path.suffix + ".pid")
    pid_path.write_text(f"{pid}\n")
    print(f"detached pid {pid}")
    print(f"  log {log_path}")
    print(f"  pid {pid_path}")
    print(f"  verify: ps -o pid,pgid,sess,command -p {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
