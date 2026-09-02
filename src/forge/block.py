"""The block runner (FORGE_SPEC_2026-09-01.md Section 2f).

The only way a Forge pilot touches the emulator: a bounded, watchdog-gated
unattended block with hard abort, per the phase-3 ruling (``PROGRESS.md``).
Piece (d) calls ``run_block``; ``run_block`` never calls piece (d).

``wrongful_reset`` is the one check, four conditions, evaluated in this
fixed order between consecutive progress-file reads or lock reads:

  1. ``cells`` went down since the previous read.
  2. ``solutions`` went down since the previous read.
  3. the child's root-state hash differs from the granted ``root_state_sha``.
  4. the child's own run-lock (``<out>/.run.lock``, written by the solver
     itself -- ``scripts/go_explore_solve.py:3646-3671``) is held by a pid
     other than the child's.

Each comparison is strict ``<``, never ``<=``: two consecutive polls that
land on the SAME unwritten row (the runner polls faster than the child's
60-s progress cadence) must read as flat, not as a decrease -- a ``<=``
regression trips on that flat pair and never reaches the real decrease it
exists to catch (test_forge_block.py's own revert-verify for
``test_positive_control_injected_reset_is_caught``).

``run_block(plan)`` launches ``plan["cmd"]`` the way the engine launches
every other action (``scripts/detach.py``, ``start_new_session=True`` so
the child leads its own process group) and polls it at ``poll_interval``
until one of: a wrongful reset trips (``stop:"abort"``), the block's own
``max_secs``/``max_steps`` bound is reached (``stop:"budget"``), or the
child exits on its own (``stop:"complete"``). A trip runs the hard abort
(SIGTERM the child's process GROUP via ``os.killpg`` -- ``pgid == pid`` by
construction of ``start_new_session`` -- then SIGKILL if it is still alive
after ``term_grace_s``), releases the child's own ``.run.lock`` (the
child's atexit unlink never fires under SIGKILL, so the runner does it),
and returns with ``banked: []``: banking only happens after a clean stop
with ``watchdog_trips: 0``, so an aborted block has nothing to bank by
construction. The root-state hash (condition 3) is re-read from disk on
EVERY poll, never cached from before launch -- a wrongful reset that
swaps the root file mid-block, not only a plan whose ``root_state_sha``
was wrong to begin with, must be reachable.

Before launch, if the plan's ``cmd`` names a ``--root-state``, that file
is copied once into ``<child_out>/sandbox/`` and the child's own command
is rewritten to read the copy -- never the path the plan named. For a
real block that path is a shared archive under ``runs/`` (other blocks,
other walls, and the grant itself all read it), so nothing this runner
does past that one read may write to it. If the plan also carries a
``root_state_sha`` (the grant's recorded value), it is checked against
the bytes just read from that shared file before the copy is trusted
for anything -- a plan whose sha has drifted from what is actually on
disk is refused (``root_state_sha_mismatch``) before launch, never
silently accepted. ``root_state_sha`` for the mid-poll comparison below
is then recomputed from the copy's own bytes at the moment it is taken,
not the plan's value carried forward.

``plan["inject_wrongful_reset"]`` is not cosmetic: when set, ``run_block``
itself corrupts the child's on-disk root-state file -- the sandboxed
copy described above, not the file the plan named -- shortly after
confirming the child is alive and has produced at least one real signal
of being up, so the watchdog catches a reset the runner manufactured
against a live child -- regardless of whether that particular child's
own telemetry would ever have decreased on its own. This is what lets
the positive control mean something against an arbitrary well-behaved
child, synthetic or real, not only a fixture that is scripted to
misbehave, and it is what keeps that same injection from ever touching
the file other blocks depend on. ``positive_control.caught`` in the
receipt is true only when a trip occurred AND the block was told to
inject one; an uninjected block that trips (a genuine wrongful reset
against a real run) reports ``injected:false, caught:false`` -- a
positive control that was never run is never credited as caught.
``banked_from_reset`` counts artifacts banked as part of that reset,
which is always 0 on the abort path (``banked`` is fixed to ``[]``
first) -- never a progress-row field a downstream reader could mistake
for "something was banked".

Nothing here touches CLAIMS.md except reading it once, to confirm the
plan's ``grant_entry`` anchor is present (LG rule 9's refusal). Nothing
here writes an EXHIBITION or FORGE ledger entry -- piece (e) does that,
from this block's own receipt.

That receipt is written to ``<child_out>/block_receipt.json`` on every
exit path that got as far as reading the plan's ``--out`` flag, refusals
included, and its path comes back in the returned dict as
``receipt_path``. ``write_block_receipt`` refuses a receipt missing any
field the two FORGE-GRANT entries fix, so a partial receipt never
reaches disk. Alongside the grant's fields it records the SHARED root
state's full sha256 before launch and after the block ends -- the
reading ruling 17 judges a control on -- which is a different value from
the 16-character fingerprint the poll loop compares. ``append_attended``
is the hook that puts rows in the ``attended_log`` the ratio divides by:
without it every live block divides by zero and reports ``ratio_ok:
false`` regardless of who sat with the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Callable, Optional

from src.utils import run_lock

REPO = Path(__file__).resolve().parent.parent.parent

DEFAULT_CLAIMS_PATH = REPO / "CLAIMS.md"
DEFAULT_GRANT_STATE_PATH = REPO / "runs" / "forge" / "grant_state.json"
DEFAULT_ATTENDED_LOG = REPO / "runs" / "forge" / "attended.jsonl"

#: Machine-hours-under-lock per attended-hour the ruling requires
#: (PROGRESS.md's phase-3 ruling: "logged attended hours beside run-lock
#: hours at ≥6 machine-h per attended-h"). Reported via `ratio_ok`, never
#: enforced as a refusal (LG open question 2): the grant is judged on the
#: cycle receipt, not the block.
RATIO_FLOOR = 6.0

POLL_INTERVAL_S = 0.5
TERM_GRACE_S = 10.0
KILL_GRACE_S = 10.0
_KILL_POLL_S = 0.02

_STOP_VALUES = ("budget", "stalled", "abort", "complete")

#: The file every block writes its own receipt to, inside the block's
#: own ``--out`` directory. The grant entries fix the receipt's FIELDS
#: and say the runner writes it; before this, ``run_block`` returned the
#: dict and wrote nothing, so a live block left no receipt on disk at
#: all (the 2026-09-02 cv_hall dry run's first blocker).
BLOCK_RECEIPT_NAME = "block_receipt.json"

#: Every field the two FORGE-GRANT entries fix for a block receipt
#: (``CLAIMS.md``'s "Receipt fields, fixed by the ruling and written by
#: ``src/forge/block.py::run_block``", spec Section 2f). Presence is
#: what is checked, not truthiness: ``abort_reason: null`` on a clean
#: block is a recorded field, an ABSENT ``abort_reason`` is a receipt
#: that cannot answer the question the grant asks of it.
GRANT_RECEIPT_FIELDS = (
    "wall_id", "cycle_id", "grant_entry", "started", "ended", "stop",
    "attended_hours", "run_lock_hours", "ratio_machine_per_attended",
    "ratio_ok", "watchdog_trips", "positive_control",
    "fabricated_clears_unretracted", "banked", "aborted", "abort_reason",
)

#: The three sub-fields the grant fixes inside ``positive_control``.
GRANT_POSITIVE_CONTROL_FIELDS = ("injected", "caught", "banked_from_reset")


class ReceiptFieldError(ValueError):
    """Raised by ``write_block_receipt`` when a receipt is missing a
    field the grant requires. Carries the full list, not just the
    first, and is raised BEFORE any byte is written, so a partial
    receipt never reaches disk to be read later as if it were whole."""

    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__(
            "block receipt missing grant-required field(s): "
            + ", ".join(self.missing))


def write_block_receipt(receipt: dict, out_dir: Path) -> Path:
    """Write ``receipt`` to ``<out_dir>/block_receipt.json`` and return
    the path, after checking it carries every field the grant fixes.

    Refuses (``ReceiptFieldError``) rather than writing a receipt that
    is missing one: a receipt is the only thing a later reader has, so
    one that silently omits ``attended_hours`` or ``watchdog_trips``
    would be indistinguishable from a block that recorded them as
    absent. ``receipt["receipt_path"]`` is set to the path before the
    write, so the file names itself and the returned dict carries the
    same string.
    """
    missing = [f for f in GRANT_RECEIPT_FIELDS if f not in receipt]
    control = receipt.get("positive_control")
    if isinstance(control, dict):
        missing += [f"positive_control.{f}"
                    for f in GRANT_POSITIVE_CONTROL_FIELDS if f not in control]
    elif "positive_control" not in missing:
        missing.append("positive_control (not a mapping)")
    if missing:
        raise ReceiptFieldError(missing)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BLOCK_RECEIPT_NAME
    receipt["receipt_path"] = str(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def append_attended(attended_log: Path, *, who: str, checked: str,
                     start: Optional[str] = None, end: Optional[str] = None,
                     now_fn: Optional[Callable[[], str]] = None) -> Path:
    """Append one attended row to ``attended_log`` and return its path.

    The ratio the grant reports divides run-lock hours by attended
    hours, and ``_attended_hours`` below reads those hours from this
    file -- so with nothing writing it, every live block divides by
    zero and reports ``ratio_ok: false`` no matter how long a person
    actually sat with the run. This is the hook the operator (or a
    driver acting for one) calls at the start and end of each interval
    they are actually at the keyboard: ``who`` names them, ``checked``
    names what they looked at, and ``start``/``end`` are local-clock
    ``%Y-%m-%dT%H:%M:%S`` stamps in the shape ``_attended_hours``
    parses. Omitting ``start`` records a zero-length checkpoint (a
    timestamped note that contributes no hours), which is the honest
    reading of a row whose interval nobody stated.
    """
    now_fn = now_fn or (
        lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    end = end or now_fn()
    start = start or end
    path = Path(attended_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"start": start, "end": end, "who": who, "checked": checked}
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _pid_alive(pid: int) -> bool:
    """True iff `pid` is still running. `run_block` launches the child
    directly (via `scripts/detach.py`'s `subprocess.Popen`), so it is
    OUR waitpid-able child even though `start_new_session=True` moves
    it into its own process GROUP -- an exited-but-unreaped child stays
    a zombie, and a bare `os.kill(pid, 0)` keeps reporting a zombie as
    alive until something reaps it. The `waitpid(WNOHANG)` reap below
    is what turns a real "complete" stop into a fast one instead of a
    wait for `max_secs` to time it out."""
    try:
        reaped_pid, _status = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sha_of_file(path: Path) -> Optional[str]:
    """First 16 hex chars of the file's sha256, or None if it can't be
    read (missing, permission). A short fingerprint, not the full hash:
    ``root_state_sha`` is compared for equality only, never displayed as
    a full digest anywhere a human reads it."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _full_sha_of_file(path: Optional[Path]) -> Optional[str]:
    """The file's whole sha256 as hex, or None if it can't be read. The
    grant records the shared root state's FULL digest, and the ruling 17
    control is judged on that digest reading the same before and after a
    block -- ``_sha_of_file`` above returns the 16-character fingerprint
    the poll loop compares, which is not the value the grant names."""
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _corrupt_root_state(path: Path) -> bool:
    """Best-effort mid-block wrongful-reset injection: appends
    distinguishing bytes to whatever file `path` names so its sha256
    changes under it. Returns True iff the write happened. This is what
    makes ``inject_wrongful_reset`` a real action against whatever child
    is running -- synthetic fixture or the real solver -- rather than a
    description the child's own script has to act out. `run_block` never
    calls this with the path a plan named directly: by the time it is
    reachable, `root_state_path` has already been reassigned to this
    block's own sandboxed copy (row 0b correction), so a write here can
    only ever land on that copy, never on a file other blocks or walls
    depend on. Never raises: a failed injection is reported by the trip
    never firing, which the positive-control test reads directly
    (``caught`` stays False), not by an exception escaping the poll
    loop."""
    try:
        with open(path, "ab") as f:
            f.write(b"\x00FORGE_INJECTED_WRONGFUL_RESET\x00")
        return True
    except OSError:
        return False


def _flag_value(cmd: list, flag: str) -> Optional[str]:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return None


def _set_flag_value(cmd: list, flag: str, new_value: str) -> list:
    """A new cmd list with `flag`'s value replaced by `new_value`. Used
    to point the launched child at this block's own sandboxed copy of
    `--root-state`, never at the path the plan named directly."""
    out = list(cmd)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = str(new_value)
            break
    return out


def _resolve(repo: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (repo / p)


def _tail_row(progress_path: Path) -> Optional[dict]:
    """The last well-formed JSON row in a progress.jsonl-shaped file, or
    None if the file is absent, empty, or every line is unparseable (a
    read racing the child's own single ``f.write(json.dumps(line) +
    "\\n")`` call -- go_explore_solve.py:8645 -- so a torn tail is read
    as "no new row yet", never as a decrease)."""
    try:
        lines = Path(progress_path).read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def wrongful_reset(
    prev_row: Optional[dict],
    cur_row: Optional[dict],
    *,
    root_state_sha: Optional[str] = None,
    current_root_sha: Optional[str] = None,
    lock_holder_pid: Optional[int] = None,
    child_pid: Optional[int] = None,
) -> dict:
    """``{"reason": <str>}`` naming the first tripped condition (checked
    in the module docstring's fixed order), else ``{"reason": None}``.

    Every comparison here is independently gated on both sides being
    present and numeric/comparable -- a missing or malformed field reads
    as "nothing to compare", never as a trip, so a torn read or an
    absent lock file cannot itself manufacture a wrongful-reset verdict.
    """
    if prev_row is not None and cur_row is not None:
        pc, cc = prev_row.get("cells"), cur_row.get("cells")
        if isinstance(pc, (int, float)) and isinstance(cc, (int, float)) \
                and cc < pc:
            return {"reason": "cells_decreased"}
        ps, cs = prev_row.get("solutions"), cur_row.get("solutions")
        if isinstance(ps, (int, float)) and isinstance(cs, (int, float)) \
                and cs < ps:
            return {"reason": "solutions_decreased"}
    if root_state_sha is not None and current_root_sha is not None \
            and current_root_sha != root_state_sha:
        return {"reason": "root_state_mismatch"}
    if lock_holder_pid is not None and child_pid is not None \
            and lock_holder_pid != child_pid:
        return {"reason": "lock_holder_mismatch"}
    return {"reason": None}


def _grant_anchor_present(grant_entry: Optional[str], claims_path: Path) -> bool:
    if not grant_entry:
        return False
    anchor = grant_entry.split("#", 1)[-1] if "#" in grant_entry else grant_entry
    try:
        text = Path(claims_path).read_text()
    except OSError:
        return False
    return anchor in text


def _grant_ended(grant_state_path: Path) -> bool:
    try:
        data = json.loads(Path(grant_state_path).read_text())
    except (OSError, ValueError):
        return False
    return data.get("status") == "GRANT_ENDED"


def _mark_grant_ended(grant_state_path: Path, wall_id: str, cycle_id: str,
                       reason: str) -> None:
    """Writes GRANT_ENDED, per the ruling: "the moment a wrongful reset
    banks an artifact, the runner writes GRANT_ENDED to
    runs/forge/grant_state.json and refuses every later block." Called
    only from the defensive check in `run_block` below -- an aborted
    block never banks by construction, so this should never fire in
    correct operation; it exists as the backstop if that invariant is
    ever broken."""
    grant_state_path.parent.mkdir(parents=True, exist_ok=True)
    grant_state_path.write_text(json.dumps({
        "status": "GRANT_ENDED", "wall_id": wall_id, "cycle_id": cycle_id,
        "reason": reason, "t": _now_iso(),
    }, indent=2) + "\n")


def _attended_hours(attended_log: Path) -> float:
    """Sum of ``end - start`` (hours) over every well-formed
    ``{"start": iso, "end": iso}`` row in ``attended_log``. Missing file
    or an unparseable row contributes 0, never raises -- an attended-log
    read is advisory telemetry for the ratio, not a gate input that can
    refuse a block (see ``run_block``'s ratio handling)."""
    try:
        lines = Path(attended_log).read_text().splitlines()
    except OSError:
        return 0.0
    total = 0.0
    fmt = "%Y-%m-%dT%H:%M:%S"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            start = time.strptime(row["start"], fmt)
            end = time.strptime(row["end"], fmt)
            delta = time.mktime(end) - time.mktime(start)
            if delta > 0:
                total += delta / 3600.0
        except (ValueError, KeyError, TypeError):
            continue
    return round(total, 4)


def _bank_solutions(child_out: Path, repo: Path) -> list:
    """Copies every file under ``<child_out>/solutions/`` into
    ``<child_out>/sandbox/`` and returns the copies' repo-relative
    paths. Called only after a clean stop with `watchdog_trips == 0` --
    never on the abort path, so `banked` is `[]` by construction
    whenever a wrongful reset was caught."""
    src_dir = Path(child_out) / "solutions"
    if not src_dir.is_dir():
        return []
    dest_dir = Path(child_out) / "sandbox"
    dest_dir.mkdir(parents=True, exist_ok=True)
    banked = []
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        dest = dest_dir / f.name
        shutil.copy2(f, dest)
        try:
            rel = dest.relative_to(repo)
        except ValueError:
            rel = dest
        banked.append(str(rel).replace(os.sep, "/"))
    return banked


def _default_launch(cmd: list, log_path: Path, cwd: Path) -> int:
    """Launches `cmd` the way the engine launches every action
    (`scripts/engine_driver.py:1073-1081`): via `scripts/detach.py`, in
    its own session, so its pgid equals its pid and a later
    `os.killpg` reaches the whole group, not one pid."""
    import sys as _sys
    if str(REPO) not in _sys.path:
        _sys.path.insert(0, str(REPO))
    from scripts.detach import launch as _detach_launch
    return _detach_launch([str(c) for c in cmd], log_path, cwd=cwd)


def _hard_abort(pid: int, *, term_grace_s: float = TERM_GRACE_S,
                 kill_grace_s: float = KILL_GRACE_S,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 wall_clock_fn: Callable[[], float] = time.time,
                 poll_s: float = _KILL_POLL_S) -> bool:
    """SIGTERM the child's process GROUP, then SIGKILL if it is still
    alive after `term_grace_s`. Returns True iff the child was
    confirmed dead within `term_grace_s + kill_grace_s`. `os.killpg`,
    never a bare `os.kill` -- the child launches detached in its OWN
    session (`start_new_session=True`), so `pgid == pid` and a bare
    `os.kill` would miss any descendant the child itself spawned."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    deadline = wall_clock_fn() + term_grace_s
    while _pid_alive(pid) and wall_clock_fn() < deadline:
        sleep_fn(poll_s)
    if not _pid_alive(pid):
        return True
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = wall_clock_fn() + kill_grace_s
    while _pid_alive(pid) and wall_clock_fn() < deadline:
        sleep_fn(poll_s)
    return not _pid_alive(pid)


def run_block(
    plan: dict,
    *,
    repo: Path = REPO,
    claims_path: Optional[Path] = None,
    grant_state_path: Optional[Path] = None,
    poll_interval: float = POLL_INTERVAL_S,
    term_grace_s: float = TERM_GRACE_S,
    kill_grace_s: float = KILL_GRACE_S,
    launch_fn: Optional[Callable[[list, Path, Path], int]] = None,
    attended_hours_fn: Optional[Callable[[], float]] = None,
    run_lock_hours_fn: Optional[Callable[[], float]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    wall_clock_fn: Callable[[], float] = time.time,
) -> dict:
    """Runs one bounded, watchdog-gated block per `plan` (the input
    shape FORGE_SPEC_2026-09-01.md Section 2f fixes: `wall_id`,
    `cycle_id`, `cmd`, `root_state_sha`, `max_secs`, `max_steps`,
    `grant_entry`, `attended_log`, `inject_wrongful_reset`) and returns
    the block receipt (the JSON shape the same section fixes),
    having first written that receipt to
    ``<--out>/block_receipt.json`` and recorded its path in the
    dict's own ``receipt_path``.

    `repo`, `claims_path`, `grant_state_path`, `launch_fn`,
    `attended_hours_fn`, `run_lock_hours_fn`, `sleep_fn` and
    `wall_clock_fn` all default to the real repo / real clock / real
    subprocess launch; tests override them to run in milliseconds
    against synthetic fixtures without waiting on real wall-clock hours
    or spawning the real emulator.
    """
    repo = Path(repo)
    claims_path = Path(claims_path) if claims_path is not None else DEFAULT_CLAIMS_PATH
    grant_state_path = (Path(grant_state_path) if grant_state_path is not None
                        else DEFAULT_GRANT_STATE_PATH)

    wall_id = plan["wall_id"]
    cycle_id = plan["cycle_id"]
    cmd = [str(c) for c in plan["cmd"]]
    grant_entry = plan.get("grant_entry")
    root_state_sha = plan.get("root_state_sha")
    max_secs = float(plan.get("max_secs", 1200))
    max_steps = int(plan.get("max_steps", 2_000_000))
    injected = bool(plan.get("inject_wrongful_reset", False))
    attended_log = _resolve(repo, plan.get("attended_log")) or DEFAULT_ATTENDED_LOG

    started = _now_iso()
    receipt = {
        "wall_id": wall_id, "cycle_id": cycle_id, "grant_entry": grant_entry,
        "started": started, "ended": None, "stop": None,
        "attended_hours": 0.0, "run_lock_hours": 0.0,
        "ratio_machine_per_attended": None, "ratio_ok": False,
        "watchdog_trips": 0,
        "positive_control": {"injected": injected, "caught": False,
                             "banked_from_reset": 0},
        "fabricated_clears_unretracted": 0, "banked": [],
        "aborted": False, "abort_reason": None,
        # Beyond the grant's fixed field list: the SHARED root state's
        # full digest, read from the file the plan named (never the
        # sandbox copy), before launch and again after the block ends.
        # Ruling 17 judges a control on those two reading the same and
        # both equalling the grant's; a receipt that omits them cannot
        # settle that question later.
        "root_state_path": None,
        "root_state_sha256_before": None,
        "root_state_sha256_after": None,
        "receipt_path": None,
    }

    #: Set as soon as the block's own --out directory is known; the
    #: receipt is written there on EVERY exit path from here on,
    #: refusals included. A refusal that precedes the --out flag being
    #: read (no grant anchor, ended grant, no --out at all) has no
    #: directory of its own to write to and is returned unwritten.
    out_state: dict = {"dir": None}

    def _finish(rcpt: dict) -> dict:
        if out_state["dir"] is not None:
            write_block_receipt(rcpt, out_state["dir"])
        return rcpt

    def _refuse(reason: str) -> dict:
        receipt["ended"] = _now_iso()
        receipt["stop"] = "abort"
        receipt["aborted"] = True
        receipt["abort_reason"] = reason
        return _finish(receipt)

    # ---- grant checks, before anything launches (LG rule 9) ----------
    if not _grant_anchor_present(grant_entry, claims_path):
        return _refuse("no_grant_anchor")
    if _grant_ended(grant_state_path):
        return _refuse("grant_ended")

    out_value = _flag_value(cmd, "--out")
    if out_value is None:
        return _refuse("cmd_missing_out_flag")
    child_out = _resolve(repo, out_value)
    child_out.mkdir(parents=True, exist_ok=True)
    out_state["dir"] = child_out

    root_state_path = _resolve(repo, _flag_value(cmd, "--root-state"))
    # Kept pointing at the file the PLAN named, across the sandbox
    # rewrite below that moves `root_state_path` to this block's own
    # copy: the before/after digests the receipt records are about the
    # shared file, which nothing this runner does may change.
    shared_root_path = root_state_path
    receipt["root_state_path"] = (str(shared_root_path)
                                  if shared_root_path is not None else None)

    # Row 0b correction: the child is never pointed at the file the plan
    # names directly. For a real block that file is a shared archive
    # under runs/ (e.g. the cv_hall wall's entrance state) -- other
    # blocks, other walls, and the grant itself all read it. Copy it
    # once, before launch, into this block's own sandbox; rewrite the
    # child's own command to read the copy; every later reference to
    # `root_state_path` below -- the mid-poll re-read and the injection
    # in `_corrupt_root_state` -- then acts on that copy alone. The path
    # the plan named is opened here only for reading, exactly once. The
    # plan's own `root_state_sha` (the grant's recorded value) is still
    # enforced here, against the bytes just read from the file the plan
    # named, before the copy is trusted for anything: a shared root that
    # has drifted from the grant's sha must refuse before launch, not
    # go unnoticed because the runner started comparing against its own
    # copy instead.
    if root_state_path is not None:
        try:
            _original_root_bytes = Path(root_state_path).read_bytes()
        except OSError as exc:
            return _refuse(f"root_state_unreadable:{exc}")
        receipt["root_state_sha256_before"] = hashlib.sha256(
            _original_root_bytes).hexdigest()
        _computed_root_sha = receipt["root_state_sha256_before"][:16]
        if root_state_sha is not None and root_state_sha != _computed_root_sha:
            return _refuse("root_state_sha_mismatch")
        sandbox_dir = child_out / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        root_state_copy = sandbox_dir / f"root_state{Path(root_state_path).suffix}"
        root_state_copy.write_bytes(_original_root_bytes)
        cmd = _set_flag_value(cmd, "--root-state", str(root_state_copy))
        root_state_path = root_state_copy
        # Already checked equal to the plan's value above (or the plan
        # supplied none, in which case there was nothing to enforce).
        # Reassigned here only so the mid-poll re-read below always
        # compares the copy against its own known-good starting sha.
        root_state_sha = _computed_root_sha

    launch_fn = launch_fn or _default_launch
    t_start = wall_clock_fn()
    try:
        child_pid = launch_fn(cmd, child_out / "block.log", repo)
    except OSError as exc:
        return _refuse(f"launch_failed:{exc}")

    progress_path = child_out / "progress.jsonl"
    lock_path = child_out / ".run.lock"

    prev_row: Optional[dict] = None
    stop: Optional[str] = None
    tripped = {"reason": None}
    injected_done = False

    while True:
        cur_row = _tail_row(progress_path)
        holder = run_lock.read_lock(lock_path)
        holder_pid = holder.pid if holder is not None else None
        # Re-read every poll, never cached from before launch: a
        # wrongful reset that swaps the root file mid-block (including
        # the one this loop injects below) must be reachable, not only
        # a plan whose sha was wrong from the start.
        current_root_sha = _sha_of_file(root_state_path) if root_state_path else None

        check = wrongful_reset(
            prev_row, cur_row,
            root_state_sha=root_state_sha, current_root_sha=current_root_sha,
            lock_holder_pid=holder_pid, child_pid=child_pid,
        )
        if check["reason"] is not None:
            tripped = check
            receipt["watchdog_trips"] = 1
            stop = "abort"
            break

        if cur_row is not None:
            prev_row = cur_row
            steps = cur_row.get("steps")
            if isinstance(steps, (int, float)) and steps >= max_steps:
                stop = "budget"
                break

        if (wall_clock_fn() - t_start) >= max_secs:
            stop = "budget"
            break

        if not _pid_alive(child_pid):
            stop = "complete"
            break

        # Positive-control injection: only after a real progress row
        # has been observed (proof the child is genuinely up and
        # progress.jsonl has actually been parsed, not an instant
        # pre-launch trip), and only once. Requires the plan to name a
        # root state (`root_state_path`) and a `root_state_sha` to
        # compare against -- the only condition of the four this
        # runner can manufacture against an arbitrary child without
        # needing that child's cooperation.
        if (injected and not injected_done and root_state_path is not None
                and root_state_sha is not None and prev_row is not None):
            injected_done = _corrupt_root_state(root_state_path)

        sleep_fn(poll_interval)

    if stop == "abort":
        _hard_abort(child_pid, term_grace_s=term_grace_s,
                   kill_grace_s=kill_grace_s, sleep_fn=sleep_fn,
                   wall_clock_fn=wall_clock_fn)
        run_lock.release(lock_path)
        receipt["aborted"] = True
        receipt["banked"] = []
        # `caught` credits the positive control only when the block was
        # actually told to inject one -- a genuine wrongful reset on an
        # uninjected block is a real trip, not a demonstrated catch.
        receipt["positive_control"]["caught"] = injected
        receipt["positive_control"]["banked_from_reset"] = len(receipt["banked"])
        receipt["abort_reason"] = tripped["reason"]
        # Defensive backstop (module docstring on _mark_grant_ended):
        # an aborted block banks nothing above, so this branch should
        # be unreachable in correct operation.
        if receipt["watchdog_trips"] > 0 and receipt["banked"]:
            _mark_grant_ended(grant_state_path, wall_id, cycle_id,
                             reason="banked_artifact_from_wrongful_reset")
    else:
        if _pid_alive(child_pid):
            _hard_abort(child_pid, term_grace_s=term_grace_s,
                       kill_grace_s=kill_grace_s, sleep_fn=sleep_fn,
                       wall_clock_fn=wall_clock_fn)
        receipt["banked"] = _bank_solutions(child_out, repo)

    receipt["ended"] = _now_iso()
    receipt["stop"] = stop

    rl_hours = (run_lock_hours_fn() if run_lock_hours_fn is not None
               else round((wall_clock_fn() - t_start) / 3600.0, 4))
    at_hours = (attended_hours_fn() if attended_hours_fn is not None
               else _attended_hours(attended_log))
    receipt["run_lock_hours"] = rl_hours
    receipt["attended_hours"] = at_hours
    if at_hours > 0:
        ratio = rl_hours / at_hours
        receipt["ratio_machine_per_attended"] = round(ratio, 4)
        receipt["ratio_ok"] = ratio >= RATIO_FLOOR
    else:
        receipt["ratio_machine_per_attended"] = None
        receipt["ratio_ok"] = False

    # Re-read from the shared file, not from the sandbox copy the poll
    # loop watched: this is the reading that shows the file other blocks
    # and the grant itself depend on came through untouched.
    receipt["root_state_sha256_after"] = _full_sha_of_file(shared_root_path)

    return _finish(receipt)
