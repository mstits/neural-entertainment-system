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

``plan["inject_wrongful_reset"]`` is not cosmetic: when set, ``run_block``
itself corrupts the child's on-disk root-state file (``--root-state`` in
``plan["cmd"]``) shortly after confirming the child is alive and has
produced at least one real signal of being up, so the watchdog catches a
reset the runner manufactured against a live child -- regardless of
whether that particular child's own telemetry would ever have decreased
on its own. This is what lets the positive control mean something against
an arbitrary well-behaved child, synthetic or real, not only a fixture
that is scripted to misbehave. ``positive_control.caught`` in the receipt
is true only when a trip occurred AND the block was told to inject one;
an uninjected block that trips (a genuine wrongful reset against a real
run) reports ``injected:false, caught:false`` -- a positive control that
was never run is never credited as caught. ``banked_from_reset`` counts
artifacts banked as part of that reset, which is always 0 on the abort
path (``banked`` is fixed to ``[]`` first) -- never a progress-row field
a downstream reader could mistake for "something was banked".

Nothing here touches CLAIMS.md except reading it once, to confirm the
plan's ``grant_entry`` anchor is present (LG rule 9's refusal). Nothing
here writes an EXHIBITION or FORGE ledger entry -- piece (e) does that,
from this block's own receipt.
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


def _corrupt_root_state(path: Path) -> bool:
    """Best-effort mid-block wrongful-reset injection: appends
    distinguishing bytes to the child's own root-state file so its
    sha256 changes under it. Returns True iff the write happened. This
    is what makes ``inject_wrongful_reset`` a real action against
    whatever child is running -- synthetic fixture or the real solver
    -- rather than a description the child's own script has to act out.
    Never raises: a failed injection is reported by the trip never
    firing, which the positive-control test reads directly (``caught``
    stays False), not by an exception escaping the poll loop."""
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
    the block receipt (the JSON shape the same section fixes).

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
    }

    def _refuse(reason: str) -> dict:
        receipt["ended"] = _now_iso()
        receipt["stop"] = "abort"
        receipt["aborted"] = True
        receipt["abort_reason"] = reason
        return receipt

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

    root_state_path = _resolve(repo, _flag_value(cmd, "--root-state"))

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

    return receipt
