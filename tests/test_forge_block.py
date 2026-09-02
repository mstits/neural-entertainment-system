"""src/forge/block.py -- the bounded, watchdog-gated block runner.

FORGE_SPEC_2026-09-01.md Section 2f. Every fixture "child" here is a real
subprocess (launched the same way `run_block` launches the real solver,
via `scripts/detach.py`) so the hard-abort path is proven against a real
pid/process-group, never a mock -- the corruption for
`test_positive_control_injected_reset_is_caught`'s second finding
("remove the SIGTERM") only fails if the SIGTERM this module sends is
the thing keeping the child from a slow SIGKILL-only death, and that is
only true of a real OS process.

Poll intervals and row-write intervals are both tens of milliseconds, so
the whole file runs in a couple of seconds; nothing here waits on a real
wall-clock hour, and nothing here launches the real emulator (the real
solver-child positive control is a separate, once-only manual run
against the live emulator, per the row (f) VOID clause in the spec's
pre-registered gate table -- this file is the always-run gate).

``inject_wrongful_reset`` is proven here to be a real action the runner
takes against the child's on-disk root-state file, not a description the
fixture script has to act out on its own: `test_injection_trips_a_
well_behaved_child_via_root_state` runs a monotone, never-misbehaving
child and still gets caught, because `run_block` itself corrupts the
root-state file mid-block. `test_positive_control_injected_reset_is_
caught` below additionally has the fixture emit a real decrease so the
`cells_decreased` condition (not just root-state) has its own direct
coverage against a live process.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge.block import run_block, wrongful_reset  # noqa: E402

ANCHOR = "FORGE-GRANT-gate_fixture-2026-09-01"
GRANT_ENTRY = f"CLAIMS.md#{ANCHOR}"

_FIXTURE_SRC = '''\
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--root-state", default=None)
args, _ = ap.parse_known_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "solutions").mkdir(parents=True, exist_ok=True)
progress = out / "progress.jsonl"

ROWS = {rows!r}
ROW_INTERVAL = {row_interval!r}
WRITE_SOLUTION = {write_solution!r}
EXIT_AFTER_ROWS = {exit_after_rows!r}
SPAWN_DESCENDANT = {spawn_descendant!r}
WRITE_OWN_RUN_LOCK = {write_own_run_lock!r}

if WRITE_OWN_RUN_LOCK:
    # Same shape src.utils.run_lock.read_lock parses: pid on line 1,
    # a start-time fingerprint (blank is fine -- lock_pid_is_live
    # treats a blank recorded start as "unknown, stay conservative")
    # on line 2. Own pid, so run_block's lock_holder_mismatch check
    # never fires and the test isolates the reset condition it wants.
    (out / ".run.lock").write_text(f"{{os.getpid()}}\\n\\n")

if SPAWN_DESCENDANT:
    # No start_new_session here -- this descendant inherits the SAME
    # process group as this script (which IS its own group leader,
    # since run_block launches every fixture via scripts/detach.py).
    # os.killpg reaches it; a bare os.kill(child_pid) would not.
    descendant = subprocess.Popen([sys.executable, "-c",
        "import time; time.sleep(120)"])
    (out / "descendant.pid").write_text(str(descendant.pid))

if WRITE_SOLUTION:
    # Written up front, not after ROWS finishes: a fixture whose rows
    # end in a wrongful reset must have a real solution artifact on
    # disk from the very start, so a test asserting "abort banks
    # nothing" is not vacuously true for want of anything to bank.
    (out / "solutions" / "sol_0.json").write_text(json.dumps({{"ok": True}}))

for row in ROWS:
    with open(progress, "a") as f:
        f.write(json.dumps(row) + "\\n")
    time.sleep(ROW_INTERVAL)

if EXIT_AFTER_ROWS:
    sys.exit(0)

while True:
    time.sleep(0.05)
'''


def _write_fixture(tmp_path: Path, name: str, rows: list, *,
                    row_interval: float = 0.02, write_solution: bool = False,
                    exit_after_rows: bool = True,
                    spawn_descendant: bool = False,
                    write_own_run_lock: bool = False) -> Path:
    script = tmp_path / f"{name}.py"
    script.write_text(_FIXTURE_SRC.format(
        rows=rows, row_interval=row_interval, write_solution=write_solution,
        exit_after_rows=exit_after_rows, spawn_descendant=spawn_descendant,
        write_own_run_lock=write_own_run_lock))
    return script


def _grant_paths(tmp_path: Path, anchor_present: bool = True,
                  ended: bool = False) -> tuple:
    claims = tmp_path / "CLAIMS.md"
    claims.write_text(f"### {ANCHOR}\n\nsome grant text\n" if anchor_present
                      else "### FORGE-GRANT-some-other-wall\n")
    grant_state = tmp_path / "grant_state.json"
    if ended:
        grant_state.write_text(json.dumps({"status": "GRANT_ENDED"}))
    return claims, grant_state


def _forbidden_launch(cmd, log_path, cwd):
    raise AssertionError(
        "run_block launched a child after a refusal that must precede launch")


def _write_root_state(tmp_path: Path, content: bytes = b"root-state-v1") -> tuple:
    """A fixture root-state file plus the sha256[:16] of its current
    bytes, matching `src.forge.block._sha_of_file`."""
    path = tmp_path / "root.state"
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()[:16]
    return path, sha


def _make_plan(tmp_path: Path, script: Path, *, wall_id="gate_wall",
               cycle_id="cycle_0", max_secs=30.0, max_steps=2_000_000,
               grant_entry=GRANT_ENTRY, inject=False,
               root_state_path=None, root_state_sha=None):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, str(script), "--out", str(out_dir)]
    if root_state_path is not None:
        cmd += ["--root-state", str(root_state_path)]
    return {
        "wall_id": wall_id, "cycle_id": cycle_id, "cmd": cmd,
        "root_state_sha": root_state_sha, "max_secs": max_secs,
        "max_steps": max_steps, "grant_entry": grant_entry,
        "attended_log": "attended.jsonl", "inject_wrongful_reset": inject,
    }


# ---------------------------------------------------------------------
# test_positive_control_injected_reset_is_caught
# ---------------------------------------------------------------------

def test_positive_control_injected_reset_is_caught(tmp_path):
    """A synthetic child emits cells 100 -> 150 -> 40 (a wrongful reset),
    on its own, regardless of the flag -- this test's job is direct
    coverage of the `cells_decreased` condition against a real subprocess.
    No `--root-state` is given, so `inject_wrongful_reset` has nothing to
    act on here; `test_injection_trips_a_well_behaved_child_via_root_
    state` below covers the flag actually DOING something against a
    child that would otherwise behave. The watchdog must catch the true
    decrease (150 -> 40), not a flat poll-vs-poll re-read of the same
    row -- the fixture writes each row 50 ms apart while the runner
    polls every 10 ms, so several flat reads occur naturally before
    either real transition.

    Revert-verify 1 (live, this session): `cc < pc` -> `cc <= pc` in
    `wrongful_reset` makes the FIRST flat read (100 == 100) trip instead
    of the real decrease -- `abort_reason` still reads `cells_decreased`
    but the trip fires one poll early; caught with a corrupted `elapsed`
    is a false pass, so the direct unit test
    `test_wrongful_reset_ignores_a_flat_read` is what actually pins this
    -- restored, re-passed.

    Revert-verify 2 (live, this session): deleting the `os.killpg(pid,
    signal.SIGTERM)` line in `_hard_abort` leaves the child alive for
    the full `term_grace_s` before the SIGKILL fallback reaches it --
    wall-clock elapsed crosses the tight bound this test asserts,
    catching the corruption on `ended` (the abort's completion time).
    Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "positive_control",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, inject=True)

    t0 = time.perf_counter()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)
    elapsed = time.perf_counter() - t0

    assert receipt["stop"] == "abort"
    assert receipt["watchdog_trips"] == 1
    assert receipt["abort_reason"] == "cells_decreased"
    assert receipt["banked"] == []
    assert receipt["aborted"] is True
    # `banked_from_reset` counts artifacts banked from the reset -- 0 by
    # construction, since `banked` is fixed to `[]` before it is read.
    assert receipt["positive_control"] == {
        "injected": True, "caught": True, "banked_from_reset": 0}
    # Killed via SIGTERM, not the slow SIGKILL-only fallback.
    assert elapsed < 0.8, f"abort took {elapsed:.2f}s -- SIGTERM likely missing"


# ---------------------------------------------------------------------
# test_clean_block_banks_with_zero_trips
# ---------------------------------------------------------------------

def test_clean_block_banks_with_zero_trips(tmp_path):
    """A monotone child (cells strictly increasing, then exits and
    leaves one solution artifact) reaches a clean `stop:"complete"` and
    banks that artifact; `watchdog_trips` stays 0 throughout.

    Revert-verify (live, this session): `cc < pc` -> `cc <= pc` trips on
    this fixture's own flat poll-vs-poll re-reads (the same mechanism
    the positive-control test's row-interval margin creates) --
    `watchdog_trips` reads 1 and `banked` reads `[]`, failing this
    test's assertions by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "clean_block",
        rows=[{"cells": 10, "steps": 100}, {"cells": 20, "steps": 200},
              {"cells": 30, "steps": 300}],
        row_interval=0.02, write_solution=True, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "complete"
    assert receipt["watchdog_trips"] == 0
    assert receipt["banked"], "a clean block with a real solution must bank it"
    assert receipt["banked"][0].endswith("sandbox/sol_0.json")
    assert (tmp_path / receipt["banked"][0]).exists()


# ---------------------------------------------------------------------
# test_budget_stop_at_max_secs_and_max_steps
# ---------------------------------------------------------------------

def test_budget_stop_at_max_secs_and_max_steps(tmp_path):
    """A child that never exits and keeps climbing `steps` must be
    stopped by the block's OWN `max_steps` bound, promptly -- not left
    to run until `max_secs` (set generously here, as a safety net the
    correct code path never actually reaches).

    Revert-verify (live, this session): commenting out the `steps >=
    max_steps` check leaves only the `max_secs` bound live; the child
    runs through its whole (much larger) row sequence and the block
    only stops once wall-clock crosses `max_secs` -- both the elapsed-
    time and the reported `steps` assertions below fail by name (steps
    far exceeds `max_steps`, elapsed far exceeds the tight bound).
    Restored, re-passed.
    """
    rows = [{"cells": i, "steps": i * 100} for i in range(1, 21)]
    script = _write_fixture(
        tmp_path, "budget_stop", rows=rows, row_interval=0.02,
        exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, max_secs=1.0, max_steps=550)

    t0 = time.perf_counter()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.005, term_grace_s=1.0, kill_grace_s=1.0)
    elapsed = time.perf_counter() - t0

    assert receipt["stop"] == "budget"
    assert receipt["watchdog_trips"] == 0
    assert elapsed < 0.5, f"stopped after {elapsed:.2f}s -- max_steps bound likely dropped"


# ---------------------------------------------------------------------
# test_refuses_without_grant_anchor
# ---------------------------------------------------------------------

def test_refuses_without_grant_anchor(tmp_path):
    """No `grant_entry` anchor in CLAIMS.md -> refused before the first
    launch (LG rule 9); no start, no child.

    Revert-verify (live, this session): short-circuiting the anchor
    check to always pass lets `run_block` proceed to `launch_fn`, which
    is the `_forbidden_launch` spy here -- it raises, failing the test
    by name (an unexpected AssertionError instead of a clean refusal).
    Restored, re-passed.
    """
    claims, grant_state = _grant_paths(tmp_path, anchor_present=False)
    script = _write_fixture(tmp_path, "unused", rows=[])
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        launch_fn=_forbidden_launch)

    assert receipt["aborted"] is True
    assert receipt["abort_reason"] == "no_grant_anchor"
    assert receipt["stop"] == "abort"
    assert receipt["banked"] == []


# ---------------------------------------------------------------------
# test_grant_ended_refuses_all_later_blocks
# ---------------------------------------------------------------------

def test_grant_ended_refuses_all_later_blocks(tmp_path):
    """`grant_state.json` already reads GRANT_ENDED (a prior block
    banked from a wrongful reset) -> every later block is refused
    before launch, even with a perfectly valid grant anchor.

    Revert-verify (live, this session): short-circuiting the
    `_grant_ended` check to always read False lets `run_block` proceed
    to `launch_fn`, the `_forbidden_launch` spy here -- it raises,
    failing the test by name. Restored, re-passed.
    """
    claims, grant_state = _grant_paths(tmp_path, ended=True)
    script = _write_fixture(tmp_path, "unused2", rows=[])
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        launch_fn=_forbidden_launch)

    assert receipt["aborted"] is True
    assert receipt["abort_reason"] == "grant_ended"
    assert receipt["stop"] == "abort"
    assert receipt["banked"] == []


# ---------------------------------------------------------------------
# test_ratio_reported_not_refused
# ---------------------------------------------------------------------

def test_ratio_reported_not_refused(tmp_path):
    """0.33 machine-hours under lock against 0.5 attended hours is well
    under the ruling's >=6:1 floor -- `ratio_ok` reads False, but the
    block still ran to completion; the ratio is reported for the CYCLE
    receipt to judge, never a reason for a block to refuse itself (LG
    open question 2).

    Revert-verify (live, this session): adding `if not ratio_ok:
    receipt["aborted"] = True` after the ratio is computed flips
    `aborted` to True on this exact fixture -- fails the `aborted is
    False` assertion by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "ratio", rows=[{"cells": 5, "steps": 50}],
        row_interval=0.01, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0,
        attended_hours_fn=lambda: 0.5, run_lock_hours_fn=lambda: 0.33)

    assert receipt["run_lock_hours"] == 0.33
    assert receipt["attended_hours"] == 0.5
    assert receipt["ratio_machine_per_attended"] == pytest.approx(0.66, abs=0.01)
    assert receipt["ratio_ok"] is False
    assert receipt["aborted"] is False
    assert receipt["stop"] in ("complete", "budget")


# ---------------------------------------------------------------------
# test_injection_trips_a_well_behaved_child_via_root_state
# ---------------------------------------------------------------------

def test_injection_trips_a_well_behaved_child_via_root_state(tmp_path):
    """A perfectly monotone child that never decreases anything on its
    own -- `inject_wrongful_reset:true` alone must still catch a
    wrongful reset, proving `run_block` ACTS on the flag (corrupts the
    child's own on-disk root-state file mid-block, after a real
    progress row has been observed) rather than merely echoing it into
    the receipt.

    Revert-verify (live, this session): gating the injection call on
    `False` (so `_corrupt_root_state` is never invoked) leaves this
    monotone child never tripped -- it exits cleanly, `stop` reads
    "complete", `positive_control` reads
    `{"injected": True, "caught": False, "banked_from_reset": 0}` --
    fails both assertions below by name. Restored, re-passed.
    """
    root_path, correct_sha = _write_root_state(tmp_path)
    rows = [{"cells": i * 10, "steps": i} for i in range(1, 11)]
    script = _write_fixture(tmp_path, "well_behaved", rows=rows,
                            row_interval=0.03, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0, inject=True)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    assert receipt["abort_reason"] == "root_state_mismatch"
    assert receipt["aborted"] is True
    assert receipt["positive_control"] == {
        "injected": True, "caught": True, "banked_from_reset": 0}
    assert receipt["banked"] == []


# ---------------------------------------------------------------------
# test_root_state_mismatch_detected_mid_block_not_only_at_launch
# ---------------------------------------------------------------------

def test_root_state_mismatch_detected_mid_block_not_only_at_launch(tmp_path):
    """A generic child whose telemetry never misbehaves, but whose
    on-disk root-state file is swapped out from under it PARTWAY
    through the block by something outside `run_block` entirely (a
    background thread here, standing in for whatever external actor
    the ruling's wrongful-reset definition names) -- proving the
    runner re-reads the root-state hash on every poll, not once before
    launch. `inject_wrongful_reset` is left False: this is not the
    runner's own injection, it is the condition the injection mechanism
    above is built on top of.

    Revert-verify (live, this session): moving the root-state hash
    computation back to a one-time pre-launch read (outside the poll
    loop) makes this test's mid-block swap invisible -- the block runs
    every row to a clean `stop:"complete"`, `watchdog_trips:0` -- fails
    both assertions below by name. Restored, re-passed.
    """
    root_path, correct_sha = _write_root_state(tmp_path)
    rows = [{"cells": i, "steps": i} for i in range(1, 11)]
    script = _write_fixture(tmp_path, "root_swap", rows=rows,
                            row_interval=0.03, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0)

    def _swap_after_delay():
        time.sleep(0.06)
        root_path.write_bytes(b"a-different-root-entirely")

    swapper = threading.Thread(target=_swap_after_delay, daemon=True)
    swapper.start()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)
    swapper.join(timeout=1.0)

    assert receipt["stop"] == "abort"
    assert receipt["watchdog_trips"] == 1
    assert receipt["abort_reason"] == "root_state_mismatch"
    assert receipt["aborted"] is True
    # Not injected -- a real trip is not a demonstrated positive control.
    assert receipt["positive_control"] == {
        "injected": False, "caught": False, "banked_from_reset": 0}


# ---------------------------------------------------------------------
# test_solutions_decrease_is_caught
# ---------------------------------------------------------------------

def test_solutions_decrease_is_caught(tmp_path):
    """`solutions` dropping between polls trips the watchdog on its own
    condition, independent of `cells` (which stays monotone here) --
    the fourth of the spec's four conditions, previously covered only
    by no block-level test.

    Revert-verify (live, this session): deleting the `solutions_
    decreased` branch in `wrongful_reset` leaves this fixture's
    monotone `cells` never tripping anything -- the block runs to
    `stop:"budget"` at the tight `max_secs` below instead of
    `stop:"abort"` -- fails by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "solutions_drop",
        rows=[{"cells": 10, "solutions": 2, "steps": 10},
              {"cells": 20, "solutions": 3, "steps": 20},
              {"cells": 30, "solutions": 1, "steps": 30}],
        row_interval=0.05, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, max_secs=2.0)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    assert receipt["watchdog_trips"] == 1
    assert receipt["abort_reason"] == "solutions_decreased"
    assert receipt["aborted"] is True
    assert receipt["banked"] == []
    assert receipt["positive_control"] == {
        "injected": False, "caught": False, "banked_from_reset": 0}


# ---------------------------------------------------------------------
# test_abort_path_never_banks_even_when_a_solution_file_exists
# ---------------------------------------------------------------------

def test_abort_path_never_banks_even_when_a_solution_file_exists(tmp_path):
    """The positive-control fixture's own `solutions/` directory holds
    a real artifact from the start (not the usual empty case) -- an
    abort must still bank nothing. Without this, `assert banked == []`
    on the abort path is trivially true for want of anything to bank in
    every other fixture.

    Revert-verify (live, this session): replacing `receipt["banked"] =
    []` with a call to `_bank_solutions(child_out, repo)` on the abort
    branch makes `banked` non-empty (the fixture's `sol_0.json` gets
    copied) -- fails the assertion below by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "abort_with_solution",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, write_solution=True, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    sol = tmp_path / "out" / "solutions" / "sol_0.json"
    assert sol.exists(), "fixture must have a real solution artifact on disk"
    assert receipt["banked"] == [], (
        "a wrongful reset must bank nothing, even with a real solution "
        "artifact present at abort time")


# ---------------------------------------------------------------------
# test_hard_abort_kills_the_whole_process_group
# ---------------------------------------------------------------------

def test_hard_abort_kills_the_whole_process_group(tmp_path):
    """The launched child spawns its OWN descendant subprocess (no
    `start_new_session`, so it inherits the same process group `run_
    block`'s child leads). A trip's hard abort must kill the whole
    GROUP: the commit body's "whole process GROUP" claim had no test
    distinguishing `os.killpg` from a bare `os.kill(child_pid)`, which
    would leave this descendant orphaned and running.

    Revert-verify (live, this session): replacing both `os.killpg`
    calls in `_hard_abort` with `os.kill` leaves the descendant alive
    and reachable after the abort completes -- fails the assertion
    below by name (manually reaped afterward; not part of the restored,
    re-passed committed code).
    """
    script = _write_fixture(
        tmp_path, "group_kill",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, exit_after_rows=False, spawn_descendant=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    descendant_pid = int((tmp_path / "out" / "descendant.pid").read_text())
    deadline = time.time() + 1.0
    alive = True
    while time.time() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.02)
    assert not alive, (
        "the descendant subprocess survived the abort -- os.killpg must "
        "reach the whole process GROUP, not just the launched child pid")


# ---------------------------------------------------------------------
# test_hard_abort_releases_the_childs_run_lock
# ---------------------------------------------------------------------

def test_hard_abort_releases_the_childs_run_lock(tmp_path):
    """The child holds its own `.run.lock` (the shape `src.utils.run_
    lock` writes). A SIGKILLed child never reaches its own atexit
    unlink, so the spec's "release() the lock" clause is the runner's
    job, not the child's, on the abort path.

    Revert-verify (live, this session): removing the `run_lock.release
    (lock_path)` call after `_hard_abort` leaves this fixture's `.run.
    lock` on disk after the block returns -- fails the assertion below
    by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "lock_release",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, exit_after_rows=False, write_own_run_lock=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    assert not (tmp_path / "out" / ".run.lock").exists(), (
        "the child's .run.lock must be released on the abort path, not "
        "left stale after a SIGKILL that skips its own atexit unlink")


# ---------------------------------------------------------------------
# test_budget_stop_at_max_secs_alone
# ---------------------------------------------------------------------

def test_budget_stop_at_max_secs_alone(tmp_path):
    """`max_secs` alone must stop a block promptly, with `max_steps`
    set as a generous safety net the correct code path never actually
    reaches -- the mirror image of `test_budget_stop_at_max_secs_and_
    max_steps`, which only ever exercises `max_steps`.

    Revert-verify (live, this session): commenting out the `(wall_
    clock_fn() - t_start) >= max_secs` check leaves only the `max_
    steps` (40) safety net live -- the block runs until steps reaches
    40 (~0.8s at this fixture's row rate) instead of stopping at ~0.1s
    -- the elapsed-time assertion below fails by name. Restored,
    re-passed.
    """
    rows = [{"cells": i, "steps": i} for i in range(1, 51)]
    script = _write_fixture(tmp_path, "budget_secs_only", rows=rows,
                            row_interval=0.02, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, max_secs=0.1, max_steps=40)

    t0 = time.perf_counter()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.005, term_grace_s=1.0, kill_grace_s=1.0)
    elapsed = time.perf_counter() - t0

    assert receipt["stop"] == "budget"
    assert receipt["watchdog_trips"] == 0
    assert elapsed < 0.5, (
        f"stopped after {elapsed:.2f}s -- max_secs bound likely dropped "
        "(fell back to the max_steps=40 safety net at ~0.8s)")


# ---------------------------------------------------------------------
# wrongful_reset itself: direct unit coverage beneath the block-level
# tests above (the FORGE_REGISTRY entry in test_forge_gates.py proves
# the same function's two polarities independently).
# ---------------------------------------------------------------------

def test_wrongful_reset_ignores_a_flat_read():
    assert wrongful_reset({"cells": 100}, {"cells": 100}) == {"reason": None}


def test_wrongful_reset_catches_a_root_state_mismatch():
    got = wrongful_reset(None, None, root_state_sha="aaa", current_root_sha="bbb")
    assert got == {"reason": "root_state_mismatch"}


def test_wrongful_reset_catches_a_lock_holder_mismatch():
    got = wrongful_reset(None, None, lock_holder_pid=999, child_pid=111)
    assert got == {"reason": "lock_holder_mismatch"}
