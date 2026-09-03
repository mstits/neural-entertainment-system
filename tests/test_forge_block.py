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
caught` below runs BOTH halves of the gate against live processes: an
inert injection (no `--root-state`, so `run_block` has nothing to
corrupt) against a child that trips on its own must report
`caught:False`, and a live injection against a monotone child must
report `caught:True`. The first half is the anti-vacuity case -- a
gate that cannot say no certifies nothing -- and it is the case the
version of this test committed before FORGE-FIX-2 asserted the other
way round, pinning `caught:True` on a block whose injection its own
docstring said had nothing to act on.

`positive_control_caught` is unit-tested directly in
`test_positive_control_caught_requires_the_injections_own_sha`, one
input at a time: which combinations a block can actually reach depends
on poll timing, so the third condition (the fingerprint at the trip is
the one the injection wrote, not an outside actor's) is pinned there
rather than through a raced fixture.
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

import src.forge.block as block_module  # noqa: E402
from src.forge.block import (  # noqa: E402
    GRANT_POSITIVE_CONTROL_FIELDS,
    GRANT_RECEIPT_FIELDS,
    DEFAULT_CLAIMS_PATH,
    ReceiptFieldError,
    _attended_hours,
    _grant_anchor_present,
    append_attended,
    positive_control_caught,
    run_block,
    write_block_receipt,
    wrongful_reset,
)

ANCHOR = "FORGE-GRANT-gate_wall-2026-09-01"
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
RELEASE_RUN_LOCK_ON_EXIT = {release_run_lock_on_exit!r}

if WRITE_OWN_RUN_LOCK:
    # Same shape src.utils.run_lock.read_lock parses: pid on line 1,
    # a start-time fingerprint (blank is fine -- lock_pid_is_live
    # treats a blank recorded start as "unknown, stay conservative")
    # on line 2. Own pid, so run_block's lock_holder_mismatch check
    # never fires and the test isolates the reset condition it wants.
    (out / ".run.lock").write_text(f"{{os.getpid()}}\\n\\n")
    if RELEASE_RUN_LOCK_ON_EXIT:
        # What the real solver does (`scripts/go_explore_solve.py`
        # registers the unlink with atexit): a child that stops on its
        # own takes its lock with it. Only a SIGKILL skips this.
        import atexit
        atexit.register(lambda: (out / ".run.lock").unlink(missing_ok=True))

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
                    write_own_run_lock: bool = False,
                    release_run_lock_on_exit: bool = False) -> Path:
    script = tmp_path / f"{name}.py"
    script.write_text(_FIXTURE_SRC.format(
        rows=rows, row_interval=row_interval, write_solution=write_solution,
        exit_after_rows=exit_after_rows, spawn_descendant=spawn_descendant,
        write_own_run_lock=write_own_run_lock,
        release_run_lock_on_exit=release_run_lock_on_exit))
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
    """The gate has to be able to say NO. Two live blocks, one
    assertion each way.

    Half 1 (anti-vacuity, the FORGE-FIX-2 case): a synthetic child
    emits cells 100 -> 150 -> 40 on its own, and the plan sets
    `inject_wrongful_reset:True` with no `--root-state` for the
    injection to act on. The watchdog must catch the true decrease
    (150 -> 40), not a flat poll-vs-poll re-read of the same row -- the
    fixture writes each row 50 ms apart while the runner polls every
    10 ms, so several flat reads occur naturally before either real
    transition -- and the receipt must report `caught:False`, because
    the child tripped itself and this runner injected nothing. Before
    FORGE-FIX-2 this same block reported `caught:True`: `caught` was
    assigned from `injected`, the plan's own request flag, inside the
    abort branch (`src/forge/block.py:577` at HEAD `df33765`).

    Half 2: the same flag against a monotone child WITH a root state,
    where the injection does fire, reports `caught:True` -- and the
    receipt carries the inference's inputs, so `caught` is re-derivable
    from the receipt without trusting the field: `injected_done` true,
    `abort_reason` `root_state_mismatch`, and `root_sha_at_trip` equal
    to `injected_sha`.

    Revert-verify 1 (live, this session): restoring the old assignment
    (`receipt["positive_control"]["caught"] = injected` in the abort
    branch, in place of the `positive_control_caught(...)` call) makes
    half 1's inert-injection block report `caught:True` again --
    `assert pc["caught"] is False` fails by name. Half 2 still passes,
    which is exactly why the old code shipped twice. Restored,
    re-passed.

    Revert-verify 2 (live, this session): `cc < pc` -> `cc <= pc` in
    `wrongful_reset` makes the FIRST flat read (100 == 100) trip
    instead of the real decrease -- `abort_reason` still reads
    `cells_decreased` but the trip fires one poll early; caught with a
    corrupted `elapsed` is a false pass, so the direct unit test
    `test_wrongful_reset_ignores_a_flat_read` is what actually pins
    this -- restored, re-passed.

    Revert-verify 3 (live, this session): deleting the `os.killpg(pid,
    signal.SIGTERM)` line in `_hard_abort` leaves the child alive for
    the full `term_grace_s` before the SIGKILL fallback reaches it --
    wall-clock elapsed crosses the tight bound half 1 asserts. Restored,
    re-passed.
    """
    claims, grant_state = _grant_paths(tmp_path)

    # --- half 1: the injection is inert, the child trips itself -------
    script = _write_fixture(
        tmp_path, "positive_control",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, exit_after_rows=False)
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
    pc = receipt["positive_control"]
    assert pc["injected"] is True, "the plan's request is still recorded"
    assert pc["injected_done"] is False, (
        "no --root-state was given, so _corrupt_root_state can never "
        "have run: an injection that did not fire is not a catch")
    assert pc["caught"] is False, (
        "the child tripped itself on cells_decreased; crediting the "
        "positive control here is the FORGE-FIX-2 defect")
    assert pc["injected_sha"] is None
    assert pc["banked_from_reset"] == 0
    # `banked_from_reset` counts artifacts banked from the reset -- 0 by
    # construction, since `banked` is fixed to `[]` before it is read.
    assert receipt["positive_control"] == {
        "injected": True, "injected_done": False, "injected_sha": None,
        "root_sha_at_trip": None, "caught": False, "banked_from_reset": 0}
    # Killed via SIGTERM, not the slow SIGKILL-only fallback.
    assert elapsed < 0.8, f"abort took {elapsed:.2f}s -- SIGTERM likely missing"

    # --- half 2: the injection fires, and IS the cause ----------------
    root_path, correct_sha = _write_root_state(
        tmp_path, content=b"root-state-for-the-live-half")
    live_script = _write_fixture(
        tmp_path, "positive_control_live",
        rows=[{"cells": i * 10, "steps": i} for i in range(1, 11)],
        row_interval=0.03, exit_after_rows=True)
    live_plan = _make_plan(tmp_path, live_script, root_state_path=root_path,
                           root_state_sha=correct_sha, max_secs=5.0,
                           inject=True)
    live_plan["cmd"] = [str(c) for c in live_plan["cmd"]]
    live_plan["cmd"][live_plan["cmd"].index("--out") + 1] = str(
        tmp_path / "out_live")

    live = run_block(
        live_plan, repo=tmp_path, claims_path=claims,
        grant_state_path=grant_state, poll_interval=0.01,
        term_grace_s=1.0, kill_grace_s=1.0)

    assert live["stop"] == "abort"
    assert live["abort_reason"] == "root_state_mismatch"
    live_pc = live["positive_control"]
    assert live_pc["injected"] is True
    assert live_pc["injected_done"] is True
    assert live_pc["caught"] is True
    assert live_pc["injected_sha"] is not None
    assert live_pc["root_sha_at_trip"] == live_pc["injected_sha"], (
        "the fingerprint the watchdog read at the trip must be the one "
        "the injection wrote, not a third value")
    # Re-derive the verdict from the receipt alone.
    assert positive_control_caught(
        injected_done=live_pc["injected_done"],
        trip_reason=live["abort_reason"],
        injected_sha=live_pc["injected_sha"],
        root_sha_at_trip=live_pc["root_sha_at_trip"]) is True
    # The persisted receipt carries the same six fields: a reader
    # holding only the file on disk re-derives the same verdict.
    on_disk = json.loads(Path(live["receipt_path"]).read_text())
    assert on_disk["positive_control"] == live_pc


# ---------------------------------------------------------------------
# test_uninjected_trip_reports_no_positive_control
# ---------------------------------------------------------------------

def test_uninjected_trip_reports_no_positive_control(tmp_path):
    """A block that never asked for an injection and trips for a real
    reason reports `injected:False, caught:False`. The root state here
    is present and correct throughout, so `root_sha_at_trip` is a real
    fingerprint (equal to the block's own recorded `root_state_sha`)
    rather than the `None` the no-root-state cases produce, and the
    trip is `cells_decreased`: the one combination none of the other
    block-level tests reach, and the one a `caught = injected` style
    assignment gets right by accident while getting everything else
    wrong.

    Revert-verify (live, this session): dropping the
    `trip_reason == "root_state_mismatch"` condition from
    `positive_control_caught` does NOT redden this test (nothing was
    injected, so condition 1 already fails) -- it reddens
    `test_positive_control_caught_requires_the_injections_own_sha`
    instead. What reddens this one is replacing the
    `positive_control_caught(...)` call with `caught = True`:
    `assert pc["caught"] is False` fails by name. Restored, re-passed.
    """
    root_path, correct_sha = _write_root_state(
        tmp_path, content=b"root-state-untouched-all-block")
    original_bytes = root_path.read_bytes()
    script = _write_fixture(
        tmp_path, "uninjected_trip",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0, inject=False)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    assert receipt["watchdog_trips"] == 1
    assert receipt["abort_reason"] == "cells_decreased"
    assert receipt["banked"] == []
    pc = receipt["positive_control"]
    assert pc["injected"] is False
    assert pc["injected_done"] is False
    assert pc["caught"] is False, (
        "a genuine wrongful reset on an uninjected block is a real "
        "trip, never a demonstrated positive control")
    assert pc["injected_sha"] is None
    # The root state was never touched, so the watchdog's read at the
    # trip is the block's own starting fingerprint.
    assert pc["root_sha_at_trip"] == correct_sha
    assert pc["banked_from_reset"] == 0
    assert root_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------
# test_positive_control_caught_requires_the_injections_own_sha
# ---------------------------------------------------------------------

def test_positive_control_caught_requires_the_injections_own_sha():
    """`positive_control_caught` directly, one input at a time. Which
    combinations a live block can reach depends on poll timing, so the
    third condition -- the fingerprint the watchdog read at the trip is
    the one the injection wrote -- is pinned here rather than through a
    fixture raced against a background swapper.

    The `outside actor` row is the one that matters: the injection did
    fire, the watchdog did trip on `root_state_mismatch`, and the
    block is still not credited, because the value that tripped it is
    not the value the injection wrote. `injected_done and
    reason == "root_state_mismatch"` alone would credit that block.

    Revert-verify (live, this session): deleting the final
    `return root_sha_at_trip == injected_sha` and returning True in its
    place makes the `outside actor` and `no sha recorded` rows pass
    when they must fail -- this test fails by name on those two rows.
    Separately, deleting the `trip_reason != "root_state_mismatch"`
    guard reddens the `wrong trip reason` rows. Restored, re-passed.
    """
    ours = "aaaaaaaaaaaaaaaa"
    theirs = "bbbbbbbbbbbbbbbb"
    cases = [
        # (label, injected_done, trip_reason, injected_sha,
        #  root_sha_at_trip, expected)
        ("the injection caused it", True, "root_state_mismatch", ours,
         ours, True),
        ("never injected", False, "root_state_mismatch", None, ours, False),
        ("injection asked for but inert", False, "cells_decreased", None,
         None, False),
        ("wrong trip reason: cells", True, "cells_decreased", ours, ours,
         False),
        ("wrong trip reason: solutions", True, "solutions_decreased", ours,
         ours, False),
        ("wrong trip reason: lock", True, "lock_holder_mismatch", ours,
         ours, False),
        ("no trip at all", True, None, ours, ours, False),
        ("outside actor swapped it in the same window", True,
         "root_state_mismatch", ours, theirs, False),
        ("no sha recorded for the injection", True, "root_state_mismatch",
         None, ours, False),
        ("no sha read at the trip", True, "root_state_mismatch", ours,
         None, False),
    ]
    for label, done, reason, inj_sha, trip_sha, expected in cases:
        got = positive_control_caught(
            injected_done=done, trip_reason=reason,
            injected_sha=inj_sha, root_sha_at_trip=trip_sha)
        assert got is expected, (
            f"positive_control_caught({label!r}) read {got}, expected "
            f"{expected}")


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
# test_plan_budget_above_grant_is_clamped_to_the_grant
# ---------------------------------------------------------------------

def test_plan_budget_above_grant_is_clamped_to_the_grant(tmp_path):
    """CLAIMS.md fixes the grant's bound at `max_steps` 2,000,000 (FORGE-
    GRANT-cv_hall-2026-09-01 and FORGE-GRANT-contra_wall-2026-09-01 both
    read `max_secs` 1200 and `max_steps` 2000000 -- "Block bounds"). A
    plan is a REQUEST; it must never be able to ask for more than the
    grant allows and get it. Here the plan asks for `max_steps=
    50_000_000`, 25x the grant -- if that request were honored, the
    child (which only ever reaches 2,500,000) would run to completion
    with no budget stop at all, well past the point the grant caps it.

    Revert-verify (live, this session): with the clamp in place, the
    block stops (`stop:"budget"`) at the row whose `steps` first reaches
    the GRANT's 2,000,000, before the fixture's five rows finish
    writing. Reverting `run_block` to read the plan's `max_steps`
    directly (as it did before FORGE-FIX-12) makes the fixture's finite
    row sequence exhaust and the child exit on its own well under the
    inflated bound -- `stop` reads `"complete"`, not `"budget"`, and this
    assertion fails by name. Restored, re-passed.
    """
    rows = [{"cells": i, "steps": i * 500_000} for i in range(1, 6)]
    script = _write_fixture(
        tmp_path, "budget_above_grant", rows=rows, row_interval=0.02,
        exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    # Far above the grant's 1200s / 2_000_000-step bound in both dims;
    # if either request value survived un-clamped, the child's finite,
    # fast-finishing row sequence would exit cleanly long before either
    # requested bound was ever reached.
    plan = _make_plan(tmp_path, script, max_secs=999.0, max_steps=50_000_000)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.005, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "budget", (
        f"stop={receipt['stop']!r} -- plan's inflated max_steps was honored "
        "instead of clamped to the grant's 2,000,000")
    assert receipt["watchdog_trips"] == 0


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
# test_grant_anchor_gate_refuses_empty_foreign_and_body_only (FORGE-FIX-3)
# ---------------------------------------------------------------------

def _claims_with(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "CLAIMS.md"
    path.write_text(body)
    return path


@pytest.mark.parametrize("case,grant_entry,wall_id,claims_body", [
    # 1. The empty fragment. "#" and "CLAIMS.md#" both split to "", and
    #    "" is a substring of every file, so the old test passed against
    #    a CLAIMS.md holding no grant at all.
    ("bare_hash", "#", "gate_wall", f"### {ANCHOR}\n\nsome grant text\n"),
    ("empty_fragment", "CLAIMS.md#", "gate_wall",
     f"### {ANCHOR}\n\nsome grant text\n"),
    ("empty_fragment_empty_claims", "CLAIMS.md#", "gate_wall", "\n"),
    # 2a. A foreign wall's landed grant. The heading is real and exact;
    #     it is another wall's, and nothing tied the anchor to the plan.
    ("foreign_wall", "CLAIMS.md#FORGE-GRANT-other_wall-2026-09-01", "gate_wall",
     "### FORGE-GRANT-other_wall-2026-09-01\n\nanother wall's grant\n"),
    # 2b. A truncated prefix of this wall's own real heading.
    ("truncated_prefix", "CLAIMS.md#FORGE-GRANT-gate_wall", "gate_wall",
     f"### {ANCHOR}\n\nsome grant text\n"),
    # 3. The grant was withdrawn by deleting its heading, but the body
    #    still quotes its own `grant_entry` the way both landed grants
    #    do (CLAIMS.md:1001-1002, :1064-1065).
    ("body_citation_only", GRANT_ENTRY, "gate_wall",
     'Each block\'s plan carries `grant_entry:\n'
     f'"{GRANT_ENTRY}"`; a plan without it is refused\n'
     "before anything launches.\n"),
    # 3b. Same shape, with the anchor sitting inside a paragraph rather
    #     than in a code span.
    ("prose_mention_only", GRANT_ENTRY, "gate_wall",
     f"The withdrawn grant was {ANCHOR} and it is gone.\n"),
], ids=["bare_hash", "empty_fragment", "empty_fragment_empty_claims",
        "foreign_wall", "truncated_prefix", "body_citation_only",
        "prose_mention_only"])
def test_grant_anchor_gate_refuses_empty_foreign_and_body_only(
        tmp_path, case, grant_entry, wall_id, claims_body):
    """Seven ways a plan reached the emulator through a gate that was one
    ``anchor in text`` substring test (FORGE-FIX-3, and the last two are
    FORGE-FIX-36). Every one is refused before ``launch_fn`` is reached:
    the spy raises if it is called, so a regression fails by name here
    rather than by silently starting a child.

    Revert-verify (live, this session), one corruption per new clause
    and one for the pair. Each failure is the launch spy firing:
    ``AssertionError: run_block launched a child after a refusal that
    must precede launch``.
      * Deleting ``if not wall_id or wall_id not in anchor`` reddens
        ``[foreign_wall]`` alone, 1 failed / 31 passed.
      * Replacing the heading regex with the old ``return anchor in
        text`` reddens ``[truncated_prefix]``, ``[body_citation_only]``
        and ``[prose_mention_only]``, 3 failed / 29 passed.
      * Deleting both, which is the function exactly as it stands at
        HEAD 4986043, reddens all seven, 7 failed / 25 passed. The three
        empty-fragment cases show up only here: an empty anchor carries
        no ``wall_id`` AND matches no heading, so either clause alone
        closes that hole and neither is individually necessary for it.
        There is deliberately no third ``if not anchor`` guard, since it
        would have no case of its own.
    All restored, all 32 re-passed.
    """
    claims = _claims_with(tmp_path, claims_body)
    grant_state = tmp_path / "grant_state.json"
    script = _write_fixture(tmp_path, f"unused_{case}", rows=[])
    plan = _make_plan(tmp_path, script, wall_id=wall_id,
                      grant_entry=grant_entry)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims,
        grant_state_path=grant_state, launch_fn=_forbidden_launch)

    assert receipt["abort_reason"] == "no_grant_anchor"
    assert receipt["aborted"] is True
    assert receipt["stop"] == "abort"
    assert receipt["banked"] == []


# ---------------------------------------------------------------------
# test_landed_grants_still_pass_the_tightened_anchor_gate (FORGE-FIX-3)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("wall_id,anchor", [
    ("cv_hall", "FORGE-GRANT-cv_hall-2026-09-01"),
    ("contra_wall", "FORGE-GRANT-contra_wall-2026-09-01"),
])
def test_landed_grants_still_pass_the_tightened_anchor_gate(wall_id, anchor):
    """The anti-vacuity half. A gate that refuses everything is not a
    gate, so the two grants actually landed in this repo's own CLAIMS.md
    have to keep passing, read from the real file at the real default
    path with the real ``grant_entry`` string each grant names for
    itself (CLAIMS.md:1001, :1064).

    Revert-verify (live, this session): narrowing the heading depth from
    ``#{1,6}`` to ``#{1,3}`` reddens both cases and nothing else, 2
    failed / 30 passed, since both landed grants are ``####`` while this
    file's own fixture grant is ``###``. Restored, both re-passed.
    """
    assert _grant_anchor_present(f"CLAIMS.md#{anchor}",
                                 DEFAULT_CLAIMS_PATH, wall_id) is True


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

    Row 0b correction: the file `_corrupt_root_state` writes to must be
    this block's own sandboxed COPY of `root_path`, never `root_path`
    itself -- for a real block, `root_path` stands in for the shared
    archive under `runs/` that other blocks and other walls also read.
    This test asserts both halves: the copy under `<out>/sandbox/` was
    corrupted (proof the injection ran), and `root_path`'s own bytes are
    exactly what `_write_root_state` wrote, untouched.

    Revert-verify (live, this session): gating the injection call on
    `False` (so `_corrupt_root_state` is never invoked) leaves this
    monotone child never tripped -- it exits cleanly, `stop` reads
    "complete", `positive_control` reads `injected:True,
    injected_done:False, caught:False` -- fails both assertions below
    by name. Restored, re-passed.
    """
    root_path, correct_sha = _write_root_state(tmp_path)
    original_bytes = root_path.read_bytes()
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
    assert receipt["positive_control"]["injected"] is True
    assert receipt["positive_control"]["injected_done"] is True
    assert receipt["positive_control"]["caught"] is True
    assert receipt["positive_control"]["banked_from_reset"] == 0
    assert (receipt["positive_control"]["root_sha_at_trip"]
            == receipt["positive_control"]["injected_sha"])
    assert receipt["banked"] == []
    # The injection landed on the block's own copy, not on root_path.
    assert root_path.read_bytes() == original_bytes, (
        "inject_wrongful_reset must never write to the file the plan "
        "named via --root-state")
    copy_path = tmp_path / "out" / "sandbox" / f"root_state{root_path.suffix}"
    assert copy_path.exists(), "run_block must sandbox a copy of root_path"
    assert copy_path.read_bytes() != original_bytes, (
        "the sandboxed copy should carry the injection's corruption")


# ---------------------------------------------------------------------
# test_injection_never_touches_the_shared_root_state
# ---------------------------------------------------------------------

def test_injection_never_touches_the_shared_root_state(tmp_path):
    """Row 0b correction, isolated: for a real block, `--root-state`
    names a shared archive under `runs/` that other blocks, other
    walls, and the grant itself all read from -- one positive-control
    injection must never be able to corrupt it. `root_path` here stands
    in for that shared file; it is asserted byte-for-byte unchanged
    after a block that both injects a reset AND catches it, which is
    the strongest case for the bug this replaces (the pre-row-0b runner
    appended bytes directly to whatever `--root-state` named).

    Revert-verify (live, this session): reassigning `root_state_path`
    to `root_state_copy` is what routes both the mid-poll re-read and
    `_corrupt_root_state` onto the copy; commenting out that one
    reassignment (`root_state_path = root_state_copy`) in `run_block`
    leaves `root_state_path` bound to the ORIGINAL file from the
    `_resolve(...)` call above it. The child itself still launches
    pointed at the copy (the `cmd` rewrite is a separate, untouched
    statement), so the watchdog now polls and corrupts `root_path`
    directly while the child reads a copy it never sees change --
    `positive_control.caught` still reads True (the corrupted original's
    sha still mismatches the recorded `root_state_sha` and trips the
    watchdog on schedule), but the shared file itself comes back
    corrupted: `root_path.read_bytes() == original_bytes` fails by name,
    ending in `b'...FUL_RESET\\x00' == b'...chain-poweron'`. Restored,
    re-passed.
    """
    root_path, correct_sha = _write_root_state(
        tmp_path, content=b"shared-archive-runs-cv-chain-poweron")
    original_bytes = root_path.read_bytes()
    rows = [{"cells": i * 5, "steps": i} for i in range(1, 21)]
    script = _write_fixture(tmp_path, "shared_root", rows=rows,
                            row_interval=0.02, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0, inject=True)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    assert receipt["positive_control"]["injected"] is True
    assert receipt["positive_control"]["caught"] is True
    assert root_path.read_bytes() == original_bytes, (
        "inject_wrongful_reset corrupted the shared file the plan "
        "named via --root-state instead of a private sandboxed copy")


# ---------------------------------------------------------------------
# test_refuses_before_launch_when_plan_root_state_sha_is_stale
# ---------------------------------------------------------------------

def test_refuses_before_launch_when_plan_root_state_sha_is_stale(tmp_path):
    """The plan's `root_state_sha` is the grant's recorded value
    (CLAIMS.md's FORGE-GRANT text names it, e.g.
    `00a93d0aae5b27d2`); the file `--root-state` actually names has
    drifted since -- the exact hazard ruling 13 names, and the one the
    row 0b sandboxing fix must not silently drop. Before this fix,
    `run_block` recomputed `root_state_sha` from the copy's own bytes
    unconditionally, so a plan carrying a stale sha was never compared
    to anything and the block ran anyway. `launch_fn` here is the
    `_forbidden_launch` spy: any launch at all fails the test by name.

    Revert-verify (live, this session): removing the
    `if root_state_sha is not None and root_state_sha !=
    _computed_root_sha` check (leaving only the unconditional
    `root_state_sha = _computed_root_sha` reassignment) makes this
    test's mismatched plan sha go uncompared -- `run_block` proceeds
    to `launch_fn`, the `_forbidden_launch` spy raises, failing the
    test by name (an unexpected AssertionError instead of a clean
    refusal). Restored, re-passed.
    """
    root_path, actual_sha = _write_root_state(
        tmp_path, content=b"shared-archive-runs-cv-chain-hw2")
    original_bytes = root_path.read_bytes()
    script = _write_fixture(tmp_path, "stale_sha", rows=[])
    claims, grant_state = _grant_paths(tmp_path)
    stale_sha = "0" * 16
    assert stale_sha != actual_sha
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=stale_sha)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        launch_fn=_forbidden_launch)

    assert receipt["aborted"] is True
    assert receipt["abort_reason"] == "root_state_sha_mismatch"
    assert receipt["stop"] == "abort"
    assert receipt["watchdog_trips"] == 0
    assert receipt["banked"] == []
    assert root_path.read_bytes() == original_bytes
    assert not (tmp_path / "out" / "sandbox").exists(), (
        "a refused block must not even copy the shared file into a "
        "sandbox, let alone launch against it")


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

    Row 0b correction: the child is launched pointed at `run_block`'s
    own sandboxed COPY of `root_path` (`<out>/sandbox/root_state<ext>`),
    never at `root_path` itself, so "the file the child's root-state
    hash is read from" is that copy from the moment the block starts.
    The swap below targets the copy for exactly that reason -- swapping
    `root_path` after this correction would swap a file nothing reads
    from anymore. `root_path` is asserted untouched at the end.

    Revert-verify (live, this session): moving the root-state hash
    computation back to a one-time pre-launch read (outside the poll
    loop) makes this test's mid-block swap invisible -- the block runs
    every row to a clean `stop:"complete"`, `watchdog_trips:0` -- fails
    both assertions below by name. Restored, re-passed.
    """
    root_path, correct_sha = _write_root_state(tmp_path)
    original_bytes = root_path.read_bytes()
    rows = [{"cells": i, "steps": i} for i in range(1, 11)]
    script = _write_fixture(tmp_path, "root_swap", rows=rows,
                            row_interval=0.03, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0)
    copy_path = tmp_path / "out" / "sandbox" / f"root_state{root_path.suffix}"

    def _swap_after_delay():
        deadline = time.time() + 1.0
        while not copy_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        copy_path.write_bytes(b"a-different-root-entirely")

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
    # Not injected -- a real trip is not a demonstrated positive
    # control, even when the trip reason is the one an injection would
    # have produced. The external swapper wrote this file, not us.
    assert receipt["positive_control"]["injected"] is False
    assert receipt["positive_control"]["injected_done"] is False
    assert receipt["positive_control"]["caught"] is False
    assert receipt["positive_control"]["injected_sha"] is None
    assert receipt["positive_control"]["banked_from_reset"] == 0
    assert root_path.read_bytes() == original_bytes, (
        "the external swap targeted the sandboxed copy; root_path "
        "itself must be untouched by anything this block did")


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
    assert receipt["positive_control"]["injected"] is False
    assert receipt["positive_control"]["injected_done"] is False
    assert receipt["positive_control"]["caught"] is False
    assert receipt["positive_control"]["banked_from_reset"] == 0


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
    # The artifact was real and is still on disk, moved out of the
    # directory banking reads rather than deleted (see
    # test_abort_quarantines_the_solutions_directory for that half).
    quarantined = receipt["quarantined_solutions"]
    assert quarantined is not None
    assert (Path(quarantined) / "sol_0.json").exists(), (
        "fixture must have had a real solution artifact on disk")
    assert receipt["banked"] == [], (
        "a wrongful reset must bank nothing, even with a real solution "
        "artifact present at abort time")


# ---------------------------------------------------------------------
# test_abort_quarantines_the_solutions_directory
# ---------------------------------------------------------------------

def test_abort_quarantines_the_solutions_directory(tmp_path):
    """`banked: []` on the abort path is a statement about one receipt,
    in memory. The artifacts have to leave the directory the next block
    on this same `--out` banks from, or that block reports them as its
    own. Moved, not deleted: the evidence of a wrongful reset is the
    thing a later reader most needs.

    Revert-verify (live, this session): deleting the
    `_quarantine_solutions` call from the abort branch of `run_block`
    leaves `quarantined_solutions` None and `out/solutions/sol_0.json`
    in place -- fails `assert quarantined is not None` by name.
    Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "abort_quarantine",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, write_solution=True, exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "abort"
    quarantined = receipt["quarantined_solutions"]
    assert quarantined is not None, (
        "an aborted block must move its solutions/ out of the bank path")
    assert (Path(quarantined) / "sol_0.json").exists(), (
        "the artifact must be preserved, not destroyed")
    assert not (tmp_path / "out" / "solutions").exists(), (
        "solutions/ must no longer sit where _bank_solutions reads")


# ---------------------------------------------------------------------
# test_next_block_on_the_same_out_does_not_bank_an_aborted_blocks_work
# ---------------------------------------------------------------------

def test_next_block_on_the_same_out_does_not_bank_an_aborted_blocks_work(
        tmp_path):
    """The defect FORGE-FIX-9 names, end to end: `<--out>` is reused
    (`mkdir(exist_ok=True)`, and `loop._out_dir_for` keys the directory
    on `cycle_id`, so a retried cycle is the same path). A block that
    aborts on a wrongful reset leaves its artifacts in `solutions/`; the
    next block on that directory stops cleanly, banks them, and reports
    `watchdog_trips: 0`. No single receipt then carries both the trip
    and the bank, so the grant's "one wrongful reset that banks an
    artifact ends the grant" (CLAIMS.md:1014-1015) can never fire.

    The second block writes no solution of its own, so anything in
    `banked` came from the first. Its rows climb from ABOVE the first
    block's last row (100, 200 against the first's final 40) because
    `progress.jsonl` is appended to, not cleared, on a reused `--out`:
    a second child starting from a lower cell count trips the watchdog
    against the previous run's tail. That is a separate defect
    (FORGE-FIX-26) and this test steps around it rather than proving
    it, so that a failure here is about banking and nothing else.

    Revert-verify (live, this session): with BOTH guards removed --
    the `_quarantine_solutions` call deleted from the abort branch AND
    `since=bank_floor` dropped from the `_bank_solutions` call -- the
    second block's `banked` reads `['out/sandbox/sol_0.json']` and the
    final assertion fails by name. Each guard alone also holds this
    test green, which is the point of having two; each is pinned on its
    own by `test_abort_quarantines_the_solutions_directory` and
    `test_bank_floor_excludes_an_artifact_predating_launch`. Restored,
    re-passed.
    """
    claims, grant_state = _grant_paths(tmp_path)

    aborting = _write_fixture(
        tmp_path, "first_aborts",
        rows=[{"cells": 100, "steps": 10}, {"cells": 150, "steps": 20},
              {"cells": 40, "steps": 30}],
        row_interval=0.05, write_solution=True, exit_after_rows=False)
    first = run_block(
        _make_plan(tmp_path, aborting), repo=tmp_path, claims_path=claims,
        grant_state_path=grant_state, poll_interval=0.01,
        term_grace_s=1.0, kill_grace_s=1.0)
    assert first["stop"] == "abort"
    assert first["banked"] == []

    clean = _write_fixture(
        tmp_path, "second_is_clean",
        rows=[{"cells": 100, "steps": 100}, {"cells": 200, "steps": 200}],
        row_interval=0.02, write_solution=False, exit_after_rows=True)
    second = run_block(
        _make_plan(tmp_path, clean, cycle_id="cycle_0"), repo=tmp_path,
        claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert second["stop"] == "complete"
    assert second["watchdog_trips"] == 0
    assert second["banked"] == [], (
        "a clean block must not bank the artifacts an earlier aborted "
        "block left in the same --out")


# ---------------------------------------------------------------------
# test_bank_floor_excludes_an_artifact_predating_launch
# ---------------------------------------------------------------------

def test_bank_floor_excludes_an_artifact_predating_launch(tmp_path):
    """The second, independent guard: an artifact whose mtime predates
    this block's own launch was not produced by this block, whatever
    left it there. This holds even if the earlier block never ran its
    own quarantine (killed between the abort and the move, or a
    directory written by something else entirely).

    The block's own artifact, written after launch, still banks -- the
    floor must exclude the stale file without excluding the real one.

    Revert-verify (live, this session): dropping `since=bank_floor` from
    the `_bank_solutions` call in `run_block` puts `out/sandbox/
    sol_stale.json` in `banked` -- fails the `sol_stale.json not in`
    assertion by name. Restored, re-passed.
    """
    stale_dir = tmp_path / "out" / "solutions"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale = stale_dir / "sol_stale.json"
    stale.write_text(json.dumps({"from": "an earlier aborted block"}))
    old = time.time() - 3600
    os.utime(stale, (old, old))

    script = _write_fixture(
        tmp_path, "clean_over_stale",
        rows=[{"cells": 10, "steps": 100}, {"cells": 20, "steps": 200}],
        row_interval=0.02, write_solution=True, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)

    receipt = run_block(
        _make_plan(tmp_path, script), repo=tmp_path, claims_path=claims,
        grant_state_path=grant_state, poll_interval=0.01,
        term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "complete"
    assert stale.exists(), "the stale artifact is left alone, not destroyed"
    banked = receipt["banked"]
    assert any(b.endswith("sandbox/sol_0.json") for b in banked), (
        "this block's OWN artifact must still bank")
    assert not any(b.endswith("sol_stale.json") for b in banked), (
        "an artifact predating launch is not this block's to bank")


# ---------------------------------------------------------------------
# test_out_dir_that_cannot_take_the_bank_floor_marker_is_refused
# ---------------------------------------------------------------------

def test_out_dir_that_cannot_take_the_bank_floor_marker_is_refused(
        tmp_path, monkeypatch):
    """No floor, no banking. An `--out` that cannot take the marker
    would otherwise fall back to banking whatever it found, which is
    the behaviour the floor exists to remove.

    The unwritable directory is simulated by patching
    `_make_bank_floor` to its own OSError return rather than by
    chmodding `--out`: a genuinely read-only `--out` also blocks
    `write_block_receipt`, so the block would die on the receipt write
    and the refusal under test would never be the observed cause.

    Revert-verify (live, this session): replacing the
    `bank_floor_unwritable` refusal with `bank_floor = None` (bank
    everything) launches the child, which `_forbidden_launch` turns
    into an AssertionError -- fails by name. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "never_launched",
        rows=[{"cells": 10, "steps": 100}], exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script)

    monkeypatch.setattr(block_module, "_make_bank_floor", lambda _out: None)
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims,
        grant_state_path=grant_state, launch_fn=_forbidden_launch,
        poll_interval=0.01)

    assert receipt["stop"] == "abort"
    assert receipt["abort_reason"] == "bank_floor_unwritable"
    assert receipt["banked"] == []


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


# ---------------------------------------------------------------------
# The receipt on disk (FORGE-FIX-1)
# ---------------------------------------------------------------------

def test_block_receipt_is_written_to_disk_with_every_grant_field(tmp_path):
    """A clean block leaves ``block_receipt.json`` in its own ``--out``
    directory carrying every field the two FORGE-GRANT entries fix,
    with attended hours beside run-lock hours, the watchdog-trip count,
    the positive control, the banked list, and the SHARED root state's
    full sha256 read before launch and again after the block ended.
    The returned dict names the same path.

    Before this, ``run_block`` returned the receipt and wrote nothing,
    so a live block under the grant left no receipt on disk at all.

    Revert-verify (live, this session): delete the
    ``write_block_receipt(rcpt, out_state["dir"])`` call in
    ``run_block``'s ``_finish`` and this fails by name on the first
    assertion -- the file does not exist. Restored, re-passed.
    """
    script = _write_fixture(
        tmp_path, "receipt_on_disk",
        rows=[{"cells": 10, "steps": 100}, {"cells": 20, "steps": 200}],
        row_interval=0.02, write_solution=True, exit_after_rows=True)
    claims, grant_state = _grant_paths(tmp_path)
    root_path, root_sha = _write_root_state(tmp_path)
    shared_before = hashlib.sha256(root_path.read_bytes()).hexdigest()
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=root_sha)
    # Two real attended rows so the ratio divides by a real number.
    attended = tmp_path / "attended.jsonl"
    append_attended(attended, who="operator", checked="launch preflight",
                    start="2026-09-02T12:00:00", end="2026-09-02T12:30:00")
    plan["attended_log"] = str(attended)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    receipt_file = tmp_path / "out" / "block_receipt.json"
    assert receipt_file.exists(), (
        "run_block returned a receipt but wrote none to disk")
    assert receipt["receipt_path"] == str(receipt_file)
    on_disk = json.loads(receipt_file.read_text())

    missing = [f for f in GRANT_RECEIPT_FIELDS if f not in on_disk]
    assert missing == [], f"receipt on disk is missing grant fields: {missing}"
    for sub in GRANT_POSITIVE_CONTROL_FIELDS:
        assert sub in on_disk["positive_control"]

    assert on_disk["stop"] == "complete"
    assert on_disk["watchdog_trips"] == 0
    assert on_disk["banked"], "a clean block with a solution must record it"
    # Attended hours beside run-lock hours, both real numbers.
    assert on_disk["attended_hours"] == 0.5
    assert on_disk["run_lock_hours"] >= 0.0
    assert on_disk["ratio_machine_per_attended"] is not None
    # The shared root state, unchanged by the block: the reading ruling
    # 17 judges a control on. Full digests, not the 16-char fingerprint.
    assert on_disk["root_state_sha256_before"] == shared_before
    assert on_disk["root_state_sha256_after"] == shared_before
    assert len(on_disk["root_state_sha256_before"]) == 64


def test_receipt_missing_a_grant_required_field_is_refused(tmp_path):
    """``write_block_receipt`` refuses a receipt that omits any field
    the grant fixes, names every one it is missing, and writes no file
    -- a partial receipt must never reach disk to be read later as if
    it were whole. A missing ``positive_control`` sub-field is refused
    on the same footing as a missing top-level field.

    Revert-verify (live, this session): drop the ``missing`` check (or
    make ``write_block_receipt`` return the path unconditionally) and
    both ``pytest.raises`` blocks fail to raise, and the
    "no file written" assertions fail -- the partial receipt lands.
    Restored, re-passed.
    """
    whole = {f: None for f in GRANT_RECEIPT_FIELDS}
    whole["positive_control"] = {f: None for f in GRANT_POSITIVE_CONTROL_FIELDS}
    out = tmp_path / "out"

    # The whole receipt is accepted, so the refusals below are about the
    # missing field and not about the fixture being wrong throughout.
    assert write_block_receipt(dict(whole), out).exists()
    (out / "block_receipt.json").unlink()

    partial = dict(whole)
    del partial["attended_hours"]
    del partial["watchdog_trips"]
    with pytest.raises(ReceiptFieldError) as exc_info:
        write_block_receipt(partial, out)
    assert "attended_hours" in exc_info.value.missing
    assert "watchdog_trips" in exc_info.value.missing
    assert not (out / "block_receipt.json").exists(), (
        "a refused receipt was written to disk anyway")

    no_control = dict(whole)
    no_control["positive_control"] = {"injected": True}
    with pytest.raises(ReceiptFieldError) as exc_info:
        write_block_receipt(no_control, out)
    assert "positive_control.caught" in exc_info.value.missing
    assert "positive_control.banked_from_reset" in exc_info.value.missing
    assert not (out / "block_receipt.json").exists()


def test_attended_hook_records_who_and_what_was_checked(tmp_path):
    """``append_attended`` writes rows ``_attended_hours`` can read, so
    attended hours are recorded by the person who was there rather than
    asserted afterwards: two half-hour intervals read as 1.0 hours, and
    each row carries who was at the keyboard and what they checked. A
    row with no ``start`` is a timestamped checkpoint contributing no
    hours, never a silent interval.

    Revert-verify (live, this session): change ``append_attended``'s
    ``open(path, "a")`` to ``open(path, "w")`` and the second-row
    assertions fail -- the file holds one row and ``_attended_hours``
    reads 0.5, not 1.0. Restored, re-passed.
    """
    log = tmp_path / "cycle" / "attended.jsonl"

    append_attended(log, who="matthew", checked="preflight shas and lock dir",
                    start="2026-09-02T12:00:00", end="2026-09-02T12:30:00")
    append_attended(log, who="matthew", checked="null block progress rows",
                    start="2026-09-02T13:00:00", end="2026-09-02T13:30:00")

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(rows) == 2, "append_attended overwrote an earlier row"
    assert [r["who"] for r in rows] == ["matthew", "matthew"]
    assert rows[0]["checked"] == "preflight shas and lock dir"
    assert rows[1]["checked"] == "null block progress rows"
    assert _attended_hours(log) == 1.0

    append_attended(log, who="matthew", checked="mid-block glance",
                    now_fn=lambda: "2026-09-02T14:00:00")
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert rows[2]["start"] == rows[2]["end"] == "2026-09-02T14:00:00"
    assert _attended_hours(log) == 1.0, (
        "a checkpoint row with no interval must contribute no hours")


# ---------------------------------------------------------------------
# FORGE-FIX-4: an unreadable watchdog input is a trip, not a blind spot
# ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["unlink", "chmod"])
def test_root_state_made_unreadable_mid_block_trips_the_watchdog(tmp_path, mode):
    """A monotone, never-misbehaving child with a valid `root_state_sha`,
    whose sandboxed root-state copy is DELETED (or made unreadable) out
    from under the poll loop partway through the block.

    Before FORGE-FIX-4 this switched condition 3 off for the rest of the
    block instead of tripping it: `_sha_of_file` returns None on any
    OSError (`src/forge/block.py:257-258` at HEAD `4986043`) and
    `wrongful_reset` skipped the comparison whenever `current_root_sha`
    was None (`:381`), so the block ran to `stop:"complete"` and reached
    `_bank_solutions`. `write_solution=True` here so `banked == []` is a
    real assertion and not vacuously true for want of anything to bank.

    Revert-verify (live, this session), both modes: deleting the
    `if root_state_seen and current_root_sha is None` branch from
    `wrongful_reset` puts the block back on `stop:"complete"`,
    `watchdog_trips:0`, `banked:["out/sandbox/sol_0.json"]` -- fails on
    the first assertion by name. Restored, re-passed. Separately,
    initialising `root_state_seen = False` instead of
    `root_state_path is not None` in `run_block` fails the same way
    (nothing ever sets the latch once the file stops reading).
    """
    root_path, correct_sha = _write_root_state(tmp_path)
    rows = [{"cells": i, "steps": i} for i in range(1, 41)]
    script = _write_fixture(tmp_path, "root_lost", rows=rows,
                            row_interval=0.03, write_solution=True,
                            exit_after_rows=False)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, root_state_path=root_path,
                      root_state_sha=correct_sha, max_secs=5.0)
    copy_path = tmp_path / "out" / "sandbox" / f"root_state{root_path.suffix}"

    def _break_after_delay():
        deadline = time.time() + 2.0
        while not copy_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        if mode == "unlink":
            copy_path.unlink()
        else:
            os.chmod(copy_path, 0o000)

    breaker = threading.Thread(target=_break_after_delay, daemon=True)
    breaker.start()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)
    breaker.join(timeout=2.0)
    if mode == "chmod" and copy_path.exists():
        os.chmod(copy_path, 0o644)

    assert receipt["stop"] == "abort"
    assert receipt["abort_reason"] == "root_state_lost"
    assert receipt["watchdog_trips"] == 1
    assert receipt["aborted"] is True
    assert receipt["banked"] == []
    assert receipt["positive_control"]["caught"] is False


def test_child_run_lock_made_unreadable_mid_block_trips_the_watchdog(tmp_path):
    """The same blind spot on condition 4. The child writes its own
    `.run.lock`, the runner reads it, and then the lock is truncated to
    garbage out from under the loop while the child is still running.

    `run_lock.read_lock` returns None for a missing OR unparseable lock
    (`src/utils/run_lock.py:100-101`), so before FORGE-FIX-4 the holder
    comparison went off for the rest of the block. Truncation rather
    than deletion here so the case being pinned is the "garbled" half of
    that `None`, which no unlink test would reach.

    Revert-verify (live, this session): deleting the `if lock_unreadable`
    branch from `wrongful_reset` runs the block to `stop:"budget"` with
    `watchdog_trips:0` -- fails the first assertion by name. Restored,
    re-passed.
    """
    rows = [{"cells": i, "steps": i} for i in range(1, 41)]
    script = _write_fixture(tmp_path, "lock_lost", rows=rows,
                            row_interval=0.03, exit_after_rows=False,
                            write_own_run_lock=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, max_secs=5.0)
    lock_path = tmp_path / "out" / ".run.lock"

    def _garble_after_delay():
        deadline = time.time() + 2.0
        while not lock_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(0.05)  # let at least one poll read it intact
        lock_path.write_text("not-a-pid\n")

    garbler = threading.Thread(target=_garble_after_delay, daemon=True)
    garbler.start()
    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)
    garbler.join(timeout=2.0)

    assert receipt["stop"] == "abort"
    assert receipt["abort_reason"] == "lock_unreadable"
    assert receipt["watchdog_trips"] == 1


def test_child_releasing_its_own_lock_on_a_clean_exit_is_not_a_trip(tmp_path):
    """The regression the `lock_seen` latch could have introduced, and
    the reason it is dropped again when the child is gone.

    A well-behaved child unlinks its own `.run.lock` as it exits (the
    real solver registers that unlink with atexit). The poll after that
    exit reads the lock as None having read it as the child's pid a
    moment earlier, which is the same `None` the new condition trips
    on. It must not: the child is dead, the lock went with it, the block
    stopped cleanly and its solution is banked. This is why
    `lock_unreadable` is `holder is None AND lock_path.exists()` and not
    a latch on "the lock was readable once": the first draft of this fix
    used the latch, de-latched on `not _pid_alive(child_pid)`, and this
    test caught it failing intermittently -- atexit unlinks the lock
    BEFORE the process dies, so a poll can land inside that window and
    see a live child with no lock.

    Revert-verify (live, this session): widening the condition to
    `lock_unreadable = holder is None` (dropping the `and
    lock_path.exists()`) turns every clean-exiting child that holds a
    lock into `stop:"abort"`, `abort_reason:"lock_unreadable"`,
    `banked:[]` -- fails all three assertions below by name. Restored,
    re-passed.
    """
    script = _write_fixture(
        tmp_path, "lock_clean_exit",
        rows=[{"cells": 10, "steps": 10}, {"cells": 20, "steps": 20}],
        row_interval=0.03, write_solution=True, exit_after_rows=True,
        write_own_run_lock=True, release_run_lock_on_exit=True)
    claims, grant_state = _grant_paths(tmp_path)
    plan = _make_plan(tmp_path, script, max_secs=5.0)

    receipt = run_block(
        plan, repo=tmp_path, claims_path=claims, grant_state_path=grant_state,
        poll_interval=0.01, term_grace_s=1.0, kill_grace_s=1.0)

    assert receipt["stop"] == "complete"
    assert receipt["watchdog_trips"] == 0
    assert receipt["banked"] == ["out/sandbox/sol_0.json"]


def test_wrongful_reset_catches_a_root_state_that_stopped_reading():
    got = wrongful_reset(None, None, root_state_sha="aaa",
                         current_root_sha=None, root_state_seen=True)
    assert got == {"reason": "root_state_lost"}


def test_wrongful_reset_catches_a_lock_that_stopped_reading():
    got = wrongful_reset(None, None, lock_holder_pid=None, child_pid=111,
                         lock_unreadable=True)
    assert got == {"reason": "lock_unreadable"}


def test_wrongful_reset_still_ignores_a_file_it_has_never_read():
    """The default both ways round: a caller that has not yet read the
    root state or the lock keeps the old "nothing to compare" reading,
    so a pre-launch or unit call site cannot manufacture a trip out of
    a file that was simply never there."""
    assert wrongful_reset(None, None, root_state_sha="aaa",
                          current_root_sha=None) == {"reason": None}
    assert wrongful_reset(None, None, lock_holder_pid=None,
                          child_pid=111) == {"reason": None}
