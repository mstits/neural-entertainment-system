"""scripts/engine_driver.py — the unattended engine's safety properties.

The bar is three weeks alone. Every test here pins a behaviour whose
absence would turn that into three weeks of nothing: emitting a command
that cannot run, launching a second emulator job onto a busy machine,
retrying a poisoned action forever, or halting on a corrupt state file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.engine_driver as ed  # noqa: E402
from tests.skip_gates import requires  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect all engine state to tmp AND neuter the launcher.

    Stubbing `launch` here rather than per-test is not tidiness. Several
    tests call `tick()` for its decision, and `tick` launches through
    `detach`, which starts a process in its OWN session that outlives the
    test run. One pass of this file left fourteen detached full-suite
    pytest processes alive, one of them at 713% CPU, which took the
    machine to a load of 20 and stalled the git commit that followed.

    A test suite must not be able to start background work by accident,
    so the capability is removed for every test in the file and the fake
    records what would have been launched.
    """
    monkeypatch.setattr(ed, "ENGINE_DIR", tmp_path / "engine")
    monkeypatch.setattr(ed, "JOURNAL", tmp_path / "engine" / "journal.jsonl")
    monkeypatch.setattr(ed, "STATE_PATH", tmp_path / "engine" / "state.json")
    launched: list = []
    monkeypatch.setattr(
        ed, "launch",
        lambda action, repo=ed.REPO: (launched.append(action), 4242)[1])
    monkeypatch.setattr(ed, "_test_launched", launched, raising=False)


def _script(tmp_path: Path, name: str, flags: list[str]) -> Path:
    p = tmp_path / "scripts" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'ap.add_argument("{f}")' for f in flags)
    p.write_text(f"import argparse\nap = argparse.ArgumentParser()\n{body}\n")
    return p


def _act(cmd, **kw):
    return ed.Action(id=kw.pop("id", "a"), kind="k", cmd=cmd,
                     needs_emulator=True, gate="g", **kw)


# ---- validation: the guard that separates an engine from a liability ----

def test_action_naming_a_missing_script_is_rejected(tmp_path):
    ok, why = ed.validate_action(_act(["scripts/nope.py"]), tmp_path)
    assert not ok and "no such script" in why


def test_action_using_an_undeclared_flag_is_rejected(tmp_path):
    _script(tmp_path, "s.py", ["--real"])
    ok, why = ed.validate_action(
        _act(["scripts/s.py", "--invented", "x"]), tmp_path)
    assert not ok and "--invented" in why


def test_action_with_declared_flags_passes(tmp_path):
    _script(tmp_path, "s.py", ["--real"])
    ok, why = ed.validate_action(_act(["scripts/s.py", "--real", "x"]),
                                 tmp_path)
    assert ok, why


def test_action_with_a_missing_input_path_is_rejected(tmp_path):
    _script(tmp_path, "s.py", ["--profile"])
    ok, why = ed.validate_action(
        _act(["scripts/s.py", "--profile", "configs/gone.yaml"]), tmp_path)
    assert not ok and "input path missing" in why


def test_script_flags_reads_declarations_without_executing(tmp_path):
    """Discovering an interface must never run the script."""
    p = _script(tmp_path, "s.py", ["--alpha", "--beta"])
    p.write_text(p.read_text() + "\nraise SystemExit('must not run')\n")
    declared, _required = ed.script_flags(p)
    assert declared == {"--alpha", "--beta"}


# ---- the physics budget ----

def test_never_launches_an_emulator_action_while_one_is_live(monkeypatch):
    """The physics budget: one emulator job, always.

    The engine may still do token-bound work beside a campaign — that is
    the brief's own scheduling rule and most of why the loop felt stalled
    — but whatever it picks must not need the emulator.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    rec = ed.tick({"completed": {}, "attempts": {}})
    launched = ed._test_launched
    assert all(not a.needs_emulator for a in launched), launched
    if rec["decision"] == "wait":
        assert "4242" in rec["reason"]


def test_waits_when_busy_and_no_token_bound_work_remains(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    monkeypatch.setattr(ed, "plan",
                        lambda s, repo=ed.REPO, emulator_only=None:
                        _act(["scripts/x.py"]) if emulator_only is None
                        else None)
    rec = ed.tick({"completed": {}, "attempts": {}, "last_run": {}})
    assert rec["decision"] == "wait", rec
    assert "4242" in rec["reason"] and "token-bound" in rec["reason"]


def test_token_bound_work_is_offered_while_the_emulator_is_busy():
    a = ed.plan({"completed": {}, "attempts": {}}, emulator_only=False)
    assert a is not None and not a.needs_emulator


def test_unknown_process_table_is_treated_as_busy(monkeypatch):
    """Failing to read ps must never be read as 'the machine is free'."""
    def boom(*a, **k):
        raise OSError("no ps")
    monkeypatch.setattr(ed.subprocess, "run", boom)
    assert ed.emulator_busy() == -1


# ---- circuit breaker, caps, floors ----

def test_circuit_breaker_blocks_after_consecutive_failures(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    state = {"completed": {}, "attempts": {},
             "consecutive_failures": ed.CONSECUTIVE_FAILURE_LIMIT}
    rec = ed.tick(state)
    assert rec["decision"] == "blocked" and "circuit breaker" in rec["reason"]


def test_disk_floor_blocks_launching(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "disk_free_gb", lambda *a: 1.0)
    rec = ed.tick({"completed": {}, "attempts": {}})
    assert rec["decision"] == "blocked" and "disk floor" in rec["reason"]


def test_explicit_halt_blocks_launching(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    rec = ed.tick({"completed": {}, "attempts": {}, "halted": "by hand"})
    assert rec["decision"] == "blocked" and "by hand" in rec["reason"]


def test_attempt_cap_abandons_rather_than_retrying_forever(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "plan", lambda s, repo=ed.REPO: _act(
        ["scripts/x.py"], id="stuck"))
    state = {"completed": {}, "attempts": {"stuck": ed.MAX_ATTEMPTS_PER_ACTION}}
    rec = ed.tick(state)
    assert rec["decision"] == "abandon"
    assert state["completed"]["stuck"]["status"] == "abandoned"


# ---- planning ----

def test_plan_skips_an_unrunnable_action_instead_of_emitting_it(monkeypatch):
    """Degrade to the work that can be done; never emit a fake command."""
    monkeypatch.setattr(ed, "validate_action",
                        lambda a, repo=ed.REPO: (False, "broken"))
    assert ed.plan({"completed": {}, "attempts": {}}) is None


def test_shelf_dispositions_missing_files_are_journaled_not_silent(tmp_path):
    """Regression: the shelf-loop's `continue`s used to bypass offer() and
    the `skipped` journal entirely, so a missing profile/rom/start-state/
    checkpoint for a shelf item (e.g. shelf_joint_1_1, the owed
    >=100-episode interference follow-up) vanished from the plan with
    zero trace in journal.jsonl, unlike every other rejection in plan().
    """
    result = ed.plan({"completed": {}, "attempts": {}, "last_run": {}},
                     tmp_path)
    assert result is None        # tmp_path has none of the real scripts
    assert ed.JOURNAL.exists(), "plan_skipped must have been journaled"
    rows = [json.loads(line) for line in ed.JOURNAL.read_text().splitlines()
           if line.strip()]
    skip_rows = [r for r in rows if r.get("type") == "plan_skipped"]
    assert skip_rows, "no plan_skipped row was written"
    all_skipped = [s for r in skip_rows for s in r.get("skipped", [])]
    assert any(s.startswith("shelf_joint_1_1:") for s in all_skipped), (
        f"shelf_joint_1_1's missing profile vanished with no trace: "
        f"{all_skipped}")
    assert any(s.startswith("shelf_1_4_endpoint:") for s in all_skipped)


def test_plan_does_not_repeat_a_completed_nonrecurring_action():
    done: dict = {}
    seen = set()
    for _ in range(6):
        a = ed.plan({"completed": done, "attempts": {}})
        if a is None:
            break
        if a.recurring:            # recurring actions are meant to repeat
            done[a.id] = {"status": "ok"}
            continue
        assert a.id not in seen, f"{a.id} re-offered after completion"
        seen.add(a.id)
        done[a.id] = {"status": "ok"}


def test_a_recurring_action_is_re_offered_after_completion(tmp_path,
                                                            monkeypatch):
    """Capping the suite check would silence the watchdog that caught a
    test red for three days.

    Run against a synthetic repo holding only run_suite_check.py, so the
    recurring action is the sole valid candidate — the real repo offers a
    dozen mission actions ahead of it now that maintenance ranks last,
    and draining them would make this test a minute long for no gain.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    sc = tmp_path / "scripts" / "run_suite_check.py"
    sc.parent.mkdir(parents=True)
    sc.write_text('import argparse\nap = argparse.ArgumentParser()\n'
                  'ap.add_argument("--out")\nap.add_argument("--timeout")\n')

    first = ed.plan({"completed": {}, "attempts": {}, "last_run": {}},
                    tmp_path)
    assert first is not None and first.id == "suite_check", first
    assert first.recurring

    again = ed.plan({"completed": {"suite_check": {"status": "succeeded"}},
                     "attempts": {}, "last_run": {}}, tmp_path)
    assert again is not None and again.id == "suite_check", (
        "a completed recurring action was not re-offered")


def test_suite_check_outer_timeout_exceeds_its_own_inner_timeout(monkeypatch, tmp_path):
    """suite_check's --timeout flag is run_suite_check.py's OWN
    subprocess.run timeout; the outer engine_driver timeout_h reaper must
    give that inner timeout the chance to fire and report cleanly first.
    At timeout_h=1.0 against --timeout 5400 (1.5h), a merely-slow suite
    was SIGKILLed 1500s before its own timeout could trigger, and the
    kill counted as an action-level failure toward the circuit breaker
    for a suite that was never actually stuck."""
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    sc = tmp_path / "scripts" / "run_suite_check.py"
    sc.parent.mkdir(parents=True)
    sc.write_text('import argparse\nap = argparse.ArgumentParser()\n'
                  'ap.add_argument("--out")\nap.add_argument("--timeout")\n')

    action = ed.plan({"completed": {}, "attempts": {}, "last_run": {}}, tmp_path)
    assert action is not None and action.id == "suite_check"

    inner_timeout_s = int(action.cmd[action.cmd.index("--timeout") + 1])
    outer_timeout_s = action.timeout_h * 3600
    assert outer_timeout_s > inner_timeout_s, (
        f"outer timeout_h ({action.timeout_h}h = {outer_timeout_s}s) must "
        f"exceed the inner --timeout ({inner_timeout_s}s) it wraps, or the "
        "reaper kills the action before the inner script's own timeout "
        "ever gets a chance to fire"
    )


def test_a_recurring_action_is_exempt_from_the_attempt_cap(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "plan", lambda s, repo=ed.REPO,
                        emulator_only=None: ed.Action(
                            id="rec", kind="k", cmd=["scripts/x.py"],
                            needs_emulator=False, gate="g", recurring=True))
    state = {"completed": {}, "attempts": {"rec": 99}}
    rec = ed.tick(state)
    assert rec["decision"] == "launched"


# ---- honest-eval detection ----

def _eval_json(d: Path, name: str, game: str, seed: int, eps: int = 50):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(
        {"game": game, "eval_seed": seed, "n_episodes": eps,
         "sticky_prob": 0.25}))


def test_two_seeds_are_required_to_count_as_scored(tmp_path):
    runs = tmp_path / "runs" / "anywhere"
    _eval_json(runs, "final_eval_seed7.json", "mario_9_9_online_v1", 7)
    assert not ed.honest_eval_done("9-9", tmp_path)
    _eval_json(runs, "final_eval_seed101.json", "mario_9_9_online_v1", 101)
    assert ed.honest_eval_done("9-9", tmp_path)


def test_detection_is_by_content_not_by_directory(tmp_path):
    """1-3's receipts live under consol2_1_3_round2, not online_1_3."""
    odd = tmp_path / "runs" / "some_unrelated_dir"
    _eval_json(odd, "final_eval_seed7.json", "mario_9_9_online_v1", 7)
    _eval_json(odd, "final_eval_seed101.json", "mario_9_9_online_v1", 101)
    assert ed.honest_eval_done("9-9", tmp_path)


def test_a_short_probe_does_not_count_as_an_honest_eval(tmp_path):
    """30-episode probes overstate; they are not the protocol."""
    runs = tmp_path / "runs" / "x"
    _eval_json(runs, "a_seed7.json", "mario_9_9_online_v1", 7, eps=30)
    _eval_json(runs, "a_seed101.json", "mario_9_9_online_v1", 101, eps=30)
    assert not ed.honest_eval_done("9-9", tmp_path)


# ---- durability ----

def test_corrupt_state_file_resets_instead_of_wedging(tmp_path):
    ed.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ed.STATE_PATH.write_text("{ this is not json")
    st = ed.load_state()
    assert st["completed"] == {} and st["consecutive_failures"] == 0


def test_state_is_written_atomically(tmp_path):
    ed.save_state({"completed": {}, "attempts": {}, "marker": 1})
    assert json.loads(ed.STATE_PATH.read_text())["marker"] == 1
    assert not ed.STATE_PATH.with_suffix(".tmp").exists()


def test_intent_is_journalled_before_the_action_runs(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "plan", lambda s, repo=ed.REPO: _act(
        ["scripts/x.py"], id="j"))
    ed.tick({"completed": {}, "attempts": {}}, dry=True)
    kinds = [json.loads(l)["type"] for l in ed.JOURNAL.read_text().splitlines()]
    assert "launch_intent" in kinds


# ---- the reaper: an exit is not an outcome ----

def test_reap_marks_success_only_when_the_marker_exists(tmp_path):
    marker = tmp_path / "done.json"
    marker.write_text("{}")
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 2,
             "running": {"id": "a", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "done.json",
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "succeeded"
    assert state["consecutive_failures"] == 0, "success must reset the breaker"
    assert state["completed"]["a"]["status"] == "succeeded"
    assert not ed.running_slots(state), "slot not released"


def test_reap_counts_a_missing_marker_as_failure_and_arms_the_breaker(tmp_path):
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "never.json",
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "failed"
    assert state["consecutive_failures"] == 1


def test_reap_will_not_infer_an_outcome_without_a_marker(tmp_path):
    """A process ending is not evidence the work succeeded."""
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": None,
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "finished"
    assert state["consecutive_failures"] == 0, "no marker, no verdict"


def test_reap_leaves_a_live_action_alone(tmp_path):
    state = {"completed": {}, "attempts": {},
             "running": {"id": "a", "pid": os.getpid(), "started": time.time(),
                         "timeout_h": 10.0, "done_marker": None,
                         "recurring": False}}
    assert ed.reap(state, tmp_path) is None
    assert state["running"] is not None


def test_reap_kills_and_fails_an_action_past_its_timeout(tmp_path, monkeypatch):
    killed_groups = []
    monkeypatch.setattr(ed.os, "killpg",
                        lambda pgid, sig: killed_groups.append((pgid, sig)))
    monkeypatch.setattr(ed.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(ed, "pid_alive", lambda pid: True)
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 4242, "started": 0.0,
                         "timeout_h": 0.001, "done_marker": None,
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "timeout"
    assert (4242, 9) in killed_groups
    assert state["consecutive_failures"] == 1


def test_reap_falls_back_to_plain_kill_if_getpgid_fails(tmp_path, monkeypatch):
    """A pid with no resolvable process group (already reaped, foreign
    namespace, ...) must still be reaped -- os.kill is the fallback, not
    a second unhandled OSError."""
    plain_killed = []
    monkeypatch.setattr(ed.os, "getpgid",
                        lambda pid: (_ for _ in ()).throw(OSError("no such pgid")))
    monkeypatch.setattr(ed.os, "kill",
                        lambda pid, sig: plain_killed.append((pid, sig)))
    monkeypatch.setattr(ed, "pid_alive", lambda pid: True)
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 4242, "started": 0.0,
                         "timeout_h": 0.001, "done_marker": None,
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "timeout"
    assert (4242, 9) in plain_killed


# ---- regression: killing a session leader must kill its children too ----
#
# detach.py launches every engine_driver-managed process with
# start_new_session=True (os.setsid() in the child), which makes that
# process a session/process-group leader. A wall-clock timeout that then
# calls plain os.kill(pid, 9) kills only the leader -- run_online_
# campaign.py, say -- and leaves its own child (train_game.py, the actual
# emulator lane) running as an orphan. The ps-based busy check then
# reports that lane held externally forever: a silent stall with no
# further diagnostic, exactly the failure a multi-day unattended run
# exists to survive. This test forks a REAL session leader with a REAL
# child and proves the fix kills both; reverting to plain os.kill leaves
# the child alive and this test fails.

def test_killing_a_session_leader_also_kills_its_child(tmp_path):
    """Exercises ed.reap() itself, not a hand-rolled killpg call -- a test
    that calls os.killpg directly proves nothing about what reap() does
    and would pass unchanged if reap() still called plain os.kill. Caught
    exactly that way while writing this: the first version passed with
    the fix reverted."""
    import subprocess
    import time as _time

    marker = tmp_path / "child_alive"
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time; "
         "p = subprocess.Popen([sys.executable, '-c', "
         "'import time,sys\\nwhile True:\\n open(sys.argv[1], \"w\").close()\\n time.sleep(0.05)', "
         f"{str(marker)!r}]); "
         "time.sleep(3600)"],
        start_new_session=True,
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            _time.sleep(0.05)
        assert marker.exists(), "child never started writing its liveness marker"

        state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
                 "running": {"id": "a", "pid": leader.pid, "started": 0.0,
                             "timeout_h": 0.0, "done_marker": None,
                             "recurring": False}}
        rec = ed.reap(state, tmp_path)
        assert rec["outcome"] == "timeout"
        leader.wait(timeout=5)

        before = marker.stat().st_mtime
        _time.sleep(0.5)
        after = marker.stat().st_mtime
        assert after == before, (
            "child kept writing after ed.reap() killed the session leader "
            "-- reap() did not reach it (plain os.kill semantics)"
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def test_a_recurring_action_is_never_recorded_as_completed(tmp_path):
    marker = tmp_path / "d.json"
    marker.write_text("{}")
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "suite", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "d.json",
                         "recurring": True}}
    ed.reap(state, tmp_path)
    assert "suite" not in state["completed"]


# ---- regression: a single crash must not out-run the attempt cap ----
#
# _reap_one used to write into state["completed"] on EVERY non-recurring
# exit — success, verified failure, and the ambiguous "finished" (no
# done_marker) case alike. offer() excludes anything already in
# `completed` before tick()'s MAX_ATTEMPTS_PER_ACTION check ever runs, so
# a single crash (transient OOM, a momentarily-locked ROM, a disk
# hiccup) permanently retired the action id with no further attempt and
# no diagnostic. honest_eval_action and the onboarding steps
# (config_/mint_/select_) all carry no done_marker, so they hit this
# exactly on their very first attempt.

def test_reap_leaves_a_crashed_no_marker_action_retryable(tmp_path):
    """No done_marker means the exit is unverifiable, not successful."""
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "honest_eval_2_1_seed7", "pid": 999999,
                         "started": 0.0, "timeout_h": 1.0,
                         "done_marker": None, "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "finished"
    assert "honest_eval_2_1_seed7" not in state["completed"], (
        "an unverifiable exit must not permanently retire the action; "
        "it must stay eligible for tick()'s attempt-cap retry")


def test_reap_leaves_a_verified_failure_retryable(tmp_path):
    """A verified failure gets to retry up to the cap, not zero retries."""
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "config_2_2", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "never.json",
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "failed"
    assert "config_2_2" not in state["completed"], (
        "a failed action must stay retryable, not retire on attempt one")


def test_reap_leaves_a_timed_out_action_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(ed.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(ed, "pid_alive", lambda pid: True)
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "resume_1_4_phase5", "pid": 4242,
                         "started": 0.0, "timeout_h": 0.001,
                         "done_marker": None, "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "timeout"
    assert "resume_1_4_phase5" not in state["completed"], (
        "a timeout-kill must stay retryable too, for the same reason")


def test_reap_still_retires_a_verified_success(tmp_path):
    """The fix must not weaken the case that DOES have a verdict."""
    marker = tmp_path / "done.json"
    marker.write_text("{}")
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "done.json",
                         "recurring": False}}
    ed.reap(state, tmp_path)
    assert state["completed"]["a"]["status"] == "succeeded"


def test_a_crash_can_be_retried_up_to_the_attempt_cap_before_abandoning(
        tmp_path, monkeypatch):
    """End-to-end: the retry/abandon logic in tick() is now reachable.

    Before the fix, the very first crash retired the action id in
    `completed`, so the second call below would see plan() return None
    (offer() excludes anything in `completed`) instead of relaunching,
    and the attempt cap's own abandon branch (tick(), MAX_ATTEMPTS_PER_
    ACTION) never ran at all.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "pid_alive", lambda pid: False)  # crashes fast
    action = ed.Action(id="flaky", kind="k", cmd=["scripts/x.py"],
                       needs_emulator=False, gate="g")
    monkeypatch.setattr(
        ed, "plan",
        lambda s, repo=ed.REPO, emulator_only=None: (
            None if "flaky" in s.get("completed", {}) else action))
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0}

    rec1 = ed.tick(state, tmp_path)
    assert rec1["decision"] == "launched"
    assert state["attempts"]["flaky"] == 1

    rec2 = ed.tick(state, tmp_path)      # reaps the crash, then relaunches
    assert "flaky" not in state["completed"], (
        "one crash must not have permanently retired the action")
    assert rec2["decision"] == "launched"
    assert state["attempts"]["flaky"] == 2

    rec3 = ed.tick(state, tmp_path)      # reaps the 2nd crash, hits the cap
    assert rec3["decision"] == "abandon"
    assert state["completed"]["flaky"]["status"] == "abandoned"


def test_reap_runs_before_anything_is_planned(monkeypatch):
    """Otherwise a finished action still looks live and blocks the tick."""
    calls = []
    monkeypatch.setattr(ed, "reap",
                        lambda s, repo=ed.REPO: calls.append("reap"))
    monkeypatch.setattr(ed, "emulator_busy",
                        lambda: (calls.append("busy"), None)[1])
    ed.tick({"completed": {}, "attempts": {}})
    assert calls[0] == "reap"


# ---- interrupted campaigns must self-heal ----

def _campaign_log(tmp_path, rows):
    d = tmp_path / "runs" / "online_2_1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "campaign.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_interrupted_campaign_reports_its_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    d = _campaign_log(tmp_path, [
        {"type": "campaign_start", "start_phase": 5},
        {"type": "phase_start", "phase": 5, "name": "consolidation"},
        {"type": "probe", "env_steps": 8.6e7, "median_max_x": 2559},
    ])
    assert ed.campaign_interrupted(d, tmp_path) == 5


def test_a_campaign_that_recorded_its_end_is_not_interrupted(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    d = _campaign_log(tmp_path, [
        {"type": "campaign_start"},
        {"type": "phase_start", "phase": 5},
        {"type": "campaign_complete"},
    ])
    assert ed.campaign_interrupted(d, tmp_path) is None


def test_a_live_campaign_is_never_called_interrupted(tmp_path, monkeypatch):
    """Liveness comes from ps; the OUTCOME still comes only from the log."""
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    d = _campaign_log(tmp_path, [
        {"type": "campaign_start"},
        {"type": "phase_start", "phase": 3},
    ])
    assert ed.campaign_interrupted(d, tmp_path) is None


def test_an_aborted_campaign_is_not_resumed_blindly(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    d = _campaign_log(tmp_path, [
        {"type": "campaign_start"},
        {"type": "phase_start", "phase": 3},
        {"type": "abort", "reason": "budget exhausted"},
    ])
    assert ed.campaign_interrupted(d, tmp_path) is None


def test_resume_carries_the_phase_rather_than_restarting_at_zero(
        tmp_path, monkeypatch):
    """Restarting at 0 re-litigates gates already earned.

    2-1 earned its deterministic rung gate 10/10 in phase 1; a resume
    that began again at phase 0 would put that back at risk for nothing.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    _campaign_log(tmp_path, [
        {"type": "campaign_start", "start_phase": 5},
        {"type": "phase_start", "phase": 5, "name": "consolidation"},
    ])
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "campaign_2_1.yaml").write_text("x")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "run_online_campaign.py").write_text(
        'import argparse\nap = argparse.ArgumentParser()\n'
        'ap.add_argument("--campaign-config")\n'
        'ap.add_argument("--start-phase")\n')

    acts = [a for a in [ed.plan({"completed": {}, "attempts": {}}, tmp_path)]
            if a is not None]
    resume = [a for a in acts if a.id.startswith("resume_")]
    if not resume:                       # suite_check may outrank it
        done = {a.id: {} for a in acts}
        nxt = ed.plan({"completed": done, "attempts": {}}, tmp_path)
        resume = [nxt] if nxt and nxt.id.startswith("resume_") else []
    assert resume, "an interrupted campaign was not offered a resume"
    a = resume[0]
    assert "--start-phase" in a.cmd
    assert a.cmd[a.cmd.index("--start-phase") + 1] == "5"


# ---- one action at a time, and maintenance never outranks the mission ----

def test_tick_will_not_launch_into_an_occupied_lane():
    """Fifteen suite checks were started in one session, four at once.

    The rule is one action per LANE, not one globally: a global slot
    fixed that pile-up and then left the engine idle for five and a half
    hours while a campaign held the machine.
    """
    state = {"completed": {}, "attempts": {}, "last_run": {},
             "running": {"token": {"id": "prev", "pid": os.getpid(),
                                   "started": time.time(), "timeout_h": 10.0,
                                   "done_marker": None, "recurring": True,
                                   "needs_emulator": False}}}
    ed.tick(state)
    assert all(a.needs_emulator for a in ed._test_launched), (
        "launched into the occupied token lane")


def test_both_lanes_occupied_means_wait():
    state = {"completed": {}, "attempts": {}, "last_run": {},
             "running": {
                 "token": {"id": "t", "pid": os.getpid(),
                           "started": time.time(), "timeout_h": 10.0,
                           "done_marker": None, "recurring": False,
                           "needs_emulator": False},
                 "emulator": {"id": "e", "pid": os.getpid(),
                              "started": time.time(), "timeout_h": 10.0,
                              "done_marker": None, "recurring": False,
                              "needs_emulator": True}}}
    rec = ed.tick(state)
    assert rec["decision"] == "wait"
    assert not ed._test_launched


def test_token_work_launches_while_the_emulator_lane_is_held(monkeypatch):
    """The capability the global slot destroyed."""
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    state = {"completed": {}, "attempts": {}, "last_run": {},
             "running": {"emulator": {"id": "campaign", "pid": os.getpid(),
                                      "started": time.time(),
                                      "timeout_h": 10.0, "done_marker": None,
                                      "recurring": False,
                                      "needs_emulator": True}}}
    ed.tick(state)
    assert ed._test_launched, "engine idled with a free token lane"
    assert all(not a.needs_emulator for a in ed._test_launched)


def test_legacy_single_slot_state_migrates_to_the_SAFE_lane():
    """A pre-lane record has no needs_emulator key.

    The live case was a running campaign: read permissively it lands in
    the token lane and the engine believes the machine is free.
    """
    legacy = {"running": {"id": "old", "pid": 1, "started": 0.0,
                          "timeout_h": 1.0, "done_marker": None,
                          "recurring": False}}
    slots = ed.running_slots(legacy)
    assert list(slots) == ["emulator"], slots
    assert slots["emulator"]["id"] == "old"


def test_an_explicit_token_record_stays_in_the_token_lane():
    rec = {"running": {"token": {"id": "t", "pid": 1, "started": 0.0,
                                 "timeout_h": 1.0, "done_marker": None,
                                 "recurring": True, "needs_emulator": False}}}
    assert list(ed.running_slots(rec)) == ["token"]


def test_recurring_action_is_suppressed_inside_its_cooldown():
    state = {"completed": {}, "attempts": {},
             "last_run": {"suite_check": time.time()}}
    a = ed.plan(state)
    assert a is None or a.id != "suite_check"


def test_recurring_action_returns_after_its_cooldown():
    old = time.time() - (ed.RECURRING_COOLDOWN_H + 1) * 3600
    state = {"completed": {}, "attempts": {}, "last_run": {"suite_check": old}}
    seen = set()
    a = ed.plan(state)
    while a is not None and a.id not in seen:
        seen.add(a.id)
        if a.id == "suite_check":
            break
        state.setdefault("completed", {})[a.id] = {"status": "ok"}
        a = ed.plan(state)
    assert "suite_check" in seen


def test_launching_records_a_last_run_stamp(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "plan", lambda s, repo=ed.REPO,
                        emulator_only=None: ed.Action(
                            id="r", kind="k", cmd=["scripts/x.py"],
                            needs_emulator=False, gate="g", recurring=True))
    state = {"completed": {}, "attempts": {}, "last_run": {}}
    ed.tick(state)
    assert state["last_run"]["r"] > 0


def test_mission_work_outranks_maintenance():
    """Housekeeping must never be chosen over the result the engine exists
    to produce."""
    a = ed.plan({"completed": {}, "attempts": {}, "last_run": {}})
    if a is not None and a.recurring:
        # only acceptable when there is genuinely no mission work left
        rest = ed.plan({"completed": {a.id: {}}, "attempts": {},
                        "last_run": {}})
        assert rest is None, (
            f"maintenance {a.id} was offered ahead of {rest.id}")


def test_a_campaign_is_not_offered_for_a_level_that_only_has_configs(tmp_path,
                                                                     monkeypatch):
    """A config is cheap and early; readiness is the ladder, dmap and anchor.

    config_2_2 ran and produced both configs for a level with none of its
    artifacts. Offering a campaign there aborts in preflight and burns an
    attempt for nothing.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "campaign_2_2.yaml").write_text("x")
    (tmp_path / "configs" / "mario_2_2_online_v1.yaml").write_text("x")
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "run_online_campaign.py").write_text(
        'import argparse\nap = argparse.ArgumentParser()\n'
        'ap.add_argument("--campaign-config")\nap.add_argument("--start-phase")\n')
    seen, state = set(), {"completed": {}, "attempts": {}, "last_run": {}}
    for _ in range(8):
        a = ed.plan(state, tmp_path)
        if a is None:
            break
        seen.add(a.id)
        state["completed"][a.id] = {"status": "ok"}
    assert "campaign_2_2" not in seen, seen


def test_validation_rejects_an_action_missing_a_required_flag(tmp_path):
    """The hole that silently skipped the hazard Phase-1 gate."""
    p = tmp_path / "scripts" / "s.py"
    p.parent.mkdir(parents=True)
    p.write_text('import argparse\n'
                 'ap = argparse.ArgumentParser()\n'
                 'ap.add_argument("--states", required=True)\n'
                 'ap.add_argument("--benchmark", action="store_true")\n')
    bad = _act(["scripts/s.py", "--benchmark"])
    ok, why = ed.validate_action(bad, tmp_path)
    assert not ok and "--states" in why and "requires" in why

    good = _act(["scripts/s.py", "--benchmark", "--states", "x"])
    assert ed.validate_action(good, tmp_path)[0]


def test_script_flags_reports_declared_and_required_separately(tmp_path):
    p = tmp_path / "s.py"
    p.write_text('import argparse\n'
                 'ap = argparse.ArgumentParser()\n'
                 'ap.add_argument("--a", required=True)\n'
                 'ap.add_argument("--b")\n')
    declared, required = ed.script_flags(p)
    assert declared == {"--a", "--b"} and required == {"--a"}


def test_script_flags_survives_an_unparseable_script(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("def (((")
    assert ed.script_flags(p) == (set(), set())


# validate_action checks that the action's input paths exist. Those two
# inputs are gitignored (.gitignore:10, 71), so on a clean clone the
# assertion below reports a missing dump instead of the validator verdict
# it exists to pin. Gate on the ladder DIRECTORY so a machine that has it
# still fails loudly when the states have gone.
@requires("roms/Super Mario Bros. (World).nes",
          "checkpoints/online_1_2/restart_states")
def test_the_real_hazard_action_now_validates():
    """Regression on the live planner, not a synthetic script."""
    ok, why = ed.validate_action(ed.Action(
        id="hazard_phase1", kind="benchmark", needs_emulator=True,
        cmd=["scripts/hazard_collect.py", "--benchmark",
             "--profile", "configs/mario_1_2_online_v2.yaml",
             "--rom", "roms/Super Mario Bros. (World).nes",
             "--states", "checkpoints/online_1_2/restart_states"],
        gate="g"))
    assert ok, why


@requires("roms/Super Mario Bros. (World).nes",
          "runs/ge_1_2_div_s1/solutions",
          "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum")
def test_the_real_hazard_collect_full_action_still_validates():
    """Regression on the second live hazard call site.

    Both --root-state and --out are supplied correctly here, so this
    must keep validating even once conditionally-required flags are
    enforced.
    """
    ok, why = ed.validate_action(ed.Action(
        id="hazard_collect_full", kind="collect", needs_emulator=True,
        cmd=["scripts/hazard_collect.py",
             "--profile", "configs/mario_1_2_online_v2.yaml",
             "--rom", "roms/Super Mario Bros. (World).nes",
             "--states", "runs/ge_1_2_div_s1/solutions/sol_000.actions.npy",
             "--root-state",
             "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state",
             "--forks-per-state", "120", "--out",
             "runs/engine/hazard_labels.npz"],
        gate="g"))
    assert ok, why


def _cond_script(tmp_path: Path) -> Path:
    """A script shaped exactly like hazard_collect.py's two conditional
    flags: --root-state (required when --states is a .npy tape) and
    --out (required unless --benchmark)."""
    p = tmp_path / "scripts" / "cond.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        'import argparse\n'
        'ap = argparse.ArgumentParser()\n'
        'ap.add_argument("--states", required=True)\n'
        'ap.add_argument("--root-state", default=None,\n'
        '                help="Required when --states is a .npy solution tape.")\n'
        'ap.add_argument("--out", default=None,\n'
        '                help="Output path. Required unless --benchmark.")\n'
        'ap.add_argument("--benchmark", action="store_true")\n')
    return p


def test_conditional_requirements_reads_when_suffix_and_unless(tmp_path):
    p = _cond_script(tmp_path)
    cond = ed.conditional_requirements(p)
    assert cond["--root-state"] == ("when_suffix", "states", "npy")
    assert cond["--out"] == ("unless", "benchmark", None)


def test_validation_rejects_npy_tape_missing_root_state(tmp_path):
    """The exact structural gap the audit flagged: a .npy --states tape
    passed without --root-state must not silently validate."""
    _cond_script(tmp_path)
    (tmp_path / "solve.npy").write_text("x")
    bad = _act(["scripts/cond.py", "--states", "solve.npy",
                "--benchmark"])
    ok, why = ed.validate_action(bad, tmp_path)
    assert not ok and "--root-state" in why

    good = _act(["scripts/cond.py", "--states", "solve.npy",
                 "--root-state", "solve.npy", "--benchmark"])
    assert ed.validate_action(good, tmp_path)[0]


def test_validation_does_not_require_root_state_for_a_directory_tape(tmp_path):
    """A directory of *.state files is not a .npy tape: --root-state
    stays optional, matching hazard_collect.py's own semantics."""
    _cond_script(tmp_path)
    states_dir = tmp_path / "states"
    states_dir.mkdir()
    ok, why = ed.validate_action(
        _act(["scripts/cond.py", "--states", "states", "--benchmark"]),
        tmp_path)
    assert ok, why


def test_validation_rejects_missing_out_without_benchmark(tmp_path):
    """The second half of the structural gap: --out is required unless
    --benchmark is set, and validate_action must enforce that too."""
    _cond_script(tmp_path)
    states_dir = tmp_path / "states"
    states_dir.mkdir()
    bad = _act(["scripts/cond.py", "--states", "states"])
    ok, why = ed.validate_action(bad, tmp_path)
    assert not ok and "--out" in why

    good = _act(["scripts/cond.py", "--states", "states", "--benchmark"])
    assert ed.validate_action(good, tmp_path)[0]

    also_good = _act(["scripts/cond.py", "--states", "states",
                      "--out", "out.npz"])
    assert ed.validate_action(also_good, tmp_path)[0]


def _interrupted(tmp_path, tag):
    d = tmp_path / "runs" / f"online_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "campaign.jsonl").write_text(
        json.dumps({"type": "campaign_start"}) + "\n"
        + json.dumps({"type": "phase_start", "phase": 5}) + "\n")
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / f"campaign_{tag}.yaml").write_text("x")
    sc = tmp_path / "scripts" / "run_online_campaign.py"
    sc.parent.mkdir(parents=True, exist_ok=True)
    sc.write_text('import argparse\nap = argparse.ArgumentParser()\n'
                  'ap.add_argument("--campaign-config")\n'
                  'ap.add_argument("--start-phase")\n')


def _scored(tmp_path, tag):
    d = tmp_path / "checkpoints" / f"mario_{tag}_online_v1"
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"game": f"mario_{tag}_online_v1", "eval_seed": s,
                        "n_episodes": 50, "sticky_prob": 0.25})
            for s in (7, 101)]
    (d / "eval.jsonl").write_text("\n".join(rows) + "\n")


def test_an_unscored_levels_interrupted_campaign_is_high_priority(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    _interrupted(tmp_path, "2_1")
    a = ed.plan({"completed": {}, "attempts": {}, "last_run": {}}, tmp_path)
    assert a is not None and a.id.startswith("resume_2_1"), a


def test_a_scored_levels_resume_ranks_below_other_work(tmp_path, monkeypatch):
    """1-4, banked at 51%, held the machine four hours on this rule."""
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    _interrupted(tmp_path, "1_4")
    _scored(tmp_path, "1_4")
    _interrupted(tmp_path, "2_1")          # unscored: must win
    a = ed.plan({"completed": {}, "attempts": {}, "last_run": {}}, tmp_path)
    assert a is not None and a.id.startswith("resume_2_1"), a


def test_a_scored_levels_resume_is_still_eventually_offered(tmp_path,
                                                            monkeypatch):
    """Deferred, not discarded — re-consolidation is a real experiment."""
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    _interrupted(tmp_path, "1_4")
    _scored(tmp_path, "1_4")
    state = {"completed": {}, "attempts": {}, "last_run": {}}
    seen = set()
    for _ in range(6):
        a = ed.plan(state, tmp_path)
        if a is None:
            break
        seen.add(a.id)
        state["completed"][a.id] = {"status": "ok"}
    assert any(i.startswith("resume_1_4") for i in seen), seen


def test_a_misfiled_lane_is_corrected_on_read():
    """Trusting the stored key made one bad classification permanent."""
    bad = {"running": {"token": {"id": "campaign", "pid": 1, "started": 0.0,
                                 "timeout_h": 1.0, "done_marker": None,
                                 "recurring": False,
                                 "needs_emulator": True}}}
    slots = ed.running_slots(bad)
    assert list(slots) == ["emulator"], slots


# ---- benchmarks may only run on a quiet machine ----

def test_a_benchmark_is_deferred_on_a_loaded_machine(monkeypatch):
    """The false KILL: 100.1 steps/s twelve minutes after a 13h campaign,
    versus 2318.6 on a settled machine. A 23x difference from timing
    alone, on a verdict that would have abandoned the research."""
    monkeypatch.setattr(ed, "load_average", lambda: ed.QUIET_LOAD_MAX + 5)
    ok, why = ed.machine_quiet({})
    assert not ok and "load" in why


def test_a_benchmark_is_deferred_until_the_machine_has_settled(monkeypatch):
    monkeypatch.setattr(ed, "load_average", lambda: 0.1)
    ok, why = ed.machine_quiet({"last_heavy_finish": time.time()})
    assert not ok and "since the last heavy job" in why


def test_a_quiet_settled_machine_admits_a_benchmark(monkeypatch):
    monkeypatch.setattr(ed, "load_average", lambda: 0.5)
    old = time.time() - (ed.QUIET_SETTLE_S + 60)
    ok, why = ed.machine_quiet({"last_heavy_finish": old})
    assert ok, why


def test_an_unreadable_load_average_is_not_quiet(monkeypatch):
    def boom():
        raise OSError("no loadavg")
    monkeypatch.setattr(ed.os, "getloadavg", boom)
    assert ed.load_average() == float("inf")
    assert not ed.machine_quiet({})[0]


def test_finishing_an_emulator_job_stamps_the_settle_clock(tmp_path):
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"emulator": {
                 "id": "campaign", "pid": 999999, "started": 0.0,
                 "timeout_h": 1.0, "done_marker": None, "recurring": False,
                 "needs_emulator": True}}}
    ed.reap(state, tmp_path)
    assert state.get("last_heavy_finish", 0) > 0


def test_a_timed_out_heavy_job_also_stamps_the_settle_clock(tmp_path,
                                                             monkeypatch):
    """Regression: the timeout-kill path used to skip this stamp entirely.

    A 14-hour needs_emulator campaign that hangs and gets SIGKILLed by
    the timeout branch loaded the machine exactly as much as one that
    exits cleanly. Without this stamp, machine_quiet() has no record of
    the load and can wave a benchmark through seconds after the kill —
    reproducing the false-negative (100.1 vs 2,318.6 steps/s) the settle
    window exists to prevent.
    """
    monkeypatch.setattr(ed.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(ed, "pid_alive", lambda pid: True)
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"emulator": {
                 "id": "campaign_2_2", "pid": 4242, "started": 0.0,
                 "timeout_h": 0.001, "done_marker": None, "recurring": False,
                 "needs_emulator": True}}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "timeout"
    assert state.get("last_heavy_finish", 0) > 0, (
        "a timeout-killed heavy job must arm the settle window too")


def test_a_quiet_requiring_action_is_skipped_not_failed(monkeypatch):
    """Deferred means it comes back, not that it burns an attempt."""
    monkeypatch.setattr(ed, "load_average", lambda: ed.QUIET_LOAD_MAX + 5)
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    state = {"completed": {}, "attempts": {}, "last_run": {}}
    ed.plan(state)
    assert state.get("attempts", {}).get("hazard_phase1", 0) == 0


# ---- an eval is stale once the policy changes under it ----

def _scored_at(tmp_path, tag, ts):
    d = tmp_path / "checkpoints" / f"mario_{tag}_online_v1"
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"game": f"mario_{tag}_online_v1", "eval_seed": s,
                        "n_episodes": 50, "sticky_prob": 0.25,
                        "timestamp": ts}) for s in (7, 101)]
    (d / "eval.jsonl").write_text("\n".join(rows) + "\n")
    return d


def test_an_eval_older_than_the_checkpoint_is_stale(tmp_path):
    d = _scored_at(tmp_path, "9_9", time.time() - 10_000)
    ck = d / "vanilla_ppo_iter_00100.pt"
    ck.write_text("x")
    assert ed.honest_eval_done("9-9", tmp_path)
    assert not ed.honest_eval_current("9-9", tmp_path)


def test_an_eval_newer_than_the_checkpoint_is_current(tmp_path):
    d = _scored_at(tmp_path, "9_9", time.time() + 10_000)
    (d / "vanilla_ppo_iter_00100.pt").write_text("x")
    assert ed.honest_eval_current("9-9", tmp_path)


def test_probe_rows_do_not_make_a_stale_eval_look_current(tmp_path):
    """eval_game appends every probe to the same file, so the file's
    mtime refreshes constantly and cannot be the staleness signal."""
    d = _scored_at(tmp_path, "9_9", time.time() - 10_000)
    (d / "vanilla_ppo_iter_00100.pt").write_text("x")
    with open(d / "eval.jsonl", "a") as f:      # a fresh 30-episode probe
        f.write(json.dumps({"game": "mario_9_9_online_v1", "eval_seed": 1,
                            "n_episodes": 30, "sticky_prob": 0.25,
                            "timestamp": time.time() + 50_000}) + "\n")
    assert not ed.honest_eval_current("9-9", tmp_path)


def test_a_glob_argument_is_not_checked_as_a_path(tmp_path):
    """replay_sweep_full was permanently un-runnable: a glob never exists
    as a file, so the input-path check rejected it every tick."""
    p = tmp_path / "scripts" / "s.py"
    p.parent.mkdir(parents=True)
    p.write_text('import argparse\nap = argparse.ArgumentParser()\n'
                 'ap.add_argument("--glob")\n')
    a = _act(["scripts/s.py", "--glob", "runs/**/solutions/*.json"])
    ok, why = ed.validate_action(a, tmp_path)
    assert ok, why


# ---- plan()'s decision table, pinned end to end (audit: legibility) ----

def _park(tmp_path, tag):
    """An in-progress campaign dir with no log: invisible to tiers 1, 3b
    and 4 alike, so it cannot become anyone's candidate at any step."""
    (tmp_path / "runs" / f"online_{tag}").mkdir(parents=True, exist_ok=True)


def _put(tmp_path, rel, text="x"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_plan_priority_order_is_pinned_to_the_documented_table(tmp_path,
                                                                monkeypatch):
    """Walks every tier of plan()'s decision-table docstring, in order.

    One synthesized state makes all nine tiers simultaneously runnable.
    The test then crosses tiers off from the top exactly as `offer()`
    would -- by completing an id, or by dropping the artifact a chained
    sub-step needs -- and asserts the next winner is the next tier down.
    If plan()'s real precedence ever stops matching the table at the top
    of the function, THIS is the test that fails, not a human noticing
    the docstring quietly went stale.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "load_average", lambda: 0.1)   # tier 2: quiet

    rom = "roms/Super Mario Bros. (World).nes"
    _put(tmp_path, rom)

    # Every SMB level not deliberately wired into a tier below is
    # parked, so it cannot contaminate which candidate wins at any step.
    for level in ed.SMB_LEVELS:
        if level not in ("1-1", "1-2", "1-4", "2-1"):
            _park(tmp_path, ed.tag_of(level))

    # eval_game.py is the script BOTH tier 1's honest-eval and tier 2c's
    # shelf dispositions launch -- one stub, declared once, covers both.
    _script(tmp_path, "eval_game.py",
            ["--game", "--profile", "--rom", "--checkpoint", "--episodes",
             "--max-steps", "--sequential", "--level-clear", "--start-state",
             "--eval-seed", "--sticky-prob", "--start-jitter",
             "--eval-workers", "--eval-rng"])

    # --- Tier 1: 1-1's campaign ended and has never been scored. ---
    (tmp_path / "runs" / "online_1_1").mkdir(parents=True)
    (tmp_path / "runs" / "online_1_1" / "campaign.jsonl").write_text(
        json.dumps({"type": "campaign_start"}) + "\n"
        + json.dumps({"type": "campaign_complete"}) + "\n")
    _put(tmp_path, "configs/mario_1_1_online_v1.yaml",
         f'rom_path: "{rom}"\n'
         'start_state_path: "checkpoints/start_states/1_1_entrance.state"\n')
    _put(tmp_path, "checkpoints/start_states/1_1_entrance.state")
    _put(tmp_path, "checkpoints/mario_1_1_online_v1/vanilla_ppo_iter_00000.pt")

    # --- Tier 2: hazard_phase1, plus everything 2b's chain needs once
    #     its own predecessor artifact exists. ---
    _script(tmp_path, "hazard_collect.py",
            ["--benchmark", "--profile", "--rom", "--states",
             "--root-state", "--forks-per-state", "--out"])
    _put(tmp_path, "configs/mario_1_2_online_v2.yaml", f'rom_path: "{rom}"\n')
    (tmp_path / "checkpoints" / "online_1_2" / "restart_states").mkdir(
        parents=True)
    _put(tmp_path, "runs/engine/logs/hazard_phase1.log",
         "... benchmark ...\nGATE: PASS\n")
    _put(tmp_path, "runs/ge_1_2_div_s1/solutions/sol_000.actions.npy")
    _put(tmp_path,
         "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/"
         "stage_03.state")
    _script(tmp_path, "train_hazard.py", ["--data", "--out", "--gate"])

    # --- Tier 2c: one shelf disposition (shelf_joint_1_1); the other
    #     two self-skip on a missing profile, which is itself part of
    #     what this tier is for and is covered by
    #     test_shelf_dispositions_missing_files_are_journaled_not_silent. ---
    _put(tmp_path, "configs/mario_1_1_backward.yaml", f'rom_path: "{rom}"\n')
    _put(tmp_path, "runs/interference/joint.pt")
    _put(tmp_path, "runs/live_show/smb_4_4_micro/entrance_start.state")

    # --- Tier 3: replay_sweep_full. ---
    _script(tmp_path, "replay_sweep.py", ["--glob", "--out"])

    # --- Tier 3b: 2-1's campaign is interrupted and never scored. ---
    _interrupted(tmp_path, "2_1")

    # --- Tier 4: 1-2 has no config yet -- the first level in
    #     SMB_LEVELS order once 1-1, 1-4 and 2-1 are excluded. ---
    _script(tmp_path, "make_campaign_config.py", ["--level"])

    # --- Tier 4b: 1-4's campaign is interrupted, but ALREADY scored. ---
    _interrupted(tmp_path, "1_4")
    _scored(tmp_path, "1_4")

    # --- Tier 5: suite_check. ---
    _script(tmp_path, "run_suite_check.py", ["--out", "--timeout"])

    state = {"completed": {}, "attempts": {}, "last_run": {}}

    def step(expect_prefix):
        a = ed.plan(state, tmp_path)
        assert a is not None and a.id.startswith(expect_prefix), \
            f"expected an id starting {expect_prefix!r}, got {a}"
        state["completed"][a.id] = {"status": "ok"}
        return a

    step("honest_eval_1_1_seed7")
    step("honest_eval_1_1_seed101")
    step("hazard_phase1")
    step("hazard_collect_full")
    _put(tmp_path, "runs/engine/hazard_labels.npz")          # 2b's output
    step("hazard_phase2_train")
    _put(tmp_path, "runs/engine/hazard/hazard_report.json")  # closes 2b
    step("shelf_joint_1_1_seed7")
    step("shelf_joint_1_1_seed101")
    step("replay_sweep_full")
    step("resume_2_1_phase5")
    step("config_1_2")
    step("resume_1_4_phase5")
    step("suite_check")

    # The fixture is exhaustive, not just the cascade: silence
    # suite_check's own recurring cooldown and confirm nothing else was
    # left standing underneath it.
    state["last_run"]["suite_check"] = time.time()
    assert ed.plan(state, tmp_path) is None


# ---------------------------------------------------------------------------
# Blocked-engine push notification + the --halt kill switch (2026-08-29
# audit: the breaker "halts silently ... in the one subsystem whose own
# docstring says the point is surviving three weeks with nobody
# watching"; state["halted"] was checked but never set anywhere).
# ---------------------------------------------------------------------------


def test_breaker_trip_notifies_once_then_latches(monkeypatch):
    calls: list = []
    monkeypatch.setattr(ed, "_notify", lambda t, m: calls.append((t, m)))
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    state = {"completed": {}, "attempts": {},
             "consecutive_failures": ed.CONSECUTIVE_FAILURE_LIMIT}
    rec1 = ed.tick(state)
    rec2 = ed.tick(state)
    assert rec1["decision"] == rec2["decision"] == "blocked"
    assert len(calls) == 1, (
        "one banner per blocked episode — a 10-minute tick loop must "
        "not push a notification every tick")
    assert "circuit breaker" in calls[0][1]


def test_notification_latch_rearms_when_the_guard_clears(monkeypatch):
    calls: list = []
    monkeypatch.setattr(ed, "_notify", lambda t, m: calls.append(t))
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    monkeypatch.setattr(ed, "plan",
                        lambda s, repo=ed.REPO, **kw: None)
    state = {"completed": {}, "attempts": {},
             "consecutive_failures": ed.CONSECUTIVE_FAILURE_LIMIT}
    ed.tick(state)                       # trips, notifies
    state["consecutive_failures"] = 0
    ed.tick(state)                       # unblocked tick clears the latch
    assert "blocked_notified" not in state
    state["consecutive_failures"] = ed.CONSECUTIVE_FAILURE_LIMIT
    ed.tick(state)                       # a NEW episode notifies again
    assert len(calls) == 2


def test_notify_failure_never_breaks_a_tick(monkeypatch):
    """_notify's guard must swallow ANY failure in the notification
    path — a broken notifications module can never fail a tick. The
    module is stubbed with one whose notify_macos raises (also keeps
    real osascript banners out of test runs)."""
    import sys as _sys
    import types as _types
    broken = _types.ModuleType("src.training.notifications")
    def _boom(*a, **kw):
        raise RuntimeError("osascript exploded")
    broken.notify_macos = _boom
    monkeypatch.setitem(_sys.modules, "src.training.notifications", broken)
    ed._notify("t", "m")   # must not raise
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    state = {"completed": {}, "attempts": {},
             "consecutive_failures": ed.CONSECUTIVE_FAILURE_LIMIT}
    rec = ed.tick(state)
    assert rec["decision"] == "blocked"


def test_halt_cli_sets_state_and_blocks_ticks(monkeypatch, capsys):
    assert ed.main(["--halt", "operator stop: verifying disk"]) == 0
    state = ed.load_state()
    assert state.get("halted") == "operator stop: verifying disk"
    monkeypatch.setattr(ed, "emulator_busy", lambda: None)
    rec = ed.tick(state)
    assert rec["decision"] == "blocked" and "operator stop" in rec["reason"]


def test_clear_halt_cli_unsets_and_rearms_latch(capsys):
    assert ed.main(["--halt", "x"]) == 0
    assert ed.main(["--clear-halt"]) == 0
    state = ed.load_state()
    assert not state.get("halted")
    assert "blocked_notified" not in state


def test_halt_and_clear_are_journaled(capsys):
    ed.main(["--halt", "why"])
    ed.main(["--clear-halt"])
    rows = [json.loads(l) for l in
            ed.JOURNAL.read_text().splitlines()]
    kinds = [r.get("type") for r in rows]
    assert "halt" in kinds and "halt_cleared" in kinds
