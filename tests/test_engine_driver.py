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
                  'ap.add_argument("--out")\n')

    first = ed.plan({"completed": {}, "attempts": {}, "last_run": {}},
                    tmp_path)
    assert first is not None and first.id == "suite_check", first
    assert first.recurring

    again = ed.plan({"completed": {"suite_check": {"status": "succeeded"}},
                     "attempts": {}, "last_run": {}}, tmp_path)
    assert again is not None and again.id == "suite_check", (
        "a completed recurring action was not re-offered")


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
    killed = []
    monkeypatch.setattr(ed.os, "kill",
                        lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(ed, "pid_alive", lambda pid: True)
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "a", "pid": 4242, "started": 0.0,
                         "timeout_h": 0.001, "done_marker": None,
                         "recurring": False}}
    rec = ed.reap(state, tmp_path)
    assert rec["outcome"] == "timeout"
    assert (4242, 9) in killed
    assert state["consecutive_failures"] == 1


def test_a_recurring_action_is_never_recorded_as_completed(tmp_path):
    marker = tmp_path / "d.json"
    marker.write_text("{}")
    state = {"completed": {}, "attempts": {}, "consecutive_failures": 0,
             "running": {"id": "suite", "pid": 999999, "started": 0.0,
                         "timeout_h": 1.0, "done_marker": "d.json",
                         "recurring": True}}
    ed.reap(state, tmp_path)
    assert "suite" not in state["completed"]


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
