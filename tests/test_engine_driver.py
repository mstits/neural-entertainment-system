"""scripts/engine_driver.py — the unattended engine's safety properties.

The bar is three weeks alone. Every test here pins a behaviour whose
absence would turn that into three weeks of nothing: emitting a command
that cannot run, launching a second emulator job onto a busy machine,
retrying a poisoned action forever, or halting on a corrupt state file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.engine_driver as ed  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "ENGINE_DIR", tmp_path / "engine")
    monkeypatch.setattr(ed, "JOURNAL", tmp_path / "engine" / "journal.jsonl")
    monkeypatch.setattr(ed, "STATE_PATH", tmp_path / "engine" / "state.json")


def _script(tmp_path: Path, name: str, flags: list[str]) -> Path:
    p = tmp_path / "scripts" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'    ap.add_argument("{f}")' for f in flags)
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
    assert ed.script_flags(p) == {"--alpha", "--beta"}


# ---- the physics budget ----

def test_never_launches_an_emulator_action_while_one_is_live(monkeypatch):
    """The physics budget: one emulator job, always.

    The engine may still do token-bound work beside a campaign — that is
    the brief's own scheduling rule and most of why the loop felt stalled
    — but whatever it picks must not need the emulator.
    """
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    launched = []
    monkeypatch.setattr(ed, "launch", lambda a, repo=ed.REPO:
                        (launched.append(a), 1)[1])
    rec = ed.tick({"completed": {}, "attempts": {}})
    assert all(not a.needs_emulator for a in launched), launched
    if rec["decision"] == "wait":
        assert "4242" in rec["reason"]


def test_waits_when_busy_and_no_token_bound_work_remains(monkeypatch):
    monkeypatch.setattr(ed, "emulator_busy", lambda: 4242)
    monkeypatch.setattr(ed, "plan",
                        lambda s, repo=ed.REPO, emulator_only=None:
                        _act(["scripts/x.py"]) if emulator_only is None
                        else None)
    rec = ed.tick({"completed": {}, "attempts": {}})
    assert rec["decision"] == "wait" and "token-bound" in rec["reason"]


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


def test_plan_does_not_repeat_a_completed_action():
    a = ed.plan({"completed": {}, "attempts": {}})
    assert a is not None
    b = ed.plan({"completed": {a.id: {"status": "ok"}}, "attempts": {}})
    assert b is None or b.id != a.id


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
