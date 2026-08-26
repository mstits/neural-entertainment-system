"""scripts/replay_sweep.py — banked tapes must be re-verifiable.

Covers the pre-registered gate's semantics (ERROR is a failure, not a
skip; a third failure beyond the two quarantined tapes is a new finding),
the tick-aligned batch replay with its verdict freeze, and the
consumer-index provenance recovery including the basename-collision bug
that made its first version confidently wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_sweep import (  # noqa: E402
    KNOWN_BAD, TapeSpec, Verdict, build_consumer_index, build_report,
    discover_tapes, evaluate_gate, read_tape, resolve_hw_flags,
    clear_predicate, resolve_profile, resolve_from_siblings,
    resolve_start_key, sibling_profiles, spec_problems,
    verify_ram_trace,
)


def _spec(name="t.json"):
    return TapeSpec(tape=name, actions="a.npy", root_state="r.state",
                    profile="configs/p.yaml", start_wd=[0, 0],
                    clear_wd=[0, 1], steps=3, core_sha16="abc")


def test_trace_that_reaches_a_clear_passes():
    v = verify_ram_trace([{"c": 0}, {"c": 0}, {"c": 1}], _spec(),
                         lambda r: bool(r["c"]))
    assert v.status == "PASS" and v.replayed_steps == 2


def test_trace_that_never_clears_fails():
    v = verify_ram_trace([{"c": 0}] * 4, _spec(), lambda r: bool(r["c"]))
    assert v.status == "FAIL" and "never satisfied" in v.reason


def test_verdict_is_taken_at_the_first_clearing_frame():
    """A tape can clear and then step back out; scoring the last frame
    would report a false negative on a good tape."""
    trace = [{"c": 0}, {"c": 1}, {"c": 0}, {"c": 0}]
    v = verify_ram_trace(trace, _spec(), lambda r: bool(r["c"]))
    assert v.status == "PASS" and v.replayed_steps == 1


def test_empty_trace_fails_rather_than_raising():
    v = verify_ram_trace([], _spec(), lambda r: True)
    assert v.status == "FAIL"


def test_gate_passes_when_only_known_bad_tapes_fail():
    vs = [Verdict(KNOWN_BAD[0], "FAIL", "x"), Verdict("good.json", "PASS", "y")]
    ok, msg = evaluate_gate(vs)
    assert ok
    assert "1 known-bad" in msg and "0 NEW" in msg


def test_gate_fails_on_a_third_failure():
    vs = [Verdict(KNOWN_BAD[0], "FAIL", "x"),
          Verdict("fresh.json", "FAIL", "y")]
    ok, msg = evaluate_gate(vs)
    assert not ok and "fresh.json" in msg


def test_error_counts_as_failure_never_as_skip():
    """'Cannot be checked' and 'passes' are different states."""
    ok, msg = evaluate_gate([Verdict("x.json", "ERROR", "no root")])
    assert not ok


def test_report_records_gate_and_counts():
    r = build_report([Verdict("a", "PASS", "x"), Verdict("b", "FAIL", "y")],
                     stamp="S", core_sha16="deadbeef")
    assert r["stamp"] == "S" and r["replaying_core_sha16"] == "deadbeef"
    assert r["counts"] == {"PASS": 1, "FAIL": 1}
    assert r["gate_passed"] is False
    assert r["known_bad"] == list(KNOWN_BAD)
    json.dumps(r)  # must serialize


def test_read_tape_extracts_core_sha16_from_hw_provenance(tmp_path):
    """Tapes carry binary provenance under "hw_provenance", never "hw"
    (checked: 0/355 banked tapes have "hw", 148 have "hw_provenance"
    populated). Reading the wrong key silently drops core_sha16 to None
    on every tape, which is the whole binary-provenance check gone dead."""
    p = tmp_path / "sol_000.json"
    p.write_text(json.dumps({
        "hw_provenance": {"nes_core": {"sha256_16": "0320f3be9080b9f8"}},
    }))
    spec = read_tape(p, root=tmp_path)
    assert spec.core_sha16 == "0320f3be9080b9f8"


def test_resolve_hw_flags_reads_the_tapes_own_recorded_lineage(tmp_path):
    """A tape's own hw_provenance.hw_flags must win over the profile's
    defaults even when its root has no .state.json sidecar to fall back
    on (checked: 0/355 banked tapes have a top-level "hw" key, 148 have
    "hw_provenance"). Reading "hw" instead of "hw_provenance" silently
    drops this to the default on every tape — the exact false-FAIL bug
    that hit three Castlevania tapes."""
    root_state = tmp_path / "checkpoints" / "root.state"
    root_state.parent.mkdir(parents=True)
    root_state.write_bytes(b"")
    tape = tmp_path / "sol_000.json"
    tape.write_text(json.dumps({
        "hw_provenance": {"hw_flags": [
            "reset_alignment", "mmio_read_timing",
            "dmc_stall_timing", "nmi_poll_timing"]},
    }))
    spec = TapeSpec(tape="sol_000.json", actions="a.npy",
                    root_state=str(root_state.relative_to(tmp_path)),
                    profile="configs/p.yaml", start_wd=[0, 0],
                    clear_wd=[0, 1], steps=3, core_sha16=None)
    got = resolve_hw_flags(spec, default_hw_flags=(), root=tmp_path)
    assert got == ("reset_alignment", "mmio_read_timing",
                   "dmc_stall_timing", "nmi_poll_timing")


def test_resolve_hw_flags_falls_back_to_sidecar_then_default(tmp_path):
    root_state = tmp_path / "checkpoints" / "root.state"
    root_state.parent.mkdir(parents=True)
    root_state.write_bytes(b"")
    tape = tmp_path / "sol_000.json"
    tape.write_text(json.dumps({}))
    spec = TapeSpec(tape="sol_000.json", actions="a.npy",
                    root_state=str(root_state.relative_to(tmp_path)),
                    profile="configs/p.yaml", start_wd=[0, 0],
                    clear_wd=[0, 1], steps=3, core_sha16=None)
    assert resolve_hw_flags(spec, default_hw_flags=("default",),
                            root=tmp_path) == ("default",)

    (tmp_path / (str(root_state.relative_to(tmp_path)) + ".json")
     ).write_text(json.dumps({"hw_flags": ["from_sidecar"]}))
    assert resolve_hw_flags(spec, default_hw_flags=("default",),
                            root=tmp_path) == ("from_sidecar",)


def test_spec_problems_names_each_missing_input(tmp_path):
    s = TapeSpec("t", "a.npy", "", "configs/none.yaml", [0, 0], None, 1, None)
    probs = spec_problems(s, root=tmp_path)
    joined = " ".join(probs)
    assert "root_state not recorded" in joined
    assert "actions missing" in joined
    assert "profile missing" in joined
    assert "clear_wd not recorded" in joined


def test_discover_excludes_quarantined_paths(tmp_path):
    for rel in ("runs/good/solutions/sol_000.json",
                "runs/x_INVALID_y/solutions/sol_000.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    got = [str(p) for p in discover_tapes("runs/**/solutions/*.json", tmp_path)]
    assert any("good" in g for g in got)
    assert not any("INVALID" in g for g in got)


def test_consumer_index_keys_on_resolved_path_not_basename(tmp_path):
    """The bug that made the first index confidently wrong.

    Two levels both name their tape sol_000.actions.npy. A basename index
    maps one level's tape to the other level's profile.
    """
    for lvl, prof in (("l13", "configs/a.yaml"), ("l21", "configs/b.yaml")):
        d = tmp_path / "checkpoints" / lvl
        d.mkdir(parents=True)
        (tmp_path / prof).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / prof).write_text("x")
        act = tmp_path / "runs" / lvl / "solutions" / "sol_000.actions.npy"
        act.parent.mkdir(parents=True)
        act.write_text("")
        (d / "manifest.json").write_text(json.dumps(
            {"source_solution": {"actions": str(act), "profile": prof}}))
    idx = build_consumer_index(tmp_path)
    assert len(idx) == 2, "basename collision collapsed two distinct tapes"
    profs = {v[0] for v in idx.values()}
    assert profs == {"configs/a.yaml", "configs/b.yaml"}


def test_resolve_profile_marks_recorded_vs_recovered(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/real.yaml").write_text("x")
    act = tmp_path / "runs/s/solutions/sol_000.actions.npy"
    act.parent.mkdir(parents=True)
    act.write_text("")

    recorded = TapeSpec("t", "runs/s/solutions/sol_000.actions.npy", "r",
                        "configs/real.yaml", None, None, None, None)
    out = resolve_profile(recorded, {}, tmp_path)
    assert out.profile_source == "recorded"

    missing = TapeSpec("t", "runs/s/solutions/sol_000.actions.npy", "r",
                       "", None, None, None, None)
    idx = {act.resolve(): ("configs/real.yaml", "m.json")}
    out = resolve_profile(missing, idx, tmp_path)
    assert out.profile == "configs/real.yaml"
    assert out.profile_source.startswith("recovered from")


def test_recovery_does_not_invent_a_profile_that_is_absent(tmp_path):
    act = tmp_path / "a.npy"
    act.write_text("")
    spec = TapeSpec("t", "a.npy", "r", "", None, None, None, None)
    idx = {act.resolve(): ("configs/gone.yaml", "m.json")}
    out = resolve_profile(spec, idx, tmp_path)
    assert out.profile == "" and out.profile_source == ""


def test_sibling_recovery_fills_a_gap_in_a_resolved_directory(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/x.yaml").write_text("x")
    resolved = TapeSpec("runs/s/solutions/sol_000.json", "a.npy", "r",
                        "configs/x.yaml", None, None, None, None,
                        profile_source="recorded")
    gap = TapeSpec("runs/s/solutions/sol_001.json", "b.npy", "r",
                   "", None, None, None, None)
    sibs = sibling_profiles([resolved, gap])
    out = resolve_from_siblings(gap, sibs, tmp_path)
    assert out.profile == "configs/x.yaml"
    assert "sibling" in out.profile_source


def test_sibling_recovery_refuses_an_ambiguous_directory(tmp_path):
    """Two different profiles in one solve dir is not evidence."""
    a = TapeSpec("runs/s/solutions/sol_000.json", "a", "r", "configs/a.yaml",
                 None, None, None, None, profile_source="recorded")
    b = TapeSpec("runs/s/solutions/sol_001.json", "b", "r", "configs/b.yaml",
                 None, None, None, None, profile_source="recorded")
    gap = TapeSpec("runs/s/solutions/sol_002.json", "c", "r", "",
                   None, None, None, None)
    assert sibling_profiles([a, b, gap]) == {}
    out = resolve_from_siblings(gap, {}, tmp_path)
    assert out.profile == "" and out.profile_source == ""


def test_sibling_recovery_does_not_cross_directories(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/x.yaml").write_text("x")
    other = TapeSpec("runs/OTHER/solutions/sol_000.json", "a", "r",
                     "configs/x.yaml", None, None, None, None,
                     profile_source="recorded")
    gap = TapeSpec("runs/MINE/solutions/sol_000.json", "b", "r", "",
                   None, None, None, None)
    out = resolve_from_siblings(gap, sibling_profiles([other, gap]), tmp_path)
    assert out.profile == ""


def test_a_root_that_already_clears_is_unscorable_not_a_pass():
    """111 of 171 passes were 'cleared at frame 0' — nothing verified."""
    v = verify_ram_trace([{"c": 1}, {"c": 1}], _spec(), lambda r: bool(r["c"]))
    assert v.status == "UNSCORABLE"
    assert "before the tape acts" in v.reason


def test_a_clear_after_the_root_still_passes():
    v = verify_ram_trace([{"c": 0}, {"c": 1}], _spec(), lambda r: bool(r["c"]))
    assert v.status == "PASS" and v.replayed_steps == 1


def test_unscorable_is_neither_a_pass_nor_a_gate_failure():
    ok, msg = evaluate_gate([Verdict("a", "UNSCORABLE", "root"),
                             Verdict("b", "PASS", "ok")])
    assert ok, msg
    assert "1 UNSCORABLE" in msg
    assert msg.startswith("1/2"), msg


# --------------------------------------------------------------------------
# start_wd: [] is a RECORDED key, not a missing one (2026-08-26)
# --------------------------------------------------------------------------

def test_an_empty_recorded_start_key_is_not_treated_as_missing():
    """`start_wd: []` is what every `level_key: []` profile banks — 40 of
    the 45 solve profiles. A truthiness test collapsed it onto None and the
    sweep reported `ERROR: start_wd not recorded` for a tape that had
    recorded it, which evaluate_gate then counted as a failure."""
    spec = TapeSpec(tape="t.json", actions="a.npy", root_state="r.state",
                    profile="configs/p.yaml", start_wd=[], clear_wd=[],
                    steps=3, core_sha16="abc")
    assert resolve_start_key(spec) == ()


def test_a_genuinely_absent_start_key_is_still_None():
    """The mirror: the fix must not make "not recorded" unreportable."""
    spec = TapeSpec(tape="t.json", actions="a.npy", root_state="r.state",
                    profile="configs/p.yaml", start_wd=None, clear_wd=None,
                    steps=3, core_sha16="abc")
    assert resolve_start_key(spec) is None


def test_a_populated_start_key_round_trips():
    assert resolve_start_key(_spec()) == (0, 0)


def test_an_empty_key_tape_reaches_the_unscorable_handler():
    """The consequence of the bug, tested end-to-end through the verifier:
    with start_wd == clear_wd == [] the correct verdict is UNSCORABLE
    ("banked under a predicate this verifier does not implement"), which
    the ERROR branch used to pre-empt by `continue`-ing first."""
    spec = TapeSpec(tape="t.json", actions="a.npy", root_state="r.state",
                    profile="configs/p.yaml", start_wd=[], clear_wd=[],
                    steps=3, core_sha16="abc")
    v = verify_ram_trace([{"c": 0}, {"c": 0}], spec, lambda r: bool(r["c"]))
    assert v.status == "UNSCORABLE", v
    assert "does not implement" in v.reason


def test_the_real_detector_gate_tapes_carry_an_empty_recorded_key():
    """Anchors the two tests above to the artifacts that reproduce it, so
    this cannot become a synthetic-only regression test."""
    found = 0
    for name in ("kirby", "double_dragon"):
        p = (ROOT / "runs" / "detector_gate_20260810" / name /
             "solutions" / "sol_000.json")
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        assert meta.get("start_wd") == [], p
        assert not meta["start_wd"], "must be falsy — that was the trap"
        assert resolve_start_key(
            TapeSpec("t", "a", "r", "p", meta["start_wd"],
                     meta.get("clear_wd"), None, None)) == ()
        found += 1
    if not found:
        pytest.skip("detector_gate_20260810 tapes not present (runs/ is "
                    "gitignored); the synthetic cases above still cover it")


# --- the win test: is_clear OR is_finale -----------------------------------
# `GenericGame` reaches a win two ways. `is_clear` opens with
# `level_key(ram) > tuple(start_key)`; `is_finale` opens with
# `tuple(start_key) == tuple(f["level_key"])`. On the empty key that 40 of
# 45 solve profiles ship, the first is `() > ()` (False, always) and the
# second is `() == ()` (True), leaving a live byte test. Asking only the
# first scores a finale-hooked tape `FAIL: never satisfied is_clear`.


class _FinaleOnlyGame:
    """A profile whose only reachable hook is `finale:` — Excitebike's
    shape. `is_clear` is the `() > ()` identity and never fires."""
    def is_clear(self, start_key, ram):
        return tuple() > tuple(start_key)

    def is_finale(self, start_key, ram):
        return bool(ram["ready"] == 2)


class _NoFinaleGame:
    """A duck-typed adapter that predates `is_finale` entirely."""
    def is_clear(self, start_key, ram):
        return bool(ram.get("c"))


def test_a_finale_only_win_is_not_scored_as_a_failure():
    fires = clear_predicate(_FinaleOnlyGame(), ())
    v = verify_ram_trace([{"ready": 0}, {"ready": 0}, {"ready": 2}],
                         _spec(), fires)
    assert v.status == "PASS" and v.replayed_steps == 2, v


def test_the_finale_arm_can_fail():
    """Anti-vacuity: the same predicate on a tape that never reaches the
    finale byte must still FAIL. Without this the test above would pass
    just as well against a predicate hardwired to True."""
    fires = clear_predicate(_FinaleOnlyGame(), ())
    v = verify_ram_trace([{"ready": 0}] * 4, _spec(), fires)
    assert v.status == "FAIL", v


def test_is_clear_still_carries_the_verdict_on_its_own():
    """The finale arm is additive — it must not shadow `is_clear`."""
    fires = clear_predicate(_NoFinaleGame(), ())
    assert verify_ram_trace([{"c": 0}, {"c": 1}], _spec(), fires).status == "PASS"
    assert verify_ram_trace([{"c": 0}] * 3, _spec(), fires).status == "FAIL"


def test_an_adapter_without_is_finale_degrades_instead_of_raising():
    """One duck-typed input must never end a 298-tape sweep."""
    fires = clear_predicate(_NoFinaleGame(), ())
    assert fires({"c": 0}) is False


def test_a_real_finale_profile_exists_and_its_is_clear_arm_is_inert():
    """Anchors the synthetic cases to a shipped config, so this cannot
    become a synthetic-only regression test. Excitebike is the only
    `finale:`-hooked profile on the roster; its `level_key` is empty, so
    the arm the sweep used to ask exclusively is the `() > ()` identity."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "excitebike.yaml").read_text())
    solve = cfg["solve"]
    assert solve["level_key"] == []
    assert solve["finale"]["level_key"] == []
    assert tuple(solve["level_key"]) == tuple(solve["finale"]["level_key"])
    # strict-greater is dead on this key; equality is not
    assert not (tuple(solve["level_key"]) > tuple(solve["level_key"]))
