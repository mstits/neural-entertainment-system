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
    KNOWN_BAD, TapeSpec, Verdict, batch, build_consumer_index, build_report,
    discover_tapes, evaluate_gate, read_tape, replay_batch, resolve_profile,
    spec_problems,
)


class StubPool:
    """Reproduces nes_core.Pool's step_all 4-tuple shape.

    `clear_at` maps worker -> the tick at which its RAM starts reporting a
    clear, so a tape can be made to clear early, exactly at its end, or
    never.
    """

    def __init__(self, n: int, clear_at: dict[int, int] | None = None):
        self.n = n
        self.clear_at = clear_at or {}
        self.t = 0
        self.loaded: dict[int, bytes] = {}
        self.actions_seen: dict[int, list[int]] = {i: [] for i in range(n)}

    def load_worker_state(self, i: int, blob: bytes) -> None:
        self.loaded[i] = blob

    def step_all(self, acts):
        self.t += 1
        for i in range(min(len(acts), self.n)):
            self.actions_seen[i].append(int(acts[i]))
        out = []
        for i in range(self.n):
            cleared = i in self.clear_at and self.t >= self.clear_at[i]
            out.append((None, None, {"cleared": cleared, "t": self.t}, None))
        return out


def _clear_fn(ram, spec):
    return bool(ram["cleared"])


def _key_fn(ram):
    return [ram["t"]]


def test_batch_chunks_including_remainder():
    assert batch([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        batch([1], 0)


def _spec(name="t.json"):
    return TapeSpec(tape=name, actions="a.npy", root_state="r.state",
                    profile="configs/p.yaml", start_wd=[0, 0],
                    clear_wd=[0, 1], steps=3, core_sha16="abc")


def test_a_tape_that_clears_within_its_actions_passes():
    pool = StubPool(1, clear_at={0: 2})
    v = replay_batch(pool, [_spec()], [[1, 1, 1]], [b"r"], _clear_fn, _key_fn)
    assert v[0].status == "PASS" and v[0].replayed_steps == 2


def test_a_tape_that_never_clears_fails_not_errors():
    pool = StubPool(1)
    v = replay_batch(pool, [_spec()], [[1, 1, 1]], [b"r"], _clear_fn, _key_fn)
    assert v[0].status == "FAIL"
    assert "exhausted" in v[0].reason


def test_verdict_is_frozen_at_the_clear_not_at_tape_end():
    """A short tape must not keep stepping past its own clear.

    Without the freeze a policy that clears then walks back out would be
    scored FAIL — a false negative on a genuinely good tape.
    """
    pool = StubPool(1, clear_at={0: 1})
    v = replay_batch(pool, [_spec()], [[1, 1, 1, 1, 1]], [b"r"],
                     lambda ram, s: ram["t"] == 1, _key_fn)
    assert v[0].status == "PASS" and v[0].replayed_steps == 1


def test_shorter_tape_is_padded_with_noop_so_workers_stay_aligned():
    pool = StubPool(2, clear_at={1: 4})
    specs = [_spec("short.json"), _spec("long.json")]
    v = replay_batch(pool, specs, [[3], [7, 7, 7, 7]], [b"a", b"b"],
                     _clear_fn, _key_fn)
    by = {x.tape: x for x in v}
    assert by["short.json"].status == "FAIL"
    assert by["long.json"].status == "PASS"
    # worker 0 was padded with NOOP for ticks 2..4, never re-fed its action
    assert pool.actions_seen[0] == [3, 0, 0, 0]


def test_empty_tape_is_a_failure():
    pool = StubPool(1)
    v = replay_batch(pool, [_spec()], [[]], [b"r"], _clear_fn, _key_fn)
    assert v[0].status == "FAIL" and "empty" in v[0].reason


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        replay_batch(StubPool(1), [_spec()], [], [b"r"], _clear_fn, _key_fn)


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
