"""Tests for scripts/go_explore_solve.py's pure helpers and GenericGame.

Kept separate from tests/test_go_explore.py (which covers the archive
in src/training/go_explore.py). GenericGame's __init__ only reads the
profile dict (no ROM/Pool I/O), so it's constructible here with a
minimal in-memory profile; Solver itself needs a real ROM/Pool and is
only exercised via progress_line() through a duck-typed stand-in.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts.go_explore_solve import (
    GenericGame,
    Solver,
    apply_hw_flags,
    available_hw_flags,
    check_state_sidecar,
    column_extremes,
    count_wmax,
    hw_provenance,
    inversion_armed,
    ortho_armed,
    ortho_pool,
    resolve_hw_flags,
    resolve_inversion_pin_secs,
    resolve_verify_bank,
    stamp_stats_provenance,
    update_stall,
    write_state_sidecar,
)


def _fresh_stall() -> dict:
    return {"last_cells": 0, "last_t": 0.0, "flat_windows": 0}


def test_update_stall_resets_on_growth() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    assert stall == {"last_cells": 10, "last_t": 60.0, "flat_windows": 0}


def test_update_stall_increments_on_flat_window() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    assert stall["flat_windows"] == 1


def test_update_stall_reaches_two_window_stall_threshold() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    update_stall(stall, 10, 180.0)
    assert stall["flat_windows"] == 2


def test_update_stall_recovers_once_new_cells_appear() -> None:
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 10, 120.0)
    assert stall["flat_windows"] == 1
    update_stall(stall, 11, 180.0)
    assert stall["flat_windows"] == 0
    assert stall["last_cells"] == 11


def test_update_stall_a_shrinking_count_still_counts_as_flat() -> None:
    # An archive never shrinks in practice, but the check is `<=` by
    # design (matches "no NEW cells", not "changed") — pin that here so
    # a future refactor can't accidentally flip it to strict equality
    # and start flagging every static run as instantly stalled.
    stall = _fresh_stall()
    update_stall(stall, 10, 60.0)
    update_stall(stall, 5, 120.0)
    assert stall["flat_windows"] == 1


def _fake_solver(tmp_path, n_cells: int) -> SimpleNamespace:
    # Duck-typed stand-in for Solver: only the attributes progress_line()
    # actually reads. Regression test for a real bug (2026-08-06): the
    # stall-watchdog refactor left a stale bare `stall` reference in
    # progress_line()'s dict literal instead of `self._stall`, which
    # passed every existing test (none called progress_line() with real
    # data) and only surfaced live, on real search runs.
    return SimpleNamespace(
        archive=list(range(n_cells)),
        _stall={"last_cells": 0, "last_t": 0.0, "flat_windows": 0},
        max_area=0, max_gx_in_area={}, max_sect=0,
        n_solutions=0, best_sol_len=0, steps_done=100,
        door_weight=0, transition_macros=[], ortho_mode="off",
        out=tmp_path,
    )


def test_progress_line_does_not_crash_and_reports_stall_state(tmp_path) -> None:
    fake = _fake_solver(tmp_path, n_cells=5)
    Solver.progress_line(fake, 10.0)
    lines = (tmp_path / "progress.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["stall_flat_windows"] == 0


def _confluence_game(lives_addr: int = 0x00) -> GenericGame:
    # GenericGame.__init__ only reads the profile dict (no ROM/Pool I/O),
    # so a minimal in-memory profile is enough to exercise is_clear().
    profile = {
        "solve": {
            "rom": "roms/does-not-need-to-exist.nes",
            "progress": {"lo": 0x01},
            "y": 0x02,
            "level_key": [],
            "lives": lives_addr,
            "clear": {"mode": "confluence"},
        }
    }
    return GenericGame(profile)


class _FakeDetector:
    """Stands in for clear_detect.StreamingConfluenceDetector: fires on
    every push(), so the test isolates is_clear()'s lives-drop veto
    rather than the real detector's own signal logic."""

    def push(self, ram) -> bool:
        return True


def _ram(lives: int) -> bytearray:
    buf = bytearray(2048)
    buf[0x00] = lives
    return buf


def test_confluence_clear_fires_when_lives_are_unchanged() -> None:
    # Regression coverage for the Gradius false-positive (2026-08-06): a
    # real death was recorded as a fake win because is_clear() is checked
    # before is_dead() in Solver.observe(). Verify the non-death path
    # still fires — the fix must not turn the detector into a permanent
    # no-op.
    game = _confluence_game()
    ctx = {"_clear_det": _FakeDetector()}
    game.is_clear((), _ram(lives=3), ctx)   # first call establishes prev_lives
    assert game.is_clear((), _ram(lives=3), ctx) is True


def test_confluence_clear_is_vetoed_on_the_same_step_lives_drop() -> None:
    game = _confluence_game()
    ctx = {"_clear_det": _FakeDetector()}
    game.is_clear((), _ram(lives=3), ctx)   # establish prev_lives=3
    assert game.is_clear((), _ram(lives=2), ctx) is False   # died this step


def test_confluence_clear_fires_again_once_lives_stabilize_post_death() -> None:
    game = _confluence_game()
    ctx = {"_clear_det": _FakeDetector()}
    game.is_clear((), _ram(lives=3), ctx)
    assert game.is_clear((), _ram(lives=2), ctx) is False   # vetoed
    # lives no longer DROPPING (2 -> 2): the veto is per-step, not sticky.
    assert game.is_clear((), _ram(lives=2), ctx) is True


# ---------------------------------------------------------------------
# hw-flag selection + lineage provenance
#
# A savestate blob carries STATE, never timing CONFIG, so loading a
# lineage's entrance into a stock Pool silently runs a different
# machine. These pin the three properties that make the opt-in safe:
# the default really is empty (existing seeded solves unchanged), a
# typo'd flag fails loudly instead of running the wrong machine, and a
# blob's recorded lineage is compared against the run's flags.
# ---------------------------------------------------------------------


def test_resolve_hw_flags_defaults_to_empty_when_the_profile_omits_the_key():
    # The non-negotiable one: no key, no CLI -> no setter is ever called,
    # so every pre-existing seeded solve stays bit-identical.
    assert resolve_hw_flags({}, None) == []
    assert resolve_hw_flags({"solve": {}}, None) == []
    assert resolve_hw_flags({"solve": {"hw_flags": None}}, None) == []
    assert resolve_hw_flags({"solve": {"hw_flags": []}}, None) == []


def test_resolve_hw_flags_reads_the_profile_list_in_order():
    profile = {"solve": {"hw_flags": ["mmio_read_timing", "nmi_poll_timing"]}}
    assert resolve_hw_flags(profile, None) == [
        "mmio_read_timing", "nmi_poll_timing"]


def test_resolve_hw_flags_cli_overrides_the_profile():
    profile = {"solve": {"hw_flags": ["mmio_read_timing"]}}
    assert resolve_hw_flags(profile, "nmi_poll_timing") == ["nmi_poll_timing"]


def test_resolve_hw_flags_cli_none_forces_the_empty_set():
    profile = {"solve": {"hw_flags": ["mmio_read_timing", "nmi_poll_timing"]}}
    assert resolve_hw_flags(profile, "none") == []
    assert resolve_hw_flags(profile, "NONE") == []
    assert resolve_hw_flags(profile, "") == []


def test_resolve_hw_flags_dedupes_and_tolerates_the_method_spelling():
    profile = {"solve": {"hw_flags": [
        "mmio_read_timing", "set_hw_mmio_read_timing", " nmi_poll_timing "]}}
    assert resolve_hw_flags(profile, None) == [
        "mmio_read_timing", "nmi_poll_timing"]


def test_resolve_hw_flags_rejects_an_unknown_name_loudly():
    # Silently ignoring a typo would run the WRONG machine while the
    # receipt claims the right one — strictly worse than not asking.
    with pytest.raises(SystemExit) as e:
        resolve_hw_flags({"solve": {"hw_flags": ["mmio_read_timming"]}}, None)
    assert "mmio_read_timming" in str(e.value)


def test_available_hw_flags_covers_the_cv_lineage_set():
    have = set(available_hw_flags())
    for name in ("reset_alignment", "mmio_read_timing",
                 "dmc_stall_timing", "nmi_poll_timing"):
        assert name in have, f"{name} missing from the installed core"


class _FakePool:
    """Records set_hw_* calls in order. Only the names the real Pool
    exposes are settable, matching resolve_hw_flags' validation."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("set_hw_"):
            raise AttributeError(name)
        return lambda on: self.calls.append((name, on))


def test_apply_hw_flags_sets_each_named_flag_in_order():
    pool = _FakePool()
    apply_hw_flags(pool, ["mmio_read_timing", "nmi_poll_timing"])
    assert pool.calls == [("set_hw_mmio_read_timing", True),
                          ("set_hw_nmi_poll_timing", True)]


def test_apply_hw_flags_is_a_no_op_on_the_empty_default():
    pool = _FakePool()
    apply_hw_flags(pool, [])
    assert pool.calls == []


def test_state_sidecar_round_trips_next_to_the_blob(tmp_path):
    blob = tmp_path / "entrance_after_2.state"
    blob.write_bytes(b"\x00" * 8)
    prov = hw_provenance(["mmio_read_timing", "nmi_poll_timing"], 4)
    path = write_state_sidecar(blob, prov, {"settled_key": [3]})
    assert path == tmp_path / "entrance_after_2.state.json"
    rec = json.loads(path.read_text())
    assert rec["hw_flags"] == ["mmio_read_timing", "nmi_poll_timing"]
    assert rec["frame_skip"] == 4
    assert rec["blob"] == "entrance_after_2.state"
    assert rec["settled_key"] == [3]
    assert "nes_core" in rec


def test_check_state_sidecar_passes_when_there_is_nothing_to_check(tmp_path):
    blob = tmp_path / "unlabelled.state"
    blob.write_bytes(b"\x00")
    assert check_state_sidecar(blob, ["mmio_read_timing"]) is True


def test_check_state_sidecar_accepts_a_matching_lineage(tmp_path):
    blob = tmp_path / "e.state"
    blob.write_bytes(b"\x00")
    write_state_sidecar(blob, hw_provenance(["nmi_poll_timing",
                                             "mmio_read_timing"], 4))
    # Order-insensitive: the flags are a set of independent toggles.
    assert check_state_sidecar(blob, ["mmio_read_timing",
                                      "nmi_poll_timing"]) is True


def test_check_state_sidecar_flags_a_lineage_mismatch(tmp_path, capsys):
    # The cv_chain_hw2 failure mode: a 4-flag entrance loaded into a
    # stock pool. Must be reported, and must name both sets.
    blob = tmp_path / "entrance_after_2.state"
    blob.write_bytes(b"\x00")
    write_state_sidecar(blob, hw_provenance(
        ["reset_alignment", "mmio_read_timing",
         "dmc_stall_timing", "nmi_poll_timing"], 4))
    assert check_state_sidecar(blob, []) is False
    out = capsys.readouterr().out
    assert "HW-FLAG LINEAGE MISMATCH" in out
    assert "nmi_poll_timing" in out


def test_stamp_stats_provenance_preserves_the_search_fields(tmp_path):
    # The search-content fields must survive untouched — the provenance
    # block is additive, never a rewrite of the archive's own stats.
    archive = tmp_path / "archive.pkl"
    stats = tmp_path / "archive.stats.json"
    search_fields = {"cells": 454, "frontier": 404, "best_score": 694,
                     "records": 5680, "new_cells": 454, "improvements": 362}
    stats.write_text(json.dumps(search_fields))
    prov = hw_provenance(["mmio_read_timing"], 4)
    stamp_stats_provenance(archive, prov)
    got = json.loads(stats.read_text())
    assert {k: got[k] for k in search_fields} == search_fields
    assert got["hw_provenance"]["hw_flags"] == ["mmio_read_timing"]
    # The machine block must NOT land on `provenance`: that key is the
    # reserved honest-origin marker, and one token cannot mean a string
    # in one artifact and a dict in its sibling.
    assert "provenance" not in got
    # Idempotent: a second flush must not churn the file.
    before = stats.read_bytes()
    stamp_stats_provenance(archive, prov)
    assert stats.read_bytes() == before


def test_stamp_stats_provenance_is_silent_when_there_is_no_stats_file(tmp_path):
    # The archive's disk-space guard can skip a save entirely; stamping
    # must not be the thing that raises in that path.
    stamp_stats_provenance(tmp_path / "archive.pkl", hw_provenance([], 4))
    assert not (tmp_path / "archive.stats.json").exists()


def test_stamp_stats_provenance_uses_a_tmp_name_the_archive_never_touches(tmp_path):
    # GoExploreArchive.save() writes archive.stats.json.tmp; colliding
    # on that name would race the background flush thread.
    archive = tmp_path / "archive.pkl"
    (tmp_path / "archive.stats.json").write_text("{}")
    stamp_stats_provenance(archive, hw_provenance([], 4))
    assert not (tmp_path / "archive.stats.json.tmp").exists()
    assert not (tmp_path / "archive.stats.json.prov.tmp").exists()


def test_frame_anchor_becomes_selectable_once_the_core_exposes_it():
    # Pending the nes_core rebuild that carries Pool::set_hw_frame_anchor
    # (nes_core/src/pool.rs). Until then the name must be REJECTED rather
    # than silently ignored — a run that thinks it anchored frames but
    # didn't is exactly the certification hole this whole change closes.
    if "frame_anchor" not in available_hw_flags():
        with pytest.raises(SystemExit) as e:
            resolve_hw_flags({"solve": {"hw_flags": ["frame_anchor"]}}, None)
        assert "rebuild" in str(e.value)
        pytest.skip("installed nes_core predates Pool::set_hw_frame_anchor "
                    "— rebuild (maturin + the stale-.so copy) to activate")
    assert resolve_hw_flags(
        {"solve": {"hw_flags": ["frame_anchor"]}}, None) == ["frame_anchor"]


def test_solver_cli_hw_flags_defaults_to_none_so_nothing_is_set():
    # Guards the default-off contract at the argparse layer: absent
    # --hw-flags must arrive as None (profile decides), NOT as "" or [].
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    assert '"--hw-flags", type=str, default=None' in src
    ns = SimpleNamespace(hw_flags=None)
    assert resolve_hw_flags({}, getattr(ns, "hw_flags", None)) == []


# ---------------------------------------------------------------------
# solution-receipt shape
#
# Regression tests for a real, silent defect (2026-08-07): the machine
# block was added to _dump_solution's dict literal under the key
# `provenance`, which the same literal already used for the string
# `"search"` — the honest-origin marker every banked receipt carries and
# that CLAIMS.md audits compare against literally. Python keeps the LAST
# binding, so the marker was deleted with no error, no warning, and no
# failing test. Nothing in the suite read _dump_solution's JSON.
# ---------------------------------------------------------------------

def _dump_one_solution(tmp_path, flags=("mmio_read_timing",),
                       verify_bank=False) -> dict:
    """Drive Solver._dump_solution through a duck-typed stand-in and
    return the receipt it wrote. Only the attributes the method actually
    reads are supplied. verify_bank=False keeps these receipt-shape tests
    off the replay path (which needs a real ROM/Pool); the verification
    behavior itself is covered in tests/test_confluence_v2.py."""
    (tmp_path / "solutions").mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        best_sol_len=10**9, sol_counter=0, n_solutions=0,
        verify_bank=verify_bank, verify_checks=0, verify_rejections=0,
        out=tmp_path,
        args=SimpleNamespace(profile="configs/castlevania.yaml", workers=8,
                             hw_flags=",".join(flags) or None),
        roots={"entrance": {"path": "runs/cv_chain_hw2/entrances/"
                                    "entrance_after_2.state"}},
        provenance=hw_provenance(list(flags), 4),
        start_wd=(0, 2),
        game=SimpleNamespace(level_key=lambda ram: (0, 3)),
    )
    Solver._dump_solution(fake, "entrance", [1, 2, 3], bytearray(2048), 3)
    return json.loads((tmp_path / "solutions" / "sol_000.json").read_text())


def test_solution_receipt_keeps_the_honest_origin_marker(tmp_path):
    rec = _dump_one_solution(tmp_path)
    # THE regression: this is the integrity line the no-cheating course
    # correction rests on, and an audit spells it exactly like this.
    assert rec["provenance"] == "search"
    assert isinstance(rec["provenance"], str)


def test_solution_receipt_records_the_machine_under_hw_provenance(tmp_path):
    rec = _dump_one_solution(tmp_path, flags=("reset_alignment",
                                              "nmi_poll_timing"))
    assert rec["hw_provenance"]["hw_flags"] == ["reset_alignment",
                                                "nmi_poll_timing"]
    assert rec["hw_provenance"]["frame_skip"] == 4
    assert "nes_core" in rec["hw_provenance"]


def test_solution_receipt_is_type_compatible_with_the_banked_corpus(tmp_path):
    # go_explore_2_1.py:209 and replay_to_demos.py:110 emit the same
    # string marker; 207 banked receipts hold it. New receipts must stay
    # readable by the same audit expression across every tool.
    rec = _dump_one_solution(tmp_path)
    assert rec["provenance"] == "search"          # the audit expression
    for key in ("solver_args", "root_id", "root_state", "start_wd",
                "clear_wd", "steps", "actions"):
        assert key in rec, key


def test_solution_receipt_records_whether_the_clear_was_replay_verified(
        tmp_path):
    # An audit has to be able to separate a v2 verified receipt from a
    # pre-v2 (or --no-verify-bank) one without re-running anything.
    assert _dump_one_solution(tmp_path)["replay_verified"] is False


# ---------------------------------------------------------------------
# banking trust: --verify-bank
#
# The detector fabricated wins on 3 of 7 games (2026-08-06). Replaying a
# candidate from its root before writing it is the gate; it is default ON
# because it is cheap (measured ~2.7 ms/action: 887 actions 2.4 s, 1,735
# actions 4.7 s) against 25-45 minute solves.
# ---------------------------------------------------------------------

def test_solver_cli_verify_bank_defaults_on(monkeypatch):
    assert _parse_solver_argv(monkeypatch).verify_bank is True


def test_solver_cli_no_verify_bank_is_the_explicit_escape(monkeypatch):
    assert _parse_solver_argv(monkeypatch, "--no-verify-bank").verify_bank is False


def test_verify_bank_is_a_json_type_so_receipts_keep_recording_it(monkeypatch):
    # solver_args in every receipt is filtered to (str, int, float, bool);
    # a flag that falls out of that filter is unverifiable after the fact.
    args = _parse_solver_argv(monkeypatch)
    assert isinstance(args.verify_bank, bool)


def test_resolve_verify_bank_defaults_on_when_the_attribute_is_absent():
    # THE direction that matters, and the opposite of every other opt-in knob
    # in this file: an args namespace that predates the flag must still be
    # verified. Flipping this default silently un-gates the show's own solver.
    assert resolve_verify_bank(SimpleNamespace()) is True


def test_resolve_verify_bank_honors_an_explicit_opt_out():
    assert resolve_verify_bank(SimpleNamespace(verify_bank=False)) is False
    assert resolve_verify_bank(SimpleNamespace(verify_bank=True)) is True


def test_an_older_in_process_call_site_still_gets_verification():
    # scripts/live_solve_show.py builds the args namespace by hand and does
    # not know about this flag. Banking trust must not be something a stale
    # call site opts out of by omission.
    from scripts.live_solve_show import solver_args

    ns, _ = solver_args("configs/mario.yaml", "root.state", Path("/tmp/x"),
                        minutes=1.0, workers=2)
    assert resolve_verify_bank(ns) is True


# ---------------------------------------------------------------------
# CLI surface: the saturation arms
#
# The solver's argparse block lives inside main(), so these drive main()
# with a stub Solver that captures the parsed namespace and bails before
# any ROM/Pool work. The point of every default assertion below is the
# reproducibility gate: a banked receipt records solver_args, and a
# replay that omits the new flags must sample exactly as the receipted
# run did.
# ---------------------------------------------------------------------

class _StubbedOut(Exception):
    """Raised by the stub Solver once it has captured the namespace."""


def _parse_solver_argv(monkeypatch, *extra) -> SimpleNamespace:
    import scripts.go_explore_solve as ges

    captured = {}

    class _StubSolver:
        def __init__(self, args):
            captured["args"] = args
            raise _StubbedOut

    monkeypatch.setattr(ges, "Solver", _StubSolver)
    monkeypatch.setattr(sys, "argv",
                        ["go_explore_solve", "--out", "/tmp/unused",
                         "--root-state", "unused.state",
                         "--profile", "unused.yaml", *extra])
    with pytest.raises(_StubbedOut):
        ges.main()
    return captured["args"]


def test_solver_cli_inversion_pin_secs_defaults_to_the_receipted_constant(
        monkeypatch):
    # 180 s is the hardcoded value the verified 32-level SMB clear ran
    # under; the flag exists to move it, not to change it by default.
    assert _parse_solver_argv(monkeypatch).inversion_pin_secs == 180.0


def test_solver_cli_inversion_pin_secs_takes_the_disable_sentinel(monkeypatch):
    args = _parse_solver_argv(monkeypatch, "--inversion-pin-secs", "-1")
    assert args.inversion_pin_secs == -1.0
    assert resolve_inversion_pin_secs(args) < 0.0     # arm off


def test_resolve_inversion_pin_secs_defaults_when_the_attribute_is_absent():
    # In-process constructions (the live show) build a namespace that
    # predates the flag — they must keep the receipted 180 s.
    assert resolve_inversion_pin_secs(SimpleNamespace()) == 180.0


def test_resolve_inversion_pin_secs_coerces_to_float():
    assert resolve_inversion_pin_secs(
        SimpleNamespace(inversion_pin_secs=45)) == 45.0
    assert resolve_inversion_pin_secs(
        SimpleNamespace(inversion_pin_secs="0")) == 0.0


def _init_assignment(target: str) -> ast.AST | None:
    """The value expression `Solver.__init__` assigns to `self.<target>`,
    or None if it never does. AST rather than substring so the check
    survives reformatting but not a rewrite."""
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and tgt.attr == target
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                return node.value
    return None


def test_solver_init_resolves_the_pin_from_argv_and_not_a_hardcode():
    # THE load-bearing wiring, and the one assertion the lane was missing:
    # nothing else connects argparse to the gate. Verified by mutation —
    # replacing this assignment with `self._inv_pin_secs = 180.0` (i.e.
    # --inversion-pin-secs silently ignored end to end, -1 no longer
    # disabling) left the whole file green. A hardcode, a typo'd resolver
    # name, or reading some other namespace now fails here.
    val = _init_assignment("_inv_pin_secs")
    assert isinstance(val, ast.Call), "not resolved from args at all"
    assert isinstance(val.func, ast.Name)
    assert val.func.id == "resolve_inversion_pin_secs"
    assert [a.id for a in val.args if isinstance(a, ast.Name)] == ["args"]


def test_solver_init_wires_every_ortho_knob_to_its_own_flag():
    # Same mutation class for the vertical arm: each knob has to come off
    # the parsed namespace under the flag's own dest, so a copy-paste that
    # points two knobs at one flag (or hardcodes a default) is caught here
    # rather than by a campaign that quietly sampled the wrong band.
    for attr, dest in (("ortho_mode", "ortho"),
                       ("ortho_pin_secs", "ortho_pin_secs"),
                       ("ortho_bias", "ortho_bias"),
                       ("ortho_band", "ortho_band"),
                       ("ortho_weight", "ortho_weight"),
                       ("ortho_macro_p", "ortho_macro_p")):
        val = _init_assignment(attr)
        assert val is not None, f"{attr} is never assigned"
        names = [n.value for n in ast.walk(val)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert dest in names, f"{attr} does not read args.{dest}"


def _legacy_inversion_gate(pin_gx, gx, floor, elapsed) -> bool:
    """The receipted inline expression, before it was extracted — with
    180.0 in place of the flag. The equivalence grid below is what lets
    the extraction ship without re-verifying the 32-level clear."""
    return (pin_gx > 400 and gx >= 0 and floor <= gx <= pin_gx + 60
            and elapsed >= 180.0)


@pytest.mark.parametrize("pin_gx", [0, 400, 401, 2064])
@pytest.mark.parametrize("gx", [-1, 0, 1764, 2064, 2124, 2125])
@pytest.mark.parametrize("floor", [0, 1764, 2200])
@pytest.mark.parametrize("elapsed", [0.0, 179.9, 180.0, 1e9])
def test_the_default_pin_is_boolean_identical_to_the_receipted_gate(
        pin_gx, gx, floor, elapsed):
    # 288 combinations spanning both window edges, the 400 px frontier
    # floor, the garbage-gx sentinel and the pin boundary. At the shipped
    # default the extracted gate must decide EXACTLY what the campaign's
    # inline expression decided; anything else is a silent behaviour
    # change dressed up as a refactor.
    assert (inversion_armed(180.0, pin_gx, gx, floor, elapsed)
            is _legacy_inversion_gate(pin_gx, gx, floor, elapsed))


@pytest.mark.parametrize("pin_secs", [-1.0, -0.001, -1e9])
def test_a_negative_pin_never_arms_however_long_the_frontier_sits(pin_secs):
    # The disable sentinel has to be checked BEFORE the clock: every
    # other conjunct is satisfied here, and `elapsed >= -1` would
    # otherwise make a "disabled" arm the always-on arm.
    assert inversion_armed(pin_secs, 2064, 2064, 0, 1e9) is False
    assert inversion_armed(pin_secs, 2064, 2064, 0, 0.0) is False


def test_a_zero_pin_arms_immediately_and_the_window_still_binds():
    # 0 is "arm as soon as the frontier exists", not "arm always" — the
    # gx window is an independent conjunct and must survive the shortcut.
    assert inversion_armed(0.0, 2064, 2064, 1764, 0.0) is True
    assert inversion_armed(0.0, 2064, 2125, 1764, 0.0) is False   # past +60
    assert inversion_armed(0.0, 2064, 1763, 1764, 0.0) is False   # below floor
    assert inversion_armed(0.0, 400, 400, 0, 0.0) is False        # no frontier
    assert inversion_armed(0.0, 2064, -1, 0, 0.0) is False        # garbage gx


def test_inversion_gate_calls_the_helper_with_the_resolved_flag():
    # The call site itself: explore()'s per-worker loop must pass the
    # resolved attribute (not a literal, not args.*) and must read the
    # pin clock once per STEP — a per-worker time.time() is what the
    # old short-circuit was buying, and the extraction gives it back.
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    gate = src.split("_w = (self.inv_weights", 1)[1]
    gate = gate.split("else self.weights", 1)[0]
    assert "inversion_armed(self._inv_pin_secs" in gate
    assert "_pin_elapsed" in gate
    assert "180.0" not in gate
    assert "time.time()" not in gate
    body = src.split("    def explore(", 1)[1].split("for i, c in enumerate", 1)[0]
    assert "_pin_elapsed = time.time() - self._pin_time" in body


def test_the_shows_in_process_namespace_keeps_the_receipted_pin(tmp_path):
    # live_solve_show builds its Solver namespace by hand (no argparse),
    # so it carries no inversion_pin_secs — the show must keep sampling
    # exactly as the banked campaign did.
    from scripts.live_solve_show import solver_args

    sargs, _ = solver_args("configs/mario.yaml", "root.state", tmp_path,
                           minutes=1.0, workers=2)
    assert not hasattr(sargs, "inversion_pin_secs")
    assert resolve_inversion_pin_secs(sargs) == 180.0


def test_solver_cli_ortho_arm_is_off_by_default(monkeypatch):
    # THE inertness contract: every ortho knob parses, and with --ortho
    # absent the arm is off, so sampling matches the banked campaign.
    args = _parse_solver_argv(monkeypatch)
    assert args.ortho == "off"
    assert args.ortho_pin_secs == 120.0
    assert args.ortho_bias == 0.30
    assert args.ortho_band == 1
    assert args.ortho_weight == 4.0
    assert args.ortho_macro_p == 0.0


def test_solver_cli_ortho_accepts_both_vertical_directions(monkeypatch):
    assert _parse_solver_argv(monkeypatch, "--ortho", "up").ortho == "up"
    assert _parse_solver_argv(monkeypatch, "--ortho", "down").ortho == "down"


def test_solver_cli_ortho_rejects_an_unknown_direction(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_solver_argv(monkeypatch, "--ortho", "sideways")


def test_solver_cli_ortho_knobs_round_trip_their_values(monkeypatch):
    args = _parse_solver_argv(
        monkeypatch, "--ortho", "up", "--ortho-pin-secs", "30",
        "--ortho-bias", "0.75", "--ortho-band", "3",
        "--ortho-weight", "2.5", "--ortho-macro-p", "0.1")
    assert (args.ortho_pin_secs, args.ortho_bias, args.ortho_band,
            args.ortho_weight, args.ortho_macro_p) == (30.0, 0.75, 3,
                                                       2.5, 0.1)


def test_solver_cli_help_renders_every_new_flag(monkeypatch, capsys):
    import scripts.go_explore_solve as ges

    monkeypatch.setattr(sys, "argv", ["go_explore_solve", "--help"])
    with pytest.raises(SystemExit) as e:
        ges.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--inversion-pin-secs", "--ortho", "--ortho-pin-secs",
                 "--ortho-bias", "--ortho-band", "--ortho-weight",
                 "--ortho-macro-p"):
        assert flag in out, flag


def test_new_solver_flags_are_json_types_so_receipts_keep_recording_them():
    # _dump_solution filters solver_args to (str, int, float, bool); a
    # knob typed outside that set would vanish from the receipt and make
    # a run unreproducible from its own JSON.
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    for decl in ('"--inversion-pin-secs", type=float, default=180.0',
                 '"--ortho", choices=("off", "up", "down"), default="off"',
                 '"--ortho-pin-secs", type=float, default=120.0',
                 '"--ortho-bias", type=float, default=0.30',
                 '"--ortho-band", type=int, default=1',
                 '"--ortho-weight", type=float, default=4.0',
                 '"--ortho-macro-p", type=float, default=0.0'):
        assert decl in src, decl


def test_no_dict_literal_in_the_solver_shadows_one_of_its_own_keys():
    # Generic guard for the whole bug CLASS: Python silently drops the
    # earlier binding and there is no linter configured in this repo, so
    # this AST scan is the gate. Covers every dict literal in both
    # solver entry points, not just _dump_solution's.
    import ast

    import scripts.go_explore_chain as gec
    import scripts.go_explore_solve as ges

    offenders = []
    for mod in (ges, gec):
        path = Path(mod.__file__)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = set()
            for k in node.keys:
                if not isinstance(k, ast.Constant):
                    continue          # **splat or computed key
                if k.value in seen:
                    offenders.append(f"{path.name}:{k.lineno} "
                                     f"duplicate key {k.value!r}")
                seen.add(k.value)
    assert offenders == []


# ---------------------------------------------------------------------
# orthogonal (vertical) progress arm
#
# The heuristic-inversion arm un-prunes left/down once gx saturates; the
# ortho arm is its vertical counterpart for walls that are shafts rather
# than runs of ground. Every property below is measured off our OWN
# rollouts (column extremes of the archive, the solver's self-measured
# pin clock) — no map, no route. The arm is opt-in, so these pin both
# halves of the contract: the semantics when it is on, and that the
# shipped defaults leave the sampler exactly as the receipted campaign
# ran it.
# ---------------------------------------------------------------------

def _ocell(gx: int, yb: int, *, area: int = 0, barren: int = 0,
           chosen: int = 0):
    """A duck-typed archive cell. The arm reads key[0] (sect), key[-5]
    (area), key[-2] (y band), key[-1] (gx bucket) and the dynamic
    `barren` counter — nothing else, so a namespace is enough."""
    key = (0, 0, 0, (), 0, ()) + (area, 3, 1, yb, gx)
    return SimpleNamespace(key=key, state=b"s", best_score=1.0,
                           best_steps=1, visits=1, times_chosen=chosen,
                           explored=False, barren=barren)


def _ortho_solver(cells, **over):
    """Duck-typed Solver for the selection arms: only the attributes
    select()/_refresh_sel_cache() read, plus the real methods bound to
    the namespace (Solver itself needs a ROM and a Pool). deep_bias 0
    and ortho_bias 1 isolate the ortho arm from the deep-frontier one."""
    f = SimpleNamespace(
        args=SimpleNamespace(deep_bias=0.0),
        rng=np.random.default_rng(0),
        archive=SimpleNamespace(cells={c.key: c for c in cells}),
        max_area=max((c.key[-5] for c in cells), default=0), max_sect=0,
        sel_mode="legacy", frontier_throttle=0,
        door_weight=0.0, _doors=frozenset(), _key_ids={},
        ortho_mode="up", ortho_pin_secs=0.0, ortho_bias=1.0,
        ortho_band=1, ortho_weight=4.0, _pin_time=0.0,
        _ortho_pool=[], _ortho_ids=set(), _ortho_ext={},
        _ortho_deep_yband=None, _ortho_selections=0,
        _ortho_cols_improved=0,
        _sel_cells=None, _sel_n=0, _sel_area=None)
    for k, v in over.items():
        setattr(f, k, v)
    for name in ("_refresh_sel_cache", "_ortho_armed", "select"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def test_column_extremes_takes_the_ceiling_going_up_and_the_floor_going_down():
    # The NES y axis grows DOWNWARD, so "up" is the min band. Getting
    # this backwards would aim the whole arm at the floor of the shaft.
    keys = [_ocell(gx, yb).key
            for gx, yb in ((4, 7), (4, 2), (4, 5), (9, 3), (9, 8))]
    assert column_extremes(keys, "up") == {4: 2, 9: 3}
    assert column_extremes(keys, "down") == {4: 7, 9: 8}
    assert column_extremes([], "up") == {}


def test_ortho_band_zero_is_the_extreme_row_only_and_the_window_is_per_column():
    one_column = [_ocell(4, 2), _ocell(4, 3), _ocell(4, 6)]
    assert [c.key[-2] for c in ortho_pool(one_column, "up", 0)] == [2]
    assert [c.key[-2] for c in ortho_pool(one_column, "up", 1)] == [2, 3]
    # THE aliasing trap: a GLOBAL cutoff (min y over every cell) would
    # let column 4's ceiling at y=2 define the whole level and drop
    # column 9's top-of-stack cell at y=8 — the arm would then never
    # climb the one column that has not been climbed yet, which is
    # precisely the column that needs it. Membership is per column.
    two_columns = [_ocell(4, 2), _ocell(9, 8), _ocell(9, 9)]
    assert {(c.key[-1], c.key[-2])
            for c in ortho_pool(two_columns, "up", 0)} == {(4, 2), (9, 8)}


def test_an_empty_ortho_pool_falls_through_instead_of_crashing():
    assert ortho_pool([], "up", 1) == []
    # Armed, bias 1.0, but the cached pool is empty (every top-of-column
    # state evicted): the arm must decline, not index into an empty list
    # or return None to a caller that treats None as "archive exhausted".
    f = _ortho_solver([_ocell(4, 3)])
    f._refresh_sel_cache()
    f._ortho_pool, f._ortho_ids = [], set()
    assert f._ortho_armed() is True
    assert f.select() is not None
    assert f._ortho_selections == 0


def test_ortho_armed_is_inclusive_at_the_pin_boundary_and_off_stays_off():
    # Inclusive: a pin of exactly pin_secs arms. An exclusive `>` would
    # need one extra poll tick to fire, and the caller polls per worker
    # reassignment, not on a clock.
    assert ortho_armed("up", 100.0, 220.0, 120.0) is True
    assert ortho_armed("up", 100.0, 219.9, 120.0) is False
    assert ortho_armed("down", 100.0, 1e9, 120.0) is True
    # The MODE gate short-circuits regardless of how long the frontier
    # has been pinned — that is what makes the default inert.
    for mode in ("off", "", None):
        assert ortho_armed(mode, 0.0, 1e9, 0.0) is False


def test_count_wmax_is_the_exact_legacy_ceiling_and_composes_both_multipliers():
    # 1.1 is the count arm's rejection-sampling ceiling as receipted.
    # Both multipliers floor at 1.0, so an arm that is off cannot shrink
    # the ceiling and silently truncate the prior.
    assert count_wmax(0.0, 1.0) == 1.1
    assert count_wmax(0.0, 0.0) == 1.1
    assert count_wmax(1.0, 1.0) == 1.1
    # Doors (5x) and the ortho frontier (4x) composed: exact, not 22.0
    # +- epsilon — an understated Wmax truncates the prior in silence.
    assert count_wmax(5.0, 4.0) == 22.0
    assert count_wmax(4.0, 5.0) == 22.0


def test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults(monkeypatch):
    # Runtime mirror of the hw-flags inertness test: absent flags must
    # leave the machinery unset, not merely unused. The pin clock here
    # is wide open (pinned since t=0), so the ONLY thing keeping the arm
    # down is the default mode — exactly the property a stalled run
    # would otherwise trip into silently.
    args = _parse_solver_argv(monkeypatch)
    f = _ortho_solver([_ocell(4, 1), _ocell(4, 6), _ocell(9, 2)],
                      ortho_mode=str(args.ortho),
                      ortho_pin_secs=float(args.ortho_pin_secs),
                      ortho_bias=float(args.ortho_bias),
                      ortho_band=int(args.ortho_band),
                      ortho_weight=float(args.ortho_weight))
    f._refresh_sel_cache()
    assert f._ortho_armed() is False
    assert f._ortho_pool == [] and f._ortho_ids == set()
    assert f._ortho_ext == {} and f._ortho_deep_yband is None
    assert [f.select() for _ in range(200)].count(None) == 0
    assert f._ortho_selections == 0


# --- the headline deliverable: default-off BYTE identity --------------
# "Inert" above is about outcomes — no ortho pick, no marked wall. That
# is strictly weaker than what every banked receipt depends on: with the
# arm off, select() must consume the SAME RNG DRAWS in the same order as
# the pre-ortho sampler, or replaying a receipt's solver_args walks a
# different stream and reproduces nothing. The gate is one short-circuit
# away from breaking that silently: reordering it to
#   `if self.rng.random() < self.ortho_bias and armed and self._ortho_pool:`
# leaves every behavioural test in this file green (verified) while
# shifting the default stream from pick #0 in both sel_modes. The two
# checks below are the guard — one counts draws, one pins the ordering.

class _DrawTally:
    """A numpy Generator proxy that tallies draws by method name. Only
    the two methods select() actually calls are forwarded, so a third
    kind of draw appearing in the hot path fails loudly here rather than
    slipping through a permissive __getattr__."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = {"random": 0, "integers": 0}

    def random(self, *a, **k):
        self.calls["random"] += 1
        return self._rng.random(*a, **k)

    def integers(self, *a, **k):
        self.calls["integers"] += 1
        return self._rng.integers(*a, **k)


def _ortho_gate_test() -> ast.AST:
    """The `if` condition guarding the ortho arm inside Solver.select,
    located by AST so the check survives reformatting but not a rewrite."""
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "select")
    gates = [n.test for n in ast.walk(fn)
             if isinstance(n, ast.If)
             and any(isinstance(x, ast.Attribute) and x.attr == "_ortho_pool"
                     for x in ast.walk(n.test))]
    assert len(gates) == 1, f"expected one ortho gate, found {len(gates)}"
    return gates[0]


def test_the_disarmed_arm_draws_no_randomness_at_all(monkeypatch):
    # Draw-count ledger, exact and stream-independent by construction:
    # deep_bias 0 kills the deep arm's second draw, and every cell shares
    # one score with times_chosen reset each round, so the count arm's
    # rejection loop provably accepts on its first candidate (w/wmax ==
    # 1.0). The remaining budget is therefore FIXED per select — 1 gate
    # draw + 0 ortho + the arm's own — and any extra draw the disarmed
    # ortho gate makes shows up as an integer mismatch, not a flake.
    args = _parse_solver_argv(monkeypatch)
    assert str(args.ortho) == "off", "shipped default is no longer 'off'"
    n = 64

    for sel_mode, want in (("legacy", {"random": n, "integers": n}),
                           ("count", {"random": 2 * n, "integers": n})):
        cells = [_ocell(gx, yb) for gx in range(6) for yb in range(3)]
        tally = _DrawTally(0)
        f = _ortho_solver(cells, sel_mode=sel_mode, rng=tally,
                          ortho_mode=str(args.ortho),
                          ortho_bias=float(args.ortho_bias),
                          ortho_pin_secs=0.0)
        # Pool deliberately non-empty and the pin clock wide open: the
        # MODE alone has to hold the gate down, so a draw ordered ahead
        # of the `armed` conjunct has nothing else stopping it.
        f._ortho_pool = list(cells[:4])
        f._ortho_ids = {c.key for c in f._ortho_pool}
        for _ in range(n):
            for c in cells:
                c.times_chosen = 0
            assert f.select() is not None
        assert f._ortho_selections == 0
        assert f._ortho_pool, "the pool must survive a disarmed refresh"
        assert tally.calls == want, f"{sel_mode} stream shifted: {tally.calls}"

    # Third leg: ARMED, but the pool is empty (deepest area holds no
    # state-bearing cell — the castle-spiral case). Emptiness has to
    # short-circuit the draw away as well, or an on-arm run's stream
    # depends on whether the pool happened to be populated that tick.
    cells = [_ocell(gx, yb) for gx in range(6) for yb in range(3)]
    tally = _DrawTally(0)
    f = _ortho_solver(cells, sel_mode="legacy", rng=tally, max_area=1,
                      ortho_mode="up", ortho_bias=0.30, ortho_pin_secs=0.0)
    for _ in range(n):
        assert f.select() is not None
    assert f._ortho_armed() is True and f._ortho_pool == []
    assert tally.calls == {"random": n, "integers": n}, \
        f"an empty pool cost RNG: {tally.calls}"


def test_the_ortho_gate_short_circuits_before_it_touches_the_rng():
    # Source-level companion to the ledger, aimed at the exact edit that
    # breaks default-off identity: the draw must be the LAST conjunct of
    # a plain `and` chain. Stated structurally because the ledger can
    # only observe the orderings its own fixtures reach, while this holds
    # for every archive shape the campaign will ever see.
    gate = _ortho_gate_test()
    assert isinstance(gate, ast.BoolOp) and isinstance(gate.op, ast.And), \
        "the ortho gate is no longer a short-circuiting `and` chain"

    def _draws(node) -> bool:
        return any(isinstance(x, ast.Attribute) and x.attr == "rng"
                   for x in ast.walk(node))

    drawing = [i for i, v in enumerate(gate.values) if _draws(v)]
    assert drawing == [len(gate.values) - 1], (
        "the rng draw must be the last conjunct of the ortho gate; "
        f"conjuncts touching self.rng: {drawing} of {len(gate.values)}")
    # And the cheap predicates it hides behind are the two that make the
    # arm off-by-default: the pin/mode gate and a non-empty pool.
    guarded = ast.unparse(ast.BoolOp(op=ast.And(), values=gate.values[:-1]))
    assert "armed" in guarded and "_ortho_pool" in guarded, guarded


def test_the_ortho_arm_enforces_the_frontier_throttle_on_its_own_pool():
    # R2 lesson, ported to the vertical arm: a column ceiling whose
    # bursts came back empty `throttle` times running is a wall, and the
    # arm would otherwise pound it forever — being the ceiling is what
    # keeps re-selecting it. The skip has to live INSIDE the arm; the
    # pool cache is shared with the count arm's up-weight.
    walls = [_ocell(4, 0, barren=3), _ocell(9, 0, barren=3)]
    floor = [_ocell(4, 6), _ocell(9, 6)]
    f = _ortho_solver(walls + floor, frontier_throttle=3)
    picks = [f.select() for _ in range(200)]
    assert {c.key for c in f._ortho_pool} == {w.key for w in walls}
    assert f._ortho_selections == 0
    assert not any(w.explored for w in walls)
    assert all(p is not None for p in picks)
    # Positive control: the same archive with the throttle disarmed —
    # the arm fires and every pick is a top-of-column cell, so the skip
    # above is the throttle and not a dead arm.
    g = _ortho_solver([_ocell(4, 0, barren=3), _ocell(9, 0, barren=3),
                       _ocell(4, 6), _ocell(9, 6)], frontier_throttle=0)
    gpicks = [g.select() for _ in range(200)]
    assert g._ortho_selections == 200
    assert all(p.key[-2] == 0 for p in gpicks)


def test_the_ortho_pool_is_the_deepest_area_and_the_column_ceiling_composed():
    # Membership is an AND of two independent filters, and each one has
    # to hold on its own: a stale column from a SHALLOWER area is a
    # ceiling nobody is climbing (it would define this wall's frontier
    # and freeze the arm), and a deep cell six bands below its own
    # column ceiling is just the floor of the shaft.
    deep_top = _ocell(4, 1, area=2)
    deep_mid = _ocell(4, 7, area=2)
    deep_other = _ocell(9, 5, area=2)
    stale_taller = _ocell(4, 0, area=1)
    f = _ortho_solver([deep_top, deep_mid, deep_other, stale_taller])
    f._refresh_sel_cache()
    assert {c.key for c in f._ortho_pool} == {deep_top.key, deep_other.key}
    assert f._ortho_ids == {deep_top.key, deep_other.key}
    assert f._ortho_ext == {4: 1, 9: 5}
    # ...but the restriction is scoped to the ARM. The stale cell is not
    # a ceiling anybody is climbing, yet it is still a perfectly good
    # root for the count/legacy arms — see the strand regression below
    # for what happens when this filter is applied to the whole pool.
    assert stale_taller.key in {c.key for c in f._sel_cells}


def test_an_unpopulated_deepest_area_does_not_strand_the_selection_pool():
    # THE strand. record() bumps self.max_area BEFORE its `loops > 6`
    # archive-eligibility early-out, so a castle-maze spiral (4-4/7-4/8-4,
    # the exact case the loop counter exists for) that crosses an area
    # boundary advances max_area with ZERO state-bearing cells in the new
    # area. When the deepest-area filter was applied to self._sel_cells,
    # the pool emptied, select() returned None, and _assign reset EVERY
    # worker to the entrance root — permanently, because entrance bursts
    # archive into the earlier area that the same filter discards. The
    # ortho arm must degrade to "no arm", never to "no search": with the
    # flag off this archive is fully selectable, so with it on it must be
    # too.
    cells = [_ocell(10 + i, 5, area=3) for i in range(6)]
    for mode in ("off", "up", "down"):
        f = _ortho_solver(cells, ortho_mode=mode, max_area=4)
        f._refresh_sel_cache()
        assert len(f._sel_cells) == 6, mode
        assert [f.select() for _ in range(50)].count(None) == 0, mode
        # The arm itself correctly finds nothing to climb in an area with
        # no cells, and says so rather than borrowing a stale ceiling.
        if mode != "off":
            assert f._ortho_pool == [] and f._ortho_ids == set(), mode
            assert f._ortho_ext == {} and f._ortho_deep_yband is None, mode
            assert f._ortho_armed() is True and f._ortho_selections == 0
    # And it recovers the moment the new area is actually populated.
    arrived = _ocell(3, 2, area=4)
    g = _ortho_solver(cells + [arrived], ortho_mode="up", max_area=4)
    g._refresh_sel_cache()
    assert len(g._sel_cells) == 7
    assert {c.key for c in g._ortho_pool} == {arrived.key}


def test_the_shallower_area_stays_selectable_by_the_other_arms():
    # The count arm is what carries the run while the ortho arm has
    # nothing to climb, so it must see the SAME pool it sees with the
    # flag off — the area filter is the arm's, not the sampler's.
    shallow = [_ocell(4 + i, 5, area=1) for i in range(8)]
    deep = [_ocell(2, 3, area=2)]
    on = _ortho_solver(shallow + deep, sel_mode="count", ortho_bias=0.0,
                       ortho_mode="up")
    off = _ortho_solver(shallow + deep, sel_mode="count", ortho_bias=0.0,
                        ortho_mode="off")
    on._refresh_sel_cache()
    off._refresh_sel_cache()
    assert {c.key for c in on._sel_cells} == {c.key for c in off._sel_cells}
    assert on._sel_maxscore == off._sel_maxscore
    assert {c.key for c in on._sel_deep} == {c.key for c in off._sel_deep}
    assert on._sel_topgx == off._sel_topgx
    picks = {c.key[-5] for c in (on.select() for _ in range(400))}
    assert picks == {1, 2}


# --- the arm's three downstream call sites ---------------------------
# Pool membership above is only half the mechanism: the count arm has to
# up-weight the same set with an exact Wmax, a fresh climb has to
# invalidate the selection cache the moment it happens, and the bursts
# rooted there have to roll the profile's hold macros at the ortho rate.
# Each of those is a different call site, and each is silently inert if
# it is wired wrong — the arm would still "work" and still find nothing.

def test_the_count_arm_up_weights_the_pool_with_a_matching_wmax():
    # The multiplier and the Wmax it is bounded by must move together;
    # splitting them turns exact rejection sampling into a silently
    # truncated prior. Source-level because the pair is an inline
    # expression in the hot selection loop.
    import scripts.go_explore_solve as ges

    arm = Path(ges.__file__).read_text().split(
        'if self.sel_mode == "count":', 1)[1].split("# Legacy:", 1)[0]
    assert "wmax = count_wmax(dw, ow)" in arm
    assert "1.1 * max(dw, 1.0)" not in arm     # the literal it replaced
    assert "ow = self.ortho_weight if armed else 1.0" in arm
    assert "w *= ow" in arm


def test_the_count_arm_pulls_toward_the_pool_only_while_the_arm_is_armed():
    # Behavioural check of the same pair, and of the count prior's actual
    # shape: visits accumulate against w/sqrt(n+1), so a weight w settles
    # at w**(2/3) times an equally-scored ordinary cell's visits (~2.5x
    # at the shipped 4.0, NOT 4x — the prior damps its own multiplier).
    # Unarmed, the same archive has to spread visits evenly.
    # ortho_bias 0 isolates the count arm from the ortho arm itself.
    def _run(mode, weight):
        # One column: its ceiling (y band 0) is the pool, the 20 cells
        # further down it are not.
        cells = [_ocell(4, 0)] + [_ocell(4, 6 + i) for i in range(20)]
        f = _ortho_solver(cells, sel_mode="count", ortho_bias=0.0,
                          ortho_mode=mode, ortho_weight=weight)
        for _ in range(4000):
            f.select()
        others = [c.times_chosen for c in cells[1:]]
        return cells[0].times_chosen, sum(others) / len(others)

    top_on, other_on = _run("up", 4.0)
    top_off, other_off = _run("off", 4.0)
    assert top_on > 2.0 * other_on
    assert top_off < 1.3 * other_off


def test_the_refresh_reports_the_ceiling_inside_the_deep_band_only():
    # ortho_deep_yband is the pre-registered gate figure, so it must be
    # the extreme y-band among the gx buckets the PRIMARY arm samples
    # (>= topgx-24). A climb far behind the frontier cannot flatter it —
    # which is the whole point of quoting it rather than the global best.
    f = _ortho_solver([_ocell(1, 2), _ocell(40, 9)])
    f._refresh_sel_cache()
    assert f._ortho_deep_yband == 9
    f = _ortho_solver([_ocell(1, 2), _ocell(40, 9), _ocell(40, 7)])
    f._refresh_sel_cache()
    assert f._ortho_deep_yband == 7


def test_the_refresh_counts_the_column_ceilings_this_run_has_pushed():
    # Cumulative and monotone: the arm's partial-progress signal (the
    # pre-registered partial gate is >= 8 columns), independent of
    # whether a solution ever lands.
    f = _ortho_solver([_ocell(4, 9), _ocell(9, 9)])
    f._refresh_sel_cache()
    assert f._ortho_cols_improved == 0          # first pass = the baseline
    climbed = _ocell(4, 6)
    f.archive.cells[climbed.key] = climbed
    f._refresh_sel_cache()
    assert f._ortho_cols_improved == 1
    f._refresh_sel_cache()                      # nothing new
    assert f._ortho_cols_improved == 1


def test_a_new_best_yband_forces_the_selection_cache_to_rebuild():
    # observe() needs a real Pool and archive, so guard it at the source.
    # Without the invalidation a freshly-climbed cell waits for the
    # 2%-growth trigger (~2 min at CV's cell rate) before it can be
    # selected at all — the arm could never compound its own progress.
    import scripts.go_explore_solve as ges

    body = Path(ges.__file__).read_text().split(
        "    def observe(", 1)[1].split("    def _dump_solution(", 1)[0]
    blk = body.split('if self.ortho_mode != "off":', 1)[1].split(
        "# Domination score", 1)[0]
    assert "self._ortho_best = yb" in blk
    assert "self._sel_cells = None" in blk


def test_ortho_rooted_bursts_are_flagged_and_roll_the_ortho_macro_rate():
    # A stair mount IS a sustained hold, and the profile-declared macros
    # fire at p=0.02 — the one maneuver a climb needs is the rarest thing
    # the sampler emits, exactly where it matters. The ctx flag is set at
    # assignment; the roll reads it per step (cheap checks first, so the
    # default path costs one float comparison).
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    assert '"ortho": cell.key in self._ortho_ids' in src
    roll = src.split("_mp = (self.ortho_macro_p", 1)[1].split(
        "mi = int(self.rng.choice", 1)[0]
    assert 'c.get("ortho")' in roll
    assert "self._ortho_armed()" in roll
    assert "self.rng.random() < _mp" in roll


def _ortho_progress_solver(tmp_path, mode: str) -> SimpleNamespace:
    fake = _fake_solver(tmp_path, n_cells=5)
    fake.ortho_mode = mode
    fake._ortho_best = 4
    fake._ortho_deep_yband = 9
    fake._ortho_pool = [1, 2, 3]
    fake._ortho_selections = 17
    fake._ortho_cols_improved = 2
    fake._pin_time = 0.0
    return fake


def test_progress_line_emits_the_ortho_telemetry_when_the_arm_is_on(tmp_path):
    # The kill criterion is read straight off these fields (45 min armed,
    # >20k selections, ceiling unmoved => the arm is inert, revert), so a
    # missing field is a run nobody can adjudicate.
    Solver.progress_line(_ortho_progress_solver(tmp_path, "up"), 10.0)
    line = json.loads(
        (tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert line["ortho_best_yband"] == 4
    assert line["ortho_deep_yband"] == 9
    assert line["ortho_pool"] == 3
    assert line["ortho_selections"] == 17
    assert line["ortho_cols_improved"] == 2
    assert line["pinned_secs"] > 0


def test_progress_line_stays_silent_about_ortho_when_the_arm_is_off(tmp_path):
    # Every banked run's progress.jsonl schema is unchanged by default.
    Solver.progress_line(_ortho_progress_solver(tmp_path, "off"), 10.0)
    line = json.loads(
        (tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert not [k for k in line if k.startswith("ortho")]
    assert "pinned_secs" not in line


# ---------------------------------------------------------------------
# R2 barren bookkeeping: the CREDIT half of the frontier throttle
#
# --frontier-throttle N retires a cell "whose bursts came back empty N
# times IN A ROW". _assign() debits the source cell on every burst and
# reads prev["yielded"] to credit the productive ones — but nothing in
# the solver ever WROTE that key, so the credit branch was dead and
# `barren` was really "times this cell was ever selected". Every cell in
# the archive therefore went permanently barren after N selections,
# which silently retired the deep-frontier band AND the orthogonal arm:
# the ortho candidate loop skips barren cells and then has no pick left
# to return, so it falls through with zero selections, forever.
#
# Receipt: runs/bubble_bobble/r68_retry_ortho.log — --sel-mode count
# --frontier-throttle 3 --ortho up --ortho-pin-secs 60, armed for the
# full 1,800 s with ortho_pool 18, and ortho_selections 0 in every one
# of the 30 progress lines. Nothing about it was count-specific: the arm
# sits above the sel_mode dispatch and starved identically in both.
# ---------------------------------------------------------------------

SEL_MODES = ("legacy", "count")


class _StubArchive:
    """Archive stand-in for the burst loop: holds the duck-typed cells
    select() samples, and reports "the archive learned something" on
    every `learn_every`-th record. learn_every 25 is the receipt's own
    regime — an archive flat at 96 cells that was still banking ~670
    dominating improvements; 0 is a genuinely dead frontier."""

    def __init__(self, cells, learn_every: int) -> None:
        self.cells = {c.key: c for c in cells}
        self.learn_every = learn_every
        self.records = 0

    def record(self, ram, state, score, steps, key=None) -> bool:
        self.records += 1
        return self.learn_every > 0 and self.records % self.learn_every == 0


# observe() calls exactly these. A profile-backed GenericGame would drag
# a ROM-shaped RAM map in for no extra coverage: cell_fn returns the same
# key _ocell(4, 0) carries, so each burst re-reaches a known cell and the
# credit is decided by the archive, which is the half under test.
_BURST_GAME = SimpleNamespace(
    is_dead=lambda ram, lives: False,
    is_finale=lambda wd, ram: False,
    is_clear=lambda wd, ram, ctx: False,
    level_key=lambda ram: (0,),
    progress=lambda ram: 68,
    progress_cap=10_000,
    area=lambda ram: 0,
    y=lambda ram: 0,
    cell_fn=lambda ram: (0, 3, 1, 0, 4),
    score_bonus=lambda ram: 0.0,
)


def _burst_cells():
    """The receipt's shape: a dozen columns, eight bands deep, one area."""
    return [_ocell(gx, yb) for gx in range(12) for yb in range(8)]


def _burst_solver(cells, *, learn_every: int = 25, **over):
    """_ortho_solver plus the burst loop the credit actually flows
    through: observe() produces it, _assign() consumes it. max_gx_in_area
    is pre-seeded past the stub's progress so observe() never re-arms the
    pin clock mid-test (a moving frontier would disarm the arm for its
    own, legitimate reason and mask the one under test)."""
    f = _ortho_solver(cells, **over)
    f.archive = _StubArchive(cells, learn_every)
    f.traces = {c.key: ("entrance", b"", 0, (), 0, (), 0) for c in cells}
    f.pool = SimpleNamespace(load_worker_state=lambda *a: None,
                             save_worker_state=lambda wid: b"blob")
    f.weights = np.array([0.5, 0.5])
    f.args.burst = 200
    f.args.root_state = "unused.state"
    f.game = _BURST_GAME
    f.start_wd, f.start_lives = (0,), 5
    f.max_gx_in_area = {0: 1000}
    f.time_bins = f.kill_key = False
    f._recorded_new = False
    f._ortho_best, f._ortho_time = None, 0.0
    for name in ("_assign", "observe"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def _run_bursts(f, bursts: int, *, burst_len: int = 4,
                strip_credit: bool = False) -> list:
    """Drive observe() -> _assign() the way explore() does: one ctx per
    burst, novelty credited through the ctx that burst carries, a fresh
    ctx on reassignment. `strip_credit` deletes the key on the way out,
    which is precisely the pre-fix code path — so the same harness
    measures both sides of the defect. Returns the keys picked."""
    ram = bytes(2048)
    picked = []
    ctx = f._assign(0)
    for _ in range(bursts):
        for step in range(burst_len):
            f.observe(0, ram, ctx["trace"], step, ctx["root"], ctx["loops"],
                      ctx["sig"], ctx["sect"], ctx["psig"], ctx=ctx)
        if strip_credit:
            ctx.pop("yielded", None)
        ctx = f._assign(0, prev=ctx)
        picked.append(ctx["key"])
    return picked


def test_assign_credits_a_productive_burst_and_debits_a_dry_one():
    # The consumer half of the contract, stated on its own: "N times IN A
    # ROW" is a run length, so any novelty has to zero the counter rather
    # than merely stop incrementing it.
    cell = _ocell(4, 0)
    f = _burst_solver([cell])
    dry = {"key": cell.key}
    f._assign(0, prev=dry)
    f._assign(0, prev=dry)
    assert cell.barren == 2
    f._assign(0, prev=dict(dry, yielded=True))
    assert cell.barren == 0
    f._assign(0, prev=dry)
    assert cell.barren == 1


def test_observe_credits_the_burst_that_taught_the_archive_something():
    # THE producer, and THE bug: `yielded` had a reader and no writer, so
    # the counter above could only ever climb. Novelty is exactly what
    # the archive reports back — a new cell or a dominating improvement.
    f = _burst_solver([_ocell(4, 0)], learn_every=1)
    ctx: dict = {}
    assert f.observe(0, bytes(2048), [0], 1, "entrance", ctx=ctx) == "live"
    assert ctx.get("yielded") is True

    # A burst that only re-visits: no credit, so the debit stands.
    g = _burst_solver([_ocell(4, 0)], learn_every=0)
    dry: dict = {}
    assert g.observe(0, bytes(2048), [0], 1, "entrance", ctx=dry) == "live"
    assert "yielded" not in dry

    # The seed call passes ctx=None (scripts/go_explore_solve.py:1597).
    assert f.observe(0, bytes(2048), [0], 1, "entrance") == "live"


@pytest.mark.parametrize("sel_mode", SEL_MODES)
def test_the_throttle_does_not_permanently_retire_the_ortho_arm(sel_mode):
    # THE reproduction, end to end at the layer the receipt measured:
    # observe -> ctx -> _assign -> select, under the r68 flags. The
    # warm-up is long enough that every cell has been picked well past
    # the throttle, which is the state the live run reached inside its
    # first 60 s — i.e. before the pin gate even opened.
    f = _burst_solver(_burst_cells(), sel_mode=sel_mode,
                      frontier_throttle=3, ortho_band=2, learn_every=25)
    _run_bursts(f, 1000)
    warm = f._ortho_selections
    _run_bursts(f, 3000)
    assert f._ortho_selections > warm, (
        f"{sel_mode}: the ortho arm made no selection after warm-up "
        "while the archive was still learning")

    # And the pre-fix path through the identical harness: zero, forever,
    # in both sel modes — the receipt's signature.
    g = _burst_solver(_burst_cells(), sel_mode=sel_mode,
                      frontier_throttle=3, ortho_band=2, learn_every=25)
    _run_bursts(g, 1000, strip_credit=True)
    stalled = g._ortho_selections
    _run_bursts(g, 3000, strip_credit=True)
    assert g._ortho_selections == stalled
    assert g._ortho_armed() is True and g._ortho_pool


@pytest.mark.parametrize("sel_mode", SEL_MODES)
def test_a_genuinely_barren_frontier_still_retires_the_arm(sel_mode):
    # The other direction: the throttle is not being disabled. An archive
    # that learns NOTHING is a real wall, and the arm must still stop
    # pounding it — otherwise the fix would just delete R2.
    f = _burst_solver(_burst_cells(), sel_mode=sel_mode,
                      frontier_throttle=3, ortho_band=2, learn_every=0)
    _run_bursts(f, 1000)
    warm = f._ortho_selections
    _run_bursts(f, 2000)
    assert f._ortho_selections == warm


@pytest.mark.parametrize("sel_mode", SEL_MODES)
def test_the_armed_ortho_arm_fires_under_every_sel_mode(sel_mode):
    # The hypothesis the receipt invited — "the arm is only wired into
    # the legacy path" — pinned false. The arm sits ABOVE the sel_mode
    # dispatch in select(), so both modes reach it identically.
    cells = [_ocell(4, 0), _ocell(9, 0), _ocell(4, 6), _ocell(9, 6)]
    f = _ortho_solver(cells, sel_mode=sel_mode)
    picks = [f.select() for _ in range(200)]
    assert f._ortho_selections == 200
    assert all(p is not None and p.key[-2] == 0 for p in picks)


def test_the_arming_matrix_covers_every_sel_mode_the_cli_offers():
    # A third mode added without extending SEL_MODES would ship an arm
    # nobody proved fires there — which is the exact class of gap the
    # r68 log took 30 minutes of live compute to surface.
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--sel-mode"):
            choices = next(k.value for k in node.keywords
                           if k.arg == "choices")
            assert tuple(ast.literal_eval(choices)) == SEL_MODES
            break
    else:
        pytest.fail("--sel-mode is no longer a CLI flag")


@pytest.mark.parametrize("sel_mode", SEL_MODES)
def test_the_credit_is_inert_at_the_shipped_throttle_default(
        monkeypatch, sel_mode):
    # Default-off identity for the credit itself: --frontier-throttle
    # defaults to 0, and nothing reads `barren` at 0, so writing the key
    # cannot move a single pick on any run that did not opt into R2.
    # Same seed, same archive, credit on vs credit stripped: identical
    # pick sequences, not merely similar ones.
    args = _parse_solver_argv(monkeypatch)
    assert args.frontier_throttle == 0

    def _picks(strip: bool) -> list:
        f = _burst_solver(_burst_cells(), sel_mode=sel_mode,
                          frontier_throttle=args.frontier_throttle)
        return _run_bursts(f, 400, strip_credit=strip)

    assert _picks(True) == _picks(False)
