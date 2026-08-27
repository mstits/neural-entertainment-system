"""Tests for scripts/go_explore_solve.py's pure helpers and GenericGame.

Kept separate from tests/test_go_explore.py (which covers the archive
in src/training/go_explore.py). GenericGame's __init__ only reads the
profile dict (no ROM/Pool I/O), so it's constructible here with a
minimal in-memory profile; Solver itself needs a real ROM/Pool and is
only exercised via progress_line() through a duck-typed stand-in.
"""

from __future__ import annotations

import ast
import copy
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
    band_cell_count,
    band_growth_stalled,
    check_state_sidecar,
    column_extremes,
    count_wmax,
    gate_armed,
    gate_axes_sidecar_sha,
    gate_counters,
    gate_run_header,
    gate_suppress_trace,
    hw_provenance,
    in_lock_key,
    inversion_armed,
    lock_armed,
    macro_slot_owner,
    merge_gate_axes,
    ortho_armed,
    ortho_pool,
    resolve_gate_pin_secs,
    resolve_hw_flags,
    resolve_inversion_pin_secs,
    resolve_verify_bank,
    stamp_stats_provenance,
    update_stall,
    write_state_sidecar,
)
from src.training import interaction_basis as ib
from src.training.profile_utils import action_space_to_bitmasks


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
        # D1: _refresh_sel_cache's deep-arm relax. 0 = the pre-D1
        # `key[0] == max_sect` filter exactly.
        transit_deep_relax=0,
        door_weight=0.0, _doors=frozenset(), _key_ids={},
        ortho_mode="up", ortho_pin_secs=0.0, ortho_bias=1.0,
        ortho_band=1, ortho_weight=4.0, _pin_time=0.0,
        # The gate-opener arm at its shipped defaults: off, and its count
        # -arm multiplier at the identity so Wmax is the legacy ceiling.
        gate_mode="off", gate_weight=1.0,
        _ortho_pool=[], _ortho_ids=set(), _ortho_ext={},
        _ortho_deep_yband=None, _ortho_selections=0,
        _ortho_cols_improved=0,
        # Banked torn-read buckets the resume refreeze refused: empty on
        # every run that did not resume, which is every fixture here.
        _gx_phantoms=set(),
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
    assert "wmax = count_wmax(dw, ow, gw)" in arm
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
        self._merit: dict = {}

    def record(self, ram, state, score, steps, key=None, merit=None) -> bool:
        # `merit` mirrors GoExploreArchive.record()'s contract just enough
        # that observe() (which always passes the kwarg now, None off the
        # lock arm) can call this stand-in unmodified; it is not asserted
        # on by the pre-lock tests, which all run with lock_mode absent
        # (=> "off" => merit=None on every call).
        if merit is not None and key is not None:
            self._merit[key] = merit
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
    # D2: observe()'s score-site literal, now an attribute. 10000 is the
    # pre-D2 hardcoded constant (TRANSIT_SCORE_WEIGHT).
    f.transit_weight = 10000
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


# ---------------------------------------------------------------------
# The death-blip debounce (Rygar, 2026-08-26)
# ---------------------------------------------------------------------
#
# MEASURED, not assumed. Replaying the deepest banked Rygar trajectory
# (runs/rygar_ceiling_2026-08-26, 3,607 actions, odometer x 0 -> 5,093,
# reproduced to the pixel) against the profile's declared death byte
# $0303 gives two cleanly separated shapes:
#
#   * 17 TRANSIENT dead reads, every one exactly 2 steps long, after
#     which the byte returns to 1 and the camera keeps scrolling. The
#     first lands at x = 1,536 — the exact odometer position the earlier
#     un-debounced run walled at.
#   * TERMINAL dead reads that never recover: a scripted forward hold
#     dies at step 138 and the byte still reads 0 at step 6,000, with
#     RAM churn frozen at a single value and `right` no longer scrolling
#     the camera.
#
# So a 2-step dead read is a transition blip and a >= 3-step one is a
# death, and the >= 3 threshold sits exactly on that boundary with one
# step of margin. `_dead_mm` had no test at all before this: deleting
# the debounce broke nothing in the suite while costing the search
# 5,093 - 1,536 = 3,557 px of frontier, a 3.3x deeper reach, on the one
# game that was chosen BECAUSE its progress signal is sound.

_ALIVE = bytes(2048)
_DEAD = b"\x01" + bytes(2047)


def _blip_solver():
    """_burst_solver whose death predicate is driven by the RAM byte the
    caller feeds, so a dead/alive pattern can be scripted step by step.
    The archive's cell key matches _BURST_GAME.cell_fn and outscores the
    banked cell, so EVERY observation that reaches the recording path
    records — `archive.records` is therefore a direct count of the steps
    observe() actually banked."""
    f = _burst_solver([_ocell(4, 0)], learn_every=0)
    f.game = SimpleNamespace(**vars(_BURST_GAME))
    f.game.is_dead = lambda ram, lives: ram[0] == 1
    return f


def _drive(f, pattern, ctx):
    return [f.observe(0, _DEAD if d else _ALIVE, [], i, "entrance", ctx=ctx)
            for i, d in enumerate(pattern)]


def test_observe_rides_out_a_two_step_death_blip_and_banks_none_of_it():
    # Rygar's door: 2 dead reads, then live play resumes. Both halves
    # matter — the lineage must SURVIVE (or the search walls at the first
    # door), and the blip steps must bank NOTHING (or the archive fills
    # with cells minted mid-transition).
    f = _blip_solver()
    ctx: dict = {}
    assert _drive(f, [0, 1, 1, 0, 0], ctx) == ["live"] * 5
    assert f.archive.records == 3, "the two blip steps must not be banked"
    assert "_dead_cause" not in ctx
    # ...and the run counter is a RUN counter: live play rearms it, so a
    # second door later in the same lineage gets the same two steps.
    assert ctx["_dead_mm"] == 0
    assert _drive(f, [1, 1, 0], ctx) == ["live"] * 3


def test_observe_kills_a_lineage_on_the_third_consecutive_dead_read():
    # The other side of the same threshold: a real death must still end
    # the lineage, and must say WHY, because _assign() retires the source
    # cell only for a "lives" death (a "key" death is a warp).
    f = _blip_solver()
    ctx: dict = {}
    assert _drive(f, [1, 1, 1], ctx) == ["live", "live", "dead"]
    assert ctx["_dead_cause"] == "lives"
    assert f.archive.records == 0


def test_the_burst_loop_actually_passes_a_ctx_to_observe():
    # A debounce that exists but is not wired is worth nothing: the whole
    # mechanism lives behind `if ctx is not None`, so explore()'s call is
    # the load-bearing line. Pinned at the source because explore() needs
    # a ROM and a Pool to run.
    import inspect

    body = inspect.getsource(Solver.explore)
    call = body.split("status = self.observe(", 1)[1].split(")", 1)[0]
    assert "ctx=c" in call, (
        "explore() must hand observe() the burst ctx — without it every "
        "transition blip reads as a death on its first frame")
    # And the seed observation is the ONE deliberate exception: a single
    # observation of the entrance has no run to debounce.
    assert 'self.observe(0, r, [], 0, "entrance")' in inspect.getsource(
        Solver.seed)


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


# ---------------------------------------------------------------------
# GATE-OPENER ARM
#
# The arm enumerates CONTROLLER INTERACTIONS off archive savestates at a
# pinned boundary and ledgers what they move in RAM. Everything it reads
# is the run's own: archive blobs as roots, the profile's declared action
# list as the alphabet, a paired NOOP control as the nuisance filter, the
# run's own boundary histogram as the prior. No address allowlist, no
# map, no answer key — the known bytes exist in this campaign only as
# grading truth behind the tuner/grader wall, and none of them appears
# anywhere below.
#
# Two contracts are pinned here and they are not the same: the SEMANTICS
# when the arm is on, and BYTE IDENTITY when it is off. The second is the
# one every banked receipt depends on.
# ---------------------------------------------------------------------

def _arm_gate(f, **over):
    """Give a duck-typed solver the gate-opener attribute surface. The
    defaults leave the arm inert (mode off, pin -1 = never arms); pass
    `gate_mode="enumerate"` to turn the machinery on."""
    d = dict(gate_mode="off", gate_pin_secs=-1.0, gate_target_typed=False,
             gate_band=24, gate_sweep_frac=0.10, gate_sweep_roots=16,
             gate_sweep_repeats=1, gate_sham_roots=1, contact_bits=3,
             gate_weight=1.0, gate_axes=[], gate_axes_sha=None,
             _gate_rng=np.random.default_rng(0), _gate_basis=[],
             _gate_phases=(8,), _gate_swept={}, _gate_admitted=[],
             _gate_shadow={}, _gate_positions=set(), _gate_band_hist=[],
             _gate_marks_kind="pattern", _gate_obs_n=0, _gate_fdr_m=None,
             _boundary_hist=None, _boundary_hist_total=0,
             _boundary_rows=None, _gate_boundary_hit=False,
             _gate_change=None, _gate_change_n=0, _gate_prev_vals=None,
             # The sweep's own copies of both counters (D-PRIOR repair):
             # the live band sampler cannot reach the band tail the sweep
             # roots in, so the sweep feeds the same statistics itself.
             _gate_change_sweep=None, _gate_change_sweep_n=0,
             _gate_baseline_frames=0, _gate_rank_stats={},
             # ...and the de-replicator that keeps those counts honest:
             # every program at a root replays the same undirected
             # frames, so they are folded once per (trajectory, index).
             _gate_baseline_len={}, _gate_baseline_prev={},
             _gate_root_sig={},
             gate_arm_cadence=60.0, _gate_last_ckpt=0.0, _gate_band_last=0,
             _gate_inject=[], _gate_inject_i=0,
             bitmasks=[0, 1],
             _gate_next_sweep=0, _gate_last_pin=0.0, _gate_disarmed=False,
             _gate_armed_secs=0.0, _gate_armed_since=None,
             _gate_axes_live=None, steps_done=0, _pin_time=0.0,
             _gate_counters=gate_counters())
    d.update(over)
    for k, v in d.items():
        setattr(f, k, v)
    for name in ("_gate_armed", "_gate_observe", "_gate_novelty",
                 "_gate_farmability", "_gate_farm_sources", "_gate_roots",
                 "_gate_maybe_sweep", "_gate_hist_alloc",
                 "_gate_baseline_sample", "_gate_baseline_fold",
                 "_gate_root_signature", "_gate_liveness",
                 "_gate_checkpoint",
                 "_gate_sweep", "_gate_wave", "_gate_pass_b", "_gate_admit",
                 "_gate_queue_injections", "shadow_yield"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    f._gate_rank = Solver._gate_rank          # a staticmethod: pure
    return f


# --- 1/8 solver: the arming predicate ---------------------------------

def test_the_gate_arm_short_circuits_on_every_conjunct_it_declares():
    # Five independent reasons not to arm, each sufficient on its own.
    # The pin-secs one is the subtle member: `now - pin >= -1` is true
    # forever, so the disable sentinel needs its own non-negativity test
    # and not merely the elapsed comparison (the exact shape that made
    # --inversion-pin-secs -1 a real disable).
    ok = dict(mode="enumerate", pin_time=100.0, now=800.0, pin_secs=600.0,
              typed=True, band_growth_ok=True)
    assert gate_armed(**ok) is True
    assert gate_armed(**dict(ok, mode="off")) is False
    assert gate_armed(**dict(ok, mode=None)) is False
    assert gate_armed(**dict(ok, pin_secs=-1.0)) is False
    assert gate_armed(**dict(ok, now=699.9)) is False
    assert gate_armed(**dict(ok, typed=False)) is False
    assert gate_armed(**dict(ok, band_growth_ok=False)) is False
    # Inclusive at the boundary, like ortho_armed: the caller polls on a
    # burst boundary, not on a clock.
    assert gate_armed(**dict(ok, now=700.0)) is True


def test_band_growth_and_band_counting_are_measured_off_our_own_archive():
    # The self-arming bookkeeping §3 specifies: band_cells is one pass
    # over the keys, and the conjunct reads the last three checkpoints
    # against the series' OWN peak (scale-free, so it cannot be tuned by
    # picking a target).
    keys = [(0, 0, 0, (), 0, ()) + (0, 0, 0, 1, gx) for gx in (10, 40, 50)]
    assert band_cell_count(keys, band=24) == 2      # 50 and 40, not 10
    assert band_cell_count(keys, top_gx=50, band=50) == 3
    assert band_cell_count([], band=24) == 0
    # Too short to judge -> False. An arm that fires on one checkpoint
    # is arming on noise.
    assert band_growth_stalled([100, 100, 100]) is False
    assert band_growth_stalled([100, 100, 100, 100]) is True
    # 5% of the peak (1000) is 50; a 60-cell step is still growth.
    assert band_growth_stalled([1000, 1010, 1020, 1080]) is False
    assert band_growth_stalled([1000, 1010, 1020, 1030]) is True


# --- 2/8 solver: the sweep never touches the search stream ------------

def _gate_methods() -> list:
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    return [n for n in cls.body
            if isinstance(n, (ast.FunctionDef,))
            and (n.name.startswith("_gate") or n.name == "shadow_yield")]


def test_no_gate_method_can_reach_the_search_rng():
    # THE isolation invariant, stated structurally so it holds for every
    # archive shape rather than only the ones a fixture reaches: the
    # sweep draws from its own Generator, so scheduling a sweep cannot
    # shift one draw of the search. A single `self.rng` anywhere in the
    # gate methods breaks A/B comparability silently.
    offenders = []
    for fn in _gate_methods():
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and node.attr == "rng"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                offenders.append(f"{fn.name}:{node.lineno}")
    assert offenders == [], f"gate methods touching self.rng: {offenders}"
    # ...and they DO have their own stream.
    assert any(isinstance(n, ast.Attribute) and n.attr == "_gate_rng"
               for fn in _gate_methods() for n in ast.walk(fn))


# --- 3/8 solver: every knob comes off its own flag --------------------

def test_solver_init_wires_every_gate_knob_to_its_own_flag():
    # Same mutation class the ortho lane pinned: a copy-paste that points
    # two knobs at one dest, or hardcodes a default, is caught here and
    # not by a campaign that quietly swept the wrong band.
    for attr, dest in (("gate_mode", "gate_opener"),
                       ("gate_target_typed", "gate_target_typed"),
                       ("gate_band", "gate_band"),
                       ("gate_sweep_frac", "gate_sweep_frac"),
                       ("gate_sweep_roots", "gate_sweep_roots"),
                       ("gate_sweep_repeats", "gate_sweep_repeats"),
                       ("gate_sham_roots", "gate_sham_roots"),
                       ("contact_bits", "contact_bits")):
        val = _init_assignment(attr)
        assert val is not None, f"{attr} is never assigned"
        names = [n.value for n in ast.walk(val)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert dest in names, f"{attr} does not read args.{dest}"
    # The pin is resolved through the helper (so the disable sentinel and
    # the "required when armed" contract live in one place), not inlined.
    pin = _init_assignment("gate_pin_secs")
    assert isinstance(pin, ast.Call) and pin.func.id == "resolve_gate_pin_secs"


# --- 4/8 solver: count_wmax keeps its ceiling exact -------------------

def test_count_wmax_composes_the_gate_multiplier_without_moving_the_default():
    # The default-arg form is the whole point: the 2-argument contract
    # above (line ~863) still holds unchanged, and a third armed
    # multiplier scales the ceiling EXACTLY rather than truncating the
    # prior in silence.
    assert count_wmax(0.0, 1.0) == 1.1
    assert count_wmax(0.0, 1.0, 1.0) == 1.1
    assert count_wmax(0.0, 1.0, 0.0) == 1.1        # floors at 1.0
    assert count_wmax(5.0, 4.0, 2.0) == 44.0
    assert count_wmax(1.0, 1.0, 3.0) == pytest.approx(3.3)


# --- 5/8 solver: the cell key arity is frozen -------------------------

def test_the_armed_arm_never_changes_the_cell_key_arity():
    # E's shadow ledger, and the reason it exists: candidates are
    # recorded OUTSIDE the key, so `cells` and `new_cells` stay
    # comparable between the armed run and its control. An axis that
    # entered the key in-run would make every A/B number meaningless.
    off = _arm_gate(_burst_solver([_ocell(4, 0)], learn_every=1))
    on = _arm_gate(_burst_solver([_ocell(4, 0)], learn_every=1),
                   gate_mode="enumerate", _gate_admitted=[0x0004],
                   _sel_topgx=4)
    ram = bytearray(2048)
    ram[0x0004] = 8
    keys_off, keys_on = [], []
    for f, out in ((off, keys_off), (on, keys_on)):
        ctx: dict = {}
        for step in range(64):
            f.observe(0, bytes(ram), [0], step, "entrance", ctx=ctx)
        out += [len(k) for k in f.archive.cells]
    assert keys_on == keys_off
    assert set(keys_on) == {11}
    # ...and the ledger still saw the candidate, off the key entirely.
    assert on._gate_shadow.get(0x0004)
    assert on.shadow_yield() > 0.0
    assert off._gate_shadow == {}


# --- 6/8 solver: DEFAULT-OFF BYTE IDENTITY (the headline) -------------

def test_the_gate_costs_zero_rng_draws_whether_it_is_off_or_merely_disarmed():
    # Draw-count ledger, exact by construction (same fixture the ortho
    # lane uses). Two legs, because "off" and "armed-capable but not
    # armed" are different code paths and BOTH have to be free:
    #   * mode off      — every gate branch short-circuits on the compare
    #   * mode enumerate, pin -1 — the machinery is live, the arm is not
    # Mutation-proof: moving any draw into a gate branch (an injection
    # probability, a sham-root pick off self.rng, a sampled sweep
    # cadence) shows up here as an integer mismatch, not a flake.
    class _BurstTally(_DrawTally):
        """_DrawTally plus the third draw the record/reassign path makes,
        so the ledger stays EXHAUSTIVE — a draw slipped into a gate
        branch has nowhere to hide."""

        def __init__(self, seed: int = 0) -> None:
            super().__init__(seed)
            self.calls["choice"] = 0

        def choice(self, *a, **k):
            self.calls["choice"] += 1
            return self._rng.choice(*a, **k)

    def _tally(mode: str) -> dict:
        t = _BurstTally(0)
        f = _arm_gate(_burst_solver(_burst_cells(), sel_mode="count"),
                      gate_mode=mode, _sel_topgx=11)
        f.rng = t
        _run_bursts(f, 120)
        return dict(t.calls)

    off, disarmed = _tally("off"), _tally("enumerate")
    assert off == disarmed, f"the gate shifted the stream: {off} vs {disarmed}"
    assert off["random"] > 0 and off["integers"] > 0   # the fixture ran


def test_every_gate_hot_path_short_circuits_on_the_mode_first():
    # Source-level companion, aimed at the exact edit that breaks
    # identity: each gate branch in the two hot loops must lead with the
    # mode compare, so a disarmed run never evaluates anything more
    # expensive (or more stateful) than one string comparison.
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    gates = []
    for fname in ("observe", "explore", "_assign"):
        fn = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == fname)
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            src = ast.unparse(node.test)
            if "gate_mode" in src or "gate_macro" in src:
                gates.append((fname, node, src))
    assert len(gates) >= 4, f"gate branches not found: {gates}"
    for fname, node, src in gates:
        test = node.test
        first = test.values[0] if isinstance(test, ast.BoolOp) else test
        assert "gate_mode" in ast.unparse(first), \
            f"{fname}: gate branch does not lead with the mode: {src}"
        assert "self.rng" not in src, f"{fname}: gate branch draws: {src}"
        # ...and the BODY may not draw either. The condition being clean
        # is not enough: the arm only has to be free on the paths a
        # default run walks, but arm C's whole comparability argument is
        # that taking the macro slot costs the SEARCH stream nothing, and
        # that claim lives inside the branch, not in its test.
        body = "\n".join(ast.unparse(s) for s in node.body)
        assert "self.rng" not in body, \
            f"{fname}: the gate branch body draws from the search stream"


def test_the_sweep_can_only_be_triggered_at_a_burst_boundary():
    # Rev 4 §10-O7. A sweep that fires mid-burst throws away in-flight
    # steps, chops in-flight macros in half and mis-debits `barren` — so
    # the trigger is conjoined with a flag explore() raises ONLY where a
    # worker's burst has just ended, and consuming the flag clears it.
    # Stated structurally because the failure is a deleted conjunct, and
    # a deleted conjunct makes the sweep MORE eager, not less: it would
    # show up as better coverage of the sweep in every other test.
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "explore")
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and "_gate_maybe_sweep" in "\n".join(ast.unparse(s)
                                                   for s in n.body)]
    assert len(guards) == 1, "the sweep trigger moved or multiplied"
    guard = guards[0]
    assert "_gate_boundary_hit" in ast.unparse(guard.test), \
        "the sweep can fire mid-burst: the boundary conjunct is gone"
    assert "_gate_boundary_hit = False" in "\n".join(
        ast.unparse(s) for s in guard.body), \
        "the boundary flag is not cleared as it is consumed"
    # The flag is raised in exactly one place, and it is the burst-ended
    # branch — not the top of the step loop.
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and "_gate_boundary_hit" in ast.unparse(n)
              and ast.unparse(n).endswith("True")]
    assert len(raises) == 1


def test_the_runs_deadline_binds_the_sweep_and_the_clock_is_re_read(tmp_path):
    # A pass-A sweep is tens of thousands of worker-steps and its pass-B
    # tail runs 654-step programs; fired one burst boundary before
    # --minutes expires it overruns by a whole sweep. T1 scores three
    # arms at EQUAL WALL-CLOCK and the arm that can overrun is always
    # the armed one, so the overrun lands on the deciding metric's own
    # side. The deadline therefore reaches the sweep, and explore()
    # re-reads the clock afterwards instead of deciding the run's last
    # branch on a reading taken before thousands of steps ran.
    late = _sweep_solver(tmp_path, n_roots=2)
    late.pool.load_worker_state = lambda i, b: pytest.fail(
        "a sweep started after the run's deadline")
    ctx = [{"key": ("a",)}, {"key": ("b",)}]
    late._gate_maybe_sweep(ctx, 1e9, deadline=1e9)      # inclusive
    late._gate_maybe_sweep(ctx, 1e9 + 5, deadline=1e9)
    assert late._gate_counters["sweeps"] == 0
    assert ctx == [{"key": ("a",)}, {"key": ("b",)}]

    # Inside the window it still fires, and with no deadline at all
    # (--minutes 0) nothing changes.
    inside = _sweep_solver(tmp_path, n_roots=2)
    inside._gate_maybe_sweep([{"key": ("a",)}, {"key": ("b",)}], 1e9 - 1,
                             deadline=1e9)
    assert inside._gate_counters["sweeps"] == 1
    forever = _sweep_solver(tmp_path, n_roots=2)
    forever._gate_maybe_sweep([{"key": ("a",)}, {"key": ("b",)}], 1e9)
    assert forever._gate_counters["sweeps"] == 1

    # The wiring, at the one call site: explore() passes its own
    # deadline and re-reads `now` before the branches that consume it.
    import scripts.go_explore_solve as ges

    body = Path(ges.__file__).read_text().split("def explore(self)", 1)[1]
    call = body[body.index("self._gate_maybe_sweep("):]
    assert call.startswith("self._gate_maybe_sweep(ctx, now, deadline)")
    after = call[:call.index("if deadline and now >= deadline")]
    assert "now = time.time()" in after, \
        "the deadline is decided on a clock reading taken before the sweep"


# --- 7/8 solver: an empty archive is a fall-through, not a crash ------

def test_an_empty_or_stateless_archive_falls_through_instead_of_sweeping():
    # The sweep borrows worker state; it must never be the thing that
    # reassigns a worker. With no state-bearing cell there is nothing to
    # root at, so it declines — no pool call, no ctx mutation, no
    # counter movement.
    f = _arm_gate(_ortho_solver([]), gate_mode="enumerate",
                  gate_pin_secs=0.0, gate_target_typed=True,
                  _gate_band_hist=[10, 10, 10, 10])
    f.pool = SimpleNamespace(
        save_worker_state=lambda i: pytest.fail("touched the pool"),
        load_worker_state=lambda i, b: pytest.fail("touched the pool"))
    f.args.workers = 2
    ctx = [{"key": None}, {"key": None}]
    assert f._gate_armed(1e9) is True
    assert f._gate_roots() == []
    f._gate_maybe_sweep(ctx, 1e9)
    assert ctx == [{"key": None}, {"key": None}]
    assert f._gate_counters["sweeps"] == 0
    # Stateless cells (evicted blobs) are the same case.
    g = _arm_gate(_ortho_solver([_ocell(4, 0)]), gate_mode="enumerate")
    for c in g.archive.cells.values():
        c.state = None
    assert g._gate_roots() == []


# --- 8/8 solver: telemetry appears only once the arm exists -----------

def test_gate_telemetry_is_absent_from_a_default_progress_line(tmp_path):
    # A progress line that grows fields on every run is a line no
    # historical reader can parse. The gate block is silent while the arm
    # is off and complete the moment it is on.
    fake = _fake_solver(tmp_path, n_cells=5)
    Solver.progress_line(fake, 10.0)
    line = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert [k for k in line if k.startswith("gate_")] == []
    assert "band_cells" not in line

    class _Arch(dict):
        """len() for the stall watchdog, .cells for the band count."""

        @property
        def cells(self):
            return self

    (tmp_path / "on").mkdir()
    armed = _fake_solver(tmp_path / "on", n_cells=5)
    armed.archive = _Arch({(0, 0, 0, (), 0, ()) + (0, 0, 0, 1, gx): None
                           for gx in range(30)})
    _arm_gate(armed, gate_mode="enumerate")
    Solver.progress_line(armed, 10.0)
    line = json.loads(
        (tmp_path / "on" / "progress.jsonl").read_text().splitlines()[-1])
    for key in ("gate_armed", "gate_armed_secs", "gate_sweeps",
                "gate_programs", "gate_candidates", "gate_admitted",
                "gate_sham_yield", "gate_cross", "gate_lift_active",
                "gate_lift_ctrl", "gate_sweep_steps", "gate_shadow_yield",
                # arm C's reach: how many admitted candidates the shared
                # macro slot could not carry, and the denominator behind
                # every farmability number in the receipts.
                "gate_injections", "gate_inexpressible",
                # ...and the one field that VOIDS the A/B pair when it is
                # nonzero: a worker the sweep could not hand back intact.
                # It is only useful if a reader sees it without opening a
                # receipt, so it is pinned in the line like the rest.
                "gate_restore_failed",
                "gate_farm_samples", "band_cells"):
        assert key in line, key
    assert line["band_cells"] == 25          # buckets 5..29 within band 24
    assert line["gate_armed"] is False       # pin -1: machinery on, arm off


# ---------------------------------------------------------------------
# interaction basis (6)
# ---------------------------------------------------------------------

_TARGET_BASIS = (("configs/castlevania.yaml", 16, 120, 64, 32, 24),
                 ("configs/bubble_bobble.yaml", 8, 72, 32, 16, 24),
                 ("configs/contra.yaml", 11, 90, 44, 22, 24))


def _profile_actions(rel: str) -> list:
    import yaml

    return yaml.safe_load((Path(__file__).resolve().parents[1] / rel)
                          .read_text())["action_space"]


def test_the_basis_reproduces_its_published_generation_table():
    # FIXTURE DATA, not a recomputation: the ladder is a fixed design
    # constant declared in the module, and this is the table a reader of
    # the campaign doc can check the code against by eye. A three-action
    # space is small enough to write out in full.
    space = [[], ["right"], ["A"]]
    got = [(p.name, p.family, len(p.masks)) for p in ib.interaction_basis(space)]
    want = ([(f"hold_a{a}_h{h}", "hold", h)
             for a in range(3) for h in (4, 12, 36, 108)]
            + [(f"tap_a{a}_{on}on{off}off_x{r}", "tap", (on + off) * r)
               for a in range(3) for on, off, r in ((2, 2, 8), (2, 10, 4))]
            + [(f"combo_a1+a2_h{h}", "combo", h) for h in (4, 12, 36, 108)])
    assert got == want
    # Deterministic: same input, same list, every call.
    assert ib.interaction_basis(space) == ib.interaction_basis(space)


@pytest.mark.parametrize("rel,n_actions,total,n_hold,n_tap,n_combo",
                         _TARGET_BASIS, ids=[t[0] for t in _TARGET_BASIS])
def test_the_per_target_basis_counts_are_exactly_the_declared_ladder(
        rel, n_actions, total, n_hold, n_tap, n_combo):
    # 4|A| holds + 2|A| taps + a 24-combo cap. These are the numbers §6
    # budgets the sweep against, so a ladder change that quietly moves
    # them turns every budget line into fiction.
    space = _profile_actions(rel)
    assert len(space) == n_actions
    basis = ib.interaction_basis(space)
    fams = [p.family for p in basis]
    assert (len(basis), fams.count("hold"), fams.count("tap"),
            fams.count("combo")) == (total, n_hold, n_tap, n_combo)


@pytest.mark.parametrize("rel,n_masks", (("configs/castlevania.yaml", 24),
                                         ("configs/bubble_bobble.yaml", 6),
                                         ("configs/contra.yaml", 20)))
def test_the_combo_cap_buys_distinct_masks_before_it_buys_durations(
        rel, n_masks):
    # FAMILY COLLAPSE. The COMBO family exists to ask which BUTTON
    # COMBINATIONS were never tried at the boundary; the HOLD family
    # already asked the duration question, at every rung, for every
    # declared action. Enumerated pair-outer, COMBO_CAP=24 is exhausted
    # by the first six novel pairs at four durations apiece on EVERY
    # target — 24 programs carrying six masks, with 31 of Castlevania's
    # 37 novel combinations never pressed and the campaign's "we
    # enumerated the interactions" claim false in exactly the direction
    # that matters. Rung-outer spends the same 24 on 24 masks where the
    # space has them, and falls back to depth only where it does not.
    space = _profile_actions(rel)
    basis = ib.interaction_basis(space)
    combos = [p for p in basis if p.family == "combo"]
    assert len(combos) == ib.COMBO_CAP           # the count does not move
    assert len({p.masks[0] for p in combos}) == n_masks
    # Bubble Bobble is the honest exception: eight actions supply six
    # novel pairwise masks in total, so depth is all its cap can buy.
    novel = {a | b for a in action_space_to_bitmasks(space)
             for b in action_space_to_bitmasks(space)} \
        - set(action_space_to_bitmasks(space))
    assert n_masks == min(ib.COMBO_CAP, len(novel))
    # Rung-outer, stated as the order it actually emits in: durations
    # are non-decreasing down the family, and a rung is only opened once
    # the one before it has offered every pair it has.
    lens = [len(p.masks) for p in combos]
    assert lens == sorted(lens)
    assert set(lens) <= set(ib.HOLD_STEPS)
    for rung in sorted(set(lens))[:-1]:
        assert lens.count(rung) == len(novel), (rel, rung)


def test_the_basis_covers_every_declared_action_in_both_solo_families():
    # COVER: no declared input may be unreachable. A basis that skips an
    # action cannot rule that action out, and "we enumerated the
    # interactions" would be false in the one direction that matters.
    for rel, _n, *_ in _TARGET_BASIS:
        space = _profile_actions(rel)
        masks = set(action_space_to_bitmasks(space))
        basis = ib.interaction_basis(space)
        for fam in ("hold", "tap"):
            covered = {p.masks[0] for p in basis if p.family == fam}
            assert covered == masks, (rel, fam)


def test_the_basis_never_invents_a_button():
    # PURITY, mechanically: every mask is a declared mask or the OR of
    # exactly two of them. A basis that reached outside the profile's own
    # action list would be pressing buttons the agent was never given.
    for rel, _n, *_ in _TARGET_BASIS:
        space = _profile_actions(rel)
        masks = list(action_space_to_bitmasks(space))
        legal = set(masks) | {a | b for a in masks for b in masks}
        for p in ib.interaction_basis(space):
            assert set(p.masks) <= legal, (rel, p.name)


def test_every_pass_a_program_is_the_one_uniform_length():
    # 153 = max settle 21 + max pattern 108 + tail 24. Uniform because a
    # whole wave is stepped in lockstep by one step_all loop; the PAD
    # rides past the tail, so the tail keeps its declared length and the
    # capture points stay per-program.
    assert ib.PROGRAM_LEN_PASS_A == 153
    space = _profile_actions("configs/castlevania.yaml")
    for p in ib.interaction_basis(space):
        for phase in ib.SETTLE_PHASES_PASS_A:
            prog = p.program(phase)
            assert len(prog) == 153
            t0, t1, t2 = ib.capture_points(p, phase)
            assert t0 == phase and t1 == phase + len(p.masks)
            assert t2 == t1 + ib.TAIL_PASS_A <= 153
            assert set(prog[:t0]) == {0}                 # settle is NOOP
            assert tuple(prog[t0:t1]) == tuple(p.masks)  # then the pattern
            assert set(prog[t1:]) == {0}                 # tail + pad
    # Pass B's longer tail fits its own uniform length.
    assert ib.PROGRAM_LEN_PASS_B == 34 + 108 + 512


def test_slot_zero_is_the_all_noop_paired_control():
    # NOOP subtraction is meaningless without it, so slot 0 is a
    # GUARANTEE and not an accident of the YAML ordering: every fleet
    # profile happens to declare [] first, and a space that does not
    # still gets a control hoisted in front.
    for rel, _n, *_ in _TARGET_BASIS:
        basis = ib.interaction_basis(_profile_actions(rel))
        assert set(basis[0].masks) == {0}
        assert set(basis[0].program(21)) == {0}
        assert sum(1 for p in basis if set(p.masks) == {0}) == 6   # 4+2 NOOP
    noopless = ib.interaction_basis([["right"], ["A"]])
    assert noopless[0].family == "control" and set(noopless[0].masks) == {0}


# ---------------------------------------------------------------------
# sweep + rank (5)
# ---------------------------------------------------------------------

class _GatePool:
    """Deterministic pool stand-in. Each worker owns a 2 KB RAM holding
    three bytes the instrument has to tell apart:

      $0010  a FAST free-runner — ticks every step whatever the pad does.
      $0011  a SLOW free-runner — input-independent too, but with a
             period of 100 steps, so it is invisible inside a short
             program's window and moves inside a long one. This is the
             nuisance a 4-step paired control cannot subtract from a
             108-step pattern, and it is exactly the shape (moves once,
             stays moved) that classes LATCHED and scores like a gate.
      $0040  the real thing — LATCHES once a nonzero mask has been held
             eight steps running, so only the LONG holds reach it and
             the short ones are the honest no-effect majority every
             sweep is mostly made of. It also takes ONE input-independent
             bump at step 600, which no pass-A program is long enough to
             reach and every pass-B program is: the confirmation pass
             re-classes survivors over a 20x longer tail, so a survivor
             address that free-runs out there gets a nuisance-driven
             class unless the pass re-runs its length twin too.
    """

    SLOW_PERIOD = 100
    LATE_TICK = 600

    def __init__(self, workers: int = 2) -> None:
        self.rams = [bytearray(2048) for _ in range(workers)]
        self.held = [0] * workers
        self.t = [0] * workers
        self.loads: list = []

    def save_worker_state(self, wid: int) -> bytes:
        return bytes(self.rams[wid])

    def load_worker_state(self, wid: int, blob) -> None:
        self.rams[wid] = bytearray(blob)
        self.held[wid] = 0
        self.t[wid] = 0
        self.loads.append(wid)

    def step_all(self, acts):
        out = []
        for i in range(len(self.rams)):
            a = int(acts[i])
            r = self.rams[i]
            self.t[i] += 1
            r[0x10] = (r[0x10] + 1) & 0xFF
            r[0x11] = (self.t[i] // self.SLOW_PERIOD) & 0xFF
            self.held[i] = self.held[i] + 1 if a else 0
            if self.held[i] >= 8:
                r[0x40] = 8
            if self.t[i] == self.LATE_TICK:
                r[0x40] = (r[0x40] + 1) & 0xFF
            out.append((None, None, np.frombuffer(bytes(r), dtype=np.uint8)))
        return out


def _sweep_solver(tmp_path, n_roots: int = 6, columns: int = 12,
                  band: int = 5):
    # Twelve columns and a band of 5: buckets 6-11 are the band the sweep
    # roots at, 0-5 are the freely-advancing region the sham roots (K1's
    # null) come from. SIX band roots because BH is charged against the
    # full addr x family grid: with three, the strongest possible
    # candidate in this fixture reaches p=1e-3 and the instrument would
    # be pinned admitting rows a 6,144-comparison correction refuses.
    # `columns`/`band` widen that geometry for the tests that need the
    # K4 screen to reach its DISTINCT-step floor: a root is worth one
    # all-NOOP trajectory (152 pairs), not one per program.
    cells = [_ocell(gx, 0) for gx in range(columns)]
    for i, c in enumerate(cells):
        c.state = bytes([i]) + bytes(2047)
    f = _arm_gate(_ortho_solver(cells), gate_mode="enumerate",
                  gate_pin_secs=0.0, gate_target_typed=True, gate_band=band,
                  gate_sweep_roots=n_roots, gate_sham_roots=0,
                  _gate_band_hist=[5, 5, 5, 5])
    space = [[], ["right"]]
    full = ib.interaction_basis(space)
    # THREE controls and three candidates, PAIRED BY LENGTH: the ladder
    # emits an all-NOOP program at every length it uses, and NOOP
    # subtraction is only valid between programs that measured the same
    # window. A single 4-step control against a 108-step pattern is the
    # defect this pairing exists to close. Two candidate FAMILIES,
    # because the ranker's background rate is leave-family-out and a
    # single-family sweep can never reach significance.
    f._gate_basis = [full[0], full[3], full[8], full[4], full[7], full[10]]
    assert [(p.name, p.family, len(p.masks)) for p in f._gate_basis] == [
        ("hold_a0_h4", "hold", 4), ("hold_a0_h108", "hold", 108),
        ("tap_a0_2on2off_x8", "tap", 32),
        ("hold_a1_h4", "hold", 4), ("hold_a1_h108", "hold", 108),
        ("tap_a1_2on2off_x8", "tap", 32)]
    assert [not any(p.masks) for p in f._gate_basis] == [
        True, True, True, False, False, False]
    f.bitmasks = list(action_space_to_bitmasks(space))
    f.pool = _GatePool(2)
    f.args.workers = 2
    f.out = tmp_path
    # The sweep reads the profile's OWN position telemetry during settle
    # (the CONTACT admission): gx frozen, y frozen => contact.
    f.game = SimpleNamespace(progress=lambda ram: 100, y=lambda ram: 40,
                             label=lambda k: "-".join(str(x) for x in k))
    return f


def test_the_sweep_restores_worker_state_ctx_and_the_search_rng_verbatim(
        tmp_path):
    # THE state-isolation contract, in its three parts. The sweep runs at
    # a burst boundary and borrows the whole pool; if any of these three
    # moves, the search after a sweep is not the search before it and the
    # A/B pair is void. Stated as three assertions because they fail
    # independently: `_assign` would break all three, a missing ctx
    # deepcopy only the second, an rng draw only the third.
    f = _sweep_solver(tmp_path)
    f.pool.rams[0][0x100] = 0xAB
    f.pool.rams[1][0x100] = 0xCD
    ctx = [{"key": ("a",), "left": 0, "trace": [1, 2, 3], "nested": {"k": 1}},
           {"key": ("b",), "left": 0, "trace": [], "nested": {"k": 2}}]
    want_ctx = copy.deepcopy(ctx)
    want_blobs = [f.pool.save_worker_state(i) for i in range(2)]
    want_rng = copy.deepcopy(f.rng.bit_generator.state)
    cells = list(f.archive.cells.values())
    want_book = [(c.times_chosen, c.explored, c.barren) for c in cells]

    # DIRTY the ctx from inside the sweep. Without this the restore is
    # untestable by inspection — the current sweep happens not to touch
    # ctx, so "ctx[i] = snap" could be deleted and every assertion below
    # would still pass (verified by mutation). A sweep that ever grows a
    # ctx write — a queued injection, a marks append, a bookkeeping
    # counter — must still hand the search back exactly what it borrowed.
    inner = f.pool.step_all

    def _dirty(acts):
        ctx[0]["trace"].append(99)
        ctx[0]["nested"]["k"] = 999
        ctx[1]["left"] = -7
        return inner(acts)

    f.pool.step_all = _dirty
    ids_before = [id(c) for c in ctx]

    ranked = f._gate_sweep(ctx)

    assert [id(c) for c in ctx] != ids_before, "ctx was never restored"

    # Positive control: the sweep really ran the instrument end to end —
    # the stand-in's latched byte survives, neither free-runner does.
    assert 0x40 in {r["addr"] for r in ranked}
    assert 0x10 not in {r["addr"] for r in ranked}
    assert 0x11 not in {r["addr"] for r in ranked}

    assert ctx == want_ctx                                    # 1: ctx dicts
    assert ctx[0]["nested"] is not want_ctx[0]["nested"]      #    (a copy)
    assert [f.pool.save_worker_state(i) for i in range(2)] == want_blobs  # 2
    assert f.rng.bit_generator.state == want_rng              # 3: search rng
    # ...and no archive bookkeeping was debited on the way through.
    assert [(c.times_chosen, c.explored, c.barren) for c in cells] == want_book
    assert f._gate_counters["sweeps"] == 1
    assert f._gate_counters["steps"] > 0


def test_a_worker_the_sweep_cannot_bank_is_never_half_restored(
        tmp_path, capsys):
    # THE DESYNC. The restore guarded the machine half on the blob and
    # handed back the ctx unconditionally, so a worker whose savestate
    # could not be banked came out of the sweep with the search
    # believing it was mid-burst at `snap` while the emulator sat at the
    # end of the last sweep program. Every action after that is appended
    # to a trace that does not describe the frames that ran — and
    # observe() banks that trace as an archive cell, which is a
    # FABRICATED trajectory, the one failure class the whole replay
    # harness exists to catch after the fact. Both or neither.
    f = _sweep_solver(tmp_path, n_roots=2)
    real_save = f.pool.save_worker_state
    f.pool.save_worker_state = lambda i: None if i == 1 else real_save(i)
    reassigned: list = []

    def _assign(i, prev=None):
        reassigned.append((i, prev))
        return {"key": ("fresh",), "left": 7}

    f._assign = _assign
    ctx = [{"key": ("a",), "left": 3}, {"key": ("b",), "left": 3}]
    f._gate_sweep(ctx)

    # Worker 0 banked, so it is handed back verbatim.
    assert ctx[0] == {"key": ("a",), "left": 3}
    # Worker 1 could not be: NEITHER half is restored, and the pair is
    # resynced from a real archive root instead of left disagreeing.
    # `prev` rides along because the abandoned burst is a FINISHED burst
    # as far as R2's credit signal is concerned — the search debits or
    # credits the source cell of every burst it ends, and dropping prev
    # here would exempt exactly the roots the sweep interrupted.
    assert reassigned == [(1, {"key": ("b",), "left": 3})]
    assert ctx[1] == {"key": ("fresh",), "left": 7}
    assert f._gate_counters["restore_failed"] == 1
    # Loudly, because it costs an RNG draw and voids the A/B pair.
    err = capsys.readouterr().err
    assert "worker 1" in err and "ABANDONED" in err


def test_a_failed_reassignment_is_reported_without_masking_the_real_error(
        tmp_path, capsys):
    # The repair runs in a FINALLY, so an exception out of it replaces
    # whatever brought the sweep down: the operator gets a select()
    # traceback from the cleanup and never sees the emulator error that
    # actually ended the run. Both directions are pinned here.
    def _rigged(n_roots=2):
        f = _sweep_solver(tmp_path, n_roots=n_roots)
        real_save = f.pool.save_worker_state
        f.pool.save_worker_state = lambda i: None if i == 1 else real_save(i)

        def _boom_assign(i, prev=None):
            raise ValueError("select() blew up")

        f._assign = _boom_assign
        return f

    # 1. Something else is already unwinding: it propagates untouched,
    #    and the repair failure is shouted with its traceback instead.
    f = _rigged()
    f.pool.step_all = lambda acts: (_ for _ in ()).throw(
        RuntimeError("the emulator died mid-sweep"))
    ctx = [{"key": ("a",), "left": 3}, {"key": ("b",), "left": 3}]
    with pytest.raises(RuntimeError, match="the emulator died mid-sweep"):
        f._gate_sweep(ctx)
    err = capsys.readouterr().err
    assert "select() blew up" in err and "ValueError" in err
    # ...and the workers that COULD be handed back still were: a repair
    # that throws must not abort the rest of the restore either.
    assert ctx[0] == {"key": ("a",), "left": 3}

    # 2. Nothing else in flight: the failure is not swallowed. It is
    #    re-raised wrapped, with the original chained so the traceback
    #    survives.
    g = _rigged()
    ctx = [{"key": ("a",), "left": 3}, {"key": ("b",), "left": 3}]
    with pytest.raises(RuntimeError, match="worker 1 could not be restored"
                       ) as ei:
        g._gate_sweep(ctx)
    assert isinstance(ei.value.__cause__, ValueError)
    assert ctx[0] == {"key": ("a",), "left": 3}


def test_the_sweep_draws_only_from_its_own_reproducible_stream(tmp_path):
    # Independence, behaviourally: two solvers seeded alike sweep alike,
    # and a sweep leaves the search stream bit-identical however many
    # sham roots it drew. (The structural half — no `self.rng` anywhere
    # in a gate method — is pinned above.)
    outs = []
    for _ in range(2):
        f = _sweep_solver(tmp_path, n_roots=2)
        f.gate_sham_roots = 1
        f._gate_rng = np.random.default_rng(1234)
        f.rng = np.random.default_rng(99)
        before = copy.deepcopy(f.rng.bit_generator.state)
        roots = f._gate_roots()
        outs.append([(c.key, sham) for c, sham in roots])
        assert f.rng.bit_generator.state == before
    assert outs[0] == outs[1]
    assert any(sham for _k, sham in outs[0]), "no sham root was drawn"


def test_the_wave_labels_every_capture_with_the_window_it_measured(tmp_path):
    # The sweep is the only place that knows which program a capture came
    # from, so it is the only place that can say "this one was a control"
    # and "this one spanned N pattern steps". Those two labels are the
    # ranker's ONLY means of pairing an observation with a control that
    # covers the same t0..t2 — drop either and the pairing silently
    # degrades to one bucket per root.
    f = _sweep_solver(tmp_path)
    cell = next(iter(f.archive.cells.values()))
    jobs = [(cell, False, slot, pat, 8, 0)
            for slot, pat in enumerate(f._gate_basis[:2])]
    obs = f._gate_wave(jobs)
    assert [(o["pattern"], o["control"], o["plen"]) for o in obs] == [
        ("hold_a0_h4", True, 4), ("hold_a0_h108", True, 108)]
    jobs = [(cell, False, 3, f._gate_basis[3], 8, 0),
            (cell, False, 5, f._gate_basis[5], 8, 0)]
    obs = f._gate_wave(jobs)
    assert [(o["pattern"], o["control"], o["plen"]) for o in obs] == [
        ("hold_a1_h4", False, 4), ("tap_a1_2on2off_x8", False, 32)]
    # The window the labels describe is the one capture_points computes.
    for o, (_c, _s, _sl, pat, phase, _r) in zip(obs, jobs):
        t0, t1, t2 = ib.capture_points(pat, phase)
        assert (t1 - t0, o["plen"]) == (len(pat.masks), len(pat.masks))


def test_contact_can_be_admitted_at_every_registered_settle_phase(tmp_path):
    # CONTACT_K=8 frozen steps needs NINE samples and the root frame is
    # not readable (the pool hands back RAM only as the result of a
    # step), so a settle-only window of 8 steps is one short: at the
    # shorter of the two registered phases the CONTACT family could
    # never be admitted at all, and half the lattice's phase replication
    # was measuring a family that structurally could not exist. Closing
    # the window one frame into the pattern makes both phases live and
    # can only ever REFUSE an admission (open ground has already moved
    # by then), never manufacture one.
    f = _sweep_solver(tmp_path)
    cell = next(iter(f.archive.cells.values()))
    long_hold = f._gate_basis[4]
    assert long_hold.name == "hold_a1_h108"
    for phase in ib.SETTLE_PHASES_PASS_A:
        assert phase <= ib.CONTACT_K + 1 or phase > ib.CONTACT_K
        obs = f._gate_wave([(cell, False, 4, long_hold, phase, 0)])
        assert (obs[0]["contact"], obs[0]["family"]) == (True, ib.CONTACT), \
            f"contact is unadmittable at the registered phase {phase}"
    # A MOVING observable still refuses it at both phases — the relabel
    # reads the profile's own telemetry, it is not a rename of "held".
    f.game = SimpleNamespace(progress=lambda ram: int(ram[0x10]),
                             y=lambda ram: 40,
                             label=lambda k: "k")
    for phase in ib.SETTLE_PHASES_PASS_A:
        obs = f._gate_wave([(cell, False, 4, long_hold, phase, 0)])
        assert (obs[0]["contact"], obs[0]["family"]) == (False, ib.HOLD)


def test_contact_is_refused_when_the_observable_moves_after_the_pattern_starts(
        tmp_path):
    # WHAT THE CLOSING FRAME IS FOR, at every phase and not just the one
    # that needs it as padding. A root standing still on open ground is
    # frozen for the whole settle exactly like a root pressed against a
    # wall — the pad is doing nothing in both cases. The two only diverge
    # one frame after the pattern starts pressing, so that frame is the
    # entire discriminating content of the admission. Sample it only
    # where a settle-only window comes up short (phase 8) and CONTACT at
    # 13/21/34 degenerates into "the pad was idle", which every root in
    # the archive satisfies.
    f = _sweep_solver(tmp_path)
    cell = next(iter(f.archive.cells.values()))
    long_hold = f._gate_basis[4]
    assert long_hold.name == "hold_a1_h108"
    phases = sorted(set(ib.SETTLE_PHASES_PASS_A)
                    | set(ib.SETTLE_PHASES_PASS_B))
    assert phases == [8, 13, 21, 34], "the registered phases moved"
    # $0010 is the stand-in's step counter, so this profile reads frozen
    # for exactly `phase` steps and then advances one unit per frame:
    # motion that begins with the pattern's first frame, i.e. open
    # ground. Pass-B lengths throughout, because a 34-step settle plus a
    # 108-step hold does not fit the pass-A program.
    for phase in phases:
        f.game = SimpleNamespace(
            progress=lambda ram, p=phase: max(0, int(ram[0x10]) - p),
            y=lambda ram: 40, label=lambda k: "k")
        obs = f._gate_wave([(cell, False, 4, long_hold, phase, 0)],
                           tail=ib.TAIL_PASS_B,
                           length=ib.PROGRAM_LEN_PASS_B)
        assert (obs[0]["contact"], obs[0]["family"]) == (False, ib.HOLD), \
            f"motion that starts with the pattern was admitted as CONTACT " \
            f"at phase {phase}"
        # Positive control at the SAME phase and the same program: hold
        # the observable still and the family comes back. Without this
        # the assertion above is also satisfied by a window that can
        # never close.
        f.game = SimpleNamespace(progress=lambda ram: 100, y=lambda ram: 40,
                                 label=lambda k: "k")
        obs = f._gate_wave([(cell, False, 4, long_hold, phase, 0)],
                           tail=ib.TAIL_PASS_B,
                           length=ib.PROGRAM_LEN_PASS_B)
        assert (obs[0]["contact"], obs[0]["family"]) == (True, ib.CONTACT), \
            f"contact is unadmittable at the registered phase {phase}"


def test_the_wave_anchors_the_contact_window_to_the_settle_it_measured(
        tmp_path):
    # The wave is the only caller that knows where settle ENDS, so it is
    # the only one that can stop the freeze window sliding past it: it
    # labels the samples with n_settle=t0. Unlabelled, the window is the
    # last k+1 of everything, which at 13/21/34 buys a frame of the
    # pattern by dropping the earliest settle transition the test covers
    # — and that transition is the entire evidence that this root was
    # still moving when the program began.
    f = _sweep_solver(tmp_path)
    cell = next(iter(f.archive.cells.values()))
    long_hold = f._gate_basis[4]
    for phase in (13, 21, 34):          # the phases with settle to spare
        # ONE step of motion, at the earliest transition the K-step
        # window covers, then frozen through the pattern's first frame.
        f.game = SimpleNamespace(
            progress=lambda ram, p=phase: int(
                int(ram[0x10]) >= p - ib.CONTACT_K + 1),
            y=lambda ram: 40, label=lambda k: "k")
        obs = f._gate_wave([(cell, False, 4, long_hold, phase, 0)],
                           tail=ib.TAIL_PASS_B,
                           length=ib.PROGRAM_LEN_PASS_B)
        assert (obs[0]["contact"], obs[0]["family"]) == (False, ib.HOLD), \
            f"the window slid past a moving settle sample at phase {phase}"


def test_the_contact_freeze_window_never_slides_off_the_end_of_settle():
    # The same rule as a pure function. `n_settle` says how many of the
    # samples are settle; anything past it is the pattern already running.
    k = 4
    frozen = [(10, 5)] * 12

    # Settle can supply k+1 samples, so the freeze window is settle-only
    # and the frame past it is a SEPARATE closing check.
    assert ib.contact_admitted(frozen, eps=0, k=k, n_settle=11)
    moved = list(frozen)
    moved[-1] = (11, 5)                       # the pattern's first frame
    assert not ib.contact_admitted(moved, eps=0, k=k, n_settle=11)

    # THE SLIDE. A root still decelerating when settle begins: the
    # earliest transition the window covers is the only evidence of it,
    # and taking the last k+1 of ALL the samples drops exactly that
    # transition to make room for a frame the window did not need. The
    # dropped evidence is the difference between a refusal and an
    # admission, and it would have made the verdict phase-dependent.
    early = list(frozen)
    early[6] = (9, 5)
    assert not ib.contact_admitted(early, eps=0, k=k, n_settle=11)
    assert ib.contact_admitted(early[-(k + 1):], eps=0, k=k), \
        "the slid window is what this test exists to rule out"

    # Settle alone one sample short: NOW the window extends into the
    # pattern's first frame, because otherwise there is no window at all.
    short = [(10, 5)] * 5
    assert ib.contact_admitted(short, eps=0, k=k, n_settle=4)
    # ...and the extension buys exactly one sample, never two.
    assert not ib.contact_admitted(short[:4], eps=0, k=k, n_settle=3)
    # Degenerate accept, and the settle-only caller is unchanged.
    assert ib.contact_admitted([], k=0)
    assert ib.contact_admitted(frozen[:9], eps=0, k=8)


def _obs(root, slot, family, moves, *, base=None, n=ib.RAM_SIZE):
    """One capture triple. `moves` maps addr -> (v1, v2) relative to a
    zero baseline, so a test states exactly what moved and when."""
    r0 = np.zeros(n, dtype=np.uint8) if base is None else np.array(base,
                                                                   np.uint8)
    r1, r2 = r0.copy(), r0.copy()
    for addr, (v1, v2) in moves.items():
        r1[addr], r2[addr] = v1, v2
    return {"root": root, "slot": slot, "family": family,
            "ram0": r0, "ram1": r1, "ram2": r2}


def test_noop_subtraction_removes_whatever_moves_in_the_paired_control():
    # The nuisance filter, and why it is MEASURED rather than assumed:
    # frame counters, RNG, timers and music free-run whatever the pad
    # does. Same address, same movement, in the control from the same
    # root => not a candidate, at any effect size.
    obs = []
    for root in ("r0", "r1", "r2"):
        obs.append(_obs(root, 0, "control", {0x10: (1, 2)}))
        obs.append(_obs(root, 1, "hold", {0x10: (1, 2), 0x40: (8, 8)}))
    ranked = Solver._gate_rank(obs)
    addrs = {r["addr"] for r in ranked}
    assert 0x40 in addrs
    assert 0x10 not in addrs, "a free-runner survived NOOP subtraction"
    # The survivor is classed by persistence: it moved and it stayed.
    assert next(r for r in ranked if r["addr"] == 0x40)["class"] == "latched"


def test_a_candidate_must_reproduce_from_three_distinct_roots():
    # Cross-root invariance. One spectacular hit at one root is exactly
    # what a coincidence looks like, and the archive has millions of
    # roots to find coincidences at.
    two = [_obs(r, 0, "control", {}) for r in ("r0", "r1")] + \
          [_obs(r, 1, "hold", {0x40: (8, 8)}) for r in ("r0", "r1")]
    assert [r["addr"] for r in Solver._gate_rank(two)] == []
    three = two + [_obs("r2", 0, "control", {}),
                   _obs("r2", 1, "hold", {0x40: (8, 8)})]
    assert [r["addr"] for r in Solver._gate_rank(three)] == [0x40]
    # ...and it is the DISTINCT root count, not the observation count:
    # ten repeats at one root still fail.
    many = [_obs("r0", 0, "control", {})] + \
           [_obs("r0", i + 1, "hold", {0x40: (8, 8)}) for i in range(10)]
    assert Solver._gate_rank(many) == []


def _win_obs(root, slot, family, plen, control, moves, phase=8):
    """A capture triple that also states the WINDOW it measured over —
    (phase, pattern length), which is what fixes t0/t1/t2."""
    ob = _obs(root, slot, family, moves)
    ob.update(plen=plen, phase=phase, control=control)
    return ob


def test_a_paired_control_only_subtracts_over_the_window_it_measured():
    # THE nuisance-filter defect, stated in full. Slot 0 is a 4-step
    # hold, so at phase 8 it measures steps 8..36; hold_a0_h108 at the
    # same phase measures 8..140. Any input-independent byte with a
    # period longer than the control's window is therefore invisible to
    # the control and fully visible to the pattern — and because a clock
    # reproduces at EVERY root and moves once and stays moved, it sails
    # through 3-root invariance and classes LATCHED. K0's blind grade and
    # G1-DISCOVERY both rest on this subtraction, so a mismatched pair
    # does not degrade the instrument, it inverts it.
    def rows(with_twin: bool) -> list:
        out = []
        for root in ("r0", "r1", "r2"):
            out.append(_win_obs(root, 0, "control", 4, True, {}))
            if with_twin:
                out.append(_win_obs(root, 1, "control", 108, True,
                                    {0x11: (1, 1)}))
            out.append(_win_obs(root, 2, "hold", 108, False,
                                {0x11: (1, 1), 0x40: (8, 8)}))
        return out

    # Short control only: the 100-period byte is receipted as a
    # discovery, scoring exactly as well as the real latch.
    unpaired = {r["addr"]: r for r in Solver._gate_rank(rows(False))}
    assert unpaired[0x11]["class"] == "latched"
    assert unpaired[0x11]["score"] == unpaired[0x40]["score"] == 1.0

    # Its own length twin — the all-NOOP program the ladder already
    # emits at 108 — removes it, and leaves the real latch alone.
    paired = {r["addr"]: r for r in Solver._gate_rank(rows(True))}
    assert 0x11 not in paired, "a slow free-runner survived NOOP subtraction"
    assert paired[0x40]["class"] == "latched"


def test_a_long_control_cannot_veto_what_a_short_window_found():
    # The rule runs BOTH ways, and this direction is the one a
    # union-of-all-controls shortcut gets wrong. A 108-step control sees
    # free-runners a 4-step pattern never had time to show; fold every
    # control at a root into one mask and that pattern's real candidate
    # is vetoed for a movement it could not have caused. Matching is on
    # the window, not on "subtract everything we ever saw idle".
    rows = []
    for root in ("r0", "r1", "r2", "r3"):
        rows.append(_win_obs(root, 0, "control", 4, True, {}))
        rows.append(_win_obs(root, 1, "control", 108, True, {0x11: (1, 1)}))
        # Inside ITS window the short pattern is the only thing that
        # moved 0x11 — the long control's late tick is out of scope.
        rows.append(_win_obs(root, 2, "hold", 4, False, {0x11: (1, 1)}))
        rows.append(_win_obs(root, 3, "tap", 32, False, {0x42: (5, 5)}))
    got = {r["addr"] for r in Solver._gate_rank(rows)}
    assert 0x11 in got, "a long control vetoed a short window's candidate"
    assert 0x42 in got


def test_the_ladder_emits_an_all_noop_control_at_every_length_it_uses():
    # Why the pairing above is affordable: the NOOP action rides the same
    # hold and tap rungs as every other action, so the basis ALREADY
    # contains a control for every window it generates. Nothing has to be
    # added to the 120/72/90 counts §6 budgets against.
    for rel, _n, total, *_ in _TARGET_BASIS:
        basis = ib.interaction_basis(_profile_actions(rel))
        ctl = ib.control_slots(basis)
        assert set(ctl) == {len(p.masks) for p in basis}
        # Six of them: the four hold rungs and the two tap duties.
        assert sorted(ctl) == [4, 12, 32, 36, 48, 108]
        assert len(basis) == total          # and the counts did not move
        # First occurrence wins, and slot 0 is still the 4-step control.
        assert ctl[4] == 0
        # Exactly ONE control per length, because dedupe is on the mask
        # tuple and (0,)*n is unique in n — so the lookup can never be
        # ambiguous about which program certifies a window.
        lens = [len(p.masks) for p in basis if not any(p.masks)]
        assert len(lens) == len(set(lens)) == 6


def test_an_all_noop_program_is_never_ranked_as_a_candidate():
    # The other half: the five NOOP programs past slot 0 are CONTROLS,
    # not hold-family patterns. Ranking them would enter a program that
    # pressed nothing into the hold family's cross-root count and let a
    # nuisance be "discovered" by the empty action.
    rows = []
    for root in ("r0", "r1", "r2"):
        rows.append(_win_obs(root, 0, "control", 4, True, {}))
        rows.append(_win_obs(root, 3, "hold", 108, True, {0x11: (1, 1)}))
    assert Solver._gate_rank(rows) == []


def test_the_ranking_is_invariant_under_a_permutation_of_its_input():
    # The sweep collects observations in wave order, which depends on
    # worker count and scheduling. A ranking that depended on that order
    # would report a different candidate table on an 8-worker box than on
    # a 4-worker one, from the same measurements.
    obs = []
    for root in ("r0", "r1", "r2", "r3"):
        obs.append(_obs(root, 0, "control", {0x10: (1, 2)}))
        obs.append(_obs(root, 1, "hold", {0x40: (8, 8)}))
        obs.append(_obs(root, 2, "tap", {0x41: (3, 0)}))
        obs.append(_obs(root, 3, "combo", {0x42: (1, 2)}))
    base = Solver._gate_rank(obs)
    assert len(base) >= 3
    rng = np.random.default_rng(0)
    for _ in range(8):
        shuffled = [obs[i] for i in rng.permutation(len(obs))]
        assert Solver._gate_rank(shuffled) == base
    # The order is the declared one: score desc, then (addr, family).
    assert base == sorted(base, key=lambda r: (-r["score"], r["addr"],
                                               r["family"]))


# ---------------------------------------------------------------------
# K4 farmability: MEASURED, never a receipted constant
# ---------------------------------------------------------------------

def _sig_rows(n_roots: int = 6) -> list:
    """Observations that clear BH-FDR on their own, so the ADMISSION path
    is exercised and not merely the ranking. Three families are the
    minimum that gives every address a defined leave-family-out
    background rate; a single-family fixture falls back to the
    conservative coin flip and nothing is ever significant.

    SIX roots, not three or four, because BH's denominator is the whole
    addr x family grid (2,048 x 3 here) and not the survivor list. The
    smallest p an address can reach is rate**k with rate floored at
    1/(grand+1), so four roots bottom out near 3e-5 — comfortably inside
    a bare q=0.01 and nowhere near the 1.6e-6 the grid actually charges.
    That gap IS the defect the grid denominator closes, and a fixture
    sitting inside it would have gone on certifying it."""
    rows = []
    for root in [f"r{i}" for i in range(n_roots)]:
        rows.append(_obs(root, 0, "control", {}))
        rows.append(_obs(root, 1, "hold", {0x40: (8, 8)}))
        rows.append(_obs(root, 2, "tap", {0x41: (3, 0)}))
        rows.append(_obs(root, 3, "combo", {0x42: (1, 2)}))
    return rows

def test_farmability_is_measured_off_the_runs_own_band_sampling():
    # K4's second refusal is "more than one event per 1k random steps",
    # and the boundary histogram cannot answer it: it records the VALUES
    # a byte took and is structurally silent about how often it moved
    # between them. So the change count is its own counter, over the same
    # stride-sampled band observations, and the rate is reported in the
    # units the kill criterion is written in.
    f = _arm_gate(_burst_solver([_ocell(4, 0)], learn_every=1),
                  gate_mode="enumerate", _sel_topgx=4)
    ram = bytearray(2048)
    n_samples = ib.FARM_MIN_SAMPLES + 1
    for i in range(n_samples * ib.BAND_SAMPLE_STRIDE):
        ram[0x20] = i & 0xFF                 # differs at every sample
        ram[0x21] = 7                        # never moves
        f._gate_observe(bytes(ram), _ocell(4, 0).key)
    assert f._gate_change_n == n_samples - 1
    # A byte that differs at every sample saturates at 1000/stride: far
    # above the 1-per-1k threshold, so K4 refuses it.
    assert f._gate_farmability(0x20) == pytest.approx(
        1000.0 / ib.BAND_SAMPLE_STRIDE)
    assert f._gate_farmability(0x20) > ib.MAX_FARM_EVENTS_PER_1K
    assert f._gate_farmability(0x21) == 0.0

    # Below the minimum sample count the estimate cannot resolve the
    # threshold it exists to decide, so it reports NOTHING rather than a
    # comfortable zero.
    assert ib.farm_rate(0, ib.FARM_MIN_SAMPLES - 1) is None
    assert ib.farm_rate(0, ib.FARM_MIN_SAMPLES) == 0.0
    # Unbiased where the decision is made: in the rare regime the chance
    # of catching a change inside a gap is stride x the per-step rate, so
    # 4 changes over 125 sampled gaps reads back as exactly the 1-event
    # -per-1k-steps threshold K4 is written against.
    assert ib.farm_rate(4, 125) == pytest.approx(ib.MAX_FARM_EVENTS_PER_1K)
    # A run that never sampled the band has no reading at all.
    cold = _arm_gate(SimpleNamespace(), gate_mode="enumerate")
    assert Solver._gate_farmability(cold, 0x20) is None


def test_an_unmeasured_farmability_refuses_the_axis_instead_of_scoring_it():
    # THE fabricated-measurement guard (§8 PURITY). A ranker handed a
    # farmability probe that has no reading yet must not write 0.0 into a
    # receipt column called "farmability" — that is a measured-looking
    # constant no probe produced, and it silently converts K4's refusal
    # into a pass.
    rows = _sig_rows()

    unmeasured = ib.rank_candidates(rows, farmability=lambda a: None)
    assert {r["farmability"] for r in unmeasured} == {None}
    assert {r["refused"] for r in unmeasured} == {"farm_unmeasured"}
    assert ib.admitted_candidates(unmeasured) == []

    # A measured, farmable axis is refused by the OTHER half of K4 —
    # the half that could never fire while farm was pinned at 0.0.
    farmable = ib.rank_candidates(rows, farmability=lambda a: 4.0)
    assert {r["refused"] for r in farmable} == {"farmable"}
    assert {r["farmability"] for r in farmable} == {4.0}
    assert ib.admitted_candidates(farmable) == []

    # Measured and clean: admitted, and the rate is carried through the
    # score as (1 - farmability) rather than dropped.
    clean = ib.rank_candidates(rows, farmability=lambda a: 0.25)
    latch = next(r for r in clean if r["addr"] == 0x40)
    assert latch["refused"] is None and latch["significant"] is True
    assert latch["score"] == pytest.approx(0.75)      # 1.0 latched x 0.75
    assert ib.admitted_candidates(clean)[0]["addr"] == 0x40

    # No probe supplied at all (the pure fixture path) reports None and
    # screens nothing — it never claims a measurement either.
    none = ib.rank_candidates(rows)
    assert {r["farmability"] for r in none} == {None}
    assert {r["refused"] for r in none} == {None}


def test_bh_fdr_is_charged_against_the_full_addr_by_family_grid():
    # THE multiple-comparisons defect. The differ runs one hypothesis
    # per (addr, family) cell of the grid — thousands of them, R1's
    # comparison flood — and then keeps the rows that reproduced from
    # >= 3 roots. Handing BH only those survivors lets a filter chosen
    # BECAUSE it keeps the interesting rows define the family of tests
    # those rows were selected from: m collapses from thousands to a
    # handful and every survivor is graded against a bar orders of
    # magnitude too generous. q is the same 0.01 either way, which is
    # why the receipt looks identical while meaning something else.
    assert ib.comparison_grid(3) == ib.RAM_SIZE * 3
    assert ib.comparison_grid(0) == ib.RAM_SIZE          # never divides by 0

    # The correction, in one p-value: 5e-3 clears a bare q=0.01 and is
    # refused the moment the real denominator is charged.
    assert ib.bh_significant([0.005], q=0.01) == [True]
    assert ib.bh_significant([0.005], q=0.01, m=ib.RAM_SIZE) == [False]
    # An m smaller than the batch handed in cannot shrink the family.
    assert ib.bh_significant([0.005], q=0.01, m=1) == [True]
    # ...and a genuinely tiny p still passes the grid-wide bar.
    assert ib.bh_significant([1e-9], q=0.01, m=ib.RAM_SIZE * 4) == [True]

    # Wired: four roots is precisely the regime the survivor-only
    # denominator called significant. p = 3.5e-5 — inside q=0.01, and
    # twenty times outside the grid's first-rank bar of 1.6e-6.
    four = ib.rank_candidates(_sig_rows(4))
    assert [r["addr"] for r in four] == [0x40, 0x41, 0x42]
    assert not any(r["significant"] for r in four)
    assert ib.admitted_candidates(four) == []
    # Six roots reaches 2.1e-8 and is admitted on the same q.
    six = ib.rank_candidates(_sig_rows(6))
    assert all(r["significant"] for r in six)
    assert [r["addr"] for r in ib.admitted_candidates(six)] == [0x40, 0x41]

    # A CALLER CAN STATE THE DENOMINATOR. A sweep that ranks its wall and
    # its sham null in two calls performed ONE grid of comparisons, and
    # neither call can see it: each counts only the families its own half
    # carried. The override is charged to the verdict and reported on the
    # row, so the two halves are comparable and a receipt can name the
    # bar it was judged against.
    wide = ib.rank_candidates(_sig_rows(6), m=ib.RAM_SIZE * 4096)
    assert {r["fdr_m"] for r in wide} == {ib.RAM_SIZE * 4096}
    assert not any(r["significant"] for r in wide)
    assert {r["fdr_m"] for r in six} == {ib.comparison_grid(3)}


def test_every_live_ranking_call_site_passes_the_farmability_probe():
    # Wiring, structurally. The estimator existing is worth nothing if
    # the sweep keeps calling the ranker without it: (1-farmability)
    # would stay inert, K4's farmable half would stay dead, and every
    # receipt row would carry the same constant. Pinned at the source so
    # a future call site cannot be added without one.
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    calls = [n for n in ast.walk(cls)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_gate_rank"]
    assert len(calls) >= 3, "the live ranking call sites moved"
    for call in calls:
        kw = {k.arg for k in call.keywords}
        assert "farmability" in kw, (
            f"_gate_rank at line {call.lineno} ranks without measuring "
            f"farmability")
        assert "novelty" in kw
    # ...and the two calls that split one sweep into a wall and its sham
    # null charge the SAME BH denominator. One sweep performed one grid
    # of comparisons; pinned as "the same name" so the pair cannot drift
    # back into two locally-derived m's that make K1 compare two bars.
    sweep = next(n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)
                 and n.name == "_gate_sweep")
    ms = [kw.value.id
          for n in ast.walk(sweep)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
          and n.func.attr == "_gate_rank"
          for kw in n.keywords
          if kw.arg == "m" and isinstance(kw.value, ast.Name)]
    assert len(ms) == 2 and len(set(ms)) == 1, (
        "the wall and its sham null are not ranked against one m")


def _swept_with_samples(tmp_path):
    """A sweep-ready solver that has already sampled enough band steps
    for a farmability reading, so the ADMISSION path is live."""
    f = _sweep_solver(tmp_path)
    ram = bytearray(2048)
    for _ in range((ib.FARM_MIN_SAMPLES + 1) * ib.BAND_SAMPLE_STRIDE):
        f._gate_observe(bytes(ram), _ocell(5, 0).key)
    return f


def test_the_confirmation_pass_re_runs_each_survivors_length_twin(tmp_path):
    # Pass B stretches the tail 24 -> 512 to settle the persistence
    # class, which is the term carrying the most weight in the rank. Over
    # a window that long the machine's slow clocks are all awake, so a
    # survivor that comes back WITHOUT the all-NOOP program of its own
    # length gets its class from a capture nothing controlled: the
    # stand-in's latched byte takes one input-independent bump at step
    # 600, and an unmatched pass reads latch-then-bump as a monotone
    # COUNTER — weight 0, score 0, candidate discarded.
    f = _swept_with_samples(tmp_path)
    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    row = next(r for r in ranked if r["addr"] == 0x40)
    assert row["class"] == "latched" and row["score"] > 0.0
    assert [r["addr"] for r in ib.admitted_candidates(ranked)] == [0x40]
    # The twin was actually run: the confirmation pass saw the bump in
    # its OWN control and subtracted it, so it declined to re-class.
    assert row["pass_b"] is False
    twins = ib.control_slots(f._gate_basis)
    assert twins == {4: 0, 108: 1, 32: 2}


def test_the_candidate_receipt_reports_how_farmability_was_measured(tmp_path):
    # End to end: the sweep writes a receipt, and a reader can tell a
    # measured zero from an unmeasured one without re-running anything.
    #
    # D-PRIOR REPAIR, wired. The LIVE band sampler has fired zero times
    # here, and at a pinned wall that is the ordinary case rather than a
    # corner: the sweep roots in the band tail the search does not
    # re-enter, so the graded Castlevania run ended 24 minutes with
    # farm_samples=0 and refused every significant row `farm_unmeasured`.
    # The sweep now measures the screen off its OWN undirected frames, so
    # the reading below is a MEASUREMENT taken on the population the
    # instrument is rooted in — not a default and not a zero nothing
    # produced.
    # Eight roots: the screen's floor is charged in DISTINCT steps, and
    # every program at a root replays one all-NOOP trajectory (repair
    # R5), so six roots cannot resolve one event per thousand.
    f = _sweep_solver(tmp_path, n_roots=8, columns=14, band=7)
    assert f._gate_change_n == 0             # the live sampler never fired
    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    assert ranked and all(r["farmability"] == 0.0 for r in ranked)
    assert f._gate_change_sweep_n > 0
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert [s["name"] for s in rec["farm_sources"]] == ["sweep_undirected"]
    assert rec["farm_steps_total"] >= ib.FARM_MIN_STEPS
    assert rec["farm_samples"] == 0          # ...and the live half says so
    # ...so the axis is screened and admitted on a real number, which is
    # the outcome the empty-by-construction admissible list denied.
    assert rec["admitted"] == [0x40]
    assert f._gate_counters["admitted"] == 1
    # A run whose sweep has ALSO never sampled has no reading at all, and
    # the ranker refuses the axis rather than scoring it (K4's second
    # half). "Unmeasured" and "measured quiet" stay distinguishable.
    cold = _arm_gate(SimpleNamespace(), gate_mode="enumerate")
    assert Solver._gate_farmability(cold, 0x40) is None

    # Now give it live samples too, and the receipt states BOTH sources
    # with the denominator behind each.
    ram = bytearray(2048)
    for i in range((ib.FARM_MIN_SAMPLES + 1) * ib.BAND_SAMPLE_STRIDE):
        f._gate_observe(bytes(ram), _ocell(5, 0).key)
    f._gate_disarmed = False
    f._gate_swept = {}
    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    assert [r["addr"] for r in ranked if r["significant"]] == [0x40]
    assert all(r["farmability"] == 0.0 for r in ranked)   # measured, quiet
    rec = json.loads((tmp_path / "gate" / "candidates_002.json").read_text())
    assert [s["name"] for s in rec["farm_sources"]] == ["live_band",
                                                        "sweep_undirected"]
    assert rec["farm_samples"] == ib.FARM_MIN_SAMPLES
    assert rec["farm_stride"] == ib.BAND_SAMPLE_STRIDE
    assert rec["farm_min_steps"] == ib.FARM_MIN_STEPS
    assert rec["admitted"] == [0x40]


def test_a_null_sweep_writes_the_receipt_that_is_its_primary_outcome(tmp_path):
    # A SWEEP THAT ADMITS NOTHING IS THE PRE-REGISTERED PRIMARY OUTCOME.
    # The arm's whole claim is that it can come back empty from a wall it
    # swept exhaustively, and K0 grades it blind on exactly that. Written
    # only on the hit path, the null survived as a counter in a progress
    # line and nothing else — after the fact, "the instrument found
    # nothing" was indistinguishable from "it never ran", "it threw" and
    # "somebody deleted the receipt", and no artifact stated what had
    # been swept or against which bar.
    f = _sweep_solver(tmp_path, n_roots=2)      # < MIN_INVARIANT_ROOTS
    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    assert ranked == [] and f._gate_counters["admitted"] == 0
    # Nothing found, so nothing is disarmed and nothing is queued: the
    # null path leaves the arm exactly as armed as it was.
    assert f._gate_disarmed is False and f._gate_inject == []

    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert (rec["admitted"], rec["ranked"], rec["queued_injections"]) == (
        [], [], [])
    # The same fields a hit carries, because a null is only readable
    # against what produced it: how much was swept, from how many roots,
    # over which phases, at which q.
    assert rec["sweep"] == 1
    assert (rec["roots"], rec["sham_roots"]) == (2, 0)
    assert rec["basis"] == len(f._gate_basis)
    assert rec["phases"] == list(f._gate_phases)
    assert rec["program_len"] == ib.PROGRAM_LEN_PASS_A
    assert (rec["fdr_q"], rec["rank_cutoff"]) == (ib.FDR_Q, ib.RANK_CUTOFF)
    assert rec["farm_samples"] == 0
    assert rec["farm_stride"] == ib.BAND_SAMPLE_STRIDE
    # ...and the BH denominator, which is the one number a null CANNOT
    # carry on a ranked row, because it has no rows.
    assert rec["fdr_m"] == ib.comparison_grid(3) == f._gate_fdr_m


def test_the_wall_and_its_sham_null_are_charged_one_bh_denominator(tmp_path):
    # K1 compares two yields, so the two have to be graded on one bar.
    # Left to derive its own, each rank call counts only the families ITS
    # half of the observations carried — and the halves differ by
    # construction, because CONTACT is a state-dependent relabel that
    # fires where the roots are pressed against something and not where
    # they are free. Here it is the SHAM roots that freeze (the wall
    # roots are the ones still advancing), so the wall's own view of the
    # grid is a family short and every wall p-value would be charged a
    # denominator two thousand comparisons too small.
    f = _sweep_solver(tmp_path)
    f.gate_sham_roots = 2
    # $0000 is the root's own index in this stand-in, so the band cells
    # (6..11) read as still advancing and the free ones (0..5) as frozen.
    f.game = SimpleNamespace(
        progress=lambda ram: 100 if int(ram[0x00]) < 6 else int(ram[0x10]),
        y=lambda ram: 40, label=lambda k: "k")
    cells = {int(c.key[-1]): c for c in f.archive.cells.values()}
    long_hold = f._gate_basis[4]
    # The split, demonstrated: the same program is HOLD at a wall root
    # and CONTACT at a sham one, so the two halves of one sweep really do
    # see different family sets.
    wall = f._gate_wave([(cells[7], False, 4, long_hold, 8, 0)])
    free = f._gate_wave([(cells[1], True, 4, long_hold, 8, 0)])
    assert (wall[0]["family"], free[0]["family"]) == (ib.HOLD, ib.CONTACT)

    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    assert {r["family"] for r in ranked} == {"hold"}, \
        "the fixture stopped producing wall rows"
    # hold + tap at the wall, and contact only over at the null: three
    # families in the sweep, two in the wall's own view of it.
    grid = ib.comparison_grid(3)
    assert f._gate_fdr_m == grid > ib.comparison_grid(2)
    assert {r["fdr_m"] for r in ranked} == {grid}


# ---------------------------------------------------------------------
# gate_marks + suppression (3)
# ---------------------------------------------------------------------

def test_gate_marks_ride_as_the_eighth_trace_element(tmp_path):
    # Hand-built ctx + trace: an armed run records where it injected, so
    # T2 can later mask exactly those frames. Element 8 exists ONLY when
    # the arm is on, so a default-path traces.pkl is unchanged.
    ram = bytes(2048)
    on = _arm_gate(_burst_solver([_ocell(4, 0)], learn_every=1),
                   gate_mode="enumerate")
    ctx = {"gate_marks": [(12, 0x40, "pattern")]}
    assert on.observe(0, ram, [0, 1], 2, "entrance", ctx=ctx) == "live"
    rec = next(iter(on.traces.values()))
    assert len(rec) == 8 and rec[7] == ((12, 0x40, "pattern"),)

    off = _burst_solver([_ocell(4, 0)], learn_every=1)
    assert off.observe(0, ram, [0, 1], 2, "entrance", ctx={}) == "live"
    assert len(next(iter(off.traces.values()))) == 7


def test_suppressing_the_marked_window_changes_the_emitted_bytes():
    # T2's deterministic ablation. It has to mask the marked frames and
    # ONLY those (a blanket NOOP is not an ablation, it is a different
    # trajectory), and it must not mutate the trace the archive holds.
    trace = [3, 3, 3, 3, 3, 3]
    marks = [(1, 0x40, "pattern"), (2, 0x40, "pattern"),
             (4, 0x40, "other")]
    masked = gate_suppress_trace(trace, marks)
    assert masked == [3, 0, 0, 3, 3, 3]
    assert masked != trace and trace == [3] * 6
    assert bytes(masked) != bytes(trace)
    # An out-of-range mark (a truncated lineage) is ignored, not fatal.
    assert gate_suppress_trace([1, 1], [(99, 0, "pattern")]) == [1, 1]
    # No marks at all is the identity, so an unarmed lineage ablates to
    # itself and the T2 receipt says "nothing to suppress" honestly.
    assert gate_suppress_trace(trace, []) == trace


def test_the_ablation_masks_the_whole_injected_window_not_one_frame():
    # T2's NECESSITY test, and why the mark needs a duration. An
    # injection owns macro_hold + 6 frames (six settle NOOPs, then the
    # hold); a mark that records only its start blanks ONE of them, so
    # the "ablated" trace still contains 113 of a 108-step hold's 114
    # frames and T2 compares the program against itself. A test that
    # cannot fail is worse than no test: it reports NECESSITY-CONFIRMED
    # for a mechanism it never removed.
    trace = [3] * 20
    masked = gate_suppress_trace(trace, [(4, 0x40, "pattern", 8)])
    assert masked == [3] * 4 + [0] * 8 + [3] * 8
    assert trace == [3] * 20                     # never mutated in place
    # Two injections, and the frames between them are untouched — a
    # blanket NOOP is a different trajectory, not an ablation.
    two = gate_suppress_trace(trace, [(1, 0x40, "pattern", 2),
                                      (10, 0x41, "pattern", 3)])
    assert two == [3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3,
                   3, 3]
    # The other kind is still skipped whole, span and all.
    assert gate_suppress_trace(trace, [(0, 0, "other", 9)]) == trace
    # A span running off the end of a truncated lineage clips.
    assert gate_suppress_trace([1, 1, 1], [(1, 0, "pattern", 99)]) == \
        [1, 0, 0]
    # A pre-span 3-tuple mark is honoured as written (one frame) rather
    # than guessed at, so a resumed lineage ablates what it recorded.
    assert gate_suppress_trace(trace, [(4, 0x40, "pattern")]) == \
        [3] * 4 + [0] + [3] * 15


def test_the_recorded_mark_carries_the_window_the_injection_owns():
    # The writer's half of the same contract, pinned at the source: the
    # branch that takes the shared macro slot writes macro_hold + 6 into
    # macro_left, and the mark it appends records THAT span. A mark that
    # only carried a step index would make the ablation above unable to
    # know what to remove, however correct the ablation itself is.
    import scripts.go_explore_solve as ges

    body = Path(ges.__file__).read_text().split("def explore(self)", 1)[1]
    branch = body[body.index('c.get("gate_macro")'):
                  body.index("self.transition_macros and")]
    assert 'c["macro_left"] = c["macro_hold"] + 6' in branch
    mark = branch[branch.index("gate_marks"):]
    assert 'int(c["macro_left"])' in mark, \
        "the injection mark records no duration: T2 can ablate one frame"
    # ...and it rides as a 4-tuple through the trace element.
    ram = bytes(2048)
    on = _arm_gate(_burst_solver([_ocell(4, 0)], learn_every=1),
                   gate_mode="enumerate")
    ctx = {"gate_marks": [(12, 0x40, "pattern", 18)]}
    assert on.observe(0, ram, [0, 1], 2, "entrance", ctx=ctx) == "live"
    assert next(iter(on.traces.values()))[7] == ((12, 0x40, "pattern", 18),)


def test_a_resumed_seven_tuple_archive_loads_with_no_marks_and_no_error():
    # Every banked archive on disk predates the 8th element. Resuming one
    # must be a no-op, not a crash and not a silent re-key.
    cell = _ocell(4, 0)
    f = _burst_solver([cell])
    f.traces = {cell.key: ("entrance", b"\x01\x02", 0, (), 0, (), 0)}
    ctx = f._assign(0)
    assert ctx["gate_marks"] == []
    assert ctx["trace"] == [1, 2] and ctx["kills"] == 0
    f.traces[cell.key] = ("entrance", b"\x01", 0, (), 0, (), 3,
                          ((7, 0x40, "pattern"),))
    ctx = f._assign(0)
    assert ctx["gate_marks"] == [(7, 0x40, "pattern")] and ctx["kills"] == 3


# ---------------------------------------------------------------------
# macro-slot precedence (2)
# ---------------------------------------------------------------------

def test_the_shared_macro_slot_goes_gate_then_transition_then_hold():
    # ONE slot, three claimants. Precedence is decided at ACQUISITION
    # only — and an in-flight macro outranks all of them, which is what
    # keeps Castlevania's declared up-hold (p=0.02, the rarest thing the
    # sampler emits) from being chopped in half by an arm that arrived
    # mid-hold.
    assert macro_slot_owner(False, True, True, True) == "gate"
    assert macro_slot_owner(False, False, True, True) == "transition"
    assert macro_slot_owner(False, False, False, True) == "hold"
    assert macro_slot_owner(False, False, False, False) is None
    for g in (True, False):
        for t in (True, False):
            for h in (True, False):
                assert macro_slot_owner(True, g, t, h) == "in_flight"


def _assign_solver(cells, chosen, queue, **over):
    """A solver whose selection is pinned, so the band rule and the
    round-robin can be stated without depending on which cell the
    weighted sampler happened to draw."""
    kw = dict(gate_mode="enumerate", gate_band=2, _sel_topgx=40,
              _gate_inject=list(queue))
    kw.update(over)
    f = _arm_gate(_burst_solver(cells), **kw)
    f.select = lambda: chosen
    return f


def test_the_gate_program_channel_has_a_writer_and_not_only_a_reader():
    # THE dead-code guard. `gate_macro`/`gate_cand` were once reachable
    # only from tests — nothing in the solver ever WROTE them — so the
    # injection branch, the gate_injections counter, gate_marks and
    # gate_suppress_trace were live under pytest and nowhere else, and
    # T1's arm C was arm B plus a step tax while T3 had nothing to be
    # sufficient about. The writer also has to live in `_assign` and only
    # there: that is what makes "at a burst boundary, never mid-burst"
    # structural instead of a comment.
    import scripts.go_explore_solve as ges

    tree = ast.parse(Path(ges.__file__).read_text())

    def _written(scope) -> set:
        return {t.slice.value for n in ast.walk(scope)
                if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and isinstance(t.slice.value, str)}

    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Solver")
    assert {"gate_macro", "gate_cand"} <= _written(cls), \
        "nothing writes a gate program: arm C and T3 are dead code"
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "_assign")
    assert {"gate_macro", "gate_cand"} <= _written(fn)


def test_only_a_declared_hold_can_be_handed_to_the_shared_macro_slot():
    # The channel's REACH, stated so the campaign cannot quietly claim
    # more than it injects. The slot is (action index, hold count) and
    # the trace alphabet the archive and replay_verify speak is action
    # indices, so a tap duty cycle and a combo mask — an OR that is by
    # construction not any declared action — have no in-run path at all.
    space = _profile_actions("configs/castlevania.yaml")
    masks = list(action_space_to_bitmasks(space))
    fams: dict = {}
    for p in ib.interaction_basis(space):
        fams.setdefault(p.family, []).append(p)
    for p in fams["hold"]:
        shot = ib.macro_injection(p, masks)
        if not any(p.masks):
            assert shot is None, "the paired control presses nothing"
            continue
        ai, hold = shot
        assert masks[ai] == p.masks[0] and hold == len(p.masks)
    assert all(ib.macro_injection(p, masks) is None
               for p in fams["tap"] + fams["combo"])
    # THE EXACT SPLIT, pinned so no receipt can round it in the arm's
    # favour: 60 of Castlevania's 120 patterns can be MEASURED and never
    # INJECTED — 32 taps + 24 combos + the 4 all-NOOP holds. The sweep
    # measures twice what the channel can carry, and both docstrings that
    # quote the figure are checked here rather than trusted.
    basis = ib.interaction_basis(space)
    injectable = [p for p in basis if ib.macro_injection(p, masks)]
    assert (len(fams["tap"]) + len(fams["combo"])) == 56
    assert len(basis) - len(injectable) == 60 == len(injectable)
    for src, quote in ((ib.macro_injection.__doc__, "60 of the 120"),
                       (Solver._gate_queue_injections.__doc__,
                        "can be handed 60 of")):
        assert quote in " ".join(src.split())


def test_an_inexpressible_candidate_is_counted_and_not_silently_dropped():
    # R2's rule: a gate the basis can measure but the channel cannot
    # carry is a DECLARED limitation, not an absence. The refusal is
    # counted into its own telemetry field and written onto the receipt
    # row, so a null from arm C can be read against how much of the
    # basis ever reached the search.
    space = [[], ["right"], ["A"]]
    basis = ib.interaction_basis(space)
    masks = list(action_space_to_bitmasks(space))
    f = _arm_gate(SimpleNamespace(), gate_mode="enumerate",
                  _gate_basis=basis, bitmasks=masks)
    hold = next(i for i, p in enumerate(basis)
                if p.family == "hold" and any(p.masks))
    combo = next(i for i, p in enumerate(basis) if p.family == "combo")
    admitted = [{"addr": 0x40, "slots": [hold]},
                {"addr": 0x41, "slots": [combo]}]
    f._gate_queue_injections(admitted)
    assert f._gate_inject == [(1, 4, 0x40)]
    assert f._gate_counters["inexpressible"] == 1
    assert admitted[0]["inject"] == [1, 4]
    assert admitted[1]["inject"] is None


def test_an_admitted_candidate_becomes_a_program_the_search_will_run(tmp_path):
    # End to end, off the real sweep: measure -> rank -> admit -> QUEUE.
    # The queue is the missing link that turns a receipt row into
    # something the search does.
    f = _swept_with_samples(tmp_path)
    f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}])
    assert f._gate_counters["admitted"] == 1
    # The winning interaction was a 108-step hold of declared action 1 —
    # the pattern that actually MOVED the byte, read back off the row's
    # own contributing slots rather than re-derived from the address.
    assert f._gate_inject == [(1, 108, 0x40)]
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["queued_injections"] == [[1, 108, 0x40]]
    assert next(r for r in rec["ranked"]
                if r["addr"] == 0x40)["inject"] == [1, 108]


def test_a_band_rooted_burst_is_handed_the_queued_program_at_assignment():
    # The hand-off rule: only inside the same --gate-band the sweep
    # rooted in, round-robin so two workers never carry the same
    # candidate, and never at all with the arm off.
    inband, outband = _ocell(40, 0), _ocell(0, 0)
    queue = [(1, 4, 0x40), (2, 12, 0x41)]
    f = _assign_solver([inband, outband], inband, queue)
    c = f._assign(0)
    assert c["gate_macro"] == (1, 4) and c["gate_cand"] == 0x40
    assert [f._assign(0)["gate_macro"] for _ in range(3)] == [
        (2, 12), (1, 4), (2, 12)]

    g = _assign_solver([inband, outband], outband, queue)
    assert "gate_macro" not in g._assign(0)

    off = _assign_solver([inband], inband, queue, gate_mode="off")
    assert "gate_macro" not in off._assign(0)
    # An empty queue is the same inert path as an unarmed run.
    empty = _assign_solver([inband], inband, [])
    assert "gate_macro" not in empty._assign(0)


def test_the_injection_hand_off_costs_the_search_not_one_draw():
    # A queued program is a receipted candidate, not a sampled one. If
    # acquiring the slot moved the stream, arm C's search would diverge
    # from arm B's for a reason that has nothing to do with the
    # treatment, and C-B would measure the divergence instead of the
    # injection.
    states = []
    for queue in ([], [(1, 4, 0x40), (2, 12, 0x41)]):
        f = _assign_solver([_ocell(40, 0)], _ocell(40, 0), queue)
        f.rng = np.random.default_rng(7)
        for _ in range(8):
            f._assign(0)
        states.append(f.rng.bit_generator.state)
    assert states[0] == states[1]


def test_the_gate_injection_takes_the_slot_without_preempting_a_burst():
    # The live wiring of the same rule, run through explore()'s own
    # branch order on a hand-built ctx. A queued gate program is a
    # receipted candidate, not a sampled one, so acquiring the slot costs
    # no RNG draw and leaves a mark for the ablation.
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    body = src.split("def explore(self)", 1)[1]
    gate_at = body.index('c.get("gate_macro")')
    trans_at = body.index("self.transition_macros and")
    route_at = body.index('c.get("route_dir")')
    hold_at = body.index("self.macros\n") if "self.macros\n" in body \
        else body.index("and self.macros")
    assert gate_at < trans_at < route_at < hold_at, \
        "the macro arms are no longer ordered gate > transition > " \
        "route > hold"
    # Every arm is guarded by the same "slot is free" test, which is what
    # makes an in-flight macro un-preemptable by construction. Four arms
    # since the room-graph route-follow OR-term (T3, ROOMGRAPH_ENGINE
    # 2026-08-24 §3 row 9) joined gate/transition/hold in the slot.
    window = body[gate_at - 400:hold_at + 200]
    assert window.count('c.get("macro_left", 0) <= 0') == 4

    f = _arm_gate(SimpleNamespace(), gate_mode="enumerate")
    c = {"macro_left": 0, "steps": 41, "gate_macro": (2, 12),
         "gate_cand": 0x40, "gate_marks": []}
    owner = macro_slot_owner(c["macro_left"] > 0, bool(c.get("gate_macro")),
                             True, True)
    assert owner == "gate"
    gi, gh = c.pop("gate_macro")
    c["macro_a"], c["macro_hold"] = gi, gh
    c["macro_left"] = gh + 6
    c["gate_marks"].append((c["steps"], c.pop("gate_cand"), "pattern"))
    assert c["gate_marks"] == [(41, 0x40, "pattern")]
    # Mid-hold, no arm may claim it again.
    assert macro_slot_owner(c["macro_left"] > 0, True, True, True) \
        == "in_flight"


# ---------------------------------------------------------------------
# receipts + provenance (4; one lives in tests/test_wall_taxonomy.py and
# one in tests/test_go_explore_chain.py)
# ---------------------------------------------------------------------

def _sidecar(tmp_path, axes) -> Path:
    p = tmp_path / "gate_axes_target.json"
    p.write_text(json.dumps({"target": "t", "tuned_on": "contra",
                             "revisions": 1, "axes": axes}) + "\n")
    return p


def test_a_sidecar_axis_without_a_probe_receipt_sha_is_refused_at_merge(
        tmp_path):
    # THE injection guard: an axis nothing measured is an address an
    # agent hand-wrote, and the whole campaign's purity claim is that
    # every address came out of a receipted probe of our own rollouts.
    prof = {"solve": {"state_sig": [{"addr": 1, "match": [1]}]}}
    bad = _sidecar(tmp_path, [{"addr": 4, "match": [8]}])
    with pytest.raises(SystemExit, match="no probe-receipt sha"):
        merge_gate_axes(dict(prof), bad, 3)

    good = _sidecar(tmp_path, [{"addr": 4, "match": [8],
                                "receipt_sha": "deadbeefcafe"}])
    p = copy.deepcopy(prof)
    axes, sha = merge_gate_axes(p, good, 3)
    # APPENDED LAST, so a resumed archive's existing bit indices survive.
    assert p["solve"]["state_sig"] == [{"addr": 1, "match": [1]},
                                       {"addr": 4, "match": [8], "mod": 0}]
    assert axes[0]["receipt_sha"] == "deadbeefcafe"
    assert sha == gate_axes_sidecar_sha(good) and len(sha) == 16
    # No sidecar at all is the byte-identical default path.
    assert merge_gate_axes(dict(prof), None, 3) == ([], None)
    # The caps are refusals, not truncations.
    with pytest.raises(SystemExit, match="--contact-bits"):
        merge_gate_axes(copy.deepcopy(prof),
                        _sidecar(tmp_path, [{"addr": a, "match": [1],
                                             "receipt_sha": "x"}
                                            for a in range(4)]), 3)
    wide = {"solve": {"state_sig": [{"addr": i, "match": [1]}
                                    for i in range(7)]}}
    with pytest.raises(SystemExit, match="cap 8"):
        merge_gate_axes(wide, _sidecar(tmp_path,
                                       [{"addr": 9, "match": [1],
                                         "receipt_sha": "x"},
                                        {"addr": 10, "match": [1],
                                         "receipt_sha": "x"}]), 3)
    # An SMB-engine profile has no solve: section — merging would flip
    # the adapter as a side effect, so it is refused instead.
    with pytest.raises(SystemExit, match="solve"):
        merge_gate_axes({}, good, 3)


def test_a_negative_gate_pin_disables_the_arm_and_the_flag_is_required(
        monkeypatch):
    # Two halves of the same contract. v1 DERIVES NOTHING: there is no
    # advance-interval history to derive a pin from, and a saturated wall
    # can log zero advances all session, so the operator states it. A
    # negative value is the disable sentinel --inversion-pin-secs already
    # established.
    args = _parse_solver_argv(monkeypatch)
    assert args.gate_opener == "off" and args.gate_pin_secs is None
    assert resolve_gate_pin_secs(args) == -1.0
    assert resolve_gate_pin_secs(SimpleNamespace()) == -1.0
    assert resolve_gate_pin_secs(SimpleNamespace(gate_pin_secs="600")) == 600.0

    armed = _parse_solver_argv(monkeypatch, "--gate-opener", "enumerate",
                               "--gate-pin-secs", "-1",
                               "--gate-target-typed")
    assert resolve_gate_pin_secs(armed) == -1.0
    assert gate_armed("enumerate", 0.0, 1e9, resolve_gate_pin_secs(armed),
                      True, True) is False
    # ...and arming without stating a pin is a hard CLI error, never a
    # silent default.
    with pytest.raises(SystemExit):
        _parse_solver_argv(monkeypatch, "--gate-opener", "enumerate")


def test_the_gate_flags_are_json_types_so_receipts_keep_recording_them():
    # Same reproducibility gate the ortho knobs carry: _dump_solution
    # filters solver_args to (str, int, float, bool), so a knob typed
    # outside that set vanishes from the receipt and the run cannot be
    # replayed from its own JSON.
    import scripts.go_explore_solve as ges

    src = Path(ges.__file__).read_text()
    for decl in ('"--gate-opener", choices=("off", "enumerate")',
                 '"--gate-sweep-frac", type=float, default=0.10',
                 '"--gate-sweep-roots", type=int, default=16',
                 '"--gate-sweep-repeats", type=int, default=2',
                 '"--gate-pin-secs", type=float, default=None',
                 '"--gate-target-typed", action="store_true"',
                 '"--gate-band", type=int, default=24',
                 '"--gate-sham-roots", type=int, default=4',
                 '"--gate-axes", type=str, default=None',
                 '"--contact-bits", type=int, default=3'):
        assert decl in src, decl
    # And the header records what an A/B pair is judged on.
    hdr = gate_run_header(
        SimpleNamespace(seed=1, workers=8, burst=64, gate_opener="enumerate",
                        gate_pin_secs=600.0, gate_target_typed=True,
                        gate_band=24),
        commit="abc123", hw_flags=["mmio_read_timing"], root_sha="0" * 16,
        sidecar_sha="1" * 16, axes=[{"addr": 4}], active_predicate="p")
    for key in ("argv", "commit", "seed", "workers", "burst", "hw_flags",
                "root_sha", "sidecar_sha", "gate_axes", "active_predicate",
                "loadavg", "contact"):
        assert key in hdr, key
    # eps and K alone do not identify the test that ran: the same two
    # constants describe a settle-only window, a window slid a frame into
    # the pattern, and settle plus a closing frame. The header states the
    # window, so two arms that applied different ones cannot produce
    # headers that agree.
    assert hdr["contact"] == {
        "observable": "profile gx/y telemetry", "eps": 0, "k": 8,
        "window": ("last k+1 settle samples, closed by the pattern's "
                   "first frame; extends into that frame only when "
                   "settle alone cannot supply k+1")}
    json.dumps(hdr)          # the whole header must be serializable


# ---------------------------------------------------------------------
# LOCK OBJECTIVE — LEX-YIELD (B-O4, CONTRA_WALL_2026-08-27.md Route B).
#
# The mandatory inertness guard, in the same shape the ortho/gate/R2
# arms above already carry: T1 proves the shipped default ("off") is
# byte-identical to a run that never heard of the lock objective at
# all; T2 proves an ARMED "yield" run leaves every cell outside its own
# self-measured lock untouched; T3 is the mutation test — remove the
# guard and watch T2's own claim break, so the guard is proven load-
# bearing rather than merely present; T4 is the non-vacuity / abort
# criterion, run OFFLINE against the banked wall archive: does the
# objective actually make selection PREFER one wall state over another,
# beyond what a shuffled (label-randomized) control would produce by
# chance alone.
#
# LEX-YIELD's merit is a property of the CELL AS A ROOT (how often
# bursts launched from it taught the archive something), not of the
# specific RAM snapshot being recorded — so, unlike the survival/latch
# objectives, `observe()`'s own bookkeeping never touches `_lock_bursts`
# /`_lock_yields` (that happens in `_assign`, off the emulator's own
# burst-accounting, see the B-O4 comment there). The helpers below drive
# `observe()` directly and inject burst/yield counts by hand, which
# tests exactly the half of the mechanism `observe()` owns (merit
# lookup + the domination/record wiring) without needing a live burst
# loop for every case.
# ---------------------------------------------------------------------


def _lock_game():
    """A game whose cell_fn/area/progress are read straight off two RAM
    bytes the test controls (gx=ram[0], area=ram[2]) so a single
    observe() call stream can visit both in-lock and definitely-not-
    in-lock cells inside one archive. cell_fn's shape, (area, 3, 1, 0,
    gx), matches _ocell's convention exactly."""
    return SimpleNamespace(
        is_dead=lambda ram, lives: False,
        is_finale=lambda wd, ram: False,
        is_clear=lambda wd, ram, ctx: False,
        level_key=lambda ram: (0,),
        progress=lambda ram: ram[0],
        progress_cap=10_000,
        area=lambda ram: ram[2],
        y=lambda ram: 0,
        cell_fn=lambda ram: (ram[2], 3, 1, 0, ram[0]),
        score_bonus=lambda ram: 0.0,
    )


def _lock_ram(gx: int, area: int) -> bytes:
    b = bytearray(2048)
    b[0] = gx
    b[2] = area
    return bytes(b)


def _lock_key(gx: int, area: int) -> tuple:
    """The full 11-slot key observe() builds for `_lock_ram(gx, area)`:
    (sect, tb, kk, psig, loops, route_sig) all default + _lock_game's
    cell_fn output. key[-5] is area, key[-1] is gx — same slots the
    OBJECTIVE DESIGN's arity table names."""
    return (0, 0, 0, (), 0, (), area, 3, 1, 0, gx)


class _SpyArchive:
    """Archive stand-in that ALWAYS reports novelty (so every observe()
    call reaches the record path) and captures the (key, score, steps,
    merit) of every record() call, instead of a StubArchive's periodic-
    learn_every trick. Carries `_merit` because observe()'s domination
    peek reads `self.archive._merit` directly whenever the lock arm
    hands it a non-None candidate merit — the real GoExploreArchive
    does too (src/training/go_explore.py); this stand-in mirrors that
    exactly rather than special-casing it away."""

    def __init__(self) -> None:
        self.cells: dict = {}
        self._merit: dict = {}
        self.calls: list = []

    def record(self, ram, state, score, steps, key=None, merit=None) -> bool:
        self.calls.append((key, score, steps, merit))
        existing = self.cells.get(key)
        if existing is None:
            self.cells[key] = SimpleNamespace(
                key=key, state=(state or b"s"), best_score=score,
                best_steps=steps, visits=1, times_chosen=0, explored=False,
                barren=0)
        else:
            existing.best_score = score
            existing.best_steps = steps
            existing.state = state or existing.state
        if merit is not None:
            self._merit[key] = merit
        return True


def _lock_solver(*, lock_mode="off", lock_pin_secs=0.0, lock_band=0,
                  lock_weight=4.0, archive=None, **over):
    """_burst_solver, aimed at the lock objective: _lock_game() instead
    of _BURST_GAME (so cell_fn actually varies with the ram the test
    feeds it), ortho_mode forced off (its "up" test default would
    otherwise fire inside select() and confound a lock-only measurement
    — irrelevant to observe()-only tests but load-bearing for T4's
    select() calls), and the lock objective's own state + bound
    methods, mirroring exactly what Solver.__init__ sets up."""
    over.setdefault("ortho_mode", "off")
    f = _burst_solver([_ocell(0, 0)], learn_every=0, **over)
    f.game = _lock_game()
    if archive is not None:
        f.archive = archive
    f.lock_mode = lock_mode
    f.lock_pin_secs = lock_pin_secs
    f.lock_band = lock_band
    f.lock_weight = lock_weight
    f._lock_bursts = {}
    f._lock_yields = {}
    for name in ("_lock_armed", "_in_lock", "_lock_merit_for"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def test_lock_objective_cli_default_is_off(monkeypatch):
    args = _parse_solver_argv(monkeypatch)
    assert args.lock_objective == "off"
    assert args.lock_pin_secs == 300.0
    assert args.lock_band == 0
    assert args.lock_weight == 4.0


def test_t1_the_default_lock_stream_is_byte_identical_with_the_arm_off():
    # Same reproducibility contract as
    # test_the_credit_is_inert_at_the_shipped_throttle_default: identical
    # seed, identical archive, and the ONLY difference between the two
    # solvers is whether `lock_mode` exists on the namespace at all.
    # "off" (explicit, with every other lock_* knob set to armed-
    # looking values) must be indistinguishable from "the attribute was
    # never set" — proving the guard is the MODE STRING, not merely an
    # absent knob.
    def _picks(explicit_off: bool) -> list:
        f = _burst_solver(_burst_cells(), sel_mode="count", ortho_mode="off")
        if explicit_off:
            f.lock_mode = "off"
            f.lock_pin_secs = 0.0
            f.lock_band = 0
            f.lock_weight = 4.0
            f._lock_bursts = {}
            f._lock_yields = {}
        return _run_bursts(f, 400)

    assert _picks(True) == _picks(False)


def test_t1b_off_mode_never_writes_the_yield_bookkeeping():
    f = _lock_solver(lock_mode="off", lock_pin_secs=0.0, sel_mode="count")
    _run_bursts(f, 200)
    assert f._lock_bursts == {} and f._lock_yields == {}


def _t2_scenario(*, lock_pin_secs=0.0):
    """Seeds an archive with two area-0 (never in-lock, by area alone)
    and three area-1 cells (deepest area; only gx>=200 is in-lock),
    freezes the selection cache so _sel_topgx == 200, then observes
    three FRESH keys — (area=0, gx=51): not in lock; (area=1, gx=101):
    same area as the frontier but short of it; (area=1, gx=210): at/
    past the frontier, with a burst/yield history banked under EXACTLY
    that key — and returns (spy, results) where results maps each
    fresh key to the merit `record()` was called with."""
    spy = _SpyArchive()
    f = _lock_solver(lock_mode="yield", lock_pin_secs=lock_pin_secs,
                     archive=spy)
    for gx, area in ((50, 0), (60, 0), (100, 1), (150, 1), (200, 1)):
        f.observe(0, _lock_ram(gx, area), [], 1, "entrance", ctx={})
    f._sel_cells = None
    f._refresh_sel_cache()
    assert f.max_area == 1
    assert f._sel_topgx == 200
    f._lock_bursts[_lock_key(210, 1)] = 10
    f._lock_yields[_lock_key(210, 1)] = 7
    spy.calls.clear()

    fresh = ((51, 0), (101, 1), (210, 1))
    for gx, area in fresh:
        f.observe(0, _lock_ram(gx, area), [], 1, "entrance", ctx={})
    by_key = {c[0]: c[3] for c in spy.calls}
    return spy, {(gx, area): by_key[_lock_key(gx, area)] for gx, area in fresh}


def test_t2_yield_merit_is_none_off_lock_and_a_real_float_on_it():
    _, merits = _t2_scenario()
    assert merits[(51, 0)] is None       # different area entirely
    assert merits[(101, 1)] is None      # right area, short of the frontier
    lock_merit = merits[(210, 1)]
    assert lock_merit is not None        # AT the frontier: non-vacuous
    assert lock_merit == pytest.approx((7.0 + 1.0) / (10.0 + 2.0))
    assert 0.0 < lock_merit < 1.0        # Laplace-smoothed: never 0 or 1


def test_t2b_unarmed_yield_leaves_even_the_frontier_cell_at_none():
    # Same scenario, but the pin clock has not run long enough — the
    # SAME cell that scored a real merit above must fall back to None
    # the moment "armed" stops being true, proving the pin gate (not
    # just the area/gx predicate) is load-bearing on its own.
    _, merits = _t2_scenario(lock_pin_secs=1e9)
    assert merits == {(51, 0): None, (101, 1): None, (210, 1): None}


def test_t3_removing_the_in_lock_guard_breaks_t2s_own_claim(monkeypatch):
    import scripts.go_explore_solve as ges

    # Baseline (already proven above): off-lock keys get merit=None.
    _, before = _t2_scenario()
    assert before[(51, 0)] is None and before[(101, 1)] is None

    # THE MUTATION: revert the scope predicate to "everything is in the
    # lock". If in_lock_key were not load-bearing, this would change
    # nothing; it must break the T2 claim for BOTH previously-excluded
    # keys.
    monkeypatch.setattr(ges, "in_lock_key", lambda *a, **k: True)
    _, after = _t2_scenario()
    assert after[(51, 0)] is not None
    assert after[(101, 1)] is not None


def test_t3b_removing_the_armed_guard_breaks_t2bs_own_claim(monkeypatch):
    import scripts.go_explore_solve as ges

    # Baseline: an unarmed run (pin clock nowhere near lock_pin_secs)
    # gives merit=None even AT the frontier cell (T2b).
    _, before = _t2_scenario(lock_pin_secs=1e9)
    assert before[(210, 1)] is None

    # THE MUTATION: the arming clock always reports armed. If
    # lock_armed's result were not load-bearing, this would change
    # nothing; it must turn the frontier cell's merit non-None despite
    # the pin clock never having run.
    monkeypatch.setattr(ges, "lock_armed", lambda *a, **k: True)
    _, after = _t2_scenario(lock_pin_secs=1e9)
    assert after[(210, 1)] is not None


# ---------------------------------------------------------------------
# T4 — the non-vacuity / abort criterion. Offline, no emulation: loads
# the banked wall archive from the Contra Route-A/wall campaign and
# asks the real select() code path whether a discriminating merit
# vector actually concentrates picks, with a shuffle control so "the
# statistic moved" cannot be explained by anything other than the real
# key-to-merit mapping.
# ---------------------------------------------------------------------

_WALL_ARCHIVE = (Path(__file__).resolve().parent.parent
                / "runs/play_one_well/contra/solve20/archive.pkl")


def _load_wall_cells() -> list:
    import pickle

    with open(_WALL_ARCHIVE, "rb") as fh:
        cells = pickle.load(fh)
    wall = [c for c in cells.values() if c.key[-1] == 192]
    assert len({c.key[0] for c in wall}) == 1   # single max_sect, as banked
    assert len({c.key[-5] for c in wall}) == 1  # single max_area, as banked
    return wall


def _wall_solver(wall_cells, *, deep_bias: float, sel_mode: str,
                 lock_mode: str = "yield"):
    f = SimpleNamespace(
        args=SimpleNamespace(deep_bias=deep_bias),
        rng=np.random.default_rng(0),
        archive=SimpleNamespace(cells={c.key: c for c in wall_cells}),
        max_area=wall_cells[0].key[-5], max_sect=wall_cells[0].key[0],
        sel_mode=sel_mode, frontier_throttle=0,
        # D1: see _ortho_solver's identical comment.
        transit_deep_relax=0,
        door_weight=0.0, _doors=frozenset(), _key_ids={},
        ortho_mode="off", ortho_pin_secs=1e18, ortho_bias=0.0,
        ortho_band=1, ortho_weight=4.0, _pin_time=0.0,
        gate_mode="off", gate_weight=1.0,
        _ortho_pool=[], _ortho_ids=set(), _ortho_ext={},
        _ortho_deep_yband=None, _ortho_selections=0,
        _ortho_cols_improved=0, _gx_phantoms=set(),
        _sel_cells=None, _sel_n=0, _sel_area=None,
        lock_mode=lock_mode, lock_pin_secs=0.0, lock_band=0,
        lock_weight=4.0, _lock_bursts={}, _lock_yields={})
    for name in ("_refresh_sel_cache", "_ortho_armed", "select",
                "_lock_armed", "_in_lock", "_lock_merit_for"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def _draw_picks(f, n: int) -> dict:
    counts: dict = {}
    for _ in range(n):
        c = f.select()
        counts[c.key] = counts.get(c.key, 0) + 1
    return counts


def _tv_distance(a: dict, b: dict, keys) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def _decile_ratio(counts: dict, merit: dict, keys: list) -> float:
    ordered = sorted(keys, key=lambda k: merit[k])
    n = max(1, len(ordered) // 10)
    bottom, top = ordered[:n], ordered[-n:]
    bot = sum(counts.get(k, 0) for k in bottom) or 1
    topc = sum(counts.get(k, 0) for k in top)
    return topc / bot


@pytest.mark.skipif(not _WALL_ARCHIVE.exists(),
                    reason="banked Contra wall archive not present "
                          "(gitignored runs/ artifact)")
def test_t4_lex_yield_actually_discriminates_among_wall_states():
    wall = _load_wall_cells()
    keys = [c.key for c in wall]
    rng = np.random.default_rng(7)
    # A pre-registered bimodal merit: the top decile is a clear high-
    # yield population, everything else is low-yield — exactly the
    # shape a real "some roots are far more generative than others"
    # regime would produce, without tuning to the specific archive.
    order = list(keys)
    rng.shuffle(order)
    n_hi = max(1, len(order) // 10)
    hi, lo = set(order[:n_hi]), set(order[n_hi:])
    bursts, yields = {}, {}
    for k in keys:
        b = 20
        y = 19 if k in hi else 0
        bursts[k], yields[k] = b, y
    merit = {k: (yields[k] + 1.0) / (bursts[k] + 2.0) for k in keys}

    def _armed(deep_bias, sel_mode, shuffled=False):
        f = _wall_solver(wall, deep_bias=deep_bias, sel_mode=sel_mode)
        if shuffled:
            shuffled_keys = list(keys)
            rng.shuffle(shuffled_keys)
            f._lock_bursts = dict(zip(shuffled_keys, (bursts[k] for k in keys)))
            f._lock_yields = dict(zip(shuffled_keys, (yields[k] for k in keys)))
        else:
            f._lock_bursts, f._lock_yields = dict(bursts), dict(yields)
        f._refresh_sel_cache()
        return _draw_picks(f, 20_000)

    off = _draw_picks(_wall_solver(wall, deep_bias=1.0, sel_mode="legacy",
                                   lock_mode="off"), 20_000)
    real = _armed(1.0, "legacy")
    sham = _armed(1.0, "legacy", shuffled=True)

    tv_real = _tv_distance(real, off, keys)
    ratio_real = _decile_ratio(real, merit, keys)
    ratio_sham = _decile_ratio(sham, merit, keys)

    assert tv_real >= 0.20, f"deep arm TV distance too small: {tv_real}"
    assert ratio_real >= 3.0, f"deep arm decile ratio too small: {ratio_real}"
    # The shuffle control: TV-from-off is NOT the right statistic to
    # compare here — a shuffled run still boosts *some* random 10% of
    # cells, so it departs from "off" by a similar aggregate amount
    # (tv_sham stays large; the boost happened, just not where it was
    # supposed to). What must collapse under shuffling is the
    # CORRESPONDENCE between the ORIGINAL merit ranking and where the
    # picks actually landed: real merit concentrates picks on the cells
    # that ARE the top decile; a shuffled key-to-merit mapping decouples
    # that correspondence, so ratio computed against the pre-shuffle
    # merit dict must collapse toward "no preference" (a ratio of 1).
    assert ratio_real > 3 * ratio_sham, (ratio_real, ratio_sham)
    assert ratio_sham < 2.0, f"shuffle control did not collapse: {ratio_sham}"

    # Secondary check: the count arm (the OTHER wired call site) also
    # discriminates, at a lighter bar — its multiplier composes with the
    # pre-existing score/count terms rather than replacing them outright.
    real_count = _armed(0.0, "count")
    ratio_count = _decile_ratio(real_count, merit, keys)
    assert ratio_count >= 1.5, f"count arm decile ratio too small: {ratio_count}"
