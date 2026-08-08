"""Tests for scripts/go_explore_solve.py's pure helpers and GenericGame.

Kept separate from tests/test_go_explore.py (which covers the archive
in src/training/go_explore.py). GenericGame's __init__ only reads the
profile dict (no ROM/Pool I/O), so it's constructible here with a
minimal in-memory profile; Solver itself needs a real ROM/Pool and is
only exercised via progress_line() through a duck-typed stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.go_explore_solve import (
    GenericGame,
    Solver,
    apply_hw_flags,
    available_hw_flags,
    check_state_sidecar,
    hw_provenance,
    resolve_hw_flags,
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
        door_weight=0, transition_macros=[],
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

def _dump_one_solution(tmp_path, flags=("mmio_read_timing",)) -> dict:
    """Drive Solver._dump_solution through a duck-typed stand-in and
    return the receipt it wrote. Only the attributes the method actually
    reads are supplied."""
    (tmp_path / "solutions").mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        best_sol_len=10**9, sol_counter=0, n_solutions=0,
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
