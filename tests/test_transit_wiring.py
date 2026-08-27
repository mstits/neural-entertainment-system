"""D1 -- the transition dimension's SOLVER WIRING.

Two other files already exist and this one deliberately does not repeat
them:
  * `tests/test_transition_witness.py` proves the blank-fold RUN COUNTER
    is sound on Rygar's own rollouts.
  * `tests/test_transit_dimension.py` proves the NOVELTY-GATED DIMENSION
    (`transit_dimension`/`lex_score`) is safe to search on, over
    pre-extracted verdict streams.

Neither exercises the solver at all. Per
`docs/research/RYGAR_TRANSITIONS_2026-08-27.md`'s own accounting: "this
suite tests the DECISION RULE, not the solver wiring -- the wiring does
not exist yet." This file is that wiring's test: `GenericGame.transit_source
/ .area_key / .min_blank_frames` (the `solve:` config surface) and
`Solver._tw_seed` / `Solver._blank_transit_step` (the two new methods that
occupy the room_id()-inequality's exact slot in the per-worker loop,
scripts/go_explore_solve.py:explore()).

ANTI-VACUITY
------------
`TestRealReplay` is the load-bearing class: it drives the REAL emulator
through the banked R1 tape (docs/receipts/rygar/r1_tape_gx6242.json, 6,018
actions, the same tape docs/receipts/rygar/clear_predicate_REFUTED.md and
tests/test_rygar_r1_tape.py already certify) through the two new Solver
methods, and in the SAME pass through the ORIGINAL room_id()-inequality
test those methods replace -- proving the mechanism, on real hardware
surface, and proving what its absence (the pre-existing default,
`transit_source: room_id`) produces on the identical steps: 0. Skips
without the ROM (gitignored, not distributable), exactly like
tests/test_rygar_r1_tape.py::TestTapeReplays; the structural classes below
it do not skip.

Ledger: EXHIBITION. No policy is trained or evaluated by anything here.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.go_explore_solve import GenericGame, Solver

REPO = Path(__file__).resolve().parent.parent
PROFILE = REPO / "configs/rygar.yaml"
TAPE = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"

N_NOVEL = 2   # same figure tests/test_transit_dimension.py pins


def _rygar_profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


# -------------------------------------------------------------------------
# The config surface. No ROM needed.
# -------------------------------------------------------------------------
class TestConfigSurface:
    def test_rygar_is_armed_for_blank_run_with_its_own_calibration(self):
        game = GenericGame(_rygar_profile())
        assert game.transit_source == "blank_run"
        assert game.area_key_addrs == (0x0014, 0x001C)
        assert game.min_blank_frames == 40

    def test_the_default_is_room_id_and_every_other_profile_is_untouched(self):
        # Every profile that has never heard of this axis (all 70 others)
        # must parse to the pre-existing constant behaviour.
        prof = _rygar_profile()
        del prof["solve"]["transit_source"]
        del prof["solve"]["area_key"]
        del prof["solve"]["min_blank_frames"]
        game = GenericGame(prof)
        assert game.transit_source == "room_id"
        assert game.area_key_addrs == ()
        assert game.min_blank_frames == 0

    def test_blank_run_without_an_area_key_refuses_to_construct(self):
        prof = _rygar_profile()
        prof["solve"]["area_key"] = []
        with pytest.raises(SystemExit):
            GenericGame(prof)

    def test_blank_run_without_a_calibrated_floor_refuses_to_construct(self):
        prof = _rygar_profile()
        prof["solve"]["min_blank_frames"] = 0
        with pytest.raises(SystemExit):
            GenericGame(prof)

    def test_an_unknown_transit_source_refuses_to_construct(self):
        prof = _rygar_profile()
        prof["solve"]["transit_source"] = "bogus"
        with pytest.raises(SystemExit):
            GenericGame(prof)

    def test_area_key_reads_the_named_bytes_in_order(self):
        game = GenericGame(_rygar_profile())
        ram = np.zeros(2048, dtype=np.uint8)
        ram[0x0014] = 3
        ram[0x001C] = 7
        assert game.area_key(ram) == (3, 7)
        # room_id() stays the constant function it was measured to be --
        # this axis routes AROUND it, it does not fix it.
        assert game.room_id(ram) == (0,)
        ram[0x0014] = 9
        assert game.room_id(ram) == (0,)


# -------------------------------------------------------------------------
# The per-worker loop actually calls the new methods, under the mode flag.
# Structural, so a future edit that quietly un-wires this fails loudly
# rather than only showing up as a flat frontier weeks later.
# -------------------------------------------------------------------------
class TestModeSwitchIsWired:
    def _explore_source(self) -> ast.FunctionDef:
        import scripts.go_explore_solve as ges
        tree = ast.parse(Path(ges.__file__).read_text())
        cls = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "Solver")
        return next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "explore")

    def test_explore_calls_blank_transit_step_under_the_mode_flag(self):
        names = {n.attr for n in ast.walk(self._explore_source())
                if isinstance(n, ast.Attribute)}
        assert "_blank_transit_step" in names
        assert "_transit_blank_run" in names
        # room_id() must still be the OTHER branch, not deleted -- the
        # contract is a mode switch, not a replacement.
        assert "room_id" in names

    def test_assign_seeds_the_witness_on_both_the_root_and_cell_branches(self):
        import scripts.go_explore_solve as ges
        tree = ast.parse(Path(ges.__file__).read_text())
        cls = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "Solver")
        fn = next(n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_assign")
        calls = [n.func.attr for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_tw_seed"]
        assert len(calls) == 2, (
            "expected exactly 2 call sites (entrance root + restored "
            f"cell), found {len(calls)}")


# -------------------------------------------------------------------------
# Real emulator, real banked tape. Skips without the ROM.
# -------------------------------------------------------------------------
class TestRealReplay:
    def test_wiring_recovers_two_novel_areas_where_room_id_recovers_none(
            self):
        assert TAPE.exists(), f"{TAPE} is missing"
        tape = json.loads(TAPE.read_text())
        prof = _rygar_profile()
        rom = REPO / prof["solve"]["rom"]
        start = REPO / prof["start_state_path"]
        if not rom.exists() or not start.exists():
            pytest.skip(f"ROM or start state absent ({rom.name}); the "
                        "TestConfigSurface/TestModeSwitchIsWired "
                        "assertions above still ran")

        import nes_core
        from src.training.profile_utils import action_space_to_bitmasks

        game = GenericGame(prof)
        assert game.transit_source == "blank_run"
        bm = action_space_to_bitmasks(prof["action_space"])
        pool = nes_core.Pool(rom_path=str(rom), num_workers=1,
                             frame_skip=prof["frame_skip"])
        try:
            pool.set_headless(True)
            pool.set_skip_preprocess(True)
            pool.set_odometer_enabled(True)
            pool.reset_all()
            pool.load_worker_state(0, start.read_bytes())
            ram0 = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
            lives0 = int(ram0[0x0303])

            # A stand-in, not a full Solver -- same convention the rest of
            # this file's siblings use for duck-typed method tests. It
            # carries exactly what _tw_seed/_blank_transit_step read.
            stand_in = SimpleNamespace(
                game=game, frame_skip=prof["frame_skip"],
                start_lives=lives0, _transit_blank_run=True,
                _odo_blank_now=[])
            c: dict = {}
            Solver._tw_seed(stand_in, c, ())

            sect_blank_run = 0
            sect_room_id = 0   # THE REVERTED MECHANISM, replayed in lockstep
            p0750 = None
            for a in tape["actions"]:
                ram = pool.step_all(
                    np.array([bm[int(a)]], dtype=np.uint8))[0][2]
                stand_in._odo_blank_now = (
                    pool.get_odometer_blank_per_worker())
                if Solver._blank_transit_step(stand_in, 0, c, ram):
                    sect_blank_run += 1
                _rid = game.room_id(ram)
                if p0750 is not None and _rid != p0750:
                    sect_room_id += 1
                p0750 = _rid
        finally:
            pool.shutdown()

        assert sect_blank_run == N_NOVEL, (
            f"expected exactly {N_NOVEL} novel-area arrivals on the "
            f"banked tape, got {sect_blank_run}")
        # THE MECHANISM ABSENT (the pre-existing default, byte-identical
        # to every run before this axis existed): room_id() is a constant
        # function on this profile, so its inequality test never fires,
        # on the SAME 6,018 real emulator steps that just banked 2 above.
        assert sect_room_id == 0, (
            "room_id() fired on Rygar -- the profile's own constancy "
            "proof (docs/receipts/rygar/clear_predicate_REFUTED.md) no "
            "longer holds and this whole axis's premise needs re-checking")


# -------------------------------------------------------------------------
# THE RESUME REFUSAL (landed 2026-08-27 with
# docs/research/RYGAR_TRANSITIONS_2026-08-27.md, from a defect the
# adjudication of runs/rygar_transitions/D2/run1 found in banked output).
#
# `seen` is the 9th trace element. An archive written before this axis
# existed has 7- or 8-tuples and therefore no occupied-area set, so every
# lineage restored from one starts empty and re-banks rooms it has already
# visited -- the design's "treadmill one restore deep", live. The error
# direction is FABRICATION, so the resume is refused rather than warned.
# No ROM needed by anything in this class.
# -------------------------------------------------------------------------
class TestLegacyArchiveResumeIsRefused:
    def test_a_full_arity_archive_resumes(self):
        from scripts.go_explore_solve import check_transit_resume
        rec = (0, b"", 0, (), 0, (), 0, (), ((0, 14),))
        assert len(rec) == 9
        check_transit_resume({("k",): rec}, Path("/nowhere"))   # no raise

    def test_an_empty_archive_resumes(self):
        from scripts.go_explore_solve import check_transit_resume
        check_transit_resume({}, Path("/nowhere"))              # no raise

    @pytest.mark.parametrize("arity", [7, 8])
    def test_a_pre_axis_archive_is_refused_and_says_how_many(self, arity):
        from scripts.go_explore_solve import check_transit_resume
        legacy = tuple(range(arity))
        full = (0, b"", 0, (), 0, (), 0, (), ((0, 14),))
        loaded = {("a",): legacy, ("b",): legacy, ("c",): full}
        with pytest.raises(SystemExit) as exc:
            check_transit_resume(loaded, Path("/prev"))
        msg = str(exc.value)
        assert "2 of 3" in msg
        assert "transit_source" in msg

    def test_seed_calls_the_check_under_the_mode_flag(self):
        # Structural: the refusal is worthless if `seed()` stops calling
        # it. Reverting the call site makes exactly this fail.
        import scripts.go_explore_solve as ges
        tree = ast.parse(Path(ges.__file__).read_text())
        cls = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "Solver")
        fn = next(n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "seed")
        calls = [n.func.id for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "check_transit_resume"]
        assert len(calls) == 1, (
            f"expected exactly 1 call site in Solver.seed(), found "
            f"{len(calls)}")
        # The gate is `getattr(self, "_transit_blank_run", False)`, so
        # the flag name is a string constant here, not an attribute node.
        lits = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert "_transit_blank_run" in (lits | names), (
            "the check must be gated on the mode flag, or every room_id "
            "profile's resume would start refusing legacy archives")

    def test_it_would_have_caught_the_run_that_found_the_defect(self):
        # The actual banked output. runs/ is gitignored, so this skips
        # off this machine -- the synthetic cases above do not.
        import pickle
        from scripts.go_explore_solve import check_transit_resume
        prev = REPO / "runs/rygar_transitions/D2/run1"
        if not (prev / "traces.pkl").exists():
            pytest.skip("runs/rygar_transitions/D2/run1 is not on this "
                        "machine (runs/ is gitignored)")
        with open(prev / "traces.pkl", "rb") as f:
            loaded = pickle.load(f)
        with pytest.raises(SystemExit) as exc:
            check_transit_resume(loaded, prev)
        # The exact shape of the defect: a majority-full archive whose
        # minority of legacy records is enough to poison the gate.
        assert "493 of 1647" in str(exc.value)
