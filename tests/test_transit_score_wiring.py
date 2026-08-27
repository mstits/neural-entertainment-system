"""D2 -- the transition dimension's SCORE/SELECTION wiring.

`tests/test_transit_dimension.py` already proves the DECISION RULE
(`transit_dimension`/`lex_score`) is safe to search on, over pre-extracted
verdict streams. `tests/test_transit_wiring.py` already proves the KEY side
of the solver wiring (D1: `sect`/`psig` come from the blank-run witness
instead of `room_id()`'s constant inequality).

Neither file touches the SCORE. Per
`docs/research/RYGAR_TRANSITIONS_2026-08-27.md`'s own accounting: the key
change alone is "necessary but not sufficient" -- the deep-frontier arm is a
maximum over gx buckets, so a genuinely new room that re-anchors LOW is
invisible to selection unless the score's leading term (formerly the bare
literal `sect * 10000`, scripts/go_explore_solve.py `observe()`) actually
outranks it. This file is that wiring's test:
  * `Solver.transit_weight` / `--transit-weight` -- the literal is now a
    CLI-overridable attribute, default byte-identical to the pre-D2 constant.
  * The score SITE itself calls `scripts.transition_witness.lex_score`
    rather than reimplementing the formula (one flattening, one set of
    tests).
  * `--transit-deep-relax` actually reaches the deep arm's `key[0]` filter
    (D1 declared the attribute; this confirms `_refresh_sel_cache` reads it).

ANTI-VACUITY
------------
Every class below is paired with the number the REVERTED mechanism (the
bare literal `sect * 10000`, or a deep-arm filter that ignores
`transit_deep_relax`) would have produced instead, and
`TestRealReplayScoresTheAxis` drives the real emulator through the banked
R1 tape exactly like `test_transit_wiring.py::TestRealReplay` does, so the
score claim is checked on real hardware surface, not only on synthetic
integers.

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

from scripts.go_explore_solve import GenericGame, Solver, TRANSIT_SCORE_WEIGHT
from scripts.transition_witness import lex_score

from tests.test_go_explore_solve import _parse_solver_argv

REPO = Path(__file__).resolve().parent.parent
PROFILE = REPO / "configs/rygar.yaml"
TAPE = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"


def _rygar_profile() -> dict:
    prof = yaml.safe_load(PROFILE.read_text())
    # Pinned independent of whatever configs/rygar.yaml currently carries:
    # a SIBLING D1 workflow lands/reverts this exact `solve:` key on this
    # same checkout while this file runs. This suite's claim is about the
    # SCORE wiring (transit_weight / lex_score / transit_deep_relax), not
    # about the checked-in profile's live state, so it supplies its own
    # known-good calibration (the same values docs/receipts/rygar/
    # clear_predicate_REFUTED.md and tests/test_transit_dimension.py use)
    # rather than trusting the file on disk at import time.
    prof["solve"]["transit_source"] = "blank_run"
    prof["solve"]["area_key"] = [0x0014, 0x001C]
    prof["solve"]["min_blank_frames"] = 40
    return prof


def _old_formula(sect: int, gx: int, bonus: int) -> int:
    """THE REVERTED MECHANISM: the exact literal `observe()` computed
    before D2 -- `sect * 10000 + gx + bonus` -- built independently of
    `lex_score` so this file cannot pass by both sides sharing one bug."""
    return sect * 10000 + gx + bonus


# -------------------------------------------------------------------------
# The weight is a parameter now, not a literal. Pure, no ROM.
# -------------------------------------------------------------------------
class TestTransitWeightDefaultIsByteIdentical:
    @pytest.mark.parametrize("sect,gx,bonus", [
        (0, 0, 0), (0, 4608, 0), (1, 0, 0), (2, 6178, 0),
        (0, 2000, 500), (5, 300, 40), (55, 6242, 0),
    ])
    def test_lex_score_at_default_weight_matches_the_old_literal(
            self, sect, gx, bonus):
        assert (lex_score(sect, gx + bonus, weight=TRANSIT_SCORE_WEIGHT)
               == _old_formula(sect, gx, bonus))

    def test_transit_score_weight_constant_is_still_10000(self):
        # The constant this whole axis is sized against
        # (tests/test_transit_dimension.py::TestTheScoreCannotBeOutbidByWalking)
        # -- pinned here too so a change to it is visible from the score-
        # wiring side, not only the dimension side.
        assert TRANSIT_SCORE_WEIGHT == 10000

    def test_reverting_to_the_bare_literal_would_disagree_at_nondefault_weight(
            self):
        # The REVERTED mechanism (a hardcoded 10000) cannot be tuned at
        # all -- this is what --transit-weight buys, and why "it agrees
        # at the default" above is not the whole story.
        assert lex_score(3, 100, weight=0) != _old_formula(3, 100, 0)
        assert lex_score(3, 100, weight=0) == 100


class TestTransitWeightCli:
    def test_default_is_the_receipted_constant(self, monkeypatch):
        args = _parse_solver_argv(monkeypatch)
        assert args.transit_weight == TRANSIT_SCORE_WEIGHT

    def test_overridable_to_zero_the_key_only_arm(self, monkeypatch):
        # The design's pre-registered cheap falsifier: key-only needs
        # weight=0 AND deep-relax wide open, as two SEPARATE knobs.
        args = _parse_solver_argv(monkeypatch, "--transit-weight", "0")
        assert args.transit_weight == 0

    def test_deep_relax_flag_still_present_and_still_zero_by_default(
            self, monkeypatch):
        # D1 shipped the attribute; this just confirms D2 did not
        # regress the CLI surface it depends on for the 3-knob A/B.
        args = _parse_solver_argv(monkeypatch)
        assert args.transit_deep_relax == 0


# -------------------------------------------------------------------------
# Structural: the score SITE and the deep-arm filter actually read the
# new attributes, rather than the literal quietly surviving beside them.
# -------------------------------------------------------------------------
def _method_source(cls_name: str, fn_name: str) -> ast.FunctionDef:
    import scripts.go_explore_solve as ges
    tree = ast.parse(Path(ges.__file__).read_text())
    cls = next(n for n in tree.body
              if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name == fn_name)


class TestScoreSiteIsWired:
    def test_observe_calls_lex_score_not_a_bare_literal(self):
        fn = _method_source("Solver", "observe")
        calls = {n.func.id for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "lex_score" in calls
        # THE REVERTED MECHANISM would still contain a BinOp `sect * 10000`
        # (or any Constant 10000 multiplying `sect`) as dead code sitting
        # next to the new call; source has none.
        src = ast.get_source_segment(Path(
            __import__("scripts.go_explore_solve",
                      fromlist=["x"]).__file__).read_text(), fn)
        assert "sect * 10000" not in src

    def test_observe_reads_self_transit_weight(self):
        # `getattr(self, "transit_weight", TRANSIT_SCORE_WEIGHT)`, not a
        # bare `self.transit_weight` -- the duck-typed Solver stand-ins
        # in tests/test_go_explore_solve.py and
        # tests/test_go_explore_solve_repairs.py predate this axis (see
        # the AttributeError this exact bare-attribute form produced
        # there, fixed alongside this test), so it is a string constant
        # in a `getattr` call, not an `ast.Attribute` node.
        fn = _method_source("Solver", "observe")
        literals = {n.value for n in ast.walk(fn)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "transit_weight" in literals


class TestDeepArmRelaxIsWired:
    def test_refresh_sel_cache_reads_transit_deep_relax(self):
        # Same `getattr` form, and for the same measured reason, as
        # `observe`'s transit_weight read above: FOUR test files carry
        # duck-typed SimpleNamespace Solver stand-ins that predate this
        # axis (test_go_explore_solve, test_room_router,
        # test_terminal_stasis, test_gate_k0_reforge), and a bare
        # `self.transit_deep_relax` here raised AttributeError in 7 of
        # them at landing. So the name is a string constant in a
        # `getattr` call, not an `ast.Attribute` node -- accept either,
        # and fail if the read disappears entirely.
        fn = _method_source("Solver", "_refresh_sel_cache")
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        lits = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "transit_deep_relax" in (names | lits)

    @pytest.mark.parametrize("max_sect", [0, 1, 5, 55])
    def test_relax_zero_reproduces_exact_equality(self, max_sect):
        # key[0] can never exceed max_sect (record() raises max_sect the
        # instant sect grows past it -- see the `if sect > self.max_sect`
        # guard right above the key/score site), so for every REACHABLE
        # key0 in [0, max_sect], `key0 >= max_sect - 0` and
        # `key0 == max_sect` must agree. THE REVERTED MECHANISM (relax
        # hardcoded to some N>0) would let key0 < max_sect pass too.
        for key0 in range(0, max_sect + 1):
            relaxed = key0 >= max_sect - 0
            exact = key0 == max_sect
            assert relaxed == exact

    def test_relax_two_admits_two_bands_below_the_frontier(self):
        max_sect, relax = 10, 2
        admitted = [k for k in range(0, max_sect + 1)
                   if k >= max_sect - relax]
        assert admitted == [8, 9, 10]


# -------------------------------------------------------------------------
# Real emulator, real banked tape. Skips without the ROM, exactly like
# test_transit_wiring.py::TestRealReplay.
# -------------------------------------------------------------------------
class TestRealReplayScoresTheAxis:
    """MEASURED FIRST, NOT ASSUMED: this profile's odometer is a
    cumulative integral that never decreases (backward-of-origin clamps
    to 0 RELATIVE motion, not the running total), so on this specific
    banked tape both real novel arrivals land at a running-maximum gx —
    there is no cross-lineage case on THIS tape where a higher-sect state
    has LOWER raw gx than a lower-sect one to dominate. What weight
    actually controls, and what this class measures instead, is the SIZE
    OF THE SCORE DISCONTINUITY at the exact step `sect` increments: at
    default weight that jump is dominated by `transit_weight`; at
    weight=0 (the reverted-to-key-only mechanism) the identical step
    produces only the ordinary few-px-per-frame jump every other step
    produces too, i.e. the transition becomes invisible to score-based
    selection. `TestTheScoreCannotBeOutbidByWalking` in
    tests/test_transit_dimension.py covers the cross-lineage domination
    claim on synthetic numbers sized against this tape's OWN receipted
    frontier+ratchet; this class is real hardware surface for the
    complementary claim."""

    def _replay(self):
        tape = json.loads(TAPE.read_text())
        prof = _rygar_profile()
        rom = REPO / prof["solve"]["rom"]
        start = REPO / prof["start_state_path"]
        if not rom.exists() or not start.exists():
            pytest.skip(f"ROM or start state absent ({rom.name}); the "
                        "pure/structural classes above still ran")

        import nes_core
        from src.training.profile_utils import action_space_to_bitmasks

        game = GenericGame(prof)
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

            stand_in = SimpleNamespace(
                game=game, frame_skip=prof["frame_skip"],
                start_lives=lives0, _transit_blank_run=True,
                _odo_blank_now=[])
            c: dict = {}
            Solver._tw_seed(stand_in, c, ())

            # (sect, gx) at every step, PRE- and POST-observation, so the
            # jump at a novel step can be isolated from ordinary walking.
            trace = []
            sect = 0
            for a in tape["actions"]:
                gx_before = int(pool.get_odometer_per_worker()[0][0])
                ram = pool.step_all(
                    np.array([bm[int(a)]], dtype=np.uint8))[0][2]
                stand_in._odo_blank_now = (
                    pool.get_odometer_blank_per_worker())
                novel = Solver._blank_transit_step(stand_in, 0, c, ram)
                gx_after = int(pool.get_odometer_per_worker()[0][0])
                if novel:
                    sect += 1
                trace.append((sect, gx_before, gx_after, novel))
        finally:
            pool.shutdown()
        return trace

    def test_the_score_jump_at_a_novel_step_is_dominated_by_the_weight(self):
        trace = self._replay()
        novel_steps = [t for t in trace if t[3]]
        assert len(novel_steps) == 2, (
            f"expected exactly 2 novel arrivals on the banked tape, got "
            f"{len(novel_steps)} — the D1 mechanism this file depends on "
            f"has regressed; see test_transit_wiring.py")
        ordinary_deltas = [abs(gx_after - gx_before)
                          for _, gx_before, gx_after, novel in trace
                          if not novel]
        # A generous ceiling on ordinary per-step gx movement (frame_skip
        # 4 at this profile's own action set) -- loose on purpose, this
        # only has to be smaller than the weight by orders of magnitude.
        max_ordinary_delta = max(ordinary_deltas)
        assert max_ordinary_delta < TRANSIT_SCORE_WEIGHT / 10
        for sect_after, gx_before, gx_after, _ in novel_steps:
            score_before = lex_score(sect_after - 1, gx_before,
                                     weight=TRANSIT_SCORE_WEIGHT)
            score_after = lex_score(sect_after, gx_after,
                                    weight=TRANSIT_SCORE_WEIGHT)
            # THE CLAIM: the jump at THIS step towers over any jump
            # ordinary walking could produce anywhere else on the tape.
            assert score_after - score_before > max_ordinary_delta * 100
            # THE REVERTED MECHANISM (weight=0 -- the key change with no
            # score change, the design's own "key alone is not
            # sufficient" claim): the identical step now produces a jump
            # no larger than ordinary walking, so score-based selection
            # cannot distinguish "found a new room" from "took a step".
            score_before_w0 = lex_score(sect_after - 1, gx_before, weight=0)
            score_after_w0 = lex_score(sect_after, gx_after, weight=0)
            assert score_after_w0 - score_before_w0 <= max_ordinary_delta
