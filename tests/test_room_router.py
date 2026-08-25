"""Room-graph engine T3: router arm + telemetry
(ROOMGRAPH_ENGINE_2026-08-24 §3 rows 7-9, 12).

Everything here runs without a ROM or a Pool: the router is
selection-side only, so the arm is exercised through the same
duck-typed Solver stand-ins the ortho-arm suite uses (real methods
bound to a namespace), and the graph math through its pure helpers.
The done-when pair from §9/T3 lives here: arm-off byte-identity (the
draw ledger + the pick-stream parity check) and the synthetic-archive
selection tests (frontier membership, articulation/U(r) up-weights,
aliased down-weight, all under the pure count prior).
"""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts.go_explore_solve import (
    ROOM_UNKNOWN,
    RoomIndex,
    Solver,
    aliased_rooms,
    derive_direction_macros,
    room_boundaries,
    room_cell_ord,
    room_frontier,
    room_weight,
    route_near_side,
)

#: room_sig [0x804, 0x805]'s psig extraction triple, as Solver.__init__
#: derives it: (lo offset from tail end, hi offset, room_sig arity).
PSIG_OFF = (-2, -1, 2)

ACTS = [["noop"], ["right"], ["right", "a"], ["left"], ["up"],
        ["down", "b"]]


def _rcell(gx, yb, *, ord_=None, area=0, barren=0, chosen=0, sect=1):
    """A duck-typed archive cell whose psig tail carries a room ordinal
    the way a room_sig [0x804, 0x805] profile threads it: room_id() =
    level_key + (area,) + (ordinal lo, ordinal hi)."""
    psig = (() if ord_ is None
            else (area, ord_ & 0xFF, (ord_ >> 8) & 0xFF))
    key = (sect, 0, 0, psig, 0, ()) + (area, 3, 1, yb, gx)
    return SimpleNamespace(key=key, state=b"s", best_score=1.0,
                           best_steps=1, visits=1, times_chosen=chosen,
                           explored=False, barren=barren)


def _room_solver(cells, index=None, **over):
    """Duck-typed Solver for the router arm: the attributes select() /
    _refresh_sel_cache() read, with every OTHER arm at its inert
    setting (deep_bias 0, ortho off, count off => legacy fallback) so
    the router is the only arm that can fire."""
    f = SimpleNamespace(
        args=SimpleNamespace(deep_bias=0.0),
        rng=np.random.default_rng(0),
        archive=SimpleNamespace(cells={c.key: c for c in cells}),
        max_area=max((c.key[-5] for c in cells), default=0), max_sect=0,
        sel_mode="legacy", frontier_throttle=0,
        door_weight=0.0, _doors=frozenset(), _key_ids={},
        gate_mode="off", gate_weight=1.0,
        ortho_mode="off", ortho_pin_secs=0.0, ortho_bias=0.0,
        ortho_band=1, ortho_weight=4.0, _pin_time=0.0,
        _ortho_pool=[], _ortho_ids=set(), _ortho_ext={},
        _ortho_deep_yband=None, _ortho_selections=0,
        _ortho_cols_improved=0, _gx_phantoms=set(),
        _sel_cells=None, _sel_n=0, _sel_area=None,
        room_bias=1.0, room_artic_weight=2.0, room_exit_weight=1.0,
        room_recent_k=4, room_fp={"max_rooms": 64}, room_index=index,
        transition_near=24, _room_psig_off=PSIG_OFF,
        _room_pools={}, _room_sides={}, _room_bounds={}, _room_U={},
        _room_V={}, _room_out_dirs={}, _room_degree={},
        _room_artic=set(), _room_aliased=set(),
        _room_router_picks=0, _route_pick=None)
    for k, v in over.items():
        setattr(f, k, v)
    f._articulation_points = Solver._articulation_points
    for name in ("_refresh_sel_cache", "_ortho_armed", "select"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def _prime(f, cells):
    """Bypass _refresh_sel_cache: hand-built caches stay exactly as the
    test set them (the refresh would rebuild pools from the index)."""
    f._sel_cells = list(cells)
    f._sel_n = len(cells)
    f._sel_area = f.max_area
    f._sel_maxscore = 1.0
    f._sel_deep, f._sel_band24, f._sel_lowl_band24 = [], [], []
    f._sel_topgx = 0


def _index(n_rooms: int) -> RoomIndex:
    idx = RoomIndex(cap=64, config_sha="t3")
    for o in range(n_rooms):
        assert idx.intern(1000 + o) == o
    return idx


# ---------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------


def test_derive_direction_macros_maps_every_side_to_its_dpad_holds():
    dm = derive_direction_macros(ACTS, steps=15)
    assert dm["E"] == [(1, 15), (2, 15)]
    assert dm["W"] == [(3, 15)]
    assert dm["N"] == [(4, 15)]
    assert dm["S"] == [(5, 15)]


def test_derive_direction_macros_leaves_an_inexpressible_side_empty():
    dm = derive_direction_macros([["noop"], ["right"]], steps=20)
    assert dm["E"] == [(1, 20)]
    assert dm["W"] == [] and dm["N"] == [] and dm["S"] == []


def test_room_cell_ord_reads_the_psig_tail_little_endian():
    assert room_cell_ord(_rcell(0, 0, ord_=300).key, PSIG_OFF) == 300
    assert room_cell_ord(_rcell(0, 0, ord_=0).key, PSIG_OFF) == 0


def test_room_cell_ord_refuses_blank_short_and_unknown_tails():
    # No transit yet: psig () — the cell belongs to no room.
    assert room_cell_ord(_rcell(0, 0).key, PSIG_OFF) is None
    # ROOM_UNKNOWN threaded by a transit that fired before the worker's
    # fingerprint settled: unroutable, never a pool.
    assert room_cell_ord(_rcell(0, 0, ord_=ROOM_UNKNOWN).key,
                         PSIG_OFF) is None
    # A tail shorter than the room_sig arity, and a run with no
    # extraction at all (room_sig not on 0x804/0x805).
    key = (1, 0, 0, (7,), 0, ()) + (0, 3, 1, 0, 0)
    assert room_cell_ord(key, PSIG_OFF) is None
    assert room_cell_ord(_rcell(0, 0, ord_=3).key, None) is None


def _edge(idx, s, d, kind="pan", direction="E", times=1):
    for _ in range(times):
        idx.record_edge(s, d, kind, direction, 10)


def test_aliased_rooms_needs_two_established_dsts_on_one_exit():
    idx = _index(4)
    _edge(idx, 0, 1, times=3)
    _edge(idx, 0, 2, times=3)          # same (pan, E) exit, second dst
    assert aliased_rooms(idx.adj) == {0}


def test_aliased_rooms_ignores_thin_or_direction_split_fanout():
    idx = _index(4)
    _edge(idx, 0, 1, times=3)
    _edge(idx, 0, 2, times=2)          # below the 3-traversal floor
    _edge(idx, 1, 2, direction="E", times=3)
    _edge(idx, 1, 3, direction="W", times=3)   # different exits: fine
    assert aliased_rooms(idx.adj) == set()


def test_room_boundaries_split_sides_at_the_cell_extent():
    cells = ([_rcell(gx, 5, ord_=1) for gx in range(0, 11)]
             + [_rcell(5, yb, ord_=1) for yb in (2, 8)])
    sides, bbox = room_boundaries(cells, near=0)
    assert bbox == (0, 10, 2, 8)
    assert {c.key[-1] for c in sides["E"]} == {10}
    assert {c.key[-1] for c in sides["W"]} == {0}
    # NES y grows downward: N is the LOW band, S the high one.
    assert {c.key[-2] for c in sides["N"]} == {2}
    assert {c.key[-2] for c in sides["S"]} == {8}
    wide, _ = room_boundaries(cells, near=2)
    assert {c.key[-1] for c in wide["E"]} == {8, 9, 10}


def test_room_weight_composes_artic_exit_visits_and_the_aliased_quarter():
    assert room_weight(False, False, 0, 0.0, 2.0, 1.0) == 1.0
    assert room_weight(True, False, 0, 0.0, 2.0, 1.0) == 3.0
    assert room_weight(False, False, 2, 0.0, 2.0, 1.0) == 3.0
    assert room_weight(False, False, 0, 3.0, 2.0, 1.0) == 0.5
    assert room_weight(False, True, 0, 0.0, 2.0, 1.0) == 0.25
    assert room_weight(True, True, 2, 3.0, 2.0, 1.0) == \
        pytest.approx(0.25 * 5.0 / 2.0)


def test_room_frontier_admits_each_clause_and_only_those():
    degree = {o: 2 for o in range(10)}
    degree[3] = 1                        # leaf
    front = room_frontier(range(10), recent_k=2, degree=degree,
                          artic={5}, aliased=set(), u={2: 1})
    # recency o >= 9-2 => 7,8,9 | leaf 3 | artic 5 | U>0 => 2
    assert front == [2, 3, 5, 7, 8, 9]
    # An aliased room cannot enter through the U clause — but recency
    # still admits it (the down-weight, not exclusion, handles it).
    front = room_frontier(range(10), recent_k=0, degree=degree,
                          artic=set(), aliased={2, 9}, u={2: 1, 9: 4})
    assert front == [3, 9]
    assert room_frontier([], 4, {}, set(), set(), {}) == []


def test_route_near_side_bands_are_inclusive_and_direction_true():
    bbox = (0, 10, 2, 8)
    assert route_near_side("E", bbox, 10, 5, 0)
    assert route_near_side("E", bbox, 8, 5, 2)
    assert not route_near_side("E", bbox, 7, 5, 2)
    assert route_near_side("W", bbox, 0, 5, 0)
    assert not route_near_side("W", bbox, 3, 5, 2)
    assert route_near_side("S", bbox, 5, 8, 0)
    assert route_near_side("N", bbox, 5, 2, 0)
    assert not route_near_side("N", bbox, 5, 5, 2)
    assert not route_near_side(None, bbox, 10, 8, 24)


# ---------------------------------------------------------------------
# _refresh_sel_cache: the room cache rides the same single scan
# ---------------------------------------------------------------------


def _chain_index():
    """0 -pan E-> 1 -pan E-> 2, plus 1 -pan E-> 3: room 1 is both the
    chain's articulation point and (fan-out on one exit) aliased."""
    idx = _index(4)
    _edge(idx, 0, 1, times=3)
    _edge(idx, 1, 2, times=3)
    _edge(idx, 1, 3, times=3)
    return idx


def test_refresh_builds_room_pools_from_psig_tails():
    cells = ([_rcell(gx, 1, ord_=0, chosen=2) for gx in range(3)]
             + [_rcell(gx, 1, ord_=1) for gx in (5, 6, 7)]
             + [_rcell(10, 1, ord_=2)]
             + [_rcell(0, 1), _rcell(1, 1, ord_=ROOM_UNKNOWN),
                _rcell(2, 1, ord_=50)])   # no room / unknown / uninterned
    f = _room_solver(cells, _chain_index())
    f._refresh_sel_cache()
    assert set(f._room_pools) == {0, 1, 2}
    assert len(f._room_pools[0]) == 3 and len(f._room_pools[1]) == 3
    assert f._room_V == {0: 6.0, 1: 0.0, 2: 0.0}
    assert f._room_bounds[1] == (5, 7, 1, 1)
    assert f._room_degree == {0: 1, 1: 3, 2: 1, 3: 1}
    assert f._room_artic == {1}
    # The scan never touches the main pool (the ortho lesson).
    assert len(f._sel_cells) == len(cells)


def test_refresh_computes_u_from_boundary_sides_minus_out_edges():
    cells = ([_rcell(gx, 1, ord_=0) for gx in range(3)]
             + [_rcell(5, 1, ord_=2)])
    f = _room_solver(cells, _chain_index())
    f._refresh_sel_cache()
    # Every side of a populated bbox has boundary cells at near=24, so
    # U = 4 - |out dirs|: room 0 exits E only, room 2 exits nowhere.
    assert f._room_U == {0: 3, 2: 4}
    assert f._room_out_dirs.get(0) == {"E"}


def test_refresh_marks_aliasing_in_the_index_monotonically():
    idx = _chain_index()
    f = _room_solver([_rcell(0, 1, ord_=0)], idx)
    f._refresh_sel_cache()
    assert f._room_aliased == {1}
    assert idx.meta[1]["aliased"] is True
    # Monotone: a later refresh over a quieter graph never clears it.
    f2 = _room_solver([_rcell(0, 1, ord_=0)], idx)
    f2._refresh_sel_cache()
    assert idx.meta[1]["aliased"] is True and f2._room_aliased == {1}


def test_refresh_skips_the_room_cache_when_the_arm_is_off():
    cells = [_rcell(gx, 1, ord_=0) for gx in range(3)]
    for over in ({"room_bias": 0.0}, {"room_index": None}):
        f = _room_solver(cells, _chain_index(), **over)
        if "room_index" in over:
            f.room_index = None
        f._refresh_sel_cache()
        assert f._room_pools == {} and f._room_artic == set()


# ---------------------------------------------------------------------
# the router arm in select()
# ---------------------------------------------------------------------


def test_router_arm_samples_only_frontier_rooms():
    # Room 0: old, degree 2, no artic, U 0, unaliased => NOT frontier.
    # Room 9: recent (within K of the newest ordinal) => frontier.
    old = [_rcell(gx, 1, ord_=0) for gx in range(4)]
    new = [_rcell(gx, 1, ord_=9) for gx in (8, 9)]
    f = _room_solver(old + new)
    _prime(f, old + new)
    f._room_pools = {0: old, 9: new}
    f._room_degree = {0: 2, 9: 1}
    for _ in range(50):
        pick = f.select()
        assert pick in new
    assert f._room_router_picks == 50
    assert f._route_pick[0] == 9


def test_router_arm_prefers_articulation_rooms_by_the_stated_weight():
    a = [_rcell(1, 1, ord_=8)]
    b = [_rcell(2, 1, ord_=9)]
    f = _room_solver(a + b)
    _prime(f, a + b)
    f._room_pools = {8: a, 9: b}
    f._room_artic = {8}
    hits = {8: 0, 9: 0}
    for _ in range(400):
        a[0].times_chosen = b[0].times_chosen = 0
        pick = f.select()
        hits[8 if pick is a[0] else 9] += 1
    # w(8) = 1 + artic_w = 3, w(9) = 1: expect ~300/100.
    assert 250 < hits[8] < 350, hits


def test_router_arm_pays_the_unexplored_exit_term():
    a = [_rcell(1, 1, ord_=8)]
    b = [_rcell(2, 1, ord_=9)]
    f = _room_solver(a + b)
    _prime(f, a + b)
    f._room_pools = {8: a, 9: b}
    f._room_U = {8: 2}
    hits = {8: 0, 9: 0}
    for _ in range(400):
        a[0].times_chosen = b[0].times_chosen = 0
        hits[8 if f.select() is a[0] else 9] += 1
    # w(8) = 1 + exit_w*2 = 3, w(9) = 1.
    assert 250 < hits[8] < 350, hits


def test_aliased_rooms_are_quartered_in_the_router():
    a = [_rcell(1, 1, ord_=8)]
    b = [_rcell(2, 1, ord_=9)]
    f = _room_solver(a + b)
    _prime(f, a + b)
    f._room_pools = {8: a, 9: b}
    f._room_aliased = {8}
    hits = {8: 0, 9: 0}
    for _ in range(500):
        a[0].times_chosen = b[0].times_chosen = 0
        hits[8 if f.select() is a[0] else 9] += 1
    # w(8) = 0.25 vs w(9) = 1: expect ~20% of picks.
    assert 50 < hits[8] < 150, hits


def test_router_pick_stashes_a_route_dir_from_an_open_side_only():
    cells = [_rcell(gx, 1, ord_=9) for gx in range(4)]
    f = _room_solver(cells)
    _prime(f, cells)
    f._room_pools = {9: cells}
    f._room_sides = {9: {"E": [cells[-1]]}}
    f._room_bounds = {9: (0, 3, 1, 1)}
    f._room_out_dirs = {}
    f.select()
    assert f._route_pick == (9, "E", (0, 3, 1, 1))
    # The same side already has an out-edge: routed, but no dir.
    f._room_out_dirs = {9: {"E"}}
    f.select()
    assert f._route_pick == (9, None, (0, 3, 1, 1))


def test_router_boundary_cells_are_biased_at_p_040():
    cells = [_rcell(gx, 1, ord_=9) for gx in range(11)]
    f = _room_solver(cells)
    _prime(f, cells)
    f._room_pools = {9: cells}
    f._room_sides = {9: {"E": [cells[-1]]}}
    f._room_bounds = {9: (0, 10, 1, 1)}
    n = 400
    hits = 0
    for _ in range(n):
        for c in cells:
            c.times_chosen = 0
        hits += f.select() is cells[-1]
    # P(boundary cell) = 0.40 + 0.60/11 ~ 0.455; a uniform arm would
    # sit at ~0.09 — the band separates the two by a wide margin.
    assert 130 < hits < 240, hits


def test_router_falls_through_when_the_room_is_all_wall():
    walls = [_rcell(gx, 1, ord_=9, barren=5) for gx in range(3)]
    other = [_rcell(0, 1)]
    f = _room_solver(walls + other, frontier_throttle=3)
    _prime(f, walls + other)
    f._room_pools = {9: walls}
    pick = f.select()
    assert pick is not None            # legacy arm still answers
    assert f._room_router_picks == 0
    assert f._route_pick is None


# ---------------------------------------------------------------------
# arm-off byte-identity (the done-when headline)
# ---------------------------------------------------------------------


class _DrawTally:
    """Numpy Generator proxy tallying draws by method name; a draw kind
    the off path should never make fails loudly instead of slipping
    through a permissive __getattr__ (the ortho suite's guard)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = {"random": 0, "integers": 0}

    def random(self, *a, **k):
        self.calls["random"] += 1
        return self._rng.random(*a, **k)

    def integers(self, *a, **k):
        self.calls["integers"] += 1
        return self._rng.integers(*a, **k)


def test_the_disarmed_router_draws_no_randomness_at_all():
    # Legacy budget per select is exactly 1 random (deep gate) + 1
    # integers (uniform pick); any extra draw from the room gate shifts
    # the ledger by an integer, not a flake. Pools deliberately
    # populated so ONLY the bias gate holds the arm down.
    n = 64
    cells = [_rcell(gx, yb, ord_=1) for gx in range(4) for yb in range(3)]
    tally = _DrawTally(0)
    f = _room_solver(cells, room_bias=0.0, rng=tally)
    _prime(f, cells)
    f._room_pools = {1: list(cells)}
    for _ in range(n):
        assert f.select() is not None
    assert f._room_router_picks == 0
    assert tally.calls == {"random": n, "integers": n}


def test_an_empty_pool_set_short_circuits_the_armed_draw_away():
    # Armed (bias 1.0) but nothing fingerprinted yet: the gate must
    # not spend a draw, or an on-arm run's stream would depend on
    # whether the cache happened to be populated that tick — the exact
    # ortho gate-ordering lesson.
    n = 64
    cells = [_rcell(gx, 1) for gx in range(6)]
    tally = _DrawTally(0)
    f = _room_solver(cells, rng=tally)
    _prime(f, cells)
    assert f._room_pools == {}
    for _ in range(n):
        assert f.select() is not None
    assert tally.calls == {"random": n, "integers": n}


def test_bias_zero_walks_the_exact_pre_router_pick_stream():
    # Stream parity: a fixture carrying every room attribute at bias 0
    # must pick the same cells in the same order as one built the way
    # the pre-roomgraph suite builds them (no room attrs at all) —
    # the unit-level shape of the flags-off byte-identity requirement.
    cells_a = [_rcell(gx, yb, ord_=2) for gx in range(5) for yb in range(2)]
    cells_b = [_rcell(gx, yb, ord_=2) for gx in range(5) for yb in range(2)]
    on = _room_solver(cells_a, room_bias=0.0)
    on._room_pools = {2: list(cells_a)}
    bare = _room_solver(cells_b)
    for name in ("room_bias", "room_artic_weight", "room_exit_weight",
                 "room_recent_k", "room_fp", "room_index",
                 "_room_pools", "_room_sides", "_room_bounds", "_room_U",
                 "_room_V", "_room_out_dirs", "_room_degree",
                 "_room_artic", "_room_aliased", "_room_router_picks",
                 "_route_pick", "_room_psig_off"):
        delattr(bare, name)
    _prime(on, cells_a)
    _prime(bare, cells_b)
    seq_on = [cells_a.index(on.select()) for _ in range(200)]
    bare.rng = np.random.default_rng(0)
    on_keys = seq_on
    seq_bare = [cells_b.index(bare.select()) for _ in range(200)]
    assert on_keys == seq_bare


# ---------------------------------------------------------------------
# _assign: the route handoff
# ---------------------------------------------------------------------


def _assign_fixture(cell, stash=None):
    f = SimpleNamespace(
        archive=SimpleNamespace(cells={cell.key: cell}),
        traces={cell.key: ("root", b"", 0, (), 1, cell.key[3], 0)},
        pool=SimpleNamespace(load_worker_state=lambda *a: None),
        args=SimpleNamespace(burst=64, root_state="unused"),
        rng=np.random.default_rng(0), weights=np.array([1.0]),
        _ortho_ids=set(), gate_mode="off", _gate_inject=[],
        frontier_throttle=0)
    def _select():
        if stash is not None:
            f._route_pick = stash
        return cell
    f.select = _select
    f._assign = MethodType(Solver._assign, f)
    return f


def test_assign_attaches_route_tags_only_on_router_picks():
    cell = _rcell(5, 3, ord_=9)
    f = _assign_fixture(cell, stash=(9, "E", (0, 5, 0, 3)))
    c = f._assign(0)
    assert c["route_room"] == 9
    assert c["route_dir"] == "E" and c["route_bbox"] == (0, 5, 0, 3)
    assert f._route_pick is None       # consumed, never reattached
    c2 = f._assign(0)                  # stub re-stashes every select
    assert c2["route_room"] == 9


def test_assign_without_a_router_pick_leaves_the_ctx_byte_identical():
    cell = _rcell(5, 3, ord_=9)
    tagged = f = _assign_fixture(cell)
    c = f._assign(0)
    assert "route_room" not in c and "route_dir" not in c \
        and "route_bbox" not in c
    assert tagged._route_pick is None


def test_assign_routed_without_an_open_side_tags_the_room_only():
    cell = _rcell(5, 3, ord_=9)
    f = _assign_fixture(cell, stash=(9, None, (0, 5, 0, 3)))
    c = f._assign(0)
    assert c["route_room"] == 9
    assert "route_dir" not in c and "route_bbox" not in c


# ---------------------------------------------------------------------
# status-line telemetry (§3 row 12)
# ---------------------------------------------------------------------


def _line_fixture(tmp_path, **over):
    f = SimpleNamespace(
        archive=[1], _stall={"last_cells": 0, "last_t": 0.0,
                             "flat_windows": 0},
        max_area=0, max_gx_in_area={}, max_sect=0, n_solutions=0,
        best_sol_len=0, steps_done=1, door_weight=0,
        transition_macros=[], ortho_mode="off", out=tmp_path)
    for k, v in over.items():
        setattr(f, k, v)
    return f


def test_progress_line_carries_the_room_graph_block(tmp_path):
    idx = _index(3)
    idx.record_edge(0, 1, "pan", "E", 10)
    idx.record_edge(1, 2, "fade", None, 30)
    idx.record_warp(0, 2, 2, (0, 0))
    idx.meta[2]["aliased"] = True
    f = _line_fixture(
        tmp_path, room_fp={"settle": 3}, room_index=idx,
        _room_settle_rejects=4, _room_artic={1},
        _room_router_picks=7, _room_route_injections=2,
        _room_edges_committed=5, _room_edges_dropped=1,
        _room_restore_transits=0)
    Solver.progress_line(f, 5.0)
    line = json.loads((tmp_path / "progress.jsonl").read_text()
                      .splitlines()[-1])
    assert line["rooms"] == 3
    assert line["edges_pan"] == 1 and line["edges_fade"] == 1
    assert line["warps_vetoed"] == 1
    assert line["aliased"] == 1
    assert line["artic"] == 1
    assert line["settle_rejects"] == 4
    assert line["router_picks"] == 7
    assert line["route_macros_injected"] == 2
    assert line["room_edges_committed"] == 5
    assert line["room_edges_dropped"] == 1
    assert line["room_restore_transits"] == 0


def test_progress_line_without_room_fp_prints_no_room_keys(tmp_path):
    Solver.progress_line(_line_fixture(tmp_path), 5.0)
    line = json.loads((tmp_path / "progress.jsonl").read_text()
                      .splitlines()[-1])
    for k in ("rooms", "edges_pan", "edges_fade", "warps_vetoed",
              "aliased", "artic", "settle_rejects", "router_picks",
              "route_macros_injected"):
        assert k not in line
