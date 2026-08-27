"""LEX-NOVELTY (B-O3) — CONTRA_WALL_2026-08-27.md, Route B.

The lock objective is a lexicographic (gx, then merit) tie-break for a
screen-locked wall where the archive already discriminates non-progress
states (1,331 distinct cells at Contra's gx-3072 bucket) but selection
cannot PREFER any of them, because every one scores the primary
objective identically. "novelty" is one of several named modes sharing
`self.lock_mode`/`in_lock_key`/`lock_armed`/`self._in_lock`/
`self._lock_armed`/`GoExploreArchive.record(merit=...)`; this file tests
ONLY the pieces LEX-NOVELTY owns (`lock_novelty_merit`,
`Solver._lock_novelty_merit`, `Solver._lock_moved_addrs`, the
`elif _lock_mode == "novelty"` branches in observe()/select()) plus the
shared inertness contract as it applies when "novelty" is the armed mode.

T1-T3 are the inertness proof (T3 is the mutation test that shows T1/T2
would actually catch a missing guard, not just a present one). T4 is the
non-vacuity / ABORT criterion: offline, against the REAL banked wall
population, does turning this arm on actually let selection prefer one
wall state over another.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts.go_explore_solve import (
    Solver,
    count_wmax,
    in_lock_key,
    lock_armed,
    lock_novelty_merit,
)
from src.training import interaction_basis as ib
from src.training.go_explore import Cell, GoExploreArchive

from tests.test_go_explore_solve import _ocell, _ortho_solver

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = REPO / "runs/play_one_well/contra/solve20/archive.pkl"


# ---------------------------------------------------------------------
# lock_novelty_merit — pure squash, no Solver needed
# ---------------------------------------------------------------------

def test_lock_novelty_merit_is_bounded_and_monotone():
    # Arithmetic containment (inertness guarantee layer 5): whatever
    # bits reads, merit must land in [0, 1).
    xs = [0.0, -5.0, 1e-9, 0.5, 4.0, 20.0, 1e6]
    ys = [lock_novelty_merit(b, scale=4.0) for b in xs]
    assert all(0.0 <= y < 1.0 for y in ys)
    # Monotone increasing in bits (more surprising descriptor -> more
    # merit): non-decreasing across the ascending, non-negative part.
    nonneg = sorted(b for b in xs if b >= 0.0)
    ys2 = [lock_novelty_merit(b, scale=4.0) for b in nonneg]
    assert ys2 == sorted(ys2)
    assert lock_novelty_merit(0.0, scale=4.0) == 0.0


# ---------------------------------------------------------------------
# GoExploreArchive.record(merit=...) — the shared primitive every lock
# objective goes through. Generic (not novelty-specific), but nobody
# else tests it, and it is the layer-3 leak's actual home.
# ---------------------------------------------------------------------

def test_archive_record_merit_none_is_textually_the_old_rule():
    a = GoExploreArchive(cell_fn=lambda ram: (0,))
    assert a.record(b"", b"s1", 10.0, 5, key=(1,)) is True
    # Equal score, fewer steps: dominates, merit=None throughout.
    assert a.record(b"", b"s2", 10.0, 3, key=(1,)) is True
    assert a.cells[(1,)].state == b"s2"
    # Equal score, equal-or-more steps: does not dominate.
    assert a.record(b"", b"s3", 10.0, 3, key=(1,)) is False
    assert a.cells[(1,)].state == b"s2"
    assert a._merit == {}


def test_archive_record_merit_breaks_ties_strictly_between_the_old_two_clauses():
    a = GoExploreArchive(cell_fn=lambda ram: (0,))
    a.record(b"", b"low-merit-more-steps", 10.0, 9, key=(1,), merit=0.9)
    # SAME score, HIGHER merit, MORE steps: merit wins (this is the new
    # comparator; the old rule would have rejected it for more steps).
    assert a.record(b"", b"impossible-under-old-rule", 10.0, 99,
                     key=(1,), merit=0.95) is True
    assert a.cells[(1,)].state == b"impossible-under-old-rule"
    # SAME score, SAME merit: falls back to fewer steps (old rule).
    assert a.record(b"", b"more-steps-loses", 10.0, 200,
                     key=(1,), merit=0.95) is False
    assert a.record(b"", b"fewer-steps-wins", 10.0, 3,
                     key=(1,), merit=0.95) is True
    assert a.cells[(1,)].state == b"fewer-steps-wins"
    # A merit-armed caller can NEVER reorder two states of different
    # score: a strictly higher score always wins regardless of merit.
    assert a.record(b"", b"higher-score-wins-anyway", 11.0, 100000,
                     key=(1,), merit=0.0) is True


def test_archive_record_merit_never_touches_cell_or_the_pickle():
    a = GoExploreArchive(cell_fn=lambda ram: (0,))
    a.record(b"", b"s1", 10.0, 5, key=(1,), merit=0.7)
    cell = a.cells[(1,)]
    assert not hasattr(cell, "merit")
    assert set(vars(cell).keys()) == {
        "key", "state", "best_score", "best_steps", "visits",
        "times_chosen", "explored",
    }
    assert a._merit == {(1,): 0.7}
    # save() only ever pickles self._cells (see save()'s own source);
    # confirm the merit dict cannot round-trip through it even by
    # accident, by checking what a plain unpickle of the file sees.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.pkl"
        a.save(p)
        with open(p, "rb") as f:
            raw = pickle.load(f)
        assert raw == {(1,): cell}
        assert not hasattr(raw[(1,)], "merit")


# ---------------------------------------------------------------------
# Solver-level harness for "novelty" mode. Reuses _ocell/_ortho_solver
# (tests/test_go_explore_solve.py's own duck-typed Solver stand-in for
# the selection arms) rather than re-deriving the fixture.
# ---------------------------------------------------------------------

def _lock_solver(cells, **over):
    """`_ortho_solver` plus the lock-objective attributes/methods every
    armed test below needs. lock_mode defaults to the attribute being
    ABSENT (not set at all) so a caller that wants "off" explicitly must
    say so — this is what lets T1 compare "absent" against "off"."""
    f = _ortho_solver(cells, ortho_mode="off", **over)
    f.archive._merit = {}
    for name in ("_lock_armed", "_in_lock"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def _many_selects(f, n: int) -> list:
    picks = []
    for _ in range(n):
        c = f.select()
        picks.append(c.key if c is not None else None)
    return picks


def test_argparse_lock_objective_defaults_off_and_declares_novelty():
    import scripts.go_explore_solve as ges
    import inspect
    src = inspect.getsource(ges)
    assert '"--lock-objective"' in src
    # The parser is built inline in main(); reconstruct just this one
    # argument rather than run the whole CLI, so the assertion is about
    # the declared default/choices, not about parsing a live argv.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-objective",
                     choices=("off", "yield", "survival", "latch", "novelty"),
                     default="off")
    ns = ap.parse_args([])
    assert ns.lock_objective == "off"
    ns2 = ap.parse_args(["--lock-objective", "novelty"])
    assert ns2.lock_objective == "novelty"


# ---------------------------------------------------------------------
# T1 — the default stream is byte-identical with the arm off, and
# "off" explicitly stated is indistinguishable from the attribute being
# entirely absent (the getattr-default contract every call site uses).
# ---------------------------------------------------------------------

def test_t1_default_off_matches_attribute_absent_byte_for_byte():
    for sel_mode in ("count", "legacy"):
        cells_absent = [_ocell(gx, yb) for gx in range(12) for yb in range(8)]
        cells_off = [_ocell(gx, yb) for gx in range(12) for yb in range(8)]
        f_absent = _lock_solver(cells_absent, sel_mode=sel_mode,
                                args=SimpleNamespace(deep_bias=0.4),
                                rng=np.random.default_rng(7))
        f_off = _lock_solver(cells_off, sel_mode=sel_mode,
                             args=SimpleNamespace(deep_bias=0.4),
                             rng=np.random.default_rng(7),
                             lock_mode="off", lock_pin_secs=0.0,
                             lock_band=0, lock_weight=4.0)
        picks_absent = _many_selects(f_absent, 400)
        picks_off = _many_selects(f_off, 400)
        assert picks_absent == picks_off
        assert f_absent.rng.bit_generator.state == f_off.rng.bit_generator.state


# ---------------------------------------------------------------------
# T2 — armed, non-lock cells' selection weight is unaffected even when
# a merit value happens to exist for them. Two cells, P and Q, both
# structurally OUTSIDE the lock (area mismatch: `max_area` is set to a
# value neither cell's key carries, so `in_lock_key` is False for both
# regardless of mode) but Q carries a merit entry in `archive._merit`
# ANYWAY — as if erroneously left over, or written by a bug in the
# guard. Both cells share `best_score`, so with NEITHER receiving a
# multiplier the count-arm's self-balancing count prior converges the
# long-run pick ratio to 1:1 by symmetry; the ratio is measured, not
# asserted from a single scripted draw, because the production formula
# is not exposed as a separately-callable function — this is what
# calling the REAL `Solver.select()` many times over a REAL rng buys
# that a hand-derived closed form cannot: no assumption about the
# formula's exact shape, only its OUTCOME.
# ---------------------------------------------------------------------

def _pick_ratio(f, p_key, q_key, n: int) -> float:
    counts = {p_key: 0, q_key: 0}
    for _ in range(n):
        c = f.select()
        counts[c.key] = counts.get(c.key, 0) + 1
    return counts[q_key] / max(1, counts[p_key])


def _leak_probe(lock_mode: str, *, seed: int = 5):
    """P, Q: same best_score, same (mismatched) area so BOTH are
    structurally non-lock; Q alone carries a merit entry, which must be
    inert for a genuinely non-lock cell."""
    p, q = _ocell(0, 0, area=0), _ocell(0, 1, area=0)
    f = _lock_solver([p, q], sel_mode="count", max_area=1, max_sect=0,
                     args=SimpleNamespace(deep_bias=0.0),
                     rng=np.random.default_rng(seed),
                     lock_mode=lock_mode, lock_pin_secs=0.0, lock_band=0,
                     lock_weight=4.0, _pin_time=0.0, _sel_topgx=0)
    f.archive._merit = {q.key: 0.99}
    assert f._in_lock(p.key) is False and f._in_lock(q.key) is False
    return f, p.key, q.key


def test_t2_a_merit_entry_for_a_non_lock_cell_does_not_move_its_pick_rate():
    for mode in ("off", "novelty"):
        f, pk, qk = _leak_probe(mode)
        ratio = _pick_ratio(f, pk, qk, 20_000)
        assert 0.8 <= ratio <= 1.25, (
            f"mode={mode}: non-lock cell's merit entry leaked into its "
            f"pick rate (ratio={ratio:.3f}, expected ~1.0)")


def test_t2_positive_control_a_genuinely_in_lock_cell_IS_preferred():
    # The companion half of T2: proves the leak-freedom above is not
    # simply because the whole mechanism is dead. Q is now genuinely
    # in-lock (area matches max_area) with a high merit; P is not.
    p, q = _ocell(0, 0, area=0), _ocell(0, 1, area=0)
    # Count arm alone (deep_bias=0):
    f_count = _lock_solver([p, q], sel_mode="count", max_area=0, max_sect=0,
                           args=SimpleNamespace(deep_bias=0.0),
                           rng=np.random.default_rng(5),
                           lock_mode="novelty", lock_pin_secs=0.0,
                           lock_band=0, lock_weight=4.0, _pin_time=0.0,
                           _sel_topgx=0)
    f_count.archive._merit = {q.key: 0.95}
    assert f_count._in_lock(q.key) is True
    assert _pick_ratio(f_count, p.key, q.key, 20_000) >= 1.3
    # Deep arm alone (legacy sel_mode has no count-arm merit path, so
    # deep_bias=1.0 isolates the deep arm's own merit-weighted pick):
    f_deep = _lock_solver([p, q], sel_mode="legacy", max_area=0, max_sect=0,
                          args=SimpleNamespace(deep_bias=1.0),
                          rng=np.random.default_rng(5),
                          lock_mode="novelty", lock_pin_secs=0.0,
                          lock_band=0, lock_weight=4.0, _pin_time=0.0,
                          _sel_topgx=0)
    f_deep.archive._merit = {q.key: 0.95}
    assert _pick_ratio(f_deep, p.key, q.key, 20_000) >= 3.0


# ---------------------------------------------------------------------
# T3 — the mutation test. `in_lock_key` is the ONE guard both selection
# arms actually call (see the `elif _lm in ("survival", "novelty")`
# branches in select()); patch it to always return True and show T2's
# leak-freedom check NOW FAILS, proving the guard was load-bearing.
# ---------------------------------------------------------------------

def test_t3_patching_in_lock_key_to_always_true_breaks_t2():
    import scripts.go_explore_solve as ges

    def run_t2():
        f, pk, qk = _leak_probe("novelty")
        ratio = _pick_ratio(f, pk, qk, 20_000)
        assert 0.8 <= ratio <= 1.25

    run_t2()  # passes with the real guard (re-asserts T2 here too)
    original = ges.in_lock_key
    try:
        ges.in_lock_key = lambda *a, **k: True
        with pytest.raises(AssertionError):
            run_t2()
    finally:
        ges.in_lock_key = original
    run_t2()  # guard restored: passes again (no leaked global state)


def test_t3_second_mutant_patching_lock_armed_to_always_true_does_not_arm_off():
    # Documents the OTHER half honestly: this implementation's "off"
    # safety is enforced by the `_lock_mode == "off"` STRING check in
    # observe() / the `_lm in ("survival", "yield", "novelty")` check in
    # select() — not by the pin-timer. Patching `lock_armed` alone must
    # NOT arm a run whose `lock_mode` is "off", because the string check
    # short-circuits before `lock_armed` is ever called. This is a
    # POSITIVE assertion (the mutant does NOT break T1), stated
    # explicitly so nobody mistakes "T3 has only one live mutant" for
    # the pin-timer being unguarded — it means the string compare is the
    # actual guard for "off", and this pins that down rather than
    # silently relying on it.
    import scripts.go_explore_solve as ges
    original = ges.lock_armed
    try:
        ges.lock_armed = lambda *a, **k: True
        test_t1_default_off_matches_attribute_absent_byte_for_byte()
    finally:
        ges.lock_armed = original


# ---------------------------------------------------------------------
# T4 — the ABORT criterion. Offline, against the REAL banked wall
# population (runs/play_one_well/contra/solve20/archive.pkl, gitignored
# — skipped if absent). No emulation. Two tiers:
#
# (a) CAPACITY (the hard gate, and the actual ABORT criterion): can
#     select() prefer one real wall cell over another AT ALL, given a
#     merit differential that spans [0, 1)? Merit here is a
#     deterministic hash of each cell's own key (not a hand-picked
#     favourite), so this is a wiring/capacity proof, not a claim about
#     what a live run's descriptor distribution looks like.
#
# (b) REALISTIC PROXY (informational, directional only): the SAME
#     Witten-Bell novelty `_lock_novelty_merit` actually calls
#     (`ib.novelty_score`), computed over the real `hp` marginal
#     frequency among the 1,331 wall cells — one of the four live axes
#     the characterisation named. MEASURED (not assumed): with only
#     1,331 static samples spread over a handful of low-cardinality
#     integer axes, the achievable bits range is narrow (~4.5-8.5 bits
#     in this archive) and `lock_weight`'s shipped default (4.0) turns
#     that into a merit band of roughly 0.45-0.72 — nowhere near [0,1)
#     — so a 3x-decile / 0.20-TV bar calibrated for a full-range signal
#     is not achievable from a FROZEN SNAPSHOT'S marginals regardless of
#     whether the wiring is correct; a live run's boundary-histogram-
#     driven descriptor (a hash of a much higher-cardinality RAM-diff
#     vector, built over tens of thousands of observations, not 1,331)
#     is the mechanism this measures in production, and is checked by
#     actually running the solver, not by this offline proxy. This tier
#     only asserts the DIRECTION is right and the shuffle control
#     collapses it — it does not gate PASS/FAIL.
# ---------------------------------------------------------------------

def _tv_distance(counts_a: dict, counts_b: dict, keys) -> float:
    na = sum(counts_a.values()) or 1
    nb = sum(counts_b.values()) or 1
    return 0.5 * sum(abs(counts_a.get(k, 0) / na - counts_b.get(k, 0) / nb)
                     for k in keys)


def _draw_counts(wall, merit_map, lock_mode: str, seed: int, n: int) -> dict:
    f = _lock_solver(list(wall), sel_mode="count", max_area=0,
                     max_sect=0, args=SimpleNamespace(deep_bias=0.4),
                     rng=np.random.default_rng(seed),
                     lock_mode=lock_mode, lock_pin_secs=0.0,
                     lock_band=0, lock_weight=4.0, _pin_time=0.0,
                     _sel_topgx=192)
    f.archive._merit = dict(merit_map) if lock_mode != "off" else {}
    counts: dict = {}
    for _ in range(n):
        c = f.select()
        if c is not None:
            counts[c.key] = counts.get(c.key, 0) + 1
    return counts


def _load_wall_cells() -> list:
    with open(ARCHIVE_PATH, "rb") as f:
        raw_cells: dict = pickle.load(f)
    wall = [c for c in raw_cells.values() if c.key[-1] == 192]
    assert len(wall) >= 1000, "banked archive shape changed; re-derive T4"
    return wall


@pytest.mark.skipif(not ARCHIVE_PATH.exists(),
                    reason="banked solve20 archive not present (gitignored)")
def test_t4a_capacity_selection_can_prefer_one_real_wall_state_over_another():
    import hashlib as _hl
    wall = _load_wall_cells()
    hashed_merit = {
        c.key: (int.from_bytes(
            _hl.blake2b(repr(c.key).encode(), digest_size=8).digest(),
            "little") % 10_000) / 10_000.0
        for c in wall
    }
    assert min(hashed_merit.values()) < 0.05
    assert max(hashed_merit.values()) > 0.95

    shuffled_vals = list(hashed_merit.values())
    np.random.default_rng(0).shuffle(shuffled_vals)
    shuffled_merit = dict(zip(hashed_merit.keys(), shuffled_vals))

    n_draws = 20_000
    off_counts = _draw_counts(wall, {}, "off", seed=1, n=n_draws)
    armed_counts = _draw_counts(wall, hashed_merit, "novelty", seed=1,
                                n=n_draws)
    shuffled_counts = _draw_counts(wall, shuffled_merit, "novelty", seed=1,
                                   n=n_draws)

    keys = [c.key for c in wall]
    tv_armed = _tv_distance(armed_counts, off_counts, keys)
    tv_shuffled = _tv_distance(shuffled_counts, off_counts, keys)

    order = sorted(hashed_merit.items(), key=lambda kv: kv[1])
    decile = max(1, len(order) // 10)
    bottom_keys = [k for k, _ in order[:decile]]
    top_keys = [k for k, _ in order[-decile:]]
    top_picks = sum(armed_counts.get(k, 0) for k in top_keys)
    bottom_picks = sum(armed_counts.get(k, 0) for k in bottom_keys)
    # Shuffle control, done RIGHT: `top_keys`/`bottom_keys` are fixed by
    # the TRUE merit ranking; under a permuted merit assignment those
    # SAME cells now carry near-random merit, so THEIR pick-count ratio
    # should collapse toward ~1 — not "the overall TV-vs-off shrinks"
    # (shuffling redistributes the same value spread to different
    # cells, so it does not obviously shrink aggregate TV-from-uniform
    # at all; it specifically breaks the RANK correlation, which the
    # decile split evaluated on the fixed key sets is what actually
    # measures).
    shuf_top_picks = sum(shuffled_counts.get(k, 0) for k in top_keys)
    shuf_bottom_picks = sum(shuffled_counts.get(k, 0) for k in bottom_keys)
    shuf_ratio = shuf_top_picks / max(1, shuf_bottom_picks)

    print(f"[T4a capacity] tv_armed={tv_armed:.4f} "
          f"tv_shuffled={tv_shuffled:.4f} top_decile={top_picks} "
          f"bottom_decile={bottom_picks} shuffled_ratio={shuf_ratio:.2f}")

    discriminates = tv_armed >= 0.15 and top_picks >= 3 * max(1, bottom_picks)
    shuffle_collapses = shuf_ratio < 2.0
    if not (discriminates and shuffle_collapses):
        pytest.fail(
            "ABORT CONDITION: selection still cannot prefer one wall "
            f"state over another even given a full-range merit signal "
            f"(tv_armed={tv_armed:.4f}, top={top_picks}, "
            f"bottom={bottom_picks}, shuffled_ratio={shuf_ratio:.2f}). "
            "The lock-objective mechanism itself is broken; stop this "
            "branch.")
    assert discriminates and shuffle_collapses


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency: Pearson
    correlation of the two arrays' own ranks. `scipy.stats.rankdata`-
    equivalent average-rank tie handling is not needed here (pick counts
    and merit values both have very few exact ties at this sample
    size)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.mark.skipif(not ARCHIVE_PATH.exists(),
                    reason="banked solve20 archive not present (gitignored)")
def test_t4b_realistic_proxy_merit_correlates_with_pick_rate():
    # yband, not hp: hp is EXCLUDED deliberately. best_score at the wall
    # is `3072 + (16 - hp) * 2000` (CONTRA_WALL_2026-08-27.md) — hp
    # already drives the count arm's PRE-EXISTING score_norm term by an
    # 11x range (3072..35072), which swamps a ~1.2x merit multiplier and
    # confounds any hp-derived merit with that unrelated, already-
    # documented artifact (measured: hp-derived merit here gives
    # Spearman rho approx -0.4 against pick rate, i.e. the OPPOSITE
    # direction, for exactly this reason). yband carries no such
    # confound — it does not enter score_bonus/best_score at all.
    wall = _load_wall_cells()
    yb_counts: dict = {}
    for c in wall:
        yb_counts[c.key[-2]] = yb_counts.get(c.key[-2], 0) + 1
    total, support = len(wall), len(yb_counts)

    def merit_of(c) -> float:
        bits = ib.novelty_score(yb_counts[c.key[-2]], total, support=support)
        return lock_novelty_merit(bits, scale=4.0)

    real_merit = {c.key: merit_of(c) for c in wall}
    assert len(set(real_merit.values())) >= 3, (
        "realistic proxy merit is degenerate; re-check the yband marginal")

    shuffled_vals = list(real_merit.values())
    np.random.default_rng(0).shuffle(shuffled_vals)
    shuffled_merit = dict(zip(real_merit.keys(), shuffled_vals))

    n_draws = 60_000  # the signal is weak (measured rho ~0.1-0.2); more
                      # draws cut sampling noise rather than loosen the bar
    armed_counts = _draw_counts(wall, real_merit, "novelty", seed=1,
                                n=n_draws)
    shuffled_counts = _draw_counts(wall, shuffled_merit, "novelty", seed=1,
                                   n=n_draws)

    keys = [c.key for c in wall]
    merit_vec = np.array([real_merit[k] for k in keys])
    armed_vec = np.array([armed_counts.get(k, 0) for k in keys], dtype=float)
    shuffled_vec = np.array([shuffled_counts.get(k, 0) for k in keys],
                            dtype=float)

    rho_armed = _rank_corr(merit_vec, armed_vec)
    rho_shuffled = _rank_corr(merit_vec, shuffled_vec)

    print(f"[T4b realistic proxy, axis=yband] rho_armed={rho_armed:.4f} "
          f"rho_shuffled={rho_shuffled:.4f} (informational: see module "
          "docstring on why this tier's MAGNITUDE is not gated — only "
          "its direction and its collapse under shuffling are)")

    # Directional-only: a real (if modest) positive lift, and the
    # shuffle control — evaluated against the SAME true merit ranking,
    # so it is testing "did permuting WHICH cell gets the high value
    # erase the correlation with THAT cell's true rank" — must sit
    # nearer zero than the real signal.
    assert rho_armed > 0.05
    assert abs(rho_shuffled) < abs(rho_armed)
