"""Every `--lock-objective` name this CLI accepts must actually run.

The defect this file exists to catch, found on main during the landing of
`docs/research/CONTRA_LOCK2_2026-08-27.md`: `--lock-objective latch` was
a declared choice with **no dispatch branch anywhere in the solver**. It
parsed, it printed as the armed mode on every progress line
(`lock_mode: latch`, non-zero `lock_cells`), and it changed not one
draw — measured at 3,000 selections byte-identical to `off`. An operator
running it would have read a null result as "the objective did not help"
when in fact nothing ran. That is the seventh instance of the vacuity
failure this codebase has shipped -- the first six are logged in
MISTAKES.md, and this one landed inside the campaign auditing for them.

The guard is BEHAVIOURAL, not a grep. For each declared name the two
probes below ask what the solver actually does:

  P1 (selection) — an armed run's pick sequence differs from `off`'s.
     A name with no branch takes the legacy line, consumes the identical
     RNG, and lands on the identical cells, so it is caught here.
  P2 (observation) — an armed burst loop leaves a merit footprint that
     an `off` run does not: either `archive._merit` (the per-observation
     modes, B-O1 survival / B-O3 novelty, which record merit through
     `GoExploreArchive.record(merit=...)`) or `_lock_bursts` (the
     per-key mode, B-O4 yield, whose counters `_assign()` keeps). A name
     with no branch writes neither.

WHAT THIS REPORTS WHEN THE MECHANISM IS ABSENT is not asserted by
inspection — `test_the_roster_probe_reports_a_fabricated_name_inert`
runs both probes against a name that deliberately does not exist and
requires them to come back inert. If that test ever passes a fake name,
this whole file is decoration.
"""

from __future__ import annotations

import argparse
import pathlib
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import src.training.interaction_basis as ib
from scripts.go_explore_solve import Solver, lock_armed, lock_clocks

from tests.test_go_explore_solve import (
    _burst_cells,
    _burst_solver,
    _ocell,
    _ortho_solver,
    _parse_solver_argv,
    _run_bursts,
)

# Lock knobs at their shipped values except the mode, so arming is
# uniform across every probed name (pin clock already elapsed).
_LOCK_KW = dict(lock_pin_secs=0.0, lock_band=0, lock_weight=4.0,
                lock_survival_scale=64.0, lock_novelty_scale=4.0,
                lock_desc_buckets=1 << 20, _pin_time=-10_000.0)

# A name that is not, and must never become, a lock objective. It stands
# in for "a choice someone added to the tuple without wiring a branch".
FABRICATED = "no_such_objective"


def _declared_lock_objective(monkeypatch) -> dict:
    """The REAL argparse declaration, captured from the real `main()`.

    Deliberately not a reconstructed parser: a hand-copied tuple in a
    test cannot drift-detect the tuple in the shipped file, which is
    exactly how a dead choice survives.
    """
    seen: dict = {}
    real_add = argparse.ArgumentParser.add_argument

    def _spy(self, *names, **kw):
        if names and names[0] == "--lock-objective":
            seen["choices"] = tuple(kw.get("choices") or ())
            seen["default"] = kw.get("default")
        return real_add(self, *names, **kw)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", _spy)
    _parse_solver_argv(monkeypatch)
    assert seen, "--lock-objective is no longer declared in main()"
    return seen


TOPGX = 11          # matches _burst_cells()' gx range

# Methods the per-mode branches call on `self`. The shared duck-typed
# stand-ins bind only _assign/observe, so bind the rest FROM THE REAL
# CLASS — a reimplemented _in_lock would let this probe pass on a solver
# whose real one is broken.
_LOCK_METHODS = ("_in_lock", "_lock_armed", "_lock_merit_for",
                 "_lock_novelty_merit", "_lock_moved_addrs")


def _bind_lock_methods(f):
    for name in _LOCK_METHODS:
        if hasattr(Solver, name):
            setattr(f, name, MethodType(getattr(Solver, name), f))
    # The per-key counters LEX-YIELD's _assign() bookkeeping writes into;
    # the real Solver.__init__ creates them, the stand-ins predate them.
    for name in ("_lock_bursts", "_lock_yields"):
        if not hasattr(f, name):
            setattr(f, name, {})
    return f


def _select_stream(mode: str) -> list:
    """P1: 400 picks through the deep-frontier arm (ortho disabled so it
    cannot intercept), fresh cells per run because `times_chosen`
    mutates in place. The archive carries a real `_merit` map so the
    per-observation modes read a populated one rather than an empty
    default — a name is being probed for reaching its branch, and an
    empty merit map would still reach it, but a populated one also
    exercises the weighting arithmetic."""
    cells = [_ocell(gx, yb) for gx in range(TOPGX + 1) for yb in range(4)]
    merit = {c.key: 0.7 for c in cells if c.key[-1] == TOPGX}
    f = _ortho_solver(cells, rng=np.random.default_rng(7),
                      args=SimpleNamespace(deep_bias=1.0), ortho_mode="off",
                      lock_mode=mode, **_LOCK_KW,
                      archive=SimpleNamespace(
                          cells={c.key: c for c in cells}, _merit=merit))
    # LEX-YIELD reads per-key burst counters instead of archive._merit;
    # seed a discriminating pair so both plumbing shapes are exercised.
    _bind_lock_methods(f)
    f._lock_bursts = {c.key: 6 for c in cells if c.key[-1] == TOPGX}
    f._lock_yields = {c.key: (5 if c.key[-2] % 2 else 0)
                      for c in cells if c.key[-1] == TOPGX}
    out = []
    for _ in range(400):
        c = f.select()
        out.append(c.key if c is not None else None)
    return out


def _merit_footprint(mode: str) -> tuple:
    """P2: (# archive merit entries, # per-key burst counters) after a
    real observe() -> _assign() burst loop whose observed cell lands on
    this run's own top gx bucket, i.e. inside the lock by construction.
    Off the lock nothing writes, which is the point of the other file's
    leak test; here the write must happen."""
    f = _burst_solver(_burst_cells(), sel_mode="count",
                      args=SimpleNamespace(deep_bias=0.4),
                      lock_mode=mode, **_LOCK_KW)
    f.game = SimpleNamespace(
        is_dead=lambda ram, lives: False,
        is_finale=lambda wd, ram: False,
        is_clear=lambda wd, ram, ctx: False,
        level_key=lambda ram: (0,),
        progress=lambda ram: 68,
        progress_cap=10_000,
        area=lambda ram: 0,
        y=lambda ram: 0,
        cell_fn=lambda ram: (0, 3, 1, 0, TOPGX),
        score_bonus=lambda ram: 0.0,
    )
    f.max_gx_in_area = {0: 1000}
    f._pin_time = -10_000.0
    # LEX-NOVELTY reads the run's own boundary histogram to decide which
    # addresses are live at this boundary. Seed a small one so the mode
    # is probed with real live axes rather than through its documented
    # "no --gate-opener, so always indifferent" floor.
    f._boundary_hist = np.zeros((ib.RAM_SIZE, 256), dtype=np.uint32)
    f._boundary_hist[16:48, 0] = 3
    f._boundary_hist[16:48, 7] = 2
    f._boundary_hist_total = 5
    f._lock_moved_cache = None
    f._lock_desc_hist = {}
    f._lock_desc_total = 0
    _bind_lock_methods(f)
    _run_bursts(f, 120, burst_len=10)
    return (len(getattr(f.archive, "_merit", {}) or {}),
            len(getattr(f, "_lock_bursts", {}) or {}))


def _probe(mode: str) -> dict:
    off_stream = _select_stream("off")
    off_merit, off_bursts = _merit_footprint("off")
    on_stream = _select_stream(mode)
    on_merit, on_bursts = _merit_footprint(mode)
    return {
        "selection_differs": on_stream != off_stream,
        "observation_differs": (on_merit, on_bursts) != (off_merit, off_bursts),
        "off_footprint": (off_merit, off_bursts),
        "on_footprint": (on_merit, on_bursts),
    }


# ---------------------------------------------------------------------
# The roster itself
# ---------------------------------------------------------------------

def test_the_cli_default_is_off_and_off_is_declared(monkeypatch):
    decl = _declared_lock_objective(monkeypatch)
    assert decl["default"] == "off"
    assert "off" in decl["choices"]


def test_latch_is_not_offered_on_this_branch(monkeypatch):
    # B-O2's LEX-LATCH lives only on the unmerged branch
    # `contra-lock-b-o2`. Offering the name here without the
    # implementation is the exact defect this file was written for; if
    # that branch merges, the name comes back WITH its dispatch and the
    # roster test below is what proves it live.
    assert "latch" not in _declared_lock_objective(monkeypatch)["choices"]


def test_every_declared_lock_objective_actually_dispatches(monkeypatch):
    decl = _declared_lock_objective(monkeypatch)
    dead = []
    for name in decl["choices"]:
        if name == "off":
            continue
        r = _probe(name)
        if not (r["selection_differs"] and r["observation_differs"]):
            dead.append((name, r))
    assert not dead, (
        "--lock-objective names that parse but change nothing "
        f"(they would print as armed and run as off): {dead}")


def test_the_roster_probe_reports_a_fabricated_name_inert():
    # THE NON-VACUITY DIRECTION. The probe above is only worth running
    # if it comes back negative when the mechanism is absent. A name
    # with no branch must fail BOTH halves — same picks, same merit
    # footprint as `off`.
    r = _probe(FABRICATED)
    assert not r["selection_differs"], (
        "a mode with no dispatch changed the pick stream — the probe is "
        "measuring something other than the objective")
    assert not r["observation_differs"], (
        "a mode with no dispatch left a merit footprint — the probe "
        "cannot tell armed from unarmed")
    assert r["on_footprint"] == r["off_footprint"] == (0, 0)


@pytest.mark.parametrize("mode", ["yield", "survival", "novelty"])
def test_each_shipped_objective_is_live_on_both_halves(mode, monkeypatch):
    # The per-mode statement, so a regression names the mode it broke
    # rather than failing the aggregate. Skips rather than fails if a
    # mode is no longer offered, since dropping one is a legitimate
    # choice the aggregate test above already covers.
    if mode not in _declared_lock_objective(monkeypatch)["choices"]:
        pytest.skip(f"--lock-objective {mode} is not declared on this branch")
    r = _probe(mode)
    assert r["selection_differs"], f"{mode} does not reach select()"
    assert r["observation_differs"], f"{mode} does not reach observe()/_assign()"


# ---------------------------------------------------------------------
# The other way a mode reads as armed when it is not: the clock
# ---------------------------------------------------------------------

def test_the_two_lock_clocks_are_not_the_same_number():
    # `lock_armed_secs` used to be `now - _pin_time` — time since the
    # frontier last moved, which begins accruing --lock-pin-secs BEFORE
    # the objective steers anything. Two lock2 reports read it as armed
    # time and overstated their runs by 2.4-2.5x. The two clocks must
    # differ by exactly the arming threshold while pinned.
    pinned, armed = lock_clocks(pin_time=0.0, now=500.0, pin_secs=300.0)
    assert (pinned, armed) == (500, 200)


def test_armed_secs_is_zero_exactly_while_the_objective_is_not_armed():
    # THE CROSS-CHECK, and the reason this is not a restatement of the
    # formula: `lock_armed()` is the predicate select()/observe()
    # actually gate on, written independently of the telemetry. If
    # either drifts, the reported clock stops meaning "the objective was
    # running", which is the only thing a reader uses it for.
    for elapsed in (0.0, 1.0, 299.0, 299.999, 300.0, 300.001, 900.0):
        _, armed = lock_clocks(0.0, elapsed, 300.0)
        live = lock_armed("survival", 0.0, elapsed, 300.0)
        # No false positives: a non-zero armed clock must never appear
        # while the predicate the arms gate on says unarmed.
        assert not (armed > 0 and not live), (
            f"reported {armed}s of arming at {elapsed}s while "
            f"lock_armed() is False")
        # And unarmed must read exactly zero, not "small".
        if not live:
            assert armed == 0, (
                f"unarmed at {elapsed}s but the clock reads {armed}s")
    # And the boundary itself: armed exactly at the threshold, but zero
    # seconds of arming have yet elapsed.
    assert lock_armed("survival", 0.0, 300.0, 300.0) is True
    assert lock_clocks(0.0, 300.0, 300.0)[1] == 0


def test_an_unselected_mode_reports_no_armed_time_regardless_of_pin():
    # The negative direction: "off" never arms no matter how long the
    # frontier has been pinned, so a progress line that prints a
    # non-zero lock_armed_secs for it would be lying. The telemetry
    # block is entered only when lock_mode != "off", source-checked here
    # so this cannot pass by the clock alone.
    #
    # Read the FILE, not inspect.getsource(Solver.progress_line): that
    # resolves a method through linecache and returned a DIFFERENT
    # method's body when this file ran inside the full suite, passing in
    # isolation and failing under ordering. A source assertion that
    # depends on suite order is worse than no source assertion.
    import scripts.go_explore_solve as ges
    src = pathlib.Path(ges.__file__).read_text()
    assert "def progress_line" in src
    body = src.split("def progress_line", 1)[1]
    guard = 'getattr(self, "lock_mode", "off") != "off"'
    assert guard in body, "the lock telemetry block lost its mode guard"
    # ...and the guard precedes the clocks it protects, in that block.
    assert body.index(guard) < body.index('line["lock_armed_secs"]')
    assert lock_armed("off", 0.0, 10_000.0, 300.0) is False
