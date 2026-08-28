"""Anti-vacuity tests for the V31 in-run dose ceiling (abort A4).

`V31_REDO_SURGICAL_2026-08-27.md` §3 moves v30's post-hoc V6 over-dose
check (`scripts/redo_arm_gate.py`) into the training loop itself: the
untreated dormant tail is measured to drift DOWN across training
(`docs/proposals/V30_REDO_ARMED_2026-08-27.md` §5/§1.3), so a tau that is
surgical at iter 30 can become a partial network reset by iter 200. A
post-hoc-only check discovers that after burning a full run; this ceiling
catches it live.

Every test here is a REPLAY against banked receipts (the same per-iteration
counts `scripts/redo_arm_gate.py`'s own tests replay), not a synthetic
story invented for this file — three real positives (the pilots that DID
overdose), two real negatives (the controls that never fired past the
threshold), and one synthetic negative (a genuine surgical trace that must
never trip). Deleting the ceiling must make the positive cases fail; that
deletion is executed and recorded in the campaign receipts, not asserted
here (a permanent test cannot delete the code it is testing).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.redo import (  # noqa: E402
    DOSE_CEILING_FRAC,
    DOSE_CEILING_WINDOW,
    dose_ceiling_trips,
    dose_fraction,
)

PILOT = REPO / "runs" / "v30_premise_falsifier_2026-08-27"

# Mirrors scripts/redo_arm_gate.py's ITER_RE exactly — a second, purpose-
# built parser would risk drifting from the log format the gate itself
# reads; this one is copied, not imported, so a change to the gate's
# regex cannot silently change what this ceiling test replays.
ITER_RE = re.compile(
    r"\[redo\] iter (\d+): dormant fc1 (\d+)/(\d+) fc2 (\d+)/(\d+) "
    r"recycled (\d+) cum (\d+) agree ([0-9.]+) max_dlogit ([0-9.eE+-]+)"
)


def _replay_first_trip(path: Path) -> int | None:
    """Feed one banked log's dormancy checks through the ceiling in
    iteration order; return the iter at which it first trips, or None.
    """
    history: list[float] = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
            m = ITER_RE.search(line)
            if m is None:
                continue
            it = int(m.group(1))
            hidden, trunk = int(m.group(3)), int(m.group(5))
            d1, d2 = int(m.group(2)), int(m.group(4))
            history.append(dose_fraction(d1, hidden, d2, trunk))
            if dose_ceiling_trips(history):
                return it
    return None


@pytest.mark.parametrize(
    "log,expect_trip_iter",
    [
        ("pilot_tau0.25_h64.log", 9),
        ("pilot_tau0.25_h96.log", 9),
        ("pilot_tau0.15_h64.log", 10),
        ("control_tau0.025_h64_VOIDED.log", None),
        ("armed_check_inert_case_VOIDED.log", None),
    ],
)
def test_ceiling_replays_the_banked_pilots_correctly(log, expect_trip_iter):
    """Three real overdoses trip at the iterations V31 §3.3 names; the two
    untreated/inert controls never trip in the whole banked window.

    REVERT-VERIFIED FAILURE (executed and recorded in the campaign
    receipts, not asserted here): comment out the `RuntimeError` raise in
    `src/training/trainer.py`'s dose-ceiling block, or replace
    `dose_ceiling_trips` with a function that always returns False — the
    three positive cases below then report `None`, i.e. an
    e.g. 62%-of-the-trunk-per-iteration run would burn to iter 250
    uninterrupted instead of aborting at iter ~9.
    """
    path = PILOT / log
    if not path.is_file():
        pytest.skip(f"pilot receipt not present: {path}")
    assert _replay_first_trip(path) == expect_trip_iter


def test_a_surgical_2_to_4_unit_trace_never_trips():
    """The ceiling is not vacuous in the other direction: a genuine
    surgical dose (2-4 of 32 trunk units per firing event, the v31
    operating regime) never trips across a full 250-iteration horizon.
    """
    history: list[float] = []
    import itertools
    # 2-4 units/32 per firing event, firing every other check from
    # iter 16 on — the v31 §1.2/§1.3 expected shape.
    doses = itertools.cycle([0.0, 2 / 32, 0.0, 3 / 32, 0.0, 4 / 32])
    for it in range(250):
        frac = 0.0 if it < 16 else next(doses)
        history.append(frac)
        assert not dose_ceiling_trips(history), (
            f"surgical trace falsely tripped the ceiling at iter {it}"
        )


def test_fewer_than_window_checks_never_trips():
    """`dose_ceiling_trips` must not fire before it has a full trailing
    window to compute a median over — even a single 100%-dose check
    must not abort a run 2 checks in.

    REVERT-VERIFIED FAILURE: drop the `len(history) < window` guard and
    a single early spike (e.g. cold-start noise before the network has
    seen any data) aborts a run that never had a chance to equilibrate.
    """
    for n in range(DOSE_CEILING_WINDOW):
        history = [1.0] * n
        assert not dose_ceiling_trips(history)


def test_dose_fraction_is_worst_hit_layer_not_pooled():
    """20/32 fc2 (62.5%) with fc1 untouched must read 0.625, not the
    pooled 20/96 = 20.8% that would silently pass the 0.25 ceiling.

    REVERT-VERIFIED FAILURE: change `dose_fraction` to
    `(dormant_fc1 + dormant_fc2) / (hidden_dim + trunk_dim)` and this
    assertion fails (0.2083... != 0.625) — the exact vacuity V6 itself
    was built to avoid, reintroduced at the ceiling instead of the gate.
    """
    assert dose_fraction(0, 64, 20, 32) == pytest.approx(0.625)


def test_registered_numerals_are_the_ones_in_the_module():
    """V31 §3.1 fixes these two numbers by name: window=10 checks,
    ceiling=0.25, the SAME numeral scripts/redo_arm_gate.py's V6 uses at
    verdict time, so in-run and post-hoc can never disagree.
    """
    assert DOSE_CEILING_WINDOW == 10
    assert DOSE_CEILING_FRAC == 0.25


def test_strict_inequality_admits_exactly_8_of_32():
    """§3.1: 8/32 = 0.25 survives (strict `>`), 9/32 = 0.28125 aborts —
    the registered one-unit margin at the §1.3 equilibrium prediction.
    """
    history_at_ceiling = [8 / 32] * DOSE_CEILING_WINDOW
    history_over_ceiling = [9 / 32] * DOSE_CEILING_WINDOW
    assert not dose_ceiling_trips(history_at_ceiling)
    assert dose_ceiling_trips(history_over_ceiling)


def test_trainer_wires_the_ceiling_into_the_ppo_loop():
    """Static check that the ceiling is actually reachable on the vanilla
    PPO training path (not just importable from redo.py) — six signals
    have been built and wired to nothing before in this campaign lineage.

    REVERT-VERIFIED FAILURE: remove the wiring lines below from
    `src/training/trainer.py` (leaving `dose_ceiling_trips` defined but
    uncalled) and this test fails while `test_redo_dose_ceiling.py`'s
    other tests (which exercise `redo.py` directly) keep passing —
    exactly the "defined but not wired" failure mode this test exists to
    catch.
    """
    src = (REPO / "src" / "training" / "trainer.py").read_text()
    assert "_redo_dose_ceiling_trips(" in src
    assert "_redo_dose_ceiling_history" in src
    assert "VOID-OVERDOSE" in src
    assert "_REDO_DOSE_CEILING = 0.25" in src
    # Wired inside the same hook the arming deadline lives in, not a
    # dead branch elsewhere.
    hook = src.index("_redo_maybe_check(")
    ceiling_site = src.index("_redo_dose_ceiling_trips(", hook)
    deadline_site = src.index("_REDO_ARM_DEADLINE_ITERS\n", hook)
    assert hook < ceiling_site < deadline_site
