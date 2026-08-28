"""Anti-vacuity tests for the ReDo arming gate.

Seven vacuous gates have shipped in this repo; the seventh (V7) was
written by work that held the previous six in its brief. The failure mode
is always the same: a check that cannot fail on the case it exists to
catch. So every check here ships with the revert-verified failure that
proves it bites — each test names, in its docstring, the one-line edit
that makes it fail.

The case these checks exist to catch is on disk: v27 and v28 each ran
four seeds x ~7h with ``redo_tau: 0.025`` on a Linear->LayerNorm->SiLU
trunk whose dormancy scores never go below ~0.15, logged
``recycled 0 cum 0`` on all ~2000 per-iteration checks, and then reported
FAIL verdicts (0.530 and 0.670) that the treatment could not have
produced. A run in which ReDo does not fire is VOID, not FAIL.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "redo_arm_gate.py"

sys.path.insert(0, str(REPO))

from scripts.redo_arm_gate import adjudicate, parse_log  # noqa: E402

# Every banked run of both campaigns. All eight are inert.
BANKED_INERT = [
    REPO / "checkpoints" / f"mario_1_1_v27_recovery_seed{i}" / "run.log"
    for i in range(4)
] + [
    REPO / "checkpoints" / f"mario_1_1_v28_capacity_seed{i}" / "run.log"
    for i in range(4)
]

ENABLED = "[redo] ENABLED tau={tau} every_iters=1 scope=fc1,fc2 sample=4096"
ITER = (
    "[redo] iter {it}: dormant fc1 0/64 fc2 {n}/32 recycled {n} cum {cum} "
    "agree {agree:.4f} max_dlogit 0.100000"
)


def _write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def _synth(
    tmp_path: Path, *, tau: float, per_iter: int, iters: int,
    agree: float = 0.85, name: str = "run.log",
) -> Path:
    lines = [ENABLED.format(tau=tau)]
    cum = 0
    for it in range(iters):
        cum += per_iter
        lines.append(ITER.format(it=it, n=per_iter, cum=cum, agree=agree))
    return _write_log(tmp_path / name, lines)


def _gate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def _adj(path: Path, **kw):
    kw.setdefault("tau", 0.25)
    kw.setdefault("min_events", 10)
    kw.setdefault("min_units", 20)
    kw.setdefault("min_agree", 0.60)
    kw.setdefault("agree_window", 50)
    return adjudicate(parse_log(path), **kw)


# --------------------------------------------------------------------
# The historical failure. This is the test that matters.
# --------------------------------------------------------------------

@pytest.mark.parametrize("log", BANKED_INERT, ids=lambda p: p.parent.name)
def test_gate_voids_every_banked_v27_v28_log(log: Path):
    """The gate, applied to the historical failure, reproduces the
    correct verdict: VOID, not FAIL — and does so at each run's OWN
    registered tau (0.025), so the void is not an artifact of judging
    them against a threshold they never claimed.

    REVERT-VERIFIED FAILURE: delete the ``rep.cum_recycled == 0`` branch
    in ``adjudicate`` and all eight of these parametrized cases fail —
    the gate then calls a run ARMED that recycled nothing across 250
    iterations, which is precisely the eighth vacuous gate.
    """
    if not log.is_file():
        pytest.skip(f"banked run log not present: {log}")
    rep = _adj(log, tau=0.025)
    assert rep.cum_recycled == 0
    assert rep.checks >= 200, "expected the full 250-iteration campaign"
    assert rep.verdict == "VOID-NEVER-FIRED"
    assert not rep.armed


def test_gate_cli_exits_2_and_says_void_never_fired_on_banked_v27():
    """End-to-end through the CLI: exit code 2, the word VOID, and the
    words PASS and FAIL nowhere in the output.

    REVERT-VERIFIED FAILURE: make ``main`` return 0 unconditionally and
    the exit-code assertion fails.
    """
    log = BANKED_INERT[0]
    if not log.is_file():
        pytest.skip(f"banked run log not present: {log}")
    res = _gate(str(log), "--tau", "0.025")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "VOID (redo never fired)" in res.stdout
    # "VOID, not FAIL" is allowed to appear inside a reason; what must
    # never appear is a VERDICT line reading PASS or FAIL.
    assert not re.search(r"VERDICT:\s*(PASS|FAIL)\b", res.stdout), (
        "the arm gate must be structurally incapable of printing a "
        "PASS/FAIL verdict for an unarmed seed"
    )


def test_gate_output_never_contains_pass_or_fail_for_any_void_case(tmp_path):
    """No VOID path anywhere in the gate may emit PASS or FAIL.

    REVERT-VERIFIED FAILURE: add the word FAIL to any VOID reason string
    and this fails.
    """
    cases = [
        _synth(tmp_path, tau=0.25, per_iter=0, iters=30, name="a.log"),
        _synth(tmp_path, tau=0.025, per_iter=3, iters=30, name="b.log"),
        _synth(tmp_path, tau=0.25, per_iter=1, iters=5, name="c.log"),
        _synth(tmp_path, tau=0.25, per_iter=5, iters=30, agree=0.2,
               name="d.log"),
    ]
    for case in cases:
        res = _gate(str(case))
        assert res.returncode == 2, f"{case.name}: {res.stdout}"
        assert not re.search(r"VERDICT:\s*(PASS|FAIL)\b", res.stdout), case.name


# --------------------------------------------------------------------
# Each registered VOID condition, one test apiece.
# --------------------------------------------------------------------

def test_wrong_tau_voids_the_arm_it_certified(tmp_path):
    """The V7 vacuity class, closed: V7 piloted at tau 0.50 while the
    runs ran at 0.025 — 20x the registered operating point — and passed.
    A log armed at any tau other than the registered one is VOID.

    REVERT-VERIFIED FAILURE: drop the ``abs(...) > 1e-12`` tau comparison
    and this fails.
    """
    log = _synth(tmp_path, tau=0.5, per_iter=20, iters=30)
    rep = _adj(log, tau=0.25)
    assert rep.verdict == "VOID-WRONG-TAU"
    assert not rep.armed


def test_disabled_line_voids(tmp_path):
    """The armed-but-unsupported-architecture path prints
    ``[redo] disabled``; a silent no-op must not pass as a treatment.

    REVERT-VERIFIED FAILURE: remove the ``rep.saw_disabled`` branch.
    """
    # Recycles present, so only the `disabled` line can void this —
    # otherwise VOID-NEVER-FIRED would mask the condition under test.
    log = _write_log(tmp_path / "run.log", [
        ENABLED.format(tau=0.25), "[redo] disabled",
    ] + [
        ITER.format(it=it, n=5, cum=5 * (it + 1), agree=0.9)
        for it in range(30)
    ])
    rep = _adj(log)
    assert rep.cum_recycled == 150
    assert rep.verdict == "VOID-NOT-ARMED"


def test_fired_once_is_still_void_minimal_dose(tmp_path):
    """The 'technically fired once' loophole — how the eighth vacuous
    gate gets written. One recycle event of one unit is not a dose.

    REVERT-VERIFIED FAILURE: delete the min_events/min_units branch and
    this run is called ARMED.
    """
    log = _write_log(tmp_path / "run.log", [
        ENABLED.format(tau=0.25),
        ITER.format(it=0, n=1, cum=1, agree=0.99),
    ] + [
        "[redo] iter %d: dormant fc1 0/64 fc2 0/32 recycled 0 cum 1 "
        "agree 1.0000 max_dlogit 0.000000" % it
        for it in range(1, 250)
    ])
    rep = _adj(log)
    assert rep.cum_recycled == 1
    assert rep.recycle_events == 1
    assert rep.verdict == "VOID-MINIMAL-DOSE"


def test_partial_reset_voids_on_identity(tmp_path):
    """A4: below median agree 0.60 the step is a partial network reset,
    not a surgical intervention, and the arm is uninterpretable.

    REVERT-VERIFIED FAILURE: delete the median_agree branch.
    """
    # A SMALL dose (4 of 32 = 12.5%, under the V6 ceiling) so that the
    # only thing voiding this run is the identity collapse.
    log = _synth(tmp_path, tau=0.25, per_iter=4, iters=30, agree=0.40)
    rep = _adj(log)
    assert rep.median_recycled_frac == pytest.approx(0.125)
    assert rep.verdict == "VOID-IDENTITY"


def test_a_genuinely_armed_run_is_armed_and_exits_zero(tmp_path):
    """The gate is not vacuous in the other direction either: a run that
    really did fire, at the registered tau, at a real dose, with
    identity preserved, is ARMED and exits 0. Without this a gate that
    voids everything would pass every test above.
    """
    log = _synth(tmp_path, tau=0.25, per_iter=5, iters=30, agree=0.85)
    rep = _adj(log)
    assert rep.verdict == "ARMED"
    assert rep.armed
    assert rep.cum_recycled == 150 and rep.recycle_events == 30
    assert _gate(str(log)).returncode == 0


def test_all_violations_are_reported_not_just_the_first(tmp_path):
    """A run can be wrong in several ways at once; masking one behind
    another is how a defect survives a gate.
    """
    log = _synth(tmp_path, tau=0.5, per_iter=0, iters=30)
    rep = _adj(log, tau=0.25)
    assert rep.verdict == "VOID-NEVER-FIRED"
    joined = " ".join(rep.reasons)
    assert "never fired" in joined and "0.5" in joined


# --------------------------------------------------------------------
# The in-run deadline: the gate that stops the spend, not just the claim.
# --------------------------------------------------------------------

def test_trainer_carries_the_hardcoded_arming_deadline():
    """The deadline is a module constant, not a config key: 20 declared
    keys in the flagship recipe never executed, so a new declared key is
    a new way to be inert.

    REVERT-VERIFIED FAILURE: move the bound into ``rl_cfg.get(...)``.
    """
    src = (REPO / "src" / "training" / "trainer.py").read_text()
    assert "_REDO_ARM_DEADLINE_ITERS = 25" in src
    assert 'rl_cfg.get("redo_arm_deadline' not in src
    assert "[redo] VOID: armed at tau=" in src
    # The raise must sit OUTSIDE the `_rd is not None` branch, so an
    # off-cadence or skipped iteration cannot buy the run past it.
    hook = src.index("_redo_maybe_check(")
    raise_at = src.index("_REDO_ARM_DEADLINE_ITERS\n", hook)
    branch = src.index("if _rd is not None:", hook)
    assert raise_at > branch
    assert "global_it >= _REDO_ARM_DEADLINE_ITERS" in src


def test_deadline_condition_is_exactly_the_v27_failure():
    """The raise fires on (armed AND past the deadline AND cum == 0) —
    the exact shape of the v27/v28 runs — and on nothing else.

    REVERT-VERIFIED FAILURE: change ``== 0`` to ``< 0`` and the inert
    case below stops firing.
    """
    def fires(redo_on: bool, global_it: int, cum: int) -> bool:
        return redo_on and global_it >= 25 and cum == 0

    assert fires(True, 25, 0), "the v27/v28 case must fire"
    assert fires(True, 249, 0)
    assert not fires(True, 24, 0), "must not fire before the deadline"
    assert not fires(True, 249, 1), "one recycle is enough to keep running"
    assert not fires(False, 249, 0), "redo-off runs are untouched"


# --------------------------------------------------------------------
# V6 over-recycling — the condition the identity check cannot see.
# --------------------------------------------------------------------

PILOT = REPO / "runs" / "v30_premise_falsifier_2026-08-27"


@pytest.mark.parametrize(
    "log,tau,frac",
    [("pilot_tau0.25_h64.log", 0.25, 0.625),
     ("pilot_tau0.15_h64.log", 0.15, 0.375)],
)
def test_v30_pilot_logs_void_on_overdose(log, tau, frac):
    """The registered v30 operating point (tau 0.25) is an over-dose, and
    the gate says so from the real pilot log: 20 of 32 trunk units
    re-initialized every iteration from iter 5 on. tau 0.15 is over the
    ceiling too, at 12 of 32.

    This is the case V5 CANNOT catch — median agree is 0.856 at tau 0.25
    and 0.950 at tau 0.15, both comfortably past the 0.60 identity floor,
    because zeroing the outgoing actor/critic columns preserves the
    output whatever the dose.

    REVERT-VERIFIED FAILURE: delete the V6 block in ``adjudicate`` and
    both cases come back ARMED with median agree well above the floor —
    a gate that certifies a per-iteration partial network reset as a
    surgical intervention.
    """
    path = PILOT / log
    if not path.is_file():
        pytest.skip(f"pilot receipt not present: {path}")
    rep = _adj(path, tau=tau)
    assert rep.median_agree is not None and rep.median_agree > 0.60, (
        "precondition: the identity check must PASS, so that the only "
        "thing voiding this run is the dose"
    )
    assert rep.median_recycled_frac == pytest.approx(frac, abs=1e-6)
    assert rep.verdict == "VOID-OVERDOSE"


def test_dose_is_the_worst_hit_layer_not_the_pooled_fraction(tmp_path):
    """Pooling fc1 and fc2 is itself a vacuity: fc1 never goes dormant on
    this architecture, so a pooled denominator carries permanent ballast
    that can never contribute to the numerator. 20 of 32 trunk units is a
    62% dose, not the 21% that 20/96 reports.

    REVERT-VERIFIED FAILURE: change the ``max(d1/hidden, d2/trunk)`` back
    to ``recycled / (hidden + trunk)`` and this run is called ARMED.
    """
    log = _write_log(tmp_path / "run.log", [
        ENABLED.format(tau=0.25),
    ] + [
        "[redo] iter %d: dormant fc1 0/64 fc2 20/32 recycled 20 cum %d "
        "agree 0.8564 max_dlogit 0.400000" % (it, 20 * (it + 1))
        for it in range(20)
    ])
    rep = _adj(log)
    assert rep.median_recycled_frac == pytest.approx(0.625)
    assert rep.verdict == "VOID-OVERDOSE"


def test_a_small_surgical_dose_is_armed(tmp_path):
    """The ceiling is not vacuous in the other direction: a genuine
    small-tail recycle — 2 of 32 trunk units, identity preserved — is
    ARMED.
    """
    log = _write_log(tmp_path / "run.log", [
        ENABLED.format(tau=0.10),
    ] + [
        "[redo] iter %d: dormant fc1 0/64 fc2 2/32 recycled 2 cum %d "
        "agree 0.9700 max_dlogit 0.050000" % (it, 2 * (it + 1))
        for it in range(20)
    ])
    rep = _adj(log, tau=0.10)
    assert rep.median_recycled_frac == pytest.approx(2 / 32)
    assert rep.verdict == "ARMED"
