"""The armed-but-inert check, tested in BOTH directions.

A checker that reports INERT for everything is as useless as one that
reports FIRED for everything, and either would sail through a one-sided
test suite. So every verdict here is pinned against its opposite on the
same shaped input:

    FIRED  <-> INERT          (same log, counter moves vs counter frozen)
    ARMED  <-> NOT_ARMED      (banner present vs absent)
    INERT  <-> UNAUDITABLE    (counter read N>0 times vs never read)

The last pair is the one that matters most and is the easiest to get
wrong. "The counter is zero" and "there is no counter" are different
facts: the first is a null result, the second is ignorance. A checker
that collapses them would have called `hazard_mask` -- which emits no
veto counter at all -- a clean pass, which is precisely how an unaudited
mechanism stays unaudited.

Two tests run against the REAL banked receipts rather than fixtures, so
the check is anchored to the artifacts it was built for:

  * `runs/v27_fresh_recovery` must read redo INERT. That is the defect a
    human audit needed a day to find, and this reproduces it in ~10ms.
  * `checkpoints/mario_1_2_online_v2` must read sil/kl_anchor/backward all
    FIRED. This is the POSITIVE CONTROL: without it, a checker that
    returned INERT unconditionally would still pass every negative test
    above, and the 1-2 campaign -- whose mechanisms were genuinely live --
    would be defamed by its own tooling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_mechanism_receipt import (  # noqa: E402
    FIRED,
    INERT,
    NOT_ARMED,
    UNAUDITABLE,
    check_run,
    main,
)

PREFIX = "00:22:03 [INFO] src.training.trainer: "


def _verdict(report, name: str) -> str:
    return next(r.verdict for r in report.readings if r.name == name)


def _write_log(d: Path, lines: list[str]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.log").write_text("".join(PREFIX + ln + "\n" for ln in lines))
    return d


def _redo_lines(cums: list[int]) -> list[str]:
    """`[redo] ENABLED` + one per-iteration check line per cumulative count."""
    out = ["[redo] ENABLED tau=0.025 every_iters=1 scope=fc1,fc2"]
    out += [
        f"[redo] iter {i}: dormant fc1 0/64 fc2 0/32 recycled 0 cum {c} "
        f"agree 1.0000 max_dlogit 0.000000"
        for i, c in enumerate(cums)
    ]
    return out


# ---------------------------------------------------------------------------
# FIRED <-> INERT: the same log shape, one counter moving and one frozen
# ---------------------------------------------------------------------------

def test_counter_that_never_moves_is_inert(tmp_path: Path) -> None:
    r = check_run(_write_log(tmp_path, _redo_lines([0] * 250)))
    assert _verdict(r, "redo") == INERT
    assert r.void_for == ["redo"]


def test_the_same_log_with_a_moving_counter_is_fired(tmp_path: Path) -> None:
    """The positive control for the negative above: one recycle at the very
    end is enough, and nothing else about the log changes."""
    r = check_run(_write_log(tmp_path, _redo_lines([0] * 249 + [1])))
    assert _verdict(r, "redo") == FIRED
    assert r.void_for == []


def test_unarmed_mechanism_is_not_reported_as_a_defect(tmp_path: Path) -> None:
    lines = [ln for ln in _redo_lines([0] * 5) if "ENABLED" not in ln]
    r = check_run(_write_log(tmp_path, lines + ["[vanilla_ppo] iter 0"]))
    assert _verdict(r, "redo") == NOT_ARMED
    assert r.void_for == []


def test_explicit_disable_line_beats_a_stale_enabled_banner(tmp_path: Path) -> None:
    r = check_run(_write_log(tmp_path, _redo_lines([0] * 5) + ["[redo] disabled"]))
    assert _verdict(r, "redo") == NOT_ARMED


# ---------------------------------------------------------------------------
# INERT <-> UNAUDITABLE: a zero counter is a null, no counter is ignorance
# ---------------------------------------------------------------------------

def test_armed_with_no_counter_at_all_is_unauditable_not_fired(
    tmp_path: Path,
) -> None:
    """`hazard_mask` announces itself and then never reports a veto.

    This must NOT read FIRED. An armed mechanism whose effect cannot be
    observed is the raw material of the next vacuous gate, and the checker
    says so out loud.
    """
    r = check_run(_write_log(tmp_path, [
        "[hazard-mask] ARMED threshold=0.6 from runs/engine/hazard/model.pt",
        "[vanilla_ppo] iter 0 throughput: 400 env-steps/s",
    ]))
    assert _verdict(r, "hazard_mask") == UNAUDITABLE
    assert "hazard_mask" in r.void_for


def test_armed_with_a_counter_that_was_never_observed_is_unauditable(
    tmp_path: Path,
) -> None:
    """ReDo armed, but the run recorded not one check line.

    Calling that INERT would be inventing a null result out of missing
    data -- the exact move that makes a gate vacuous.
    """
    r = check_run(_write_log(tmp_path, [
        "[redo] ENABLED tau=0.025 every_iters=1 scope=fc1,fc2",
        "[vanilla_ppo] iter 0 throughput: 400 env-steps/s",
    ]))
    reading = next(x for x in r.readings if x.name == "redo")
    assert reading.verdict == UNAUDITABLE
    assert reading.observations == 0


def test_inert_and_unauditable_are_distinguished_on_observation_count(
    tmp_path: Path,
) -> None:
    """The two VOID verdicts must not be interchangeable."""
    seen = check_run(_write_log(tmp_path / "seen", _redo_lines([0] * 12)))
    unseen = check_run(_write_log(tmp_path / "unseen", _redo_lines([])))
    assert _verdict(seen, "redo") == INERT
    assert next(x for x in seen.readings if x.name == "redo").observations == 12
    assert _verdict(unseen, "redo") == UNAUDITABLE


# ---------------------------------------------------------------------------
# Ladder semantics: frozen mid-walk is inert, parked at the entrance is not
# ---------------------------------------------------------------------------

def test_ladder_frozen_at_a_nonzero_rung_is_inert(tmp_path: Path) -> None:
    lines = ["[backward] ENABLED: 785 states (0 MB) from ckpt/backward_states"]
    lines += [f"[backward] iter {i}: tau=744/785 (step 0 frame 0 gx 40) "
              f"trailing 26/30=0.87" for i in range(40)]
    assert _verdict(check_run(_write_log(tmp_path, lines)), "backward") == INERT


def test_ladder_parked_at_the_entrance_is_armed_and_arrived(
    tmp_path: Path,
) -> None:
    """Entrance-pinned consolidation holds tau at 0 for a whole run by
    design -- the 1-2 campaign's own shape. Armed and arrived, not inert."""
    lines = ["[backward] ENABLED: 6 states (0 MB) from ckpt/restart_states"]
    lines += [f"[backward] iter {i}: tau=0/5 (step 0 frame 0 gx 0) "
              f"trailing 26/30=0.87" for i in range(40)]
    assert _verdict(check_run(_write_log(tmp_path, lines)), "backward") == FIRED


def test_ladder_that_walks_is_fired(tmp_path: Path) -> None:
    lines = ["[backward] ENABLED: 785 states (0 MB) from ckpt/backward_states"]
    lines += [f"[backward] iter {i}: tau={744 - i * 20}/785 (step 0 frame 0 "
              f"gx 40) trailing 26/30=0.87" for i in range(20)]
    assert _verdict(check_run(_write_log(tmp_path, lines)), "backward") == FIRED


# ---------------------------------------------------------------------------
# metrics.jsonl-backed mechanisms
# ---------------------------------------------------------------------------

def _write_metrics(d: Path, rows: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_demo_anchor_armed_but_zero_loss_is_inert(tmp_path: Path) -> None:
    """The v27/v28 shape, had the coefficient actually been armed."""
    r = check_run(_write_metrics(tmp_path, [
        {"demo_anchor_coef": 0.5, "demo_anchor_loss": 0.0} for _ in range(250)]))
    assert _verdict(r, "demo_anchor") == INERT


def test_demo_anchor_with_a_zero_coefficient_is_simply_not_armed(
    tmp_path: Path,
) -> None:
    """What v27/v28 ACTUALLY recorded: coef 0.0 on all 250 rows.

    The distinction is load-bearing for how the run gets described. "Armed
    and inert" is a VOID; "never armed" is just a mechanism this run did
    not use, and it is not a defect.
    """
    r = check_run(_write_metrics(tmp_path, [
        {"demo_anchor_coef": 0.0, "demo_anchor_loss": 0.0} for _ in range(250)]))
    assert _verdict(r, "demo_anchor") == NOT_ARMED
    assert r.void_for == []


def test_sil_counter_moving_is_fired(tmp_path: Path) -> None:
    r = check_run(_write_metrics(tmp_path, [
        {"sil_loss": 0.2, "sil_clears_total": i * 7} for i in range(1, 40)]))
    assert _verdict(r, "sil") == FIRED


def test_sil_armed_with_a_pinned_clear_count_is_inert(tmp_path: Path) -> None:
    r = check_run(_write_metrics(tmp_path, [
        {"sil_loss": 0.2, "sil_clears_total": 0} for _ in range(40)]))
    assert _verdict(r, "sil") == INERT


# ---------------------------------------------------------------------------
# Anti-vacuity: the checker must never certify what it did not read
# ---------------------------------------------------------------------------

def test_empty_directory_is_an_error_not_a_pass(tmp_path: Path) -> None:
    d = tmp_path / "nothing"
    d.mkdir()
    r = check_run(d)
    assert r.error is not None
    assert r.readings == []


def test_cli_exits_1_on_an_unreadable_run(tmp_path: Path, capsys) -> None:
    d = tmp_path / "nothing"
    d.mkdir()
    assert main([str(d)]) == 1


def test_cli_exits_2_on_an_inert_mechanism(tmp_path: Path, capsys) -> None:
    assert main([str(_write_log(tmp_path, _redo_lines([0] * 50)))]) == 2


def test_cli_exits_0_when_every_armed_mechanism_fired(
    tmp_path: Path, capsys,
) -> None:
    assert main([str(_write_log(tmp_path, _redo_lines([0, 0, 3])))]) == 0


def test_require_voids_a_run_that_never_armed_the_named_mechanism(
    tmp_path: Path, capsys,
) -> None:
    """A registration naming ReDo as a variable cannot be satisfied by a run
    that never switched it on."""
    d = _write_log(tmp_path, ["[vanilla_ppo] iter 0 throughput: 400 env-steps/s"])
    assert main([str(d)]) == 0
    assert main([str(d), "--require", "redo"]) == 2


def test_require_rejects_an_unknown_mechanism_name(
    tmp_path: Path, capsys,
) -> None:
    d = _write_log(tmp_path, _redo_lines([0, 0, 3]))
    assert main([str(d), "--require", "no_such_mechanism"]) == 2


def test_json_output_is_machine_readable(tmp_path: Path, capsys) -> None:
    main([str(_write_log(tmp_path, _redo_lines([0] * 20))), "--json"])
    blob = json.loads(capsys.readouterr().out)
    assert blob[0]["void_for"] == ["redo"]


# ---------------------------------------------------------------------------
# The real banked receipts
# ---------------------------------------------------------------------------

V27 = REPO / "runs" / "v27_fresh_recovery"
ONLINE_1_2 = REPO / "checkpoints" / "mario_1_2_online_v2"


@pytest.mark.skipif(not V27.exists(), reason="v27 receipts not present")
def test_real_v27_receipts_read_redo_as_inert() -> None:
    """The finding, reproduced from the artifacts in milliseconds.

    ReDo was registered as one of two variables in v27's AMENDMENT 1 and
    logged `cum 0` on every check across all four seeds. v27 and v28 were
    therefore SINGLE-variable arms. Their FAIL verdicts are untouched by
    this -- neither depended on ReDo doing anything -- but the sentence
    "we tested ReDo and it did not help" was never true.
    """
    r = check_run(V27)
    reading = next(x for x in r.readings if x.name == "redo")
    assert reading.verdict == INERT
    assert reading.peak == 0.0
    assert reading.observations > 500, "four seeds x ~250 checks"


@pytest.mark.skipif(not ONLINE_1_2.exists(), reason="1-2 receipts not present")
def test_real_1_2_campaign_receipts_read_every_mechanism_as_live() -> None:
    """THE POSITIVE CONTROL.

    Without this, a checker hard-wired to return INERT would pass every
    other test in this file. The 1-2 online campaign's registered
    mechanisms were genuinely live in its own receipts, and the checker
    must agree.
    """
    r = check_run(ONLINE_1_2)
    assert r.error is None
    for name in ("sil", "kl_anchor", "backward"):
        assert _verdict(r, name) == FIRED, name
    assert r.void_for == []
