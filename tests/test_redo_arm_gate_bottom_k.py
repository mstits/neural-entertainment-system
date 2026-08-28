"""Bottom-k B1-B4 arming gate — V32_REDO_BOTTOM_K_2026-08-28.md §3/§12
item 3. Replaces F1/F2/F3 wholesale for a bottom_k run; see
scripts/redo_arm_gate.py's module docstring for why F1/F2 alone would
be the tenth vacuous gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.redo_arm_gate import (  # noqa: E402
    adjudicate_bottom_k, parse_log_bottom_k,
)

ENABLED = (
    "[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096 "
    "reset_moments=true mode=bottom_k k=2 recycle_scope=fc2\n"
)


def _event_lines(iter_: int, cum: int, fc2_indices: list[int],
                  scores: list[float]) -> str:
    return (
        f"[redo] iter {iter_}: dormant fc1 0/64 fc2 2/32 recycled 2 "
        f"cum {cum} agree 0.9900 max_dlogit 0.010000 tail fc1 "
        "0.30/0.40/0.44 fc2 0.20/0.30/0.34\n"
        f"[redo] recycled unit indices: fc1=[] fc2={fc2_indices}\n"
        f"[redo] fc2 scores: {scores}\n"
    )


def _healthy_scores(low_a: int, low_b: int) -> list[float]:
    """A 32-value fc2 score vector whose two lowest indices are exactly
    `low_a`/`low_b` — real bottom-k, unambiguous ordering."""
    scores = [1.0 + 0.01 * i for i in range(32)]
    scores[low_a], scores[low_b] = 0.05, 0.06
    return scores


def test_healthy_turning_over_trace_arms(tmp_path):
    # Three events, each recycling a DIFFERENT pair (real turnover), the
    # logged indices always genuinely the two lowest of the logged
    # scores — must ARM at min_events<=3.
    log = tmp_path / "run.log"
    text = ENABLED
    text += _event_lines(0, 2, [9, 13], _healthy_scores(9, 13))
    text += _event_lines(5, 4, [20, 25], _healthy_scores(20, 25))
    text += _event_lines(10, 6, [9, 17], _healthy_scores(9, 17))
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=3,
    )
    assert rep.armed, rep.reasons
    assert rep.artifact_match_frac == 1.0
    assert rep.repeat_rate == 0.0
    # {9, 13, 20, 25, 9, 17} -- unit 9 repeats across events 1 and 3, so
    # 5 distinct indices over 6 recycled-unit-events, not 6.
    assert rep.distinct_fc2_indices == 5


def test_artifact_mismatch_voids(tmp_path):
    # The logged indices do NOT match the two lowest scores in the
    # logged vector — Lane A's lesson: a guard on the artifact, not the
    # pipeline. This is the synthetic mismatch trace the registration
    # requires.
    log = tmp_path / "run.log"
    text = ENABLED
    # Real bottom-2 of this vector is [9, 13] (0.05/0.06); log a
    # different pair instead.
    text += _event_lines(0, 2, [1, 2], _healthy_scores(9, 13))
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=1,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-ARTIFACT-MISMATCH"
    assert rep.artifact_match_frac == 0.0


def test_zero_turnover_voids_at_repeat_rate_exactly_one(tmp_path):
    # The exact same pair recycled on every event -> repeat_rate == 1.00
    # -> VOID-NO-TURNOVER, nothing else gated (this is the synthetic
    # repeat_rate==1.00 trace the registration requires, and it must VOID
    # even though B1/B2/B3 all pass).
    log = tmp_path / "run.log"
    text = ENABLED
    for i, it in enumerate((0, 5, 10, 15)):
        text += _event_lines(it, 2 * (i + 1), [16, 5], _healthy_scores(16, 5))
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=4,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-NO-TURNOVER"
    assert rep.repeat_rate == 1.00
    # B1-B3 all genuinely passed on this trace -- turnover is the ONLY
    # thing wrong with it.
    assert rep.artifact_match_frac == 1.0


def test_partial_turnover_is_reported_but_does_not_void(tmp_path):
    # repeat_rate strictly below 1.00 is reported with no verdict
    # attached (§6.2: "no fraction below 1.0 is asserted").
    log = tmp_path / "run.log"
    text = ENABLED
    text += _event_lines(0, 2, [9, 13], _healthy_scores(9, 13))
    text += _event_lines(5, 4, [9, 20], _healthy_scores(9, 20))  # shares 9
    text += _event_lines(10, 6, [3, 4], _healthy_scores(3, 4))   # no share
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=3,
    )
    assert rep.armed
    assert abs(rep.repeat_rate - 0.5) < 1e-9


def test_wrong_operating_point_voids(tmp_path):
    log = tmp_path / "run.log"
    wrong = (
        "[redo] ENABLED tau=0.025 every_iters=1 scope=fc1,fc2 sample=4096 "
        "reset_moments=true mode=bottom_k k=4 recycle_scope=fc2\n"
    )
    text = wrong + _event_lines(0, 4, [1, 2, 3, 4],
                                 [0.05, 0.06, 0.07, 0.08] + [1.0] * 28)
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=1,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-WRONG-POINT"


def test_disabled_line_voids(tmp_path):
    log = tmp_path / "run.log"
    text = ENABLED + "[redo] disabled\n"
    text += _event_lines(0, 2, [9, 13], _healthy_scores(9, 13))
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=1,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-NOT-ARMED"


def test_fc1_recycle_voids_dose(tmp_path):
    # Scope must be fc2-only; any fc1 recycle under bottom_k is a
    # registration/code defect, not a dose question.
    log = tmp_path / "run.log"
    text = ENABLED
    text += (
        # cum held at k*events=2 so B1's NOT-REACHED condition does not
        # also fire — isolating the fc1-recycle defect as a pure B3/DOSE
        # violation, which is the property this test checks.
        "[redo] iter 0: dormant fc1 1/64 fc2 2/32 recycled 3 cum 2 "
        "agree 0.9900 max_dlogit 0.01 tail fc1 0.1/0.1/0.1 "
        "fc2 0.2/0.3/0.3\n"
        f"[redo] recycled unit indices: fc1=[7] fc2={[9, 13]}\n"
        f"[redo] fc2 scores: {_healthy_scores(9, 13)}\n"
    )
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=1,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-DOSE"


def test_overdose_line_voids(tmp_path):
    log = tmp_path / "run.log"
    text = ENABLED
    text += _event_lines(0, 2, [9, 13], _healthy_scores(9, 13))
    text += "[redo] VOID-OVERDOSE: trailing-10-check median dose ...\n"
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=1,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-DOSE"


def test_too_few_events_voids_not_reached(tmp_path):
    log = tmp_path / "run.log"
    text = ENABLED + _event_lines(0, 2, [9, 13], _healthy_scores(9, 13))
    log.write_text(text)

    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log), k=2, cadence=5, min_events=48,
    )
    assert not rep.armed
    assert rep.verdict == "VOID-NOT-REACHED"


def test_real_smoke_log_arms_at_its_own_event_count():
    # runs/v32_redo_bottom_k_2026-08-28/smoke/smoke_stdout.log — the
    # actual production-path smoke receipt (§13.1). At min_events=3 (its
    # own count, not the full campaign's 48) it must ARM.
    smoke = (ROOT / "runs" / "v32_redo_bottom_k_2026-08-28" / "smoke"
              / "smoke_stdout.log")
    if not smoke.is_file():
        import pytest
        pytest.skip("smoke receipt not present in this checkout")
    rep = adjudicate_bottom_k(
        parse_log_bottom_k(smoke), k=2, cadence=5, min_events=3,
    )
    assert rep.armed, rep.reasons
    assert rep.artifact_match_frac == 1.0
    assert rep.repeat_rate == 0.0
