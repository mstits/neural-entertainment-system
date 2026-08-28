"""Phase R adjudication — V32_REDO_BOTTOM_K_2026-08-28.md §7."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adjudicate_phase_r import adjudicate_phase_r  # noqa: E402

ENABLED = (
    "[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096 "
    "reset_moments=true mode=bottom_k k=2 recycle_scope=fc2\n"
)


def _healthy_scores(low_a: int, low_b: int) -> list[float]:
    scores = [1.0 + 0.01 * i for i in range(32)]
    scores[low_a], scores[low_b] = 0.05, 0.06
    return scores


def _event(iter_: int, cum: int, fc2: list[int], scores: list[float]) -> str:
    return (
        f"[redo] iter {iter_}: dormant fc1 0/64 fc2 2/32 recycled 2 "
        f"cum {cum} agree 0.99 max_dlogit 0.01 tail fc1 0.3/0.4/0.4 "
        "fc2 0.2/0.3/0.3\n"
        f"[redo] recycled unit indices: fc1=[] fc2={fc2}\n"
        f"[redo] fc2 scores: {scores}\n"
    )


def test_healthy_turning_over_12_events_is_go(tmp_path):
    log = tmp_path / "run.log"
    text = ENABLED
    pairs = [(9, 13), (20, 25), (9, 17), (3, 4), (5, 6), (7, 8),
              (10, 11), (12, 14), (15, 16), (18, 19), (21, 22), (23, 24)]
    for i, (a, b) in enumerate(pairs):
        text += _event(i * 5, 2 * (i + 1), [a, b], _healthy_scores(a, b))
    log.write_text(text)

    result = adjudicate_phase_r(log, k=2, cadence=5, min_events=10)
    assert result["go"] is True
    assert result["decision"] == "GO"
    assert result["r1_reached"] and result["r2_artifact_match"]
    assert result["r3_dose"] and result["r4_turnover"]
    assert len(result["recovery_curve"]) == 24  # 12 events x k=2


def test_zero_turnover_across_12_events_no_gos_on_r4(tmp_path):
    log = tmp_path / "run.log"
    text = ENABLED
    for i in range(12):
        text += _event(i * 5, 2 * (i + 1), [16, 5], _healthy_scores(16, 5))
    log.write_text(text)

    result = adjudicate_phase_r(log, k=2, cadence=5, min_events=10)
    assert result["go"] is False
    assert result["decision"].startswith("NO-GO-R4")
    assert result["r1_reached"] and result["r2_artifact_match"] and result["r3_dose"]
    assert not result["r4_turnover"]


def test_artifact_mismatch_is_stop_not_escalate(tmp_path):
    # R2 failing is an implementation defect -> STOP, never the ladder.
    log = tmp_path / "run.log"
    text = ENABLED
    for i in range(10):
        text += _event(i * 5, 2 * (i + 1), [1, 2], _healthy_scores(9, 13))
    log.write_text(text)

    result = adjudicate_phase_r(log, k=2, cadence=5, min_events=10)
    assert result["go"] is False
    assert result["decision"].startswith("STOP")
    assert "R2" in result["decision"]


def test_real_smoke_log_gos_at_its_own_floor():
    smoke = (ROOT / "runs" / "v32_redo_bottom_k_2026-08-28" / "smoke"
              / "smoke_stdout.log")
    if not smoke.is_file():
        import pytest
        pytest.skip("smoke receipt not present in this checkout")
    result = adjudicate_phase_r(smoke, k=2, cadence=5, min_events=3)
    assert result["go"] is True
    assert result["recovery_curve"]
