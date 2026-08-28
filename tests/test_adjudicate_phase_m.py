"""Tests for the Phase M adjudicator (V31_REDO_SURGICAL_2026-08-27.md §5.2)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.adjudicate_phase_m import adjudicate  # noqa: E402

ENABLED = "[redo] ENABLED tau=0.10 every_iters=1 scope=fc1,fc2 sample=4096"
ITER = (
    "[redo] iter {it}: dormant fc1 0/64 fc2 {n}/32 recycled {n} cum {cum} "
    "agree 0.9700 max_dlogit 0.050000"
)


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "phase_m.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def _healthy_60_iter_log(tmp_path: Path, *, n_per_fire: int = 2) -> Path:
    lines = [ENABLED]
    cum = 0
    for it in range(60):
        n = n_per_fire if it >= 16 else 0
        cum += n
        lines.append(ITER.format(it=it, n=n, cum=cum))
        if n:
            fc2 = [(it * n + k) % 32 for k in range(n)]
            lines.append(f"[redo] recycled unit indices: fc1=[] fc2={fc2}")
    return _write(tmp_path, lines)


def test_healthy_phase_m_log_is_go(tmp_path):
    log = _healthy_60_iter_log(tmp_path)
    result = adjudicate(log)
    assert result["verdict"] == "GO"
    assert result["M1_fires"]
    assert result["M4_ge_8_firing"]
    assert result["M5_healthy"]
    assert result["M3_not_drifting"]


def test_never_fires_is_nogo_on_m1_and_m4(tmp_path):
    lines = [ENABLED] + [
        ITER.format(it=it, n=0, cum=0) for it in range(60)
    ]
    log = _write(tmp_path, lines)
    result = adjudicate(log)
    assert result["verdict"] == "NO-GO"
    assert not result["M1_fires"]
    assert not result["M4_ge_8_firing"]


def test_overdose_trip_is_nogo_on_m2(tmp_path):
    log = _healthy_60_iter_log(tmp_path)
    with log.open("a") as f:
        f.write(
            "\n[redo] VOID-OVERDOSE: trailing-10-check median dose "
            "exceeded 0.25 at iter 40\n"
        )
    result = adjudicate(log)
    assert result["ceiling_tripped"]
    assert not result["M2_under_ceiling"]
    assert result["verdict"] == "NO-GO"


def test_climbing_dose_fails_m3():
    pass  # covered functionally below


def test_drifting_dose_fails_m3(tmp_path):
    lines = [ENABLED]
    cum = 0
    for it in range(60):
        # Dose climbs steadily: 1 unit at iter 26-30, 5 units at 56-60.
        n = 0
        if 26 <= it <= 30:
            n = 1
        elif 56 <= it <= 60:
            n = 5
        elif it >= 16:
            n = 1
        cum += n
        lines.append(ITER.format(it=it, n=n, cum=cum))
        if n:
            fc2 = [(it * n + k) % 32 for k in range(n)]
            lines.append(f"[redo] recycled unit indices: fc1=[] fc2={fc2}")
    log = _write(tmp_path, lines)
    result = adjudicate(log)
    assert result["M3_drift_units"] == 4.0
    assert not result["M3_not_drifting"]
    assert result["verdict"] == "NO-GO"


def test_single_index_lesion_fails_m5(tmp_path):
    lines = [ENABLED]
    cum = 0
    for it in range(60):
        n = 2 if it >= 16 else 0
        cum += n
        lines.append(ITER.format(it=it, n=n, cum=cum))
        if n:
            lines.append("[redo] recycled unit indices: fc1=[] fc2=[3, 7]")
    log = _write(tmp_path, lines)
    result = adjudicate(log)
    assert result["M5_distinct_fc2"] == 2
    assert not result["M5_healthy"]
    assert result["verdict"] == "NO-GO"


def test_equilibrium_law_corroboration_flag(tmp_path):
    """§1.3's registered prediction: tau=0.10 -> 7.6-8.0/32 units, i.e.
    inside [0.20, 0.30] of the trunk. Build a log whose tail settles at
    exactly 8/32 = 0.25 and confirm the corroboration flag fires.
    """
    log = _healthy_60_iter_log(tmp_path, n_per_fire=8)
    result = adjudicate(log)
    assert result["equilibrium_frac_tail10"] == 0.25
    assert result["eq_law_prediction_0.20_0.30_corroborated"]
