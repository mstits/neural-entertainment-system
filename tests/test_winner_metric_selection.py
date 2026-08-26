"""P1-1 — winner-retention metric selection (non-ladder/non-consolidate).

The else branch of the winner-retention block (trainer.py ~8836) keyed
`save_winner` on the RAW in-training `vppo_success_rate` for every mode
without ladder/consolidate/backward metrics. That rate is inflated
relative to the honest cold probe (PR-MDP incident: winners/ retained a
"best" checkpoint at in-training 0.25 while the honest cold probe read
0.0 at all nine sampled checkpoints). The selection logic now lives in
`_select_winner_metric`, which prefers the honest cold-probe rate when
one is in scope (PLR mode populates `last_cold_metrics`) and emits a
loud warning whenever it must fall back to the inflated in-training
rate. These tests pin that contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.training.trainer import _select_winner_metric

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_SRC = (ROOT / "src" / "training" / "trainer.py").read_text()

_INFLATED_MSG = (
    "WINNER METRIC = in-training clear_rate (INFLATED; not "
    "honest-probe verified)"
)


def test_honest_cold_rate_wins_over_in_training_rate() -> None:
    """An honest cold-probe rate in scope MUST key the winner, even when the
    in-training rate is higher (the exact PR-MDP inversion: 0.25 in-training
    vs 0.0 honest)."""
    val, name = _select_winner_metric(0.25, cold_rate=0.0, bwd_snapshot=None)
    assert val == 0.0
    assert name == "cold_seq_clear_rate"

    val, name = _select_winner_metric(0.25, cold_rate=0.63, bwd_snapshot=None)
    assert val == 0.63
    assert name == "cold_seq_clear_rate"


def test_honest_selection_does_not_log_inflated_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        _select_winner_metric(0.25, cold_rate=0.4, bwd_snapshot=None)
    assert _INFLATED_MSG not in caplog.text


def test_missing_cold_rate_falls_back_and_logs_loudly(caplog) -> None:
    """No honest probe in scope (PR-MDP / plain vanilla runs never probe):
    the raw rate is still used — save_winner must keep working — but the
    fallback is flagged unmistakably in the log."""
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        val, name = _select_winner_metric(
            0.25, cold_rate=None, bwd_snapshot=None
        )
    assert val == 0.25
    assert name == "clear_rate"
    assert _INFLATED_MSG in caplog.text


def test_sentinel_cold_rate_is_not_treated_as_honest(caplog) -> None:
    """`last_cold_metrics` carries -1.0 sentinels when a probe scored no
    episodes; a sentinel must not silently become the winner metric."""
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        val, name = _select_winner_metric(
            0.25, cold_rate=-1.0, bwd_snapshot=None
        )
    assert val == 0.25
    assert name == "clear_rate"
    assert _INFLATED_MSG in caplog.text


def test_backward_mode_at_entrance_keys_on_trailing_rate(caplog) -> None:
    """Backward mode keeps its deliberate re-keying: at the entrance, the
    trailing entrance rate is the metric (no inflated-fallback warning)."""
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        val, name = _select_winner_metric(
            0.9, cold_rate=None,
            bwd_snapshot={"at_entrance": True, "rate": 0.12},
        )
    assert val == 0.12
    assert name == "entrance_trailing_rate"
    assert _INFLATED_MSG not in caplog.text


def test_backward_mode_pre_entrance_suppresses_winner(caplog) -> None:
    """Pre-entrance nets are never the deliverable: no metric, no save, and
    no fallback warning either."""
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        val, _name = _select_winner_metric(
            0.9, cold_rate=None,
            bwd_snapshot={"at_entrance": False, "rate": 0.12},
        )
    assert val is None
    assert _INFLATED_MSG not in caplog.text


def test_retention_else_branch_routes_through_selector_in_source() -> None:
    """Anchor the wiring: the retention block's else branch must call the
    selector with the honest cold rate from `last_cold_metrics`, not read
    `vppo_success_rate` bare."""
    assert "_wm_val, _wm_name = _select_winner_metric(" in _TRAINER_SRC
    assert "cold_rate=last_cold_metrics.get(" in _TRAINER_SRC


def test_plr_per_level_probe_failure_uses_sentinel_not_zero() -> None:
    """PLR cold-probe winner-selection block (~trainer.py:8474-8494): a
    failed `cold_probe.probe()` call for one level (e.g. a transiently
    unreadable entry-state file) returns `cold_seq_clear_rate=None`. That
    must not collapse into a fabricated 0.0 — indistinguishable from a
    genuine 0% clear rate, and able to force `_weakest` to 0.0 for a round
    that never actually measured the level. Every other cold-rate site in
    this file (see the ladder-path sibling, and
    `test_sentinel_cold_rate_is_not_treated_as_honest` above) uses the
    -1.0 failure sentinel instead; the PLR per-level/holdout loops must
    match."""
    assert (
        '_per_level[_lvl] = float(_c.get("cold_seq_clear_rate") or 0.0)'
        not in _TRAINER_SRC
    )
    assert (
        '_hold[_lvl] = float(_c.get("cold_seq_clear_rate") or 0.0)'
        not in _TRAINER_SRC
    )
    assert (
        "_per_level[_lvl] = (\n"
        "                        float(_lvl_rate) if _lvl_rate is not None else -1.0\n"
        "                    )"
    ) in _TRAINER_SRC
    assert (
        "_hold[_lvl] = (\n"
        "                        float(_hold_rate) if _hold_rate is not None else -1.0\n"
        "                    )"
    ) in _TRAINER_SRC
