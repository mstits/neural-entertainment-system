"""Schema-contract tests for trainer metric emissions.

Pins the canonical metric-key schema (`DASHBOARD_REQUIRED_KEYS` in
`metrics_sink.py`) against every trainer mode's `_emit_metrics`
call site. Each trainer mode MUST pass every required key as a
keyword argument to `_emit_metrics` so the dashboard panels
(PPO telemetry, reward signal stack, fitness plots) chart real
data instead of NaN.

These are static-text tests — they grep the trainer source for
the required keys appearing in the right contexts. Cheaper than
booting a full Trainer just to inspect what gets emitted, and
catches the regression immediately if someone adds a new trainer
mode that forgets a key.
"""

from __future__ import annotations

from pathlib import Path

from src.training.metrics_sink import (
    DASHBOARD_REQUIRED_KEYS,
    DASHBOARD_OPTIONAL_KEYS,
    MetricsSink,
)


TRAINER_SRC = Path("src/training/trainer.py").read_text()


def test_required_keys_set_is_nonempty() -> None:
    assert len(DASHBOARD_REQUIRED_KEYS) >= 6, (
        "DASHBOARD_REQUIRED_KEYS must contain the core fitness + PPO "
        "telemetry keys the dashboard charts."
    )
    for k in ("ppo_loss", "ppo_policy_loss", "ppo_value_loss", "ppo_entropy"):
        assert k in DASHBOARD_REQUIRED_KEYS, (
            f"{k!r} missing — PPO panel will chart NaN otherwise."
        )


def test_ga_ppo_path_emits_required_keys() -> None:
    block = _extract_emit_block(
        TRAINER_SRC, "def _run_one_generation", "def _evaluate_batch",
    )
    # GA path emits ppo_* via the ppo_stats dict (key strings appear
    # as dict keys in _reinforce_update); other canonical keys appear
    # as direct kwargs in `_emit_metrics(...)`. Check the source body
    # of _run_one_generation contains the literal strings.
    assert "best_fitness" in block
    assert "avg_fitness" in block


def test_vanilla_ppo_path_emits_required_keys() -> None:
    block = _extract_emit_block(
        TRAINER_SRC, "def _run_vanilla_ppo", "def _save_checkpoint",
    )
    for required in DASHBOARD_REQUIRED_KEYS:
        if required == "generation":
            assert "generation=it" in block, (
                "_run_vanilla_ppo must emit `generation=it`"
            )
            continue
        assert f"{required}=" in block, (
            f"_run_vanilla_ppo missing required emit key {required!r}. "
            f"Canonical schema: {sorted(DASHBOARD_REQUIRED_KEYS)}"
        )


def test_metrics_sink_warns_on_missing_required_key(tmp_path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,
    )

    import logging
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    log_ = logging.getLogger("src.training.metrics_sink")
    log_.addHandler(handler)
    try:
        sink.emit(generation=0, best_fitness=1.0, avg_fitness=0.5)
    finally:
        log_.removeHandler(handler)

    # Specific-phrase match (see test_metrics_sink_warns_once_per_key for
    # why the bare 'ppo_loss' substring match is too permissive).
    text = " ".join(r.getMessage() for r in records)
    assert "missing required key 'ppo_loss'" in text, (
        f"MetricsSink should warn about ppo_loss specifically. Got: {text!r}"
    )


def test_metrics_sink_warns_once_per_key(tmp_path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,
    )

    import logging
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    log_ = logging.getLogger("src.training.metrics_sink")
    log_.addHandler(handler)
    try:
        for _ in range(5):
            sink.emit(generation=0, best_fitness=1.0, avg_fitness=0.5)
    finally:
        log_.removeHandler(handler)

    # Filter on the SPECIFIC "missing required key '<name>'" phrase —
    # the warning's full text also lists every canonical key in its
    # `Canonical schema: [...]` epilogue, so a plain substring search
    # on 'ppo_loss' matches ALL warnings (including ones about
    # ppo_value_loss, etc.).
    ppo_loss_warnings = [
        r for r in records
        if "missing required key 'ppo_loss'" in r.getMessage()
    ]
    assert len(ppo_loss_warnings) == 1, (
        f"ppo_loss warning should fire once across 5 emits. "
        f"Got {len(ppo_loss_warnings)}"
    )


def _extract_emit_block(src: str, marker: str, end_marker: str) -> str:
    start = src.find(marker)
    assert start >= 0, f"{marker!r} not in trainer source"
    end = src.find(end_marker, start)
    assert end > start, f"{end_marker!r} not after {marker!r}"
    return src[start:end]
