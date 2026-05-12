"""Tests for the JSONL + queue + TB metrics fan-out sink."""

from __future__ import annotations

import json
import queue as _queue
from pathlib import Path

from src.training.metrics_sink import MetricsSink


def test_emit_appends_jsonl_line_with_timestamp(tmp_path: Path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,  # skip torch.utils.tensorboard import
    )
    sink.emit(generation=0, fitness=1.5)
    sink.emit(generation=1, fitness=2.0)

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["generation"] == 0
    assert row0["fitness"] == 1.5
    assert "timestamp" in row0
    assert row1["generation"] == 1


def test_truncate_clears_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    p.write_text("stale\nstale\n")
    sink = MetricsSink(metrics_path=p, tb_log_dir=tmp_path / "tb", tb_enabled=False)
    sink.truncate()
    assert p.read_text() == ""


def test_emit_pushes_to_queue(tmp_path: Path) -> None:
    q: _queue.Queue = _queue.Queue(maxsize=4)
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        queue=q,
        tb_enabled=False,
    )
    sink.emit(generation=0, x=42)
    row = q.get_nowait()
    assert row["generation"] == 0
    assert row["x"] == 42


def test_emit_drops_silently_when_queue_full(tmp_path: Path) -> None:
    """Queue full must not block the training thread — drop instead."""
    q: _queue.Queue = _queue.Queue(maxsize=1)
    q.put_nowait({"prefill": True})  # fill the queue
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        queue=q,
        tb_enabled=False,
    )
    sink.emit(generation=0)  # would raise queue.Full without the guard
    # JSONL line still landed even though queue dropped it.
    assert (tmp_path / "metrics.jsonl").read_text().strip() != ""


def test_close_is_idempotent(tmp_path: Path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,
    )
    sink.close()
    sink.close()  # must not raise
