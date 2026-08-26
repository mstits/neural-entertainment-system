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


# ---------------------------------------------------------------------------
# emit_episode() — the per-episode sibling of emit(). See the module
# docstring: metrics.jsonl is per-generation aggregates, episodes.jsonl is
# one row per completed episode, and the two must never share a file.
# ---------------------------------------------------------------------------
def test_emit_episode_appends_separate_jsonl_with_right_shape(tmp_path: Path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,
    )
    sink.emit_episode(
        generation=3, worker_id=1, episode_return=12.5, episode_length=44,
        final_x=680, env_done=False, reward_done=True,
    )
    sink.emit_episode(
        generation=3, worker_id=0, episode_return=8.0, episode_length=20,
        final_x=303, env_done=True, reward_done=False,
    )

    episodes_path = tmp_path / "episodes.jsonl"
    assert episodes_path.exists()
    lines = episodes_path.read_text().splitlines()
    assert len(lines) == 2

    row0 = json.loads(lines[0])
    assert row0["generation"] == 3
    assert row0["worker_id"] == 1
    assert row0["episode_return"] == 12.5
    assert row0["episode_length"] == 44
    assert row0["final_x"] == 680
    assert row0["env_done"] is False
    assert row0["reward_done"] is True
    assert "timestamp" in row0
    assert row0["schema_version"] == 1

    row1 = json.loads(lines[1])
    assert row1["worker_id"] == 0

    # metrics.jsonl must stay untouched — the two row shapes never mix.
    # emit_episode() alone never even creates it (only emit()/truncate() do).
    assert not (tmp_path / "metrics.jsonl").exists()


def test_emit_episode_does_not_touch_queue_or_tb(tmp_path: Path) -> None:
    q: _queue.Queue = _queue.Queue(maxsize=1)
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        queue=q,
        tb_enabled=False,
    )
    sink.emit_episode(generation=0, worker_id=0, episode_return=1.0, episode_length=1)
    assert q.empty(), "emit_episode() must not fan out to the GUI queue"


def test_truncate_also_clears_episodes_jsonl(tmp_path: Path) -> None:
    sink = MetricsSink(
        metrics_path=tmp_path / "metrics.jsonl",
        tb_log_dir=tmp_path / "tb",
        tb_enabled=False,
    )
    sink.emit_episode(generation=0, worker_id=0, episode_return=1.0, episode_length=1)
    assert (tmp_path / "episodes.jsonl").read_text() != ""
    sink.truncate()
    assert (tmp_path / "episodes.jsonl").read_text() == ""
