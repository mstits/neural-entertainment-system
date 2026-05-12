"""Per-generation metrics sink.

Owns the three persistence surfaces the trainer uses to publish
per-gen scalars:
  * `metrics.jsonl` on disk (one JSON line per generation, the
    canonical source of truth for post-run analysis)
  * the in-process queue feeding the GUI dashboard (if any)
  * a TensorBoard writer (lazy — `from torch.utils.tensorboard` is
    only imported on first `emit()` so headless / test runs don't
    pay the import cost)

Extracted from Trainer so the file IO and the TB lazy-init logic
can be exercised in isolation; the trainer no longer carries
`_tb_writer` / `_tb_enabled` on `self`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional


log = logging.getLogger(__name__)


class MetricsSink:
    """JSONL + queue + optional TensorBoard fan-out for trainer metrics.

    Construct once per training run. Call `truncate()` at the top of
    a fresh run, `emit(**scalars)` each generation, and `close()` at
    shutdown to flush the TB writer.
    """

    def __init__(
        self,
        metrics_path: Path,
        tb_log_dir: Path,
        queue: Any = None,
        tb_enabled: bool = True,
    ) -> None:
        self.metrics_path = Path(metrics_path)
        self.tb_log_dir = Path(tb_log_dir)
        self._queue = queue
        self._tb_enabled = bool(tb_enabled)
        # Built lazily on first emit() so headless / test runs that
        # never write metrics don't import torch.utils.tensorboard.
        self._tb_writer: Optional[Any] = None

    def truncate(self) -> None:
        """Empty the metrics file (called at the start of a fresh run)."""
        self.metrics_path.write_text("")

    def close(self) -> None:
        """Flush + close the TB writer. Idempotent.

        Without this, the last few generations of scalars buffered in
        the writer never hit disk and the TB UI shows a truncated tail.
        """
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None

    def emit(self, **metrics: Any) -> None:
        """Append one generation's scalars to JSONL, GUI queue, and TB."""
        metrics["timestamp"] = time.time()
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._queue is not None:
            try:
                self._queue.put_nowait(metrics)
            except Exception:
                # Queue full — drop this update rather than blocking the
                # training thread.
                pass

        if not self._tb_enabled:
            return
        if self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir))
            except Exception as exc:
                log.debug("TensorBoard unavailable, disabling: %s", exc)
                self._tb_enabled = False
                return
        gen = metrics.get("generation", 0)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ("generation", "timestamp"):
                try:
                    self._tb_writer.add_scalar(k, float(v), gen)
                except Exception:
                    pass
