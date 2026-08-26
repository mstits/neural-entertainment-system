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


# The canonical metric schema every trainer mode must emit. Any panel
# in `src/gui/training_dashboard.py` reads from this set; a mode that
# omits a key silently breaks the panel that reads it. We enforce the
# contract via a once-per-session warning at emit time so a future
# trainer mode (DreamerV3, RecurrentTilePolicy, RLHF, etc.) can't ship
# with broken dashboard panels.
#
# REQUIRED — every emission must include these. Missing → warn.
# OPTIONAL — mode-specific extras. Dashboard tolerates missing
# (charts NaN); no warning emitted.
DASHBOARD_REQUIRED_KEYS: frozenset[str] = frozenset({
    "generation",
    "best_fitness",
    "avg_fitness",
    "ppo_loss",
    "ppo_policy_loss",
    "ppo_value_loss",
    "ppo_entropy",
})

# Optional keys the dashboard charts when present. Documented here so
# new trainer modes know what data their mode-specific panels can
# consume. Not enforced.
DASHBOARD_OPTIONAL_KEYS: frozenset[str] = frozenset({
    "stage", "success_rate", "episodes",
    "depth_scalar", "depth_leaf",
    "rnd_loss", "rnd_intrinsic_avg",
    "wm_total", "wm_recon", "wm_kl", "wm_reward", "wm_continue",
    # Count of policy-logit rows `_safe_sample_from_logits` substituted
    # a uniform distribution for (NaN/Inf — network divergence) this
    # generation. GA/PPO-hybrid path only; always 0 when quiet, never
    # null/missing.
    "nan_rows_this_gen",
})


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
        # Track which canonical keys we've already warned about so the
        # warning is once-per-session per key (not once-per-gen — that
        # would spam the log thousands of times).
        self._warned_missing: set[str] = set()

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
        """Append one generation's scalars to JSONL, GUI queue, and TB.

        Validates against `DASHBOARD_REQUIRED_KEYS` and warns once per
        missing key per session. Any panel in the dashboard reading a
        canonical key (PPO telemetry, reward signal stack, fitness
        plots) silently breaks if a trainer mode forgets to emit it;
        the warning ensures the regression surfaces immediately
        instead of "graph is empty, no idea why."
        """
        # Schema check (once-per-session per key — log warning if a
        # required canonical key is missing from this emission).
        for required in DASHBOARD_REQUIRED_KEYS:
            if required not in metrics and required not in self._warned_missing:
                self._warned_missing.add(required)
                log.warning(
                    "MetricsSink: emission missing required key %r "
                    "— a dashboard panel will chart NaN. Trainer mode "
                    "is likely missing it from its _emit_metrics call. "
                    "Canonical schema: %s",
                    required,
                    sorted(DASHBOARD_REQUIRED_KEYS),
                )
        metrics["timestamp"] = time.time()
        # Schema marker (start at 1) — lets a reader tell which field set
        # a line was written under across runs, since old lines have no
        # such marker at all and JSON lines from different eras of this
        # sink otherwise carry silently different key sets. `generation`
        # is a per-run counter, not this: it resets every run and says
        # nothing about which fields a given run's trainer mode emitted.
        metrics["schema_version"] = 1
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
            if (isinstance(v, (int, float))
                    and k not in ("generation", "timestamp", "schema_version")):
                try:
                    self._tb_writer.add_scalar(k, float(v), gen)
                except Exception:
                    pass
