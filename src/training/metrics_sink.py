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

A fourth, optional surface: `episodes.jsonl`, written by
`emit_episode()`. `metrics.jsonl`'s unit of work is a GENERATION —
an aggregate over every worker's episodes that completed within it
(mean/max return, mean length, ...). That is the right shape for the
per-gen dashboard, but it means per-episode identity (which worker,
which episode, its own return/length) exists only for the moment the
trainer's rollout loop holds it before folding it into the
generation's aggregate — there is no way to later ask "which worker
produced the outlier." `emit_episode()` is the sibling that keeps one
row per completed episode instead of collapsing it into an average,
in its own file so the two row shapes never mix in one reader's path.
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
    # Vanilla-PPO-path trust-region + critic-fit diagnostics
    # (V29_STABILITY_2026-08-25.md F0 — "the five missing scalars").
    # Pure observations of the K-epoch update; not on any training path
    # of the GA/PPO-hybrid or Dreamer modes, so OPTIONAL rather than
    # REQUIRED avoids a spurious once-per-session warning there.
    "vanilla_ppo_clip_fraction", "vanilla_ppo_approx_kl",
    "vanilla_ppo_grad_norm", "vanilla_ppo_adv_mean", "vanilla_ppo_adv_std",
    "vanilla_ppo_explained_variance",
})


class MetricsSink:
    """JSONL + queue + optional TensorBoard fan-out for trainer metrics.

    Construct once per training run. Call `truncate()` at the top of
    a fresh run (pass `resume=True` instead when continuing from a
    checkpoint, so the canonical JSONL isn't wiped out from under a
    continuous generation counter), `emit(**scalars)` each generation,
    and `close()` at shutdown to flush the TB writer. `emit_episode(
    **fields)` is the per-episode sibling of `emit()` — see the module
    docstring.
    """

    def __init__(
        self,
        metrics_path: Path,
        tb_log_dir: Path,
        queue: Any = None,
        tb_enabled: bool = True,
    ) -> None:
        self.metrics_path = Path(metrics_path)
        # Sidecar for emit_episode() — same checkpoint_dir as
        # metrics.jsonl (its parent), never the same file: the two
        # row shapes (per-generation aggregate vs. per-episode raw)
        # would break every existing reader of either if mixed.
        self.episodes_path = self.metrics_path.parent / "episodes.jsonl"
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

    def truncate(self, *, resume: bool = False) -> None:
        """Empty the metrics file (called at the start of a fresh run).

        Also empties the episodes.jsonl sidecar so a fresh run never
        appends onto a previous run's per-episode rows — harmless when
        `emit_episode()` is never called (the file just stays empty).

        `resume=True` is a no-op: a resumed run continues the same
        generation counter, so wiping here would silently sever the
        canonical JSONL from every generation before the resume even
        though training is continuous end-to-end. Callers must pass
        `resume=True` explicitly when continuing from a checkpoint;
        the default (`False`) preserves today's unconditional-wipe
        behavior for fresh-start callers.
        """
        if resume:
            return
        self.metrics_path.write_text("")
        self.episodes_path.write_text("")

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
        try:
            line = json.dumps(metrics)
        except Exception as exc:
            # Mirrors the queue/TB fan-outs below: one non-serializable
            # value in a single generation's dict must not abort the
            # whole emit() call (the queue + TB updates still need to
            # happen), but unlike those the JSONL file is the canonical
            # record, so the drop is logged rather than silent.
            log.warning(
                "MetricsSink: dropped one generation's JSONL line — "
                "metrics not JSON-serializable: %s", exc,
            )
        else:
            with open(self.metrics_path, "a") as f:
                f.write(line + "\n")
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

    def emit_episode(self, **fields: Any) -> None:
        """Append one row per COMPLETED EPISODE to `episodes.jsonl`.

        The per-episode sibling of `emit()` — see the module docstring
        for why this file exists separately from `metrics.jsonl`.
        Callers pass only fields already computed at the harvest site
        (generation, worker index, that episode's own return/length,
        ...); this method does not infer, aggregate, or derive
        anything — it stamps `timestamp` / `schema_version` the same
        way `emit()` does and appends.

        Deliberately NOT fanned out to the GUI queue or the TB writer:
        no dashboard panel reads per-episode rows, and the row shape
        (one line per episode, arbitrarily more or fewer than one per
        generation) doesn't match what either sink expects. Disk IO
        is this method's only side effect. Callers gate calls to this
        method behind their own opt-in flag — it is meant to sit right
        next to a hot rollout loop, so it does no schema validation or
        other work beyond the write itself.
        """
        fields["timestamp"] = time.time()
        # Own schema marker (starts at 1, like `emit()`'s) — this file
        # has a different, independent row shape from metrics.jsonl's,
        # so the two must not be conflated by a reader keying off
        # `schema_version` alone without also checking which file it
        # read the line from.
        fields["schema_version"] = 1
        with open(self.episodes_path, "a") as f:
            f.write(json.dumps(fields) + "\n")
