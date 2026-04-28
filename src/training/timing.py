"""
Gen-scoped timing telemetry.

`GenTimer` is a thin accumulator you wrap critical sections with. At
the end of a generation the trainer calls `snapshot()` to get
millisecond totals for each section, prefixes them `timing_*`, and
appends them to the metrics JSONL. Low-friction: one
`with timer.section("emulation"):` per hot path, no extra config.

Why: without per-section telemetry, every perf regression is an
overnight mystery. "Gen got slower" rarely points at the cause. With
per-section totals logged every gen, `jq '.timing_emulation_ms'` on
the metrics log tells you exactly which phase blew up.

Overhead: each `section()` enter/exit is a `time.perf_counter()` call
(~50 ns on modern CPUs) plus a dict update. For 16 workers × 1000
steps × 5 sections = 80 k enter/exits per gen, at ~2 ms total — well
under the noise floor of the training loop.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class GenTimer:
    """Per-generation timing accumulator.

    Use as:

        timer = GenTimer()
        for step in range(...):
            with timer.section("inference"):
                actions = net(obs)
            with timer.section("emulation_wait"):
                results = pool.step_all(actions)
            with timer.section("reward_compute"):
                ...
        totals = timer.snapshot()  # {"inference": ms, ...}
        timer.reset()               # zero-out between gens
    """

    def __init__(self) -> None:
        self._accum_ns: dict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self._accum_ns[name] += time.perf_counter_ns() - start

    def add(self, name: str, ns: int) -> None:
        """Manually record a duration (for sections that can't be
        wrapped with a context manager, e.g. producer-side emit hooks
        the trainer doesn't own)."""
        self._accum_ns[name] += int(ns)

    def snapshot(self) -> dict[str, float]:
        """Returns per-section totals in milliseconds. Safe to call
        any number of times; totals keep accumulating until
        `reset()`."""
        return {
            f"timing_{name}_ms": ns / 1e6
            for name, ns in self._accum_ns.items()
        }

    def reset(self) -> None:
        self._accum_ns.clear()
