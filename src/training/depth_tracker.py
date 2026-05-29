"""
Depth tracking for auto-curriculum.

Thin Python façade over `nes_core.DepthTracker`. Per-game RAM readers
(zelda/mario/generic) and lexicographic max-depth logic live in
`nes_core/src/depth_tracker.rs`; this module handles the JSONL memo
file I/O (Rust doesn't need to know about file paths).

What depth-tracking gives us:

* **New-record narration.** When a worker pushes past the previous
  all-time depth, the tracker emits a record dict; the trainer
  forwards the caption to the Narrator queue.
* **Stream bragging rights.** The memo file at
  `<checkpoint_dir>/depth_memo.jsonl` is a per-run log of every new
  record.
* **Foundation for auto-promotion.** Deepest-ever (worker, step) is
  the candidate start-state for the next curriculum stage.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import nes_core


log = logging.getLogger(__name__)


class DepthTracker:
    """Stateful max-depth tracker — delegates to `nes_core.DepthTracker`.

    `observe(ram, worker_id, genome_name, generation)` returns a record
    dict on a strictly-greater depth key; `None` otherwise.
    """

    def __init__(self, game: str, memo_path: Optional[Path] = None) -> None:
        self.game = game
        self._inner = nes_core.DepthTracker(game)
        # Bounded ring — the durable record is the memo JSONL
        # (_append_memo); this in-memory copy only backs the optional
        # dump(). An unbounded list grew for the whole run with no
        # eviction (dump() is rarely/never called), so cap it.
        self._history: deque = deque(maxlen=2000)
        self._memo_path = memo_path
        self._disabled_memo: bool = False

    @property
    def best(self) -> Optional[tuple[int, int, int]]:
        b = self._inner.best
        return tuple(b) if b is not None else None

    def observe(
        self,
        ram: bytes,
        worker_id: int,
        genome_name: str,
        generation: int,
    ) -> Optional[dict]:
        rec = self._inner.observe(bytes(ram), int(worker_id), str(genome_name), int(generation))
        if rec is None:
            return None
        # Normalise the timestamp on the Python side — Rust doesn't
        # need wall-clock time; the memo reader does.
        import time as _time
        rec = dict(rec)
        rec["key"] = list(rec["key"])
        rec["timestamp"] = _time.time()
        self._history.append(rec)
        self._append_memo(rec)
        return rec

    def _append_memo(self, record: dict) -> None:
        if self._memo_path is None or self._disabled_memo:
            return
        try:
            self._memo_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._memo_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            log.debug("depth memo write failed, disabling: %s", exc)
            self._disabled_memo = True

    def dump(self, path: Path) -> None:
        """Overwrite `path` with the full history as JSONL."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for record in self._history:
                    f.write(json.dumps(record) + "\n")
        except Exception as exc:
            log.warning("depth memo dump failed: %s", exc)
