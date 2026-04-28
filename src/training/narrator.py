"""
Event-driven narrator — turns per-step reward-function activity into
human-readable captions for the stream/GUI.

Thin façade over `nes_core.Narrator`. Detection logic (delta scan,
rate limiting, first-ever dedup, combo-kill tracker) lives in
`nes_core/src/narrator.rs`; this module owns the caption templates
and the queue push (Rust doesn't need to know about either).

Call `Narrator.observe(worker_id, name, prev_breakdown, new_breakdown,
done, success)` once per step. Call `drain()` per generation to pull
a list of freshly-emitted events for broadcast.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import nes_core


log = logging.getLogger(__name__)


# Event kind strings as the Rust side spells them.
EVENT_DUNGEON_ENTER = "dungeon_enter"
EVENT_NEW_ITEM = "new_item"
EVENT_HEART_CONTAINER = "heart_container"
EVENT_TRIFORCE = "triforce"
EVENT_KEY = "key"
EVENT_MAP_COMPASS = "map_compass"
EVENT_DEATH = "death"
EVENT_SUCCESS = "success"
EVENT_COMBO_KILL = "combo_kill"


_CAPTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    EVENT_DUNGEON_ENTER: (
        "{name} ducks into a dungeon",
        "{name} enters the labyrinth",
        "{name} found a dungeon mouth",
    ),
    EVENT_NEW_ITEM: (
        "{name} grabbed a new item!",
        "{name} just picked something up",
        "{name} pockets a new artifact",
    ),
    EVENT_HEART_CONTAINER: (
        "{name} got a heart container!",
        "{name} heals up — heart container found",
    ),
    EVENT_TRIFORCE: (
        "{name} SECURED A TRIFORCE PIECE!",
        "{name} just claimed a triforce piece!",
    ),
    EVENT_KEY: (
        "{name} pockets a key",
        "{name} unlocks something",
    ),
    EVENT_MAP_COMPASS: (
        "{name} found the map / compass",
        "{name} knows the dungeon layout now",
    ),
    EVENT_DEATH: (
        "{name} went down fighting",
        "RIP {name}",
        "{name} fell in battle",
    ),
    EVENT_SUCCESS: (
        "{name} CLEARS the stage!",
        "{name} just won the level",
    ),
    EVENT_COMBO_KILL: (
        "{name} is on a killing spree!",
        "{name} chained a combo",
    ),
}


_FIRST_EVER_PREFIX = "FIRST EVER — "


@dataclass(frozen=True)
class NarratorEvent:
    worker_id: int
    genome_name: str
    kind: str
    caption: str
    first_ever: bool
    timestamp: float


class Narrator:
    """Stateful per-trainer-run event detector backed by `nes_core.Narrator`."""

    def __init__(
        self,
        min_event_gap_s: float = 1.5,
        max_caption_queue: int = 256,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._inner = nes_core.Narrator(float(min_event_gap_s))
        self._events: deque[NarratorEvent] = deque(maxlen=max_caption_queue)
        self._rng = random.Random(rng_seed)

    def observe(
        self,
        worker_id: int,
        genome_name: str,
        prev_breakdown: dict,
        new_breakdown: dict,
        done: bool = False,
        success: bool = False,
    ) -> None:
        """Diff reward breakdowns and emit events for each novel signal."""
        prev_pairs = [(str(k), float(v or 0.0)) for k, v in prev_breakdown.items()]
        new_pairs = [(str(k), float(v or 0.0)) for k, v in new_breakdown.items()]
        raw_events = self._inner.observe(
            int(worker_id), time.monotonic(), prev_pairs, new_pairs, bool(done), bool(success)
        )
        for ev in raw_events:
            kind = ev["kind"]
            templates = _CAPTION_TEMPLATES.get(kind)
            if not templates:
                continue
            caption = self._rng.choice(templates).format(name=genome_name)
            if ev["first_ever"]:
                caption = _FIRST_EVER_PREFIX + caption
            self._events.append(NarratorEvent(
                worker_id=int(ev["worker_id"]),
                genome_name=genome_name,
                kind=kind,
                caption=caption,
                first_ever=bool(ev["first_ever"]),
                timestamp=float(ev["timestamp"]),
            ))

    def drain(self) -> list[NarratorEvent]:
        """Pop every pending event, oldest first."""
        out: list[NarratorEvent] = []
        while self._events:
            out.append(self._events.popleft())
        return out


def push_events_to_queue(
    events: Iterable[NarratorEvent],
    queue,
) -> None:
    """Push drained events onto a thread-safe Queue non-blockingly. Full queue
    drops events rather than stalling the trainer — captions are
    ephemeral and the GUI tolerates loss."""
    if queue is None:
        return
    for ev in events:
        try:
            queue.put_nowait({
                "worker_id": ev.worker_id,
                "genome_name": ev.genome_name,
                "kind": ev.kind,
                "caption": ev.caption,
                "first_ever": ev.first_ever,
                "timestamp": ev.timestamp,
            })
        except Exception:
            break
