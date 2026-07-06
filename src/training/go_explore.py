"""Go-Explore cell archive — first-return-then-explore for NES games.

The exploration wall this addresses: the current stack reaches later
levels only via per-level hand-authored dense reward ladders compiled
into `nes_core::rewards` (~1 hr of human labor per level, ceiling at SMB
world 1). Go-Explore (Ecoffet et al. 2021, "First return, then explore",
Nature) replaces that manual labor with a general mechanism:

  1. Discretize state into *cells* (a coarse abstraction — e.g. level +
     bucketed position, or a downsampled RAM signature).
  2. Keep an *archive*: for each visited cell, remember the best-known
     way to get there (a saved emulator state + the score/steps it took).
  3. To explore: pick a promising cell, RETURN to it by restoring its
     saved state, THEN explore from there (random / policy actions).
  4. Any new cell discovered, or any better path to a known cell, updates
     the archive. Repeat.

This repo already owns the two hard prerequisites — deterministic
`pool.save_worker_state` / `load_worker_state` and a disk state cache —
so the missing piece is exactly this archive + return-selection policy.
It generalizes across game types (SMB, Contra, Zelda, Tetris, Bubble
Bobble): only the `cell_fn` — the map from a RAM snapshot to a cell key —
is game-specific, and it is injected, not hard-coded, so the archive
itself stays game-agnostic (same design stance as `curriculum.py`).

This module is the archive + selection core, deliberately decoupled from
the trainer so it is unit-testable in isolation. The trainer integration
(swap the SMB-hardcoded warm-start block in `_run_vanilla_ppo` for
archive-driven returns) is a separate, thin step.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, Optional

# A cell key is any hashable produced from a RAM snapshot. Games inject
# their own abstraction (see cell function helpers below).
CellKey = Hashable
CellFn = Callable[[bytes], CellKey]


@dataclass
class Cell:
    """One archive entry: the best-known way to reach a discretized state."""

    key: CellKey
    # Opaque emulator state blob (from pool.save_worker_state) that lands
    # the env in this cell. `None` is allowed for the synthetic start cell.
    state: Optional[bytes]
    # Domain score of the best trajectory that reached this cell. Higher
    # is better; ties broken by fewer steps.
    best_score: float
    # Steps taken to reach the cell on the best trajectory.
    best_steps: int
    # How many distinct times this cell has been reached (novelty signal).
    visits: int = 1
    # How many times this cell has been CHOSEN as a return target. The
    # selection weight decays with this so exploration spreads out.
    times_chosen: int = 0
    # Whether this cell has ever been chosen and explored from. Fresh
    # cells (frontier) are favored until explored at least once.
    explored: bool = False


class GoExploreArchive:
    """A cell archive with first-return-then-explore selection.

    Game-agnostic: the only game-specific input is `cell_fn`, the map from
    a 2 KB RAM snapshot to a cell key. Everything else — insertion,
    domination, return-target selection, persistence — is generic.
    """

    def __init__(
        self,
        cell_fn: CellFn,
        *,
        seed: int = 0,
    ) -> None:
        self.cell_fn = cell_fn
        self._cells: dict[CellKey, Cell] = {}
        self._rng = random.Random(seed)
        # Monotone counters for reporting / convergence checks.
        self.total_records = 0
        self.total_new_cells = 0
        self.total_improvements = 0

    # ---- construction / mutation ------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> dict[CellKey, Cell]:
        return self._cells

    def record(
        self,
        ram: bytes,
        state: Optional[bytes],
        score: float,
        steps: int,
    ) -> bool:
        """Observe a reached state. Returns True if this created a new
        cell or improved (dominated) an existing one — i.e. the archive
        learned something. `state` is the blob that reproduces this exact
        position (from pool.save_worker_state)."""
        key = self.cell_fn(ram)
        self.total_records += 1
        existing = self._cells.get(key)
        if existing is None:
            self._cells[key] = Cell(
                key=key, state=state, best_score=score, best_steps=steps
            )
            self.total_new_cells += 1
            return True

        existing.visits += 1
        # Domination: a strictly higher score, or the same score reached
        # in fewer steps, replaces the stored trajectory. This is the
        # Go-Explore "keep the best cell representative" rule and is what
        # lets the archive ratchet toward better play without any
        # hand-authored reward shaping.
        if score > existing.best_score + 1e-9 or (
            abs(score - existing.best_score) <= 1e-9 and steps < existing.best_steps
        ):
            existing.best_score = score
            existing.best_steps = steps
            existing.state = state
            self.total_improvements += 1
            return True
        return False

    # ---- selection --------------------------------------------------

    def _selection_weight(self, cell: Cell) -> float:
        """Higher = more likely to be chosen as a return target.

        Go-Explore biases toward cells that are under-explored. We use the
        paper's count-based form: weight decays with how often the cell
        has been chosen, with a bonus for never-yet-explored (frontier)
        cells so newly discovered territory is expanded promptly.
        """
        base = 1.0 / math.sqrt(cell.times_chosen + 1)
        frontier_bonus = 2.0 if not cell.explored else 1.0
        return base * frontier_bonus

    def select_return_cell(self) -> Optional[Cell]:
        """Pick a cell to return to and explore from, weighted toward the
        frontier. Marks it chosen/explored. Returns None on an empty
        archive."""
        if not self._cells:
            return None
        cells = list(self._cells.values())
        weights = [self._selection_weight(c) for c in cells]
        chosen = self._rng.choices(cells, weights=weights, k=1)[0]
        chosen.times_chosen += 1
        chosen.explored = True
        return chosen

    def select_return_states(self, n: int) -> list[Optional[bytes]]:
        """Convenience for the whole-pool warm-start: pick `n` return
        targets (with replacement) and return their state blobs. The
        trainer restores each env from one of these instead of the single
        SMB-hardcoded stage anchor."""
        out: list[Optional[bytes]] = []
        for _ in range(n):
            cell = self.select_return_cell()
            out.append(cell.state if cell is not None else None)
        return out

    # ---- reporting --------------------------------------------------

    def frontier_size(self) -> int:
        """Cells discovered but never yet explored from."""
        return sum(1 for c in self._cells.values() if not c.explored)

    def best_score(self) -> float:
        return max((c.best_score for c in self._cells.values()), default=0.0)

    def stats(self) -> dict:
        return {
            "cells": len(self._cells),
            "frontier": self.frontier_size(),
            "best_score": self.best_score(),
            "records": self.total_records,
            "new_cells": self.total_new_cells,
            "improvements": self.total_improvements,
        }

    # ---- persistence ------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the archive (cells + state blobs) to disk. Pickle is
        used because cell keys can be arbitrary hashables and state blobs
        are raw bytes; a sidecar JSON holds human-readable stats."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._cells, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path.with_suffix(".stats.json"), "w") as f:
            json.dump(self.stats(), f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(Path(path), "rb") as f:
            self._cells = pickle.load(f)


# ---------------------------------------------------------------------
# Cell-function helpers. `cell_fn` maps a 2 KB RAM snapshot to a cell key;
# the coarser the abstraction, the fewer cells and the more aggressive the
# generalization. These cover the common cases; games can supply their own.
# ---------------------------------------------------------------------


def ram_bytes_cell(addresses: list[int], bucket: int = 1) -> CellFn:
    """Cell key = a tuple of selected RAM bytes, optionally bucketed.

    The game-specific but declarative option: name the RAM addresses that
    define "where am I" (e.g. SMB: level byte $0760, x-page $006D, x-lo
    $0086 bucketed) and the archive discretizes on exactly those. Coarser
    `bucket` collapses nearby positions into one cell.
    """

    def _fn(ram: bytes) -> CellKey:
        return tuple((ram[a] // bucket) for a in addresses)

    return _fn


def ram_downsample_cell(stride: int = 64, bucket: int = 16) -> CellFn:
    """Fully game-agnostic cell key: coarsely downsampled RAM signature.

    Samples every `stride`-th byte and buckets its value. No per-game
    knowledge — the fallback for games without a known position layout.
    Trades precision (noisy bytes like timers/RNG inflate the cell count)
    for zero configuration; pair with a larger `bucket` to compensate.
    """

    def _fn(ram: bytes) -> CellKey:
        return tuple((ram[i] // bucket) for i in range(0, len(ram), stride))

    return _fn
