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
from typing import Callable, Hashable, Optional, Sequence, Union

# A cell key is any hashable produced from a RAM snapshot. Games inject
# their own abstraction (see cell function helpers below).
CellKey = Hashable
CellFn = Callable[[bytes], CellKey]
# Maps a cell key to its horizontal-neighbor keys (see
# `horizontal_neighbors`). Injected like `cell_fn`, so games with an
# unusual key layout can supply their own adjacency.
NeighborFn = Callable[[CellKey], tuple]

# Coefficient of the neighbor-aware location bonus in `_selection_weight`
# (Ecoffet et al. 2021, Nature: W_location = (2 - h) / 10 where h is the
# number of horizontal neighbors already present in the archive). Set to
# 0.0 to disable the bonus and recover the pure count-based weight.
W_LOCATION_COEF = 0.1

# Contract: a first task-success is NOT terminal for exploration. The
# archive's domination rule in `record()` (higher score wins; at equal
# score, FEWER steps win) means post-clear exploration keeps ratcheting
# elites toward shorter trajectories — Go-Explore's "keep improving after
# the first solution" behavior (Ecoffet et al. 2021). Nothing in this
# module early-outs on success; harvest drivers should gate their loops
# on `keep_exploring()` rather than on "found a clear".
EXPLORE_AFTER_FIRST_CLEAR = True

# Backward-algorithm start window, in FRAMES (Salimans & Chen 2018,
# arXiv:1812.03381: start states sampled from the trailing 160 frames
# behind the anchor, not from one frozen state).
START_WINDOW_FRAMES = 160


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
        neighbor_fn: Optional[NeighborFn] = None,
    ) -> None:
        self.cell_fn = cell_fn
        # Adjacency used by the W_location selection bonus. Defaults to
        # the `horizontal_neighbors` convention (last key component = x
        # bucket); pass an explicit callable for custom key layouts, or
        # zero W_LOCATION_COEF to disable the bonus entirely.
        self.neighbor_fn: NeighborFn = (
            neighbor_fn if neighbor_fn is not None else horizontal_neighbors
        )
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
        position (from pool.save_worker_state).

        Domination is (higher score, then FEWER steps at equal score), so
        a task-success is never terminal for the archive: post-clear
        records keep replacing elites with shorter trajectories (see
        EXPLORE_AFTER_FIRST_CLEAR / `keep_exploring`)."""
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

        On top of the count term, the paper's neighbor-aware location
        bonus (Ecoffet et al. 2021): W_location = (2 - h) / 10, where h is
        the number of the cell's horizontal neighbors (same area, x bucket
        +/- 1) already present in the archive. Cells at the edge of
        explored space — missing one or both neighbors — get an additive
        boost, so the frontier is expanded preferentially. The coefficient
        is the module constant W_LOCATION_COEF (zero it to disable).
        """
        base = 1.0 / math.sqrt(cell.times_chosen + 1)
        frontier_bonus = 2.0 if not cell.explored else 1.0
        weight = base * frontier_bonus
        if W_LOCATION_COEF > 0.0:
            neighbors = self.neighbor_fn(cell.key)
            if neighbors:
                h = sum(1 for k in neighbors if k in self._cells)
                weight += W_LOCATION_COEF * (len(neighbors) - h)
        return weight

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


def smb_gx_phase_cell(gx_bucket: int = 16, y_bucket: int = 32) -> CellFn:
    """SMB cell key: (area, y-band, enemy phase, gx bucket).

    The gx bucket is LAST so `horizontal_neighbors` sees the x component
    per its convention. The phase component is ((ram[0x0009] >> 2) & 7):
    bits 2-4 of the frame counter. At frame_skip 4 consecutive decisions
    land 4 frames apart, so bits 0-1 are constant within a lineage and a
    raw mod-8 key collapses to 2-4 residues; bits 2-4 are the granularity
    at which enemy spawn/walk cycles genuinely differ. Eight phase
    classes keeps arrival-context diversity in the archive — the same
    (area, y, gx) reached on a different enemy phase is a DIFFERENT
    cell, so the archive holds one return state per arrival context —
    without fragmenting the key space 64-way.
    """

    def _fn(ram: bytes) -> CellKey:
        gx = (ram[0x006D] << 8) | ram[0x0086]
        return (
            ram[0x0760],
            ram[0x00CE] // y_bucket,
            (ram[0x0009] >> 2) & 7,
            gx // gx_bucket,
        )

    return _fn


# ---------------------------------------------------------------------
# Pure helpers for the Nature-spec selection / robustification quartet.
# All are side-effect-free and unit-tested in tests/test_go_explore.py.
# ---------------------------------------------------------------------


def horizontal_neighbors(key: CellKey) -> tuple:
    """The two same-area cells one x-bucket left/right of `key`.

    Convention (matches `ram_bytes_cell` usage across the repo): the key
    is a tuple whose LAST component is the horizontal-position bucket and
    whose leading components identify the area (world/level/area bytes).
    The neighbors therefore share every leading component and differ by
    +/-1 in the last. Keys without that structure (non-tuple, empty, or a
    non-int last component) have no known adjacency and yield `()`, which
    disables the W_location bonus for them.
    """
    if (
        isinstance(key, tuple)
        and key
        and isinstance(key[-1], int)
        and not isinstance(key[-1], bool)
    ):
        return (key[:-1] + (key[-1] - 1,), key[:-1] + (key[-1] + 1,))
    return ()


def keep_exploring(
    clears_found: int,
    want_clears: int,
    *,
    explore_after_first_clear: bool = EXPLORE_AFTER_FIRST_CLEAR,
) -> bool:
    """Should a harvest driver keep sampling episodes?

    Go-Explore's exploration phase does not stop at the first solution
    (Ecoffet et al. 2021): further clears keep improving the archive
    because `record()`'s domination rule replaces elites with equal-
    progress, fewer-step trajectories — the demo-shortening that makes
    later backward robustification tractable. Drivers should therefore
    loop on this predicate instead of early-outing on the first clear:

        while keep_exploring(len(demos), want) and seen < cap: ...

    With `explore_after_first_clear=False` (legacy behavior), the first
    clear is treated as terminal.
    """
    if clears_found <= 0:
        return True
    if not explore_after_first_clear:
        return False
    return clears_found < want_clears


def start_window(
    archive_or_trajectory: Union["GoExploreArchive", dict, Sequence],
    anchor_index: int,
    window_frames: int = START_WINDOW_FRAMES,
    *,
    frames_per_step: int = 1,
) -> list:
    """Candidate start entries in the trailing window behind an anchor.

    Backward robustification a la Salimans & Chen (arXiv:1812.03381)
    starts each episode from a state sampled inside a 160-FRAME window
    ending at the anchor, not from one frozen state — the single-anchor
    weld arrives with one exact enemy/firebar phase and the clone
    overfits to it (the phase curse). Callers (robustify_level.py ladder
    rungs, trainer warm starts) should draw uniformly from the returned
    list each episode instead of always loading the anchor blob.

    Two input shapes:

    * A per-step TRAJECTORY (any sequence — state blobs, (frame, blob)
      pairs, demo steps). `anchor_index` indexes the sequence (negative
      OK, Python-style); entries within `window_frames // frames_per_step`
      steps before the anchor, anchor inclusive, are returned. Pass
      `frames_per_step` = the emulator frame_skip (4 in the robustifier)
      so the window is measured in real frames.
    * A `GoExploreArchive` or its `.cells` dict. Cells are ordered by
      `best_steps` (ascending, i.e. by depth along the elite path);
      `anchor_index` indexes that ordering and the returned cells are
      those whose `best_steps` lies within the window behind the
      anchor's, anchor inclusive.

    Pure: no mutation of the archive or trajectory; raises IndexError on
    an out-of-range anchor.
    """
    if window_frames < 0:
        raise ValueError(f"window_frames must be >= 0, got {window_frames}")
    if frames_per_step < 1:
        raise ValueError(f"frames_per_step must be >= 1, got {frames_per_step}")
    window_steps = window_frames // frames_per_step

    cells_map: Optional[dict] = None
    if isinstance(archive_or_trajectory, GoExploreArchive):
        cells_map = archive_or_trajectory.cells
    elif isinstance(archive_or_trajectory, dict):
        cells_map = archive_or_trajectory

    if cells_map is not None:
        ordered = sorted(cells_map.values(), key=lambda c: c.best_steps)
        anchor = ordered[anchor_index]  # IndexError propagates
        lo = anchor.best_steps - window_steps
        return [c for c in ordered if lo <= c.best_steps <= anchor.best_steps]

    seq = list(archive_or_trajectory)
    n = len(seq)
    idx = anchor_index + n if anchor_index < 0 else anchor_index
    if not 0 <= idx < n:
        raise IndexError(
            f"anchor_index {anchor_index} out of range for trajectory of "
            f"length {n}"
        )
    return seq[max(0, idx - window_steps): idx + 1]


def learnable(p_success: float, lo: float = 0.1, hi: float = 0.9) -> bool:
    """Is a start state inside the Florensa learnability band?

    Florensa et al. (arXiv:1707.05300, reverse curriculum generation)
    keep only "goals of intermediate difficulty": starts whose estimated
    success probability lies in [lo, hi]. Below `lo` the start is
    hopeless (no learning signal); above `hi` it is mastered (wasted
    samples). Bounds are inclusive.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo} hi={hi}")
    return lo <= p_success <= hi


def filter_learnable(
    starts: Sequence,
    p_success: Sequence[float],
    lo: float = 0.1,
    hi: float = 0.9,
) -> list:
    """Band-filter a start-state collection by the Florensa criterion.

    `p_success[i]` is the success estimate for `starts[i]` (e.g. a
    rolling greedy clear rate). Returns the starts whose estimate lies in
    [lo, hi] — retiring both mastered and hopeless starts (see
    `learnable`). Pure; the inputs are not mutated.
    """
    ps = list(p_success)
    if len(ps) != len(starts):
        raise ValueError(
            f"starts ({len(starts)}) and p_success ({len(ps)}) lengths differ"
        )
    return [s for s, p in zip(starts, ps) if learnable(p, lo, hi)]
