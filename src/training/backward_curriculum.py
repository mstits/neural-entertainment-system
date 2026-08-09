"""Backward start-state curriculum — reverse curriculum from a solver tape.

Salimans & Chen (arXiv:1812.03381, "Learning Montezuma's Revenge from a
Single Demonstration") do NOT imitate the demonstration: they use it only
as a source of START STATES. Training begins near the end of the tape and
the restart cursor walks BACKWARD as the policy earns each rung, until the
agent is starting from the true entrance and the demonstration has been
fully consumed. The reward is the only learning signal at every rung — no
action labels ever reach the network.

That distinction is why this module exists at all. Naive behavior cloning
on the same tapes was ELIMINATED for this stack in Dossier v3 (2026-07-23):
clone accuracy 1.0 collapsed to 0.00 success under the honest sticky
protocol. The tape is a curriculum, not a teacher.

Three pieces, all pure so they unit-test without an emulator:

* `snapshot_steps` — which action-steps a minter should snapshot, given a
  tape length and a frame stride. `scripts/mint_backward_states.py` is the
  only production caller.
* `TauScheduler` — the backward cursor. Holds a trailing success window at
  the current rung and walks tau back by a fixed number of tape entries
  once that window clears the advance threshold over enough attempts.
  Small-sample gates are the known failure mode on this project (3-episode
  gates were measured to be mirages behind a true success rate under
  1/400), so the attempt floor is a hard precondition, not a suggestion.
* `draw_restart` — the length-balanced draw over {window entries, true
  entrance}. The true entrance is ALWAYS in the pool, so the terminal
  curriculum (tau == 0) is exactly the honest start distribution and the
  in-training entrance success rate is measurable throughout the run.

The window itself is `go_explore.start_window` — one implementation of
"trailing window behind an anchor" for the whole repo, already tested.

Index format (`index.json`, written by the minter, read by `load_index`):

    {"level": "1-1", "frame_skip": 4, "every_frames": 4,
     "stride_steps": 1, "n_actions": 903, ...,
     "entries": [{"step": 0, "frame": 0, "gx": 40, "area": 0,
                  "file": "s_000000.state"}, ...]}
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

INDEX_NAME = "index.json"
STATE_PREFIX = "s_"
STATE_SUFFIX = ".state"

# Defaults mirror the trainer's `reinforce.backward_curriculum` block so a
# config that omits a key and a caller that omits an argument agree.
DEFAULT_EVERY_FRAMES = 4
DEFAULT_WINDOW_FRAMES = 160
DEFAULT_ADVANCE_ACTIONS = 40
DEFAULT_ADVANCE_THRESHOLD = 0.2
DEFAULT_MIN_ATTEMPTS = 30
DEFAULT_ENTRANCE_WEIGHT = 1.0
# Within-segment gx dip (pixels) treated as sub-tile jitter rather than a
# real regression by `gx_report`. One SMB tile.
DEFAULT_GX_TOLERANCE = 16
# A drop that LANDS at or below this gx is the coordinate re-basing at a
# new region's origin (pipe / vine / castle), not a regression. One page.
DEFAULT_GX_RESET_MAX = 256

# Sentinel returned by `draw_restart` for "use the true level entrance".
ENTRANCE = -1

# Schema tag on `TauScheduler.state_dict()`. Bumped only if the meaning of
# a field changes; readers tolerate its absence.
STATE_VERSION = 1


@dataclass(frozen=True)
class StateEntry:
    """One minted savestate: where on the tape it was taken."""

    step: int      # action index the state was captured BEFORE
    frame: int     # NES frames elapsed from the tape root
    gx: int        # global x at capture (telemetry / monotonicity gate)
    area: int      # area byte at capture (segments the gx gate)
    file: str      # basename inside the states dir

    def as_dict(self) -> dict:
        return {"step": self.step, "frame": self.frame, "gx": self.gx,
                "area": self.area, "file": self.file}


# ---- minting (pure planning half) -----------------------------------


def snapshot_steps(
    n_actions: int,
    *,
    frames_per_step: int = 4,
    every_frames: int = DEFAULT_EVERY_FRAMES,
) -> list[int]:
    """Action-step indices to snapshot for a tape of `n_actions`.

    The emulator only exists at step granularity (one policy action spans
    `frames_per_step` NES frames), so a request for a state "every 4
    frames" at frame_skip 4 resolves to every step, and "every 16 frames"
    to every 4th step. A stride finer than one step is rounded UP to one
    step rather than silently duplicating states.

    The final post-clear state is deliberately NOT emitted: a restart
    there has already won and teaches nothing.
    """
    if n_actions < 0:
        raise ValueError(f"n_actions must be >= 0, got {n_actions}")
    if frames_per_step < 1:
        raise ValueError(f"frames_per_step must be >= 1, got {frames_per_step}")
    if every_frames < 1:
        raise ValueError(f"every_frames must be >= 1, got {every_frames}")
    stride = max(1, every_frames // frames_per_step)
    return list(range(0, n_actions, stride))


def stride_steps(frames_per_step: int, every_frames: int) -> int:
    """Tape steps between consecutive minted states (>= 1)."""
    if frames_per_step < 1 or every_frames < 1:
        raise ValueError("frames_per_step and every_frames must be >= 1")
    return max(1, every_frames // frames_per_step)


def state_filename(step: int) -> str:
    """Stable, sortable basename for the state minted at `step`."""
    return f"{STATE_PREFIX}{int(step):06d}{STATE_SUFFIX}"


def gx_report(
    entries: Sequence[StateEntry],
    tolerance: int = DEFAULT_GX_TOLERANCE,
    reset_max: int = DEFAULT_GX_RESET_MAX,
) -> dict:
    """Monotonicity report over a minted index's progress coordinate.

    A forward solution's global x rises, but two kinds of drop are normal
    and neither says anything about tape quality:

    * The odometer RESTARTS. Entering a pipe, a vine or a castle re-bases
      x at the new region's origin. Sometimes the area byte moves with it
      (1-2 step 115: area 1 -> 2, gx 160 -> 0) and sometimes it does not
      (1-2 step 971: area 2 throughout, gx 2656 -> 0). Either way the
      coordinate wrapped, so comparisons restart: a drop BEYOND
      `tolerance` that LANDS at or below `reset_max` opens a new segment
      instead of counting. Both halves matter — without the size test,
      ordinary jitter near a level's origin reads as a re-base.
    * Sub-tile jitter — deceleration, landing recoil, a bumped wall —
      dips x by a few pixels. Measured on the banked tapes: 47 dips on
      1-1 (max 4 px), 91 on 1-2. `tolerance` (one tile by default)
      separates that from a real regression, i.e. a tape doubling back.

    `monotone` applies both rules and is the mint gate. `strict_monotone`
    applies neither beyond the segmenting, so a caller can still see the
    raw picture. `max_drop` is the largest WITHIN-segment dip in pixels.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    if reset_max < 0:
        raise ValueError(f"reset_max must be >= 0, got {reset_max}")
    decreases = 0
    over_tol = 0
    max_drop = 0
    segment_starts: list[int] = [entries[0].step] if entries else []
    areas: list[int] = []
    for e in entries:
        if not areas or areas[-1] != e.area:
            areas.append(e.area)
    for prev, cur in zip(entries, entries[1:]):
        drop = prev.gx - cur.gx
        if prev.area != cur.area or (drop > tolerance and cur.gx <= reset_max):
            segment_starts.append(cur.step)
            continue
        if drop > 0:
            decreases += 1
            max_drop = max(max_drop, drop)
            if drop > tolerance:
                over_tol += 1
    return {
        "n": len(entries),
        "tolerance": int(tolerance),
        "reset_max": int(reset_max),
        "decreases": decreases,
        "decreases_over_tolerance": over_tol,
        "max_drop": max_drop,
        "monotone": over_tol == 0,
        "strict_monotone": decreases == 0,
        "areas": areas,
        "segments": len(segment_starts),
        "segment_starts": segment_starts,
        "gx_first": entries[0].gx if entries else 0,
        "gx_last": entries[-1].gx if entries else 0,
    }


# ---- index IO -------------------------------------------------------


def write_index(states_dir, entries: Sequence[StateEntry], meta: dict) -> Path:
    """Write `index.json` for a minted states dir. Returns its path."""
    out = Path(states_dir) / INDEX_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(meta)
    rec["entries"] = [e.as_dict() for e in entries]
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return out


def load_index(states_dir) -> tuple[list[StateEntry], dict]:
    """Read a minted states dir. Returns (entries, meta-without-entries).

    Entries come back sorted by tape step, which is the order the backward
    cursor indexes them in. Raises on an empty or missing index rather
    than degrading to a silent no-curriculum run.
    """
    path = Path(states_dir) / INDEX_NAME
    rec = json.loads(path.read_text())
    raw = rec.get("entries") or []
    entries = sorted(
        (StateEntry(step=int(e["step"]), frame=int(e["frame"]),
                    gx=int(e.get("gx", 0)), area=int(e.get("area", 0)),
                    file=str(e["file"]))
         for e in raw),
        key=lambda e: e.step,
    )
    if not entries:
        raise ValueError(f"{path}: index has no entries")
    meta = {k: v for k, v in rec.items() if k != "entries"}
    meta["dir"] = str(Path(states_dir))
    return entries, meta


def load_blobs(states_dir, entries: Iterable[StateEntry]) -> list[bytes]:
    """Read every minted savestate into memory, in `entries` order.

    A tape's worth of SMB states is ~20 KB each — a 1200-step level is
    ~25 MB, cheaper to hold than to re-read on every one of the thousands
    of episode restarts an iter performs.
    """
    root = Path(states_dir)
    return [(root / e.file).read_bytes() for e in entries]


# ---- the backward cursor --------------------------------------------


class TauScheduler:
    """Backward restart cursor over a minted tape.

    `tau` indexes the tape's minted entries; episodes restart from the
    trailing window ENDING at tau (see `go_explore.start_window`), so tau
    is "how far into the solution the curriculum currently starts". It
    begins at the last entry (`tau_init=-1`, right before the clear) and
    walks toward 0 (the level entrance).

    Advance rule: the trailing window of attempts AT THE CURRENT TAU must
    hold at least `min_attempts` samples and a success rate of at least
    `advance_threshold` before tau moves back by `advance_entries`. Both
    conditions are hard — the project's history with small-sample gates
    is that they certify mirages.

    Attempts that started from the true entrance are NOT tau attempts and
    must be recorded through `record_entrance` instead: mixing them in
    would drag the rung's rate down with a different distribution's
    outcomes and stall the curriculum. Their rate is tracked separately
    because it is the honest in-training signal worth logging.
    """

    __slots__ = ("n_entries", "advance_entries", "advance_threshold",
                 "min_attempts", "trailing", "_tau", "_window", "_advances",
                 "_ent_att", "_ent_succ")

    def __init__(
        self,
        n_entries: int,
        *,
        tau_init: int = -1,
        advance_entries: int = DEFAULT_ADVANCE_ACTIONS,
        advance_threshold: float = DEFAULT_ADVANCE_THRESHOLD,
        min_attempts: int = DEFAULT_MIN_ATTEMPTS,
        trailing: Optional[int] = None,
    ) -> None:
        if n_entries < 1:
            raise ValueError(f"n_entries must be >= 1, got {n_entries}")
        if advance_entries < 1:
            raise ValueError(
                f"advance_entries must be >= 1, got {advance_entries}")
        if not 0.0 <= advance_threshold <= 1.0:
            raise ValueError(
                f"advance_threshold must be in [0,1], got {advance_threshold}")
        if min_attempts < 1:
            raise ValueError(f"min_attempts must be >= 1, got {min_attempts}")
        tau = tau_init + n_entries if tau_init < 0 else tau_init
        if not 0 <= tau < n_entries:
            raise IndexError(
                f"tau_init {tau_init} out of range for {n_entries} entries")
        self.n_entries = int(n_entries)
        self.advance_entries = int(advance_entries)
        self.advance_threshold = float(advance_threshold)
        self.min_attempts = int(min_attempts)
        self.trailing = int(trailing) if trailing else int(min_attempts)
        self._tau = int(tau)
        self._window: deque = deque(maxlen=self.trailing)
        self._advances = 0
        self._ent_att = 0
        self._ent_succ = 0

    # -- state --------------------------------------------------------

    @property
    def tau(self) -> int:
        return self._tau

    @property
    def at_entrance(self) -> bool:
        """True once the cursor has walked all the way back."""
        return self._tau == 0

    @property
    def attempts(self) -> int:
        return len(self._window)

    @property
    def successes(self) -> int:
        return sum(self._window)

    @property
    def rate(self) -> float:
        return (self.successes / len(self._window)) if self._window else 0.0

    # -- bookkeeping --------------------------------------------------

    def record(self, success: bool, tau: Optional[int] = None) -> bool:
        """Score one episode that started inside the tau window.

        `tau` is the cursor the episode STARTED under. An episode that
        outlived an advance carries a stale tau and is dropped (returns
        False) — it measured a rung the curriculum has already left.
        """
        if tau is not None and int(tau) != self._tau:
            return False
        self._window.append(bool(success))
        return True

    def record_entrance(self, success: bool) -> None:
        """Score one episode that started from the true level entrance."""
        self._ent_att += 1
        self._ent_succ += int(bool(success))

    def maybe_advance(self) -> bool:
        """Walk tau back if this rung is earned. Returns True if it moved."""
        if self._tau == 0:
            return False
        if len(self._window) < self.min_attempts:
            return False
        if self.rate < self.advance_threshold:
            return False
        self._tau = max(0, self._tau - self.advance_entries)
        self._window.clear()
        self._advances += 1
        return True

    # -- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        """Everything a resume needs to land back on the same rung.

        A curriculum whose cursor is not checkpointed is a curriculum that
        restarts at the ladder top on every relaunch: B4 v3 resumed from
        iter 80 with tau back at 757/757, so the at-entrance winner gate
        (the mode's deliverable selector) never fired again for the rest
        of the run. Everything the advance rule and the entrance telemetry
        read lives here — the cursor, the trailing window's CONTENTS (not
        just its rate; a half-full window must resume half-full or the
        attempt floor is silently re-armed), the advance count and the
        entrance counters.

        Pure JSON/pickle-safe types only, and every container is a fresh
        copy: the caller may keep, mutate or torch.save the result without
        reaching back into the live cursor.

        Config (`advance_entries`, `advance_threshold`, `min_attempts`,
        `trailing`) is deliberately NOT stored — it comes from the config
        file, so a resume picks up retuned gates instead of freezing the
        ones the run started with. `n_entries` IS stored, as the identity
        of the tape tau indexes: see `load_state_dict`.
        """
        return {
            "version": STATE_VERSION,
            "tau": int(self._tau),
            "n_entries": int(self.n_entries),
            "window": [bool(x) for x in self._window],
            "advances": int(self._advances),
            "entrance_attempts": int(self._ent_att),
            "entrance_successes": int(self._ent_succ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a cursor saved by `state_dict`. Raises on nonsense.

        Validation is strict and the caller is expected to catch: a
        checkpoint whose tape had a different number of entries makes tau
        meaningless (entry 400 of a 757-rung tape is not entry 400 of a
        re-minted 900-rung one), and silently honoring it would restart
        the curriculum at an arbitrary point in the level. Failing loudly
        lets the trainer log it and fall back to the configured `tau_init`
        — the same place a run with no checkpoint starts.

        The window is re-bound to THIS scheduler's `trailing` length, so a
        retuned `min_attempts` takes effect on resume (an over-long saved
        window keeps its most recent attempts, as the live deque would).
        """
        try:
            tau = int(state["tau"])
        except (KeyError, TypeError) as e:
            raise KeyError(f"backward curriculum state has no usable 'tau': {e}")
        n_entries = int(state.get("n_entries", self.n_entries))
        if n_entries != self.n_entries:
            raise ValueError(
                f"state was saved against a {n_entries}-entry tape but this "
                f"scheduler has {self.n_entries}: tau is not comparable — "
                "the tape was re-minted"
            )
        if not 0 <= tau < self.n_entries:
            raise ValueError(
                f"tau {tau} out of range for {self.n_entries} entries")
        window = [bool(x) for x in (state.get("window") or ())]
        advances = int(state.get("advances", 0))
        ent_att = int(state.get("entrance_attempts", 0))
        ent_succ = int(state.get("entrance_successes", 0))
        if advances < 0 or ent_att < 0 or ent_succ < 0:
            raise ValueError(
                f"negative counters in state: advances={advances}, "
                f"entrance={ent_succ}/{ent_att}")
        if ent_succ > ent_att:
            raise ValueError(
                f"entrance successes {ent_succ} exceed attempts {ent_att}")
        # Commit only after every field validated, so a rejected state
        # leaves the cursor exactly as it was.
        self._tau = tau
        self._window = deque(window, maxlen=self.trailing)
        self._advances = advances
        self._ent_att = ent_att
        self._ent_succ = ent_succ

    def snapshot(self) -> dict:
        """Loggable state. Pure — safe to call every iter."""
        return {
            "tau": self._tau,
            "n_entries": self.n_entries,
            "attempts": self.attempts,
            "successes": self.successes,
            "rate": self.rate,
            "advances": self._advances,
            "at_entrance": self.at_entrance,
            "entrance_attempts": self._ent_att,
            "entrance_successes": self._ent_succ,
            "entrance_rate": (self._ent_succ / self._ent_att
                              if self._ent_att else 0.0),
        }


# ---- the restart draw -----------------------------------------------


def draw_restart(
    n_window: int,
    entrance_weight: float,
    rand: float,
) -> int:
    """Pick a restart from {window entries} u {true entrance}.

    Returns an index into the window, or `ENTRANCE` (-1). The draw is
    uniform over the window entries, with the entrance carrying
    `entrance_weight` entries' worth of mass — length-balanced, so the
    entrance keeps a fixed per-episode share (1/(W+1) at weight 1.0)
    instead of a share that swings with the window size. That trickle is
    what makes the honest start distribution measurable during training,
    and at tau 0 the window IS the entrance region, so the terminal
    curriculum needs no special case.

    `rand` is a caller-supplied uniform in [0, 1) — the trainer draws it
    from the same `np.random` stream everything else uses, and keeping it
    an argument makes the split testable without seeding anything.
    """
    if n_window < 0:
        raise ValueError(f"n_window must be >= 0, got {n_window}")
    if entrance_weight < 0.0:
        raise ValueError(
            f"entrance_weight must be >= 0, got {entrance_weight}")
    total = n_window + entrance_weight
    if total <= 0.0:
        return ENTRANCE
    u = float(rand) * total
    if u >= n_window:
        return ENTRANCE
    return min(int(u), n_window - 1)
