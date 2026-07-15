"""Sequential World-1 progression predicate for the SMB one-shot campaign.

`eval_game.py`'s existing clear notion is `episode_success()`, which latches on
`cleared_any` — a flag rewards.rs sets at the FIRST flagpole touch (1-1). That
latch makes a 1-1 clear indistinguishable from a full World-1 clear, so it can
never measure a *sequential* 1-1 -> 1-4 campaign. This module reconstructs the
sequential predicate directly from RAM so a cold eval can score "beat World 1
in order" independent of the 1-1 latch.

Two public pieces, both encoder-agnostic and pure (no emulator, no torch):

  * `SequentialTracker` — fed the RAM each step of an episode; exposes
    `seq_clear` (a real 1-4 castle clear crossing into World 2), `warp_taken`
    (a World rise that came from a warp pipe, NOT a castle), and the furthest
    `(world, level)` reached on the sequential chain (`furthest_seq`) vs. any
    route including warps (`furthest_any`).

  * `smb_sequential_cell(ram, x_bucket)` — the warp-admission cell key. Returns
    `None` for any state OFF the World-1 sequential chain (World != 1 or a
    non-allowlisted area byte), so warped progress never seeds the ladder, never
    enters a Go-Explore archive, and never gets captured live. Shared later by
    the sub-stage ladder's `region_of` and the GE-burst cell key.

RAM semantics (all verified against rewards.rs and the shipped curriculum save
states — see `SEQ_AREAS` below):
  * `$075F` WorldNumber — 0-indexed (World 1 == 0).
  * `$0760` AreaNumber — internal area index, NOT the displayed level.
  * `$075C` displayed level-within-world — 0-indexed (x-1..x-4 == 0..3).
  * `$006D`/`$0086` Mario X page/low -> absolute x = page*256 + low.

`seq_clear` mirrors the shipped F52 castle guard (rewards.rs:1322): a World-byte
increment whose *previous* displayed level was the x-4 castle (`$075C == 3`) is a
real castle clear; a World rise from any other displayed level is a warp (SMB
warp pipes leave from x-2, `$075C == 1`).
"""

from __future__ import annotations

from typing import Optional, Tuple

# RAM addresses (see module docstring).
RAM_WORLD = 0x075F
RAM_AREA = 0x0760
RAM_DISPLAY = 0x075C
RAM_X_PAGE = 0x006D
RAM_X_LOW = 0x0086

# Displayed level index of an x-4 castle. A World-byte increment out of this
# displayed level is a genuine castle clear; out of anything else it is a warp.
DISPLAY_LEVEL_CASTLE = 3

# World-1 sequential area bytes ($0760), in play order:
#   0 = 1-1
#   1 = 1-2 entrance (the short above-ground bit)
#   2 = 1-2 underground main (the long pipe gauntlet)
#   3 = 1-3
#   4 = 1-4 (the castle)
# EMPIRICALLY verified by loading the shipped curriculum save states and reading
# $075F/$0760/$075C: stage_01 -> area 1 / disp 1, stage_02 -> area 2 / disp 1,
# stage_03 -> area 3 / disp 2, stage_1_4 -> area 4 / disp 3. This matches
# rewards.rs `checkpoints_for()` (area 3 -> LEVEL_1_3, area 4 -> LEVEL_1_4) and
# intentionally DIFFERS from the trainer's `AREA_TO_LEVEL` telemetry table
# (which maps 5 -> "3", 6 -> "4"); that table mislabels World 1's 1-3/1-4, and
# the save-state RAM is the ground truth. Any area outside this set (a warp room
# or a World 2+ area) is off the sequential chain.
SEQ_AREAS = frozenset({0, 1, 2, 3, 4})


def _byte(ram, addr: int) -> int:
    """Read one RAM byte as a plain int (accepts np.ndarray, bytes, list)."""
    return int(ram[addr])


def _read_x(ram) -> int:
    """Absolute Mario X = page*256 + low ($006D, $0086)."""
    return (_byte(ram, RAM_X_PAGE) << 8) | _byte(ram, RAM_X_LOW)


def _displayed(world_byte: int, display_byte: int) -> Tuple[int, int]:
    """(world, level) as the game displays them: both 1-indexed."""
    return (world_byte + 1, display_byte + 1)


def smb_sequential_cell(
    ram, x_bucket: int = 8
) -> Optional[Tuple[int, int, int]]:
    """Warp-admission cell key: ``(world, area, x // x_bucket)`` or ``None``.

    `world` is the DISPLAYED world (1-indexed, so World 1 == 1); `area` is the
    raw `$0760` area byte, UNBUCKETED, so 1-2's entrance (area 1) and underground
    (area 2) stay distinct sub-stages and two different levels at the same x page
    never collide. Returns ``None`` for any state off the World-1 sequential
    chain — `world != 1` (a warp/next-world state) or an area byte not in
    `SEQ_AREAS` (a warp room). `None` is the admission veto: such a state can
    never seed the ladder, advance the frontier, or key an archive burst.
    """
    world = _byte(ram, RAM_WORLD) + 1  # displayed world (World 1 == 1)
    area = _byte(ram, RAM_AREA)
    if world != 1 or area not in SEQ_AREAS:
        return None
    bucket = x_bucket if x_bucket > 0 else 1
    return (world, area, _read_x(ram) // bucket)


class SequentialTracker:
    """Reconstruct sequential World-1 progression from per-step RAM.

    Feed `.update(ram)` once per environment step across a single episode
    (instantiate a fresh tracker, or call `.reset()`, per episode — the F52
    guard and the high-water marks are episode-scoped). Then read the
    properties. `seq_clear`/`warp_taken` are latches (set once, stay set);
    `furthest_seq`/`furthest_any` are `(world, level)` high-water marks.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Previous transition sample (mirrors rewards.rs: these advance only on
        # a world/area transition, so `_prev_display` holds the displayed level
        # of the level we are leaving — the signal the F52 castle guard needs).
        self._prev_world: Optional[int] = None
        self._prev_area: int = 0
        self._prev_display: int = 0
        self._seq_clear: bool = False
        self._warp_taken: bool = False
        self._furthest_seq: Optional[Tuple[int, int]] = None
        self._furthest_any: Optional[Tuple[int, int]] = None

    def update(self, ram) -> None:
        world = _byte(ram, RAM_WORLD)
        area = _byte(ram, RAM_AREA)
        display = _byte(ram, RAM_DISPLAY)

        if self._prev_world is None:
            # First step: seed the transition sample; no clear/warp yet.
            self._prev_world = world
            self._prev_area = area
            self._prev_display = display
        elif world != self._prev_world or area != self._prev_area:
            # World/area transition (rewards.rs:1304). A World-byte increment
            # out of the x-4 castle is a real clear; out of anything else it is
            # a warp pipe.
            if world > self._prev_world:
                if self._prev_display == DISPLAY_LEVEL_CASTLE:
                    self._seq_clear = True
                else:
                    self._warp_taken = True
            self._prev_world = world
            self._prev_area = area
            self._prev_display = display

        # High-water marks (every step). `furthest_any` counts warps and later
        # worlds ("beyond"); `furthest_seq` counts only admitted World-1 states.
        cur = _displayed(world, display)
        if self._furthest_any is None or cur > self._furthest_any:
            self._furthest_any = cur
        if world + 1 == 1 and area in SEQ_AREAS:
            if self._furthest_seq is None or cur > self._furthest_seq:
                self._furthest_seq = cur

    @property
    def seq_clear(self) -> bool:
        """True once a real World-1 castle clear (crossing into World 2) fired."""
        return self._seq_clear

    @property
    def warp_taken(self) -> bool:
        """True once a warp-pipe World rise (not a castle clear) fired."""
        return self._warp_taken

    @property
    def furthest_seq(self) -> Optional[Tuple[int, int]]:
        """Deepest displayed `(world, level)` on the sequential World-1 chain."""
        return self._furthest_seq

    @property
    def furthest_any(self) -> Optional[Tuple[int, int]]:
        """Deepest displayed `(world, level)` on ANY route, warps included."""
        return self._furthest_any


def level_label(level: Optional[Tuple[int, int]]) -> Optional[str]:
    """Format a displayed `(world, level)` tuple as e.g. ``"1-4"``."""
    if level is None:
        return None
    return f"{level[0]}-{level[1]}"
