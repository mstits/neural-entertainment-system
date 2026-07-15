"""Sub-stage ladder for the SMB one-shot campaign (World 1, 1-1 -> 1-4).

Generalizes the proven scalar area-byte curriculum
(`smb_curriculum_anchors[stage]`, a packed `world<<4|area`) into an
**ordered list of `(world, area, x-bucket)` sub-stages** along the
sequential World-1 path. Each rung is a small slice of one level; the
existing 50%/5-iter rolling advance gate is reused verbatim, but it
measures `region_of(ram)` (a sub-stage order) instead of the packed byte.

Ladder shape (18 rungs, matching the campaign milestone table):

    orders 0-2   1-1        (area byte 0), 3 x-buckets
    order  3     1-2 entry  (area byte 1), 1 x-bucket
    orders 4-6   1-2 under  (area byte 2), 3 x-buckets
    orders 7-9   1-3        (area byte 3), 3 x-buckets
    orders 10-17 1-4 castle (area byte 4), 8 x-buckets (~256 x-units each,
                            matching the pre-existing stage_1_4_bw_x* blobs)

The area-byte mapping (0=1-1, 1=1-2 entrance, 2=1-2 underground main,
3=1-3, 4=1-4) is the empirically-verified one used by
`MarioReward::checkpoints_for` in `nes_core/src/rewards.rs`; the world byte
`$075F` is 0 for World 1 (see the reward engine and `DepthTracker`).

`region_of()` credits only sequential-path states: it returns
`OFF_LADDER` (-1) for any state `smb_sequential_cell` rejects (a warp into
world 2, or an off-chain area byte), so a warp can never raise the frontier
region. Within a level it is monotone in x; across the low-road area-byte
progression (0->1->2->3->4) it is monotone too.

Seeding is disk-first: `build_ladder(seed_globs)` globs the campaign's
save-state blobs and binds each rung to the nearest-x blob of the same
level. Rungs with no disk seed get `seed_blob=None` and are live-filled by
the trainer's mid-rollout capture path.

This module is deliberately decoupled from the trainer so it is
unit-testable in isolation (same stance as `go_explore.py` /
`curriculum.py`). The `smb_sequential_cell` cell key it consumes is owned
by `smb_sequential.py`; a spec-reference fallback is provided so this
module imports and tests standalone before that lane is merged.
"""

from __future__ import annotations

import glob as _glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

# ---- shared cell key (owned by smb_sequential.py; fallback below) ----
#
# `smb_sequential_cell(ram, x_bucket=8) -> (world, area, x_bucket) | None`
# returns None for any off-chain / warped state (world != World 1, or an
# area byte outside the sequential allowlist). `region_of` uses it purely
# for that admission decision, then reads the area/x bytes directly, so it
# is agnostic to whichever world-index convention that function returns.
try:  # pragma: no cover - exercised at integration once Lane 1 lands
    from src.training.smb_sequential import smb_sequential_cell
except Exception:  # pragma: no cover - the standalone/isolated path
    smb_sequential_cell = None  # bound to the reference impl below


# ---- RAM layout (Super Mario Bros.) ---------------------------------
RAM_WORLD = 0x075F   # world number, 0-indexed (0 == World 1)
RAM_AREA = 0x0760    # internal AREA index ($0760), NOT the displayed level
RAM_DISPLAY = 0x075C  # displayed level within the world (3 == x-4 castle)
RAM_X_HI = 0x006D    # screen page
RAM_X_LO = 0x0086    # fine x within the page

# World-1 sequential area bytes, verified against `checkpoints_for`:
#   0 = 1-1, 1 = 1-2 entrance, 2 = 1-2 underground main, 3 = 1-3, 4 = 1-4.
SEQUENTIAL_AREA_ALLOWLIST = frozenset({0, 1, 2, 3, 4})
WORLD1_BYTE = 0  # raw $075F value for World 1

# Displayed-level label per sequential area byte (1-2 spans two area bytes).
AREA_TO_LEVEL_LABEL: dict[int, str] = {
    0: "1-1",
    1: "1-2",
    2: "1-2",
    3: "1-3",
    4: "1-4",
}

# Sentinel region for any state off the sequential ladder (warp / off-chain).
# Chosen below every real order so the `region_of(ram) >= k+1` advance gate
# can never be satisfied by a warp, and a warp strictly lowers the region.
OFF_LADDER = -1

# ONE global platform-shaping reward profile for the whole ladder — the
# per-level dense checkpoint ladder auto-switches by area byte inside a
# single episode (`checkpoints_for`), so the profile is scalar and shared.
DEFAULT_REWARD_PROFILE = "smb_1_4_dense"


def _read_x(ram: Sequence[int]) -> int:
    """Absolute forward x, page<<8 | fine (matches the reward engine)."""
    return (int(ram[RAM_X_HI]) << 8) | int(ram[RAM_X_LO])


def _fallback_smb_sequential_cell(
    ram: Sequence[int], x_bucket: int = 8,
) -> Optional[tuple[int, int, int]]:
    """Spec-reference `smb_sequential_cell` (Lane 1 owns the canonical one).

    Returns `(world, area, x // x_bucket)` for an on-chain World-1 state, or
    `None` for a warp (world != World 1) or an off-allowlist area byte.
    `world`/`area` are the raw RAM bytes ($075F is 0 for World 1).
    """
    world = int(ram[RAM_WORLD])
    area = int(ram[RAM_AREA])
    if world != WORLD1_BYTE or area not in SEQUENTIAL_AREA_ALLOWLIST:
        return None
    x = _read_x(ram)
    return (world, area, x // max(1, int(x_bucket)))


if smb_sequential_cell is None:
    smb_sequential_cell = _fallback_smb_sequential_cell


# ---- sub-stage ----------------------------------------------------------
@dataclass(frozen=True)
class SubStage:
    """One rung of the ordered World-1 ladder.

    A rung is a `(world, area, [x_lo, x_hi))` slice of a level. `order` is
    the rung's position in the sequential ladder (the value `region_of`
    returns and the advance gate compares against). `seed_blob` is the disk
    path of the nearest-x save state bound to this rung (or None -> the
    trainer live-fills it via mid-rollout capture).
    """

    order: int
    world: int
    area: int
    x_lo: int
    x_hi: int
    seed_blob: Optional[str] = None
    reward_profile: str = DEFAULT_REWARD_PROFILE

    @property
    def level(self) -> str:
        """Displayed level label, e.g. "1-2"."""
        return AREA_TO_LEVEL_LABEL.get(self.area, f"1-?({self.area})")

    @property
    def x_center(self) -> float:
        """Representative x used for nearest-x seed binding."""
        return (self.x_lo + self.x_hi) / 2.0

    def contains_x(self, x: int) -> bool:
        return self.x_lo <= x < self.x_hi

    @property
    def anchor(self) -> int:
        """Packed `world<<4 | area`, the legacy curriculum anchor byte."""
        return (self.world << 4) | (self.area & 0x0F)


# Structural template: (area_byte, x_lo, x_hi) per rung, in sequential
# order. The x boundaries are heuristic slices (the reward engine's dense
# checkpoints put 1-2 obstacles near x=400/1300/2300 and the 1-4 axe near
# x=2560); only the ordering and the level partition are load-bearing. The
# top bucket of each level carries a generous x_hi so an agent that runs
# past the expected level end still maps to that level's final rung
# (region stays monotone, never drops).
_LADDER_TEMPLATE: tuple[tuple[int, int, int], ...] = (
    # 1-1 (area 0) — 3 buckets: entry / mid / flagpole approach.
    (0, 0, 1100),
    (0, 1100, 2200),
    (0, 2200, 3600),
    # 1-2 entrance (area 1) — 1 short bucket.
    (1, 0, 1000),
    # 1-2 underground main (area 2) — 3 buckets: pipes / gauntlet / warp exit.
    (2, 0, 900),
    (2, 900, 1800),
    (2, 1800, 3000),
    # 1-3 (area 3) — 3 buckets.
    (3, 0, 900),
    (3, 900, 1800),
    (3, 1800, 3000),
    # 1-4 castle (area 4) — 8 buckets, ~256 x-units each; the last absorbs
    # the bridge -> axe tail (axe at x~2560).
    (4, 0, 256),
    (4, 256, 512),
    (4, 512, 768),
    (4, 768, 1024),
    (4, 1024, 1280),
    (4, 1280, 1536),
    (4, 1536, 1792),
    (4, 1792, 3000),
)

# Number of rungs, exported for callers that size arrays on it.
LADDER_SIZE = len(_LADDER_TEMPLATE)


def _iter_globs(
    seed_globs: Union[str, Sequence[str], None],
    base_dir: Union[str, Path, None],
) -> list[str]:
    """Expand one glob string / a sequence of them into matching paths."""
    if seed_globs is None:
        return []
    if isinstance(seed_globs, (str, Path)):
        patterns: Sequence[str] = [str(seed_globs)]
    else:
        patterns = [str(p) for p in seed_globs]
    out: list[str] = []
    for pat in patterns:
        p = pat
        if base_dir is not None and not Path(pat).is_absolute():
            p = str(Path(base_dir) / pat)
        out.extend(_glob.glob(p))
    return out


# Filename parsers, tried in order. Each returns (area_byte, x) or None.
_RE_BW = re.compile(r"^stage_1_4_bw_x(\d+)\.state$")
_RE_DEPTH = re.compile(r"^depth_(\d+)_(\d+)_(\d+)(?:_\d+)*\.state(?:\.bin)?$")
_RE_STAGE = re.compile(r"^stage_(\d+)\.state$")


def _parse_seed(path: str) -> Optional[tuple[int, int]]:
    """Extract `(area_byte, x)` from a save-state filename.

    Recognizes the three campaign naming schemes:
      * `stage_1_4_bw_x<X>.state`            -> (area 4, x=<X>)
      * `depth_<world>_<area>_<x>[_<ts>]...` -> (area, x) for world 0 only
      * `stage_<NN>.state` (+ `.meta.json`)  -> (area, x=0) via the anchor

    Off-chain seeds (world != World 1, or an area outside the allowlist)
    return None so they never bind to a sequential rung.
    """
    p = Path(path)
    name = p.name

    m = _RE_BW.match(name)
    if m:
        return (4, int(m.group(1)))

    m = _RE_DEPTH.match(name)
    if m:
        world, area, x = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if world != WORLD1_BYTE or area not in SEQUENTIAL_AREA_ALLOWLIST:
            return None
        return (area, x)

    m = _RE_STAGE.match(name)
    if m:
        meta = p.with_name(f"stage_{m.group(1)}.meta.json")
        if not meta.exists():
            return None
        try:
            anchor = int(json.loads(meta.read_text())["anchor"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        world, area = (anchor >> 4) & 0x0F, anchor & 0x0F
        if world != WORLD1_BYTE or area not in SEQUENTIAL_AREA_ALLOWLIST:
            return None
        # No x in the name / anchor -> treat as the area's cold entry (x=0).
        return (area, 0)

    return None


def build_ladder(
    seed_globs: Union[str, Sequence[str], None] = None,
    *,
    base_dir: Union[str, Path, None] = None,
    reward_profile: str = DEFAULT_REWARD_PROFILE,
) -> list[SubStage]:
    """Build the ordered World-1 sub-stage ladder, binding disk seeds.

    Each rung is bound to the nearest-x save-state blob of the SAME level
    (area byte), by absolute distance to the rung's `x_center`. A blob may
    bind to several rungs (each rung picks its own nearest independently);
    rungs with no same-area blob keep `seed_blob=None` for the trainer to
    live-fill. Passing no globs yields the structural ladder (all
    `seed_blob=None`), which is also what `region_of` measures against.

    Args:
        seed_globs: a glob string, a sequence of them, or None. The
            campaign passes `checkpoints/.../stage_0*.state`,
            `.../stage_1_4_bw_x*.state`, `.../depth_0_*.state.bin`.
        base_dir: optional root to resolve relative glob patterns against.
        reward_profile: the shared platform-shaping profile tag for each rung.

    Returns:
        18 `SubStage`s ordered 0..17.
    """
    # Collect (area, x, path) for every matching, on-chain seed.
    seeds: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for path in _iter_globs(seed_globs, base_dir):
        if path in seen:
            continue
        seen.add(path)
        parsed = _parse_seed(path)
        if parsed is None:
            continue
        area, x = parsed
        seeds.append((area, x, path))

    ladder: list[SubStage] = []
    for order, (area, x_lo, x_hi) in enumerate(_LADDER_TEMPLATE):
        center = (x_lo + x_hi) / 2.0
        candidates = [(x, path) for (a, x, path) in seeds if a == area]
        seed_blob: Optional[str] = None
        if candidates:
            # Nearest by x; deterministic tie-break on the path name.
            seed_blob = min(
                candidates, key=lambda xp: (abs(xp[0] - center), xp[1])
            )[1]
        ladder.append(
            SubStage(
                order=order,
                world=WORLD1_BYTE,
                area=area,
                x_lo=x_lo,
                x_hi=x_hi,
                seed_blob=seed_blob,
                reward_profile=reward_profile,
            )
        )
    return ladder


# Structural ladder (no seeds) — the default reference `region_of` measures
# against. Cached so the common `region_of(ram)` call is allocation-free.
_STRUCTURAL_LADDER: Optional[list[SubStage]] = None


def _structural_ladder() -> list[SubStage]:
    global _STRUCTURAL_LADDER
    if _STRUCTURAL_LADDER is None:
        _STRUCTURAL_LADDER = build_ladder()
    return _STRUCTURAL_LADDER


def region_of(
    ram: Sequence[int], ladder: Optional[Sequence[SubStage]] = None,
) -> int:
    """Sub-stage order for a RAM snapshot along the sequential World-1 path.

    Returns the `order` of the rung whose `(area, [x_lo, x_hi))` slice the
    state falls in, or `OFF_LADDER` (-1) for any state `smb_sequential_cell`
    rejects (a warp into world 2, or an off-chain area byte). x past a
    level's final rung clamps to that rung's order, so the value is monotone
    in x within a level and never drops on forward progress; a warp yields
    `OFF_LADDER`, strictly below every real order, so it cannot raise the
    frontier region.

    `ladder` defaults to the structural ladder; pass a built ladder to reuse
    its (identical) rung boundaries.
    """
    # Admission: reject warps / off-chain states up front. Only the boolean
    # matters here, so this is robust to the cell's world-index convention.
    if smb_sequential_cell(ram) is None:
        return OFF_LADDER

    rungs = ladder if ladder is not None else _structural_ladder()
    area = int(ram[RAM_AREA])
    x = _read_x(ram)

    area_rungs = [r for r in rungs if r.area == area]
    if not area_rungs:
        # In the allowlist (else the cell would be None) but not represented
        # in this ladder — treat as off-ladder rather than guessing.
        return OFF_LADDER
    for r in area_rungs:
        if r.contains_x(x):
            return r.order
    # Past the final rung of this area: clamp to it (monotone, no drop).
    return area_rungs[-1].order
