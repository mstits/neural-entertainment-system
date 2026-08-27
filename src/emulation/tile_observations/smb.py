"""SMB-specific tile observation decoder.

Reads CPU RAM directly to produce a tile grid centered on Mario plus
a small set of game-state scalars. RAM addresses were originally cited
to the standard NESdev SMB disassembly and SethBling's MarI/O Lua
script, but SMB is this project's one completely solved game — all 32
levels cleared on a single cold-boot tape with `state_loads=0` and the
ending frame rendered — so every address below is independently earned
by that witnessed run, not merely inherited from the external citation.
A behaviour change here would invalidate banked SMB training runs, so
this is a provenance note, not a semantics change.

Output layout (175 int8 features, all in range [-128, 127]):

* `[0:169]` — 13×13 tile grid (row-major, top-down) centered on Mario.
  Values:  0 = empty/passable, 1 = solid block, -1 = enemy.
* `[169]`  — Mario X velocity (signed byte)
* `[170]`  — Mario Y velocity (signed byte)
* `[171]`  — `on_ground` flag (0 in air, 1 on ground)
* `[172]`  — Mario powerup state (0 small, 1 big, 2 fire flower)
* `[173]`  — lives remaining
* `[174]`  — fractional X-position within current screen page
            (0-15, encodes Mario's sub-tile horizontal position)

The grid is centered on Mario's screen position, with 6 cells of context
to each side (total 13). Feature 174 captures sub-tile precision that
tile-grid alone loses — relevant for jump-arc timing.

Performance: each call reads ~50 RAM bytes and does 169 array lookups.
Fully vectorized via numpy where possible. Sub-millisecond per call,
which is well below the per-step emulation time (a few ms).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# RAM addresses (originally cited to NESdev wiki / SMB disassembly;
# independently earned by SMB's witnessed 32-level cold-boot clear —
# see module docstring)
_RAM_X_PAGE         = 0x006D  # Mario screen-page index (high byte of X)
_RAM_X_LOW          = 0x0086  # Mario X within page (low byte)
_RAM_Y              = 0x00CE  # Mario Y on screen
_RAM_VEL_X          = 0x0057  # signed X velocity
_RAM_VEL_Y          = 0x009F  # signed Y velocity
_RAM_ON_GROUND      = 0x001D  # 0 when on ground, nonzero in air
_RAM_POWERUP_STATE  = 0x0756  # 0 small, 1 big, 2 fire
_RAM_LIVES          = 0x075A
_RAM_TILE_BASE      = 0x0500  # 13×16 metatile grid, two pages

# Enemy slot bookkeeping (5 active slots, contiguous indexing).
_NUM_ENEMY_SLOTS    = 5
_RAM_ENEMY_FLAG     = 0x000F  # nonzero = enemy active in slot
_RAM_ENEMY_X_PAGE   = 0x006E  # enemy X page (parallel to Mario's)
_RAM_ENEMY_X_LOW    = 0x0087  # enemy X within page
_RAM_ENEMY_Y        = 0x00CF  # enemy Y on screen

# Output grid dimensions. 13×13 with Mario centered at (6, 6).
_GRID_W = 13
_GRID_H = 13
_MARIO_GX = 6
_MARIO_GY = 6
_TILE_SIZE = 16  # pixels per metatile in SMB

# Total feature vector length: tile grid + 6 scalars.
_NUM_SCALARS = 6
FEATURE_DIM = _GRID_W * _GRID_H + _NUM_SCALARS

# Tile encoding values.
_TILE_EMPTY = 0
_TILE_SOLID = 1
_TILE_ENEMY = -1


class SMBTileObservation:
    """Implements the `TileObservation` protocol for Super Mario Bros.

    Constructed once per worker; `extract(ram)` is called every agent
    step. Internal state-free — the same instance can be reused across
    workers without sync issues.

    The 13×13 grid sampler is fully numpy-vectorized: address arithmetic
    runs as broadcast integer ops, the 169 tile lookups happen as a
    single fancy-index `ram[addr]` gather, and out-of-bounds cells are
    masked out instead of branched per-cell. Replaced an earlier pure-
    Python double-loop that profiled at ~7% of trainer wall-time
    (815k inner iterations across a 3-gen run); the vectorized path is
    ~10× faster on the same workload.
    """

    def __init__(self, use_rust: bool = True) -> None:
        # Pre-compute the (dx, dy) offset grids once; reused on every
        # extract() call. Center cell (Mario) is at (_MARIO_GX, _MARIO_GY).
        dx_grid, dy_grid = np.meshgrid(
            np.arange(_GRID_W, dtype=np.int32) - _MARIO_GX,
            np.arange(_GRID_H, dtype=np.int32) - _MARIO_GY,
        )
        # Flatten so we can do one big gather. Order is row-major to
        # match the original Python loop (`out[dy*W + dx]`).
        self._dx_flat = dx_grid.flatten()
        self._dy_flat = dy_grid.flatten()
        # Optional native dispatch: nes_core.extract_smb_tiles is byte-
        # exact with the Python numpy path (parity-tested across 2000
        # randomized RAM snapshots in tests/test_smb_tile_extract_parity.py)
        # but ~55x faster. Available when the abi3 wheel is built —
        # falls back to the numpy path on import failure so dev environments
        # without the Rust binary still work.
        self._extract_rust = None
        if use_rust:
            try:
                import nes_core
                if hasattr(nes_core, "extract_smb_tiles"):
                    self._extract_rust = nes_core.extract_smb_tiles
            except ImportError as exc:
                log.warning(
                    "nes_core native extension unavailable (%s); falling back "
                    "to the numpy tile-extraction path (~55x slower). If a "
                    "build was expected, the compiled extension may be "
                    "missing or ABI-mismatched.",
                    exc,
                )

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def extract(self, ram_bytes: bytes) -> np.ndarray:
        """Decode a 2 KB RAM snapshot into a flat 175-int8 feature vector.

        The grid sampling matches MarI/O: world-coordinate lookup with
        wrap-around across the two-page tile RAM buffer. Out-of-bounds
        cells (above the level ceiling or off the right edge) are
        reported as empty so the agent treats them as passable.
        """
        if self._extract_rust is not None:
            return self._extract_rust(ram_bytes)
        ram = np.frombuffer(ram_bytes, dtype=np.uint8)
        out = np.zeros(FEATURE_DIM, dtype=np.int8)

        # Mario's world coordinates. X spans multiple screens, so the
        # high byte (page) and low byte (offset within page) combine
        # into the world X. Y is screen-relative since SMB doesn't
        # scroll vertically in 1-1.
        mario_x_world = (int(ram[_RAM_X_PAGE]) << 8) | int(ram[_RAM_X_LOW])
        mario_y_screen = int(ram[_RAM_Y])

        # World pixel coordinates of all 169 grid cells, computed as
        # broadcast integer arithmetic. Each entry is the cell's center.
        world_px_x = mario_x_world + self._dx_flat * _TILE_SIZE
        world_px_y = mario_y_screen + self._dy_flat * _TILE_SIZE

        # Convert to per-page sub-tile coordinates.
        sub_y = (world_px_y - 32) // _TILE_SIZE
        page = (world_px_x // 256) % 2
        sub_x = (world_px_x % 256) // _TILE_SIZE

        # Validity mask: a cell maps to a real tile-RAM byte only when
        # all of these hold (matches the original `_tile_at` branching).
        valid = (
            (world_px_x >= 0)  # left of world origin: floor-mod would wrap
            & (world_px_y >= 0)  # a negative x into a phantom page-1 tile
            & (sub_y >= 0) & (sub_y < 13)
            & (sub_x >= 0) & (sub_x < 16)
        )

        # Compute target addresses for ALL 169 cells. Invalid cells get
        # a safe 0 address; we'll ignore their fetched values via mask.
        addr = (
            _RAM_TILE_BASE
            + page * (13 * 16)
            + sub_y * 16
            + sub_x
        ).astype(np.int32)
        addr_safe = np.where(valid, addr, 0)
        # Single fancy-index gather replaces 169 sequential lookups.
        tile_bytes = ram[addr_safe]
        # Solid wherever the byte is non-zero AND the cell was valid.
        is_solid = (tile_bytes != 0) & valid
        out[:_GRID_W * _GRID_H] = is_solid.astype(np.int8)

        # Overlay enemies on the grid. Enemies are sparse (at most 5
        # active), so this stays as a scalar loop — vectorizing 5
        # iterations isn't worth the complexity, but the path is now
        # the only Python loop in the hot path.
        for slot in range(_NUM_ENEMY_SLOTS):
            if ram[_RAM_ENEMY_FLAG + slot] == 0:
                continue  # slot unused
            enemy_x_world = (
                (int(ram[_RAM_ENEMY_X_PAGE + slot]) << 8)
                | int(ram[_RAM_ENEMY_X_LOW + slot])
            )
            enemy_y_screen = int(ram[_RAM_ENEMY_Y + slot])
            # Convert to grid coordinates relative to Mario.
            ex_grid = (enemy_x_world - mario_x_world) // _TILE_SIZE + _MARIO_GX
            ey_grid = (enemy_y_screen - mario_y_screen) // _TILE_SIZE + _MARIO_GY
            if 0 <= ex_grid < _GRID_W and 0 <= ey_grid < _GRID_H:
                out[ey_grid * _GRID_W + ex_grid] = _TILE_ENEMY

        # Scalars. Velocities are signed bytes; cast through int8 view
        # to interpret correctly. The remaining are unsigned but small
        # enough to fit int8 without overflow.
        out[_GRID_W * _GRID_H + 0] = np.int8(np.int8(ram[_RAM_VEL_X]))
        out[_GRID_W * _GRID_H + 1] = np.int8(np.int8(ram[_RAM_VEL_Y]))
        # `on_ground`: SMB stores 0 = on ground, nonzero = airborne.
        # Invert to the more natural "1 means on ground" convention.
        out[_GRID_W * _GRID_H + 2] = 1 if ram[_RAM_ON_GROUND] == 0 else 0
        out[_GRID_W * _GRID_H + 3] = min(int(ram[_RAM_POWERUP_STATE]), 127)
        out[_GRID_W * _GRID_H + 4] = min(int(ram[_RAM_LIVES]), 127)
        # Sub-tile X position: encodes timing precision the grid can't
        # see. e.g. "Mario is 3 pixels into a tile" matters for jumps.
        out[_GRID_W * _GRID_H + 5] = int(ram[_RAM_X_LOW]) % _TILE_SIZE

        return out

    @staticmethod
    def _tile_at(ram: np.ndarray, world_x: int, world_y: int) -> int:
        """Look up the metatile at a world pixel coordinate.

        SMB's tile RAM is two pages of 13 rows × 16 columns each
        (416 bytes total). Pages alternate as Mario scrolls right;
        which page a given world X falls into depends on the high
        bits of the world coordinate.

        Returns `_TILE_SOLID` for any non-zero metatile (effectively
        any block, pipe, ground, brick, question-mark block...) and
        `_TILE_EMPTY` for empty space. We don't distinguish block
        types because the agent doesn't need to — "thing in the way"
        is the only useful boolean for path-planning.
        """
        # World Y above the top of the level (negative) is sky — empty.
        # Below the level (>= 13 rows × 16 px) is "solid floor" — though
        # in practice Mario can't get there anyway.
        if world_y < 0:
            return _TILE_EMPTY
        # Convert world pixels to tile coordinates within the SMB
        # vertical layout (subtract status bar offset of 32 px).
        sub_y = (world_y - 32) // _TILE_SIZE
        if sub_y < 0 or sub_y >= 13:
            return _TILE_EMPTY
        # Page-and-column math: each page is 16 cols wide. Two pages
        # alternate in the buffer so the level scrolls horizontally.
        page = (world_x // 256) % 2
        sub_x = (world_x % 256) // _TILE_SIZE
        if sub_x < 0 or sub_x >= 16:
            return _TILE_EMPTY
        addr = _RAM_TILE_BASE + page * (13 * 16) + sub_y * 16 + sub_x
        if 0 <= addr < ram.shape[0] and ram[addr] != 0:
            return _TILE_SOLID
        return _TILE_EMPTY


FEATURE_DIM_V2 = FEATURE_DIM + 3


class SMBTileObservationV2(SMBTileObservation):
    """v1 features + 3 de-aliasing scalars (opt-in: `encoder: smb_tiles_pos`).

    The 13×13 window aliases badly in precision compounds: identical
    local tiles at different level positions / hazard phases demand
    different actions, capping BC accuracy and killing argmax play at
    e.g. 2-1's pit/piranha gauntlet. v2 appends:

      * ``out[175]`` level-progress page  (global_x >> 8, clamped 0..127)
      * ``out[176]`` fine progress        ((global_x & 0xFF) >> 1)
      * ``out[177]`` frame-counter phase  (RAM[$0009] >> 1) — the phase
        driver for firebars / piranhas / walk cycles.

    v1 checkpoints keep their contract; nets trained on v2 are 178-wide.
    Rust-native dispatch (`nes_core.extract_smb_tiles_v2`) with a numpy
    fallback composed from the v1 python path.
    """

    def __init__(self, use_rust: bool = True) -> None:
        super().__init__(use_rust=use_rust)
        self._extract_rust_v2 = None
        if use_rust:
            try:
                import nes_core
                if hasattr(nes_core, "extract_smb_tiles_v2"):
                    self._extract_rust_v2 = nes_core.extract_smb_tiles_v2
            except ImportError as exc:
                log.warning(
                    "nes_core native extension unavailable (%s); falling back "
                    "to the numpy tile-extraction path (~55x slower). If a "
                    "build was expected, the compiled extension may be "
                    "missing or ABI-mismatched.",
                    exc,
                )

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM_V2

    def extract(self, ram_bytes: bytes) -> np.ndarray:
        if self._extract_rust_v2 is not None:
            return np.asarray(self._extract_rust_v2(ram_bytes))
        v1 = super().extract(ram_bytes)
        ram = np.frombuffer(ram_bytes, dtype=np.uint8)
        gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
        extra = np.array(
            [min(gx >> 8, 127), (gx & 0xFF) >> 1, int(ram[0x0009]) >> 1],
            dtype=np.int8,
        )
        return np.concatenate([v1, extra])
