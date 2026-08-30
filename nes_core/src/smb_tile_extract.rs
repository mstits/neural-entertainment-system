//! Native Rust implementation of the SMB tile-feature extractor.
//!
//! Companion to `src/emulation/tile_observations/smb.py`. The Python
//! version was profiled at ~7% of trainer wall-time (815k inner-loop
//! iterations across a 3-gen run); even after the numpy vectorization
//! shipped in commit 4ea3610, it's still ~25 μs/call. The Rust port
//! brings that down to single-digit microseconds because compiled
//! Rust eliminates the numpy ufunc dispatch + the Python interpreter
//! overhead entirely.
//!
//! Why NOT NEON intrinsics: the workload is memory-bound (169
//! gathered byte loads from a 2 KB RAM array), not compute-bound.
//! Apple Silicon's load ports issue 3-4 loads per cycle to L1 in
//! parallel, so plain scalar Rust already extracts most of the
//! available throughput. NEON intrinsics could squeeze out a bit
//! more but at the cost of platform-gated code; not worth it for
//! this inner loop.
//!
//! **Parity is the design contract**: the output of this function
//! must be byte-exact with the Python `SMBTileObservation.extract`
//! for any RAM input. A property-based test in
//! `tests/test_smb_tile_extract_parity.py` enforces that across
//! tens of thousands of random RAM snapshots.

#![allow(clippy::too_many_arguments)]

// RAM addresses — kept in sync with `src/emulation/tile_observations/smb.py`.
// If any address shifts, the Python file is the source of truth and
// this constant block must be updated to match.
const RAM_X_PAGE: usize = 0x006D;
const RAM_X_LOW: usize = 0x0086;
const RAM_Y: usize = 0x00CE;
const RAM_VEL_X: usize = 0x0057;
const RAM_VEL_Y: usize = 0x009F;
const RAM_ON_GROUND: usize = 0x001D;
const RAM_POWERUP_STATE: usize = 0x0756;
const RAM_LIVES: usize = 0x075A;
const RAM_TILE_BASE: usize = 0x0500;

const NUM_ENEMY_SLOTS: usize = 5;
const RAM_ENEMY_FLAG: usize = 0x000F;
const RAM_ENEMY_X_PAGE: usize = 0x006E;
const RAM_ENEMY_X_LOW: usize = 0x0087;
const RAM_ENEMY_Y: usize = 0x00CF;

const GRID_W: usize = 13;
const GRID_H: usize = 13;
const MARIO_GX: i32 = 6;
const MARIO_GY: i32 = 6;
const TILE_SIZE: i32 = 16;

const NUM_SCALARS: usize = 6;
pub const FEATURE_DIM: usize = GRID_W * GRID_H + NUM_SCALARS;

#[allow(dead_code)] // referenced from test module, invisible to cargo check
const TILE_EMPTY: i8 = 0;
const TILE_SOLID: i8 = 1;
const TILE_ENEMY: i8 = -1;

/// Decode a 2 KB SMB CPU RAM snapshot into a flat 175-element int8
/// feature vector. Layout matches the Python implementation:
///
/// * `out[0..169]` — 13×13 tile grid centered on Mario, row-major.
///   0 = empty, 1 = solid, -1 = enemy overlay.
/// * `out[169]` — Mario X velocity (signed byte)
/// * `out[170]` — Mario Y velocity (signed byte)
/// * `out[171]` — `on_ground` flag (1 if RAM[$001D]=0 else 0)
/// * `out[172]` — powerup state (clamped to [0, 127])
/// * `out[173]` — lives (clamped to [0, 127])
/// * `out[174]` — sub-tile X position (RAM[$0086] mod 16)
///
/// Out-of-bounds tile lookups (above the level ceiling, off the
/// page edge) report 0 (empty), matching the Python behavior.
pub fn extract(ram: &[u8]) -> [i8; FEATURE_DIM] {
    debug_assert!(
        ram.len() >= 0x0800,
        "ram slice too short — SMB needs the full 2 KB CPU RAM page"
    );
    let mut out = [0i8; FEATURE_DIM];

    // Mario's world coordinates.
    let mario_x_world: i32 =
        ((ram[RAM_X_PAGE] as i32) << 8) | (ram[RAM_X_LOW] as i32);
    let mario_y_screen: i32 = ram[RAM_Y] as i32;

    // 13×13 tile grid lookup. Each cell is a separate small computation
    // + a single byte gather from RAM. Apple Silicon's load ports issue
    // multiple loads per cycle, so the loop is roughly bandwidth-bound
    // on L1D fetches; the `out[i] = solid` write is sequential into a
    // small stack array that lives entirely in L1.
    for cell_idx in 0..(GRID_W * GRID_H) {
        let dy = (cell_idx / GRID_W) as i32 - MARIO_GY;
        let dx = (cell_idx % GRID_W) as i32 - MARIO_GX;
        let world_px_x = mario_x_world + dx * TILE_SIZE;
        let world_px_y = mario_y_screen + dy * TILE_SIZE;

        // Same validity logic as `_tile_at` in the Python version. Left of
        // world origin (world_px_x < 0) must be invalid: rem_euclid below
        // would otherwise wrap a negative x into a phantom page-1 tile.
        if world_px_x < 0 || world_px_y < 0 {
            continue;
        }
        let sub_y = (world_px_y - 32).div_euclid(TILE_SIZE);
        if sub_y < 0 || sub_y >= 13 {
            continue;
        }
        // Python uses `world_x // 256 % 2` for page; in Rust we want
        // floor-div semantics matching Python on negatives.
        let page = world_px_x.div_euclid(256).rem_euclid(2);
        let sub_x = world_px_x.rem_euclid(256).div_euclid(TILE_SIZE);
        if sub_x < 0 || sub_x >= 16 {
            continue;
        }
        let addr = RAM_TILE_BASE
            + (page as usize) * (13 * 16)
            + (sub_y as usize) * 16
            + (sub_x as usize);
        if addr < ram.len() && ram[addr] != 0 {
            out[cell_idx] = TILE_SOLID;
        }
    }

    // Enemy overlay — sparse (at most 5 active slots), so a small
    // scalar loop. Each iteration writes a single grid cell to
    // `_TILE_ENEMY` if the enemy maps inside the visible window.
    for slot in 0..NUM_ENEMY_SLOTS {
        if ram[RAM_ENEMY_FLAG + slot] == 0 {
            continue;
        }
        let enemy_x_world: i32 = ((ram[RAM_ENEMY_X_PAGE + slot] as i32) << 8)
            | (ram[RAM_ENEMY_X_LOW + slot] as i32);
        let enemy_y_screen: i32 = ram[RAM_ENEMY_Y + slot] as i32;
        let ex_grid = (enemy_x_world - mario_x_world).div_euclid(TILE_SIZE) + MARIO_GX;
        let ey_grid = (enemy_y_screen - mario_y_screen).div_euclid(TILE_SIZE) + MARIO_GY;
        if (0..GRID_W as i32).contains(&ex_grid) && (0..GRID_H as i32).contains(&ey_grid) {
            out[(ey_grid as usize) * GRID_W + (ex_grid as usize)] = TILE_ENEMY;
        }
    }

    // Scalars. Velocities are signed bytes — interpret as i8.
    out[GRID_W * GRID_H + 0] = ram[RAM_VEL_X] as i8;
    out[GRID_W * GRID_H + 1] = ram[RAM_VEL_Y] as i8;
    out[GRID_W * GRID_H + 2] = if ram[RAM_ON_GROUND] == 0 { 1 } else { 0 };
    out[GRID_W * GRID_H + 3] = (ram[RAM_POWERUP_STATE].min(127)) as i8;
    out[GRID_W * GRID_H + 4] = (ram[RAM_LIVES].min(127)) as i8;
    out[GRID_W * GRID_H + 5] = (ram[RAM_X_LOW] % (TILE_SIZE as u8)) as i8;

    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ram_yields_empty_grid_with_on_ground_flag() {
        let ram = vec![0u8; 2048];
        let out = extract(&ram);
        // Tile grid is empty.
        for i in 0..GRID_W * GRID_H {
            assert_eq!(out[i], 0, "cell {i} should be empty");
        }
        // on_ground flag is 1 because RAM[$001D] = 0 means "on ground".
        assert_eq!(out[171], 1);
        // All scalars are zero in empty RAM (except on_ground above).
        assert_eq!(out[169], 0); // vel_x
        assert_eq!(out[170], 0); // vel_y
        assert_eq!(out[172], 0); // powerup
        assert_eq!(out[173], 0); // lives
        assert_eq!(out[174], 0); // sub-tile X
    }

    #[test]
    fn mario_velocity_signed_round_trip() {
        let mut ram = vec![0u8; 2048];
        ram[RAM_VEL_X] = 0xF0; // -16 as signed byte
        let out = extract(&ram);
        assert_eq!(out[169], -16);
    }

    #[test]
    fn enemy_overlay_marks_grid_cell() {
        let mut ram = vec![0u8; 2048];
        ram[RAM_X_PAGE] = 0;
        ram[RAM_X_LOW] = 96;
        ram[RAM_Y] = 96;
        // Enemy at +2 tiles right of Mario.
        ram[RAM_ENEMY_FLAG] = 1;
        ram[RAM_ENEMY_X_PAGE] = 0;
        ram[RAM_ENEMY_X_LOW] = 96 + 32;
        ram[RAM_ENEMY_Y] = 96;
        let out = extract(&ram);
        assert_eq!(out[6 * GRID_W + 8], TILE_ENEMY);
    }

    #[test]
    fn solid_tile_below_mario() {
        let mut ram = vec![0u8; 2048];
        ram[RAM_X_PAGE] = 0;
        ram[RAM_X_LOW] = 96;
        ram[RAM_Y] = 96;
        // Place a non-zero metatile at sub_y=7 in tile RAM. The Python
        // mapping says this corresponds to grid (6, 9).
        ram[RAM_TILE_BASE + 7 * 16 + 6] = 0x42;
        let out = extract(&ram);
        assert_eq!(out[9 * GRID_W + 6], TILE_SOLID);
        assert_eq!(out[7 * GRID_W + 6], TILE_EMPTY);
    }
}

/// v2 feature width: the 175 v1 features + 3 de-aliasing scalars.
pub const FEATURE_DIM_V2: usize = FEATURE_DIM + 3;

/// v2 extractor: the v1 layout plus three scalars that dissolve the
/// observation aliasing v1 suffers in precision compounds (identical
/// local tile windows at different level positions / hazard phases):
///
/// * `out[175]` — level-progress page (`global_x / 256`, clamped 0..127):
///   coarse absolute position within the level.
/// * `out[176]` — fine progress (`(global_x % 256) / 2`, 0..127).
/// * `out[177]` — global frame counter / 2 (`RAM[$0009] >> 1`, 0..127):
///   the phase driver for firebars / piranhas / enemy walk cycles.
///
/// v1 (`extract`) is untouched — every shipped checkpoint keeps its
/// contract; v2 is opt-in per profile (`encoder: smb_tiles_pos`).
pub fn extract_v2(ram: &[u8]) -> [i8; FEATURE_DIM_V2] {
    let v1 = extract(ram);
    let mut out = [0i8; FEATURE_DIM_V2];
    out[..FEATURE_DIM].copy_from_slice(&v1);
    let gx: i32 = ((ram[RAM_X_PAGE] as i32) << 8) | (ram[RAM_X_LOW] as i32);
    out[FEATURE_DIM] = (gx >> 8).min(127) as i8;
    out[FEATURE_DIM + 1] = ((gx & 0xFF) >> 1) as i8;
    out[FEATURE_DIM + 2] = (ram[0x0009] >> 1) as i8;
    out
}
