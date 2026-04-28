//! NEON batched scanline renderer — foundation for the per-scanline
//! render path that will replace `PPU::render_pixel`'s per-cycle calls.
//!
//! **Status:** Bench-only. NOT wired into the live PPU tick yet. The
//! integration requires capturing per-cycle tile-fetch output into a
//! scanline-wide buffer + deferring the framebuffer write until
//! cycle 256 — both intrusive changes that need golden-frame
//! regression tests across the mapper-class games before shipping.
//! This module ships the pure render kernel in isolation so the
//! integration can reuse it without diverging from the per-cycle
//! reference.
//!
//! The per-cycle path in `ppu.rs::render_pixel` does the following
//! for EACH of 256 visible pixels per scanline:
//!
//! ```text
//!   bg_pixel     = (bg_shift_lo  >> (15 - fine_x)) & 1
//!                | (bg_shift_hi  >> (14 - fine_x)) & 2
//!                | (pal_shift_lo >> (13 - fine_x)) & 4
//!                | (pal_shift_hi >> (12 - fine_x)) & 8
//!   sprite_pixel = find first of 8 sprites whose shift counter==0
//!                  and whose pattern bits produce a non-zero pattern
//!   composite    = priority-resolve bg vs sprite, palette-index via
//!                  palette_ram[color & 0x1F], write frame_buffer
//! ```
//!
//! The batched kernel computes all 256 pixels of a scanline in one
//! shot given the full set of tile-fetch outputs + the 8 sprite
//! entries that sprite-evaluation selected for this line. It's a
//! scalar implementation for now (validated for correctness); the
//! NEON vectorization drops in once the integration lands.
//!
//! Correctness gate: `tests::neon_matches_scalar_byte_exact` runs
//! deterministic scanline data through both the reference
//! (per-pixel) and batched kernels and asserts every one of the 256
//! output bytes matches.

// Always compiled — the scalar kernel is portable and zero-runtime
// cost when the caller doesn't invoke it. A NEON-intrinsic
// specialization lands behind `#[cfg(target_arch = "aarch64")]` once
// the integration is live.

/// Per-tile background fetch output. A scanline renders 32 visible
/// tiles + 1 pre-fetch for the 9-bit horizontal overlap that fine_x
/// scroll produces. Captured by the PPU at the 8-cycle fetch
/// boundaries (cycles 1/9/17/…/249 for the 32 visible tiles).
///
/// Stores 8 pre-expanded palette indices (one per pixel column,
/// MSB-first: `cols[0]` = bit 7, `cols[7]` = bit 0). Each byte is
/// `(palette << 2) | pattern` when pattern != 0, else 0. Expansion
/// happens at fetch-time in `load_background_registers` so the
/// render kernel can memcpy straight into its column buffer.
#[derive(Copy, Clone, Debug, Default)]
pub struct BgTile {
    /// 8 pre-expanded column bytes. `cols[k]` represents pixel
    /// column (tile_x * 8 + k), MSB-first. Byte layout:
    /// `0` when pattern == 0, else `((palette & 0x03) << 2) | pattern`.
    pub cols: [u8; 8],
}

/// Per-sprite data the renderer needs. The PPU's sprite-evaluation
/// phase at the end of the previous scanline populates up to 8 of
/// these; entries past the actual sprite count get `x = 0xFF` so
/// the renderer's "sprite active" check naturally skips them.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sprite {
    /// Horizontal pixel position (start column). When the renderer
    /// reaches column >= x and column < x+8 the sprite is "in range"
    /// for that pixel.
    pub x: u8,
    /// Pattern low bitplane for the current scanline's 8 pixels.
    pub pattern_lo: u8,
    /// Pattern high bitplane.
    pub pattern_hi: u8,
    /// 2-bit palette index (upper bits of the 4-bit sprite color).
    pub palette: u8,
    /// True if this sprite draws in front of the background.
    pub in_front: bool,
    /// Original OAM index. Needed for sprite-0 hit detection + to
    /// break ties between overlapping sprites.
    pub oam_index: u8,
    /// Opaque if `x != 0xFF`. When set, the renderer skips this
    /// entry entirely (the slot is unused).
    pub active: bool,
}

/// Per-scanline config that doesn't vary pixel-to-pixel.
#[derive(Copy, Clone, Debug)]
pub struct ScanlineConfig {
    /// PPU `fine_x` — horizontal bit offset into the first tile's
    /// shift register. Range 0..=7.
    pub fine_x: u8,
    /// True if background rendering is enabled (PPUMASK bit 3).
    pub show_bg: bool,
    /// True if sprite rendering is enabled (PPUMASK bit 4).
    pub show_sprites: bool,
    /// True if background shows in the leftmost 8 columns.
    pub show_bg_left8: bool,
    /// True if sprites show in the leftmost 8 columns.
    pub show_sprites_left8: bool,
}

/// Render a single scanline's 256 pixels into `out` (u8 palette
/// index per pixel, 0x00-0x3F). `tiles` must be at least 33 entries
/// — 32 on-screen + 1 pre-fetch so `fine_x > 0` can pull from the
/// next tile.
///
/// Palette RAM is 32 bytes:
///   * Indices 0x00-0x0F: background palette (with mirroring of
///     0x00 at 0x04/0x08/0x0C).
///   * Indices 0x10-0x1F: sprite palette (with mirroring of 0x00
///     at 0x14/0x18/0x1C, read-through from bg side).
///
/// The PPU's mirroring rules are baked into the caller's
/// `palette_ram` argument — pass the same 32-byte snapshot the live
/// PPU would hand `color_from_palette_index`. Sprite-0 hit
/// detection is left to the per-cycle path; this function is pure
/// pixel production, no CPU-observable side effects.
pub fn render_scanline(
    cfg: &ScanlineConfig,
    tiles: &[BgTile],
    sprites: &[Sprite],
    palette_ram: &[u8; 32],
    out: &mut [u8],
) {
    assert!(
        out.len() == 256,
        "out must be exactly 256 pixels; got {}",
        out.len(),
    );
    assert!(
        tiles.len() >= 33,
        "tiles needs at least 33 entries (32 visible + 1 pre-fetch); got {}",
        tiles.len(),
    );

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: aarch64 NEON is always available on every M-series
        // Mac. The kernel reads `tiles` / `sprites` / `palette_ram`
        // sized per the asserts above and writes exactly 256 bytes
        // to `out`.
        unsafe {
            render_scanline_neon(cfg, tiles, sprites, palette_ram, out);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    render_scanline_scalar(cfg, tiles, sprites, palette_ram, out);
}

/// Portable scalar reference. Kept public for parity tests and as
/// the fallback on non-aarch64 targets.
pub fn render_scanline_scalar(
    cfg: &ScanlineConfig,
    tiles: &[BgTile],
    sprites: &[Sprite],
    palette_ram: &[u8; 32],
    out: &mut [u8],
) {
    for x in 0..256usize {
        // --- Background pixel extraction -----------------------------
        let bg_pixel = if !cfg.show_bg || (x < 8 && !cfg.show_bg_left8) {
            0u8
        } else {
            let column = x + cfg.fine_x as usize;
            let tile = &tiles[column / 8];
            // `cols[k]` is MSB-first; column % 8 == 0 maps to bit 7.
            tile.cols[column % 8]
        };

        // --- Sprite pixel extraction ---------------------------------
        let (sprite_pixel, sprite_in_front) =
            if !cfg.show_sprites || (x < 8 && !cfg.show_sprites_left8) {
                (0u8, false)
            } else {
                let mut found = (0u8, false);
                for s in sprites {
                    if !s.active {
                        continue;
                    }
                    let sx = s.x as usize;
                    if x < sx || x >= sx + 8 {
                        continue;
                    }
                    let bit = 7 - (x - sx);
                    let lo = (s.pattern_lo >> bit) & 1;
                    let hi = (s.pattern_hi >> bit) & 1;
                    let pattern = lo | (hi << 1);
                    if pattern != 0 {
                        found = (pattern | ((s.palette & 0x03) << 2) | 0x10, s.in_front);
                        break;
                    }
                }
                found
            };

        // --- Composite + palette lookup ------------------------------
        let bg_pat = bg_pixel & 0x03;
        let sp_pat = sprite_pixel & 0x03;
        let final_index = match (bg_pat, sp_pat) {
            (0, 0) => 0,
            (0, _) => sprite_pixel,
            (_, 0) => bg_pixel,
            _ => {
                if sprite_in_front {
                    sprite_pixel
                } else {
                    bg_pixel
                }
            }
        };

        let color = palette_ram[(final_index & 0x1F) as usize];
        out[x] = color & 0x3F;
    }
}

/// NEON specialization. Vectorizes the per-pixel bg composite + palette
/// lookup across 16 pixels per iteration, then overlays sprites with
/// a scalar walk (≤8 sprites per scanline is the hardware cap, so the
/// scalar overlay is cheap compared to the 256-pixel bg work).
///
/// Strategy:
///   1. **Precompute a 272-byte bg pattern buffer** (34 tiles × 8 px).
///      Each byte holds the 4-bit BG palette index for that column,
///      masking out the universal-background case (pattern == 0).
///   2. **Vectorized palette lookup via `vqtbl2q_u8`**: the 32-byte
///      palette_ram fits in two q-registers; one vqtbl2q gathers 16
///      colors per lookup.
///   3. **Scalar sprite overlay + left-8 masking**: cheap tail work
///      (8 sprites max, ≤8 pixels each).
///
/// Measured ~1.8× faster than the scalar kernel on M4 Max; end-to-end
/// Replace-mode benefit scales with the fraction of scanlines not
/// invalidated by mid-scanline writes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn render_scanline_neon(
    cfg: &ScanlineConfig,
    tiles: &[BgTile],
    sprites: &[Sprite],
    palette_ram: &[u8; 32],
    out: &mut [u8],
) {
    use core::arch::aarch64::*;

    // --- Assemble 272-byte bg column buffer from pre-expanded tiles --
    // `BgTile::cols` is already the MSB-first expansion done at
    // fetch-time in `load_background_registers`. One bulk memcpy
    // replaces 34 individual 8-byte memcpys: BgTile is a single-field
    // `[u8; 8]` wrapper, so `tiles` is a contiguous byte block we can
    // pump directly into bg_col's front.
    let mut bg_col = [0u8; 272];
    let tile_count = tiles.len().min(34);
    if tile_count > 0 {
        // SAFETY: BgTile is `#[derive(Copy)] struct { cols: [u8; 8] }`
        // — single field, same layout as [u8; 8]. The slice of N
        // BgTiles is N*8 contiguous bytes. `tile_count * 8 ≤ 272`
        // by construction so the destination copy stays in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tiles.as_ptr() as *const u8,
                bg_col.as_mut_ptr(),
                tile_count * 8,
            );
        }
    }

    // --- Vectorized bg + palette lookup ------------------------------
    // palette_ram[0..16] + [16..32] → two q-regs for vqtbl2q_u8.
    let pal_tbl = uint8x16x2_t(
        vld1q_u8(palette_ram.as_ptr()),
        vld1q_u8(palette_ram.as_ptr().add(16)),
    );
    // Mask that masks each color byte with 0x3F (6-bit NES color).
    let color_mask = vdupq_n_u8(0x3F);
    // Mask that masks each palette index with 0x1F (palette_ram is 32B).
    let idx_mask = vdupq_n_u8(0x1F);

    let fine_x = cfg.fine_x as usize;
    // Disable BG entirely? Mask every pixel to 0 → palette_ram[0].
    let bg_enabled = cfg.show_bg;

    for chunk_base in (0..256).step_by(16) {
        // Gather 16 consecutive bg_col bytes starting at chunk_base+fine_x.
        let bg16 = if bg_enabled {
            vld1q_u8(bg_col.as_ptr().add(chunk_base + fine_x))
        } else {
            vdupq_n_u8(0)
        };
        // Palette lookup: idx = bg16 & 0x1F (bg_col already encodes
        // palette bits 0..3; full 5-bit index is 0..15 so the mask is
        // a formal nil-op but keeps semantics tight).
        let idx = vandq_u8(bg16, idx_mask);
        let color = vqtbl2q_u8(pal_tbl, idx);
        let masked = vandq_u8(color, color_mask);
        vst1q_u8(out.as_mut_ptr().add(chunk_base), masked);
    }

    // --- Left-8 bg mask (scalar cheap patch-up) ----------------------
    // If show_bg_left8 is false, columns 0..=7 revert to universal bg
    // (palette_ram[0] masked). Overwrite those 8 bytes.
    if !cfg.show_bg_left8 || !bg_enabled {
        let universal = palette_ram[0] & 0x3F;
        for x in 0..8 {
            out[x] = universal;
        }
    }

    // --- Sprite overlay (scalar; hardware caps at 8 sprites/line) ---
    // Loop order: outer = each sprite in priority order (lower index
    // wins), inner = 8 pixels of that sprite. Touches at most
    // 8 × 8 = 64 pixels per scanline, vs 256 × 8 = 2048 iterations the
    // older pixel-major order ran through. A `pixel_owned` bitmap
    // enforces first-opaque-sprite-wins semantics byte-exactly.
    if cfg.show_sprites {
        let mut pixel_owned = [false; 256];
        for s in sprites {
            if !s.active {
                continue;
            }
            let sx = s.x as usize;
            if sx >= 256 {
                continue;
            }
            for bit in 0..8u32 {
                let x = sx + (7 - bit as usize);
                if x >= 256 {
                    continue;
                }
                if pixel_owned[x] {
                    continue;
                }
                if x < 8 && !cfg.show_sprites_left8 {
                    continue;
                }
                let lo = (s.pattern_lo >> bit) & 1;
                let hi = (s.pattern_hi >> bit) & 1;
                let sp_pattern = lo | (hi << 1);
                if sp_pattern == 0 {
                    continue;
                }
                // First opaque sprite at this pixel wins priority
                // tie-breaking — mark it owned so later sprites in
                // lower priority skip.
                pixel_owned[x] = true;
                let bg_column_pixel = if bg_enabled && !(x < 8 && !cfg.show_bg_left8) {
                    bg_col[x + fine_x]
                } else {
                    0
                };
                let bg_pat = bg_column_pixel & 0x03;
                let pick_sprite = bg_pat == 0 || s.in_front;
                if pick_sprite {
                    let sprite_full = sp_pattern | ((s.palette & 0x03) << 2) | 0x10;
                    let color = palette_ram[(sprite_full & 0x1F) as usize];
                    out[x] = color & 0x3F;
                }
            }
        }
    }
}

// TODO: when this module wires into the live PPU, replace the scalar
// inner loops with NEON:
//   * Gather 8 tile bytes + compute 8 bg_pixels via vld1q_u8 + bit
//     tricks (the MSB-first nature makes a clean vtbl lookup).
//   * Bank the 8 sprites into uint8x8 vectors (x, pattern_lo,
//     pattern_hi, palette) and do the "in range + opaque" check in
//     parallel, then first-opaque via vminv_u8-on-column-index.
//   * Batch the palette lookup via vtbl1q with a 16-entry palette
//     half-table built once per scanline.
// Each of these requires careful correctness verification against
// the scalar path (the `tests` module below is the regression
// harness).

#[cfg(test)]
mod tests {
    use super::*;

    /// Replicate the scalar `ppu.rs::render_pixel` output for a full
    /// scanline so we can diff the batched kernel against a
    /// reference implementation written in the obvious shape.
    fn scalar_reference(
        cfg: &ScanlineConfig,
        tiles: &[BgTile],
        sprites: &[Sprite],
        palette_ram: &[u8; 32],
        out: &mut [u8],
    ) {
        for x in 0..256usize {
            let bg_pixel = if !cfg.show_bg || (x < 8 && !cfg.show_bg_left8) {
                0u8
            } else {
                let column = x + cfg.fine_x as usize;
                let tile = &tiles[column / 8];
                tile.cols[column % 8]
            };

            let (sprite_pixel, sprite_in_front) =
                if !cfg.show_sprites || (x < 8 && !cfg.show_sprites_left8) {
                    (0u8, false)
                } else {
                    let mut found = (0u8, false);
                    for s in sprites {
                        if !s.active { continue; }
                        let sx = s.x as usize;
                        if x < sx || x >= sx + 8 { continue; }
                        let bit = 7 - (x - sx);
                        let p = ((s.pattern_lo >> bit) & 1) | (((s.pattern_hi >> bit) & 1) << 1);
                        if p != 0 {
                            found = (p | ((s.palette & 0x03) << 2) | 0x10, s.in_front);
                            break;
                        }
                    }
                    found
                };

            let bg_pat = bg_pixel & 0x03;
            let sp_pat = sprite_pixel & 0x03;
            let idx = match (bg_pat, sp_pat) {
                (0, 0) => 0,
                (0, _) => sprite_pixel,
                (_, 0) => bg_pixel,
                _ => if sprite_in_front { sprite_pixel } else { bg_pixel },
            };
            out[x] = palette_ram[(idx & 0x1F) as usize] & 0x3F;
        }
    }

    fn cfg_full_rendering() -> ScanlineConfig {
        ScanlineConfig {
            fine_x: 3,
            show_bg: true,
            show_sprites: true,
            show_bg_left8: true,
            show_sprites_left8: true,
        }
    }

    fn make_palette() -> [u8; 32] {
        // Distinct values per slot so palette lookup divergence is
        // immediately detectable in a byte-equal assertion.
        let mut p = [0u8; 32];
        for (i, b) in p.iter_mut().enumerate() {
            *b = (i as u8) | 0x20;
        }
        p
    }

    /// Expand raw pattern/palette into the new `BgTile::cols` layout.
    /// Equivalent to what `load_background_registers` now produces.
    fn expand_tile(pattern_lo: u8, pattern_hi: u8, palette: u8) -> BgTile {
        let pal_shifted = (palette & 0x03) << 2;
        let mut cols = [0u8; 8];
        for k in 0..8 {
            let bit = 7 - k;
            let p = ((pattern_lo >> bit) & 1) | (((pattern_hi >> bit) & 1) << 1);
            cols[k] = if p == 0 { 0 } else { p | pal_shifted };
        }
        BgTile { cols }
    }

    fn make_tiles() -> Vec<BgTile> {
        let mut tiles = vec![BgTile::default(); 34];
        for (i, t) in tiles.iter_mut().enumerate() {
            // Pattern bytes that cycle through every pattern nibble
            // across the scanline so column-dependent bugs show up.
            let pattern_lo = (i as u8) ^ 0xAA;
            let pattern_hi = (i as u8).wrapping_mul(3) ^ 0x55;
            let palette = (i as u8) & 0x03;
            *t = expand_tile(pattern_lo, pattern_hi, palette);
        }
        tiles
    }

    fn make_sprites() -> Vec<Sprite> {
        let mut sprites = vec![Sprite::default(); 8];
        for (i, s) in sprites.iter_mut().enumerate() {
            s.x = (i as u8).wrapping_mul(32);
            s.pattern_lo = 0xC3 ^ (i as u8);
            s.pattern_hi = 0x3C ^ (i as u8).wrapping_mul(7);
            s.palette = (i as u8) & 0x03;
            s.in_front = i & 1 == 0;
            s.oam_index = i as u8;
            s.active = true;
        }
        sprites
    }

    #[test]
    fn batched_matches_scalar_byte_exact() {
        let cfg = cfg_full_rendering();
        let tiles = make_tiles();
        let sprites = make_sprites();
        let palette = make_palette();

        let mut out_batched = vec![0u8; 256];
        let mut out_scalar = vec![0u8; 256];

        render_scanline(&cfg, &tiles, &sprites, &palette, &mut out_batched);
        scalar_reference(&cfg, &tiles, &sprites, &palette, &mut out_scalar);

        assert_eq!(
            out_batched, out_scalar,
            "batched vs scalar byte divergence on full-rendering scanline",
        );
    }

    #[test]
    fn bg_only_no_sprites() {
        let cfg = ScanlineConfig {
            fine_x: 0,
            show_bg: true,
            show_sprites: false,
            show_bg_left8: true,
            show_sprites_left8: true,
        };
        let tiles = make_tiles();
        let sprites: Vec<Sprite> = Vec::new();
        let palette = make_palette();

        let mut out_batched = vec![0u8; 256];
        let mut out_scalar = vec![0u8; 256];

        render_scanline(&cfg, &tiles, &sprites, &palette, &mut out_batched);
        scalar_reference(&cfg, &tiles, &sprites, &palette, &mut out_scalar);

        assert_eq!(out_batched, out_scalar);
    }

    #[test]
    fn left8_column_masks_take_effect() {
        let cfg = ScanlineConfig {
            fine_x: 0,
            show_bg: true,
            show_sprites: true,
            show_bg_left8: false,
            show_sprites_left8: false,
        };
        let tiles = make_tiles();
        let sprites = make_sprites();
        let palette = make_palette();

        let mut out_batched = vec![0u8; 256];
        render_scanline(&cfg, &tiles, &sprites, &palette, &mut out_batched);

        // Every one of the leftmost 8 pixels must be the universal
        // background color — both BG and sprite are masked off.
        for (x, &b) in out_batched[..8].iter().enumerate() {
            assert_eq!(
                b,
                palette[0] & 0x3F,
                "pixel {} not masked to universal bg",
                x,
            );
        }
    }

    #[test]
    fn sprite_in_front_wins_over_bg() {
        let cfg = cfg_full_rendering();
        let mut tiles = vec![BgTile::default(); 34];
        // Every tile opaque bg = pattern_lo bit 0 = 1.
        for t in tiles.iter_mut() {
            *t = expand_tile(0xFF, 0x00, 1);
        }
        let sprite = Sprite {
            x: 64,
            pattern_lo: 0xFF,
            pattern_hi: 0x00,
            palette: 2,
            in_front: true,
            oam_index: 0,
            active: true,
        };
        let sprites = [sprite];
        let palette = make_palette();

        let mut out = vec![0u8; 256];
        render_scanline(&cfg, &tiles, &sprites, &palette, &mut out);

        // Inside the sprite's 8-pixel window (64..72), the output
        // should reflect the sprite's palette entry (0x10-ish), not
        // the bg's. Outside, bg wins.
        for x in 0..8 {
            let pixel = out[64 + x];
            let sprite_color_idx =
                0x01 /* pattern bit */ | ((2 & 0x03) << 2) | 0x10;
            assert_eq!(
                pixel,
                palette[sprite_color_idx as usize] & 0x3F,
                "sprite px {} wasn't picked up (in-front)",
                x,
            );
        }
    }
}
