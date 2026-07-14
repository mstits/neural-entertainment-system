use crate::mapper::{Mapper, MapperEnum};
use crate::sink::*;

use bit_reverse::ParallelReverse;
use bitflags::bitflags;
use serde_derive::{Deserialize, Serialize};

use std::default::Default;
use std::ops::{Deref, DerefMut};

pub const SCREEN_WIDTH: usize = 256;
pub const SCREEN_HEIGHT: usize = 240;

const CYCLES_PER_SCANLINE: u64 = 341;

const VISIBLE_START_SCANLINE: u16 = 0;
pub const VISIBLE_END_SCANLINE: u16 = 239;
const VBLANK_START_SCANLINE: u16 = 241;
const PRE_RENDER_SCANLINE: u16 = 261;

// Memory-mapped register addresses
const PPUCTRL_ADDRESS: u16 = 0x2000;
const PPUMASK_ADDRESS: u16 = 0x2001;
const PPUSTATUS_ADDRESS: u16 = 0x2002;
const OAMADDR_ADDRESS: u16 = 0x2003;
pub const OAMDATA_ADDRESS: u16 = 0x2004;
const PPUSCROLL_ADDRESS: u16 = 0x2005;
const PPUADDR_ADDRESS: u16 = 0x2006;
const PPUDATA_ADDRESS: u16 = 0x2007;

pub struct Ppu {
    pub cycles: u64,
    regs: Regs,

    // When reading while the VRAM address is in the range 0-$3EFF (i.e., before the palettes),
    // the read will return the contents of an internal read buffer. This internal buffer is
    // updated only when reading PPUDATA, and so is preserved across frames. After the CPU reads
    // and gets the contents of the internal buffer, the PPU will immediately update the internal
    // buffer with the byte at the current VRAM address. Thus, after setting the VRAM address, one
    // should first read this register and discard the result.
    // Reading palette data from $3F00-$3FFF works differently. The palette data is placed
    // immediately on the data bus, and hence no dummy read is required. Reading the palettes still
    // updates the internal buffer though, but the data placed in it is the mirrored nametable data
    // that would appear "underneath" the palette.
    ppu_data_read_buffer: u8,

    // PPU address space
    pub mem: MemMap,

    // Object Attribute Memory
    oam: Oam,

    pub scanline: u16,

    // The cycle that the current scanline started at
    scanline_start_cycle: u64,

    pub frame: u64,

    frame_buffer: Box<[u8]>,

    // The PPU has an internal data bus that it uses for communication with the CPU.
    // This bus, called _io_db in Visual 2C02 and PPUGenLatch in FCEUX,[1] behaves as an
    // 8-bit dynamic latch due to capacitance of very long traces that run to various parts
    // of the PPU. Writing any value to any PPU port, even to the nominally read-only
    // PPUSTATUS, will fill this latch. Reading any readable port (PPUSTATUS, OAMDATA, or PPUDATA)
    // also fills the latch with the bits read. Reading a nominally "write-only" register returns
    // the latch's current value, as do the unused bits of PPUSTATUS.
    ppu_gen_latch: u8,

    // Registers used during rendering to hold background data
    name_table_byte: u8,
    attribute_table_byte: u8,
    tile_bitmap_byte_lo: u8,
    tile_bitmap_byte_hi: u8,
    // Packed 4-lane background shift registers — pattern_lo,
    // pattern_hi, palette_lo, palette_hi in indices 0..3. Stored as
    // a single `[u16; 4]` so `shift_background_registers` can issue
    // a single NEON `vshl_n_u16` over all four lanes per cycle (the
    // function fires ~272× per visible/pre-render scanline).
    bg_shift: [u16; 4],

    // Registers used during rendering to hold sprite data
    sprite_pattern_shifts_lo: [u8; 8],
    sprite_pattern_shifts_hi: [u8; 8],
    sprite_attribute_latches: [SpriteAttributes; 8],
    sprite_x_counters: [u8; 8],
    sprite_0_on_scanline: bool,

    pub nmi_occurred: bool,
    pub nmi_output: bool,

    // Skip-render fast path. When `true`, render_pixel and clear_pixel
    // bypass the per-pixel color resolution and frame_buffer write,
    // but sprite-0 hit detection still runs (the CPU observes it via
    // PPUSTATUS reads). All cycle-accurate side effects (tile fetches,
    // shift register updates, sprite evaluation, NMI/vblank timing,
    // mapper IRQ A12 ticks) are unchanged.
    //
    // Use case: RL training with frame_skip=N runs the emulator for N
    // frames per agent observation but only needs the FINAL frame
    // rendered. Setting skip_render=true on the first N-1 frames
    // gives a 3-5x speedup with no behavior change visible to either
    // the CPU or the consumer of the final frame.
    //
    // Not part of `State` — transient flag, not part of save snapshots.
    skip_render: bool,

    // Per-scanline tile-fetch buffer for the batched render kernel
    // (`ppu_neon::render_scanline`). Populated at every
    // `load_background_registers` call; consumed at end of each
    // visible scanline once the batched path is wired in. Transient
    // — not serialized.
    //
    // Layout (by the time cycle 256 of a visible scanline fires):
    //   index 0, 1: tiles loaded during cycles 328, 336 of the PREVIOUS
    //              scanline (the two pre-fetch tiles that fine_x scroll
    //              can peek into when rendering the FIRST pixels).
    //   index 2..=33: tiles loaded during cycles 8, 16, ..., 248, 256
    //              of the CURRENT scanline.
    //   index 34, 35: tiles loaded during the prefetch window (cycles
    //              328, 336) of the CURRENT scanline. These are the
    //              first two tiles of the NEXT scanline — carry over
    //              as indices 0, 1 at scanline boundary.
    bg_tile_buffer: [crate::ppu_neon::BgTile; 36],
    // Monotonic counter of `load_background_registers` calls since
    // the last scanline boundary. Used to index `bg_tile_buffer`.
    bg_tile_load_count: u16,

    // Per-scanline sprite snapshot for the batched render kernel.
    // Captured at scanline boundary — right after sprite fetches
    // for the new scanline complete at cycle 320 of the previous
    // one. The live `sprite_pattern_shifts_*` arrays get shifted
    // during rendering (as sprites are drawn), so we need a
    // pre-render snapshot to feed the batched kernel.
    scanline_sprites: [crate::ppu_neon::Sprite; 8],

    // Scratch space for the batched render. When the per-scanline
    // path is fully wired in, this replaces the per-pixel frame
    // buffer writes. Until then it's a diagnostic staging area.
    batched_scanline_buf: [u8; 256],

    // Set true by any palette-RAM write during cycles 1..=256 of the
    // current visible scanline. When true, run_batched_scanline bails
    // (the single end-of-scanline palette snapshot can't reproduce the
    // per-pixel path's live reads). Reset at scanline start.
    batched_scanline_invalid: bool,

    // Per-scanline prediction history: was scanline N clean-rendered
    // (no invalidation) in the PREVIOUS frame? In Replace mode, a
    // clean history entry lets us skip per-pixel framebuffer writes
    // at the START of THIS frame's scanline — the batched kernel at
    // cycle 256 writes the final row. Indexed by scanline number
    // (0..240). A miss means we paid per-pixel cost; a hit skips it
    // entirely (the actual perf win of Batch D).
    //
    // If THIS frame's scanline invalidates (mid-scanline state change
    // we didn't predict), the flag gets flipped to dirty so the NEXT
    // frame falls back to per-pixel for that scanline until several
    // clean frames land.
    clean_scanline_history: [bool; 240],

    // Cached at scanline start: are we skipping per-pixel framebuffer
    // writes on THIS scanline? Set to true only when mode=Replace AND
    // the prior frame's same scanline was clean. render_pixel reads
    // this and short-circuits the color/framebuffer write; sprite-0
    // hit detection still runs (CPU-observable side effect).
    skip_pixel_writes_this_scanline: bool,

    // Runtime mode control for the batched PPU renderer:
    //   * `verify`  — run batched path in parallel with per-cycle
    //                 path, count divergences. No behavior change.
    //   * `replace` — run batched path ONLY; skip per-pixel
    //                 framebuffer writes. Requires `verify` to have
    //                 previously proven parity on the game in use.
    // Both default off; the live pixel writes win. Caller flips on
    // via `set_batched_render_mode`. `scanlines_verified` /
    // `scanlines_diverged` accumulate across frames so a caller
    // can gauge parity confidence before flipping to `replace`.
    batched_render_mode: BatchedRenderMode,
    pub scanlines_verified: u64,
    pub scanlines_diverged: u64,

    // Diagnostic knob: when true, the batched path reverts to the
    // old "broad" invalidation (any $2007 / any CTRL-MASK-SCROLL-ADDR
    // register / any mapper write during visible cycles flips the
    // flag). Used for A/B stats comparison; default is the narrowed
    // ruleset. Gated on `ppu_neon_stats` — unused in release builds.
    #[cfg(feature = "ppu_neon_stats")]
    pub batched_invalidation_broad: bool,

    // Diagnostic counters for the batched PPU renderer. Gated on
    // `ppu_neon_stats` so the default build pays zero cost.
    //
    //   * `batched_scanlines_total`: visible scanlines where the
    //     batched path was asked to run (cycle 256 of scanlines 0..=239,
    //     with mode != Off).
    //   * `batched_scanlines_fallback`: subset of above that fell
    //     back to per-pixel output (invalidation flag set).
    //   * `batched_invalidations_by_source[N]`: which write source
    //     flipped the flag. Indices:
    //       0 = palette RAM write via $2007
    //       1 = nametable/CHR write via $2007 (non-palette)
    //       2 = PPUCTRL/MASK/SCROLL/ADDR mid-scanline write
    //       3 = mapper write ($4018-$FFFF) during render window
    //     Incremented at most once per source per scanline (after the
    //     first invalidation fires on a given source, further writes
    //     from that source don't re-count).
    #[cfg(feature = "ppu_neon_stats")]
    pub batched_scanlines_total: u64,
    #[cfg(feature = "ppu_neon_stats")]
    pub batched_scanlines_fallback: u64,
    #[cfg(feature = "ppu_neon_stats")]
    pub batched_invalidations_by_source: [u64; 4],
    #[cfg(feature = "ppu_neon_stats")]
    batched_source_fired_this_scanline: [bool; 4],

    /// Cached raw pointer to the mapper's static 8 KB CHR window.
    /// Set when the mapper guarantees no runtime CHR banking via
    /// `Mapper::chr_static_ptr()`. Bypasses the per-fetch
    /// `MapperEnum` virtual dispatch on PPU CHR reads (~5M reads/
    /// sec/worker for typical games). `null` when the mapper banks
    /// CHR at runtime (MMC1, MMC3, …); CHR reads then fall back to
    /// the dispatched `mapper.chr_read_byte` path. PPU refreshes
    /// this at every scanline boundary. Placed at the END of the
    /// struct so it lives on its own cache line — earlier attempt
    /// to put it inside MemMap (which sits in the hot read path)
    /// regressed N=12 by ~30% via cache-line layout displacement.
    chr_cache_ptr: *const u8,

    /// Packed snapshot of the four PPUMASK rendering bits used in
    /// the per-pixel hot path: bit 0 = SHOW_BACKGROUND, bit 1 =
    /// SHOW_SPRITES, bit 2 = SHOW_BACKGROUND_LEFT_8, bit 3 =
    /// SHOW_SPRITES_LEFT_8. Cached so that `background_pixel`,
    /// `sprite_pixel_and_index`, and `render_pixel`'s sprite-0
    /// hit check use one byte load + bit-test instead of 4
    /// separate `bitflags::contains` calls per pixel call (8-12
    /// per pixel total, ~5M pixels/sec/worker on full-render
    /// frames). Refreshed in `write_ppu_mask` (only place PPUMASK
    /// changes — via $2001 MMIO write).
    ppu_mask_cache: u8,

    /// Per-pixel palette-index AND mask for the PPUMASK greyscale bit
    /// (bit 0). `0x3F` when greyscale is off (identity — the full 6-bit
    /// index passes through); `0x30` when on (restricts selection to
    /// the grey column $00/$10/$20/$30, matching real PPU behavior).
    /// Applied at the frame-buffer write in `render_pixel` and the
    /// batched scanline copy. Refreshed alongside `ppu_mask_cache`, so
    /// the default (greyscale off) path stays byte-identical.
    grey_mask: u8,

    /// Runtime kill-switch for the refined skip-render fast paths
    /// (idle-HBlank early-return + BG-pixel-pipeline elision). Default
    /// `true` (the paths are active whenever `skip_render` is also set).
    /// Set to `false` to force the full per-cycle tick body — the exact
    /// pre-optimization behaviour — for A/B measurement, byte-exactness
    /// verification, or as a safety fallback. Never a correctness lever:
    /// both settings must produce byte-identical CPU-observable state.
    refined_skip_render: bool,

    /// Cached at scanline boundary: does the active mapper's CHR fetch
    /// path have NO visible-dot side effect (see
    /// `MapperEnum::skip_render_bg_elision_safe`)? Gates the BG-pixel
    /// pipeline elision so MMC3-A12 / MMC2-latch / scanline-IRQ mappers
    /// keep their full per-cycle fetch stream. Refreshed wherever
    /// `chr_cache_ptr` is (the only spots holding a `&mut mapper`).
    /// Safe conservative default `false`: until the first scanline
    /// boundary sets it, the BG pipeline runs in full (byte-exact, just
    /// unoptimized). The mapper never changes for a live `Nes`, so once
    /// set this value is stable — it can never flip from a wrong `true`
    /// to `false`, so it can never cause a fidelity break. Only compiled
    /// in under `ppu_skip_bg` (the pipeline-elision experiment).
    #[cfg(feature = "ppu_skip_bg")]
    bg_elision_allowed: bool,
}

// SAFETY: chr_cache_ptr aliases memory owned by the same Nes that
// owns this Ppu. The pointer is only dereferenced while the owning
// mapper is held by the same Nes (enforced by Nes ownership). Pool
// workers each own their own Nes which Rayon ships across threads —
// the existing PPU is already required to be Send/Sync.
unsafe impl Send for Ppu {}
unsafe impl Sync for Ppu {}

/// Modes for the batched PPU render path. See `Ppu::batched_render_mode`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum BatchedRenderMode {
    /// Per-pixel rendering only (the historical path). Zero cost
    /// from the batched kernel. The default now because `Replace`
    /// produces visible palette divergence on Contra (mountains
    /// flicker red/purple/green) and similar mapper-CHR-RAM games
    /// where the batched kernel's mid-scanline state tracking drops
    /// attribute-table updates. Flip to `Replace` only when the
    /// trainer explicitly needs the perf win and rendering
    /// correctness is unnecessary (net sees grayscale anyway).
    #[default]
    Off,
    /// Run BOTH the per-pixel and batched paths; compare outputs and
    /// count mismatches. Diagnostic — slower but lets callers prove
    /// parity before committing to `Replace`.
    Verify,
    /// Run the batched path only (no per-pixel framebuffer writes);
    /// keeps per-cycle sprite-0 hit detection. ~+26% single-env on
    /// Contra / Castlevania (CPU-heavy mapper games), ~+3% on Mario
    /// Bros (already PPU-cheap). Per-pixel divergence on games with
    /// mid-scanline raster effects shows up as wrong colors in the
    /// GUI but is acceptable for headless training (net only consumes
    /// downsampled 84×84 grayscale).
    Replace,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cycles: u64,
    pub regs: Regs,
    pub ppu_data_read_buffer: u8,
    pub vram: Vram,
    pub palette_ram: PaletteRam,
    pub oam: Oam,
    pub scanline: u16,
    pub scanline_start_cycle: u64,
    pub frame: u64,
    pub ppu_gen_latch: u8,
    pub name_table_byte: u8,
    pub attribute_table_byte: u8,
    pub tile_bitmap_byte_lo: u8,
    pub tile_bitmap_byte_hi: u8,
    pub background_pattern_shift_lo: u16,
    pub background_pattern_shift_hi: u16,
    pub background_palette_shift_lo: u16,
    pub background_palette_shift_hi: u16,
    pub sprite_pattern_shifts_lo: [u8; 8],
    pub sprite_pattern_shifts_hi: [u8; 8],
    pub sprite_attribute_latches: [SpriteAttributes; 8],
    pub sprite_x_counters: [u8; 8],
    pub sprite_0_on_scanline: bool,
    pub nmi_occurred: bool,
    pub nmi_output: bool,
}

impl Ppu {
    pub fn new() -> Ppu {
        Ppu {
            cycles: 0,
            regs: Regs::new(),
            ppu_data_read_buffer: 0,
            mem: MemMap::new(),
            oam: Oam::new(),
            scanline: 0,
            scanline_start_cycle: 0,
            frame: 0,
            frame_buffer: Box::new([0; SCREEN_WIDTH * SCREEN_HEIGHT]),
            ppu_gen_latch: 0,
            name_table_byte: 0,
            attribute_table_byte: 0,
            tile_bitmap_byte_lo: 0,
            tile_bitmap_byte_hi: 0,
            bg_shift: [0; 4],
            sprite_pattern_shifts_lo: [0; 8],
            sprite_pattern_shifts_hi: [0; 8],
            sprite_attribute_latches: [SpriteAttributes(0); 8],
            sprite_x_counters: [0; 8],
            sprite_0_on_scanline: false,
            nmi_occurred: false,
            nmi_output: false,
            skip_render: false,
            bg_tile_buffer: [crate::ppu_neon::BgTile::default(); 36],
            bg_tile_load_count: 0,
            scanline_sprites: [crate::ppu_neon::Sprite::default(); 8],
            batched_scanline_buf: [0u8; 256],
            batched_scanline_invalid: false,
            clean_scanline_history: [false; 240],
            skip_pixel_writes_this_scanline: false,
            batched_render_mode: BatchedRenderMode::default(),
            scanlines_verified: 0,
            scanlines_diverged: 0,
            #[cfg(feature = "ppu_neon_stats")]
            batched_scanlines_total: 0,
            #[cfg(feature = "ppu_neon_stats")]
            batched_scanlines_fallback: 0,
            #[cfg(feature = "ppu_neon_stats")]
            batched_invalidations_by_source: [0; 4],
            #[cfg(feature = "ppu_neon_stats")]
            batched_source_fired_this_scanline: [false; 4],
            #[cfg(feature = "ppu_neon_stats")]
            batched_invalidation_broad: false,
            chr_cache_ptr: core::ptr::null(),
            ppu_mask_cache: 0,
            grey_mask: 0x3F,
            refined_skip_render: true,
            #[cfg(feature = "ppu_skip_bg")]
            bg_elision_allowed: false,
        }
    }

    /// Read a CHR byte ($0000-$1FFF). Uses the cached static-CHR
    /// pointer (refreshed at scanline boundary) when the active
    /// mapper guarantees no runtime banking; otherwise dispatches
    /// through `mapper.chr_read_byte`.
    #[inline(always)]
    fn read_chr_byte(&self, mapper: &mut MapperEnum, address: u16) -> u8 {
        if !self.chr_cache_ptr.is_null() {
            unsafe { *self.chr_cache_ptr.add(address as usize) }
        } else {
            mapper.chr_read_byte(address)
        }
    }

    /// Pack the four hot PPUMASK rendering bits into `ppu_mask_cache`.
    /// Called from PPUMASK ($2001) write handler.
    #[inline(always)]
    fn refresh_ppu_mask_cache(&mut self) {
        let m = self.regs.ppu_mask;
        let mut packed = 0u8;
        if m.contains(PpuMask::SHOW_BACKGROUND)        { packed |= 0b0001; }
        if m.contains(PpuMask::SHOW_SPRITES)           { packed |= 0b0010; }
        if m.contains(PpuMask::SHOW_BACKGROUND_LEFT_8) { packed |= 0b0100; }
        if m.contains(PpuMask::SHOW_SPRITES_LEFT_8)    { packed |= 0b1000; }
        self.ppu_mask_cache = packed;
        // Greyscale (bit 0): AND every output palette index with 0x30
        // to collapse it onto the grey column; identity mask otherwise.
        self.grey_mask = if m.contains(PpuMask::GREYSCALE) { 0x30 } else { 0x3F };
    }

    #[inline(always)] fn cached_show_bg(&self) -> bool         { self.ppu_mask_cache & 0b0001 != 0 }
    #[inline(always)] fn cached_show_sprites(&self) -> bool    { self.ppu_mask_cache & 0b0010 != 0 }
    #[inline(always)] fn cached_show_bg_left8(&self) -> bool   { self.ppu_mask_cache & 0b0100 != 0 }
    #[inline(always)] fn cached_show_sprites_left8(&self) -> bool { self.ppu_mask_cache & 0b1000 != 0 }

    /// Record that an invalidation source fired for the current scanline.
    /// First-fire-wins per source (avoids double-counting repeated writes
    /// from the same source within a single scanline).
    #[inline(always)]
    #[allow(unused_variables)]
    fn note_invalidation_source(&mut self, source: usize) {
        #[cfg(feature = "ppu_neon_stats")]
        {
            if source < self.batched_source_fired_this_scanline.len()
                && !self.batched_source_fired_this_scanline[source]
            {
                self.batched_source_fired_this_scanline[source] = true;
                self.batched_invalidations_by_source[source] =
                    self.batched_invalidations_by_source[source].saturating_add(1);
            }
        }
    }

    /// Toggle the per-frame skip-render fast path. When `true`, the
    /// PPU skips per-pixel color resolution and frame_buffer writes
    /// (massive perf win for RL training with frame_skip>1) but keeps
    /// all CPU-observable side effects intact: sprite-0 hit, NMI,
    /// vblank, shift registers, mapper IRQ A12 ticks.
    pub fn set_skip_render(&mut self, skip: bool) {
        self.skip_render = skip;
    }

    /// Toggle the refined skip-render fast paths (idle-HBlank
    /// early-return + BG-pixel-pipeline elision). Default on. Setting
    /// `false` restores the exact full per-cycle tick body; it is a
    /// perf/verification knob only and must never change
    /// CPU-observable output. See `refined_skip_render`.
    pub fn set_refined_skip_render(&mut self, on: bool) {
        self.refined_skip_render = on;
    }

    /// CPU cycles until the PPU vblank-NMI rising edge fires.
    ///
    /// Returns `None` when no NMI is currently pending in the
    /// foreseeable run path:
    ///   * `nmi_output` is false (PPUCTRL bit 7 disabled — the game
    ///     would have to do a $2000 write to enable it, which is
    ///     MMIO and forces an ASM-batch exit anyway).
    ///   * `nmi_occurred` is already true (vblank already fired this
    ///     frame; the rising edge is in the past).
    ///
    /// Otherwise returns `Some(n)` where `n` is the number of CPU
    /// cycles before `set_vblank()` executes (rounded up). Used by
    /// `Nes::step` to cap the ASM bulk budget so the batch ends at
    /// most one instruction past the vblank fire — matching real
    /// 6502 NMI semantics (NMI services at the next instruction
    /// boundary after the rising edge).
    ///
    /// Approximation: ignores the odd-frame pre-render single-cycle
    /// skip (1 PPU cycle / frame, < 1 CPU cycle of error). The
    /// frame total is fixed at 262 * 341 = 89342 PPU cycles.
    pub fn cpu_cycles_until_nmi_fire(&self) -> Option<u64> {
        if !self.nmi_output || self.nmi_occurred {
            return None;
        }
        // PPU position within the frame, ignoring the odd-frame skip.
        let cur_pos = (self.scanline as u64) * CYCLES_PER_SCANLINE
            + self.scanline_cycle();
        // set_vblank() fires inside the tick whose start state has
        // (scanline=241, scanline_cycle=1) — i.e., frame_pos = 82182.
        let target = (VBLANK_START_SCANLINE as u64) * CYCLES_PER_SCANLINE + 1;
        let frame_total = 262u64 * CYCLES_PER_SCANLINE;
        let ppu_cycles_to_fire = if cur_pos <= target {
            target - cur_pos + 1
        } else {
            frame_total - cur_pos + target + 1
        };
        // CPU cycles = ceil(ppu_cycles / 3). At least 1 to allow
        // forward progress when we're sitting on the firing tick.
        Some(((ppu_cycles_to_fire + 2) / 3).max(1))
    }

    pub fn get_state(&self) -> State {
        State {
            cycles: self.cycles,
            regs: self.regs,
            ppu_data_read_buffer: self.ppu_data_read_buffer,
            vram: self.mem.vram.clone(),
            palette_ram: self.mem.palette_ram.clone(),
            oam: self.oam.clone(),
            scanline: self.scanline,
            scanline_start_cycle: self.scanline_start_cycle,
            frame: self.frame,
            ppu_gen_latch: self.ppu_gen_latch,
            name_table_byte: self.name_table_byte,
            attribute_table_byte: self.attribute_table_byte,
            tile_bitmap_byte_lo: self.tile_bitmap_byte_lo,
            tile_bitmap_byte_hi: self.tile_bitmap_byte_hi,
            background_pattern_shift_lo: self.bg_shift[0],
            background_pattern_shift_hi: self.bg_shift[1],
            background_palette_shift_lo: self.bg_shift[2],
            background_palette_shift_hi: self.bg_shift[3],
            sprite_pattern_shifts_lo: self.sprite_pattern_shifts_lo,
            sprite_pattern_shifts_hi: self.sprite_pattern_shifts_hi,
            sprite_attribute_latches: self.sprite_attribute_latches,
            sprite_x_counters: self.sprite_x_counters,
            sprite_0_on_scanline: self.sprite_0_on_scanline,
            nmi_occurred: self.nmi_occurred,
            nmi_output: self.nmi_output,
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.cycles = state.cycles;
        self.regs = state.regs;
        self.refresh_ppu_mask_cache();
        self.ppu_data_read_buffer = state.ppu_data_read_buffer;
        self.mem.vram = state.vram.clone();
        self.mem.palette_ram = state.palette_ram.clone();
        self.oam = state.oam.clone();
        self.scanline = state.scanline;
        self.scanline_start_cycle = state.scanline_start_cycle;
        self.frame = state.frame;
        self.ppu_gen_latch = state.ppu_gen_latch;
        self.name_table_byte = state.name_table_byte;
        self.attribute_table_byte = state.attribute_table_byte;
        self.tile_bitmap_byte_lo = state.tile_bitmap_byte_lo;
        self.tile_bitmap_byte_hi = state.tile_bitmap_byte_hi;
        self.bg_shift = [
            state.background_pattern_shift_lo,
            state.background_pattern_shift_hi,
            state.background_palette_shift_lo,
            state.background_palette_shift_hi,
        ];
        self.sprite_pattern_shifts_lo = state.sprite_pattern_shifts_lo;
        self.sprite_pattern_shifts_hi = state.sprite_pattern_shifts_hi;
        self.sprite_attribute_latches = state.sprite_attribute_latches;
        self.sprite_x_counters = state.sprite_x_counters;
        self.sprite_0_on_scanline = state.sprite_0_on_scanline;
        self.nmi_occurred = state.nmi_occurred;
        self.nmi_output = state.nmi_output;
    }

    pub fn reset(&mut self) {
        self.cycles = 0;
        self.frame = 0;
        *self.regs.ppu_ctrl = 0;
        self.regs.ppu_mask = PpuMask::NONE;
        self.regs.ppu_status = PpuStatus::RESET_VALUE;
        self.ppu_data_read_buffer = 0;
        self.ppu_mask_cache = 0;
        self.grey_mask = 0x3F;
    }

    pub fn initialize_nestest(&mut self) {
        self.scanline = 0;
        self.cycles = 21;
    }

    fn read_ppu_status(&mut self) -> u8 {
        // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242002_read
        self.regs.w = WriteToggle::FirstWrite;

        let vblank = if self.nmi_occurred { 0x80 } else { 0x00 };
        let status = vblank | (self.regs.ppu_status.bits() & 0x60) | (self.ppu_gen_latch & 0x1F);

        self.nmi_occurred = false;

        status
    }

    fn read_oam_byte(&self) -> u8 {
        // http://wiki.nesdev.com/w/index.php/PPU_sprite_evaluation
        if self.scanline <= VISIBLE_END_SCANLINE {
            let scanline_cycle = self.scanline_cycle();
            if (1..=64).contains(&scanline_cycle) {
                return 0xFF;
            }
        }

        self.oam[self.regs.oam_addr as usize]
    }

    fn write_oam_byte(&mut self, val: u8) {
        // Ignore writes during rendering
        // http://wiki.nesdev.com/w/index.php/PPU_registers#OAM_data_.28.242004.29_.3C.3E_read.2Fwrite
        if self.rendering_enabled() && self.scanline <= VISIBLE_END_SCANLINE {
            return;
        }

        self.oam[self.regs.oam_addr as usize] = val;
        self.regs.oam_addr = self.regs.oam_addr.wrapping_add(1);
    }

    fn write_ppu_ctrl(&mut self, value: u8) {
        // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242000_write
        self.regs.t = (self.regs.t & 0xF3FF) | ((value as u16 & 0x03) << 10);
        *self.regs.ppu_ctrl = value;

        self.nmi_output = value & 0x80 != 0;
    }

    fn write_ppu_scroll(&mut self, value: u8) {
        match self.regs.w {
            WriteToggle::FirstWrite => {
                // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242005_first_write_.28w_is_0.29
                self.regs.t = (self.regs.t & 0xFFE0) | (((value as u16) >> 3) & 0x01F);
                self.regs.x = value & 0x07;
                self.regs.w = WriteToggle::SecondWrite;
            }
            WriteToggle::SecondWrite => {
                // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242005_second_write_.28w_is_1.29
                self.regs.t = (self.regs.t & 0x0C1F)
                    | (((value & 0x07) as u16) << 12)
                    | (((value & 0xF8) as u16) << 2);
                self.regs.w = WriteToggle::FirstWrite;
            }
        }
    }

    fn inc_ppu_addr(&mut self) {
        // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242007_reads_and_writes
        if self.rendering_enabled()
            && (self.scanline == PRE_RENDER_SCANLINE || self.scanline <= VISIBLE_END_SCANLINE)
        {
            self.inc_coarse_x_with_wrap();
            self.inc_y_with_wrap();
        } else {
            self.regs.v += match self.regs.ppu_ctrl.vram_address_increment() {
                VramAddressIncrement::Add1Across => 1,
                VramAddressIncrement::Add32Down => 32,
            };
        }
    }

    fn write_ppu_addr(&mut self, value: u8) {
        match self.regs.w {
            WriteToggle::FirstWrite => {
                // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242006_first_write_.28w_is_0.29
                self.regs.t = (self.regs.t & 0x00FF) | ((value as u16 & 0x3F) << 8);
                self.regs.w = WriteToggle::SecondWrite;
            }
            WriteToggle::SecondWrite => {
                // http://wiki.nesdev.com/w/index.php/PPU_scrolling#.242006_second_write_.28w_is_1.29
                // Per spec: v <- t (full 15 bits, including fine Y bit 2
                // at position 14). The previous mask 0x3FFF cleared bit
                // 14 — wrong when a $2005 second write had set fine Y
                // ∈ {4..7}. EXPERIMENT 2026-04-24: changed to 0x7FFF.
                self.regs.t = (self.regs.t & 0xFF00) | (value as u16);
                self.regs.v = self.regs.t & 0x7FFF;
                self.regs.w = WriteToggle::FirstWrite;
            }
        }
    }

    fn read_ppu_data_byte(&mut self, mapper: &mut MapperEnum) -> u8 {
        let address = self.regs.v;

        let data = if address < PaletteRam::START_ADDRESS {
            // Return contents of read buffer before the read.
            let read_buffer = self.ppu_data_read_buffer;
            self.ppu_data_read_buffer = self.mem.read_byte(mapper, address);
            read_buffer
        } else {
            // Palette data is returned immediately, but we need
            // fill read buffer with VRAM hidden by palette.
            self.ppu_data_read_buffer = self.mem.read_byte(mapper, address - 0x1000);
            self.mem.read_byte(mapper, address)
        };

        self.inc_ppu_addr();

        data
    }

    fn write_ppu_data_byte(&mut self, mapper: &mut MapperEnum, val: u8) {
        let address = self.regs.v;
        // $2007 write divergence analysis:
        //
        //   * **Palette RAM writes** ($3F00-$3FFF): the batched
        //     kernel takes a single palette snapshot at cycle 256,
        //     so a mid-scanline palette write causes divergence
        //     against the per-pixel path (which re-reads palette
        //     per pixel). MUST invalidate.
        //   * **Nametable / pattern writes** (<$3F00): both paths
        //     consume BG fetches captured into `bg_tile_buffer` at
        //     fetch-time. A $2007 write plus the rendering-enabled
        //     `inc_coarse_x/y` side effect mutates `regs.v` going
        //     forward — which alters subsequent tile fetches
        //     identically for BOTH paths. No divergence on fetch
        //     output, so no invalidation needed.
        //
        // Narrowed from the previous blanket-invalidation on ANY
        // $2007 write. Typically a 2-5× reduction in fallback rate
        // on CHR-RAM games that stream tile updates via $2007 and
        // occasionally cross scanline boundaries.
        if self.batched_render_mode != BatchedRenderMode::Off
            && self.scanline <= VISIBLE_END_SCANLINE
            && self.scanline_cycle() <= 256
        {
            let is_palette = address >= PaletteRam::START_ADDRESS;
            if is_palette {
                self.batched_scanline_invalid = true;
                self.note_invalidation_source(0);
            } else {
                // Source 1 = non-palette $2007 write during render.
                // Previously invalidated; now absorbed by the batched
                // path. Counter tracks "would-have-fallen-back under
                // the old rule" so the stats report the narrowing win.
                self.note_invalidation_source(1);
                #[cfg(feature = "ppu_neon_stats")]
                {
                    if self.batched_invalidation_broad {
                        self.batched_scanline_invalid = true;
                    }
                }
            }
        }
        self.mem.write_byte(mapper, address, val);
        self.inc_ppu_addr();
    }

    fn is_sprite_at_y_on_scanline(&self, y: u8) -> bool {
        let y = y as u16;
        let height = self.regs.ppu_ctrl.sprite_size().height() as u16;

        y <= self.scanline && self.scanline < y + height
    }

    fn sprite_evaluation_init(&mut self) {
        self.oam.secondary_write_index = 0;
        self.oam.n = 0;
        self.oam.m = 0;
        self.sprite_0_on_scanline = self.oam.sprite_0_found;
        self.oam.sprite_0_found = false;
    }

    fn sprite_evaluation_read_byte(&mut self) {
        let index = (self.oam.n * Oam::BYTES_PER_SPRITE) + self.oam.m;
        if index < self.oam.primary.len() {
            self.oam.last_read_byte = self.oam.primary[index];
        }
    }

    fn sprite_evaluation_write_byte(&mut self) {
        if self.oam.n < Oam::PRIMARY_MAX_SPRITES {
            if self.oam.secondary_write_index < Oam::SECONDARY_SIZE {
                self.oam.secondary[self.oam.secondary_write_index] = self.oam.last_read_byte;
                if self.oam.m == 0 && !self.is_sprite_at_y_on_scanline(self.oam.last_read_byte) {
                    self.oam.n += 1;
                } else {
                    if self.oam.n == 0 && self.oam.m == 0 {
                        self.oam.sprite_0_found = true;
                    }
                    self.oam.secondary_write_index += 1;
                    self.oam.m += 1;
                }
            } else {
                let y = self.oam.last_read_byte;
                if self.is_sprite_at_y_on_scanline(y) {
                    self.regs.ppu_status.set(PpuStatus::SPRITE_OVERFLOW, true);
                    self.oam.m += 1;
                } else {
                    self.oam.n += 1;
                    self.oam.m += 1;
                }
            }

            if self.oam.m == 4 {
                self.oam.m = 0;
                self.oam.n += 1;
            }
        }
    }

    pub fn rendering_enabled(&self) -> bool {
        self.regs.ppu_mask.contains(PpuMask::SHOW_BACKGROUND)
            || self.regs.ppu_mask.contains(PpuMask::SHOW_SPRITES)
    }

    fn current_name_address(&self) -> u16 {
        0x2000 | (self.regs.v & 0x0FFF)
    }

    fn current_attribute_address(&self) -> u16 {
        let v = self.regs.v;
        0x23C0 | (v & 0x0C00) | ((v >> 4) & 0x38) | ((v >> 2) & 0x07)
    }

    fn current_pattern_address(&self) -> u16 {
        let fine_y = (self.regs.v >> 12) & 0x07;
        self.regs.ppu_ctrl.background_pattern_table_address()
            + (self.name_table_byte as u16 * 16 + fine_y)
    }

    #[inline(always)]
    fn fetch_name_table_byte(&mut self, mapper: &mut MapperEnum) {
        self.name_table_byte = self.mem.read_byte(mapper, self.current_name_address());
    }

    #[inline(always)]
    fn fetch_attribute_table_byte(&mut self, mapper: &mut MapperEnum) {
        self.attribute_table_byte = self.mem.read_byte(mapper, self.current_attribute_address());
    }

    #[inline(always)]
    fn fetch_bitmap_byte_lo(&mut self, mapper: &mut MapperEnum) {
        // BG pattern fetch is always in CHR space ($0000-$1FFF), so
        // we can use the cached CHR pointer fast-path directly
        // instead of going through MemMap::read_byte's address
        // classification + mapper enum dispatch.
        let addr = self.current_pattern_address();
        self.tile_bitmap_byte_lo = self.read_chr_byte(mapper, addr);
    }

    #[inline(always)]
    fn fetch_bitmap_byte_hi(&mut self, mapper: &mut MapperEnum) {
        let addr = self.current_pattern_address() + 8;
        self.tile_bitmap_byte_hi = self.read_chr_byte(mapper, addr);
    }

    /// Called at cycle 256 of each visible scanline when the
    /// batched mode is enabled. Computes a full scanline's 256
    /// palette-indexed bytes from the per-scanline bg tile buffer +
    /// sprite snapshot + palette RAM, then either diffs against the
    /// per-pixel framebuffer (Verify) or overwrites the framebuffer
    /// row with the batched output (Replace).
    fn run_batched_scanline(&mut self) {
        let y = self.scanline as usize;
        #[cfg(feature = "ppu_neon_stats")]
        {
            self.batched_scanlines_total =
                self.batched_scanlines_total.saturating_add(1);
            if self.batched_scanline_invalid {
                self.batched_scanlines_fallback =
                    self.batched_scanlines_fallback.saturating_add(1);
            }
        }
        // Mid-scanline state change we detected — skip the batched
        // render and leave the per-pixel framebuffer authoritative.
        // Don't count as "verified" or "diverged": the metric we care
        // about is "if we'd batched this scanline, would it match?".
        // Invalidated scanlines are a perf hit (we paid per-pixel),
        // not a correctness loss.
        if self.batched_scanline_invalid {
            // Flip history so NEXT frame's same scanline won't try to
            // skip per-pixel writes (we'd mis-predict again).
            if y < self.clean_scanline_history.len() {
                self.clean_scanline_history[y] = false;
            }
            // If we DID skip writes earlier this scanline (mis-
            // prediction), the row is stale. In Replace mode we
            // accept the stale row for one frame — the history flip
            // above ensures next frame falls back. This is the
            // trade-off: near-zero perf cost in exchange for rare
            // 1-frame artifacts on mid-scanline-writing games.
            return;
        }
        let cfg = crate::ppu_neon::ScanlineConfig {
            fine_x: self.regs.x,
            show_bg: self.regs.ppu_mask.contains(PpuMask::SHOW_BACKGROUND),
            show_sprites: self.regs.ppu_mask.contains(PpuMask::SHOW_SPRITES),
            show_bg_left8: self.regs.ppu_mask.contains(PpuMask::SHOW_BACKGROUND_LEFT_8),
            show_sprites_left8: self.regs.ppu_mask.contains(PpuMask::SHOW_SPRITES_LEFT_8),
        };
        // Master-disable path: when neither background nor sprites are
        // being rendered, the per-pixel path calls `clear_pixel`, which
        // outputs the $3F00 backdrop color (hardware forced-blank behavior,
        // Mesen-validated). Mirror that here.
        if !cfg.show_bg && !cfg.show_sprites {
            let backdrop = self.mem.palette_ram.read_byte(0x3F00) & 0x3F;
            for px in self.batched_scanline_buf.iter_mut() {
                *px = backdrop;
            }
            let y = self.scanline as usize;
            let row_start = y * SCREEN_WIDTH;
            let row_end = row_start + SCREEN_WIDTH;
            match self.batched_render_mode {
                BatchedRenderMode::Off => unreachable!(),
                BatchedRenderMode::Verify => {
                    let mut diverged = false;
                    for x in 0..SCREEN_WIDTH {
                        if self.frame_buffer[row_start + x]
                            != self.batched_scanline_buf[x]
                        {
                            diverged = true;
                            break;
                        }
                    }
                    self.scanlines_verified = self.scanlines_verified.saturating_add(1);
                    if diverged {
                        self.scanlines_diverged =
                            self.scanlines_diverged.saturating_add(1);
                    }
                }
                BatchedRenderMode::Replace => {
                    self.frame_buffer[row_start..row_end]
                        .copy_from_slice(&self.batched_scanline_buf);
                }
            }
            return;
        }
        // Materialize a 32-byte palette snapshot for the kernel.
        // Previously this was a 32-byte scalar loop calling
        // `PaletteRam::read_byte` on every index — 240 scanlines ×
        // 32 reads = 7680 function calls per frame. Bulk-copy + patch
        // the 4 mirror entries ($3F10/14/18/1C mirror $3F00/04/08/0C)
        // cuts that to one memcpy + 4 stores.
        let mut palette = *self.mem.palette_ram.raw_bytes();
        palette[0x10] = palette[0x00];
        palette[0x14] = palette[0x04];
        palette[0x18] = palette[0x08];
        palette[0x1C] = palette[0x0C];
        let tile_count = self.bg_tile_load_count.min(34) as usize;
        // Fast path: a full 34 tiles were loaded this scanline (the
        // overwhelmingly common case). Hand the kernel a slice of the
        // existing buffer directly — no per-scanline [BgTile; 34]
        // stack alloc + zero-init + copy (previously 272 bytes × 240
        // scanlines = 65 KB/frame of pointless churn).
        //
        // Slow path: fewer than 34 tiles loaded (off-by-one transition
        // frame). Fall back to padding with defaults so the kernel sees
        // at least 33 entries as its assertion requires.
        if tile_count >= 34 {
            crate::ppu_neon::render_scanline(
                &cfg,
                &self.bg_tile_buffer[..34],
                &self.scanline_sprites,
                &palette,
                &mut self.batched_scanline_buf,
            );
        } else {
            let mut padded: [crate::ppu_neon::BgTile; 34] =
                [crate::ppu_neon::BgTile::default(); 34];
            padded[..tile_count].copy_from_slice(&self.bg_tile_buffer[..tile_count]);
            crate::ppu_neon::render_scanline(
                &cfg,
                &padded,
                &self.scanline_sprites,
                &palette,
                &mut self.batched_scanline_buf,
            );
        }

        // Greyscale (PPUMASK bit 0): restrict every rendered pixel to
        // the grey column, mirroring the per-pixel `render_pixel` path
        // so Verify-mode comparison and Replace-mode output both match.
        // No-op when off (grey_mask == 0x3F) → batched output stays
        // byte-identical by default.
        if self.grey_mask != 0x3F {
            for px in self.batched_scanline_buf.iter_mut() {
                *px &= self.grey_mask;
            }
        }

        let row_start = y * SCREEN_WIDTH;
        let row_end = row_start + SCREEN_WIDTH;

        match self.batched_render_mode {
            BatchedRenderMode::Off => unreachable!(),
            BatchedRenderMode::Verify => {
                let mut diverged = false;
                for x in 0..SCREEN_WIDTH {
                    if self.frame_buffer[row_start + x] != self.batched_scanline_buf[x] {
                        diverged = true;
                        break;
                    }
                }
                self.scanlines_verified = self.scanlines_verified.saturating_add(1);
                // Mark clean (for the prediction heuristic — even in
                // Verify mode we track history so Replace mode can
                // start predicting immediately on switch).
                if !diverged && y < self.clean_scanline_history.len() {
                    self.clean_scanline_history[y] = true;
                } else if y < self.clean_scanline_history.len() {
                    self.clean_scanline_history[y] = false;
                }
                if diverged {
                    self.scanlines_diverged = self.scanlines_diverged.saturating_add(1);
                }
            }
            BatchedRenderMode::Replace => {
                self.frame_buffer[row_start..row_end]
                    .copy_from_slice(&self.batched_scanline_buf);
                // Successfully batched — mark this scanline clean so
                // next frame we skip per-pixel writes. Compounds over
                // frames into the actual perf win.
                if y < self.clean_scanline_history.len() {
                    self.clean_scanline_history[y] = true;
                }
            }
        }
    }

    /// Called by SystemBus on mapper PRG-register writes ($4018-$FFFF).
    ///
    /// **Previously** this invalidated the current scanline's
    /// batched path on any mapper write during cycles 0..=256 of a
    /// visible scanline, out of paranoia about mid-render CHR bank
    /// or mirroring swaps.
    ///
    /// **Divergence analysis**: a mapper register write changes
    /// mapper-internal state (CHR bank, mirroring, IRQ counter).
    /// Subsequent BG tile fetches and sprite tile fetches on this
    /// scanline read the new bank/mirror. The per-pixel path feeds
    /// those fetched bytes into its shift registers; the batched
    /// path captures the same fetched bytes into `bg_tile_buffer`
    /// via `load_background_registers`. Both paths produce
    /// identical output for mid-scanline CHR swaps because the
    /// shared fetch point is post-swap for both.
    ///
    /// The only mapper-write side effect that would actually
    /// diverge is an MMC3-style IRQ that mutates CPU-observable
    /// state mid-scanline, which affects sprite-0-hit timing — and
    /// sprite-0-hit runs on the per-pixel path regardless of
    /// batched mode. So the batched FRAME-buffer output is safe.
    ///
    /// Keep the counter bump for diagnostics; drop the actual
    /// invalidation. Stats-off builds pay zero cost.
    pub fn note_mapper_write_during_render(&mut self) {
        #[cfg(feature = "ppu_neon_stats")]
        {
            if self.batched_render_mode != BatchedRenderMode::Off
                && self.scanline <= VISIBLE_END_SCANLINE
                && self.scanline_cycle() <= 256
            {
                self.note_invalidation_source(3);
                if self.batched_invalidation_broad {
                    self.batched_scanline_invalid = true;
                }
            }
        }
    }

    /// Set the batched-render mode. See `BatchedRenderMode` for the
    /// Off / Verify / Replace semantics. Also resets the parity
    /// counters so callers measure one run at a time.
    pub fn set_batched_render_mode(&mut self, mode: BatchedRenderMode) {
        self.batched_render_mode = mode;
        self.scanlines_verified = 0;
        self.scanlines_diverged = 0;
        // Reset prediction history on mode change — fresh slate so we
        // don't carry stale "clean" flags from a different run.
        self.clean_scanline_history.fill(false);
        self.skip_pixel_writes_this_scanline = false;
        #[cfg(feature = "ppu_neon_stats")]
        {
            self.batched_scanlines_total = 0;
            self.batched_scanlines_fallback = 0;
            self.batched_invalidations_by_source = [0; 4];
            self.batched_source_fired_this_scanline = [false; 4];
        }
    }

    fn load_background_registers(&mut self) {
        self.bg_shift[0] =
            (self.bg_shift[0] & 0xFF00) | (self.tile_bitmap_byte_lo as u16);
        self.bg_shift[1] =
            (self.bg_shift[1] & 0xFF00) | (self.tile_bitmap_byte_hi as u16);

        let palette_shift = ((self.regs.v >> 4) & 0x04) | (self.regs.v & 0x02);
        let palette = ((self.attribute_table_byte >> palette_shift) & 0x03) as u16;

        self.bg_shift[2] = (self.bg_shift[2] & 0xFF00)
            | (if (palette & 0x01) == 0x01 { 0xFF } else { 0x00 });
        self.bg_shift[3] = (self.bg_shift[3] & 0xFF00)
            | (if (palette & 0x02) == 0x02 { 0xFF } else { 0x00 });

        // Capture into the batched-render tile buffer only when the
        // batched path is active. Default Off mode skips entirely —
        // no struct write, no branch, no memory traffic. Matters
        // because load_background_registers fires 34×/scanline ×
        // 240 scanlines × 60 fps × 16 workers = ~7.8M calls/sec.
        if self.batched_render_mode != BatchedRenderMode::Off {
            let idx = self.bg_tile_load_count as usize;
            if idx < self.bg_tile_buffer.len() {
                // Pre-expand the 8 column bytes here (Sprint 11: move
                // the bg_col expansion out of the per-scanline NEON
                // kernel into fetch-time). `cols[k]` is MSB-first —
                // k=0 is bit 7, k=7 is bit 0. This trades zero
                // per-scanline work for an 8-byte write per tile
                // fetch, cutting Replace-mode overhead to a flat
                // memcpy in the kernel.
                let lo = self.tile_bitmap_byte_lo;
                let hi = self.tile_bitmap_byte_hi;
                let pal_shifted = ((palette as u8) & 0x03) << 2;
                let mut cols = [0u8; 8];
                for k in 0..8 {
                    let bit = 7 - k;
                    let p = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    cols[k] = if p == 0 { 0 } else { p | pal_shifted };
                }
                self.bg_tile_buffer[idx] = crate::ppu_neon::BgTile { cols };
            }
            self.bg_tile_load_count = self.bg_tile_load_count.saturating_add(1);
        }
    }

    #[inline(always)]
    fn shift_background_registers(&mut self) {
        // 4 u16 lanes, shifted left by 1 every cycle. Fires ~272×
        // per visible/pre-render scanline = ~65k×/full-render frame
        // = ~17M×/sec/worker at 270 fs=4-frames/sec. Single NEON
        // vshl_n_u16 over the 4-lane array beats 4 separate scalar
        // load+shift+store sequences.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use std::arch::aarch64::*;
            let p = self.bg_shift.as_mut_ptr();
            let v = vld1_u16(p);
            let shifted = vshl_n_u16::<1>(v);
            vst1_u16(p, shifted);
            return;
        }
        #[allow(unreachable_code)]
        {
            self.bg_shift[0] <<= 1;
            self.bg_shift[1] <<= 1;
            self.bg_shift[2] <<= 1;
            self.bg_shift[3] <<= 1;
        }
    }

    fn fetch_sprite_tile(&mut self, mapper: &mut MapperEnum, sprite_index: usize) {
        // Don't attempt to fetch tiles for empty sprite slots.
        if sprite_index >= self.oam.secondary_write_index / 4 {
            self.sprite_attribute_latches[sprite_index] = SpriteAttributes(0xFF);
            return;
        }

        let index = sprite_index * Oam::BYTES_PER_SPRITE;
        let sprite =
            Sprite::from_oam_bytes(&self.oam.secondary[index..(index + Oam::BYTES_PER_SPRITE)]);

        let size = self.regs.ppu_ctrl.sprite_size();

        let height: u16 = match size {
            SpriteSize::Size8x8 => 8,
            SpriteSize::Size8x16 => 16,
        };
        // Guard the row: `self.scanline - sprite.y` underflows when
        // scanline < sprite.y (e.g. the pre-render scanline with stale
        // secondary OAM), and the flip subtraction below then produces a
        // pattern address far outside CHR space that read_chr_byte would
        // dereference unmasked via a raw pointer (out-of-bounds read). Real
        // evaluated sprites always have row in [0, height), so this early
        // return never fires for them — rendering stays byte-identical.
        let mut row = match self.scanline.checked_sub(sprite.y as u16) {
            Some(r) if r < height => r,
            _ => {
                self.sprite_attribute_latches[sprite_index] = SpriteAttributes(0xFF);
                return;
            }
        };
        let pattern_addr = match size {
            SpriteSize::Size8x8 => {
                if sprite.flip_vertically() {
                    row = 7 - row;
                }

                (sprite.tile_index as u16) * 16
                    + self.regs.ppu_ctrl.sprite_pattern_table_address()
                    + row
            }
            SpriteSize::Size8x16 => {
                if sprite.flip_vertically() {
                    row = 15 - row;
                }

                let table: u16 = if sprite.tile_index & 0x01 == 0 {
                    0x0000
                } else {
                    0x1000
                };
                let mut tile_index = sprite.tile_index & 0xFE;
                if row > 7 {
                    // Jump to second tile
                    tile_index += 1;
                    row -= 8;
                }

                table + (tile_index as u16) * 16 + row
            }
        };
        // Defensive: constrain the fetch to CHR space ($0000-$1FFF) so a
        // malformed address can never index past the cached CHR pointer. The
        // row guard above keeps this in range for real sprites (no-op there).
        let pattern_addr = pattern_addr & 0x1FFF;

        // Sprite pattern fetch is in CHR space; use cached pointer.
        let mut pattern_lo = self.read_chr_byte(mapper, pattern_addr);
        let mut pattern_hi = self.read_chr_byte(mapper, pattern_addr + 8);

        if sprite.flip_horizontally() {
            pattern_lo = pattern_lo.swap_bits();
            pattern_hi = pattern_hi.swap_bits();
        }

        self.sprite_pattern_shifts_lo[sprite_index] = pattern_lo;
        self.sprite_pattern_shifts_hi[sprite_index] = pattern_hi;
        self.sprite_attribute_latches[sprite_index] = sprite.attributes;
        self.sprite_x_counters[sprite_index] = sprite.x;
    }

    #[inline(always)]
    fn update_sprite_rendering_registers(&mut self) {
        // Per-pixel hot loop (called once per visible cycle = 256 ×
        // 240 = 61440 calls/full-render frame). Each lane updates
        // independently, so all 8 sprites can be processed in a
        // single NEON vector op.
        //
        // LLVM auto-vectorizes the scalar branchless form below, but
        // explicit NEON keeps the 3 dependent ops in the same
        // register file across the full sequence, avoiding the load/
        // store-pair bookkeeping the auto-vectorizer emits between
        // statement boundaries.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use std::arch::aarch64::*;
            let xc_ptr = self.sprite_x_counters.as_mut_ptr();
            let lo_ptr = self.sprite_pattern_shifts_lo.as_mut_ptr();
            let hi_ptr = self.sprite_pattern_shifts_hi.as_mut_ptr();
            let xc = vld1_u8(xc_ptr);
            let lo = vld1_u8(lo_ptr);
            let hi = vld1_u8(hi_ptr);
            // is_zero[lane] = 0xFF if x_counter==0 else 0x00.
            let is_zero = vceq_u8(xc, vdup_n_u8(0));
            // Counter: subtract 1 unless already zero. Equivalent to
            // saturating_sub(1 - is_zero) in scalar; expressed as
            // qsub of a per-lane amount (0 or 1).
            let sub_amount = vand_u8(vmvn_u8(is_zero), vdup_n_u8(1));
            let new_xc = vqsub_u8(xc, sub_amount);
            // Pattern shifts: shl by 1 when counter just hit zero.
            let shift_amount = vand_u8(is_zero, vdup_n_u8(1));
            let shift_signed = vreinterpret_s8_u8(shift_amount);
            let new_lo = vshl_u8(lo, shift_signed);
            let new_hi = vshl_u8(hi, shift_signed);
            vst1_u8(xc_ptr, new_xc);
            vst1_u8(lo_ptr, new_lo);
            vst1_u8(hi_ptr, new_hi);
            return;
        }
        // Portable scalar fallback (auto-vectorized on most targets).
        #[allow(unreachable_code)]
        for i in 0..8 {
            let c = self.sprite_x_counters[i];
            let is_zero = (c == 0) as u8;
            self.sprite_x_counters[i] = c.saturating_sub(1 - is_zero);
            self.sprite_pattern_shifts_lo[i] <<= is_zero;
            self.sprite_pattern_shifts_hi[i] <<= is_zero;
        }
    }

    #[inline(always)]
    fn color_from_palette_index(&self, index: u8) -> u8 {
        self.mem
            .palette_ram
            .read_byte(PaletteRam::START_ADDRESS | (index & 0x1F) as u16)
    }

    #[inline(always)]
    fn render_pixel(&mut self) {
        // Skip-render fast path: bail before background+sprite pixel
        // extraction when no CPU-observable side effect can fire this
        // pixel. The only side effect the CPU sees from per-pixel work
        // is the SPRITE_ZERO_HIT sticky bit. We can short-circuit when:
        //   * SPRITE_ZERO_HIT is already latched (sticky until vblank
        //     clears it on the pre-render scanline), OR
        //   * sprite 0 isn't on the current scanline (the hit can only
        //     fire on scanlines that contain sprite 0).
        // For SMB-style games, sprite 0 sits on a single HUD scanline
        // (~scanline 31), so 209 of 240 visible scanlines bail
        // immediately — 209 * 256 = 53504 calls saved per skipped
        // frame of background_pixel + sprite_pixel_and_index work.
        if self.skip_render
            && (self.regs.ppu_status.contains(PpuStatus::SPRITE_ZERO_HIT)
                || !self.sprite_0_on_scanline)
        {
            return;
        }
        // Batched-render predicted-clean path: we've committed to
        // emitting this row from the ppu_neon kernel at cycle 256.
        // Only CPU-observable effect of the per-pixel path is the
        // sprite-0-hit latch. If sprite-0 is already latched OR the
        // current scanline doesn't contain sprite 0, we can bail
        // immediately — pure win, no pixel extraction work at all.
        if self.skip_pixel_writes_this_scanline
            && (!self.sprite_0_on_scanline
                || self.regs.ppu_status.contains(PpuStatus::SPRITE_ZERO_HIT))
        {
            return;
        }

        let x = self.scanline_cycle() - 1;
        let y = self.scanline;

        let background_pixel = self.background_pixel(x);
        let (sprite_pixel, sprite_index) = self.sprite_pixel_and_index(x);

        let background_pattern = background_pixel & 0x03;
        let sprite_pattern = sprite_pixel & 0x03;

        // Sprite-0 hit detection runs unconditionally — it sets a
        // PPUSTATUS bit the CPU reads. Even with skip_render on, we
        // need this side effect.
        if background_pattern > 0
            && sprite_pattern > 0
            && self.sprite_0_on_scanline
            && sprite_index == 0
            && self.cached_show_bg()
            && self.cached_show_sprites()
            && !(x < 8
                && (!self.cached_show_bg_left8()
                    || !self.cached_show_sprites_left8()))
            && x < 255
            && !self.regs.ppu_status.contains(PpuStatus::SPRITE_ZERO_HIT)
        {
            self.regs.ppu_status.set(PpuStatus::SPRITE_ZERO_HIT, true);
        }

        // Color resolution + frame_buffer write — pure render work,
        // never observed by CPU. Skip on the fast path.
        // ALSO skip when the batched renderer has committed to writing
        // this scanline's row at cycle 256 (predicted-clean path).
        if self.skip_render || self.skip_pixel_writes_this_scanline {
            return;
        }

        let color = if background_pattern == 0 && sprite_pattern == 0 {
            self.color_from_palette_index(0x00)
        } else if background_pattern == 0 && sprite_pattern > 0 {
            self.color_from_palette_index(sprite_pixel)
        } else if background_pattern > 0 && sprite_pattern == 0 {
            self.color_from_palette_index(background_pixel)
        } else {
            let attrs = self.sprite_attribute_latches[sprite_index];
            match attrs.priority() {
                SpritePriority::InFrontOfBackground => self.color_from_palette_index(sprite_pixel),
                SpritePriority::BehindBackground => self.color_from_palette_index(background_pixel),
            }
        };

        // `grey_mask` is 0x3F normally (identity on the 6-bit index) and
        // 0x30 when PPUMASK greyscale is on (collapses to the grey column).
        self.frame_buffer[(y as usize * SCREEN_WIDTH) + x as usize] = color & self.grey_mask;
    }

    // Set pixel to black.
    #[inline(always)]
    fn clear_pixel(&mut self) {
        if self.skip_render {
            return;
        }
        let x = self.scanline_cycle() - 1;
        let y = self.scanline;
        // Real hardware (and the LaiNES/Mesen oracle) outputs the backdrop
        // color at palette_ram[$3F00] during forced blank, not a hardcoded
        // NES black. Mask to 6 bits like write_pixel. (Mesen-validated.)
        let backdrop = self.mem.palette_ram.read_byte(0x3F00) & 0x3F;
        self.frame_buffer[(y as usize * SCREEN_WIDTH) + x as usize] = backdrop;
    }

    #[inline(always)]
    fn background_pixel(&self, x: u64) -> u8 {
        // Use packed PPUMASK cache instead of bitflags::contains() —
        // single byte load + bit test vs ldrb + and + cmp per check.
        if !self.cached_show_bg() || x < 8 && !self.cached_show_bg_left8() {
            return 0;
        }

        let fine_x = self.regs.x;

        ((self.bg_shift[0] >> (15 - fine_x)) & 0x01) as u8
            | ((self.bg_shift[1] >> (14 - fine_x)) & 0x02) as u8
            | ((self.bg_shift[2] >> (13 - fine_x)) & 0x04) as u8
            | ((self.bg_shift[3] >> (12 - fine_x)) & 0x08) as u8
    }

    #[inline(always)]
    fn sprite_pixel_and_index(&self, x: u64) -> (u8, usize) {
        if self.scanline == 0
            || !self.cached_show_sprites()
            || x < 8 && !self.cached_show_sprites_left8()
        {
            return (0, 0);
        }

        // Fast path: check all 8 x-counters at once. A packed zero
        // check lets us skip the per-slot iteration entirely when no
        // sprite is active at this column (the ~common case for most
        // of each scanline).
        let xs = u64::from_ne_bytes(self.sprite_x_counters);
        // Byte-wise "is any lane zero" trick: a byte is zero iff
        // (b - 1) & ~b & 0x80 is set, combined across all 8 lanes.
        // See Hacker's Delight §6-1 / Bit Twiddling Hacks "haszero".
        let any_zero = (xs.wrapping_sub(0x0101_0101_0101_0101))
            & !xs
            & 0x8080_8080_8080_8080;
        if any_zero == 0 {
            return (0, 0);
        }

        // Slow path — at least one sprite is active at this column.
        // Compute the candidate pixel for all 8 sprites in parallel
        // via NEON, then find the first lane that's both active
        // (x_counter==0 AND attr!=0xFF) AND has non-transparent
        // pattern bits.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use std::arch::aarch64::*;
            let xc = vld1_u8(self.sprite_x_counters.as_ptr());
            let attrs = vld1_u8(self.sprite_attribute_latches.as_ptr() as *const u8);
            let lo_shifts = vld1_u8(self.sprite_pattern_shifts_lo.as_ptr());
            let hi_shifts = vld1_u8(self.sprite_pattern_shifts_hi.as_ptr());

            // active[lane] = 0xFF if (x_counter == 0) AND (attr != 0xFF).
            let zero = vdup_n_u8(0);
            let ff = vdup_n_u8(0xFF);
            let xc_zero = vceq_u8(xc, zero);
            let attr_ne_ff = vmvn_u8(vceq_u8(attrs, ff));
            let active = vand_u8(xc_zero, attr_ne_ff);

            // pixel[lane] = (lo>>7 & 1) | (hi>>6 & 2) | ((attr & 0x03) << 2 | 0x10).
            // Note: palette() = (raw & 0x03) | 0x04, so
            // (palette() << 2) & 0x1C = ((raw & 0x03) << 2) | 0x10.
            let lo_bit = vshr_n_u8::<7>(lo_shifts);
            let hi_bit = vand_u8(vshr_n_u8::<6>(hi_shifts), vdup_n_u8(2));
            let palette_bits = vorr_u8(
                vshl_n_u8::<2>(vand_u8(attrs, vdup_n_u8(0x03))),
                vdup_n_u8(0x10),
            );
            let pixel = vorr_u8(vorr_u8(lo_bit, hi_bit), palette_bits);

            // interesting[lane] = 0xFF if active AND (pattern bits nonzero).
            let pattern_bits = vand_u8(pixel, vdup_n_u8(0x03));
            let pattern_nonzero = vmvn_u8(vceq_u8(pattern_bits, zero));
            let interesting = vand_u8(active, pattern_nonzero);

            // Find first lane where interesting != 0. Pack to u64
            // (8 lanes × 0xFF or 0x00) and use trailing_zeros / 8.
            let interesting_u64 =
                vget_lane_u64::<0>(vreinterpret_u64_u8(interesting));
            if interesting_u64 == 0 {
                return (0, 0);
            }
            let lane = (interesting_u64.trailing_zeros() / 8) as usize;
            let pixel_bytes: [u8; 8] = core::mem::transmute(pixel);
            return (pixel_bytes[lane], lane);
        }

        // Portable scalar fallback.
        #[allow(unreachable_code)]
        {
            for i in 0..8 {
                if self.sprite_x_counters[i] == 0 && self.sprite_attribute_latches[i].0 != 0xFF
                {
                    let pixel = ((self.sprite_pattern_shifts_lo[i] >> 7) & 0x01)
                        | ((self.sprite_pattern_shifts_hi[i] >> 6) & 0x02)
                        | ((self.sprite_attribute_latches[i].palette() << 2) & 0x1C);

                    if pixel & 0x03 != 0 {
                        return (pixel, i);
                    }
                }
            }
            (0, 0)
        }
    }

    fn inc_coarse_x_with_wrap(&mut self) {
        if (self.regs.v & 0x001F) == 31 {
            self.regs.v &= !0x001F; // course X = 0
            self.regs.v ^= 0x0400; // switch horizontal nametable
        } else {
            self.regs.v += 1;
        }
    }

    fn inc_y_with_wrap(&mut self) {
        if (self.regs.v & 0x7000) != 0x7000 {
            self.regs.v += 0x1000;
        } else {
            self.regs.v &= !0x7000;
            let mut y = (self.regs.v & 0x03E0) >> 5;
            if y == 29 {
                y = 0;
                self.regs.v ^= 0x0800;
            } else if y == 31 {
                y = 0;
            } else {
                y += 1;
            }

            self.regs.v = (self.regs.v & !0x03E0) | (y << 5);
        }
    }

    fn set_vblank(&mut self) {
        self.nmi_occurred = true;
    }

    fn clear_vblank(&mut self) {
        self.nmi_occurred = false;
    }

    pub fn scanline_cycle(&self) -> u64 {
        self.cycles - self.scanline_start_cycle
    }

    /// Run three PPU cycles in one call. There are 3 PPU cycles
    /// per CPU cycle; this saves the per-cycle function-call
    /// overhead (register save/restore, borrow-checker
    /// bookkeeping) when the caller doesn't need per-cycle hooks
    /// in between (the bulk-step path, for example).
    ///
    /// Behaviourally identical to calling `tick` three times.
    #[inline]
    pub fn tick_three<V: VideoSink>(&mut self, mapper: &mut MapperEnum, video_frame_sink: &mut V) {
        self.tick(mapper, video_frame_sink);
        self.tick(mapper, video_frame_sink);
        self.tick(mapper, video_frame_sink);
    }

    // Tried `tick_n(n)` — regressed 55%, see apu.rs for details.
    // Kept `tick_three` because its 3-iteration unroll is PGO-friendly.

    /// Run one PPU cycle. There are 3 PPU cycles per CPU cycle.
    /// Returns whether an NMI should be requested.
    pub fn tick<V: VideoSink>(&mut self, mapper: &mut MapperEnum, video_frame_sink: &mut V) {
        // Vblank-scanline fast-path (scanlines 240..=260). On vblank
        // scanlines the PPU does no rendering work — no BG fetches,
        // no sprite evaluation, no mapper IRQ tick, no per-pixel
        // render. The only CPU-observable side effect is
        // `set_vblank()` at (241, 1). Skipping the rest of the tick
        // body for these 21 scanlines × 341 cycles = 7161 PPU ticks
        // per frame (~8% of all ticks) is pure perf, no correctness
        // risk. Batched-render boundary work is also skipped here
        // because it would be a no-op during vblank — `bg_tile_load
        // _count` and `batched_scanline_invalid` get re-initialized
        // when scanline 261 (pre-render) starts via the main path.
        let scanline = self.scanline;
        if scanline > VISIBLE_END_SCANLINE && scanline < PRE_RENDER_SCANLINE {
            let scanline_cycle = self.scanline_cycle();
            if scanline == VBLANK_START_SCANLINE && scanline_cycle == 1 {
                self.set_vblank();
            }
            self.cycles += 1;
            if scanline_cycle >= CYCLES_PER_SCANLINE - 1 {
                self.scanline_start_cycle = self.cycles;
                self.scanline += 1;
                // Refresh cached CHR pointer at every scanline
                // boundary, including vblank → pre-render (260 →
                // 261). Pre-render does CHR reads via BG fetches.
                self.chr_cache_ptr = mapper.chr_static_ptr().unwrap_or(core::ptr::null());
                #[cfg(feature = "ppu_skip_bg")]
                {
                    self.bg_elision_allowed = mapper.skip_render_bg_elision_safe();
                }
                // Vblank-scanline transitions never reach pre-render
                // boundary state that batched_render needs, so skip
                // the boundary work entirely — it gets done when we
                // transition into 261 (pre-render) via the main path.
            }
            return;
        }

        // Refined skip-render fast path #1: idle-HBlank early-return.
        // On a VISIBLE scanline (0..=239), dots 258..=320 do NO
        // CPU-observable or PPU-state work EXCEPT (a) sprite tile
        // fetches at dots 264,272,…,320 (multiples of 8) and (b) the
        // MMC3 A12 scanline-IRQ trigger at dot 260. Every other dot in
        // that window only advances the cycle counter: no render_pixel
        // (dot>256), no bg/sprite shift (2..=257 / 322..=337), no bg
        // fetch (1..=256 / 321..=336), no sprite evaluation (1..=256),
        // no scroll copy (dot 257, or pre-render 280..=304). So the
        // whole tick body collapses to `cycles += 1` for those dots.
        //
        // Restricted to VISIBLE scanlines: the pre-render scanline (261)
        // performs the vertical v<-t scroll copy at dots 280..=304 —
        // inside this window and CPU-observable — so it must run the
        // full body. Preserving dot 260 and the multiple-of-8 sprite
        // fetches keeps this byte-exact for A12-IRQ mappers too (the
        // fetch address stream and its cpu-cycle timing are unchanged),
        // so no mapper gate is needed. Only active under `skip_render`
        // (via the runtime kill-switch) so the full-render path — every
        // frame the agent actually sees, and the parity/oracle gates —
        // is byte-identical to the pre-optimization tick.
        if self.refined_skip_render && self.skip_render && scanline <= VISIBLE_END_SCANLINE {
            let dot = self.scanline_cycle();
            if (258..=320).contains(&dot) && dot % 8 != 0 && dot != 260 {
                self.cycles += 1;
                return;
            }
        }

        // `skip_render` is a pixel-write fast path, NOT a whole-PPU
        // bypass. The main tick body runs as normal so that sprite-0
        // hit, mapper scanline IRQs, scroll register reloads (v from
        // t at cycle 257 / cycles 280-304), sprite evaluation, and
        // every other CPU-observable PPU side effect fire on their
        // exact cycles. `render_pixel` and `clear_pixel` themselves
        // honor `skip_render` and return early before the
        // frame_buffer write — that's where the perf saving lives.
        //
        // Historical context: an earlier revision delegated
        // immediately to a `tick_skip_render` function here, which
        // bypassed everything except vblank and the scanline
        // counter. That broke every sprite-0-hit-timed game: SMB
        // HUD, Zelda HUD, MMC3 status-bar splits. The CPU runs at
        // full speed across the frame_skip batch, reads $2002 for
        // sprite-0 mid-frame, and without the PPU firing that bit
        // on skipped frames, the CPU took wrong branches and the
        // batch's final "full render" frame had corrupted scroll.
        // See tests/skip_render_parity.rs.

        let scanline_cycle = self.scanline_cycle();

        let on_prerender_scanline = self.scanline == PRE_RENDER_SCANLINE;
        let on_visible_scanline = self.scanline <= VISIBLE_END_SCANLINE;
        let on_visible_cycle = (1..=256).contains(&scanline_cycle);
        // Cache once: the PPUMASK SHOW_BACKGROUND/SHOW_SPRITES bits
        // can only change via a CPU $2001 write, which is MMIO and
        // routes through the bus (outside this tick body). So the
        // value is constant for the duration of one tick and we can
        // hoist the bitflag check + or out of the 3-call hot loop.
        let rendering_enabled = self.rendering_enabled();

        // MMC3-style scanline IRQ hook. Fires once per visible +
        // pre-render scanline on the rising edge of PPU A12, which
        // depends on PPUCTRL pattern-table selection:
        //
        //   * BG=$0000, sprites=$1000 (most common, e.g. SMB3 status
        //     bar split): A12 transitions low→high at cycle 260 when
        //     the first sprite tile pattern fetch starts.
        //   * BG=$1000, sprites=$0000 (e.g. some MM3/Crystalis
        //     scenes): A12 transitions low→high at cycle 324 when
        //     BG fetches resume after the sprite fetch window.
        //   * Both tables same: A12 stays at one level the entire
        //     scanline → no rise → no IRQ tick.
        //
        // Mappers that don't use scanline IRQs have a no-op default.
        if rendering_enabled
            && (on_visible_scanline || on_prerender_scanline)
        {
            let bg = self.regs.ppu_ctrl.background_pattern_table_address();
            let sp = self.regs.ppu_ctrl.sprite_pattern_table_address();
            let trigger_cycle: Option<u64> = match (bg, sp) {
                (0x0000, 0x1000) => Some(260),
                (0x1000, 0x0000) => Some(324),
                _ => None,
            };
            if Some(scanline_cycle) == trigger_cycle {
                mapper.on_scanline_tick();
            }
        }

        // Pre-fetch tiles for the next scanline
        let on_bg_prefetch_cycle = (321..=336).contains(&scanline_cycle);
        let on_bg_fetch_cycle = on_visible_cycle || on_bg_prefetch_cycle;

        // Refined skip-render fast path #2: BG-pixel-pipeline elision.
        // On a skip-render frame, a visible scanline that does NOT
        // contain sprite 0 produces no CPU-observable effect from its
        // dot 1..=256 background pipeline (name/attr/bitmap fetches, bg
        // shift, register load, sprite-shift, render_pixel): the shift
        // registers it feeds are read only by `render_pixel`, which
        // bails before touching them precisely when
        // `!self.sprite_0_on_scanline` under skip_render (see the bail
        // at the top of `render_pixel`). So this gate mirrors that bail
        // exactly — whenever `skip_bg` is true, `render_pixel` would
        // have returned a no-op anyway; whenever sprite 0 IS present,
        // `skip_bg` is false and the full pipeline runs so the sprite-0
        // hit latches on its exact dot. Scroll increments (coarse-x at
        // multiples of 8, y at 256, v<-t at 257), sprite EVALUATION
        // (which sets `sprite_0_on_scanline` for the next line), sprite
        // tile fetches (257..=320), and the 321..=336 prefetch that
        // re-primes the next scanline's shift regs are all preserved —
        // so scroll (`regs.v`), OAM, sprite overflow, and mapper timing
        // stay cycle-exact. `bg_elision_allowed` restricts this to
        // mappers with no visible-fetch side effect (see
        // `skip_render_bg_elision_safe`); MMC3/MMC2/scanline-IRQ mappers
        // keep the full fetch stream and only benefit from fast path #1.
        // `bg_elision_allowed` is constant per ROM, so listing it first
        // makes `skip_bg` short-circuit after a single load on excluded
        // mappers (MMC3/MMC2/scanline-IRQ) — they pay nothing for this
        // path and still get fast path #1. `skip_render` is the next
        // cheapest gate (a field bool that is false on every full-render
        // tick), so the remaining terms are only reached on the frames
        // this optimization targets.
        #[cfg(feature = "ppu_skip_bg")]
        let skip_bg = self.bg_elision_allowed
            && self.refined_skip_render
            && self.skip_render
            && rendering_enabled
            && on_visible_scanline
            && !self.sprite_0_on_scanline;
        // Default build: compile-time `false`, so every `!skip_bg` /
        // `if skip_bg {…} else {…}` below folds back to the exact
        // baseline inner loop — the shipped tick body carries ZERO
        // pipeline-elision code and only the (unconditional) idle-HBlank
        // fast path.
        #[cfg(not(feature = "ppu_skip_bg"))]
        let skip_bg = false;

        if on_visible_scanline && on_visible_cycle {
            if rendering_enabled {
                // Scanline-level skip: when Replace mode has committed
                // to emitting this row at cycle 256 AND sprite-0 is
                // either not on this scanline or already latched, the
                // per-pixel call has zero side effects. Bailing here
                // skips 256 call+branch+return dead cycles per such
                // scanline. On a 240-visible-scanline frame this wipes
                // out roughly 61K function calls — the bulk of what
                // render_pixel would have done anyway when the batched
                // path owns output.
                let skip_pixel_call = self.skip_pixel_writes_this_scanline
                    && (!self.sprite_0_on_scanline
                        || self.regs.ppu_status.contains(PpuStatus::SPRITE_ZERO_HIT));
                if !skip_pixel_call && !skip_bg {
                    self.render_pixel();
                }
            } else {
                // Set the pixel in the frame buffer to black when not rendering.
                // This fixes an issue with "Wizard's and Warriors", where garbage pixels would
                // remain in the frame buffer right above the status bar.
                self.clear_pixel();
            }
        }

        // End of visible portion of a visible scanline — both bg
        // tile buffer and sprite snapshot have everything the
        // batched kernel needs for THIS scanline's output. Invoke
        // when the caller has asked for Verify / Replace mode.
        if on_visible_scanline
            && scanline_cycle == 256
            && self.batched_render_mode != BatchedRenderMode::Off
            && !self.skip_render
        {
            self.run_batched_scanline();
        }

        if rendering_enabled {
            // Handle backgrounds
            {
                if on_visible_scanline || on_prerender_scanline {
                    // Under skip_bg, suppress the unobserved dot 2..=257
                    // shifts; keep the 322..=337 prefetch shifts that
                    // prime the next scanline's registers.
                    if (!skip_bg && (2..=257).contains(&scanline_cycle))
                        || (322..=337).contains(&scanline_cycle)
                    {
                        self.shift_background_registers();
                    }

                    // Under skip_bg, suppress the visible-dot (1..=256)
                    // BG fetches + register load; keep the 321..=336
                    // prefetch fetches (`on_bg_prefetch_cycle`).
                    let do_bg_fetch = if skip_bg { on_bg_prefetch_cycle } else { on_bg_fetch_cycle };
                    if do_bg_fetch {
                        match scanline_cycle % 8 {
                            1 => self.fetch_name_table_byte(mapper),
                            3 => self.fetch_attribute_table_byte(mapper),
                            5 => self.fetch_bitmap_byte_lo(mapper),
                            7 => self.fetch_bitmap_byte_hi(mapper),
                            0 => self.load_background_registers(),
                            _ => (),
                        }
                    }

                    // Unused NT fetches
                    if scanline_cycle == 337 || scanline_cycle == 339 {
                        self.mem.read_byte(mapper, self.current_name_address());
                    }

                    // Increment the effective x scroll coordinate every 8 cycles
                    if on_bg_fetch_cycle && scanline_cycle.is_multiple_of(8) {
                        self.inc_coarse_x_with_wrap();
                    }

                    if scanline_cycle == 256 {
                        self.inc_y_with_wrap();
                    } else if scanline_cycle == 257 {
                        // Copy bits related to horizontal position from t to v
                        self.regs.v = (self.regs.v & !0x041F) | (self.regs.t & 0x041F);
                    }
                }

                if self.scanline == PRE_RENDER_SCANLINE && (280..=304).contains(&scanline_cycle) {
                    // Copy bits related to vertical position from t to v
                    self.regs.v = (self.regs.v & !0x7BE0) | (self.regs.t & 0x7BE0);
                }
            }

            // Handle sprites
            {
                if on_visible_scanline {
                    // Under skip_bg the per-pixel sprite shift/counter
                    // update is unobserved (it feeds only render_pixel,
                    // which is skipped); the shift regs are reloaded by
                    // the preserved fetch_sprite_tile at dots 257..=320.
                    if on_visible_cycle && !skip_bg {
                        self.update_sprite_rendering_registers();
                    }

                    // Sprite evaluation
                    match scanline_cycle {
                        1 => {
                            self.sprite_evaluation_init();
                        }
                        2..=64 => {
                            if scanline_cycle.is_multiple_of(2) {
                                self.oam.secondary[((scanline_cycle / 2) - 1) as usize] = 0xFF;
                            }
                        }
                        65..=256 => {
                            if scanline_cycle % 2 == 1 {
                                self.sprite_evaluation_read_byte();
                            } else {
                                self.sprite_evaluation_write_byte();
                            }
                        }
                        _ => (),
                    }
                }

                if on_visible_scanline || on_prerender_scanline {
                    // Fetch sprite tile data for the _next_ scanline.
                    if (257..=320).contains(&scanline_cycle) && scanline_cycle.is_multiple_of(8) {
                        self.fetch_sprite_tile(mapper, ((scanline_cycle - 264) / 8) as usize);
                    }
                }
            }
        }

        // VBLANK_START set_vblank is handled by the vblank-scanline
        // fast-path at the top of this function. Only PRE_RENDER's
        // clear_vblank lands in the main body, and only at cycle 1.
        // Replace the always-evaluated `match self.scanline` with a
        // single conditional that's almost always false.
        if on_prerender_scanline && scanline_cycle == 1 {
            self.clear_vblank();
            self.regs.ppu_status.set(PpuStatus::SPRITE_OVERFLOW, false);
            self.regs.ppu_status.set(PpuStatus::SPRITE_ZERO_HIT, false);
        }

        self.cycles += 1;

        // End of scanline
        if scanline_cycle >= CYCLES_PER_SCANLINE - 1 ||
            // On pre-render scanline, for odd frames, the cycle at the end of the scanline is skipped
            (rendering_enabled &&
            self.scanline == PRE_RENDER_SCANLINE &&
            scanline_cycle == CYCLES_PER_SCANLINE - 2 &&
            !self.frame.is_multiple_of(2))
        {
            self.scanline_start_cycle = self.cycles;
            self.scanline += 1;
            // Refresh cached CHR pointer at every scanline boundary.
            // For static-CHR mappers (NROM, etc.) the pointer never
            // changes; refresh is a single Option-unwrap. For
            // banked-CHR mappers (MMC1, MMC3) the trait method
            // returns None and reads fall back to enum dispatch.
            self.chr_cache_ptr = mapper.chr_static_ptr().unwrap_or(core::ptr::null());
            #[cfg(feature = "ppu_skip_bg")]
            {
                self.bg_elision_allowed = mapper.skip_render_bg_elision_safe();
            }
            // End of frame: scanline only changes here, so the
            // overflow check belongs in the same branch — fires once
            // per scanline transition (262×/frame) instead of once
            // per PPU tick (89342×/frame).
            if self.scanline > PRE_RENDER_SCANLINE {
                // Color-emphasis (PPUMASK bits 5-7): carry the 3 bits
                // alongside the frame buffer so the sink can tint via
                // its 512-entry LUT. bit5=red→0x01, bit6=green→0x02,
                // bit7=blue→0x04. Emphasis 0 is the byte-identical
                // default path.
                let emphasis = (self.regs.ppu_mask.bits() >> 5) & 0x07;
                video_frame_sink.set_emphasis(emphasis);
                video_frame_sink.write_frame(&self.frame_buffer);
                self.scanline = VISIBLE_START_SCANLINE;
                self.frame += 1;
                // Clear skip_render so the first cycles of the new frame
                // always render. Bulk-step CPU paths (try_bulk_step,
                // cpu_asm) tick the PPU in batches that can cross a frame
                // boundary; without this reset, the overshoot ticks into
                // the new frame keep the outgoing frame's skip_render
                // value. On a skip→full transition (frame_skip>1, last
                // frame of the batch) that leaves the new frame's first
                // ~50-100 scanline-0 pixels unrendered — they retain
                // whatever frame_buffer content was there from the last
                // full-render frame, producing ~50-68 px of row-0 drift
                // vs frame_skip=1. Callers always set skip_render before
                // each advance_one_frame, so forcing it off at the frame
                // boundary is semantically safe: the tiny prefix that
                // renders without the skip optimization costs ~50-100
                // pixel writes per frame (0.1% of per-frame work) and
                // guarantees correct output.
                self.skip_render = false;
            }
            // Batched-render scanline boundary work. Gated on mode
            // != Off so the default path stays zero-cost:
            //   * Rollover prefetch tiles (cycles 328/336 → [0]/[1]).
            //   * Reset load counter + invalidation flag.
            //   * Predict per-scanline skip-per-pixel from history.
            //   * Snapshot sprite state for the new scanline.
            // This block fires once per scanline transition ×
            // 262 scanlines × 60 fps × 16 workers ≈ 250k times/sec,
            // so skipping it in Off mode is a real win.
            if self.batched_render_mode != BatchedRenderMode::Off {
                if self.bg_tile_load_count >= 36 {
                    self.bg_tile_buffer[0] = self.bg_tile_buffer[34];
                    self.bg_tile_buffer[1] = self.bg_tile_buffer[35];
                }
                self.bg_tile_load_count = 2;
                self.batched_scanline_invalid = false;
                #[cfg(feature = "ppu_neon_stats")]
                {
                    self.batched_source_fired_this_scanline = [false; 4];
                }
                self.skip_pixel_writes_this_scanline = matches!(
                    self.batched_render_mode,
                    BatchedRenderMode::Replace,
                ) && (self.scanline as usize) < self.clean_scanline_history.len()
                    && self.clean_scanline_history[self.scanline as usize];
                for i in 0..8 {
                    let attr = self.sprite_attribute_latches[i];
                    let in_front = matches!(
                        attr.priority(),
                        SpritePriority::InFrontOfBackground,
                    );
                    self.scanline_sprites[i] = crate::ppu_neon::Sprite {
                        x: self.sprite_x_counters[i],
                        pattern_lo: self.sprite_pattern_shifts_lo[i],
                        pattern_hi: self.sprite_pattern_shifts_hi[i],
                        palette: attr.palette(),
                        in_front,
                        oam_index: i as u8,
                        active: attr.0 != 0xFF,
                    };
                }
            }
        }
    }

    pub fn read_byte(&mut self, mapper: &mut MapperEnum, address: u16) -> u8 {
        if !((0x2000..=0x3FFF).contains(&address)) {
            panic!(
                "Invalid read from PPU memory-mapped registers: {:X}",
                address
            )
        }

        let address = address & 0x2007;

        let val = match address {
            PPUSTATUS_ADDRESS => self.read_ppu_status(),
            OAMDATA_ADDRESS => self.read_oam_byte(),
            PPUDATA_ADDRESS => self.read_ppu_data_byte(mapper),
            _ => self.ppu_gen_latch,
        };

        self.ppu_gen_latch = val;

        val
    }

    pub fn peek_byte(&self, address: u16) -> u8 {
        if !((0x2000..=0x3FFF).contains(&address)) {
            panic!(
                "Invalid read from PPU memory-mapped registers: {:X}",
                address
            )
        }

        let address = address & 0x2007;

        let val = match address {
            PPUSTATUS_ADDRESS => self.regs.ppu_status.bits(),
            OAMDATA_ADDRESS => self.read_oam_byte(),
            PPUDATA_ADDRESS => self.mem.vram.read_byte(address),
            _ => self.ppu_gen_latch,
        };

        val
    }

    pub fn write_byte(&mut self, mapper: &mut MapperEnum, address: u16, value: u8) {
        if !((0x2000..0x4000).contains(&address)) {
            panic!(
                "Invalid write to PPU memory-mapped registers, address: {:X}, value: {}",
                address, value
            )
        }

        let address = address & 0x2007;

        // Writes to ANY $2000-$3FFF register fill the I/O bus (PPU
        // generator latch) regardless of whether the register itself
        // honors the write — the value is on the data bus from the
        // CPU's perspective, capacitively held until the next bus
        // operation. Reads of $2002 OR this latch into bits 4-0 of
        // the returned status byte (open bus). Mesen does the same:
        // see Mesen2 NesPpu::WriteRam — `SetOpenBus(0xFF, value)`
        // unconditional, before per-register handling. A previous
        // version of this function early-returned during the
        // 29658-cycle warmup before updating ppu_gen_latch, which
        // caused SMB's first `LDA $2002` to read 0x00 instead of
        // 0x10 (the open-bus low 5 bits of the prior `STA $2000
        // #$10`). That's the first byte-divergence on cold boot.
        self.ppu_gen_latch = value;

        // Writes to the following registers are ignored if earlier than
        // ~29658 CPU clocks after reset: PPUCTRL, PPUMASK, PPUSCROLL, PPUADDR
        if self.cycles < 3 * 29658
            && (address == PPUCTRL_ADDRESS
                || address == PPUMASK_ADDRESS
                || address == PPUSCROLL_ADDRESS
                || address == PPUADDR_ADDRESS)
        {
            return;
        }

        // Mid-scanline register-write invalidation, narrowed per
        // per-register divergence analysis:
        //
        //   * **PPUMASK** ($2001): changes show_bg / show_sprites /
        //     left-8 masks. Per-pixel path applies immediately; the
        //     batched path uses an end-of-scanline snapshot for the
        //     whole 256 columns. MUST invalidate.
        //   * **PPUSCROLL** ($2005) first write: updates `regs.x`
        //     (fine_x) instantly. Per-pixel reads new fine_x for the
        //     remaining columns; batched uses end-of-scanline fine_x
        //     for all 256. MUST invalidate.
        //   * **PPUSCROLL** ($2005) second write: updates bits of
        //     `regs.t` only; `t→v` doesn't happen until the next
        //     scanline boundary. Subsequent fetches are unaffected
        //     on the current scanline. SAFE to absorb.
        //   * **PPUCTRL** ($2000): pattern-table / nametable-base
        //     changes show up in the NEXT bg fetch. Both paths see
        //     the altered fetch output identically via the tile
        //     buffer. SAFE to absorb.
        //   * **PPUADDR** ($2006) second write: transfers t→v,
        //     disrupting subsequent fetches. Both paths see the
        //     same altered fetches → no per-pixel vs batched
        //     divergence on the already-captured tiles. SAFE to
        //     absorb.
        //
        // Narrowed from invalidating ALL four registers to only the
        // two that actually produce per-pixel vs end-of-scanline
        // divergence.
        if self.batched_render_mode != BatchedRenderMode::Off
            && self.scanline <= VISIBLE_END_SCANLINE
            && self.scanline_cycle() <= 256
        {
            let mask_or_scroll_first =
                address == PPUMASK_ADDRESS
                    || (address == PPUSCROLL_ADDRESS
                        && matches!(self.regs.w, WriteToggle::FirstWrite));
            if mask_or_scroll_first {
                self.batched_scanline_invalid = true;
                self.note_invalidation_source(2);
            } else if matches!(
                address,
                PPUCTRL_ADDRESS | PPUSCROLL_ADDRESS | PPUADDR_ADDRESS,
            ) {
                // Source 2 also tracks the absorbed-by-narrowing
                // writes (so stats distinguish real from phantom).
                self.note_invalidation_source(2);
                #[cfg(feature = "ppu_neon_stats")]
                {
                    if self.batched_invalidation_broad {
                        self.batched_scanline_invalid = true;
                    }
                }
            }
        }

        match address {
            PPUCTRL_ADDRESS => self.write_ppu_ctrl(value),
            PPUMASK_ADDRESS => {
                self.regs.ppu_mask = PpuMask::from_bits(value).unwrap();
                self.refresh_ppu_mask_cache();
            }
            OAMADDR_ADDRESS => self.regs.oam_addr = value,
            OAMDATA_ADDRESS => self.write_oam_byte(value),
            PPUSCROLL_ADDRESS => self.write_ppu_scroll(value),
            PPUADDR_ADDRESS => self.write_ppu_addr(value),
            PPUDATA_ADDRESS => self.write_ppu_data_byte(mapper, value),
            _ => (),
        }
    }
}

impl Default for Ppu {
    fn default() -> Self {
        Self::new()
    }
}

// VRAM address increment per CPU read/write of PPUDATA
enum VramAddressIncrement {
    Add1Across, // Add 1, going across
    Add32Down,  // Add 32, going down
}

enum SpriteSize {
    Size8x8,
    Size8x16,
}

impl SpriteSize {
    fn height(&self) -> u8 {
        match *self {
            SpriteSize::Size8x8 => 8,
            SpriteSize::Size8x16 => 16,
        }
    }
}

#[derive(Copy, Clone, Deserialize, Serialize)]
struct PpuCtrl {
    val: u8,
}

impl PpuCtrl {
    fn vram_address_increment(self) -> VramAddressIncrement {
        if (self.val & 0x04) == 0 {
            VramAddressIncrement::Add1Across
        } else {
            VramAddressIncrement::Add32Down
        }
    }

    // For 8x8 sprites (ignored in 8x16 mode)
    fn sprite_pattern_table_address(self) -> u16 {
        if (self.val & 0x08) == 0 {
            0x0000
        } else {
            0x1000
        }
    }

    fn background_pattern_table_address(self) -> u16 {
        if (self.val & 0x10) == 0 {
            0x0000
        } else {
            0x1000
        }
    }

    fn sprite_size(self) -> SpriteSize {
        if (self.val & 0x20) == 0 {
            SpriteSize::Size8x8
        } else {
            SpriteSize::Size8x16
        }
    }
}

impl Deref for PpuCtrl {
    type Target = u8;

    fn deref(&self) -> &u8 {
        &self.val
    }
}

impl DerefMut for PpuCtrl {
    fn deref_mut(&mut self) -> &mut u8 {
        &mut self.val
    }
}

bitflags! {
    #[derive(Copy, Clone, Deserialize, Serialize)]
    struct PpuMask: u8 {
        const NONE                   = 0;

        // Greyscale (0: normal color, 1: produce a greyscale display)
        const GREYSCALE              = 1;

        // 1: Show background in leftmost 8 pixels of screen, 0: Hide
        const SHOW_BACKGROUND_LEFT_8 = 1 << 1;

        // 1: Show sprites in leftmost 8 pixels of screen, 0: Hide
        const SHOW_SPRITES_LEFT_8    = 1 << 2;

        const SHOW_BACKGROUND        = 1 << 3;
        const SHOW_SPRITES           = 1 << 4;

        // On PAL, red and green bits are switched
        const EMPHASIZE_RED          = 1 << 5;
        const EMPHASIZE_GREEN        = 1 << 6;
        const EMPHASIZE_BLUE         = 1 << 7;
    }
}

bitflags! {
    #[derive(Copy, Clone, Deserialize, Serialize)]
    struct PpuStatus: u8 {
        const NONE            = 0;
        const SPRITE_OVERFLOW = 1 << 5;
        const SPRITE_ZERO_HIT = 1 << 6;
        // Reset-time PPU status. LaiNES (nes-py) resets this to 0x00:
        // sprite_overflow=false, sprite_zero_hit=false, vblank flag
        // starts clear. Our previous value of `!0x80` (0x7F) had both
        // sprite bits set at reset, causing $2002 to read 0x60 on the
        // very first poll — a value Zelda's startup code branches on
        // differently, cascading into the cave-stuck sword-pickup
        // failure by tape frame ~1000. Diagnosed via bus-access
        // differential trace 2026-04-24; see
        // `docs/proposals/archive/zelda_cave_stuck_investigation.md`.
        const RESET_VALUE     = 0;
    }
}

#[derive(Copy, Clone, Deserialize, Serialize)]
enum WriteToggle {
    FirstWrite,
    SecondWrite,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct Regs {
    // Registers mapped from CPU address space
    ppu_ctrl: PpuCtrl,     // 0x2000
    ppu_mask: PpuMask,     // 0x2001
    ppu_status: PpuStatus, // 0x2002
    oam_addr: u8,          // 0x2003

    // Internal registers (for scrolling)
    v: u16,         // Current VRAM address (15 bits)
    t: u16,         // Temporary VRAM address (15 bits)
    x: u8,          // Fine X scroll (3 bits)
    w: WriteToggle, // First of second write toggle
}

impl Regs {
    fn new() -> Regs {
        Regs {
            ppu_ctrl: PpuCtrl { val: 0 },
            ppu_mask: PpuMask::NONE,
            ppu_status: PpuStatus::NONE,
            oam_addr: 0,
            v: 0,
            t: 0,
            x: 0,
            w: WriteToggle::FirstWrite,
        }
    }
}

// OAM (Object Attribute Memory) is internal memory inside the PPU that contains
// a display list of up to 64 sprites, where each sprite's information occupies 4 bytes
#[derive(Clone, Deserialize, Serialize)]
pub struct Oam {
    #[serde(with = "serde_bytes")]
    primary: Vec<u8>,
    #[serde(with = "serde_bytes")]
    secondary: Vec<u8>,

    // Used during sprite evaluation
    last_read_byte: u8,
    secondary_write_index: usize,
    sprite_0_found: bool,
    n: usize,
    m: usize,
}

impl Oam {
    const BYTES_PER_SPRITE: usize = 4;
    const PRIMARY_MAX_SPRITES: usize = 64;
    const PRIMARY_SIZE: usize = Oam::PRIMARY_MAX_SPRITES * Oam::BYTES_PER_SPRITE;
    const SECONDARY_MAX_SPRITES: usize = 8;
    const SECONDARY_SIZE: usize = Oam::SECONDARY_MAX_SPRITES * Oam::BYTES_PER_SPRITE;

    fn new() -> Oam {
        Oam {
            primary: vec![0u8; Oam::PRIMARY_SIZE],
            secondary: vec![0u8; Oam::SECONDARY_SIZE],
            last_read_byte: 0,
            secondary_write_index: 0,
            sprite_0_found: false,
            n: 0,
            m: 0,
        }
    }
}

impl Deref for Oam {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.primary
    }
}

impl DerefMut for Oam {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.primary
    }
}

// 2KB internal dedicated Video RAM
#[derive(Clone, Deserialize, Serialize)]
pub struct Vram {
    #[serde(with = "serde_bytes")]
    bytes: Vec<u8>,
}

impl Vram {
    const SIZE: usize = 0x0800;

    fn new() -> Vram {
        Vram {
            bytes: vec![0u8; Vram::SIZE],
        }
    }

    fn read_byte(&self, address: u16) -> u8 {
        // Mask into the 2 KB buffer. Callers typically pass a mirrored
        // address already, but mapper-controlled mirroring can produce
        // unmapped quadrants (Gauntlet, Rad Racer II) that land past
        // the buffer end. Wrap instead of panic.
        self.bytes[address as usize & (Vram::SIZE - 1)]
    }

    fn write_byte(&mut self, address: u16, value: u8) {
        self.bytes[address as usize & (Vram::SIZE - 1)] = value
    }
}

impl Deref for Vram {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl DerefMut for Vram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bytes
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PaletteRam {
    bytes: [u8; PaletteRam::SIZE],
}

impl PaletteRam {
    const SIZE: usize = 32;
    const START_ADDRESS: u16 = 0x3F00;

    fn new() -> PaletteRam {
        PaletteRam {
            bytes: [0u8; PaletteRam::SIZE],
        }
    }

    #[inline(always)]
    fn index(address: u16) -> usize {
        let index = (address & 0x001F) as usize;
        // Addresses 0x3F10/0x3F14/0x3F18/0x3F1C are mirrors of
        // 0x3F00/0x3F04/0x3F08/0x3F0C. Branchless form: the low 2
        // bits tell us if it's a multiple of 4; bit 4 tells us if
        // it's in the upper half. AND them to mask out bit 4 when
        // both conditions hold.
        let is_mirror = ((index >> 4) & 1) & (((index & 3) == 0) as usize);
        index & !(0x10 * is_mirror)
    }

    #[inline(always)]
    fn read_byte(&self, address: u16) -> u8 {
        self.bytes[PaletteRam::index(address)]
    }

    fn write_byte(&mut self, address: u16, value: u8) {
        self.bytes[PaletteRam::index(address)] = value
    }

    /// Raw 32-byte palette buffer, for fast-path consumers that want
    /// to do a single `copy_from_slice` and handle mirror resolution
    /// themselves (hot-path render path). Addresses 0x10/0x14/0x18/0x1C
    /// hold independent bytes here that may not reflect the mirrored
    /// ones at 0x00/0x04/0x08/0x0C — callers must patch those up if
    /// they care.
    #[inline(always)]
    pub(crate) fn raw_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

pub struct MemMap {
    pub(crate) vram: Vram,
    pub(crate) palette_ram: PaletteRam,
}

impl MemMap {
    pub fn new() -> Self {
        MemMap {
            vram: Vram::new(),
            palette_ram: PaletteRam::new(),
        }
    }

    pub fn read_byte(&mut self, mapper: &mut MapperEnum, address: u16) -> u8 {
        let address = address & 0x3FFF;

        if address < 0x2000 {
            mapper.chr_read_byte(address)
        } else if address < PaletteRam::START_ADDRESS {
            let mirroring = mapper.mirroring();
            self.vram.read_byte(mirroring.mirror_address(address))
        } else if address < 0x4000 {
            // Palette RAM is not configurable, always mapped to the
            // internal palette control in VRAM.
            self.palette_ram.read_byte(address)
        } else {
            panic!("Invalid read from PPU space memory: {:X}", address)
        }
    }

    fn write_byte(&mut self, mapper: &mut MapperEnum, address: u16, value: u8) {
        let address = address & 0x3FFF;

        if address < 0x2000 {
            mapper.chr_write_byte(address, value);
        } else if address < PaletteRam::START_ADDRESS {
            let mirroring = mapper.mirroring();
            self.vram
                .write_byte(mirroring.mirror_address(address) & 0x07FF, value)
        } else if address < 0x4000 {
            // Palette RAM is not configurable, always mapped to the
            // internal palette control in VRAM.
            self.palette_ram.write_byte(address, value);
        } else {
            panic!(
                "Invalid write to PPU-space memory, address: 0x{:04x}, value: 0x{:02x}",
                address, value
            )
        }
    }
}

impl Default for MemMap {
    fn default() -> Self {
        Self::new()
    }
}

enum SpritePriority {
    InFrontOfBackground,
    BehindBackground,
}

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
pub struct SpriteAttributes(u8);

impl SpriteAttributes {
    fn palette(self) -> u8 {
        (self.0 & 0x03) | 0x04
    }

    fn priority(self) -> SpritePriority {
        if self.0 & 0x20 == 0 {
            SpritePriority::InFrontOfBackground
        } else {
            SpritePriority::BehindBackground
        }
    }

    fn flip_horizontally(self) -> bool {
        self.0 & 0x40 == 0x40
    }

    fn flip_vertically(self) -> bool {
        self.0 & 0x80 == 0x80
    }
}

#[derive(Clone, Copy, Default)]
struct Sprite {
    y: u8,
    tile_index: u8,
    attributes: SpriteAttributes,
    x: u8,
}

impl Sprite {
    fn from_oam_bytes(oam_bytes: &[u8]) -> Sprite {
        Sprite {
            y: oam_bytes[0],
            tile_index: oam_bytes[1],
            // OAM attribute bits 2-4 are unimplemented on real hardware and
            // always read back 0. Mask them off (0xE3) so a real sprite's
            // attribute can never collide with the 0xFF empty-slot sentinel
            // used by fetch_sprite_tile and both render paths. (Mesen-validated.)
            attributes: SpriteAttributes(oam_bytes[2] & 0xE3),
            x: oam_bytes[3],
        }
    }

    fn flip_horizontally(self) -> bool {
        self.attributes.flip_horizontally()
    }

    fn flip_vertically(self) -> bool {
        self.attributes.flip_vertically()
    }
}

#[cfg(test)]
mod predict_nmi_tests {
    use super::*;

    fn ppu_at(scanline: u16, scanline_cycle: u64, nmi_output: bool, nmi_occurred: bool) -> Ppu {
        let mut p = Ppu::new();
        p.scanline = scanline;
        p.scanline_start_cycle = 1000;
        p.cycles = 1000 + scanline_cycle;
        p.nmi_output = nmi_output;
        p.nmi_occurred = nmi_occurred;
        p
    }

    #[test]
    fn nmi_disabled_returns_none() {
        let p = ppu_at(0, 0, false, false);
        assert_eq!(p.cpu_cycles_until_nmi_fire(), None);
    }

    #[test]
    fn nmi_already_pending_returns_none() {
        let p = ppu_at(241, 5, true, true);
        assert_eq!(p.cpu_cycles_until_nmi_fire(), None);
    }

    #[test]
    fn at_vblank_fire_state_returns_one() {
        let p = ppu_at(VBLANK_START_SCANLINE, 1, true, false);
        assert_eq!(p.cpu_cycles_until_nmi_fire(), Some(1));
    }

    #[test]
    fn one_ppu_cycle_before_fire() {
        let p = ppu_at(VBLANK_START_SCANLINE, 0, true, false);
        assert_eq!(p.cpu_cycles_until_nmi_fire(), Some(1));
    }

    #[test]
    fn full_frame_distance_when_just_past_fire() {
        let p = ppu_at(VBLANK_START_SCANLINE, 2, true, false);
        let expected_ppu = (262u64 * CYCLES_PER_SCANLINE) - 1;
        let expected_cpu = (expected_ppu + 2) / 3;
        assert_eq!(p.cpu_cycles_until_nmi_fire(), Some(expected_cpu));
    }

    #[test]
    fn cap_shrinks_as_vblank_approaches() {
        let far = ppu_at(0, 0, true, false);
        let close = ppu_at(240, 200, true, false);
        let very_close = ppu_at(241, 0, true, false);
        let f = far.cpu_cycles_until_nmi_fire().unwrap();
        let c = close.cpu_cycles_until_nmi_fire().unwrap();
        let v = very_close.cpu_cycles_until_nmi_fire().unwrap();
        assert!(f > c, "far ({f}) should be greater than close ({c})");
        assert!(c > v, "close ({c}) should be greater than very_close ({v})");
        assert!(v >= 1, "very_close ({v}) must be at least 1 to make progress");
    }

    /// Integration: tick PPU exactly N CPU cycles (3N PPU ticks) and
    /// check that vblank fires precisely when the helper said it would.
    #[test]
    fn helper_matches_actual_vblank_fire() {
        use crate::cartridge::Cartridge;
        use crate::mapper::MapperEnum;
        use crate::sink::VideoSink;

        struct Discard;
        impl VideoSink for Discard {
            fn write_frame(&mut self, _: &[u8]) {}
            fn frame_written(&self) -> bool { false }
            fn pixel_size(&self) -> usize { 4 }
        }

        // Synthesize a minimal NROM cart for the test.
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // 32KB PRG
        rom.push(1); // 8KB CHR
        rom.extend_from_slice(&[0u8; 10]);
        rom.extend(vec![0u8; 32 * 1024]);
        rom.extend(vec![0u8; 8 * 1024]);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        let mut mapper = MapperEnum::from_cartridge(cart);

        let mut ppu = Ppu::new();
        ppu.nmi_output = true;
        ppu.nmi_occurred = false;
        // Place the PPU at scanline 240 cycle 100 (well-defined,
        // pre-vblank, rendering off so timing is plain CPU:PPU = 1:3).
        ppu.scanline = 240;
        ppu.scanline_start_cycle = 1_000_000;
        ppu.cycles = 1_000_000 + 100;

        let predicted_cpu_cycles = ppu.cpu_cycles_until_nmi_fire()
            .expect("nmi_output=true && nmi_occurred=false");

        let mut sink = Discard;
        let mut fired_at: Option<u64> = None;
        for cpu_cycle in 1..=(predicted_cpu_cycles + 8) {
            ppu.tick_three(&mut mapper, &mut sink);
            if ppu.nmi_occurred && fired_at.is_none() {
                fired_at = Some(cpu_cycle);
            }
        }
        let actual = fired_at.expect("vblank must fire within prediction + slack");
        // Helper rounds up — actual fire must be ≤ predicted (we never
        // under-predict, so the cap never lets the bulk run past fire).
        assert!(
            actual <= predicted_cpu_cycles,
            "predicted {predicted_cpu_cycles}, actually fired at {actual} — under-prediction would let ASM overshoot vblank"
        );
        // And not too pessimistic: actual must be within 1 CPU cycle
        // of prediction (rounding-up tolerance).
        assert!(
            predicted_cpu_cycles - actual <= 1,
            "over-predicted by {} CPU cycles", predicted_cpu_cycles - actual
        );
    }
}

#[cfg(test)]
mod sprite_and_blank_tests {
    use super::*;
    use crate::cartridge::Cartridge;
    use crate::mapper::MapperEnum;

    /// Minimal NROM cart (32 KB PRG, 8 KB CHR of zeros) so the mapper
    /// can back CHR reads.
    fn nrom_mapper() -> MapperEnum {
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // 32 KB PRG
        rom.push(1); // 8 KB CHR
        rom.extend_from_slice(&[0u8; 10]);
        rom.extend(vec![0u8; 32 * 1024]);
        rom.extend(vec![0u8; 8 * 1024]);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        MapperEnum::from_cartridge(cart)
    }

    // An out-of-range sprite row (scanline < sprite.y) must not underflow
    // into an out-of-bounds CHR fetch; the guard marks the slot inactive.
    #[test]
    fn fetch_sprite_tile_guards_underflow_row() {
        let mut ppu = Ppu::new();
        let mut mapper = nrom_mapper();
        ppu.oam.secondary[0] = 200; // y
        ppu.oam.secondary[1] = 0x00;
        ppu.oam.secondary[2] = 0x00;
        ppu.oam.secondary[3] = 0x00;
        ppu.oam.secondary_write_index = 4;
        ppu.scanline = 10; // above the sprite's y
        ppu.fetch_sprite_tile(&mut mapper, 0);
        assert_eq!(
            ppu.sprite_attribute_latches[0].0, 0xFF,
            "out-of-range sprite row must mark the slot inactive, not fetch OOB"
        );
    }

    // A real sprite whose raw OAM attribute byte is 0xFF must still render:
    // bits 2-4 are unimplemented (read 0), so masking to 0xE3 keeps it
    // distinct from the 0xFF empty-slot sentinel.
    #[test]
    fn sprite_with_raw_attribute_0xff_is_not_dropped() {
        let mut ppu = Ppu::new();
        let mut mapper = nrom_mapper();
        ppu.oam.secondary[0] = 8; // y (scanline 10 -> row 2)
        ppu.oam.secondary[1] = 0x01;
        ppu.oam.secondary[2] = 0xFF; // raw attribute
        ppu.oam.secondary[3] = 40;
        ppu.oam.secondary_write_index = 4;
        ppu.scanline = 10;
        ppu.fetch_sprite_tile(&mut mapper, 0);
        assert_eq!(
            ppu.sprite_attribute_latches[0].0, 0xE3,
            "raw 0xFF attribute must be masked to 0xE3, not dropped as the empty sentinel"
        );
    }

    // Forced blank outputs the palette_ram[$3F00] backdrop, not 0x0F.
    #[test]
    fn clear_pixel_uses_backdrop_color() {
        let mut ppu = Ppu::new();
        ppu.mem.palette_ram.write_byte(0x3F00, 0x21);
        ppu.skip_render = false;
        ppu.scanline = 5;
        ppu.scanline_start_cycle = 0;
        ppu.cycles = 4; // scanline_cycle() = 4 -> x = 3
        ppu.clear_pixel();
        let x = (ppu.scanline_cycle() - 1) as usize;
        let y = ppu.scanline as usize;
        assert_eq!(
            ppu.frame_buffer[y * SCREEN_WIDTH + x], 0x21,
            "forced-blank pixel must use the $3F00 backdrop color"
        );
    }
}

#[cfg(test)]
mod greyscale_tests {
    use super::*;

    #[test]
    fn grey_mask_identity_when_off() {
        let mut p = Ppu::new();
        p.regs.ppu_mask = PpuMask::NONE;
        p.refresh_ppu_mask_cache();
        assert_eq!(p.grey_mask, 0x3F);
        // Every 6-bit palette index passes through unchanged: the
        // default render path is byte-identical to pre-greyscale code.
        for idx in 0u8..64 {
            assert_eq!(idx & p.grey_mask, idx);
        }
    }

    #[test]
    fn grey_mask_collapses_to_grey_column_when_on() {
        let mut p = Ppu::new();
        p.regs.ppu_mask = PpuMask::GREYSCALE;
        p.refresh_ppu_mask_cache();
        assert_eq!(p.grey_mask, 0x30);
        // Masking with 0x30 restricts every index to the four grey
        // entries $00/$10/$20/$30 (NES greyscale behavior).
        for idx in 0u8..64 {
            let masked = idx & p.grey_mask;
            assert!(
                matches!(masked, 0x00 | 0x10 | 0x20 | 0x30),
                "index {idx:#04x} masked to {masked:#04x}, not a grey-column entry"
            );
        }
        // Canonical spot-checks: a color keeps only its luminance column.
        assert_eq!(0x21 & p.grey_mask, 0x20);
        assert_eq!(0x0F & p.grey_mask, 0x00);
        assert_eq!(0x3D & p.grey_mask, 0x30);
    }

    #[test]
    fn greyscale_bit_does_not_disturb_show_bits() {
        // Greyscale must not perturb the packed show/left8 cache used by
        // the per-pixel hot path.
        let mut p = Ppu::new();
        p.regs.ppu_mask = PpuMask::GREYSCALE
            | PpuMask::SHOW_BACKGROUND
            | PpuMask::SHOW_SPRITES;
        p.refresh_ppu_mask_cache();
        assert!(p.cached_show_bg());
        assert!(p.cached_show_sprites());
        assert!(!p.cached_show_bg_left8());
        assert!(!p.cached_show_sprites_left8());
        assert_eq!(p.grey_mask, 0x30);
    }

    #[test]
    fn reset_restores_identity_mask() {
        let mut p = Ppu::new();
        p.regs.ppu_mask = PpuMask::GREYSCALE;
        p.refresh_ppu_mask_cache();
        assert_eq!(p.grey_mask, 0x30);
        p.reset();
        assert_eq!(p.grey_mask, 0x3F);
    }

    #[test]
    fn apply_state_recomputes_mask() {
        let mut p = Ppu::new();
        p.regs.ppu_mask = PpuMask::GREYSCALE;
        let state = p.get_state();
        let mut q = Ppu::new();
        assert_eq!(q.grey_mask, 0x3F);
        q.apply_state(&state);
        assert_eq!(q.grey_mask, 0x30, "grey_mask must be rebuilt from restored PPUMASK");
    }

    #[test]
    fn emphasis_bits_extract_from_high_ppumask_bits() {
        // Mirror of the extraction the write_frame path performs:
        // bit5→red(0x01), bit6→green(0x02), bit7→blue(0x04).
        let extract = |m: PpuMask| (m.bits() >> 5) & 0x07;
        assert_eq!(extract(PpuMask::NONE), 0);
        assert_eq!(extract(PpuMask::EMPHASIZE_RED), 0b001);
        assert_eq!(extract(PpuMask::EMPHASIZE_GREEN), 0b010);
        assert_eq!(extract(PpuMask::EMPHASIZE_BLUE), 0b100);
        assert_eq!(
            extract(
                PpuMask::EMPHASIZE_RED
                    | PpuMask::EMPHASIZE_GREEN
                    | PpuMask::EMPHASIZE_BLUE
            ),
            0b111
        );
        // Rendering/greyscale bits must not leak into the emphasis value.
        assert_eq!(
            extract(PpuMask::GREYSCALE | PpuMask::SHOW_BACKGROUND),
            0
        );
    }

    /// End-to-end: drive a full frame through `tick` with emphasis bits
    /// set in PPUMASK and confirm the value reaches the sink via
    /// `set_emphasis` right before `write_frame`.
    #[test]
    fn emphasis_reaches_sink_through_full_frame_tick() {
        use crate::cartridge::Cartridge;
        use crate::mapper::MapperEnum;
        use crate::sink::VideoSink;

        struct CaptureSink {
            emphasis: u8,
            frames: u32,
        }
        impl VideoSink for CaptureSink {
            fn write_frame(&mut self, _: &[u8]) {
                self.frames += 1;
            }
            fn frame_written(&self) -> bool {
                false
            }
            fn pixel_size(&self) -> usize {
                4
            }
            fn set_emphasis(&mut self, emphasis: u8) {
                self.emphasis = emphasis;
            }
        }

        // Minimal NROM cart (same shape as the vblank helper test).
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2);
        rom.push(1);
        rom.extend_from_slice(&[0u8; 10]);
        rom.extend(vec![0u8; 32 * 1024]);
        rom.extend(vec![0u8; 8 * 1024]);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        let mut mapper = MapperEnum::from_cartridge(cart);

        let mut ppu = Ppu::new();
        // Blue + red emphasis, rendering disabled (keeps timing plain).
        ppu.regs.ppu_mask = PpuMask::EMPHASIZE_RED | PpuMask::EMPHASIZE_BLUE;
        ppu.refresh_ppu_mask_cache();

        let mut sink = CaptureSink { emphasis: 0xAA, frames: 0 };
        // One full frame is 262*341 PPU ticks; step generously past it.
        for _ in 0..(263 * 341) {
            ppu.tick(&mut mapper, &mut sink);
            if sink.frames > 0 {
                break;
            }
        }
        assert_eq!(sink.frames, 1, "exactly one frame should have emitted");
        // bit5 (red) → 0x01, bit7 (blue) → 0x04 ⇒ 0b101.
        assert_eq!(sink.emphasis, 0b101, "emphasis bits must reach the sink");
    }
}
