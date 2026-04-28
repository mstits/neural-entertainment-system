//! Per-frame fallback-rate report for BatchedRenderMode::Replace.
//!
//! Runs a ROM for N frames in Replace mode and reports:
//!   * Total visible scanlines evaluated by the batched path.
//!   * Scanlines that fell back to per-pixel (invalidation flag set).
//!   * Per-source invalidation counts (palette, nametable $2007,
//!     register writes, mapper writes).
//!
//! Requires `--features "asm_cpu ppu_neon_stats"`.
//!
//! Usage:
//!   cargo run --release --features "asm_cpu ppu_neon_stats" \
//!     --example ppu_neon_stats -- <rom-path> [frames]

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{self, BatchedRenderMode};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../roms/Contra (USA).nes".into());
    let target_frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let bytes = std::fs::read(&path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    println!(
        "ppu_neon_stats: {} (mapper={}, {} frames)",
        path, cart.mapper, target_frames,
    );

    let broad = std::env::var("PPU_NEON_STATS_BROAD")
        .ok()
        .map(|s| s == "1")
        .unwrap_or(false);

    let mut nes = Nes::new(cart);
    nes.ppu.set_batched_render_mode(BatchedRenderMode::Replace);
    #[cfg(feature = "ppu_neon_stats")]
    {
        nes.ppu.batched_invalidation_broad = broad;
        println!(
            "invalidation ruleset: {}",
            if broad { "BROAD (baseline)" } else { "NARROW (current)" },
        );
    }

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = NullAudio;

    // Warmup — title screens with animation load patterns that push
    // the PPU through its common register-write patterns.
    for _ in 0..30 {
        let mut v = Xrgb8888VideoSink::new(&mut video_buf);
        while !v.frame_written() {
            nes.step(&mut v, &mut audio);
        }
    }

    // Reset counters so the measurement window is exactly target_frames.
    nes.ppu.set_batched_render_mode(BatchedRenderMode::Replace);
    #[cfg(feature = "ppu_neon_stats")]
    {
        nes.ppu.batched_invalidation_broad = broad;
    }

    for _ in 0..target_frames {
        let mut v = Xrgb8888VideoSink::new(&mut video_buf);
        while !v.frame_written() {
            nes.step(&mut v, &mut audio);
        }
    }

    #[cfg(feature = "ppu_neon_stats")]
    {
        let total = nes.ppu.batched_scanlines_total;
        let fallback = nes.ppu.batched_scanlines_fallback;
        let by_source = nes.ppu.batched_invalidations_by_source;
        let pct = if total > 0 {
            100.0 * fallback as f64 / total as f64
        } else {
            0.0
        };
        println!("batched scanlines total      = {}", total);
        println!("batched scanlines fallback   = {} ({:.3}%)", fallback, pct);
        println!("  by-source invalidation counts (scanline-unique):");
        println!("    [0] palette RAM write ($2007 >= $3F00)  = {}", by_source[0]);
        println!(
            "    [1] nametable/pattern $2007 (< $3F00)   = {}  (absorbed by narrowing)",
            by_source[1],
        );
        println!(
            "    [2] PPUCTRL/MASK/SCROLL/ADDR mid-scanl. = {}  (MASK + SCROLL-first still invalidate; CTRL/SCROLL-2nd/ADDR absorbed)",
            by_source[2],
        );
        println!(
            "    [3] mapper write $4018-$FFFF in render  = {}  (absorbed by narrowing)",
            by_source[3],
        );
    }
    #[cfg(not(feature = "ppu_neon_stats"))]
    {
        eprintln!("error: build with --features ppu_neon_stats to populate counters.");
        std::process::exit(2);
    }
}
