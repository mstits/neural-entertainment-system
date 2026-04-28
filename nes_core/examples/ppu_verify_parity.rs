//! Runs a ROM in BatchedRenderMode::Verify and reports per-scanline
//! divergence between the per-pixel reference path and the batched
//! ppu_neon kernel. Exit code 0 = byte-exact, nonzero = divergence.
//!
//! Usage: cargo run --release --features asm_cpu --example
//!        ppu_verify_parity -- <rom-path> [frames]

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
        .unwrap_or_else(|| "../roms/zelda.nes".to_string());
    let target_frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let bytes = std::fs::read(&path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    println!(
        "Verify PPU parity: {} (mapper={}, {} frames)",
        path, cart.mapper, target_frames
    );

    let mut nes = Nes::new(cart);
    let mode = std::env::var("PPU_NEON_MODE").unwrap_or_else(|_| "verify".into());
    nes.ppu.set_batched_render_mode(match mode.as_str() {
        "replace" => BatchedRenderMode::Replace,
        _ => BatchedRenderMode::Verify,
    });
    #[cfg(feature = "ppu_neon_stats")]
    {
        let broad = std::env::var("PPU_NEON_STATS_BROAD")
            .ok()
            .map(|s| s == "1")
            .unwrap_or(false);
        nes.ppu.batched_invalidation_broad = broad;
        if broad {
            println!("  (BROAD invalidation ruleset enabled)");
        }
    }

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = NullAudio;
    let mut frames_done = 0usize;

    let start = std::time::Instant::now();
    while frames_done < target_frames {
        let mut video = Xrgb8888VideoSink::new(&mut video_buf);
        while !video.frame_written() {
            nes.step(&mut video, &mut audio);
        }
        frames_done += 1;
    }
    let dur = start.elapsed();

    let verified = nes.ppu.scanlines_verified;
    let diverged = nes.ppu.scanlines_diverged;
    println!(
        "{} frames in {:.2}s | scanlines_verified={} diverged={} ({:.4}%)",
        frames_done,
        dur.as_secs_f64(),
        verified,
        diverged,
        if verified > 0 {
            100.0 * diverged as f64 / verified as f64
        } else {
            0.0
        },
    );

    if diverged > 0 {
        eprintln!("DIVERGENCE DETECTED");
        std::process::exit(2);
    } else {
        println!("PARITY CLEAN");
    }
}
