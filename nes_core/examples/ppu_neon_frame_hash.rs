//! Frame-hash parity check between BatchedRenderMode::Off and Replace.
//! Runs a ROM for N frames in each mode with identical input state,
//! hashes each frame's pixel buffer, and reports how many frames
//! produced identical vs divergent hashes.
//!
//! Usage:
//!   cargo run --release --features asm_cpu --example \
//!     ppu_neon_frame_hash -- <rom-path> [frames]

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

fn hash_frame(buf: &[u32]) -> u64 {
    // Cheap non-crypto FNV-1a 64-bit. Good enough to spot divergence.
    let mut h: u64 = 0xcbf29ce484222325;
    for &px in buf {
        for b in px.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn run(bytes: &[u8], mode: BatchedRenderMode, frames: usize) -> Vec<u64> {
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes.to_vec()))
        .expect("parse iNES");
    let mut nes = Nes::new(cart);
    nes.ppu.set_batched_render_mode(mode);
    #[cfg(feature = "ppu_neon_stats")]
    {
        let broad = std::env::var("PPU_NEON_STATS_BROAD")
            .ok()
            .map(|s| s == "1")
            .unwrap_or(false);
        nes.ppu.batched_invalidation_broad = broad;
    }

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = NullAudio;
    let mut hashes = Vec::with_capacity(frames);

    for _ in 0..frames {
        let mut v = Xrgb8888VideoSink::new(&mut video_buf);
        while !v.frame_written() {
            nes.step(&mut v, &mut audio);
        }
        hashes.push(hash_frame(&video_buf));
    }
    hashes
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../roms/Contra (USA).nes".into());
    let frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let bytes = std::fs::read(&path).expect("read rom");
    println!("ppu_neon_frame_hash: {} ({} frames)", path, frames);

    let off = run(&bytes, BatchedRenderMode::Off, frames);
    let rep = run(&bytes, BatchedRenderMode::Replace, frames);

    let mut matches = 0usize;
    let mut diffs = Vec::new();
    for (i, (o, r)) in off.iter().zip(rep.iter()).enumerate() {
        if o == r {
            matches += 1;
        } else {
            diffs.push(i);
        }
    }
    println!("matching frames: {}/{}", matches, frames);
    if !diffs.is_empty() {
        println!("diverging frames ({}): first 10 = {:?}",
            diffs.len(),
            &diffs[..diffs.len().min(10)]);
    }
}
