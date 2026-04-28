//! Throughput bench: BatchedRenderMode::Off vs Replace on a real ROM.
//! Measures the skip-per-pixel perf win end-to-end.
//!
//! Usage: cargo run --release --features asm_cpu --example
//!        bench_batched_ppu -- <rom-path> [frames]

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{self, BatchedRenderMode};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn run(rom_bytes: &[u8], mode: BatchedRenderMode, frames: usize) -> f64 {
    let cart = Cartridge::load(&mut std::io::Cursor::new(rom_bytes.to_vec()))
        .expect("parse iNES");
    let mut nes = Nes::new(cart);
    nes.ppu.set_batched_render_mode(mode);

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = NullAudio;

    // Warmup so the prediction history stabilizes (Replace needs a
    // few clean frames to flip the heuristic on).
    for _ in 0..30 {
        let mut v = Xrgb8888VideoSink::new(&mut video_buf);
        while !v.frame_written() {
            nes.step(&mut v, &mut audio);
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..frames {
        let mut v = Xrgb8888VideoSink::new(&mut video_buf);
        while !v.frame_written() {
            nes.step(&mut v, &mut audio);
        }
    }
    let dur = start.elapsed().as_secs_f64();
    frames as f64 / dur
}

fn main() {
    let rom_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../roms/zelda.nes".into());
    let frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let rom = std::fs::read(&rom_path).expect("read rom");

    println!("== batched_ppu bench: {} ({} frames each) ==", rom_path, frames);
    let off_fps = run(&rom, BatchedRenderMode::Off, frames);
    let rep_fps = run(&rom, BatchedRenderMode::Replace, frames);

    println!("  Off:     {:7.1} fps", off_fps);
    println!("  Replace: {:7.1} fps  ({:+.1}%)",
        rep_fps, 100.0 * (rep_fps - off_fps) / off_fps);
}
