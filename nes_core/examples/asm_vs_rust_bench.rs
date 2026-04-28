//! Benchmark: run N frames of a ROM end-to-end (CPU+PPU+APU, full
//! rendering to a video buffer) and report wall time.
//!
//! Run twice for comparison:
//!   cargo run --release --example asm_vs_rust_bench           # pure Rust
//!   cargo run --release --features asm_cpu --example asm_vs_rust_bench  # ASM
//!
//! Optional first arg: ROM path (default Mario Bros).
//! Optional second arg: frame count (default 600 = 10s @ 60fps).

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

struct Sink {
    samples: usize,
    peak: f32,
}
impl AudioSink for Sink {
    fn write_sample(&mut self, s: f32) {
        if s.abs() > self.peak { self.peak = s.abs(); }
        self.samples += 1;
    }
    fn samples_written(&self) -> usize { self.samples }
}

fn main() {
    let rom_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../roms/Mario Bros.nes".to_string());
    let target_frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let bytes = std::fs::read(&rom_path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    let mapper = cart.mapper;
    let mut nes = Nes::new(cart);

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = Sink { samples: 0, peak: 0.0 };

    #[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
    let asm_before = nes_core::cpu_asm::ASM_HITS
        .load(std::sync::atomic::Ordering::Relaxed);

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

    // Simple hash of final frame buffer so we can verify byte-exact
    // rendering between builds (asm_cpu vs pure Rust) without needing
    // two Python wheels loaded at once.
    let hash = {
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a 64
        for p in &video_buf {
            h ^= *p as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    let nonzero = video_buf.iter().filter(|&&p| p != 0).count();
    let fps = frames_done as f64 / dur.as_secs_f64();
    let speed_vs_60hz = fps / 60.0;

    let build_label = {
        #[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
        { "ASM (AArch64 + asm_cpu)" }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm_cpu")))]
        { "pure Rust interpreter" }
    };

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" NES core benchmark — {}", build_label);
    println!("═══════════════════════════════════════════════════════════════════");
    println!("ROM:                  {} (mapper {})", rom_path, mapper);
    println!("Frames rendered:      {}", frames_done);
    println!("Wall time:            {:.3} s", dur.as_secs_f64());
    println!("FPS:                  {:.1}", fps);
    println!("Real-time multiplier: {:.2}× (vs 60 Hz)", speed_vs_60hz);
    println!("CPU cycles:           {}", nes.cycles);
    println!("Audio samples:        {} (peak {:.3})", audio.samples_written(), audio.peak);
    println!("Nonzero pixels (last frame): {} / {}", nonzero, FRAME_PIXELS);
    println!("Final frame FNV-1a:   {:016x}", hash);

    #[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
    {
        let hits = nes_core::cpu_asm::ASM_HITS
            .load(std::sync::atomic::Ordering::Relaxed) - asm_before;
        println!("ASM instructions executed: {}", hits);
    }

    println!("═══════════════════════════════════════════════════════════════════");
}
