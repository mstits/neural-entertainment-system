// Diagnostic: pure-Rust rayon scaling test. No PyO3, no GIL, no
// pybindings. Isolates whether the terrible parallel efficiency the
// Python bench sees is:
//   a) Python / PyO3 overhead (collect_results, PyList/PyBytes)
//   b) Memory allocator contention
//   c) Something actually inside Nes::step
//
// Usage:
//   cargo run --release --features asm_cpu --example rayon_scaling -- \
//       path/to/rom.nes [frame_skip] [steps]
use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

const FRAME_PIXELS: usize = 256 * 240;

struct NopAudio {
    n: usize,
}
impl AudioSink for NopAudio {
    fn write_sample(&mut self, _s: f32) {
        self.n += 1;
    }
    fn samples_written(&self) -> usize {
        self.n
    }
}

struct Worker {
    nes: Nes,
    video_buf: Vec<u32>,
    audio: NopAudio,
}

impl Worker {
    fn new(cart: Cartridge) -> Self {
        let mut nes = Nes::new(cart);
        // Match training-pool defaults: drop audio sample gen for
        // non-observed workers.
        nes.set_audio_output_enabled(false);
        Self {
            nes,
            video_buf: vec![0u32; FRAME_PIXELS],
            audio: NopAudio { n: 0 },
        }
    }

    fn advance_one_frame(&mut self) {
        let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);
        while !video.frame_written() {
            self.nes.step(&mut video, &mut self.audio);
        }
    }

    fn step(&mut self, frame_skip: u32) {
        for i in 0..frame_skip {
            let is_last = i + 1 == frame_skip;
            self.nes.set_skip_render(!is_last);
            self.advance_one_frame();
        }
        self.nes.set_skip_render(false);
    }
}

fn bench(rom_bytes: &[u8], workers: usize, frame_skip: u32, steps: usize) -> f64 {
    let mut ws = Vec::with_capacity(workers);
    for _ in 0..workers {
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom_bytes)).unwrap();
        ws.push(Mutex::new(Worker::new(cart)));
    }
    // Reset + warm up.
    for mu in &ws {
        let mut w = mu.lock().unwrap();
        w.nes.reset();
        w.advance_one_frame();
    }
    for _ in 0..2 {
        ws.par_iter().for_each(|mu| {
            mu.lock().unwrap().step(frame_skip);
        });
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        ws.par_iter().for_each(|mu| {
            mu.lock().unwrap().step(frame_skip);
        });
    }
    t0.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} ROM [frame_skip=16] [steps=100]",
            args.first().map(String::as_str).unwrap_or("rayon_scaling")
        );
        std::process::exit(1);
    }
    let rom = &args[1];
    let frame_skip: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let rom_bytes = std::fs::read(rom).expect("read rom");

    println!(
        "== pure-Rust rayon-scaling bench ==\n  rom:        {rom}\n  steps:      {steps} per config\n  frame_skip: {frame_skip}\n  num_cpus:   {}\n",
        num_cpus::get()
    );
    println!(
        "{:>8}{:>10}{:>14}{:>18}{:>14}",
        "workers", "ms", "total sps", "per-worker sps", "efficiency"
    );
    println!("{}", "-".repeat(64));

    let counts = [1usize, 2, 4, 8, 12, 16, 20, 24];
    let mut baseline: Option<f64> = None;
    for &n in &counts {
        let ms = bench(&rom_bytes, n, frame_skip, steps);
        let total_sps = (n * steps) as f64 / (ms / 1000.0);
        let per_worker = total_sps / n as f64;
        if baseline.is_none() && n == 1 {
            baseline = Some(per_worker);
        }
        let eff = baseline.map(|b| per_worker / b * 100.0).unwrap_or(100.0);
        println!(
            "{:>8}{:>10.0}{:>14.0}{:>18.1}{:>13.1}%",
            n, ms, total_sps, per_worker, eff
        );
    }
}
