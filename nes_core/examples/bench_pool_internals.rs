//! Pure-Rust bench that breaks down `Pool::step_all` into phases.
//! Bypasses PyO3 and Python GIL paths so we see exactly where the
//! emulator hot loop spends its time.
//!
//! Usage: cargo run --release --features asm_cpu --example
//!        bench_pool_internals -- [rom] [workers] [frame_skip] [steps]

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;
const RAM_SIZE: usize = 2048;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

struct Worker {
    nes: Nes,
    video_buf: Vec<u32>,
}

impl Worker {
    fn new(cart: Cartridge) -> Self {
        Self {
            nes: Nes::new(cart),
            video_buf: vec![0u32; FRAME_PIXELS],
        }
    }

    fn step(&mut self, frame_skip: u32) {
        for i in 0..frame_skip {
            let is_last = i + 1 == frame_skip;
            self.nes.set_skip_render(!is_last);
            let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);
            let mut audio = NullAudio;
            while !video.frame_written() {
                self.nes.step(&mut video, &mut audio);
            }
        }
        self.nes.set_skip_render(false);
    }
}

fn xrgb_to_rgb(src: &[u32], dst: &mut [u8]) {
    for (i, &px) in src.iter().enumerate() {
        let r = ((px >> 16) & 0xFF) as u8;
        let g = ((px >> 8) & 0xFF) as u8;
        let b = (px & 0xFF) as u8;
        dst[i * 3] = r;
        dst[i * 3 + 1] = g;
        dst[i * 3 + 2] = b;
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rom = args.next().unwrap_or_else(|| "../roms/zelda.nes".into());
    let workers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let frame_skip: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(500);

    let bytes = std::fs::read(&rom).expect("read rom");
    let pool: Vec<Mutex<Worker>> = (0..workers)
        .map(|_| {
            let cart = Cartridge::load(&mut std::io::Cursor::new(bytes.clone()))
                .expect("parse iNES");
            Mutex::new(Worker::new(cart))
        })
        .collect();

    // Warmup so bench numbers reflect steady-state.
    for _ in 0..30 {
        pool.par_iter().for_each(|mu| {
            let mut w = mu.lock().unwrap();
            w.step(frame_skip);
        });
    }

    let mut t_emu_ns: u64 = 0;
    let mut t_rgb_ns: u64 = 0;
    let mut t_pp_ns: u64 = 0;
    let mut t_ram_ns: u64 = 0;
    let mut t_total_ns: u64 = 0;

    let bench_start = Instant::now();
    for _ in 0..steps {
        let step_start = Instant::now();
        let per_thread: Vec<(u64, u64, u64, u64)> = pool
            .par_iter()
            .map(|mu| {
                let mut w = mu.lock().unwrap();
                let t0 = Instant::now();
                w.step(frame_skip);
                let t1 = Instant::now();

                let mut rgb = vec![0u8; FRAME_PIXELS * 3];
                xrgb_to_rgb(&w.video_buf, &mut rgb);
                let t2 = Instant::now();

                let _pp = nes_core::preprocess::preprocess_frame(&rgb);
                let t3 = Instant::now();

                let _ram: Vec<u8> = w.nes.system_ram().as_ref().to_vec();
                let t4 = Instant::now();

                (
                    (t1 - t0).as_nanos() as u64,
                    (t2 - t1).as_nanos() as u64,
                    (t3 - t2).as_nanos() as u64,
                    (t4 - t3).as_nanos() as u64,
                )
            })
            .collect();
        t_total_ns += step_start.elapsed().as_nanos() as u64;
        for (e, r, p, m) in per_thread {
            t_emu_ns += e;
            t_rgb_ns += r;
            t_pp_ns += p;
            t_ram_ns += m;
        }
    }
    let bench_elapsed = bench_start.elapsed().as_secs_f64();

    // Per-phase totals are SUMS across all worker threads; divide by
    // workers count to get "effective ms per step" on the critical
    // path (rayon parallelizes perfectly at 16 cores).
    let steps_f = steps as f64;
    let emu_ms = t_emu_ns as f64 / 1e6 / workers as f64;
    let rgb_ms = t_rgb_ns as f64 / 1e6 / workers as f64;
    let pp_ms = t_pp_ns as f64 / 1e6 / workers as f64;
    let ram_ms = t_ram_ns as f64 / 1e6 / workers as f64;
    let total_ms = t_total_ns as f64 / 1e6;

    let sps = (steps_f * workers as f64) / bench_elapsed;

    println!("\n== pool internals bench ==");
    println!("  rom:        {}", rom);
    println!("  workers:    {}", workers);
    println!("  frame_skip: {}", frame_skip);
    println!("  steps:      {}", steps);
    println!();
    println!("critical-path ms per step_all (parallel wall equivalent):");
    println!("  emulate (nes.step×fs)       {:8.3}  {:5.1}%", emu_ms / steps_f, 100.0 * emu_ms / total_ms);
    println!("  xrgb→rgb                    {:8.3}  {:5.1}%", rgb_ms / steps_f, 100.0 * rgb_ms / total_ms);
    println!("  preprocess_frame            {:8.3}  {:5.1}%", pp_ms / steps_f, 100.0 * pp_ms / total_ms);
    println!("  ram copy                    {:8.3}  {:5.1}%", ram_ms / steps_f, 100.0 * ram_ms / total_ms);
    println!("  TOTAL step_all wall         {:8.3}  100.0%", total_ms / steps_f);
    println!();
    println!("throughput: {:.0} sps ({:.1}× realtime across {} envs)",
        sps, sps / 60.0, workers);
}
