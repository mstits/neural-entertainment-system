//! PPU-state cost profiler. Boots zelda, applies the start state used
//! by the baseline, then replicates python.rs::advance_one_frame's
//! cycle-locked stepping at several frame_skip values so we can
//! decompose CPU+APU vs PPU-state vs PPU-pixel exactly like
//! scripts/bench_emulator_phases.py — but from a plain Rust binary that
//! `sample` can attach to without a Python interpreter in the way.
//!
//! Env vars:
//!   PROF_MODE = phases | spin   (default phases)
//!   PROF_FRAMES = frames per timed config (default 4000)
//!   PROF_FS = frame_skip for spin mode (default 64)

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;
const CPU_CYCLES_PER_FRAME: usize = 29781;

struct Null;
impl AudioSink for Null {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn load_state(nes: &mut Nes, path: &str) {
    let raw = std::fs::read(path).expect("state file");
    // Strip repeated NCST\x01 prefixes (mirror python.rs loader).
    let magic = b"NCST\x01";
    let mut body = &raw[..];
    while body.len() >= magic.len() && &body[..magic.len()] == magic {
        body = &body[magic.len()..];
    }
    let state: nes_core::nes::State =
        bincode::deserialize(body).expect("bincode state");
    nes.apply_state(&state);
}

/// One frame of the cycle-locked advance, mirroring
/// python.rs::advance_one_frame (bulk-step to a cycle target, then tick
/// the remainder). `skip` toggles the PPU pixel-write fast path.
#[inline]
fn advance_one_frame(
    nes: &mut Nes,
    video_buf: &mut [u32],
    audio: &mut Null,
    target: &mut usize,
    skip: bool,
) {
    nes.set_skip_render(skip);
    let mut video = Xrgb8888VideoSink::new(video_buf);
    *target += CPU_CYCLES_PER_FRAME;
    let margin = nes.asm_bulk_cycles_margin();
    while nes.cycles + margin < *target {
        if nes.oam_dma.active {
            nes.tick(&mut video, audio);
        } else {
            nes.step(&mut video, audio);
        }
    }
    while nes.cycles < *target {
        nes.tick(&mut video, audio);
    }
}

fn run_config(
    nes: &mut Nes,
    video_buf: &mut [u32],
    audio: &mut Null,
    target: &mut usize,
    frame_skip: usize,
    steps: usize,
) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        for i in 0..frame_skip {
            let is_last = i + 1 == frame_skip;
            advance_one_frame(nes, video_buf, audio, target, !is_last);
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let rom = std::env::var("PROF_ROM")
        .unwrap_or_else(|_| "roms/zelda.nes".into());
    let state = std::env::var("PROF_STATE")
        .unwrap_or_else(|_| "roms/zelda_start_ctrl.state.bin".into());
    let mode = std::env::var("PROF_MODE").unwrap_or_else(|_| "phases".into());
    let steps: usize = std::env::var("PROF_FRAMES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(4000);

    let bytes = std::fs::read(&rom).expect("rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("ines");
    let mut nes = Nes::new(cart);
    load_state(&mut nes, &state);

    // PROF_REFINED=0 forces the refined skip-render paths off (full
    // per-cycle tick body), giving a same-binary A/B against the default
    // (on). Any other value / unset leaves the production default on.
    if std::env::var("PROF_REFINED").as_deref() == Ok("0") {
        nes.ppu.set_refined_skip_render(false);
        eprintln!("[refined skip-render OFF]");
    }

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = Null;
    let mut target = nes.cycles;

    // Warm-up: settle mapper/APU/PPU, get into steady rendering state.
    for _ in 0..30 {
        advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, false);
    }

    if mode == "hash" {
        // Render one full frame and print an FNV-1a hash of the pixel
        // buffer, to prove byte-exactness across builds.
        advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, false);
        let mut h: u64 = 0xcbf29ce484222325;
        for p in &video_buf {
            h ^= *p as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // Run 300 more frames rendering every one, accumulate hash, to
        // exercise many scanline configs / sprite-0 positions.
        for _ in 0..300 {
            advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, false);
            for p in &video_buf {
                h ^= *p as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("frame-hash {h:016x}");
        return;
    }

    if mode == "hashfs" {
        // Run fs=16 (skip-render on the first 15 of each 16) and hash
        // ONLY the rendered frames. Proves that dropping unobserved
        // work on skip frames doesn't corrupt the rendered frame that
        // the agent actually sees.
        let fs: usize = std::env::var("PROF_FS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let mut h: u64 = 0xcbf29ce484222325;
        for _ in 0..400 {
            for i in 0..fs {
                let is_last = i + 1 == fs;
                advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, !is_last);
            }
            for p in &video_buf {
                h ^= *p as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("hashfs(fs={fs}) {h:016x}");
        return;
    }

    if mode == "spin" {
        let fs: usize = std::env::var("PROF_FS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        eprintln!("[spin] fs={fs}; sample me now (Ctrl-C to stop)");
        let mut n = 0u64;
        loop {
            for i in 0..fs {
                let is_last = i + 1 == fs;
                advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, !is_last);
            }
            n += 1;
            if n % 2000 == 0 { eprintln!("[spin] {} steps", n); }
        }
    }

    // phases mode: three configs, 3 repeats each, report spread.
    let configs = [1usize, 16, 64];
    println!("== ppu_state_profile phases ==  rom={rom} steps={steps}/config");
    let mut t = [[0f64; 3]; 3];
    for (ci, &fs) in configs.iter().enumerate() {
        let cfg_steps = steps / fs.max(1);
        for r in 0..3 {
            t[ci][r] = run_config(&mut nes, &mut video_buf, &mut audio, &mut target, fs, cfg_steps)
                / cfg_steps as f64; // ms/step
        }
    }
    let med = |a: [f64; 3]| { let mut v = a; v.sort_by(|x, y| x.partial_cmp(y).unwrap()); v[1] };
    let t1 = med(t[0]);
    let t16 = med(t[1]);
    let t64 = med(t[2]);
    // Per-frame decomposition (same algebra as bench_emulator_phases).
    let a_us = ((t16 - t1) / 15.0) * 1000.0; // CPU+APU+PPU(skip)
    let b_us = (t1 - (t16 - t1) / 15.0) * 1000.0; // PPU pixel work
    println!("fs=1  ms/step: {:.3} {:.3} {:.3}", t[0][0], t[0][1], t[0][2]);
    println!("fs=16 ms/step: {:.3} {:.3} {:.3}", t[1][0], t[1][1], t[1][2]);
    println!("fs=64 ms/step: {:.3} {:.3} {:.3}", t[2][0], t[2][1], t[2][2]);
    println!("median: fs1={t1:.4} fs16={t16:.4} fs64={t64:.4}");
    println!("CPU+APU+PPU(skip): {a_us:.1} us/frame");
    println!("PPU pixel work:    {b_us:.1} us/frame");
    let pred64 = 63.0 * (a_us / 1000.0) + (b_us / 1000.0);
    println!("cross-check fs=64 predicted {pred64:.3} vs measured {t64:.3} ms/step");
}
