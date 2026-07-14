//! A/B microbench for the MMC1 ($8000-$FFFF) PRG read hot path.
//!
//! Single deterministic env, cycle-locked fs=16 env-steps on an MMC1
//! ROM (Zelda by default). Times the emulate loop per env-step and
//! reports the min + mean, plus an FNV-1a hash of every rendered frame
//! so the two builds can be proven byte-identical end-to-end.
//!
//! Build BOTH sides with the production feature set and compare:
//!   # window path (shipped default)
//!   cargo build --profile release-with-debug --features asm_cpu \
//!       --example bench_mmc1_prg
//!   # bank-math baseline
//!   cargo build --profile release-with-debug \
//!       --features "asm_cpu,mmc1_prg_bankmath_ab" --example bench_mmc1_prg
//!
//! Usage: bench_mmc1_prg [rom] [warmup_steps] [measure_steps] [frame_skip]

use nes_core::cartridge::Cartridge;
use nes_core::input::Button;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use std::time::Instant;

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

/// Deterministic scripted input so both builds execute the identical
/// instruction stream. Boots through the title/file-select screens,
/// then walks Link around (bank-switch-heavy room transitions).
fn apply_input(nes: &mut Nes, step: usize) {
    let pad = nes.game_pad_1();
    pad.set_button_pressed(Button::Start, (2..5).contains(&step) || (8..10).contains(&step));
    pad.set_button_pressed(Button::A, (5..8).contains(&step) || step % 4 == 0);
    pad.set_button_pressed(Button::Right, step >= 10 && (step / 8) % 2 == 0);
    pad.set_button_pressed(Button::Down, step >= 10 && (step / 8) % 2 == 1);
    pad.set_button_pressed(Button::B, step % 6 == 0);
}

fn env_step(nes: &mut Nes, video_buf: &mut [u32], frame_skip: u32) {
    for i in 0..frame_skip {
        let is_last = i + 1 == frame_skip;
        nes.set_skip_render(!is_last);
        let mut video = Xrgb8888VideoSink::new(video_buf);
        let mut audio = NullAudio;
        while !video.frame_written() {
            nes.step(&mut video, &mut audio);
        }
    }
    nes.set_skip_render(false);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rom = args.next().unwrap_or_else(|| "../roms/zelda.nes".into());
    let warmup: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let measure: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);
    let frame_skip: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    let bytes = std::fs::read(&rom).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    let mapper = cart.mapper;
    let prg_len = cart.prg_rom.len();
    let mut nes = Nes::new(cart);
    let mut video_buf = vec![0u32; FRAME_PIXELS];

    // Running FNV-1a over every rendered frame — an end-to-end
    // equivalence signature that must match across window/bank-math.
    let mut frame_hash: u64 = 0xcbf29ce484222325;
    let mut step = 0usize;

    for _ in 0..warmup {
        apply_input(&mut nes, step);
        env_step(&mut nes, &mut video_buf, frame_skip);
        step += 1;
    }

    let mut min_ns = u64::MAX;
    let mut sum_ns: u64 = 0;
    for _ in 0..measure {
        apply_input(&mut nes, step);
        let t0 = Instant::now();
        env_step(&mut nes, &mut video_buf, frame_skip);
        let dt = t0.elapsed().as_nanos() as u64;
        min_ns = min_ns.min(dt);
        sum_ns += dt;
        for &p in &video_buf {
            frame_hash ^= p as u64;
            frame_hash = frame_hash.wrapping_mul(0x100000001b3);
        }
        step += 1;
    }

    let build = if cfg!(feature = "mmc1_prg_bankmath_ab") {
        "BANKMATH"
    } else {
        "WINDOW"
    };
    let min_ms = min_ns as f64 / 1e6;
    let mean_ms = (sum_ns as f64 / measure as f64) / 1e6;
    let min_us_frame = min_ns as f64 / 1e3 / frame_skip as f64;
    println!(
        "[{}] rom={} mapper={} prg={}KB fs={} steps={}",
        build, rom, mapper, prg_len / 1024, frame_skip, measure
    );
    println!(
        "[{}] env-step  min={:.4} ms  mean={:.4} ms  |  per-frame min={:.2} us  |  frame_hash={:016x}",
        build, min_ms, mean_ms, min_us_frame, frame_hash
    );
}
