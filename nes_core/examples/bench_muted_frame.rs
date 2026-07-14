//! Single-env fs=16 throughput A/B for the muted-worker perf wins
//! (APU channel-timer skip + CPU bulk=1 fixed-overhead trim).
//!
//! Mimics one training worker: `skip_render = true`, audio optionally
//! muted. A "pool step" advances 16 NES frames (fs=16) worth of CPU
//! cycles, applying a deterministic gameplay input schedule per frame,
//! exactly the hot path (`Nes::step` → ASM CPU dispatch → per-cycle
//! `Apu::tick` / `Ppu::tick`) that both wins touch.
//!
//! Usage:
//!   bench_muted_frame <rom> [audio=0|1] [warm_frames] [timed_steps] [reps]
//! Reports per-config min / mean / spread in us/pool-step and us/frame.
//! Build: cargo build --release --features asm_cpu --example bench_muted_frame

use std::time::Instant;

use nes_core::cartridge::Cartridge;
use nes_core::input::Button;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

const CYCLES_PER_FRAME: usize = 29781;
const FS: usize = 16;

struct NullV;
impl VideoSink for NullV {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool {
        false
    }
    fn pixel_size(&self) -> usize {
        4
    }
}
struct NullA;
impl AudioSink for NullA {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

fn schedule_mask(frame: usize) -> u8 {
    if frame < 900 {
        if frame % 32 < 2 {
            0x08
        } else {
            0x00
        }
    } else {
        let mut x = (frame / 8) as u32 ^ 0x9E37_79B9;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x & 0xF3) as u8
    }
}

fn apply_mask(nes: &mut Nes, mask: u8) {
    let pad = nes.game_pad_1();
    pad.set_button_pressed(Button::A, mask & 0x01 != 0);
    pad.set_button_pressed(Button::B, mask & 0x02 != 0);
    pad.set_button_pressed(Button::Select, mask & 0x04 != 0);
    pad.set_button_pressed(Button::Start, mask & 0x08 != 0);
    pad.set_button_pressed(Button::Up, mask & 0x10 != 0);
    pad.set_button_pressed(Button::Down, mask & 0x20 != 0);
    pad.set_button_pressed(Button::Left, mask & 0x40 != 0);
    pad.set_button_pressed(Button::Right, mask & 0x80 != 0);
}

/// Advance one NES frame worth of CPU cycles (cycle-locked to
/// CYCLES_PER_FRAME), applying `mask` for the frame.
fn advance_frame(nes: &mut Nes, frame: usize, v: &mut NullV, a: &mut NullA) {
    apply_mask(nes, schedule_mask(frame));
    let target = nes.cycles + CYCLES_PER_FRAME;
    while nes.cycles < target {
        nes.step(v, a);
    }
}

fn main() {
    let rom = std::env::args().nth(1).expect("rom path required");
    let audio_on = std::env::args()
        .nth(2)
        .map(|s| s == "1")
        .unwrap_or(false);
    let warm: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let timed_steps: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let reps: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let bytes = std::fs::read(&rom).expect("rom read");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("ines");
    let mut nes = Nes::new(cart);
    nes.reset();
    nes.set_skip_render(true);
    nes.set_audio_output_enabled(audio_on);

    let mut v = NullV;
    let mut a = NullA;

    // Warm into gameplay so DMC / APU frame IRQ / scrolling are active.
    let mut frame = 0usize;
    for _ in 0..warm {
        advance_frame(&mut nes, frame, &mut v, &mut a);
        frame += 1;
    }

    // Timed reps: each rep times `timed_steps` pool-steps (16 frames
    // each). Report min / mean / spread of per-pool-step wall time.
    let mut rep_means_us: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..timed_steps {
            for _ in 0..FS {
                advance_frame(&mut nes, frame, &mut v, &mut a);
                frame += 1;
            }
        }
        let elapsed = t0.elapsed().as_secs_f64() * 1e6; // us
        rep_means_us.push(elapsed / timed_steps as f64);
    }

    let min = rep_means_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rep_means_us
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = rep_means_us.iter().sum::<f64>() / reps as f64;
    let audio_label = if audio_on { "audio-ON " } else { "audio-OFF" };
    println!(
        "{rom} [{audio_label}] fs={FS}: min={:.1} mean={:.1} spread={:.1} us/pool-step | \
         {:.2} us/frame (min) | reps={reps} timed_steps={timed_steps} warm={warm}",
        min,
        mean,
        max - min,
        min / FS as f64,
    );
}
