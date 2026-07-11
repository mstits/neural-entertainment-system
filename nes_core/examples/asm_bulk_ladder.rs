//! Throughput rung for the ASM bulk-budget ladder (MMC1 / UxROM).
//!
//! Replays the Pool's cycle-locked `advance_one_frame` loop (29781
//! CPU cycles per frame, headless: skip-render on, audio off) at a
//! given `asm_bulk_cycles` budget and reports frames/sec. Run the
//! ladder on a cool machine:
//!
//!   cargo run --release --features asm_cpu --example asm_bulk_ladder -- ../roms/zelda.nes 1
//!   cargo run --release --features asm_cpu --example asm_bulk_ladder -- ../roms/zelda.nes 8
//!   cargo run --release --features asm_cpu --example asm_bulk_ladder -- ../roms/zelda.nes 16
//!
//! Throughput is only half the rung — the budget must ALSO pass the
//! lockstep tests (`tests/asm_bulk_override.rs -- --ignored`), the
//! Mesen oracle, and `make parity` before a profile enables it.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

const CPU_CYCLES_PER_FRAME: usize = 29781;

struct V;
impl VideoSink for V {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
struct A;
impl AudioSink for A {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn advance_one_frame(nes: &mut Nes, target: &mut usize) {
    let mut v = V;
    let mut a = A;
    *target += CPU_CYCLES_PER_FRAME;
    let margin = nes.asm_bulk_cycles_margin();
    while nes.cycles + margin < *target {
        if nes.oam_dma.active {
            nes.tick(&mut v, &mut a);
        } else {
            nes.step(&mut v, &mut a);
        }
    }
    while nes.cycles < *target {
        nes.tick(&mut v, &mut a);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rom = args.next().unwrap_or_else(|| "../roms/zelda.nes".to_string());
    let budget: i64 = args.next().map(|s| s.parse().expect("budget")).unwrap_or(1);
    let frames: usize = args.next().map(|s| s.parse().expect("frames")).unwrap_or(3000);

    let bytes = std::fs::read(&rom).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    println!("{rom}: mapper={} budget={budget} frames={frames}", cart.mapper);

    let mut nes = Nes::new(cart);
    nes.reset();
    nes.set_asm_bulk_cycles_override(budget);
    nes.set_audio_output_enabled(false);
    nes.set_skip_render(true);

    let mut target = nes.cycles;
    // Warmup past cold boot / attract-mode setup.
    for _ in 0..300 {
        advance_one_frame(&mut nes, &mut target);
    }
    let start = std::time::Instant::now();
    for _ in 0..frames {
        advance_one_frame(&mut nes, &mut target);
    }
    let dur = start.elapsed();
    let fps = frames as f64 / dur.as_secs_f64();
    println!(
        "budget={budget}: {frames} frames in {:.3}s -> {:.0} fps ({:.1}x realtime, {:.1} us/frame)",
        dur.as_secs_f64(),
        fps,
        fps / 60.0,
        dur.as_micros() as f64 / frames as f64,
    );
}
