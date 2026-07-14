//! APU muted-worker channel-timer skip: CPU-observable byte-exactness.
//!
//! `Apu::step_timer` / `Apu::step_frame_counter` skip the pulse /
//! triangle / noise timers + envelope / sweep / linear-counter clocking
//! when `sample_output_enabled == false` (the 15 non-audio training
//! workers). DMC, the frame counter, the length counters, and the
//! $4015 / IRQ semantics stay fully live because they are
//! CPU-observable (DMC steals CPU cycles; length counters + the frame
//! IRQ flag surface through $4015 and the IRQ line).
//!
//! The skipped state (duty phase, timer values, envelope volume, sweep
//! period, linear counter) feeds ONLY `Apu::output()` /
//! `generate_sample()`, which is already gated off while audio is
//! muted. So a muted run must be *bit-identical, in every CPU-observable
//! respect*, to a full-tick run of the same ROM + inputs.
//!
//! This test proves exactly that by lockstep-running two instances of
//! the same ROM — one audio-ON (full timer path), one audio-OFF
//! (muted-skip path) — under an identical deterministic input schedule,
//! comparing at every instruction boundary:
//!   * all six CPU registers + the running cycle count,
//!   * the `$4015` status byte (length counters + DMC length + frame
//!     IRQ + DMC IRQ) via a side-effect-free peek,
//!   * the CPU IRQ line (`apu.irq_pending() || mapper.irq_pending()`),
//! plus a full 2 KB RAM compare at every frame boundary.
//!
//! ROMs are chosen so the ONLY per-cycle difference between the two
//! instances is the skip under test: all three use mappers whose
//! `tick_audio()` is a no-op (UxROM / MMC3 / MMC1 have no expansion
//! audio), so audio-ON's `mapper.tick_audio()` and `generate_sample()`
//! are both CPU-observable-neutral and the comparison isolates the
//! channel-timer skip. Coverage:
//!   * Contra (UxROM)  — DMC-active (gunfire), APU frame IRQ.
//!   * SMB 3 (MMC3)    — DMC + APU frame IRQ + heavy mapper scanline IRQ.
//!   * Zelda (MMC1)    — APU frame IRQ, battery SRAM.
//!
//! ROMs are git-ignored; the test skips gracefully when absent (it runs
//! for real on the main tree, where `roms/` is populated).

use nes_core::cartridge::Cartridge;
use nes_core::input::Button;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const CONTRA: &str = "../roms/Contra (USA).nes";
const SMB3: &str = "../roms/Super Mario Bros. 3 (USA) (Rev A).nes";
const ZELDA: &str = "../roms/zelda.nes";

const CYCLES_PER_FRAME: usize = 29781;

struct V;
impl VideoSink for V {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool {
        false
    }
    fn pixel_size(&self) -> usize {
        4
    }
}
struct A;
impl AudioSink for A {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

fn load_rom(path: &str) -> Option<Nes> {
    let f = File::open(path).ok()?;
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    Some(Nes::new(cart))
}

/// Same deterministic schedule shape as `asm_bulk_override.rs`: Start
/// pulses through the title/menu for the first ~900 frames, then
/// pseudo-random d-pad + A/B gameplay input. Drives both instances out
/// of attract mode into gameplay (scrolling, DMC gunfire, sprite-0
/// splits, MMC3 scanline IRQ, APU frame IRQ).
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

fn step_one_instruction(nes: &mut Nes, v: &mut V, a: &mut A) {
    nes.step(v, a);
    while !nes.cpu.at_instruction_boundary() {
        nes.step(v, a);
    }
}

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Frame count for the lockstep window; env-tunable for longer
/// main-tree gate runs (`APU_MUTED_SKIP_FRAMES=5000`).
fn frame_budget() -> usize {
    std::env::var("APU_MUTED_SKIP_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1200)
}

/// Lockstep an audio-ON (full timer path) instance against an audio-OFF
/// (muted-skip path) instance of the same ROM. Any CPU-observable
/// divergence — registers, cycles, $4015, IRQ line, or RAM — fails the
/// test with the exact frame + first mismatching RAM byte.
fn assert_muted_skip_matches_full(rom: &str, label: &str) {
    let (mut full, mut muted) = match (load_rom(rom), load_rom(rom)) {
        (Some(f), Some(m)) => (f, m),
        _ => {
            eprintln!("skipping {label}: {rom} not present");
            return;
        }
    };
    full.reset();
    muted.reset();
    // full = the reference full-tick path (audio ON, every channel
    // timer + envelope/sweep/linear steps). muted = the path under
    // test (audio OFF → channel timers + env/sweep/linear skipped).
    full.set_audio_output_enabled(true);
    muted.set_audio_output_enabled(false);

    let mut v_f = V;
    let mut a_f = A;
    let mut v_m = V;
    let mut a_m = A;

    let frames = frame_budget();
    let mut last_frame = usize::MAX;
    let mut checks: u64 = 0;

    loop {
        let frame = full.cycles / CYCLES_PER_FRAME;
        if frame >= frames {
            break;
        }
        if frame != last_frame {
            last_frame = frame;
            let mask = schedule_mask(frame);
            apply_mask(&mut full, mask);
            apply_mask(&mut muted, mask);

            // Full-RAM compare once per frame (belt-and-suspenders on
            // top of the per-instruction register check below).
            let hf = fnv1a((0..0x800u16).map(|addr| full.system_ram_byte(addr)));
            let hm = fnv1a((0..0x800u16).map(|addr| muted.system_ram_byte(addr)));
            if hf != hm {
                let bad = (0..0x800u16)
                    .find(|&addr| full.system_ram_byte(addr) != muted.system_ram_byte(addr));
                panic!(
                    "{label}: RAM divergence at frame {frame}: first mismatch {:?}",
                    bad.map(|addr| format!(
                        "${addr:04X}: full={:02X} muted={:02X}",
                        full.system_ram_byte(addr),
                        muted.system_ram_byte(addr),
                    )),
                );
            }
        }

        step_one_instruction(&mut full, &mut v_f, &mut a_f);
        step_one_instruction(&mut muted, &mut v_m, &mut a_m);

        let rf = full.cpu.regs();
        let rm = muted.cpu.regs();
        // Side-effect-free $4015 peek (does NOT clear the frame IRQ
        // flag the way a real bus read would) so probing it can't
        // itself perturb either instance.
        let status_f = full.apu.peek_byte(0x4015);
        let status_m = muted.apu.peek_byte(0x4015);
        let irq_f = full.cpu.irq_line_low;
        let irq_m = muted.cpu.irq_line_low;

        if rf.pc != rm.pc
            || rf.a != rm.a
            || rf.x != rm.x
            || rf.y != rm.y
            || rf.sp != rm.sp
            || full.cycles != muted.cycles
            || status_f != status_m
            || irq_f != irq_m
        {
            panic!(
                "{label}: CPU-observable divergence at frame {frame} (check #{checks}):\n\
                 full : PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={} $4015={:02X} irq={}\n\
                 muted: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={} $4015={:02X} irq={}",
                rf.pc, rf.a, rf.x, rf.y, rf.sp, full.cycles, status_f, irq_f,
                rm.pc, rm.a, rm.x, rm.y, rm.sp, muted.cycles, status_m, irq_m,
            );
        }
        checks += 1;
    }

    // Guard against a silent no-op run (ROM present but never advanced).
    assert!(
        checks > 10_000,
        "{label}: only {checks} instruction checks — ROM did not run"
    );
    println!(
        "{label}: muted-skip == full-tick across {} frames, {checks} instruction-boundary checks",
        frames
    );
}

#[test]
fn contra_dmc_muted_skip_is_cpu_observable_identical() {
    assert_muted_skip_matches_full(CONTRA, "Contra (UxROM, DMC-active)");
}

#[test]
fn smb3_mmc3_irq_muted_skip_is_cpu_observable_identical() {
    assert_muted_skip_matches_full(SMB3, "SMB3 (MMC3, mapper+frame IRQ)");
}

#[test]
fn zelda_mmc1_muted_skip_is_cpu_observable_identical() {
    assert_muted_skip_matches_full(ZELDA, "Zelda (MMC1, frame IRQ)");
}
