//! Shadow oracle for the event-driven-PPU campaign (Rung 1 / Level A).
//!
//! Runs two full `Nes` instances from an identical ROM + start-state
//! under an identical input script, differing ONLY in the scanline-
//! granular `advance` gate — one with `set_ppu_scanline_advance(true)`
//! (the fast path under test), one with it forced off (the pure per-dot
//! reference). Both hold `skip_render` on every frame so `advance` is
//! continuously engaged. After every step the two are compared at each
//! PPU scanline boundary over long runs; any divergence is a hard fail
//! reported with the (frame, scanline, dot) triple.
//!
//! What is compared: the full CPU-observable machine state — CPU RAM +
//! registers, mapper (banks / IRQ counters / CHR latch), the PPU
//! register file (v/t/x/w, PPUCTRL/MASK/STATUS, OAMADDR, buffers), OAM
//! (primary + secondary + evaluation scratch), palette RAM, the
//! frame/scanline/cycle counters, and the sprite-0 / vblank / NMI flags.
//! VRAM + full RAM are additionally compared at every frame boundary.
//! Deliberately EXCLUDED: the render pipeline (BG/sprite shift + fetch
//! latches) that `skip_render` discards and `advance` does not
//! reproduce — no CPU-observable state depends on it.
//!
//! Frame count per ROM is `PPU_ORACLE_FRAMES` (default 250 for a fast
//! `cargo test`; the campaign gate runs it at 2000). ROMs default to the
//! campaign set (SMB / Zelda / Contra / Punch-Out MMC2 control / SMB3
//! MMC3 control) but any missing ROM is skipped, not failed.

use std::io::Cursor;

use nes_core::cartridge::Cartridge;
use nes_core::input::Button;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

/// Video sink that discards output and never signals a written frame —
/// frame completion is detected via `ppu.frame` instead, so the driver
/// controls exactly one frame per outer step.
struct DiscardVideo;
impl VideoSink for DiscardVideo {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}

fn frames_env() -> u64 {
    std::env::var("PPU_ORACLE_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(250)
}

fn load(path: &str) -> Option<Nes> {
    let bytes = std::fs::read(path).ok()?;
    let cart = Cartridge::load(&mut Cursor::new(bytes)).ok()?;
    Some(Nes::new(cart))
}

fn load_state(nes: &mut Nes, path: &str) -> bool {
    let Ok(raw) = std::fs::read(path) else { return false };
    let magic = b"NCST\x01";
    let mut body = &raw[..];
    while body.len() >= magic.len() && &body[..magic.len()] == magic {
        body = &body[magic.len()..];
    }
    match bincode::deserialize::<nes_core::nes::State>(body) {
        Ok(state) => { nes.apply_state(&state); true }
        Err(_) => false,
    }
}

fn set_buttons(nes: &mut Nes, buttons: &[Button]) {
    for b in [
        Button::A, Button::B, Button::Select, Button::Start,
        Button::Up, Button::Down, Button::Left, Button::Right,
    ] {
        nes.game_pad_1().set_button_pressed(b, false);
    }
    for &b in buttons {
        nes.game_pad_1().set_button_pressed(b, true);
    }
}

/// Compact CPU-observable digest, minus the render pipeline. `full` adds
/// VRAM + RAM (used at frame boundaries).
fn digest(nes: &Nes, full: bool) -> Vec<u8> {
    let s = nes.get_state();
    let p = &s.ppu;
    let mut d = bincode::serialize(&(
        (&s.mapper, &s.cpu, &s.apu, &s.input, s.cycles),
        (&p.regs, &p.palette_ram, &p.oam, p.ppu_data_read_buffer, p.ppu_gen_latch),
        (p.cycles, p.scanline, p.scanline_start_cycle, p.frame,
         p.sprite_0_on_scanline, p.nmi_occurred, p.nmi_output),
    )).expect("serialize observable state");
    if full {
        d.extend(bincode::serialize(&(&s.ram, &p.vram)).expect("serialize ram/vram"));
    }
    d
}

/// Drive both instances through exactly one frame, comparing observable
/// state at every scanline boundary. Panics with the (frame, scanline,
/// dot) triple on the first divergence.
fn step_frame_lockstep(on: &mut Nes, off: &mut Nes, rom: &str, frame_idx: u64) {
    let target = on.ppu.frame + 1;
    let mut prev_sl = on.ppu.scanline;
    let mut steps: u64 = 0;
    while on.ppu.frame < target {
        {
            let mut v = DiscardVideo; let mut a = NullAudio;
            on.step(&mut v, &mut a);
        }
        {
            let mut v = DiscardVideo; let mut a = NullAudio;
            off.step(&mut v, &mut a);
        }
        steps += 1;
        // Safety valve against a runaway divergence that never lands a
        // frame boundary.
        assert!(steps < 200_000, "{rom}: frame {frame_idx} never completed (divergence?)");

        let crossed = on.ppu.scanline != prev_sl || on.ppu.frame >= target;
        if crossed {
            let frame_boundary = on.ppu.frame >= target;
            let da = digest(on, frame_boundary);
            let db = digest(off, frame_boundary);
            if da != db {
                panic!(
                    "SHADOW ORACLE DIVERGENCE on {rom} at frame {frame_idx}, \
                     scanline {} (advance dot {}) vs off scanline {} dot {}: \
                     advance is NOT observably exact",
                    on.ppu.scanline, on.ppu.scanline_cycle(),
                    off.ppu.scanline, off.ppu.scanline_cycle(),
                );
            }
            prev_sl = on.ppu.scanline;
        }
    }
}

fn run_oracle(rom: &str, state: Option<&str>, script: impl Fn(u64) -> Vec<Button>) {
    let (Some(mut on), Some(mut off)) = (load(rom), load(rom)) else {
        eprintln!("[shadow-oracle] SKIP (missing ROM): {rom}");
        return;
    };
    if let Some(st) = state {
        // A state that fails to load leaves the fresh boot — still a
        // valid (if less targeted) run, so only warn.
        let a = load_state(&mut on, st);
        let b = load_state(&mut off, st);
        if a != b { panic!("state loaded inconsistently for {rom}"); }
        if !a { eprintln!("[shadow-oracle] WARN (no state {st}) — running from boot"); }
    }
    on.ppu.set_ppu_scanline_advance(true);
    off.ppu.set_ppu_scanline_advance(false);

    let frames = frames_env();
    let _bufs = (vec![0u32; FRAME_PIXELS], vec![0u32; FRAME_PIXELS]);
    for f in 0..frames {
        let buttons = script(f);
        set_buttons(&mut on, &buttons);
        set_buttons(&mut off, &buttons);
        // Keep advance continuously engaged: re-arm skip_render each
        // frame (the frame boundary auto-clears it).
        on.set_skip_render(true);
        off.set_skip_render(true);
        step_frame_lockstep(&mut on, &mut off, rom, f);
    }
    eprintln!("[shadow-oracle] OK ({frames} frames): {rom}");
}

const SMB: &str = "../roms/Super Mario Bros. (World).nes";
const ZELDA: &str = "../roms/Legend of Zelda, The (USA) (Rev A).nes";
const ZELDA_STATE: &str = "../roms/zelda_start_ctrl.state.bin";
const CONTRA: &str = "../roms/Contra (USA).nes";
const CONTRA_STATE: &str = "../roms/Contra (USA)_start.state.bin";
const PUNCHOUT: &str = "../roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A).nes";
const PUNCHOUT_STATE: &str = "../roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A)_start.state.bin";
const SMB3: &str = "../roms/Super Mario Bros. 3 (USA).nes";
const KIRBY: &str = "../roms/Kirby's Adventure (USA) (Rev A).nes"; // MMC3 control

/// Walk right + jump — exercises scroll (v/t reloads) and the sprite-0
/// HUD split, the two hardest observable boundaries.
fn play_script(step: u64) -> Vec<Button> {
    match step % 8 {
        0..=2 => vec![Button::Right],
        3 => vec![Button::Right, Button::A],
        4..=6 => vec![Button::Right, Button::B],
        _ => vec![Button::Start, Button::Right],
    }
}

#[test]
fn shadow_oracle_smb_nrom() {
    run_oracle(SMB, None, play_script);
}

#[test]
fn shadow_oracle_zelda_mmc1() {
    run_oracle(ZELDA, Some(ZELDA_STATE), play_script);
}

#[test]
fn shadow_oracle_contra_uxrom() {
    run_oracle(CONTRA, Some(CONTRA_STATE), play_script);
}

#[test]
fn shadow_oracle_punchout_mmc2_control() {
    run_oracle(PUNCHOUT, Some(PUNCHOUT_STATE), play_script);
}

#[test]
fn shadow_oracle_smb3_mmc3_control() {
    run_oracle(SMB3, None, play_script);
}

/// MMC3 control (Kirby stands in for SMB3, which is absent from this
/// tree): a scanline-IRQ mapper always takes `advance`'s reference path
/// — this pins that path byte-exact through the boot + title animation,
/// including the mapper's own IRQ-counter mutations.
#[test]
fn shadow_oracle_kirby_mmc3_control() {
    run_oracle(KIRBY, None, play_script);
}
