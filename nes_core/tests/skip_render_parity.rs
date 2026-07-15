//! Prove-It pattern regression test for the skip_render correctness bug.
//!
//! `Nes::set_skip_render(true)` is supposed to be a perf optimization that
//! skips pixel-production work on intermediate frames of a frame_skip
//! batch, under the assumption that "the rendered frame is correct because
//! the LAST frame in the batch runs the full tick path."
//!
//! That assumption is wrong for any game that relies on PPU side effects
//! during gameplay — sprite-0 hit, MMC3 scanline IRQ, scroll-register
//! updates, etc. The CPU runs at full speed across all 4 frames; the CPU
//! reads $2002 mid-frame to time scroll splits. If sprite-0 hit never
//! fires on the skipped frames, the CPU takes different branches than it
//! would under real hardware, and by the time the 4th (full-render) frame
//! starts, the CPU has written wrong scroll values. The "final frame
//! renders correctly" invariant collapses.
//!
//! These tests compare final framebuffers between:
//!   A. 120 frames via frame_skip=1 (all full ticks, never skip_render)
//!   B. 120 frames via frame_skip=4 (3 skip_render + 1 full per batch)
//!
//! Byte-exact match means skip_render preserves CPU-observable state.
//! Divergence means it's wrong.

use std::io::Cursor;

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn load(path: &str) -> Nes {
    let bytes = std::fs::read(path).expect("read rom");
    let cart = Cartridge::load(&mut Cursor::new(bytes)).expect("parse iNES");
    Nes::new(cart)
}

/// Advance exactly one NES video frame. Returns the filled XRGB8888 buffer.
fn step_one_frame(nes: &mut Nes, video_buf: &mut Vec<u32>) {
    let mut video = Xrgb8888VideoSink::new(video_buf);
    let mut audio = NullAudio;
    while !video.frame_written() {
        nes.step(&mut video, &mut audio);
    }
}

/// Run `frames` frames with skip_render bracketing: for each outer group of
/// `skip_group` frames, set skip_render=true for the first N-1 frames and
/// false on the last. This mirrors `pool.rs::step` + `python.rs::step`.
fn run_with_skip(nes: &mut Nes, frames: usize, skip_group: usize, buf: &mut Vec<u32>) {
    assert!(frames % skip_group == 0, "frames must be divisible by skip_group");
    for _ in 0..frames / skip_group {
        for i in 0..skip_group {
            let is_last = i + 1 == skip_group;
            nes.ppu.set_skip_render(skip_group > 1 && !is_last);
            step_one_frame(nes, buf);
        }
    }
    nes.ppu.set_skip_render(false);
}

/// Run `frames` frames batched into `skip_group` outer steps. On each
/// outer step, button state is latched from `per_step_button(step_idx)`
/// and held across all `skip_group` inner frames — matching the Python
/// `NESEnvironment::step(action, frame_skip)` semantics. That API
/// doesn't support mid-step button changes, so matching it is what
/// reproduces the in-game divergence we care about.
fn run_batched(
    nes: &mut Nes,
    frames: usize,
    skip_group: usize,
    per_step_button: impl Fn(usize) -> bool,
    buf: &mut Vec<u32>,
) {
    use nes_core::input::Button;
    assert!(frames.is_multiple_of(skip_group));
    let outer_steps = frames / skip_group;
    for step_idx in 0..outer_steps {
        let pressed = per_step_button(step_idx);
        nes.game_pad_1().set_button_pressed(Button::Start, pressed);
        for i in 0..skip_group {
            let is_last = i + 1 == skip_group;
            nes.ppu.set_skip_render(skip_group > 1 && !is_last);
            step_one_frame(nes, buf);
        }
    }
    nes.ppu.set_skip_render(false);
}

fn frame_diff(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn smb_title_idle_skip_render_matches_full_render() {
    let rom = "../roms/Super Mario Bros. (World).nes";
    let mut nes_full = load(rom);
    let mut nes_skip = load(rom);

    let mut buf_full = vec![0u32; FRAME_PIXELS];
    let mut buf_skip = vec![0u32; FRAME_PIXELS];

    // 120 frames, idle. fs=1 all-full vs fs=4 3-skip-1-full per batch.
    run_with_skip(&mut nes_full, 120, 1, &mut buf_full);
    run_with_skip(&mut nes_skip, 120, 4, &mut buf_skip);

    let diff = frame_diff(&buf_full, &buf_skip);
    assert_eq!(
        diff, 0,
        "skip_render drift on idle SMB title screen: {} / {} pixels differ",
        diff, FRAME_PIXELS,
    );
}

#[test]
fn smb_in_game_skip_render_matches_full_render() {
    // Load-bearing test. SMB's in-game HUD split is sprite-0-hit timed.
    // If skip_render breaks sprite-0-hit on intermediate frames, the
    // CPU's mid-frame scroll-update logic takes different branches on
    // every skipped frame, and by the 4th (full-render) frame the
    // scroll state is corrupted. Python-side reproduction shows 59k
    // pixels differ at 240 frames — effectively the whole screen.
    let rom = "../roms/Super Mario Bros. (World).nes";
    let mut nes_full = load(rom);
    let mut nes_skip = load(rom);

    let mut buf_full = vec![0u32; FRAME_PIXELS];
    let mut buf_skip = vec![0u32; FRAME_PIXELS];

    // 240 frames, Start held on outer steps whose start-frame falls in
    // [60, 70). Matches the Python repro pattern (button latched once
    // per step). Step covers frame_skip frames; fs=4 → steps 15..18
    // land in the window, so Start is held on frames 60-71.
    let start_held = |step_idx: usize, skip_group: usize| -> bool {
        let t = step_idx * skip_group;
        (60..70).contains(&t)
    };
    run_batched(&mut nes_full, 240, 1, |s| start_held(s, 1), &mut buf_full);
    run_batched(&mut nes_skip, 240, 4, |s| start_held(s, 4), &mut buf_skip);

    let diff = frame_diff(&buf_full, &buf_skip);
    assert_eq!(
        diff, 0,
        "skip_render drift on SMB gameplay (sprite-0-hit split): {} / {} pixels differ",
        diff, FRAME_PIXELS,
    );
}

/// Regression for the "bulk-step overshoot" bug. Background:
///
///   `Nes::step` runs one CPU instruction, including via `try_bulk_step`
///   or the AArch64 ASM path that batches multiple CPU cycles in one
///   burst. Those ticks advance the PPU in a `for _ in 0..n` loop; when
///   the PPU crosses a frame boundary mid-burst, `write_frame` fires and
///   `self.frame += 1`, then the PPU keeps ticking into scanline 0 of
///   the new frame for the remainder of the burst. The first ~50-100
///   cycles of scanline 0 therefore render with the OUTGOING frame's
///   `skip_render` value, not the incoming frame's.
///
///   That only matters at a skip→full transition (frame_skip > 1, last
///   frame of the batch). The overshoot ticks render with skip_render =
///   true → render_pixel bails before writing frame_buffer, and the
///   affected pixels retain whatever stale content was there from the
///   last full-render frame. Visible result: 50-100 px of row-0 drift
///   vs the reference frame_skip=1 run.
///
///   Fix (in `Ppu::tick`): clear `self.skip_render` inside the
///   `if self.scanline > PRE_RENDER_SCANLINE` block so the overshoot
///   ticks always render correctly. Callers re-set skip_render before
///   every advance_one_frame so this is semantically a no-op for them.
///
/// Repro: mimic NESEnvironment by advancing one boot frame with
/// skip_render=false (the `reset()` warm-up), then run 240 in-game
/// frames with the [T,T,T,F] fs=4 pattern. Pre-fix this produced 50 px
/// on SMB scanline 0 cols 22-71. Post-fix it's byte-exact.
#[test]
fn smb_in_game_skip_render_with_reset_advance_is_byte_exact() {
    let rom = "../roms/Super Mario Bros. (World).nes";
    let mut nes_full = load(rom);
    let mut nes_skip = load(rom);

    let mut buf_full = vec![0u32; FRAME_PIXELS];
    let mut buf_skip = vec![0u32; FRAME_PIXELS];

    // Mirror NESEnvironment::reset()'s advance_one_frame.
    nes_full.ppu.set_skip_render(false);
    nes_skip.ppu.set_skip_render(false);
    step_one_frame(&mut nes_full, &mut buf_full);
    step_one_frame(&mut nes_skip, &mut buf_skip);

    let start_held = |step_idx: usize, skip_group: usize| -> bool {
        let t = step_idx * skip_group;
        (60..70).contains(&t)
    };
    run_batched(&mut nes_full, 240, 1, |s| start_held(s, 1), &mut buf_full);
    run_batched(&mut nes_skip, 240, 4, |s| start_held(s, 4), &mut buf_skip);

    let diff = frame_diff(&buf_full, &buf_skip);
    assert_eq!(
        diff, 0,
        "skip_render overshoot regression: {} / {} pixels differ after reset-advance + 240f gameplay",
        diff, FRAME_PIXELS,
    );
}

// ---------------------------------------------------------------------
// Refined skip-render fast-path guards.
//
// The refined skip-render path adds, on top of plain skip_render, the
// idle-HBlank early-return — dots 258..=320 on visible scanlines
// collapse to `cycles += 1` (shipped ON, all mappers). It is an
// unobserved-work elision: it must leave every CPU-observable side
// effect (sprite-0 hit, scroll v/t, OAM, sprite overflow, NMI, mapper
// IRQ/CHR state, RAM) byte-identical. These guards pin that — the
// refined path is fully byte-exact, including the shift registers.
// ---------------------------------------------------------------------

/// Load an `NCST\x01`-framed save-state onto `nes` (mirrors the loader in
/// `python.rs` / the ppu_state_profile harness).
fn load_state(nes: &mut Nes, path: &str) {
    let raw = std::fs::read(path).expect("read state file");
    let magic = b"NCST\x01";
    let mut body = &raw[..];
    while body.len() >= magic.len() && &body[..magic.len()] == magic {
        body = &body[magic.len()..];
    }
    let state: nes_core::nes::State =
        bincode::deserialize(body).expect("bincode state");
    nes.apply_state(&state);
}

/// Serialize ONLY the CPU-observable machine state: CPU RAM + registers,
/// mapper (banks / IRQ counters / CHR latch), APU, controller latch, and
/// the PPU register file (v/t/x/w, PPUCTRL/MASK/STATUS, OAMADDR, read
/// buffer), OAM, VRAM (nametables), palette RAM, the frame/scanline/cycle
/// counters, and the sprite-0 / NMI flags. DELIBERATELY excludes the BG +
/// sprite fetch pipeline (name/attr/bitmap latches, bg shift registers,
/// sprite pattern/attribute/x registers): those are pure render
/// intermediates read only by `render_pixel`. If any of the INCLUDED
/// state ever diverges between the refined path and the full tick body,
/// the CPU could observe it.
fn observable_digest(nes: &Nes) -> Vec<u8> {
    let s = nes.get_state();
    let p = &s.ppu;
    let mut d =
        bincode::serialize(&(&s.ram, &s.mapper, &s.cpu, &s.apu, &s.input, s.cycles))
            .expect("serialize nes-level observable state");
    d.extend(
        bincode::serialize(&(
            &p.regs,
            &p.vram,
            &p.palette_ram,
            &p.oam,
            p.ppu_data_read_buffer,
            p.ppu_gen_latch,
            p.cycles,
            p.scanline,
            p.scanline_start_cycle,
            p.frame,
            p.sprite_0_on_scanline,
            p.nmi_occurred,
            p.nmi_output,
        ))
        .expect("serialize ppu observable state"),
    );
    d
}

/// Run `nes` for one outer step of `skip_group` frames with the trainer's
/// skip-render bracketing (skip the first N-1, render the last), holding
/// the given buttons across the whole step.
fn step_group(nes: &mut Nes, skip_group: usize, buf: &mut Vec<u32>, buttons: &[nes_core::input::Button]) {
    use nes_core::input::Button;
    // Release everything, then press the requested buttons. (`Button`
    // is Copy but not PartialEq, so we clear-then-set rather than test
    // membership.)
    for b in [
        Button::A, Button::B, Button::Select, Button::Start,
        Button::Up, Button::Down, Button::Left, Button::Right,
    ] {
        nes.game_pad_1().set_button_pressed(b, false);
    }
    for &b in buttons {
        nes.game_pad_1().set_button_pressed(b, true);
    }
    for i in 0..skip_group {
        let is_last = i + 1 == skip_group;
        nes.ppu.set_skip_render(skip_group > 1 && !is_last);
        step_one_frame(nes, buf);
    }
    nes.ppu.set_skip_render(false);
}

/// Core guard: run two copies of the same ROM/state through an identical
/// input script under fs=`skip_group` skip-render bracketing — one with
/// the refined skip-render path ON (the shipped default), one with it
/// forced OFF (`set_refined_skip_render(false)` = the exact full per-cycle
/// tick body) — and assert their full observable state is byte-identical
/// at EVERY frame-group boundary.
fn refined_off_vs_on_state_parity(
    rom: &str,
    state: Option<&str>,
    skip_group: usize,
    steps: usize,
    script: impl Fn(usize) -> Vec<nes_core::input::Button>,
) {
    let mut on = load(rom);
    let mut off = load(rom);
    if let Some(st) = state {
        load_state(&mut on, st);
        load_state(&mut off, st);
    }
    // `on` keeps the production default (refined ON); force `off` to the
    // full tick body.
    off.ppu.set_refined_skip_render(false);

    let mut buf = vec![0u32; FRAME_PIXELS];
    for step in 0..steps {
        let buttons = script(step);
        step_group(&mut on, skip_group, &mut buf, &buttons);
        step_group(&mut off, skip_group, &mut buf, &buttons);
        let da = observable_digest(&on);
        let db = observable_digest(&off);
        assert!(
            da == db,
            "refined skip-render observable-state divergence on {rom} at step {step} \
             (skip_group={skip_group}): the fast path changed CPU-observable state",
        );
    }
}

/// SMB (NROM) is the sprite-0 canary — its in-game HUD split is timed off
/// the sprite-0 hit, the exact side effect the refined path must preserve.
/// Boot, press Start into gameplay, then hold Right; refined off vs on
/// must stay byte-identical in observable state every frame.
#[test]
fn smb_refined_skip_render_state_parity_off_vs_on() {
    let script = |step: usize| {
        use nes_core::input::Button;
        if (15..18).contains(&step) {
            vec![Button::Start]
        } else if step >= 20 {
            vec![Button::Right]
        } else {
            vec![]
        }
    };
    refined_off_vs_on_state_parity(
        "../roms/Super Mario Bros. (World).nes",
        None,
        4,
        60,
        script,
    );
}

/// Zelda (MMC1) from an in-game save-state — HUD sprite-0 active from
/// frame 0, banked CHR. Walk around; observable state must match refined
/// off vs on.
#[test]
fn zelda_refined_skip_render_state_parity_off_vs_on() {
    let script = |step: usize| {
        use nes_core::input::Button;
        match step % 4 {
            0 | 1 => vec![Button::Right],
            _ => vec![Button::Left, Button::B],
        }
    };
    refined_off_vs_on_state_parity(
        "../roms/Legend of Zelda, The (USA) (Rev A).nes",
        Some("../roms/zelda_start_ctrl.state.bin"),
        16,
        30,
        script,
    );
}

/// Zelda (MMC1) frame-buffer parity: fs=1 (skip_render never set →
/// refined never fires) vs fs=16 (refined fires on 15/16 frames) must
/// produce a byte-identical rendered frame — the MMC1 sprite-0-HUD analog
/// of the SMB frame-buffer guards above. A constant held button keeps the
/// per-frame input identical across both frame-skip rates, so the ONLY
/// variable is the refined skip-render path.
#[test]
fn zelda_in_game_skip_render_matches_full_render() {
    use nes_core::input::Button;
    let rom = "../roms/Legend of Zelda, The (USA) (Rev A).nes";
    let state = "../roms/zelda_start_ctrl.state.bin";
    let mut nes_full = load(rom);
    let mut nes_skip = load(rom);
    load_state(&mut nes_full, state);
    load_state(&mut nes_skip, state);

    let mut buf_full = vec![0u32; FRAME_PIXELS];
    let mut buf_skip = vec![0u32; FRAME_PIXELS];

    // Hold Right the whole run: scrolls the world (exercises scroll v/t
    // reloads) with the HUD sprite-0 active every frame.
    let held = [Button::Right];
    // 320 frames: fs=1 all-full vs fs=16 (15 skip + 1 full) per step.
    for _ in 0..320 {
        step_group(&mut nes_full, 1, &mut buf_full, &held);
    }
    for _ in 0..20 {
        step_group(&mut nes_skip, 16, &mut buf_skip, &held);
    }

    let diff = frame_diff(&buf_full, &buf_skip);
    assert_eq!(
        diff, 0,
        "refined skip_render drift on Zelda (MMC1) gameplay: {} / {} pixels differ",
        diff, FRAME_PIXELS,
    );
}
