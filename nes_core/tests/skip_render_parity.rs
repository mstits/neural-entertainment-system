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
