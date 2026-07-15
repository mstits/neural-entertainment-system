//! Rung-0 batchable-fraction profiler for the event-driven-PPU
//! catch-up campaign (docs/proposals/ppu_event_driven_catchup.md).
//!
//! Boots a ROM, applies an optional in-game start state, warms up, then
//! runs the skip-render (training) workload while the `ppu_batch_stats`
//! instrumentation classifies every visible scanline by the
//! observable-event classes that would force a per-dot advance. Prints
//! the per-class histogram and the resulting batchable fraction — the
//! share of visible lines a scanline-granular `advance` (Rung 1/2/3)
//! could fast-forward in closed form. These numbers validate or refute
//! the Rung 1-2 gain projections.
//!
//! Build + run (the feature is required — a plain build prints a hint):
//!   cargo run --release --features ppu_batch_stats \
//!     --example ppu_batch_profile
//!
//! Env vars:
//!   PROF_ROM    = path to the .nes ROM (required in practice)
//!   PROF_STATE  = path to an NCST start state, or "none"/unset for
//!                 power-on (a longer warm-up is used then)
//!   PROF_FRAMES = visible frames to measure (default 900)
//!   PROF_FS     = frame_skip cadence for the skip-render loop (16)
//!   PROF_WARM   = warm-up frames before the measurement window
//!                 (default 120 with a state, 400 without)

#[cfg(feature = "ppu_batch_stats")]
mod imp {
    use nes_core::cartridge::Cartridge;
    use nes_core::nes::Nes;
    use nes_core::ppu;
    use nes_core::sink::{AudioSink, Xrgb8888VideoSink};

    const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;
    const CPU_CYCLES_PER_FRAME: usize = 29781;

    struct Null;
    impl AudioSink for Null {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize {
            0
        }
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

    /// One frame of cycle-locked advance, mirroring
    /// python.rs::advance_one_frame. `skip` toggles the PPU pixel-write
    /// fast path (the training / spectator-subframe behaviour).
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

    fn pct(num: u64, den: u64) -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    }

    pub fn run() {
        let rom = std::env::var("PROF_ROM")
            .unwrap_or_else(|_| "roms/Super Mario Bros. (World).nes".into());
        let state = std::env::var("PROF_STATE").unwrap_or_default();
        let have_state = !state.is_empty() && state != "none";
        let frames: usize = std::env::var("PROF_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(900);
        let fs: usize = std::env::var("PROF_FS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
            .max(1);
        let warm: usize = std::env::var("PROF_WARM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if have_state { 120 } else { 400 });

        let bytes = std::fs::read(&rom).unwrap_or_else(|e| panic!("read rom {rom}: {e}"));
        let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("ines");
        let mut nes = Nes::new(cart);
        if have_state {
            load_state(&mut nes, &state);
        }

        let mut video_buf = vec![0u32; FRAME_PIXELS];
        let mut audio = Null;
        let mut target = nes.cycles;

        // Warm-up: reach steady in-game rendering before measuring.
        for _ in 0..warm {
            advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, false);
        }

        // Measure the skip-render workload: skip on all but the last
        // frame of each frame_skip batch (the trainer / spectator path).
        nes.ppu.reset_batch_stats();
        for i in 0..frames {
            let is_last = (i % fs) == fs - 1;
            advance_one_frame(&mut nes, &mut video_buf, &mut audio, &mut target, !is_last);
        }

        let s = nes.ppu.batch_stats();
        let v = s.visible_lines;

        println!("== ppu_batch_profile (Rung-0 batchable fraction) ==");
        println!("rom:    {rom}");
        println!(
            "state:  {}",
            if have_state { state.as_str() } else { "(power-on)" }
        );
        println!(
            "frames: {} measured ({} visible scanlines; fs={fs}, warm={warm})",
            s.frames, v
        );
        println!();
        println!(
            "  {:<26} {:>12} {:>9} {:>10}",
            "event class", "lines", "%visible", "per-frame"
        );
        let row = |name: &str, n: u64| {
            let pf = if s.frames == 0 { 0.0 } else { n as f64 / s.frames as f64 };
            println!("  {name:<26} {n:>12} {:>8.2}% {pf:>10.2}", pct(n, v));
        };
        row("mid-line MMIO (any)", s.lines_mmio);
        row("  -- $2000-7 reg write", s.lines_reg_write);
        row("  -- $2002 status read", s.lines_status_read);
        row("  -- rendering toggle", s.lines_render_toggle);
        row("sprite-0 pre-hit line", s.lines_sprite0_prehit);
        row("sprite-overflow line", s.lines_overflow);
        row("A12 scanline-IRQ line", s.lines_a12);
        row("mapper write (non-block)", s.lines_mapper_write);
        println!();
        println!("  HEADLINE — batchable fraction (visible lines a slice could fast-forward):");
        println!(
            "    strict  (Rung 1; A12 -> per-dot)   {:>12} / {v}  = {:>6.2}%",
            s.lines_batchable_strict,
            pct(s.lines_batchable_strict, v)
        );
        println!(
            "    horizon (Rung 2/3; A12 batched)    {:>12} / {v}  = {:>6.2}%",
            s.lines_batchable_with_a12,
            pct(s.lines_batchable_with_a12, v)
        );
    }
}

#[cfg(feature = "ppu_batch_stats")]
fn main() {
    imp::run();
}

#[cfg(not(feature = "ppu_batch_stats"))]
fn main() {
    eprintln!(
        "ppu_batch_profile requires the ppu_batch_stats feature:\n  \
         cargo run --release --features ppu_batch_stats --example ppu_batch_profile"
    );
}
