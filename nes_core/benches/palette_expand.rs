//! criterion port of `examples/bench_metal_vs_neon.rs`'s CPU side.
//!
//! Proof-of-pattern for the audit finding: every bench_*.rs /
//! *_bench.rs under nes_core/examples/ is a hand-rolled
//! `Instant::now()` timing loop with no black_box() and no
//! statistical confidence interval, so a regression smaller than
//! run-to-run noise goes unnoticed. This is the first of those
//! ported onto criterion as the pattern other benches should follow;
//! the other twelve are intentionally left alone (see the audit —
//! this is establishing the pattern, not a migration).
//!
//! Picked `Xrgb8888VideoSink::write_frame` because it's a pure,
//! self-contained hot-path function (palette-index bytes in, XRGB8888
//! pixels out) that needs no Cartridge/Nes/Pool to exercise — unlike
//! most of the other examples/*bench*.rs files, which drive a full
//! ROM. On aarch64 this calls the real shipped NEON gather/store
//! path (`xrgb8888_write_neon`), not a re-implementation of it, so
//! the number this produces (once measured honestly, see below) is
//! the actual per-frame palette-expansion cost paid on every
//! rendered frame.
//!
//! IMPORTANT — NO BASELINE HAS BEEN CAPTURED FROM THIS BENCH.
//! This was ported and verified to compile + run (`cargo bench
//! --no-run`, then a short live run) on a machine that was, at the
//! time, running multiple concurrent agent workflows plus a
//! just-finished multi-seed training run. Any ns/iter number
//! measured under that load is noise, not signal — this project
//! already threw out one perf baseline (RG-1d) that was contaminated
//! this same way. Do NOT treat any number this bench has printed so
//! far as real. Before this bench's output means anything:
//!   1. Confirm the machine is idle (no other cargo/python/training
//!      processes competing for cores).
//!   2. Run `cargo bench --bench palette_expand` to let criterion
//!      collect its own baseline (criterion always compares against
//!      the previous local run — the first idle run establishes the
//!      first trustworthy one).
//!   3. Only then treat `cargo bench --bench palette_expand` deltas
//!      as meaningful regression signal.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;

fn bench_xrgb8888_write_frame(c: &mut Criterion) {
    // Deterministic synthetic frame: palette indices cycling 0..63 so
    // every LUT entry is exercised at least once, matching the input
    // shape used in examples/bench_metal_vs_neon.rs.
    let frame_buffer: Vec<u8> = (0..FRAME_PIXELS).map(|i| (i % 64) as u8).collect();
    let mut out = vec![0u32; FRAME_PIXELS];

    c.bench_function("xrgb8888_write_frame_256x240", |b| {
        b.iter(|| {
            let mut sink = Xrgb8888VideoSink::new(&mut out);
            // black_box on the input defeats constant-folding the
            // whole palette lookup at compile time; black_box on the
            // sink return value defeats dead-store elimination of the
            // write loop, since nothing outside the closure otherwise
            // reads `out`.
            sink.write_frame(black_box(&frame_buffer));
            black_box(&out);
        });
    });
}

criterion_group!(benches, bench_xrgb8888_write_frame);
criterion_main!(benches);
