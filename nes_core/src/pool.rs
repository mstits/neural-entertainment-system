//! Rust-native worker pool — drop-in replacement for Python's
//! `src/emulation/parallel_pool.py` with zero IPC overhead.
//!
//! The Python pool uses `multiprocessing.Process` per worker, action
//! / reset commands flowing through `mp.Queue`, frames published via
//! `FrameTransport` shared memory, and results collected via another
//! queue. Every step crosses the process boundary 16 times (one per
//! worker) with serialize/deserialize overhead at each hop.
//!
//! This pool keeps all N NES instances in-process. `step_all` iterates
//! them with rayon's thread pool (M4 has plenty of performance cores).
//! Returning to Python is a single PyO3 call per generation step,
//! carrying `num_workers` frames + RAM snapshots + done flags in one
//! shot. Multiprocessing, queues, shm — all of that plumbing is gone.
//!
//! Intentionally minimal — no curriculum-aware start-state path
//! switching, no demo-worker slot, no audio-rate introspection (those
//! were Python-layer concerns tied to the multiprocessing worker
//! lifetime). If we need them, they're additive features on this
//! surface.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

use crate::cartridge::Cartridge;
use crate::input::Button;
use crate::nes::Nes;
use crate::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use crate::sink::{AudioSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;
const RAM_SIZE: usize = 2048;

/// Versioned save-state magic. Mirrors `python.rs::STATE_MAGIC` —
/// pool snapshots and env snapshots use the same header so blobs are
/// interchangeable across the two code paths.
const POOL_STATE_MAGIC: &[u8; 5] = b"NCST\x01";

// Button bitmask layout — matches the nes-py convention that the rest
// of the Python ecosystem already uses (`src/emulation/frame_utils.py`,
// `src/training/trainer.py` action table, legacy `.state.bin`
// recordings). Bit order reads "right→A" high-to-low. Historical
// pre-alignment: nes_core used the reversed layout (A=bit0,
// right=bit7). That silently swapped every button in the trainer →
// pool path. Fixed 2026-04-21 in the audit sweep.
const BUTTON_RIGHT: u8 = 1 << 7;
const BUTTON_LEFT: u8 = 1 << 6;
const BUTTON_DOWN: u8 = 1 << 5;
const BUTTON_UP: u8 = 1 << 4;
const BUTTON_START: u8 = 1 << 3;
const BUTTON_SELECT: u8 = 1 << 2;
const BUTTON_B: u8 = 1 << 1;
const BUTTON_A: u8 = 1 << 0;

/// Per-worker state. Stored in `UnsafeCell` so rayon's `par_iter`
/// can hand out exclusive per-index access without paying the
/// atomic lock/unlock cost of a `Mutex`. Each rayon task touches
/// exactly one worker at its own index, so there is no aliasing;
/// the `Mutex` previously used here was uncontended overhead
/// (measured ~20-50ns × 16 workers × 1700+ sps = ~27K atomic ops/s).
///
/// All sequential access (reset_all, save_worker_state,
/// set_worker_pace, set_worker_audio, drain_audio,
/// set_batched_render_mode,
/// load_worker_state, load_start_state) runs on the Python trainer
/// thread and never overlaps with the parallel `step_all`/`reset_all`
/// rayon closures — those are the only two methods that dispatch
/// into the rayon pool, and they are themselves called sequentially
/// from Python.
///
/// `dead` is flipped when the worker's step panics (via
/// `catch_unwind` in `step_all`/`reset_all`). Dead workers are
/// skipped on subsequent steps, keeping the 15 healthy workers
/// alive while a faulty ROM/state combination is isolated. The
/// trainer side sees the dead worker's result as (frame-of-zeros,
/// ram-of-zeros, done=true) so its episode cleanly terminates.
struct Worker {
    nes: Nes,
    // RGB palette-index → XRGB8888 conversion scratch.
    video_buf: Vec<u32>,
    // Captured audio samples since the last drain.
    audio: Vec<i16>,
    // Marks this worker as a spectator (GUI-visible / live-audio)
    // worker. Pacing itself is pool-level (see `Pool::pace_pool_step`)
    // so rayon threads never sleep; this flag feeds the pool's
    // "any worker paced?" check and controls sub-frame rendering
    // (see `spectator_subframe_skip`).
    realtime_pace: bool,
    // Cached start-state snapshot for fast `reset()` (<=ms restore).
    start_state_snapshot: Option<Vec<u8>>,
    // Set to `true` after a panic in this worker's step/reset. Once
    // dead, all future step/reset calls short-circuit to a zero
    // frame + done=true.
    dead: bool,
    // Gray-intermediate scratch reused across every preprocess call.
    // Without this the preprocess pipeline allocated a 61 KB gray Vec
    // per step per worker — at 16 workers × 1700 sps that was ~1.5 GB/s
    // of allocator churn on a buffer that gets overwritten every step.
    gray_scratch: Vec<u8>,
    // Absolute CPU cycle target for the next frame boundary. Mirrors
    // the cycle-lock in `NESEnvironment::advance_one_frame` — without
    // it, training workers run the old "until PPU writes the frame"
    // model and re-introduce the Zelda cave-stuck / sword-pickup defect
    // that the cycle-lock fix closed for Play-window / parity. See
    // `python.rs::advance_one_frame` for the rationale.
    frame_cycle_target: Option<usize>,
    // Hardware-true frame anchoring (config): when set,
    // `advance_one_frame` runs to the next real PPU frame boundary
    // (29,780/29,781 CPU cycles alternating with the odd-frame dot
    // skip) instead of the legacy fixed 29,781-cycle block. Mirror of
    // `NESEnvironment::hw_frame_anchor` — see that field's doc for the
    // drift receipt. Pool needs its own copy because the anchor is a
    // frame-advance policy, not `Nes` state: without it a tape built
    // or replayed on a Pool structurally cannot match Mesen's
    // inputPolled schedule, so a Pool-built tape can never be
    // certified against Mesen on its own machine. Worker field, not
    // `Nes` field, so it is deliberately NOT part of `State` and
    // survives `load_worker_state` as sticky config — same semantics
    // as every other hw flag. Default OFF: training/parity paths
    // depend on the cycle-locked step.
    hw_frame_anchor: bool,
    /// Peak SMB world-x position seen during the most recent
    /// `step()` call. Computed inside the frame_skip loop so we
    /// capture transient peaks that a final-frame-only RAM read
    /// would miss (Mario could reach x=1810 mid-jump and end the
    /// step at x=1700 after a death-fall, with the reward function
    /// otherwise seeing only 1700 and never crediting the 1810
    /// crossing toward checkpoint or max_x_visited). The trainer
    /// reads this via `peek_max_x_per_worker` and overrides the
    /// reward function's x-position view. SMB-specific addresses
    /// hard-coded; harmless for other games (always reports the
    /// final-frame x there).
    last_max_x: u32,
}

impl Worker {
    fn new(cart: Cartridge) -> Self {
        let mut nes = Nes::new(cart);
        // Training workers default to audio-off. Only `set_worker_pace`
        // (audio-solo-X mode) flips this back on for the observed
        // worker, mirroring the realtime_pace toggle. Saves the
        // 43 KHz × N-workers sample-gen + mix work on all the
        // discarded ones.
        nes.set_audio_output_enabled(false);
        Self {
            nes,
            video_buf: vec![0u32; FRAME_PIXELS],
            audio: Vec::new(),
            realtime_pace: false,
            start_state_snapshot: None,
            dead: false,
            gray_scratch: vec![0u8; FRAME_PIXELS],
            frame_cycle_target: None,
            hw_frame_anchor: false,
            last_max_x: 0,
        }
    }

    fn reset(&mut self) {
        if let Some(snap) = &self.start_state_snapshot {
            match bincode::deserialize::<crate::nes::State>(snap) {
                Ok(state) => self.nes.apply_state(&state),
                Err(e) => {
                    // Cached snapshot bytes are corrupted somehow.
                    // Falling back to a cold reset is correct behavior
                    // (better than panicking), but we MUST surface the
                    // failure — without this, training silently runs
                    // every episode from the title screen and looks
                    // exactly like "policy isn't moving" to the
                    // trainer, which is impossible to diagnose without
                    // peering at frames. Log and clear the cached
                    // snapshot so subsequent resets don't keep paying
                    // the deserialization cost on a known-bad blob.
                    eprintln!(
                        "[nes_core::Pool] Worker::reset: cached start-state \
                         snapshot deserialization failed ({}); cold-reset and \
                         clearing cached snapshot.",
                        e
                    );
                    self.start_state_snapshot = None;
                    self.nes.reset();
                }
            }
        } else {
            self.nes.reset();
        }
        self.audio.clear();
        self.frame_cycle_target = None;
        self.advance_one_frame();
    }

    fn advance_one_frame(&mut self) {
        let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);
        let mut sink = AudioVecSink { buf: &mut self.audio };
        // Cycle-locked frame advance — exact 29781 CPU cycles per call,
        // matching nes-py / LaiNES upstream. Mirror of the fix in
        // `python.rs::advance_one_frame`. The bulk-step margin must
        // be at least the mapper's `asm_bulk_cycles` (32 for NROM /
        // SMB; 1-4 for everything else); a hard-coded 7 was too low
        // and let `nes.step()` overshoot target by 18-25 cycles each
        // frame, breaking cycle alignment between training workers
        // and the parity reference.
        const CPU_CYCLES_PER_FRAME: usize = 29781;
        if self.hw_frame_anchor {
            // Hardware-true frame boundary: run to the next PPU frame
            // increment (29,780/29,781 CPU cycles alternating). Input
            // bytes then land on real frames, matching Mesen's
            // inputPolled schedule on tape replays. Cap at ~2 frames
            // of cycles so a wedged PPU can't loop forever. Verbatim
            // port of `python.rs::advance_one_frame`'s anchored branch
            // — keep the two in sync.
            let start_frame = self.nes.ppu.frame;
            let cap = self.nes.cycles + 2 * CPU_CYCLES_PER_FRAME + 64;
            while self.nes.ppu.frame == start_frame && self.nes.cycles < cap {
                if self.nes.oam_dma.active {
                    self.nes.tick(&mut video, &mut sink);
                } else {
                    self.nes.step(&mut video, &mut sink);
                }
            }
            self.frame_cycle_target = None;
            return;
        }
        let target = self
            .frame_cycle_target
            .map(|t| t + CPU_CYCLES_PER_FRAME)
            .unwrap_or_else(|| self.nes.cycles + CPU_CYCLES_PER_FRAME);
        let margin = self.nes.asm_bulk_cycles_margin();
        while self.nes.cycles + margin < target {
            // `margin` only bounds a normal `step()` (<= mapper
            // `asm_bulk_cycles`). It does NOT bound a step taken while
            // an OAM DMA is active: `Nes::step` falls to its slow path
            // (`while !tick() {}`) and consumes the entire ~513-514
            // cycle DMA transfer PLUS the following instruction in one
            // call, overshooting `target` by ~500 cycles and breaking
            // the per-frame cycle lock that SMB/Zelda/Contra parity
            // depends on. When a DMA is in progress, advance one CPU
            // cycle at a time so the transfer drains without
            // overshooting; bulk-stepping resumes automatically once
            // `oam_dma.active` clears. This is a strict no-op whenever
            // no DMA straddles the bulk window (the common case): the
            // same `tick()` calls run, just from this loop instead of
            // inside `step()`, landing on identical state at `target`.
            if self.nes.oam_dma.active {
                self.nes.tick(&mut video, &mut sink);
            } else {
                self.nes.step(&mut video, &mut sink);
            }
        }
        while self.nes.cycles < target {
            self.nes.tick(&mut video, &mut sink);
        }
        self.frame_cycle_target = Some(target);
    }

    fn step(&mut self, action: u8, frame_skip: u32, spectator_subframe_skip: bool) -> bool {
        self.apply_buttons(action);
        // Reset peak tracker. Read x BEFORE the first frame so the
        // baseline includes the entry state (otherwise a step that
        // moves Mario backward would report last_max_x=0 instead of
        // pre-step x).
        let initial_ram = self.nes.system_ram();
        self.last_max_x = 256 * initial_ram[0x006D] as u32 + initial_ram[0x0086] as u32;
        // Sub-frame render skip. Unpaced training workers always skip
        // the first N-1 sub-frames (proven emulation-neutral — see
        // tests/skip_render_parity.rs; sprite-0 / NMI / IRQ / RAM are
        // bit-identical, only framebuffer writes are elided). Spectator
        // (paced) workers historically rendered every sub-frame even
        // though only the final one ships to Python; with
        // `spectator_subframe_skip` (pool default ON) they use the same
        // final-frame-only path, cutting their PPU pixel work ~N×.
        let use_skip = !self.realtime_pace || spectator_subframe_skip;
        for i in 0..frame_skip {
            let is_last = i + 1 == frame_skip;
            self.nes.set_skip_render(use_skip && !is_last);
            self.advance_one_frame();
            // Sample SMB x position after each frame. Cheap read —
            // 2 RAM bytes — vs the cost of the NES frame just run.
            // For non-SMB games this still fires but the values are
            // meaningless; the trainer only consults it when running
            // a Mario reward profile.
            let ram = self.nes.system_ram();
            let x = 256 * ram[0x006D] as u32 + ram[0x0086] as u32;
            if x > self.last_max_x {
                self.last_max_x = x;
            }
        }
        self.nes.set_skip_render(false);
        // Realtime pacing is pool-level (`Pool::pace_pool_step`), NOT
        // per-worker: a sleep here would park one of the ~12 rayon
        // threads per paced worker, capping the number of realtime
        // spectator workers at the thread count. The pool sleeps ONCE
        // on the calling thread after the parallel step completes.
        //
        // Nes doesn't expose a per-step done flag in this codebase;
        // done-ness is rewards-driven in the trainer. Always return
        // false here to match existing worker behaviour.
        false
    }

    fn apply_buttons(&mut self, mask: u8) {
        let pad = self.nes.game_pad_1();
        pad.set_button_pressed(Button::Right, mask & BUTTON_RIGHT != 0);
        pad.set_button_pressed(Button::Left, mask & BUTTON_LEFT != 0);
        pad.set_button_pressed(Button::Down, mask & BUTTON_DOWN != 0);
        pad.set_button_pressed(Button::Up, mask & BUTTON_UP != 0);
        pad.set_button_pressed(Button::Start, mask & BUTTON_START != 0);
        pad.set_button_pressed(Button::Select, mask & BUTTON_SELECT != 0);
        pad.set_button_pressed(Button::B, mask & BUTTON_B != 0);
        pad.set_button_pressed(Button::A, mask & BUTTON_A != 0);
    }
}

/// Acquire a mutex, recovering the inner value even if a previous
/// holder panicked (poisoned the lock). A poisoned worker is
/// already marked `dead` and will short-circuit; we never want to
/// propagate the poison panic further.
/// Configure rayon's global thread pool for Apple Silicon M-series
/// chips. M4 Max / Pro ship with both performance cores (P) and
/// efficiency cores (E); rayon's default `num_cpus::get()` returns
/// the full count and happily schedules compute-bound tasks onto
/// E-cores where they run at ~40% of P-core throughput. Measured on
/// Zelda at frame_skip=16: 16 rayon threads → 1144 total sps; 12
/// rayon threads → 1410 total sps. Cap to the P-core count so every
/// worker thread lands on a fast core.
///
/// The cap is soft — RAYON_NUM_THREADS wins if set. Callers can also
/// override by constructing their own `rayon::ThreadPoolBuilder` in
/// the main thread before creating a Pool.
fn init_rayon_pool_for_apple_silicon(num_workers: usize) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return;
        }
        let total = num_cpus::get();
        // Apple Silicon M-series has P cores + E cores. E cores run
        // compute-bound tasks at ~40% of P-core throughput.
        //
        // P-core counts (estimated as `total - 4` since every current
        // M-series chip has 4 E cores; Ultra has 8):
        //   M1/M2/M3 base:    4 P   M4 base: 4 P
        //   M1/M2/M3 Pro:     6-8 P M4 Pro: 8 P
        //   M4 Max:           12 P
        //   M2/M3 Ultra:      16 P
        //
        // Sizing heuristic:
        //   * If num_workers ≤ P-cores: use P-cores. Every worker
        //     lands on a fast core — measured 1300 sps at 12 workers
        //     vs 1020 sps if we queue 16 workers onto 12 threads.
        //   * If num_workers > P-cores: rayon must match num_workers
        //     (up to total cores). Queueing workers onto fewer threads
        //     than requested is worse than paying E-core tax — measured
        //     1113 sps at 16 threads with 16 workers vs 1020 sps with
        //     12 threads + 4 queued workers.
        //
        // Init-once on first Pool creation so repeat constructions
        // (eg the scaling bench) don't try to resize the global pool.
        let p_cores = total.saturating_sub(4).max(1);
        let threads = if num_workers <= p_cores {
            p_cores
        } else {
            num_workers.min(total)
        };
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("nes-pool-{i}"))
            .build_global();
    });
}

/// Unpack a flat slice of XRGB8888 pixels (`0x00RRGGBB` in a u32) into
/// tightly-packed RGB8 bytes. Dst must be exactly 3× src len.
///
/// On aarch64 we use NEON's `vld4q_u8` to deinterleave 16 u32 pixels
/// (64 bytes) into 4 byte vectors — B, G, R, Pad — then re-interleave
/// the first three with `vst3q_u8` for 48 bytes of output per loop
/// iteration. Measured 6-8× faster than the per-pixel scalar loop on
/// M4 Max; sustained training throughput is sensitive to this since
/// the rayon collect path runs once per `step_all` across every
/// worker.
#[inline]
fn xrgb_to_rgb(src: &[u32], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), src.len() * 3);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { xrgb_to_rgb_neon(src, dst) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, &px) in src.iter().enumerate() {
            dst[i * 3] = ((px >> 16) & 0xFF) as u8;
            dst[i * 3 + 1] = ((px >> 8) & 0xFF) as u8;
            dst[i * 3 + 2] = (px & 0xFF) as u8;
        }
    }
}

/// Go directly from XRGB8888 to grayscale u8 — headless mode's
/// equivalent of xrgb_to_rgb + rgb_to_grayscale fused into one pass.
/// Skips the intermediate 184 KB RGB Vec entirely (~6 GB/s saved at
/// 16 workers × 2200 sps). Uses the same 77/150/29 luma weights as
/// preprocess::rgb_to_grayscale so the downsample output is
/// byte-identical to the two-stage path.
#[inline]
fn xrgb_to_gray(src: &[u32], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { xrgb_to_gray_neon(src, dst) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, &px) in src.iter().enumerate() {
            let r = ((px >> 16) & 0xFF) as u32;
            let g = ((px >> 8) & 0xFF) as u32;
            let b = (px & 0xFF) as u32;
            dst[i] = ((77 * r + 150 * g + 29 * b) >> 8).min(255) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xrgb_to_gray_neon(src: &[u32], dst: &mut [u8]) {
    use core::arch::aarch64::*;
    let n = src.len();
    let full = n & !15; // 16-pixel blocks
    let src_u8 = src.as_ptr() as *const u8;
    let dst_p = dst.as_mut_ptr();
    // Luma weights as u16 lanes.
    let wr = vdupq_n_u16(77);
    let wg = vdupq_n_u16(150);
    let wb = vdupq_n_u16(29);
    let mut i = 0usize;
    while i < full {
        let quad = unsafe { vld4q_u8(src_u8.add(i * 4)) };
        let b8 = quad.0;
        let g8 = quad.1;
        let r8 = quad.2;
        // Widen each u8 vector into two u16 halves.
        let r_lo = vmovl_u8(vget_low_u8(r8));
        let r_hi = vmovl_u8(vget_high_u8(r8));
        let g_lo = vmovl_u8(vget_low_u8(g8));
        let g_hi = vmovl_u8(vget_high_u8(g8));
        let b_lo = vmovl_u8(vget_low_u8(b8));
        let b_hi = vmovl_u8(vget_high_u8(b8));
        // Luma = (77*r + 150*g + 29*b) >> 8.
        let y_lo = vshrq_n_u16::<8>(
            vaddq_u16(vaddq_u16(vmulq_u16(r_lo, wr), vmulq_u16(g_lo, wg)), vmulq_u16(b_lo, wb))
        );
        let y_hi = vshrq_n_u16::<8>(
            vaddq_u16(vaddq_u16(vmulq_u16(r_hi, wr), vmulq_u16(g_hi, wg)), vmulq_u16(b_hi, wb))
        );
        // Narrow both halves back to u8 and store.
        let y8 = vcombine_u8(vqmovn_u16(y_lo), vqmovn_u16(y_hi));
        unsafe { vst1q_u8(dst_p.add(i), y8) };
        i += 16;
    }
    // Scalar tail.
    while i < n {
        let px = unsafe { *src.as_ptr().add(i) };
        let r = ((px >> 16) & 0xFF) as u32;
        let g = ((px >> 8) & 0xFF) as u32;
        let b = (px & 0xFF) as u32;
        unsafe { *dst_p.add(i) = ((77 * r + 150 * g + 29 * b) >> 8).min(255) as u8; }
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xrgb_to_rgb_neon(src: &[u32], dst: &mut [u8]) {
    use core::arch::aarch64::*;
    let n = src.len();
    let full = n & !15; // 16-pixel blocks
    let src_u8 = src.as_ptr() as *const u8;
    let dst_p = dst.as_mut_ptr();
    let mut i = 0usize;
    while i < full {
        // 64 input bytes = 16 XRGB pixels. vld4q loads as 4 byte lanes
        // deinterleaved: v.0=B, v.1=G, v.2=R, v.3=padding.
        let quad = unsafe { vld4q_u8(src_u8.add(i * 4)) };
        let rgb = uint8x16x3_t(quad.2, quad.1, quad.0); // R, G, B
        unsafe { vst3q_u8(dst_p.add(i * 3), rgb) };
        i += 16;
    }
    // Scalar tail for the last <16 pixels.
    while i < n {
        let px = unsafe { *src.as_ptr().add(i) };
        unsafe {
            *dst_p.add(i * 3) = ((px >> 16) & 0xFF) as u8;
            *dst_p.add(i * 3 + 1) = ((px >> 8) & 0xFF) as u8;
            *dst_p.add(i * 3 + 2) = (px & 0xFF) as u8;
        }
        i += 1;
    }
}

/// Interior-mutability wrapper around a `Worker`. The inner
/// `UnsafeCell` is not `Sync`; we assert `Sync` manually because all
/// access is discipline-based (see the `Pool` `Send + Sync` SAFETY
/// comment below).
///
/// `#[repr(align(128))]` (M4's cache line size) rather than the
/// default 8-byte alignment `Worker`'s fields would otherwise imply.
/// `Vec<WorkerCell>` (`Pool::workers`) is the one contiguous,
/// worker_id-indexed allocation where every element is written by a
/// DIFFERENT rayon thread on every `step_all`/`reset_all` call — and
/// each `step()` dirties close to the whole `Worker` (RAM writes,
/// PPU/OAM/mapper state, CPU registers), not just one field. Without
/// this, `Worker`'s natural 8-byte alignment lets consecutive
/// elements land with only an 8-byte-granular boundary, so the tail
/// bytes of worker[i] and the head bytes of worker[i+1] can share a
/// physical cache line — two different threads then write that same
/// line every step, forcing MESI invalidation traffic back and forth
/// between cores on every boundary pair. Same failure shape as the
/// `ASM_HITS` regression (see `nes.rs`'s note near the ASM dispatch
/// path), just at the worker-array boundary instead of a single
/// global counter. `#[repr(align(128))]` pads `WorkerCell`'s size up
/// to a 128-byte multiple and 128-byte-aligns the whole `Vec`'s
/// backing allocation, so no two workers' elements can ever share a
/// line — a few dozen bytes of padding per multi-KB `Worker`, not a
/// logic change. (Dropping `#[repr(transparent)]`: nothing here
/// transmutes or pointer-casts `WorkerCell` to `UnsafeCell<Worker>`
/// — every access goes through `.0.get()` / `WorkerCell::new()` — so
/// the ABI-transparency guarantee wasn't load-bearing, only cosmetic.)
#[repr(align(128))]
struct WorkerCell(UnsafeCell<Worker>);

// SAFETY: sharing `&WorkerCell` across threads is sound because the
// only way to mint a `&mut Worker` from it is `worker_mut`, whose
// callers (rayon `par_iter` at unique indices + sequential Python
// trainer thread) uphold unique-owner access. See `Pool` SAFETY note.
unsafe impl Sync for WorkerCell {}

impl WorkerCell {
    #[inline(always)]
    fn new(w: Worker) -> Self {
        Self(UnsafeCell::new(w))
    }
}

/// Obtain an exclusive `&mut Worker` from its `WorkerCell` wrapper.
///
/// SAFETY: caller must ensure exclusive access. Rayon's `par_iter`
/// provides this for per-index access across workers. For sequential
/// access (reset_all, save_worker_state, set_worker_pace, drain_audio,
/// set_batched_render_mode, load_worker_state, load_start_state), the
/// trainer is responsible for not calling these concurrently with
/// step_all — which holds in the current architecture (all Pool methods
/// run on the Python trainer thread, never overlapping).
#[inline(always)]
unsafe fn worker_mut(cell: &WorkerCell) -> &mut Worker {
    &mut *cell.0.get()
}

struct AudioVecSink<'a> {
    buf: &'a mut Vec<i16>,
}
impl AudioSink for AudioVecSink<'_> {
    fn write_sample(&mut self, s: f32) {
        self.buf.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    fn samples_written(&self) -> usize {
        self.buf.len()
    }
}

/// Python-facing worker pool. Replaces `ParallelPool` from
/// `src/emulation/parallel_pool.py` with an in-process, rayon-parallel,
/// zero-IPC implementation.
///
/// Usage mirrors the Python pool closely so the trainer can switch
/// via env_spec resolution:
///
/// ```python
/// pool = nes_core.Pool(
///     rom_path='roms/zelda.nes',
///     num_workers=16,
///     frame_skip=4,
///     start_state_path=None,
/// )
/// pool.reset_all()  # returns list of (frame, ram, done)
/// pool.step_all([0]*16)
/// pool.shutdown()
/// ```
// Send + Sync are required so Python can pass a Pool ref across threads
// (the async-pipeline optimization calls step_all from a worker thread
// concurrent with a main-thread MPS forward) and so the rayon closures
// inside step_all/reset_all can capture `&self.workers` across worker
// threads. `WorkerCell` (an `UnsafeCell<Worker>` newtype) has a manual
// `unsafe impl Sync` so `&WorkerCell` is `Send`; see `WorkerCell` above.
//
// SAFETY: The only parallel access into `self.workers` happens inside
// the rayon `par_iter().enumerate()` closures in `step_all` and
// `reset_all` (plus `collect_results`'s `par_iter().map()`). Each
// rayon task receives a unique index / unique `&WorkerCell` and uses
// `unsafe { worker_mut(cell) }` to mint an exclusive `&mut Worker`
// for just that element — indices are not shared, so no two tasks
// ever alias the same Worker. All other methods
// (`save_worker_state`, `set_worker_pace`, `set_worker_audio`,
// `drain_audio`,
// `load_worker_state`, `set_batched_render_mode`, `load_start_state`)
// run sequentially on the Python trainer thread and never overlap
// with an in-flight `step_all`/`reset_all` — Python holds the GIL
// across each of our PyO3 entry points, and the async-pipeline
// optimization releases it only inside `py.allow_threads(|| ...)`
// for the duration of a single step. Therefore unique-owner access
// is upheld at every point of use.
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

#[pyclass(module = "nes_core")]
pub struct Pool {
    workers: Vec<WorkerCell>,
    num_workers: usize,
    frame_skip: u32,
    /// When true, step_all returns a 1×1×3 dummy frame instead of
    /// the full 256×240×3 RGB frame — saves the 184 KB unpack + Vec
    /// alloc per worker per step (~3 MB/step × 16 workers × 2200 sps
    /// = ~6 GB/s of malloc/memcpy). Training paths only use
    /// `preprocessed` (the 84×84 grayscale), so the full frame is
    /// dead weight unless a GUI frame_sink is attached.
    headless: std::sync::atomic::AtomicBool,
    /// When true, `preprocessed` is emitted as IEEE 754 float16 bits
    /// normalized to [0, 1] instead of uint8 [0, 255]. The Python side
    /// receives the buffer as `np.ndarray(dtype=np.float16)` and can
    /// skip the `.float().div_(255.0)` cast/divide that otherwise
    /// costs a GPU kernel launch per batch on unified-memory M-series
    /// hardware. Off by default — enabling it changes numerical
    /// semantics (uint8 → fp16-rounded normalized floats), so the
    /// caller opts in via `set_preprocess_f16(true)`.
    preprocess_f16: std::sync::atomic::AtomicBool,
    /// When true, skip the 84×84 grayscale+resize preprocess entirely
    /// in headless mode. Tile-encoder games (e.g. SMB tile mode) read
    /// only `ram_snapshot` and never consume `preprocessed`, so the
    /// per-worker gray+resize kernel is pure wasted CPU. With it off the
    /// preprocess runs every step (60 workers × 1024 steps/iter). The
    /// caller opts in via `set_skip_preprocess(true)` only when the
    /// policy reads RAM, not pixels. `preprocessed` is then returned as
    /// a 0-length (0, 0) sentinel array (the adapter substitutes a
    /// shared zeros((84, 84)) singleton; the trainer ignores it).
    skip_preprocess: std::sync::atomic::AtomicBool,
    /// When true (default), every worker closure in `step_all` /
    /// `reset_all` is wrapped in `catch_unwind(AssertUnwindSafe(...))`
    /// so a panic in one worker marks that worker dead without taking
    /// the rayon thread down. That wrap costs ~50–200 ns on Apple
    /// Silicon; at 16 workers × 1700+ sps = ~27K calls/s it's
    /// 0.5–1% of throughput.
    ///
    /// When false, the wrap is bypassed and worker bodies run
    /// directly on the rayon thread. This is the "production
    /// training" fast path: the trainer takes responsibility for
    /// stable ROMs/states, and in exchange a panic will unwind past
    /// the rayon worker (unsound — may abort the process). Default
    /// stays `true` so existing behaviour is preserved unless the
    /// trainer explicitly opts in via `set_panic_isolation(False)`.
    panic_isolation: std::sync::atomic::AtomicBool,
    /// Per-worker "episode done" mask. The trainer sets this once a
    /// genome's episode terminates so subsequent `step_all` calls
    /// short-circuit that worker — no NES emulation, no preprocess
    /// kernel, just the cached final state. With 30 workers and a
    /// genome that dies at step 50 of 1500, this saves ~5800 NES
    /// frames of dead-time emulation per episode. Cleared on
    /// `reset_all` so the next episode starts everyone fresh.
    ///
    /// Cache-alignment audit note: this IS a contiguous,
    /// worker_id-indexed array touched by every rayon task on every
    /// `step_all` (the `load` in the `step_all` closure), so it looks
    /// like the `ASM_HITS` pattern at first glance. It isn't one:
    /// every access inside the parallel closure is a read
    /// (`Ordering::Acquire` load), and concurrent reads of a shared
    /// cache line don't invalidate each other under MESI — only
    /// writes do. The only writers (`store` in `set_worker_done` and
    /// the `reset_all` clear loop) run sequentially on the Python
    /// trainer thread and, per the `Pool` SAFETY note, never overlap
    /// an in-flight `step_all`. No concurrent writer ⇒ no per-step
    /// cache-line ping-pong ⇒ left un-padded on purpose. Padding
    /// `AtomicBool` entries to 128B each would cost real memory
    /// (128× vs the current 1 byte per worker) to guard against a
    /// contention pattern that cannot occur here.
    worker_done: Vec<std::sync::atomic::AtomicBool>,
    /// Pool-level pacing speed multiplier as raw IEEE 754 f64 bits
    /// (atomics have no f64 flavor). 1.0 = realtime (frame_skip / 60 s
    /// per step); 2.0 = double speed; clamped to [0.25, 16.0] by
    /// `set_pace_multiplier`. Only scales the pool pacing budget —
    /// emulation semantics are untouched.
    pace_multiplier_bits: std::sync::atomic::AtomicU64,
    /// When true (default), paced spectator workers render only the
    /// FINAL sub-frame of each step — the only frame that ships to
    /// Python — via the same `skip_render` mechanism unpaced workers
    /// already use (emulation-neutral; see tests/skip_render_parity.rs).
    /// Set false to restore the historical render-every-sub-frame
    /// behaviour for paced workers.
    spectator_subframe_skip: std::sync::atomic::AtomicBool,
    /// Absolute wall-clock target for the next paced `step_all`
    /// completion. Pool-level replacement for the old per-worker
    /// `pace_next_target`: ONE sleep on the `step_all` calling thread
    /// after the rayon join, instead of N sleeps that would each park
    /// a rayon thread (hard ~thread-count ceiling on paced workers).
    /// Mutex is uncontended — locked once per `step_all` on the
    /// (sequential) trainer thread.
    pool_pace_next_target: std::sync::Mutex<Option<std::time::Instant>>,
}

#[pymethods]
impl Pool {
    #[new]
    #[pyo3(signature = (
        rom_path, num_workers = 16, frame_skip = 4, start_state_path = None,
    ))]
    fn new(
        rom_path: std::path::PathBuf,
        num_workers: usize,
        frame_skip: u32,
        start_state_path: Option<std::path::PathBuf>,
    ) -> PyResult<Self> {
        if num_workers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_workers must be >= 1",
            ));
        }
        init_rayon_pool_for_apple_silicon(num_workers);
        let rom_bytes = std::fs::read(&rom_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to read ROM {}: {e}",
                rom_path.display()
            ))
        })?;

        // Each worker owns an independent Cartridge + Nes. Parse the
        // iNES bytes once, clone the cart per worker — cheap, since
        // PRG/CHR buffers are deterministic per ROM and the per-worker
        // cart only mutates its own mirroring/CHR-RAM state.
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let cart = Cartridge::load(&mut std::io::Cursor::new(&rom_bytes))
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "failed to parse iNES: {e:?}"
                    ))
                })?;
            let mapper_num = cart.mapper;
            let worker = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Worker::new(cart)
            })).map_err(|e| {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "mapper {mapper_num} initialization panicked: {msg}"
                ))
            })?;
            workers.push(WorkerCell::new(worker));
        }

        let worker_done = (0..num_workers)
            .map(|_| std::sync::atomic::AtomicBool::new(false))
            .collect();
        let pool = Self {
            workers,
            num_workers,
            frame_skip: frame_skip.max(1),
            headless: std::sync::atomic::AtomicBool::new(false),
            preprocess_f16: std::sync::atomic::AtomicBool::new(false),
            skip_preprocess: std::sync::atomic::AtomicBool::new(false),
            panic_isolation: std::sync::atomic::AtomicBool::new(true),
            worker_done,
            pace_multiplier_bits: std::sync::atomic::AtomicU64::new(1.0f64.to_bits()),
            spectator_subframe_skip: std::sync::atomic::AtomicBool::new(true),
            pool_pace_next_target: std::sync::Mutex::new(None),
        };

        if let Some(p) = start_state_path {
            pool.load_start_state(&p)?;
        }
        Ok(pool)
    }

    /// Reset every worker. Returns a list of per-worker
    /// `(frame, ram, done)` tuples. Parallel across rayon's pool.
    /// Worker panics are caught and that worker is marked dead
    /// (remains in the pool but short-circuits to zero-frame on
    /// future calls), so one bad worker can't poison the other 15.
    fn reset_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Load the flag once outside the rayon closure — per-closure
        // atomic loads would reintroduce the overhead this gate exists
        // to avoid. Acquire pairs with the Release store in
        // `set_panic_isolation`.
        let panic_isolation = self
            .panic_isolation
            .load(std::sync::atomic::Ordering::Acquire);
        // Clear the per-worker done mask so the next episode starts
        // with everyone active. Without this, a worker that finished
        // last episode would short-circuit forever.
        for done in &self.worker_done {
            done.store(false, std::sync::atomic::Ordering::Release);
        }
        // Re-anchor pool pacing — mirrors the old per-worker
        // `pace_next_target = None` in `Worker::reset` so the first
        // step after a reset never sleeps against a stale target.
        self.clear_pool_pace_target();
        py.allow_threads(|| {
            // SAFETY: rayon's `par_iter().enumerate()` gives each task
            // a unique index; no two tasks touch the same worker cell.
            self.workers.par_iter().enumerate().for_each(|(idx, cell)| {
                let w = unsafe { worker_mut(cell) };
                // A dead worker (panicked on a prior step/reset) gets a chance
                // to REVIVE here: clearing the flag and resetting to the clean
                // start state recovers from a transient / corrupt-state panic.
                // If reset panics again it is re-marked dead below. Without
                // this, a single edge-case panic left the env emitting
                // zero-frames + done=false into PPO for the rest of the run,
                // silently poisoning the policy on a permanently-dead cohort.
                w.dead = false;
                let res: Result<(), Box<dyn std::any::Any + Send>> = if panic_isolation {
                    catch_unwind(AssertUnwindSafe(|| w.reset()))
                } else {
                    // Direct call. Any panic will unwind past the
                    // rayon worker, which is unsound — but the
                    // trainer takes responsibility for stable
                    // ROMs/states when this flag is off.
                    Ok(w.reset())
                };
                if res.is_err() {
                    eprintln!("[nes_core::Pool] worker {idx} panicked on reset; marking dead");
                    w.dead = true;
                }
            });
        });
        self.collect_results(py)
    }

    /// Step every worker with its respective action bitmask. Returns
    /// the same shape as `reset_all` — `num_workers` tuples of
    /// `(frame, ram, done)`.
    ///
    /// Accepts either a numpy `uint8` 1-D array (preferred, zero-copy
    /// O(1) FFI) or a Python list/sequence of ints (fallback, one
    /// unbox + alloc per element). Training hot path passes the numpy
    /// array to shave ~4000 per-generation list unboxings.
    fn step_all<'py>(
        &self,
        py: Python<'py>,
        actions: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Zero-copy fast path: numpy uint8 array. Borrows the underlying
        // buffer for the duration of this call — no Vec alloc, no
        // per-element PyLong unbox. `to_vec()` copies into a local
        // owned buffer so we can drop the readonly guard before the
        // `py.allow_threads` block (numpy guards touch the GIL on drop).
        // The copy is still cheap — 16 bytes — and keeps the rayon
        // closure from having to hold onto the numpy guard.
        let actions_vec: Vec<u8> = if let Ok(arr) =
            actions.extract::<numpy::PyReadonlyArray1<'py, u8>>()
        {
            arr.as_slice()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "actions numpy array not contiguous: {e}"
                    ))
                })?
                .to_vec()
        } else {
            // Fallback: list / sequence of ints. Keeps legacy callers
            // (tests, bench scripts passing `[0]*16`) working.
            actions.extract::<Vec<u8>>()?
        };
        if actions_vec.len() != self.num_workers {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "actions length {} != num_workers {}",
                actions_vec.len(),
                self.num_workers
            )));
        }
        // The heavy lifting (rayon dispatch + pool pacing) lives in
        // `step_all_native` so cargo tests can drive it without a
        // Python interpreter. GIL released for the whole native step —
        // including the pacing sleep — so Qt / trainer-control threads
        // aren't starved while a spectator pool paces at realtime.
        let collected = py.allow_threads(|| self.step_all_native(&actions_vec));
        self.build_result_list(py, collected)
    }

    /// Dump a worker's current 32-byte PPU palette RAM for diagnostic
    /// purposes. Returns the raw_bytes() buffer, which contains the
    /// game-written palette indices at $3F00-$3F1F (before mirror
    /// resolution). Added 2026-04-22 to diagnose the Contra palette
    /// flicker — if palette RAM changes per frame that explains the
    /// visible flicker; if it's stable but rendered pixels differ,
    /// the render path is mis-mapping indices to XRGB colors.
    fn palette_ram<'py>(
        &self,
        py: Python<'py>,
        worker_id: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if worker_id >= self.num_workers {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "worker_id out of range",
            ));
        }
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        let bytes = *w.nes.ppu.mem.palette_ram.raw_bytes();
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Drain accumulated audio samples for a specific worker. Used by
    /// the audio mixer's live-PCM path. Safe to call on any worker.
    fn drain_audio<'py>(
        &self,
        py: Python<'py>,
        worker_id: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyByteArray>> {
        if worker_id >= self.num_workers {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "worker_id out of range",
            ));
        }
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        let samples: Vec<i16> = std::mem::take(&mut w.audio);
        // int16 samples as raw bytes — caller reinterprets via
        // `numpy.frombuffer(..., dtype=np.int16)` or equivalent.
        let byte_len = samples.len() * 2;
        let bytes_ptr = samples.as_ptr() as *const u8;
        // Safety: samples is a Vec<i16>, layout-compatible with a
        // byte buffer of 2*len bytes; we copy into a PyByteArray and
        // drop the Vec at scope end.
        let slice = unsafe { std::slice::from_raw_parts(bytes_ptr, byte_len) };
        Ok(pyo3::types::PyByteArray::new_bound(py, slice))
    }

    /// Toggle realtime pacing on a specific worker — used for
    /// GUI-visible spectator workers so audio production matches
    /// playback rate. Other workers stay unpaced. Pacing is enforced
    /// at pool level (one sleep per `step_all` on the calling thread),
    /// so any number of workers can be paced without parking rayon
    /// threads.
    ///
    /// Back-compat side effect: also flips this worker's audio output
    /// to match (`set_audio_output_enabled(on)`), preserving the
    /// historical pace↔audio welding. To control audio independently,
    /// call `set_worker_audio` — between the two, the LAST call wins
    /// for the audio-enable state.
    fn set_worker_pace(&self, worker_id: usize, on: bool) {
        if worker_id >= self.num_workers {
            return;
        }
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        if w.realtime_pace != on {
            w.realtime_pace = on;
            // Flipping pace ↔ audio: the paced worker is one whose
            // audio the mixer plays, so it should do the sample-gen +
            // mix work. Re-align audio output to match. (Overridable
            // afterwards via `set_worker_audio` — last call wins.)
            w.nes.set_audio_output_enabled(on);
            // Re-anchor pool pacing on any membership change — mirrors
            // the old per-worker `pace_next_target = None` on flip, so
            // the next step paces from "now" instead of a stale target.
            self.clear_pool_pace_target();
        }
    }

    /// Enable/disable APU sample output for a specific worker WITHOUT
    /// touching its realtime pacing. Lets the mixer's Mode::All hear
    /// many workers while only the pool-level pacer controls speed.
    /// Channel timers + frame counter still tick when off, so IRQ/DMC
    /// emulation semantics are bit-identical either way.
    ///
    /// Interaction with `set_worker_pace`: that method also flips the
    /// same audio-enable flag as a back-compat side effect. Between
    /// the two, the LAST call wins. Out-of-range ids are ignored.
    fn set_worker_audio(&self, worker_id: usize, on: bool) {
        if worker_id >= self.num_workers {
            return;
        }
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        w.nes.set_audio_output_enabled(on);
    }

    /// Set the pool pacing speed multiplier. 1.0 (default) = realtime:
    /// each paced `step_all` targets `frame_skip / 60` seconds of wall
    /// clock. 2.0 = double speed, 0.5 = half speed. Clamped to
    /// [0.25, 16.0]; non-finite input resets to 1.0. Only scales the
    /// pacing budget — emulation semantics are untouched, and unpaced
    /// pools ignore it entirely.
    fn set_pace_multiplier(&self, mult: f64) {
        let m = if mult.is_finite() {
            mult.clamp(0.25, 16.0)
        } else {
            1.0
        };
        self.pace_multiplier_bits
            .store(m.to_bits(), std::sync::atomic::Ordering::Release);
    }

    /// Toggle final-frame-only rendering for paced spectator workers
    /// (default ON). When on, a paced worker renders only the LAST
    /// sub-frame of each step — the only frame shipped to Python —
    /// using the same `skip_render` mechanism the unpaced training
    /// path has always used (emulation-neutral: RAM / NMI / IRQ /
    /// sprite-0 are bit-identical; see tests/skip_render_parity.rs).
    /// Set false to restore the historical render-every-sub-frame
    /// behaviour.
    fn set_spectator_subframe_skip(&self, on: bool) {
        self.spectator_subframe_skip
            .store(on, std::sync::atomic::Ordering::Release);
    }

    /// Enable headless mode — `step_all` returns a dummy 1×1×3
    /// frame instead of the full 256×240×3 RGB. Training-only
    /// code paths use `preprocessed` (84×84 grayscale) and don't
    /// touch `frame`, so this is pure perf win (~6 GB/s of RGB
    /// unpack/alloc saved). GUI callers with a live frame_sink
    /// should keep this off (default).
    fn set_headless(&self, on: bool) {
        self.headless
            .store(on, std::sync::atomic::Ordering::Release);
    }

    /// Mark a worker as episode-done so subsequent `step_all` calls
    /// skip its NES emulation and preprocess kernel. The worker's
    /// previous frame and RAM are returned unchanged. Cleared
    /// automatically on `reset_all`. Idempotent — set on the same
    /// worker twice in a row is a no-op.
    ///
    /// Out-of-range indices are silently ignored (matches the
    /// trainer's "best effort" pattern for late-binding done
    /// genomes from a closing episode loop).
    fn set_worker_done(&self, worker_id: usize, done: bool) {
        if let Some(flag) = self.worker_done.get(worker_id) {
            flag.store(done, std::sync::atomic::Ordering::Release);
        }
    }

    /// Peak SMB world-x reached during the most recent `step_all`
    /// call, per worker. The trainer overrides the reward function's
    /// x view with this value so transient mid-frame_skip peaks
    /// (e.g. Mario reaches x=1810 mid-jump then dies and ends the
    /// step at x=1700) get credited toward `max_x_visited` and
    /// checkpoint detection. Reading is sequential on the Python
    /// thread after `step_all` completes — rayon's join introduces
    /// the memory barrier we need so writes from the worker threads
    /// are visible.
    fn peek_max_x_per_worker(&self) -> Vec<u32> {
        self.workers
            .iter()
            .map(|cell| {
                // Safety: caller is the Python trainer thread,
                // which serializes with `step_all`/`reset_all` (both
                // join their rayon tasks before returning). No other
                // mutator runs between those returning and this
                // read.
                let w = unsafe { &*cell.0.get() };
                w.last_max_x
            })
            .collect()
    }

    /// Toggle IEEE 754 half-precision emission on the preprocess path.
    /// When on, `step_all`/`reset_all` return `preprocessed` as a
    /// `numpy.float16` (84, 84) array already normalized to [0, 1].
    /// The trainer can then skip the `torch.from_numpy(...).float().div_(255.0)`
    /// cast/divide on MPS — a per-batch GPU kernel launch is eliminated.
    /// Off by default to preserve numerical semantics for existing
    /// training runs.
    fn set_preprocess_f16(&self, on: bool) {
        self.preprocess_f16
            .store(on, std::sync::atomic::Ordering::Release);
    }

    /// Skip the 84×84 grayscale+resize preprocess in headless mode.
    /// Set true for tile-encoder games whose policy reads `ram_snapshot`
    /// and never consumes `preprocessed` — the kernel is wasted CPU
    /// otherwise (60 workers × 1024 steps/iter). `preprocessed` is then
    /// returned as a 0-length (0, 0) sentinel array; the Python adapter
    /// substitutes a shared zeros((84, 84)) singleton.
    fn set_skip_preprocess(&self, on: bool) {
        self.skip_preprocess
            .store(on, std::sync::atomic::Ordering::Release);
    }

    /// Toggle per-worker panic isolation in `step_all` / `reset_all`.
    /// Default on — every worker closure is wrapped in
    /// `catch_unwind` so a panic in one worker marks it dead
    /// without taking the rayon thread down.
    ///
    /// When off, the wrap is bypassed: the trainer takes
    /// responsibility for stable ROMs/states and in exchange gets
    /// back the ~50–200 ns per-call overhead (0.5–1% of throughput
    /// at 16 workers × 1700+ sps). A panic in this mode will
    /// unwind past the rayon worker — unsound, may abort the
    /// process. Use only in production training phases with
    /// validated ROMs.
    fn set_panic_isolation(&self, enabled: bool) -> PyResult<()> {
        self.panic_isolation
            .store(enabled, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Runtime opt-out from the AArch64 ASM CPU on all Pool
    /// workers. asm_cpu wins ~15% on single-env headless cold-boot
    /// but at parallel scale (12 workers) the larger ASM dispatch
    /// code may compete for L1i; experimental toggle for benching.
    /// Off by default (asm_cpu stays on, matches GUI / single-env).
    fn set_disable_asm_cpu(&self, on: bool) -> PyResult<()> {
        // SAFETY: called sequentially from Python — never overlaps
        // with an in-flight step_all/reset_all rayon dispatch.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.disable_asm_cpu = on;
        }
        Ok(())
    }

    /// Opt-in ASM bulk-step budget (CPU cycles per ASM invocation)
    /// on all Pool workers. Only the batch-safe mappers honor it —
    /// MMC1 (Zelda, Metroid, Kid Icarus, Mega Man 2, …) and UxROM
    /// (Contra, Castlevania, DuckTales, …); every other mapper
    /// ignores the call. Default budget is 1 (one instruction per
    /// invocation) — the shipped path, so leaving this untouched is
    /// parity-safe by construction.
    /// Raising it amortizes the per-invocation ASM setup (SystemBus
    /// construction, TLS tick install, state init/writeback, FFI,
    /// NMI prediction) over 2-5 instructions, but coarsens the tick
    /// granularity that timed $2002 polling (sprite-0 waits) and
    /// mid-batch APU frame/DMC IRQ service can observe. Enable
    /// per-game ONLY after the lockstep + Mesen-oracle + parity
    /// gate passes at that budget. Measured ladder: 1 -> 8 -> 16.
    fn set_asm_bulk_cycles(&self, cycles: i64) -> PyResult<()> {
        if !(1..=16).contains(&cycles) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("asm_bulk_cycles must be in 1..=16, got {cycles}"),
            ));
        }
        // SAFETY: called sequentially from Python — never overlaps
        // with an in-flight step_all/reset_all rayon dispatch.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.set_asm_bulk_cycles_override(cycles);
        }
        Ok(())
    }

    /// Hardware-true PPU-register read timing on every worker —
    /// single-env mirror is `NESEnvironment::set_hw_mmio_read_timing`;
    /// see the Cpu field doc for the ground-truth receipt. Default
    /// OFF (legacy LaiNES-parity early commit).
    fn set_hw_mmio_read_timing(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.cpu.hw_mmio_read_timing = on;
        }
    }

    /// Hardware-true boot alignment on every worker — single-env
    /// mirror is `NESEnvironment::set_hw_reset_alignment`; see the
    /// `Nes` field doc. Takes effect on each worker's next reset.
    /// Default OFF (legacy boot accounting).
    fn set_hw_reset_alignment(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.hw_reset_alignment = on;
        }
    }

    /// Hardware-true DMC DMA stall on every worker — single-env
    /// mirror is `NESEnvironment::set_hw_dmc_stall_timing`; see the
    /// `Apu` field doc. Default OFF (legacy flat-4 stall).
    fn set_hw_dmc_stall_timing(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.apu.hw_dmc_stall_timing = on;
        }
    }

    /// Hardware-true frame anchoring on every worker — single-env
    /// mirror is `NESEnvironment::set_hw_frame_anchor`; see the
    /// `Worker` field doc. Each `step()` sub-frame then advances to a
    /// real PPU frame boundary instead of a fixed 29,781-cycle block,
    /// which is what a tape replay needs to line up with Mesen's
    /// inputPolled schedule. Config, not state: survives
    /// `load_worker_state`. Default OFF.
    fn set_hw_frame_anchor(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.hw_frame_anchor = on;
            w.frame_cycle_target = None;
        }
    }

    /// Hardware-true NMI service timing on every worker — single-env
    /// mirror is `NESEnvironment::set_hw_nmi_poll_timing`; see the
    /// Cpu field doc. Disables ASM/bulk fast paths while on.
    fn set_hw_nmi_poll_timing(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.cpu.hw_nmi_poll_timing = on;
        }
    }

    /// Hardware-true NMI-line sample phase (φ2 latch) on every worker
    /// — single-env mirror is
    /// `NESEnvironment::set_hw_nmi_subcycle_phase`; see the `Nes`
    /// field doc. Disables ASM/bulk fast paths while on. Default OFF.
    fn set_hw_nmi_subcycle_phase(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.hw_nmi_subcycle_phase = on;
        }
    }

    /// Hardware-true `$2002`/vblank read race on every worker —
    /// single-env mirror is
    /// `NESEnvironment::set_hw_vblank_read_race`; see the `Nes` field
    /// doc for the modeled window. Default OFF. Refuses an enable whose
    /// bus phase would make the race inert, for the same reason the
    /// single-env setter does; the prerequisites are checked on every
    /// worker BEFORE any worker is mutated, so a refusal leaves the pool
    /// uniformly configured.
    fn set_hw_vblank_read_race(&self, on: bool) -> PyResult<()> {
        // SAFETY: as above — sequential from Python.
        if on {
            for (i, cell) in self.workers.iter().enumerate() {
                let w = unsafe { worker_mut(cell) };
                if !w.nes.vblank_read_race_prereqs_met() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "hw_vblank_read_race needs the sub-cycle read phase, \
                         missing on worker {i}: enable hw_event_ppu + \
                         hw_mmio_read_timing and set ppu_read_dot_offset=0 \
                         FIRST (see NESEnvironment.set_hw_vblank_read_race)"
                    )));
                }
            }
        }
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.set_hw_vblank_read_race(on);
        }
        Ok(())
    }

    /// Do the bus-phase prerequisites for `set_hw_vblank_read_race` hold
    /// on every worker? Single-env mirror is
    /// `NESEnvironment::vblank_read_race_prereqs_met`.
    fn vblank_read_race_prereqs_met(&self) -> bool {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            if !w.nes.vblank_read_race_prereqs_met() {
                return false;
            }
        }
        true
    }

    /// Hardware-true PPU-register write timing on every worker —
    /// single-env mirror is `NESEnvironment::set_hw_mmio_write_timing`.
    fn set_hw_mmio_write_timing(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.cpu.hw_mmio_write_timing = on;
        }
    }

    /// Event-driven PPU catch-up on every worker — single-env mirror is
    /// `NESEnvironment::set_hw_event_ppu`; see the `Nes` field doc.
    /// Observably identical to the legacy inline interleave. Default OFF.
    fn set_hw_event_ppu(&self, on: bool) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.hw_event_ppu = on;
        }
    }

    /// Sub-cycle bus-access dot offset for deferred PPU-register READS on
    /// every worker — single-env mirror is
    /// `NESEnvironment::set_ppu_read_dot_offset` (event-driven PPU
    /// Stage 3). 0..=2 (clamped); default 2 = byte-identical.
    fn set_ppu_read_dot_offset(&self, offset: u8) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.ppu_read_dot_offset = offset.min(2);
        }
    }

    /// Write-side mirror of `set_ppu_read_dot_offset` on every worker —
    /// single-env mirror is `NESEnvironment::set_ppu_write_dot_offset`.
    fn set_ppu_write_dot_offset(&self, offset: u8) {
        // SAFETY: as above — sequential from Python.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.ppu_write_dot_offset = offset.min(2);
        }
    }

    /// Enable the batched PPU renderer on every worker. Trades
    /// (rare) single-frame stale-row artifacts on mid-scanline-
    /// writing games for a measured +10-27% single-env throughput.
    /// Accepts "off", "verify", "replace".
    fn set_batched_render_mode(&self, mode: &str) -> PyResult<()> {
        let bmode = match mode {
            "off" => crate::ppu::BatchedRenderMode::Off,
            "verify" => crate::ppu::BatchedRenderMode::Verify,
            "replace" => crate::ppu::BatchedRenderMode::Replace,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("unknown batched_render_mode: {mode}"),
            )),
        };
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            w.nes.ppu.set_batched_render_mode(bmode);
        }
        Ok(())
    }

    /// Save a specific worker's NES state as an opaque bytes blob.
    /// Matches `src/emulation/parallel_pool.py::save_worker_state`
    /// for drop-in compatibility with the trainer's auto-curriculum
    /// snapshot path.
    fn save_worker_state<'py>(
        &self,
        py: Python<'py>,
        worker_id: usize,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        if worker_id >= self.num_workers {
            return Ok(None);
        }
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        let state = w.nes.get_state();
        match bincode::serialize(&state) {
            Ok(body) => {
                let mut out = Vec::with_capacity(POOL_STATE_MAGIC.len() + body.len());
                out.extend_from_slice(POOL_STATE_MAGIC);
                out.extend_from_slice(&body);
                Ok(Some(PyBytes::new_bound(py, &out)))
            }
            Err(_) => Ok(None),
        }
    }

    /// Load an opaque bytes blob into a specific worker. Counterpart
    /// to `save_worker_state`; use for start-state broadcast or
    /// curriculum-driven restore. Accepts both the versioned
    /// `NCST\x01` prefix and legacy naked-bincode blobs.
    fn load_worker_state(
        &self,
        worker_id: usize,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        if worker_id >= self.num_workers {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "worker_id out of range",
            ));
        }
        let raw = data.as_bytes();
        let body: &[u8] = if raw.starts_with(POOL_STATE_MAGIC) {
            // Strip repeated NCST prefixes, matching the loader
            // semantics in `load_start_state` + `python.rs::load_state`.
            // A historical play_window.py bug wrote double-prefix
            // files; this loader also tolerates them so Pool-driven
            // curriculum restores can consume the same on-disk files.
            let mut body = raw;
            while body.starts_with(POOL_STATE_MAGIC) {
                body = &body[POOL_STATE_MAGIC.len()..];
            }
            body
        } else if raw.len() >= 4 && &raw[..4] == b"NCST" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "load_worker_state: unsupported NCST version byte {:#04x}",
                raw[4],
            )));
        } else {
            raw
        };
        let state: crate::nes::State = bincode::deserialize(body).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "load_worker_state deserialize failed: {e}"
            ))
        })?;
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        // Guard apply_state: a bincode-valid but mapper-mismatched or
        // invariant-violating state (e.g. a stale auto-curriculum snapshot
        // after a profile swap) panics ("Invalid mapper state enum
        // variant"). Without catch_unwind that panic crosses the PyO3
        // boundary as an uncatchable PanicException and takes down the
        // entire training run. Mirror load_start_state: return a catchable
        // PyValueError so the trainer can degrade this worker to a cold
        // boot instead of the whole overnight run dying.
        let res = catch_unwind(AssertUnwindSafe(|| {
            w.nes.apply_state(&state);
        }));
        if let Err(panic_payload) = res {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "load_worker_state: state is not compatible with the current \
                 ROM's mapper — apply_state panicked: {msg}. This usually \
                 means a stale/mismatched curriculum state; the trainer falls \
                 back to a cold boot for this worker."
            )));
        }
        w.audio.clear();
        // CRITICAL: clear frame_cycle_target. Without this, the next
        // `advance_one_frame` computes `target = stale_value + 29781`
        // where stale_value was set by the worker's previous frame
        // (typically ~30k from cold-boot reset). The loaded state's
        // `nes.cycles` is far larger (millions), so `cycles >> target`
        // and BOTH `while cycles < target` loops are immediately false
        // — the NES never ticks, PPU never renders, and the worker
        // appears frozen forever from the trainer's perspective. The
        // existing `reset()` method clears this for the same reason
        // (line ~175). Without this fix, save-state-based curriculum
        // (load_worker_state into mid-game state) is structurally
        // broken: RAM reads return the saved values but the screen
        // shows cold-boot pixels and the game can't advance.
        w.frame_cycle_target = None;
        Ok(())
    }

    #[getter]
    fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Count of workers currently marked dead (panicked and not yet
    /// revived). `reset_all` attempts to revive them each iteration, so a
    /// persistently-nonzero value flags a worker that keeps re-panicking —
    /// the trainer can log/act on it instead of training on a dead cohort.
    #[getter]
    fn num_dead(&self) -> usize {
        let mut n = 0;
        for cell in &self.workers {
            // Sequential getter — never overlaps step_all/reset_all.
            if unsafe { worker_mut(cell).dead } {
                n += 1;
            }
        }
        n
    }

    #[getter]
    fn frame_skip(&self) -> u32 {
        self.frame_skip
    }

    /// No-op; kept for surface parity with ParallelPool. Workers are
    /// dropped when the pool goes out of scope.
    fn shutdown(&self) {}
}

impl Pool {
    /// Rayon-parallel step across all workers + pool-level pacing.
    /// Python-free core of `step_all` — factored out so cargo tests
    /// can exercise stepping/pacing without a live interpreter. Must
    /// be called with the GIL released (`py.allow_threads`) from the
    /// Python entry point; the pacing sleep happens here.
    fn step_all_native(&self, actions_vec: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)> {
        debug_assert_eq!(actions_vec.len(), self.num_workers);
        let frame_skip = self.frame_skip;
        let pp_dim = crate::preprocess::PP_SIZE;
        let headless = self.headless.load(std::sync::atomic::Ordering::Acquire);
        let f16_mode = self.preprocess_f16.load(std::sync::atomic::Ordering::Acquire);
        let skip_preprocess = self.skip_preprocess.load(std::sync::atomic::Ordering::Acquire);
        let spectator_subframe_skip = self
            .spectator_subframe_skip
            .load(std::sync::atomic::Ordering::Acquire);
        // Load the panic-isolation flag once outside the rayon closure.
        // Acquire pairs with the Release store in `set_panic_isolation`.
        let panic_isolation = self
            .panic_isolation
            .load(std::sync::atomic::Ordering::Acquire);
        // When f16 mode is on, `preprocessed` holds raw IEEE 754 half-
        // precision bytes: 2 bytes × 84×84 = 14112 bytes. The Python
        // side reinterprets via `np.frombuffer(..., dtype=np.float16)`.
        let pp_byte_len = if f16_mode { pp_dim * pp_dim * 2 } else { pp_dim * pp_dim };
        // Fused step + collect in one rayon pass. Previously this
        // was two separate par_iter runs — a for_each that stepped,
        // then a map in `collect_results` that unpacked frames.
        // Merging halves the mutex acquisitions, reuses the hot
        // worker-state cache while it's still in L1 after the step,
        // and cuts one rayon join barrier off the critical path.
        //
        // SAFETY: rayon's `par_iter().enumerate()` gives each task
        // a unique index; no two tasks touch the same worker cell.
        // The `&mut Worker` reborrowed from the UnsafeCell is valid
        // for the lifetime of this closure — no aliasing, so it
        // crosses the `catch_unwind` boundary soundly.
        let collected: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)> = self
            .workers
            .par_iter()
            .zip(actions_vec.par_iter())
            .enumerate()
            .map(|(idx, (cell, a))| {
                let w = unsafe { worker_mut(cell) };
                // In skip_preprocess mode nobody reads `preprocessed`
                // (tile policies consume `ram_snapshot`), so ship a
                // 0-length sentinel instead of a dead 7-14 KB zero
                // buffer. `build_result_list` turns it into a (0, 0)
                // array and the Python adapter substitutes a shared
                // zeros((84, 84)) singleton. At 60 workers × 1024
                // steps/iter the dead buffer was ~430 MB of zeroing
                // + ~61K numpy allocations per iteration.
                let empty_or_zero_pp = || {
                    if skip_preprocess {
                        Vec::new()
                    } else {
                        vec![0u8; pp_byte_len]
                    }
                };
                if w.dead {
                    return (
                        vec![0u8; FRAME_PIXELS * 3],
                        empty_or_zero_pp(),
                        vec![0u8; RAM_SIZE],
                        true,
                    );
                }
                // Trainer-flagged episode-done short-circuit. Skip
                // the entire step (no Nes::step, no preprocess). The
                // worker's RAM is returned as-is, but the frame ships
                // as a 0-length sentinel (build_result_list emits it
                // as a (1, 1, 3) placeholder) — we do NOT re-render or
                // copy the ~180 KB prior framebuffer for a slot the
                // trainer already ignores (see done_flags in
                // _evaluate_batch). The Python adapter reuses the
                // worker's previous live frame so a spectator tile
                // keeps its last image instead of going black. The
                // saved work is the ~4× frame_skip NES cycles per
                // remaining-step that would otherwise burn on this
                // dead worker.
                if self.worker_done[idx]
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    let ram_bytes: Vec<u8> =
                        w.nes.system_ram().as_ref().to_vec();
                    return (
                        Vec::new(),
                        empty_or_zero_pp(),
                        ram_bytes,
                        true,
                    );
                }
                let action = *a;
                // The worker body is identical in both branches;
                // only the `catch_unwind` wrap differs. Both
                // branches produce the same `res` type so the
                // downstream match arms stay unchanged.
                let mut work = || {
                    w.step(action, frame_skip, spectator_subframe_skip);
                    // Tile-encoder games read `ram_snapshot`, not
                    // `preprocessed`, so in skip mode the buffer is a
                    // 0-length sentinel and every gray/resize write
                    // below is gated on `!skip_preprocess` (writing
                    // through the raw-pointer f16 path on an empty
                    // Vec would be UB).
                    let mut preprocessed = empty_or_zero_pp();
                    let Worker { gray_scratch, video_buf, nes, .. } = &mut *w;
                    let rgb: Vec<u8> = if headless {
                        if !skip_preprocess {
                            xrgb_to_gray(video_buf, gray_scratch);
                            if f16_mode {
                                let out_u16 = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        preprocessed.as_mut_ptr() as *mut u16,
                                        pp_dim * pp_dim,
                                    )
                                };
                                crate::preprocess::resize_gray_to_f16_norm(
                                    gray_scratch, out_u16,
                                );
                            } else {
                                crate::preprocess::resize_area_into(
                                    gray_scratch, 240, 256, &mut preprocessed,
                                    crate::preprocess::PP_SIZE,
                                );
                            }
                        } else {
                            let _ = (&video_buf, &gray_scratch);
                        }
                        Vec::new()
                    } else {
                        // vec![0u8; N], not with_capacity + set_len: a
                        // reference to uninitialized memory is UB even for
                        // u8. xrgb_to_rgb fully overwrites it below, so the
                        // zero-init is free in practice.
                        let mut rgb: Vec<u8> = vec![0u8; FRAME_PIXELS * 3];
                        xrgb_to_rgb(video_buf, &mut rgb);
                        if !skip_preprocess {
                            if f16_mode {
                                let out_u16 = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        preprocessed.as_mut_ptr() as *mut u16,
                                        pp_dim * pp_dim,
                                    )
                                };
                                crate::preprocess::preprocess_frame_into_f16(
                                    &rgb, gray_scratch, out_u16,
                                );
                            } else {
                                crate::preprocess::preprocess_frame_into(
                                    &rgb, gray_scratch, &mut preprocessed,
                                );
                            }
                        }
                        rgb
                    };
                    let ram_bytes: Vec<u8> = nes.system_ram().as_ref().to_vec();
                    (rgb, preprocessed, ram_bytes, false)
                };
                // catch_unwind is now UNCONDITIONAL. A panic unwinding
                // past a rayon worker crosses the PyO3 boundary as a
                // PanicException and kills the whole training process —
                // the "crashed and stayed dead" failure class. Always
                // catching it means one bad worker degrades to
                // dead+zero-frame (revived by reset_all) instead of
                // taking down 60 envs. `panic_isolation` now only
                // controls how loudly the event is logged; measured
                // always-on cost was 0.5-1%.
                let _ = panic_isolation;
                let res: Result<(Vec<u8>, Vec<u8>, Vec<u8>, bool), Box<dyn std::any::Any + Send>> =
                    catch_unwind(AssertUnwindSafe(work));
                match res {
                    Ok(tuple) => tuple,
                    Err(_) => {
                        eprintln!(
                            "[nes_core::Pool] worker {idx} panicked on step (action=0x{:02x}); marking dead",
                            action,
                        );
                        w.dead = true;
                        (
                            vec![0u8; FRAME_PIXELS * 3],
                            empty_or_zero_pp(),
                            vec![0u8; RAM_SIZE],
                            true,
                        )
                    }
                }
            })
            .collect();
        // Pool-level pacing: ONE sleep on this (calling) thread after
        // the parallel block joins. Rayon threads never sleep, so N
        // paced workers cost the same as one — no ~thread-count
        // ceiling on realtime spectator workers.
        self.pace_pool_step(frame_skip);
        collected
    }

    /// True if any worker is flagged for realtime pacing AND still
    /// live this episode. Dead workers and trainer-flagged done
    /// workers short-circuit before `Worker::step` runs, so under the
    /// old per-worker pacer they never slept — a paced spectator whose
    /// episode ended stopped throttling immediately. Preserve exactly
    /// that: a paced worker only counts while it is neither dead nor
    /// `worker_done` (otherwise a finished GUI-solo worker would drag
    /// the WHOLE pool down to realtime until `reset_all`, since the
    /// trainer calls `set_worker_done` as episodes terminate).
    ///
    /// SAFETY: read sequentially on the `step_all` calling thread
    /// after the rayon join (same pattern as `peek_max_x_per_worker`);
    /// never overlaps a parallel dispatch.
    fn any_worker_paced(&self) -> bool {
        self.workers.iter().enumerate().any(|(idx, cell)| {
            let w = unsafe { &*cell.0.get() };
            w.realtime_pace
                && !w.dead
                && !self.worker_done[idx].load(std::sync::atomic::Ordering::Acquire)
        })
    }

    /// Re-anchor pool pacing so the next paced step measures from
    /// "now" instead of a stale target (mirrors the old per-worker
    /// `pace_next_target = None` semantics on reset / pace flip).
    fn clear_pool_pace_target(&self) {
        let mut slot = self
            .pool_pace_next_target
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        *slot = None;
    }

    /// Pool-level realtime pacing, run once per `step_all` on the
    /// calling thread after the rayon join. Sleeps so paced steps
    /// average `frame_skip / (60 × multiplier)` seconds of wall clock,
    /// with the same catch-up clamp as the old per-worker pacer: if
    /// emulation ran slower than the budget, the next target is
    /// re-anchored to `now + step_dur` so lag debt never accumulates
    /// past one step. No-op (and target cleared) when no live,
    /// not-done worker is paced.
    fn pace_pool_step(&self, frame_skip: u32) {
        let mut slot = self
            .pool_pace_next_target
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !self.any_worker_paced() {
            *slot = None;
            return;
        }
        let mult = f64::from_bits(
            self.pace_multiplier_bits
                .load(std::sync::atomic::Ordering::Acquire),
        );
        let step_dur = std::time::Duration::from_nanos(
            ((frame_skip as f64) * 1_000_000_000.0 / (60.0 * mult)) as u64,
        );
        let now = std::time::Instant::now();
        let target = match *slot {
            Some(t) if t > now => t,
            _ => now,
        };
        if target > now {
            std::thread::sleep(target - now);
        }
        let advanced = target + step_dur;
        let max_lag = now + step_dur;
        *slot = Some(if advanced > max_lag { advanced } else { max_lag });
    }

    fn load_start_state(&self, path: &std::path::Path) -> PyResult<()> {
        let raw = std::fs::read(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to read start state {}: {e}",
                path.display()
            ))
        })?;
        const NCST_MAGIC: &[u8] = b"NCST\x01";
        let snapshot: Vec<u8> = if raw.starts_with(NCST_MAGIC) {
            // Strip every repeated NCST prefix. A historical bug in
            // `src/gui/play_window.py::_save_state` was adding a
            // second `NCST\x01` on top of the one already baked into
            // `NESEnvironment.save_state()`, producing a double-
            // prefix blob. Pre-fix users' files are now recoverable
            // without hand-editing. The save path is fixed; this
            // loop is a belt-and-suspenders compatibility shim.
            let mut body = raw.as_slice();
            while body.starts_with(NCST_MAGIC) {
                body = &body[NCST_MAGIC.len()..];
            }
            eprintln!(
                "[nes_core::Pool] start state {}: NCST binary format ({} body bytes, {} prefix strips)",
                path.display(),
                body.len(),
                (raw.len() - body.len()) / NCST_MAGIC.len(),
            );
            body.to_vec()
        } else {
            // Legacy action-replay: run the replay ONCE on worker 0,
            // then snapshot the resulting NES state and distribute
            // to every worker. Avoids paying the 5-15s replay cost
            // per-worker.
            eprintln!(
                "[nes_core::Pool] start state {}: legacy action-replay ({} frames = ~{:.1}s). \
                 Replaying once then broadcasting snapshot to all {} workers.",
                path.display(),
                raw.len(),
                raw.len() as f64 / 60.0,
                self.num_workers,
            );
            // Legacy recordings use the nes-py bit layout. Now that
            // `BUTTON_*` constants above match that layout too, no
            // reversal is needed — bytes pass through directly.
            let t0 = std::time::Instant::now();
            // SAFETY: called sequentially from Python — never overlaps
            // with an in-flight step_all/reset_all rayon dispatch.
            let w = unsafe { worker_mut(&self.workers[0]) };
            for &byte in &raw {
                w.apply_buttons(byte);
                w.advance_one_frame();
            }
            eprintln!(
                "[nes_core::Pool] replay took {:.2}s",
                t0.elapsed().as_secs_f64()
            );
            bincode::serialize(&w.nes.get_state()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "snapshot post-replay failed: {e}"
                ))
            })?
        };

        // Apply the snapshot to every worker + cache for fast resets.
        let state: crate::nes::State = bincode::deserialize(&snapshot).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "start-state snapshot unreadable: {e}"
            ))
        })?;
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        //
        // Guard the apply_state call with catch_unwind: if the saved
        // mapper state variant doesn't match the current cartridge's
        // mapper (e.g. loading a Contra/mapper-2 state into a freshly
        // booted NROM/mapper-0 cart), the mapper's own apply_state
        // panics at "Invalid mapper state enum variant". Without this
        // guard the panic propagates through PyO3 and kills the whole
        // Python process (observed 2026-04-21 when a stale
        // auto-curriculum state was restored after a profile switch).
        for cell in &self.workers {
            let w = unsafe { worker_mut(cell) };
            let res = catch_unwind(AssertUnwindSafe(|| {
                w.nes.apply_state(&state);
            }));
            if let Err(panic_payload) = res {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "start state {} is not compatible with the current ROM's \
                     mapper — apply_state panicked: {msg}. This typically \
                     happens when a save-state from a different game (e.g. a \
                     stale auto-curriculum snapshot) is passed to a freshly \
                     loaded cartridge. Clear the start_state_path in your \
                     profile or delete the offending .state.bin.",
                    path.display()
                )));
            }
            w.start_state_snapshot = Some(snapshot.clone());
        }
        Ok(())
    }

    /// Collect per-worker (frame RGB uint8, ram bytes, done) into a
    /// Python list. Releases the Python GIL during the heavy repack
    /// so other Python threads (e.g. the Qt event loop reading from
    /// queues) aren't starved.
    fn collect_results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Repack each worker's XRGB8888 video_buf into a flat RGB
        // uint8 Vec, build the 84×84 preprocessed observation for the
        // policy network (Rust-side, NEON SIMD grayscale + area
        // downsample), copy the RAM snapshot. Done outside the GIL so
        // rayon is free to saturate the M4's performance cores and
        // Python-side threads (Qt, trainer-control) don't starve.
        //
        // Dead workers (those that panicked earlier) return all-zero
        // frame+preprocessed+ram + done=true so the caller's episode
        // cleanly terminates without touching potentially corrupt
        // emulator state.
        let pp_dim = crate::preprocess::PP_SIZE;
        let f16_mode = self.preprocess_f16.load(std::sync::atomic::Ordering::Relaxed);
        let skip_preprocess = self.skip_preprocess.load(std::sync::atomic::Ordering::Acquire);
        let pp_byte_len = if f16_mode { pp_dim * pp_dim * 2 } else { pp_dim * pp_dim };
        // Mirror of step_all: in skip_preprocess mode `preprocessed`
        // is a 0-length sentinel (tile policies read `ram_snapshot`);
        // build_result_list emits it as a (0, 0) array and the Python
        // adapter substitutes a shared zeros((84, 84)) singleton.
        let empty_or_zero_pp = || {
            if skip_preprocess {
                Vec::new()
            } else {
                vec![0u8; pp_byte_len]
            }
        };
        let collected: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)> = py.allow_threads(|| {
            // SAFETY: rayon's `par_iter()` gives each task a unique
            // element of `self.workers`; no two tasks touch the same
            // worker cell. The `&mut Worker` reborrowed here is valid
            // across the catch_unwind boundary — single-owner only.
            self.workers
                .par_iter()
                .map(|cell| {
                    let w = unsafe { worker_mut(cell) };
                    if w.dead {
                        return (
                            vec![0u8; FRAME_PIXELS * 3],
                            empty_or_zero_pp(),
                            vec![0u8; RAM_SIZE],
                            true,
                        );
                    }
                    // Peeking RAM / unpacking frame can panic on a
                    // broken mapper; guard with catch_unwind and
                    // demote the worker so collect_results never
                    // propagates a panic to Python.
                    let res = catch_unwind(AssertUnwindSafe(|| {
                        // vec![0u8; N], not with_capacity + set_len (UB on the
                        // uninitialized &mut slice); xrgb_to_rgb overwrites it.
                        let mut rgb: Vec<u8> = vec![0u8; FRAME_PIXELS * 3];
                        xrgb_to_rgb(&w.video_buf, &mut rgb);
                        let mut preprocessed = empty_or_zero_pp();
                        let Worker { gray_scratch, nes, .. } = &mut *w;
                        if !skip_preprocess {
                            if f16_mode {
                                let out_u16 = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        preprocessed.as_mut_ptr() as *mut u16,
                                        pp_dim * pp_dim,
                                    )
                                };
                                crate::preprocess::preprocess_frame_into_f16(
                                    &rgb, gray_scratch, out_u16,
                                );
                            } else {
                                crate::preprocess::preprocess_frame_into(
                                    &rgb, gray_scratch, &mut preprocessed,
                                );
                            }
                        }
                        let ram_bytes: Vec<u8> = nes.system_ram().as_ref().to_vec();
                        (rgb, preprocessed, ram_bytes, false)
                    }));
                    match res {
                        Ok(tuple) => tuple,
                        Err(_) => {
                            w.dead = true;
                            (
                                vec![0u8; FRAME_PIXELS * 3],
                                empty_or_zero_pp(),
                                vec![0u8; RAM_SIZE],
                                true,
                            )
                        }
                    }
                })
                .collect()
        });

        self.build_result_list(py, collected)
    }

    /// Wrap a `Vec<(rgb, preprocessed, ram, done)>` into the Python
    /// list of tuples `step_all` / `reset_all` return. Factored out
    /// so both methods share one GIL-held wrap path.
    ///
    /// This is NOT a per-worker framebuffer memcpy. `into_pyarray_bound`
    /// on an owned `Array` *moves* the Rust `Vec` into the NumPy array
    /// (`from_owned_array_bound` → `from_raw_parts` + `PySliceContainer`
    /// base), so the ~180 KB frame + 84×84 preprocess buffers are handed
    /// to Python by ownership transfer — zero copy. The XRGB→RGB / resize
    /// work that fills those `Vec`s already ran in the GIL-released rayon
    /// phase of `step_all_native`. The only real copy here is the 2 KB
    /// RAM → `PyBytes` per worker (~2.5 µs at 60 workers). Measured object
    /// churn for this whole loop is ~0.03 ms at 60 workers (≈0.3% of a
    /// 60-worker `step_all`), and it is irreducible GIL-bound Python
    /// object creation — there is no bulk frame copy to hoist into the
    /// parallel phase. (Verified 2026-07-12; see the perf analysis.)
    fn build_result_list<'py>(
        &self,
        py: Python<'py>,
        collected: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)>,
    ) -> PyResult<Bound<'py, PyList>> {
        let pp_dim = crate::preprocess::PP_SIZE;
        let f16_mode = self.preprocess_f16.load(std::sync::atomic::Ordering::Relaxed);
        let out = PyList::empty_bound(py);
        for (rgb, preprocessed, ram, done) in collected {
            let frame = if rgb.is_empty() {
                numpy::ndarray::Array3::zeros((1, 1, 3))
            } else {
                numpy::ndarray::Array3::from_shape_vec(
                    (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                    rgb,
                )
                .expect("frame shape")
            };
            use numpy::IntoPyArray;
            let frame_py = frame.into_pyarray_bound(py);
            let pp_py: Bound<'py, pyo3::PyAny> = if preprocessed.is_empty() {
                // skip_preprocess sentinel: tile-mode policies read
                // `ram_snapshot` and never consume `preprocessed`, so
                // ship a 0-length (0, 0) uint8 array instead of a fresh
                // 7-14 KB zero buffer per worker per step. The Python
                // adapter substitutes a shared zeros((84, 84)) singleton
                // to keep the StepResult shape invariant. `pp_byte_len`
                // is always > 0 on the live paths, so empty ⇔ skip mode.
                numpy::ndarray::Array2::<u8>::zeros((0, 0))
                    .into_pyarray_bound(py)
                    .into_any()
            } else if f16_mode {
                debug_assert_eq!(preprocessed.len(), pp_dim * pp_dim * 2);
                let pp = numpy::ndarray::Array2::from_shape_vec((pp_dim, pp_dim * 2), preprocessed)
                    .expect("preprocessed f16 shape");
                pp.into_pyarray_bound(py).into_any()
            } else {
                let pp = numpy::ndarray::Array2::from_shape_vec((pp_dim, pp_dim), preprocessed)
                    .expect("preprocessed shape");
                pp.into_pyarray_bound(py).into_any()
            };
            let ram_py = PyBytes::new_bound(py, &ram);
            let tup = pyo3::types::PyTuple::new_bound(
                py,
                &[
                    frame_py.into_any(),
                    pp_py,
                    ram_py.into_any(),
                    pyo3::types::PyBool::new_bound(py, done).to_owned().into_any(),
                ],
            );
            out.append(tup)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod bugfix_tests {
    use super::*;

    /// Build a synthetic 32 KB-PRG NROM (mapper 0) iNES ROM whose PRG
    /// is filled with a chosen instruction pattern. Reset vector at
    /// $FFFC → $8000 (start of the PRG window).
    pub(super) fn build_nrom(fill: &[u8]) -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(0); // CHR = 0 → CHR-RAM
        rom.push(0); // flags6: mapper 0 low nibble, H-mirror
        rom.push(0); // flags7: mapper 0 high nibble
        rom.extend_from_slice(&[0u8; 8]); // header padding 8..=15

        let mut prg = vec![0u8; 32 * 1024];
        if !fill.is_empty() {
            let mut i = 0;
            while i + fill.len() <= prg.len() {
                prg[i..i + fill.len()].copy_from_slice(fill);
                i += fill.len();
            }
        }
        // Reset vector $FFFC-$FFFD → $8000. Written AFTER the fill so
        // it isn't clobbered by the pattern.
        let n = prg.len();
        prg[n - 4] = 0x00; // low byte of $8000
        prg[n - 3] = 0x80; // high byte of $8000
        rom.extend(prg);
        rom
    }

    pub(super) fn worker_from_rom(rom: Vec<u8>) -> Worker {
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom))
            .expect("synthetic NROM should parse");
        Worker::new(cart)
    }

    /// BUG 3 — cycle-locked `advance_one_frame` must consume EXACTLY
    /// 29781 CPU cycles per frame even when an OAM DMA is in flight.
    ///
    /// The ROM writes $4014 (STA absolute → `8D 14 40`) in a tight
    /// loop, so an OAM DMA (~513-514 cycle stall) is almost always
    /// active. Before the fix, the bulk-step loop calls `nes.step()`
    /// while a DMA is active; that single call runs the whole transfer
    /// plus the next instruction (~517 cycles), overshooting the frame
    /// target by ~500 cycles and breaking the per-frame cycle lock.
    /// After the fix, DMA cycles are advanced one at a time and every
    /// frame lands exactly on target.
    #[test]
    fn advance_one_frame_stays_cycle_locked_across_oam_dma() {
        const CPU_CYCLES_PER_FRAME: usize = 29781;
        let mut w = worker_from_rom(build_nrom(&[0x8D, 0x14, 0x40]));

        // Anchor: the first `advance_one_frame` targets
        // `nes.cycles + 29781` (frame_cycle_target starts None).
        let start = w.nes.cycles;
        for k in 1..=8usize {
            w.advance_one_frame();
            let expected = start + k * CPU_CYCLES_PER_FRAME;
            assert_eq!(
                w.nes.cycles, expected,
                "frame {k}: OAM DMA broke the cycle lock — cycles={} \
                 expected={} (overshoot={})",
                w.nes.cycles,
                expected,
                w.nes.cycles as i64 - expected as i64,
            );
        }
        // Sanity: the ROM actually exercises OAM DMA (otherwise the
        // test would pass trivially and prove nothing).
        assert!(
            w.nes.oam_dma.active || w.nes.oam_dma.count > 0,
            "expected OAM DMA activity from the STA $4014 loop",
        );
    }

    /// BUG 1 — a corrupt / mapper-mismatched state blob must be handled
    /// without unwinding across the (PyO3) boundary. `load_worker_state`
    /// wraps `Nes::apply_state` in `catch_unwind` for exactly this. We
    /// cannot drive `load_worker_state` directly from a unit test (its
    /// `&Bound<PyBytes>` signature needs a live interpreter, which the
    /// `extension-module` build does not link), so this exercises the
    /// same framing → deserialize → guarded-apply flow at the `Worker`
    /// level: a length-mismatched `ram` field makes `apply_state` panic
    /// (`copy_from_slice`), and the `catch_unwind` guard turns that
    /// panic into a recoverable `Err` instead of aborting the process.
    #[test]
    fn corrupt_state_blob_is_caught_not_unwound() {
        let mut w = worker_from_rom(build_nrom(&[0xEA])); // NOP-filled PRG

        // Start from a genuinely valid snapshot, then corrupt it so the
        // blob still deserializes cleanly but detonates inside
        // apply_state (mismatched `ram` length → copy_from_slice panic).
        let mut state = w.nes.get_state();
        assert_eq!(state.ram.len(), RAM_SIZE);
        state.ram.truncate(100);
        let corrupt = bincode::serialize(&state).expect("serialize corrupt state");

        // Frame it exactly like a real curriculum blob and strip the
        // magic exactly like `load_worker_state` does.
        let mut blob = POOL_STATE_MAGIC.to_vec();
        blob.extend_from_slice(&corrupt);
        let mut body: &[u8] = &blob;
        while body.starts_with(POOL_STATE_MAGIC) {
            body = &body[POOL_STATE_MAGIC.len()..];
        }
        let decoded: crate::nes::State =
            bincode::deserialize(body).expect("corrupt blob still deserializes");
        assert_eq!(decoded.ram.len(), 100);

        // The guarded apply (as the fix performs it) must NOT propagate
        // the panic — it returns Err so the caller can mark the worker
        // dead and surface a clean error.
        let res = catch_unwind(AssertUnwindSafe(|| {
            w.nes.apply_state(&decoded);
        }));
        assert!(
            res.is_err(),
            "corrupt ram-length blob was expected to panic apply_state; \
             if this stops panicking the guard is no longer exercised",
        );

        // Mirror the fix's recovery: isolate the worker. The process is
        // still alive (we got here without a SIGABRT), which is the
        // whole point of the guard.
        w.dead = true;
        assert!(w.dead);
    }
}

/// Spectator-scale pool features: pool-level pacing (no per-worker
/// sleeps inside rayon), pace multiplier, pace-independent audio
/// enable, and paced-worker sub-frame render skip.
///
/// Timing tests use generous ±35% tolerances — macOS sleep overshoot
/// and CI load vary — and the whole suite is expected to run with
/// `--test-threads=1` (the repo's standard; parallel timing tests
/// would fight each other for the rayon pool anyway).
#[cfg(test)]
mod spectator_tests {
    use super::bugfix_tests::{build_nrom, worker_from_rom};
    use super::*;

    /// Write a synthetic NOP-filled NROM into target/test-scratch and
    /// build a Pool from it. Unique file per tag so tests don't race
    /// on the ROM file if ever run in parallel.
    fn pool_from_nrom(num_workers: usize, frame_skip: u32, tag: &str) -> Pool {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch");
        std::fs::create_dir_all(&dir).expect("create test-scratch dir");
        let path = dir.join(format!("spectator_{tag}.nes"));
        std::fs::write(&path, build_nrom(&[0xEA])).expect("write synthetic NROM");
        Pool::new(path, num_workers, frame_skip, None).expect("Pool::new")
    }

    fn run_steps(pool: &Pool, steps: usize) -> f64 {
        let actions = vec![0u8; pool.num_workers];
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let _ = pool.step_all_native(&actions);
        }
        t0.elapsed().as_secs_f64()
    }

    fn assert_wall_within(wall: f64, expected: f64, what: &str) {
        let lo = expected * 0.65;
        let hi = expected * 1.35;
        assert!(
            wall >= lo && wall <= hi,
            "{what}: wall {wall:.3}s outside [{lo:.3}, {hi:.3}] (expected ~{expected:.3}s)",
        );
    }

    /// (a) Pool pacing budget at mult=4 with EVERY worker paced. With
    /// the old per-worker in-rayon sleeps this would still pass at 4
    /// workers (4 < thread count) — the load-bearing all-paced scaling
    /// check is `sixteen_paced_workers_do_not_serialize` below. This
    /// one pins the budget arithmetic: 30 steps × 4 frames / (60 × 4)
    /// = 0.5 s.
    #[test]
    fn pool_pacing_budget_all_paced_mult4() {
        let pool = pool_from_nrom(4, 4, "budget_mult4");
        for i in 0..4 {
            pool.set_worker_pace(i, true);
        }
        pool.set_pace_multiplier(4.0);
        let wall = run_steps(&pool, 30);
        assert_wall_within(wall, 30.0 * 4.0 / 240.0, "4 paced workers @ mult 4");
    }

    /// (a, no-serialization) 16 paced workers on a rayon pool that is
    /// almost certainly smaller than 16 threads (P-core cap). Under
    /// the old per-worker pacing, each paced worker slept ~66.7 ms
    /// INSIDE its rayon task, so 16 paced workers on ≤12 threads took
    /// ≥2 sleep waves ≈ 2× the budget per step (~1.3 s here). Pool-
    /// level pacing sleeps once on the caller: wall stays at 1× the
    /// budget (10 steps × 4/60 ≈ 0.667 s) no matter how many workers
    /// pace. Runs at mult=1 so the budget comfortably covers debug-
    /// build emulation time for 16 workers queued on fewer threads —
    /// at tighter budgets this test would measure emulation speed,
    /// not pacing.
    #[test]
    fn sixteen_paced_workers_do_not_serialize() {
        let pool = pool_from_nrom(16, 4, "no_serialize");
        for i in 0..16 {
            pool.set_worker_pace(i, true);
        }
        pool.set_pace_multiplier(1.0);
        let wall = run_steps(&pool, 10);
        assert_wall_within(wall, 10.0 * 4.0 / 60.0, "16 paced workers @ mult 1");
    }

    /// (b) Default multiplier (1.0) + exactly one paced worker = the
    /// pre-change GUI solo mode. Budget must match the historical
    /// per-worker pacer: frame_skip / 60 s per step.
    #[test]
    fn single_paced_worker_matches_realtime_budget() {
        let pool = pool_from_nrom(4, 4, "solo_realtime");
        pool.set_worker_pace(0, true);
        let wall = run_steps(&pool, 10);
        assert_wall_within(wall, 10.0 * 4.0 / 60.0, "1 paced worker @ mult 1");
    }

    /// Unpaced pools must not pace at all — no worker flagged, so
    /// `pace_pool_step` is a no-op and 30 steps of a 4-worker NROM
    /// pool complete in a tiny fraction of the realtime budget.
    #[test]
    fn unpaced_pool_never_sleeps() {
        let pool = pool_from_nrom(4, 4, "unpaced");
        let wall = run_steps(&pool, 30);
        let budget = 30.0 * 4.0 / 60.0; // 2 s realtime equivalent
        assert!(
            wall < budget * 0.25,
            "unpaced pool took {wall:.3}s — pacing appears active (budget {budget:.3}s)",
        );
    }

    /// A paced worker whose episode the trainer has flagged done
    /// (`set_worker_done`) must stop throttling the pool IMMEDIATELY.
    /// Old per-worker semantics: the done short-circuit in the step
    /// closure returned before `Worker::step`, so `pace_to_realtime`
    /// never ran and the pool sprinted from the moment the spectator's
    /// episode ended until `reset_all`. The trainer relies on this
    /// (trainer.py calls `set_worker_done` as episodes terminate) — a
    /// pool-level pacer keyed on `realtime_pace` alone would drag ALL
    /// workers down to realtime (~15 steps/s at fs=4) during that
    /// window.
    #[test]
    fn done_paced_worker_stops_pacing() {
        let pool = pool_from_nrom(4, 4, "done_paced");
        pool.set_worker_pace(0, true);
        // A couple of paced steps first so a pace target exists — the
        // regression must clear/ignore it, not just never create one.
        let paced_wall = run_steps(&pool, 3);
        assert_wall_within(paced_wall, 3.0 * 4.0 / 60.0, "3 paced warm-up steps");
        pool.set_worker_done(0, true);
        let wall = run_steps(&pool, 10);
        let budget = 10.0 * 4.0 / 60.0;
        assert!(
            wall < budget * 0.25,
            "done paced worker still throttles the pool: {wall:.3}s \
             (realtime budget {budget:.3}s)",
        );
        // reset_all clears worker_done, so pacing must resume on the
        // next episode. Drive the native reset path (no interpreter):
        // clear the flag exactly as reset_all does, then re-measure.
        pool.set_worker_done(0, false);
        let resumed = run_steps(&pool, 5);
        assert_wall_within(resumed, 5.0 * 4.0 / 60.0, "pacing resumes after done clears");
    }

    /// Same as above for a worker that died (panicked): dead workers
    /// short-circuit before `Worker::step`, so under the old semantics
    /// they never paced. A paced-then-dead spectator must not throttle
    /// the pool.
    #[test]
    fn dead_paced_worker_stops_pacing() {
        let pool = pool_from_nrom(4, 4, "dead_paced");
        pool.set_worker_pace(0, true);
        {
            // SAFETY: sequential test-thread access, no rayon dispatch
            // in flight — same discipline as drain_audio.
            let w0 = unsafe { worker_mut(&pool.workers[0]) };
            w0.dead = true;
        }
        let wall = run_steps(&pool, 10);
        let budget = 10.0 * 4.0 / 60.0;
        assert!(
            wall < budget * 0.25,
            "dead paced worker still throttles the pool: {wall:.3}s \
             (realtime budget {budget:.3}s)",
        );
    }

    /// Guard against over-eager eligibility: a DIFFERENT worker being
    /// done must not stop pacing while the paced spectator is alive.
    #[test]
    fn other_worker_done_keeps_pacing() {
        let pool = pool_from_nrom(4, 4, "other_done");
        pool.set_worker_pace(0, true);
        pool.set_worker_done(1, true);
        let wall = run_steps(&pool, 10);
        assert_wall_within(wall, 10.0 * 4.0 / 60.0, "paced worker 0, done worker 1");
    }

    /// (c) `set_worker_audio(id, true)` on an UNPACED worker: samples
    /// accumulate (the APU emits at ~44.1 kHz even in silence) and the
    /// worker is NOT paced — steps stay far under the realtime budget.
    #[test]
    fn set_worker_audio_accumulates_without_pacing() {
        let pool = pool_from_nrom(4, 4, "audio_unpaced");
        pool.set_worker_audio(1, true);
        let wall = run_steps(&pool, 10);

        // SAFETY: sequential test-thread access, no rayon dispatch in
        // flight — same discipline as drain_audio.
        let w1 = unsafe { worker_mut(&pool.workers[1]) };
        assert!(
            !w1.audio.is_empty(),
            "worker 1 audio ring empty after 10 steps with audio enabled",
        );
        assert!(
            !w1.realtime_pace,
            "set_worker_audio must not flip realtime_pace",
        );
        let w0 = unsafe { worker_mut(&pool.workers[0]) };
        assert!(
            w0.audio.is_empty(),
            "worker 0 audio should stay disabled (default off)",
        );
        let budget = 10.0 * 4.0 / 60.0;
        assert!(
            wall < budget * 0.25,
            "audio-enabled unpaced worker got paced: wall {wall:.3}s vs realtime budget {budget:.3}s",
        );
    }

    /// `set_worker_pace`'s audio side effect is preserved, and
    /// `set_worker_audio` afterwards wins (documented last-call-wins).
    #[test]
    fn pace_audio_weld_last_call_wins() {
        let pool = pool_from_nrom(2, 4, "weld");
        pool.set_worker_pace(0, true); // welds audio ON
        pool.set_worker_audio(0, false); // last call: audio OFF, still paced
        let actions = vec![0u8; 2];
        for _ in 0..3 {
            let _ = pool.step_all_native(&actions);
        }
        let w0 = unsafe { worker_mut(&pool.workers[0]) };
        assert!(w0.realtime_pace, "worker 0 should still be paced");
        assert!(
            w0.audio.is_empty(),
            "set_worker_audio(0, false) after set_worker_pace(0, true) must win",
        );
    }

    #[test]
    fn pace_multiplier_clamps_and_rejects_non_finite() {
        let pool = pool_from_nrom(1, 4, "clamp");
        let get = |p: &Pool| {
            f64::from_bits(
                p.pace_multiplier_bits
                    .load(std::sync::atomic::Ordering::Acquire),
            )
        };
        assert_eq!(get(&pool), 1.0, "default multiplier");
        pool.set_pace_multiplier(100.0);
        assert_eq!(get(&pool), 16.0, "upper clamp");
        pool.set_pace_multiplier(0.01);
        assert_eq!(get(&pool), 0.25, "lower clamp");
        pool.set_pace_multiplier(f64::NAN);
        assert_eq!(get(&pool), 1.0, "NaN resets to default");
        pool.set_pace_multiplier(2.0);
        assert_eq!(get(&pool), 2.0, "in-range value passes through");
    }

    /// (d) Spectator sub-frame skip is emulation-neutral: a paced
    /// worker stepped with final-frame-only rendering must land on
    /// bit-identical RAM, identical CPU cycle count, AND an identical
    /// final framebuffer vs one rendering every sub-frame. Uses the
    /// real SMB ROM when present (sprite-0-timed HUD split is the
    /// historically risky path — see tests/skip_render_parity.rs),
    /// falling back to the synthetic NROM used by the other pool
    /// tests.
    #[test]
    fn spectator_subframe_skip_is_emulation_neutral() {
        let rom_bytes = std::fs::read("../roms/Super Mario Bros. (World).nes")
            .unwrap_or_else(|_| build_nrom(&[0xEA]));
        let mut w_skip = worker_from_rom(rom_bytes.clone());
        let mut w_full = worker_from_rom(rom_bytes);
        w_skip.realtime_pace = true;
        w_full.realtime_pace = true;

        // Hold Start on steps 15..18 (frames 60..72) to leave the SMB
        // title screen, then run right — enters the sprite-0 HUD-split
        // gameplay path. Harmless input pattern on the NROM fallback.
        for step_idx in 0..60usize {
            let action = if (15..18).contains(&step_idx) {
                BUTTON_START
            } else {
                BUTTON_RIGHT
            };
            w_skip.step(action, 4, true);
            w_full.step(action, 4, false);
        }

        assert_eq!(
            w_skip.nes.cycles, w_full.nes.cycles,
            "CPU cycle counts diverged — sub-frame skip touched emulation timing",
        );
        assert_eq!(
            w_skip.nes.system_ram().as_ref(),
            w_full.nes.system_ram().as_ref(),
            "RAM diverged — sub-frame skip is not emulation-neutral",
        );
        let px_diff = w_skip
            .video_buf
            .iter()
            .zip(w_full.video_buf.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            px_diff, 0,
            "final framebuffer diverged on {px_diff} / {FRAME_PIXELS} pixels",
        );
    }
}

/// Pool-side hardware-true frame anchoring.
///
/// `set_hw_frame_anchor` existed only on `NESEnvironment`, so a Pool
/// always ran the legacy fixed-29,781-cycle frame block. That block
/// drifts ~+1/3 CPU cycle per frame against the true PPU frame, which
/// means a tape built or replayed on a Pool structurally cannot line
/// up with Mesen's `inputPolled` schedule — a Pool-built tape can
/// never be certified against Mesen on its own machine. These tests
/// pin both halves of the fix: the anchored path really tracks PPU
/// frame boundaries, and the default (flag never touched) path is
/// byte-identical to the pre-change cycle lock.
#[cfg(test)]
mod frame_anchor_tests {
    use super::bugfix_tests::{build_nrom, worker_from_rom};
    use super::*;

    const CPU_CYCLES_PER_FRAME: usize = 29781;

    fn pool_from_nrom(num_workers: usize, tag: &str) -> Pool {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch");
        std::fs::create_dir_all(&dir).expect("create test-scratch dir");
        let path = dir.join(format!("frame_anchor_{tag}.nes"));
        std::fs::write(&path, build_nrom(&[0xEA])).expect("write synthetic NROM");
        Pool::new(path, num_workers, 1, None).expect("Pool::new")
    }

    /// The new field must not perturb the default path. A worker that
    /// never sees `set_hw_frame_anchor` still consumes EXACTLY 29,781
    /// CPU cycles per `advance_one_frame`, DMA in flight or not — the
    /// same invariant `advance_one_frame_stays_cycle_locked_across_oam_dma`
    /// pins, re-asserted here because the anchored branch sits directly
    /// above the cycle-lock code it must not disturb.
    #[test]
    fn default_path_is_still_exactly_cycle_locked() {
        for fill in [&[0xEA][..], &[0x8D, 0x14, 0x40][..]] {
            let mut w = worker_from_rom(build_nrom(fill));
            assert!(!w.hw_frame_anchor, "frame anchor must default OFF");
            let start = w.nes.cycles;
            for k in 1..=8usize {
                w.advance_one_frame();
                assert_eq!(
                    w.nes.cycles,
                    start + k * CPU_CYCLES_PER_FRAME,
                    "fill {fill:02X?} frame {k}: default path left the cycle lock",
                );
            }
        }
    }

    /// With the anchor ON, every call crosses exactly one PPU frame,
    /// and the cycle budget tracks the real 89,342-dot frame
    /// (29,780.67 CPU cycles) instead of the fixed 29,781 block.
    ///
    /// This is the whole point of the flag: over 180 frames (a
    /// multiple of 3, so the true frame is a whole 5,360,520 cycles)
    /// the fixed block spends 5,360,580 — 60 cycles of drift, ~+1/3
    /// per frame. Per-frame deltas can't be asserted tightly because
    /// `nes.step()` runs a whole instruction and overshoots the
    /// boundary by 0-6 cycles; the accumulated total can, and it is
    /// the quantity that desynced the CV tape from Mesen's
    /// `inputPolled` schedule.
    #[test]
    fn anchored_path_tracks_the_real_ppu_frame_not_a_fixed_block() {
        const FRAMES: usize = 180;
        // 3 PPU frames = 3 × 89,342 dots = 89,342 CPU cycles exactly.
        const TRUE_CYCLES: usize = FRAMES / 3 * 89_342;
        // Worst-case instruction overshoot at each end of the window.
        const SLOP: i64 = 8;

        let mut w = worker_from_rom(build_nrom(&[0xEA]));
        w.hw_frame_anchor = true;
        // First call re-syncs from wherever boot left the PPU to the
        // next frame boundary; measure from there.
        w.advance_one_frame();

        let f0 = w.nes.ppu.frame;
        let c0 = w.nes.cycles;
        for k in 1..=FRAMES {
            let before = w.nes.ppu.frame;
            w.advance_one_frame();
            assert_eq!(
                w.nes.ppu.frame,
                before + 1,
                "frame {k}: anchored advance did not cross exactly one PPU frame",
            );
        }
        assert_eq!(w.nes.ppu.frame, f0 + FRAMES as u64);

        let spent = (w.nes.cycles - c0) as i64;
        assert!(
            (spent - TRUE_CYCLES as i64).abs() <= SLOP,
            "anchored {FRAMES} frames spent {spent} CPU cycles, expected \
             {TRUE_CYCLES} ± {SLOP}",
        );
        // And it is measurably NOT the fixed block the default path runs.
        let fixed = (FRAMES * CPU_CYCLES_PER_FRAME) as i64;
        assert!(
            fixed - spent > SLOP,
            "anchored total {spent} is indistinguishable from the fixed \
             block {fixed} — the anchor is not tracking the PPU",
        );
    }

    /// The anchored branch must drain an in-flight OAM DMA one CPU
    /// cycle at a time, exactly like the cycle-locked branch does, so
    /// a DMA straddling the frame boundary can't run the PPU past it.
    #[test]
    fn anchored_path_crosses_one_frame_with_oam_dma_in_flight() {
        let mut w = worker_from_rom(build_nrom(&[0x8D, 0x14, 0x40]));
        w.hw_frame_anchor = true;
        for k in 1..=8usize {
            let f0 = w.nes.ppu.frame;
            w.advance_one_frame();
            assert_eq!(
                w.nes.ppu.frame,
                f0 + 1,
                "frame {k}: OAM DMA overshot the anchored frame boundary",
            );
        }
        assert!(
            w.nes.oam_dma.active || w.nes.oam_dma.count > 0,
            "expected OAM DMA activity from the STA $4014 loop",
        );
    }

    /// The anchor lives on `Worker`, not on `Nes`, so it is
    /// deliberately absent from `State` — it must survive an
    /// `apply_state` (the `load_worker_state` path) as sticky config,
    /// matching every other hw flag. If someone ever moves it into
    /// `Nes`/`State` this test fails and tells them the savestate
    /// format just grew a config field.
    #[test]
    fn frame_anchor_is_config_not_state() {
        let mut w = worker_from_rom(build_nrom(&[0xEA]));
        w.hw_frame_anchor = true;
        w.advance_one_frame();
        let snapshot = w.nes.get_state();

        // A state captured on an anchored machine must not carry the
        // flag back into an unanchored one.
        let mut fresh = worker_from_rom(build_nrom(&[0xEA]));
        fresh.nes.apply_state(&snapshot);
        assert!(
            !fresh.hw_frame_anchor,
            "apply_state leaked frame-anchor config out of a savestate",
        );

        // ...and loading a state into an anchored worker must not
        // silently clear the flag.
        w.nes.apply_state(&snapshot);
        assert!(
            w.hw_frame_anchor,
            "apply_state clobbered sticky frame-anchor config",
        );
    }

    /// `Pool::set_hw_frame_anchor` sets every worker, and clears the
    /// stale `frame_cycle_target` so a mid-run flip can't compute a
    /// target from the other policy's bookkeeping.
    #[test]
    fn pool_setter_reaches_every_worker() {
        let pool = pool_from_nrom(4, "setter");
        let actions = vec![0u8; pool.num_workers];
        let _ = pool.step_all_native(&actions);
        for cell in &pool.workers {
            let w = unsafe { worker_mut(cell) };
            assert!(!w.hw_frame_anchor, "flag must default OFF on every worker");
            assert!(w.frame_cycle_target.is_some(), "cycle-locked path sets a target");
        }

        pool.set_hw_frame_anchor(true);
        for cell in &pool.workers {
            let w = unsafe { worker_mut(cell) };
            assert!(w.hw_frame_anchor, "setter missed a worker");
            assert!(
                w.frame_cycle_target.is_none(),
                "setter left a stale cycle target behind",
            );
        }

        // Anchored stepping keeps every worker on real frame boundaries.
        let before: Vec<u64> = pool
            .workers
            .iter()
            .map(|c| unsafe { worker_mut(c) }.nes.ppu.frame)
            .collect();
        let _ = pool.step_all_native(&actions);
        for (i, cell) in pool.workers.iter().enumerate() {
            let w = unsafe { worker_mut(cell) };
            assert_eq!(
                w.nes.ppu.frame,
                before[i] + 1,
                "worker {i}: anchored step_all did not advance exactly one frame",
            );
        }

        pool.set_hw_frame_anchor(false);
        for cell in &pool.workers {
            let w = unsafe { worker_mut(cell) };
            assert!(!w.hw_frame_anchor, "setter could not turn the flag back off");
        }
    }
}
