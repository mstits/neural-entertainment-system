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
/// set_worker_pace, drain_audio, set_batched_render_mode,
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
    // Realtime pacing for the live-audio worker. Unused for the
    // 15 non-audio workers; keeps them running at max speed.
    realtime_pace: bool,
    pace_next_target: Option<std::time::Instant>,
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
            pace_next_target: None,
            start_state_snapshot: None,
            dead: false,
            gray_scratch: vec![0u8; FRAME_PIXELS],
            frame_cycle_target: None,
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
        self.pace_next_target = None;
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
        let target = self
            .frame_cycle_target
            .map(|t| t + CPU_CYCLES_PER_FRAME)
            .unwrap_or_else(|| self.nes.cycles + CPU_CYCLES_PER_FRAME);
        let margin = self.nes.asm_bulk_cycles_margin();
        while self.nes.cycles + margin < target {
            self.nes.step(&mut video, &mut sink);
        }
        while self.nes.cycles < target {
            self.nes.tick(&mut video, &mut sink);
        }
        self.frame_cycle_target = Some(target);
    }

    fn step(&mut self, action: u8, frame_skip: u32) -> bool {
        self.apply_buttons(action);
        let use_skip = !self.realtime_pace;
        for i in 0..frame_skip {
            let is_last = i + 1 == frame_skip;
            self.nes.set_skip_render(use_skip && !is_last);
            self.advance_one_frame();
        }
        self.nes.set_skip_render(false);
        if self.realtime_pace {
            self.pace_to_realtime(frame_skip);
        }
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

    fn pace_to_realtime(&mut self, frame_skip: u32) {
        let step_dur = std::time::Duration::from_nanos(
            ((frame_skip as u64) * 1_000_000_000) / 60,
        );
        let now = std::time::Instant::now();
        let target = match self.pace_next_target {
            Some(t) if t > now => t,
            _ => now,
        };
        if target > now {
            std::thread::sleep(target - now);
        }
        let advanced = target + step_dur;
        let max_lag = now + step_dur;
        self.pace_next_target = Some(if advanced > max_lag {
            advanced
        } else {
            max_lag
        });
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
/// comment below). This is a transparent newtype so rayon can hand
/// out `&WorkerCell` items across worker threads.
#[repr(transparent)]
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
// (`save_worker_state`, `set_worker_pace`, `drain_audio`,
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
    worker_done: Vec<std::sync::atomic::AtomicBool>,
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
            panic_isolation: std::sync::atomic::AtomicBool::new(true),
            worker_done,
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
        py.allow_threads(|| {
            // SAFETY: rayon's `par_iter().enumerate()` gives each task
            // a unique index; no two tasks touch the same worker cell.
            self.workers.par_iter().enumerate().for_each(|(idx, cell)| {
                let w = unsafe { worker_mut(cell) };
                if w.dead {
                    return;
                }
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
        let frame_skip = self.frame_skip;
        let pp_dim = crate::preprocess::PP_SIZE;
        let headless = self.headless.load(std::sync::atomic::Ordering::Acquire);
        let f16_mode = self.preprocess_f16.load(std::sync::atomic::Ordering::Acquire);
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
        // Merging halves the mutex acquisitions, reuses the hot
        // worker-state cache while it's still in L1 after the step,
        // and cuts one rayon join barrier off the critical path.
        let collected: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)> = py.allow_threads(|| {
            // SAFETY: rayon's `par_iter().enumerate()` gives each task
            // a unique index; no two tasks touch the same worker cell.
            // The `&mut Worker` reborrowed from the UnsafeCell is valid
            // for the lifetime of this closure — no aliasing, so it
            // crosses the `catch_unwind` boundary soundly.
            self.workers
                .par_iter()
                .zip(actions_vec.par_iter())
                .enumerate()
                .map(|(idx, (cell, a))| {
                    let w = unsafe { worker_mut(cell) };
                    if w.dead {
                        return (
                            vec![0u8; FRAME_PIXELS * 3],
                            vec![0u8; pp_byte_len],
                            vec![0u8; RAM_SIZE],
                            true,
                        );
                    }
                    // Trainer-flagged episode-done short-circuit. Skip
                    // the entire step (no Nes::step, no preprocess) —
                    // the worker's prior frame and RAM stay untouched
                    // and we return them as-is. The trainer is going
                    // to ignore this slot anyway (see done_flags in
                    // _evaluate_batch); the saved work is the ~4×
                    // frame_skip NES cycles per remaining-step that
                    // would otherwise burn on this dead worker.
                    if self.worker_done[idx]
                        .load(std::sync::atomic::Ordering::Acquire)
                    {
                        let ram_bytes: Vec<u8> =
                            w.nes.system_ram().as_ref().to_vec();
                        return (
                            Vec::new(),
                            vec![0u8; pp_byte_len],
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
                        w.step(action, frame_skip);
                        let mut preprocessed = vec![0u8; pp_byte_len];
                        let Worker { gray_scratch, video_buf, nes, .. } = &mut *w;
                        let rgb: Vec<u8> = if headless {
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
                            Vec::new()
                        } else {
                            let mut rgb: Vec<u8> = Vec::with_capacity(FRAME_PIXELS * 3);
                            unsafe { rgb.set_len(FRAME_PIXELS * 3); }
                            xrgb_to_rgb(video_buf, &mut rgb);
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
                            rgb
                        };
                        let ram_bytes: Vec<u8> = nes.system_ram().as_ref().to_vec();
                        (rgb, preprocessed, ram_bytes, false)
                    };
                    let res: Result<(Vec<u8>, Vec<u8>, Vec<u8>, bool), Box<dyn std::any::Any + Send>> =
                        if panic_isolation {
                            catch_unwind(AssertUnwindSafe(work))
                        } else {
                            // Direct call. Any panic will unwind past
                            // the rayon worker, which is unsound — but
                            // the trainer takes responsibility for
                            // stable ROMs/states when this flag is off.
                            Ok(work())
                        };
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
                                vec![0u8; pp_byte_len],
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

    /// Toggle realtime pacing on a specific worker — used for the
    /// live-audio worker (mixer Solo X mode) so audio production
    /// matches playback rate. Other workers stay unpaced.
    fn set_worker_pace(&self, worker_id: usize, on: bool) {
        if worker_id >= self.num_workers {
            return;
        }
        // SAFETY: called sequentially from Python — never overlaps with
        // an in-flight step_all/reset_all rayon dispatch.
        let w = unsafe { worker_mut(&self.workers[worker_id]) };
        if w.realtime_pace != on {
            w.realtime_pace = on;
            w.pace_next_target = None;
            // Flipping pace ↔ audio: the paced worker is the one
            // whose audio the mixer plays, so only it should do the
            // sample-gen + mix work. Re-align audio output to match.
            w.nes.set_audio_output_enabled(on);
        }
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
        w.nes.apply_state(&state);
        w.audio.clear();
        Ok(())
    }

    #[getter]
    fn num_workers(&self) -> usize {
        self.num_workers
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
        let pp_byte_len = if f16_mode { pp_dim * pp_dim * 2 } else { pp_dim * pp_dim };
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
                            vec![0u8; pp_byte_len],
                            vec![0u8; RAM_SIZE],
                            true,
                        );
                    }
                    // Peeking RAM / unpacking frame can panic on a
                    // broken mapper; guard with catch_unwind and
                    // demote the worker so collect_results never
                    // propagates a panic to Python.
                    let res = catch_unwind(AssertUnwindSafe(|| {
                        let mut rgb: Vec<u8> = Vec::with_capacity(FRAME_PIXELS * 3);
                        unsafe { rgb.set_len(FRAME_PIXELS * 3); }
                        xrgb_to_rgb(&w.video_buf, &mut rgb);
                        let mut preprocessed = vec![0u8; pp_byte_len];
                        let Worker { gray_scratch, nes, .. } = &mut *w;
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
                        let ram_bytes: Vec<u8> = nes.system_ram().as_ref().to_vec();
                        (rgb, preprocessed, ram_bytes, false)
                    }));
                    match res {
                        Ok(tuple) => tuple,
                        Err(_) => {
                            w.dead = true;
                            (
                                vec![0u8; FRAME_PIXELS * 3],
                                vec![0u8; pp_byte_len],
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
    /// so both methods share one GIL-held copy path.
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
            let pp_py: Bound<'py, pyo3::PyAny> = if f16_mode {
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
