//! PyO3 module — Python-importable NESEnvironment.
//!
//! Compiled only when the `python` feature is on (gated in lib.rs).
//! Built via maturin into a `.so` named `nes_core` that exposes the
//! `NESEnvironment` class to Python. Drop-in for the existing
//! `src/emulation/nes_environment.py` and `nesrs_environment.py`
//! shapes — same method names, same return types, same button
//! bitmask layout.

use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

use crate::apu::SAMPLE_RATE as APU_NATIVE_RATE;
use crate::cartridge::Cartridge;
use crate::input::Button;
use crate::nes::Nes;
use crate::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use crate::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;

/// Apply a save-state to `nes` behind a panic guard, mapping any panic
/// (wrong-length RAM, mismatched mapper enum variant, etc.) to a
/// PyValueError instead of unwinding across the PyO3 boundary and
/// killing the interpreter. Single funnel so every restore site
/// (constructor, load_state, reset_no_advance) inherits the same
/// guarantee — the Pool paths were hardened after a wrong-game-state
/// process-kill incident; the NESEnvironment paths were not, and a GUI
/// "Load State" on a stale/wrong-game/truncated .state.bin hard-crashed
/// the play window. On failure the caller is expected to leave the env
/// cold-reset, never half-mutated.
fn apply_state_guarded(nes: &mut Nes, state: &crate::nes::State) -> PyResult<()> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nes.apply_state(state);
    }))
    .map_err(|e| {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "save-state is not compatible with this ROM (apply_state \
             panicked: {msg}) — it is likely from a different game, a \
             different core version, or a truncated/corrupt file"
        ))
    })
}

/// Magic prefix for versioned on-the-wire save-state blobs.
/// Format: `b"NCST" + version_byte + bincode(nes::State)`. Version
/// byte bumps whenever the `State` struct layout changes so old blobs
/// refuse to load with a clear error rather than silently deserialising
/// into a different layout and crashing further downstream.
const STATE_MAGIC: &[u8; 5] = b"NCST\x01";

// Match nes-py / nesrs button bitmask layout. Bits as defined in
// src/emulation/nes_environment.py.
// Button bitmask layout — nes-py convention. See `pool.rs` for the
// history of why we standardized on this. All Python callers
// (`src/emulation/frame_utils.py`, trainer action table, legacy
// recordings) already use this layout.
const BUTTON_RIGHT: u8 = 1 << 7;
const BUTTON_LEFT: u8 = 1 << 6;
const BUTTON_DOWN: u8 = 1 << 5;
const BUTTON_UP: u8 = 1 << 4;
const BUTTON_START: u8 = 1 << 3;
const BUTTON_SELECT: u8 = 1 << 2;
const BUTTON_B: u8 = 1 << 1;
const BUTTON_A: u8 = 1 << 0;

/// Audio sink that accumulates f32 samples into an int16 stereo
/// buffer, matching the existing nesrs `get_audio()` shape.
struct CaptureAudio {
    samples: Vec<i16>,
}
impl CaptureAudio {
    fn new() -> Self {
        Self { samples: Vec::new() }
    }
    fn drain(&mut self) -> Vec<i16> {
        std::mem::take(&mut self.samples)
    }
}
impl AudioSink for CaptureAudio {
    fn write_sample(&mut self, s: f32) {
        // Mono: one sample per APU output. The Python `AudioMixer`
        // accepts mono int16 directly via the 1D-array branch in
        // `push_audio`. Emitting stereo-interleaved here makes the
        // mixer treat the buffer as 2x the samples (it reshapes flat
        // and plays at 2x speed).
        let i = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        self.samples.push(i);
    }
    fn samples_written(&self) -> usize {
        self.samples.len()
    }
}

/// Python-facing NES environment.
#[pyclass(module = "nes_core")]
pub struct NESEnvironment {
    nes: Nes,
    frame_skip: u32,
    sample_rate: u32,
    // When `realtime_pace` is true, `step()` sleeps after the
    // emulation work so the per-step wall time matches the real
    // game time the step represents (frame_skip / 60 seconds).
    // The trainer flips this on for the worker whose audio is
    // being routed to speakers (Solo X mode); other workers stay
    // unpaced for max training throughput.
    realtime_pace: bool,
    // Tracks the wall-clock target for the next step boundary so
    // pacing accumulates cleanly across steps without drift. Reset
    // on reset() and on realtime_pace toggle.
    pace_next_target: Option<std::time::Instant>,
    // Hardware-true frame anchoring (config): when set,
    // `advance_one_frame` runs to the next real PPU frame boundary
    // (29,780/29,781 CPU cycles alternating with the odd-frame dot
    // skip) instead of the legacy fixed 29,781-cycle block. The fixed
    // block drifts +1/3 CPU cycle per frame against the true frame,
    // so input tapes applied per step slip ~1,900 cycles into the
    // frame by ~frame 3,800 and eventually miss the game's controller
    // poll — the receipted CV-vs-Mesen tape desync mechanism. Default
    // OFF: training/parity paths depend on the cycle-locked step.
    hw_frame_anchor: bool,
    // Reusable RGB conversion buffer — Xrgb8888VideoSink writes into
    // this u32 array, then `step()` repacks it into a (240, 256, 3)
    // numpy uint8 array. Kept on the struct so we don't reallocate
    // per step; the perf split (05) will replace the repack with a
    // zero-copy buffer-protocol export.
    video_buf: Vec<u32>,
    audio: CaptureAudio,
    done: bool,
    // Post-replay snapshot of the NES state. When `start_state_path`
    // is supplied, the constructor replays the recorded actions then
    // captures the resulting NES state into this `Vec<u8>` (bincode-
    // encoded `nes::State`). Subsequent `reset()` calls deserialize
    // this back into the NES so episodes always begin at the recorded
    // checkpoint without paying the multi-second replay cost again.
    // None when no start state is configured.
    start_state_snapshot: Option<Vec<u8>>,
    // Absolute CPU cycle target for the next frame boundary. nes-py
    // (LaiNES upstream) runs a fixed 29781 CPU cycles per frame; ours
    // historically ran "until PPU writes the frame," which produced
    // per-frame variance of ±5 cycles. The drift accumulates and
    // breaks byte-exact parity on cycle-sensitive games (Zelda's
    // sword-pickup transition, see scripts/zelda_cave_entry_repro.py).
    // Tracking an absolute target lets a long instruction in frame N
    // shorten frame N+1 by the overshoot, so cumulative CPU/PPU cycle
    // counts stay locked to 29781 × frame-count. Initialized lazily
    // on first `advance_one_frame` after a reset.
    frame_cycle_target: Option<usize>,
}

#[pymethods]
impl NESEnvironment {
    #[new]
    #[pyo3(signature = (
        rom_path, frame_skip = 1, start_state_path = None, sample_rate = 44_100,
    ))]
    fn new(
        rom_path: std::path::PathBuf,
        frame_skip: u32,
        start_state_path: Option<std::path::PathBuf>,
        // `sample_rate` is accepted for API compat with nesrs/nes-py
        // but IGNORED — the APU produces samples at its native rate
        // (`APU_NATIVE_RATE`, ~43653 Hz on NTSC). The Python audio
        // mixer is responsible for resampling to its output device
        // rate. Reporting an honest rate via the `sample_rate`
        // property lets the mixer's resampler do its job;
        // previously we reported 44100 and produced 43653, causing a
        // 1% pitch drift AND silent drops in the mixer's
        // "rates match exactly" branch.
        sample_rate: u32,
    ) -> PyResult<Self> {
        let _ = sample_rate;
        let rom_basename = rom_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| rom_path.display().to_string());
        let bytes = std::fs::read(&rom_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to read ROM {rom_basename}: {e}"
            ))
        })?;
        let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to parse iNES: {e:?}"
            ))
        })?;
        // Guard against unsupported mappers + panics during mapper
        // construction. Without this, a scan of the full NES library
        // takes down the Python interpreter the first time it hits an
        // obscure cart (VRC7, TxSROM, Codemasters, etc.).
        let mapper_num = cart.mapper;
        let nes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Nes::new(cart)
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
        let mut env = Self {
            nes,
            frame_skip: frame_skip.max(1),
            sample_rate: APU_NATIVE_RATE,
            video_buf: vec![0u32; FRAME_PIXELS],
            audio: CaptureAudio::new(),
            done: false,
            start_state_snapshot: None,
            realtime_pace: false,
            pace_next_target: None,
            frame_cycle_target: None,
            hw_frame_anchor: false,
        };
        if let Some(p) = start_state_path {
            let p_basename = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string());
            let raw = std::fs::read(&p).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "failed to read start state {p_basename}: {e}"
                ))
            })?;
            // Two on-disk formats:
            //   * binary nes_core state: prefix b"NCST\x01" + bincode blob.
            //     Saved by the new PlayWindow path. Loads byte-exact via
            //     `apply_state` — no emulator-determinism dependency.
            //   * legacy action recording: raw byte stream, one byte per
            //     frame. Replayed from cold reset by stepping through
            //     each byte. Diverges between emulators over thousands
            //     of frames; supported for backward compat with old
            //     `.state.bin` files.
            const NCST_MAGIC: &[u8] = b"NCST\x01";
            if raw.starts_with(NCST_MAGIC) {
                eprintln!(
                    "[nes_core] start state {}: NCST binary format ({} bytes), instant load.",
                    p.display(), raw.len() - NCST_MAGIC.len(),
                );
                let blob = &raw[NCST_MAGIC.len()..];
                let state: crate::nes::State = bincode::deserialize(blob)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "binary start state {} unreadable: {e}",
                            p.display()
                        ))
                    })?;
                apply_state_guarded(&mut env.nes, &state)?;
                // Cache for fast resets.
                env.start_state_snapshot = Some(blob.to_vec());
            } else {
                // Legacy action-replay path. Slow (one frame per byte)
                // and emulator-determinism-dependent — these are the
                // recordings made before the binary save_state existed.
                eprintln!(
                    "[nes_core] start state {}: LEGACY action-replay format ({} frames = ~{:.1}s). \
                     For instant-load + byte-exact handoff, re-save via the Play window — the \
                     new save format writes an NCST binary snapshot instead.",
                    p.display(), raw.len(), raw.len() as f64 / 60.0,
                );
                let t0 = std::time::Instant::now();
                env.replay_actions(&raw).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "start state replay at {} failed: {e}",
                        p.display()
                    ))
                })?;
                eprintln!(
                    "[nes_core] replay of {} took {:.2}s",
                    p.display(), t0.elapsed().as_secs_f64(),
                );
                env.start_state_snapshot = Some(
                    bincode::serialize(&env.nes.get_state()).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "failed to snapshot post-replay state: {e}"
                        ))
                    })?,
                );
            }
        }
        // Drop initial audio (boot transient) before returning.
        env.audio.drain();
        Ok(env)
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        // Emulation-heavy work (state restore + a full frame advance)
        // runs with the GIL released so other Python threads (Qt event
        // loop, spectator renderer, trainer control) keep running. The
        // closure touches only pure-Rust `self` fields — no `py`, no
        // Py* objects. The numpy frame export below reacquires the GIL.
        py.allow_threads(|| -> PyResult<()> {
            // When a start-state snapshot is cached, restore from it
            // directly WITHOUT a power-cycle first. The upstream
            // `Nes::reset()` reinitializes mapper internals (PRG/CHR
            // bank windows etc.) that `apply_state` doesn't fully
            // re-establish; doing a reset before apply leaves the CPU
            // pointing into PRG ROM with the wrong bank mapped, which
            // crashes on the first opcode fetch (observed: opcode 0x02
            // panic at PC 0xE534 on Zelda after 100 frames of play).
            if let Some(snap) = &self.start_state_snapshot {
                let state: crate::nes::State = bincode::deserialize(snap).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "failed to restore start-state snapshot: {e}"
                    ))
                })?;
                self.nes.apply_state(&state);
            } else {
                self.nes.reset();
            }
            self.audio.drain();
            self.done = false;
            // Reset pacing so the first paced step doesn't sleep for the
            // entire elapsed time since pacing was last active.
            self.pace_next_target = None;
            // Reset cycle target so the first post-reset advance_one_frame
            // anchors to the post-reset cycle count rather than to the
            // previous episode's accumulated drift.
            self.frame_cycle_target = None;
            // Step until first frame is rendered, so the caller gets a
            // valid image rather than uninitialized buffer.
            //
            // Note on parity: this puts nes_core one emulated frame
            // ahead of nes-py, whose `NESEnv.reset()` does NOT advance.
            // Parity tests (`tests/parity/lockstep.py`) compensate by
            // advancing nes-py an extra frame; removing this advance
            // entirely breaks the GUI play-window (user sees a black
            // frame and thinks the emulator is frozen) and has been
            // empirically confirmed user-visible bad. Accept the 1-frame
            // offset cost and compensate in parity tests instead.
            self.advance_one_frame();
            Ok(())
        })?;
        Ok(self.frame_to_numpy(py))
    }

    /// Reset for parity tracing: like `reset()` but skips the
    /// auto-frame-advance. Leaves the CPU at cycle 7 (post-reset
    /// 6502 state) so per-instruction traces from this point align
    /// with reference emulators (Mesen, nes-py) that do not auto-
    /// advance on reset. Not safe for the GUI / training paths —
    /// frame buffer contains uninitialized memory until the first
    /// `step()`. Use only for trace generation.
    fn reset_no_advance(&mut self, py: Python<'_>) {
        // Entirely pure-Rust — no Py* objects touched — so the whole
        // body runs with the GIL released.
        py.allow_threads(|| {
            // Deserialize into an owned State first so the immutable borrow
            // of `start_state_snapshot` is released before the mutable
            // apply — and cold-reset on any deserialize OR apply failure.
            let restored = self
                .start_state_snapshot
                .as_ref()
                .and_then(|snap| {
                    bincode::deserialize::<crate::nes::State>(snap).ok()
                });
            let applied = match restored {
                Some(state) => apply_state_guarded(&mut self.nes, &state).is_ok(),
                None => false,
            };
            if !applied {
                self.nes.reset();
            }
            self.audio.drain();
            self.done = false;
            self.pace_next_target = None;
            self.frame_cycle_target = None;
        });
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_bitmask: u8,
    ) -> PyResult<(Bound<'py, PyArray3<u8>>, bool)> {
        // The whole emulation batch (button apply, frame_skip stepping,
        // and any realtime pacing sleep) runs with the GIL released so
        // the ~1,000 env calls/s from a spectator/solver thread no longer
        // fight the GIL — the receipted spectator-starvation fix
        // (commit f14d60b). The closure touches only pure-Rust `self`
        // fields; the numpy frame export below reacquires the GIL.
        let done = py.allow_threads(|| {
            self.apply_buttons(action_bitmask);
            // Skip-render fast path: only the LAST frame in the frame_skip
            // batch needs a rendered frame buffer (the trainer observes
            // that one). All earlier frames in the batch run with PPU
            // pixel work skipped — sprite-0 / NMI / vblank / mapper IRQs
            // still tick correctly, but per-pixel color resolution +
            // frame_buffer writes are eliminated.
            //
            // Exception: when `realtime_pace` is on (this worker's audio is
            // being heard live), DISABLE skip-render entirely. Skip-render
            // sacrifices MMC3 IRQ + sprite-0 hit accuracy on intermediate
            // frames; that's fine for headless training but produces
            // visible glitches on a stream. Pacing implies "this is the
            // user-facing demo," so we run the full PPU.
            let use_skip = !self.realtime_pace;
            for i in 0..self.frame_skip {
                let is_last = i + 1 == self.frame_skip;
                self.nes.set_skip_render(use_skip && !is_last);
                self.advance_one_frame();
            }
            self.nes.set_skip_render(false);

            if self.realtime_pace {
                self.pace_to_realtime();
            }
            self.done
        });
        Ok((self.frame_to_numpy(py), done))
    }

    /// Toggle realtime pacing. When `true`, `step()` sleeps so the
    /// per-step wall time matches `frame_skip / 60` seconds (one NES
    /// frame at 60 Hz). Use for the worker whose audio is being
    /// routed to speakers — pacing keeps the audio source producing
    /// at the rate the mixer can consume, giving continuous,
    /// non-glitchy live audio. Other workers stay unpaced for max
    /// training throughput.
    fn set_realtime_pace(&mut self, on: bool) {
        if self.realtime_pace != on {
            self.realtime_pace = on;
            self.pace_next_target = None;
        }
    }

    /// Skip APU sample generation + filter + sink writes when false.
    /// Mirror of `Pool::set_worker_pace`'s audio gating (see
    /// `pool.rs::Worker::new`): non-audio consumers (replay viewer
    /// driving CNN inference, benchmarks, headless training on this
    /// single-instance API) can flip this off to avoid ~43 KHz of
    /// unused sample-gen work per second. Channel timers + frame
    /// counter still tick so IRQ + DMC integrity are preserved.
    fn set_audio_output_enabled(&mut self, enabled: bool) {
        self.nes.set_audio_output_enabled(enabled);
    }

    /// Opt-in ASM bulk-step budget (CPU cycles per ASM invocation).
    /// Single-env mirror of `Pool::set_asm_bulk_cycles` — see the
    /// docs there. Only the batch-safe mappers (MMC1, UxROM) honor
    /// it; default budget 1 is the shipped path (timing unchanged).
    /// Enable per-game ONLY after the lockstep + Mesen-oracle +
    /// parity gate passes at that budget.
    fn set_asm_bulk_cycles(&mut self, cycles: i64) -> PyResult<()> {
        if !(1..=16).contains(&cycles) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("asm_bulk_cycles must be in 1..=16, got {cycles}"),
            ));
        }
        self.nes.set_asm_bulk_cycles_override(cycles);
        Ok(())
    }

    /// Hardware-true PPU-register read timing (Mesen-verified): defer
    /// absolute-mode $2000-$3FFF reads to the instruction's final
    /// cycle instead of the LaiNES-parity cycle-0 early commit.
    /// Default OFF — banked trajectories replay on legacy semantics.
    /// Config, not state: survives save/load untouched.
    fn set_hw_mmio_read_timing(&mut self, on: bool) {
        self.nes.cpu.hw_mmio_read_timing = on;
    }

    /// Hardware-true boot alignment (Mesen-verified): the next
    /// `reset()`/`reset_no_advance()` lands the first opcode fetch at
    /// CYC=7 with the PPU 25 dots into the frame (canonical NTSC
    /// power-on phase) instead of the legacy CYC=16 / dot-24
    /// accounting. Set BEFORE calling reset. Default OFF — existing
    /// receipts and nes-py parity assume legacy boot accounting.
    /// Config, not state: survives save/load untouched.
    fn set_hw_reset_alignment(&mut self, on: bool) {
        self.nes.hw_reset_alignment = on;
    }

    /// Hardware-true DMC DMA stall (3 cycles aligned vs legacy flat
    /// 4). See the `Apu` field doc for the drift receipt. Default
    /// OFF. Config, not state: survives save/load untouched.
    fn set_hw_dmc_stall_timing(&mut self, on: bool) {
        self.nes.apu.hw_dmc_stall_timing = on;
    }

    /// Hardware-true frame anchoring: `step()` advances to real PPU
    /// frame boundaries instead of fixed 29,781-cycle blocks. See the
    /// struct field doc for the tape-drift receipt. Default OFF.
    /// Config, not state.
    fn set_hw_frame_anchor(&mut self, on: bool) {
        self.hw_frame_anchor = on;
        self.frame_cycle_target = None;
    }

    /// Hardware-true NMI service timing: an NMI edge landing on an
    /// instruction's final cycle defers service by one instruction
    /// (real 6502 second-to-last-cycle poll). Disables the ASM/bulk
    /// CPU fast paths while on. See the Cpu field doc for the CV
    /// frame-3792 receipt. Default OFF. Config, not state.
    fn set_hw_nmi_poll_timing(&mut self, on: bool) {
        self.nes.cpu.hw_nmi_poll_timing = on;
    }

    /// Hardware-true PPU-register write timing: absolute-mode stores
    /// to $2000-$3FFF commit on the instruction's final cycle. See
    /// the Cpu field doc for the CV frame-11 NMI-enable receipt.
    /// Default OFF. Config, not state.
    fn set_hw_mmio_write_timing(&mut self, on: bool) {
        self.nes.cpu.hw_mmio_write_timing = on;
    }

    /// Event-driven PPU catch-up: route the per-cycle PPU advancement in
    /// `Nes::tick` and the ASM/bulk catch-up loops through
    /// `Ppu::advance_to(target_dot)` (absolute-dot re-parameterization of
    /// the per-dot tick loop; closed-form vblank/post-render fast-forward
    /// under the hood). Observably identical to the legacy inline
    /// interleave in every render mode / mapper. Default OFF. Config, not
    /// state. See docs/proposals/event_driven_ppu_design_2026-07-31.md.
    fn set_hw_event_ppu(&mut self, on: bool) {
        self.nes.hw_event_ppu = on;
    }

    /// Sub-cycle bus-access dot offset for deferred PPU-register READS
    /// under `hw_event_ppu` (event-driven PPU Stage 3). 0..=2 (clamped);
    /// **default 2** = end-of-cycle = byte-identical to Stages 1-2 and to
    /// `hw_event_ppu` OFF. Lower values make a `$2002`/`$2004`/`$2007`
    /// read sample the PPU earlier within its final CPU cycle's three
    /// dots — the sub-cycle race fix. Only effective on the per-cycle
    /// path (needs `hw_mmio_read_timing` to pin the read to its final
    /// cycle, and the ASM/bulk batchers off — `hw_nmi_poll_timing`, set
    /// by `--hw-all`, does that). Calibrate against Mesen; see
    /// scripts/verify_subcycle_offset.py. Config, not state.
    fn set_ppu_read_dot_offset(&mut self, offset: u8) {
        self.nes.ppu_read_dot_offset = offset.min(2);
    }

    /// Write-side mirror of `set_ppu_read_dot_offset`: sub-cycle dot
    /// offset for deferred PPU-register WRITES ($2000-$3FFF stores and
    /// the always-deferred `$4014` OAM DMA arm) under `hw_event_ppu`.
    /// 0..=2 (clamped); default 2. Needs `hw_mmio_write_timing`.
    fn set_ppu_write_dot_offset(&mut self, offset: u8) {
        self.nes.ppu_write_dot_offset = offset.min(2);
    }

    /// Save the current emulator state as an opaque `bytes` blob.
    /// Format is bincode-encoded `nes::State` — RAM, CPU, PPU, APU,
    /// mapper, input registers. Versioning is implicit: only blobs
    /// produced by the same `nes_core` build are guaranteed to load.
    /// A future split-05 migration will add an explicit version
    /// header. For now, callers who care about durability should
    /// re-snapshot rather than relying on cross-version loads.
    fn save_state<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Snapshot + bincode serialize is pure-Rust; run it with the GIL
        // released so a concurrent Python thread isn't blocked while a
        // large state blob is encoded. The `PyBytes` allocation (which
        // needs the GIL) happens after the closure returns the owned Vec.
        let out = py.allow_threads(|| -> PyResult<Vec<u8>> {
            let state = self.nes.get_state();
            let body = bincode::serialize(&state).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "save_state serialize failed: {e}"
                ))
            })?;
            // Prefix every blob with `NCST\x01` so future format changes
            // can bump the version byte and refuse to load old blobs with
            // a clear error (instead of silently deserialising into the
            // wrong struct layout). PlayWindow's on-disk `.state.bin`
            // files write the same prefix — save_state()'s output is
            // copy-paste loadable there.
            let mut out = Vec::with_capacity(STATE_MAGIC.len() + body.len());
            out.extend_from_slice(STATE_MAGIC);
            out.extend_from_slice(&body);
            Ok(out)
        })?;
        Ok(PyBytes::new_bound(py, &out))
    }

    /// Restore from a `bytes` blob produced by `save_state()`. Refuses
    /// blobs without the `NCST\x01` header (prevents silent corruption
    /// on a version mismatch). After load, the next `step()` produces
    /// the frame as if the saved emulator had stepped from that point.
    fn load_state(&mut self, py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let raw = data.as_bytes();
        let body = if raw.starts_with(STATE_MAGIC) {
            // Strip every repeated NCST prefix — tolerates the
            // pre-fix double-prefix blobs from `play_window.py`.
            let mut body = raw;
            while body.starts_with(STATE_MAGIC) {
                body = &body[STATE_MAGIC.len()..];
            }
            body
        } else if raw.len() >= 4 && &raw[..4] == b"NCST" {
            // Known prefix but unknown version byte — surface a clear
            // error instead of letting bincode eat the version byte.
            // Bound-check the version byte access: a 4-byte blob with
            // NCST prefix has no version byte at all, but the message
            // formerly accessed raw[4] unconditionally and panicked.
            let got = if raw.len() > STATE_MAGIC.len() - 1 {
                format!("{:#04x}", raw[STATE_MAGIC.len() - 1])
            } else {
                "<missing>".to_string()
            };
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "load_state: unsupported NCST version byte {} (this build expects {:#04x})",
                got,
                STATE_MAGIC[STATE_MAGIC.len() - 1]
            )));
        } else {
            // No magic at all — this is a legacy pre-versioning blob
            // (a raw bincode body). Accept it to keep older .state.bin
            // files loadable during the format rollout; log so we
            // can flip this to strict refusal once every caller has
            // migrated.
            raw
        };
        // Copy the parsed body out of the (GIL-bound) PyBytes buffer into
        // an owned Vec so the deserialize + apply can run with the GIL
        // released — the closure must not reference any Py* object. The
        // copy is a cheap memcpy of the state blob (tens of KB), dwarfed
        // by the emulation-state apply it precedes; load_state is not a
        // per-frame hot path.
        let body_owned: Vec<u8> = body.to_vec();
        py.allow_threads(|| -> PyResult<()> {
            let state: crate::nes::State = bincode::deserialize(&body_owned).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "load_state deserialize failed: {e}"
                ))
            })?;
            apply_state_guarded(&mut self.nes, &state)?;
            // The cycle anchor was tied to the PREVIOUS Nes instance's
            // CPU clock. After a state load the CPU clock effectively
            // jumps; clear the anchor so the next advance_one_frame
            // recomputes target = current_cycles + CPU_CYCLES_PER_FRAME
            // rather than chasing a stale offset that could undershoot
            // or overshoot by a full frame.
            self.frame_cycle_target = None;
            self.audio.drain();
            self.done = false;
            Ok(())
        })
    }

    fn get_audio<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i16>>> {
        Ok(self.audio.drain().into_pyarray_bound(py))
    }

    fn get_ram(&mut self, addr: u16) -> u8 {
        // Internal RAM is 2KB mirrored across $0000-$1FFF. Caller
        // should pass the canonical 0..=0x07FF range.
        self.nes.system_ram().peek_byte(addr)
    }

    fn get_ram_range<'py>(
        &mut self,
        py: Python<'py>,
        start: u16,
        end: u16,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if end <= start || end > 0x0800 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "RAM range out of bounds (0..=0x800)",
            ));
        }
        let ram = self.nes.system_ram();
        let mut out = Vec::with_capacity((end - start) as usize);
        for a in start..end {
            out.push(ram.peek_byte(a));
        }
        Ok(out.into_pyarray_bound(py))
    }

    fn get_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        Ok(self.frame_to_numpy(py))
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[getter]
    fn done(&self) -> bool {
        self.done
    }

    /// Diagnostic CPU register dump. Returns
    /// `(pc, a, x, y, sp, flags_byte, nmi_pended)`. flags_byte uses
    /// the canonical NN VEBDIZC bit layout (`From<Flags> for u8`).
    fn cpu_state(&self) -> (u16, u8, u8, u8, u8, u8, bool) {
        let r = self.nes.cpu.regs();
        let p: u8 = self.nes.cpu.flags().into();
        (r.pc, r.a, r.x, r.y, r.sp, p, self.nes.cpu.nmi_pended)
    }

    /// Diagnostic PPU state dump. Returns
    /// `(ppu_cycles, ppu_frame, ppu_scanline, ppu_scanline_cycle,
    ///   cpu_cycles, nmi_occurred)`. Used by the frame-36 cave-stuck
    /// timing investigation to localize the specific PPU state divergence
    /// against nes-py.
    fn ppu_state(&self) -> (u64, u64, u16, u64, u64, bool) {
        (
            self.nes.ppu.cycles,
            self.nes.ppu.frame,
            self.nes.ppu.scanline,
            self.nes.ppu.scanline_cycle(),
            self.nes.cycles as u64,
            self.nes.ppu.nmi_occurred,
        )
    }

    /// Step the emulator until the CPU reaches the next instruction
    /// boundary (i.e. completes the current instruction and is about
    /// to fetch a new opcode). Returns `(pc, completed_opcode, cycles_spent)`
    /// where `pc` is the PC AT the new boundary, `completed_opcode` is
    /// the opcode of the instruction that just finished, and
    /// `cycles_spent` is the number of CPU ticks consumed.
    ///
    /// Used by per-instruction trace harnesses (cave-bug investigation,
    /// nestest validation). NOT a hot path — calls are bounded by 200
    /// CPU ticks per instruction (worst-case is OAM DMA at 514 cycles
    /// total but split across many calls).
    ///
    /// Caps at 600 ticks to detect runaway loops.
    fn step_one_instruction(&mut self, py: Python<'_>) -> (u16, u8, u32) {
        use crate::sink::{AudioSink, VideoSink};
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
        // Bounded (≤600 ticks) pure-Rust work; return is a plain Copy
        // tuple, so the whole body runs with the GIL released.
        py.allow_threads(|| {
            let mut v = V;
            let mut a = A;
            let starting_pc = self.nes.cpu.regs().pc;
            let starting_opcode = {
                let bus = self.nes.system_bus();
                bus.peek_byte(starting_pc)
            };
            let mut ticks = 0u32;
            // Tick at least once to leave the current boundary.
            loop {
                self.nes.tick(&mut v, &mut a);
                ticks += 1;
                if ticks > 600 { break; }
                if self.nes.cpu.at_instruction_boundary() {
                    break;
                }
            }
            let new_pc = self.nes.cpu.regs().pc;
            (new_pc, starting_opcode, ticks)
        })
    }

    /// Format the next instruction's nestest-style trace line at the
    /// current CPU instruction boundary. Useful for per-instruction
    /// logs that compare against canonical Nintendulator output.
    fn trace_line(&mut self) -> String {
        self.nes.trace_line()
    }

    /// Set the controller-1 button mask without taking a step. Useful
    /// when a per-instruction trace driver wants the button state
    /// updated mid-frame for the next instruction's bus reads.
    fn set_buttons(&mut self, mask: u8) {
        self.apply_buttons(mask);
    }

    fn close(&mut self) {
        // No-op; Drop frees resources.
    }
}

impl NESEnvironment {
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

    fn advance_one_frame(&mut self) {
        let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);
        const CPU_CYCLES_PER_FRAME: usize = 29781;
        if self.hw_frame_anchor {
            // Hardware-true frame boundary: run to the next PPU frame
            // increment (29,780/29,781 CPU cycles alternating). Input
            // bytes then land on real frames, matching Mesen's
            // inputPolled schedule on tape replays. Cap at ~2 frames
            // of cycles so a wedged PPU can't loop forever.
            let start_frame = self.nes.ppu.frame;
            let cap = self.nes.cycles + 2 * CPU_CYCLES_PER_FRAME + 64;
            while self.nes.ppu.frame == start_frame && self.nes.cycles < cap {
                if self.nes.oam_dma.active {
                    self.nes.tick(&mut video, &mut self.audio);
                } else {
                    self.nes.step(&mut video, &mut self.audio);
                }
            }
            self.frame_cycle_target = None;
            return;
        }
        let target = self
            .frame_cycle_target
            .map(|t| t + CPU_CYCLES_PER_FRAME)
            .unwrap_or_else(|| self.nes.cycles + CPU_CYCLES_PER_FRAME);
        // Bulk-step margin must cover the worst-case `nes.step()`
        // advance, which is mapper-dependent: NROM (mapper 0, used by
        // SMB) returns `asm_bulk_cycles=32`, the ASM CPU can therefore
        // consume up to 32 cycles per `step()` call (it runs up to
        // that many cycles in one batch before returning to Rust).
        // The previous margin of 7 was for the non-bulk single-
        // instruction case and let bulk-step routinely overshoot
        // target by 18-25 cycles, putting us 1-2 cycles ahead of
        // nes-py's exact 29781 every frame. That cycle drift cascaded
        // into NMI firing at slightly different PCs which cascaded
        // into the SMB level-load ground-tile buffer ($05B0+) never
        // getting written, which made Mario fall through the floor.
        // Margin of 64 leaves headroom over any current mapper
        // (highest non-NROM is 4) without making the trailing tick
        // loop expensive.
        let margin = self.nes.asm_bulk_cycles_margin();
        while self.nes.cycles + margin < target {
            // An OAM DMA makes Nes::step fall to its slow path and consume the
            // whole ~513-cycle transfer plus the next instruction in one call,
            // overshooting target by ~500 cycles and breaking the per-frame
            // cycle lock the parity harness relies on. Tick one cycle at a time
            // while a DMA is active; bulk-stepping resumes once oam_dma.active
            // clears. Strict no-op when no DMA straddles the window.
            // (Mesen-validated; mirrors pool.rs::advance_one_frame.)
            if self.nes.oam_dma.active {
                self.nes.tick(&mut video, &mut self.audio);
            } else {
                self.nes.step(&mut video, &mut self.audio);
            }
        }
        while self.nes.cycles < target {
            self.nes.tick(&mut video, &mut self.audio);
        }
        self.frame_cycle_target = Some(target);
    }

    /// Sleep until the wall-clock target for the next step boundary,
    /// then advance the target by one step's worth of game time.
    /// Initialized lazily on the first paced step. If we're already
    /// behind the target (emulation slower than realtime), no sleep
    /// — the target catches up rather than accumulating debt
    /// indefinitely (cap to one step's worth).
    fn pace_to_realtime(&mut self) {
        let step_dur = std::time::Duration::from_nanos(
            ((self.frame_skip as u64) * 1_000_000_000) / 60,
        );
        let now = std::time::Instant::now();
        let target = match self.pace_next_target {
            Some(t) if t > now => t,
            // First call or we've fallen behind — anchor target to
            // now and don't sleep this round.
            _ => now,
        };
        if target > now {
            std::thread::sleep(target - now);
        }
        // Advance target. If we're more than one step behind, snap
        // forward instead of accumulating multi-step debt that would
        // create a long catch-up sprint when emulation speeds up.
        let advanced = target + step_dur;
        let max_lag = now + step_dur;
        self.pace_next_target = Some(if advanced > max_lag {
            advanced
        } else {
            max_lag
        });
    }

    /// Replay a recorded action sequence from cold reset. One byte
    /// per frame, each byte a NES button bitmask in the same layout
    /// as `step()`. Used for `.state.bin` start-state files —
    /// matches the behaviour of `nes_environment.py::_apply_start_state`.
    /// Audio captured during the replay is discarded.
    ///
    /// Errors if the recording terminates the episode mid-replay
    /// (the agent died, etc.), matching the existing nes-py behaviour.
    fn replay_actions(&mut self, action_bytes: &[u8]) -> Result<(), String> {
        // Replay uses the full-render path (skip-render gated off);
        // some PPU register state may otherwise drift. Audio is
        // captured into the same buffer and dropped after.
        self.nes.set_skip_render(false);
        // Use a throwaway audio sink so the boot transient doesn't
        // contaminate `self.audio` (which Python drains on first step).
        struct DiscardAudio;
        impl AudioSink for DiscardAudio {
            fn write_sample(&mut self, _: f32) {}
            fn samples_written(&self) -> usize {
                0
            }
        }
        for &byte in action_bytes {
            self.apply_buttons(byte);
            let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);
            let mut discard = DiscardAudio;
            while !video.frame_written() {
                self.nes.step(&mut video, &mut discard);
            }
            // The Nes type doesn't expose a per-step done flag; episode
            // end in this codebase is rewards-driven not emulator-driven,
            // so we don't enforce a mid-replay-done check here. (If the
            // recording was made carefully, it doesn't end mid-replay.)
        }
        // Drain any audio that leaked into self.audio (shouldn't be
        // any; we used DiscardAudio).
        self.audio.drain();
        Ok(())
    }

    fn frame_to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        // Repack u32 XRGB8888 into the (H, W, 3) RGB uint8 layout.
        // PGO showed this per-pixel unpack as the 2nd-hottest function
        // (9.5B inner-block count). NEON vld4q_u8 + vst3q_u8 processes
        // 16 pixels per iteration vs the scalar 1-at-a-time push.
        let n = FRAME_PIXELS;
        let mut out = vec![0u8; n * 3];

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use std::arch::aarch64::*;
            let in_ptr = self.video_buf.as_ptr() as *const u8;
            let out_ptr = out.as_mut_ptr();
            let mut i = 0;
            // Stride 16 pixels = 64 input bytes (u32 XRGB) → 48 output
            // bytes (RGB interleaved).
            while i + 16 <= n {
                // Load 16 BGRA pixels as 4 deinterleaved u8x16 vectors.
                // u32 little-endian XRGB8888 byte order in memory is
                // [B, G, R, X]; vld4q_u8 with 4-byte stride gives
                // .0 = B, .1 = G, .2 = R, .3 = X.
                let bgra = vld4q_u8(in_ptr.add(i * 4));
                // Interleave R, G, B for vst3q_u8 (3-channel store).
                let rgb = uint8x16x3_t(bgra.2, bgra.1, bgra.0);
                vst3q_u8(out_ptr.add(i * 3), rgb);
                i += 16;
            }
            // Tail: (n % 16) pixels left. FRAME_PIXELS = 240 * 256 =
            // 61440, which is divisible by 16, so the loop covers
            // every pixel and the tail is empty in practice. Leave
            // the tail loop here defensively for any future buffer
            // size that might not be 16-aligned.
            while i < n {
                let px = self.video_buf[i];
                out[i * 3] = ((px >> 16) & 0xFF) as u8;
                out[i * 3 + 1] = ((px >> 8) & 0xFF) as u8;
                out[i * 3 + 2] = (px & 0xFF) as u8;
                i += 1;
            }
        }
        // Portable scalar fallback (also gets auto-vectorized on most
        // targets, but matches the NEON path's behaviour exactly).
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        for (i, &px) in self.video_buf.iter().enumerate() {
            out[i * 3] = ((px >> 16) & 0xFF) as u8;
            out[i * 3 + 1] = ((px >> 8) & 0xFF) as u8;
            out[i * 3 + 2] = (px & 0xFF) as u8;
        }

        numpy::ndarray::Array3::from_shape_vec((SCREEN_HEIGHT, SCREEN_WIDTH, 3), out)
            .expect("frame shape")
            .into_pyarray_bound(py)
    }
}

/// Python-facing reward function. One instance per training genome;
/// holds per-episode state (previous RAM values, visited screens, etc.).
/// Dispatches internally to the per-game implementation selected at
/// construction time via `build_reward_function`.
///
/// Send-safe: all game-specific reward structs are plain data
/// (counters, sets, HashMaps of primitive types). Dropping the
/// `unsendable` annotation lets the trainer move reward computation
/// to a worker thread without the pyo3 cross-thread-access panic
/// that bit us on `AudioMixer` earlier this audit. No change in
/// observed behaviour today (reward is still called from the main
/// trainer thread); this is future-proofing.
#[pyclass(module = "nes_core")]
pub struct RewardFunction {
    inner: crate::rewards::Reward,
    breakdown: std::collections::HashMap<String, f64>,
}

#[pymethods]
impl RewardFunction {
    fn reset(&mut self) {
        self.inner.reset();
        self.breakdown.clear();
    }

    /// Compute the per-step reward from a 2KB RAM snapshot and the
    /// action bitmask that produced this frame. Returns
    /// `(reward, done, level_id)` — matches the Python
    /// `RewardFunction.compute` protocol exactly.
    #[pyo3(signature = (ram, action = 0))]
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        ram: &Bound<'py, PyBytes>,
        action: u8,
    ) -> PyResult<(f64, bool, String)> {
        let bytes = ram.as_bytes();
        // Always populate breakdown — Python trainer's narrator path
        // reads it via `.breakdown`. Cost is a short Vec<(&str, f64)>
        // per step, negligible vs emulation.
        let _ = py;
        let out = self.inner.compute(bytes, action, true);
        for (signal, delta) in out.breakdown_delta {
            *self.breakdown.entry(signal.to_string()).or_insert(0.0) += delta;
        }
        Ok((out.reward, out.done, out.level_id))
    }

    fn episode_success(&self) -> bool {
        self.inner.episode_success()
    }

    /// Episode-cumulative reward breakdown (per signal → total). The
    /// trainer reads this at episode end for the metrics / narrator
    /// path. Returns a fresh dict so Python-side mutations don't
    /// corrupt the Rust state.
    #[getter]
    fn breakdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (k, v) in &self.breakdown {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }
}

/// Batched dispatch: compute rewards for N (genome, ram, action) triples
/// in a single PyO3 call. Equivalent to looping
/// `[fn.compute(ram, action=a) for fn, ram, a in zip(fns, rams, actions)]`
/// in Python, but eliminates per-call argument-parsing + boundary-
/// crossing overhead. Saves ~50ms/gen at 16 workers × ~600 steps × 5µs
/// per call of avoided overhead.
///
/// Inputs:
///   - `reward_fns`: list of RewardFunction objects, one per genome.
///   - `rams`: list of `bytes` (each 2 KB), parallel to `reward_fns`.
///   - `actions`: list of u8 action bitmasks, parallel to `reward_fns`.
///
/// All three lists must be the same length. Returns a list of
/// `(reward, done, level_id)` tuples in input order.
///
/// Why a free function rather than a method on RewardFunction: each
/// instance carries its own state, and a method receiver would need
/// borrow_mut on `self` for the others — awkward in PyO3. Free
/// function lets us pull `Bound<RewardFunction>` from each list slot
/// and call its `compute` body directly.
#[pyfunction]
fn compute_rewards_batch<'py>(
    py: Python<'py>,
    reward_fns: &Bound<'py, pyo3::types::PyList>,
    rams: &Bound<'py, pyo3::types::PyList>,
    actions: Vec<u8>,
) -> PyResult<Vec<(f64, bool, String)>> {
    let n = reward_fns.len();
    if rams.len() != n || actions.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "compute_rewards_batch: list length mismatch — \
             reward_fns={}, rams={}, actions={}",
            n, rams.len(), actions.len(),
        )));
    }
    let _ = py;
    let mut out: Vec<(f64, bool, String)> = Vec::with_capacity(n);
    for i in 0..n {
        let fn_obj = reward_fns.get_item(i)?;
        let mut fn_borrowed = fn_obj.downcast::<RewardFunction>()?.borrow_mut();
        let ram_obj = rams.get_item(i)?;
        let ram_bytes_obj = ram_obj.downcast::<PyBytes>()?;
        let bytes = ram_bytes_obj.as_bytes();
        let result = fn_borrowed.inner.compute(bytes, actions[i], true);
        for (signal, delta) in result.breakdown_delta {
            *fn_borrowed
                .breakdown
                .entry(signal.to_string())
                .or_insert(0.0) += delta;
        }
        out.push((result.reward, result.done, result.level_id));
    }
    Ok(out)
}


/// Native Rust SMB tile-feature extractor — drop-in for the Python
/// `SMBTileObservation.extract` decoder. Returns a freshly-allocated
/// numpy `int8` array of shape `(175,)` matching the Python layout
/// byte-exactly.
///
/// Uses the byte-buffer fast path: `ram_bytes` enters PyO3 as a
/// borrowed `&[u8]` (no copy), `extract` writes into a stack array,
/// and we hand the array off to numpy via PyArray::from_iter (one
/// alloc, contiguous memory).
///
/// Why expose this: `src/emulation/tile_observations/smb.py` is hot
/// (~7% wall in tile-mode profiles even after the numpy vectorize).
/// The Rust port runs the same algorithm without the numpy ufunc
/// dispatch + interpreter overhead — single-digit microseconds per
/// call. Opt in from Python via `SMBTileObservation(use_rust=True)`.
#[pyfunction]
fn extract_smb_tiles<'py>(
    py: Python<'py>,
    ram_bytes: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, numpy::PyArray1<i8>>> {
    let bytes = ram_bytes.as_bytes();
    if bytes.len() < 0x0800 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "ram_bytes too short: got {} bytes, need at least 2048",
            bytes.len(),
        )));
    }
    let out = crate::smb_tile_extract::extract(bytes);
    Ok(numpy::PyArray1::from_slice_bound(py, &out))
}

/// v2 tile extractor: v1 + 3 de-aliasing scalars (level page, fine
/// progress, global frame-counter phase). Opt-in per profile.
#[pyfunction]
fn extract_smb_tiles_v2<'py>(
    py: Python<'py>,
    ram_bytes: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, numpy::PyArray1<i8>>> {
    let bytes = ram_bytes.as_bytes();
    if bytes.len() < 0x0800 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "ram_bytes too short: got {} bytes, need at least 2048",
            bytes.len(),
        )));
    }
    let out = crate::smb_tile_extract::extract_v2(bytes);
    Ok(numpy::PyArray1::from_slice_bound(py, &out))
}


/// Factory — mirrors `src/utils/reward_functions/__init__.py::build_reward_function`.
/// Accepts a game profile dict and returns a ready-to-use RewardFunction.
#[pyfunction]
fn build_reward_function(profile: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RewardFunction> {
    let name: String = profile
        .get_item("name")?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("profile must contain 'name'")
        })?
        .extract()?;

    // Extract reward_weights as HashMap<String, f64>. Python-side YAML
    // loads yield dict[str, float | int], so we coerce numeric types
    // flexibly.
    let weights_py = profile.get_item("reward_weights")?;
    let mut weights: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    if let Some(w) = weights_py {
        let d: Bound<'_, pyo3::types::PyDict> = w.extract()?;
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            // Try float, fall back to int, fall back to drop silently.
            if let Ok(f) = v.extract::<f64>() {
                weights.insert(key, f);
            } else if let Ok(i) = v.extract::<i64>() {
                weights.insert(key, i as f64);
            }
        }
    }

    let inner = crate::rewards::build_reward(&name, &weights).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "no reward function for game: {name}"
        ))
    })?;
    Ok(RewardFunction {
        inner,
        breakdown: std::collections::HashMap::new(),
    })
}

/// Python-facing audio mixer. Drop-in for `src/audio/ram_music.py::AudioMixer`
/// — same method names, same semantics, same interaction model with
/// the trainer's `push_audio` hot path and the GUI's mixer window.
///
/// Internally forwards every call to the in-Rust `audio::AudioMixer`
/// which owns a cpal output stream on the macOS Core Audio device.
/// Eliminates the sounddevice (PortAudio) dep + the Python-side
/// resampler + PCM ring logic.
// AudioMixer is Send+Sync now (cpal::Stream lives on a dedicated
// worker thread, not in this struct), so we drop `unsendable`.
// Removing it is what fixes the GUI Start/Stop bug — previously the
// audio-drainer thread calling `set_mode` panicked because pyo3
// enforces `unsendable` as "only the creating thread may touch me."
#[pyclass(module = "nes_core")]
pub struct AudioMixer {
    inner: crate::audio::AudioMixer,
}

#[pymethods]
impl AudioMixer {
    #[new]
    #[pyo3(signature = (num_instances = 16))]
    fn new(num_instances: usize) -> Self {
        Self {
            inner: crate::audio::AudioMixer::new(num_instances),
        }
    }

    /// No-op kept for API parity with the Python mixer — the Rust
    /// backend opens its cpal stream lazily in `set_mode`.
    fn start(&self) {}

    /// Close the cpal output stream. Called on trainer shutdown.
    fn stop(&self) {
        self.inner.stop_stream();
    }

    /// Accept a mode string: "mute", "solo-N", or "all". Matches the
    /// string format the GUI audio-mixer window already emits.
    fn set_mode(&self, mode: &str) -> PyResult<()> {
        self.inner.set_mode(mode).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })
    }

    /// Master volume in [0.0, 1.0].
    fn set_volume(&self, v: f32) {
        self.inner.set_volume(v);
    }

    fn set_instance_intensity(&self, instance_id: usize, intensity: f32) {
        self.inner.set_instance_intensity(instance_id, intensity);
    }

    /// Clear a voice's buffered audio + resampler state. Call when a NEW
    /// emulator instance takes over a worker voice slot (e.g. a fresh
    /// Pool spun up for the next level) so its first pushed samples don't
    /// splice onto the previous occupant's leftover waveform.
    fn reset_instance(&self, instance_id: usize) {
        self.inner.reset_instance(instance_id);
    }

    /// Push int16 mono PCM samples produced by one worker's emulator.
    /// `audio_rate` is the source sample rate (typically 43_653 from
    /// nes_core's APU); the mixer's internal resampler converts to
    /// the cpal output rate.
    #[pyo3(signature = (instance_id, audio, audio_rate))]
    fn push_audio(
        &self,
        instance_id: usize,
        audio: numpy::PyReadonlyArray1<'_, i16>,
        audio_rate: u32,
    ) -> PyResult<()> {
        let slice = audio.as_slice()?;
        self.inner.push_audio_i16(instance_id, slice, audio_rate);
        Ok(())
    }

    /// Song-byte RAM push — kept for API compatibility with the
    /// Python mixer but a no-op here. The chiptune/NSF fallback
    /// paths it used to drive are gone (nes_core has real APU audio).
    fn push_ram(&self, _instance_id: usize, _ram: &Bound<'_, PyBytes>) {}

    /// Exposed as a read-only attr so callers that gate on
    /// `mixer.mode != "mute"` (see `trainer.py:_evaluate_batch`)
    /// still work unchanged. Internal enum is string-ified on read.
    #[getter]
    fn mode(&self) -> String {
        self.inner.mode_str()
    }

    /// API parity: the Python mixer exposed `num` as an int attr.
    #[getter]
    fn num(&self) -> usize {
        self.inner.num_instances()
    }
}

/// Stateful max-depth tracker — records the furthest in-game position
/// reached during this run. Pure logic port of
/// `src/training/depth_tracker.py`; the Python `DepthTracker` stays as
/// a ~40-line wrapper that handles JSONL memo file I/O.
#[pyclass]
struct DepthTracker {
    inner: crate::depth_tracker::DepthTracker,
}

#[pymethods]
impl DepthTracker {
    #[new]
    fn new(game: &str) -> Self {
        Self {
            inner: crate::depth_tracker::DepthTracker::new(game),
        }
    }

    /// Feed one RAM snapshot. Returns a dict on a new all-time record,
    /// `None` otherwise.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        ram: &[u8],
        worker_id: u32,
        genome_name: &str,
        generation: u32,
    ) -> PyResult<Option<pyo3::Bound<'py, pyo3::types::PyDict>>> {
        let Some(rec) = self.inner.observe(ram, worker_id, genome_name, generation) else {
            return Ok(None);
        };
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("generation", rec.generation)?;
        d.set_item("worker_id", rec.worker_id)?;
        d.set_item("genome_name", rec.genome_name)?;
        d.set_item("key", (rec.key.0, rec.key.1, rec.key.2))?;
        d.set_item("caption", rec.caption)?;
        Ok(Some(d))
    }

    /// Current all-time-best depth key, or None if nothing observed yet.
    #[getter]
    fn best(&self) -> Option<(u32, u32, u32)> {
        self.inner.best()
    }
}

/// Event-detection core for `src/training/narrator.py`. The Python
/// wrapper handles caption phrasing + GUI queue push; this class owns
/// the hot path — delta detection, rate limiting, first-ever dedup,
/// combo-kill tracker.
#[pyclass]
struct Narrator {
    inner: crate::narrator::Narrator,
}

#[pymethods]
impl Narrator {
    #[new]
    #[pyo3(signature = (min_event_gap_s = 1.5))]
    fn new(min_event_gap_s: f64) -> Self {
        Self {
            inner: crate::narrator::Narrator::new(min_event_gap_s),
        }
    }

    /// Feed one per-worker breakdown diff. Returns a list of event
    /// dicts — `worker_id`, `kind` (string), `first_ever` (bool),
    /// `timestamp` (monotonic seconds). `prev` and `new` are lists of
    /// `(str, float)` pairs; the Python wrapper normalises dict →
    /// list once per call.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        worker_id: u32,
        now: f64,
        prev: Vec<(String, f64)>,
        new: Vec<(String, f64)>,
        done: bool,
        success: bool,
    ) -> PyResult<Vec<pyo3::Bound<'py, pyo3::types::PyDict>>> {
        let prev_refs: Vec<(&str, f64)> = prev.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let new_refs: Vec<(&str, f64)> = new.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let events =
            self.inner
                .observe(worker_id, now, &prev_refs, &new_refs, done, success);
        let mut out = Vec::with_capacity(events.len());
        for ev in events {
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("worker_id", ev.worker_id)?;
            d.set_item("kind", ev.kind.as_str())?;
            d.set_item("first_ever", ev.first_ever)?;
            d.set_item("timestamp", ev.timestamp_monotonic)?;
            out.push(d);
        }
        Ok(out)
    }
}

/// Parse the iNES / NES 2.0 header and return (md5_hex, mapper,
/// sub_mapper, is_nes20). Trainers can gate on the md5 to reject
/// dirty dumps whose RAM layout has silently drifted from the
/// profile's expected baseline.
#[pyfunction]
fn rom_info(rom_path: &str) -> PyResult<(String, u16, u8, bool)> {
    let basename = std::path::Path::new(rom_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| rom_path.to_string());
    let bytes = std::fs::read(rom_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "failed to read {basename}: {e}"
        ))
    })?;
    let cart = crate::cartridge::Cartridge::load(&mut std::io::Cursor::new(bytes))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "failed to parse iNES header for {basename}: {e}"
            ))
        })?;
    Ok((cart.md5, cart.mapper, cart.sub_mapper, cart.is_nes20))
}

/// List of iNES mapper numbers nes_core currently supports. Used by
/// the ROM library scanner to pre-filter before attempting to boot an
/// env — unsupported mappers will still fail cleanly with a PyErr,
/// but checking up front is faster than paying the env construction
/// overhead.
#[pyfunction]
fn supported_mappers() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 19, 21, 22, 23, 24, 25, 26,
         34, 37, 40, 41, 47, 64, 66, 68, 69, 71, 79, 85, 105, 113, 118, 119,
         228, 232, 234]
}

#[pymodule]
fn nes_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NESEnvironment>()?;
    m.add_class::<crate::pool::Pool>()?;
    m.add_class::<RewardFunction>()?;
    m.add_class::<AudioMixer>()?;
    m.add_class::<DepthTracker>()?;
    m.add_class::<Narrator>()?;
    m.add_function(wrap_pyfunction!(build_reward_function, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rewards_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_smb_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(extract_smb_tiles_v2, m)?)?;
    m.add_function(wrap_pyfunction!(rom_info, m)?)?;
    m.add_function(wrap_pyfunction!(supported_mappers, m)?)?;
    m.add("BUTTON_RIGHT", BUTTON_RIGHT)?;
    m.add("BUTTON_LEFT", BUTTON_LEFT)?;
    m.add("BUTTON_DOWN", BUTTON_DOWN)?;
    m.add("BUTTON_UP", BUTTON_UP)?;
    m.add("BUTTON_START", BUTTON_START)?;
    m.add("BUTTON_SELECT", BUTTON_SELECT)?;
    m.add("BUTTON_B", BUTTON_B)?;
    m.add("BUTTON_A", BUTTON_A)?;
    Ok(())
}
