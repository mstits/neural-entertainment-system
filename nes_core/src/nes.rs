use crate::apu::{self, Apu};
use crate::cartridge::Cartridge;
use crate::cpu::Cpu;
use crate::game_genie::Cheat;
use crate::input::{GamePad, Input};
use crate::mapper::{Mapper, MapperEnum};
use crate::memory::{Memory, Ram};
use crate::oam_dma::OamDma;
use crate::ppu::{self, Vram};
use crate::ppu::{OAMDATA_ADDRESS, Ppu};
use crate::sink::*;
use crate::system_bus::SystemBus;
use crate::{cpu, input, mapper};

use serde_derive::{Deserialize, Serialize};

use std::collections::HashMap;

pub struct Nes {
    ram: Ram,
    pub mapper: MapperEnum,
    pub cpu: Cpu,
    pub ppu: Ppu,
    pub apu: Apu,
    pub oam_dma: OamDma,
    pub input: Input,
    cheats: HashMap<u16, Cheat>,
    pub cycles: usize,

    // Debugging
    pub trace: bool,

    /// Runtime opt-out from the AArch64 ASM CPU fast path. When set,
    /// `Nes::step` skips the ASM dispatch and falls through to
    /// `try_bulk_step` / per-cycle slow path. Pool workers may set
    /// this if benching shows the ASM path wastes L1i / vector
    /// register file at parallel scale (12+ workers). Default
    /// `false` so single-env GUI / tests keep the ASM win.
    pub disable_asm_cpu: bool,

    /// Hardware-true boot alignment (config, not savestate). When set,
    /// `reset()` lands the first opcode fetch at CYC=7 with the PPU 25
    /// dots into the frame — Mesen's canonical NTSC power-on phase —
    /// instead of the legacy accounting (construction + reset each
    /// discarding 8 cycles → first fetch at CYC=16, PPU dot 24). The
    /// legacy path leaves an ODD cycle-counter offset vs Mesen, which
    /// flips OAM-DMA 513/514 alignment parity at every DMA and makes
    /// long input-tape replays drift (receipted: CV tape desyncs at
    /// ~frame 3992 via a sprite-0 poll flip). Default `false`: nes-py
    /// parity suites and all existing solver receipts assume legacy
    /// boot accounting.
    pub hw_reset_alignment: bool,

    /// Event-driven PPU catch-up (config, not savestate). When set, the
    /// per-cycle PPU advancement in `Nes::tick` and the ASM/bulk catch-up
    /// loops route through `Ppu::advance_to(target_dot)` — an absolute-dot
    /// re-parameterization of the per-dot `tick` loop — instead of the
    /// inline `3× ppu.tick` / `tick_three` sequences. Stage 1 is
    /// equal-by-construction (the routed path is the same tick body);
    /// Stage 2 lets `advance_to` fast-forward event-free dot ranges in
    /// closed form (vblank/post-render lines) via the proven `advance`
    /// batcher. Default `false`: every legacy path stays byte-identical
    /// (the parity suites + banked receipts assume the inline interleave).
    /// See docs/proposals/event_driven_ppu_design_2026-07-31.md.
    pub hw_event_ppu: bool,

    /// Sub-cycle bus-access dot offsets (event-driven PPU Stage 3;
    /// config, not savestate). A 6502 bus access resolves on the LAST
    /// cycle of the memory phase; in our 3-dots-per-CPU-cycle model that
    /// cycle spans PPU dot-slots 0, 1, 2. These pick which slot a
    /// *deferred* PPU-register READ (`ppu_read_dot_offset`) or WRITE
    /// (`ppu_write_dot_offset`) samples the PPU at: when `hw_event_ppu`
    /// is on, `Nes::tick` advances `offset + 1` of the cycle's three dots
    /// BEFORE `cpu.tick` services the access, then the remaining
    /// `2 - offset` dots after — so `$2002` polls / `$2000` NMI-enables
    /// land at the exact intra-instruction dot instead of the end of the
    /// CPU cycle. Range 0..=2, **default 2** = end-of-cycle = the shipped
    /// model (receipt: 93% of accesses already land at `abs_dot%3==2`),
    /// so the default is byte-identical to Stages 1-2 and to `hw_event_ppu`
    /// OFF. Only effective on the per-cycle path: the access must be pinned
    /// to its final cycle by `hw_mmio_read_timing` / `hw_mmio_write_timing`
    /// (always-deferred for `$4014`), and that final-cycle pin is only
    /// reached per-cycle when the ASM/bulk batchers are off — which
    /// `hw_nmi_poll_timing` (set by `--hw-all`) does. Calibrate against a
    /// Mesen master-clock trace post-rebuild; see
    /// scripts/verify_subcycle_offset.py and §3.4 of the design doc.
    pub ppu_read_dot_offset: u8,
    pub ppu_write_dot_offset: u8,

    /// Hardware-true NMI-line sample phase (config, not savestate).
    /// The physical 6502 samples its NMI input on φ2 — 2 PPU dots into
    /// each 3-dot CPU cycle — while the bulk 3:1 interleave lets the
    /// sample after the cycle's LAST dot reach the edge latch, so a
    /// dot-2 edge becomes CPU-visible one cycle early. When set,
    /// `Nes::tick` suppresses the `set_nmi_line` call after dot
    /// index 2 (0-indexed); the edge is then picked up by the NEXT
    /// cycle's dot-0 sample — the φ2 point, the NMI-pin manifestation
    /// of `ppu_read_dot_offset = 1`. External research round
    /// (2026-08-06) root-caused the residual ±2-3-cycle NMI-taken
    /// quantization vs Mesen (the CV-tape fork past frame 4884) to
    /// the end-of-cycle sample. Forces the per-cycle path: the
    /// ASM/bulk batchers sample the line once per batch and cannot
    /// place a mid-cycle latch. This flag moves WHEN an edge becomes
    /// visible AND, when on, which boundary snapshot the service poll
    /// consumes: the live `nmi_pended` — which under the φ2 latch is
    /// exactly the pended state as of φ2 of the FINAL cycle, the
    /// snapshot Mesen's before-every-bus-transaction catch-up reads at
    /// its end-of-instruction check. Stacking the second-to-last-cycle
    /// `nmi_poll_latch` on top of the φ2 latch double-quantizes —
    /// measured (2026-08-07) as a constant +1-cycle NMI-entry offset
    /// vs Mesen with anti-phase OAM-DMA start parities from CV tape
    /// frame 3423, the feedback loop behind the frame-3435 state fork
    /// — so `poll_interrupts` bypasses the latch while this flag is
    /// set; `cpu.hw_nmi_poll_timing`'s latch convention applies only
    /// without it. Standalone (poll timing off) the φ2 latch still
    /// defers a dot-2 final-cycle edge by one instruction vs legacy.
    /// Default OFF: byte-identical to the shipped interleave.
    pub hw_nmi_subcycle_phase: bool,

    /// Cached `mapper.prg_asm_ptr()` result. Stable for the mapper's
    /// lifetime — all mapper PRG-ASM windows are fixed-size 32 KB
    /// `Vec<u8>`s mutated only via slice indexing (never resized),
    /// so the underlying allocation pointer doesn't move on bank
    /// switches. Refreshed at construction + reset + apply_state.
    /// Bypasses the per-Nes::step `MapperEnum` virtual dispatch
    /// (PGO showed 366M calls/run).
    cached_prg_asm_ptr: Option<*const u8>,

    /// Cached `mapper.asm_bulk_cycles()` budget. A per-mapper constant
    /// (NROM = 64, most others = 1) that only ever changes through
    /// `set_asm_bulk_cycles_override` — never per-step or on bank
    /// switches — so caching it lets `Nes::step` skip a `MapperEnum`
    /// virtual call on every instruction. Refreshed at construction +
    /// reset + apply_state + override, exactly like `cached_prg_asm_ptr`.
    cached_asm_bulk_cycles: i64,

    /// Coarse "this mapper could be batchable by `Ppu::advance`" hint
    /// (event-driven-PPU campaign, Rung 1). `mapper.chr_static_ptr()
    /// .is_some() && !mapper.uses_scanline_irq()` sampled at the mapper's
    /// current banking. Purely a perf gate on the catch-up loops: when
    /// `false` (MMC3/MMC5 scanline-IRQ mappers, MMC2/MMC4 CHR-latch
    /// mappers, and MMC1 while its static window is inactive) the loops
    /// run the verbatim `tick_three` path with zero `advance` overhead.
    /// Correctness never rests on this: `Ppu::advance` re-checks the live
    /// mapper every call and self-selects the reference path, so a stale
    /// hint only ever costs a batching opportunity, never fidelity.
    /// Refreshed alongside `cached_asm_bulk_cycles`.
    cached_ppu_batchable: bool,
}

// SAFETY: cached_prg_asm_ptr aliases memory owned by the same Nes
// (the mapper's prg_asm_window Vec). Pool workers each own their
// own Nes which Rayon ships across threads.
unsafe impl Send for Nes {}
unsafe impl Sync for Nes {}

#[derive(Deserialize, Serialize)]
pub struct State {
    #[serde(with = "serde_bytes")]
    pub ram: Vec<u8>,
    pub mapper: mapper::State,
    pub cpu: cpu::State,
    pub ppu: ppu::State,
    pub apu: apu::State,
    pub input: input::State,
    pub cycles: usize,
}

impl Nes {
    pub fn new(cartridge: Cartridge) -> Nes {
        let mut nes = Self {
            ram: Ram::new(),
            mapper: MapperEnum::from_cartridge(cartridge),
            cpu: Cpu::new(),
            ppu: Ppu::new(),
            apu: Apu::new(),
            oam_dma: OamDma::new(),
            input: Input::new(),
            cheats: HashMap::new(),
            cycles: 0,
            trace: false,
            disable_asm_cpu: false,
            hw_reset_alignment: false,
            hw_event_ppu: false,
            ppu_read_dot_offset: 2,
            ppu_write_dot_offset: 2,
            hw_nmi_subcycle_phase: false,
            cached_prg_asm_ptr: None,
            cached_asm_bulk_cycles: 1,
            cached_ppu_batchable: false,
        };
        nes.cached_prg_asm_ptr = nes.mapper.prg_asm_ptr();
        nes.cached_asm_bulk_cycles = nes.mapper.asm_bulk_cycles();
        nes.cached_ppu_batchable = nes.ppu_batchable();

        // Install the ASM dispatch + opcode-cycle tables once, at
        // construction, instead of on the per-instruction hot path.
        // `install_opcode_table_once` is a global `Once`, so the first
        // Nes built in the process pays it and every later worker's
        // call is a single atomic-load fast return — but hoisting it
        // out of `Nes::step` removes that load from every instruction.
        // `asm_opcode_cycles` (read in `step` to decide the ASM entry)
        // depends on the cycle table this populates, so it MUST run
        // before the first step.
        #[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
        crate::cpu_asm::install_opcode_table_once();

        // Constructor-time reset initializes CPU PC / SP / flags so the
        // state is immediately usable (some harnesses poke state without
        // calling `reset()`). Python callers generally call `env.reset()`
        // right after construction, which invokes `Nes::reset()` again
        // — double-reset is idempotent for PC/SP/flags and harmless.
        // See docs/proposals/archive/zelda_cave_stuck_investigation.md for why
        // this was once suspected as a cave-stuck contributor (it isn't).
        nes.reset();

        nes
    }

    // Runs all hardware for a single CPU instruction and returns the number of cycles that
    // were run.
    pub fn step<A: AudioSink, V: VideoSink + Sized>(
        &mut self,
        video_frame_sink: &mut V,
        audio_frame_sink: &mut A,
    ) -> usize {
        let prev_cycles = self.cycles;

        if self.trace {
            let bus = SystemBus::new(
                &mut self.ram,
                &mut self.mapper,
                &mut self.ppu,
                &mut self.apu,
                &mut self.oam_dma,
                &mut self.input,
                &self.cheats,
            );
            println!("{}", self.cpu.trace(&bus));
        }

        // Bulk-step fast path — block interpreter dispatch from
        // `docs/proposals/cpu_bulk_stepping.md`. Phase 1 covers
        // LDA/LDX/LDY imm+zp, STA/STX/STY zp, TAX/TAY/TXA/TYA.
        // Phase 2 expands coverage; see `Cpu::try_bulk_step`.
        //
        // Guard-cost optimization: the full guard set (stall, OAM
        // DMA, interrupts, trace) runs only when we're at an
        // instruction boundary AND about to fetch an opcode. The
        // 95%+ of calls not at a boundary pay 2 field loads + 2
        // branches, comparable to the baseline's own checks inside
        // `Cpu::tick`.
        if self.cpu.cycle == 0
            && self.cpu.instruction.is_none()
            && !self.trace
            && !self.oam_dma.active
            && self.cpu.stall_cycles == 0
            && !self.cpu.nmi_pended
            && !(self.cpu.irq_line_low && !self.cpu.flags.i)
            // Hardware NMI poll timing needs the per-cycle latch
            // pipeline in Cpu::tick; the ASM/bulk paths batch cycles
            // and would take NMIs at the wrong boundary. Fidelity
            // lane trades speed for exactness.
            && !self.cpu.hw_nmi_poll_timing
            // φ2 NMI-line sampling needs the per-dot interleave in
            // Nes::tick: the ASM/bulk paths sample the line once at
            // batch end — coarser even than end-of-cycle — and cannot
            // place the latch 2 dots into a cycle. See the
            // `hw_nmi_subcycle_phase` field doc.
            && !self.hw_nmi_subcycle_phase
        {
            let pc = self.cpu.regs.pc;
            // Fast opcode peek. Almost all code executes from PRG
            // ROM ($8000-$FFFF) → mapper's `prg_peek_byte` is a
            // straight array index for NROM/MMC1/MMC3/etc.
            let opcode = match pc {
                0x0000..=0x1FFF => self.ram.peek_byte(pc),
                0x8000..=0xFFFF => self.mapper.prg_peek_byte(pc),
                _ => 0xFF,
            };
            // AArch64 ASM fast path — try before Rust bulk-step. Runs
            // ONE instruction through threaded-dispatch assembly, no
            // Rust per-opcode match. On unported opcodes or MMIO the
            // ASM exits cleanly and we fall through to try_bulk_step.
            #[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
            {
                let ram_ptr = self.ram.as_mut_ptr();
                // Cached at construction / reset / apply_state. The
                // underlying prg_asm_window Vec is fixed-size 32 KB
                // mutated only via slice indexing; pointer is stable.
                let prg_ptr = self.cached_prg_asm_ptr;
                // Cached per-mapper budget (see the field doc). Avoids a
                // `MapperEnum` virtual call on every instruction.
                let raw_bulk_cycles = self.cached_asm_bulk_cycles;
                // Predict NMI fire and cap the bulk so the batch ends
                // at most one instruction past the vblank rising edge
                // (matches real 6502 NMI delivery — services between
                // instructions). Without the cap, multi-instruction
                // bulks accumulate cycles past vblank without ticking
                // PPU between them, so NMI fires K instructions late
                // (~15-74k diff lines in smb_write_trace at bulk=4..32
                // before this cap). With the cap, NROM can safely run
                // bulk_cycles=16+ — the cap shrinks dynamically as
                // vblank approaches.
                //
                // At raw==1 (every MMC1/UxROM/MMC3 game — the majority
                // of the library) the cap is provably dead: its only
                // shrinking arm needs `n < 1` (i.e. n==0), which
                // `.max(1)`s straight back to 1, and every other arm
                // yields raw==1. So the batch is always exactly 1 and
                // the per-instruction `cpu_cycles_until_nmi_fire()`
                // query is discarded work — skip it. Control flow for
                // raw > 1 is byte-identical to the prior match.
                let bulk_cycles = if raw_bulk_cycles == 1 {
                    1
                } else {
                    match self.ppu.cpu_cycles_until_nmi_fire() {
                        Some(n) if (n as i64) < raw_bulk_cycles => {
                            (n as i64).max(1)
                        }
                        _ => raw_bulk_cycles,
                    }
                };
                // Only enter the ASM path if the mapper exposes a PRG
                // window AND the opcode at PC has an ASM handler. This
                // avoids ASM setup overhead when we'd immediately bail.
                let asm_handler_cycles = if prg_ptr.is_some() {
                    crate::cpu_asm::asm_opcode_cycles(opcode)
                } else {
                    0
                };
                let _ = (bulk_cycles, asm_handler_cycles);
                // Construct a bus so the ASM MMIO callback can route
                // reads/writes through the live PPU/APU/mapper. The
                // bus borrows ram/mapper/ppu/apu/oam_dma/input, so the
                // ram_ptr raw pointer aliases with the bus's ram borrow
                // — only one accesses any given byte per ASM invocation
                // (ASM uses ram_ptr for $0000-$1FFF, bus for $2000+).
                if asm_handler_cycles > 0 && !self.disable_asm_cpu {
                let mut bus = SystemBus::new(
                    &mut self.ram,
                    &mut self.mapper,
                    &mut self.ppu,
                    &mut self.apu,
                    &mut self.oam_dma,
                    &mut self.input,
                    &self.cheats,
                );
                let bus_ptr = &mut bus as *mut _ as *mut core::ffi::c_void;
                // Monomorphized tick fn + ctx: install into thread-local
                // so MMIO callback ticks PPU/APU with real sinks.
                struct SinkCtx<'v, 'a, V: crate::sink::VideoSink, A: crate::sink::AudioSink> {
                    video: &'v mut V,
                    audio: &'a mut A,
                }
                unsafe extern "C" fn tick_impl<V: crate::sink::VideoSink, A: crate::sink::AudioSink>(
                    bus_ptr: *mut core::ffi::c_void,
                    ctx_ptr: *mut core::ffi::c_void,
                    cycles: u32,
                ) {
                    let bus = unsafe { &mut *(bus_ptr as *mut SystemBus) };
                    let ctx = unsafe { &mut *(ctx_ptr as *mut SinkCtx<V, A>) };
                    bus.tick_cpu_cycles_with(cycles, ctx.video, ctx.audio);
                }
                let mut sink_ctx: SinkCtx<V, A> = SinkCtx {
                    video: video_frame_sink,
                    audio: audio_frame_sink,
                };
                crate::cpu_asm::set_asm_tick(
                    &mut sink_ctx as *mut _ as *mut core::ffi::c_void,
                    tick_impl::<V, A>,
                );
                let result = crate::cpu_asm::try_step_asm(
                    &mut self.cpu, ram_ptr, prg_ptr, Some(bus_ptr),
                    bulk_cycles,
                );
                crate::cpu_asm::clear_asm_tick();
                drop(bus);
                if let Some(r) = result {
                    // MMIO callbacks already ticked PPU/APU for
                    // `cycles_ticked_in_callback` cycles via real sinks.
                    // Remainder (non-MMIO cycles) gets ticked here.
                    //
                    // The per-cycle set_nmi_line / update_irq_line
                    // calls that used to live inside this loop are
                    // hoisted to the single end-of-loop call below.
                    // Both signals are sticky-while-asserted: NMI is
                    // a rising-edge latch on `cpu.nmi_pended` (set
                    // when `nmi_output && nmi_occurred` transitions
                    // low→high; cleared only by NMI service or PPU
                    // $2002 read), and IRQ is a level line driven by
                    // the mapper's `irq_pending()`. Since the catch-
                    // up loop processes no MMIO (no $2002 reads, no
                    // mapper register writes), neither signal can
                    // de-assert mid-loop. So the end-of-loop sample
                    // produces an identical `nmi_pended` /
                    // `irq_line_low` to the prior per-cycle pattern.
                    // The poll_interrupts call after services any
                    // pending interrupt at the next instruction
                    // boundary — exactly as before.
                    let remaining = r.cycles_consumed
                        .saturating_sub(r.cycles_ticked_in_callback);
                    // Event-driven PPU catch-up (Stage 1/2). APU stays
                    // per-cycle (audio / DMC / frame-IRQ), then the PPU
                    // fast-forwards to the absolute batch-end dot in one
                    // `advance_to` — observably identical to the
                    // interleaved `tick_three` loop (APU/PPU do not observe
                    // each other mid-batch; IRQ/NMI lines are sampled only
                    // at batch end below). Works for every render mode /
                    // mapper: `advance_to` self-selects closed-form vs the
                    // verbatim per-dot reference. Legacy branches unchanged.
                    if self.hw_event_ppu {
                        for _ in 0..remaining {
                            self.apu.tick(&mut self.mapper, audio_frame_sink);
                        }
                        let target = self.ppu.cycles + remaining as u64 * 3;
                        self.ppu.advance_to(
                            target,
                            &mut self.mapper,
                            video_frame_sink,
                        );
                    } else if self.ppu.is_skip_render()
                        && self.ppu.scanline_advance_enabled()
                        && self.cached_ppu_batchable
                    {
                        // Rung-1 scanline-granular PPU catch-up. Only under
                        // skip_render (advance omits pixel work), a
                        // batchable mapper, and the runtime gate.
                        for _ in 0..remaining {
                            self.apu.tick(&mut self.mapper, audio_frame_sink);
                        }
                        self.ppu.advance(
                            remaining as u64 * 3,
                            &mut self.mapper,
                            video_frame_sink,
                        );
                    } else {
                        for _ in 0..remaining {
                            self.apu.tick(&mut self.mapper, audio_frame_sink);
                            self.ppu.tick_three(&mut self.mapper, video_frame_sink);
                        }
                    }
                    self.cycles += r.cycles_consumed as usize;
                    self.cpu.set_nmi_line(
                        self.ppu.nmi_output && self.ppu.nmi_occurred,
                    );
                    self.update_irq_line();
                    self.cpu.poll_interrupts();
                    // NOTE: `ASM_HITS` used to be bumped here, but the atomic
                    // RMW on a single cache line serialized rayon workers
                    // catastrophically (~5% parallel efficiency at 16 workers
                    // on M4 Max). Gate the diagnostic behind a feature flag;
                    // production builds skip the counter entirely so all 16
                    // workers scale linearly.
                    #[cfg(feature = "asm_hit_counter")]
                    crate::cpu_asm::ASM_HITS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return self.cycles - prev_cycles;
                }
                } // end asm_handler_cycles > 0
                // No pre-tick to unwind — fall through to try_bulk_step
                // / per-cycle path cleanly.
            }
            let mut bus = SystemBus::new(
                &mut self.ram,
                &mut self.mapper,
                &mut self.ppu,
                &mut self.apu,
                &mut self.oam_dma,
                &mut self.input,
                &self.cheats,
            );
            if let Some(cycles) = self.cpu.try_bulk_step(&mut bus, opcode) {
                drop(bus);
                // APU + PPU catch-up. The per-cycle path calls
                // `set_nmi_line` after every PPU tick to detect the
                // rising edge when the PPU asserts NMI (scanline
                // 241 cycle 1). We skipped that and got black
                // screens — the vblank wait loop never saw the edge.
                //
                // Hoisted version: since the opcodes in
                // `Cpu::try_bulk_step` never touch MMIO (all
                // addressing modes are imm / zp / stack / relative),
                // the PPU's `nmi_occurred` can only transition
                // high-to-low via a CPU $2002 read — which can't
                // happen inside a bulk instruction. Therefore
                // `nmi_output && nmi_occurred` is monotonic within
                // the instruction (once true, stays true), so a
                // single end-of-bulk edge check detects the NMI
                // correctly. Same argument for `update_irq_line` —
                // only read by `poll_interrupts` at instruction end.
                //
                // Measured impact of hoisting: recovers most of the
                // ~25% throughput the correctness fix cost, without
                // reintroducing the black-screen bug. Verified via
                // the frame-render regression test inline in the
                // bulk_step_bench module below.
                // Event-driven PPU catch-up (Stage 1/2; see the ASM
                // remainder loop above for the invariant). Legacy
                // scanline-granular + verbatim interleaved branches
                // unchanged below.
                if self.hw_event_ppu {
                    for _ in 0..cycles {
                        self.apu.tick(&mut self.mapper, audio_frame_sink);
                    }
                    let target = self.ppu.cycles + cycles as u64 * 3;
                    self.ppu.advance_to(
                        target,
                        &mut self.mapper,
                        video_frame_sink,
                    );
                } else if self.ppu.is_skip_render()
                    && self.ppu.scanline_advance_enabled()
                    && self.cached_ppu_batchable
                {
                    for _ in 0..cycles {
                        self.apu.tick(&mut self.mapper, audio_frame_sink);
                    }
                    self.ppu.advance(
                        cycles as u64 * 3,
                        &mut self.mapper,
                        video_frame_sink,
                    );
                } else {
                    for _ in 0..cycles {
                        self.apu.tick(&mut self.mapper, audio_frame_sink);
                        self.ppu.tick_three(&mut self.mapper, video_frame_sink);
                    }
                }
                self.cycles += cycles as usize;
                // Detect NMI edge once, at the end. `set_nmi_line`
                // handles the low-to-high / high-to-low transition
                // internally; calling it here gives the same
                // `nmi_pended` result as the per-tick path for the
                // monotonic case we guarantee above.
                self.cpu.set_nmi_line(
                    self.ppu.nmi_output && self.ppu.nmi_occurred,
                );
                self.update_irq_line();
                self.cpu.poll_interrupts();
                return self.cycles - prev_cycles;
            }
        }

        // Slow path: run cycles until a CPU instruction completes.
        while !self.tick(video_frame_sink, audio_frame_sink) {}
        self.cycles - prev_cycles
    }

    /// Bulk-step fast path for LDA-immediate (opcode 0xA9). Part of
    /// the block-interpreter design in
    /// `docs/proposals/cpu_bulk_stepping.md`. Executes the whole
    /// instruction in one Rust call + batches PPU/APU cycle
    /// advancement. Phase 1 measured 4.5× speedup on the isolated
    /// bench. This is the first production integration.
    ///
    /// Caller must have verified (in `Nes::step`):
    /// - opcode at PC is 0xA9
    /// - no interrupt is pending
    /// - CPU not mid-instruction (`cpu.cycle == 0`,
    ///   `cpu.instruction.is_none()`)
    /// - `cpu.stall_cycles == 0`
    ///
    /// Returns the 2 cycles consumed.
    pub fn step_lda_immediate_fast<A: AudioSink, V: VideoSink + Sized>(
        &mut self,
        video_frame_sink: &mut V,
        audio_frame_sink: &mut A,
    ) -> usize {
        let mut bus = SystemBus::new(
            &mut self.ram,
            &mut self.mapper,
            &mut self.ppu,
            &mut self.apu,
            &mut self.oam_dma,
            &mut self.input,
            &self.cheats,
        );
        let cpu_cycles = self.cpu.step_lda_immediate_bulk(&mut bus) as usize;
        drop(bus);
        // APU + PPU catch-up. Per-cycle loop keeps behaviour
        // bit-identical to the slow path; `bulk_tick(n)` helpers in
        // PPU/APU are a future optimization (Phase 3 of the plan).
        for _ in 0..cpu_cycles {
            self.apu.tick(&mut self.mapper, audio_frame_sink);
            for _ in 0..3 {
                self.ppu.tick(&mut self.mapper, video_frame_sink);
            }
        }
        self.cycles += cpu_cycles;
        // CPU may have new interrupt state after the instruction
        // completes — poll for NMI/IRQ now, matching what the
        // per-cycle path does via `Cpu::poll_interrupts` on instr end.
        self.cpu.poll_interrupts();
        cpu_cycles
    }

    /// Test-only alias to keep the bench module in cpu.rs working.
    #[cfg(test)]
    pub fn step_one_lda_immediate_bulk<A: AudioSink, V: VideoSink + Sized>(
        &mut self,
        video_frame_sink: &mut V,
        audio_frame_sink: &mut A,
    ) -> usize {
        self.step_lda_immediate_fast(video_frame_sink, audio_frame_sink)
    }

    // Run all hardware for the duration of a single CPU cycle. Returns whether a CPU instruction completed in this cycle.
    //
    // Note: tried `#[inline(always)]` on this + `Apu::tick` + `Ppu::tick`
    // — regressed from 1289 → 548 sps (52% slower) post-PGO. The icache
    // pressure from inlining a large function × 16 workers dominated
    // any register-allocation win. PGO knows better. Don't re-add.
    pub fn tick<A: AudioSink, V: VideoSink + Sized>(
        &mut self,
        video_frame_sink: &mut V,
        audio_frame_sink: &mut A,
    ) -> bool {
        // Keep the CPU's φ2-latch mirror in sync (tests and tools set
        // the `Nes` field directly, so setters can't be the sync
        // point). Unconditional store: cheaper than the branch.
        self.cpu.hw_nmi_subcycle_phase = self.hw_nmi_subcycle_phase;

        // There is 1 APU cycle per CPU cycle.
        let cpu_stall_cycles = self.apu.tick(&mut self.mapper, audio_frame_sink);
        if cpu_stall_cycles > 0 {
            self.cpu.stall(cpu_stall_cycles);
        }

        // There are 3 PPU cycles per CPU cycle.
        //
        // Sub-cycle bus catch-up (event-driven PPU Stage 3): when a
        // deferred PPU-register access will fire in `cpu.tick` below, hold
        // the PPU back at the calibrated sub-cycle dot (`lead` = offset+1
        // of the three dots) so the access samples that exact dot; the
        // remaining `3 - lead` dots run AFTER the access. `lead == 3` for
        // every non-access cycle and for the default end-of-cycle offset,
        // making this path byte-identical to the pre-Stage-3 interleave.
        let lead = if self.hw_event_ppu {
            self.ppu_bus_dot_lead()
        } else {
            3
        };
        if self.hw_event_ppu {
            // Event-driven routing: advance one dot at a time through
            // `advance_to` so the per-dot `set_nmi_line` edge sampling is
            // preserved exactly (`advance_to(cycles+1)` is one `tick`).
            // Byte-identical to the else branch by construction.
            let target = self.ppu.cycles + lead;
            while self.ppu.cycles < target {
                self.ppu
                    .advance_to(self.ppu.cycles + 1, &mut self.mapper, video_frame_sink);
                // φ2 sample phase (see the else branch): the cycle's
                // dot index 2 lands in THIS loop only when lead == 3
                // (no deferred access splits the cycle) and this is
                // the last iteration; lead < 3 cycles reach index 2
                // in the catch-up tail below.
                if lead < 3 || self.ppu.cycles < target || !self.hw_nmi_subcycle_phase {
                    self.cpu
                        .set_nmi_line(self.ppu.nmi_output && self.ppu.nmi_occurred);
                }
            }
        } else {
            // φ2 NMI-line sample phase: the physical 6502 latches its
            // NMI input on φ2 — 2 PPU dots into the 3-dot CPU cycle —
            // so under `hw_nmi_subcycle_phase` the sample after the
            // cycle's LAST dot (index 2) must not reach the edge
            // latch; an edge asserting on that dot becomes visible at
            // the NEXT cycle's dot-0 sample, one CPU cycle later.
            // Keeping the dot-0/1 samples is equivalent to a single
            // φ2 sample: the line cannot both assert (241,1) and
            // deassert (261,1 / `$2002` read — which happens inside
            // `cpu.tick`, not between dots) within one CPU cycle.
            // Flag OFF short-circuits to the identical call sequence.
            for dot in 0..3 {
                self.ppu.tick(&mut self.mapper, video_frame_sink);
                if dot < 2 || !self.hw_nmi_subcycle_phase {
                    self.cpu
                        .set_nmi_line(self.ppu.nmi_output && self.ppu.nmi_occurred);
                }
            }
        }

        // Push current CPU cycle to the mapper so it can implement
        // cycle-sensitive quirks (MMC1 RMW consecutive-write filter).
        self.mapper.set_cpu_cycle(self.cycles as u64);

        self.update_irq_line();

        let completed_instruction = if self.oam_dma.active {
            self.handle_oam_dma();
            false
        } else {
            let mut bus = SystemBus::new(
                &mut self.ram,
                &mut self.mapper,
                &mut self.ppu,
                &mut self.apu,
                &mut self.oam_dma,
                &mut self.input,
                &self.cheats,
            );
            self.cpu.tick(&mut bus)
        };

        // Sub-cycle bus catch-up tail: advance the `3 - lead` dots that
        // were deferred past a sub-cycle PPU-register access this cycle
        // (zero unless `hw_event_ppu` held the PPU back at a non-default
        // offset above). Runs before the post-cycle IRQ sample so the
        // mapper sees the whole cycle's fetches. `lead == 3` → no-op.
        if self.hw_event_ppu && lead < 3 {
            let target = self.ppu.cycles + (3 - lead);
            while self.ppu.cycles < target {
                self.ppu
                    .advance_to(self.ppu.cycles + 1, &mut self.mapper, video_frame_sink);
                // φ2 sample phase: the tail's final dot is the cycle's
                // dot index 2 — suppress its sample when the flag is
                // on. These tail dots run after `cpu.tick` already
                // took this cycle's `nmi_poll_latch`, so the
                // suppression is latch-invisible under
                // `hw_nmi_poll_timing`; it keeps `nmi_line_low`
                // dot-exact so the NEXT cycle's edge detection matches
                // the φ2 model.
                if self.ppu.cycles < target || !self.hw_nmi_subcycle_phase {
                    self.cpu
                        .set_nmi_line(self.ppu.nmi_output && self.ppu.nmi_occurred);
                }
            }
        }

        self.update_irq_line();

        self.cycles += 1;

        completed_instruction
    }

    /// Dots to advance BEFORE this CPU cycle's bus work under
    /// `hw_event_ppu`: 3 (end-of-cycle — the default and every
    /// non-access cycle), or `offset + 1` when the next `cpu.tick`
    /// performs a deferred PPU-register access, so the access samples the
    /// PPU at the calibrated sub-cycle dot (`ppu_read_dot_offset` /
    /// `ppu_write_dot_offset`). The remaining `3 - lead` dots run after
    /// the access. See the `ppu_read_dot_offset` field doc and §3.4 of
    /// docs/proposals/event_driven_ppu_design_2026-07-31.md.
    #[inline]
    fn ppu_bus_dot_lead(&self) -> u64 {
        // Default (end-of-cycle) offsets, OAM DMA (its per-cycle $2004
        // writes are the island, §1.10), and DMC/OAM stall cycles never
        // sub-cycle: return the full 3 so the interleave is byte-identical.
        if (self.ppu_read_dot_offset >= 2 && self.ppu_write_dot_offset >= 2)
            || self.oam_dma.active
            || self.cpu.stall_cycles > 0
        {
            return 3;
        }
        match self.cpu.pending_deferred_ppu_access() {
            Some(is_write) => {
                let off = if is_write {
                    self.ppu_write_dot_offset
                } else {
                    self.ppu_read_dot_offset
                };
                (off.min(2) as u64) + 1
            }
            None => 3,
        }
    }

    #[inline(always)]
    fn update_irq_line(&mut self) {
        self.cpu
            .set_irq_line_low(self.apu.irq_pending() || self.mapper.irq_pending());
    }

    fn handle_oam_dma(&mut self) {
        if self.oam_dma.dummy_read {
            // Needs an extra cycle to align on odd cycles.
            if !self.cycles.is_multiple_of(2) {
                self.oam_dma.dummy_read = false;
            }
            return;
        }

        // ON even cycles read the source byte. Route the fetch through
        // the SAME full CPU bus dispatch a normal read uses (RAM mirror
        // < $2000, PPU/APU/input MMIO, mapper PRG-RAM/ROM >= $4020) so a
        // game DMAing sprites out of PRG-RAM ($6000-$7FFF) or a static
        // PRG-ROM table transfers the correct bytes — the OAM DMA unit
        // drives the real CPU read line on hardware, it does not read
        // internal RAM directly. The common case (page $02 → internal
        // RAM) stays byte-identical: `SystemBus::read_byte` dispatches
        // $0000-$1FFF straight to `Ram::read_byte` with the same mirror
        // mask. This changes only WHICH byte is fetched for source pages
        // >= $20; it does NOT change the read/write cadence, so the DMA
        // stall length (513/514 CPU cycles) is unaffected.
        if self.cycles.is_multiple_of(2) {
            let addr = ((self.oam_dma.page as u16) << 8) | (self.oam_dma.count & 0xFF);
            let data = {
                let mut bus = SystemBus::new(
                    &mut self.ram,
                    &mut self.mapper,
                    &mut self.ppu,
                    &mut self.apu,
                    &mut self.oam_dma,
                    &mut self.input,
                    &self.cheats,
                );
                bus.read_byte(addr)
            };
            self.oam_dma.data = data;
        }
        // ON odd cycles write to PPU OAMDATA.
        else {
            self.ppu
                .write_byte(&mut self.mapper, OAMDATA_ADDRESS, self.oam_dma.data);
            self.oam_dma.count += 1;
            if self.oam_dma.count == 256 {
                self.oam_dma.active = false;
            }
        }
    }

    /// Maximum CPU cycles a single `step()` call may consume. Used by
    /// the cycle-locked `advance_one_frame` paths in `python.rs` and
    /// `pool.rs` so they stop bulk-stepping with enough headroom that
    /// the trailing single-cycle `tick()` loop can land EXACTLY on the
    /// 29781-cycle frame target. The ASM CPU runs up to
    /// `mapper.asm_bulk_cycles()` CPU cycles per call (NROM = 32, most
    /// others = 1-4); without this margin the bulk phase routinely
    /// overshoots target by 18-25 cycles, drifting cycle alignment from
    /// nes-py and breaking SMB / Zelda / Contra (Mario falls through
    /// floor, Link can't transition screens, etc.) — see today's
    /// session memory.
    pub fn asm_bulk_cycles_margin(&self) -> usize {
        // +1 for the slow-path single-cycle tick that may also fire,
        // and to absorb any future +1 cycle adjustment.
        (self.mapper.asm_bulk_cycles().max(7) as usize) + 1
    }

    pub fn get_state(&self) -> State {
        State {
            ram: self.ram.to_vec(),
            mapper: self.mapper.get_state(),
            cpu: self.cpu.get_state(),
            ppu: self.ppu.get_state(),
            apu: self.apu.get_state(),
            input: self.input.get_state(),
            cycles: self.cycles,
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.ram.copy_from_slice(&state.ram);
        self.mapper.apply_state(&state.mapper);
        // mapper.apply_state may rebuild the prg_asm_window Vec
        // (e.g., after a fresh deserialize). Refresh the cached
        // ptr so any subsequent Nes::step uses the new allocation.
        self.cached_prg_asm_ptr = self.mapper.prg_asm_ptr();
        self.cached_asm_bulk_cycles = self.mapper.asm_bulk_cycles();
        self.cached_ppu_batchable = self.ppu_batchable();
        self.cpu.apply_state(&state.cpu);
        self.ppu.apply_state(&state.ppu);
        self.apu.apply_state(&state.apu);
        self.input.apply_state(&state.input);
        self.cycles = state.cycles;
    }

    /// nestest-style trace line for the instruction the CPU is about
    /// to execute. Call only at an instruction boundary (see
    /// `Cpu::at_instruction_boundary`); otherwise the asm
    /// disassembly is for the in-progress instruction's opcode and
    /// the register values reflect mid-instruction state.
    pub fn trace_line(&mut self) -> String {
        let bus = SystemBus::new(
            &mut self.ram, &mut self.mapper, &mut self.ppu, &mut self.apu,
            &mut self.oam_dma, &mut self.input, &self.cheats,
        );
        self.cpu.trace(&bus)
    }

    pub fn system_bus(&'_ mut self) -> SystemBus<'_> {
        SystemBus::new(
            &mut self.ram,
            &mut self.mapper,
            &mut self.ppu,
            &mut self.apu,
            &mut self.oam_dma,
            &mut self.input,
            &self.cheats,
        )
    }

    // ---- test-only CPU+RAM accessors for the ASM diff harness ----
    // Used by `cpu_asm::tests` to set up identical state on both the
    // Rust reference and the ASM path, then compare outcomes.
    #[cfg(any(test, feature = "asm_cpu"))]
    pub fn cpu_state_for_diff_test(&self) -> crate::cpu::State {
        self.cpu.get_state()
    }
    #[cfg(any(test, feature = "asm_cpu"))]
    pub fn cpu_apply_state_for_diff_test(&mut self, s: &crate::cpu::State) {
        self.cpu.apply_state(s);
    }
    #[cfg(any(test, feature = "asm_cpu"))]
    pub fn ram_for_diff_test(&self) -> &[u8] {
        self.ram.as_slice()
    }
    #[cfg(any(test, feature = "asm_cpu"))]
    pub fn ram_mut_for_diff_test(&mut self) -> &mut [u8] {
        self.ram.as_mut_slice()
    }

    /// Forward to `Ppu::set_skip_render`. Toggle the per-frame
    /// skip-render fast path — see PPU docs for the semantics.
    /// Forward to `Apu::set_sample_output_enabled`. Skips sample
    /// generation + mixing for training workers that don't consume
    /// audio.
    pub fn set_audio_output_enabled(&mut self, enabled: bool) {
        self.apu.set_sample_output_enabled(enabled);
    }

    pub fn set_skip_render(&mut self, skip: bool) {
        self.ppu.set_skip_render(skip);
    }

    /// Forward to `Mapper::set_asm_bulk_cycles_override`. Opt-in ASM
    /// bulk budget for the batch-safe mappers (MMC1, UxROM); no-op on
    /// every other mapper. Default budget is 1 — timing on default
    /// settings is unchanged. The cycle-locked `advance_one_frame`
    /// margins in `python.rs`/`pool.rs` query
    /// `asm_bulk_cycles_margin()` per frame, so they track the new
    /// budget automatically.
    pub fn set_asm_bulk_cycles_override(&mut self, cycles: i64) {
        self.mapper.set_asm_bulk_cycles_override(cycles);
        // Keep the cached budget in lockstep with the mapper — this is
        // the only path (besides construction / reset / apply_state)
        // that changes what `asm_bulk_cycles()` returns.
        self.cached_asm_bulk_cycles = self.mapper.asm_bulk_cycles();
    }

    /// Coarse batchability sample for `cached_ppu_batchable`. See the
    /// field doc: a perf hint only, re-checked authoritatively inside
    /// `Ppu::advance`.
    #[inline]
    fn ppu_batchable(&self) -> bool {
        self.mapper.chr_static_ptr().is_some() && !self.mapper.uses_scanline_irq()
    }

    pub fn reset(&mut self) {
        {
            let mut bus = SystemBus::new(
                &mut self.ram,
                &mut self.mapper,
                &mut self.ppu,
                &mut self.apu,
                &mut self.oam_dma,
                &mut self.input,
                &self.cheats,
            );
            self.cpu.reset(&mut bus);
        }
        self.ppu.reset();
        self.apu.reset();
        // Mapper has been reset (during cpu.reset above? actually
        // mapper isn't reset here — that's Mapper::reset and isn't
        // called from Nes::reset). But re-cache anyway so first
        // ASM dispatch sees the right pointer.
        self.cached_prg_asm_ptr = self.mapper.prg_asm_ptr();
        self.cached_asm_bulk_cycles = self.mapper.asm_bulk_cycles();
        self.cached_ppu_batchable = self.ppu_batchable();

        // Model the 8-cycle 6502 reset sequence: real hardware spends
        // 8 CPU cycles reading the reset vector + setting up internal
        // state before the first instruction fetch. During those 8
        // cycles the PPU runs normally (3 PPU per CPU = 24 PPU cycles).
        // Mesen models this in NesCpu::Reset() (loops 8 StartCpuCycle/
        // EndCpuCycle pairs after the vector read). Without modeling,
        // our PPU starts ~25 cycles behind Mesen's at the first
        // instruction, drifting all $2002 / vblank timing relative to
        // the gold-standard reference and capping how tight the
        // tests/parity/test_mesen_lockstep.py ceilings can go.
        //
        // Side effect: nes-py / LaiNES also do not model the reset
        // cycles, so this fix tightens nes_core-vs-Mesen parity at
        // the cost of nes_core-vs-nes-py parity. The Mesen ceilings
        // are the better gate (Mesen is more accurate); nes-py
        // ceilings are tracked separately in test_library_buckets
        // and may need re-bucketing.
        let mut bus = SystemBus::new(
            &mut self.ram,
            &mut self.mapper,
            &mut self.ppu,
            &mut self.apu,
            &mut self.oam_dma,
            &mut self.input,
            &self.cheats,
        );
        if self.hw_reset_alignment {
            // Hardware-true boot: 7-cycle reset sequence, then the PPU
            // sits 4 dots ahead of 3×CPU (Mesen canonical power-on
            // alignment → first opcode fetch at CYC=7, PPU dot 25).
            // `self.cycles` is SET (not incremented) so a reset after
            // construction can't stack a second discard's worth of
            // cycles — the odd 16-vs-7 offset is exactly what flips
            // OAM-DMA parity against Mesen on tape replays.
            bus.tick_cpu_cycles_discard(7);
            bus.tick_ppu_dots_discard(4);
            drop(bus);
            self.cycles = 7;
        } else {
            bus.tick_cpu_cycles_discard(8);
            drop(bus);
            self.cycles += 8;
        }
    }

    pub fn initialize_nestest(&mut self) {
        self.cpu.initialize_nestest();
        self.ppu.initialize_nestest();
    }

    pub fn add_cheat(&mut self, cheat: Cheat) {
        self.cheats.insert(cheat.address(), cheat);
    }

    pub fn remove_cheat(&mut self, cheat: Cheat) {
        self.cheats.remove(&cheat.address());
    }

    pub fn clear_cheats(&mut self) {
        self.cheats.clear();
    }

    pub fn system_ram_byte(&self, addr: u16) -> u8 {
        self.ram.peek_byte(addr)
    }

    pub fn system_ram(&mut self) -> &mut Ram {
        &mut self.ram
    }

    pub fn video_ram(&mut self) -> &mut Vram {
        &mut self.ppu.mem.vram
    }

    pub fn save_ram(&mut self) -> *mut u8 {
        self.mapper.sram()
    }

    pub fn save_ram_size(&self) -> usize {
        self.mapper.sram_size()
    }

    pub fn game_pad_1(&mut self) -> &mut GamePad {
        &mut self.input.game_pad_1
    }

    pub fn game_pad_2(&mut self) -> &mut GamePad {
        &mut self.input.game_pad_2
    }
}

#[cfg(test)]
mod oam_dma_bus_tests {
    use super::*;
    use crate::cartridge::Cartridge;

    const OAMADDR_ADDRESS: u16 = 0x2003;

    struct NullSinks;
    impl VideoSink for NullSinks {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool {
            false
        }
        fn pixel_size(&self) -> usize {
            4
        }
    }
    impl AudioSink for NullSinks {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize {
            0
        }
    }

    /// Synthetic 32 KB NROM (mapper 0) iNES 1.0 ROM. `flags8 = 0` →
    /// `max(1, 0)` = one 8 KB PRG-RAM bank at $6000-$7FFF (see
    /// `cartridge.rs` PRG-RAM sizing). Reset vector points at $C000.
    fn build_nrom_with_prg_ram() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(0); // CHR = 0 (CHR-RAM)
        rom.push(0); // flags6: mapper 0 low nibble, H-mirror
        rom.push(0); // flags7: mapper 0 high nibble (iNES 1.0)
        rom.extend_from_slice(&[0u8; 8]); // bytes 8..=15 (flags8=0 → 8 KB PRG-RAM)
        let mut prg = vec![0u8; 32 * 1024];
        let n = prg.len();
        prg[n - 4] = 0x00; // reset vector low  → $C000
        prg[n - 3] = 0xC0; // reset vector high → $C000
        rom.extend(prg);
        rom
    }

    fn build_nes() -> Nes {
        let rom = build_nrom_with_prg_ram();
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom))
            .expect("synthetic NROM should parse");
        Nes::new(cart)
    }

    /// Drive an OAM DMA (started via a $4014 write) to completion by
    /// ticking the whole machine, then read the 256 OAM bytes back.
    /// Returns the OAM contents at indices 0..256.
    fn run_dma_and_read_oam(nes: &mut Nes, page: u8) -> Vec<u8> {
        // STA $4014 = `page` (plus OAMADDR = 0 so the transfer lands at
        // OAM[0..256]) through the real bus — exactly the CPU path.
        {
            let mut bus = SystemBus::new(
                &mut nes.ram,
                &mut nes.mapper,
                &mut nes.ppu,
                &mut nes.apu,
                &mut nes.oam_dma,
                &mut nes.input,
                &nes.cheats,
            );
            bus.write_byte(OAMADDR_ADDRESS, 0x00);
            bus.write_byte(crate::system_bus::OAMDMA_ADDRESS, page);
        }
        assert!(nes.oam_dma.active, "$4014 write must arm OAM DMA");

        // A DMA is 513/514 CPU cycles; cap generously and assert it ends.
        let mut v = NullSinks;
        let mut a = NullSinks;
        let mut ticks = 0;
        while nes.oam_dma.active {
            nes.tick(&mut v, &mut a);
            ticks += 1;
            assert!(ticks < 1024, "OAM DMA never completed");
        }

        // Read OAM back via $2004. Park the PPU on the post-render
        // scanline (240 > VISIBLE_END_SCANLINE) so `read_oam_byte`
        // returns the true byte instead of the $FF sprite-eval value.
        nes.ppu.scanline = 240;
        let mut out = vec![0u8; 256];
        let mut bus = SystemBus::new(
            &mut nes.ram,
            &mut nes.mapper,
            &mut nes.ppu,
            &mut nes.apu,
            &mut nes.oam_dma,
            &mut nes.input,
            &nes.cheats,
        );
        for (i, slot) in out.iter_mut().enumerate() {
            bus.write_byte(OAMADDR_ADDRESS, i as u8);
            *slot = bus.read_byte(OAMDATA_ADDRESS);
        }
        out
    }

    /// F66: DMA from a PRG-RAM source page ($6000-$7FFF) must copy the
    /// PRG-RAM contents into OAM. The pre-fix code masked the source
    /// address into the 2 KB internal RAM ($6000 & 0x07FF = $0000), so
    /// it would have copied the poisoned RAM byte instead.
    #[test]
    fn oam_dma_from_prg_ram_page_transfers_prg_ram() {
        let mut nes = build_nes();
        nes.reset();

        // Poison ALL internal RAM: the buggy masked-read path folds page
        // $60 into $0000-$00FF, so it would yield 0xAA everywhere.
        for a in 0u16..0x0800 {
            nes.ram.write_byte(a, 0xAA);
        }
        // Fill PRG-RAM $6000-$60FF with a position-dependent pattern.
        for i in 0u16..256 {
            nes.mapper.prg_write_byte(0x6000 + i, (i as u8) ^ 0x5A);
        }

        let oam = run_dma_and_read_oam(&mut nes, 0x60);

        for i in 0..256 {
            let expected = (i as u8) ^ 0x5A;
            assert_eq!(
                oam[i], expected,
                "OAM[{i}] = {:#04X}, expected PRG-RAM byte {:#04X} (got the \
                 masked-RAM value {:#04X} instead? = bus routing broken)",
                oam[i], expected, 0xAAu8,
            );
        }
    }

    /// CRITICAL regression guard: the overwhelmingly common case is a
    /// DMA from internal RAM (SMB DMAs $0200 every frame). Routing the
    /// fetch through the bus MUST leave this byte-identical.
    #[test]
    fn oam_dma_from_internal_ram_page_unchanged() {
        let mut nes = build_nes();
        nes.reset();

        // Distinct data in the $0200 page; different filler elsewhere so
        // a wrong mask would be caught.
        for a in 0u16..0x0800 {
            nes.ram.write_byte(a, 0x33);
        }
        for i in 0u16..256 {
            nes.ram.write_byte(0x0200 + i, i as u8);
        }

        let oam = run_dma_and_read_oam(&mut nes, 0x02);

        for i in 0..256 {
            assert_eq!(
                oam[i], i as u8,
                "OAM[{i}] = {:#04X}, expected internal-RAM byte {:#04X}",
                oam[i], i as u8,
            );
        }
    }

    /// The internal-RAM mirror (pages $08-$1F alias $00-$07) must still
    /// resolve through the bus: page $0A ($0A00) mirrors to $0200.
    #[test]
    fn oam_dma_from_mirrored_ram_page_resolves_mirror() {
        let mut nes = build_nes();
        nes.reset();

        for a in 0u16..0x0800 {
            nes.ram.write_byte(a, 0x00);
        }
        // Write the pattern at the canonical $0200 page.
        for i in 0u16..256 {
            nes.ram.write_byte(0x0200 + i, (i as u8).wrapping_add(7));
        }

        // DMA from page $0A ($0A00) — mirrors to $0200 in 2 KB RAM.
        let oam = run_dma_and_read_oam(&mut nes, 0x0A);

        for i in 0..256 {
            assert_eq!(
                oam[i],
                (i as u8).wrapping_add(7),
                "OAM[{i}] from mirrored page $0A did not resolve to $0200",
            );
        }
    }
}

#[cfg(test)]
mod event_ppu_routing_tests {
    //! `hw_event_ppu` is a routing gate: ON must produce byte-identical
    //! machine state to OFF (the legacy inline `3× ppu.tick` / catch-up
    //! interleave). These tests drive real instruction streams through
    //! both flag states in lockstep and assert full-`Nes`-state equality,
    //! exercising `Nes::tick` (slow path), the ASM/bulk catch-up loops
    //! (call sites C/D), and — via a $2000 write that enables the NMI —
    //! the vblank / NMI edge the routing must not misplace.
    use super::*;
    use crate::cartridge::Cartridge;

    struct NullSinks;
    impl VideoSink for NullSinks {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool {
            false
        }
        fn pixel_size(&self) -> usize {
            4
        }
    }
    impl AudioSink for NullSinks {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize {
            0
        }
    }

    /// Synthetic 32 KB NROM whose reset routine at $C000 enables the
    /// vblank NMI (`LDA #$80 ; STA $2000`) and then spins on NOPs. The
    /// NMI handler ($E000, wired via the vector) is a bare `RTI`. This
    /// keeps the CPU emitting bulk/ASM-eligible opcodes (so the catch-up
    /// loops run) while forcing a real NMI edge every frame (so the
    /// routing's edge handling is exercised).
    fn build_nop_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(1); // CHR = 1 × 8 KB
        rom.push(0); // flags6: mapper 0 low nibble, H-mirror
        rom.push(0); // flags7: mapper 0 high nibble (iNES 1.0)
        rom.extend_from_slice(&[0u8; 8]);
        let mut prg = vec![0xEAu8; 32 * 1024]; // NOP fill
        // Reset routine at $C000 (PRG offset 0x0000): LDA #$80; STA $2000.
        prg[0x0000] = 0xA9; // LDA #imm
        prg[0x0001] = 0x80; // #$80 — PPUCTRL bit 7 (NMI enable)
        prg[0x0002] = 0x8D; // STA abs
        prg[0x0003] = 0x00; // $2000 lo
        prg[0x0004] = 0x20; // $2000 hi
        // rest of PRG stays NOP; PC free-runs into the NOP field and wraps.
        // NMI handler at $E000 (PRG offset 0x2000): RTI.
        prg[0x2000] = 0x40; // RTI
        // Vectors at $FFFA-$FFFF (PRG offset 0x7FFA..0x7FFF).
        prg[0x7FFA] = 0x00; // NMI  → $E000
        prg[0x7FFB] = 0xE0;
        prg[0x7FFC] = 0x00; // RESET → $C000
        prg[0x7FFD] = 0xC0;
        prg[0x7FFE] = 0x00; // IRQ  → $E000
        prg[0x7FFF] = 0xE0;
        let mut chr = vec![0u8; 8 * 1024];
        for (i, b) in chr.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        rom.extend(prg);
        rom.extend(chr);
        rom
    }

    fn build_nes(hw_event_ppu: bool) -> Nes {
        let cart = Cartridge::load(&mut std::io::Cursor::new(build_nop_rom()))
            .expect("synthetic NROM should parse");
        let mut nes = Nes::new(cart);
        nes.hw_event_ppu = hw_event_ppu;
        nes
    }

    fn state_digest(nes: &Nes) -> Vec<u8> {
        bincode::serialize(&nes.get_state()).expect("serialize nes state")
    }

    /// Step both machines instruction-for-instruction; the routing gate
    /// must keep them byte-identical the whole way. `skip_render` chosen
    /// by the caller so both the full-render and skip-render code paths
    /// through `advance_to` are covered.
    fn assert_routing_equiv(steps: usize, skip_render: bool) {
        let mut on = build_nes(true);
        let mut off = build_nes(false);
        on.set_skip_render(skip_render);
        off.set_skip_render(skip_render);
        let mut v = NullSinks;
        let mut a = NullSinks;
        assert_eq!(
            state_digest(&on),
            state_digest(&off),
            "post-reset state must match (skip_render={skip_render})"
        );
        for i in 0..steps {
            on.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            // Compare every step for the first 6000 (tight around boot +
            // first vblank), then sample to keep the test fast.
            if i < 6000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&on),
                    state_digest(&off),
                    "hw_event_ppu ON diverged from OFF at instruction {i} \
                     (skip_render={skip_render}): routing is NOT byte-identical"
                );
            }
        }
    }

    #[test]
    fn routing_on_equals_off_skip_render() {
        // ~90k instructions ≈ several frames — crosses vblank set, the
        // NMI edge + RTI, vblank clear, and odd/even frame skips many
        // times, through the ASM/bulk catch-up loops.
        assert_routing_equiv(90_000, true);
    }

    #[test]
    fn routing_on_equals_off_full_render() {
        // Full render forces `advance_to`'s per-dot fallback inside the
        // catch-up loops; must still match the legacy interleave exactly.
        assert_routing_equiv(30_000, false);
    }

    #[test]
    fn routing_on_equals_off_asm_disabled() {
        // With the ASM path disabled the machine runs the Rust bulk-step
        // + slow (`Nes::tick`) paths; the per-cycle `advance_to(cycles+1)`
        // routing in `Nes::tick` must match the legacy `3× ppu.tick`.
        let mut on = build_nes(true);
        let mut off = build_nes(false);
        on.disable_asm_cpu = true;
        off.disable_asm_cpu = true;
        let mut v = NullSinks;
        let mut a = NullSinks;
        for i in 0..30_000usize {
            on.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            if i < 4000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&on),
                    state_digest(&off),
                    "hw_event_ppu ON diverged from OFF (asm disabled) at {i}"
                );
            }
        }
    }

    /// Synthetic MMC3 (mapper 4) ROM. MMC3 is a NON-batchable mapper
    /// (scanline IRQ), so under `hw_event_ppu` the catch-up loops still
    /// tick the APU separately and let `advance_to` run the PPU per-dot
    /// via the reference path — the one configuration the NROM routing
    /// tests don't cover (they exercise the closed-form batcher instead).
    /// Code lives entirely in MMC3's fixed last 8 KB bank ($E000-$FFFF)
    /// so no bank register init is needed to boot.
    fn build_mmc3_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 128 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(8); // PRG = 8 × 16 KB = 128 KB
        rom.push(1); // CHR = 1 × 8 KB
        rom.push(0x40); // flags6: mapper 4 low nibble (H-mirror)
        rom.push(0x00); // flags7: mapper 4 high nibble = 0 (iNES 1.0)
        rom.extend_from_slice(&[0u8; 8]);
        let prg_len = 128 * 1024;
        let mut prg = vec![0xEAu8; prg_len]; // NOP fill
        // The fixed last 8 KB bank maps to $E000-$FFFF = PRG tail.
        let last = prg_len - 8 * 1024; // offset of $E000
        // Reset routine at $E000: LDA #$80; STA $2000 (enable NMI); NOPs.
        prg[last] = 0xA9;
        prg[last + 1] = 0x80;
        prg[last + 2] = 0x8D;
        prg[last + 3] = 0x00;
        prg[last + 4] = 0x20;
        // NMI/IRQ handler at $F000 (offset last + 0x1000): RTI.
        prg[last + 0x1000] = 0x40; // RTI
        // Vectors at $FFFA-$FFFF (PRG tail).
        prg[prg_len - 6] = 0x00; // NMI → $F000
        prg[prg_len - 5] = 0xF0;
        prg[prg_len - 4] = 0x00; // RESET → $E000
        prg[prg_len - 3] = 0xE0;
        prg[prg_len - 2] = 0x00; // IRQ → $F000
        prg[prg_len - 1] = 0xF0;
        let mut chr = vec![0u8; 8 * 1024];
        for (i, b) in chr.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        rom.extend(prg);
        rom.extend(chr);
        rom
    }

    #[test]
    fn routing_on_equals_off_mmc3_non_batchable() {
        let load = |ev: bool| {
            let cart = Cartridge::load(&mut std::io::Cursor::new(build_mmc3_rom()))
                .expect("synthetic MMC3 should parse");
            let mut nes = Nes::new(cart);
            nes.hw_event_ppu = ev;
            nes
        };
        let mut on = load(true);
        let mut off = load(false);
        on.set_skip_render(true);
        off.set_skip_render(true);
        let mut v = NullSinks;
        let mut a = NullSinks;
        // Cross several frames so the MMC3 A12 scanline-IRQ counter clocks
        // and (if it fires) both machines take the IRQ identically — the
        // routing must keep them byte-identical regardless.
        for i in 0..60_000usize {
            on.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            if i < 6000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&on),
                    state_digest(&off),
                    "hw_event_ppu ON diverged from OFF on MMC3 at instruction {i}: \
                     APU/PPU catch-up separation is not byte-identical for a \
                     non-batchable mapper"
                );
            }
        }
    }

    // ================================================================
    // Stage 3 — sub-cycle bus-access catch-up (READ_DOT_OFFSET).
    // ================================================================

    /// Synthetic 32 KB NROM whose entire PRG is `LDA $2002` (`AD 02 20`),
    /// so the CPU polls PPUSTATUS forever, walking straight up through PRG
    /// from the reset vector. RESET points at `$8000` — PRG offset 0, the
    /// fixed NROM base for any bank count, so the first fetch is
    /// pattern-aligned and every instruction is a clean `LDA $2002`. No
    /// `$2000` write means `nmi_output` stays false (no NMI ever fires)
    /// and `LDA` never branches on the read value, so the instruction
    /// stream is fully deterministic and independent of the sub-cycle
    /// offset — letting the tests attribute any change in the read's PPU
    /// dot purely to the offset.
    fn build_status_poll_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(1); // CHR = 1 × 8 KB
        rom.push(0); // flags6: mapper 0, H-mirror
        rom.push(0); // flags7: mapper 0 (iNES 1.0)
        rom.extend_from_slice(&[0u8; 8]);
        let mut prg = vec![0u8; 32 * 1024];
        for (i, b) in prg.iter_mut().enumerate() {
            *b = match i % 3 {
                0 => 0xAD, // LDA abs
                1 => 0x02, // $2002 lo
                _ => 0x20, // $2002 hi
            };
        }
        // Vectors (PRG tail). All point at $8000 (pattern-aligned); the
        // NMI/IRQ vectors are never taken in the tests' pre-NMI window.
        prg[0x7FFA] = 0x00;
        prg[0x7FFB] = 0x80;
        prg[0x7FFC] = 0x00;
        prg[0x7FFD] = 0x80;
        prg[0x7FFE] = 0x00;
        prg[0x7FFF] = 0x80;
        let chr = vec![0u8; 8 * 1024];
        rom.extend(prg);
        rom.extend(chr);
        rom
    }

    /// Run the status-poll ROM for `steps` instructions and return the PPU
    /// dot at which the last `$2002` read was serviced. `hw_nmi_poll_timing`
    /// is always on to force the per-cycle path (disabling the ASM/bulk
    /// batchers), the only path that positions a deferred access sub-cycle.
    fn poll_status_read_dot(
        hw_event_ppu: bool,
        mmio_read_timing: bool,
        read_offset: u8,
        steps: usize,
    ) -> u64 {
        let cart = Cartridge::load(&mut std::io::Cursor::new(build_status_poll_rom()))
            .expect("synthetic status-poll NROM should parse");
        let mut nes = Nes::new(cart);
        nes.hw_event_ppu = hw_event_ppu;
        nes.cpu.hw_mmio_read_timing = mmio_read_timing;
        nes.cpu.hw_nmi_poll_timing = true;
        nes.ppu_read_dot_offset = read_offset;
        nes.set_skip_render(true);
        let mut v = NullSinks;
        let mut a = NullSinks;
        for _ in 0..steps {
            nes.step(&mut v, &mut a);
        }
        nes.ppu.last_status_read_dot
    }

    #[test]
    fn subcycle_read_offset_shifts_status_poll_landing() {
        // The deferred `$2002` read must sample the PPU 2 / 1 / 0 dots
        // earlier as `ppu_read_dot_offset` drops 2 → 1 → 0. The CPU-cycle
        // base is identical across offsets (the PPU always advances 3 dots
        // per CPU cycle), so the landings differ by exactly (2 - offset).
        let steps = 60;
        let d2 = poll_status_read_dot(true, true, 2, steps);
        let d1 = poll_status_read_dot(true, true, 1, steps);
        let d0 = poll_status_read_dot(true, true, 0, steps);
        assert!(d2 >= 3, "sanity: end-of-cycle landing should be well past boot (d2={d2})");
        assert_eq!(
            d2 - d1, 1,
            "read offset 1 must land one dot before end-of-cycle (d2={d2} d1={d1})"
        );
        assert_eq!(
            d2 - d0, 2,
            "read offset 0 must land two dots before end-of-cycle (d2={d2} d0={d0})"
        );
    }

    #[test]
    fn subcycle_default_offset_matches_event_ppu_off() {
        // hw_event_ppu ON at the default end-of-cycle offset (2) must land
        // the `$2002` read at the SAME PPU dot as hw_event_ppu OFF — the
        // Stage 3 default is byte-identical to the legacy interleave.
        let steps = 60;
        let off = poll_status_read_dot(false, true, 2, steps);
        let on2 = poll_status_read_dot(true, true, 2, steps);
        assert_eq!(
            off, on2,
            "default offset must reproduce the OFF-path read dot (off={off} on2={on2})"
        );
    }

    #[test]
    fn subcycle_offset_inert_without_mmio_read_timing() {
        // Without hw_mmio_read_timing the `$2002` read is early-committed
        // at cycle 0 (cycle_zero_early_commit), so no final-cycle handle
        // exists and the sub-cycle offset cannot move the landing:
        // offset 0 lands at the same dot as offset 2. This documents that
        // Stage 3 rides the existing exact-cycle pin, not a new one.
        let steps = 60;
        let d0 = poll_status_read_dot(true, false, 0, steps);
        let d2 = poll_status_read_dot(true, false, 2, steps);
        assert_eq!(
            d0, d2,
            "offset must be inert when the read is not deferred to its final cycle \
             (d0={d0} d2={d2})"
        );
    }

    #[test]
    fn pending_deferred_ppu_access_fires_only_on_final_read_cycle() {
        // `Cpu::pending_deferred_ppu_access` must flag Some(false) exactly
        // on the tick that runs the deferred `$2002` read (the final cycle
        // of `LDA $2002`) and None on every other cycle.
        let cart = Cartridge::load(&mut std::io::Cursor::new(build_status_poll_rom()))
            .expect("synthetic status-poll NROM should parse");
        let mut nes = Nes::new(cart);
        nes.cpu.hw_mmio_read_timing = true;
        nes.set_skip_render(true);
        let mut v = NullSinks;
        let mut a = NullSinks;
        // Step past boot so the next fetch is a clean, pattern-aligned
        // `LDA $2002` at an instruction boundary.
        for _ in 0..5 {
            nes.step(&mut v, &mut a);
        }
        let mut seen = Vec::new();
        loop {
            seen.push(nes.cpu.pending_deferred_ppu_access());
            if nes.tick(&mut v, &mut a) {
                break; // instruction completed on this cycle
            }
        }
        let somes = seen.iter().filter(|x| x.is_some()).count();
        assert_eq!(
            somes, 1,
            "exactly one final-cycle read handle per instruction; saw {seen:?}"
        );
        assert_eq!(
            seen.last().copied().flatten(),
            Some(false),
            "the handle must land on the LAST cycle and be a read; saw {seen:?}"
        );
    }

    // ================================================================
    // φ2 NMI-line sample phase (hw_nmi_subcycle_phase).
    // ================================================================

    /// Synthetic 32 KB NROM for the φ2 NMI-phase tests. The reset
    /// routine at $C000 (PRG offset 0x4000 — NROM-256 decodes
    /// `addr & 0x7FFF`) waits out three vblanks with the idiomatic
    /// `BIT $2002 / BPL` loops — PPUCTRL writes land ~57-87k cycles in,
    /// safely past the PPU warm-up gate that silently drops them below
    /// 3×29658 dots (`Ppu::write_register`) — then enables the vblank
    /// NMI (`LDA #$80 ; STA $2000`) and parks in a 3-cycle `JMP $C014`
    /// self-loop. The NMI handler at $E000 (offset 0x6000) is a bare
    /// `RTI`, and the CPU reaches $E000 ONLY through the NMI vector
    /// (IRQ shares the vector but is never taken: reset sets the I
    /// flag and nothing CLIs), making `pc == $E000` after a step an
    /// unambiguous "NMI sequence just completed" marker for
    /// cycle-stamping the service point. With rendering off there is
    /// no odd-frame dot skip, so each frame is 89342 dots ≡ 2 (mod 3)
    /// and the vblank edge's dot slot within its CPU cycle rotates
    /// across frames — every slot, including the φ2-critical slot 2,
    /// recurs every 3 NMIs.
    fn build_nmi_spin_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(1); // CHR = 1 × 8 KB
        rom.push(0); // flags6: mapper 0, H-mirror
        rom.push(0); // flags7: mapper 0 (iNES 1.0)
        rom.extend_from_slice(&[0u8; 8]);
        let mut prg = vec![0xEAu8; 32 * 1024]; // NOP fill (never executed)
        // Reset routine at $C000 (PRG offset 0x4000): three vblank
        // waits (3rd guards the case where the flag is already set at
        // entry, which would collapse a wait to zero frames and land
        // the PPUCTRL write back under the warm-up gate).
        for w in 0..3usize {
            let base = 0x4000 + w * 5;
            prg[base] = 0x2C; // BIT abs
            prg[base + 1] = 0x02; // $2002 lo
            prg[base + 2] = 0x20; // $2002 hi
            prg[base + 3] = 0x10; // BPL (bit 7 → N: loop until vblank)
            prg[base + 4] = 0xFB; // -5 → back to the BIT
        }
        prg[0x400F] = 0xA9; // LDA #imm
        prg[0x4010] = 0x80; // #$80 — PPUCTRL bit 7 (NMI enable)
        prg[0x4011] = 0x8D; // STA abs
        prg[0x4012] = 0x00; // $2000 lo
        prg[0x4013] = 0x20; // $2000 hi
        prg[0x4014] = 0x4C; // JMP abs — 3-cycle self-loop
        prg[0x4015] = 0x14; // $C014 lo
        prg[0x4016] = 0xC0; // $C014 hi
        // NMI handler at $E000 (PRG offset 0x6000): RTI.
        prg[0x6000] = 0x40;
        // Vectors at $FFFA-$FFFF (PRG offsets 0x7FFA..0x7FFF).
        prg[0x7FFA] = 0x00; // NMI  → $E000
        prg[0x7FFB] = 0xE0;
        prg[0x7FFC] = 0x00; // RESET → $C000
        prg[0x7FFD] = 0xC0;
        prg[0x7FFE] = 0x00; // IRQ  → $E000 (I stays set — never taken)
        prg[0x7FFF] = 0xE0;
        let chr = vec![0u8; 8 * 1024];
        rom.extend(prg);
        rom.extend(chr);
        rom
    }

    fn build_spin_nes(subcycle: bool) -> Nes {
        let cart = Cartridge::load(&mut std::io::Cursor::new(build_nmi_spin_rom()))
            .expect("synthetic NMI-spin NROM should parse");
        let mut nes = Nes::new(cart);
        nes.hw_nmi_subcycle_phase = subcycle;
        nes.set_skip_render(true);
        nes
    }

    /// Run the spin ROM and stamp `nes.cycles` at each completed NMI
    /// service (the step after which PC sits at the $E000 handler
    /// entry), returning the first `n` stamps. `poll_timing` picks
    /// which boundary snapshot `poll_interrupts` consumes, so the φ2
    /// flag is exercised against both service models; either flag
    /// forces the per-cycle path via the `Nes::step` guard.
    fn nmi_service_stamps(subcycle: bool, poll_timing: bool, n: usize) -> Vec<usize> {
        let mut nes = build_spin_nes(subcycle);
        nes.cpu.hw_nmi_poll_timing = poll_timing;
        let mut v = NullSinks;
        let mut a = NullSinks;
        let mut stamps = Vec::with_capacity(n);
        for _ in 0..2_000_000usize {
            nes.step(&mut v, &mut a);
            if nes.cpu.regs.pc == 0xE000 {
                stamps.push(nes.cycles);
                if stamps.len() == n {
                    return stamps;
                }
            }
        }
        panic!(
            "expected {n} NMI services, saw {} — the spin ROM's NMI enable \
             never took effect",
            stamps.len()
        );
    }

    /// Shared assertions for the φ2 A/B: in the homogeneous 3-cycle
    /// JMP loop the modeled shift is exactly one instruction (one JMP
    /// = 3 cycles), and only for edges on the model's critical dot
    /// slot; the other slots are untouched (+0). The DIRECTION is the
    /// caller's call: standalone (poll timing off) the φ2 latch can
    /// only defer (+3, a dot-2 final-cycle edge loses its end-of-cycle
    /// visibility); composed with `hw_nmi_poll_timing` the flag also
    /// switches the boundary poll from the second-to-last-cycle latch
    /// to the φ2-of-final-cycle live value, so a final-cycle dot-0/1
    /// edge is serviced one JMP EARLIER (-3). Phases realign after
    /// each NMI (the shift is exactly one loop period), so the deltas
    /// stay per-edge.
    fn assert_phi2_shift_profile(
        on: &[usize],
        off: &[usize],
        moved: i64,
        label: &str,
    ) {
        assert_ne!(
            on, off,
            "{label}: φ2 latch never moved an NMI service point — the flag \
             is not reaching the per-dot sample suppression"
        );
        let mut zeros = 0usize;
        let mut moves = 0usize;
        for (i, (&s_on, &s_off)) in on.iter().zip(off.iter()).enumerate() {
            let d = s_on as i64 - s_off as i64;
            if d == 0 {
                zeros += 1;
            } else if d == moved {
                moves += 1;
            } else {
                panic!(
                    "{label}: nmi {i} shifted by {d} cycles (on={s_on} \
                     off={s_off}); the φ2 model predicts exactly 0 or one \
                     JMP ({moved:+})"
                );
            }
        }
        assert!(
            moves >= 1,
            "{label}: no NMI moved by one JMP ({moved:+}) — the critical \
             edge slot was never exercised (zeros={zeros})"
        );
        assert!(
            zeros >= 1,
            "{label}: every NMI shifted — non-critical edge slots must be \
             untouched (moves={moves})"
        );
    }

    #[test]
    fn nmi_subcycle_phase_shifts_service_under_poll_timing() {
        // Both machines on the per-cycle path with hardware poll timing
        // (the lockstep-lane composition); the ONLY difference is the
        // φ2 flag — which here both suppresses the dot-2 sample AND
        // switches the boundary poll to the φ2-of-final-cycle live
        // value (Mesen's snapshot). Net vs the second-to-last-cycle
        // latch: a final-cycle dot-0/1 edge is serviced one JMP
        // EARLIER (-3); a second-to-last-cycle dot-2 edge — deferred
        // +3 by suppression alone — is caught back by the later poll
        // point (net 0). The edge slot rotates across frames (see
        // build_nmi_spin_rom), so 12 NMIs cover every slot several
        // times.
        let n = 12;
        let off = nmi_service_stamps(false, true, n);
        let on = nmi_service_stamps(true, true, n);
        assert_phi2_shift_profile(&on, &off, -3, "poll_timing=on");
    }

    #[test]
    fn nmi_subcycle_phase_shifts_service_standalone() {
        // Unlike the Stage-3 dot offsets (inert without their
        // hw_mmio_* prerequisite), the φ2 latch acts standalone on the
        // legacy live-`nmi_pended` poll: a dot-2 edge on an
        // instruction's FINAL cycle is visible to the boundary poll
        // when sampled end-of-cycle but not at φ2, deferring service
        // by one instruction: shift profile {0, +3}. (Both machines
        // use the live boundary poll here, so unlike the poll-timing
        // composition there is no snapshot switch to advance service.)
        let n = 12;
        let off = nmi_service_stamps(false, false, n);
        let on = nmi_service_stamps(true, false, n);
        assert_phi2_shift_profile(&on, &off, 3, "poll_timing=off");
    }

    #[test]
    fn nmi_subcycle_event_ppu_twin_matches_legacy() {
        // The φ2 suppression must be identical through the
        // hw_event_ppu per-dot twin loops (lead loop + Stage-3 tail)
        // and the legacy `3× ppu.tick` loop: flag ON with event
        // routing ON vs OFF must stay byte-identical across several
        // NMIs. At default offsets `lead` is always 3 (tail empty), so
        // the suppression lands on the lead loop's last iteration —
        // the exact branch this test pins.
        let mut on = build_spin_nes(true);
        let mut off = build_spin_nes(true);
        on.hw_event_ppu = true;
        let mut v = NullSinks;
        let mut a = NullSinks;
        // ~60k JMP-loop instructions ≈ 6 frames — two full rotations
        // of the edge's dot slot, so the suppressed slot-2 sample is
        // exercised on both routings.
        for i in 0..60_000usize {
            on.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            if i < 6000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&on),
                    state_digest(&off),
                    "hw_event_ppu twin diverged from legacy interleave at \
                     instruction {i} with hw_nmi_subcycle_phase on: the φ2 \
                     suppression is not routing-invariant"
                );
            }
        }
    }

    #[test]
    fn nmi_subcycle_inert_without_nmi_edges() {
        // build_status_poll_rom never writes $2000, so `nmi_output`
        // stays false and the NMI line never rises: with no edge to
        // sample, the φ2 latch must be a pure no-op. Flag ON (which
        // also forces the per-cycle path via the `Nes::step` guard)
        // must stay byte-identical to flag OFF — locking that the
        // flag's ONLY observable is NMI-line sample placement (cycle
        // accounting and `$2002` polling cancel out).
        //
        // `disable_asm_cpu` on BOTH machines: under `--features
        // asm_cpu` the flag-OFF machine would otherwise run `LDA
        // $2002` through `try_step_asm`, which does not maintain the
        // interpreter's serialized mid-instruction scratch fields
        // (`opcode`/`addr_abs`/`fetched_data` — cpu_asm leaves them at
        // defaults), so a cross-path full-digest comparison diverges
        // by construction with or without this flag. ASM-vs-slow
        // parity is the asm_vs_slow_* suites' jurisdiction; this test
        // isolates the φ2 flag on a single execution path.
        let load = |flag: bool| {
            let cart =
                Cartridge::load(&mut std::io::Cursor::new(build_status_poll_rom()))
                    .expect("synthetic status-poll NROM should parse");
            let mut nes = Nes::new(cart);
            nes.hw_nmi_subcycle_phase = flag;
            nes.disable_asm_cpu = true;
            nes.set_skip_render(true);
            nes
        };
        let mut on = load(true);
        let mut off = load(false);
        let mut v = NullSinks;
        let mut a = NullSinks;
        for i in 0..15_000usize {
            on.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            if i < 4000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&on),
                    state_digest(&off),
                    "hw_nmi_subcycle_phase ON diverged from OFF at {i} with \
                     no NMI edge in the stream: the flag has an observable \
                     beyond NMI-line sampling"
                );
            }
        }
    }

    #[test]
    fn nmi_subcycle_off_is_default_and_byte_identical() {
        // Flag-OFF contract: `hw_nmi_subcycle_phase` defaults to false,
        // and OFF is the untouched legacy interleave — the suppression
        // conditions short-circuit on `!flag` and the `Nes::step` guard
        // term passes through, so OFF compiles to the identical call
        // sequence. Within one build the strongest expressible lock is
        // default-constructed vs explicitly-OFF digest lockstep across
        // many NMIs; the cross-build OFF gates are every other test in
        // this suite (all flag-OFF: nestest CYC lock, the routing
        // tests, the parity fleets) plus the line-[1] trajectory SHA of
        // scripts/verify_subcycle_offset.py compared across .so builds.
        let mut default_nes = {
            let cart =
                Cartridge::load(&mut std::io::Cursor::new(build_nmi_spin_rom()))
                    .expect("synthetic NMI-spin NROM should parse");
            let mut nes = Nes::new(cart);
            nes.set_skip_render(true);
            nes
        };
        assert!(
            !default_nes.hw_nmi_subcycle_phase,
            "hw_nmi_subcycle_phase must default OFF"
        );
        let mut off = build_spin_nes(false);
        let mut v = NullSinks;
        let mut a = NullSinks;
        // ~45k JMP-loop instructions ≈ 4-5 frames — several NMI edges.
        for i in 0..45_000usize {
            default_nes.step(&mut v, &mut a);
            off.step(&mut v, &mut a);
            if i < 4000 || i % 37 == 0 {
                assert_eq!(
                    state_digest(&default_nes),
                    state_digest(&off),
                    "explicit flag-OFF diverged from a default-constructed \
                     machine at instruction {i}"
                );
            }
        }
    }
}
