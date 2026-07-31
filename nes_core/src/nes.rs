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
                    // Rung-1 scanline-granular PPU catch-up. Only under
                    // skip_render (advance omits pixel work), a batchable
                    // mapper, and the runtime gate. APU stays per-cycle
                    // (audio / DMC / frame-IRQ), then the PPU fast-forwards
                    // in one call — observably identical to the interleaved
                    // `tick_three` loop below, which is the verbatim,
                    // byte-identical fallback for every other case.
                    if self.ppu.is_skip_render()
                        && self.ppu.scanline_advance_enabled()
                        && self.cached_ppu_batchable
                    {
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
                // Rung-1 scanline-granular PPU catch-up (see the ASM
                // remainder loop above for the invariant). Verbatim
                // interleaved fallback in the else arm.
                if self.ppu.is_skip_render()
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
        // There is 1 APU cycle per CPU cycle.
        let cpu_stall_cycles = self.apu.tick(&mut self.mapper, audio_frame_sink);
        if cpu_stall_cycles > 0 {
            self.cpu.stall(cpu_stall_cycles);
        }

        // There are 3 PPU cycles per CPU cycle.
        for _ in 0..3 {
            self.ppu.tick(&mut self.mapper, video_frame_sink);
            self.cpu
                .set_nmi_line(self.ppu.nmi_output && self.ppu.nmi_occurred);
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

        self.update_irq_line();

        self.cycles += 1;

        completed_instruction
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
