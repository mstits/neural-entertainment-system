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

    /// Cached `mapper.prg_asm_ptr()` result. Stable for the mapper's
    /// lifetime — all mapper PRG-ASM windows are fixed-size 32 KB
    /// `Vec<u8>`s mutated only via slice indexing (never resized),
    /// so the underlying allocation pointer doesn't move on bank
    /// switches. Refreshed at construction + reset + apply_state.
    /// Bypasses the per-Nes::step `MapperEnum` virtual dispatch
    /// (PGO showed 366M calls/run).
    cached_prg_asm_ptr: Option<*const u8>,
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
            cached_prg_asm_ptr: None,
        };
        nes.cached_prg_asm_ptr = nes.mapper.prg_asm_ptr();

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
                crate::cpu_asm::install_opcode_table_once();
                let ram_ptr = self.ram.as_mut_ptr();
                // Cached at construction / reset / apply_state. The
                // underlying prg_asm_window Vec is fixed-size 32 KB
                // mutated only via slice indexing; pointer is stable.
                let prg_ptr = self.cached_prg_asm_ptr;
                let raw_bulk_cycles = self.mapper.asm_bulk_cycles();
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
                let bulk_cycles = match self.ppu.cpu_cycles_until_nmi_fire() {
                    Some(n) if (n as i64) < raw_bulk_cycles => {
                        (n as i64).max(1)
                    }
                    _ => raw_bulk_cycles,
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
                    for _ in 0..remaining {
                        self.apu.tick(&mut self.mapper, audio_frame_sink);
                        self.ppu.tick_three(&mut self.mapper, video_frame_sink);
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
                for _ in 0..cycles {
                    self.apu.tick(&mut self.mapper, audio_frame_sink);
                    self.ppu.tick_three(&mut self.mapper, video_frame_sink);
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

        // ON even cycles read from CPU RAM.
        if self.cycles.is_multiple_of(2) {
            let addr = ((self.oam_dma.page as u16) << 8) | (self.oam_dma.count & 0xFF);
            self.oam_dma.data = self.ram.read_byte(addr);
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
        bus.tick_cpu_cycles_discard(8);
        drop(bus);
        self.cycles += 8;
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
        &mut self.input.game_pad_1
    }
}
