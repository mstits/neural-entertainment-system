use crate::system_bus::SystemBus;

use serde_derive::{Deserialize, Serialize};

use std::fmt;
use std::fmt::{Debug, Formatter};

pub type CycleFn = fn(&mut Cpu, bus: &mut SystemBus) -> bool;

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Instruction {
    name: &'static str,
    length: u8,
    mode: AddressMode,
    official: bool,
    cycles: &'static [CycleFn],
}

impl Instruction {
    /// Base cycle count for this opcode (number of per-cycle
    /// functions the Rust interpreter runs to complete it).
    pub fn base_cycles(&self) -> usize {
        self.cycles.len()
    }

    pub fn nestest_asm_op(&self, cpu: &Cpu, bus: &SystemBus) -> String {
        let ops = [
            bus.peek_byte(cpu.regs.pc + 1),
            bus.peek_byte(cpu.regs.pc + 2),
        ];

        match self.mode {
            AddressMode::Immediate => format!("#${:02X}", ops[0]),

            AddressMode::ZeroPage => {
                let addr = ops[0];
                let val = bus.peek_byte(addr as u16);
                format!("${:02X} = {:02X}", addr, val)
            }

            AddressMode::ZeroPageIndexed(reg) => {
                let addr = ops[0];
                let reg_val = cpu.get_register(reg);
                let target = addr.wrapping_add(reg_val);
                let val = bus.peek_byte(target as u16);
                format!("${:02X},{:?} @ {:02X} = {:02X}", addr, reg, target, val)
            }

            AddressMode::Absolute => {
                let addr = u16::from_le_bytes([ops[0], ops[1]]);
                if self.name == "JMP" || self.name == "JSR" {
                    format!("${:04X}", addr)
                } else {
                    let val = bus.peek_byte(addr);
                    format!("${:04X} = {:02X}", addr, val)
                }
            }

            AddressMode::AbsoluteIndexed(reg) => {
                let base_addr = u16::from_le_bytes([ops[0], ops[1]]);
                let reg_val = cpu.get_register(reg) as u16;
                let target = base_addr.wrapping_add(reg_val);
                let val = bus.peek_byte(target);
                format!(
                    "${:04X},{:?} @ {:04X} = {:02X}",
                    base_addr, reg, target, val
                )
            }

            AddressMode::Indirect => {
                let ptr = u16::from_le_bytes([ops[0], ops[1]]);
                // 6502 JMP Indirect bug: if ptr is $xxFF, it wraps to $xx00 for the high byte.
                let hi_ptr = if (ptr & 0x00FF) == 0x00FF {
                    ptr & 0xFF00
                } else {
                    ptr + 1
                };
                let target = u16::from_le_bytes([bus.peek_byte(ptr), bus.peek_byte(hi_ptr)]);
                format!("(${:04X}) = {:04X}", ptr, target)
            }

            AddressMode::IndexedIndirect(reg) => {
                let base = ops[0];
                let ptr = base.wrapping_add(cpu.get_register(reg));
                let lo = bus.peek_byte(ptr as u16);
                let hi = bus.peek_byte(ptr.wrapping_add(1) as u16);
                let target = u16::from_le_bytes([lo, hi]);
                let val = bus.peek_byte(target);
                format!(
                    "(${:02X},X) @ {:02X} = {:04X} = {:02X}",
                    base, ptr, target, val
                )
            }

            AddressMode::IndirectIndexed(reg) => {
                let ptr = ops[0];
                let lo = bus.peek_byte(ptr as u16);
                let hi = bus.peek_byte(ptr.wrapping_add(1) as u16);
                let base_target = u16::from_le_bytes([lo, hi]);
                let target = base_target.wrapping_add(cpu.get_register(reg) as u16);
                let val = bus.peek_byte(target);
                format!(
                    "(${:02X}),Y = {:04X} @ {:04X} = {:02X}",
                    ptr, base_target, target, val
                )
            }

            AddressMode::Relative => {
                let offset = ops[0] as i8;
                let target = cpu.regs.pc.wrapping_add(2).wrapping_add(offset as u16);
                format!("${:04X}", target)
            }

            AddressMode::Accumulator => "A".to_string(),
            AddressMode::Implied => "".to_string(),
        }
    }
}

pub const CPU_FREQUENCY: u64 = 1_789_773;

const NMI_VECTOR: u16 = 0xFFFA;
const IRQ_VECTOR: u16 = 0xFFFE;
const RESET_VECTOR: u16 = 0xFFFC;
const BRK_VECTOR: u16 = 0xFFFE;

#[derive(Copy, Clone, Default, Deserialize, Serialize)]
pub struct Flags {
    // `pub(crate)` on every flag — the bulk-step dispatch reads/
    // writes these during opcode execution + IRQ-mask checking.
    pub(crate) c: bool, // Carry
    pub(crate) z: bool, // Zero
    pub(crate) i: bool, // Interrupt inhibit
    pub(crate) d: bool, // Decimal (not used on NES)
    pub(crate) b: bool, // Break Command
    pub(crate) e: bool, // Expansion
    pub(crate) v: bool, // Overflow
    pub(crate) n: bool, // Negative
}

impl From<Flags> for u8 {
    fn from(flags: Flags) -> Self {
        (flags.c as u8)
            | ((flags.z as u8) << 1)
            | ((flags.i as u8) << 2)
            | ((flags.d as u8) << 3)
            | ((flags.b as u8) << 4)
            | ((flags.e as u8) << 5)
            | ((flags.v as u8) << 6)
            | ((flags.n as u8) << 7)
    }
}

impl From<u8> for Flags {
    fn from(bits: u8) -> Self {
        Flags {
            c: (bits & 0x01) != 0,
            z: (bits & 0x02) != 0,
            i: (bits & 0x04) != 0,
            d: (bits & 0x08) != 0,
            b: (bits & 0x10) != 0,
            e: (bits & 0x20) != 0,
            v: (bits & 0x40) != 0,
            n: (bits & 0x80) != 0,
        }
    }
}

impl Debug for Flags {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "N: {}, V: {}, e: {}, b: {}, d: {}, I: {}, Z: {}, C: {}",
            self.n as u8,
            self.v as u8,
            self.e as u8,
            self.b as u8,
            self.d as u8,
            self.i as u8,
            self.z as u8,
            self.c as u8
        )
    }
}

#[derive(Copy, Clone, Default, Deserialize, Serialize)]
pub struct Regs {
    pub pc: u16,
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
}

impl Debug for Regs {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "pc: {:04X}, a: {:02X}, x: {:02X}, y: {:02X}, sp: {:02X}",
            self.pc, self.a, self.x, self.y, self.sp
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Register8 {
    A,
    X,
    Y,
    Sp,
    Status,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AddressMode {
    Immediate,
    Absolute,
    ZeroPage,
    AbsoluteIndexed(Register8),
    ZeroPageIndexed(Register8),
    IndexedIndirect(Register8),
    IndirectIndexed(Register8),
    Accumulator,
    Implied,
    Relative,
    Indirect,
}

#[derive(Default)]
pub struct Cpu {
    // `pub(crate)` on the hot fields so `Nes::step`'s bulk-step fast
    // path (see `docs/proposals/cpu_bulk_stepping.md`) can check the
    // guard conditions (clean state, no stall, no pending interrupt)
    // before taking the bulk path. These are still not part of the
    // external crate API.
    pub(crate) stall_cycles: u8,
    pub(crate) regs: Regs,
    pub(crate) flags: Flags,
    pub(crate) instruction: Option<Instruction>,
    nmi_line_low: bool,
    pub(crate) nmi_pended: bool,
    pub irq_line_low: bool,
    cycles_total: u64,

    // Temporary mid-instruction data to be able to run one cycle at a time.
    opcode: u8,
    pub(crate) cycle: u8,
    addr_abs: u16,
    temp_addr_low: u8,
    base_addr: u8,
    rel_offset: i8,
    fetched_data: u8,

    /// Hardware-true MMIO read timing (config, not savestate state):
    /// when set, absolute-mode reads of PPU registers ($2000-$3FFF)
    /// commit on the instruction's FINAL cycle (real 6502 / Mesen
    /// behavior) instead of the LaiNES-parity cycle-0 early commit.
    /// Ground truth receipt (2026-07-30): CV boot vblank-wait
    /// `LDA $2002` whose read lands at PPU 241,8 must see the vblank
    /// flag (Mesen A=80); early-commit samples pre-241,1 and misses
    /// it, shifting every edge-straddling poll one loop iteration
    /// late and displacing lag frames vs hardware. Default OFF: every
    /// banked trajectory (SMB campaign, curriculum states, chains)
    /// replays on the legacy semantics it was recorded under.
    pub hw_mmio_read_timing: bool,

    /// Hardware-true PPU-register WRITE timing (config, not
    /// savestate): defer absolute-mode stores to $2000-$3FFF to the
    /// instruction's final cycle instead of the LaiNES-parity cycle-0
    /// early commit — the write-side counterpart of
    /// `hw_mmio_read_timing`, using the `$4014` late-write plumbing.
    /// Ground-truth receipt (2026-07-31): CV frame 11 re-enables NMI
    /// via `STA $2000` mid-vblank; the early-committed write raises
    /// the NMI edge 3 cycles before the instruction boundary so the
    /// poll services it immediately, where hardware/Mesen (write on
    /// final cycle) defer service by one instruction. That one-
    /// instruction idle-loop phase offset seeds every later NMI-
    /// quantization flip on the tape. Default OFF.
    pub hw_mmio_write_timing: bool,

    /// Hardware-true NMI service timing (config, not savestate):
    /// the real 6502 decides "service an interrupt after this
    /// instruction" at the poll on the instruction's second-to-last
    /// cycle. An NMI edge that lands during the FINAL cycle is too
    /// late — the next instruction runs to completion first. Legacy
    /// behavior services the NMI at the very next boundary no matter
    /// which cycle the edge landed on, entering handlers up to one
    /// instruction (2-3 cycles) early. Ground-truth receipt
    /// (2026-07-31): CV frame-3792 NMI entry at PPU 241,23 (ours)
    /// vs 241,31 (Mesen) — the per-frame ±3-cycle jitter that flips
    /// OAM-DMA parity and forks long tape replays. Default OFF.
    pub hw_nmi_poll_timing: bool,

    /// IRQ-mask counterpart of `hw_nmi_poll_timing`, and gated for the
    /// same reason it is: turning a hardware-accuracy timing quirk on by
    /// default changes what EVERY banked artifact replays as. CLI/SEI/PLP
    /// write `flags.i` on their own final cycle, so hardware sees the mask
    /// change one instruction later than a live read of `flags.i` does
    /// (blargg cpu_interrupts_v2 / 1-cli_latency). With this off the
    /// legacy immediate-service behaviour is preserved byte-for-byte.
    ///
    /// Default OFF deliberately. The accuracy fix is real and tested, but
    /// this project replays banked solver tapes (scripts/replay_sweep.py)
    /// and re-scores banked training runs against them, and an IRQ-timing
    /// change is exactly the class that silently invalidates those. Set it
    /// via the same --hw-all path that arms `hw_nmi_poll_timing`.
    pub hw_irq_poll_timing: bool,

    /// Mirror of `Nes::hw_nmi_subcycle_phase` (config, not savestate;
    /// synced every `Nes::tick`). When the φ2 latch is active, the
    /// live `nmi_pended` at an instruction boundary IS the value as of
    /// φ2 of the final cycle — the cycle's three dots run before its
    /// execution and the dot-2 sample is suppressed — and that is the
    /// snapshot Mesen's before-every-bus-transaction PPU catch-up
    /// consumes at its end-of-instruction check. Consuming the
    /// second-to-last-cycle `nmi_poll_latch` on top of the φ2 latch
    /// double-quantizes: measured as a CONSTANT +1-cycle NMI-entry
    /// offset vs Mesen and anti-phase OAM-DMA start parities from CV
    /// tape frame 3423 (2026-08-07, dma_parity_probe.py), the feedback
    /// loop behind the frame-3435 state fork. So `poll_interrupts`
    /// uses the live value whenever this is set.
    pub hw_nmi_subcycle_phase: bool,

    /// `nmi_pended` as of the end of the previous CPU cycle — the
    /// value the hardware poll point actually sees. Runtime latch,
    /// deliberately not serialized: at savestate boundaries the CPU
    /// sits between instructions where the latch re-syncs within one
    /// cycle.
    nmi_poll_latch: bool,

    /// `flags.i` as of the end of the previous CPU cycle — the IRQ-
    /// mask counterpart of `nmi_poll_latch`. Real hardware polls
    /// interrupts at an instruction's second-to-last cycle, but
    /// CLI/SEI/PLP write `flags.i` on their own FINAL cycle, so the
    /// mask state they set is not visible to that poll — a pending
    /// IRQ is serviced (or withheld) one whole instruction later than
    /// the write, not immediately after it. This is the classic 6502
    /// "interrupt hijacking" / CLI-latency quirk (blargg's
    /// `cpu_interrupts_v2/1-cli_latency`). Without this latch,
    /// `poll_interrupts` reads the live post-write `flags.i` and
    /// services a pending IRQ one instruction early. Runtime latch,
    /// deliberately not serialized: re-synced from `flags.i` on
    /// restore, same as `nmi_poll_latch`.
    i_poll_latch: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct State {
    pub stall_cycles: u8,
    pub regs: Regs,
    pub flags: Flags,
    pub nmi_line_low: bool,
    pub nmi_pended: bool,
    pub irq_line_low: bool,
    pub cycles_total: u64,
    pub opcode: u8,
    pub cycle: u8,
    pub addr_abs: u16,
    pub temp_addr_low: u8,
    pub base_addr: u8,
    pub rel_offset: i8,
    pub fetched_data: u8,
    pub active_interrupt: Option<Interrupt>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum Interrupt {
    Nmi,
    Irq,
}

impl Cpu {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_state(&self) -> State {
        let active_interrupt = self
            .instruction
            .map(|inst| match inst.name {
                "NMI" => Some(Interrupt::Nmi),
                "IRQ" => Some(Interrupt::Irq),
                _ => None,
            })
            .flatten();

        State {
            stall_cycles: self.stall_cycles,
            regs: self.regs,
            flags: self.flags,
            nmi_line_low: self.nmi_line_low,
            nmi_pended: self.nmi_pended,
            irq_line_low: self.irq_line_low,
            cycles_total: self.cycles_total,
            opcode: self.opcode,
            cycle: self.cycle,
            addr_abs: self.addr_abs,
            temp_addr_low: self.temp_addr_low,
            base_addr: self.base_addr,
            rel_offset: self.rel_offset,
            fetched_data: self.fetched_data,
            active_interrupt,
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.stall_cycles = state.stall_cycles;
        self.regs = state.regs;
        self.flags = state.flags;
        self.nmi_line_low = state.nmi_line_low;
        self.nmi_pended = state.nmi_pended;
        // Not serialized: re-sync so a restored pending NMI isn't
        // spuriously deferred at the first post-load boundary.
        self.nmi_poll_latch = state.nmi_pended;
        // Same re-sync for the IRQ-mask latch (see `i_poll_latch`).
        self.i_poll_latch = state.flags.i;
        self.irq_line_low = state.irq_line_low;
        self.cycles_total = state.cycles_total;
        self.opcode = state.opcode;
        self.cycle = state.cycle;
        self.addr_abs = state.addr_abs;
        self.temp_addr_low = state.temp_addr_low;
        self.base_addr = state.base_addr;
        self.rel_offset = state.rel_offset;
        self.fetched_data = state.fetched_data;

        if let Some(interrupt) = &state.active_interrupt {
            match interrupt {
                Interrupt::Nmi => self.instruction = Some(NMI_INTERRUPT),
                Interrupt::Irq => self.instruction = Some(IRQ_INTERRUPT),
            }
        } else if state.cycle == 0 {
            // BUG FIX (vs upstream RustedNES): when the saved state
            // was at cycle == 0 with no active interrupt, the CPU was
            // BETWEEN instructions — `poll_interrupts` had cleared
            // `self.instruction` to None to signal "fetch a new
            // opcode next tick." If we instead restore `instruction`
            // from the previous opcode here, the CPU re-runs that
            // instruction with stale operand state and the program
            // counter quickly walks off into garbage (observed:
            // PC=$E534 fetching 0x02 on Zelda after a save+load
            // round-trip). Setting instruction = None here makes
            // the next tick fetch fresh, matching the saved CPU
            // state's intent.
            self.instruction = None;
        } else {
            // Mid-instruction: restore the OPCODES dispatch entry so
            // the in-flight micro-op chain resumes from `state.cycle`.
            self.instruction = OPCODES[self.opcode as usize];
        }
    }

    pub fn stall(&mut self, cycles: u8) {
        self.stall_cycles += cycles;
    }

    pub fn regs(&self) -> Regs {
        self.regs
    }

    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// True iff the CPU is between instructions and the next tick
    /// would fetch a new opcode (i.e. no in-progress instruction and
    /// no remaining cycles of the current one). Used by external
    /// validators (nestest harness) to know when to capture trace
    /// state vs when to keep ticking.
    pub fn at_instruction_boundary(&self) -> bool {
        self.cycle == 0 && self.instruction.is_none()
    }

    pub fn reset(&mut self, bus: &mut SystemBus) {
        self.regs.pc = bus.read_word(RESET_VECTOR);
        self.regs.sp = 0xFD;
        self.flags = Flags {
            c: false,
            z: false,
            i: true,
            d: false,
            b: false,
            e: true,
            v: false,
            n: false,
        };
    }

    pub fn initialize_nestest(&mut self) {
        self.regs.pc = 0xC000;
        self.regs.sp = 0xFD;
        self.regs.a = 0x00;
        self.regs.x = 0x00;
        self.regs.y = 0x00;
        self.flags = 0x24.into();
        self.cycles_total = 7;
    }

    // Run a single CPU cycle. Returns whether an instruction completed.
    pub fn tick(&mut self, bus: &mut SystemBus) -> bool {
        if self.stall_cycles > 0 {
            self.stall_cycles -= 1;
            // The interrupt-poll pipeline keeps sampling while the CPU
            // is halted for DMA/DMC stalls.
            self.nmi_poll_latch = self.nmi_pended;
            self.i_poll_latch = self.flags.i;
            return false;
        }

        if self.cycle == 0 {
            self.cycle = 1;
            self.cycles_total += 1;

            if self.instruction.is_none() {
                // Fetch new instruction.
                self.opcode = self.next_pc_byte(bus);
                self.instruction = OPCODES[self.opcode as usize];
                // LaiNES-parity early commit for MMIO-touching
                // absolute-addressing load/store opcodes. See
                // `cycle_zero_early_commit` rustdoc.
                self.cycle_zero_early_commit(bus);
            }

            self.nmi_poll_latch = self.nmi_pended;
            self.i_poll_latch = self.flags.i;
            return false;
        }

        // Continue in-progress instruction.
        let instr = match self.instruction {
            Some(instr) => instr,
            None => {
                // Defensive fallback for the 16 opcodes that have no
                // `Instruction` entry in `OPCODES`: the 12 KIL/JAM
                // slots (0x02/0x12/0x22/0x32/0x42/0x52/0x62/0x72/0x92/
                // 0xB2/0xD2/0xF2, all 1-byte) plus the unstable TAS
                // $9B / LAS $BB / SHA $9F (3-byte, absolute,Y) and SHA
                // $93 (2-byte, (indirect),Y). The opcode byte itself
                // was already consumed by `next_pc_byte` during fetch,
                // so PC must advance by exactly the operand bytes
                // still owed for the real instruction length, or the
                // decode stream desyncs from the next real instruction
                // (a previous flat `+1` assumed every one of these was
                // 2 bytes total, which is wrong for both the 1-byte
                // KIL/JAM slots and the 3-byte TAS/LAS/SHA $9F forms).
                // Crucially, still poll interrupts so a pending NMI/IRQ
                // is serviced at this instruction boundary rather than
                // being silently deferred by one whole instruction.
                let operand_bytes: u16 = match self.opcode {
                    0x9B | 0xBB | 0x9F => 2,
                    0x93 => 1,
                    _ => 0,
                };
                self.regs.pc = self.regs.pc.wrapping_add(operand_bytes);
                self.cycle = 0;
                self.opcode = 0;
                self.poll_interrupts();
                return true;
            }
        };

        let cycle_fn = instr.cycles[self.cycle as usize - 1];
        let completed = cycle_fn(self, bus);

        self.cycle += 1;
        self.cycles_total += 1;

        if completed || self.cycle as usize > instr.cycles.len() {
            self.cycle = 0;
            // NOTE: poll_interrupts consumes `nmi_poll_latch` and
            // `i_poll_latch` (state as of the end of the second-to-
            // last cycle) and re-syncs them — do not update either
            // latch before this call.
            self.poll_interrupts();
            true
        } else {
            self.nmi_poll_latch = self.nmi_pended;
            self.i_poll_latch = self.flags.i;
            false
        }
    }

    pub fn trace(&self, bus: &SystemBus) -> String {
        let opcode = bus.peek_byte(self.regs.pc);
        let instruction = match OPCODES[opcode as usize] {
            Some(instruction) => instruction,
            None => panic!("Unimplemented opcode: {:#04x}", opcode),
        };

        let mut hex_extra = Vec::new();
        for i in 1..instruction.length {
            hex_extra.push(bus.peek_byte(self.regs.pc + i as u16));
        }

        let hex_extra = hex_extra
            .iter()
            .map(|z| format!("{:02X}", z))
            .collect::<Vec<String>>()
            .join(" ");

        let asm_op = instruction.nestest_asm_op(self, bus);

        format!(
            "{:04X}  {:02X} {:<5} {}{:<3} {:<27} A:{:02X} X:{:02X} Y:{:02X} P:{:02X} SP:{:02X} PPU:{:>3},{:>3} CYC:{}",
            self.regs.pc,
            opcode,
            hex_extra,
            if instruction.official { ' ' } else { '*' },
            instruction.name,
            asm_op,
            self.regs.a,
            self.regs.x,
            self.regs.y,
            u8::from(self.flags),
            self.regs.sp,
            bus.ppu_scanline(),
            bus.ppu_scanline_cycle(),
            self.cycles_total,
        )
    }

    pub(crate) fn poll_interrupts(&mut self) {
        // Hardware polls interrupts at the end of the instruction's
        // second-to-last cycle: under `hw_nmi_poll_timing` the service
        // decision uses `nmi_poll_latch` (the pended state as of the
        // end of the PREVIOUS cycle), so an edge landing during the
        // final cycle defers service by one instruction. Legacy path
        // uses the live `nmi_pended` (immediate service). Under the φ2
        // latch (`hw_nmi_subcycle_phase`) the live value already
        // carries the sub-cycle quantization — it equals pended as of
        // φ2 of the FINAL cycle, the snapshot Mesen consumes — so the
        // second-to-last-cycle latch must NOT stack on top of it (see
        // the field doc for the CV receipt).
        let take_nmi = if self.hw_nmi_poll_timing && !self.hw_nmi_subcycle_phase {
            self.nmi_poll_latch
        } else {
            self.nmi_pended
        };
        if take_nmi {
            self.instruction = Some(NMI_INTERRUPT);
        } else if self.irq_line_low && !(if self.hw_irq_poll_timing {
            self.i_poll_latch
        } else {
            self.flags.i
        }) {
            // Use `i_poll_latch` (flags.i as of the end of the
            // PREVIOUS cycle), not the live `flags.i`: CLI/SEI/PLP
            // write flags.i on their own final cycle, and the poll
            // happens at that same instruction boundary. Reading the
            // live value would let the mask change take effect one
            // instruction earlier than hardware — the "interrupt
            // hijacking" / CLI-latency quirk (blargg's
            // `cpu_interrupts_v2/1-cli_latency`). See the
            // `i_poll_latch` field doc.
            self.instruction = Some(IRQ_INTERRUPT);
        } else {
            // No interrupt? Then we proceed to fetch a normal opcode.
            // We set instruction to None (or a Fetch state) so Cycle 0
            // knows to read from PC.
            self.instruction = None;
        }
        // A deferred edge (pended but not latched at the poll point)
        // becomes visible to the NEXT boundary.
        self.nmi_poll_latch = self.nmi_pended;
        self.i_poll_latch = self.flags.i;
    }

    #[inline(always)]
    fn next_pc_byte(&mut self, bus: &mut SystemBus) -> u8 {
        let b = bus.read_byte(self.regs.pc);
        self.regs.pc = self.regs.pc.wrapping_add(1);
        b
    }

    ///////////////////////
    // Flag helpers
    ///////////////////////

    #[inline(always)]
    fn set_zero_negative(&mut self, result: u8) {
        self.flags.z = result == 0;
        self.flags.n = (result & 0x80) != 0;
    }

    ///////////////////////
    // Register helpers
    ///////////////////////

    #[inline(always)]
    fn get_register(&self, r: Register8) -> u8 {
        use self::Register8::*;
        match r {
            A => self.regs.a,
            X => self.regs.x,
            Y => self.regs.y,
            Sp => self.regs.sp,
            Status => self.flags.into(),
        }
    }

    #[inline(always)]
    fn set_register(&mut self, r: Register8, val: u8) {
        use self::Register8::*;
        match r {
            A => self.regs.a = val,
            X => self.regs.x = val,
            Y => self.regs.y = val,
            Sp => self.regs.sp = val,
            Status => self.flags = val.into(),
        }
    }

    //////////////////////
    // Instruction helpers
    //////////////////////

    #[inline(always)]
    fn add_value(&mut self, value: u8) {
        let result = self.regs.a as u32 + value as u32 + self.flags.c as u32;

        self.flags.c = (result & 0x100) != 0;
        let result = result as u8;
        self.flags.v =
            ((self.regs.a & 0x80) == (value & 0x80)) && (self.regs.a & 0x80 != result & 0x80);
        self.set_zero_negative(result);

        self.regs.a = result;
    }

    #[inline(always)]
    fn sub_value(&mut self, value: u8) {
        let result = self.regs.a as i32 - value as i32 - (!self.flags.c) as i32;

        self.flags.c = result >= 0;

        let result = result as u8;
        self.flags.v =
            ((self.regs.a ^ value) & 0x80 != 0) && (((self.regs.a ^ result) & 0x80) != 0);
        self.set_zero_negative(result);

        self.regs.a = result;
    }

    #[inline(always)]
    fn and_value(&mut self, value: u8) -> u8 {
        let result = value & self.regs.a;
        self.set_zero_negative(result);
        self.regs.a = result;
        result
    }

    #[inline(always)]
    fn ora_value(&mut self, value: u8) {
        let result = value | self.regs.a;
        self.set_zero_negative(result);
        self.regs.a = result;
    }

    #[inline(always)]
    fn eor_value(&mut self, value: u8) {
        let result = value ^ self.regs.a;
        self.set_zero_negative(result);
        self.regs.a = result;
    }

    // Push byte onto the stack
    #[inline(always)]
    fn push_byte(&mut self, bus: &mut SystemBus, val: u8) {
        let s = self.regs.sp;
        bus.write_byte(0x0100 | (s as u16), val);
        self.regs.sp = s.wrapping_sub(1);
    }

    // Pull byte from the stack
    #[inline(always)]
    fn pull_byte(&mut self, bus: &mut SystemBus) -> u8 {
        self.regs.sp = self.regs.sp.wrapping_add(1);

        bus.read_byte(0x0100 | (self.regs.sp as u16))
    }

    ///////////////
    // Interrupts
    ///////////////

    // Request NMI to be run on next instruction fetch attempt.
    pub fn set_nmi_line(&mut self, is_low: bool) {
        if !self.nmi_line_low && is_low {
            self.nmi_pended = true;
        }
        self.nmi_line_low = is_low;
    }

    // Request IRQ to be run on next instruction fetch attempt.
    #[inline(always)]
    pub fn set_irq_line_low(&mut self, val: bool) {
        self.irq_line_low = val;
    }

    ///////////////////////////////////////
    // Official Instruction Cycle Functions
    ///////////////////////////////////////

    fn fetch_rel_offset_and_check_branch_n(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        !self.flags.n
    }

    fn fetch_rel_offset_and_check_branch_not_n(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        self.flags.n
    }

    fn fetch_rel_offset_and_check_branch_c(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        !self.flags.c
    }

    fn fetch_rel_offset_and_check_branch_not_c(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        self.flags.c
    }

    fn fetch_rel_offset_and_check_branch_z(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        !self.flags.z
    }

    fn fetch_rel_offset_and_check_branch_not_z(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        self.flags.z
    }

    fn fetch_rel_offset_and_check_branch_v(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        !self.flags.v
    }

    fn fetch_rel_offset_and_check_branch_not_v(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.rel_offset = self.next_pc_byte(bus) as i8;
        self.flags.v
    }

    fn take_branch(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let old_pc = self.regs.pc;
        let new_pc = self.regs.pc.wrapping_add_signed(self.rel_offset as i16);

        // Dummy fetch using using old high byte and new low byte of the pc.
        let dummy_addr = (old_pc & 0xFF00) | (new_pc & 0x00FF);
        let _ = bus.read_byte(dummy_addr);

        self.regs.pc = new_pc;

        // If not crossing a page, this is the last cycle.
        if (old_pc & 0xFF00) == (new_pc & 0xFF00) {
            return true;
        }

        // When there is a page cross, there is an extra cycle to fix up the page.
        // We've already set the PC correctly, but we still need the extra cycle.
        false
    }

    // Only called if there was a page cross.
    fn branch_page_fixup(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        // Read from corrected PC.
        let _ = bus.read_byte(self.regs.pc);
        true
    }

    fn fetch_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetched_data = self.next_pc_byte(bus);
        false
    }

    fn fetch_abs_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.addr_abs = self.next_pc_byte(bus) as u16;
        false
    }

    fn fetch_abs_high(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let high = (self.next_pc_byte(bus) as u16) << 8;
        self.addr_abs |= high;
        false
    }

    fn fetch_base_addr(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.base_addr = self.next_pc_byte(bus);
        false
    }

    fn dummy_read_base(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.base_addr as u16);
        false
    }

    fn dummy_read(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.regs.pc);
        false
    }

    fn dummy_fetch_and_inc_pc(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.regs.pc);
        self.regs.pc = self.regs.pc.wrapping_add(1);
        false
    }

    fn indexed_x_dummy_read_and_add(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.base_addr as u16);
        self.base_addr = self.base_addr.wrapping_add(self.regs.x);
        false
    }

    fn indexed_fetch_ptr_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.temp_addr_low = bus.read_byte(self.base_addr as u16);
        false
    }

    fn indexed_fetch_ptr_high(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let addr_high = bus.read_byte(self.base_addr.wrapping_add(1) as u16);
        self.addr_abs = ((addr_high as u16) << 8) | (self.temp_addr_low as u16);
        false
    }

    fn lda_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.regs.a = self.fetched_data;
        self.set_zero_negative(self.regs.a);
        true
    }

    /// Unified bulk-step dispatch — executes one full CPU
    /// instruction in one Rust call if the opcode has a fast-path
    /// implementation, returning the cycle count consumed. Returns
    /// `None` when the opcode isn't (yet) in the bulk-step set — the
    /// caller falls back to the per-cycle path.
    ///
    /// See `docs/proposals/cpu_bulk_stepping.md`. Phase 1 landed
    /// 0xA9 (LDA imm). This module grows one match arm per opcode
    /// in Phase 2, prioritized by instruction frequency on real
    /// NES code (LDA/STA/LDX/LDY zero-page + absolute, ALU ops,
    /// branches, JMP/JSR/RTS, etc.).
    ///
    /// Correctness invariants the caller must establish BEFORE
    /// calling (checked in `Nes::step`'s guard block, not here for
    /// inlining reasons):
    /// 1. `self.cycle == 0 && self.instruction.is_none()` — at a
    ///    clean instruction boundary.
    /// 2. `self.stall_cycles == 0` — no DMC DMA stall pending.
    /// 3. No interrupt is pending (NMI / IRQ).
    /// 4. `!oam_dma.active`.
    /// 5. Opcode-specific: operand addresses don't land in MMIO
    ///    (`$2000-$401F` or mapper-specific MMIO). Checked per-arm
    ///    when needed.
    ///
    /// The caller is responsible for advancing APU + PPU by the
    /// returned cycle count after this returns.
    #[inline]
    pub fn try_bulk_step(&mut self, bus: &mut SystemBus, opcode: u8) -> Option<u8> {
        match opcode {
            // --- Immediate-mode loads (2 cycles, no memory read
            //     beyond PC+1 which is always code-space) ---
            0xA9 => {
                // LDA immediate
                self.regs.pc = self.regs.pc.wrapping_add(1); // skip opcode
                let imm = self.next_pc_byte(bus);
                self.regs.a = imm;
                self.set_zero_negative(imm);
                self.cycles_total += 2;
                Some(2)
            }
            0xA2 => {
                // LDX immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.regs.x = imm;
                self.set_zero_negative(imm);
                self.cycles_total += 2;
                Some(2)
            }
            0xA0 => {
                // LDY immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.regs.y = imm;
                self.set_zero_negative(imm);
                self.cycles_total += 2;
                Some(2)
            }

            // --- Zero-page loads (3 cycles; zero-page is always
            //     $00-$FF which is RAM — never MMIO, safe to bulk) ---
            0xA5 => {
                // LDA zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let val = bus.read_byte(zp as u16);
                self.regs.a = val;
                self.set_zero_negative(val);
                self.cycles_total += 3;
                Some(3)
            }
            0xA6 => {
                // LDX zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let val = bus.read_byte(zp as u16);
                self.regs.x = val;
                self.set_zero_negative(val);
                self.cycles_total += 3;
                Some(3)
            }
            0xA4 => {
                // LDY zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let val = bus.read_byte(zp as u16);
                self.regs.y = val;
                self.set_zero_negative(val);
                self.cycles_total += 3;
                Some(3)
            }

            // --- Zero-page stores (3 cycles; always RAM) ---
            0x85 => {
                // STA zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                bus.write_byte(zp as u16, self.regs.a);
                self.cycles_total += 3;
                Some(3)
            }
            0x86 => {
                // STX zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                bus.write_byte(zp as u16, self.regs.x);
                self.cycles_total += 3;
                Some(3)
            }
            0x84 => {
                // STY zero-page
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                bus.write_byte(zp as u16, self.regs.y);
                self.cycles_total += 3;
                Some(3)
            }

            // --- Register transfers (2 cycles, register-only) ---
            0xAA => {
                // TAX
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.x = self.regs.a;
                self.set_zero_negative(self.regs.x);
                self.cycles_total += 2;
                Some(2)
            }
            0xA8 => {
                // TAY
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.y = self.regs.a;
                self.set_zero_negative(self.regs.y);
                self.cycles_total += 2;
                Some(2)
            }
            0x8A => {
                // TXA
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.a = self.regs.x;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x98 => {
                // TYA
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.a = self.regs.y;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0xBA => {
                // TSX — transfer stack pointer to X
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.x = self.regs.sp;
                self.set_zero_negative(self.regs.x);
                self.cycles_total += 2;
                Some(2)
            }
            0x9A => {
                // TXS — transfer X to stack pointer (no flags)
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.sp = self.regs.x;
                self.cycles_total += 2;
                Some(2)
            }

            // --- Flag ops (2 cycles, register-only) ---
            0x18 => {
                // CLC
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.c = false;
                self.cycles_total += 2;
                Some(2)
            }
            0x38 => {
                // SEC
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.c = true;
                self.cycles_total += 2;
                Some(2)
            }
            0x58 => {
                // CLI — enables interrupts. Don't take fast path if
                // this would un-mask a pending IRQ (safer to let the
                // per-cycle path handle the transition).
                if self.irq_line_low {
                    None
                } else {
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.flags.i = false;
                    self.cycles_total += 2;
                    Some(2)
                }
            }
            0x78 => {
                // SEI
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.i = true;
                self.cycles_total += 2;
                Some(2)
            }
            0xD8 => {
                // CLD (no-op on NES)
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.d = false;
                self.cycles_total += 2;
                Some(2)
            }
            0xF8 => {
                // SED (no-op-ish on NES)
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.d = true;
                self.cycles_total += 2;
                Some(2)
            }
            0xB8 => {
                // CLV
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.v = false;
                self.cycles_total += 2;
                Some(2)
            }

            // --- Implied register ops ---
            0xE8 => {
                // INX
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.x = self.regs.x.wrapping_add(1);
                self.set_zero_negative(self.regs.x);
                self.cycles_total += 2;
                Some(2)
            }
            0xC8 => {
                // INY
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.y = self.regs.y.wrapping_add(1);
                self.set_zero_negative(self.regs.y);
                self.cycles_total += 2;
                Some(2)
            }
            0xCA => {
                // DEX
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.x = self.regs.x.wrapping_sub(1);
                self.set_zero_negative(self.regs.x);
                self.cycles_total += 2;
                Some(2)
            }
            0x88 => {
                // DEY
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.regs.y = self.regs.y.wrapping_sub(1);
                self.set_zero_negative(self.regs.y);
                self.cycles_total += 2;
                Some(2)
            }
            0xEA => {
                // NOP
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.cycles_total += 2;
                Some(2)
            }

            // --- Branches (2 cycles base; +1 if taken, +1 if page
            //     cross). All relative to PC after operand fetch. ---
            0x10 => Some(branch(self, bus, !self.flags.n)), // BPL
            0x30 => Some(branch(self, bus, self.flags.n)),  // BMI
            0x50 => Some(branch(self, bus, !self.flags.v)), // BVC
            0x70 => Some(branch(self, bus, self.flags.v)),  // BVS
            0x90 => Some(branch(self, bus, !self.flags.c)), // BCC
            0xB0 => Some(branch(self, bus, self.flags.c)),  // BCS
            0xD0 => Some(branch(self, bus, !self.flags.z)), // BNE
            0xF0 => Some(branch(self, bus, self.flags.z)),  // BEQ

            // --- JMP absolute (3 cycles) ---
            0x4C => {
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.next_pc_byte(bus) as u16;
                let hi = self.next_pc_byte(bus) as u16;
                self.regs.pc = (hi << 8) | lo;
                self.cycles_total += 3;
                Some(3)
            }

            // --- Immediate-mode compares (2 cycles) ---
            0xC9 => {
                // CMP immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                let a = self.regs.a;
                self.flags.c = a >= imm;
                self.set_zero_negative(a.wrapping_sub(imm));
                self.cycles_total += 2;
                Some(2)
            }
            0xE0 => {
                // CPX immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                let x = self.regs.x;
                self.flags.c = x >= imm;
                self.set_zero_negative(x.wrapping_sub(imm));
                self.cycles_total += 2;
                Some(2)
            }
            0xC0 => {
                // CPY immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                let y = self.regs.y;
                self.flags.c = y >= imm;
                self.set_zero_negative(y.wrapping_sub(imm));
                self.cycles_total += 2;
                Some(2)
            }

            // --- Immediate-mode ALU (2 cycles, reg-only after
            //     operand fetch) ---
            0x29 => {
                // AND immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.regs.a &= imm;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x09 => {
                // ORA immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.regs.a |= imm;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x49 => {
                // EOR immediate
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.regs.a ^= imm;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }

            // --- ADC / SBC immediate (2 cycles, NES ignores
            //     decimal-mode flag — behaves as binary always) ---
            0x69 => {
                // ADC immediate — reuse the existing `add_value`
                // helper so carry/overflow flag logic matches the
                // per-cycle path bit-for-bit.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.add_value(imm);
                self.cycles_total += 2;
                Some(2)
            }
            0xE9 | 0xEB => {
                // SBC immediate (0xE9 official, 0xEB illegal alias).
                // SBC is ADC of one's complement.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let imm = self.next_pc_byte(bus);
                self.add_value(imm ^ 0xFF);
                self.cycles_total += 2;
                Some(2)
            }

            // --- Zero-page ALU (3 cycles; zp is always RAM) ---
            0x25 => {
                // AND zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                self.regs.a &= v;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 3;
                Some(3)
            }
            0x05 => {
                // ORA zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                self.regs.a |= v;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 3;
                Some(3)
            }
            0x45 => {
                // EOR zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                self.regs.a ^= v;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 3;
                Some(3)
            }
            0xC5 => {
                // CMP zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                let a = self.regs.a;
                self.flags.c = a >= v;
                self.set_zero_negative(a.wrapping_sub(v));
                self.cycles_total += 3;
                Some(3)
            }
            0xE4 => {
                // CPX zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                let x = self.regs.x;
                self.flags.c = x >= v;
                self.set_zero_negative(x.wrapping_sub(v));
                self.cycles_total += 3;
                Some(3)
            }
            0xC4 => {
                // CPY zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                let y = self.regs.y;
                self.flags.c = y >= v;
                self.set_zero_negative(y.wrapping_sub(v));
                self.cycles_total += 3;
                Some(3)
            }
            0x24 => {
                // BIT zp — N = bit7(m), V = bit6(m), Z = (A & m) == 0.
                // No effect on A.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16);
                self.flags.z = (self.regs.a & v) == 0;
                self.flags.n = (v & 0x80) != 0;
                self.flags.v = (v & 0x40) != 0;
                self.cycles_total += 3;
                Some(3)
            }

            // --- Absolute-mode loads (4 cycles; address must not
            //     hit MMIO — checked at runtime after fetching the
            //     operand word). ---
            0xAD => {
                // LDA absolute
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.next_pc_byte(bus) as u16;
                let hi = self.next_pc_byte(bus) as u16;
                let addr = (hi << 8) | lo;
                if (0x2000..=0x401F).contains(&addr) {
                    // MMIO read — must take the per-cycle path for
                    // side-effect ordering. Rewind PC and bail so the
                    // slow path re-fetches the opcode cleanly.
                    self.regs.pc = self.regs.pc.wrapping_sub(3);
                    return None;
                }
                let v = bus.read_byte(addr);
                self.regs.a = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0xAE => {
                // LDX absolute
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.next_pc_byte(bus) as u16;
                let hi = self.next_pc_byte(bus) as u16;
                let addr = (hi << 8) | lo;
                if (0x2000..=0x401F).contains(&addr) {
                    self.regs.pc = self.regs.pc.wrapping_sub(3);
                    return None;
                }
                let v = bus.read_byte(addr);
                self.regs.x = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0xAC => {
                // LDY absolute
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.next_pc_byte(bus) as u16;
                let hi = self.next_pc_byte(bus) as u16;
                let addr = (hi << 8) | lo;
                if (0x2000..=0x401F).contains(&addr) {
                    self.regs.pc = self.regs.pc.wrapping_sub(3);
                    return None;
                }
                let v = bus.read_byte(addr);
                self.regs.y = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }

            // Absolute-mode stores + absolute,X/Y loads were tried
            // as bulk-path arms. Net: -2.6% regression on canonical
            // Zelda bench (PGO puts these as cold code because
            // Zelda rarely uses them), +6.5% on Contra. Kept out of
            // the match to preserve the common-case perf — the
            // fewer arms in `try_bulk_step`, the tighter the
            // dispatch. A future per-game-specialized core could
            // selectively enable these based on the profile
            // workload, but that's complex infrastructure for
            // marginal gain.

            // --- Zero-page INC/DEC (5 cycles; zp = RAM) ---
            0xE6 => {
                // INC zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16).wrapping_add(1);
                bus.write_byte(zp as u16, v);
                self.set_zero_negative(v);
                self.cycles_total += 5;
                Some(5)
            }
            0xC6 => {
                // DEC zp
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let v = bus.read_byte(zp as u16).wrapping_sub(1);
                bus.write_byte(zp as u16, v);
                self.set_zero_negative(v);
                self.cycles_total += 5;
                Some(5)
            }

            // --- Zero-page,X addressing (4 cycles; zp+X wraps in
            //     zero-page = always RAM). HIGH FREQUENCY on games
            //     that use X-indexed tables. ---
            0xB5 => {
                // LDA zp,X
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.x) as u16;
                let v = bus.read_byte(addr);
                self.regs.a = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0xB4 => {
                // LDY zp,X
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.x) as u16;
                let v = bus.read_byte(addr);
                self.regs.y = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0xB6 => {
                // LDX zp,Y
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.y) as u16;
                let v = bus.read_byte(addr);
                self.regs.x = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0x95 => {
                // STA zp,X
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.x) as u16;
                bus.write_byte(addr, self.regs.a);
                self.cycles_total += 4;
                Some(4)
            }
            0x94 => {
                // STY zp,X
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.x) as u16;
                bus.write_byte(addr, self.regs.y);
                self.cycles_total += 4;
                Some(4)
            }
            0x96 => {
                // STX zp,Y
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let zp = self.next_pc_byte(bus);
                let addr = zp.wrapping_add(self.regs.y) as u16;
                bus.write_byte(addr, self.regs.x);
                self.cycles_total += 4;
                Some(4)
            }

            // --- Stack ops (PHA/PLA/PHP/PLP) — stack is $0100-$01FF
            //     = RAM, always safe. 3-4 cycles. ---
            0x48 => {
                // PHA (3)
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.push_byte(bus, self.regs.a);
                self.cycles_total += 3;
                Some(3)
            }
            0x68 => {
                // PLA (4)
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let v = self.pull_byte(bus);
                self.regs.a = v;
                self.set_zero_negative(v);
                self.cycles_total += 4;
                Some(4)
            }
            0x08 => {
                // PHP (3) — pushes flags with B=1, E=1 conventions.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let mut f = self.flags;
                f.b = true;
                f.e = true;
                self.push_byte(bus, u8::from(f));
                self.cycles_total += 3;
                Some(3)
            }
            0x28 => {
                // PLP (4) — pull flags; B/E are ignored (stay as
                // hardware-defined values).
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let byte = self.pull_byte(bus);
                self.flags = byte.into();
                self.flags.e = true;
                self.flags.b = false;
                self.cycles_total += 4;
                Some(4)
            }

            // --- JSR absolute (6 cycles) + RTS (6 cycles) ---
            // These are extremely hot in real 6502 code — every
            // subroutine call + return.
            0x20 => {
                // JSR $abs. 6502 quirk: pushes PC-1 (address of the
                // high byte of the JSR operand), then RTS pops and
                // adds 1.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.next_pc_byte(bus) as u16;
                // DO NOT fetch high byte yet — PC is currently at
                // high-byte address. That's what we push.
                let push_pc = self.regs.pc; // high-byte address
                self.push_byte(bus, (push_pc >> 8) as u8);
                self.push_byte(bus, (push_pc & 0xFF) as u8);
                let hi = self.next_pc_byte(bus) as u16;
                self.regs.pc = (hi << 8) | lo;
                self.cycles_total += 6;
                Some(6)
            }
            0x60 => {
                // RTS — pull PC-1, then add 1.
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let lo = self.pull_byte(bus) as u16;
                let hi = self.pull_byte(bus) as u16;
                self.regs.pc = ((hi << 8) | lo).wrapping_add(1);
                self.cycles_total += 6;
                Some(6)
            }

            // --- Accumulator shifts/rotates (2 cycles, reg-only) ---
            0x0A => {
                // ASL A
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.c = (self.regs.a & 0x80) != 0;
                self.regs.a = self.regs.a.wrapping_shl(1);
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x4A => {
                // LSR A
                self.regs.pc = self.regs.pc.wrapping_add(1);
                self.flags.c = (self.regs.a & 0x01) != 0;
                self.regs.a = self.regs.a.wrapping_shr(1);
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x2A => {
                // ROL A
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let carry_in = if self.flags.c { 1 } else { 0 };
                self.flags.c = (self.regs.a & 0x80) != 0;
                self.regs.a = (self.regs.a.wrapping_shl(1)) | carry_in;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }
            0x6A => {
                // ROR A
                self.regs.pc = self.regs.pc.wrapping_add(1);
                let carry_in = if self.flags.c { 0x80 } else { 0 };
                self.flags.c = (self.regs.a & 0x01) != 0;
                self.regs.a = (self.regs.a.wrapping_shr(1)) | carry_in;
                self.set_zero_negative(self.regs.a);
                self.cycles_total += 2;
                Some(2)
            }

            _ => None,
        }
    }

    /// Legacy wrapper around the unified dispatch for the Phase 1
    /// bench — kept so `step_one_lda_immediate_bulk` and the
    /// bulk_step_bench module don't break. Real callers should use
    /// `try_bulk_step` directly.
    #[inline]
    pub fn step_lda_immediate_bulk(&mut self, bus: &mut SystemBus) -> u8 {
        self.try_bulk_step(bus, 0xA9).expect("LDA-imm has a bulk impl")
    }

    fn ldx_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.regs.x = self.fetched_data;
        self.set_zero_negative(self.regs.x);
        true
    }

    fn ldy_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.regs.y = self.fetched_data;
        self.set_zero_negative(self.regs.y);
        true
    }

    fn lda_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ld_abs_indexed_optimistic(bus, Register8::A, Register8::X)
    }

    fn lda_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ld_abs_indexed_optimistic(bus, Register8::A, Register8::Y)
    }

    fn ldx_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ld_abs_indexed_optimistic(bus, Register8::X, Register8::Y)
    }

    fn ldy_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ld_abs_indexed_optimistic(bus, Register8::Y, Register8::X)
    }

    #[inline(always)]
    // Returns true if page crossed.
    fn indexed_optimistic_fetch(&mut self, bus: &mut SystemBus, index: u8) -> bool {
        let speculative_addr =
            (self.addr_abs & 0xFF00) | ((self.addr_abs as u8).wrapping_add(index) as u16);
        self.addr_abs = self.addr_abs.wrapping_add(index as u16);
        self.fetched_data = bus.read_byte(speculative_addr);

        (speculative_addr & 0xFF00) != (self.addr_abs & 0xFF00)
    }

    #[inline(always)]
    fn ld_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        ld_reg: Register8,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.set_register(ld_reg, self.fetched_data);
            self.set_zero_negative(self.get_register(ld_reg));
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn lda_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.regs.a = bus.read_byte(self.addr_abs);
        self.set_zero_negative(self.regs.a);
        true
    }

    fn ldx_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.regs.x = bus.read_byte(self.addr_abs);
        self.set_zero_negative(self.regs.x);
        true
    }

    fn ldy_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.regs.y = bus.read_byte(self.addr_abs);
        self.set_zero_negative(self.regs.y);
        true
    }

    fn lda_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x as u16;
        let addr = (self.base_addr as u16 + index) % 0x0100;
        self.regs.a = bus.read_byte(addr);
        self.set_zero_negative(self.regs.a);
        true
    }

    fn ldx_indexed_y_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y as u16;
        let addr = (self.base_addr as u16 + index) % 0x0100;
        self.regs.x = bus.read_byte(addr);
        self.set_zero_negative(self.regs.x);
        true
    }

    fn ldy_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x as u16;
        let addr = (self.base_addr as u16 + index) % 0x0100;
        self.regs.y = bus.read_byte(addr);
        self.set_zero_negative(self.regs.y);
        true
    }

    fn sta_write_byte_abs(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.regs.a);
        true
    }

    /// No-op cycle handler. Used by opcodes whose entire work runs
    /// synchronously at the opcode-fetch tick (see
    /// `cycle_zero_early_commit`). These noops exist only to consume
    /// the correct total cycle count — PPU ticks 3x per CPU cycle in
    /// the outer loop, so the instruction still occupies its spec-
    /// accurate number of CPU cycles.
    fn noop_cycle(_self: &mut Cpu, _bus: &mut SystemBus) -> bool {
        false
    }

    /// Special last-cycle handler for STA $abs — performs the deferred
    /// $4014 (OAM DMA) write at the END of the instruction. For all
    /// other absolute addresses, the write happened at cycle 0 via
    /// `cycle_zero_early_commit` and this acts as a noop.
    ///
    /// Why deferred: with early-commit, slow's DMA `dummy_read` first
    /// tick fires at self.cycles=N+1 (cycle 0 of STA + 1 nes.tick
    /// increment) while ASM's fires at N+4 (after the full 4-cycle
    /// STA via ASM completes). The 3-cycle parity flip changes the
    /// alignment-tick count by 1 and breaks asm-vs-slow lockstep on
    /// every OAM DMA. Deferring slow's $4014 write to N+4 too brings
    /// both paths into agreement (and matches real-hardware timing,
    /// where DMA stall is post-instruction).
    ///
    /// Cost: shifts nes-py parity for ROMs that hit OAM DMA on a
    /// timing-sensitive scanline. Re-baselined against Mesen.
    fn sta_abs_4014_late_write(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.addr_abs == 0x4014 || self_.defer_ppu_write() {
            bus.write_byte(self_.addr_abs, self_.regs.a);
        }
        false
    }

    /// STX/STY $abs analogues of `sta_abs_4014_late_write`. Nearly
    /// every game arms OAM DMA with STA, but Konami CNROM titles use
    /// STY (Gradius: `8C 14 40` at $8087). Without the deferral the
    /// slow path commits the $4014 write at cycle 0 while ASM commits
    /// after the full 4-cycle store — the DMA dummy_read parity flips
    /// and asm-vs-slow lockstep diverges by 1 cycle on every DMA,
    /// exactly the STA failure mode fixed 2026-04-26.
    fn stx_abs_4014_late_write(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.addr_abs == 0x4014 || self_.defer_ppu_write() {
            bus.write_byte(self_.addr_abs, self_.regs.x);
        }
        false
    }

    fn sty_abs_4014_late_write(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.addr_abs == 0x4014 || self_.defer_ppu_write() {
            bus.write_byte(self_.addr_abs, self_.regs.y);
        }
        false
    }

    /// LaiNES-parity early commit: for the MMIO-reading load opcodes
    /// and MMIO-writing store opcodes with absolute addressing,
    /// execute the entire instruction work (operand fetch + MMIO
    /// read/write) during the opcode-fetch tick (CPU cycle N of the
    /// instruction). The per-cycle `cycles` array runs noops for
    /// cycles N+1..N+k to maintain the spec-correct total cycle count.
    ///
    /// Why: LaiNES's `skip_cycles` model does all instruction work at
    /// dispatch (effectively CPU cycle N), then idles. PPU observes
    /// MMIO side effects at PPU cycle 3*N. Our previous per-cycle
    /// model did the MMIO access at the LAST cycle (PPU cycle
    /// 3*(N+k)). For $2002 polling loops this shifted our "first
    /// vblank observed" to an earlier CPU-op position than nes-py's,
    /// causing Zelda's NMI-enable / cave-transition sequence to take
    /// a different code path and break sword pickup at tape f1011.
    /// Diagnosed via per-bus-access diff
    /// (`scripts/tracing/diff_bus_traces.py`) 2026-04-24. See
    /// `docs/proposals/archive/zelda_cave_stuck_investigation.md`.
    fn cycle_zero_early_commit(&mut self, bus: &mut SystemBus) {
        match self.opcode {
            // STA $abs (4 cycles): fetch 2 operand bytes, write A.
            //
            // EXCEPT $4014 — see `sta_abs_4014_late_write` rationale.
            // OAM DMA timing aligns with where the write happens; if
            // slow path commits at cycle 0 but ASM at cycle 3, the
            // DMA dummy_read parity flips and the two paths diverge
            // by 1 cycle on every DMA. Defer slow to match ASM (and
            // real hardware).
            0x8D => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if self.addr_abs != 0x4014 && !self.defer_ppu_write() {
                    bus.write_byte(self.addr_abs, self.regs.a);
                }
            }
            // STX $abs (4 cycles). $4014 deferred like STA — see
            // `stx_abs_4014_late_write`.
            0x8E => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if self.addr_abs != 0x4014 && !self.defer_ppu_write() {
                    bus.write_byte(self.addr_abs, self.regs.x);
                }
            }
            // STY $abs (4 cycles). $4014 deferred like STA — see
            // `sty_abs_4014_late_write`.
            0x8C => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if self.addr_abs != 0x4014 && !self.defer_ppu_write() {
                    bus.write_byte(self.addr_abs, self.regs.y);
                }
            }
            // LDA $abs (4 cycles): fetch 2 operand bytes, read into A.
            // PPU-register reads defer to the final cycle under
            // hw_mmio_read_timing — see `abs_late_ppu_read_a`.
            0xAD => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if !self.defer_ppu_read() {
                    self.regs.a = bus.read_byte(self.addr_abs);
                    self.set_zero_negative(self.regs.a);
                }
            }
            // LDX $abs (4 cycles).
            0xAE => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if !self.defer_ppu_read() {
                    self.regs.x = bus.read_byte(self.addr_abs);
                    self.set_zero_negative(self.regs.x);
                }
            }
            // LDY $abs (4 cycles).
            0xAC => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if !self.defer_ppu_read() {
                    self.regs.y = bus.read_byte(self.addr_abs);
                    self.set_zero_negative(self.regs.y);
                }
            }
            // BIT $abs (4 cycles): the canonical vblank-poll idiom
            // `BIT $2002 / BPL loop` lives here. Matching LaiNES's
            // read-at-T0 semantics for BIT specifically is what closes
            // the Zelda cave-stuck root cause identified via bus-
            // trace diff (`diff_bus_traces.py`).
            0x2C => {
                let lo = self.next_pc_byte(bus) as u16;
                let hi = (self.next_pc_byte(bus) as u16) << 8;
                self.addr_abs = hi | lo;
                if !self.defer_ppu_read() {
                    let m = bus.read_byte(self.addr_abs);
                    self.flags.n = (m & 0x80) != 0;
                    self.flags.v = (m & 0x40) != 0;
                    self.flags.z = (m & self.regs.a) == 0;
                }
            }
            _ => {}
        }
    }

    /// True when the just-decoded absolute-mode MMIO read must wait
    /// for the instruction's final cycle (hardware-true PPU register
    /// read timing; see the `hw_mmio_read_timing` field doc).
    #[inline]
    fn defer_ppu_read(&self) -> bool {
        self.hw_mmio_read_timing && (0x2000..=0x3FFF).contains(&self.addr_abs)
    }

    /// True when the deferred access this cycle is a READ of PPUSTATUS
    /// (`$2002`, including its every-8-bytes mirrors up to `$3FFF`).
    /// Narrower than `pending_deferred_ppu_access`, which reports any
    /// `$2000-$3FFF` access: the `$2002`/vblank read race
    /// (`Nes::hw_vblank_read_race`) is a property of the PPUSTATUS latch
    /// specifically — a deferred `$2007` read at the same dot must not
    /// drain it.
    #[inline]
    pub(crate) fn pending_deferred_status_read(&self) -> bool {
        self.pending_deferred_ppu_access() == Some(false) && (self.addr_abs & 0x2007) == 0x2002
    }

    /// Write-side mirror of `defer_ppu_read` — see the
    /// `hw_mmio_write_timing` field doc.
    fn defer_ppu_write(&self) -> bool {
        self.hw_mmio_write_timing && (0x2000..=0x3FFF).contains(&self.addr_abs)
    }

    /// Sub-cycle bus-catch-up hook (event-driven PPU Stage 3). Returns
    /// `Some(is_write)` when the CPU is poised to run the FINAL cycle of
    /// an absolute-mode load/store whose deferred bus access targets a
    /// PPU-observable register ($2000-$3FFF) or `$4014` OAM DMA — i.e.
    /// the next `Cpu::tick` will perform that access through a late-commit
    /// handler (`abs_late_ppu_read_*` / `*_abs_4014_late_write`).
    /// Otherwise `None`.
    ///
    /// `Nes::tick` queries this BEFORE `cpu.tick` so it can hold the PPU
    /// at the exact intra-cycle dot the access samples instead of the end
    /// of the CPU cycle. It only fires on the deferred-access cycle the
    /// `hw_mmio_read_timing` / `hw_mmio_write_timing` (and always-deferred
    /// `$4014`) machinery already pins, inheriting that exact-cycle
    /// placement and adding only the dot offset within the cycle. With
    /// those flags OFF a PPU-register access is early-committed at cycle 0
    /// (`cycle_zero_early_commit`) and this returns `None`, so the legacy
    /// end-of-cycle timing is untouched.
    #[inline]
    pub(crate) fn pending_deferred_ppu_access(&self) -> Option<bool> {
        let instr = self.instruction?;
        // The late-commit handler is the last entry in the cycle table;
        // `Cpu::tick` runs it on the tick it enters with
        // `cycle == cycles.len()` (see `Cpu::tick` dispatch).
        if self.cycle as usize != instr.base_cycles() {
            return None;
        }
        match self.opcode {
            // LDA/LDX/LDY/BIT $abs — deferred PPU-register reads.
            0xAD | 0xAE | 0xAC | 0x2C if self.defer_ppu_read() => Some(false),
            // STA/STX/STY $abs — deferred PPU-register writes and the
            // always-deferred `$4014` OAM DMA arm.
            0x8D | 0x8E | 0x8C if self.defer_ppu_write() || self.addr_abs == 0x4014 => {
                Some(true)
            }
            _ => None,
        }
    }

    /// Final-cycle handlers for the deferred PPU-register reads — the
    /// read-side mirror of `sta_abs_4014_late_write`. When the read
    /// was NOT deferred (flag off or non-PPU target) these are noops:
    /// the work already happened at cycle 0.
    fn abs_late_ppu_read_a(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.defer_ppu_read() {
            self_.regs.a = bus.read_byte(self_.addr_abs);
            self_.set_zero_negative(self_.regs.a);
        }
        false
    }

    fn abs_late_ppu_read_x(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.defer_ppu_read() {
            self_.regs.x = bus.read_byte(self_.addr_abs);
            self_.set_zero_negative(self_.regs.x);
        }
        false
    }

    fn abs_late_ppu_read_y(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.defer_ppu_read() {
            self_.regs.y = bus.read_byte(self_.addr_abs);
            self_.set_zero_negative(self_.regs.y);
        }
        false
    }

    fn abs_late_ppu_read_bit(self_: &mut Cpu, bus: &mut SystemBus) -> bool {
        if self_.defer_ppu_read() {
            let m = bus.read_byte(self_.addr_abs);
            self_.flags.n = (m & 0x80) != 0;
            self_.flags.v = (m & 0x40) != 0;
            self_.flags.z = (m & self_.regs.a) == 0;
        }
        false
    }

    fn stx_write_byte_abs(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.regs.x);
        true
    }

    fn sty_write_byte_abs(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.regs.y);
        true
    }

    fn sta_write_byte_abs_indexed_x(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        bus.write_byte(addr, self.regs.a);
        true
    }

    fn stx_write_byte_abs_indexed_y(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        bus.write_byte(addr, self.regs.x);
        true
    }

    fn sty_write_byte_abs_indexed_x(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        bus.write_byte(addr, self.regs.y);
        true
    }

    fn st_abs_indexed_x_dummy_read(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x as u16;
        let base_addr = self.addr_abs;
        self.addr_abs = base_addr + index;
        bus.read_byte((base_addr & 0xFF00) | (self.addr_abs & 0x00FF));

        false
    }

    fn st_abs_indexed_y_dummy_read(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y as u16;
        let base_addr = self.addr_abs;
        self.addr_abs = base_addr + index;
        bus.read_byte((base_addr & 0xFF00) | (self.addr_abs & 0x00FF));

        false
    }

    fn adc_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.adc_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn adc_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.adc_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    fn adc_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.add_value(self.fetched_data);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn adc_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.add_value(self.fetched_data);
        true
    }

    fn adc_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.add_value(value);
        true
    }

    fn adc_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.add_value(value);
        true
    }

    fn sbc_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.sbc_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn sbc_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.sbc_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    fn sbc_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.sub_value(self.fetched_data);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn sbc_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.sub_value(self.fetched_data);
        true
    }

    fn sbc_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.sub_value(value);
        true
    }

    fn sbc_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.sub_value(value);
        true
    }

    fn and_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.and_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn and_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.and_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    #[inline(always)]
    fn and_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.and_value(self.fetched_data);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn and_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.and_value(self.fetched_data);
        true
    }

    fn and_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.and_value(value);
        true
    }

    fn and_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.and_value(value);
        true
    }

    fn ora_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ora_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn ora_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.ora_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    #[inline(always)]
    fn ora_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.ora_value(self.fetched_data);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn ora_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.ora_value(self.fetched_data);
        true
    }

    fn ora_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.ora_value(value);
        true
    }

    fn ora_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.ora_value(value);
        true
    }

    fn eor_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.eor_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn eor_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.eor_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    fn eor_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.eor_value(self.fetched_data);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn eor_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.eor_value(self.fetched_data);
        true
    }

    fn eor_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.eor_value(value);
        true
    }

    fn eor_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.eor_value(value);
        true
    }

    fn sec(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.c = true;
        true
    }

    fn clc(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.c = false;
        true
    }

    fn sei(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.i = true;
        true
    }

    fn cli(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.i = false;
        true
    }

    fn sed(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.d = true;
        true
    }

    fn cld(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.d = false;
        true
    }

    fn clv(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.flags.v = false;
        true
    }

    fn cmp_addr_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.cmp_addr_abs_indexed_optimistic(bus, Register8::X)
    }

    fn cmp_addr_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        self.cmp_addr_abs_indexed_optimistic(bus, Register8::Y)
    }

    fn cmp_addr_abs_indexed_optimistic(
        &mut self,
        bus: &mut SystemBus,
        index_reg: Register8,
    ) -> bool {
        let index = self.get_register(index_reg);
        if !self.indexed_optimistic_fetch(bus, index) {
            // No page cross, so the instruction can finish without an extra cycle.
            self.compare_value(self.fetched_data, self.regs.a);
            true
        } else {
            // Page cross. We need  an extra cycle
            false
        }
    }

    fn compare_value(&mut self, value: u8, reg_value: u8) {
        let result = reg_value.wrapping_sub(value);
        self.set_zero_negative(result);
        self.flags.c = value <= reg_value;
    }

    fn cmp_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.compare_value(self.fetched_data, self.regs.a);
        true
    }

    fn cpx_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.compare_value(self.fetched_data, self.regs.x);
        true
    }

    fn cpy_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetch_immediate(bus);
        self.compare_value(self.fetched_data, self.regs.y);
        true
    }

    fn cmp_zero_page_indexed_x_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        let value = bus.read_byte(addr);
        self.compare_value(value, self.regs.a);
        true
    }

    fn cmp_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.compare_value(value, self.regs.a);
        true
    }

    fn cpx_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.compare_value(value, self.regs.x);
        true
    }

    fn cpy_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = bus.read_byte(self.addr_abs);
        self.compare_value(value, self.regs.y);
        true
    }

    fn bit_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let m = bus.read_byte(self.addr_abs);
        let a = self.regs.a;

        self.flags.n = (m & 0x80) != 0;
        self.flags.v = (m & 0x40) != 0;
        self.flags.z = (m & a) == 0;

        true
    }

    fn read_abs_addr_data(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetched_data = bus.read_byte(self.addr_abs);
        false
    }

    fn read_zero_page_indexed_x_data(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        self.addr_abs = (self.base_addr as u16 + index as u16) % 0x0100;
        self.fetched_data = bus.read_byte(self.addr_abs);
        false
    }

    fn rmw_abs_indexed_x_dummy_read(&mut self, bus: &mut SystemBus) -> bool {
        let lo = self.addr_abs.wrapping_add(self.regs.x as u16) & 0x00FF;
        let uncorrected_addr = (self.addr_abs & 0xFF00) | lo;
        bus.read_byte(uncorrected_addr);
        false
    }

    fn rmw_abs_indexed_y_dummy_read(&mut self, bus: &mut SystemBus) -> bool {
        let lo = self.addr_abs.wrapping_add(self.regs.y as u16) & 0x00FF;
        let uncorrected_addr = (self.addr_abs & 0xFF00) | lo;
        bus.read_byte(uncorrected_addr);
        false
    }

    fn read_abs_indexed_x_data(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.x;
        self.addr_abs = self.addr_abs.wrapping_add(index as u16);
        self.fetched_data = bus.read_byte(self.addr_abs);
        false
    }

    fn read_abs_indexed_y_data(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y;
        self.addr_abs = self.addr_abs.wrapping_add(index as u16);
        self.fetched_data = bus.read_byte(self.addr_abs);
        false
    }

    fn addr_abs_fetched_data_dummy_write(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.fetched_data);
        false
    }

    fn inc_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.fetched_data.wrapping_add(1);
        self.set_zero_negative(value);
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn dec_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.fetched_data.wrapping_sub(1);
        self.set_zero_negative(value);
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn inx(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        let val = self.regs.x.wrapping_add(1);
        self.set_zero_negative(val);
        self.regs.x = val;
        true
    }

    fn iny(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        let val = self.regs.y.wrapping_add(1);
        self.set_zero_negative(val);
        self.regs.y = val;
        true
    }

    fn dex(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        let val = self.regs.x.wrapping_sub(1);
        self.set_zero_negative(val);
        self.regs.x = val;
        true
    }

    fn dey(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        let val = self.regs.y.wrapping_sub(1);
        self.set_zero_negative(val);
        self.regs.y = val;
        true
    }

    fn tax(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.set_zero_negative(self.regs.a);
        self.regs.x = self.regs.a;
        true
    }

    fn txa(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.set_zero_negative(self.regs.x);
        self.regs.a = self.regs.x;
        true
    }

    fn tay(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.set_zero_negative(self.regs.a);
        self.regs.y = self.regs.a;
        true
    }

    fn tya(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.set_zero_negative(self.regs.y);
        self.regs.a = self.regs.y;
        true
    }

    fn tsx(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.set_zero_negative(self.regs.sp);
        self.regs.x = self.regs.sp;
        true
    }

    fn txs(&mut self, bus: &mut SystemBus) -> bool {
        self.dummy_read(bus);
        self.regs.sp = self.regs.x;
        true
    }

    fn dummy_stack_read(&mut self, bus: &mut SystemBus) -> bool {
        let _ = bus.read_byte(0x0100 | (self.regs.sp as u16));
        false
    }

    fn push_pc_high(&mut self, bus: &mut SystemBus) -> bool {
        self.push_byte(bus, (self.regs.pc >> 8) as u8);
        false
    }

    fn push_pc_low(&mut self, bus: &mut SystemBus) -> bool {
        self.push_byte(bus, (self.regs.pc & 0xFF) as u8);
        false
    }

    fn brk_push_status(&mut self, bus: &mut SystemBus) -> bool {
        let mut status = self.flags;
        status.b = true;
        status.e = true;
        self.push_byte(bus, status.into());
        self.flags.i = true;
        false
    }

    fn interrupt_push_status(&mut self, bus: &mut SystemBus) -> bool {
        let mut status = self.flags;
        status.b = false;
        status.e = true;
        self.push_byte(bus, status.into());
        // Per 6502 spec, IRQ entry sets the I flag AFTER pushing P,
        // mirroring brk_push_status. Without this, mapper IRQs that
        // hold the IRQ line asserted (MMC3 scanline IRQ, MMC5,
        // FME-7) keep re-entering the handler each instruction
        // boundary, draining the stack until underflow. Symptom in
        // the wild: Crash 'n' the Boys cold-boot trap at $FFF7
        // within 19 frames; see KNOWN_ISSUES.md.
        self.flags.i = true;
        false
    }

    fn pull_low(&mut self, bus: &mut SystemBus) -> bool {
        self.addr_abs = self.pull_byte(bus) as u16;
        false
    }

    fn pull_high(&mut self, bus: &mut SystemBus) -> bool {
        let lo = self.addr_abs;
        let hi = self.pull_byte(bus) as u16;
        self.addr_abs = (hi << 8) | lo;
        false
    }

    fn pull_status(&mut self, bus: &mut SystemBus) -> bool {
        self.flags = self.pull_byte(bus).into();
        self.flags.e = true;
        self.flags.b = false;
        false
    }

    fn jsr_finish(&mut self, bus: &mut SystemBus) -> bool {
        let addr_lo = self.addr_abs;
        let addr_hi = self.next_pc_byte(bus) as u16;
        self.regs.pc = (addr_hi << 8) | addr_lo;
        true
    }

    fn rts_finish(&mut self, _bus: &mut SystemBus) -> bool {
        self.regs.pc = self.addr_abs + 1;
        true
    }

    fn pha_finish(&mut self, bus: &mut SystemBus) -> bool {
        self.push_byte(bus, self.regs.a);
        true
    }

    fn inc_sp(&mut self, _bus: &mut SystemBus) -> bool {
        self.regs.sp = self.regs.sp.wrapping_add(1);
        false
    }

    fn pla_finish(&mut self, bus: &mut SystemBus) -> bool {
        let val = bus.read_byte(0x0100 | (self.regs.sp as u16));
        self.set_zero_negative(val);
        self.regs.a = val;
        true
    }

    fn php_finish(&mut self, bus: &mut SystemBus) -> bool {
        let mut status = self.flags;
        status.b = true;
        status.e = true;
        self.push_byte(bus, status.into());
        true
    }

    fn plp_finish(&mut self, bus: &mut SystemBus) -> bool {
        let val = bus.read_byte(0x0100 | (self.regs.sp as u16));
        let flags: u8 = self.flags.into();
        self.flags = ((val & 0xCF) | (flags & 0x30)).into();
        true
    }

    fn lsr_accumulator_finish(&mut self, _bus: &mut SystemBus) -> bool {
        self.flags.c = (self.regs.a & 0x01) != 0;
        self.regs.a = (self.regs.a >> 1) & 0x7F;
        self.set_zero_negative(self.regs.a);
        true
    }

    fn lsr_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = (self.fetched_data >> 1) & 0x7F;
        self.set_zero_negative(value);
        self.flags.c = (self.fetched_data & 0x01) != 0;
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn asl_accumulator_finish(&mut self, _bus: &mut SystemBus) -> bool {
        self.flags.c = (self.regs.a & 0x80) != 0;
        self.regs.a = (self.regs.a << 1) & 0xFE;
        self.set_zero_negative(self.regs.a);
        true
    }

    fn asl_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.flags.c = (self.fetched_data & 0x80) != 0;
        let value = (self.fetched_data << 1) & 0xFE;
        self.set_zero_negative(value);
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn ror_accumulator_finish(&mut self, _bus: &mut SystemBus) -> bool {
        let carry: u8 = if self.flags.c { 1 << 7 } else { 0 };
        self.flags.c = (self.regs.a & 0x01) != 0;
        self.regs.a = ((self.regs.a >> 1) & 0x7F) | carry;
        self.set_zero_negative(self.regs.a);
        true
    }

    fn ror_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let carry: u8 = if self.flags.c { 1 << 7 } else { 0 };
        self.flags.c = (self.fetched_data & 0x01) != 0;
        let value = ((self.fetched_data >> 1) & 0x7F) | carry;
        self.set_zero_negative(value);
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn rol_accumulator_finish(&mut self, _bus: &mut SystemBus) -> bool {
        let carry: u8 = self.flags.c.into();
        self.flags.c = (self.regs.a & 0x80) != 0;
        self.regs.a = ((self.regs.a << 1) & 0xFE) | carry;
        self.set_zero_negative(self.regs.a);
        true
    }

    fn rol_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let carry: u8 = self.flags.c.into();
        self.flags.c = (self.fetched_data & 0x80) != 0;
        let value = ((self.fetched_data << 1) & 0xFE) | carry;
        self.set_zero_negative(value);
        bus.write_byte(self.addr_abs, value);
        true
    }

    fn read_brk_vector_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let vector_addr = if self.nmi_pended {
            NMI_VECTOR // Hijacked by NMI
        } else {
            BRK_VECTOR
        };
        self.temp_addr_low = bus.read_byte(vector_addr);
        false
    }

    fn brk_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let vector_addr = if self.nmi_pended {
            NMI_VECTOR // Hijacked by NMI
        } else {
            BRK_VECTOR
        };
        let hi = bus.read_byte(vector_addr + 1) as u16;
        self.regs.pc = (hi << 8) | (self.temp_addr_low as u16);

        // If hijacked by NMI, we need to clear it.
        if self.nmi_pended {
            self.nmi_pended = false;
        }

        true
    }

    fn read_nmi_vector_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.nmi_pended = false;
        self.temp_addr_low = bus.read_byte(NMI_VECTOR);
        false
    }

    fn nmi_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let hi = bus.read_byte(NMI_VECTOR + 1) as u16;
        self.regs.pc = (hi << 8) | (self.temp_addr_low as u16);
        true
    }

    fn read_irq_vector_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.temp_addr_low = bus.read_byte(IRQ_VECTOR);
        false
    }

    fn irq_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let hi = bus.read_byte(IRQ_VECTOR + 1) as u16;
        self.regs.pc = (hi << 8) | (self.temp_addr_low as u16);
        true
    }

    fn rti_finish(self: &mut Cpu, _bus: &mut SystemBus) -> bool {
        self.regs.pc = self.addr_abs;
        true
    }

    fn nop_implied(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.regs.pc);
        true
    }

    fn jmp_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let hi = self.next_pc_byte(bus) as u16;
        self.regs.pc = (hi << 8) | self.addr_abs;
        true
    }

    fn jmp_indirect_read_pointer_low(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.fetched_data = bus.read_byte(self.addr_abs);
        false
    }

    fn jump_indirect_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        // There is a hardware bug in the JNI instruction. If the 16-bit argument of an indirect JMP is
        // located between 2 pages (0x01FF and 0x0200 for example), then the LSB will be read from
        // 0x01FF and the MSB will be read from 0x0100.
        let msb = bus.read_byte(if (self.addr_abs & 0xFF) == 0xFF {
            self.addr_abs & 0xff00
        } else {
            self.addr_abs + 1
        });

        let lsb = self.fetched_data;

        self.regs.pc = ((msb as u16) << 8) | (lsb as u16);

        true
    }

    ///////////////////////////////////////
    // Unofficial Instruction Cycle Functions
    ///////////////////////////////////////

    fn isb_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.fetched_data.wrapping_add(1);
        bus.write_byte(self.addr_abs, value);
        self.sub_value(value);
        true
    }

    fn skb(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.regs.pc);
        self.regs.pc = self.regs.pc.wrapping_add(1);
        true
    }

    fn alr(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.next_pc_byte(bus);
        self.and_value(value);
        self.lsr_accumulator_finish(bus);
        true
    }

    fn anc(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.next_pc_byte(bus);
        self.and_value(value);
        self.flags.c = self.flags.n;
        true
    }

    fn arr(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let imm = self.next_pc_byte(bus);
        self.regs.a = ((self.flags.c as u8) << 7) | ((self.regs.a & imm) >> 1);

        let a = self.regs.a;
        self.set_zero_negative(a);
        self.flags.c = (a & 0x40) != 0;
        self.flags.v = ((a ^ (a << 1)) & 0x40) != 0;
        true
    }

    fn axs(&mut self, bus: &mut SystemBus) -> bool {
        let imm = self.next_pc_byte(bus);
        let a_and_x = self.regs.a & self.regs.x;
        let result = (a_and_x).wrapping_sub(imm);
        self.set_zero_negative(result);
        self.flags.c = imm <= a_and_x;
        self.regs.x = result;
        true
    }

    fn lax_abs_indexed_y_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        let completed = self.ld_abs_indexed_optimistic(bus, Register8::A, Register8::Y);
        self.regs.x = self.regs.a;
        completed
    }

    fn lax_immediate(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.lda_immediate(bus);
        self.regs.x = self.regs.a;
        true
    }

    fn lax_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.lda_addr_abs_finish(bus);
        self.regs.x = self.regs.a;
        true
    }

    fn lax_indexed_y_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y as u16;
        let addr = (self.base_addr as u16 + index) % 0x0100;
        self.regs.a = bus.read_byte(addr);
        self.set_zero_negative(self.regs.a);
        self.regs.x = self.regs.a;
        true
    }

    fn dcp_fetched_data_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = self.fetched_data.wrapping_sub(1);
        bus.write_byte(self.addr_abs, value);
        self.compare_value(value, self.regs.a);
        true
    }

    fn ign_abs_indexed_x_optimistic(&mut self, bus: &mut SystemBus) -> bool {
        !self.indexed_optimistic_fetch(bus, self.regs.x)
    }

    fn ign_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.read_byte(self.addr_abs);
        true
    }

    // Used by "Gaau Hok Gwong Cheung (Ch)"
    // This instruction can be unpredictable.
    // See http://visual6502.org/wiki/index.php?title=6502_Opcode_8B_%28XAA,_ANE%29
    fn xaa(&mut self, bus: &mut SystemBus) -> bool {
        self.regs.a = self.regs.a & self.regs.x & self.next_pc_byte(bus);
        true
    }

    fn rla_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let carry: u8 = self.flags.c.into();
        self.flags.c = (self.fetched_data & 0x80) != 0;
        let value = ((self.fetched_data << 1) & 0xFE) | carry;
        bus.write_byte(self.addr_abs, value);
        self.and_value(value);
        true
    }

    fn rra_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let carry: u8 = if self.flags.c { 1 << 7 } else { 0 };
        self.flags.c = (self.fetched_data & 0x01) != 0;
        let value = ((self.fetched_data >> 1) & 0x7F) | carry;
        bus.write_byte(self.addr_abs, value);
        self.add_value(value);
        true
    }

    fn slo_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        self.flags.c = (self.fetched_data & 0x80) != 0;
        let value = (self.fetched_data << 1) & 0xFE;
        bus.write_byte(self.addr_abs, value);
        self.ora_value(value);
        true
    }

    fn sre_addr_abs_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let value = (self.fetched_data >> 1) & 0x7F;
        self.flags.c = (self.fetched_data & 0x01) != 0;
        bus.write_byte(self.addr_abs, value);
        self.eor_value(value);
        true
    }

    fn sax_write_byte_abs(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.regs.a & self.regs.x);
        true
    }

    fn sax_write_byte_abs_indexed_y(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        let index = self.regs.y;
        let addr = (self.base_addr as u16 + index as u16) % 0x0100;
        bus.write_byte(addr, self.regs.a & self.regs.x);
        true
    }

    fn shy_logic(self: &mut Cpu, _bus: &mut SystemBus) -> bool {
        let base = self.addr_abs;
        let value = self.regs.y;
        let index = self.regs.x;

        let addr = base + index as u16;

        self.fetched_data = value & ((base >> 8) + 1) as u8;

        self.addr_abs = if ((base ^ addr) & 0x100) != 0 {
            // Page crossed
            (addr & ((value as u16) << 8)) | (addr & 0x00FF)
        } else {
            addr
        };

        false
    }

    fn shy_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.fetched_data);
        true
    }

    fn shx_logic(self: &mut Cpu, _bus: &mut SystemBus) -> bool {
        let base = self.addr_abs;
        let value = self.regs.x;
        let index = self.regs.y;

        let addr = base + index as u16;

        self.fetched_data = value & ((base >> 8) + 1) as u8;

        self.addr_abs = if ((base ^ addr) & 0x100) != 0 {
            // Page crossed
            (addr & ((value as u16) << 8)) | (addr & 0x00FF)
        } else {
            addr
        };

        false
    }

    fn shx_finish(self: &mut Cpu, bus: &mut SystemBus) -> bool {
        bus.write_byte(self.addr_abs, self.fetched_data);
        true
    }
}

/// Shared helper for the 8 relative-branch opcodes
/// (BPL/BMI/BVC/BVS/BCC/BCS/BNE/BEQ) used by `Cpu::try_bulk_step`.
/// PC points at the branch opcode on entry; this advances past the
/// opcode + signed offset byte, then either takes the branch (with
/// correct cycle accounting for taken + page-cross) or falls through.
///
/// Returns the cycle count: 2 if not taken, 3 if taken same-page,
/// 4 if taken crossing a page boundary.
#[inline]
fn branch(cpu: &mut Cpu, bus: &mut SystemBus, take: bool) -> u8 {
    cpu.regs.pc = cpu.regs.pc.wrapping_add(1); // skip opcode
    let offset = cpu.next_pc_byte(bus) as i8;
    if !take {
        cpu.cycles_total += 2;
        return 2;
    }
    let old_pc = cpu.regs.pc;
    let new_pc = old_pc.wrapping_add(offset as u16);
    cpu.regs.pc = new_pc;
    let page_cross = (old_pc & 0xFF00) != (new_pc & 0xFF00);
    let cycles = if page_cross { 4 } else { 3 };
    cpu.cycles_total += cycles as u64;
    cycles
}

#[cfg(test)]
mod bulk_step_bench {
    //! Proof-of-concept perf comparison for the block-interpreter
    //! design in `docs/proposals/cpu_bulk_stepping.md`.
    //!
    //! Builds a minimal in-memory NES running a synthetic program of
    //! LDA-immediate instructions and times two execution modes:
    //!
    //! - **per-cycle**: the existing `Nes::step` hot path —
    //!   `while !nes.tick(...) {}` loop that advances 1 APU + 3 PPU +
    //!   1 CPU sub-cycle per iteration.
    //! - **bulk**: `Cpu::step_lda_immediate_bulk` directly — one call
    //!   per LDA instruction with a single arithmetic advance of
    //!   PPU/APU counter state.
    //!
    //! **Gate** (docs/proposals/cpu_bulk_stepping.md Phase 1): if
    //! bulk wins by ≥1.3× on this synthetic, commit to expanding
    //! the block-interpreter to the top ~20 opcodes in Phase 2.
    //! Otherwise pivot.
    //!
    //! Run: `cargo test --release --lib bulk_step_bench -- --nocapture`

    use super::*;
    use crate::cartridge::Cartridge;
    use crate::nes::Nes;
    use crate::sink::{AudioSink, VideoSink};
    use std::time::Instant;

    const N_ITERS: usize = 200_000;

    /// Build a synthetic iNES ROM whose reset vector points at a
    /// loop of LDA-immediate instructions. 32KB PRG so NROM mapper 0
    /// applies; CHR size set to 0 (CHR-RAM). Reset vector is at
    /// $FFFC-$FFFD pointing to $C000 (start of PRG).
    fn build_lda_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        // iNES header: "NES\x1a" + 2×16KB PRG + 0×8KB CHR + flags.
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(0); // CHR = 0 (CHR-RAM)
        rom.push(0); // flags6: mapper 0 low nibble, H-mirror
        rom.push(0); // flags7: mapper 0 high nibble
        rom.extend_from_slice(&[0u8; 8]); // padding bytes 8..=15

        // PRG contents: 32 KB of `A9 42` (LDA #$42) at every address,
        // with the reset vector at $FFFC → $C000.
        let mut prg = vec![0u8; 32 * 1024];
        // Fill most of PRG with LDA #$42 (A9 42) — tight loop of
        // 2-byte instructions that will execute in sequence. Won't
        // infinite-loop the bench because we stop after a fixed
        // iteration count.
        for i in (0..prg.len()).step_by(2) {
            prg[i] = 0xA9; // LDA immediate
            if i + 1 < prg.len() {
                prg[i + 1] = 0x42;
            }
        }
        // Reset vector at $FFFC-$FFFD (= offset 32K-4, 32K-3 within PRG).
        let reset_lo = prg.len() - 4;
        prg[reset_lo] = 0x00; // low byte of $C000
        prg[reset_lo + 1] = 0xC0; // high byte of $C000
        rom.extend(prg);
        rom
    }

    struct DiscardSinks;
    impl VideoSink for DiscardSinks {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool {
            false
        }
        fn pixel_size(&self) -> usize {
            4
        }
    }
    impl AudioSink for DiscardSinks {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize {
            0
        }
    }

    fn build_nes() -> Nes {
        let rom = build_lda_rom();
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom))
            .expect("synthetic iNES should parse");
        Nes::new(cart)
    }

    #[test]
    fn bulk_vs_per_cycle_lda_immediate() {
        // --- per-cycle path ---
        let mut nes = build_nes();
        nes.reset();
        let mut vsink = DiscardSinks;
        let mut asink = DiscardSinks;
        // Each Nes::step advances ONE full CPU instruction (tick-loop
        // until `completed_instruction`). Perfect for per-instruction
        // timing.
        // Warm up to prime icache + branch predictor.
        for _ in 0..2000 {
            nes.step(&mut vsink, &mut asink);
        }
        let t0 = Instant::now();
        for _ in 0..N_ITERS {
            nes.step(&mut vsink, &mut asink);
        }
        let per_cycle_ns = t0.elapsed().as_nanos() as f64 / N_ITERS as f64;

        // --- bulk path ---
        // Same setup, but bypass Nes::step's per-cycle loop by calling
        // `Cpu::step_lda_immediate_bulk` directly + manually advancing
        // PPU/APU. This is the target shape of the full block
        // interpreter's fast path for non-MMIO opcodes.
        let mut nes = build_nes();
        nes.reset();
        // Warm up with per-cycle to match the baseline's cold state.
        for _ in 0..2000 {
            nes.step(&mut vsink, &mut asink);
        }
        let t0 = Instant::now();
        for _ in 0..N_ITERS {
            nes.step_one_lda_immediate_bulk(&mut vsink, &mut asink);
        }
        let bulk_ns = t0.elapsed().as_nanos() as f64 / N_ITERS as f64;

        let speedup = per_cycle_ns / bulk_ns;
        // Not an `assert!` — this is a diagnostic bench, not a pass/
        // fail. Print the numbers so `cargo test -- --nocapture`
        // shows them.
        eprintln!();
        eprintln!("=== CPU bulk-step PoC (LDA #imm, N={N_ITERS}) ===");
        eprintln!("  per-cycle:  {per_cycle_ns:6.1} ns/instr");
        eprintln!("  bulk:       {bulk_ns:6.1} ns/instr");
        eprintln!("  speedup:    {speedup:.2}×");
        eprintln!(
            "  gate (1.3×): {}",
            if speedup >= 1.3 {
                "PASS — commit to Phase 2"
            } else if speedup >= 1.0 {
                "marginal — value uncertain"
            } else {
                "FAIL — do not proceed"
            }
        );
        // Sanity: neither path panicked and both left the CPU in
        // non-interrupt state. Register A should be 0x42 (the LDA
        // immediate value).
        assert!(bulk_ns > 0.0);
        assert!(per_cycle_ns > 0.0);
    }
}

pub const OPCODES: [Option<Instruction>; 256] = {
    let mut opcodes = [None; 256];

    opcodes[0xA9] = Some(Instruction {
        name: "LDA",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::lda_immediate],
    });

    opcodes[0xA5] = Some(Instruction {
        name: "LDA",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::lda_addr_abs_finish],
    });

    opcodes[0xB5] = Some(Instruction {
        name: "LDA",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::lda_indexed_x_finish,
        ],
    });

    opcodes[0xAD] = Some(Instruction {
        name: "LDA",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit`.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::abs_late_ppu_read_a],
    });

    opcodes[0xBD] = Some(Instruction {
        name: "LDA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::lda_abs_indexed_x_optimistic,
            Cpu::lda_addr_abs_finish,
        ],
    });

    opcodes[0xB9] = Some(Instruction {
        name: "LDA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::lda_abs_indexed_y_optimistic,
            Cpu::lda_addr_abs_finish,
        ],
    });

    opcodes[0xA1] = Some(Instruction {
        name: "LDA",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::lda_addr_abs_finish,
        ],
    });

    opcodes[0xB1] = Some(Instruction {
        name: "LDA",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::lda_abs_indexed_y_optimistic,
            Cpu::lda_addr_abs_finish,
        ],
    });

    opcodes[0xA2] = Some(Instruction {
        name: "LDX",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::ldx_immediate],
    });

    opcodes[0xA6] = Some(Instruction {
        name: "LDX",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::ldx_addr_abs_finish],
    });

    opcodes[0xB6] = Some(Instruction {
        name: "LDX",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::ldx_indexed_y_finish,
        ],
    });

    opcodes[0xAE] = Some(Instruction {
        name: "LDX",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit`.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::abs_late_ppu_read_x],
    });

    opcodes[0xBE] = Some(Instruction {
        name: "LDX",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ldx_abs_indexed_y_optimistic,
            Cpu::ldx_addr_abs_finish,
        ],
    });

    opcodes[0xA0] = Some(Instruction {
        name: "LDY",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::ldy_immediate],
    });

    opcodes[0xA4] = Some(Instruction {
        name: "LDY",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::ldy_addr_abs_finish],
    });

    opcodes[0xB4] = Some(Instruction {
        name: "LDY",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::ldy_indexed_x_finish,
        ],
    });

    opcodes[0xAC] = Some(Instruction {
        name: "LDY",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit`.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::abs_late_ppu_read_y],
    });

    opcodes[0xBC] = Some(Instruction {
        name: "LDY",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ldy_abs_indexed_x_optimistic,
            Cpu::ldy_addr_abs_finish,
        ],
    });

    opcodes[0x85] = Some(Instruction {
        name: "STA",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::sta_write_byte_abs],
    });

    opcodes[0x95] = Some(Instruction {
        name: "STA",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::sta_write_byte_abs_indexed_x,
        ],
    });

    opcodes[0x8D] = Some(Instruction {
        name: "STA",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit` for all
        // addresses except $4014 (OAM DMA). $4014's write is deferred
        // to the LAST cycle here so DMA-stall alignment matches the
        // asm_cpu path. See `sta_abs_4014_late_write` rationale.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::sta_abs_4014_late_write],
    });

    opcodes[0x9D] = Some(Instruction {
        name: "STA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::st_abs_indexed_x_dummy_read,
            Cpu::sta_write_byte_abs,
        ],
    });

    opcodes[0x99] = Some(Instruction {
        name: "STA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::st_abs_indexed_y_dummy_read,
            Cpu::sta_write_byte_abs,
        ],
    });

    opcodes[0x81] = Some(Instruction {
        name: "STA",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::sta_write_byte_abs,
        ],
    });

    opcodes[0x91] = Some(Instruction {
        name: "STA",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::st_abs_indexed_y_dummy_read,
            Cpu::sta_write_byte_abs,
        ],
    });

    opcodes[0x86] = Some(Instruction {
        name: "STX",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::stx_write_byte_abs],
    });

    opcodes[0x96] = Some(Instruction {
        name: "STX",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::stx_write_byte_abs_indexed_y,
        ],
    });

    opcodes[0x8E] = Some(Instruction {
        name: "STX",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit` for all
        // addresses except $4014 (OAM DMA) — deferred to the LAST cycle
        // so DMA-stall alignment matches the asm_cpu path. See
        // `sta_abs_4014_late_write` rationale.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::stx_abs_4014_late_write],
    });

    opcodes[0x84] = Some(Instruction {
        name: "STY",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::sty_write_byte_abs],
    });

    opcodes[0x94] = Some(Instruction {
        name: "STY",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::sty_write_byte_abs_indexed_x,
        ],
    });

    opcodes[0x8C] = Some(Instruction {
        name: "STY",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit` for all
        // addresses except $4014 (OAM DMA) — deferred to the LAST cycle
        // so DMA-stall alignment matches the asm_cpu path. See
        // `sta_abs_4014_late_write` rationale.
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::sty_abs_4014_late_write],
    });

    opcodes[0x69] = Some(Instruction {
        name: "ADC",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::adc_immediate],
    });

    opcodes[0x65] = Some(Instruction {
        name: "ADC",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::adc_addr_abs_finish],
    });

    opcodes[0x75] = Some(Instruction {
        name: "ADC",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::adc_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0x6D] = Some(Instruction {
        name: "ADC",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::adc_addr_abs_finish,
        ],
    });

    opcodes[0x7D] = Some(Instruction {
        name: "ADC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::adc_addr_abs_indexed_x_optimistic,
            Cpu::adc_addr_abs_finish,
        ],
    });

    opcodes[0x79] = Some(Instruction {
        name: "ADC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::adc_addr_abs_indexed_y_optimistic,
            Cpu::adc_addr_abs_finish,
        ],
    });

    opcodes[0x61] = Some(Instruction {
        name: "ADC",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::adc_addr_abs_finish,
        ],
    });

    opcodes[0x71] = Some(Instruction {
        name: "ADC",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::adc_addr_abs_indexed_y_optimistic,
            Cpu::adc_addr_abs_finish,
        ],
    });

    opcodes[0xE9] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::sbc_immediate],
    });

    opcodes[0xE5] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::sbc_addr_abs_finish],
    });

    opcodes[0xF5] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::sbc_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0xED] = Some(Instruction {
        name: "SBC",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::sbc_addr_abs_finish,
        ],
    });

    opcodes[0xFD] = Some(Instruction {
        name: "SBC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::sbc_addr_abs_indexed_x_optimistic,
            Cpu::sbc_addr_abs_finish,
        ],
    });

    opcodes[0xF9] = Some(Instruction {
        name: "SBC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::sbc_addr_abs_indexed_y_optimistic,
            Cpu::sbc_addr_abs_finish,
        ],
    });

    opcodes[0xE1] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::sbc_addr_abs_finish,
        ],
    });

    opcodes[0xF1] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::sbc_addr_abs_indexed_y_optimistic,
            Cpu::sbc_addr_abs_finish,
        ],
    });

    opcodes[0x29] = Some(Instruction {
        name: "AND",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::and_immediate],
    });

    opcodes[0x25] = Some(Instruction {
        name: "AND",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::and_addr_abs_finish],
    });

    opcodes[0x35] = Some(Instruction {
        name: "AND",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::and_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0x2D] = Some(Instruction {
        name: "AND",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::and_addr_abs_finish,
        ],
    });

    opcodes[0x3D] = Some(Instruction {
        name: "AND",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::and_addr_abs_indexed_x_optimistic,
            Cpu::and_addr_abs_finish,
        ],
    });

    opcodes[0x39] = Some(Instruction {
        name: "AND",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::and_addr_abs_indexed_y_optimistic,
            Cpu::and_addr_abs_finish,
        ],
    });

    opcodes[0x21] = Some(Instruction {
        name: "AND",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::and_addr_abs_finish,
        ],
    });

    opcodes[0x31] = Some(Instruction {
        name: "AND",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::and_addr_abs_indexed_y_optimistic,
            Cpu::and_addr_abs_finish,
        ],
    });

    opcodes[0x09] = Some(Instruction {
        name: "ORA",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::ora_immediate],
    });

    opcodes[0x05] = Some(Instruction {
        name: "ORA",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::ora_addr_abs_finish],
    });

    opcodes[0x15] = Some(Instruction {
        name: "ORA",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::ora_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0x0D] = Some(Instruction {
        name: "ORA",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ora_addr_abs_finish,
        ],
    });

    opcodes[0x1D] = Some(Instruction {
        name: "ORA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ora_addr_abs_indexed_x_optimistic,
            Cpu::ora_addr_abs_finish,
        ],
    });

    opcodes[0x19] = Some(Instruction {
        name: "ORA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ora_addr_abs_indexed_y_optimistic,
            Cpu::ora_addr_abs_finish,
        ],
    });

    opcodes[0x01] = Some(Instruction {
        name: "ORA",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::ora_addr_abs_finish,
        ],
    });

    opcodes[0x11] = Some(Instruction {
        name: "ORA",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::ora_addr_abs_indexed_y_optimistic,
            Cpu::ora_addr_abs_finish,
        ],
    });

    opcodes[0x49] = Some(Instruction {
        name: "EOR",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::eor_immediate],
    });

    opcodes[0x45] = Some(Instruction {
        name: "EOR",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::eor_addr_abs_finish],
    });

    opcodes[0x55] = Some(Instruction {
        name: "EOR",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::eor_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0x4D] = Some(Instruction {
        name: "EOR",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::eor_addr_abs_finish,
        ],
    });

    opcodes[0x5D] = Some(Instruction {
        name: "EOR",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::eor_addr_abs_indexed_x_optimistic,
            Cpu::eor_addr_abs_finish,
        ],
    });

    opcodes[0x59] = Some(Instruction {
        name: "EOR",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::eor_addr_abs_indexed_y_optimistic,
            Cpu::eor_addr_abs_finish,
        ],
    });

    opcodes[0x41] = Some(Instruction {
        name: "EOR",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::eor_addr_abs_finish,
        ],
    });

    opcodes[0x51] = Some(Instruction {
        name: "EOR",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::eor_addr_abs_indexed_y_optimistic,
            Cpu::eor_addr_abs_finish,
        ],
    });

    opcodes[0x38] = Some(Instruction {
        name: "SEC",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::sec],
    });

    opcodes[0x18] = Some(Instruction {
        name: "CLC",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::clc],
    });

    opcodes[0x78] = Some(Instruction {
        name: "SEI",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::sei],
    });

    opcodes[0x58] = Some(Instruction {
        name: "CLI",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::cli],
    });

    opcodes[0xF8] = Some(Instruction {
        name: "SED",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::sed],
    });

    opcodes[0xD8] = Some(Instruction {
        name: "CLD",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::cld],
    });

    opcodes[0xB8] = Some(Instruction {
        name: "CLV",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::clv],
    });

    opcodes[0x4C] = Some(Instruction {
        name: "JMP",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::jmp_abs_finish],
    });

    opcodes[0x6C] = Some(Instruction {
        name: "JMP",
        length: 3,
        mode: AddressMode::Indirect, // Pure Indirect (non-indexed)
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::jmp_indirect_read_pointer_low,
            Cpu::jump_indirect_finish,
        ],
    });

    opcodes[0x30] = Some(Instruction {
        name: "BMI",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_n,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0x10] = Some(Instruction {
        name: "BPL",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_not_n,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0x90] = Some(Instruction {
        name: "BCC",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_not_c,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0xB0] = Some(Instruction {
        name: "BCS",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_c,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0xF0] = Some(Instruction {
        name: "BEQ",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_z,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0xD0] = Some(Instruction {
        name: "BNE",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_not_z,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0x70] = Some(Instruction {
        name: "BVS",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_v,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0x50] = Some(Instruction {
        name: "BVC",
        length: 2,
        mode: AddressMode::Relative,
        official: true,
        cycles: &[
            Cpu::fetch_rel_offset_and_check_branch_not_v,
            Cpu::take_branch,
            Cpu::branch_page_fixup,
        ],
    });

    opcodes[0xC9] = Some(Instruction {
        name: "CMP",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::cmp_immediate],
    });

    opcodes[0xC5] = Some(Instruction {
        name: "CMP",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::cmp_addr_abs_finish],
    });

    opcodes[0xD5] = Some(Instruction {
        name: "CMP",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::cmp_zero_page_indexed_x_finish,
        ],
    });

    opcodes[0xCD] = Some(Instruction {
        name: "CMP",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::cmp_addr_abs_finish,
        ],
    });

    opcodes[0xDD] = Some(Instruction {
        name: "CMP",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::cmp_addr_abs_indexed_x_optimistic,
            Cpu::cmp_addr_abs_finish,
        ],
    });

    opcodes[0xD9] = Some(Instruction {
        name: "CMP",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::cmp_addr_abs_indexed_y_optimistic,
            Cpu::cmp_addr_abs_finish,
        ],
    });

    opcodes[0xC1] = Some(Instruction {
        name: "CMP",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::cmp_addr_abs_finish,
        ],
    });

    opcodes[0xD1] = Some(Instruction {
        name: "CMP",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::cmp_addr_abs_indexed_y_optimistic,
            Cpu::cmp_addr_abs_finish,
        ],
    });

    opcodes[0xE0] = Some(Instruction {
        name: "CPX",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::cpx_immediate],
    });

    opcodes[0xE4] = Some(Instruction {
        name: "CPX",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::cpx_addr_abs_finish],
    });

    opcodes[0xEC] = Some(Instruction {
        name: "CPX",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::cpx_addr_abs_finish,
        ],
    });

    opcodes[0xC0] = Some(Instruction {
        name: "CPY",
        length: 2,
        mode: AddressMode::Immediate,
        official: true,
        cycles: &[Cpu::cpy_immediate],
    });

    opcodes[0xC4] = Some(Instruction {
        name: "CPY",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::cpy_addr_abs_finish],
    });

    opcodes[0xCC] = Some(Instruction {
        name: "CPY",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::cpy_addr_abs_finish,
        ],
    });

    opcodes[0x24] = Some(Instruction {
        name: "BIT",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[Cpu::fetch_abs_low, Cpu::bit_addr_abs_finish],
    });

    opcodes[0x2C] = Some(Instruction {
        name: "BIT",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        // Early-commit at cycle 0 via `cycle_zero_early_commit`.
        // Critical for vblank-polling loops (`BIT $2002 / BPL loop`).
        cycles: &[Cpu::noop_cycle, Cpu::noop_cycle, Cpu::abs_late_ppu_read_bit],
    });

    opcodes[0xE6] = Some(Instruction {
        name: "INC",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::inc_addr_abs_finish,
        ],
    });

    opcodes[0xF6] = Some(Instruction {
        name: "INC",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::inc_addr_abs_finish,
        ],
    });

    opcodes[0xEE] = Some(Instruction {
        name: "INC",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::inc_addr_abs_finish,
        ],
    });

    opcodes[0xFE] = Some(Instruction {
        name: "INC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::inc_addr_abs_finish,
        ],
    });

    opcodes[0xC6] = Some(Instruction {
        name: "DEC",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dec_addr_abs_finish,
        ],
    });

    opcodes[0xD6] = Some(Instruction {
        name: "DEC",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dec_addr_abs_finish,
        ],
    });

    opcodes[0xCE] = Some(Instruction {
        name: "DEC",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dec_addr_abs_finish,
        ],
    });

    opcodes[0xDE] = Some(Instruction {
        name: "DEC",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dec_addr_abs_finish,
        ],
    });

    opcodes[0xE8] = Some(Instruction {
        name: "INX",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::inx],
    });

    opcodes[0xC8] = Some(Instruction {
        name: "INY",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::iny],
    });

    opcodes[0xCA] = Some(Instruction {
        name: "DEX",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dex],
    });

    opcodes[0x88] = Some(Instruction {
        name: "DEY",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dey],
    });

    opcodes[0xAA] = Some(Instruction {
        name: "TAX",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::tax],
    });

    opcodes[0x8A] = Some(Instruction {
        name: "TXA",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::txa],
    });

    opcodes[0xA8] = Some(Instruction {
        name: "TAY",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::tay],
    });

    opcodes[0x98] = Some(Instruction {
        name: "TYA",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::tya],
    });

    opcodes[0xBA] = Some(Instruction {
        name: "TSX",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::tsx],
    });

    opcodes[0x9A] = Some(Instruction {
        name: "TXS",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::txs],
    });

    opcodes[0x20] = Some(Instruction {
        name: "JSR",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::dummy_stack_read,
            Cpu::push_pc_high,
            Cpu::push_pc_low,
            Cpu::jsr_finish,
        ],
    });

    opcodes[0x60] = Some(Instruction {
        name: "RTS",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[
            Cpu::dummy_read,
            Cpu::pull_low,
            Cpu::pull_high,
            Cpu::dummy_read,
            Cpu::rts_finish,
        ],
    });

    opcodes[0x48] = Some(Instruction {
        name: "PHA",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dummy_read, Cpu::pha_finish],
    });

    opcodes[0x68] = Some(Instruction {
        name: "PLA",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dummy_read, Cpu::inc_sp, Cpu::pla_finish],
    });

    opcodes[0x08] = Some(Instruction {
        name: "PHP",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dummy_read, Cpu::php_finish],
    });

    opcodes[0x28] = Some(Instruction {
        name: "PLP",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::dummy_read, Cpu::inc_sp, Cpu::plp_finish],
    });

    opcodes[0x4A] = Some(Instruction {
        name: "LSR",
        length: 1,
        mode: AddressMode::Accumulator,
        official: true,
        cycles: &[Cpu::lsr_accumulator_finish],
    });

    opcodes[0x46] = Some(Instruction {
        name: "LSR",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::lsr_addr_abs_finish,
        ],
    });

    opcodes[0x56] = Some(Instruction {
        name: "LSR",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::lsr_addr_abs_finish,
        ],
    });

    opcodes[0x4E] = Some(Instruction {
        name: "LSR",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::lsr_addr_abs_finish,
        ],
    });

    opcodes[0x5E] = Some(Instruction {
        name: "LSR",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::lsr_addr_abs_finish,
        ],
    });

    opcodes[0x0A] = Some(Instruction {
        name: "ASL",
        length: 1,
        mode: AddressMode::Accumulator,
        official: true,
        cycles: &[Cpu::asl_accumulator_finish],
    });

    opcodes[0x06] = Some(Instruction {
        name: "ASL",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::asl_addr_abs_finish,
        ],
    });

    opcodes[0x16] = Some(Instruction {
        name: "ASL",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::asl_addr_abs_finish,
        ],
    });

    opcodes[0x0E] = Some(Instruction {
        name: "ASL",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::asl_addr_abs_finish,
        ],
    });

    opcodes[0x1E] = Some(Instruction {
        name: "ASL",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::asl_addr_abs_finish,
        ],
    });

    opcodes[0x6A] = Some(Instruction {
        name: "ROR",
        length: 1,
        mode: AddressMode::Accumulator,
        official: true,
        cycles: &[Cpu::ror_accumulator_finish],
    });

    opcodes[0x66] = Some(Instruction {
        name: "ROR",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::ror_addr_abs_finish,
        ],
    });

    opcodes[0x76] = Some(Instruction {
        name: "ROR",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::ror_addr_abs_finish,
        ],
    });

    opcodes[0x6E] = Some(Instruction {
        name: "ROR",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::ror_addr_abs_finish,
        ],
    });

    opcodes[0x7E] = Some(Instruction {
        name: "ROR",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::ror_addr_abs_finish,
        ],
    });

    opcodes[0x2A] = Some(Instruction {
        name: "ROL",
        length: 1,
        mode: AddressMode::Accumulator,
        official: true,
        cycles: &[Cpu::rol_accumulator_finish],
    });

    opcodes[0x26] = Some(Instruction {
        name: "ROL",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rol_addr_abs_finish,
        ],
    });

    opcodes[0x36] = Some(Instruction {
        name: "ROL",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rol_addr_abs_finish,
        ],
    });

    opcodes[0x2E] = Some(Instruction {
        name: "ROL",
        length: 3,
        mode: AddressMode::Absolute,
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rol_addr_abs_finish,
        ],
    });

    opcodes[0x3E] = Some(Instruction {
        name: "ROL",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: true,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rol_addr_abs_finish,
        ],
    });

    opcodes[0x00] = Some(Instruction {
        name: "BRK",
        length: 2,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[
            Cpu::dummy_fetch_and_inc_pc,
            Cpu::push_pc_high,
            Cpu::push_pc_low,
            Cpu::brk_push_status,
            Cpu::read_brk_vector_low,
            Cpu::brk_finish,
        ],
    });

    opcodes[0x40] = Some(Instruction {
        name: "RTI",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[
            Cpu::dummy_read,
            Cpu::pull_status,
            Cpu::pull_low,
            Cpu::pull_high,
            Cpu::rti_finish,
        ],
    });

    opcodes[0xEA] = Some(Instruction {
        name: "NOP",
        length: 1,
        mode: AddressMode::Implied,
        official: true,
        cycles: &[Cpu::nop_implied],
    });

    // Unofficial opcodes.

    opcodes[0x4B] = Some(Instruction {
        name: "ALR",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::alr],
    });

    opcodes[0x0B] = Some(Instruction {
        name: "ANC",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::anc],
    });

    opcodes[0x2B] = Some(Instruction {
        name: "ANC",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::anc],
    });

    opcodes[0x6B] = Some(Instruction {
        name: "ARR",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::arr],
    });

    opcodes[0xCB] = Some(Instruction {
        name: "AXS",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::axs],
    });

    opcodes[0xA3] = Some(Instruction {
        name: "LAX",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::lax_addr_abs_finish,
        ],
    });

    opcodes[0xA7] = Some(Instruction {
        name: "LAX",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[Cpu::fetch_abs_low, Cpu::lax_addr_abs_finish],
    });

    opcodes[0xAB] = Some(Instruction {
        name: "LAX",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::lax_immediate],
    });

    opcodes[0xAF] = Some(Instruction {
        name: "LAX",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::lax_addr_abs_finish,
        ],
    });

    opcodes[0xB3] = Some(Instruction {
        name: "LAX",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::lax_abs_indexed_y_optimistic,
            Cpu::lax_addr_abs_finish,
        ],
    });

    opcodes[0xB7] = Some(Instruction {
        name: "LAX",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::lax_indexed_y_finish,
        ],
    });

    opcodes[0xBF] = Some(Instruction {
        name: "LAX",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::lax_abs_indexed_y_optimistic,
            Cpu::lax_addr_abs_finish,
        ],
    });

    opcodes[0x83] = Some(Instruction {
        name: "SAX",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::sax_write_byte_abs,
        ],
    });

    opcodes[0x87] = Some(Instruction {
        name: "SAX",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[Cpu::fetch_abs_low, Cpu::sax_write_byte_abs],
    });

    opcodes[0x8F] = Some(Instruction {
        name: "SAX",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::sax_write_byte_abs,
        ],
    });

    opcodes[0x97] = Some(Instruction {
        name: "SAX",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::sax_write_byte_abs_indexed_y,
        ],
    });

    opcodes[0xC7] = Some(Instruction {
        name: "DCP",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xCF] = Some(Instruction {
        name: "DCP",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xD7] = Some(Instruction {
        name: "DCP",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xDF] = Some(Instruction {
        name: "DCP",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xDB] = Some(Instruction {
        name: "DCP",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xC3] = Some(Instruction {
        name: "DCP",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xD3] = Some(Instruction {
        name: "DCP",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::dcp_fetched_data_finish,
        ],
    });

    opcodes[0xE3] = Some(Instruction {
        name: "ISB",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xE7] = Some(Instruction {
        name: "ISB",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xF3] = Some(Instruction {
        name: "ISB",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xF7] = Some(Instruction {
        name: "ISB",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xFB] = Some(Instruction {
        name: "ISB",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xEF] = Some(Instruction {
        name: "ISB",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xFF] = Some(Instruction {
        name: "ISB",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::isb_addr_abs_finish,
        ],
    });

    opcodes[0xEB] = Some(Instruction {
        name: "SBC",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::sbc_immediate],
    });

    opcodes[0x23] = Some(Instruction {
        name: "RLA",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x27] = Some(Instruction {
        name: "RLA",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x2F] = Some(Instruction {
        name: "RLA",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x33] = Some(Instruction {
        name: "RLA",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x37] = Some(Instruction {
        name: "RLA",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x3B] = Some(Instruction {
        name: "RLA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x3F] = Some(Instruction {
        name: "RLA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rla_addr_abs_finish,
        ],
    });

    opcodes[0x63] = Some(Instruction {
        name: "RRA",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x67] = Some(Instruction {
        name: "RRA",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x6F] = Some(Instruction {
        name: "RRA",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x73] = Some(Instruction {
        name: "RRA",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x77] = Some(Instruction {
        name: "RRA",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x7B] = Some(Instruction {
        name: "RRA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x7F] = Some(Instruction {
        name: "RRA",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::rra_addr_abs_finish,
        ],
    });

    opcodes[0x03] = Some(Instruction {
        name: "SLO",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x07] = Some(Instruction {
        name: "SLO",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x0F] = Some(Instruction {
        name: "SLO",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x13] = Some(Instruction {
        name: "SLO",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x17] = Some(Instruction {
        name: "SLO",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x1B] = Some(Instruction {
        name: "SLO",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x1F] = Some(Instruction {
        name: "SLO",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::slo_addr_abs_finish,
        ],
    });

    opcodes[0x43] = Some(Instruction {
        name: "SRE",
        length: 2,
        mode: AddressMode::IndexedIndirect(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_x_dummy_read_and_add,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x47] = Some(Instruction {
        name: "SRE",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x4F] = Some(Instruction {
        name: "SRE",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::read_abs_addr_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x53] = Some(Instruction {
        name: "SRE",
        length: 2,
        mode: AddressMode::IndirectIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::indexed_fetch_ptr_low,
            Cpu::indexed_fetch_ptr_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x5B] = Some(Instruction {
        name: "SRE",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_y_dummy_read,
            Cpu::read_abs_indexed_y_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x57] = Some(Instruction {
        name: "SRE",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::dummy_read_base,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    opcodes[0x5F] = Some(Instruction {
        name: "SRE",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::rmw_abs_indexed_x_dummy_read,
            Cpu::read_abs_indexed_x_data,
            Cpu::addr_abs_fetched_data_dummy_write,
            Cpu::sre_addr_abs_finish,
        ],
    });

    let nop_implied = Instruction {
        name: "NOP",
        length: 1,
        mode: AddressMode::Implied,
        official: false,
        cycles: &[Cpu::nop_implied],
    };
    opcodes[0x1A] = Some(nop_implied);
    opcodes[0x3A] = Some(nop_implied);
    opcodes[0x5A] = Some(nop_implied);
    opcodes[0x7A] = Some(nop_implied);
    opcodes[0xDA] = Some(nop_implied);
    opcodes[0xFA] = Some(nop_implied);

    let skb = Instruction {
        name: "NOP",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::skb],
    };
    opcodes[0x80] = Some(skb);
    opcodes[0x82] = Some(skb);
    opcodes[0x89] = Some(skb);
    opcodes[0xC2] = Some(skb);
    opcodes[0xE2] = Some(skb);

    opcodes[0x0C] = Some(Instruction {
        name: "NOP",
        length: 3,
        mode: AddressMode::Absolute,
        official: false,
        cycles: &[Cpu::fetch_abs_low, Cpu::fetch_abs_high, Cpu::ign_finish],
    });

    let ign_abs_x = Instruction {
        name: "NOP",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::ign_abs_indexed_x_optimistic,
            Cpu::ign_finish,
        ],
    };
    opcodes[0x1C] = Some(ign_abs_x);
    opcodes[0x3C] = Some(ign_abs_x);
    opcodes[0x5C] = Some(ign_abs_x);
    opcodes[0x7C] = Some(ign_abs_x);
    opcodes[0xDC] = Some(ign_abs_x);
    opcodes[0xFC] = Some(ign_abs_x);

    let ign_zp = Instruction {
        name: "NOP",
        length: 2,
        mode: AddressMode::ZeroPage,
        official: false,
        cycles: &[Cpu::fetch_abs_low, Cpu::ign_finish],
    };
    opcodes[0x04] = Some(ign_zp);
    opcodes[0x44] = Some(ign_zp);
    opcodes[0x64] = Some(ign_zp);

    let ign_zp_x = Instruction {
        name: "NOP",
        length: 2,
        mode: AddressMode::ZeroPageIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_base_addr,
            Cpu::read_zero_page_indexed_x_data,
            Cpu::ign_finish,
        ],
    };
    opcodes[0x14] = Some(ign_zp_x);
    opcodes[0x34] = Some(ign_zp_x);
    opcodes[0x54] = Some(ign_zp_x);
    opcodes[0x74] = Some(ign_zp_x);
    opcodes[0xD4] = Some(ign_zp_x);
    opcodes[0xF4] = Some(ign_zp_x);

    opcodes[0x8B] = Some(Instruction {
        name: "XAA",
        length: 2,
        mode: AddressMode::Immediate,
        official: false,
        cycles: &[Cpu::xaa],
    });

    opcodes[0x9C] = Some(Instruction {
        name: "SHY",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::X),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::shy_logic,
            Cpu::shy_finish,
        ],
    });

    opcodes[0x9E] = Some(Instruction {
        name: "SHX",
        length: 3,
        mode: AddressMode::AbsoluteIndexed(Register8::Y),
        official: false,
        cycles: &[
            Cpu::fetch_abs_low,
            Cpu::fetch_abs_high,
            Cpu::shx_logic,
            Cpu::shx_finish,
        ],
    });

    opcodes
};

pub const NMI_INTERRUPT: Instruction = Instruction {
    name: "NMI",
    length: 0,
    mode: AddressMode::Implied,
    official: true,
    cycles: &[
        Cpu::dummy_read,
        Cpu::push_pc_high,
        Cpu::push_pc_low,
        Cpu::interrupt_push_status,
        Cpu::read_nmi_vector_low,
        Cpu::nmi_finish,
    ],
};

pub const IRQ_INTERRUPT: Instruction = Instruction {
    name: "IRQ",
    length: 0,
    mode: AddressMode::Implied,
    official: true,
    cycles: &[
        Cpu::dummy_read,
        Cpu::push_pc_high,
        Cpu::push_pc_low,
        Cpu::interrupt_push_status,
        Cpu::read_irq_vector_low,
        Cpu::irq_finish,
    ],
};

#[cfg(test)]
mod cpu_coverage_tests {
    use super::*;
    use crate::apu::Apu;
    use crate::cartridge::Cartridge;
    use crate::game_genie::Cheat;
    use crate::input::Input;
    use crate::mapper::MapperEnum;
    use crate::memory::Ram;
    use crate::oam_dma::OamDma;
    use crate::ppu::Ppu;
    use std::collections::HashMap;
    use std::io::Cursor;

    /// Owns every peripheral so a `SystemBus` can borrow them for the
    /// lifetime of a test. Programs run out of RAM ($0000-$07FF), so
    /// the mapper/PRG contents never matter — a minimal NROM image is
    /// only needed because `MapperEnum` demands a `Cartridge`.
    struct Parts {
        ram: Ram,
        mapper: MapperEnum,
        ppu: Ppu,
        apu: Apu,
        oam_dma: OamDma,
        input: Input,
        cheats: HashMap<u16, Cheat>,
    }

    fn min_nrom_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // 2 x 16KB PRG -> mapper 0 (NROM) 32KB
        rom.push(0); // CHR-RAM
        rom.push(0); // flags6: mapper low nibble 0
        rom.push(0); // flags7: mapper high nibble 0
        rom.extend_from_slice(&[0u8; 8]);
        rom.extend(std::iter::repeat(0u8).take(32 * 1024));
        rom
    }

    fn parts() -> Parts {
        let cart = Cartridge::load(&mut Cursor::new(min_nrom_rom()))
            .expect("synthetic NROM image parses");
        Parts {
            ram: Ram::new(),
            mapper: MapperEnum::from_cartridge(cart).unwrap(),
            ppu: Ppu::new(),
            apu: Apu::new(),
            oam_dma: OamDma::new(),
            input: Input::new(),
            cheats: HashMap::new(),
        }
    }

    impl Parts {
        fn bus(&mut self) -> SystemBus<'_> {
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
    }

    /// Load consecutive bytes into RAM starting at `addr`.
    fn load_prog(bus: &mut SystemBus, addr: u16, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            bus.write_byte(addr + i as u16, *b);
        }
    }

    /// Tick a CPU sitting at an instruction boundary until it reports
    /// the instruction completed; return the number of CPU cycles
    /// consumed (the opcode-fetch cycle included, matching hardware
    /// cycle counts). Panics if the instruction never finishes so a
    /// hung dispatch can't silently pass.
    fn run_one(cpu: &mut Cpu, bus: &mut SystemBus) -> u32 {
        let mut cycles = 0u32;
        loop {
            let done = cpu.tick(bus);
            cycles += 1;
            if done {
                return cycles;
            }
            assert!(cycles <= 24, "instruction failed to complete");
        }
    }

    // ---- Gap 1: ADC overflow (V) flag, all four sign combinations ----
    // add_value is the shared ADC core. These bite any mangling of the
    // "operands share a sign AND the result flips it" overflow rule or
    // the carry-out detection.

    #[test]
    fn adc_overflow_pos_plus_pos_no_carry_no_overflow() {
        // 0x50 + 0x10 = 0x60: two positives, positive result -> no V.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = false;
        cpu.add_value(0x10);
        assert_eq!(cpu.regs.a, 0x60);
        assert!(!cpu.flags.v, "same-sign result kept sign -> no overflow");
        assert!(!cpu.flags.c, "no carry out of bit 7");
        assert!(!cpu.flags.z);
        assert!(!cpu.flags.n);
    }

    #[test]
    fn adc_overflow_pos_plus_pos_sets_v() {
        // 0x50 + 0x50 = 0xA0: positive + positive -> negative -> V set.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = false;
        cpu.add_value(0x50);
        assert_eq!(cpu.regs.a, 0xA0);
        assert!(cpu.flags.v, "two positives crossing into negative -> V");
        assert!(!cpu.flags.c);
        assert!(cpu.flags.n, "bit 7 set");
    }

    #[test]
    fn adc_overflow_neg_plus_neg_sets_v_and_carry() {
        // 0xD0 + 0x90 = 0x160 -> A=0x60, carry out, and two negatives
        // producing a positive -> V. Carry-out and V both fire here.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0xD0;
        cpu.flags.c = false;
        cpu.add_value(0x90);
        assert_eq!(cpu.regs.a, 0x60);
        assert!(cpu.flags.c, "sum exceeded 0xFF -> carry set");
        assert!(cpu.flags.v, "two negatives crossing into positive -> V");
        assert!(!cpu.flags.n);
    }

    #[test]
    fn adc_overflow_mixed_signs_never_overflow() {
        // 0x50 + 0x90 = 0xE0: opposite signs can never overflow -> V clear.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = false;
        cpu.add_value(0x90);
        assert_eq!(cpu.regs.a, 0xE0);
        assert!(!cpu.flags.v, "opposite-sign operands never overflow");
        assert!(!cpu.flags.c);
        assert!(cpu.flags.n);
    }

    #[test]
    fn adc_carry_in_is_added_and_binary_only() {
        // Carry-in participates: 0x00 + 0xFF + C=1 wraps to 0x00 with
        // carry out and Z set. The D flag must be ignored on the NES
        // (binary add), so 0x50 + 0x50 with D=1 still yields 0xA0, not
        // a BCD result.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x00;
        cpu.flags.c = true;
        cpu.add_value(0xFF);
        assert_eq!(cpu.regs.a, 0x00, "0x00+0xFF+1 wraps to 0x00");
        assert!(cpu.flags.c, "carry propagated out");
        assert!(cpu.flags.z, "result is zero");

        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = false;
        cpu.flags.d = true; // must be a no-op on the NES 6502
        cpu.add_value(0x50);
        assert_eq!(cpu.regs.a, 0xA0, "decimal mode ignored -> binary result");
    }

    // ---- Gap 2: SBC borrow/carry and V flag ----
    // sub_value is the SBC core. C=1 means "no borrow"; C acts as an
    // inverted borrow so C=0 subtracts an extra 1.

    #[test]
    fn sbc_overflow_case_sets_v() {
        // 0x50 - 0xB0 (no borrow): +80 - (-80) = +160 -> signed overflow.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = true; // no incoming borrow
        cpu.sub_value(0xB0);
        assert_eq!(cpu.regs.a, 0xA0);
        assert!(cpu.flags.v, "result sign wrong for the true difference -> V");
        assert!(!cpu.flags.c, "0x50 < 0xB0 -> borrow -> carry cleared");
        assert!(cpu.flags.n);
        assert!(!cpu.flags.z);
    }

    #[test]
    fn sbc_no_overflow_mixed_result() {
        // 0x50 - 0xF0 (no borrow): +80 - (-16) = +96, fits -> no V,
        // but a borrow occurred so carry clears.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x50;
        cpu.flags.c = true;
        cpu.sub_value(0xF0);
        assert_eq!(cpu.regs.a, 0x60);
        assert!(!cpu.flags.v, "difference in range -> no overflow");
        assert!(!cpu.flags.c, "borrow occurred");
        assert!(!cpu.flags.n);
    }

    #[test]
    fn sbc_carry_clear_is_extra_borrow() {
        // Same operands, only the incoming carry differs: C=0 subtracts
        // one more than C=1. Bites any dropped `!carry` borrow term.
        let mut with_borrow = Cpu::new();
        with_borrow.regs.a = 0x50;
        with_borrow.flags.c = false; // incoming borrow
        with_borrow.sub_value(0x10);
        assert_eq!(with_borrow.regs.a, 0x3F, "0x50 - 0x10 - 1 = 0x3F");
        assert!(with_borrow.flags.c, "0x50 >= 0x11 -> no resulting borrow");

        let mut no_borrow = Cpu::new();
        no_borrow.regs.a = 0x50;
        no_borrow.flags.c = true;
        no_borrow.sub_value(0x10);
        assert_eq!(no_borrow.regs.a, 0x40, "0x50 - 0x10 - 0 = 0x40");
    }

    #[test]
    fn sbc_equal_operands_sets_zero_and_carry() {
        // A - A with no borrow = 0, carry set (no borrow out), Z set.
        let mut cpu = Cpu::new();
        cpu.regs.a = 0x05;
        cpu.flags.c = true;
        cpu.sub_value(0x05);
        assert_eq!(cpu.regs.a, 0x00);
        assert!(cpu.flags.z);
        assert!(cpu.flags.c, "no borrow -> carry stays set");
        assert!(!cpu.flags.v);
        assert!(!cpu.flags.n);
    }

    // ---- Gap 3: page-cross extra cycle on indexed reads ----
    // The "optimistic" indexed read finishes early (no extra cycle)
    // only when the speculative address stays on the base page.

    #[test]
    fn abs_x_read_no_page_cross_is_four_cycles() {
        let mut p = parts();
        let mut bus = p.bus();
        // LDA $0010,X ; X=0x05 -> $0015, same page.
        load_prog(&mut bus, 0x0200, &[0xBD, 0x10, 0x00]);
        bus.write_byte(0x0015, 0x37);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.x = 0x05;
        let cycles = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x37);
        assert_eq!(cycles, 4, "no page cross -> base 4 cycles");
    }

    #[test]
    fn abs_x_read_page_cross_adds_a_cycle() {
        let mut p = parts();
        let mut bus = p.bus();
        // LDA $00F0,X ; X=0x20 -> $0110, crosses $00 -> $01.
        load_prog(&mut bus, 0x0200, &[0xBD, 0xF0, 0x00]);
        bus.write_byte(0x0110, 0x42);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.x = 0x20;
        let cycles = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x42);
        assert_eq!(cycles, 5, "page cross -> one extra read cycle");
    }

    #[test]
    fn abs_y_read_page_cross_delta_is_exactly_one() {
        // Both LDA $abs,Y forms, differing only in whether Y pushes the
        // effective address onto the next page. The delta must be 1.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xB9, 0x10, 0x00]); // $0010,Y
        bus.write_byte(0x0018, 0x11);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.y = 0x08; // -> $0018, same page
        let same = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x11);

        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xB9, 0xF0, 0x00]); // $00F0,Y
        bus.write_byte(0x0108, 0x22);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.y = 0x18; // -> $0108, crosses page
        let crossed = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x22);
        assert_eq!(crossed - same, 1, "exactly one extra cycle on cross");
    }

    #[test]
    fn indirect_y_read_page_cross_cycle_counts() {
        // LDA ($20),Y with the pointer's base low byte + Y deciding the
        // cross. No cross = 5 cycles, cross = 6.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xB1, 0x20]);
        bus.write_byte(0x0020, 0x00); // ptr low
        bus.write_byte(0x0021, 0x04); // ptr high -> base $0400
        bus.write_byte(0x0405, 0x5A); // $0400 + Y(5)
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.y = 0x05;
        let no_cross = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x5A);
        assert_eq!(no_cross, 5, "(ind),Y no cross -> 5 cycles");

        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xB1, 0x20]);
        bus.write_byte(0x0020, 0xF0); // ptr low
        bus.write_byte(0x0021, 0x04); // base $04F0
        bus.write_byte(0x0510, 0x6B); // $04F0 + Y(0x20) crosses to $05
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.y = 0x20;
        let cross = run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.a, 0x6B);
        assert_eq!(cross, 6, "(ind),Y cross -> 6 cycles");
    }

    // ---- Gap 4: poll_interrupts flag combinations ----

    fn pending_name(cpu: &Cpu) -> Option<&'static str> {
        cpu.instruction.map(|i| i.name)
    }

    #[test]
    fn nmi_edge_is_latched_and_serviced() {
        // A high->low transition latches nmi_pended; the default poll
        // path services it at the next boundary.
        let mut cpu = Cpu::new();
        cpu.set_nmi_line(true);
        assert!(cpu.nmi_pended, "falling edge latches the NMI");
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), Some("NMI"));
    }

    #[test]
    fn irq_is_level_gated_by_i_flag() {
        // IRQ services only while the line is low AND I is clear. The
        // gate reads `i_poll_latch` (flags.i as of the end of the
        // previous cycle), so a direct unit test of `poll_interrupts`
        // must set the latch alongside the live flag, same as the
        // `nmi_poll_latch` tests below do for `nmi_pended`.
        let mut cpu = Cpu::new();
        cpu.irq_line_low = true;
        cpu.flags.i = false;
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.i_poll_latch = false;
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), Some("IRQ"), "unmasked IRQ serviced");

        let mut cpu = Cpu::new();
        cpu.irq_line_low = true;
        cpu.flags.i = true; // masked
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.i_poll_latch = true;
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), None, "I flag masks the IRQ");
    }

    #[test]
    fn nmi_has_priority_over_irq() {
        let mut cpu = Cpu::new();
        cpu.set_nmi_line(true);
        cpu.irq_line_low = true;
        cpu.flags.i = false;
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), Some("NMI"), "NMI outranks IRQ");
    }

    #[test]
    fn nmi_poll_timing_defers_final_cycle_edge() {
        // Under hw_nmi_poll_timing (and NOT subcycle phase) the service
        // decision uses nmi_poll_latch, so an edge that only reached
        // nmi_pended (latch still false) is deferred one boundary.
        let mut cpu = Cpu::new();
        cpu.hw_nmi_poll_timing = true;
        cpu.hw_nmi_subcycle_phase = false;
        cpu.nmi_pended = true;
        cpu.nmi_poll_latch = false;
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), None, "final-cycle edge deferred");

        // Once latched, it services.
        let mut cpu = Cpu::new();
        cpu.hw_nmi_poll_timing = true;
        cpu.hw_nmi_subcycle_phase = false;
        cpu.nmi_pended = true;
        cpu.nmi_poll_latch = true;
        cpu.poll_interrupts();
        assert_eq!(pending_name(&cpu), Some("NMI"), "latched edge serviced");
    }

    #[test]
    fn subcycle_phase_uses_live_pended_over_poll_latch() {
        // hw_nmi_subcycle_phase forces the live nmi_pended path even
        // when hw_nmi_poll_timing is set — the two must not stack
        // (would double-quantize NMI entry).
        let mut cpu = Cpu::new();
        cpu.hw_nmi_poll_timing = true;
        cpu.hw_nmi_subcycle_phase = true;
        cpu.nmi_pended = true;
        cpu.nmi_poll_latch = false; // ignored under subcycle phase
        cpu.poll_interrupts();
        assert_eq!(
            pending_name(&cpu),
            Some("NMI"),
            "subcycle phase reads live pended, not the second-to-last latch"
        );
    }

    #[test]
    fn irq_poll_latency_is_off_by_default_and_preserves_legacy_service() {
        // NEGATIVE CONTROL for the gate, and the reason the gate exists.
        //
        // Same program and setup as irq_poll_latency_cli_defers_service_by_
        // one_instruction, with hw_irq_poll_timing left at its default. The
        // IRQ must be serviced AT CLI's own boundary -- one instruction
        // earlier than hardware, i.e. the pre-2026-08-26 behaviour.
        //
        // That inaccuracy is kept as the default deliberately: this project
        // replays banked solver tapes (scripts/replay_sweep.py) and re-scores
        // banked training runs against them, and an IRQ-timing change is
        // exactly the class that silently invalidates every one of those. The
        // accuracy fix is opt-in via the same --hw-all path that arms
        // hw_nmi_poll_timing.
        //
        // If this test starts failing, the gate was removed or inverted and
        // every banked artifact's replay semantics just changed underneath it.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xEA, 0x58, 0xEA, 0xEA]); // NOP ; CLI ; NOP ; NOP
        let mut cpu = Cpu::new();
        assert!(
            !cpu.hw_irq_poll_timing,
            "the IRQ-poll accuracy gate must default OFF"
        );
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD;
        cpu.flags.i = true;
        cpu.set_irq_line_low(true);

        assert_eq!(run_one(&mut cpu, &mut bus), 2, "NOP is 2 cycles");
        assert_eq!(pending_name(&cpu), None, "a masked IRQ stays masked");

        assert_eq!(run_one(&mut cpu, &mut bus), 2, "CLI is 2 cycles");
        assert!(!cpu.flags.i, "CLI cleared I on its own final cycle");
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "with the gate OFF the IRQ is serviced at CLI's own boundary \
             (legacy, knowingly one instruction early); a change here means \
             banked tape replays are no longer comparable"
        );
    }

    #[test]
    fn cli_defers_pending_irq_by_one_instruction() {
        // The classic 6502 "interrupt hijacking" / CLI-latency quirk
        // (blargg's `cpu_interrupts_v2/1-cli_latency`): CLI writes
        // flags.i on its own final cycle, but hardware polls
        // interrupts using the I flag as of the END of the PREVIOUS
        // cycle. A pending IRQ must wait until the instruction AFTER
        // CLI completes before it is serviced -- not immediately
        // after CLI itself. Without `i_poll_latch`, `poll_interrupts`
        // reads the live post-write `flags.i` and services the IRQ a
        // full instruction early.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0x58, 0xEA, 0xEA]); // CLI ; NOP ; NOP
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.flags.i = true;
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.i_poll_latch = true;
        cpu.irq_line_low = true;

        let cli_cycles = run_one(&mut cpu, &mut bus);
        assert_eq!(cli_cycles, 2, "CLI is a 2-cycle instruction");
        assert!(!cpu.flags.i, "CLI clears I on its own final cycle");
        assert_eq!(
            pending_name(&cpu),
            None,
            "the pending IRQ must NOT be serviced immediately after CLI"
        );
        assert_eq!(cpu.regs.pc, 0x0201, "next fetch must be the NOP, not the IRQ vector");

        // The NOP right after CLI must execute BEFORE the IRQ is taken.
        run_one(&mut cpu, &mut bus);
        assert_eq!(cpu.regs.pc, 0x0202, "the NOP executed first");
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "the deferred IRQ is serviced at the boundary after the NOP"
        );
    }

    // ---- Gap 5: get_state / apply_state round-trip ----

    #[test]
    fn state_round_trip_preserves_full_register_file_and_latches() {
        let mut cpu = Cpu::new();
        cpu.regs = Regs { pc: 0x1234, a: 0x11, x: 0x22, y: 0x33, sp: 0x44 };
        cpu.flags = Flags::from(0xC1); // N,V,C set
        cpu.stall_cycles = 3;
        cpu.nmi_line_low = true;
        cpu.nmi_pended = true;
        cpu.irq_line_low = true;
        cpu.cycles_total = 987_654;
        cpu.opcode = 0xBD; // a real multi-cycle opcode
        cpu.cycle = 2; // mid-instruction
        cpu.addr_abs = 0xBEEF;
        cpu.temp_addr_low = 0x77;
        cpu.base_addr = 0x88;
        cpu.rel_offset = -5;
        cpu.fetched_data = 0x99;

        let state = cpu.get_state();
        let mut restored = Cpu::new();
        restored.apply_state(&state);

        assert_eq!(restored.regs.pc, 0x1234);
        assert_eq!(restored.regs.a, 0x11);
        assert_eq!(restored.regs.x, 0x22);
        assert_eq!(restored.regs.y, 0x33);
        assert_eq!(restored.regs.sp, 0x44);
        assert_eq!(u8::from(restored.flags), 0xC1);
        assert_eq!(restored.stall_cycles, 3);
        assert!(restored.nmi_line_low);
        assert!(restored.nmi_pended);
        assert!(restored.irq_line_low);
        assert_eq!(restored.cycles_total, 987_654);
        assert_eq!(restored.opcode, 0xBD);
        assert_eq!(restored.cycle, 2);
        assert_eq!(restored.addr_abs, 0xBEEF);
        assert_eq!(restored.temp_addr_low, 0x77);
        assert_eq!(restored.base_addr, 0x88);
        assert_eq!(restored.rel_offset, -5);
        assert_eq!(restored.fetched_data, 0x99);
        // Runtime latch is re-synced to the restored pending flag.
        assert!(restored.nmi_poll_latch);
        // Mid-instruction: dispatch re-hydrated from the opcode.
        assert_eq!(pending_name(&restored), Some("LDA"));
    }

    #[test]
    fn state_round_trip_boundary_has_no_instruction() {
        // cycle==0 with no active interrupt means "between instructions":
        // apply_state must leave instruction None so the next tick
        // fetches fresh (the documented save/load fix).
        let mut cpu = Cpu::new();
        cpu.cycle = 0;
        cpu.opcode = 0xA9; // stale opcode must NOT be re-armed
        let state = cpu.get_state();
        let mut restored = Cpu::new();
        restored.apply_state(&state);
        assert!(restored.at_instruction_boundary());
        assert_eq!(pending_name(&restored), None);
    }

    #[test]
    fn state_round_trip_preserves_active_interrupt() {
        // An in-flight NMI (mid-service) must resume as an NMI, not the
        // opcode whose byte happens to sit in `opcode`.
        let mut cpu = Cpu::new();
        cpu.cycle = 3;
        cpu.opcode = 0xA9;
        cpu.instruction = Some(NMI_INTERRUPT);
        let state = cpu.get_state();
        let mut restored = Cpu::new();
        restored.apply_state(&state);
        assert_eq!(pending_name(&restored), Some("NMI"));
    }

    #[test]
    fn apply_state_resumes_execution_identically() {
        // Full reference run vs. a run interrupted mid-instruction,
        // snapshotted, and resumed on a fresh CPU. Same end state and
        // total cycle count -> apply_state is a faithful resume point.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xBD, 0xF0, 0x00]); // LDA $00F0,X (cross)
        bus.write_byte(0x0110, 0x7E);

        let mut reference = Cpu::new();
        reference.regs.pc = 0x0200;
        reference.regs.x = 0x20;
        let full = run_one(&mut reference, &mut bus);

        let mut partial = Cpu::new();
        partial.regs.pc = 0x0200;
        partial.regs.x = 0x20;
        for _ in 0..3 {
            partial.tick(&mut bus); // three cycles in, still mid-instruction
        }
        assert_ne!(partial.cycle, 0, "snapshot must be mid-instruction");
        let snap = partial.get_state();

        let mut resumed = Cpu::new();
        resumed.apply_state(&snap);
        let remaining = run_one(&mut resumed, &mut bus);

        assert_eq!(resumed.regs.a, reference.regs.a);
        assert_eq!(resumed.regs.pc, reference.regs.pc);
        assert_eq!(resumed.regs.a, 0x7E);
        assert_eq!(3 + remaining, full, "snapshot+resume matches uninterrupted cycles");
    }

    // ---- Gap 6: flag mechanics — shifts and the pushed B/unused bits ----

    fn run_accumulator_op(opcode: u8, a: u8, carry_in: bool) -> Cpu {
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[opcode]);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.a = a;
        cpu.flags.c = carry_in;
        run_one(&mut cpu, &mut bus);
        cpu
    }

    #[test]
    fn asl_accumulator_carry_from_bit7() {
        // 0x81 << 1 = 0x02, bit7 -> carry. Then 0x80 << 1 = 0x00 -> Z.
        let cpu = run_accumulator_op(0x0A, 0x81, false);
        assert_eq!(cpu.regs.a, 0x02);
        assert!(cpu.flags.c, "old bit 7 shifted into carry");
        assert!(!cpu.flags.z);
        assert!(!cpu.flags.n);

        let cpu = run_accumulator_op(0x0A, 0x80, false);
        assert_eq!(cpu.regs.a, 0x00);
        assert!(cpu.flags.c);
        assert!(cpu.flags.z);
    }

    #[test]
    fn lsr_accumulator_carry_from_bit0_clears_n() {
        // 0x03 >> 1 = 0x01, bit0 -> carry; N always cleared by LSR.
        let cpu = run_accumulator_op(0x4A, 0x03, false);
        assert_eq!(cpu.regs.a, 0x01);
        assert!(cpu.flags.c, "old bit 0 shifted into carry");
        assert!(!cpu.flags.n, "LSR feeds 0 into bit 7");
        assert!(!cpu.flags.z);

        let cpu = run_accumulator_op(0x4A, 0x01, false);
        assert_eq!(cpu.regs.a, 0x00);
        assert!(cpu.flags.c);
        assert!(cpu.flags.z);
    }

    #[test]
    fn rol_accumulator_rotates_carry_into_bit0() {
        // 0x80 ROL with C=0 -> 0x00 (bit7 -> carry, 0 -> bit0), Z set.
        let cpu = run_accumulator_op(0x2A, 0x80, false);
        assert_eq!(cpu.regs.a, 0x00);
        assert!(cpu.flags.c, "old bit 7 -> carry");
        assert!(cpu.flags.z);

        // 0x40 ROL with C=1 -> 0x81 (carry into bit0), carry out clear.
        let cpu = run_accumulator_op(0x2A, 0x40, true);
        assert_eq!(cpu.regs.a, 0x81, "incoming carry lands in bit 0");
        assert!(!cpu.flags.c, "old bit 7 was 0 -> carry clears");
        assert!(cpu.flags.n);
    }

    #[test]
    fn ror_accumulator_rotates_carry_into_bit7() {
        // 0x01 ROR with C=0 -> 0x00 (bit0 -> carry), Z set.
        let cpu = run_accumulator_op(0x6A, 0x01, false);
        assert_eq!(cpu.regs.a, 0x00);
        assert!(cpu.flags.c, "old bit 0 -> carry");
        assert!(cpu.flags.z);

        // 0x02 ROR with C=1 -> 0x81 (carry into bit7), carry out clear.
        let cpu = run_accumulator_op(0x6A, 0x02, true);
        assert_eq!(cpu.regs.a, 0x81, "incoming carry lands in bit 7");
        assert!(!cpu.flags.c, "old bit 0 was 0 -> carry clears");
        assert!(cpu.flags.n);
    }

    #[test]
    fn php_pushes_b_and_unused_bits_set() {
        // PHP pushes P with both bit4 (B) and bit5 set, regardless of
        // the live flags.
        let mut p = parts();
        let mut bus = p.bus();
        let mut cpu = Cpu::new();
        cpu.regs.sp = 0xFD;
        cpu.flags = Flags::from(0x01); // only carry set
        cpu.php_finish(&mut bus);
        let pushed = bus.peek_byte(0x01FD);
        assert_eq!(pushed, 0x31, "carry + B(0x10) + unused(0x20)");
        assert_eq!(cpu.regs.sp, 0xFC, "stack pointer decremented");
    }

    #[test]
    fn interrupt_push_clears_b_but_sets_unused_and_masks_irq() {
        // IRQ/NMI entry pushes P with B (bit4) CLEAR and bit5 set, then
        // sets the I flag. This is the only difference from PHP/BRK.
        let mut p = parts();
        let mut bus = p.bus();
        let mut cpu = Cpu::new();
        cpu.regs.sp = 0xFD;
        cpu.flags = Flags::from(0x01); // carry set, I clear
        cpu.interrupt_push_status(&mut bus);
        let pushed = bus.peek_byte(0x01FD);
        assert_eq!(pushed, 0x21, "carry + unused, but B cleared");
        assert_eq!(pushed & 0x10, 0, "hardware clears B for IRQ/NMI");
        assert!(cpu.flags.i, "I set after pushing status");
    }

    #[test]
    fn brk_push_sets_b_bit() {
        // BRK, unlike a hardware IRQ, pushes with B set.
        let mut p = parts();
        let mut bus = p.bus();
        let mut cpu = Cpu::new();
        cpu.regs.sp = 0xFD;
        cpu.flags = Flags::from(0x01);
        cpu.brk_push_status(&mut bus);
        let pushed = bus.peek_byte(0x01FD);
        assert_eq!(pushed & 0x10, 0x10, "BRK sets B in the pushed status");
        assert_eq!(pushed, 0x31);
        assert!(cpu.flags.i, "BRK sets I after pushing status");
    }

    // ---- Gap 7: the `None` fallback must skip the real instruction
    // length for opcodes with no `OPCODES` entry, not a flat +1 ----

    #[test]
    fn none_fallback_opcodes_advance_pc_by_real_instruction_length() {
        // TAS $1234,Y (0x9B) has no `Instruction` table entry, so it
        // falls into the defensive `None` arm. It is a 3-byte
        // instruction; before the fix, that arm always advanced PC by
        // exactly one more byte past the opcode fetch (i.e. treated
        // every unmodeled opcode as 2 bytes total), leaving PC
        // pointing at TAS's own second operand byte instead of the
        // real next instruction.
        let mut tas = parts();
        let mut tas_bus = tas.bus();
        load_prog(&mut tas_bus, 0x0200, &[0x9B, 0x34, 0x12, 0xEA]);
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        run_one(&mut cpu, &mut tas_bus);
        assert_eq!(
            cpu.regs.pc, 0x0203,
            "TAS is 3 bytes; PC must land on the real next instruction"
        );
        run_one(&mut cpu, &mut tas_bus);
        assert_eq!(cpu.regs.pc, 0x0204, "the following NOP must decode cleanly");

        // KIL/JAM ($02) is a 1-byte instruction with no operand. The
        // same flat +1 over-consumed a byte here too, swallowing the
        // start of the next real instruction.
        let mut kil = parts();
        let mut kil_bus = kil.bus();
        load_prog(&mut kil_bus, 0x0300, &[0x02, 0xEA]);
        let mut cpu2 = Cpu::new();
        cpu2.regs.pc = 0x0300;
        run_one(&mut cpu2, &mut kil_bus);
        assert_eq!(
            cpu2.regs.pc, 0x0301,
            "KIL/JAM is 1 byte; must not swallow the following opcode"
        );
    }
    // ---- Gap 8: CLI/SEI/PLP interrupt-poll latency (the
    // `i_poll_latch` field) ----
    //
    // Real 6502 hardware decides whether to service an interrupt
    // BEFORE an instruction's final cycle, but CLI/SEI/PLP write
    // `flags.i` ON that final cycle. The mask they write is therefore
    // invisible to that decision: the change takes effect one whole
    // instruction later. blargg's `cpu_interrupts_v2/1-cli_latency`
    // is the canonical ROM; these are the equivalent cycle-level unit
    // assertions, which the repo can run without the ROM.
    //
    // Every setup below reaches its state through the real per-cycle
    // pipeline (`Cpu::tick`) — none of these tests writes
    // `i_poll_latch` directly — so each assertion is a claim about
    // observable CPU behaviour, not a restatement of the
    // implementation. Two of them (RTI, interrupt re-entry) are
    // deliberate negative controls that must hold with OR without the
    // latch: they fail if the deferral is ever over-applied.

    /// Run the interrupt sequence the CPU has already latched and
    /// report `(cycles, pushed_return_address, pushed_status)`. The
    /// pushed return address is the load-bearing receipt: it names
    /// exactly which instruction boundary the interrupt was taken at.
    fn run_pending_interrupt(cpu: &mut Cpu, bus: &mut SystemBus) -> (u32, u16, u8) {
        let sp = cpu.regs.sp;
        let hi_addr = 0x0100 | (sp as u16);
        let lo_addr = 0x0100 | (sp.wrapping_sub(1) as u16);
        let status_addr = 0x0100 | (sp.wrapping_sub(2) as u16);
        let cycles = run_one(cpu, bus);
        let ret =
            ((bus.peek_byte(hi_addr) as u16) << 8) | (bus.peek_byte(lo_addr) as u16);
        (cycles, ret, bus.peek_byte(status_addr))
    }

    #[test]
    fn irq_poll_latency_cli_defers_service_by_one_instruction() {
        // A device holds the IRQ line low while I is set. CLI clears I
        // on its own final cycle, so the poll at that same boundary
        // still sees the OLD (masked) value and the IRQ waits for the
        // instruction AFTER CLI to finish.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xEA, 0x58, 0xEA, 0xEA]); // NOP ; CLI ; NOP ; NOP
        let mut cpu = Cpu::new();
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD;
        cpu.flags.i = true;
        cpu.set_irq_line_low(true);

        // Prime the poll pipeline through the real per-cycle path.
        assert_eq!(run_one(&mut cpu, &mut bus), 2, "NOP is 2 cycles");
        assert_eq!(pending_name(&cpu), None, "a masked IRQ stays masked");

        assert_eq!(run_one(&mut cpu, &mut bus), 2, "CLI is 2 cycles");
        assert!(!cpu.flags.i, "CLI cleared I on its own final cycle");
        assert_eq!(
            pending_name(&cpu),
            None,
            "the IRQ must NOT be serviced at CLI's own boundary"
        );
        assert_eq!(cpu.regs.pc, 0x0202, "next fetch is the NOP, not the vector");

        run_one(&mut cpu, &mut bus); // the NOP after CLI runs first
        assert_eq!(cpu.regs.pc, 0x0203, "the post-CLI NOP executed");
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "the deferred IRQ is serviced one instruction later"
        );

        let (cycles, ret, status) = run_pending_interrupt(&mut cpu, &mut bus);
        assert_eq!(cycles, 7, "IRQ entry is 7 cycles");
        assert_eq!(
            ret, 0x0203,
            "pushed return address pins the boundary: after the post-CLI NOP \
             ($0203), not after CLI itself ($0202)"
        );
        assert_eq!(status & 0x10, 0, "a hardware IRQ pushes B clear");
        assert!(cpu.flags.i, "IRQ entry masks further IRQs");
    }

    #[test]
    fn irq_poll_latency_sei_cannot_mask_an_irq_pending_at_its_final_cycle() {
        // The other half of the quirk, and the half the deferral test
        // above cannot reach: if the line is already low when hardware
        // decides, SEI's own I write comes too late to stop it — the
        // IRQ is still taken at SEI's boundary. Implementations that
        // read the live `flags.i` swallow this interrupt entirely.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0x78, 0xEA]); // SEI ; NOP
        let mut cpu = Cpu::new();
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD;
        cpu.flags.i = false; // interrupts enabled...

        assert!(
            !cpu.tick(&mut bus),
            "SEI cycle 1 (opcode fetch) does not complete the instruction"
        );
        // ...and the device asserts IRQ during SEI, before the poll.
        cpu.set_irq_line_low(true);

        assert!(cpu.tick(&mut bus), "SEI completes on cycle 2");
        assert!(cpu.flags.i, "SEI set I on its final cycle");
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "SEI cannot retroactively mask an IRQ already pending at its \
             own final cycle"
        );

        let (cycles, ret, _status) = run_pending_interrupt(&mut cpu, &mut bus);
        assert_eq!(cycles, 7, "IRQ entry is 7 cycles");
        assert_eq!(
            ret, 0x0201,
            "IRQ taken at SEI's own boundary — the following NOP has not run"
        );
    }

    #[test]
    fn irq_poll_latency_plp_defers_service_like_cli() {
        // PLP writes the whole status byte — I included — on its own
        // final cycle (`plp_finish`), so it carries the same one-
        // instruction latency as CLI.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xEA, 0x28, 0xEA, 0xEA]); // NOP ; PLP ; NOP ; NOP
        bus.write_byte(0x01FE, 0x20); // status to pull: I clear, unused set
        let mut cpu = Cpu::new();
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD; // PLP increments SP, then pulls from $01FE
        cpu.flags.i = true;
        cpu.set_irq_line_low(true);

        assert_eq!(run_one(&mut cpu, &mut bus), 2, "NOP is 2 cycles");
        assert_eq!(pending_name(&cpu), None, "a masked IRQ stays masked");

        assert_eq!(run_one(&mut cpu, &mut bus), 4, "PLP is 4 cycles");
        assert!(!cpu.flags.i, "PLP pulled a status byte with I clear");
        assert_eq!(
            pending_name(&cpu),
            None,
            "the IRQ must NOT be serviced at PLP's own boundary"
        );

        run_one(&mut cpu, &mut bus); // the NOP after PLP runs first
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "the deferred IRQ is serviced one instruction later"
        );
        let (_cycles, ret, _status) = run_pending_interrupt(&mut cpu, &mut bus);
        assert_eq!(
            ret, 0x0203,
            "pushed return address pins the boundary: after the post-PLP NOP"
        );
    }

    #[test]
    fn irq_poll_latency_rti_restores_i_with_no_delay() {
        // NEGATIVE CONTROL — must pass with OR without `i_poll_latch`.
        // blargg: "RTI affects IRQ inhibition immediately." RTI pulls
        // the status byte three cycles before it ends, so the restored
        // mask IS visible to the boundary poll. A deferral that
        // over-applied here would strand every level-held mapper IRQ
        // for an extra instruction on every handler return.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0xEA, 0x40]); // NOP ; RTI
        bus.write_byte(0x01FB, 0x20); // pulled status: I clear
        bus.write_byte(0x01FC, 0x00); // return address low
        bus.write_byte(0x01FD, 0x03); // return address high -> $0300
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFA;
        cpu.flags.i = true;
        cpu.set_irq_line_low(true);

        assert_eq!(run_one(&mut cpu, &mut bus), 2, "NOP is 2 cycles");
        assert_eq!(pending_name(&cpu), None, "a masked IRQ stays masked");

        assert_eq!(run_one(&mut cpu, &mut bus), 6, "RTI is 6 cycles");
        assert_eq!(cpu.regs.pc, 0x0300, "RTI returned to the pulled address");
        assert!(!cpu.flags.i, "RTI restored a status with I clear");
        assert_eq!(
            pending_name(&cpu),
            Some("IRQ"),
            "RTI's I restore takes effect IMMEDIATELY — no one-instruction \
             deferral"
        );
    }

    #[test]
    fn irq_entry_masks_a_held_line_and_does_not_re_enter() {
        // NEGATIVE CONTROL — must pass with OR without `i_poll_latch`.
        // A mapper IRQ holds the line low until the handler acks it.
        // Entry sets I at cycle 4 of 7, so the latch has re-synced
        // well before the entry sequence's own boundary poll. If it
        // had not, the CPU would re-enter the handler at every
        // boundary and drain the stack — the Crash 'n' the Boys
        // failure class in KNOWN_ISSUES.md.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0000, &[0xEA, 0xEA, 0xEA]); // handler at the $FFFE vector
        load_prog(&mut bus, 0x0200, &[0xEA]); // NOP
        let mut cpu = Cpu::new();
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD;
        cpu.flags.i = false;
        cpu.set_irq_line_low(true);

        run_one(&mut cpu, &mut bus);
        assert_eq!(pending_name(&cpu), Some("IRQ"), "unmasked IRQ is taken");

        let (cycles, ret, _status) = run_pending_interrupt(&mut cpu, &mut bus);
        assert_eq!(cycles, 7, "IRQ entry is 7 cycles");
        assert_eq!(ret, 0x0201, "pushed return address is the next instruction");
        assert!(cpu.flags.i, "entry set I");
        assert_eq!(cpu.regs.sp, 0xFA, "entry pushed exactly three bytes");
        assert_eq!(
            pending_name(&cpu),
            None,
            "no re-entry at the entry sequence's own boundary"
        );

        for _ in 0..3 {
            run_one(&mut cpu, &mut bus);
            assert_eq!(
                pending_name(&cpu),
                None,
                "the still-low line must stay masked inside the handler"
            );
        }
        assert_eq!(cpu.regs.sp, 0xFA, "stack did not drain");
    }

    #[test]
    fn irq_poll_latency_survives_a_mid_cli_savestate_round_trip() {
        // `i_poll_latch` is deliberately NOT serialized; `apply_state`
        // re-derives it from `state.flags.i`. That is only sound
        // because the latch and the live flag can never disagree in
        // any externally observable state (CLI/SEI/PLP write I and
        // poll in the same cycle). Pin it: snapshot mid-CLI, restore
        // into a fresh CPU, and the deferral must still happen. This
        // is the savestate class that silently killed Zelda (MMC1
        // restore write-drop) — a re-derived field that gets the
        // derivation wrong is invisible until a run is already lost.
        let mut p = parts();
        let mut bus = p.bus();
        load_prog(&mut bus, 0x0200, &[0x58, 0xEA, 0xEA]); // CLI ; NOP ; NOP
        let mut cpu = Cpu::new();
        cpu.hw_irq_poll_timing = true;  // gated like hw_nmi_poll_timing; off by default
        cpu.regs.pc = 0x0200;
        cpu.regs.sp = 0xFD;
        cpu.flags.i = true;
        cpu.set_irq_line_low(true);

        assert!(!cpu.tick(&mut bus), "CLI cycle 1 (opcode fetch)");
        let snapshot = cpu.get_state(); // mid-instruction: I still set

        let mut restored = Cpu::new();
        restored.hw_irq_poll_timing = true;  // the gate is a machine setting, not saved state
        restored.apply_state(&snapshot);
        assert!(restored.flags.i, "restored mid-CLI with I still set");

        assert!(restored.tick(&mut bus), "CLI completes on cycle 2");
        assert!(!restored.flags.i, "CLI cleared I");
        assert_eq!(
            pending_name(&restored),
            None,
            "the restored CPU must still defer the IRQ past CLI"
        );

        run_one(&mut restored, &mut bus);
        assert_eq!(
            pending_name(&restored),
            Some("IRQ"),
            "and take it one instruction later, exactly as the un-saved CPU does"
        );
        let (_cycles, ret, _status) = run_pending_interrupt(&mut restored, &mut bus);
        assert_eq!(ret, 0x0202, "pushed return address: after the post-CLI NOP");
    }
}
