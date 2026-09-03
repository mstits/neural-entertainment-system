//! Differential fuzzer: randomized 6502 instruction streams through
//! both the ASM core and the pure-Rust reference, assert byte-exact
//! state AND cycle-exact timing after every instruction.
//!
//! Generates a random sequence of opcodes drawn from the set we've
//! ported to ASM, with random operand bytes following. Builds an NROM
//! ROM with that code starting at $C000 (reset vector points there),
//! seeds RAM, then single-steps both cores one instruction at a time.
//! After EACH instruction the snapshot (A/X/Y/SP/P/PC, RAM hash, and
//! the absolute CPU cycle counter) is compared. Any divergence is
//! reported with the offending seed + instruction stream.
//!
//! Coverage highlights (vs. the original immediate/zp-only fuzzer):
//!   * Indexed addressing modes — abs,X / abs,Y / (zp),Y — which are
//!     exactly where the recently-fixed ADC/SBC page-cross cycle
//!     mischarge lived. Effective addresses are constrained to the
//!     2 KB internal RAM window so no PPU/APU MMIO side effects enter
//!     the comparison (the fuzzer runs the CPU core in isolation with
//!     a null bus pointer).
//!   * The CPU cycle counter is compared after every instruction, so a
//!     +1/-1 cycle mischarge (e.g. a missing or spurious page-cross
//!     penalty) is caught directly rather than only surfacing as a
//!     downstream PC drift.
//!
//! Usage:
//!     cargo run --release --features asm_cpu --example asm_diff_fuzz
//!
//! Arguments (positional):
//!     iterations        default 10_000   — streams to fuzz
//!     instrs_per_stream default 8        — instructions per stream
//!     seed              default random   — PRNG seed for reproducibility
//!
//! Environment:
//!     ASM_FUZZ_SKIP_ADC_SBC_INDEXED  when set, drops the ADC/SBC
//!         abs,X / abs,Y / (zp),Y opcodes from the fuzz set. Use this to
//!         keep a stamina run productive while a known page-cross cycle
//!         bug in those specific ASM handlers is outstanding.
//!
//! Exit code: 0 = no divergence, 2 = at least one divergence reported.
//!
//! Designed for a ~24h stamina run via the shell (`for`, or the
//! harness loops infinitely with --forever).

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::cartridge::Cartridge;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::cpu_asm::{install_opcode_table, nes_cpu_run_block, AsmCpuState};
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::nes::Nes;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::sink::{AudioSink, VideoSink};

/// How a generated opcode's operand bytes are produced. The 6502
/// addressing mode dictates both the operand encoding and the memory
/// range the effective address is allowed to touch — the fuzzer runs
/// the CPU core against a null bus, so every data access MUST resolve
/// to the 2 KB internal RAM (< $2000). Indexed modes therefore clamp
/// their base address so `base + index` (index ∈ 0..=255) can never
/// reach the $2000-$7FFF MMIO window.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
#[derive(Clone, Copy)]
enum Gen {
    /// `n` fully-random operand bytes: immediate, zp load, implied,
    /// accumulator, or stack ops. None of these can escape RAM.
    Simple(u8),
    /// zp store (STA/STX/STY zp): 1 operand byte constrained to
    /// $00-$EF so it can never overwrite the (zp),Y pointer table
    /// planted at zp $F0-$FF.
    StoreZp,
    /// abs,X / abs,Y READ: 2 operand bytes forming a base in
    /// $0000-$1EFF, so `base + index` ∈ $0000-$1FFE stays in RAM.
    /// The low byte is random so page crossings occur ~half the time.
    AbsIdxRead,
    /// abs,X / abs,Y STORE (STA): 2 operand bytes forming a base in
    /// $0100-$06FF. Keeping stores inside the first RAM mirror (and
    /// off page 0) guarantees `base + index` never aliases the zp
    /// pointer table after the $07FF RAM-mirror mask.
    AbsIdxStore,
    /// (zp),Y (any op): 1 operand byte selecting a planted pointer
    /// slot. The pointer targets $0100-$06FF so both the read and the
    /// STA (zp),Y store stay in RAM and off the pointer table.
    IndY,
}

/// The full fuzzable opcode set. The indexed entries are the ones the
/// prior audit flagged as untested — abs,X $1D/3D/5D/7D/9D/BD/DD/FD,
/// abs,Y $19/39/59/79/99/B9/D9/F9, and (zp),Y $11/31/51/71/91/B1/D1/F1.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
const FUZZABLE_OPCODES: &[(u8, Gen)] = &[
    // ---- immediate / zp load / implied / accumulator / stack ----
    (0xA9, Gen::Simple(1)), // LDA #imm
    (0xA2, Gen::Simple(1)), // LDX #imm
    (0xA0, Gen::Simple(1)), // LDY #imm
    (0xA5, Gen::Simple(1)), // LDA zp
    (0xA6, Gen::Simple(1)), // LDX zp
    (0xA4, Gen::Simple(1)), // LDY zp
    (0x85, Gen::StoreZp),   // STA zp
    (0x86, Gen::StoreZp),   // STX zp
    (0x84, Gen::StoreZp),   // STY zp
    (0xAA, Gen::Simple(0)), // TAX
    (0xA8, Gen::Simple(0)), // TAY
    (0x8A, Gen::Simple(0)), // TXA
    (0x98, Gen::Simple(0)), // TYA
    (0xBA, Gen::Simple(0)), // TSX
    (0x9A, Gen::Simple(0)), // TXS
    (0x09, Gen::Simple(1)), // ORA #imm
    (0x29, Gen::Simple(1)), // AND #imm
    (0x49, Gen::Simple(1)), // EOR #imm
    (0x69, Gen::Simple(1)), // ADC #imm
    (0xE9, Gen::Simple(1)), // SBC #imm
    (0xC9, Gen::Simple(1)), // CMP #imm
    (0xE0, Gen::Simple(1)), // CPX #imm
    (0xC0, Gen::Simple(1)), // CPY #imm
    (0xE8, Gen::Simple(0)), // INX
    (0xC8, Gen::Simple(0)), // INY
    (0xCA, Gen::Simple(0)), // DEX
    (0x88, Gen::Simple(0)), // DEY
    (0x18, Gen::Simple(0)), // CLC
    (0x38, Gen::Simple(0)), // SEC
    (0x58, Gen::Simple(0)), // CLI
    (0x78, Gen::Simple(0)), // SEI
    (0xB8, Gen::Simple(0)), // CLV
    (0xD8, Gen::Simple(0)), // CLD
    (0xF8, Gen::Simple(0)), // SED
    (0xEA, Gen::Simple(0)), // NOP
    (0x48, Gen::Simple(0)), // PHA
    (0x68, Gen::Simple(0)), // PLA
    (0x08, Gen::Simple(0)), // PHP
    (0x28, Gen::Simple(0)), // PLP
    (0x0A, Gen::Simple(0)), // ASL A
    (0x4A, Gen::Simple(0)), // LSR A
    (0x2A, Gen::Simple(0)), // ROL A
    (0x6A, Gen::Simple(0)), // ROR A
    // ---- abs,X reads (4/5 cycles — page-cross adds 1) ----
    (0x1D, Gen::AbsIdxRead), // ORA abs,X
    (0x3D, Gen::AbsIdxRead), // AND abs,X
    (0x5D, Gen::AbsIdxRead), // EOR abs,X
    (0x7D, Gen::AbsIdxRead), // ADC abs,X
    (0xBD, Gen::AbsIdxRead), // LDA abs,X
    (0xDD, Gen::AbsIdxRead), // CMP abs,X
    (0xFD, Gen::AbsIdxRead), // SBC abs,X
    // ---- abs,Y reads (4/5 cycles — page-cross adds 1) ----
    (0x19, Gen::AbsIdxRead), // ORA abs,Y
    (0x39, Gen::AbsIdxRead), // AND abs,Y
    (0x59, Gen::AbsIdxRead), // EOR abs,Y
    (0x79, Gen::AbsIdxRead), // ADC abs,Y
    (0xB9, Gen::AbsIdxRead), // LDA abs,Y
    (0xD9, Gen::AbsIdxRead), // CMP abs,Y
    (0xF9, Gen::AbsIdxRead), // SBC abs,Y
    // ---- abs,X / abs,Y stores (fixed 5 cycles, no page-cross) ----
    (0x9D, Gen::AbsIdxStore), // STA abs,X
    (0x99, Gen::AbsIdxStore), // STA abs,Y
    // ---- (zp),Y (5/6 cycles for reads — page-cross adds 1; STA fixed 6) ----
    (0x11, Gen::IndY), // ORA (zp),Y
    (0x31, Gen::IndY), // AND (zp),Y
    (0x51, Gen::IndY), // EOR (zp),Y
    (0x71, Gen::IndY), // ADC (zp),Y
    (0x91, Gen::IndY), // STA (zp),Y
    (0xB1, Gen::IndY), // LDA (zp),Y
    (0xD1, Gen::IndY), // CMP (zp),Y
    (0xF1, Gen::IndY), // SBC (zp),Y
];

/// zp addresses [POINTER_BASE, 0xFF] hold the (zp),Y pointer table:
/// 8 little-endian pointers at $F0/$F1, $F2/$F3, … $FE/$FF. No store
/// the fuzzer generates is allowed to write this region, so pointers
/// stay valid (and in-RAM) for the whole stream.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
const POINTER_BASE: usize = 0xF0;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
const POINTER_SLOTS: usize = 8;

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
struct XorShift64 {
    s: u64,
}
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        x
    }
    fn next_u8(&mut self) -> u8 {
        self.next_u64() as u8
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
struct DiscardSinks;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
impl VideoSink for DiscardSinks {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
impl AudioSink for DiscardSinks {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    pc: u16,
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    p: u8,
    cycles: u64,
    ram_hash: u64,
}

/// Fast 64-bit FNV-1a hash of the 2KB NES RAM — cheaper than comparing
/// 2,048 bytes on every iteration; divergences land a hash mismatch
/// and then we report the full offending bytes separately if needed.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn ram_hash(ram: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in ram {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
struct FuzzCase {
    code: Vec<u8>,        // bytes starting at $C000
    opcodes: Vec<u8>,     // opcode byte of each generated instruction, in order
    init_a: u8,
    init_x: u8,
    init_y: u8,
    init_sp: u8,
    init_p: u8,
    init_ram: [u8; 2048],
    n_instructions: usize,
}

/// The six ADC/SBC indexed opcodes (abs,X / abs,Y / (zp),Y). They can be
/// quarantined via `ASM_FUZZ_SKIP_ADC_SBC_INDEXED` while a known ASM
/// page-cross cycle bug in these handlers is outstanding, so a stamina
/// run can keep hunting the rest of the opcode space instead of aborting
/// on the first divergence.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
const ADC_SBC_INDEXED: [u8; 6] = [0x7D, 0x79, 0x71, 0xFD, 0xF9, 0xF1];

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn generate_case(rng: &mut XorShift64, opcode_set: &[(u8, Gen)], instrs: usize) -> FuzzCase {
    let mut code = Vec::with_capacity(instrs * 3);
    let mut opcodes = Vec::with_capacity(instrs);
    for _ in 0..instrs {
        let (op, mode) = opcode_set[rng.range(opcode_set.len())];
        opcodes.push(op);
        code.push(op);
        match mode {
            Gen::Simple(n) => {
                for _ in 0..n {
                    code.push(rng.next_u8());
                }
            }
            Gen::StoreZp => {
                // $00-$EF: never overwrite the zp $F0-$FF pointer table.
                code.push(rng.range(POINTER_BASE) as u8);
            }
            Gen::AbsIdxRead => {
                let lo = rng.next_u8();
                // High byte $00-$1E ⇒ base ≤ $1EFF ⇒ base+0xFF ≤ $1FFE < $2000.
                let hi = rng.range(0x1F) as u8;
                code.push(lo);
                code.push(hi);
            }
            Gen::AbsIdxStore => {
                let lo = rng.next_u8();
                // High byte $01-$06 ⇒ base ∈ $0100-$06FF ⇒ effective in the
                // first RAM mirror, off page 0 (the pointer table).
                let hi = (1 + rng.range(6)) as u8;
                code.push(lo);
                code.push(hi);
            }
            Gen::IndY => {
                // Even slot in $F0-$FE so operand and operand+1 both land in
                // the planted pointer table (no zp wrap past $FF).
                let slot = (POINTER_BASE + 2 * rng.range(POINTER_SLOTS)) as u8;
                code.push(slot);
            }
        }
    }
    for _ in 0..32 {
        code.push(0xEA);
    }

    let mut ram = [0u8; 2048];
    for b in ram.iter_mut() {
        *b = rng.next_u8();
    }
    // Plant the (zp),Y pointer table AFTER randomizing so it is well
    // formed: each pointer targets $0100-$06FF so read/store effective
    // addresses stay in RAM and never alias the table itself.
    for slot in 0..POINTER_SLOTS {
        let z = POINTER_BASE + 2 * slot;
        ram[z] = rng.next_u8(); // pointer low
        ram[z + 1] = (1 + rng.range(6)) as u8; // pointer high $01-$06
    }

    FuzzCase {
        code,
        opcodes,
        init_a: rng.next_u8(),
        init_x: rng.next_u8(),
        init_y: rng.next_u8(),
        init_sp: 0xFD,
        init_p: (rng.next_u8() & 0xCF) | 0x20,
        init_ram: ram,
        n_instructions: instrs,
    }
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn build_rom(code: &[u8]) -> Vec<u8> {
    let mut rom = Vec::with_capacity(16 + 32 * 1024);
    rom.extend_from_slice(b"NES\x1a");
    rom.push(2); // PRG 2×16KB = 32KB
    rom.push(0); // CHR 0
    rom.push(0);
    rom.push(0);
    rom.extend_from_slice(&[0u8; 8]);

    let mut prg = vec![0u8; 32 * 1024];
    let start = 0x4000;
    prg[start..start + code.len()].copy_from_slice(code);
    let last = prg.len();
    prg[last - 4] = 0x00;
    prg[last - 3] = 0xC0;
    rom.extend(prg);
    rom
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn build_prg(code: &[u8]) -> Vec<u8> {
    let mut prg = vec![0u8; 32 * 1024];
    prg[0x4000..0x4000 + code.len()].copy_from_slice(code);
    let last = prg.len();
    prg[last - 4] = 0x00;
    prg[last - 3] = 0xC0;
    prg
}

/// Where a per-instruction comparison first disagreed.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
struct Divergence {
    index: usize,
    opcode: u8,
    rust: Snapshot,
    asm: Snapshot,
}

/// Single-step both cores one instruction at a time, comparing the full
/// snapshot (registers, flags, RAM, AND the absolute cycle counter)
/// after each instruction. Returns the first divergence, or `None`.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn run_case(case: &FuzzCase) -> Option<Divergence> {
    // ---- Rust reference: instruction-driven, one CPU cycle per tick. ----
    let rom = build_rom(&case.code);
    let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
    let mut nes = Nes::new(cart);
    nes.reset();

    let mut state = nes.cpu_state_for_diff_test();
    state.regs.pc = 0xC000;
    state.regs.a = case.init_a;
    state.regs.x = case.init_x;
    state.regs.y = case.init_y;
    state.regs.sp = case.init_sp;
    state.flags = case.init_p.into();
    state.cycle = 0;
    state.opcode = 0;
    state.stall_cycles = 0;
    state.nmi_pended = false;
    state.nmi_line_low = false;
    state.irq_line_low = false;
    state.active_interrupt = None;
    nes.cpu_apply_state_for_diff_test(&state);
    nes.ram_mut_for_diff_test().copy_from_slice(&case.init_ram);

    // ---- ASM core: budget=1 per call runs exactly one instruction. ----
    let prg = build_prg(&case.code);
    let mut ram = case.init_ram;
    let mut cpu = AsmCpuState {
        pc: 0xC000,
        a: case.init_a,
        x: case.init_x,
        y: case.init_y,
        sp: case.init_sp,
        p: case.init_p,
        ..Default::default()
    };

    let mut vs = DiscardSinks;
    let mut aus = DiscardSinks;
    let mut rust_cycles: u64 = 0;

    for i in 0..case.n_instructions {
        // Advance the reference by exactly one instruction, counting the
        // CPU cycles it consumes (one per tick).
        loop {
            let completed = nes.tick(&mut vs, &mut aus);
            rust_cycles += 1;
            if completed {
                break;
            }
        }

        // Advance the ASM core by exactly one instruction. A budget of 1
        // is below every opcode's cost (min 2), so the NEXT tail exits
        // after a single instruction; `cpu.cycles` accumulates the real
        // cycle charge (including any page-cross penalty).
        let _ = unsafe {
            nes_cpu_run_block(&mut cpu as *mut _, ram.as_mut_ptr(), prg.as_ptr(), 1)
        };

        let st = nes.cpu_state_for_diff_test();
        let r = Snapshot {
            pc: st.regs.pc,
            a: st.regs.a,
            x: st.regs.x,
            y: st.regs.y,
            sp: st.regs.sp,
            p: st.flags.into(),
            cycles: rust_cycles,
            ram_hash: ram_hash(nes.ram_for_diff_test()),
        };
        let a = Snapshot {
            pc: cpu.pc,
            a: cpu.a,
            x: cpu.x,
            y: cpu.y,
            sp: cpu.sp,
            p: cpu.p,
            cycles: cpu.cycles,
            ram_hash: ram_hash(&ram),
        };
        if r != a {
            return Some(Divergence {
                index: i,
                opcode: case.opcodes[i],
                rust: r,
                asm: a,
            });
        }
    }
    None
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn main() {
    let mut args = std::env::args().skip(1);
    let iterations: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let instrs_per_stream: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!(
        "asm_diff_fuzz: iterations={} instrs_per_stream={} seed={} (indexed modes + cycle compare)",
        iterations, instrs_per_stream, seed
    );

    // Optional quarantine of the ADC/SBC indexed opcodes (see const doc).
    let skip_adc_sbc = std::env::var_os("ASM_FUZZ_SKIP_ADC_SBC_INDEXED").is_some();
    let opcode_set: Vec<(u8, Gen)> = if skip_adc_sbc {
        eprintln!(
            "WARNING: ASM_FUZZ_SKIP_ADC_SBC_INDEXED set — excluding ADC/SBC \
             abs,X/abs,Y/(zp),Y ({:02X?}) from the fuzz set.",
            ADC_SBC_INDEXED
        );
        FUZZABLE_OPCODES
            .iter()
            .copied()
            .filter(|(op, _)| !ADC_SBC_INDEXED.contains(op))
            .collect()
    } else {
        FUZZABLE_OPCODES.to_vec()
    };

    // Install the dispatch table once up front; run_case reuses it.
    install_opcode_table();

    let mut rng = XorShift64::new(seed);
    let mut divergences = 0usize;
    let start = std::time::Instant::now();

    for i in 0..iterations {
        let case = generate_case(&mut rng, &opcode_set, instrs_per_stream);
        if let Some(d) = run_case(&case) {
            divergences += 1;
            if divergences <= 5 {
                eprintln!("\n--- DIVERGENCE #{} (iter {}) ---", divergences, i);
                eprintln!("  seed={} instr_index={} opcode={:02X}", seed, d.index, d.opcode);
                eprintln!("  code bytes: {:02X?}", &case.code[..case.code.len().min(48)]);
                eprintln!(
                    "  init A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X}",
                    case.init_a, case.init_x, case.init_y, case.init_sp, case.init_p
                );
                eprintln!("  rust: {:?}", d.rust);
                eprintln!("  asm:  {:?}", d.asm);
                if d.rust.cycles != d.asm.cycles {
                    eprintln!(
                        "  >>> CYCLE MISMATCH: rust={} asm={} (Δ={})",
                        d.rust.cycles,
                        d.asm.cycles,
                        d.asm.cycles as i64 - d.rust.cycles as i64
                    );
                }
            }
        }
        if (i + 1).is_multiple_of(1_000) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            eprintln!(
                "[{:>7}/{}] {:.0} cases/s, {} divergences",
                i + 1, iterations, rate, divergences
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nComplete: {} iterations in {:.2}s ({:.0} cases/s), {} divergences",
        iterations,
        elapsed,
        iterations as f64 / elapsed,
        divergences
    );

    if divergences > 0 {
        std::process::exit(2);
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "asm_cpu")))]
fn main() {
    eprintln!("asm_diff_fuzz requires --features asm_cpu on aarch64.");
    std::process::exit(1);
}
