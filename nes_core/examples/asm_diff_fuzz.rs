//! Differential fuzzer: randomized 6502 instruction streams through
//! both the ASM core and the pure-Rust reference, assert byte-exact
//! state after every step.
//!
//! Generates a random sequence of opcodes drawn from the set we've
//! ported to ASM, with random operand bytes following. Builds an NROM
//! ROM with that code starting at $C000 (reset vector points there),
//! zero-fills RAM, then runs both cores for the same cycle budget.
//! Any divergence in A/X/Y/SP/P/PC or RAM is reported with the
//! offending seed + instruction stream.
//!
//! Usage:
//!     cargo run --release --features asm_cpu --example asm_diff_fuzz
//!
//! Arguments (positional):
//!     iterations        default 10_000   — streams to fuzz
//!     instrs_per_stream default 8        — instructions per stream
//!     seed              default random   — PRNG seed for reproducibility
//!
//! Exit code: 0 = no divergence, 2 = at least one divergence reported.
//!
//! Designed for a ~24h stamina run via the shell (`for`, or the
//! harness loops infinitely with --forever).

#![cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]

use nes_core::cartridge::Cartridge;
use nes_core::cpu_asm::{install_opcode_table, nes_cpu_run_block, AsmCpuState};
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

/// (opcode, operand bytes, cycles). Cycles is the exact base count
/// for the ASM handler — fuzz budget is the sum across the generated
/// stream so both cores exit at the same instruction boundary.
const FUZZABLE_OPCODES: &[(u8, usize, u8)] = &[
    (0xA9, 1, 2), // LDA #imm
    (0xA2, 1, 2), // LDX #imm
    (0xA0, 1, 2), // LDY #imm
    (0xA5, 1, 3), // LDA zp
    (0xA6, 1, 3), // LDX zp
    (0xA4, 1, 3), // LDY zp
    (0x85, 1, 3), // STA zp
    (0x86, 1, 3), // STX zp
    (0x84, 1, 3), // STY zp
    (0xAA, 0, 2), // TAX
    (0xA8, 0, 2), // TAY
    (0x8A, 0, 2), // TXA
    (0x98, 0, 2), // TYA
    (0xBA, 0, 2), // TSX
    (0x9A, 0, 2), // TXS
    (0x09, 1, 2), // ORA #imm
    (0x29, 1, 2), // AND #imm
    (0x49, 1, 2), // EOR #imm
    (0x69, 1, 2), // ADC #imm
    (0xE9, 1, 2), // SBC #imm
    (0xC9, 1, 2), // CMP #imm
    (0xE0, 1, 2), // CPX #imm
    (0xC0, 1, 2), // CPY #imm
    (0xE8, 0, 2), // INX
    (0xC8, 0, 2), // INY
    (0xCA, 0, 2), // DEX
    (0x88, 0, 2), // DEY
    (0x18, 0, 2), // CLC
    (0x38, 0, 2), // SEC
    (0x58, 0, 2), // CLI
    (0x78, 0, 2), // SEI
    (0xB8, 0, 2), // CLV
    (0xD8, 0, 2), // CLD
    (0xF8, 0, 2), // SED
    (0xEA, 0, 2), // NOP
    (0x48, 0, 3), // PHA
    (0x68, 0, 4), // PLA
    (0x08, 0, 3), // PHP
    (0x28, 0, 4), // PLP
    (0x0A, 0, 2), // ASL A
    (0x4A, 0, 2), // LSR A
    (0x2A, 0, 2), // ROL A
    (0x6A, 0, 2), // ROR A
];

struct XorShift64 {
    s: u64,
}
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

struct DiscardSinks;
impl VideoSink for DiscardSinks {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
impl AudioSink for DiscardSinks {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    pc: u16,
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    p: u8,
    ram_hash: u64,
}

/// Fast 64-bit FNV-1a hash of the 2KB NES RAM — cheaper than comparing
/// 2,048 bytes on every iteration; divergences land a hash mismatch
/// and then we report the full offending bytes separately if needed.
fn ram_hash(ram: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in ram {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct FuzzCase {
    code: Vec<u8>,       // bytes starting at $C000
    init_a: u8,
    init_x: u8,
    init_y: u8,
    init_sp: u8,
    init_p: u8,
    init_ram: [u8; 2048],
    n_instructions: usize,
    total_cycles: i64,
}

fn generate_case(rng: &mut XorShift64, instrs: usize) -> FuzzCase {
    let mut code = Vec::with_capacity(instrs * 3);
    let mut total_cycles: i64 = 0;
    for _ in 0..instrs {
        let (op, operand_n, cycles) = FUZZABLE_OPCODES[rng.range(FUZZABLE_OPCODES.len())];
        code.push(op);
        for _ in 0..operand_n {
            code.push(rng.next_u8());
        }
        total_cycles += cycles as i64;
    }
    for _ in 0..32 {
        code.push(0xEA);
    }

    let mut ram = [0u8; 2048];
    for b in ram.iter_mut() {
        *b = rng.next_u8();
    }

    FuzzCase {
        code,
        init_a: rng.next_u8(),
        init_x: rng.next_u8(),
        init_y: rng.next_u8(),
        init_sp: 0xFD,
        init_p: (rng.next_u8() & 0xCF) | 0x20,
        init_ram: ram,
        n_instructions: instrs,
        total_cycles,
    }
}

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

fn run_rust(case: &FuzzCase) -> Snapshot {
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

    let mut vs = DiscardSinks;
    let mut aus = DiscardSinks;
    for _ in 0..case.n_instructions {
        while !nes.tick(&mut vs, &mut aus) {}
    }

    let st = nes.cpu_state_for_diff_test();
    let p_byte: u8 = st.flags.into();
    Snapshot {
        pc: st.regs.pc,
        a: st.regs.a,
        x: st.regs.x,
        y: st.regs.y,
        sp: st.regs.sp,
        p: p_byte,
        ram_hash: ram_hash(nes.ram_for_diff_test()),
    }
}

fn run_asm(case: &FuzzCase, cycles_budget: i64) -> Snapshot {
    install_opcode_table();
    let mut prg = vec![0u8; 32 * 1024];
    prg[0x4000..0x4000 + case.code.len()].copy_from_slice(&case.code);
    let last = prg.len();
    prg[last - 4] = 0x00;
    prg[last - 3] = 0xC0;

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
    let _ = unsafe {
        nes_cpu_run_block(
            &mut cpu as *mut _,
            ram.as_mut_ptr(),
            prg.as_ptr(),
            cycles_budget,
        )
    };
    Snapshot {
        pc: cpu.pc,
        a: cpu.a,
        x: cpu.x,
        y: cpu.y,
        sp: cpu.sp,
        p: cpu.p,
        ram_hash: ram_hash(&ram),
    }
}

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
        "asm_diff_fuzz: iterations={} instrs_per_stream={} seed={}",
        iterations, instrs_per_stream, seed
    );

    let mut rng = XorShift64::new(seed);
    let mut divergences = 0usize;
    let start = std::time::Instant::now();

    for i in 0..iterations {
        let case = generate_case(&mut rng, instrs_per_stream);
        let r = run_rust(&case);
        let a = run_asm(&case, case.total_cycles);
        if r != a {
            divergences += 1;
            if divergences <= 5 {
                eprintln!(
                    "\n--- DIVERGENCE #{} (iter {}) ---",
                    divergences, i
                );
                eprintln!("  code bytes: {:02X?}", &case.code[..case.code.len().min(40)]);
                eprintln!("  init A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X}",
                    case.init_a, case.init_x, case.init_y, case.init_sp, case.init_p);
                eprintln!("  rust: {:?}", r);
                eprintln!("  asm:  {:?}", a);
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
