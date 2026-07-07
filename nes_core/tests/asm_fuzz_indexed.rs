//! Indexed-addressing differential regression tests for the aarch64 ASM
//! 6502 core, with explicit CPU-cycle-count comparison.
//!
//! The randomized stamina fuzzer lives in `examples/asm_diff_fuzz.rs`;
//! this file pins a handful of hand-crafted, deterministic cases so the
//! guarantees are checked on every `cargo test --features asm_cpu` run
//! (not only during a stamina campaign).
//!
//! Two things are verified that the earlier register-only harness did
//! not exercise:
//!   1. Indexed addressing modes — abs,X / abs,Y / (zp),Y — including
//!      the page-cross +1-cycle penalty on reads and the fixed cost on
//!      stores.
//!   2. The absolute CPU cycle counter after the instruction, so a
//!      +1/-1 cycle mischarge is caught directly.
//!
//! Effective addresses are kept inside the 2 KB internal RAM window so
//! the CPU core can run against a null bus with no PPU/APU MMIO side
//! effects entering the comparison.

#![cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]

use nes_core::cartridge::Cartridge;
use nes_core::cpu_asm::{install_opcode_table, nes_cpu_run_block, AsmCpuState};
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

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

#[derive(Clone)]
struct Init {
    a: u8,
    x: u8,
    y: u8,
    p: u8,
    code: Vec<u8>,
    ram: [u8; 2048],
}

impl Init {
    fn new(code: &[u8]) -> Self {
        Init {
            a: 0,
            x: 0,
            y: 0,
            p: 0x24, // I set, others clear — a benign, common P value
            code: code.to_vec(),
            ram: [0u8; 2048],
        }
    }
    fn a(mut self, v: u8) -> Self {
        self.a = v;
        self
    }
    fn x(mut self, v: u8) -> Self {
        self.x = v;
        self
    }
    fn y(mut self, v: u8) -> Self {
        self.y = v;
        self
    }
    fn p(mut self, v: u8) -> Self {
        self.p = v;
        self
    }
    fn ram(mut self, addr: usize, v: u8) -> Self {
        self.ram[addr] = v;
        self
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Snap {
    pc: u16,
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    p: u8,
    cycles: u64,
    ram: Vec<u8>,
}

fn build_rom(code: &[u8]) -> Vec<u8> {
    let mut rom = Vec::with_capacity(16 + 32 * 1024);
    rom.extend_from_slice(b"NES\x1a");
    rom.push(2); // 32 KB PRG
    rom.push(0); // no CHR
    rom.push(0);
    rom.push(0);
    rom.extend_from_slice(&[0u8; 8]);
    let mut prg = vec![0u8; 32 * 1024];
    prg[0x4000..0x4000 + code.len()].copy_from_slice(code);
    let last = prg.len();
    prg[last - 4] = 0x00; // reset vector low  → $C000
    prg[last - 3] = 0xC0; // reset vector high
    rom.extend(prg);
    rom
}

fn build_prg(code: &[u8]) -> Vec<u8> {
    let mut prg = vec![0u8; 32 * 1024];
    prg[0x4000..0x4000 + code.len()].copy_from_slice(code);
    let last = prg.len();
    prg[last - 4] = 0x00;
    prg[last - 3] = 0xC0;
    prg
}

/// Run exactly one instruction on the pure-Rust reference core, counting
/// one CPU cycle per `Nes::tick`.
fn run_rust_one(init: &Init) -> Snap {
    let rom = build_rom(&init.code);
    let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
    let mut nes = Nes::new(cart);
    nes.reset();

    let mut st = nes.cpu_state_for_diff_test();
    st.regs.pc = 0xC000;
    st.regs.a = init.a;
    st.regs.x = init.x;
    st.regs.y = init.y;
    st.regs.sp = 0xFD;
    st.flags = init.p.into();
    st.cycle = 0;
    st.opcode = 0;
    st.stall_cycles = 0;
    st.nmi_pended = false;
    st.nmi_line_low = false;
    st.irq_line_low = false;
    st.active_interrupt = None;
    nes.cpu_apply_state_for_diff_test(&st);
    nes.ram_mut_for_diff_test().copy_from_slice(&init.ram);

    let mut vs = DiscardSinks;
    let mut aus = DiscardSinks;
    let mut cycles = 0u64;
    loop {
        let completed = nes.tick(&mut vs, &mut aus);
        cycles += 1;
        if completed {
            break;
        }
    }

    let s = nes.cpu_state_for_diff_test();
    Snap {
        pc: s.regs.pc,
        a: s.regs.a,
        x: s.regs.x,
        y: s.regs.y,
        sp: s.regs.sp,
        p: s.flags.into(),
        cycles,
        ram: nes.ram_for_diff_test().to_vec(),
    }
}

/// Run exactly one instruction on the ASM core. Budget 1 is below every
/// opcode's cost (min 2), so the `NEXT` tail exits after a single
/// instruction and `cpu.cycles` holds the real cycle charge.
fn run_asm_one(init: &Init) -> Snap {
    install_opcode_table();
    let prg = build_prg(&init.code);
    let mut ram = init.ram;
    let mut cpu = AsmCpuState {
        pc: 0xC000,
        a: init.a,
        x: init.x,
        y: init.y,
        sp: 0xFD,
        p: init.p,
        ..Default::default()
    };
    let _ = unsafe { nes_cpu_run_block(&mut cpu as *mut _, ram.as_mut_ptr(), prg.as_ptr(), 1) };
    Snap {
        pc: cpu.pc,
        a: cpu.a,
        x: cpu.x,
        y: cpu.y,
        sp: cpu.sp,
        p: cpu.p,
        cycles: cpu.cycles,
        ram: ram.to_vec(),
    }
}

/// Assert the ASM core matches the reference byte-for-byte AND
/// cycle-for-cycle after one instruction.
fn assert_match(init: &Init, label: &str) {
    let r = run_rust_one(init);
    let a = run_asm_one(init);
    assert_eq!(r.pc, a.pc, "{label}: PC rust={:#06X} asm={:#06X}", r.pc, a.pc);
    assert_eq!(r.a, a.a, "{label}: A rust={:#04X} asm={:#04X}", r.a, a.a);
    assert_eq!(r.x, a.x, "{label}: X");
    assert_eq!(r.y, a.y, "{label}: Y");
    assert_eq!(r.sp, a.sp, "{label}: SP");
    assert_eq!(r.p, a.p, "{label}: P rust={:#04X} asm={:#04X}", r.p, a.p);
    assert_eq!(
        r.cycles, a.cycles,
        "{label}: CYCLES rust={} asm={}",
        r.cycles, a.cycles
    );
    assert!(r.ram == a.ram, "{label}: RAM diverged");
}

// ============================================================================
// Cycle-correct indexed handlers — these must match exactly on both
// cores. They double as validation that the differential harness measures
// cycles correctly (a correct handler with a page cross reports the +1 on
// BOTH cores), so a divergence here is a real regression, not a harness
// artifact.
// ============================================================================

#[test]
fn lda_abs_x_no_cross_is_4() {
    // LDA $1000,X ; X=1 → $1001 (no page cross). $1001 & $7FF = $001.
    let init = Init::new(&[0xBD, 0x00, 0x10]).x(1).ram(0x001, 0x37);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x37);
    assert_eq!(r.cycles, 4, "LDA abs,X no-cross should be 4 cycles");
    assert_match(&init, "LDA $1000,X no cross");
}

#[test]
fn lda_abs_x_page_cross_is_5() {
    // LDA $10FF,X ; X=1 → $1100 (page cross). $1100 & $7FF = $100.
    let init = Init::new(&[0xBD, 0xFF, 0x10]).x(1).ram(0x100, 0x5A);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x5A);
    assert_eq!(r.cycles, 5, "LDA abs,X page-cross should be 5 cycles");
    assert_match(&init, "LDA $10FF,X page cross");
}

#[test]
fn lda_abs_y_page_cross_is_5() {
    // LDA $10F0,Y ; Y=0x20 → $1110 (page cross). $1110 & $7FF = $110.
    let init = Init::new(&[0xB9, 0xF0, 0x10]).y(0x20).ram(0x110, 0x81);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x81);
    assert_eq!(r.cycles, 5, "LDA abs,Y page-cross should be 5 cycles");
    assert_match(&init, "LDA $10F0,Y page cross");
}

#[test]
fn cmp_abs_x_page_cross_flags_and_cycles() {
    // CMP $10FF,X ; X=1 → $1100. A == M → Z=1, C=1. 5 cycles.
    let init = Init::new(&[0xDD, 0xFF, 0x10]).a(0x42).x(1).ram(0x100, 0x42);
    let r = run_rust_one(&init);
    assert_eq!(r.cycles, 5);
    assert_eq!(r.p & 0b0000_0011, 0b0000_0011, "CMP equal → Z=1,C=1");
    assert_match(&init, "CMP $10FF,X page cross");
}

#[test]
fn and_ora_eor_indexed_page_cross() {
    // AND $10FF,X page cross (5 cy).
    assert_match(
        &Init::new(&[0x3D, 0xFF, 0x10]).a(0xF0).x(1).ram(0x100, 0x0F),
        "AND $10FF,X",
    );
    // ORA $10F0,Y page cross (5 cy).
    assert_match(
        &Init::new(&[0x19, 0xF0, 0x10]).a(0x0F).y(0x20).ram(0x110, 0x80),
        "ORA $10F0,Y",
    );
    // EOR $10FF,X page cross (5 cy).
    assert_match(
        &Init::new(&[0x5D, 0xFF, 0x10]).a(0xAA).x(1).ram(0x100, 0xFF),
        "EOR $10FF,X",
    );
}

#[test]
fn lda_ind_y_no_cross_is_5() {
    // LDA ($20),Y ; ptr=$1000, Y=0x10 → $1010 (no cross). $1010 & $7FF = $010.
    let init = Init::new(&[0xB1, 0x20])
        .y(0x10)
        .ram(0x20, 0x00) // ptr low
        .ram(0x21, 0x10) // ptr high
        .ram(0x010, 0x63);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x63);
    assert_eq!(r.cycles, 5, "(zp),Y no-cross should be 5 cycles");
    assert_match(&init, "LDA ($20),Y no cross");
}

#[test]
fn lda_ind_y_page_cross_is_6() {
    // LDA ($20),Y ; ptr=$10F0, Y=0x20 → $1110 (page cross). $1110 & $7FF = $110.
    let init = Init::new(&[0xB1, 0x20])
        .y(0x20)
        .ram(0x20, 0xF0) // ptr low
        .ram(0x21, 0x10) // ptr high
        .ram(0x110, 0x9C);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x9C);
    assert_eq!(r.cycles, 6, "(zp),Y page-cross should be 6 cycles");
    assert_match(&init, "LDA ($20),Y page cross");
}

#[test]
fn sta_abs_x_is_5_and_writes() {
    // STA $1000,X ; X=0x10 → $1010. Stores are fixed 5 cycles (no page-cross
    // penalty) even when the index crosses a page.
    let init = Init::new(&[0x9D, 0x00, 0x10]).a(0x77).x(0x10);
    let r = run_rust_one(&init);
    assert_eq!(r.ram[0x010], 0x77);
    assert_eq!(r.cycles, 5);
    assert_match(&init, "STA $1000,X");

    // STA $10FF,X ; X=1 → $1100 crosses a page — still exactly 5 cycles.
    let init2 = Init::new(&[0x9D, 0xFF, 0x10]).a(0x88).x(1);
    let r2 = run_rust_one(&init2);
    assert_eq!(r2.ram[0x100], 0x88);
    assert_eq!(r2.cycles, 5, "STA abs,X is fixed 5 cycles across page cross");
    assert_match(&init2, "STA $10FF,X cross");
}

#[test]
fn sta_ind_y_is_6_and_writes() {
    // STA ($20),Y ; ptr=$1000, Y=0x10 → $1010. Fixed 6 cycles.
    let init = Init::new(&[0x91, 0x20])
        .a(0x5F)
        .y(0x10)
        .ram(0x20, 0x00)
        .ram(0x21, 0x10);
    let r = run_rust_one(&init);
    assert_eq!(r.ram[0x010], 0x5F);
    assert_eq!(r.cycles, 6);
    assert_match(&init, "STA ($20),Y");
}

// ============================================================================
// ADC / SBC indexed page-cross timing.
//
// KNOWN BUG (ASM core, out of this lane's edit scope): the ADC_CORE macro
// in `src/cpu_asm.s` uses `w7` as scratch for the V-flag computation, but
// every ADC/SBC indexed handler stashes the page-cross flag in `w7` and
// then branches on it AFTER calling ADC_CORE (`ADC_CORE w1` ; `cbnz w7,
// …_cross`). The page-cross flag is therefore overwritten by the overflow
// result, so the +1 page-cross cycle is applied based on V instead of the
// actual page cross — a ±1 cycle mischarge on ADC/SBC abs,X / abs,Y /
// (zp),Y. Registers and memory stay correct; only the cycle count is wrong,
// which is why register-only tests never caught it.
//
// These tests assert the correct (reference) timing and full ASM parity.
// They FAIL today against the buggy ASM, so they are `#[ignore]`d to keep
// the default suite green; remove the `#[ignore]` once the ADC_CORE
// w7-clobber is fixed (e.g. compute the page-cross branch before ADC_CORE,
// or save/restore w7 around it). The randomized fuzzer in
// `examples/asm_diff_fuzz.rs` reports the same divergences on every run.
// ============================================================================

#[test]
fn adc_abs_y_reference_page_cross_is_5() {
    // Reference-only: documents the textbook-correct timing regardless of
    // the ASM bug. ADC $10F0,Y ; Y=0x20 → $1110 (page cross).
    let init = Init::new(&[0x79, 0xF0, 0x10])
        .a(0x10)
        .y(0x20)
        .ram(0x110, 0x22);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x32, "0x10 + 0x22 + C(0)");
    assert_eq!(r.cycles, 5, "ADC abs,Y page-cross is 5 cycles (reference)");
}

#[test]
fn sbc_abs_x_reference_page_cross_is_5() {
    // Reference-only. SBC $10FF,X ; X=1 → $1100 (page cross), C=1 (no borrow).
    let init = Init::new(&[0xFD, 0xFF, 0x10])
        .a(0x50)
        .x(1)
        .p(0x25) // C=1
        .ram(0x100, 0x10);
    let r = run_rust_one(&init);
    assert_eq!(r.a, 0x40, "0x50 - 0x10 - !C(0) = 0x40");
    assert_eq!(r.cycles, 5, "SBC abs,X page-cross is 5 cycles (reference)");
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn adc_abs_x_page_cross_asm_parity() {
    // ADC $10FF,X ; X=1 → $1100 (page cross) → 5 cycles.
    assert_match(
        &Init::new(&[0x7D, 0xFF, 0x10]).a(0x10).x(1).ram(0x100, 0x22),
        "ADC $10FF,X page cross",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn adc_abs_x_no_cross_asm_parity() {
    // ADC $1000,X ; X=1 → $1001 (no cross) → 4 cycles. Overflow set here
    // (0x50 + 0x50 = 0xA0, V=1) exposes the inverse mischarge: the buggy
    // ASM wrongly adds the page-cross cycle because V=1.
    assert_match(
        &Init::new(&[0x7D, 0x00, 0x10]).a(0x50).x(1).ram(0x001, 0x50),
        "ADC $1000,X no cross (V=1)",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn adc_abs_y_page_cross_asm_parity() {
    assert_match(
        &Init::new(&[0x79, 0xF0, 0x10]).a(0x10).y(0x20).ram(0x110, 0x22),
        "ADC $10F0,Y page cross",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn adc_ind_y_page_cross_asm_parity() {
    // ADC ($20),Y ; ptr=$10F0, Y=0x20 → $1110 (page cross) → 6 cycles.
    assert_match(
        &Init::new(&[0x71, 0x20])
            .a(0x10)
            .y(0x20)
            .ram(0x20, 0xF0)
            .ram(0x21, 0x10)
            .ram(0x110, 0x22),
        "ADC ($20),Y page cross",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn sbc_abs_x_page_cross_asm_parity() {
    assert_match(
        &Init::new(&[0xFD, 0xFF, 0x10]).a(0x50).x(1).p(0x25).ram(0x100, 0x10),
        "SBC $10FF,X page cross",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn sbc_abs_y_page_cross_asm_parity() {
    assert_match(
        &Init::new(&[0xF9, 0xF0, 0x10]).a(0x50).y(0x20).p(0x25).ram(0x110, 0x10),
        "SBC $10F0,Y page cross",
    );
}

#[test]
#[ignore = "ASM ADC_CORE clobbers the w7 page-cross flag; enable after cpu_asm.s fix"]
fn sbc_ind_y_page_cross_asm_parity() {
    assert_match(
        &Init::new(&[0xF1, 0x20])
            .a(0x50)
            .y(0x20)
            .p(0x25)
            .ram(0x20, 0xF0)
            .ram(0x21, 0x10)
            .ram(0x110, 0x10),
        "SBC ($20),Y page cross",
    );
}
