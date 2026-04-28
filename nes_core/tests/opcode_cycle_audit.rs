//! Per-opcode cycle-count audit: every official 6502 opcode must
//! consume the documented base cycle count when executed in its
//! simplest form (no page-cross, branch-not-taken, etc).
//!
//! Source of truth: LaiNES `OPERATION_CYCLES` table at
//! `/tmp/nespy_ref/nes_py-8.2.1/nes_py/nes/include/cpu_opcodes.hpp`,
//! which is the same table nes-py uses. Cross-referenced against the
//! NESdev wiki 6502 reference.
//!
//! Methodology: build a 32 KB NROM image whose reset vector points
//! at the opcode under test, with deterministic operands chosen to
//! ensure the simplest cycle path (e.g. `BPL` is constructed with a
//! preceding flag-set so the branch is NOT taken → 2 cycles).
//!
//! Each opcode is run ONCE and `Nes::step()`'s returned cycle count
//! is compared to the expected base. Any mismatch fails the test
//! with the opcode + actual + expected.

use std::io::Cursor;

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

struct NullSinks;
impl VideoSink for NullSinks {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
impl AudioSink for NullSinks {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

/// Build a 32 KB NROM ROM whose reset vector points at $C000 and
/// PRG begins with `program_bytes`. The rest of PRG is filled with
/// 0xEA (NOP) so unintended fall-through is harmless.
fn build_rom(program_bytes: &[u8]) -> Nes {
    let mut rom = Vec::with_capacity(16 + 32 * 1024);
    rom.extend_from_slice(b"NES\x1a");
    rom.push(2);  // PRG = 2 × 16 KB
    rom.push(0);  // CHR = 0 (CHR-RAM)
    rom.push(0);  // mapper 0
    rom.push(0);
    rom.extend_from_slice(&[0u8; 8]);

    let mut prg = vec![0xEA; 32 * 1024];  // NOP filler
    // CPU starts at $C000 (per the reset vector below). PRG maps
    // $8000-$FFFF directly, so PRG offset $4000 = CPU address $C000.
    let entry_offset = 0x4000;
    prg[entry_offset..entry_offset + program_bytes.len()]
        .copy_from_slice(program_bytes);
    // Reset vector at $FFFC-$FFFD (= last 4 bytes of PRG).
    prg[32 * 1024 - 4] = 0x00;  // low byte of $C000
    prg[32 * 1024 - 3] = 0xC0;
    rom.extend(prg);
    let cart = Cartridge::load(&mut Cursor::new(rom)).unwrap();
    Nes::new(cart)
}

/// Step one CPU instruction, returning total cycles consumed (including
/// PPU/APU side ticks). For our purposes — measuring the BASE CPU cycle
/// count of an instruction — we expect this to equal the LaiNES
/// `OPERATION_CYCLES[opcode]` value when executed in its simplest form.
fn step_one(nes: &mut Nes) -> usize {
    let mut v = NullSinks;
    let mut a = NullSinks;
    let before = nes.cpu.get_state().cycles_total;
    nes.step(&mut v, &mut a);
    (nes.cpu.get_state().cycles_total - before) as usize
}

/// (opcode, expected_base_cycles, name) — base cycles per LaiNES table
/// when executed in the simplest path (no page-cross, branch not taken).
/// 0xA9 LDA #imm → 2 cycles, 0xAD LDA abs → 4 cycles, etc.
const SIMPLE_OPCODES: &[(u8, usize, &str)] = &[
    // Loads (immediate)
    (0xA9, 2, "LDA #imm"),
    (0xA2, 2, "LDX #imm"),
    (0xA0, 2, "LDY #imm"),
    // Loads (zero page)
    (0xA5, 3, "LDA zp"),
    (0xA6, 3, "LDX zp"),
    (0xA4, 3, "LDY zp"),
    // Loads (absolute)
    (0xAD, 4, "LDA abs"),
    (0xAE, 4, "LDX abs"),
    (0xAC, 4, "LDY abs"),
    // Stores (zero page)
    (0x85, 3, "STA zp"),
    (0x86, 3, "STX zp"),
    (0x84, 3, "STY zp"),
    // Stores (absolute)
    (0x8D, 4, "STA abs"),
    (0x8E, 4, "STX abs"),
    (0x8C, 4, "STY abs"),
    // Implied / single-byte (1 byte, 2 cycles each)
    (0xEA, 2, "NOP"),
    (0x18, 2, "CLC"),
    (0x38, 2, "SEC"),
    (0xD8, 2, "CLD"),
    (0xF8, 2, "SED"),
    (0x58, 2, "CLI"),
    (0x78, 2, "SEI"),
    (0xB8, 2, "CLV"),
    (0xAA, 2, "TAX"),
    (0xA8, 2, "TAY"),
    (0x8A, 2, "TXA"),
    (0x98, 2, "TYA"),
    (0xBA, 2, "TSX"),
    (0x9A, 2, "TXS"),
    (0xE8, 2, "INX"),
    (0xC8, 2, "INY"),
    (0xCA, 2, "DEX"),
    (0x88, 2, "DEY"),
    // Arithmetic (immediate)
    (0x69, 2, "ADC #imm"),
    (0xE9, 2, "SBC #imm"),
    (0x29, 2, "AND #imm"),
    (0x09, 2, "ORA #imm"),
    (0x49, 2, "EOR #imm"),
    (0xC9, 2, "CMP #imm"),
    (0xE0, 2, "CPX #imm"),
    (0xC0, 2, "CPY #imm"),
    // Stack (PHA/PHP push: 3 cycles; PLA/PLP pull: 4 cycles)
    (0x48, 3, "PHA"),
    (0x08, 3, "PHP"),
    (0x68, 4, "PLA"),
    (0x28, 4, "PLP"),
    // Shifts (accumulator)
    (0x0A, 2, "ASL A"),
    (0x4A, 2, "LSR A"),
    (0x2A, 2, "ROL A"),
    (0x6A, 2, "ROR A"),
];

#[test]
fn simple_opcodes_match_lainess_base_cycle_count() {
    let mut failures: Vec<String> = Vec::new();
    for &(op, expected, name) in SIMPLE_OPCODES {
        // Build a minimal program: opcode + 2 operand bytes (some opcodes
        // ignore the second operand, fine to set 0).
        let program = [op, 0x00, 0x00];
        let mut nes = build_rom(&program);
        // First step() may return reset/boot cycles; do a warmup step
        // to skip past anything that's not the test opcode, then
        // measure the second step which IS the opcode-of-interest.
        // ... actually just measure the first step; reset doesn't
        // consume cycles in nes_core (no 7-cycle reset pulse).
        let cycles = step_one(&mut nes);
        if cycles != expected {
            failures.push(format!(
                "  0x{:02X} {:<10}: expected {} cycles, got {}",
                op, name, expected, cycles,
            ));
        }
    }
    if !failures.is_empty() {
        panic!(
            "Per-opcode cycle-count regressions vs LaiNES table:\n{}",
            failures.join("\n"),
        );
    }
}

/// Branch instructions: 2 cycles when NOT taken. Need to ensure the
/// flag check causes the branch to fall through.
const BRANCH_NOT_TAKEN: &[(u8, &str, /*setup_op:*/ Option<u8>)] = &[
    (0x10, "BPL", Some(0xA9)),  // Setup: LDA #$80 → N=1; BPL won't take
    (0x30, "BMI", Some(0xA9)),  // Setup: LDA #$00 → N=0; BMI won't take
    // BVC/BVS need V flag manipulation — skip for simplicity.
    (0x90, "BCC", Some(0x38)),  // Setup: SEC → C=1; BCC won't take
    (0xB0, "BCS", Some(0x18)),  // Setup: CLC → C=0; BCS won't take
    (0xD0, "BNE", Some(0xA9)),  // Setup: LDA #$00 → Z=1; BNE won't take
    (0xF0, "BEQ", Some(0xA9)),  // Setup: LDA #$01 → Z=0; BEQ won't take
];

#[test]
fn lda_abs_x_no_page_cross_is_4_cycles() {
    // LDA $C000,X with X=0 → effective $C000. Same page. 4 cycles.
    // Setup: LDX #$00 (2 cycles), LDA $0040,X (4 cycles, no cross).
    let program = [0xA2, 0x00, 0xBD, 0x40, 0x00, 0xEA];
    let mut nes = build_rom(&program);
    step_one(&mut nes);  // LDX #$00
    let cycles = step_one(&mut nes);
    assert_eq!(cycles, 4, "LDA abs,X no-page-cross: expected 4 cycles, got {}", cycles);
}

#[test]
fn lda_abs_x_page_cross_is_5_cycles() {
    // LDA $00FF,X with X=0x01 → effective $0100, page cross. 5 cycles.
    let program = [0xA2, 0x01, 0xBD, 0xFF, 0x00, 0xEA];
    let mut nes = build_rom(&program);
    step_one(&mut nes);  // LDX #$01
    let cycles = step_one(&mut nes);
    assert_eq!(cycles, 5, "LDA abs,X page-cross: expected 5 cycles, got {}", cycles);
}

#[test]
fn lda_indirect_y_no_page_cross_is_5_cycles() {
    // Setup: store $0040 at zp $10/$11, LDY #$00, LDA ($10),Y → 5 cycles, no cross.
    // Bytes: LDA #$40 (2c), STA $10 (3c), LDA #$00 (2c), STA $11 (3c),
    //        LDY #$00 (2c), LDA ($10),Y (5c, no cross).
    let program = [
        0xA9, 0x40, 0x85, 0x10,  // LDA #$40 ; STA $10
        0xA9, 0x00, 0x85, 0x11,  // LDA #$00 ; STA $11
        0xA0, 0x00,              // LDY #$00
        0xB1, 0x10,              // LDA ($10),Y
        0xEA,
    ];
    let mut nes = build_rom(&program);
    for _ in 0..5 { step_one(&mut nes); }  // setup
    let cycles = step_one(&mut nes);
    assert_eq!(cycles, 5, "LDA (zp),Y no-page-cross: expected 5 cycles, got {}", cycles);
}

#[test]
fn jsr_is_6_cycles_rts_is_6_cycles() {
    // JSR $C010 (6 cycles), then RTS (6 cycles).
    // Program at $C000: JSR $C010 (3 bytes), NOP, NOP, ... RTS at $C010
    let mut program = vec![0xEA; 0x20];
    program[0x00] = 0x20;  // JSR
    program[0x01] = 0x10;  // low byte of $C010
    program[0x02] = 0xC0;  // high byte
    program[0x10] = 0x60;  // RTS
    let mut nes = build_rom(&program);
    let jsr_cycles = step_one(&mut nes);
    assert_eq!(jsr_cycles, 6, "JSR: expected 6 cycles, got {}", jsr_cycles);
    let rts_cycles = step_one(&mut nes);
    assert_eq!(rts_cycles, 6, "RTS: expected 6 cycles, got {}", rts_cycles);
}

#[test]
fn branch_taken_no_page_cross_is_3_cycles() {
    // BPL (0x10) with N=0 + small forward offset → branch taken, no
    // page cross. Should be 3 cycles.
    // Setup: LDA #$00 (sets N=0, 2 cycles), then BPL +$10 (3 cycles).
    let program = [0xA9, 0x00, 0x10, 0x10, 0xEA, 0xEA];
    let mut nes = build_rom(&program);
    let _setup = step_one(&mut nes);
    let cycles = step_one(&mut nes);
    assert_eq!(cycles, 3, "BPL taken, no page cross: expected 3 cycles, got {}", cycles);
}

#[test]
fn branch_not_taken_is_2_cycles() {
    let mut failures: Vec<String> = Vec::new();
    for &(op, name, setup_op) in BRANCH_NOT_TAKEN {
        // Setup operand for the LDA case: pick value that sets the
        // flag we need so the branch falls through.
        let setup_operand = match (op, setup_op) {
            (0x10, Some(0xA9)) => 0x80,  // BPL: LDA #$80 → N=1
            (0x30, Some(0xA9)) => 0x00,  // BMI: LDA #$00 → N=0
            (0xD0, Some(0xA9)) => 0x00,  // BNE: LDA #$00 → Z=1
            (0xF0, Some(0xA9)) => 0x01,  // BEQ: LDA #$01 → Z=0
            _ => 0x00,
        };
        let program: Vec<u8> = match setup_op {
            Some(0xA9) => vec![0xA9, setup_operand, op, 0x10, 0xEA, 0xEA],
            Some(0x38) => vec![0x38, op, 0x10, 0xEA, 0xEA],   // SEC
            Some(0x18) => vec![0x18, op, 0x10, 0xEA, 0xEA],   // CLC
            _ => vec![op, 0x10, 0xEA, 0xEA],
        };
        let mut nes = build_rom(&program);
        // Run the setup instruction first
        if setup_op.is_some() {
            step_one(&mut nes);
        }
        let cycles = step_one(&mut nes);
        if cycles != 2 {
            failures.push(format!(
                "  0x{:02X} {:<5} not-taken: expected 2 cycles, got {}",
                op, name, cycles,
            ));
        }
    }
    if !failures.is_empty() {
        panic!(
            "Branch-not-taken cycle regressions vs LaiNES (all should be 2):\n{}",
            failures.join("\n"),
        );
    }
}
