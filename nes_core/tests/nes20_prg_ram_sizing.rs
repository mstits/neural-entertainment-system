//! Regression tests for NES 2.0 PRG-RAM sizing.
//!
//! NES 2.0 header byte 10 packs TWO nibbles:
//!   bits 0-3: volatile PRG-RAM shift count
//!   bits 4-7: non-volatile (battery) PRG-RAM shift count
//!
//! For shift count S, size = 64 << S bytes. Shift 0 = "no RAM of
//! that kind". iNES 1.0 header byte 8 is PRG-RAM size in 8 KB banks
//! (legacy zero-means-8 KB fallback).
//!
//! The historic bug: the parser read only the LOWER nibble (volatile)
//! and ignored the UPPER nibble (battery). A Zelda dump with
//! `flags10 = 0x70` (no volatile + 8 KB battery) ended up with 0 B of
//! PRG-RAM allocated. Zelda's code writes to `$6000-$7FFF` for save
//! data and title-screen scratch; with no backing, those writes hit
//! nothing and subsequent reads return open-bus. A read-then-indirect
//! -JMP eventually fetches a bogus address → CPU jumps to garbage →
//! BRK → Zelda's IRQ trap at `$FFF0`.

use nes_core::cartridge::Cartridge;
use std::io::Cursor;

/// Build a minimal NES 2.0 header + empty PRG/CHR payloads. Only the
/// sizing bytes matter for PRG-RAM sizing tests; contents are zero.
fn synth_nes20_rom(
    prg_banks_lo: u8,
    chr_banks_lo: u8,
    flags6: u8,
    flags7: u8, // set bit 3 for NES 2.0
    flags10: u8,
) -> Vec<u8> {
    let mut rom = vec![0u8; 16];
    rom[0] = b'N';
    rom[1] = b'E';
    rom[2] = b'S';
    rom[3] = 0x1A;
    rom[4] = prg_banks_lo;
    rom[5] = chr_banks_lo;
    rom[6] = flags6;
    rom[7] = flags7;
    rom[10] = flags10;
    // PRG payload: prg_banks_lo * 16 KB of zeros.
    rom.resize(
        16 + (prg_banks_lo as usize) * 16 * 1024
            + (chr_banks_lo as usize) * 8 * 1024,
        0,
    );
    rom
}

#[test]
fn nes20_zelda_style_battery_8k_only() {
    // 8 × 16 KB = 128 KB PRG, no CHR ROM, MMC1 (mapper 1), battery flag,
    // flags7 = 0x08 marks NES 2.0, flags10 = 0x70 = no volatile + 8 KB battery.
    let rom = synth_nes20_rom(8, 0, /*flags6*/ 0x12, /*flags7*/ 0x08, /*flags10*/ 0x70);
    let cart = Cartridge::load(&mut Cursor::new(rom)).expect("cart loads");
    // PRG-RAM total should be 8192 bytes so MMC1 has backing at $6000-$7FFF.
    assert_eq!(
        cart.prg_ram.len(),
        8192,
        "zelda.nes-style NES 2.0 header (flags10=0x70) must allocate 8 KB PRG-RAM",
    );
    assert!(cart.is_battery_backed, "battery flag must propagate");
}

#[test]
fn nes20_volatile_only_no_battery() {
    // flags10 = 0x07 = 8 KB volatile, no battery. flags6 = 0x10 (no battery).
    let rom = synth_nes20_rom(1, 0, /*flags6*/ 0x10, /*flags7*/ 0x08, /*flags10*/ 0x07);
    let cart = Cartridge::load(&mut Cursor::new(rom)).expect("cart loads");
    assert_eq!(cart.prg_ram.len(), 8192);
    assert!(!cart.is_battery_backed);
}

#[test]
fn nes20_both_volatile_and_battery() {
    // flags10 = 0x77 = 8 KB + 8 KB. Rare but legal.
    let rom = synth_nes20_rom(1, 0, /*flags6*/ 0x12, /*flags7*/ 0x08, /*flags10*/ 0x77);
    let cart = Cartridge::load(&mut Cursor::new(rom)).expect("cart loads");
    assert_eq!(
        cart.prg_ram.len(),
        16384,
        "sum of volatile+battery PRG-RAM sizes",
    );
}

#[test]
fn nes20_zero_prg_ram() {
    // flags10 = 0x00 = no PRG-RAM at all.
    let rom = synth_nes20_rom(1, 1, /*flags6*/ 0x10, /*flags7*/ 0x08, /*flags10*/ 0x00);
    let cart = Cartridge::load(&mut Cursor::new(rom)).expect("cart loads");
    assert_eq!(cart.prg_ram.len(), 0);
}

#[test]
fn ines_1_legacy_prg_ram_sizing_preserved() {
    // iNES 1.0: flags7 bit 3 = 0 (header bytes 8-15 are free).
    // flags8 in iNES 1.0 = PRG-RAM size in 8 KB banks; 0 legacy-means-8 KB.
    let mut rom = vec![0u8; 16];
    rom[0] = b'N';
    rom[1] = b'E';
    rom[2] = b'S';
    rom[3] = 0x1A;
    rom[4] = 1; // 16 KB PRG
    rom[5] = 1; // 8 KB CHR
    rom[6] = 0x12; // battery + mapper-LSB=1
    rom[7] = 0x00; // iNES 1.0
    rom[8] = 0x00; // legacy → 8 KB
    rom.resize(16 + 16 * 1024 + 8 * 1024, 0);
    let cart = Cartridge::load(&mut Cursor::new(rom)).expect("cart loads");
    assert_eq!(cart.prg_ram.len(), 8192);
}
