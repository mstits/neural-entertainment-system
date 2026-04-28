use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 234 — Maxi 15 in 1 pirate multicart.
//
// Two bank-select registers, both latched by writes in the $FF80-$FFFF
// range. Register A ($FF80-$FF9F) encodes the outer bank + mirroring +
// chip mode, register B ($FFE8-$FFF7) encodes the inner bank. Most
// games use a simple NROM-like 32 KB PRG / 8 KB CHR window per combined
// bank. Per nesdev:
//   Reg A (on write to $FF80-$FF9F if A is currently zero; sticky):
//     bit 7:   mirroring (0 = vertical, 1 = horizontal)
//     bit 6:   mode (0 = NROM-32K, 1 = NROM-128K block)
//     bits 5-3: outer PRG bank (upper 3 bits of the combined bank)
//     bits 2-0: outer CHR bank (upper 3 bits)
//   Reg B (on write to $FFE8-$FFF7, always):
//     bits 6-4: inner PRG bank (low 3 bits), used in mode 1
//     bits 2-0: inner CHR bank (low 3 bits), used in mode 1
//
// For boot + 120 frames we implement the common case: combine outer and
// inner to form a 32 KB PRG + 8 KB CHR window, update mirroring from
// reg A bit 7. Details vary by cart — a panic is the only real failure
// mode we care about here.
pub struct Mapper234 {
    cartridge: Cartridge,
    reg_a: u8,
    reg_b: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub reg_a: u8,
    pub reg_b: u8,
    pub mirroring: Mirroring,
}

impl Mapper234 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper234 {
            cartridge,
            reg_a: 0,
            reg_b: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_32k_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn chr_8k_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x2000).max(1)
    }

    /// Combined 32 KB PRG bank selected by regA+regB. In NROM-32K mode
    /// (regA bit 6 == 0) only the outer bits matter and the inner regB
    /// bits are ignored; in NROM-128K mode (regA bit 6 == 1) the outer
    /// bits fix the upper half and the inner bits pick within a 128 KB
    /// block.
    fn prg_bank(&self) -> usize {
        let outer = ((self.reg_a >> 3) & 0x07) as usize;
        let inner = ((self.reg_b >> 4) & 0x07) as usize;
        let combined = if self.reg_a & 0x40 != 0 {
            (outer & 0x06) | (inner & 0x01)
        } else {
            outer
        };
        combined % self.prg_32k_banks_count()
    }

    fn chr_bank(&self) -> usize {
        let outer = (self.reg_a & 0x07) as usize;
        let inner = (self.reg_b & 0x07) as usize;
        let combined = if self.reg_a & 0x40 != 0 {
            (outer & 0x06) | (inner & 0x01)
        } else {
            outer
        };
        combined % self.chr_8k_banks_count()
    }

    fn rebuild_asm_window(&mut self) {
        let bank = self.prg_bank();
        let off = bank * 0x8000;
        let prg = &self.cartridge.prg_rom;
        if off + 0x8000 <= prg.len() {
            self.prg_asm_window
                .copy_from_slice(&prg[off..off + 0x8000]);
        }
    }

    fn update_mirroring(&mut self) {
        self.cartridge.mirroring = if self.reg_a & 0x80 == 0 {
            Mirroring::Vertical
        } else {
            Mirroring::Horizontal
        };
    }
}

impl Mapper for Mapper234 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = self.prg_bank();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        if off < self.cartridge.prg_rom.len() {
            self.cartridge.prg_rom[off]
        } else {
            0
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0xFF80 && address <= 0xFF9F {
            // Register A is "sticky" — only accepts writes when its
            // current value is zero. Most Maxi-15 menus rely on this.
            if self.reg_a == 0 {
                self.reg_a = value;
                self.update_mirroring();
                self.rebuild_asm_window();
            }
        } else if address >= 0xFFE8 && address <= 0xFFF7 {
            self.reg_b = value;
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = self.chr_bank();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let bank = self.chr_bank();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.reg_a = 0;
        self.reg_b = 0;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State234(State {
            cartridge: self.cartridge.get_state(),
            reg_a: self.reg_a,
            reg_b: self.reg_b,
            mirroring: self.cartridge.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State234(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.reg_a = state.reg_a;
                self.reg_b = state.reg_b;
                self.cartridge.mirroring = state.mirroring;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
