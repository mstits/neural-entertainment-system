use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// NINA-06 / HES Multicart (mapper 113).
// Single register latched on any write in $4100-$7FFF.
//   bit 7:     mirroring (0 = horizontal, 1 = vertical)
//   bit 6:     CHR bank high bit (A17)
//   bits 5-3:  PRG bank (8 possible 32 KB banks)
//   bits 2-0:  CHR bank low bits (A14-A16)
//
// PRG bank index = (value >> 3) & 0x07
// CHR bank index = ((value >> 3) & 0x08) | (value & 0x07)   // 4-bit, 16 banks
pub struct Mapper113 {
    cartridge: Cartridge,
    prg_bank: u8,
    chr_bank: u8,
    mirroring: Mirroring,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub chr_bank: u8,
    pub mirroring: Mirroring,
}

impl Mapper113 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mirroring = cartridge.mirroring;
        let mut m = Mapper113 {
            cartridge,
            prg_bank: 0,
            chr_bank: 0,
            mirroring,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn chr_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x2000).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000;
        let prg = &self.cartridge.prg_rom;
        if off + 0x8000 <= prg.len() {
            self.prg_asm_window.copy_from_slice(&prg[off..off + 0x8000]);
        }
    }
}

impl Mapper for Mapper113 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let prg = &self.cartridge.prg_rom;
        if prg.is_empty() {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        prg[off & (prg.len() - 1)]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x4100..=0x7FFF).contains(&address) {
            self.prg_bank = (value >> 3) & 0x07;
            self.chr_bank = ((value >> 3) & 0x08) | (value & 0x07);
            self.mirroring = if value & 0x80 != 0 {
                Mirroring::Vertical
            } else {
                Mirroring::Horizontal
            };
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr = &self.cartridge.chr;
        if chr.is_empty() {
            return 0;
        }
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = ((bank * 0x2000) | (address as usize & 0x1FFF)) & (chr.len() - 1);
        chr[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_len = self.cartridge.chr.len();
        if chr_len == 0 {
            return;
        }
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = ((bank * 0x2000) | (address as usize & 0x1FFF)) & (chr_len - 1);
        self.cartridge.chr[off] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank = 0;
        self.mirroring = self.cartridge.mirroring;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State113(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
            mirroring: self.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State113(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank = state.chr_bank;
                self.mirroring = state.mirroring;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
