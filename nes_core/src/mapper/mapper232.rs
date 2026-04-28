use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 232 — Camerica BF9096 (Quattro Adventure, Quattro Arcade,
// Quattro Sports). 256 KB PRG split into four 64 KB blocks; each
// block holds four 16 KB banks. Lower window ($8000-$BFFF) is a
// switchable bank within the current block; upper window
// ($C000-$FFFF) is fixed to the last bank of the current block.
// CHR is 8 KB of CHR-RAM, no switching.
pub struct Mapper232 {
    cartridge: Cartridge,
    block: u8,       // bits 4-3 of the value written to $8000-$BFFF
    inner_bank: u8,  // bits 1-0 of the value written to $C000-$FFFF
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub block: u8,
    pub inner_bank: u8,
}

impl Mapper232 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper232 {
            cartridge,
            block: 0,
            inner_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_16k_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x4000).max(1)
    }

    fn low_bank(&self) -> usize {
        let bank = ((self.block as usize) << 2) | (self.inner_bank as usize);
        bank % self.prg_16k_banks_count()
    }

    fn high_bank(&self) -> usize {
        let bank = ((self.block as usize) << 2) | 0x03;
        bank % self.prg_16k_banks_count()
    }

    fn rebuild_asm_window(&mut self) {
        let prg = &self.cartridge.prg_rom;
        let low_off = self.low_bank() * 0x4000;
        let high_off = self.high_bank() * 0x4000;
        if low_off + 0x4000 <= prg.len() {
            self.prg_asm_window[..0x4000]
                .copy_from_slice(&prg[low_off..low_off + 0x4000]);
        }
        if high_off + 0x4000 <= prg.len() {
            self.prg_asm_window[0x4000..]
                .copy_from_slice(&prg[high_off..high_off + 0x4000]);
        }
    }
}

impl Mapper for Mapper232 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = if address < 0xC000 {
            self.low_bank()
        } else {
            self.high_bank()
        };
        let off = bank * 0x4000 | (address as usize & 0x3FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x8000..=0xBFFF => {
                // Block select (bits 4-3 of value).
                self.block = (value >> 3) & 0x03;
                self.rebuild_asm_window();
            }
            0xC000..=0xFFFF => {
                // Inner bank select (bits 1-0 of value).
                self.inner_bank = value & 0x03;
                self.rebuild_asm_window();
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.block = 0;
        self.inner_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State232(State {
            cartridge: self.cartridge.get_state(),
            block: self.block,
            inner_bank: self.inner_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State232(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.block = state.block;
                self.inner_bank = state.inner_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
