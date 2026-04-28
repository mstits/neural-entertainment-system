use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Color Dreams: single 8-bit register latched on any write to $8000-$FFFF.
// Low nibble = 32 KB PRG bank, high nibble = 8 KB CHR bank.
pub struct Mapper11 {
    cartridge: Cartridge,
    prg_bank: u8,
    chr_bank: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub chr_bank: u8,
}

impl Mapper11 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper11 {
            cartridge,
            prg_bank: 0,
            chr_bank: 0,
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

impl Mapper for Mapper11 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            self.prg_bank = value & 0x0F;
            self.chr_bank = (value >> 4) & 0x0F;
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        self.cartridge.chr[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State11(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State11(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank = state.chr_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
