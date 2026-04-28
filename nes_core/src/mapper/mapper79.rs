use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// iNES Mapper 79 — Nina-003 / NAMCOT-00301 (AVE / C&E discrete logic).
// Single register at $4100-$41FF selects a 32 KB PRG bank (bit 3) and
// an 8 KB CHR bank (bits 0-2). PRG window is fixed at $8000-$FFFF.
// Mirroring is fixed from the iNES header. Writes to $8000-$FFFF are
// ignored (PRG is ROM).
pub struct Mapper79 {
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

impl Mapper79 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper79 {
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

impl Mapper for Mapper79 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        // Register is mirrored across $4100-$41FF. The decoder asserts
        // when A13 is high and A8 is high within the $4000-$5FFF range,
        // which matches `address & 0xE100 == 0x4100`.
        if (address & 0xE100) == 0x4100 {
            self.prg_bank = (value >> 3) & 0x01;
            self.chr_bank = value & 0x07;
            self.rebuild_asm_window();
        }
        // Writes to $8000-$FFFF are ignored (PRG ROM).
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
        mapper::State::State79(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State79(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank = state.chr_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
