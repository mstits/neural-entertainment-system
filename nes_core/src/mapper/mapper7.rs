use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper7 {
    cartridge: Cartridge,
    prg_rom_bank: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_rom_bank: u8,
}

impl Mapper7 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper7 {
            cartridge,
            prg_rom_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn rebuild_asm_window(&mut self) {
        let bank_off = (self.prg_rom_bank as usize) * 0x8000;
        let prg = &self.cartridge.prg_rom;
        if bank_off + 0x8000 <= prg.len() {
            self.prg_asm_window.copy_from_slice(&prg[bank_off..bank_off + 0x8000]);
        }
    }

    fn prg_rom_address(bank: u8, address: u16) -> usize {
        (bank as usize * 0x8000) | (address as usize & 0x7FFF)
    }

    fn read_prg_rom(&self, address: u16) -> u8 {
        let rom_addr = Mapper7::prg_rom_address(self.prg_rom_bank, address);
        self.cartridge.prg_rom[rom_addr]
    }
}

impl Mapper for Mapper7 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else if address < 0x8000 {
            if self.cartridge.prg_ram.is_empty() {
                0
            } else {
                let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx]
            }
        } else {
            self.read_prg_rom(address)
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) && !self.cartridge.prg_ram.is_empty() {
            let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
            self.cartridge.prg_ram[idx] = value;
        } else if address >= 0x8000 {
            self.prg_rom_bank = value & 0x07;
            self.cartridge.mirroring = if value & 0x10 == 0 {
                Mirroring::OneScreenLower
            } else {
                Mirroring::OneScreenUpper
            };
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        self.cartridge.chr[address as usize]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        self.cartridge.chr[address as usize] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.prg_rom_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 { 4 }

    fn get_state(&self) -> mapper::State {
        mapper::State::State7(State {
            cartridge: self.cartridge.get_state(),
            prg_rom_bank: self.prg_rom_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State7(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_rom_bank = state.prg_rom_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
