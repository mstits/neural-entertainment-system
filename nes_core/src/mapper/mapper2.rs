use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};
use serde_derive::{Deserialize, Serialize};

pub struct Mapper2 {
    cartridge: Cartridge,
    switchable_bank: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub switchable_bank: u8,
}

impl Mapper2 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper2 {
            cartridge,
            switchable_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn rebuild_asm_window(&mut self) {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let low_bank = self.switchable_bank;
        let high_bank = self.cartridge.prg_rom_num_banks.saturating_sub(1);
        let prg = &self.cartridge.prg_rom;
        let low_off = (low_bank as usize) * bank_size;
        let high_off = (high_bank as usize) * bank_size;
        if low_off + bank_size <= prg.len() {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[low_off..low_off + bank_size]);
        }
        if high_off + bank_size <= prg.len() {
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[high_off..high_off + bank_size]);
        }
    }

    fn prg_rom_address(bank: u8, address: u16) -> usize {
        (bank as usize * PRG_ROM_BANK_SIZE as usize)
            | (address as usize & (PRG_ROM_BANK_SIZE as usize - 1))
    }

    fn read_prg_rom(&self, address: u16) -> u8 {
        let bank = if address < 0xC000 {
            self.switchable_bank
        } else {
            self.cartridge.prg_rom_num_banks - 1
        };

        let rom_addr = Mapper2::prg_rom_address(bank, address);
        self.cartridge.prg_rom[rom_addr]
    }
}

impl Mapper for Mapper2 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else {
            self.read_prg_rom(address)
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            self.switchable_bank =
                ((value as usize) % (self.cartridge.prg_rom_num_banks) as usize) as u8;
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
        self.switchable_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 { 1 }

    fn get_state(&self) -> mapper::State {
        mapper::State::State2(State {
            cartridge: self.cartridge.get_state(),
            switchable_bank: self.switchable_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State2(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.switchable_bank = state.switchable_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
