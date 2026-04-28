use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Camerica / Codemasters BF909x. Functionally a UxROM variant:
//   $C000-$FFFF writes → 16 KB PRG bank for the $8000-$BFFF window.
//   $8000-$BFFF fixed to the last 16 KB bank.
//   $9000-$9FFF writes (Fire Hawk only) → single-screen mirroring control.
pub struct Mapper71 {
    cartridge: Cartridge,
    switchable_bank: u8,
    // Fire Hawk uses $9000 to toggle single-screen mirroring. Other
    // Codemasters boards never write there, so treating this as a
    // latched mirroring control is safe for the whole family.
    fire_hawk_mirroring: bool,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub switchable_bank: u8,
    pub fire_hawk_mirroring: bool,
}

impl Mapper71 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper71 {
            cartridge,
            switchable_bank: 0,
            fire_hawk_mirroring: false,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_banks_count(&self) -> usize {
        self.cartridge.prg_rom_num_banks.max(1) as usize
    }

    fn rebuild_asm_window(&mut self) {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let low_bank = (self.switchable_bank as usize) % self.prg_banks_count();
        let high_bank = self.prg_banks_count().saturating_sub(1);
        let prg = &self.cartridge.prg_rom;
        let low_off = low_bank * bank_size;
        let high_off = high_bank * bank_size;
        if low_off + bank_size <= prg.len() {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[low_off..low_off + bank_size]);
        }
        if high_off + bank_size <= prg.len() {
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[high_off..high_off + bank_size]);
        }
    }
}

impl Mapper for Mapper71 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let bank = if address < 0xC000 {
            (self.switchable_bank as usize) % self.prg_banks_count()
        } else {
            self.prg_banks_count() - 1
        };
        self.cartridge.prg_rom[bank * bank_size | (address as usize & (bank_size - 1))]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x9000..=0x9FFF => {
                self.fire_hawk_mirroring = true;
                self.cartridge.mirroring = if value & 0x10 == 0 {
                    Mirroring::OneScreenLower
                } else {
                    Mirroring::OneScreenUpper
                };
            }
            0xC000..=0xFFFF | 0x8000..=0xBFFF => {
                self.switchable_bank = value & 0x0F;
                self.rebuild_asm_window();
            }
            _ => {}
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
        self.fire_hawk_mirroring = false;
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
        mapper::State::State71(State {
            cartridge: self.cartridge.get_state(),
            switchable_bank: self.switchable_bank,
            fire_hawk_mirroring: self.fire_hawk_mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State71(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.switchable_bank = state.switchable_bank;
                self.fire_hawk_mirroring = state.fire_hawk_mirroring;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
