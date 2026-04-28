use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 34 is shared by two distinct boards:
//   * BNROM (Deadly Towers, homebrew): $8000-$FFFF writes select a
//     32 KB PRG bank. CHR is a single fixed 8 KB bank (CHR-RAM or
//     a single 8 KB CHR-ROM).
//   * NINA-001 (some homebrew): $7FFD selects PRG bank, $7FFE/$7FFF
//     select two 4 KB CHR banks.
//
// Detection heuristic: header reports 0 or 1 CHR banks -> BNROM.
// Otherwise -> NINA-001.
pub struct Mapper34 {
    cartridge: Cartridge,
    is_nina: bool,
    prg_bank: u8,
    chr_bank_lo: u8,
    chr_bank_hi: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub is_nina: bool,
    pub prg_bank: u8,
    pub chr_bank_lo: u8,
    pub chr_bank_hi: u8,
}

impl Mapper34 {
    pub fn new(cartridge: Cartridge) -> Self {
        let is_nina = cartridge.chr_num_banks >= 2;
        let mut m = Mapper34 {
            cartridge,
            is_nina,
            prg_bank: 0,
            chr_bank_lo: 0,
            chr_bank_hi: 1,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn chr_4k_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x1000).max(1)
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

impl Mapper for Mapper34 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if self.is_nina {
            match address {
                0x7FFD => {
                    self.prg_bank = value;
                    self.rebuild_asm_window();
                }
                0x7FFE => {
                    self.chr_bank_lo = value;
                }
                0x7FFF => {
                    self.chr_bank_hi = value;
                }
                _ => {}
            }
        } else if address >= 0x8000 {
            // BNROM: any value, mask by PRG bank count.
            self.prg_bank = value;
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        if self.is_nina {
            let bank_count = self.chr_4k_banks_count();
            let bank = if address < 0x1000 {
                (self.chr_bank_lo as usize) % bank_count
            } else {
                (self.chr_bank_hi as usize) % bank_count
            };
            let off = bank * 0x1000 | (address as usize & 0x0FFF);
            if off < self.cartridge.chr.len() {
                self.cartridge.chr[off]
            } else {
                0
            }
        } else if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        if self.is_nina {
            let bank_count = self.chr_4k_banks_count();
            let bank = if address < 0x1000 {
                (self.chr_bank_lo as usize) % bank_count
            } else {
                (self.chr_bank_hi as usize) % bank_count
            };
            let off = bank * 0x1000 | (address as usize & 0x0FFF);
            if off < self.cartridge.chr.len() {
                self.cartridge.chr[off] = value;
            }
        } else if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank_lo = 0;
        self.chr_bank_hi = 1;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State34(State {
            cartridge: self.cartridge.get_state(),
            is_nina: self.is_nina,
            prg_bank: self.prg_bank,
            chr_bank_lo: self.chr_bank_lo,
            chr_bank_hi: self.chr_bank_hi,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State34(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.is_nina = state.is_nina;
                self.prg_bank = state.prg_bank;
                self.chr_bank_lo = state.chr_bank_lo;
                self.chr_bank_hi = state.chr_bank_hi;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
