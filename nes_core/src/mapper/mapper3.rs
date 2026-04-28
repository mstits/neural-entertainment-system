use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper3 {
    cartridge: Cartridge,
    chr_bank: u8,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub chr_bank: u8,
}

impl Mapper3 {
    pub fn new(cartridge: Cartridge) -> Self {
        Mapper3 {
            cartridge,
            chr_bank: 0,
        }
    }

    fn chr_address(&self, bank: u8, address: u16) -> usize {
        let chr_len = self.cartridge.chr.len();
        if chr_len == 0 {
            return 0;
        }
        let raw = ((bank as usize) * 0x2000) | (address as usize & 0x1FFF);
        raw & (chr_len - 1)
    }
}

impl Mapper for Mapper3 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            0
        } else {
            // 16 KB CNROM carts mirror the single PRG bank across both
            // halves of the $8000-$FFFF window (Joust, Legend of Kage).
            // The `& (len-1)` mask handles both 16 KB (mirrored) and
            // 32 KB (full) carts since PRG sizes are powers of two.
            let len = self.cartridge.prg_rom.len();
            if len == 0 { return 0; }
            self.cartridge.prg_rom[(address - 0x8000) as usize & (len - 1)]
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            // Bank-register value indexes 8 KB CHR banks, not PRG.
            // The original code divided by PRG size which masked
            // incorrectly on carts with unequal PRG/CHR sizes
            // (Legend of Kage: 32 KB PRG / 16 KB CHR → bank 3 was
            // reachable even though only 2 CHR banks exist). Store
            // the raw value; chr_address masks at fetch time.
            self.chr_bank = value;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let rom_addr = self.chr_address(self.chr_bank, address);
        self.cartridge.chr[rom_addr]
    }

    fn chr_write_byte(&mut self, _address: u16, _value: u8) {
        // Mapper 3 (CNROM) technically has read-only CHR ROM, but a
        // handful of commercial carts and most homebrew use CNROM-
        // shaped headers with CHR-RAM behaviour. Panicking here takes
        // down the whole trainer on a single buggy write — silently
        // ignore instead, matching behaviour of other read-only CHR
        // mappers.
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.chr_bank = 0;
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State3(State {
            cartridge: self.cartridge.get_state(),
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State3(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.chr_bank = state.chr_bank;
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
