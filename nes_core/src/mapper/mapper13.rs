use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// CPROM (Videomation): fixed 32 KB PRG at $8000-$FFFF. CHR is 16 KB of
// on-cart CHR-RAM split into two 4 KB banks. Lower 4 KB ($0000-$0FFF)
// is fixed to CHR-RAM bank 0; upper 4 KB ($1000-$1FFF) is switchable
// (value & 0x03) via any write to $8000-$FFFF.
const CHR_RAM_SIZE: usize = 16 * 1024;

pub struct Mapper13 {
    cartridge: Cartridge,
    chr_upper_bank: u8,
    chr_ram: Vec<u8>,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub chr_upper_bank: u8,
    #[serde(with = "serde_bytes")]
    pub chr_ram: Vec<u8>,
}

impl Mapper13 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper13 {
            cartridge,
            chr_upper_bank: 0,
            chr_ram: vec![0u8; CHR_RAM_SIZE],
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn rebuild_asm_window(&mut self) {
        let prg = &self.cartridge.prg_rom;
        let len = prg.len();
        if len == 0 {
            return;
        }
        // CPROM has a single fixed 32 KB PRG bank. If the cart is only
        // 16 KB, mirror it across the 32 KB window.
        for i in 0..0x8000 {
            self.prg_asm_window[i] = prg[i & (len - 1)];
        }
    }

    fn chr_address(&self, address: u16) -> usize {
        // Lower 4 KB fixed to bank 0, upper 4 KB switchable.
        let addr = address as usize & 0x1FFF;
        if addr < 0x1000 {
            addr
        } else {
            let bank = (self.chr_upper_bank as usize) & 0x03;
            (bank * 0x1000) | (addr & 0x0FFF)
        }
    }
}

impl Mapper for Mapper13 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let len = self.cartridge.prg_rom.len();
        if len == 0 {
            return 0;
        }
        self.cartridge.prg_rom[(address - 0x8000) as usize & (len - 1)]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            self.chr_upper_bank = value & 0x03;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let off = self.chr_address(address) & (CHR_RAM_SIZE - 1);
        self.chr_ram[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let off = self.chr_address(address) & (CHR_RAM_SIZE - 1);
        self.chr_ram[off] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.chr_upper_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State13(State {
            cartridge: self.cartridge.get_state(),
            chr_upper_bank: self.chr_upper_bank,
            chr_ram: self.chr_ram.clone(),
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State13(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.chr_upper_bank = state.chr_upper_bank;
                if state.chr_ram.len() == self.chr_ram.len() {
                    self.chr_ram.copy_from_slice(&state.chr_ram);
                }
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
