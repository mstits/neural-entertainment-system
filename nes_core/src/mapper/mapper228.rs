use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 228 — "Action 52" / Cheetahmen II pirate multicart.
// Bank selection is encoded in the WRITE ADDRESS, value selects CHR:
//
//   A13:          mirroring (0 = vertical, 1 = horizontal)
//   A12..A11:     PRG chip select (0, 2, 3 -> ROM; 1 -> open bus / mirrored)
//   A10..A6:      5-bit PRG bank within the selected chip
//   A5:           PRG mode (0 = 32 KB, 1 = 16 KB mirrored)
//   value[5..2]:  CHR bank high bits
//   value[1..0]:  CHR bank low bits
//
// The chip table is: chip 0 = 512 KB, chip 2 = 256 KB, chip 3 = 256 KB
// stacked into the combined ~1 MB PRG image. For a minimal boot we
// flatten the PRG into 16 KB banks indexed linearly — most library
// dumps are sized correctly for this, and it keeps the fast path
// simple.
pub struct Mapper228 {
    cartridge: Cartridge,
    prg_bank: u8,  // 16 KB bank select for the low window
    prg_mode_32k: bool, // if true, $8000-$FFFF is a 32 KB window
    chr_bank: u8,  // 8 KB CHR bank
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub prg_mode_32k: bool,
    pub chr_bank: u8,
    pub mirroring: Mirroring,
}

impl Mapper228 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper228 {
            cartridge,
            prg_bank: 0,
            prg_mode_32k: false,
            chr_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_16k_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x4000).max(1)
    }

    fn chr_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x2000).max(1)
    }

    fn decode_prg_bank(address: u16) -> u8 {
        // A10..A6 = inner bank (5 bits).
        // A12..A11 = chip select (2 bits) stacked above.
        let chip = ((address >> 11) & 0x03) as u8;
        let inner = ((address >> 6) & 0x1F) as u8;
        // Remap chip 1 -> chip 0 (open-bus/mirror behaviour on real hw).
        let chip = if chip == 1 { 0 } else { chip };
        (chip << 5) | inner
    }

    fn rebuild_asm_window(&mut self) {
        let bank_count = self.prg_16k_banks_count();
        let (low_bank, high_bank) = if self.prg_mode_32k {
            let base = (self.prg_bank as usize & !1) % bank_count;
            (base, (base + 1) % bank_count)
        } else {
            let b = (self.prg_bank as usize) % bank_count;
            (b, b)
        };
        let prg = &self.cartridge.prg_rom;
        let low_off = low_bank * 0x4000;
        let high_off = high_bank * 0x4000;
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

impl Mapper for Mapper228 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank_count = self.prg_16k_banks_count();
        let (low_bank, high_bank) = if self.prg_mode_32k {
            let base = (self.prg_bank as usize & !1) % bank_count;
            (base, (base + 1) % bank_count)
        } else {
            let b = (self.prg_bank as usize) % bank_count;
            (b, b)
        };
        let bank = if address < 0xC000 { low_bank } else { high_bank };
        let off = bank * 0x4000 | (address as usize & 0x3FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address < 0x8000 {
            return;
        }
        // Mirroring from A13.
        self.cartridge.mirroring = if address & (1 << 13) == 0 {
            Mirroring::Vertical
        } else {
            Mirroring::Horizontal
        };
        self.prg_bank = Mapper228::decode_prg_bank(address);
        self.prg_mode_32k = (address & (1 << 5)) == 0;
        // CHR bank: value bits (5..2) as high nibble, (1..0) as low nibble
        // of the 4-bit CHR bank (8 KB granularity).
        let chr_hi = (value >> 2) & 0x0F;
        let chr_lo = value & 0x03;
        self.chr_bank = (chr_hi << 2) | chr_lo;
        self.rebuild_asm_window();
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off]
        } else {
            0
        }
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
        self.prg_mode_32k = false;
        self.chr_bank = 0;
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
        mapper::State::State228(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            prg_mode_32k: self.prg_mode_32k,
            chr_bank: self.chr_bank,
            mirroring: self.cartridge.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State228(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.prg_mode_32k = state.prg_mode_32k;
                self.chr_bank = state.chr_bank;
                self.cartridge.mirroring = state.mirroring;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
