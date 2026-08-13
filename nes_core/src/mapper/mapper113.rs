use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// NINA-06 / HES Multicart (mapper 113).
// Single register latched on any write in $4100-$7FFF.
//   bit 7:     mirroring (0 = horizontal, 1 = vertical)
//   bit 6:     CHR bank high bit (A17)
//   bits 5-3:  PRG bank (8 possible 32 KB banks)
//   bits 2-0:  CHR bank low bits (A14-A16)
//
// PRG bank index = (value >> 3) & 0x07
// CHR bank index = ((value >> 3) & 0x08) | (value & 0x07)   // 4-bit, 16 banks
pub struct Mapper113 {
    cartridge: Cartridge,
    prg_bank: u8,
    chr_bank: u8,
    mirroring: Mirroring,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub chr_bank: u8,
    pub mirroring: Mirroring,
}

impl Mapper113 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mirroring = cartridge.mirroring;
        let mut m = Mapper113 {
            cartridge,
            prg_bank: 0,
            chr_bank: 0,
            mirroring,
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

impl Mapper for Mapper113 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let prg = &self.cartridge.prg_rom;
        if prg.is_empty() {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        prg[off & (prg.len() - 1)]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x4100..=0x7FFF).contains(&address) {
            self.prg_bank = (value >> 3) & 0x07;
            self.chr_bank = ((value >> 3) & 0x08) | (value & 0x07);
            self.mirroring = if value & 0x80 != 0 {
                Mirroring::Vertical
            } else {
                Mirroring::Horizontal
            };
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr = &self.cartridge.chr;
        if chr.is_empty() {
            return 0;
        }
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = ((bank * 0x2000) | (address as usize & 0x1FFF)) & (chr.len() - 1);
        chr[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_len = self.cartridge.chr.len();
        if chr_len == 0 {
            return;
        }
        let bank = (self.chr_bank as usize) % self.chr_banks_count();
        let off = ((bank * 0x2000) | (address as usize & 0x1FFF)) & (chr_len - 1);
        self.cartridge.chr[off] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank = 0;
        self.mirroring = self.cartridge.mirroring;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State113(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
            mirroring: self.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State113(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank = state.chr_bank;
                self.mirroring = state.mirroring;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{Cartridge, Mirroring};
    use crate::mapper::Mapper;

    // Any address in $4100-$7FFF latches the single control register.
    const LATCH: u16 = 0x4100;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 113,
            sub_mapper: 0,
            mirroring: Mirroring::Horizontal,
            default_mirroring: Mirroring::Horizontal,
            prg_rom_num_banks: prg_16k_banks,
            prg_rom: vec![0u8; prg_16k_banks as usize * 16 * 1024],
            chr_num_banks: chr_8k_banks,
            chr: vec![0u8; chr_8k_banks as usize * 8 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    // Stamp the first byte of every 32 KB PRG bank and every 8 KB CHR
    // bank with that bank's index. Sizes are powers of two so the
    // mapper's `& (len - 1)` addressing stays clean.
    fn stamped_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        let mut c = test_cart(prg_16k_banks, chr_8k_banks);
        let prg_banks = c.prg_rom.len() / 0x8000;
        for b in 0..prg_banks {
            c.prg_rom[b * 0x8000] = b as u8;
        }
        let chr_banks = c.chr.len() / 0x2000;
        for b in 0..chr_banks {
            c.chr[b * 0x2000] = b as u8;
        }
        c
    }

    // ---- PRG banking (bits 5-3 select a 32 KB bank) ----

    #[test]
    fn prg_bank_switch() {
        let mut m = Mapper113::new(stamped_cart(16, 16)); // 8 x 32 KB
        m.prg_write_byte(LATCH, 3 << 3); // PRG bank 3
        assert_eq!(m.prg_read_byte(0x8000), 3);
    }

    // A PRG bank index beyond the ROM wraps mod the bank count.
    #[test]
    fn prg_bank_wraps() {
        let mut m = Mapper113::new(stamped_cart(4, 16)); // 2 x 32 KB
        m.prg_write_byte(LATCH, 3 << 3); // 3 % 2 == 1
        assert_eq!(m.prg_read_byte(0x8000), (3 % 2) as u8);
    }

    // $8000 and $C000 read the low and high 16 KB of the same 32 KB bank.
    #[test]
    fn prg_offset_within_bank() {
        let mut c = test_cart(16, 16);
        // Marker at the very top of bank 3 (offset 0x7FFF within the bank).
        c.prg_rom[3 * 0x8000 + 0x7FFF] = 0x99;
        let mut m = Mapper113::new(c);
        m.prg_write_byte(LATCH, 3 << 3);
        assert_eq!(m.prg_read_byte(0xFFFF), 0x99);
    }

    // ---- CHR banking (bit 6 is the high bit, bits 2-0 the low bits) ----

    #[test]
    fn chr_low_bits_only() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        m.prg_write_byte(LATCH, 0x05); // chr bank = 5
        assert_eq!(m.chr_read_byte(0x0000), 5);
    }

    // Bit 6 supplies CHR bank bit 3, reaching banks 8-15.
    #[test]
    fn chr_high_bit_selects_upper_banks() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        // value 0x42: bit6 set -> +8, low bits = 2 => chr bank 10.
        m.prg_write_byte(LATCH, 0x42);
        assert_eq!(m.chr_read_byte(0x0000), 10);
    }

    // CHR bank index wraps mod the CHR bank count.
    #[test]
    fn chr_bank_wraps() {
        let mut m = Mapper113::new(stamped_cart(16, 4)); // 4 x 8 KB
        m.prg_write_byte(LATCH, 0x42); // chr bank 10 -> 10 % 4 == 2
        assert_eq!(m.chr_read_byte(0x0000), (10 % 4) as u8);
    }

    // A single register write updates PRG and CHR banks together.
    #[test]
    fn one_write_sets_prg_and_chr() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        let v = (2 << 3) | 4; // PRG bits -> 2, CHR low bits -> 4
        m.prg_write_byte(LATCH, v);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        assert_eq!(m.chr_read_byte(0x0000), 4);
    }

    // CHR RAM writes hit the currently-selected 8 KB bank.
    #[test]
    fn chr_write_read_roundtrip() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        m.prg_write_byte(LATCH, 0x05); // chr bank 5
        m.chr_write_byte(0x0010, 0xEE);
        assert_eq!(m.chr_read_byte(0x0010), 0xEE);
        // A different bank does not see the write.
        m.prg_write_byte(LATCH, 0x06); // chr bank 6
        assert_ne!(m.chr_read_byte(0x0010), 0xEE);
    }

    // ---- Mirroring (bit 7) ----

    #[test]
    fn mirroring_bit7() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(LATCH, 0x80); // bit 7 -> vertical
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(LATCH, 0x00);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }

    // ---- Address decoding of the latch window ----

    // Writes below $4100 and above $7FFF must not touch the register.
    #[test]
    fn writes_outside_window_ignored() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        m.prg_write_byte(LATCH, 3 << 3); // PRG bank 3
        assert_eq!(m.prg_read_byte(0x8000), 3);
        m.prg_write_byte(0x40FF, 0); // just below the window
        assert_eq!(m.prg_read_byte(0x8000), 3);
        m.prg_write_byte(0x8000, 0); // above the window (PRG ROM space)
        assert_eq!(m.prg_read_byte(0x8000), 3);
    }

    // Both ends of the latch window ($4100 and $7FFF) are decoded.
    #[test]
    fn latch_window_boundaries() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        m.prg_write_byte(0x4100, 1 << 3);
        assert_eq!(m.prg_read_byte(0x8000), 1);
        m.prg_write_byte(0x7FFF, 2 << 3);
        assert_eq!(m.prg_read_byte(0x8000), 2);
    }

    // ---- State + reset ----

    #[test]
    fn state_round_trip() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        // 0xDA: PRG bank 3, CHR bank 10, vertical mirroring.
        m.prg_write_byte(LATCH, 0xDA);
        assert_eq!(m.prg_read_byte(0x8000), 3);
        assert_eq!(m.chr_read_byte(0x0000), 10);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        let snap = m.get_state();

        m.prg_write_byte(LATCH, 0x00); // back to bank 0, horizontal
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);

        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 3);
        assert_eq!(m.chr_read_byte(0x0000), 10);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    #[test]
    fn reset_restores_power_on() {
        let mut m = Mapper113::new(stamped_cart(16, 16));
        m.prg_write_byte(LATCH, 0xDA); // PRG 3, CHR 10, vertical
        assert_eq!(m.prg_read_byte(0x8000), 3);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.chr_read_byte(0x0000), 0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }
}
