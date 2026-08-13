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

    fn prg_32k_bank_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let bank = (self.prg_rom_bank as usize) % self.prg_32k_bank_count();
        let bank_off = bank * 0x8000;
        let prg = &self.cartridge.prg_rom;
        if bank_off + 0x8000 <= prg.len() {
            self.prg_asm_window.copy_from_slice(&prg[bank_off..bank_off + 0x8000]);
        }
    }

    fn read_prg_rom(&self, address: u16) -> u8 {
        let bank = (self.prg_rom_bank as usize) % self.prg_32k_bank_count();
        let rom_addr = (bank * 0x8000) | (address as usize & 0x7FFF);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 7,
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

    // Stamp byte 0 of every 32 KB PRG bank with a distinct marker.
    fn stamp_prg32_banks(cart: &mut Cartridge) {
        let banks32 = cart.prg_rom.len() / 0x8000;
        for b in 0..banks32 {
            cart.prg_rom[b * 0x8000] = 0xA0 + b as u8;
        }
    }

    // AxROM swaps the entire 32 KB $8000-$FFFF window via bits 0-2.
    #[test]
    fn axrom_32k_bank_switch() {
        let mut cart = test_cart(16, 1); // 256 KB = 8 x 32 KB banks
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 0);
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        m.prg_write_byte(0x8000, 5);
        assert_eq!(m.prg_read_byte(0x8000), 0xA5);
        m.prg_write_byte(0x8000, 7);
        assert_eq!(m.prg_read_byte(0x8000), 0xA7);
    }

    // Only the low 3 bits select the bank; the mirror bit does not shift it.
    #[test]
    fn axrom_bank_uses_low_three_bits() {
        let mut cart = test_cart(16, 1);
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        // 0x15 = 0b10101 => bank bits = 5, mirror bit (0x10) also set
        m.prg_write_byte(0x8000, 0x15);
        assert_eq!(m.prg_read_byte(0x8000), 0xA5);
    }

    // On a cart smaller than the 3-bit bank select can address, an in-range
    // bank still maps correctly.
    #[test]
    fn axrom_small_rom_in_range_bank() {
        let mut cart = test_cart(8, 1); // 128 KB = 4 x 32 KB banks
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 3); // highest in-range bank
        assert_eq!(m.prg_read_byte(0x8000), 0xA3);
    }

    // On a cart smaller than 256 KB, a bank select whose value exceeds the
    // real bank count wraps modulo that count instead of indexing past the
    // ROM. Before the mask, read_prg_rom computed bank * 0x8000 straight
    // into prg_rom and this panicked out of bounds.
    #[test]
    fn axrom_small_rom_bank_wraps() {
        let mut cart = test_cart(8, 1); // 128 KB = 4 x 32 KB banks
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 6); // 6 & 0x07 == 6, 6 % 4 == 2
        assert_eq!(m.prg_read_byte(0x8000), 0xA2);
        m.prg_write_byte(0x8000, 5); // 5 % 4 == 1
        assert_eq!(m.prg_read_byte(0x8000), 0xA1);
    }

    // Bit 4 clear selects the single-screen lower nametable.
    #[test]
    fn axrom_mirroring_lower() {
        let cart = test_cart(16, 1);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 0x00);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
    }

    // Bit 4 set selects the single-screen upper nametable.
    #[test]
    fn axrom_mirroring_upper() {
        let cart = test_cart(16, 1);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 0x10);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }

    // CHR-RAM round-trips writes.
    #[test]
    fn axrom_chr_read_write() {
        let cart = test_cart(16, 1);
        let mut m = Mapper7::new(cart);
        m.chr_write_byte(0x0000, 0x71);
        m.chr_write_byte(0x1FFF, 0x72);
        assert_eq!(m.chr_read_byte(0x0000), 0x71);
        assert_eq!(m.chr_read_byte(0x1FFF), 0x72);
    }

    // reset() restores bank 0 and the cartridge's default mirroring.
    #[test]
    fn axrom_reset_restores_defaults() {
        let mut cart = test_cart(16, 1);
        cart.default_mirroring = Mirroring::OneScreenLower;
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 0x13); // bank 3, upper mirror
        assert_eq!(m.prg_read_byte(0x8000), 0xA3);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
    }

    // State snapshot preserves bank + mirroring across a later write.
    #[test]
    fn axrom_state_round_trip() {
        let mut cart = test_cart(16, 1);
        stamp_prg32_banks(&mut cart);
        let mut m = Mapper7::new(cart);
        m.prg_write_byte(0x8000, 0x14); // bank 4, upper mirror
        let snap = m.get_state();
        m.prg_write_byte(0x8000, 0x01); // bank 1, lower mirror
        assert_eq!(m.prg_read_byte(0x8000), 0xA1);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 0xA4);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }
}
