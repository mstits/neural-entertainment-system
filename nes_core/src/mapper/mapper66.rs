use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// GxROM / MHROM: one latched register on writes to $8000-$FFFF.
// Bits 5-4 = 32 KB PRG bank, bits 1-0 = 8 KB CHR bank.
pub struct Mapper66 {
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

impl Mapper66 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper66 {
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

impl Mapper for Mapper66 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            self.prg_bank = (value >> 4) & 0x03;
            self.chr_bank = value & 0x03;
            self.rebuild_asm_window();
        }
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
        mapper::State::State66(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State66(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank = state.chr_bank;
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

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 66,
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

    // Stamp each 32 KB PRG bank and each 8 KB CHR bank with a marker byte
    // at offset 0. `prg_32k` = count of 32 KB banks, `chr_8k` = 8 KB banks.
    fn stamped_mapper(prg_32k: u8, chr_8k: u8) -> Mapper66 {
        let mut cart = test_cart(prg_32k * 2, chr_8k);
        for b in 0..prg_32k as usize {
            cart.prg_rom[b * 0x8000] = 0xA0 + b as u8;
        }
        for b in 0..chr_8k as usize {
            cart.chr[b * 0x2000] = 0xC0 + b as u8;
        }
        Mapper66::new(cart)
    }

    // Bits 5-4 of the latch select the 32 KB PRG bank.
    #[test]
    fn prg_high_bits_select_32k_bank() {
        let mut m = stamped_mapper(4, 1);
        for bank in 0..4u8 {
            m.prg_write_byte(0x8000, bank << 4);
            assert_eq!(m.prg_read_byte(0x8000), 0xA0 + bank);
        }
    }

    // Bits 1-0 of the latch select the 8 KB CHR bank.
    #[test]
    fn chr_low_bits_select_8k_bank() {
        let mut m = stamped_mapper(1, 4);
        for bank in 0..4u8 {
            m.prg_write_byte(0x8000, bank);
            assert_eq!(m.chr_read_byte(0x0000), 0xC0 + bank);
        }
    }

    // PRG (bits 5-4) and CHR (bits 1-0) are decoded independently from the
    // same latch write.
    #[test]
    fn single_write_sets_both_banks() {
        let mut m = stamped_mapper(4, 4);
        m.prg_write_byte(0x8000, 0x31); // PRG bank 3, CHR bank 1
        assert_eq!(m.prg_read_byte(0x8000), 0xA0 + 3);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0 + 1);
    }

    // Bits 7-6 and 3-2 are ignored by the PRG/CHR decode masks.
    #[test]
    fn unused_latch_bits_are_ignored() {
        let mut m = stamped_mapper(4, 4);
        // 0xCC = 1100_1100: PRG field bits5-4 = 00, CHR field bits1-0 = 00,
        // with the ignored bits set. Both banks must resolve to 0.
        m.prg_write_byte(0x8000, 0xCC);
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0);
    }

    // PRG bank selection wraps modulo the bank count for undersized ROMs.
    #[test]
    fn prg_bank_wraps_modulo_count() {
        let mut m = stamped_mapper(2, 1); // only two 32 KB banks
        m.prg_write_byte(0x8000, 0x30); // PRG bank 3 % 2 == 1
        assert_eq!(m.prg_read_byte(0x8000), 0xA0 + 1);
    }

    // CHR bank selection wraps modulo the CHR bank count.
    #[test]
    fn chr_bank_wraps_modulo_count() {
        let mut m = stamped_mapper(1, 2); // only two 8 KB CHR banks
        m.prg_write_byte(0x8000, 0x03); // CHR bank 3 % 2 == 1
        assert_eq!(m.chr_read_byte(0x0000), 0xC0 + 1);
    }

    // Reads below $8000 are not decoded.
    #[test]
    fn reads_below_8000_return_zero() {
        let mut m = stamped_mapper(4, 1);
        m.prg_write_byte(0x8000, 0x30);
        assert_eq!(m.prg_read_byte(0x4020), 0);
    }

    // get_state/apply_state restores both bank latches.
    #[test]
    fn state_roundtrip_restores_banks() {
        let mut m = stamped_mapper(4, 4);
        m.prg_write_byte(0x8000, 0x31); // PRG 3, CHR 1
        let snap = m.get_state();
        m.prg_write_byte(0x8000, 0x00);
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 0xA0 + 3);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0 + 1);
    }

    // reset returns both latches to power-on bank 0.
    #[test]
    fn reset_returns_to_bank_zero() {
        let mut m = stamped_mapper(4, 4);
        m.prg_write_byte(0x8000, 0x31);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0);
    }
}
