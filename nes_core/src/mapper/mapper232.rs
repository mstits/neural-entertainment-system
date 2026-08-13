use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 232 — Camerica BF9096 (Quattro Adventure, Quattro Arcade,
// Quattro Sports). 256 KB PRG split into four 64 KB blocks; each
// block holds four 16 KB banks. Lower window ($8000-$BFFF) is a
// switchable bank within the current block; upper window
// ($C000-$FFFF) is fixed to the last bank of the current block.
// CHR is 8 KB of CHR-RAM, no switching.
pub struct Mapper232 {
    cartridge: Cartridge,
    block: u8,       // bits 4-3 of the value written to $8000-$BFFF
    inner_bank: u8,  // bits 1-0 of the value written to $C000-$FFFF
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub block: u8,
    pub inner_bank: u8,
}

impl Mapper232 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper232 {
            cartridge,
            block: 0,
            inner_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_16k_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x4000).max(1)
    }

    fn low_bank(&self) -> usize {
        let bank = ((self.block as usize) << 2) | (self.inner_bank as usize);
        bank % self.prg_16k_banks_count()
    }

    fn high_bank(&self) -> usize {
        let bank = ((self.block as usize) << 2) | 0x03;
        bank % self.prg_16k_banks_count()
    }

    fn rebuild_asm_window(&mut self) {
        let prg = &self.cartridge.prg_rom;
        let low_off = self.low_bank() * 0x4000;
        let high_off = self.high_bank() * 0x4000;
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

impl Mapper for Mapper232 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = if address < 0xC000 {
            self.low_bank()
        } else {
            self.high_bank()
        };
        let off = bank * 0x4000 | (address as usize & 0x3FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x8000..=0xBFFF => {
                // Block select (bits 4-3 of value).
                self.block = (value >> 3) & 0x03;
                self.rebuild_asm_window();
            }
            0xC000..=0xFFFF => {
                // Inner bank select (bits 1-0 of value).
                self.inner_bank = value & 0x03;
                self.rebuild_asm_window();
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.block = 0;
        self.inner_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State232(State {
            cartridge: self.cartridge.get_state(),
            block: self.block,
            inner_bank: self.inner_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State232(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.block = state.block;
                self.inner_bank = state.inner_bank;
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
            mapper: 232,
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

    fn marker(bank: usize) -> u8 {
        0x40 + bank as u8
    }

    fn stamp_prg_16k(cart: &mut Cartridge) {
        let banks = cart.prg_rom.len() / 0x4000;
        for b in 0..banks {
            cart.prg_rom[b * 0x4000] = marker(b);
        }
    }

    // The $C000 register picks the inner bank of the low ($8000) window,
    // within the current 64 KB block.
    #[test]
    fn inner_bank_selects_low_window() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        for inner in 0u8..4 {
            m.prg_write_byte(0xC000, inner); // inner = value & 3
            assert_eq!(m.prg_read_byte(0x8000), marker(inner as usize));
        }
    }

    // The high ($C000) window is fixed to the last bank of the block
    // regardless of the inner-bank selection.
    #[test]
    fn high_window_fixed_to_last_bank_of_block() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        for inner in 0u8..4 {
            m.prg_write_byte(0xC000, inner);
            assert_eq!(m.prg_read_byte(0xC000), marker(3)); // block 0 last bank
        }
        m.prg_write_byte(0x8000, 1 << 3); // block 1
        assert_eq!(m.prg_read_byte(0xC000), marker(7)); // (1<<2)|3
    }

    // The $8000 register picks the 64 KB block (its base 16 KB bank).
    #[test]
    fn block_select_shifts_window_base() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0xC000, 0); // inner 0
        for block in 0u8..4 {
            m.prg_write_byte(0x8000, block << 3); // block bits 4-3
            let base = (block as usize) << 2;
            assert_eq!(m.prg_read_byte(0x8000), marker(base));
            assert_eq!(m.prg_read_byte(0xC000), marker(base | 3));
        }
    }

    // Only bits 4-3 of the $8000 value select the block.
    #[test]
    fn block_select_uses_bits_4_and_3_only() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0xC000, 0);
        m.prg_write_byte(0x8000, 0xFF); // (0xFF>>3)&3 == 3
        assert_eq!(m.prg_read_byte(0x8000), marker(12));
        m.prg_write_byte(0x8000, 0x08); // (0x08>>3)&3 == 1
        assert_eq!(m.prg_read_byte(0x8000), marker(4));
    }

    // Only bits 1-0 of the $C000 value select the inner bank.
    #[test]
    fn inner_select_uses_low_two_bits_only() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0x8000, 0); // block 0
        m.prg_write_byte(0xC000, 0xFF); // 0xFF & 3 == 3
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        m.prg_write_byte(0xC000, 0x06); // 0x06 & 3 == 2
        assert_eq!(m.prg_read_byte(0x8000), marker(2));
    }

    // Bank indices wrap modulo the bank count on an undersized ROM.
    #[test]
    fn bank_index_wraps_on_small_rom() {
        let mut cart = test_cart(4, 1); // only block 0 present
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0x8000, 2 << 3); // block 2 -> base 8
        m.prg_write_byte(0xC000, 0);
        assert_eq!(m.prg_read_byte(0x8000), marker(0)); // 8 % 4
        assert_eq!(m.prg_read_byte(0xC000), marker(3)); // 11 % 4
    }

    // CHR is 8 KB of CHR-RAM: writes are read back verbatim.
    #[test]
    fn chr_ram_read_write_round_trips() {
        let cart = test_cart(4, 1);
        let mut m = Mapper232::new(cart);
        m.chr_write_byte(0x0000, 0xAB);
        m.chr_write_byte(0x1FFF, 0xCD);
        assert_eq!(m.chr_read_byte(0x0000), 0xAB);
        assert_eq!(m.chr_read_byte(0x1FFF), 0xCD);
    }

    // reset returns block and inner bank to zero.
    #[test]
    fn reset_restores_block_and_inner_zero() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0x8000, 3 << 3); // block 3
        m.prg_write_byte(0xC000, 2); // inner 2
        assert_eq!(m.prg_read_byte(0x8000), marker(14)); // 12 | 2
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        assert_eq!(m.prg_read_byte(0xC000), marker(3)); // block 0 last bank
    }

    // get_state/apply_state round-trips both the block and inner registers.
    #[test]
    fn state_round_trip() {
        let mut cart = test_cart(16, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper232::new(cart);
        m.prg_write_byte(0x8000, 2 << 3); // block 2
        m.prg_write_byte(0xC000, 1); // inner 1
        assert_eq!(m.prg_read_byte(0x8000), marker(9)); // 8 | 1
        let snap = m.get_state();
        m.prg_write_byte(0x8000, 0);
        m.prg_write_byte(0xC000, 0);
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), marker(9));
        assert_eq!(m.prg_read_byte(0xC000), marker(11)); // block 2 last bank
    }
}
