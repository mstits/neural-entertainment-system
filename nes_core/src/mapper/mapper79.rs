use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// iNES Mapper 79 — Nina-003 / NAMCOT-00301 (AVE / C&E discrete logic).
// Single register at $4100-$41FF selects a 32 KB PRG bank (bit 3) and
// an 8 KB CHR bank (bits 0-2). PRG window is fixed at $8000-$FFFF.
// Mirroring is fixed from the iNES header. Writes to $8000-$FFFF are
// ignored (PRG is ROM).
pub struct Mapper79 {
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

impl Mapper79 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper79 {
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

impl Mapper for Mapper79 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        // Register is mirrored across $4100-$41FF. The decoder asserts
        // when A13 is high and A8 is high within the $4000-$5FFF range,
        // which matches `address & 0xE100 == 0x4100`.
        if (address & 0xE100) == 0x4100 {
            self.prg_bank = (value >> 3) & 0x01;
            self.chr_bank = value & 0x07;
            self.rebuild_asm_window();
        }
        // Writes to $8000-$FFFF are ignored (PRG ROM).
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
        mapper::State::State79(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State79(state) => {
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

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 79,
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

    // Stamp the first byte of every 32 KB PRG bank so a read at $8000
    // reveals which bank is currently mapped in.
    fn stamp_prg_32k(cart: &mut Cartridge) {
        let banks = cart.prg_rom.len() / 0x8000;
        for b in 0..banks {
            cart.prg_rom[b * 0x8000] = 0xC0 + b as u8;
        }
    }

    // Stamp the first byte of every 8 KB CHR bank.
    fn stamp_chr_8k(cart: &mut Cartridge) {
        let banks = cart.chr.len() / 0x2000;
        for b in 0..banks {
            cart.chr[b * 0x2000] = 0xD0 + b as u8;
        }
    }

    // Bit 3 of the $4100 register picks one of two 32 KB PRG banks.
    #[test]
    fn prg_bank_bit3_selects_32k_bank() {
        let mut cart = test_cart(4, 2); // two 32 KB PRG banks
        stamp_prg_32k(&mut cart);
        let mut m = Mapper79::new(cart);
        assert_eq!(m.prg_read_byte(0x8000), 0xC0); // power-on bank 0
        m.prg_write_byte(0x4100, 0x08); // value bit 3 -> PRG bank 1
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
        m.prg_write_byte(0x4100, 0x00); // back to bank 0
        assert_eq!(m.prg_read_byte(0x8000), 0xC0);
    }

    // Bits 0-2 of the $4100 register pick the active 8 KB CHR bank.
    #[test]
    fn chr_bank_low_three_bits_select_8k_bank() {
        let mut cart = test_cart(2, 8); // eight 8 KB CHR banks
        stamp_chr_8k(&mut cart);
        let mut m = Mapper79::new(cart);
        for bank in 0u8..8 {
            m.prg_write_byte(0x4100, bank); // bits 0-2 = CHR bank, bit 3 = 0
            assert_eq!(m.chr_read_byte(0x0000), 0xD0 + bank);
        }
    }

    // The register only decodes at `addr & 0xE100 == 0x4100`; writes into
    // ROM space or to a non-matching low address must not rebank.
    #[test]
    fn register_ignores_non_matching_writes() {
        let mut cart = test_cart(4, 2);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x4100, 0x08); // select PRG bank 1
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
        m.prg_write_byte(0x8000, 0x00); // store into ROM space -> ignored
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
        m.prg_write_byte(0x4000, 0x00); // does not match the decode mask
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
    }

    // The register is mirrored across the whole $4100-$41FF window.
    #[test]
    fn register_mirrored_across_4100_41ff() {
        let mut cart = test_cart(4, 2);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x41FF, 0x08); // top of the mirror range
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
    }

    // A CHR bank index past the end of ROM wraps rather than reading OOB.
    #[test]
    fn chr_bank_wraps_when_rom_smaller() {
        let mut cart = test_cart(2, 2); // only two 8 KB CHR banks
        stamp_chr_8k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x4100, 0x05); // asks CHR bank 5 -> 5 % 2 == 1
        assert_eq!(m.chr_read_byte(0x0000), 0xD1);
    }

    // A PRG bank index past the end of ROM wraps to bank 0.
    #[test]
    fn prg_bank_wraps_single_bank_rom() {
        let mut cart = test_cart(2, 2); // single 32 KB PRG bank
        stamp_prg_32k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x4100, 0x08); // asks bank 1 -> 1 % 1 == 0
        assert_eq!(m.prg_read_byte(0x8000), 0xC0);
    }

    // reset returns to the power-on PRG 0 / CHR 0 configuration.
    #[test]
    fn reset_restores_power_on_banks() {
        let mut cart = test_cart(4, 8);
        stamp_prg_32k(&mut cart);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x4100, 0x0F); // PRG bank 1, CHR bank 7
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
        assert_eq!(m.chr_read_byte(0x0000), 0xD7);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0xC0);
        assert_eq!(m.chr_read_byte(0x0000), 0xD0);
    }

    // Mirroring is fixed by the header and unaffected by register writes.
    #[test]
    fn mirroring_is_fixed_from_header() {
        let mut cart = test_cart(2, 2);
        cart.mirroring = Mirroring::Vertical;
        let mut m = Mapper79::new(cart);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0x4100, 0x0F);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    // get_state/apply_state round-trips the selected PRG and CHR banks.
    #[test]
    fn state_round_trip_restores_banks() {
        let mut cart = test_cart(4, 8);
        stamp_prg_32k(&mut cart);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper79::new(cart);
        m.prg_write_byte(0x4100, 0x0D); // PRG bank 1 (bit 3), CHR bank 5
        let snap = m.get_state();
        m.prg_write_byte(0x4100, 0x00); // mutate away
        assert_eq!(m.prg_read_byte(0x8000), 0xC0);
        assert_eq!(m.chr_read_byte(0x0000), 0xD0);
        m.apply_state(&snap); // restore
        assert_eq!(m.prg_read_byte(0x8000), 0xC1);
        assert_eq!(m.chr_read_byte(0x0000), 0xD5);
    }
}
