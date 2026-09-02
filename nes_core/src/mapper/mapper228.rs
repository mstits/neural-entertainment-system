use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 228 — "Action 52" / Cheetahmen II pirate multicart.
// Bank selection is encoded in the WRITE ADDRESS, value selects CHR:
//
//   A13:          mirroring (0 = vertical, 1 = horizontal)
//   A12..A11:     PRG chip select (0, 1, 3 -> ROM; 2 -> unpopulated, open bus)
//   A10..A6:      5-bit PRG bank within the selected chip
//   A5:           PRG mode (0 = 32 KB, 1 = 16 KB mirrored)
//   value[5..2]:  CHR bank high bits
//   value[1..0]:  CHR bank low bits
//
// Action 52 carries three 512 KB chips on selects 0, 1 and 3; select 2
// is unpopulated. The iNES dump stores them back to back (chip 0 at
// PRG offset 0, chip 1 at 512 KB, chip 3 at 1 MB), so we flatten the
// PRG into 16 KB banks indexed linearly and alias select 3 onto the
// third 512 KB (select 2, open bus on the board, reads the same
// region here). Cheetahmen II is a single 256 KB chip, so the bank
// modulo below makes the chip bits irrelevant for it.
//
// The register has no reset line; the games expect the equivalent of
// a $00 write to $8000 at power-on (chip 0, bank 0, 32 KB mode). The
// 32 KB mode matters: bank pair 0/1 (what power-on and reset() select)
// keeps its reset vector and boot stub in bank 1, the odd bank, so
// 16 KB mode at power-on mirrors bank 0 into $C000-$FFFF and boots a
// minigame's own code instead of the multicart menu. (47 of the 48 odd
// banks read RESET=$FFD8 from their own boot stub; the fix only relies
// on bank 1, which is confirmed.)
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
            prg_mode_32k: true,
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
        // Select 3 is the third populated chip, stored third in the
        // dump; select 2 is unpopulated and reads that region too.
        let chip = if chip == 3 { 2 } else { chip };
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
        self.prg_mode_32k = true;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 228,
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

    // Compose the write address that carries the bank-select bits: A12..A11
    // = chip, A10..A6 = inner bank, A5 = PRG-mode (set for 16 KB), A13 =
    // mirroring. Bit 15 keeps the address inside PRG write space.
    fn wr_addr(chip: u16, inner: u16, mode_16k: bool, a13_high: bool) -> u16 {
        let mut a = 0x8000u16;
        if a13_high {
            a |= 1 << 13;
        }
        a |= (chip & 0x03) << 11;
        a |= (inner & 0x1F) << 6;
        if mode_16k {
            a |= 1 << 5; // A5 set => 16 KB mode
        }
        a
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

    fn stamp_chr_8k(cart: &mut Cartridge) {
        let banks = cart.chr.len() / 0x2000;
        for b in 0..banks {
            cart.chr[b * 0x2000] = b as u8;
        }
    }

    // In 16 KB mode both CPU windows mirror the same selected bank.
    #[test]
    fn prg_16k_mode_mirrors_bank_into_both_windows() {
        let mut cart = test_cart(8, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 3, true, false), 0x00); // inner 3, 16 KB
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        assert_eq!(m.prg_read_byte(0xC000), marker(3)); // mirrored high window
    }

    // In 32 KB mode $8000 sees the even base bank and $C000 the next one.
    #[test]
    fn prg_32k_mode_maps_consecutive_banks() {
        let mut cart = test_cart(8, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 4, false, false), 0x00); // inner 4, 32 KB
        assert_eq!(m.prg_read_byte(0x8000), marker(4));
        assert_eq!(m.prg_read_byte(0xC000), marker(5));
    }

    // The 2-bit chip select stacks above the 5-bit inner bank: selects
    // 0, 1 and 3 are the three chips stored in order in the dump, so
    // select 3 aliases onto the third 512 KB, and the unpopulated
    // select 2 reads that region as well.
    #[test]
    fn prg_chip_select_stacks_above_inner_bank() {
        let mut cart = test_cart(128, 1); // 2 MB so every chip is in range
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 1, true, false), 0x00);
        assert_eq!(m.prg_read_byte(0x8000), marker(1)); // chip 0, bank 1
        m.prg_write_byte(wr_addr(1, 1, true, false), 0x00);
        assert_eq!(m.prg_read_byte(0x8000), marker(33)); // (1<<5)|1, second chip
        m.prg_write_byte(wr_addr(2, 1, true, false), 0x00);
        assert_eq!(m.prg_read_byte(0x8000), marker(65)); // (2<<5)|1
        m.prg_write_byte(wr_addr(3, 1, true, false), 0x00);
        assert_eq!(m.prg_read_byte(0x8000), marker(65)); // select 3 -> third chip
    }

    // Power-on is 32 KB mode on bank pair 0/1: the high window shows
    // bank 1, where Action 52 keeps every reset vector and boot stub.
    #[test]
    fn power_on_is_32k_mode_bank_pair_0_1() {
        let mut cart = test_cart(8, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        assert_eq!(m.prg_read_byte(0xC000), marker(1));
    }

    // The CHR bank is the low six bits of the written value (bits 6-7
    // dropped); the split hi/lo nibble recombine covers the full range.
    #[test]
    fn chr_bank_from_value_low_six_bits() {
        let mut cart = test_cart(2, 64);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(0x8000, 0x05);
        assert_eq!(m.chr_read_byte(0x0000), 5);
        m.prg_write_byte(0x8000, 0x3F);
        assert_eq!(m.chr_read_byte(0x0000), 63);
        m.prg_write_byte(0x8000, 0xC5); // bits 6-7 ignored -> still bank 5
        assert_eq!(m.chr_read_byte(0x0000), 5);
    }

    // A13 of the write address selects the nametable mirroring.
    #[test]
    fn mirroring_follows_a13() {
        let cart = test_cart(2, 1);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(0x8000, 0x00); // A13 clear
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0xA000, 0x00); // A13 set
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }

    // A PRG bank index beyond the ROM wraps modulo the bank count.
    #[test]
    fn prg_bank_wraps_beyond_rom() {
        let mut cart = test_cart(4, 1);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 6, true, false), 0x00); // 6 % 4 == 2
        assert_eq!(m.prg_read_byte(0x8000), marker(2));
    }

    // reset returns to bank 0 / 32 KB mode and the default mirroring.
    #[test]
    fn reset_restores_defaults() {
        let mut cart = test_cart(8, 4);
        stamp_prg_16k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 5, true, false), 0x07); // bank 5, CHR 7, vertical
        assert_eq!(m.prg_read_byte(0x8000), marker(5));
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        assert_eq!(m.prg_read_byte(0xC000), marker(1)); // 32 KB mode again
        assert_eq!(m.mirroring(), Mirroring::Horizontal); // default
    }

    // get_state/apply_state round-trips PRG mode, banks, CHR and mirroring.
    #[test]
    fn state_round_trip() {
        let mut cart = test_cart(8, 8);
        stamp_prg_16k(&mut cart);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper228::new(cart);
        m.prg_write_byte(wr_addr(0, 4, false, true), 0x03); // 32 KB, banks 4/5, CHR 3, horizontal
        assert_eq!(m.prg_read_byte(0x8000), marker(4));
        assert_eq!(m.prg_read_byte(0xC000), marker(5));
        let snap = m.get_state();
        m.prg_write_byte(wr_addr(0, 0, true, false), 0x00); // mutate away
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), marker(4));
        assert_eq!(m.prg_read_byte(0xC000), marker(5));
        assert_eq!(m.chr_read_byte(0x0000), 3);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }
}
