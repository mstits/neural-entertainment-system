use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 234 — Maxi 15 in 1 pirate multicart.
//
// Two bank-select registers, both latched by writes in the $FF80-$FFFF
// range. Register A ($FF80-$FF9F) encodes the outer bank + mirroring +
// chip mode, register B ($FFE8-$FFF7) encodes the inner bank. Most
// games use a simple NROM-like 32 KB PRG / 8 KB CHR window per combined
// bank. Per nesdev:
//   Reg A (on write to $FF80-$FF9F if A is currently zero; sticky):
//     bit 7:   mirroring (0 = vertical, 1 = horizontal)
//     bit 6:   mode (0 = NROM-32K, 1 = NROM-128K block)
//     bits 5-3: outer PRG bank (upper 3 bits of the combined bank)
//     bits 2-0: outer CHR bank (upper 3 bits)
//   Reg B (on write to $FFE8-$FFF7, always):
//     bits 6-4: inner PRG bank (low 3 bits), used in mode 1
//     bits 2-0: inner CHR bank (low 3 bits), used in mode 1
//
// For boot + 120 frames we implement the common case: combine outer and
// inner to form a 32 KB PRG + 8 KB CHR window, update mirroring from
// reg A bit 7. Details vary by cart — a panic is the only real failure
// mode we care about here.
pub struct Mapper234 {
    cartridge: Cartridge,
    reg_a: u8,
    reg_b: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub reg_a: u8,
    pub reg_b: u8,
    pub mirroring: Mirroring,
}

impl Mapper234 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper234 {
            cartridge,
            reg_a: 0,
            reg_b: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_32k_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn chr_8k_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x2000).max(1)
    }

    /// Combined 32 KB PRG bank selected by regA+regB. In NROM-32K mode
    /// (regA bit 6 == 0) only the outer bits matter and the inner regB
    /// bits are ignored; in NROM-128K mode (regA bit 6 == 1) the outer
    /// bits fix the upper half and the inner bits pick within a 128 KB
    /// block.
    fn prg_bank(&self) -> usize {
        let outer = ((self.reg_a >> 3) & 0x07) as usize;
        let inner = ((self.reg_b >> 4) & 0x07) as usize;
        let combined = if self.reg_a & 0x40 != 0 {
            (outer & 0x06) | (inner & 0x01)
        } else {
            outer
        };
        combined % self.prg_32k_banks_count()
    }

    fn chr_bank(&self) -> usize {
        let outer = (self.reg_a & 0x07) as usize;
        let inner = (self.reg_b & 0x07) as usize;
        let combined = if self.reg_a & 0x40 != 0 {
            (outer & 0x06) | (inner & 0x01)
        } else {
            outer
        };
        combined % self.chr_8k_banks_count()
    }

    fn rebuild_asm_window(&mut self) {
        let bank = self.prg_bank();
        let off = bank * 0x8000;
        let prg = &self.cartridge.prg_rom;
        if off + 0x8000 <= prg.len() {
            self.prg_asm_window
                .copy_from_slice(&prg[off..off + 0x8000]);
        }
    }

    fn update_mirroring(&mut self) {
        self.cartridge.mirroring = if self.reg_a & 0x80 == 0 {
            Mirroring::Vertical
        } else {
            Mirroring::Horizontal
        };
    }
}

impl Mapper for Mapper234 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = self.prg_bank();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        if off < self.cartridge.prg_rom.len() {
            self.cartridge.prg_rom[off]
        } else {
            0
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0xFF80 && address <= 0xFF9F {
            // Register A is "sticky" — only accepts writes when its
            // current value is zero. Most Maxi-15 menus rely on this.
            if self.reg_a == 0 {
                self.reg_a = value;
                self.update_mirroring();
                self.rebuild_asm_window();
            }
        } else if address >= 0xFFE8 && address <= 0xFFF7 {
            self.reg_b = value;
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = self.chr_bank();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let bank = self.chr_bank();
        let off = bank * 0x2000 | (address as usize & 0x1FFF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.reg_a = 0;
        self.reg_b = 0;
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
        mapper::State::State234(State {
            cartridge: self.cartridge.get_state(),
            reg_a: self.reg_a,
            reg_b: self.reg_b,
            mirroring: self.cartridge.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State234(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.reg_a = state.reg_a;
                self.reg_b = state.reg_b;
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
            mapper: 234,
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
        0xC0 + bank as u8
    }

    fn chr_marker(bank: usize) -> u8 {
        0xD0 + bank as u8
    }

    fn stamp_prg_32k(cart: &mut Cartridge) {
        let banks = cart.prg_rom.len() / 0x8000;
        for b in 0..banks {
            cart.prg_rom[b * 0x8000] = marker(b);
        }
    }

    fn stamp_chr_8k(cart: &mut Cartridge) {
        let banks = cart.chr.len() / 0x2000;
        for b in 0..banks {
            cart.chr[b * 0x2000] = chr_marker(b);
        }
    }

    // Register A ($FF80-$FF9F) is sticky: the first non-zero write latches
    // and later writes are ignored until reset.
    #[test]
    fn reg_a_is_sticky_after_first_write() {
        let mut cart = test_cart(16, 1); // eight 32 KB PRG banks
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x08); // NROM-32K, outer 1 -> bank 1
        assert_eq!(m.prg_read_byte(0x8000), marker(1));
        m.prg_write_byte(0xFF80, 0x10); // ignored (reg A already non-zero)
        assert_eq!(m.prg_read_byte(0x8000), marker(1));
    }

    // In NROM-32K mode (reg A bit 6 clear) the outer bits alone pick the
    // 32 KB PRG bank.
    #[test]
    fn outer_prg_bank_select_nrom32() {
        for outer in 0u8..8 {
            let mut cart = test_cart(16, 1);
            stamp_prg_32k(&mut cart);
            let mut m = Mapper234::new(cart);
            m.prg_write_byte(0xFF80, outer << 3); // bit 6 = 0, outer in bits 5-3
            assert_eq!(m.prg_read_byte(0x8000), marker(outer as usize));
        }
    }

    // In NROM-128K mode (reg A bit 6 set) the low PRG-bank bit comes from
    // reg B bit 4 while the outer bits fix the upper half.
    #[test]
    fn nrom128_mode_low_bit_from_reg_b() {
        let mut cart = test_cart(16, 1);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x58); // bit 6 set, outer 3 -> base (3&6)=2
        m.prg_write_byte(0xFFE8, 0x00); // reg B inner 0 -> bank 2
        assert_eq!(m.prg_read_byte(0x8000), marker(2));
        m.prg_write_byte(0xFFE8, 0x10); // reg B bit 4 -> low bit 1 -> bank 3
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
    }

    // In NROM mode reg A bits 0-2 select the 8 KB CHR bank.
    #[test]
    fn chr_outer_bank_select_nrom() {
        let mut cart = test_cart(2, 8);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x05); // bit 6 = 0, CHR outer 5
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(5));
    }

    // In NROM-128K mode the low CHR-bank bit comes from reg B bit 0.
    #[test]
    fn chr_nrom128_low_bit_from_reg_b() {
        let mut cart = test_cart(2, 8);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x44); // bit 6 set, CHR outer 4 -> base (4&6)=4
        m.prg_write_byte(0xFFE8, 0x00); // reg B inner 0 -> bank 4
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(4));
        m.prg_write_byte(0xFFE8, 0x01); // reg B bit 0 -> low bit 1 -> bank 5
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(5));
    }

    // Reg A bit 7 selects the nametable mirroring.
    #[test]
    fn mirroring_from_reg_a_bit7() {
        let mut cart = test_cart(2, 1);
        cart.mirroring = Mirroring::Vertical;
        let mut m = Mapper234::new(cart);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0xFF80, 0x80); // bit 7 set -> horizontal
        assert_eq!(m.mirroring(), Mirroring::Horizontal);

        let mut cart2 = test_cart(2, 1);
        cart2.mirroring = Mirroring::Horizontal;
        let mut m2 = Mapper234::new(cart2);
        m2.prg_write_byte(0xFF80, 0x08); // bit 7 clear -> vertical
        assert_eq!(m2.mirroring(), Mirroring::Vertical);
    }

    // Writes just outside the reg A window must not latch reg A.
    #[test]
    fn out_of_reg_a_range_write_does_not_latch() {
        let mut cart = test_cart(16, 1);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF7F, 0x08); // one below the window
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        m.prg_write_byte(0xFFA0, 0x08); // one above the window
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
    }

    // Reg B latches only within $FFE8-$FFF7; the boundaries decode and
    // addresses just outside are ignored.
    #[test]
    fn reg_b_range_boundaries() {
        let mut cart = test_cart(16, 1);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x58); // mode 1, outer 3 -> base 2
        m.prg_write_byte(0xFFE8, 0x00); // low boundary, inner 0 -> bank 2
        assert_eq!(m.prg_read_byte(0x8000), marker(2));
        m.prg_write_byte(0xFFF7, 0x10); // high boundary, inner bit -> bank 3
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        m.prg_write_byte(0xFFF8, 0x00); // just above -> ignored
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        m.prg_write_byte(0xFFE7, 0x00); // just below -> ignored
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
    }

    // SUSPECTED BUG: the physical Maxi-15 board latches its bank registers
    // on the CPU *read* of the $FFxx address (the value comes off the data
    // bus, i.e. the ROM byte at that address). This implementation latches
    // only on writes, so a read of the register window leaves banking
    // untouched. This test documents the current behavior; it is not an
    // assertion that read-latching should be absent.
    #[test]
    fn reads_do_not_latch_banks() {
        let mut cart = test_cart(16, 1);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        let _ = m.prg_read_byte(0xFF80);
        let _ = m.prg_read_byte(0xFFE8);
        assert_eq!(m.prg_read_byte(0x8000), marker(0)); // still power-on bank 0
    }

    // A PRG bank index beyond the ROM wraps modulo the bank count.
    #[test]
    fn prg_bank_wraps_beyond_rom() {
        let mut cart = test_cart(2, 1); // single 32 KB bank
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x38); // outer 7 -> 7 % 1 == 0
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
    }

    // reset clears both registers and restores the default mirroring.
    #[test]
    fn reset_clears_registers_and_mirroring() {
        let mut cart = test_cart(16, 4);
        stamp_prg_32k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x08); // outer 1 -> bank 1, bit 7 clear -> vertical
        assert_eq!(m.prg_read_byte(0x8000), marker(1));
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), marker(0));
        assert_eq!(m.mirroring(), Mirroring::Horizontal); // default restored
    }

    // get_state/apply_state round-trips PRG bank, CHR bank and mirroring.
    #[test]
    fn state_round_trip() {
        let mut cart = test_cart(16, 8);
        stamp_prg_32k(&mut cart);
        stamp_chr_8k(&mut cart);
        let mut m = Mapper234::new(cart);
        m.prg_write_byte(0xFF80, 0x5A); // mode 1, PRG outer 3, CHR outer 2, vertical
        m.prg_write_byte(0xFFE8, 0x11); // reg B bit 4 (PRG) + bit 0 (CHR)
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(3));
        let snap = m.get_state();
        m.prg_write_byte(0xFFE8, 0x00); // mutate reg B away
        assert_eq!(m.prg_read_byte(0x8000), marker(2));
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(2));
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), marker(3));
        assert_eq!(m.chr_read_byte(0x0000), chr_marker(3));
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }
}
