use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Mapper 34 is shared by two distinct boards:
//   * BNROM (Deadly Towers, homebrew): $8000-$FFFF writes select a
//     32 KB PRG bank. CHR is a single fixed 8 KB bank (CHR-RAM or
//     a single 8 KB CHR-ROM).
//   * NINA-001 (some homebrew): $7FFD selects PRG bank, $7FFE/$7FFF
//     select two 4 KB CHR banks.
//
// Detection heuristic: header reports 0 or 1 CHR banks -> BNROM.
// Otherwise -> NINA-001.
pub struct Mapper34 {
    cartridge: Cartridge,
    is_nina: bool,
    prg_bank: u8,
    chr_bank_lo: u8,
    chr_bank_hi: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub is_nina: bool,
    pub prg_bank: u8,
    pub chr_bank_lo: u8,
    pub chr_bank_hi: u8,
}

impl Mapper34 {
    pub fn new(cartridge: Cartridge) -> Self {
        let is_nina = cartridge.chr_num_banks >= 2;
        let mut m = Mapper34 {
            cartridge,
            is_nina,
            prg_bank: 0,
            chr_bank_lo: 0,
            chr_bank_hi: 1,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_banks_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x8000).max(1)
    }

    fn chr_4k_banks_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x1000).max(1)
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

impl Mapper for Mapper34 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        self.cartridge.prg_rom[off]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if self.is_nina {
            match address {
                0x7FFD => {
                    self.prg_bank = value;
                    self.rebuild_asm_window();
                }
                0x7FFE => {
                    self.chr_bank_lo = value;
                }
                0x7FFF => {
                    self.chr_bank_hi = value;
                }
                _ => {}
            }
        } else if address >= 0x8000 {
            // BNROM: any value, mask by PRG bank count.
            self.prg_bank = value;
            self.rebuild_asm_window();
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        if self.is_nina {
            let bank_count = self.chr_4k_banks_count();
            let bank = if address < 0x1000 {
                (self.chr_bank_lo as usize) % bank_count
            } else {
                (self.chr_bank_hi as usize) % bank_count
            };
            let off = bank * 0x1000 | (address as usize & 0x0FFF);
            if off < self.cartridge.chr.len() {
                self.cartridge.chr[off]
            } else {
                0
            }
        } else if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize]
        } else {
            0
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        if self.is_nina {
            let bank_count = self.chr_4k_banks_count();
            let bank = if address < 0x1000 {
                (self.chr_bank_lo as usize) % bank_count
            } else {
                (self.chr_bank_hi as usize) % bank_count
            };
            let off = bank * 0x1000 | (address as usize & 0x0FFF);
            if off < self.cartridge.chr.len() {
                self.cartridge.chr[off] = value;
            }
        } else if (address as usize) < self.cartridge.chr.len() {
            self.cartridge.chr[address as usize] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank_lo = 0;
        self.chr_bank_hi = 1;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State34(State {
            cartridge: self.cartridge.get_state(),
            is_nina: self.is_nina,
            prg_bank: self.prg_bank,
            chr_bank_lo: self.chr_bank_lo,
            chr_bank_hi: self.chr_bank_hi,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State34(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.is_nina = state.is_nina;
                self.prg_bank = state.prg_bank;
                self.chr_bank_lo = state.chr_bank_lo;
                self.chr_bank_hi = state.chr_bank_hi;
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

    // `chr_8k >= 2` flips Mapper34 from BNROM into NINA-001 mode.
    fn cart(prg_16k: u8, chr_8k: u8) -> Cartridge {
        Cartridge {
            mapper: 34,
            sub_mapper: 0,
            mirroring: Mirroring::Horizontal,
            default_mirroring: Mirroring::Horizontal,
            prg_rom_num_banks: prg_16k,
            prg_rom: vec![0u8; prg_16k as usize * 16 * 1024],
            chr_num_banks: chr_8k,
            chr: vec![0u8; chr_8k as usize * 8 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    // Stamp byte 0 of every 32 KB PRG bank with 0xA0 + index so a read
    // through the $8000 window reveals which 32 KB bank is mapped.
    fn stamp_prg_32k(c: &mut Cartridge) {
        let banks = c.prg_rom.len() / 0x8000;
        for b in 0..banks {
            c.prg_rom[b * 0x8000] = 0xA0 + b as u8;
        }
    }

    // Stamp byte 0 of every 4 KB CHR bank with 0xC0 + index (NINA banks).
    fn stamp_chr_4k(c: &mut Cartridge) {
        let banks = c.chr.len() / 0x1000;
        for b in 0..banks {
            c.chr[b * 0x1000] = 0xC0 + b as u8;
        }
    }

    // ---- BNROM (single CHR bank) ----

    // A single 8 KB CHR bank selects the BNROM board, not NINA.
    #[test]
    fn bnrom_detected_for_one_chr_bank() {
        let m = Mapper34::new(cart(8, 1));
        assert!(!m.is_nina);
    }

    // BNROM: a $8000-$FFFF write selects the 32 KB PRG bank.
    #[test]
    fn bnrom_prg_bank_switch() {
        let mut c = cart(8, 1); // 128 KB -> 4 × 32 KB banks
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA0); // power-on = bank 0
        m.prg_write_byte(0x8000, 2);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA2);
        m.prg_write_byte(0xFFFF, 3); // any $8000+ address latches
        assert_eq!(m.prg_peek_byte(0x8000), 0xA3);
    }

    // BNROM: a bank index past the ROM wraps (masks) instead of reading OOB.
    #[test]
    fn bnrom_prg_bank_wraps() {
        let mut c = cart(8, 1);
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x8000, 6); // 6 % 4 == 2
        assert_eq!(m.prg_peek_byte(0x8000), 0xA2);
    }

    // BNROM ignores the NINA register window ($7FFD-$7FFF is below $8000).
    #[test]
    fn bnrom_ignores_nina_registers() {
        let mut c = cart(8, 1);
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x8000, 1);
        m.prg_write_byte(0x7FFD, 3); // ignored in BNROM mode
        assert_eq!(m.prg_bank, 1);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA1);
    }

    // BNROM CHR is a flat, directly-indexed 8 KB window (CHR-RAM writable),
    // with no bank switching from the NINA CHR registers.
    #[test]
    fn bnrom_chr_is_flat_window() {
        let mut m = Mapper34::new(cart(8, 1));
        m.chr_write_byte(0x0100, 0xEE);
        m.chr_write_byte(0x1FFF, 0x77);
        assert_eq!(m.chr_read_byte(0x0100), 0xEE);
        assert_eq!(m.chr_read_byte(0x1FFF), 0x77);
        // NINA CHR-select writes must not disturb the flat window.
        m.prg_write_byte(0x7FFE, 5);
        m.prg_write_byte(0x7FFF, 6);
        assert_eq!(m.chr_read_byte(0x0100), 0xEE);
    }

    // ---- NINA-001 (>= 2 CHR banks) ----

    // Two or more 8 KB CHR banks selects the NINA-001 board.
    #[test]
    fn nina_detected_for_multi_chr_banks() {
        let m = Mapper34::new(cart(4, 2));
        assert!(m.is_nina);
    }

    // NINA: $7FFD selects the 32 KB PRG bank; oversized index wraps.
    #[test]
    fn nina_prg_bank_switch() {
        let mut c = cart(4, 2); // 64 KB -> 2 × 32 KB banks
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA0);
        m.prg_write_byte(0x7FFD, 1);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA1);
        m.prg_write_byte(0x7FFD, 3); // 3 % 2 == 1
        assert_eq!(m.prg_peek_byte(0x8000), 0xA1);
    }

    // NINA: $7FFE picks the low 4 KB CHR bank, $7FFF the high 4 KB bank.
    // Power-on latches lo=0, hi=1.
    #[test]
    fn nina_chr_bank_switch() {
        let mut c = cart(4, 2); // 16 KB CHR -> 4 × 4 KB banks
        stamp_chr_4k(&mut c);
        let mut m = Mapper34::new(c);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0); // lo = 0
        assert_eq!(m.chr_read_byte(0x1000), 0xC1); // hi = 1 (power-on)
        m.prg_write_byte(0x7FFE, 2);
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
        m.prg_write_byte(0x7FFF, 3);
        assert_eq!(m.chr_read_byte(0x1000), 0xC3);
    }

    // NINA: an oversized CHR bank index wraps rather than reading OOB.
    #[test]
    fn nina_chr_bank_wraps() {
        let mut c = cart(4, 2);
        stamp_chr_4k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x7FFE, 6); // 6 % 4 == 2
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
    }

    // NINA ignores BNROM-style $8000+ PRG writes.
    #[test]
    fn nina_ignores_bnrom_write() {
        let mut c = cart(4, 2);
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x7FFD, 1);
        m.prg_write_byte(0x8000, 0); // ignored in NINA mode
        assert_eq!(m.prg_bank, 1);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA1);
    }

    // get_state/apply_state round-trips every bank latch (NINA).
    #[test]
    fn nina_state_round_trip() {
        let mut c = cart(4, 2);
        stamp_prg_32k(&mut c);
        stamp_chr_4k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x7FFD, 1);
        m.prg_write_byte(0x7FFE, 2);
        m.prg_write_byte(0x7FFF, 3);
        let snap = m.get_state();
        m.prg_write_byte(0x7FFD, 0);
        m.prg_write_byte(0x7FFE, 0);
        m.prg_write_byte(0x7FFF, 0);
        m.apply_state(&snap);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA1);
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
        assert_eq!(m.chr_read_byte(0x1000), 0xC3);
    }

    // reset() returns to the power-on latch config (prg=0, lo=0, hi=1).
    #[test]
    fn reset_restores_power_on_banks() {
        let mut c = cart(4, 2);
        stamp_prg_32k(&mut c);
        let mut m = Mapper34::new(c);
        m.prg_write_byte(0x7FFD, 1);
        m.prg_write_byte(0x7FFE, 2);
        m.prg_write_byte(0x7FFF, 3);
        m.reset();
        assert_eq!(m.prg_bank, 0);
        assert_eq!(m.chr_bank_lo, 0);
        assert_eq!(m.chr_bank_hi, 1);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA0);
    }
}
