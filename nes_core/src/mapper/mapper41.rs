use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Caltron 6-in-1 multicart. 128 KB PRG (4 × 32 KB), 128 KB CHR (16 × 8 KB).
//
// Two register regions:
//   $6000-$67FF: bit 5 = mirroring (0 = V, 1 = H),
//                bits 4-3 = CHR bank high bits,
//                bits 2-0 = PRG bank (32 KB select).
//   $8000-$FFFF: bits 1-0 = CHR bank low bits (only when the outer
//                register left the inner bank unlocked; model as always
//                latchable, which matches the Caltron carts' boot code).
//
// Composed CHR bank = ((chr_high << 2) | chr_low) & 0x0F.
pub struct Mapper41 {
    cartridge: Cartridge,
    prg_bank: u8,
    chr_bank_high: u8,
    chr_bank_low: u8,
    mirroring: Mirroring,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub chr_bank_high: u8,
    pub chr_bank_low: u8,
    pub mirroring: Mirroring,
}

impl Mapper41 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mirroring = cartridge.mirroring;
        let mut m = Mapper41 {
            cartridge,
            prg_bank: 0,
            chr_bank_high: 0,
            chr_bank_low: 0,
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

    fn composed_chr_bank(&self) -> usize {
        (((self.chr_bank_high as usize) << 2) | (self.chr_bank_low as usize)) & 0x0F
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

impl Mapper for Mapper41 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank = (self.prg_bank as usize) % self.prg_banks_count();
        let off = bank * 0x8000 | (address as usize & 0x7FFF);
        let prg = &self.cartridge.prg_rom;
        if prg.is_empty() {
            return 0;
        }
        prg[off & (prg.len() - 1)]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..=0x67FF).contains(&address) {
            self.prg_bank = value & 0x07;
            self.chr_bank_high = (value >> 3) & 0x03;
            self.mirroring = if value & 0x20 != 0 {
                Mirroring::Horizontal
            } else {
                Mirroring::Vertical
            };
            self.rebuild_asm_window();
        } else if address >= 0x8000 {
            self.chr_bank_low = value & 0x03;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr = &self.cartridge.chr;
        if chr.is_empty() {
            return 0;
        }
        let bank = self.composed_chr_bank() % self.chr_banks_count();
        let off = (bank * 0x2000) | (address as usize & 0x1FFF);
        chr[off & (chr.len() - 1)]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_len = self.cartridge.chr.len();
        if chr_len == 0 {
            return;
        }
        let bank = self.composed_chr_bank() % self.chr_banks_count();
        let off = ((bank * 0x2000) | (address as usize & 0x1FFF)) & (chr_len - 1);
        self.cartridge.chr[off] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_bank_high = 0;
        self.chr_bank_low = 0;
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
        mapper::State::State41(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_bank_high: self.chr_bank_high,
            chr_bank_low: self.chr_bank_low,
            mirroring: self.mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State41(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.chr_bank_high = state.chr_bank_high;
                self.chr_bank_low = state.chr_bank_low;
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

    fn cart(prg_16k: u8, chr_8k: u8) -> Cartridge {
        Cartridge {
            mapper: 41,
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

    // Stamp byte 0 of every 32 KB PRG bank with 0xA0 + index.
    fn stamp_prg_32k(c: &mut Cartridge) {
        let banks = c.prg_rom.len() / 0x8000;
        for b in 0..banks {
            c.prg_rom[b * 0x8000] = 0xA0 + b as u8;
        }
    }

    // Stamp byte 0 of every 8 KB CHR bank with 0xC0 + index.
    fn stamp_chr_8k(c: &mut Cartridge) {
        let banks = c.chr.len() / 0x2000;
        for b in 0..banks {
            c.chr[b * 0x2000] = 0xC0 + b as u8;
        }
    }

    // $6000-$67FF bits 2-0 select the 32 KB PRG bank.
    #[test]
    fn prg_bank_switch() {
        let mut c = cart(8, 16); // 128 KB -> 4 × 32 KB banks
        stamp_prg_32k(&mut c);
        let mut m = Mapper41::new(c);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA0);
        m.prg_write_byte(0x6000, 2);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA2);
    }

    // An oversized PRG index wraps rather than reading OOB.
    #[test]
    fn prg_bank_wraps() {
        let mut c = cart(8, 16);
        stamp_prg_32k(&mut c);
        let mut m = Mapper41::new(c);
        m.prg_write_byte(0x6000, 7); // 7 & 7 == 7, 7 % 4 == 3
        assert_eq!(m.prg_peek_byte(0x8000), 0xA3);
    }

    // $6000 bit 5 drives mirroring: set = Horizontal, clear = Vertical.
    #[test]
    fn mirroring_control() {
        let mut m = Mapper41::new(cart(8, 16));
        m.prg_write_byte(0x6000, 0x20);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(0x6000, 0x00);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    // CHR bank composes from $6000 high bits (4-3) and $8000 low bits (1-0):
    // bank = ((high << 2) | low) & 0x0F.
    #[test]
    fn chr_bank_switch_composed() {
        let mut c = cart(8, 16); // 128 KB CHR -> 16 × 8 KB banks
        stamp_chr_8k(&mut c);
        let mut m = Mapper41::new(c);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0); // power-on bank 0
        m.prg_write_byte(0x6000, 0x08); // high = (0x08 >> 3) & 3 = 1
        m.prg_write_byte(0x8000, 0x02); // low = 2 -> composed = 6
        assert_eq!(m.chr_read_byte(0x0000), 0xC6);
        m.prg_write_byte(0x6000, 0x18); // high = 3
        m.prg_write_byte(0x8000, 0x03); // low = 3 -> composed = 15
        assert_eq!(m.chr_read_byte(0x0000), 0xCF);
    }

    // The composed CHR bank wraps against the actual bank count.
    #[test]
    fn chr_bank_wraps() {
        let mut c = cart(8, 4); // 32 KB CHR -> 4 × 8 KB banks
        stamp_chr_8k(&mut c);
        let mut m = Mapper41::new(c);
        m.prg_write_byte(0x6000, 0x18); // high = 3
        m.prg_write_byte(0x8000, 0x03); // low = 3 -> composed 15, 15 % 4 == 3
        assert_eq!(m.chr_read_byte(0x0000), 0xC3);
    }

    // CHR writes land in the currently-composed bank and are visible on read.
    #[test]
    fn chr_write_through() {
        let mut m = Mapper41::new(cart(8, 16));
        m.prg_write_byte(0x6000, 0x08); // high = 1
        m.prg_write_byte(0x8000, 0x02); // low = 2 -> composed 6
        m.chr_write_byte(0x0000, 0x5A);
        assert_eq!(m.chr_read_byte(0x0000), 0x5A);
        m.prg_write_byte(0x8000, 0x00); // composed 4 -> a different bank
        assert_ne!(m.chr_read_byte(0x0000), 0x5A);
    }

    // reset() clears the bank latches and restores cartridge mirroring.
    #[test]
    fn reset_restores_power_on() {
        let mut c = cart(8, 16);
        stamp_prg_32k(&mut c);
        let mut m = Mapper41::new(c);
        m.prg_write_byte(0x6000, 0x2A); // prg=2, high=1, mirroring H
        m.prg_write_byte(0x8000, 0x03); // low = 3
        m.reset();
        assert_eq!(m.prg_bank, 0);
        assert_eq!(m.chr_bank_high, 0);
        assert_eq!(m.chr_bank_low, 0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal); // cartridge default
        assert_eq!(m.prg_peek_byte(0x8000), 0xA0);
    }

    // get_state/apply_state round-trips PRG, CHR and mirroring latches.
    #[test]
    fn state_round_trip() {
        let mut c = cart(8, 16);
        stamp_prg_32k(&mut c);
        stamp_chr_8k(&mut c);
        let mut m = Mapper41::new(c);
        m.prg_write_byte(0x6000, 0x2A); // prg=2, high=1, mirroring H
        m.prg_write_byte(0x8000, 0x02); // low=2 -> composed 6
        let snap = m.get_state();
        m.prg_write_byte(0x6000, 0x00);
        m.prg_write_byte(0x8000, 0x00);
        m.apply_state(&snap);
        assert_eq!(m.prg_peek_byte(0x8000), 0xA2);
        assert_eq!(m.chr_read_byte(0x0000), 0xC6);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }
}
