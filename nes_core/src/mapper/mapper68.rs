// Sunsoft-4 / NINA-006 (mapper 68). Two 8 KB PRG banks plus four
// 2 KB CHR banks; also has special "nametable as CHR" mode where
// the two fill nametables are sourced from CHR ROM rather than
// internal VRAM. That advanced mode isn't widely used and would
// need PPU-side wiring we don't currently have — we stub it to
// fall back on the cart's default mirroring.
//
// Unlocks After Burner (JP), Maharajah, Ripple Island, Nantettatte!! Baseball.

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

const PRG_BANK: usize = 0x4000;
const CHR_BANK_2K: usize = 0x0800;

pub struct Mapper68 {
    cartridge: Cartridge,
    prg_bank: u8,     // $8000 (low) — 16 KB
    chr_banks: [u8; 4], // 2 KB each at $0000/$0800/$1000/$1800
    nt_bank_0: u8,
    nt_bank_1: u8,
    use_chr_nametables: bool,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub chr_banks: [u8; 4],
    pub nt_bank_0: u8,
    pub nt_bank_1: u8,
    pub use_chr_nametables: bool,
}

impl Mapper68 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper68 {
            cartridge,
            prg_bank: 0,
            chr_banks: [0; 4],
            nt_bank_0: 0,
            nt_bank_1: 0,
            use_chr_nametables: false,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_bank_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK).max(1)
    }

    fn chr_bank_count(&self) -> usize {
        (self.cartridge.chr.len() / CHR_BANK_2K).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let low = self.prg_bank as usize % self.prg_bank_count();
        let high = self.prg_bank_count() - 1;
        let prg = &self.cartridge.prg_rom;
        let lo_off = low * PRG_BANK;
        let hi_off = high * PRG_BANK;
        if lo_off + PRG_BANK <= prg.len() {
            self.prg_asm_window[..PRG_BANK]
                .copy_from_slice(&prg[lo_off..lo_off + PRG_BANK]);
        }
        if hi_off + PRG_BANK <= prg.len() {
            self.prg_asm_window[PRG_BANK..]
                .copy_from_slice(&prg[hi_off..hi_off + PRG_BANK]);
        }
    }
}

impl Mapper for Mapper68 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xBFFF => {
                let bank = self.prg_bank as usize % self.prg_bank_count();
                self.cartridge.prg_rom[bank * PRG_BANK | (address as usize & 0x3FFF)]
            }
            0xC000..=0xFFFF => {
                let bank = self.prg_bank_count() - 1;
                self.cartridge.prg_rom[bank * PRG_BANK | (address as usize & 0x3FFF)]
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                if i < self.cartridge.prg_ram.len() {
                    self.cartridge.prg_ram[i] = value;
                }
            }
            0x8000..=0x8FFF => self.chr_banks[0] = value,
            0x9000..=0x9FFF => self.chr_banks[1] = value,
            0xA000..=0xAFFF => self.chr_banks[2] = value,
            0xB000..=0xBFFF => self.chr_banks[3] = value,
            0xC000..=0xCFFF => {
                self.nt_bank_0 = value & 0x7F;
            }
            0xD000..=0xDFFF => {
                self.nt_bank_1 = value & 0x7F;
            }
            0xE000..=0xEFFF => {
                // Mirroring control: low 2 bits select mode; bit 4
                // enables CHR-ROM-as-nametable (stubbed).
                self.use_chr_nametables = value & 0x10 != 0;
                self.cartridge.mirroring = match value & 0x03 {
                    0 => Mirroring::Horizontal,
                    1 => Mirroring::Vertical,
                    2 => Mirroring::OneScreenLower,
                    _ => Mirroring::OneScreenUpper,
                };
            }
            0xF000..=0xFFFF => {
                self.prg_bank = value & 0x0F;
                self.rebuild_asm_window();
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let slot = ((address >> 11) & 0x03) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_bank_count();
        self.cartridge.chr[bank * CHR_BANK_2K | (address as usize & 0x07FF)]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let slot = ((address >> 11) & 0x03) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_bank_count();
        let off = bank * CHR_BANK_2K | (address as usize & 0x07FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.chr_banks = [0; 4];
        self.nt_bank_0 = 0;
        self.nt_bank_1 = 0;
        self.use_chr_nametables = false;
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
        mapper::State::State68(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            chr_banks: self.chr_banks,
            nt_bank_0: self.nt_bank_0,
            nt_bank_1: self.nt_bank_1,
            use_chr_nametables: self.use_chr_nametables,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State68(s) => {
                self.cartridge.apply_state(&s.cartridge);
                self.prg_bank = s.prg_bank;
                self.chr_banks = s.chr_banks;
                self.nt_bank_0 = s.nt_bank_0;
                self.nt_bank_1 = s.nt_bank_1;
                self.use_chr_nametables = s.use_chr_nametables;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant for Sunsoft-4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 68,
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

    // Stamp offset 0 of every 16 KB PRG bank and every 2 KB CHR bank with
    // its own index, so a read through a window reveals which bank is live.
    fn stamped_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        let mut c = test_cart(prg_16k_banks, chr_8k_banks);
        for i in 0..prg_16k_banks as usize {
            c.prg_rom[i * PRG_BANK] = i as u8;
        }
        let chr_2k = c.chr.len() / CHR_BANK_2K;
        for i in 0..chr_2k {
            c.chr[i * CHR_BANK_2K] = i as u8;
        }
        c
    }

    // The $8000-$BFFF window maps whichever bank was written to $F000.
    #[test]
    fn prg_bank_select_low_window() {
        let mut m = Mapper68::new(stamped_cart(4, 2));
        m.prg_write_byte(0xF000, 2);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        m.prg_write_byte(0xF000, 3);
        assert_eq!(m.prg_read_byte(0x8000), 3);
    }

    // The last 16 KB PRG bank is fixed at $C000 regardless of $8000 select.
    #[test]
    fn prg_last_bank_fixed_high_window() {
        let mut m = Mapper68::new(stamped_cart(4, 2));
        m.prg_write_byte(0xF000, 0);
        assert_eq!(m.prg_read_byte(0xC000), 3);
        m.prg_write_byte(0xF000, 2);
        assert_eq!(m.prg_read_byte(0xC000), 3);
    }

    // PRG bank index (masked to 4 bits) wraps modulo the bank count.
    #[test]
    fn prg_bank_index_wraps() {
        let mut m = Mapper68::new(stamped_cart(4, 2));
        // 5 & 0x0F = 5, then 5 % 4 = 1.
        m.prg_write_byte(0xF000, 5);
        assert_eq!(m.prg_read_byte(0x8000), 1);
    }

    // Each 2 KB CHR window ($8000/$9000/$A000/$B000) selects independently.
    #[test]
    fn chr_bank_select_per_slot() {
        let mut m = Mapper68::new(stamped_cart(4, 2));
        m.prg_write_byte(0x8000, 1);
        m.prg_write_byte(0x9000, 2);
        m.prg_write_byte(0xA000, 3);
        m.prg_write_byte(0xB000, 4);
        assert_eq!(m.chr_read_byte(0x0000), 1);
        assert_eq!(m.chr_read_byte(0x0800), 2);
        assert_eq!(m.chr_read_byte(0x1000), 3);
        assert_eq!(m.chr_read_byte(0x1800), 4);
    }

    // CHR bank index wraps modulo the 2 KB bank count (here 4 banks).
    #[test]
    fn chr_bank_index_wraps() {
        let mut m = Mapper68::new(stamped_cart(1, 1));
        m.prg_write_byte(0x8000, 5); // 5 % 4 = 1
        assert_eq!(m.chr_read_byte(0x0000), 1);
    }

    // $6000-$7FFF PRG RAM writes read back through the same window.
    #[test]
    fn prg_ram_read_write() {
        let mut m = Mapper68::new(stamped_cart(2, 1));
        m.prg_write_byte(0x6000, 0xAB);
        m.prg_write_byte(0x7FFF, 0xCD);
        assert_eq!(m.prg_read_byte(0x6000), 0xAB);
        assert_eq!(m.prg_read_byte(0x7FFF), 0xCD);
    }

    // $E000 low two bits drive the mirroring mode.
    #[test]
    fn mirroring_control() {
        let mut m = Mapper68::new(stamped_cart(2, 1));
        m.prg_write_byte(0xE000, 0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(0xE000, 1);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0xE000, 2);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
        m.prg_write_byte(0xE000, 3);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }

    // $E000 bit 4 latches the CHR-as-nametable enable flag.
    #[test]
    fn chr_nametable_enable_bit() {
        let mut m = Mapper68::new(stamped_cart(2, 1));
        m.prg_write_byte(0xE000, 0x10);
        match m.get_state() {
            mapper::State::State68(s) => assert!(s.use_chr_nametables),
            _ => panic!("wrong state variant"),
        }
        m.prg_write_byte(0xE000, 0x00);
        match m.get_state() {
            mapper::State::State68(s) => assert!(!s.use_chr_nametables),
            _ => panic!("wrong state variant"),
        }
    }

    // $C000/$D000 store the two nametable bank selects masked to 7 bits.
    #[test]
    fn nametable_bank_registers_masked() {
        let mut m = Mapper68::new(stamped_cart(2, 1));
        m.prg_write_byte(0xC000, 0xFF);
        m.prg_write_byte(0xD000, 0x81);
        match m.get_state() {
            mapper::State::State68(s) => {
                assert_eq!(s.nt_bank_0, 0x7F);
                assert_eq!(s.nt_bank_1, 0x01);
            }
            _ => panic!("wrong state variant"),
        }
    }

    // Bank selections and mirroring survive a get_state/apply_state cycle.
    #[test]
    fn state_round_trip() {
        let mut m = Mapper68::new(stamped_cart(4, 2));
        m.prg_write_byte(0xF000, 2); // low PRG window -> bank 2
        m.prg_write_byte(0x8000, 3); // CHR slot 0 -> bank 3
        m.prg_write_byte(0xE000, 1); // vertical
        let snap = m.get_state();
        m.prg_write_byte(0xF000, 0);
        m.prg_write_byte(0x8000, 0);
        m.prg_write_byte(0xE000, 0);
        assert_eq!(m.prg_read_byte(0x8000), 0);
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        assert_eq!(m.chr_read_byte(0x0000), 3);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    // reset() returns to bank 0 everywhere and the cart's default mirroring.
    #[test]
    fn reset_restores_power_on() {
        let mut c = stamped_cart(4, 2);
        c.default_mirroring = Mirroring::Vertical;
        c.mirroring = Mirroring::Vertical;
        let mut m = Mapper68::new(c);
        m.prg_write_byte(0xF000, 3);
        m.prg_write_byte(0x8000, 2);
        m.prg_write_byte(0xE000, 2);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.chr_read_byte(0x0000), 0);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }
}
