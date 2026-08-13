// Nintendo MMC4 / FxROM (mapper 10). Close cousin of MMC2:
//   * CHR banking + PPU-read latch is identical to MMC2.
//   * PRG layout is DIFFERENT: 16 KB switchable at $8000-$BFFF,
//     16 KB fixed (last bank) at $C000-$FFFF. MMC2 used 8 KB slots.
//
// Unlocks Fire Emblem 1 + 2 (JP) and Famicom Wars (JP).

use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper10 {
    cartridge: Cartridge,
    prg_bank: u8, // 16 KB, low window
    latch_0: u8,
    latch_1: u8,
    chr_fd_0000_bank: u8,
    chr_fe_0000_bank: u8,
    chr_fd_1000_bank: u8,
    chr_fe_1000_bank: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub latch_0: u8,
    pub latch_1: u8,
    pub chr_fd_0000_bank: u8,
    pub chr_fe_0000_bank: u8,
    pub chr_fd_1000_bank: u8,
    pub chr_fe_1000_bank: u8,
}

impl Mapper10 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper10 {
            cartridge,
            prg_bank: 0,
            latch_0: 0xFE,
            latch_1: 0xFE,
            chr_fd_0000_bank: 0,
            chr_fe_0000_bank: 0,
            chr_fd_1000_bank: 0,
            chr_fe_1000_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_bank_count(&self) -> usize {
        (self.cartridge.prg_rom_num_banks as usize).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let low = self.prg_bank as usize % self.prg_bank_count();
        let high = self.prg_bank_count() - 1;
        let prg = &self.cartridge.prg_rom;
        let lo_off = low * bank_size;
        let hi_off = high * bank_size;
        if lo_off + bank_size <= prg.len() {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[lo_off..lo_off + bank_size]);
        }
        if hi_off + bank_size <= prg.len() {
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[hi_off..hi_off + bank_size]);
        }
    }

    fn chr_bank_for(&self, address: u16) -> u8 {
        match address {
            0x0000..=0x0FFF => {
                if self.latch_0 == 0xFD {
                    self.chr_fd_0000_bank
                } else {
                    self.chr_fe_0000_bank
                }
            }
            0x1000..=0x1FFF => {
                if self.latch_1 == 0xFD {
                    self.chr_fd_1000_bank
                } else {
                    self.chr_fe_1000_bank
                }
            }
            _ => 0,
        }
    }
}

impl Mapper for Mapper10 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xBFFF => {
                let bank = self.prg_bank as usize % self.prg_bank_count();
                self.cartridge.prg_rom[bank * bank_size | (address as usize & (bank_size - 1))]
            }
            0xC000..=0xFFFF => {
                let bank = self.prg_bank_count() - 1;
                self.cartridge.prg_rom[bank * bank_size | (address as usize & (bank_size - 1))]
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
            0xA000..=0xAFFF => {
                self.prg_bank = value & 0x0F;
                self.rebuild_asm_window();
            }
            0xB000..=0xBFFF => {
                self.chr_fd_0000_bank = value & 0x1F;
            }
            0xC000..=0xCFFF => {
                self.chr_fe_0000_bank = value & 0x1F;
            }
            0xD000..=0xDFFF => {
                self.chr_fd_1000_bank = value & 0x1F;
            }
            0xE000..=0xEFFF => {
                self.chr_fe_1000_bank = value & 0x1F;
            }
            0xF000..=0xFFFF => {
                self.cartridge.mirroring = if value & 0x01 == 0 {
                    Mirroring::Vertical
                } else {
                    Mirroring::Horizontal
                };
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = self.chr_bank_for(address);
        let off = (bank as usize * 0x1000) | (address as usize & 0x0FFF);
        let byte = self.cartridge.chr.get(off).copied().unwrap_or(0);

        // Latch update — same as MMC2. The special tile IDs 0xFD /
        // 0xFE aren't actual tile indices but addresses accessed
        // during sprite-pattern fetch for those tiles.
        match address {
            0x0FD8..=0x0FDF => self.latch_0 = 0xFD,
            0x0FE8..=0x0FEF => self.latch_0 = 0xFE,
            0x1FD8..=0x1FDF => self.latch_1 = 0xFD,
            0x1FE8..=0x1FEF => self.latch_1 = 0xFE,
            _ => {}
        }

        byte
    }

    fn chr_write_byte(&mut self, _address: u16, _value: u8) {}

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.latch_0 = 0xFE;
        self.latch_1 = 0xFE;
        self.chr_fd_0000_bank = 0;
        self.chr_fe_0000_bank = 0;
        self.chr_fd_1000_bank = 0;
        self.chr_fe_1000_bank = 0;
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
        mapper::State::State10(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            latch_0: self.latch_0,
            latch_1: self.latch_1,
            chr_fd_0000_bank: self.chr_fd_0000_bank,
            chr_fe_0000_bank: self.chr_fe_0000_bank,
            chr_fd_1000_bank: self.chr_fd_1000_bank,
            chr_fe_1000_bank: self.chr_fe_1000_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State10(s) => {
                self.cartridge.apply_state(&s.cartridge);
                self.prg_bank = s.prg_bank;
                self.latch_0 = s.latch_0;
                self.latch_1 = s.latch_1;
                self.chr_fd_0000_bank = s.chr_fd_0000_bank;
                self.chr_fe_0000_bank = s.chr_fe_0000_bank;
                self.chr_fd_1000_bank = s.chr_fd_1000_bank;
                self.chr_fe_1000_bank = s.chr_fe_1000_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant for MMC4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 10,
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

    // Stamp offset 0 of every 16 KB PRG bank and every 4 KB CHR bank with
    // its own index so a windowed read reveals the mapped bank.
    fn stamped_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        let mut c = test_cart(prg_16k_banks, chr_8k_banks);
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        for i in 0..prg_16k_banks as usize {
            c.prg_rom[i * bank_size] = i as u8;
        }
        let chr_4k = c.chr.len() / 0x1000;
        for i in 0..chr_4k {
            c.chr[i * 0x1000] = i as u8;
        }
        c
    }

    // The $8000-$BFFF window maps whichever 16 KB bank was written to $A000.
    #[test]
    fn prg_bank_select_low_window() {
        let mut m = Mapper10::new(stamped_cart(4, 2));
        m.prg_write_byte(0xA000, 2);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        m.prg_write_byte(0xA000, 1);
        assert_eq!(m.prg_read_byte(0x8000), 1);
    }

    // The last 16 KB PRG bank is fixed at $C000.
    #[test]
    fn prg_last_bank_fixed() {
        let mut m = Mapper10::new(stamped_cart(4, 2));
        m.prg_write_byte(0xA000, 0);
        assert_eq!(m.prg_read_byte(0xC000), 3);
        m.prg_write_byte(0xA000, 2);
        assert_eq!(m.prg_read_byte(0xC000), 3);
    }

    // PRG bank index (masked to 4 bits) wraps modulo the bank count.
    #[test]
    fn prg_bank_wraps() {
        let mut m = Mapper10::new(stamped_cart(4, 2));
        m.prg_write_byte(0xA000, 5); // 5 & 0x0F = 5, 5 % 4 = 1
        assert_eq!(m.prg_read_byte(0x8000), 1);
    }

    // $6000-$7FFF PRG RAM writes read back.
    #[test]
    fn prg_ram_read_write() {
        let mut m = Mapper10::new(stamped_cart(2, 2));
        m.prg_write_byte(0x6000, 0x77);
        assert_eq!(m.prg_read_byte(0x6000), 0x77);
    }

    // Power-on latch is 0xFE; a $0FD8-$0FDF fetch flips the $0000 latch to
    // FD and a $0FE8-$0FEF fetch flips it back, changing the live CHR bank.
    #[test]
    fn chr_latch_0_switches_bank() {
        let mut m = Mapper10::new(stamped_cart(2, 2)); // 4 4K CHR banks
        m.prg_write_byte(0xC000, 0); // FE bank for $0000 -> 0
        m.prg_write_byte(0xB000, 1); // FD bank for $0000 -> 1
        assert_eq!(m.chr_read_byte(0x0000), 0); // latch = FE
        let _ = m.chr_read_byte(0x0FD8); // flip to FD
        assert_eq!(m.chr_read_byte(0x0000), 1);
        let _ = m.chr_read_byte(0x0FE8); // flip back to FE
        assert_eq!(m.chr_read_byte(0x0000), 0);
    }

    // The $1000 pattern table has its own latch driven by $1FD8 / $1FE8.
    #[test]
    fn chr_latch_1_switches_bank() {
        let mut m = Mapper10::new(stamped_cart(2, 2));
        m.prg_write_byte(0xE000, 2); // FE bank for $1000 -> 2
        m.prg_write_byte(0xD000, 3); // FD bank for $1000 -> 3
        assert_eq!(m.chr_read_byte(0x1000), 2);
        let _ = m.chr_read_byte(0x1FD8);
        assert_eq!(m.chr_read_byte(0x1000), 3);
        let _ = m.chr_read_byte(0x1FE8);
        assert_eq!(m.chr_read_byte(0x1000), 2);
    }

    // The latch flip happens AFTER the triggering fetch: that fetch still
    // returns a byte read through the pre-flip bank.
    #[test]
    fn chr_latch_flip_is_post_fetch() {
        let mut c = test_cart(2, 2);
        c.chr[0x0FD8] = 0xAA; // bank 0 (FE) at offset 0x0FD8
        c.chr[0x1000 + 0x0FD8] = 0xBB; // bank 1 (FD) at offset 0x0FD8
        let mut m = Mapper10::new(c);
        m.prg_write_byte(0xC000, 0); // FE -> bank 0
        m.prg_write_byte(0xB000, 1); // FD -> bank 1
        assert_eq!(m.chr_read_byte(0x0FD8), 0xAA); // pre-flip bank, then flips
        assert_eq!(m.chr_read_byte(0x0FD8), 0xBB); // now sees FD bank
    }

    // The FD/FE CHR bank registers are masked to 5 bits.
    #[test]
    fn chr_bank_registers_masked() {
        let mut m = Mapper10::new(stamped_cart(2, 2));
        m.prg_write_byte(0xB000, 0xFF); // fd_0000 = 0x1F
        m.prg_write_byte(0xC000, 0xE1); // fe_0000 = 0x01
        match m.get_state() {
            mapper::State::State10(s) => {
                assert_eq!(s.chr_fd_0000_bank, 0x1F);
                assert_eq!(s.chr_fe_0000_bank, 0x01);
            }
            _ => panic!("wrong state variant"),
        }
    }

    // $F000 bit 0 selects mirroring (0 -> Vertical, 1 -> Horizontal).
    #[test]
    fn mirroring_control() {
        let mut m = Mapper10::new(stamped_cart(2, 2));
        m.prg_write_byte(0xF000, 0);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0xF000, 1);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }

    // PRG bank, CHR bank registers, and latch state round-trip through state.
    #[test]
    fn state_round_trip() {
        let mut m = Mapper10::new(stamped_cart(4, 2));
        m.prg_write_byte(0xA000, 2);
        m.prg_write_byte(0xC000, 0); // fe_0000 -> 0
        m.prg_write_byte(0xB000, 1); // fd_0000 -> 1
        let _ = m.chr_read_byte(0x0FD8); // latch_0 -> FD
        let snap = m.get_state();
        m.prg_write_byte(0xA000, 0);
        let _ = m.chr_read_byte(0x0FE8); // latch_0 -> FE
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        // snapshot had latch_0 = FD, so $0000 reads the FD bank (1).
        assert_eq!(m.chr_read_byte(0x0000), 1);
    }

    // reset() restores prg bank 0, both latches to FE, and default mirroring.
    #[test]
    fn reset_restores_power_on() {
        let mut c = stamped_cart(4, 2);
        c.default_mirroring = Mirroring::Horizontal;
        c.mirroring = Mirroring::Horizontal;
        let mut m = Mapper10::new(c);
        m.prg_write_byte(0xA000, 3);
        m.prg_write_byte(0xB000, 1);
        m.prg_write_byte(0xC000, 0);
        let _ = m.chr_read_byte(0x0FD8); // latch_0 -> FD
        m.prg_write_byte(0xF000, 0); // vertical
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.chr_read_byte(0x0000), 0); // latch back to FE -> bank 0
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }
}
