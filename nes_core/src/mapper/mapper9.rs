use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper9 {
    cartridge: Cartridge,

    latch_0: u8,
    latch_1: u8,
    prg_rom_switchable_bank: u8,
    prg_rom_fixed_bank_1: u8,
    prg_rom_fixed_bank_2: u8,
    prg_rom_fixed_bank_3: u8,
    chr_fd_0000_bank: u8,
    chr_fe_0000_bank: u8,
    chr_fd_1000_bank: u8,
    chr_fe_1000_bank: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub latch_0: u8,
    pub latch_1: u8,
    pub prg_rom_switchable_bank: u8,
    pub prg_rom_fixed_bank_1: u8,
    pub prg_rom_fixed_bank_2: u8,
    pub prg_rom_fixed_bank_3: u8,
    pub chr_fd_0000_bank: u8,
    pub chr_fe_0000_bank: u8,
    pub chr_fd_1000_bank: u8,
    pub chr_fe_1000_bank: u8,
}

impl Mapper9 {
    pub fn new(cartridge: Cartridge) -> Self {
        let prg_rom_fixed_bank_1 = cartridge.prg_rom_num_banks * 2 - 3;
        let prg_rom_fixed_bank_2 = cartridge.prg_rom_num_banks * 2 - 2;
        let prg_rom_fixed_bank_3 = cartridge.prg_rom_num_banks * 2 - 1;
        let mut m = Mapper9 {
            cartridge,
            prg_rom_switchable_bank: 0,
            prg_rom_fixed_bank_1,
            prg_rom_fixed_bank_2,
            prg_rom_fixed_bank_3,
            latch_0: 0,
            latch_1: 0,
            chr_fd_0000_bank: 0,
            chr_fe_0000_bank: 0,
            chr_fd_1000_bank: 0,
            chr_fe_1000_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_rom_address(bank: u8, address: u16) -> usize {
        (bank as usize * 0x2000) | (address as usize & 0x1FFF)
    }

    // Flat $8000-$FFFF image for the ASM CPU fast path: one 8 KB
    // switchable bank followed by the three fixed top banks. Rebuilt
    // in place on the (rare) $A000-$AFFF bank write, so the Vec's
    // data pointer handed out via prg_asm_ptr() stays stable.
    fn rebuild_asm_window(&mut self) {
        let bank_size = 0x2000usize;
        let banks = [
            self.prg_rom_switchable_bank,
            self.prg_rom_fixed_bank_1,
            self.prg_rom_fixed_bank_2,
            self.prg_rom_fixed_bank_3,
        ];
        let prg = &self.cartridge.prg_rom;
        for (slot, bank) in banks.iter().enumerate() {
            let off = *bank as usize * bank_size;
            if off + bank_size <= prg.len() {
                self.prg_asm_window[slot * bank_size..(slot + 1) * bank_size]
                    .copy_from_slice(&prg[off..off + bank_size]);
            }
        }
    }

    fn chr_address(&self, address: u16) -> usize {
        let bank = if address < 0x1000 {
            if self.latch_0 == 0xFD {
                self.chr_fd_0000_bank
            } else {
                self.chr_fe_0000_bank
            }
        } else if self.latch_1 == 0xFD {
            self.chr_fd_1000_bank
        } else {
            self.chr_fe_1000_bank
        };

        (bank as usize * 0x1000) | (address as usize & 0x0FFF)
    }
}

impl Mapper for Mapper9 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else if address < 0x8000 {
            if self.cartridge.prg_ram.is_empty() {
                0
            } else {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx]
            }
        } else if address < 0xA000 {
            let rom_addr = Mapper9::prg_rom_address(self.prg_rom_switchable_bank, address);
            self.cartridge.prg_rom[rom_addr]
        } else if address < 0xC000 {
            let rom_addr = Mapper9::prg_rom_address(self.prg_rom_fixed_bank_1, address);
            self.cartridge.prg_rom[rom_addr]
        } else if address < 0xE000 {
            let rom_addr = Mapper9::prg_rom_address(self.prg_rom_fixed_bank_2, address);
            self.cartridge.prg_rom[rom_addr]
        } else {
            let rom_addr = Mapper9::prg_rom_address(self.prg_rom_fixed_bank_3, address);
            self.cartridge.prg_rom[rom_addr]
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address < 0x6000 {
        } else if address < 0x8000 {
            if !self.cartridge.prg_ram.is_empty() {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx] = value;
            }
        } else if address < 0xA000 {
        } else if address < 0xB000 {
            self.prg_rom_switchable_bank = value & 0x0F;
            self.rebuild_asm_window();
        } else if address < 0xC000 {
            self.chr_fd_0000_bank = value & 0x1F;
        } else if address < 0xD000 {
            self.chr_fe_0000_bank = value & 0x1F;
        } else if address < 0xE000 {
            self.chr_fd_1000_bank = value & 0x1F;
        } else if address < 0xF000 {
            self.chr_fe_1000_bank = value & 0x1F;
        } else if value & 0x01 == 0 {
            self.cartridge.mirroring = Mirroring::Vertical;
        } else {
            self.cartridge.mirroring = Mirroring::Horizontal;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr_addr = self.chr_address(address);
        let value = self.cartridge.chr[chr_addr];

        // Latch should be updated AFTER the byte is fetched
        if address == 0x0FD8 {
            self.latch_0 = 0xFD;
        } else if address == 0x0FE8 {
            self.latch_0 = 0xFE;
        } else if (0x1FD8..=0x1FDF).contains(&address) {
            self.latch_1 = 0xFD;
        } else if (0x1FE8..=0x1FEF).contains(&address) {
            self.latch_1 = 0xFE;
        }

        value
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr] = value
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.prg_rom_switchable_bank = 0;
        self.latch_0 = 0;
        self.latch_1 = 0;
        self.chr_fd_0000_bank = 0;
        self.chr_fe_0000_bank = 0;
        self.chr_fd_1000_bank = 0;
        self.chr_fe_1000_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State9(State {
            cartridge: self.cartridge.get_state(),
            latch_0: self.latch_0,
            latch_1: self.latch_1,
            prg_rom_switchable_bank: self.prg_rom_switchable_bank,
            prg_rom_fixed_bank_1: self.prg_rom_fixed_bank_1,
            prg_rom_fixed_bank_2: self.prg_rom_fixed_bank_2,
            prg_rom_fixed_bank_3: self.prg_rom_fixed_bank_3,
            chr_fd_0000_bank: self.chr_fd_0000_bank,
            chr_fe_0000_bank: self.chr_fe_0000_bank,
            chr_fd_1000_bank: self.chr_fd_1000_bank,
            chr_fe_1000_bank: self.chr_fe_1000_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State9(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.latch_0 = state.latch_0;
                self.latch_1 = state.latch_1;
                self.prg_rom_switchable_bank = state.prg_rom_switchable_bank;
                self.prg_rom_fixed_bank_1 = state.prg_rom_fixed_bank_1;
                self.prg_rom_fixed_bank_2 = state.prg_rom_fixed_bank_2;
                self.prg_rom_fixed_bank_3 = state.prg_rom_fixed_bank_3;
                self.chr_fd_0000_bank = state.chr_fd_0000_bank;
                self.chr_fe_0000_bank = state.chr_fe_0000_bank;
                self.chr_fd_1000_bank = state.chr_fd_1000_bank;
                self.chr_fe_1000_bank = state.chr_fe_1000_bank;
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

    // Punch-Out!!-shaped cart: 128 KB PRG (16 x 8 KB banks), 128 KB CHR.
    fn test_cart() -> Cartridge {
        let prg_len = 128 * 1024;
        let mut prg_rom = vec![0u8; prg_len];
        // Tag every byte with (8 KB bank index, low offset byte) so a
        // mis-mapped window can't accidentally match.
        for (i, b) in prg_rom.iter_mut().enumerate() {
            *b = ((i / 0x2000) as u8) ^ (i as u8).rotate_left(3);
        }
        Cartridge {
            mapper: 9,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: 8,
            prg_rom,
            chr_num_banks: 16,
            chr: vec![0u8; 128 * 1024],
            prg_ram: Vec::new(),
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    fn window(m: &Mapper9) -> &[u8] {
        &m.prg_asm_window
    }

    fn assert_window_matches_peek(m: &Mapper9) {
        let win = window(m);
        for address in 0x8000..=0xFFFFu16 {
            assert_eq!(
                win[(address - 0x8000) as usize],
                m.prg_peek_byte(address),
                "asm window diverges from prg_peek_byte at ${address:04X}",
            );
        }
    }

    #[test]
    fn asm_window_tracks_prg_bank_switches() {
        let mut m = Mapper9::new(test_cart());
        assert!(m.prg_asm_ptr().is_some());
        assert_window_matches_peek(&m);
        for bank in [1u8, 7, 12, 15, 0] {
            m.prg_write_byte(0xA000, bank);
            assert_window_matches_peek(&m);
        }
        // CHR / mirroring registers must not disturb the PRG window.
        let before = window(&m).to_vec();
        m.prg_write_byte(0xB000, 3);
        m.prg_write_byte(0xE000, 5);
        m.prg_write_byte(0xF000, 1);
        assert_eq!(window(&m), &before[..]);
        assert_window_matches_peek(&m);
    }

    #[test]
    fn asm_window_rebuilt_on_reset_and_apply_state() {
        let mut m = Mapper9::new(test_cart());
        m.prg_write_byte(0xA000, 9);
        let saved = m.get_state();
        let saved_window = window(&m).to_vec();

        m.prg_write_byte(0xA000, 4);
        assert_ne!(window(&m), &saved_window[..]);

        m.apply_state(&saved);
        assert_eq!(window(&m), &saved_window[..]);
        assert_window_matches_peek(&m);

        m.reset();
        assert_eq!(m.prg_rom_switchable_bank, 0);
        assert_window_matches_peek(&m);
    }
}
