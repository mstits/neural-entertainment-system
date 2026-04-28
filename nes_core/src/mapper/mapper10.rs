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
