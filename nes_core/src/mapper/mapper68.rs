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
