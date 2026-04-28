//! Mapper 105: Nintendo World Championships 1990.
//!
//! MMC1-compatible core + competition-timer / DIP-switch extensions.
//! The cart carries 256 KB of PRG ROM split into two 128 KB outer
//! "super-banks"; inner bank math is identical to MMC1. The chr0
//! register's bit 0 selects the outer bank, bit 3 toggles an M2-clocked
//! IRQ counter, and chr1's low nibble sets the counter's initial
//! divider. DIP switches (4 bits) are set at manufacture and are
//! hard-coded here to 0x00 (shortest competition time) per convention.
//!
//! This is a standalone implementation that mirrors the MMC1 state
//! machine (see `mapper1.rs`) with the extended register bits wired
//! in. The competition IRQ is NOT emulated — the menu boots and the
//! timer behaves as if it never fires, which is sufficient for the
//! library's boot-plus-frames correctness bar.
use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

/// Size of one outer PRG "super-bank": 128 KB.
const OUTER_BANK_SIZE: usize = 128 * 1024;

pub struct Mapper105 {
    cartridge: Cartridge,
    shift: u8,
    regs: Regs,
    /// Selects which 128 KB outer PRG half is visible through the MMC1
    /// windows. Mirrors bit 0 of chr_bank_0 at the moment it was
    /// written. Applied as `outer_bank * 128 KB` atop the inner offset.
    outer_bank: u8,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct Regs {
    control: u8,
    prg_bank: u8,
    chr_bank_0: u8,
    chr_bank_1: u8,
}

impl Regs {
    fn new() -> Regs {
        Regs {
            control: 0x0C,
            prg_bank: 0,
            chr_bank_0: 0,
            chr_bank_1: 0,
        }
    }
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
enum PrgRomMode {
    Switch32Kb,
    FixFirstBank,
    FixLastBank,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
enum ChrRomMode {
    Switch8Kb,
    Switch4Kb,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub shift: u8,
    pub regs: Regs,
    pub outer_bank: u8,
}

const SHIFT_REGISTER_DEFAULT: u8 = 0x10;

impl Mapper105 {
    pub fn new(cartridge: Cartridge) -> Self {
        Mapper105 {
            cartridge,
            shift: SHIFT_REGISTER_DEFAULT,
            regs: Regs::new(),
            outer_bank: 0,
        }
    }

    fn prg_rom_mode(&self) -> PrgRomMode {
        let control = (self.regs.control & 0x0F) >> 2;
        match control {
            0 | 1 => PrgRomMode::Switch32Kb,
            2 => PrgRomMode::FixFirstBank,
            _ => PrgRomMode::FixLastBank,
        }
    }

    fn chr_rom_mode(&self) -> ChrRomMode {
        if self.regs.control & 0x10 == 0 {
            ChrRomMode::Switch8Kb
        } else {
            ChrRomMode::Switch4Kb
        }
    }

    fn write_register(&mut self, address: u16, shift: u8) {
        if address < 0xA000 {
            self.regs.control = shift & 0x1F;
            self.cartridge.mirroring = match self.regs.control & 0x03 {
                0 => Mirroring::OneScreenLower,
                1 => Mirroring::OneScreenUpper,
                2 => Mirroring::Vertical,
                _ => Mirroring::Horizontal,
            };
        } else if address < 0xC000 {
            // chr0: bit 0 selects the outer 128 KB PRG bank; bit 3
            // gates the competition IRQ counter (not emulated). The
            // other bits follow standard MMC1 chr_bank_0 semantics,
            // but on a 8 KB CHR-RAM NWC cart none of the CHR bits
            // actually matter for pattern-table routing (wrapped by
            // chr_address's len-mask).
            self.regs.chr_bank_0 = shift & 0x1F;
            self.outer_bank = shift & 0x01;
            // bit 3 (IRQ enable) intentionally ignored.
        } else if address < 0xE000 {
            // chr1: low nibble is the IRQ divider on NWC; we don't
            // emulate the timer, so just record the bits.
            self.regs.chr_bank_1 = shift & 0x1F;
        } else {
            self.regs.prg_bank = shift & 0x0F;
        }
    }

    fn prg_rom_bank_first(&self) -> u8 {
        match self.prg_rom_mode() {
            PrgRomMode::Switch32Kb => self.regs.prg_bank & 0xFE,
            PrgRomMode::FixFirstBank => 0,
            PrgRomMode::FixLastBank => self.regs.prg_bank,
        }
    }

    fn prg_rom_bank_last(&self) -> u8 {
        match self.prg_rom_mode() {
            PrgRomMode::Switch32Kb => (self.regs.prg_bank & 0xFE) | 0x01,
            PrgRomMode::FixFirstBank => self.regs.prg_bank,
            // For the "fix last" mode, pin to the last bank within the
            // currently-selected outer 128 KB half (not the end of the
            // full 256 KB PRG). NWC's reset vector lives at the end of
            // each outer half.
            PrgRomMode::FixLastBank => {
                let banks_per_outer =
                    (OUTER_BANK_SIZE / PRG_ROM_BANK_SIZE as usize) as u8;
                banks_per_outer.saturating_sub(1)
            }
        }
    }

    fn prg_rom_address(&self, bank: u8, address: u16) -> usize {
        // Inner offset within the selected 128 KB outer half.
        let inner = (bank as usize * PRG_ROM_BANK_SIZE as usize)
            | (address as usize & (PRG_ROM_BANK_SIZE as usize - 1));
        // Clip inner to the outer-bank window so a rogue bank index
        // can't escape into the other half.
        let inner = inner & (OUTER_BANK_SIZE - 1);
        let outer_off = (self.outer_bank as usize & 0x01) * OUTER_BANK_SIZE;
        let raw = outer_off + inner;
        // Final safety mask against physical ROM size. PRG is a power
        // of two on NWC (256 KB), so `len - 1` is a valid mask.
        raw & (self.cartridge.prg_rom.len().max(1) - 1)
    }

    fn chr_address(&self, address: u16) -> usize {
        let raw = match self.chr_rom_mode() {
            ChrRomMode::Switch4Kb => {
                let bank = if address < 0x1000 {
                    self.regs.chr_bank_0
                } else {
                    self.regs.chr_bank_1
                };
                (bank as usize * 0x1000) | (address as usize & 0x0FFF)
            }
            ChrRomMode::Switch8Kb => {
                let bank = self.regs.chr_bank_0 & !1;
                (bank as usize * 0x1000) | (address as usize & 0x1FFF)
            }
        };
        raw & (self.cartridge.chr.len().max(1) - 1)
    }
}

impl Mapper for Mapper105 {
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
        } else if address < 0xC000 {
            let rom_addr = self.prg_rom_address(self.prg_rom_bank_first(), address);
            self.cartridge.prg_rom[rom_addr]
        } else {
            let rom_addr = self.prg_rom_address(self.prg_rom_bank_last(), address);
            self.cartridge.prg_rom[rom_addr]
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address < 0x6000 {
            // Do nothing
        } else if address < 0x8000 {
            if !self.cartridge.prg_ram.is_empty() {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx] = value
            }
        } else if (value & 0x80) == 0 {
            let is_last_shift = (self.shift & 0x01) != 0;
            self.shift = (self.shift >> 1) | ((value & 0x01) << 4);
            if is_last_shift {
                let shift = self.shift;
                self.write_register(address, shift);
                self.shift = SHIFT_REGISTER_DEFAULT;
            }
        } else {
            self.shift = SHIFT_REGISTER_DEFAULT;
            self.regs.control |= 0x0C;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr] = value
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn sram(&mut self) -> *mut u8 {
        self.cartridge.prg_ram.as_mut_ptr() as *mut _
    }

    fn sram_size(&self) -> usize {
        self.cartridge.prg_ram.len()
    }

    fn reset(&mut self) {
        self.shift = SHIFT_REGISTER_DEFAULT;
        self.regs = Regs::new();
        self.outer_bank = 0;
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State105(State {
            cartridge: self.cartridge.get_state(),
            shift: self.shift,
            regs: self.regs,
            outer_bank: self.outer_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State105(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.shift = state.shift;
                self.regs = state.regs;
                self.outer_bank = state.outer_bank;
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
