use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Sunsoft FME-7 (mapper 69). 8 KB PRG banks × 4 windows; last fixed.
// 1 KB CHR banks × 8. 16-bit IRQ counter decremented per CPU cycle.
//
// We approximate per-CPU-cycle decrement by stepping the counter once
// per `on_scanline_tick` (≈114 CPU cycles). Good enough for every game
// that uses FME-7 IRQ for scanline-timed raster effects; tight-timing
// games get minor jitter. Sunsoft 5B extra audio is currently muted.
const PRG_BANK: usize = 0x2000; // 8 KB
const CHR_BANK: usize = 0x0400; // 1 KB

pub struct Mapper69 {
    cartridge: Cartridge,
    command: u8,
    // CHR banks 0-7 (each 1 KB at $0000, $0400, ... $1C00)
    chr_banks: [u8; 8],
    // PRG $6000 slot: bit7 = RAM enable, bit6 = use ROM (not RAM), bits 5-0 = bank
    prg_6000: u8,
    // PRG banks for windows $8000/$A000/$C000 (8 KB each)
    prg_banks: [u8; 3],
    irq_enabled: bool,
    irq_counter_enabled: bool,
    irq_counter: u16,
    irq_pending: bool,
    // Flat $8000-$FFFF view for the ASM fast path.
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub command: u8,
    pub chr_banks: [u8; 8],
    pub prg_6000: u8,
    pub prg_banks: [u8; 3],
    pub irq_enabled: bool,
    pub irq_counter_enabled: bool,
    pub irq_counter: u16,
    pub irq_pending: bool,
}

impl Mapper69 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper69 {
            cartridge,
            command: 0,
            chr_banks: [0; 8],
            prg_6000: 0,
            prg_banks: [0; 3],
            irq_enabled: false,
            irq_counter_enabled: false,
            irq_counter: 0,
            irq_pending: false,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_rom_bank_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK).max(1)
    }

    fn chr_bank_count(&self) -> usize {
        (self.cartridge.chr.len() / CHR_BANK).max(1)
    }

    fn prg_bank_offset(&self, bank_idx: u8) -> usize {
        (bank_idx as usize % self.prg_rom_bank_count()) * PRG_BANK
    }

    fn rebuild_asm_window(&mut self) {
        let last = self.prg_rom_bank_count() - 1;
        let windows = [
            self.prg_bank_offset(self.prg_banks[0]),
            self.prg_bank_offset(self.prg_banks[1]),
            self.prg_bank_offset(self.prg_banks[2]),
            last * PRG_BANK,
        ];
        let prg = &self.cartridge.prg_rom;
        for (i, off) in windows.iter().enumerate() {
            let dst = i * PRG_BANK;
            if *off + PRG_BANK <= prg.len() {
                self.prg_asm_window[dst..dst + PRG_BANK]
                    .copy_from_slice(&prg[*off..*off + PRG_BANK]);
            }
        }
    }
}

impl Mapper for Mapper69 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7FFF => {
                // $6000 window can be PRG RAM or PRG ROM. Bit 6 = 1 for
                // ROM select; bit 7 = 1 for RAM enable.
                if self.prg_6000 & 0x40 != 0 {
                    let bank = self.prg_6000 & 0x3F;
                    let off = self.prg_bank_offset(bank) | (address as usize & 0x1FFF);
                    if off < self.cartridge.prg_rom.len() {
                        self.cartridge.prg_rom[off]
                    } else {
                        0
                    }
                } else if self.prg_6000 & 0x80 != 0 {
                    let i = (address - 0x6000) as usize;
                    self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
                } else {
                    0
                }
            }
            0x8000..=0xDFFF => {
                let slot = ((address - 0x8000) / 0x2000) as usize;
                let bank = self.prg_banks[slot];
                let off = self.prg_bank_offset(bank) | (address as usize & 0x1FFF);
                self.cartridge.prg_rom[off]
            }
            0xE000..=0xFFFF => {
                let last = self.prg_rom_bank_count() - 1;
                self.cartridge.prg_rom[last * PRG_BANK | (address as usize & 0x1FFF)]
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x6000..=0x7FFF => {
                if self.prg_6000 & 0xC0 == 0x80 {
                    // RAM enabled and ROM not selected.
                    let i = (address - 0x6000) as usize;
                    if i < self.cartridge.prg_ram.len() {
                        self.cartridge.prg_ram[i] = value;
                    }
                }
            }
            0x8000..=0x9FFF => {
                self.command = value & 0x0F;
            }
            0xA000..=0xBFFF => match self.command {
                0..=7 => {
                    self.chr_banks[self.command as usize] = value;
                }
                8 => {
                    self.prg_6000 = value;
                }
                9 => {
                    self.prg_banks[0] = value & 0x3F;
                    self.rebuild_asm_window();
                }
                10 => {
                    self.prg_banks[1] = value & 0x3F;
                    self.rebuild_asm_window();
                }
                11 => {
                    self.prg_banks[2] = value & 0x3F;
                    self.rebuild_asm_window();
                }
                12 => {
                    self.cartridge.mirroring = match value & 0x03 {
                        0 => Mirroring::Vertical,
                        1 => Mirroring::Horizontal,
                        2 => Mirroring::OneScreenLower,
                        _ => Mirroring::OneScreenUpper,
                    };
                }
                13 => {
                    self.irq_enabled = value & 0x01 != 0;
                    self.irq_counter_enabled = value & 0x80 != 0;
                    self.irq_pending = false;
                }
                14 => {
                    self.irq_counter = (self.irq_counter & 0xFF00) | value as u16;
                }
                15 => {
                    self.irq_counter = (self.irq_counter & 0x00FF) | ((value as u16) << 8);
                }
                _ => {}
            },
            0xC000..=0xDFFF => {
                // Sunsoft 5B audio command register — muted for now.
            }
            0xE000..=0xFFFF => {
                // Sunsoft 5B audio data register — muted.
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_bank_count();
        let off = bank * CHR_BANK | (address as usize & 0x3FF);
        self.cartridge.chr[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_bank_count();
        let off = bank * CHR_BANK | (address as usize & 0x3FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.command = 0;
        self.chr_banks = [0; 8];
        self.prg_6000 = 0;
        self.prg_banks = [0; 3];
        self.irq_enabled = false;
        self.irq_counter_enabled = false;
        self.irq_counter = 0;
        self.irq_pending = false;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn on_scanline_tick(&mut self) {
        // Approximate: one scanline ≈ 114 CPU cycles.
        if self.irq_counter_enabled {
            let (new, wrap) = self.irq_counter.overflowing_sub(114);
            self.irq_counter = new;
            if wrap && self.irq_enabled {
                self.irq_pending = true;
            }
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_pending
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        // Small bulk to keep IRQ latency tight.
        2
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State69(State {
            cartridge: self.cartridge.get_state(),
            command: self.command,
            chr_banks: self.chr_banks,
            prg_6000: self.prg_6000,
            prg_banks: self.prg_banks,
            irq_enabled: self.irq_enabled,
            irq_counter_enabled: self.irq_counter_enabled,
            irq_counter: self.irq_counter,
            irq_pending: self.irq_pending,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State69(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.command = state.command;
                self.chr_banks = state.chr_banks;
                self.prg_6000 = state.prg_6000;
                self.prg_banks = state.prg_banks;
                self.irq_enabled = state.irq_enabled;
                self.irq_counter_enabled = state.irq_counter_enabled;
                self.irq_counter = state.irq_counter;
                self.irq_pending = state.irq_pending;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
