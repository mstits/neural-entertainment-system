// Namco 163 (mapper 19). Wide feature set — 8 KB × 4 PRG banks,
// 1 KB × 12 CHR slots (8 pattern + 4 nametable), battery RAM,
// scanline IRQ, and an 8-channel wavetable synth. Audio is left
// as a follow-up; banking + IRQ are sufficient to boot the N163
// catalog: Rolling Thunder, Mappy Land, Family Tennis, Warrior's
// Legend, Famista '89, Dragon Buster II, etc.
//
// Register layout is address-sparse but falls into four groups:
//   $4800-$4FFF: sound register data port (audio; deferred)
//   $5000-$57FF: IRQ counter low 8 bits
//   $5800-$5FFF: IRQ counter high 7 bits + enable
//   $6000-$7FFF: optional PRG RAM (write protect via $F800)
//   $8000-$B7FF: 8× 1 KB CHR bank registers (bank value 0x00-0xDF = CHR ROM,
//                0xE0-0xFF = nametable RAM when enabled)
//   $B800-$BFFF: nametable 0 bank (advanced mode)
//   $C000-$C7FF: nametable 1 bank
//   $C800-$CFFF: nametable 2 bank
//   $D000-$D7FF: nametable 3 bank
//   $E000-$E7FF: PRG bank for $8000 + sound enable bit
//   $E800-$EFFF: PRG bank for $A000 + PRG RAM high-byte flag
//   $F000-$F7FF: PRG bank for $C000
//   $F800-$FFFF: sound register select + PRG RAM protection

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

const PRG_BANK: usize = 0x2000; // 8 KB
const CHR_BANK: usize = 0x0400; // 1 KB

pub struct Mapper19 {
    cartridge: Cartridge,
    prg_banks: [u8; 3], // $8000, $A000, $C000; $E000 fixed last
    chr_banks: [u8; 8],
    nt_banks: [u8; 4],
    irq_counter: u16,
    irq_enabled: bool,
    irq_pending: bool,
    prg_ram_protect: u8,
    sound_reg_addr: u8,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_banks: [u8; 3],
    pub chr_banks: [u8; 8],
    pub nt_banks: [u8; 4],
    pub irq_counter: u16,
    pub irq_enabled: bool,
    pub irq_pending: bool,
    pub prg_ram_protect: u8,
    pub sound_reg_addr: u8,
}

impl Mapper19 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper19 {
            cartridge,
            prg_banks: [0; 3],
            chr_banks: [0; 8],
            nt_banks: [0; 4],
            irq_counter: 0,
            irq_enabled: false,
            irq_pending: false,
            prg_ram_protect: 0,
            sound_reg_addr: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK).max(1)
    }

    fn chr_count(&self) -> usize {
        (self.cartridge.chr.len() / CHR_BANK).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let last = self.prg_count() - 1;
        let slots = [
            (self.prg_banks[0] & 0x3F) as usize % self.prg_count(),
            (self.prg_banks[1] & 0x3F) as usize % self.prg_count(),
            (self.prg_banks[2] & 0x3F) as usize % self.prg_count(),
            last,
        ];
        let prg = &self.cartridge.prg_rom;
        for (i, &bank) in slots.iter().enumerate() {
            let src = bank * PRG_BANK;
            let dst = i * PRG_BANK;
            if src + PRG_BANK <= prg.len() {
                self.prg_asm_window[dst..dst + PRG_BANK]
                    .copy_from_slice(&prg[src..src + PRG_BANK]);
            }
        }
    }
}

impl Mapper for Mapper19 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x4800..=0x4FFF => 0, // sound data read — audio deferred
            0x5000..=0x57FF => (self.irq_counter & 0xFF) as u8,
            0x5800..=0x5FFF => (((self.irq_counter >> 8) & 0x7F) as u8)
                | if self.irq_enabled { 0x80 } else { 0 },
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xFFFF => {
                let slot = ((address - 0x8000) / 0x2000) as usize;
                let bank = if slot == 3 {
                    self.prg_count() - 1
                } else {
                    (self.prg_banks[slot] & 0x3F) as usize % self.prg_count()
                };
                self.cartridge.prg_rom[bank * PRG_BANK | (address as usize & 0x1FFF)]
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x4800..=0x4FFF => { /* sound register data — deferred */ }
            0x5000..=0x57FF => {
                self.irq_counter = (self.irq_counter & 0xFF00) | value as u16;
                self.irq_pending = false;
            }
            0x5800..=0x5FFF => {
                self.irq_counter = (self.irq_counter & 0x00FF) | (((value as u16) & 0x7F) << 8);
                self.irq_enabled = value & 0x80 != 0;
                self.irq_pending = false;
            }
            0x6000..=0x7FFF => {
                if self.prg_ram_protect & 0x01 == 0 {
                    let i = (address - 0x6000) as usize;
                    if i < self.cartridge.prg_ram.len() {
                        self.cartridge.prg_ram[i] = value;
                    }
                }
            }
            0x8000..=0xB7FF => {
                let slot = ((address - 0x8000) / 0x0800) as usize;
                if slot < 8 {
                    self.chr_banks[slot] = value;
                }
            }
            0xB800..=0xBFFF => self.nt_banks[0] = value,
            0xC000..=0xC7FF => self.nt_banks[1] = value,
            0xC800..=0xCFFF => self.nt_banks[2] = value,
            0xD000..=0xD7FF => self.nt_banks[3] = value,
            0xE000..=0xE7FF => {
                self.prg_banks[0] = value;
                self.rebuild_asm_window();
            }
            0xE800..=0xEFFF => {
                self.prg_banks[1] = value;
                self.rebuild_asm_window();
            }
            0xF000..=0xF7FF => {
                self.prg_banks[2] = value;
                self.rebuild_asm_window();
            }
            0xF800..=0xFFFF => {
                self.sound_reg_addr = value & 0x7F;
                self.prg_ram_protect = value >> 6;
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_count();
        self.cartridge.chr[bank * CHR_BANK | (address as usize & 0x3FF)]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_count();
        let off = bank * CHR_BANK | (address as usize & 0x3FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_banks = [0; 3];
        self.chr_banks = [0; 8];
        self.nt_banks = [0; 4];
        self.irq_counter = 0;
        self.irq_enabled = false;
        self.irq_pending = false;
        self.prg_ram_protect = 0;
        self.sound_reg_addr = 0;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn on_scanline_tick(&mut self) {
        if !self.irq_enabled {
            return;
        }
        // N163 IRQ counter increments every CPU cycle; fires when it
        // reaches 0x7FFF. Scanline approximation: +113 per scanline.
        let new = self.irq_counter.wrapping_add(113);
        if new >= 0x7FFF && self.irq_counter < 0x7FFF {
            self.irq_pending = true;
            self.irq_counter = 0x7FFF;
        } else {
            self.irq_counter = new & 0x7FFF;
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_pending
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        2
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State19(State {
            cartridge: self.cartridge.get_state(),
            prg_banks: self.prg_banks,
            chr_banks: self.chr_banks,
            nt_banks: self.nt_banks,
            irq_counter: self.irq_counter,
            irq_enabled: self.irq_enabled,
            irq_pending: self.irq_pending,
            prg_ram_protect: self.prg_ram_protect,
            sound_reg_addr: self.sound_reg_addr,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State19(s) => {
                self.cartridge.apply_state(&s.cartridge);
                self.prg_banks = s.prg_banks;
                self.chr_banks = s.chr_banks;
                self.nt_banks = s.nt_banks;
                self.irq_counter = s.irq_counter;
                self.irq_enabled = s.irq_enabled;
                self.irq_pending = s.irq_pending;
                self.prg_ram_protect = s.prg_ram_protect;
                self.sound_reg_addr = s.sound_reg_addr;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state variant for N163"),
        }
    }
}
