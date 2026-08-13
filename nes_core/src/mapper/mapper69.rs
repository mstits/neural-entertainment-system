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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 69,
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

    // Stamp offset 0 of every 8 KB PRG bank and every 1 KB CHR bank with
    // its own index so a windowed read reveals the mapped bank.
    fn stamped_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        let mut c = test_cart(prg_16k_banks, chr_8k_banks);
        let prg_8k = c.prg_rom.len() / PRG_BANK;
        for i in 0..prg_8k {
            c.prg_rom[i * PRG_BANK] = i as u8;
        }
        let chr_1k = c.chr.len() / CHR_BANK;
        for i in 0..chr_1k {
            c.chr[i * CHR_BANK] = i as u8;
        }
        c
    }

    // FME-7 register model: select the command at $8000, then the value at
    // $A000.
    fn set_reg(m: &mut Mapper69, command: u8, value: u8) {
        m.prg_write_byte(0x8000, command);
        m.prg_write_byte(0xA000, value);
    }

    fn irq_counter(m: &Mapper69) -> u16 {
        match m.get_state() {
            mapper::State::State69(s) => s.irq_counter,
            _ => panic!("wrong state variant"),
        }
    }

    // Commands 9/10/11 select the $8000/$A000/$C000 8 KB PRG windows.
    #[test]
    fn prg_bank_select_windows() {
        let mut m = Mapper69::new(stamped_cart(4, 2)); // 8 8K banks
        set_reg(&mut m, 9, 1);
        set_reg(&mut m, 10, 2);
        set_reg(&mut m, 11, 3);
        assert_eq!(m.prg_read_byte(0x8000), 1);
        assert_eq!(m.prg_read_byte(0xA000), 2);
        assert_eq!(m.prg_read_byte(0xC000), 3);
    }

    // The $E000 window is hardwired to the last 8 KB PRG bank.
    #[test]
    fn prg_last_bank_fixed() {
        let mut m = Mapper69::new(stamped_cart(4, 2)); // last bank = 7
        assert_eq!(m.prg_read_byte(0xE000), 7);
        set_reg(&mut m, 9, 5); // moving another window must not disturb $E000
        assert_eq!(m.prg_read_byte(0xE000), 7);
    }

    // PRG bank index is masked to 6 bits then wraps modulo bank count.
    #[test]
    fn prg_bank_wraps() {
        let mut m = Mapper69::new(stamped_cart(4, 2)); // 8 banks
        set_reg(&mut m, 9, 9); // 9 & 0x3F = 9, 9 % 8 = 1
        assert_eq!(m.prg_read_byte(0x8000), 1);
    }

    // Command 8 with bit 6 set maps PRG ROM into the $6000 window.
    #[test]
    fn prg_6000_rom_mode() {
        let mut m = Mapper69::new(stamped_cart(4, 2));
        set_reg(&mut m, 8, 0x40 | 3); // ROM select, bank 3
        assert_eq!(m.prg_read_byte(0x6000), 3);
    }

    // Command 8 with bit 7 set (bit 6 clear) maps writable PRG RAM at $6000.
    #[test]
    fn prg_6000_ram_mode() {
        let mut m = Mapper69::new(stamped_cart(4, 2));
        set_reg(&mut m, 8, 0x80);
        m.prg_write_byte(0x6000, 0x5A);
        assert_eq!(m.prg_read_byte(0x6000), 0x5A);
    }

    // With neither ROM nor RAM enabled the $6000 window reads 0.
    #[test]
    fn prg_6000_disabled_reads_zero() {
        let mut m = Mapper69::new(stamped_cart(4, 2));
        set_reg(&mut m, 8, 0x00);
        assert_eq!(m.prg_read_byte(0x6000), 0);
    }

    // A $6000 write only lands when RAM is enabled and ROM deselected.
    #[test]
    fn prg_6000_ram_write_gated() {
        let mut m = Mapper69::new(stamped_cart(4, 2));
        set_reg(&mut m, 8, 0x80); // RAM enabled
        m.prg_write_byte(0x6000, 0x11);
        set_reg(&mut m, 8, 0x00); // RAM disabled
        m.prg_write_byte(0x6000, 0x22); // dropped
        set_reg(&mut m, 8, 0x80); // RAM enabled again
        assert_eq!(m.prg_read_byte(0x6000), 0x11);
    }

    // Commands 0-7 each select an independent 1 KB CHR window.
    #[test]
    fn chr_bank_select_per_slot() {
        let mut m = Mapper69::new(stamped_cart(4, 2)); // 16 1K banks
        for slot in 0..8u8 {
            set_reg(&mut m, slot, slot + 1);
        }
        for slot in 0..8u16 {
            assert_eq!(m.chr_read_byte(slot * 0x400), (slot as u8) + 1);
        }
    }

    // CHR bank index wraps modulo the 1 KB bank count (here 8 banks).
    #[test]
    fn chr_bank_wraps() {
        let mut m = Mapper69::new(stamped_cart(1, 1));
        set_reg(&mut m, 0, 9); // 9 % 8 = 1
        assert_eq!(m.chr_read_byte(0x0000), 1);
    }

    // Command 12 sets mirroring (FME-7 maps 0->Vertical, 1->Horizontal).
    #[test]
    fn mirroring_control() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 12, 0);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        set_reg(&mut m, 12, 1);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        set_reg(&mut m, 12, 2);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
        set_reg(&mut m, 12, 3);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }

    // Each enabled scanline tick decrements the IRQ counter by 114.
    #[test]
    fn irq_counter_decrements_per_tick() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 14, 0xE8); // low byte
        set_reg(&mut m, 15, 0x03); // high byte -> 0x03E8 = 1000
        set_reg(&mut m, 13, 0x80); // counter enabled, IRQ line disabled
        m.on_scanline_tick();
        assert_eq!(irq_counter(&m), 1000 - 114);
    }

    // The IRQ asserts when the counter underflows and the IRQ line is on.
    #[test]
    fn irq_fires_on_underflow() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 14, 100); // < 114 -> first tick underflows
        set_reg(&mut m, 15, 0);
        set_reg(&mut m, 13, 0x81); // counter + IRQ line enabled
        assert!(!m.irq_pending());
        m.on_scanline_tick();
        assert!(m.irq_pending());
    }

    // Writing the control register (command 13) acknowledges a pending IRQ.
    #[test]
    fn irq_ack_clears_pending() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 14, 100);
        set_reg(&mut m, 15, 0);
        set_reg(&mut m, 13, 0x81);
        m.on_scanline_tick();
        assert!(m.irq_pending());
        set_reg(&mut m, 13, 0x00); // ack + disable
        assert!(!m.irq_pending());
    }

    // Underflow with the IRQ line disabled must not assert an IRQ.
    #[test]
    fn irq_line_disabled_no_fire() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 14, 50);
        set_reg(&mut m, 15, 0);
        set_reg(&mut m, 13, 0x80); // counter enabled, IRQ line disabled
        m.on_scanline_tick();
        assert!(!m.irq_pending());
    }

    // With the counter disabled it neither decrements nor fires.
    #[test]
    fn irq_counter_frozen_when_disabled() {
        let mut m = Mapper69::new(stamped_cart(2, 1));
        set_reg(&mut m, 14, 10);
        set_reg(&mut m, 15, 0);
        set_reg(&mut m, 13, 0x01); // IRQ line enabled but counter frozen
        m.on_scanline_tick();
        assert_eq!(irq_counter(&m), 10);
        assert!(!m.irq_pending());
    }

    // Bank plus IRQ counter state survive a get_state/apply_state cycle.
    #[test]
    fn state_round_trip() {
        let mut m = Mapper69::new(stamped_cart(4, 2));
        set_reg(&mut m, 9, 2); // PRG window 0 -> bank 2
        set_reg(&mut m, 0, 5); // CHR slot 0 -> bank 5
        set_reg(&mut m, 14, 0x34);
        set_reg(&mut m, 15, 0x12); // counter = 0x1234
        let snap = m.get_state();
        set_reg(&mut m, 9, 0);
        set_reg(&mut m, 0, 0);
        set_reg(&mut m, 14, 0);
        set_reg(&mut m, 15, 0);
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 2);
        assert_eq!(m.chr_read_byte(0x0000), 5);
        assert_eq!(irq_counter(&m), 0x1234);
    }

    // reset() clears banks and IRQ state and restores default mirroring.
    #[test]
    fn reset_restores_power_on() {
        let mut c = stamped_cart(4, 2);
        c.default_mirroring = Mirroring::Vertical;
        c.mirroring = Mirroring::Vertical;
        let mut m = Mapper69::new(c);
        set_reg(&mut m, 9, 3);
        set_reg(&mut m, 0, 4);
        set_reg(&mut m, 14, 5);
        set_reg(&mut m, 13, 0x81);
        m.on_scanline_tick(); // leaves IRQ pending
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.chr_read_byte(0x0000), 0);
        assert!(!m.irq_pending());
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }
}
