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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 19,
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

    // Stamp the first byte of every 8 KB PRG bank with its index so a read
    // through a mapped window reveals which bank is currently selected.
    fn stamp_prg(cart: &mut Cartridge) {
        let banks = cart.prg_rom.len() / PRG_BANK;
        for i in 0..banks {
            cart.prg_rom[i * PRG_BANK] = i as u8;
        }
    }

    // Stamp the first byte of every 1 KB CHR bank with its index.
    fn stamp_chr(cart: &mut Cartridge) {
        let banks = cart.chr.len() / CHR_BANK;
        for i in 0..banks {
            cart.chr[i * CHR_BANK] = i as u8;
        }
    }

    fn build() -> Mapper19 {
        let mut cart = test_cart(4, 2); // 8 PRG 8K-banks, 16 CHR 1K-banks
        stamp_prg(&mut cart);
        stamp_chr(&mut cart);
        Mapper19::new(cart)
    }

    // The $8000 window follows the register at $E000-$E7FF.
    #[test]
    fn prg_bank0_switches_at_8000() {
        let mut m = build();
        m.prg_write_byte(0xE000, 3);
        assert_eq!(m.prg_read_byte(0x8000), 3);
    }

    // The $A000 window follows the register at $E800-$EFFF.
    #[test]
    fn prg_bank1_switches_at_a000() {
        let mut m = build();
        m.prg_write_byte(0xE800, 5);
        assert_eq!(m.prg_read_byte(0xA000), 5);
    }

    // The $C000 window follows the register at $F000-$F7FF.
    #[test]
    fn prg_bank2_switches_at_c000() {
        let mut m = build();
        m.prg_write_byte(0xF000, 6);
        assert_eq!(m.prg_read_byte(0xC000), 6);
    }

    // The $E000 window is hard-wired to the last PRG bank.
    #[test]
    fn prg_last_bank_fixed_at_e000() {
        let mut m = build();
        assert_eq!(m.prg_read_byte(0xE000), 7); // last of 8 banks
        m.prg_write_byte(0xF000, 1); // touching another window must not move it
        assert_eq!(m.prg_read_byte(0xE000), 7);
    }

    // A bank index beyond the ROM wraps (mod bank count) instead of OOB.
    #[test]
    fn prg_bank_index_wraps() {
        let mut m = build();
        m.prg_write_byte(0xE000, 10); // 10 & 0x3F = 10, 10 % 8 == 2
        assert_eq!(m.prg_read_byte(0x8000), 2);
    }

    // Each 1 KB CHR window follows its per-slot register at $8000 + slot*0x800.
    #[test]
    fn chr_bank_switches_per_slot() {
        let mut m = build();
        m.prg_write_byte(0x8000, 5); // slot 0 -> bank 5
        assert_eq!(m.chr_read_byte(0x0000), 5);
        m.prg_write_byte(0x8800, 9); // slot 1 -> bank 9
        assert_eq!(m.chr_read_byte(0x0400), 9);
        m.prg_write_byte(0xB000, 2); // slot 6 -> bank 2
        assert_eq!(m.chr_read_byte(0x1800), 2);
    }

    // A CHR bank index beyond the ROM wraps (mod bank count).
    #[test]
    fn chr_bank_index_wraps() {
        let mut m = build();
        m.prg_write_byte(0x8000, 20); // 20 % 16 == 4
        assert_eq!(m.chr_read_byte(0x0000), 4);
    }

    // The IRQ line stays low while the enable bit is clear.
    #[test]
    fn irq_disabled_never_fires() {
        let mut m = build();
        for _ in 0..1000 {
            m.on_scanline_tick();
        }
        assert!(!m.irq_pending());
    }

    // The counter climbs to 0x7FFF, asserts the IRQ, and a port write acks it.
    #[test]
    fn irq_fires_then_acks() {
        let mut m = build();
        m.prg_write_byte(0x5000, 0xCD); // counter low
        m.prg_write_byte(0x5800, 0xFF); // counter high 0x7F + enable
        assert!(!m.irq_pending());
        m.on_scanline_tick(); // 0x7FCD + 113 crosses 0x7FFF
        assert!(m.irq_pending());
        m.prg_write_byte(0x5000, 0x00); // ack via a counter write
        assert!(!m.irq_pending());
    }

    // The IRQ counter/enable ports read back what was written.
    #[test]
    fn irq_ports_read_back() {
        let mut m = build();
        m.prg_write_byte(0x5000, 0xCD);
        m.prg_write_byte(0x5800, 0xFF);
        assert_eq!(m.prg_read_byte(0x5000), 0xCD);
        assert_eq!(m.prg_read_byte(0x5800), 0xFF); // 0x7F | enable(0x80)
    }

    // PRG RAM at $6000 is writable until $F800 sets the protect bit.
    #[test]
    fn prg_ram_write_protect() {
        let mut m = build();
        m.prg_write_byte(0x6000, 0x42);
        assert_eq!(m.prg_read_byte(0x6000), 0x42);
        m.prg_write_byte(0xF800, 0x40); // prg_ram_protect = 0x40 >> 6 == 1
        m.prg_write_byte(0x6000, 0x99); // blocked
        assert_eq!(m.prg_read_byte(0x6000), 0x42);
    }

    // Snapshot/restore returns bank and IRQ state to the captured values.
    #[test]
    fn state_round_trip() {
        let mut m = build();
        m.prg_write_byte(0xE000, 3); // prg slot 0 -> bank 3
        m.prg_write_byte(0x8000, 7); // chr slot 0 -> bank 7
        m.prg_write_byte(0x5000, 0x11);
        m.prg_write_byte(0x5800, 0x82); // enable + high bits
        let snap = m.get_state();

        m.prg_write_byte(0xE000, 1);
        m.prg_write_byte(0x8000, 0);
        m.prg_write_byte(0x5000, 0x00);
        m.prg_write_byte(0x5800, 0x00);

        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 3);
        assert_eq!(m.chr_read_byte(0x0000), 7);
        assert_eq!(m.prg_read_byte(0x5000), 0x11);
        assert_eq!(m.prg_read_byte(0x5800), 0x82);
    }

    // reset returns to power-on banking (bank 0 at $8000) with IRQ disabled.
    #[test]
    fn reset_clears_banks_and_irq() {
        let mut m = build();
        m.prg_write_byte(0xE000, 4);
        m.prg_write_byte(0x5800, 0x82);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert!(!m.irq_pending());
        assert_eq!(m.prg_read_byte(0x5800) & 0x80, 0); // enable bit cleared
    }

    // Documents the current register map: $B800 writes nametable slot 0, not
    // CHR bank register 7, so CHR pattern slot 7 ($1C00-$1FFF) stays on its
    // power-on bank and cannot be re-banked.
    // SUSPECTED BUG: on real N163 hardware $B800 is CHR bank register 7 and
    // the four nametable registers begin at $C000 ($C000/$C800/$D000/$D800).
    // Here they are shifted one slot low ($B800/$C000/$C800/$D000), leaving
    // CHR register 7 unreachable.
    #[test]
    fn b800_writes_nametable_not_chr_slot7() {
        let mut m = build();
        m.prg_write_byte(0xB800, 0x0A);
        assert_eq!(m.nt_banks[0], 0x0A);
        assert_eq!(m.chr_read_byte(0x1C00), 0); // slot 7 unchanged
    }
}
