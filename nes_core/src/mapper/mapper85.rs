// Konami VRC7 (mapper 85). Shares the VRC4-family IRQ semantics
// (8-bit counter, scanline + cycle modes, enable-after-ack). Adds
// a YM2413-derived 6-channel FM synth — audio deferred as a follow-
// up, but banking + IRQ are enough to boot Lagrange Point.
//
// Register layout (mapper 85 uses address bits A4 + A12 to select
// sub-registers; the canonical selector is address & 0xF038):
//   $8000: PRG bank for $8000-$9FFF (6 bits)
//   $8010: PRG bank for $A000-$BFFF (6 bits)
//   $9000: PRG bank for $C000-$DFFF (6 bits)
//   $9010/$9030: FM audio (skip)
//   $A000-$D010: CHR banks 0-7 (1 KB each, 8-bit bank index)
//   $E000: Mirroring (low 2 bits) + WRAM control
//   $E010: IRQ latch
//   $F000: IRQ control
//   $F010: IRQ acknowledge

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

const PRG_BANK: usize = 0x2000; // 8 KB
const CHR_BANK: usize = 0x0400; // 1 KB

pub struct Mapper85 {
    cartridge: Cartridge,
    prg_banks: [u8; 3], // $8000, $A000, $C000 (8 KB each); $E000 fixed last
    chr_banks: [u8; 8],
    // IRQ (VRC4-style)
    irq_latch: u8,
    irq_counter: u8,
    irq_mode_cycle: bool,
    irq_enable_after_ack: bool,
    irq_enabled: bool,
    irq_pending: bool,
    prescaler: i16,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_banks: [u8; 3],
    pub chr_banks: [u8; 8],
    pub irq_latch: u8,
    pub irq_counter: u8,
    pub irq_mode_cycle: bool,
    pub irq_enable_after_ack: bool,
    pub irq_enabled: bool,
    pub irq_pending: bool,
    pub prescaler: i16,
}

impl Mapper85 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper85 {
            cartridge,
            prg_banks: [0; 3],
            chr_banks: [0; 8],
            irq_latch: 0,
            irq_counter: 0,
            irq_mode_cycle: false,
            irq_enable_after_ack: false,
            irq_enabled: false,
            irq_pending: false,
            prescaler: 341,
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
            self.prg_banks[0] as usize % self.prg_count(),
            self.prg_banks[1] as usize % self.prg_count(),
            self.prg_banks[2] as usize % self.prg_count(),
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

    fn tick_counter(&mut self) {
        if self.irq_counter == 0xFF {
            self.irq_counter = self.irq_latch;
            self.irq_pending = true;
        } else {
            self.irq_counter = self.irq_counter.wrapping_add(1);
        }
    }
}

impl Mapper for Mapper85 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xFFFF => {
                let slot = ((address - 0x8000) / 0x2000) as usize;
                let bank = if slot == 3 {
                    self.prg_count() - 1
                } else {
                    self.prg_banks[slot] as usize % self.prg_count()
                };
                self.cartridge.prg_rom[bank * PRG_BANK | (address as usize & 0x1FFF)]
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) {
            let i = (address - 0x6000) as usize;
            if i < self.cartridge.prg_ram.len() {
                self.cartridge.prg_ram[i] = value;
            }
            return;
        }
        // mapper 85 selector = address & 0xF038. That gives register
        // identity that's independent of which board variant wrote
        // (standard uses 0xF010 mask).
        match address & 0xF010 {
            0x8000 => {
                self.prg_banks[0] = value & 0x3F;
                self.rebuild_asm_window();
            }
            0x8010 => {
                self.prg_banks[1] = value & 0x3F;
                self.rebuild_asm_window();
            }
            0x9000 => {
                self.prg_banks[2] = value & 0x3F;
                self.rebuild_asm_window();
            }
            0x9010 | 0x9030 => {
                // FM audio register port — deferred.
            }
            0xA000 => {
                self.chr_banks[0] = value;
            }
            0xA010 => {
                self.chr_banks[1] = value;
            }
            0xB000 => {
                self.chr_banks[2] = value;
            }
            0xB010 => {
                self.chr_banks[3] = value;
            }
            0xC000 => {
                self.chr_banks[4] = value;
            }
            0xC010 => {
                self.chr_banks[5] = value;
            }
            0xD000 => {
                self.chr_banks[6] = value;
            }
            0xD010 => {
                self.chr_banks[7] = value;
            }
            0xE000 => {
                self.cartridge.mirroring = match value & 0x03 {
                    0 => Mirroring::Vertical,
                    1 => Mirroring::Horizontal,
                    2 => Mirroring::OneScreenLower,
                    _ => Mirroring::OneScreenUpper,
                };
            }
            0xE010 => {
                self.irq_latch = value;
            }
            0xF000 => {
                self.irq_mode_cycle = value & 0x04 != 0;
                self.irq_enable_after_ack = value & 0x01 != 0;
                self.irq_enabled = value & 0x02 != 0;
                self.irq_pending = false;
                if self.irq_enabled {
                    self.irq_counter = self.irq_latch;
                    self.prescaler = 341;
                }
            }
            0xF010 => {
                self.irq_pending = false;
                self.irq_enabled = self.irq_enable_after_ack;
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
        self.irq_latch = 0;
        self.irq_counter = 0;
        self.irq_mode_cycle = false;
        self.irq_enable_after_ack = false;
        self.irq_enabled = false;
        self.irq_pending = false;
        self.prescaler = 341;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn on_scanline_tick(&mut self) {
        if !self.irq_enabled {
            return;
        }
        if self.irq_mode_cycle {
            for _ in 0..113 {
                self.tick_counter();
                if self.irq_pending {
                    break;
                }
            }
        } else {
            self.prescaler -= 113 * 3;
            while self.prescaler <= 0 {
                self.prescaler += 341;
                self.tick_counter();
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
        2
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State85(State {
            cartridge: self.cartridge.get_state(),
            prg_banks: self.prg_banks,
            chr_banks: self.chr_banks,
            irq_latch: self.irq_latch,
            irq_counter: self.irq_counter,
            irq_mode_cycle: self.irq_mode_cycle,
            irq_enable_after_ack: self.irq_enable_after_ack,
            irq_enabled: self.irq_enabled,
            irq_pending: self.irq_pending,
            prescaler: self.prescaler,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State85(s) => {
                self.cartridge.apply_state(&s.cartridge);
                self.prg_banks = s.prg_banks;
                self.chr_banks = s.chr_banks;
                self.irq_latch = s.irq_latch;
                self.irq_counter = s.irq_counter;
                self.irq_mode_cycle = s.irq_mode_cycle;
                self.irq_enable_after_ack = s.irq_enable_after_ack;
                self.irq_enabled = s.irq_enabled;
                self.irq_pending = s.irq_pending;
                self.prescaler = s.prescaler;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state variant for VRC7"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 85,
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

    fn stamp_prg(cart: &mut Cartridge) {
        let banks = cart.prg_rom.len() / PRG_BANK;
        for i in 0..banks {
            cart.prg_rom[i * PRG_BANK] = i as u8;
        }
    }

    fn stamp_chr(cart: &mut Cartridge) {
        let banks = cart.chr.len() / CHR_BANK;
        for i in 0..banks {
            cart.chr[i * CHR_BANK] = i as u8;
        }
    }

    fn build() -> Mapper85 {
        let mut cart = test_cart(4, 2); // 8 PRG 8K-banks, 16 CHR 1K-banks
        stamp_prg(&mut cart);
        stamp_chr(&mut cart);
        Mapper85::new(cart)
    }

    // The three switchable PRG windows are selected by the A4 register bit:
    // $8000 -> slot 0, $8010 -> slot 1, $9000 -> slot 2.
    #[test]
    fn prg_banks_switch_via_a4_selector() {
        let mut m = build();
        m.prg_write_byte(0x8000, 1);
        m.prg_write_byte(0x8010, 2);
        m.prg_write_byte(0x9000, 6);
        assert_eq!(m.prg_read_byte(0x8000), 1);
        assert_eq!(m.prg_read_byte(0xA000), 2);
        assert_eq!(m.prg_read_byte(0xC000), 6);
    }

    // The $E000 window is fixed to the last PRG bank.
    #[test]
    fn prg_last_bank_fixed() {
        let mut m = build();
        assert_eq!(m.prg_read_byte(0xE000), 7);
        m.prg_write_byte(0x9000, 1);
        assert_eq!(m.prg_read_byte(0xE000), 7);
    }

    // A PRG bank index beyond the ROM wraps (mod bank count).
    #[test]
    fn prg_bank_wraps() {
        let mut m = build();
        m.prg_write_byte(0x8000, 10); // 10 & 0x3F = 10, 10 % 8 == 2
        assert_eq!(m.prg_read_byte(0x8000), 2);
    }

    // The eight 1 KB CHR windows follow $A000..$D010.
    #[test]
    fn chr_banks_switch() {
        let mut m = build();
        m.prg_write_byte(0xA000, 3); // slot 0
        m.prg_write_byte(0xD010, 9); // slot 7
        assert_eq!(m.chr_read_byte(0x0000), 3);
        assert_eq!(m.chr_read_byte(0x1C00), 9);
    }

    // A CHR bank index beyond the ROM wraps (mod bank count).
    #[test]
    fn chr_bank_wraps() {
        let mut m = build();
        m.prg_write_byte(0xA000, 20); // 20 % 16 == 4
        assert_eq!(m.chr_read_byte(0x0000), 4);
    }

    // $E000 sets mirroring from the low two bits.
    #[test]
    fn mirroring_control() {
        let mut m = build();
        m.prg_write_byte(0xE000, 0);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(0xE000, 1);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(0xE000, 2);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
        m.prg_write_byte(0xE000, 3);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }

    // Cycle-mode IRQ: a latch of 0xFF reloads to 0xFF and asserts on the
    // first clocked cycle; acknowledge clears the line and (with re-enable
    // off) leaves it disabled.
    #[test]
    fn irq_cycle_mode_fires_and_acks() {
        let mut m = build();
        m.prg_write_byte(0xE010, 0xFF); // latch
        m.prg_write_byte(0xF000, 0x06); // enable + cycle mode
        assert!(!m.irq_pending());
        m.on_scanline_tick();
        assert!(m.irq_pending());
        m.prg_write_byte(0xF010, 0x00); // ack
        assert!(!m.irq_pending());
        m.on_scanline_tick(); // disabled after ack
        assert!(!m.irq_pending());
    }

    // With enable-on-ack set, acknowledging re-arms the line so it fires again.
    #[test]
    fn irq_re_enables_after_ack() {
        let mut m = build();
        m.prg_write_byte(0xE010, 0xFF);
        m.prg_write_byte(0xF000, 0x07); // enable + cycle + enable_after_ack
        m.on_scanline_tick();
        assert!(m.irq_pending());
        m.prg_write_byte(0xF010, 0x00); // ack re-enables
        assert!(!m.irq_pending());
        m.on_scanline_tick();
        assert!(m.irq_pending());
    }

    // Scanline-mode IRQ runs the 341-tick prescaler: it does not clock on
    // the first scanline (341 -> 2) and first clocks on the second, so a
    // latch of 0xFF fires only after two scanline ticks.
    #[test]
    fn irq_scanline_mode_prescaler() {
        let mut m = build();
        m.prg_write_byte(0xE010, 0xFF);
        m.prg_write_byte(0xF000, 0x02); // enable, scanline mode
        m.on_scanline_tick();
        assert!(!m.irq_pending());
        m.on_scanline_tick();
        assert!(m.irq_pending());
    }

    // Snapshot/restore returns bank, mirroring, and IRQ latch state.
    #[test]
    fn state_round_trip() {
        let mut m = build();
        m.prg_write_byte(0x8000, 3);
        m.prg_write_byte(0xA000, 7); // chr slot 0
        m.prg_write_byte(0xE000, 0); // vertical
        m.prg_write_byte(0xE010, 0x5A); // latch
        let snap = m.get_state();

        m.prg_write_byte(0x8000, 0);
        m.prg_write_byte(0xA000, 0);
        m.prg_write_byte(0xE000, 1);
        m.prg_write_byte(0xE010, 0x00);

        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 3);
        assert_eq!(m.chr_read_byte(0x0000), 7);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        assert_eq!(m.irq_latch, 0x5A);
    }

    // reset returns to power-on banking, default mirroring, and IRQ cleared.
    #[test]
    fn reset_restores_power_on() {
        let mut m = build();
        m.prg_write_byte(0x8000, 4);
        m.prg_write_byte(0xE000, 0); // vertical
        m.prg_write_byte(0xF000, 0x06);
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        assert!(!m.irq_pending());
    }
}
