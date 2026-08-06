use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// NTDEC 2722 (iNES mapper 40) — the FDS-to-cartridge conversion board
// used by the Super Mario Bros. 2 (Japan) / "Lost Levels" bootleg.
//
// PRG-ROM is banked in fixed 8 KB windows; only $C000-$DFFF is
// switchable (a 3-bit register written through $E000-$FFFF). Layout
// (nesdev.org/wiki/INES_Mapper_040):
//
//   $6000 = bank 6   $8000 = bank 4   $A000 = bank 5
//   $C000 = { $E000 register }        $E000 = bank 7
//
// CHR is a single unbanked 8 KB region (ROM, or RAM when the header
// declares no CHR). Mirroring is fixed from the iNES header.
//
// IRQ: a CD4020 acting as a 13-bit counter clocked once per M2 (CPU)
// cycle while enabled. The IRQ line is bit 12 of that counter — it
// asserts 4096 cycles after the counter starts and, if the game never
// acknowledges it, SELF-ACKNOWLEDGES 4096 cycles later when the counter
// wraps past 8191 (bit 12 falls). The line is therefore a 50%-duty
// square wave of period 8192 while enabled. A write to $8000-$9FFF
// disables + resets the counter to 0; a write to $A000-$BFFF enables.
// The count is driven from `set_cpu_cycle`, which `Nes::tick` calls once
// per CPU cycle with the running cycle counter — the delta between
// successive calls is the number of M2 pulses to add while enabled.
//
// This self-acknowledging behaviour is load-bearing for the SMB2j
// bootleg: it reuses the original FDS timer-IRQ handler, which
// acknowledges by reading the FDS status port ($4030) rather than
// writing $8000. Without the self-ack the line would latch asserted
// forever and the CPU would never leave the IRQ handler.
const PRG_BANK: usize = 0x2000; // 8 KB
const IRQ_BIT: u32 = 0x1000; // bit 12 = the IRQ line
const IRQ_WRAP: u32 = 0x2000; // 13-bit counter wraps here (self-ack)

pub struct Mapper40 {
    cartridge: Cartridge,
    // Switchable $C000-$DFFF PRG bank (register value & 0x07).
    prg_bank: u8,
    irq_enabled: bool,
    // 13-bit M2 counter; the IRQ line is bit 12 (IRQ_BIT).
    irq_counter: u32,
    // Most recent CPU cycle seen via `set_cpu_cycle`; the per-cycle
    // delta drives the counter while enabled.
    last_cpu_cycle: u64,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_bank: u8,
    pub irq_enabled: bool,
    pub irq_counter: u32,
    pub last_cpu_cycle: u64,
}

impl Mapper40 {
    pub fn new(cartridge: Cartridge) -> Self {
        Mapper40 {
            cartridge,
            prg_bank: 0,
            irq_enabled: false,
            irq_counter: 0,
            last_cpu_cycle: 0,
        }
    }

    fn prg_bank_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK).max(1)
    }

    /// Byte at 8 KB `bank` (taken modulo the ROM's actual bank count)
    /// offset by the low 13 bits of `address`.
    fn read_bank(&self, bank: usize, address: u16) -> u8 {
        let off = (bank % self.prg_bank_count()) * PRG_BANK | (address as usize & 0x1FFF);
        self.cartridge.prg_rom.get(off).copied().unwrap_or(0)
    }
}

impl Mapper for Mapper40 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            // $6000-$7FFF -> fixed bank 6
            0x6000..=0x7FFF => self.read_bank(6, address),
            // $8000-$9FFF -> fixed bank 4
            0x8000..=0x9FFF => self.read_bank(4, address),
            // $A000-$BFFF -> fixed bank 5
            0xA000..=0xBFFF => self.read_bank(5, address),
            // $C000-$DFFF -> switchable bank ($E000 register)
            0xC000..=0xDFFF => self.read_bank(self.prg_bank as usize, address),
            // $E000-$FFFF -> fixed bank 7
            0xE000..=0xFFFF => self.read_bank(7, address),
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x8000..=0x9FFF => {
                // IRQ disable + acknowledge + reset counter.
                self.irq_enabled = false;
                self.irq_counter = 0;
            }
            0xA000..=0xBFFF => {
                // IRQ enable — count M2 cycles from the current value.
                self.irq_enabled = true;
            }
            0xC000..=0xDFFF => {
                // Outer-bank register on submapper 1 only; a no-op for
                // the submapper-0 SMB2j board.
            }
            0xE000..=0xFFFF => {
                // Select the 8 KB bank mapped at $C000-$DFFF.
                self.prg_bank = value & 0x07;
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let idx = address as usize & 0x1FFF;
        self.cartridge.chr.get(idx).copied().unwrap_or(0)
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        // CHR-RAM carts write here; CHR-ROM carts don't drive writes.
        // Mirrors Mapper2's direct-index handling of the 8 KB region.
        let idx = address as usize & 0x1FFF;
        if let Some(cell) = self.cartridge.chr.get_mut(idx) {
            *cell = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn set_cpu_cycle(&mut self, cpu_cycle: u64) {
        let delta = cpu_cycle.wrapping_sub(self.last_cpu_cycle);
        self.last_cpu_cycle = cpu_cycle;

        // Advance the free-running 13-bit counter while enabled. The IRQ
        // line (bit 12) rises at 4096 and falls when the counter wraps
        // past 8191 — the automatic self-acknowledge. Masking to 13 bits
        // keeps the square wave going indefinitely without overflow.
        if self.irq_enabled {
            self.irq_counter = ((self.irq_counter as u64 + delta) % IRQ_WRAP as u64) as u32;
        }
    }

    fn irq_pending(&self) -> bool {
        // Level = bit 12 of the counter while enabled.
        self.irq_enabled && (self.irq_counter & IRQ_BIT) != 0
    }

    fn reset(&mut self) {
        self.prg_bank = 0;
        self.irq_enabled = false;
        self.irq_counter = 0;
        self.last_cpu_cycle = 0;
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State40(State {
            cartridge: self.cartridge.get_state(),
            prg_bank: self.prg_bank,
            irq_enabled: self.irq_enabled,
            irq_counter: self.irq_counter,
            last_cpu_cycle: self.last_cpu_cycle,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State40(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.prg_bank = state.prg_bank;
                self.irq_enabled = state.irq_enabled;
                self.irq_counter = state.irq_counter;
                self.last_cpu_cycle = state.last_cpu_cycle;
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 8 banks × 8 KB = 64 KB. Every byte encodes its containing 8 KB
    // bank (`bank * 37`, 37 coprime to 256, so the mapping is injective
    // over bank index) — a read from the wrong bank is observable since
    // window and bank math share the within-bank offset at a fixed
    // address.
    fn synth_mapper40() -> Mapper40 {
        let banks = 8usize;
        let prg_len = banks * PRG_BANK;
        let prg_rom: Vec<u8> = (0..prg_len)
            .map(|i| ((i / PRG_BANK) as u8).wrapping_mul(37).wrapping_add((i & 0xFF) as u8))
            .collect();
        let cart = Cartridge {
            mapper: 40,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: (prg_len / (16 * 1024)) as u8,
            prg_rom,
            chr_num_banks: 1,
            chr: vec![0u8; 8 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        };
        Mapper40::new(cart)
    }

    // First byte of 8 KB `bank` per the synthetic encoding.
    fn bank_first_byte(bank: u8) -> u8 {
        bank.wrapping_mul(37)
    }

    #[test]
    fn fixed_bank_layout() {
        let m = synth_mapper40();
        // $6000 -> 6, $8000 -> 4, $A000 -> 5, $E000 -> 7.
        assert_eq!(m.prg_peek_byte(0x6000), bank_first_byte(6));
        assert_eq!(m.prg_peek_byte(0x8000), bank_first_byte(4));
        assert_eq!(m.prg_peek_byte(0xA000), bank_first_byte(5));
        assert_eq!(m.prg_peek_byte(0xE000), bank_first_byte(7));
        // Default switchable $C000 window is bank 0.
        assert_eq!(m.prg_peek_byte(0xC000), bank_first_byte(0));
    }

    #[test]
    fn e000_switches_c000_window() {
        let mut m = synth_mapper40();
        for bank in 0u8..=7 {
            m.prg_write_byte(0xE000, bank);
            assert_eq!(m.prg_peek_byte(0xC000), bank_first_byte(bank));
        }
        // Only the low 3 bits select the bank.
        m.prg_write_byte(0xE000, 0xF9); // 0b1111_1001 -> bank 1
        assert_eq!(m.prg_peek_byte(0xC000), bank_first_byte(1));
        // Fixed windows are unaffected by the switch.
        assert_eq!(m.prg_peek_byte(0x8000), bank_first_byte(4));
        assert_eq!(m.prg_peek_byte(0xA000), bank_first_byte(5));
        assert_eq!(m.prg_peek_byte(0xE000), bank_first_byte(7));
    }

    #[test]
    fn irq_asserts_at_4096_and_self_acks_at_8192() {
        let mut m = synth_mapper40();
        let mut cyc: u64 = 100;
        m.set_cpu_cycle(cyc);
        assert!(!m.irq_pending());

        m.prg_write_byte(0xA000, 0); // enable

        // 4095 cycles: still low.
        for _ in 0..4095 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        assert!(!m.irq_pending(), "IRQ must be low before 4096 cycles");

        // Cycle 4096: line asserts (bit 12 set).
        cyc += 1;
        m.set_cpu_cycle(cyc);
        assert!(m.irq_pending(), "IRQ must assert at 4096 enabled cycles");

        // Stays asserted through the high window (bit 12 set: 4096..8191).
        // From counter=4096, 4095 more cycles land on 8191 — still high.
        for _ in 0..4095 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        assert!(m.irq_pending(), "IRQ holds through the high window");

        // The 8192nd cycle wraps the counter to 0 — self-acknowledge,
        // line falls.
        cyc += 1;
        m.set_cpu_cycle(cyc);
        assert!(!m.irq_pending(), "IRQ must self-ack when the counter wraps");
        assert_eq!(m.irq_counter, 0);

        // And it re-asserts 4096 cycles later — a periodic square wave.
        for _ in 0..4096 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        assert!(m.irq_pending(), "IRQ re-asserts one period later");
    }

    #[test]
    fn write_8000_disables_and_resets() {
        let mut m = synth_mapper40();
        let mut cyc = 0u64;
        m.set_cpu_cycle(cyc);
        m.prg_write_byte(0xA000, 0); // enable
        for _ in 0..5000 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        assert!(m.irq_pending(), "asserted inside the high window");

        // $8000 write: disable + ack + reset counter.
        m.prg_write_byte(0x8000, 0);
        assert!(!m.irq_pending());
        assert!(!m.irq_enabled);
        assert_eq!(m.irq_counter, 0);

        // Disabled: cycles no longer advance the counter or assert.
        for _ in 0..20000 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        assert!(!m.irq_pending(), "disabled counter must not assert");
        assert_eq!(m.irq_counter, 0);
    }

    #[test]
    fn state_roundtrip_preserves_irq_and_bank() {
        let mut m = synth_mapper40();
        m.prg_write_byte(0xE000, 5);
        m.prg_write_byte(0xA000, 0);
        let mut cyc = 0u64;
        for _ in 0..10 {
            cyc += 1;
            m.set_cpu_cycle(cyc);
        }
        let state = m.get_state();

        let mut m2 = synth_mapper40();
        m2.apply_state(&state);
        assert_eq!(m2.prg_bank, 5);
        assert!(m2.irq_enabled);
        assert_eq!(m2.irq_counter, m.irq_counter);
        assert_eq!(m2.last_cpu_cycle, m.last_cpu_cycle);
        assert_eq!(m2.prg_peek_byte(0xC000), bank_first_byte(5));
    }
}
