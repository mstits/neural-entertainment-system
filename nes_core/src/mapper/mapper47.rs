use crate::cartridge::{Cartridge, Mirroring};
use crate::cpu::Cpu;
use crate::mapper::{self, Mapper};
use crate::mapper::mapper4::Mapper4;
use crate::ppu::Ppu;

use serde_derive::{Deserialize, Serialize};

// Mapper 47 — NES-QJ multicart (Super Spike V'Ball + Nintendo World Cup).
// MMC3 inside with an outer-bank register at $6000-$7FFF:
//   bit 0 of value: outer bank select (0 or 1)
// Each outer bank = 128 KB PRG + 128 KB CHR. Outer=0 -> Super Spike,
// outer=1 -> Nintendo World Cup.
pub struct Mapper47 {
    inner: Mapper4,
    outer: u8,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub inner: Box<mapper::State>,
    pub outer: u8,
}

impl Mapper47 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper47 {
            inner: Mapper4::new(cartridge),
            outer: 0,
        };
        m.apply_outer();
        m
    }

    fn region(&self) -> (usize, usize, usize, usize) {
        let bank = (self.outer & 0x01) as usize;
        let prg_base = bank * 128 * 1024;
        let chr_base = bank * 128 * 1024;
        (prg_base, 128 * 1024, chr_base, 128 * 1024)
    }

    fn apply_outer(&mut self) {
        let (pb, ps, cb, cs) = self.region();
        self.inner.set_outer_region(pb, ps, cb, cs);
    }
}

impl Mapper for Mapper47 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        self.inner.prg_peek_byte(address)
    }

    fn prg_read_byte(&mut self, address: u16) -> u8 {
        self.inner.prg_read_byte(address)
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) {
            self.outer = value & 0x01;
            self.apply_outer();
            return;
        }
        self.inner.prg_write_byte(address, value);
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        self.inner.chr_read_byte(address)
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        self.inner.chr_write_byte(address, value);
    }

    fn mirroring(&self) -> Mirroring {
        self.inner.mirroring()
    }

    fn step(&mut self, cpu: &mut Cpu, ppu: &Ppu) {
        self.inner.step(cpu, ppu);
    }

    fn on_scanline_tick(&mut self) {
        self.inner.on_scanline_tick();
    }

    fn sram(&mut self) -> *mut u8 {
        self.inner.sram()
    }

    fn sram_size(&self) -> usize {
        self.inner.sram_size()
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        self.inner.prg_asm_ptr()
    }

    fn asm_bulk_cycles(&self) -> i64 {
        self.inner.asm_bulk_cycles()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.outer = 0;
        self.apply_outer();
    }

    fn irq_pending(&self) -> bool {
        self.inner.irq_pending()
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State47(State {
            inner: Box::new(self.inner.get_state()),
            outer: self.outer,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State47(s) => {
                self.inner.apply_state(&s.inner);
                self.outer = s.outer;
                self.apply_outer();
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

    const IRQ_LATCH: u16 = 0xC000;
    const IRQ_RELOAD: u16 = 0xC001;
    const IRQ_ENABLE: u16 = 0xE001;
    const IRQ_DISABLE_ACK: u16 = 0xE000;

    // NES-QJ: two 128 KB PRG + 128 KB CHR outer banks -> 256 KB each.
    fn cart() -> Cartridge {
        Cartridge {
            mapper: 47,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: 16,
            prg_rom: vec![0u8; 256 * 1024],
            chr_num_banks: 32,
            chr: vec![0u8; 256 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    fn stamp_prg_8k(c: &mut Cartridge) {
        let banks = c.prg_rom.len() / 0x2000;
        for b in 0..banks {
            c.prg_rom[b * 0x2000] = b as u8;
        }
    }

    // The outer register (bit 0) at $6000-$7FFF picks the 128 KB bank; $E000
    // reads its fixed last 8 KB bank (outer 0 -> global 15, outer 1 -> 31).
    #[test]
    fn outer_selects_prg_bank() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
        m.prg_write_byte(0x6000, 1);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
    }

    // Only bit 0 of the outer write is retained.
    #[test]
    fn outer_masks_to_bit0() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 0xFE); // bit 0 clear -> bank 0
        assert_eq!(m.outer, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
        m.prg_write_byte(0x6000, 0xFF); // bit 0 set -> bank 1
        assert_eq!(m.outer, 1);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
    }

    // The outer register also swaps the 128 KB CHR window.
    #[test]
    fn outer_selects_chr_bank() {
        let mut c = cart();
        c.chr[0] = 0xB0;
        c.chr[128 * 1024] = 0xB1;
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 0);
        assert_eq!(m.chr_read_byte(0x0000), 0xB0);
        m.prg_write_byte(0x6000, 1);
        assert_eq!(m.chr_read_byte(0x0000), 0xB1);
    }

    // Inner MMC3 PRG banking works within the selected outer bank.
    #[test]
    fn inner_mmc3_prg_bank_through_wrapper() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 0); // outer bank 0
        m.prg_write_byte(0x8000, 6); // select bank register 6
        m.prg_write_byte(0x8001, 3); // R6 = 3
        assert_eq!(m.prg_peek_byte(0x8000), 3);
    }

    // Inner MMC3 scanline IRQ arms, fires and acks through the wrapper.
    #[test]
    fn irq_passthrough() {
        let mut m = Mapper47::new(cart());
        m.prg_write_byte(IRQ_LATCH, 3);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        for _ in 0..3 {
            m.on_scanline_tick();
            assert!(!m.irq_pending());
        }
        m.on_scanline_tick();
        assert!(m.irq_pending());
        m.prg_write_byte(IRQ_DISABLE_ACK, 0);
        assert!(!m.irq_pending());
    }

    // reset() returns the outer latch to bank 0.
    #[test]
    fn reset_clears_outer() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 1);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
        m.reset();
        assert_eq!(m.outer, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
    }

    // get_state/apply_state round-trips the outer latch and inner MMC3 state.
    #[test]
    fn state_round_trip() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper47::new(c);
        m.prg_write_byte(0x6000, 1); // outer bank 1
        m.prg_write_byte(0x8000, 6);
        m.prg_write_byte(0x8001, 3); // R6=3 in bank 1 -> global bank 19
        let snap = m.get_state();
        m.prg_write_byte(0x6000, 0);
        m.apply_state(&snap);
        assert_eq!(m.outer, 1);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
        assert_eq!(m.prg_peek_byte(0x8000), 19);
    }
}
