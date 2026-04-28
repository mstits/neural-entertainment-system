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
