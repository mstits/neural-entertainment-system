use crate::cartridge::{Cartridge, Mirroring};
use crate::cpu::Cpu;
use crate::mapper::{self, Mapper};
use crate::mapper::mapper4::Mapper4;
use crate::ppu::Ppu;

use serde_derive::{Deserialize, Serialize};

// Mapper 37 — NES-ZZ multicart (Super Mario Bros. + Tetris + Nintendo
// World Cup, Europe). Standard MMC3 inside, with an outer-bank register
// mapped to $6000-$7FFF that selects one of three "chips":
//   Chip 0: 128 KB PRG + 128 KB CHR (SMB)
//   Chip 1: 128 KB PRG + 128 KB CHR (Tetris)
//   Chip 2: 256 KB PRG + 128 KB CHR (NWC)
//
// The outer register uses bits 2-0 of the written value. Chip = val >> 1
// clamped to 0..=2.
pub struct Mapper37 {
    inner: Mapper4,
    outer: u8,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub inner: Box<mapper::State>,
    pub outer: u8,
}

impl Mapper37 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper37 {
            inner: Mapper4::new(cartridge),
            outer: 0,
        };
        m.apply_outer();
        m
    }

    /// Compute PRG and CHR region (base, size) in bytes for the currently
    /// selected outer value. Layout in the ROM image: SMB (128K PRG,
    /// 128K CHR), Tetris (128K PRG, 128K CHR), NWC (256K PRG, 128K CHR).
    fn region(&self) -> (usize, usize, usize, usize) {
        let val = self.outer & 0x07;
        let chip = (val >> 1).min(2);
        let inner_bank = val & 0x01;
        let (prg_base, prg_size) = match chip {
            0 => (0, 128 * 1024),
            1 => (128 * 1024, 128 * 1024),
            _ => (256 * 1024, 256 * 1024),
        };
        let chr_base = chip as usize * 128 * 1024;
        let chr_size = 128 * 1024;
        // Inner sub-bank selection: for chips 0/1 (128 KB each), the
        // outer value's low bit could gate sub-regions; we keep the
        // full 128 KB visible to the inner MMC3 and let its bank regs
        // pick. For chip 2 (NWC, 256 KB), low bit is ignored.
        let _ = inner_bank;
        (prg_base, prg_size, chr_base, chr_size)
    }

    fn apply_outer(&mut self) {
        let (pb, ps, cb, cs) = self.region();
        self.inner.set_outer_region(pb, ps, cb, cs);
    }
}

impl Mapper for Mapper37 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        self.inner.prg_peek_byte(address)
    }

    fn prg_read_byte(&mut self, address: u16) -> u8 {
        self.inner.prg_read_byte(address)
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) {
            // Outer bank select. Updates region and re-evaluates MMC3
            // bank offsets against the new window. Suppress forwarding
            // to inner so we don't disturb PRG-RAM.
            self.outer = value & 0x07;
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
        mapper::State::State37(State {
            inner: Box::new(self.inner.get_state()),
            outer: self.outer,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State37(s) => {
                self.inner.apply_state(&s.inner);
                self.outer = s.outer;
                self.apply_outer();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
