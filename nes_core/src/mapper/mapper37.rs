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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{Cartridge, Mirroring};
    use crate::mapper::Mapper;

    // MMC3 IRQ registers (bit 0 + the $C000/$E000 split are all that matter).
    const IRQ_LATCH: u16 = 0xC000;
    const IRQ_RELOAD: u16 = 0xC001;
    const IRQ_ENABLE: u16 = 0xE001;
    const IRQ_DISABLE_ACK: u16 = 0xE000;

    // NES-ZZ needs 512 KB PRG (SMB 128K + Tetris 128K + NWC 256K) and
    // 384 KB CHR (three 128 KB chips).
    fn cart() -> Cartridge {
        Cartridge {
            mapper: 37,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: 32,
            prg_rom: vec![0u8; 512 * 1024],
            chr_num_banks: 48,
            chr: vec![0u8; 384 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    // Stamp byte 0 of every 8 KB PRG bank with its global index so a read
    // reveals which physical 8 KB bank is mapped ($E000 = the fixed last
    // bank of the active outer region).
    fn stamp_prg_8k(c: &mut Cartridge) {
        let banks = c.prg_rom.len() / 0x2000;
        for b in 0..banks {
            c.prg_rom[b * 0x2000] = b as u8;
        }
    }

    // The outer register at $6000-$7FFF selects one of three chips; $E000
    // reads the fixed last 8 KB bank of that chip's region.
    //   chip 0 = [0,128K)   -> last bank global 15
    //   chip 1 = [128K,256K)-> last bank global 31
    //   chip 2 = [256K,512K)-> last bank global 63
    #[test]
    fn outer_selects_prg_chip() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
        m.prg_write_byte(0x6000, 2);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
        m.prg_write_byte(0x6000, 4);
        assert_eq!(m.prg_peek_byte(0xE000), 63);
    }

    // The outer latch keeps only bits 2-0; chip = (val >> 1).min(2).
    #[test]
    fn outer_masks_low_three_bits() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0xFF);
        assert_eq!(m.outer, 0x07);
        assert_eq!(m.prg_peek_byte(0xE000), 63); // clamped to chip 2
    }

    // The outer register selects the CHR chip; a $0000 fetch hits the base
    // 1 KB bank of that chip's 128 KB CHR region.
    #[test]
    fn outer_selects_chr_chip() {
        let mut c = cart();
        c.chr[0] = 0xB0;
        c.chr[128 * 1024] = 0xB1;
        c.chr[256 * 1024] = 0xB2;
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0);
        assert_eq!(m.chr_read_byte(0x0000), 0xB0);
        m.prg_write_byte(0x6000, 2);
        assert_eq!(m.chr_read_byte(0x0000), 0xB1);
        m.prg_write_byte(0x6000, 4);
        assert_eq!(m.chr_read_byte(0x0000), 0xB2);
    }

    // The inner MMC3 PRG bank register still works through the wrapper:
    // within chip 0, R6=3 maps $8000-$9FFF to global 8 KB bank 3.
    #[test]
    fn inner_mmc3_prg_bank_through_wrapper() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0); // chip 0
        m.prg_write_byte(0x8000, 6); // select bank register 6
        m.prg_write_byte(0x8001, 3); // R6 = 3
        assert_eq!(m.prg_peek_byte(0x8000), 3);
    }

    // The inner MMC3 scanline IRQ arms, fires and acks through the wrapper.
    #[test]
    fn irq_passthrough() {
        let mut m = Mapper37::new(cart());
        m.prg_write_byte(IRQ_LATCH, 3);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        for _ in 0..3 {
            m.on_scanline_tick();
            assert!(!m.irq_pending());
        }
        m.on_scanline_tick(); // latch+1'th clock fires
        assert!(m.irq_pending());
        m.prg_write_byte(IRQ_DISABLE_ACK, 0);
        assert!(!m.irq_pending());
    }

    // reset() clears the outer latch back to chip 0.
    #[test]
    fn reset_clears_outer() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 4); // chip 2
        assert_eq!(m.prg_peek_byte(0xE000), 63);
        m.reset();
        assert_eq!(m.outer, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
    }

    // get_state/apply_state round-trips both the outer latch and the inner
    // MMC3 bank state.
    #[test]
    fn state_round_trip() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 4); // chip 2
        m.prg_write_byte(0x8000, 6);
        m.prg_write_byte(0x8001, 3); // R6=3 in chip 2 -> global bank 35
        let snap = m.get_state();
        m.prg_write_byte(0x6000, 0); // chip 0
        m.apply_state(&snap);
        assert_eq!(m.outer, 4);
        assert_eq!(m.prg_peek_byte(0xE000), 63);
        assert_eq!(m.prg_peek_byte(0x8000), 35);
    }
}
