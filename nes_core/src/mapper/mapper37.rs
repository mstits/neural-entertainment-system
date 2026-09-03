use crate::cartridge::{Cartridge, Mirroring};
use crate::cpu::Cpu;
use crate::mapper::{self, Mapper};
use crate::mapper::mapper4::Mapper4;
use crate::ppu::Ppu;

use serde_derive::{Deserialize, Serialize};

// Mapper 37 — NES-ZZ multicart (Super Mario Bros. + Tetris + Nintendo
// World Cup, Europe). 256 KB PRG + 256 KB CHR, standard MMC3 inside,
// with a 3-bit outer register mapped to $6000-$7FFF.
//
// The outer value's bit 2 drives A17 on both PRG and CHR. Its low two
// bits gate PRG A16: the value 3 forces A16 high, any other value ANDs
// the MMC3's own A16 with bit 2. So the windows the MMC3 sees are, per
// the NESdev wiki's iNES Mapper 037 table:
//
//   outer | PRG window             | CHR window (always 128 KB)
//   ------+------------------------+---------------------------
//   0,1,2 | $00000-$0FFFF  (64 KB) | $00000-$1FFFF
//   3     | $10000-$1FFFF  (64 KB) | $00000-$1FFFF
//   4,5,6 | $20000-$3FFFF (128 KB) | $20000-$3FFFF
//   7     | $30000-$3FFFF  (64 KB) | $20000-$3FFFF
//
// The window size matters at power-on, not just for bank arithmetic:
// the MMC3's fixed last bank at $E000-$FFFF is the last 8 KB of the
// window, so it is where the 6502 fetches its reset vector. Outer 0
// with a 64 KB window lands on $0FFFC ($FF00, the menu); the same
// register value with a 128 KB window lands on $1FFFC ($8000), which
// is Tetris's vector table, and the cart never reaches its menu.
//
// The outer register is cleared by the CIC reset line, so power-on and
// reset() are outer = 0.
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

    /// PRG and CHR window (base, size) in bytes for the currently
    /// selected outer value, per the table in this file's header.
    fn region(&self) -> (usize, usize, usize, usize) {
        let val = self.outer & 0x07;
        // Bit 2 is wired to A17 on both PRG and CHR.
        let a17 = ((val >> 2) & 0x01) as usize;
        // The low pair forces PRG A16 high when it reads 3; otherwise
        // the MMC3's A16 is ANDed with bit 2, so A16 is only free when
        // bit 2 is set.
        let force_a16 = (val & 0x03) == 0x03;
        let prg_base = (a17 << 17) | if force_a16 { 1 << 16 } else { 0 };
        let prg_size = if force_a16 || a17 == 0 {
            64 * 1024
        } else {
            128 * 1024
        };
        let chr_base = a17 << 17;
        let chr_size = 128 * 1024;
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

    // NES-ZZ is 256 KB PRG + 256 KB CHR, matching the real dump.
    fn cart() -> Cartridge {
        Cartridge {
            mapper: 37,
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

    // Stamp byte 0 of every 8 KB PRG bank with its global index so a read
    // reveals which physical 8 KB bank is mapped ($E000 = the fixed last
    // bank of the active outer window).
    fn stamp_prg_8k(c: &mut Cartridge) {
        let banks = c.prg_rom.len() / 0x2000;
        for b in 0..banks {
            c.prg_rom[b * 0x2000] = b as u8;
        }
    }

    // The outer register picks the PRG window from this file's header
    // table; $E000 reads the fixed last 8 KB bank of that window.
    //   0,1,2 -> $00000-$0FFFF (64 KB)  -> last 8 KB bank global 7
    //   3     -> $10000-$1FFFF (64 KB)  -> global 15
    //   4,5,6 -> $20000-$3FFFF (128 KB) -> global 31
    //   7     -> $30000-$3FFFF (64 KB)  -> global 31
    #[test]
    fn outer_selects_prg_window() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        for v in [0u8, 1, 2] {
            m.prg_write_byte(0x6000, v);
            assert_eq!(m.prg_peek_byte(0xE000), 7, "outer {v}");
        }
        m.prg_write_byte(0x6000, 3);
        assert_eq!(m.prg_peek_byte(0xE000), 15);
        for v in [4u8, 5, 6] {
            m.prg_write_byte(0x6000, v);
            assert_eq!(m.prg_peek_byte(0xE000), 31, "outer {v}");
        }
        m.prg_write_byte(0x6000, 7);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
    }

    // The low pair of the outer value gates PRG A16 rather than selecting
    // a chip: only the value 3 forces A16 high while bit 2 is clear, so
    // outer 2 and outer 3 land on different 64 KB halves even though both
    // have bit 2 clear. Guards against a `chip = val >> 1` style decode,
    // which would put 2 and 3 in the same window.
    #[test]
    fn low_pair_gates_a16_not_a_chip_index() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 2);
        assert_eq!(m.prg_peek_byte(0xE000), 7); // A16 = MMC3 A16 AND bit2 = 0
        m.prg_write_byte(0x6000, 3);
        assert_eq!(m.prg_peek_byte(0xE000), 15); // A16 forced high
    }

    // Bit 2 is A17 on PRG, and with the low pair clear it also frees the
    // MMC3's own A16: the window doubles to 128 KB, so an inner bank
    // register above 7 is reachable there and nowhere else.
    #[test]
    fn bit2_frees_a16_and_doubles_the_window() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x8000, 6); // select bank register 6 -> $8000-$9FFF
        m.prg_write_byte(0x8001, 9); // R6 = 9, above the 8-bank 64 KB window
        m.prg_write_byte(0x6000, 0);
        assert_eq!(m.prg_peek_byte(0x8000), 1); // 9 % 8 = 1, A16 held low
        m.prg_write_byte(0x6000, 4);
        assert_eq!(m.prg_peek_byte(0x8000), 16 + 9); // 9 % 16 = 9 inside $20000
    }

    // The outer latch keeps only bits 2-0.
    #[test]
    fn outer_masks_low_three_bits() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0xFF);
        assert_eq!(m.outer, 0x07);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
    }

    // Bit 2 is wired to CHR A17 as well: the CHR window is always 128 KB,
    // at $00000 with bit 2 clear and $20000 with it set. The low pair does
    // not move CHR at all.
    #[test]
    fn outer_bit2_selects_chr_half() {
        let mut c = cart();
        c.chr[0] = 0xB0;
        c.chr[128 * 1024] = 0xB1;
        let mut m = Mapper37::new(c);
        for v in [0u8, 1, 2, 3] {
            m.prg_write_byte(0x6000, v);
            assert_eq!(m.chr_read_byte(0x0000), 0xB0, "outer {v}");
        }
        for v in [4u8, 5, 6, 7] {
            m.prg_write_byte(0x6000, v);
            assert_eq!(m.chr_read_byte(0x0000), 0xB1, "outer {v}");
        }
    }

    // Every outer value addresses PRG and CHR that actually exist in a
    // 256 KB / 256 KB dump. HEAD mapped outer >= 4 to a 256 KB window
    // based at 256 KB, which is entirely past the end of PRG.
    #[test]
    fn every_outer_value_stays_inside_the_dump() {
        let m = Mapper37::new(cart());
        let prg_len = 256 * 1024;
        let chr_len = 256 * 1024;
        for v in 0u8..8 {
            let mut m2 = Mapper37 { inner: Mapper4::new(cart()), outer: v };
            m2.apply_outer();
            let (pb, ps, cb, cs) = m2.region();
            assert!(pb + ps <= prg_len, "outer {v}: PRG window {pb}+{ps} past {prg_len}");
            assert!(cb + cs <= chr_len, "outer {v}: CHR window {cb}+{cs} past {chr_len}");
            assert!(ps == 64 * 1024 || ps == 128 * 1024, "outer {v}: PRG window size {ps}");
        }
        let _ = m;
    }

    // The inner MMC3 PRG bank register still works through the wrapper:
    // in the power-on window, R6=3 maps $8000-$9FFF to global 8 KB bank 3.
    #[test]
    fn inner_mmc3_prg_bank_through_wrapper() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 0);
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

    // The outer register has no reset line of its own on the board, it is
    // cleared by the CIC reset line, so reset() puts it back to 0. That
    // also puts the reset vector back in the menu's 64 KB window.
    #[test]
    fn reset_clears_outer() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 4);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
        m.reset();
        assert_eq!(m.outer, 0);
        assert_eq!(m.prg_peek_byte(0xE000), 7);
    }

    // get_state/apply_state round-trips both the outer latch and the inner
    // MMC3 bank state.
    #[test]
    fn state_round_trip() {
        let mut c = cart();
        stamp_prg_8k(&mut c);
        let mut m = Mapper37::new(c);
        m.prg_write_byte(0x6000, 4); // $20000-$3FFFF, 16 banks
        m.prg_write_byte(0x8000, 6);
        m.prg_write_byte(0x8001, 3); // R6=3 in that window -> global bank 19
        let snap = m.get_state();
        m.prg_write_byte(0x6000, 0);
        m.apply_state(&snap);
        assert_eq!(m.outer, 4);
        assert_eq!(m.prg_peek_byte(0xE000), 31);
        assert_eq!(m.prg_peek_byte(0x8000), 19);
    }
}
