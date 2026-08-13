use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper3 {
    cartridge: Cartridge,
    chr_bank: u8,
    /// Flat 32 KB PRG ROM view for the AArch64 ASM CPU fast path.
    /// CNROM has no PRG banking (writes to $8000+ select CHR banks
    /// only), so this is built once at construction and is pointer-
    /// AND content-stable for the mapper's lifetime — same shape as
    /// Mapper0. 16 KB carts hold a duplicated copy (low bank at
    /// 0x0000 and 0x4000) so the ASM's 0x7FFF mask resolves the
    /// mirror exactly like `prg_peek_byte`'s `& (len-1)`.
    prg_flat_32k: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub chr_bank: u8,
}

impl Mapper3 {
    pub fn new(cartridge: Cartridge) -> Self {
        let prg_flat_32k = match cartridge.prg_rom.len() {
            n if n == 32 * 1024 => cartridge.prg_rom.clone(),
            n if n == 16 * 1024 => {
                let mut v = Vec::with_capacity(32 * 1024);
                v.extend_from_slice(&cartridge.prg_rom);
                v.extend_from_slice(&cartridge.prg_rom);
                v
            }
            _ => Vec::new(), // unsupported size — ASM path disabled
        };
        Mapper3 {
            cartridge,
            chr_bank: 0,
            prg_flat_32k,
        }
    }

    fn chr_address(&self, bank: u8, address: u16) -> usize {
        let chr_len = self.cartridge.chr.len();
        if chr_len == 0 {
            return 0;
        }
        let raw = ((bank as usize) * 0x2000) | (address as usize & 0x1FFF);
        raw & (chr_len - 1)
    }
}

impl Mapper for Mapper3 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            0
        } else {
            // 16 KB CNROM carts mirror the single PRG bank across both
            // halves of the $8000-$FFFF window (Joust, Legend of Kage).
            // The `& (len-1)` mask handles both 16 KB (mirrored) and
            // 32 KB (full) carts since PRG sizes are powers of two.
            let len = self.cartridge.prg_rom.len();
            if len == 0 { return 0; }
            self.cartridge.prg_rom[(address - 0x8000) as usize & (len - 1)]
        }
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        if self.prg_flat_32k.len() == 32 * 1024 {
            Some(self.prg_flat_32k.as_ptr())
        } else {
            None
        }
    }

    /// CNROM has no IRQ source and fully static PRG, so like NROM it
    /// is safe under multi-instruction ASM batches: the mapper-
    /// agnostic `cpu_cycles_until_nmi_fire` cap in `Nes::step` ends
    /// each batch at most one instruction past the vblank rising
    /// edge, and CHR-bank stores to $8000+ route through the ASM
    /// MMIO write callback (cumulative PPU tick before the write) —
    /// the same path AxROM bank writes already use at bulk=4.
    ///
    /// Verified at 8 by `tests/asm_vs_slow_gradius.rs` (20M-
    /// instruction lockstep: per-instruction cycle + register
    /// equality, periodic full-RAM compare) and
    /// `tests/gradius_state_diag.rs` (full serialized-state compare
    /// incl. PPU/APU internals). Wiring CNROM onto the ASM engine
    /// surfaced two engine-wide MMIO-timing misalignments (indexed
    /// stores + early-commit reads — fixed in cpu_asm.s). Raise
    /// toward the NROM-equivalent 64 only after re-running those
    /// suites and the Gradius Mesen tape at the higher value.
    fn asm_bulk_cycles(&self) -> i64 {
        8
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            // Bank-register value indexes 8 KB CHR banks, not PRG.
            // The original code divided by PRG size which masked
            // incorrectly on carts with unequal PRG/CHR sizes
            // (Legend of Kage: 32 KB PRG / 16 KB CHR → bank 3 was
            // reachable even though only 2 CHR banks exist). Store
            // the raw value; chr_address masks at fetch time.
            self.chr_bank = value;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        if self.cartridge.chr.is_empty() {
            return 0;
        }
        let rom_addr = self.chr_address(self.chr_bank, address);
        self.cartridge.chr[rom_addr]
    }

    fn chr_write_byte(&mut self, _address: u16, _value: u8) {
        // Mapper 3 (CNROM) technically has read-only CHR ROM, but a
        // handful of commercial carts and most homebrew use CNROM-
        // shaped headers with CHR-RAM behaviour. Panicking here takes
        // down the whole trainer on a single buggy write — silently
        // ignore instead, matching behaviour of other read-only CHR
        // mappers.
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.chr_bank = 0;
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State3(State {
            cartridge: self.cartridge.get_state(),
            chr_bank: self.chr_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State3(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.chr_bank = state.chr_bank;
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
            mapper: 3,
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

    // Stamp byte 0 of every 8 KB CHR bank with a distinct marker.
    fn stamp_chr_banks(cart: &mut Cartridge) {
        let bank_size = 8 * 1024;
        for b in 0..cart.chr_num_banks as usize {
            cart.chr[b * bank_size] = 0xC0 + b as u8;
        }
    }

    // Writing $8000+ selects the 8 KB CHR bank seen by CHR reads.
    #[test]
    fn cnrom_chr_bank_switch() {
        let mut cart = test_cart(2, 4);
        stamp_chr_banks(&mut cart);
        let mut m = Mapper3::new(cart);
        m.prg_write_byte(0x8000, 0);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0);
        m.prg_write_byte(0x8000, 2);
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
        // Any address in the $8000-$FFFF window is a bank-select port.
        m.prg_write_byte(0xFFFF, 3);
        assert_eq!(m.chr_read_byte(0x0000), 0xC3);
    }

    // CHR bank index wraps within CHR ROM size (bank 6 of 4 => bank 2).
    #[test]
    fn cnrom_chr_bank_wraps() {
        let mut cart = test_cart(2, 4);
        stamp_chr_banks(&mut cart);
        let mut m = Mapper3::new(cart);
        // 4 banks => 32 KB, mask 0x7FFF; 6*0x2000 & 0x7FFF = 0x4000 => bank 2.
        m.prg_write_byte(0x8000, 6);
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
    }

    // 32 KB PRG maps linearly; reads below $8000 are open bus.
    #[test]
    fn cnrom_prg_32k_mapping() {
        let mut cart = test_cart(2, 4);
        cart.prg_rom[0x0000] = 0x10;
        cart.prg_rom[0x7FFF] = 0x20;
        let mut m = Mapper3::new(cart);
        assert_eq!(m.prg_read_byte(0x7FFF), 0); // below $8000
        assert_eq!(m.prg_read_byte(0x8000), 0x10);
        assert_eq!(m.prg_read_byte(0xFFFF), 0x20);
    }

    // 16 KB PRG mirrors into the high half of $8000-$FFFF.
    #[test]
    fn cnrom_prg_16k_mirror() {
        let mut cart = test_cart(1, 4);
        cart.prg_rom[0x0000] = 0x55;
        let mut m = Mapper3::new(cart);
        assert_eq!(m.prg_read_byte(0x8000), 0x55);
        assert_eq!(m.prg_read_byte(0xC000), 0x55);
    }

    // CHR is ROM: chr_write_byte is a no-op and does not corrupt banks.
    #[test]
    fn cnrom_chr_write_ignored() {
        let mut cart = test_cart(2, 4);
        stamp_chr_banks(&mut cart);
        let mut m = Mapper3::new(cart);
        m.chr_write_byte(0x0000, 0xFF);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0); // unchanged
    }

    // reset() returns to CHR bank 0.
    #[test]
    fn cnrom_reset_selects_bank_zero() {
        let mut cart = test_cart(2, 4);
        stamp_chr_banks(&mut cart);
        let mut m = Mapper3::new(cart);
        m.prg_write_byte(0x8000, 3);
        assert_eq!(m.chr_read_byte(0x0000), 0xC3);
        m.reset();
        assert_eq!(m.chr_read_byte(0x0000), 0xC0);
    }

    // mirroring() reflects the cartridge header.
    #[test]
    fn cnrom_mirroring_from_cartridge() {
        let mut cart = test_cart(2, 4);
        cart.mirroring = Mirroring::Vertical;
        let m = Mapper3::new(cart);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    // State snapshot preserves the CHR bank across a later switch.
    #[test]
    fn cnrom_state_round_trip() {
        let mut cart = test_cart(2, 4);
        stamp_chr_banks(&mut cart);
        let mut m = Mapper3::new(cart);
        m.prg_write_byte(0x8000, 2);
        let snap = m.get_state();
        m.prg_write_byte(0x8000, 0);
        assert_eq!(m.chr_read_byte(0x0000), 0xC0);
        m.apply_state(&snap);
        assert_eq!(m.chr_read_byte(0x0000), 0xC2);
    }
}
