use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// CPROM (Videomation): fixed 32 KB PRG at $8000-$FFFF. CHR is 16 KB of
// on-cart CHR-RAM split into two 4 KB banks. Lower 4 KB ($0000-$0FFF)
// is fixed to CHR-RAM bank 0; upper 4 KB ($1000-$1FFF) is switchable
// (value & 0x03) via any write to $8000-$FFFF.
const CHR_RAM_SIZE: usize = 16 * 1024;

pub struct Mapper13 {
    cartridge: Cartridge,
    chr_upper_bank: u8,
    chr_ram: Vec<u8>,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub chr_upper_bank: u8,
    #[serde(with = "serde_bytes")]
    pub chr_ram: Vec<u8>,
}

impl Mapper13 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper13 {
            cartridge,
            chr_upper_bank: 0,
            chr_ram: vec![0u8; CHR_RAM_SIZE],
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn rebuild_asm_window(&mut self) {
        let prg = &self.cartridge.prg_rom;
        let len = prg.len();
        if len == 0 {
            return;
        }
        // CPROM has a single fixed 32 KB PRG bank. If the cart is only
        // 16 KB, mirror it across the 32 KB window.
        for i in 0..0x8000 {
            self.prg_asm_window[i] = prg[i & (len - 1)];
        }
    }

    fn chr_address(&self, address: u16) -> usize {
        // Lower 4 KB fixed to bank 0, upper 4 KB switchable.
        let addr = address as usize & 0x1FFF;
        if addr < 0x1000 {
            addr
        } else {
            let bank = (self.chr_upper_bank as usize) & 0x03;
            (bank * 0x1000) | (addr & 0x0FFF)
        }
    }
}

impl Mapper for Mapper13 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let len = self.cartridge.prg_rom.len();
        if len == 0 {
            return 0;
        }
        self.cartridge.prg_rom[(address - 0x8000) as usize & (len - 1)]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address >= 0x8000 {
            self.chr_upper_bank = value & 0x03;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let off = self.chr_address(address) & (CHR_RAM_SIZE - 1);
        self.chr_ram[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let off = self.chr_address(address) & (CHR_RAM_SIZE - 1);
        self.chr_ram[off] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.chr_upper_bank = 0;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State13(State {
            cartridge: self.cartridge.get_state(),
            chr_upper_bank: self.chr_upper_bank,
            chr_ram: self.chr_ram.clone(),
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State13(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.chr_upper_bank = state.chr_upper_bank;
                if state.chr_ram.len() == self.chr_ram.len() {
                    self.chr_ram.copy_from_slice(&state.chr_ram);
                }
                self.rebuild_asm_window();
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

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 13,
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

    // A full 32 KB PRG cart (two 16 KB banks) with distinct markers at the
    // first and last byte of the window.
    fn cprom_mapper() -> Mapper13 {
        let mut cart = test_cart(2, 0);
        cart.prg_rom[0x0000] = 0x11;
        cart.prg_rom[0x7FFF] = 0x22;
        Mapper13::new(cart)
    }

    // The 32 KB PRG bank is fixed: writes to the register do not move it.
    #[test]
    fn prg_is_fixed_32k() {
        let mut m = cprom_mapper();
        assert_eq!(m.prg_read_byte(0x8000), 0x11);
        assert_eq!(m.prg_read_byte(0xFFFF), 0x22);
        m.prg_write_byte(0x8000, 0x03); // CHR select only, PRG must not move
        assert_eq!(m.prg_read_byte(0x8000), 0x11);
        assert_eq!(m.prg_read_byte(0xFFFF), 0x22);
    }

    // Reads below $8000 are not decoded.
    #[test]
    fn reads_below_8000_return_zero() {
        let mut m = cprom_mapper();
        assert_eq!(m.prg_read_byte(0x6000), 0);
        assert_eq!(m.prg_read_byte(0x7FFF), 0);
    }

    // The lower 4 KB pattern window ($0000-$0FFF) is hard-wired to CHR-RAM
    // bank 0 and never follows the bank register.
    #[test]
    fn lower_4k_is_fixed_to_bank_zero() {
        let mut m = cprom_mapper();
        m.chr_write_byte(0x0500, 0x42);
        for sel in 0..4u8 {
            m.prg_write_byte(0x8000, sel);
            assert_eq!(m.chr_read_byte(0x0500), 0x42);
        }
    }

    // The upper 4 KB pattern window ($1000-$1FFF) switches among the four
    // CHR-RAM banks selected by the register's low two bits.
    #[test]
    fn upper_4k_switches_banks() {
        let mut m = cprom_mapper();
        // Stamp each of the four upper banks at $1000 with a distinct byte.
        for bank in 0..4u8 {
            m.prg_write_byte(0x8000, bank);
            m.chr_write_byte(0x1000, 0xD0 + bank);
        }
        // Reading back with each bank selected must return that byte.
        for bank in 0..4u8 {
            m.prg_write_byte(0x8000, bank);
            assert_eq!(m.chr_read_byte(0x1000), 0xD0 + bank);
        }
    }

    // Only the low two bits of the register matter: bank 5 aliases bank 1.
    #[test]
    fn upper_bank_masks_to_two_bits() {
        let mut m = cprom_mapper();
        m.prg_write_byte(0x8000, 0x01);
        m.chr_write_byte(0x1000, 0x9E);
        m.prg_write_byte(0x8000, 0x05); // 5 & 3 == 1 -> same physical bank
        assert_eq!(m.chr_read_byte(0x1000), 0x9E);
        m.prg_write_byte(0x8000, 0xFF); // ...as does 0xFF & 3 == 3, not 1
        assert_ne!(m.chr_read_byte(0x1000), 0x9E);
    }

    // A 16 KB half-cart is mirrored across the fixed 32 KB window.
    #[test]
    fn half_size_prg_mirrors_across_window() {
        let mut cart = test_cart(1, 0); // 16 KB PRG
        cart.prg_rom[0x0000] = 0x7C;
        let mut m = Mapper13::new(cart);
        assert_eq!(m.prg_read_byte(0x8000), 0x7C);
        assert_eq!(m.prg_read_byte(0xC000), 0x7C); // mirror of $8000
    }

    // get_state/apply_state restores both the bank register and CHR-RAM.
    #[test]
    fn state_roundtrip_restores_bank_and_ram() {
        let mut m = cprom_mapper();
        m.prg_write_byte(0x8000, 0x02);
        m.chr_write_byte(0x1000, 0x55);
        m.chr_write_byte(0x0800, 0x66);
        let snap = m.get_state();
        // Clobber selection and RAM.
        m.prg_write_byte(0x8000, 0x00);
        m.chr_write_byte(0x1000, 0x00);
        m.chr_write_byte(0x0800, 0x00);
        m.apply_state(&snap);
        m.prg_write_byte(0x8000, 0x02);
        assert_eq!(m.chr_read_byte(0x1000), 0x55);
        assert_eq!(m.chr_read_byte(0x0800), 0x66);
    }

    // reset clears the upper-bank register back to bank 0.
    #[test]
    fn reset_clears_upper_bank() {
        let mut m = cprom_mapper();
        // Put a marker in bank 0 upper and a different one in bank 2 upper.
        m.prg_write_byte(0x8000, 0x00);
        m.chr_write_byte(0x1000, 0xAA);
        m.prg_write_byte(0x8000, 0x02);
        m.chr_write_byte(0x1000, 0xBB);
        m.reset();
        assert_eq!(m.chr_read_byte(0x1000), 0xAA);
    }
}
