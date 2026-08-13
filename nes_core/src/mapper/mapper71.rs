use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

// Camerica / Codemasters BF909x. Functionally a UxROM variant:
//   $C000-$FFFF writes → 16 KB PRG bank for the $8000-$BFFF window.
//   $8000-$BFFF fixed to the last 16 KB bank.
//   $9000-$9FFF writes (Fire Hawk only) → single-screen mirroring control.
pub struct Mapper71 {
    cartridge: Cartridge,
    switchable_bank: u8,
    // Fire Hawk uses $9000 to toggle single-screen mirroring. Other
    // Codemasters boards never write there, so treating this as a
    // latched mirroring control is safe for the whole family.
    fire_hawk_mirroring: bool,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub switchable_bank: u8,
    pub fire_hawk_mirroring: bool,
}

impl Mapper71 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper71 {
            cartridge,
            switchable_bank: 0,
            fire_hawk_mirroring: false,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_banks_count(&self) -> usize {
        self.cartridge.prg_rom_num_banks.max(1) as usize
    }

    fn rebuild_asm_window(&mut self) {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let low_bank = (self.switchable_bank as usize) % self.prg_banks_count();
        let high_bank = self.prg_banks_count().saturating_sub(1);
        let prg = &self.cartridge.prg_rom;
        let low_off = low_bank * bank_size;
        let high_off = high_bank * bank_size;
        if low_off + bank_size <= prg.len() {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[low_off..low_off + bank_size]);
        }
        if high_off + bank_size <= prg.len() {
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[high_off..high_off + bank_size]);
        }
    }
}

impl Mapper for Mapper71 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x8000 {
            return 0;
        }
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let bank = if address < 0xC000 {
            (self.switchable_bank as usize) % self.prg_banks_count()
        } else {
            self.prg_banks_count() - 1
        };
        self.cartridge.prg_rom[bank * bank_size | (address as usize & (bank_size - 1))]
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x9000..=0x9FFF => {
                self.fire_hawk_mirroring = true;
                self.cartridge.mirroring = if value & 0x10 == 0 {
                    Mirroring::OneScreenLower
                } else {
                    Mirroring::OneScreenUpper
                };
            }
            0xC000..=0xFFFF | 0x8000..=0xBFFF => {
                self.switchable_bank = value & 0x0F;
                self.rebuild_asm_window();
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        self.cartridge.chr[address as usize]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        self.cartridge.chr[address as usize] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.switchable_bank = 0;
        self.fire_hawk_mirroring = false;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        4
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State71(State {
            cartridge: self.cartridge.get_state(),
            switchable_bank: self.switchable_bank,
            fire_hawk_mirroring: self.fire_hawk_mirroring,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State71(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.switchable_bank = state.switchable_bank;
                self.fire_hawk_mirroring = state.fire_hawk_mirroring;
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
            mapper: 71,
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

    // Stamp the first byte of every 16 KB PRG bank with a marker so a read
    // through the $8000/$C000 windows reveals which bank is mapped.
    fn stamped_mapper(prg_16k: u8) -> Mapper71 {
        let mut cart = test_cart(prg_16k, 1);
        for b in 0..prg_16k as usize {
            cart.prg_rom[b * 0x4000] = 0xB0 + b as u8;
        }
        Mapper71::new(cart)
    }

    const LAST: u8 = 0xB0 + 7; // marker of the last of eight 16 KB banks

    // Writes to $C000-$FFFF set the switchable 16 KB bank seen at $8000.
    #[test]
    fn c000_write_selects_low_window_bank() {
        let mut m = stamped_mapper(8);
        for bank in 0..8u8 {
            m.prg_write_byte(0xC000, bank);
            assert_eq!(m.prg_read_byte(0x8000), 0xB0 + bank);
        }
    }

    // The $C000-$FFFF window is hard-fixed to the last 16 KB bank and never
    // follows the switchable register.
    #[test]
    fn high_window_is_fixed_to_last_bank() {
        // Stamp both ends of the last (8th) bank so the $C000 and $FFFF
        // reads land on distinct, known bytes.
        let mut cart = test_cart(8, 1);
        for b in 0..8usize {
            cart.prg_rom[b * 0x4000] = 0xB0 + b as u8;
        }
        cart.prg_rom[7 * 0x4000 + 0x3FFF] = 0x7E; // last byte of last bank
        let mut m = Mapper71::new(cart);
        for bank in 0..8u8 {
            m.prg_write_byte(0xC000, bank);
            assert_eq!(m.prg_read_byte(0xC000), LAST); // start of last bank
            assert_eq!(m.prg_read_byte(0xFFFF), 0x7E); // end of last bank
        }
    }

    // Writes to $8000-$BFFF also latch the bank on these boards.
    #[test]
    fn writes_in_8000_bfff_also_select_bank() {
        let mut m = stamped_mapper(8);
        m.prg_write_byte(0xA000, 4);
        assert_eq!(m.prg_read_byte(0x8000), 0xB0 + 4);
    }

    // The bank register is masked to 4 bits and then reduced modulo the
    // bank count, so an over-range value wraps instead of reading OOB.
    #[test]
    fn bank_selection_wraps_modulo_count() {
        let mut m = stamped_mapper(8);
        m.prg_write_byte(0xC000, 0x0A); // 10 % 8 == 2
        assert_eq!(m.prg_read_byte(0x8000), 0xB0 + 2);
    }

    // The switchable low-window offset tracks the address low 14 bits.
    #[test]
    fn low_window_offset_tracks_address() {
        let mut cart = test_cart(2, 1);
        cart.prg_rom[0x0123] = 0x5A; // bank 0
        let mut m = Mapper71::new(cart);
        m.prg_write_byte(0xC000, 0); // select bank 0 at $8000
        assert_eq!(m.prg_read_byte(0x8123), 0x5A);
    }

    // $9000-$9FFF is the Fire Hawk single-screen mirroring latch: bit 4
    // clear selects the lower nametable, set selects the upper.
    #[test]
    fn fire_hawk_9000_controls_single_screen_mirroring() {
        let mut m = stamped_mapper(8);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(0x9000, 0x00);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
        m.prg_write_byte(0x9000, 0x10);
        assert_eq!(m.mirroring(), Mirroring::OneScreenUpper);
    }

    // A $9000 mirroring write must not disturb the selected PRG bank.
    #[test]
    fn mirroring_write_leaves_bank_untouched() {
        let mut m = stamped_mapper(8);
        m.prg_write_byte(0xC000, 5);
        m.prg_write_byte(0x9000, 0x10); // mirroring only
        assert_eq!(m.prg_read_byte(0x8000), 0xB0 + 5);
    }

    // Reads below $8000 are not decoded.
    #[test]
    fn reads_below_8000_return_zero() {
        let mut m = stamped_mapper(8);
        assert_eq!(m.prg_read_byte(0x6000), 0);
        assert_eq!(m.prg_read_byte(0x7FFF), 0);
    }

    // CHR is RAM on these boards: a write then read round-trips.
    #[test]
    fn chr_ram_write_read_roundtrip() {
        let mut m = stamped_mapper(8);
        m.chr_write_byte(0x0400, 0x3C);
        assert_eq!(m.chr_read_byte(0x0400), 0x3C);
    }

    // get_state/apply_state restores the bank register and mirroring.
    #[test]
    fn state_roundtrip_restores_bank_and_mirroring() {
        let mut m = stamped_mapper(8);
        m.prg_write_byte(0xC000, 3);
        m.prg_write_byte(0x9000, 0x00); // OneScreenLower
        let snap = m.get_state();
        m.reset(); // bank 0, mirroring back to default
        assert_eq!(m.prg_read_byte(0x8000), 0xB0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 0xB0 + 3);
        assert_eq!(m.mirroring(), Mirroring::OneScreenLower);
    }

    // reset clears the bank latch to 0 and restores the default mirroring.
    #[test]
    fn reset_restores_power_on_state() {
        let mut m = stamped_mapper(8);
        m.prg_write_byte(0xC000, 6);
        m.prg_write_byte(0x9000, 0x10); // OneScreenUpper
        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0xB0);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }
}
