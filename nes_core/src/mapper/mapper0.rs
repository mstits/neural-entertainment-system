use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};
use serde_derive::{Deserialize, Serialize};

pub struct Mapper0 {
    cartridge: Cartridge,
    /// Flat 32 KB PRG ROM view for the AArch64 ASM CPU fast path. For
    /// 32 KB carts this mirrors `cartridge.prg_rom`; for 16 KB carts
    /// it holds a duplicated copy (low bank at 0x0000 and 0x4000) so
    /// the ASM's 0x7FFF mask resolves correctly.
    prg_flat_32k: Vec<u8>,
}

impl Mapper0 {
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
        Mapper0 { cartridge, prg_flat_32k }
    }
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
}

impl Mapper for Mapper0 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else if address < 0x8000 {
            if self.cartridge.prg_ram.is_empty() {
                return 0;
            }
            let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
            self.cartridge.prg_ram[idx]
        } else if self.cartridge.prg_rom.len() > PRG_ROM_BANK_SIZE as usize {
            self.cartridge.prg_rom[(address & 0x7FFF) as usize]
        } else {
            // Mirror second bank to first
            self.cartridge.prg_rom[(address & 0x3FFF) as usize]
        }
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        if self.prg_flat_32k.len() == 32 * 1024 {
            Some(self.prg_flat_32k.as_ptr())
        } else {
            None
        }
    }

    /// 2026-04-26 follow-up: predict-NMI-fire cap in `Nes::step`
    /// (`Ppu::cpu_cycles_until_nmi_fire`) dynamically shrinks the
    /// effective budget as vblank approaches, so this constant is the
    /// MAXIMUM ceiling between vblank events. The cap guarantees the
    /// ASM batch ends at-most-one-instruction past the vblank rising
    /// edge — matching real 6502 NMI delivery semantics (services at
    /// next instruction boundary). 64 amortizes per-call ASM dispatch
    /// overhead well without losing meaningful resolution: most ASM
    /// batches exit on MMIO ($2002, $2007, $4014, etc.) far below
    /// this ceiling, and the cycles-exit path stays correct even
    /// for long MMIO-free stretches via the predict-fire cap.
    /// Verified by `tests/parity/test_mesen_lockstep.py` (31/31 ROMs)
    /// and `tests/parity/` (585/585 byte-exact).
    fn asm_bulk_cycles(&self) -> i64 {
        64
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) && !self.cartridge.prg_ram.is_empty() {
            // $6000-$7FFF maps to the 8 KB SRAM window. Mask by 0x1FFF
            // (NOT 0x0100, which was a historical typo that aliased
            // every write into a 256-byte stripe and corrupted save
            // states / scratchpad data). Second modulo by the actual
            // prg_ram length guards against smaller cartridges.
            let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
            self.cartridge.prg_ram[idx] = value;
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr = &self.cartridge.chr;
        if chr.is_empty() {
            return 0;
        }
        chr[(address as usize) & (chr.len() - 1)]
    }

    fn chr_static_ptr(&self) -> Option<*const u8> {
        // NROM has a single static 8 KB CHR window with no banking;
        // pointer remains valid for the mapper's lifetime.
        if self.cartridge.chr.len() >= 0x2000 {
            Some(self.cartridge.chr.as_ptr())
        } else {
            None
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr = &mut self.cartridge.chr;
        if chr.is_empty() {
            return;
        }
        let mask = chr.len() - 1;
        chr[(address as usize) & mask] = value;
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        // Nothing to reset
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State0(State {
            cartridge: self.cartridge.get_state(),
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State0(state) => {
                self.cartridge.apply_state(&state.cartridge);
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
            mapper: 0,
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

    // 32 KB PRG maps linearly across $8000-$FFFF (address & 0x7FFF).
    #[test]
    fn nrom_32k_prg_linear_mapping() {
        let mut cart = test_cart(2, 1);
        cart.prg_rom[0x0000] = 0xA0; // maps to $8000
        cart.prg_rom[0x4000] = 0xC0; // maps to $C000
        cart.prg_rom[0x7FFF] = 0xFF; // maps to $FFFF
        let mut m = Mapper0::new(cart);
        assert_eq!(m.prg_read_byte(0x8000), 0xA0);
        assert_eq!(m.prg_read_byte(0xC000), 0xC0);
        assert_eq!(m.prg_read_byte(0xFFFF), 0xFF);
    }

    // 16 KB PRG mirrors the single bank into both halves of $8000-$FFFF.
    #[test]
    fn nrom_16k_prg_mirrors_high_half() {
        let mut cart = test_cart(1, 1);
        cart.prg_rom[0x0000] = 0x42;
        cart.prg_rom[0x3FFF] = 0x99;
        let mut m = Mapper0::new(cart);
        // $8000 and $C000 fold to the same offset (address & 0x3FFF).
        assert_eq!(m.prg_read_byte(0x8000), 0x42);
        assert_eq!(m.prg_read_byte(0xC000), 0x42);
        assert_eq!(m.prg_read_byte(0xBFFF), 0x99);
        assert_eq!(m.prg_read_byte(0xFFFF), 0x99);
    }

    // Reads below $6000 are open bus (0), not ROM.
    #[test]
    fn nrom_below_6000_reads_zero() {
        let mut cart = test_cart(2, 1);
        cart.prg_rom[0] = 0xAB;
        let mut m = Mapper0::new(cart);
        assert_eq!(m.prg_read_byte(0x0000), 0);
        assert_eq!(m.prg_read_byte(0x5FFF), 0);
    }

    // PRG-RAM window $6000-$7FFF round-trips writes (masked by 0x1FFF).
    #[test]
    fn nrom_prg_ram_read_write() {
        let cart = test_cart(2, 1);
        let mut m = Mapper0::new(cart);
        m.prg_write_byte(0x6000, 0x55);
        m.prg_write_byte(0x7FFF, 0xAA);
        assert_eq!(m.prg_read_byte(0x6000), 0x55);
        assert_eq!(m.prg_read_byte(0x7FFF), 0xAA);
    }

    // CHR-RAM round-trips writes across the full 8 KB window.
    #[test]
    fn nrom_chr_read_write() {
        let cart = test_cart(2, 1);
        let mut m = Mapper0::new(cart);
        m.chr_write_byte(0x0000, 0x33);
        m.chr_write_byte(0x1FFF, 0x44);
        assert_eq!(m.chr_read_byte(0x0000), 0x33);
        assert_eq!(m.chr_read_byte(0x1FFF), 0x44);
    }

    // mirroring() reflects the cartridge header value.
    #[test]
    fn nrom_mirroring_from_cartridge() {
        let mut cart = test_cart(2, 1);
        cart.mirroring = Mirroring::Vertical;
        let m = Mapper0::new(cart);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
    }

    // get_state/apply_state restores CHR-RAM and PRG-RAM contents.
    #[test]
    fn nrom_state_round_trip() {
        let cart = test_cart(2, 1);
        let mut m = Mapper0::new(cart);
        m.chr_write_byte(0x0010, 0x7E);
        m.prg_write_byte(0x6001, 0x3C);
        let snap = m.get_state();
        // Clobber both RAM regions.
        m.chr_write_byte(0x0010, 0x00);
        m.prg_write_byte(0x6001, 0x00);
        m.apply_state(&snap);
        assert_eq!(m.chr_read_byte(0x0010), 0x7E);
        assert_eq!(m.prg_read_byte(0x6001), 0x3C);
    }
}
