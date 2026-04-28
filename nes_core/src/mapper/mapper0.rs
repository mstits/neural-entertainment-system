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
            self.cartridge.prg_ram[(address & 0x1FFF) as usize]
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
        self.cartridge.chr[address as usize]
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
        self.cartridge.chr[address as usize] = value;
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
