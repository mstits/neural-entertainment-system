use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};
use serde_derive::{Deserialize, Serialize};

pub struct Mapper2 {
    cartridge: Cartridge,
    switchable_bank: u8,
    prg_asm_window: Vec<u8>,
    /// ASM bulk-step budget in CPU cycles. Defaults to 1 (the
    /// shipped path) so default-settings timing is unchanged.
    /// Runtime perf knob — NOT console state, so it is excluded from
    /// `get_state`/`apply_state` and survives resets/state loads.
    /// Raised via `set_asm_bulk_cycles_override` (clamped 1..=16)
    /// only after the per-game lockstep + Mesen-oracle gate passes;
    /// UxROM has no IRQ line and bank-switch writes rebuild the ASM
    /// window in place, but action games that race sprite-0 via
    /// timed $2002 polling (Contra status split) can still observe
    /// the coarser tick granularity.
    asm_bulk_budget: i64,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub switchable_bank: u8,
}

impl Mapper2 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper2 {
            cartridge,
            switchable_bank: 0,
            prg_asm_window: vec![0u8; 32 * 1024],
            asm_bulk_budget: 1,
        };
        m.rebuild_asm_window();
        m
    }

    fn rebuild_asm_window(&mut self) {
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let low_bank = self.switchable_bank;
        let high_bank = self.cartridge.prg_rom_num_banks.saturating_sub(1);
        let prg = &self.cartridge.prg_rom;
        let low_off = (low_bank as usize) * bank_size;
        let high_off = (high_bank as usize) * bank_size;
        if low_off + bank_size <= prg.len() {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[low_off..low_off + bank_size]);
        }
        if high_off + bank_size <= prg.len() {
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[high_off..high_off + bank_size]);
        }
    }

    fn prg_rom_address(bank: u8, address: u16) -> usize {
        (bank as usize * PRG_ROM_BANK_SIZE as usize)
            | (address as usize & (PRG_ROM_BANK_SIZE as usize - 1))
    }

    fn read_prg_rom(&self, address: u16) -> u8 {
        let bank = if address < 0xC000 {
            self.switchable_bank
        } else {
            self.cartridge.prg_rom_num_banks - 1
        };

        let rom_addr = Mapper2::prg_rom_address(bank, address);
        self.cartridge.prg_rom[rom_addr]
    }
}

impl Mapper for Mapper2 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else if address < 0x8000 {
            // UxROM does not officially have SRAM, but a handful of carts
            // (and homebrew) use $6000-$7FFF as work RAM. Pass through if
            // the cartridge declared PRG-RAM, otherwise return open bus.
            if self.cartridge.prg_ram.is_empty() {
                0
            } else {
                let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx]
            }
        } else {
            self.read_prg_rom(address)
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) && !self.cartridge.prg_ram.is_empty() {
            let idx = (address & 0x1FFF) as usize % self.cartridge.prg_ram.len();
            self.cartridge.prg_ram[idx] = value;
        } else if address >= 0x8000 {
            self.switchable_bank =
                ((value as usize) % (self.cartridge.prg_rom_num_banks) as usize) as u8;
            self.rebuild_asm_window();
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
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        self.asm_bulk_budget
    }

    fn set_asm_bulk_cycles_override(&mut self, cycles: i64) {
        // Same ceiling rationale as Mapper1: 16 cycles = 2-5
        // instructions per batch, the top rung of the measured ladder.
        self.asm_bulk_budget = cycles.clamp(1, 16);
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State2(State {
            cartridge: self.cartridge.get_state(),
            switchable_bank: self.switchable_bank,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State2(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.switchable_bank = state.switchable_bank;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }
}
