use crate::cartridge::{self, Cartridge, Mirroring, PRG_ROM_BANK_SIZE};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper1 {
    cartridge: Cartridge,
    shift: u8,
    regs: Regs,
    /// 32 KB cached view of the currently-mapped PRG ROM, kept in sync
    /// with bank-select writes. Lets the AArch64 ASM CPU fast path
    /// read PRG directly via `prg_asm_ptr()` without having to call
    /// through the mapper's bank math on every fetch.
    prg_asm_window: Vec<u8>,
    /// Consecutive-write filter for the MMC1 shift register. Per the
    /// NESdev wiki: "If two writes to the MMC1 shift register occur
    /// on consecutive CPU cycles (such as the dummy + real writes of
    /// a Read-Modify-Write opcode like INC/DEC/ASL/LSR/ROL/ROR/SLO),
    /// the MMC1 ignores the second write."
    ///
    /// Without this, an `INC $FFFF` shifts in TWO bits instead of
    /// one — corrupting the shift-register protocol. Bill & Ted's
    /// (mapper 1) reset code at $FFE0 starts with `INC $FFFF`; the
    /// double-shift broke its MMC1 init and the CPU eventually
    /// landed in the IRQ vector trampoline.
    ///
    /// Stores the CPU cycle count of the most recent register write.
    /// A new write whose cycle == last_register_write_cycle + 1 is
    /// the second of a back-to-back pair — skipped. The cycle is
    /// tracked via `Mapper::set_cpu_cycle`, called by `Nes::tick`
    /// once per CPU cycle.
    last_register_write_cycle: u64,
    /// Most recent CPU cycle pushed via `set_cpu_cycle`.
    cur_cpu_cycle: u64,
    /// ASM bulk-step budget in CPU cycles. Defaults to 1 (one
    /// instruction per ASM invocation — the shipped path) so
    /// default-settings timing is unchanged. Runtime perf knob — NOT
    /// console state, so it is deliberately excluded from
    /// `get_state`/`apply_state` and survives resets/state loads.
    /// Raised via `set_asm_bulk_cycles_override` (clamped 1..=16)
    /// only after the per-game lockstep + Mesen-oracle gate passes;
    /// MMC1 has no IRQ line and bank-switch writes rebuild the ASM
    /// window in place, so batching is structurally safe, but timed
    /// $2002 polling (sprite-0 waits) can still observe the coarser
    /// tick granularity.
    asm_bulk_budget: i64,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct Regs {
    control: u8,
    prg_bank: u8,
    chr_bank_0: u8,
    chr_bank_1: u8,
}

impl Regs {
    fn new() -> Regs {
        Regs {
            control: 0x0C,
            prg_bank: 0,
            chr_bank_0: 0,
            chr_bank_1: 0,
        }
    }
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
enum PrgRomMode {
    Switch32Kb,   // Switch 32 KB at $8000, ignoring low bit of bank number
    FixFirstBank, // Fix first bank at $8000 and switch 16 KB bank at $C000
    FixLastBank,  // Fix last bank at $C000 and switch 16 KB bank at $8000
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
enum ChrRomMode {
    Switch8Kb, // Switch 8 KB at a time
    Switch4Kb, // Switch two separate 4 KB banks
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub shift: u8,
    pub regs: Regs,
}

// Put a 1 in bit 4 so we can detect when we've shifted enough to write to a register
const SHIFT_REGISTER_DEFAULT: u8 = 0x10;

impl Mapper1 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper1 {
            cartridge,
            shift: SHIFT_REGISTER_DEFAULT,
            regs: Regs::new(),
            prg_asm_window: vec![0u8; 32 * 1024],
            last_register_write_cycle: u64::MAX,
            cur_cpu_cycle: 0,
            asm_bulk_budget: 1,
        };
        m.rebuild_asm_window();
        m
    }

    /// Copy the currently-mapped first + last 16 KB PRG banks into the
    /// flat 32 KB ASM window. Called on every bank-select register
    /// write and on reset/apply_state.
    fn rebuild_asm_window(&mut self) {
        let first_bank = self.prg_rom_bank_first();
        let last_bank = self.prg_rom_bank_last();
        let bank_size = PRG_ROM_BANK_SIZE as usize;
        let first_off = (first_bank as usize) * bank_size;
        let last_off = (last_bank as usize) * bank_size;
        let prg = &self.cartridge.prg_rom;
        // Safely pull from the ROM even if a bank index is out of
        // range — wrap modulo the ROM's bank count.
        let banks = self.cartridge.prg_rom_num_banks as usize;
        let first = first_off % prg.len().max(1);
        let last = last_off % prg.len().max(1);
        let _ = banks; // silences unused warning in release builds
        if prg.len() >= bank_size {
            self.prg_asm_window[..bank_size]
                .copy_from_slice(&prg[first..first + bank_size]);
            self.prg_asm_window[bank_size..]
                .copy_from_slice(&prg[last..last + bank_size]);
        }
    }

    fn prg_rom_mode(&self) -> PrgRomMode {
        let control = (self.regs.control & 0x0F) >> 2;
        match control {
            0 | 1 => PrgRomMode::Switch32Kb,
            2 => PrgRomMode::FixFirstBank,
            _ => PrgRomMode::FixLastBank,
        }
    }

    fn chr_rom_mode(&self) -> ChrRomMode {
        if self.regs.control & 0x10 == 0 {
            ChrRomMode::Switch8Kb
        } else {
            ChrRomMode::Switch4Kb
        }
    }

    fn write_register(&mut self, address: u16, shift: u8) {
        if address < 0xA000 {
            self.regs.control = shift & 0x1F;
            self.cartridge.mirroring = match self.regs.control & 0x03 {
                0 => Mirroring::OneScreenLower,
                1 => Mirroring::OneScreenUpper,
                2 => Mirroring::Vertical,
                _ => Mirroring::Horizontal,
            };
        } else if address < 0xC000 {
            self.regs.chr_bank_0 = shift & 0x1F;
        } else if address < 0xE000 {
            self.regs.chr_bank_1 = shift & 0x1F;
        } else {
            self.regs.prg_bank = shift & 0x0F;
        }
    }

    /// SUROM/SXROM boards carry 512 KiB of PRG ROM — one more bank bit
    /// than the MMC1's 4-bit `prg_bank` register can express. That fifth
    /// bit (PRG A18, the high 256 KiB "block" select) is supplied by
    /// bit 4 of the CHR bank 0 register. On these boards CHR is 8 KiB
    /// RAM, so CHR bit 4 is otherwise unused and free for PRG banking.
    ///
    /// On standard boards (<= 256 KiB PRG) CHR bit 4 is a genuine CHR
    /// bank line and MUST NOT bleed into PRG addressing, so this
    /// returns 0 there.
    fn prg_a18(&self) -> u8 {
        if self.cartridge.prg_rom.len() > 256 * 1024 {
            self.regs.chr_bank_0 & 0x10
        } else {
            0
        }
    }

    fn prg_rom_bank_first(&self) -> u8 {
        let a18 = self.prg_a18();
        match self.prg_rom_mode() {
            PrgRomMode::Switch32Kb => (self.regs.prg_bank & 0xFE) | a18,
            PrgRomMode::FixFirstBank => a18,
            PrgRomMode::FixLastBank => self.regs.prg_bank | a18,
        }
    }

    fn prg_rom_bank_last(&self) -> u8 {
        let a18 = self.prg_a18();
        match self.prg_rom_mode() {
            PrgRomMode::Switch32Kb => (self.regs.prg_bank & 0xFE) | 0x01 | a18,
            PrgRomMode::FixFirstBank => self.regs.prg_bank | a18,
            PrgRomMode::FixLastBank => {
                if self.cartridge.prg_rom.len() > 256 * 1024 {
                    // The fixed bank tracks A18: it is the LAST 16 KiB
                    // bank *within the currently-selected 256 KiB block*
                    // (0x0F inside the block), not the last bank of the
                    // whole 512 KiB ROM.
                    0x0F | a18
                } else {
                    self.cartridge.prg_rom_num_banks - 1
                }
            }
        }
    }

    fn prg_rom_address(&self, bank: u8, address: u16) -> usize {
        let raw = (bank as usize * PRG_ROM_BANK_SIZE as usize)
            | (address as usize & (PRG_ROM_BANK_SIZE as usize - 1));
        // Guard for carts whose prg_bank register can address beyond
        // the actual prg_rom length (observed on MMC1 variants like
        // Mega Man 2's 256 KB PRG — the software writes bank indices
        // that extend past the on-cart ROM window). Mask instead of
        // panicking on indexing.
        raw & (self.cartridge.prg_rom.len().max(1) - 1)
    }

    fn chr_address(&self, address: u16) -> usize {
        let raw = match self.chr_rom_mode() {
            ChrRomMode::Switch4Kb => {
                let bank = if address < 0x1000 {
                    self.regs.chr_bank_0
                } else {
                    self.regs.chr_bank_1
                };
                (bank as usize * 0x1000) | (address as usize & 0x0FFF)
            }
            ChrRomMode::Switch8Kb => {
                let bank = self.regs.chr_bank_0 & !1;
                (bank as usize * 0x1000) | (address as usize & 0x1FFF)
            }
        };
        // 512KB-PRG MMC1 boards (SUROM, SXROM, etc.) repurpose the
        // chr_bank registers' high bits as the PRG bank-high bit. On
        // CHR-RAM carts (which are always 8KB) those high bits make
        // the raw chr address overflow the 8KB buffer. Wrapping by
        // chr.len() keeps both CHR-RAM (wrap to 8KB) and CHR-ROM
        // (no-op when bank is in range) correct without branching on
        // chr_is_ram.
        raw & (self.cartridge.chr.len().max(1) - 1)
    }
}

impl Mapper for Mapper1 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        if address < 0x6000 {
            0
        } else if address < 0x8000 {
            if self.cartridge.prg_ram.is_empty() {
                0
            } else {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx]
            }
        } else if address < 0xC000 {
            let rom_addr = self.prg_rom_address(self.prg_rom_bank_first(), address);
            self.cartridge.prg_rom[rom_addr]
        } else {
            let rom_addr = self.prg_rom_address(self.prg_rom_bank_last(), address);
            self.cartridge.prg_rom[rom_addr]
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address < 0x6000 {
            // Do nothing
            return;
        }
        if address < 0x8000 {
            if !self.cartridge.prg_ram.is_empty() {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx] = value
            }
            // PRG-RAM writes don't go through the shift register, so
            // they don't arm the consecutive-write filter either.
            return;
        }

        // Mapper register write at $8000-$FFFF. If the previous bus
        // operation was ALSO a register write (RMW dummy + real
        // write back-to-back), MMC1 ignores this second one — see
        // `last_op_was_register_write` doc comment for the full
        // rationale. Still mark this op as a register write so any
        // FURTHER consecutive write would also be filtered, but the
        // shift-register state is left untouched.
        if self.cur_cpu_cycle == self.last_register_write_cycle.wrapping_add(1) {
            // RMW dummy + real write back-to-back: ignore the second.
            // Update last_register_write_cycle so a 3rd consecutive
            // write (only possible from non-RMW pathological code)
            // would also be filtered correctly.
            self.last_register_write_cycle = self.cur_cpu_cycle;
            return;
        }
        self.last_register_write_cycle = self.cur_cpu_cycle;

        if (value & 0x80) == 0 {
            // If a 1 has been shifted into bit 0, it's time to write to a register
            let is_last_shift = (self.shift & 0x01) != 0;

            // Bit 0 of the value gets shifted into the shift
            // register from the left, starting at bit 4.
            self.shift = (self.shift >> 1) | ((value & 0x01) << 4);

            if is_last_shift {
                let shift = self.shift;
                self.write_register(address, shift);
                self.shift = SHIFT_REGISTER_DEFAULT;
                // Bank state just changed — rebuild the flat window
                // that the ASM CPU reads through.
                self.rebuild_asm_window();
            }
        } else {
            // Writing a value with bit 7 set clears the shift register to its initial state
            // and forces FixLastBank PRG mode. The previous code did NOT
            // rebuild the asm window after this; the ASM CPU fast path
            // would then read PRG via stale bank pointers until the
            // game's next non-reset register write triggered a rebuild.
            self.shift = SHIFT_REGISTER_DEFAULT;
            self.regs.control |= 0x0C;
            self.rebuild_asm_window();
        }
    }


    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr] = value
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn sram(&mut self) -> *mut u8 {
        self.cartridge.prg_ram.as_mut_ptr() as *mut _
    }

    fn sram_size(&self) -> usize {
        self.cartridge.prg_ram.len()
    }

    fn reset(&mut self) {
        self.shift = SHIFT_REGISTER_DEFAULT;
        self.regs = Regs::new();
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        self.asm_bulk_budget
    }

    fn set_asm_bulk_cycles_override(&mut self, cycles: i64) {
        // 16 is the top rung of the measured ladder (2-5 instructions
        // per batch); larger budgets widen the mid-batch window in
        // which timed $2002 reads / APU IRQ service slip without
        // buying meaningful extra amortization.
        self.asm_bulk_budget = cycles.clamp(1, 16);
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State1(State {
            cartridge: self.cartridge.get_state(),
            shift: self.shift,
            regs: self.regs,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State1(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.shift = state.shift;
                self.regs = state.regs;
                self.rebuild_asm_window();
                // Transient consecutive-write guard isn't persisted;
                // reset so the next write is unconditionally honored.
                self.last_register_write_cycle = u64::MAX;
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }

    fn set_cpu_cycle(&mut self, cpu_cycle: u64) {
        self.cur_cpu_cycle = cpu_cycle;
    }
}
