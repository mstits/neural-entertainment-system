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
    /// True when interpreter/PPU-driven PRG reads in $8000-$FFFF may be
    /// served straight from `prg_asm_window` instead of recomputing the
    /// bank mapping on every read. The window is rebuilt with a `% len`
    /// wrap (`rebuild_asm_window`) whereas the exact bank math wraps
    /// `& (len - 1)` (`prg_rom_address`); the two are identical only for
    /// power-of-two PRG — every commercial MMC1 board (SxROM 32 KB…
    /// 512 KB are all powers of two). Non-pow2 dumps set this `false`
    /// and keep the exact bank math. Derived once from PRG length at
    /// construction (length is fixed for the mapper's lifetime), so it
    /// is not console state and is excluded from get_state/apply_state.
    /// Perf note: under the production ASM CPU build this is ~0% (the
    /// ASM core already fetches through the same window); it pays only
    /// on interpreter builds / disable_asm_cpu. The measured MMC1 win
    /// (~4.6% env-step, Zelda fs=16) is entirely the CHR window below.
    prg_window_ok: bool,
    /// Flat 8 KB view of the currently-mapped CHR ([$0000-$0FFF][
    /// $1000-$1FFF]), rebuilt in place on every CHR bank / mode change.
    /// Lets `Ppu::read_chr_byte` serve pattern fetches straight from
    /// this buffer (via `chr_static_ptr`) instead of dispatching to
    /// `chr_read_byte` + `chr_address` bank math on every one of the
    /// ~16.8k CHR reads/frame. Because it is rebuilt *in place* (the Vec
    /// is never reallocated, only its bytes rewritten), the PPU's
    /// scanline-boundary-cached pointer stays valid and observes a
    /// mid-scanline bank switch immediately — no staleness. CHR-RAM
    /// writes update the window coherently in `chr_write_byte`.
    chr_window: Vec<u8>,
    /// True when the CHR window is usable (CHR is a whole number of
    /// 8 KB banks — every real NES CHR layout). MMC1 has no CHR-read
    /// side effects (unlike MMC3's A12 IRQ clock), so bypassing the
    /// per-fetch dispatch is safe. Derived once from CHR length; not
    /// console state.
    chr_window_active: bool,
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
        let prg_len = cartridge.prg_rom.len();
        // The flat window covers [first 16 KB][last 16 KB]; it is only
        // populated when the ROM has at least one full bank, and only
        // matches the exact bank math for power-of-two PRG (see
        // `prg_window_ok`).
        let prg_window_ok =
            prg_len.is_power_of_two() && prg_len >= PRG_ROM_BANK_SIZE as usize;
        // CHR window covers a full 8 KB, addressed as two 4 KB pages; it
        // is exact for any CHR that is a whole number of 8 KB banks (all
        // NES CHR ROM/RAM — the iNES CHR unit is 8 KB, CHR-RAM is 8 KB),
        // because `rebuild_chr_window` mirrors the same `& (len - 1)`
        // masking `chr_address` uses and copies contiguous 8 KB / 4 KB
        // spans that stay in bounds when `len` is an 8 KB multiple.
        let chr_len = cartridge.chr.len();
        let chr_window_active = chr_len >= 0x2000 && chr_len % 0x2000 == 0;
        let mut m = Mapper1 {
            cartridge,
            shift: SHIFT_REGISTER_DEFAULT,
            regs: Regs::new(),
            prg_asm_window: vec![0u8; 32 * 1024],
            last_register_write_cycle: u64::MAX,
            cur_cpu_cycle: 0,
            asm_bulk_budget: 1,
            prg_window_ok,
            chr_window: vec![0u8; 8 * 1024],
            chr_window_active,
        };
        m.rebuild_asm_window();
        m.rebuild_chr_window();
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

    /// Copy the currently-mapped CHR into the flat 8 KB window, mirroring
    /// `chr_address` exactly. In 8 KB mode the whole 8 KB is one
    /// contiguous CHR slice; in 4 KB mode the two 4 KB pages come from
    /// the `chr_bank_0` / `chr_bank_1` banks. Rewrites in place, so the
    /// PPU's cached window pointer sees the change without a re-fetch.
    fn rebuild_chr_window(&mut self) {
        if !self.chr_window_active {
            return;
        }
        let chr = &self.cartridge.chr;
        let mask = chr.len() - 1;
        match self.chr_rom_mode() {
            ChrRomMode::Switch8Kb => {
                // Base is 8 KB-aligned within CHR (matches `chr_address`
                // for $0000-$1FFF); the mapped 8 KB is contiguous.
                let base = ((self.regs.chr_bank_0 as usize & !1) * 0x1000) & mask;
                self.chr_window.copy_from_slice(&chr[base..base + 0x2000]);
            }
            ChrRomMode::Switch4Kb => {
                let base_lo = ((self.regs.chr_bank_0 as usize) * 0x1000) & mask;
                let base_hi = ((self.regs.chr_bank_1 as usize) * 0x1000) & mask;
                self.chr_window[0x0000..0x1000]
                    .copy_from_slice(&chr[base_lo..base_lo + 0x1000]);
                self.chr_window[0x1000..0x2000]
                    .copy_from_slice(&chr[base_hi..base_hi + 0x1000]);
            }
        }
    }

    /// Exact CPU read for $6000-$FFFF via per-read bank math. This is the
    /// reference path; `prg_peek_byte` serves $8000-$FFFF from the flat
    /// window when `prg_window_ok`, and falls back here otherwise
    /// (PRG-RAM window at $6000-$7FFF, non-pow2 PRG, or the A/B knob).
    fn prg_peek_bankmath(&self, address: u16) -> u8 {
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
}

impl Mapper for Mapper1 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        // $8000-$FFFF PRG ROM served from the flat 32 KB window that
        // bank-select writes already keep in sync (the same buffer the
        // AArch64 ASM CPU reads through via `prg_asm_ptr`). This
        // collapses the per-read bank recompute — PRG mode decode +
        // first/last bank select + address mul/mask/wrap + bounds
        // check — into a single masked array index. `prg_window_ok`
        // guards the pow2 wrap-equivalence caveat; the
        // `mmc1_prg_bankmath_ab` cargo feature forces the exact bank
        // math for A/B measurement (default off — window is the
        // shipped path). Byte-identical to `prg_peek_bankmath` for all
        // banking states (proven by `prg_window_matches_bankmath`).
        #[cfg(not(feature = "mmc1_prg_bankmath_ab"))]
        if address >= 0x8000 && self.prg_window_ok {
            return self.prg_asm_window[(address & 0x7FFF) as usize];
        }
        self.prg_peek_bankmath(address)
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
        //
        // `last_register_write_cycle == u64::MAX` is `apply_state`'s
        // "no previous write" sentinel (see its doc comment). Without
        // this guard, `u64::MAX.wrapping_add(1) == 0` collides with a
        // freshly-restored, never-ticked machine's `cur_cpu_cycle == 0`
        // and the sentinel meant to unconditionally honor the next
        // write instead silently drops it — see
        // `restore_consecutive_write_tests` below and
        // docs/research/ASM_CPU_STATUS_2026-08-25.md.
        if self.last_register_write_cycle != u64::MAX
            && self.cur_cpu_cycle == self.last_register_write_cycle.wrapping_add(1)
        {
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
                // Bank state just changed — rebuild the flat PRG window
                // the ASM CPU / interpreter read through and the CHR
                // window the PPU fetches through.
                self.rebuild_asm_window();
                self.rebuild_chr_window();
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
            // Reset only forces PRG FixLastBank (control bits 2-3); CHR
            // mapping is unchanged, but rebuild for symmetry — a no-op
            // copy of the same 8 KB, on the rare shift-reset write.
            self.rebuild_chr_window();
        }
    }


    fn chr_read_byte(&mut self, address: u16) -> u8 {
        // Served from the maintained flat window when active
        // (byte-identical to the bank math); MMC1 has no CHR-read side
        // effect, so this is a pure indexed load. The
        // `mmc1_chr_dispatch_ab` feature reverts to the bank math for
        // A/B measurement.
        if self.chr_window_active && !cfg!(feature = "mmc1_chr_dispatch_ab") {
            self.chr_window[(address & 0x1FFF) as usize]
        } else {
            let chr_addr = self.chr_address(address);
            self.cartridge.chr[chr_addr]
        }
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let chr_addr = self.chr_address(address);
        self.cartridge.chr[chr_addr] = value;
        // Keep the flat window coherent with CHR-RAM writes. Update
        // every window offset that currently views the written byte:
        // the slot `address` maps to, plus (in 4 KB mode) the other
        // slot if it aliases the same 4 KB page. The `chr_address`
        // guard makes this exact for both CHR modes and any bank
        // aliasing, at O(1) — two checks — per write.
        if self.chr_window_active {
            let off = (address & 0x0FFF) as usize;
            for slot_base in [0usize, 0x1000usize] {
                let w = slot_base | off;
                if self.chr_address(w as u16) == chr_addr {
                    self.chr_window[w] = value;
                }
            }
        }
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
        self.rebuild_chr_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn chr_static_ptr(&self) -> Option<*const u8> {
        // The PPU caches this at each scanline boundary and reads
        // pattern bytes straight from it, skipping the `chr_read_byte`
        // dispatch + `chr_address` bank math. Safe for MMC1 because it
        // has no CHR-read side effect, and correct across mid-scanline
        // bank switches because `rebuild_chr_window` rewrites this same
        // buffer in place (the pointer never changes). The
        // `mmc1_chr_dispatch_ab` feature forces the dispatch path for
        // A/B measurement.
        #[cfg(not(feature = "mmc1_chr_dispatch_ab"))]
        if self.chr_window_active {
            return Some(self.chr_window.as_ptr());
        }
        None
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
                self.rebuild_chr_window();
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

#[cfg(test)]
mod prg_window_tests {
    use super::*;

    /// Build a synthetic MMC1 cartridge with a `prg_len`-byte PRG ROM
    /// whose bytes encode their containing 16 KB bank so any wrong-bank
    /// read is observable: at a fixed CPU address both the window and
    /// bank-math paths use the identical within-bank offset, so the two
    /// bytes differ iff they land in different banks — and `bank * 37`
    /// (37 coprime to 256) is injective over the bank index.
    fn synth_mapper1(prg_len: usize, chr_len: usize) -> Mapper1 {
        let prg_rom: Vec<u8> = (0..prg_len)
            .map(|i| ((i >> 14) as u8).wrapping_mul(37).wrapping_add((i & 0xFF) as u8))
            .collect();
        let prg_banks = (prg_len / PRG_ROM_BANK_SIZE as usize) as u8;
        let cart = Cartridge {
            mapper: 1,
            sub_mapper: 0,
            mirroring: Mirroring::Horizontal,
            default_mirroring: Mirroring::Horizontal,
            prg_rom_num_banks: prg_banks,
            prg_rom,
            chr_num_banks: (chr_len / 8192) as u8,
            // CHR bytes encode their containing 4 KB page (`page * 37`,
            // 37 coprime to 256) so a rebuild that copies from the wrong
            // page is caught: window and bank-math share the within-page
            // offset at a given address, so only the page can differ.
            chr: (0..chr_len)
                .map(|i| ((i >> 12) as u8).wrapping_mul(37).wrapping_add((i & 0xFF) as u8))
                .collect(),
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        };
        Mapper1::new(cart)
    }

    /// Exhaustively assert the flat-window read (`prg_peek_byte`) is
    /// byte-identical to the exact bank math (`prg_peek_bankmath`)
    /// across every $8000-$FFFF address for every banking state the
    /// MMC1 registers can reach. Covers all three PRG modes (control
    /// bits 2-3), all 16 `prg_bank` values, and the CHR-bank-0 bit 4
    /// that supplies PRG A18 on 512 KB boards.
    fn assert_window_matches(prg_len: usize) {
        let mut m = synth_mapper1(prg_len, 8 * 1024);
        assert!(m.prg_window_ok, "pow2 PRG {} should enable the window", prg_len);
        for control in 0u8..0x20 {
            for prg_bank in 0u8..0x10 {
                for &chr0 in &[0x00u8, 0x10u8] {
                    m.regs.control = control;
                    m.regs.prg_bank = prg_bank;
                    m.regs.chr_bank_0 = chr0;
                    m.rebuild_asm_window();
                    for addr in 0x8000u32..=0xFFFF {
                        let a = addr as u16;
                        let win = m.prg_peek_byte(a);
                        let math = m.prg_peek_bankmath(a);
                        assert_eq!(
                            win, math,
                            "mismatch at ${:04X} (prg_len={} control={:#04X} prg_bank={:#04X} chr0={:#04X})",
                            a, prg_len, control, prg_bank, chr0
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prg_window_matches_bankmath_128k() {
        // Standard board (<= 256 KB): PRG A18 is forced 0.
        assert_window_matches(128 * 1024);
    }

    #[test]
    fn prg_window_matches_bankmath_512k() {
        // SUROM/SXROM: CHR bank 0 bit 4 feeds PRG A18 (>256 KB).
        assert_window_matches(512 * 1024);
    }

    #[test]
    fn prg_window_matches_bankmath_32k() {
        // Smallest pow2 that fills the whole 32 KB window in one shot.
        assert_window_matches(32 * 1024);
    }

    #[test]
    fn non_pow2_prg_falls_back_to_bankmath() {
        // 48 KB is not a power of two; the `% len` window build would
        // diverge from the `& (len-1)` bank math, so the window must be
        // gated off and reads served from the exact bank math.
        let mut m = synth_mapper1(48 * 1024, 8 * 1024);
        assert!(!m.prg_window_ok, "non-pow2 PRG must not use the window");
        for control in 0u8..0x20 {
            for prg_bank in 0u8..0x10 {
                m.regs.control = control;
                m.regs.prg_bank = prg_bank;
                m.rebuild_asm_window();
                for addr in 0x8000u32..=0xFFFF {
                    let a = addr as u16;
                    assert_eq!(
                        m.prg_peek_byte(a),
                        m.prg_peek_bankmath(a),
                        "non-pow2 fallback diverged at ${:04X}",
                        a
                    );
                }
            }
        }
    }

    #[test]
    fn prg_ram_window_unchanged() {
        // $6000-$7FFF PRG-RAM must still route through bank math (the
        // flat window only covers $8000-$FFFF).
        let mut m = synth_mapper1(128 * 1024, 8 * 1024);
        for (i, b) in m.cartridge.prg_ram.iter_mut().enumerate() {
            *b = (i as u8) ^ 0x5A;
        }
        for addr in 0x6000u32..=0x7FFF {
            let a = addr as u16;
            assert_eq!(m.prg_peek_byte(a), m.prg_peek_bankmath(a));
        }
        // Below $6000 reads open bus (0) on both paths.
        assert_eq!(m.prg_peek_byte(0x4020), 0);
        assert_eq!(m.prg_peek_byte(0x0000), 0);
    }

    /// Reference CHR read straight through the bank math, independent of
    /// the window — the oracle the window/static-ptr path must match.
    fn chr_bankmath(m: &Mapper1, address: u16) -> u8 {
        m.cartridge.chr[m.chr_address(address)]
    }

    /// Exhaustively assert the CHR window (`chr_read_byte` +
    /// `chr_static_ptr`) is byte-identical to the per-read bank math
    /// across both CHR modes and every bank pairing. `chr_len` picks the
    /// CHR-ROM size (bank-switching content).
    fn assert_chr_window_matches(chr_len: usize) {
        let mut m = synth_mapper1(32 * 1024, chr_len);
        assert!(m.chr_window_active, "CHR {} should enable the window", chr_len);
        // control bit 4 selects 8 KB (0) vs 4 KB (0x10) CHR mode.
        for &mode_bit in &[0x00u8, 0x10u8] {
            for chr0 in 0u8..0x20 {
                for chr1 in 0u8..0x20 {
                    m.regs.control = mode_bit;
                    m.regs.chr_bank_0 = chr0;
                    m.regs.chr_bank_1 = chr1;
                    m.rebuild_chr_window();
                    // Snapshot the window buffer (always mirrors the bank
                    // math) and the static pointer (gated by the A/B
                    // feature) before the &mut chr_read_byte calls.
                    let win: Vec<u8> = m.chr_window.clone();
                    let static_bytes: Option<Vec<u8>> = m.chr_static_ptr().map(|p| {
                        unsafe { std::slice::from_raw_parts(p, 0x2000) }.to_vec()
                    });
                    for addr in 0x0000u16..0x2000 {
                        let want = chr_bankmath(&m, addr);
                        assert_eq!(
                            m.chr_read_byte(addr),
                            want,
                            "chr_read_byte mismatch ${:04X} mode={:#04X} chr0={:#04X} chr1={:#04X}",
                            addr, mode_bit, chr0, chr1
                        );
                        assert_eq!(
                            win[addr as usize], want,
                            "window mismatch ${:04X} mode={:#04X} chr0={:#04X} chr1={:#04X}",
                            addr, mode_bit, chr0, chr1
                        );
                        if let Some(ref sb) = static_bytes {
                            assert_eq!(
                                sb[addr as usize], want,
                                "static-ptr mismatch ${:04X} mode={:#04X} chr0={:#04X} chr1={:#04X}",
                                addr, mode_bit, chr0, chr1
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chr_window_matches_bankmath_32k() {
        assert_chr_window_matches(32 * 1024);
    }

    #[test]
    fn chr_window_matches_bankmath_128k() {
        assert_chr_window_matches(128 * 1024);
    }

    #[test]
    fn chr_window_matches_bankmath_16k() {
        // Non-power-of-two page count relative to 8 KB (2 banks) — still
        // a whole number of 4 KB pages, so the window is exact.
        assert_chr_window_matches(16 * 1024);
    }

    /// CHR-RAM writes must keep the window coherent — including 4 KB-mode
    /// aliasing where both pattern-table halves map to the same 4 KB RAM
    /// page, so a write through one half must show through the other.
    #[test]
    fn chr_ram_write_keeps_window_coherent() {
        // 8 KB CHR behaves as RAM here (chr_write_byte writes it).
        let mut m = synth_mapper1(32 * 1024, 8 * 1024);
        // (control mode bit, chr_bank_0, chr_bank_1) — includes the
        // aliasing case chr0 == chr1 in 4 KB mode.
        let configs = [
            (0x00u8, 0u8, 0u8),  // 8 KB mode
            (0x00, 2, 0),        // 8 KB mode, bank ignores low bit
            (0x10, 0, 1),        // 4 KB mode, distinct pages
            (0x10, 1, 0),        // 4 KB mode, swapped pages
            (0x10, 1, 1),        // 4 KB mode, ALIASED pages
            (0x10, 3, 3),        // 4 KB mode, aliased (wraps within 8 KB)
        ];
        for (i, &(mode, chr0, chr1)) in configs.iter().enumerate() {
            m.regs.control = mode;
            m.regs.chr_bank_0 = chr0;
            m.regs.chr_bank_1 = chr1;
            m.rebuild_chr_window();
            // Scatter writes across the whole $0000-$1FFF space.
            for addr in (0x0000u16..0x2000).step_by(7) {
                let value = (addr.wrapping_mul(31).wrapping_add(i as u16)) as u8 ^ 0xA5;
                m.chr_write_byte(addr, value);
            }
            // Every window offset must still equal the bank-math read.
            for addr in 0x0000u16..0x2000 {
                assert_eq!(
                    m.chr_window[addr as usize],
                    chr_bankmath(&m, addr),
                    "window incoherent after writes: config {} ${:04X}",
                    i, addr
                );
            }
        }
    }

    #[test]
    #[cfg(not(feature = "mmc1_chr_dispatch_ab"))]
    fn chr_static_ptr_stable_across_bank_switch() {
        // In-place rebuild is what makes mid-scanline bank switches safe:
        // the pointer the PPU cached must not change when banks change.
        let mut m = synth_mapper1(32 * 1024, 32 * 1024);
        let p0 = m.chr_static_ptr().unwrap();
        m.regs.control = 0x10;
        m.regs.chr_bank_0 = 5;
        m.regs.chr_bank_1 = 2;
        m.rebuild_chr_window();
        let p1 = m.chr_static_ptr().unwrap();
        assert_eq!(p0, p1, "CHR window pointer must be stable across rebuilds");
    }
}

#[cfg(test)]
mod restore_consecutive_write_tests {
    use super::*;

    fn synth_cart() -> Cartridge {
        Cartridge {
            mapper: 1,
            sub_mapper: 0,
            mirroring: Mirroring::Horizontal,
            default_mirroring: Mirroring::Horizontal,
            prg_rom_num_banks: 8,
            prg_rom: vec![0u8; 8 * PRG_ROM_BANK_SIZE as usize],
            chr_num_banks: 1,
            chr: vec![0u8; 8 * 1024],
            prg_ram: vec![0u8; 8 * 1024],
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    /// The consecutive-write filter must never eat the FIRST register
    /// write after `apply_state`, on a machine whose CPU-cycle counter
    /// has not moved off zero yet.
    ///
    /// FIXED 2026-08-25 (was an open defect, `#[ignore]`d as a
    /// falsifier — now a permanent regression guard). Root cause:
    /// `apply_state` parks `last_register_write_cycle` at `u64::MAX`
    /// with the stated intent "reset so the next write is
    /// unconditionally honored". The filter predicate was
    /// `cur_cpu_cycle == last_register_write_cycle.wrapping_add(1)`,
    /// and `u64::MAX.wrapping_add(1) == 0`. A `Nes` that has been
    /// restored but never ticked still has `cur_cpu_cycle == 0`
    /// (`Mapper::set_cpu_cycle` is only ever pushed from `Nes::tick`,
    /// and the ASM MMIO callback path does not push it), so the
    /// sentinel that promised "always honored" delivered the exact
    /// opposite: the write was silently dropped. One dropped write
    /// desynchronizes MMC1's 5-write shift sequence, so the eventual
    /// commit lands on the wrong register with the wrong value —
    /// wrong PRG/CHR bank, dead game. Fixed by guarding the predicate
    /// with `self.last_register_write_cycle != u64::MAX &&` at its
    /// call site above. A/B-verified (guard on vs off) across 8 ROMs
    /// including Journey to Silius and Zelda; both diverge without the
    /// guard and agree with the interpreter with it. See
    /// docs/research/ASM_CPU_STATUS_2026-08-25.md.
    #[test]
    fn first_register_write_after_apply_state_is_honored_at_cycle_zero() {
        let saved = Mapper1::new(synth_cart()).get_state();

        let mut restored = Mapper1::new(synth_cart());
        // A freshly-constructed machine that has never ticked: nothing
        // has pushed a CPU cycle into the mapper yet.
        assert_eq!(restored.cur_cpu_cycle, 0);
        restored.apply_state(&saved);

        // One ordinary shift-register write (bit 7 clear, bit 0 set).
        restored.prg_write_byte(0x8000, 0x01);

        assert_eq!(
            restored.shift, 0x18,
            "first MMC1 register write after apply_state was swallowed by \
             the RMW consecutive-write filter (shift still at its default \
             {SHIFT_REGISTER_DEFAULT:#04X})"
        );
    }

    /// The filter itself must still work: two writes on back-to-back
    /// CPU cycles are the RMW dummy+real pair, and the second is
    /// ignored. This is the Bill & Ted's `INC $FFFF` case and it must
    /// survive any fix to the sentinel above.
    #[test]
    fn consecutive_cycle_writes_are_still_filtered() {
        let mut m = Mapper1::new(synth_cart());
        m.set_cpu_cycle(1000);
        m.prg_write_byte(0x8000, 0x01);
        let after_first = m.shift;
        assert_eq!(after_first, 0x18);

        m.set_cpu_cycle(1001);
        m.prg_write_byte(0x8000, 0x01);
        assert_eq!(
            m.shift, after_first,
            "RMW dummy+real pair: the second write must be ignored"
        );
    }
}
