// Konami VRC2 / VRC4 (mappers 21, 22, 23, 25).
//
// All four mapper numbers share the same internal register layout —
// they only differ in which CPU-side address pins route to each
// register's A0/A1 select bits. Upstream games ship register writes
// to the address pattern their specific board expects.
//
// We decode incoming writes through a per-mapper pin-remap, then
// dispatch to the canonical register set. This keeps the register
// logic in one place instead of four copies that drift.
//
// Canonical register map (post-remap):
//   $8000 | n :  PRG bank 0 low 5 bits (bank index for $8000 or $C000 slot depending on swap mode)
//   $9000 | 0 :  Mirroring control (0=V, 1=H, 2=1S-lower, 3=1S-upper)
//   $9000 | 2 :  PRG swap mode  (bit1: 0 = $8000 switchable + $C000 fixed; 1 = swapped)
//   $A000 | n :  PRG bank 1 low 5 bits ($A000 slot, 8 KB)
//   $B000 | 0 :  CHR0 low 4 bits
//   $B000 | 1 :  CHR0 high 5 bits
//   $B000 | 2 :  CHR1 low 4 bits
//   $B000 | 3 :  CHR1 high 5 bits
//   $C000-$E000: CHR2..7 (same low/high pair pattern)
//   $F000 | 0 :  IRQ reload low
//   $F000 | 1 :  IRQ reload high
//   $F000 | 2 :  IRQ control
//   $F000 | 3 :  IRQ acknowledge

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, Mapper};

use serde_derive::{Deserialize, Serialize};

const PRG_BANK: usize = 0x2000; // 8 KB
const CHR_BANK: usize = 0x0400; // 1 KB

#[derive(Clone, Copy)]
pub enum VrcBoard {
    // mapper 21 — VRC4a/c: reg pins = (A1|A6, A2|A7)
    M21,
    // mapper 22 — VRC2a: reg pins swapped = (A1, A0), only A0/A1 used
    M22,
    // mapper 23 — VRC4e/f or VRC2b: direct = (A0, A1) (VRC4f also ORs A2/A4/A6 + A3/A5/A7)
    M23,
    // mapper 25 — VRC4b/d or VRC2c: pins swapped = (A1, A0) (VRC4d also ORs A3/A5/A7 + A2/A4/A6)
    M25,
}

impl VrcBoard {
    #[inline]
    fn reg_bits(self, addr: u16) -> u8 {
        // Canonical (bit1, bit0) extraction per board.
        let a = addr;
        let bit = |shift: u16| ((a >> shift) & 1) as u8;
        match self {
            VrcBoard::M21 => {
                let lo = bit(1) | bit(6);
                let hi = bit(2) | bit(7);
                (hi << 1) | lo
            }
            VrcBoard::M22 => {
                let lo = bit(1);
                let hi = bit(0);
                (hi << 1) | lo
            }
            VrcBoard::M23 => {
                let lo = bit(0) | bit(2) | bit(4) | bit(6);
                let hi = bit(1) | bit(3) | bit(5) | bit(7);
                (hi << 1) | lo
            }
            VrcBoard::M25 => {
                let lo = bit(1) | bit(3) | bit(5) | bit(7);
                let hi = bit(0) | bit(2) | bit(4) | bit(6);
                (hi << 1) | lo
            }
        }
    }
}

pub struct Vrc24 {
    cartridge: Cartridge,
    board: VrcBoard,
    prg_swap_mode: bool,
    prg_bank_0: u8, // 5-bit index to 8 KB PRG bank (variable slot)
    prg_bank_1: u8, // 5-bit index to 8 KB PRG bank ($A000 slot)
    chr_banks: [u16; 8], // 9-bit indices to 1 KB CHR banks
    irq_latch: u8,
    irq_counter: u8,
    irq_mode_cycle: bool, // false = scanline mode, true = cycle mode
    irq_enable_after_ack: bool,
    irq_enabled: bool,
    irq_pending: bool,
    prescaler: i16,
    prg_asm_window: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_swap_mode: bool,
    pub prg_bank_0: u8,
    pub prg_bank_1: u8,
    pub chr_banks: [u16; 8],
    pub irq_latch: u8,
    pub irq_counter: u8,
    pub irq_mode_cycle: bool,
    pub irq_enable_after_ack: bool,
    pub irq_enabled: bool,
    pub irq_pending: bool,
    pub prescaler: i16,
}

impl Vrc24 {
    pub fn new(cartridge: Cartridge, board: VrcBoard) -> Self {
        let mut m = Vrc24 {
            cartridge,
            board,
            prg_swap_mode: false,
            prg_bank_0: 0,
            prg_bank_1: 0,
            chr_banks: [0; 8],
            irq_latch: 0,
            irq_counter: 0,
            irq_mode_cycle: false,
            irq_enable_after_ack: false,
            irq_enabled: false,
            irq_pending: false,
            prescaler: 341,
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.rebuild_asm_window();
        m
    }

    fn prg_bank_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK).max(1)
    }

    fn chr_bank_count(&self) -> usize {
        (self.cartridge.chr.len() / CHR_BANK).max(1)
    }

    // Resolve the effective 1 KB CHR bank for a slot, honoring the
    // VRC2a wiring quirk: on the VRC2a board (mapper 22) the low bit of
    // each CHR select register is NOT connected to the cartridge, so
    // software writes `bank * 2` and the hardware selects `value >> 1`.
    // VRC2b / VRC4 (mappers 21/23/25) wire the low bit normally.
    fn chr_bank_index(&self, slot: usize) -> usize {
        let raw = self.chr_banks[slot] as usize;
        let bank = match self.board {
            VrcBoard::M22 => raw >> 1,
            _ => raw,
        };
        bank % self.chr_bank_count()
    }

    fn prg_offset_for(&self, bank: u8) -> usize {
        (bank as usize % self.prg_bank_count()) * PRG_BANK
    }

    fn rebuild_asm_window(&mut self) {
        let last = self.prg_bank_count() - 1;
        let second_last = self.prg_bank_count().saturating_sub(2);
        // Four 8 KB slots at $8000/$A000/$C000/$E000.
        // $A000 is always prg_bank_1.
        // $E000 is always last bank.
        // $8000 and $C000 swap based on mode:
        //   mode 0: $8000 = prg_bank_0 (switchable), $C000 = second_last (fixed)
        //   mode 1: $8000 = second_last (fixed),     $C000 = prg_bank_0 (switchable)
        let (slot0, slot2) = if self.prg_swap_mode {
            (second_last, self.prg_bank_0 as usize % self.prg_bank_count())
        } else {
            (self.prg_bank_0 as usize % self.prg_bank_count(), second_last)
        };
        let slots = [
            slot0 * PRG_BANK,
            self.prg_offset_for(self.prg_bank_1),
            slot2 * PRG_BANK,
            last * PRG_BANK,
        ];
        let prg = &self.cartridge.prg_rom;
        for (i, off) in slots.iter().enumerate() {
            let dst = i * PRG_BANK;
            if *off + PRG_BANK <= prg.len() {
                self.prg_asm_window[dst..dst + PRG_BANK]
                    .copy_from_slice(&prg[*off..*off + PRG_BANK]);
            }
        }
    }

    fn mirroring_from_code(c: u8) -> Mirroring {
        match c & 0x03 {
            0 => Mirroring::Vertical,
            1 => Mirroring::Horizontal,
            2 => Mirroring::OneScreenLower,
            _ => Mirroring::OneScreenUpper,
        }
    }

    fn handle_irq_write(&mut self, reg: u8, value: u8) {
        match reg {
            0 => {
                self.irq_latch = (self.irq_latch & 0xF0) | (value & 0x0F);
            }
            1 => {
                self.irq_latch = (self.irq_latch & 0x0F) | ((value & 0x0F) << 4);
            }
            2 => {
                self.irq_mode_cycle = value & 0x04 != 0;
                self.irq_enable_after_ack = value & 0x01 != 0;
                self.irq_enabled = value & 0x02 != 0;
                self.irq_pending = false;
                if self.irq_enabled {
                    self.irq_counter = self.irq_latch;
                    self.prescaler = 341;
                }
            }
            3 => {
                self.irq_pending = false;
                self.irq_enabled = self.irq_enable_after_ack;
            }
            _ => {}
        }
    }

    // NOTE: per-CPU-cycle VRC prescaler tick lives in `on_scanline_tick`
    // (scanline-granularity approximation). A future `on_cpu_cycle` hook
    // would replace it with the canonical 3-PPU-cycles-per-CPU-cycle
    // decrement. Keeping this out of the hot mapper trait for now — the
    // scanline approximation is accurate enough for every VRC4 game
    // tested so far.

    fn tick_counter(&mut self) {
        if self.irq_counter == 0xFF {
            self.irq_counter = self.irq_latch;
            self.irq_pending = true;
        } else {
            self.irq_counter = self.irq_counter.wrapping_add(1);
        }
    }

    pub fn write_register(&mut self, address: u16, value: u8) {
        if address < 0x8000 {
            return;
        }
        let reg = self.board.reg_bits(address);
        match address & 0xF000 {
            0x8000 => {
                self.prg_bank_0 = value & 0x1F;
                self.rebuild_asm_window();
            }
            0x9000 => match reg {
                0 | 1 => {
                    self.cartridge.mirroring = Self::mirroring_from_code(value);
                }
                2 | 3 => {
                    let new_mode = value & 0x02 != 0;
                    if new_mode != self.prg_swap_mode {
                        self.prg_swap_mode = new_mode;
                        self.rebuild_asm_window();
                    }
                }
                _ => {}
            },
            0xA000 => {
                self.prg_bank_1 = value & 0x1F;
                self.rebuild_asm_window();
            }
            0xB000 | 0xC000 | 0xD000 | 0xE000 => {
                // Each $x000 region holds 2 CHR bank registers (low
                // nibble at reg=0/2, high nibble at reg=1/3).
                let group = ((address - 0xB000) >> 12) as usize * 2; // B→0, C→2, D→4, E→6
                let chr_idx = group + ((reg >> 1) as usize); // +0 or +1
                if chr_idx < 8 {
                    if reg & 1 == 0 {
                        self.chr_banks[chr_idx] =
                            (self.chr_banks[chr_idx] & 0x1F0) | (value & 0x0F) as u16;
                    } else {
                        self.chr_banks[chr_idx] =
                            (self.chr_banks[chr_idx] & 0x00F) | (((value & 0x1F) as u16) << 4);
                    }
                }
            }
            0xF000 => {
                self.handle_irq_write(reg, value);
            }
            _ => {}
        }
    }
}

impl Mapper for Vrc24 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xFFFF => {
                let slot = ((address - 0x8000) / 0x2000) as usize;
                let off_in_bank = address as usize & 0x1FFF;
                let bank = match slot {
                    0 => {
                        if self.prg_swap_mode {
                            self.prg_bank_count() - 2
                        } else {
                            self.prg_bank_0 as usize % self.prg_bank_count()
                        }
                    }
                    1 => self.prg_bank_1 as usize % self.prg_bank_count(),
                    2 => {
                        if self.prg_swap_mode {
                            self.prg_bank_0 as usize % self.prg_bank_count()
                        } else {
                            self.prg_bank_count() - 2
                        }
                    }
                    _ => self.prg_bank_count() - 1,
                };
                self.cartridge.prg_rom[bank * PRG_BANK | off_in_bank]
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if (0x6000..0x8000).contains(&address) {
            let i = (address - 0x6000) as usize;
            if i < self.cartridge.prg_ram.len() {
                self.cartridge.prg_ram[i] = value;
            }
            return;
        }
        self.write_register(address, value);
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_bank_index(slot);
        self.cartridge.chr[bank * CHR_BANK | (address as usize & 0x3FF)]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_bank_index(slot);
        let off = bank * CHR_BANK | (address as usize & 0x3FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_swap_mode = false;
        self.prg_bank_0 = 0;
        self.prg_bank_1 = 0;
        self.chr_banks = [0; 8];
        self.irq_latch = 0;
        self.irq_counter = 0;
        self.irq_mode_cycle = false;
        self.irq_enable_after_ack = false;
        self.irq_enabled = false;
        self.irq_pending = false;
        self.prescaler = 341;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn on_scanline_tick(&mut self) {
        // Scanline-approx IRQ: decrement prescaler by 113 (CPU cycles
        // per scanline) — good enough for the majority of VRC4 games.
        if !self.irq_enabled {
            return;
        }
        if self.irq_mode_cycle {
            for _ in 0..113 {
                self.tick_counter();
                if self.irq_pending {
                    break;
                }
            }
        } else {
            self.prescaler -= 113 * 3;
            while self.prescaler <= 0 {
                self.prescaler += 341;
                self.tick_counter();
            }
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_pending
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        2
    }

    fn get_state(&self) -> mapper::State {
        let s = State {
            cartridge: self.cartridge.get_state(),
            prg_swap_mode: self.prg_swap_mode,
            prg_bank_0: self.prg_bank_0,
            prg_bank_1: self.prg_bank_1,
            chr_banks: self.chr_banks,
            irq_latch: self.irq_latch,
            irq_counter: self.irq_counter,
            irq_mode_cycle: self.irq_mode_cycle,
            irq_enable_after_ack: self.irq_enable_after_ack,
            irq_enabled: self.irq_enabled,
            irq_pending: self.irq_pending,
            prescaler: self.prescaler,
        };
        match self.board {
            VrcBoard::M21 => mapper::State::State21(s),
            VrcBoard::M22 => mapper::State::State22(s),
            VrcBoard::M23 => mapper::State::State23(s),
            VrcBoard::M25 => mapper::State::State25(s),
        }
    }

    fn apply_state(&mut self, state: &mapper::State) {
        let inner = match state {
            mapper::State::State21(s)
            | mapper::State::State22(s)
            | mapper::State::State23(s)
            | mapper::State::State25(s) => s,
            _ => panic!("Invalid mapper state variant for VRC2/4"),
        };
        self.cartridge.apply_state(&inner.cartridge);
        self.prg_swap_mode = inner.prg_swap_mode;
        self.prg_bank_0 = inner.prg_bank_0;
        self.prg_bank_1 = inner.prg_bank_1;
        self.chr_banks = inner.chr_banks;
        self.irq_latch = inner.irq_latch;
        self.irq_counter = inner.irq_counter;
        self.irq_mode_cycle = inner.irq_mode_cycle;
        self.irq_enable_after_ack = inner.irq_enable_after_ack;
        self.irq_enabled = inner.irq_enabled;
        self.irq_pending = inner.irq_pending;
        self.prescaler = inner.prescaler;
        self.rebuild_asm_window();
    }
}

// Thin wrappers so enum_dispatch has distinct variants per mapper.
pub struct Mapper21(pub Vrc24);
pub struct Mapper22(pub Vrc24);
pub struct Mapper23(pub Vrc24);
pub struct Mapper25(pub Vrc24);

macro_rules! forward_vrc {
    ($name:ident, $board:expr) => {
        impl $name {
            pub fn new(cartridge: Cartridge) -> Self {
                Self(Vrc24::new(cartridge, $board))
            }
        }
        impl Mapper for $name {
            fn prg_peek_byte(&self, a: u16) -> u8 {
                self.0.prg_peek_byte(a)
            }
            fn prg_write_byte(&mut self, a: u16, v: u8) {
                self.0.prg_write_byte(a, v)
            }
            fn chr_read_byte(&mut self, a: u16) -> u8 {
                self.0.chr_read_byte(a)
            }
            fn chr_write_byte(&mut self, a: u16, v: u8) {
                self.0.chr_write_byte(a, v)
            }
            fn mirroring(&self) -> Mirroring {
                self.0.mirroring()
            }
            fn reset(&mut self) {
                self.0.reset()
            }
            fn on_scanline_tick(&mut self) {
                self.0.on_scanline_tick()
            }
            fn irq_pending(&self) -> bool {
                self.0.irq_pending()
            }
            fn prg_asm_ptr(&self) -> Option<*const u8> {
                self.0.prg_asm_ptr()
            }
            fn asm_bulk_cycles(&self) -> i64 {
                self.0.asm_bulk_cycles()
            }
            fn get_state(&self) -> mapper::State {
                self.0.get_state()
            }
            fn apply_state(&mut self, s: &mapper::State) {
                self.0.apply_state(s)
            }
        }
    };
}

forward_vrc!(Mapper21, VrcBoard::M21);
forward_vrc!(Mapper22, VrcBoard::M22);
forward_vrc!(Mapper23, VrcBoard::M23);
forward_vrc!(Mapper25, VrcBoard::M25);

#[cfg(test)]
mod tests {
    use super::*;

    // CHR filled so that byte 0 of 1 KB bank `b` equals `b`, letting a
    // read prove which physical bank a slot resolved to.
    fn test_cartridge() -> Cartridge {
        let mut chr = vec![0u8; 32 * CHR_BANK]; // 32 KB = 32 1 KB banks
        for b in 0..32usize {
            chr[b * CHR_BANK] = b as u8;
        }
        Cartridge {
            mapper: 22,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: 2,
            prg_rom: vec![0u8; 32 * 1024],
            chr_num_banks: 4,
            chr,
            prg_ram: Vec::new(),
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    #[test]
    fn vrc2a_chr_bank_low_bit_ignored() {
        // VRC2a (mapper 22): register holds bank*2, effective bank = value>>1.
        let mut m = Vrc24::new(test_cartridge(), VrcBoard::M22);
        m.chr_banks[0] = 6;
        // Reading slot 0 ($0000-$03FF) must resolve to physical bank 3.
        assert_eq!(m.chr_read_byte(0x0000), 3, "VRC2a must select value>>1");
    }

    #[test]
    fn vrc2b_vrc4_chr_bank_unshifted() {
        // Other boards wire the low bit normally: bank == raw register.
        for board in [VrcBoard::M21, VrcBoard::M23, VrcBoard::M25] {
            let mut m = Vrc24::new(test_cartridge(), board);
            m.chr_banks[0] = 6;
            assert_eq!(
                m.chr_read_byte(0x0000),
                6,
                "non-VRC2a boards must not shift the CHR bank"
            );
        }
    }
}
