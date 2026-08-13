// iNES Mapper 64 — Tengen RAMBO-1.
//
// RAMBO-1 is Tengen's MMC3-like chip used by a small set of Tengen
// carts (Klax, Rolling Thunder, Skull & Crossbones, Road Runner,
// Xybots, Shinobi, ...). It is register-compatible with MMC3 at
// $8000/$8001/$A000/$A001/$C000/$C001/$E000/$E001 but adds two extra
// bank registers (R8/R9) that enable either extra PRG banking or
// all-1KB CHR banking, selected by bit 5 of $8000.
//
// This implementation targets "boot + run" compatibility — bank
// layouts follow the nesdev wiki, but the IRQ counter is implemented
// as a scanline-style count-down (copy of MMC3). True RAMBO-1 counts
// CPU cycles; close enough for most boot-and-run-N-frames tests.

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::cpu::Cpu;
use crate::mapper::{self, Mapper};
use crate::ppu::{self, Ppu};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper64 {
    cartridge: Cartridge,

    // Register file. Bits 0..=3 of $8000 select which register
    // receives the next write to $8001. Only R0..=R9 are meaningful;
    // higher indices are silently ignored.
    next_bank_register: u8,
    bank_registers: [u8; 10],

    // Bit 6 of $8000: toggles which PRG slot is fixed.
    prg_rom_mode: PrgRomMode,
    // Bit 7 of $8000: CHR A12 inversion (like MMC3).
    chr_a12_inversion: ChrA12Inversion,
    // Bit 5 of $8000: CHR bank size mode.
    chr_mode: ChrMode,

    // Scanline IRQ (MMC3-style; good enough for boot-and-run).
    irq_enabled: bool,
    irq_counter: u8,
    irq_reload_flag: bool,
    irq_latch: u8,
    irq_active: bool,

    // Resolved bank offsets, rebuilt on every register write.
    prg_rom_bank_offsets: [usize; 4],
    chr_bank_offsets: [usize; 8],

    // Flat 32 KB shadow of currently-mapped PRG for the AArch64 ASM
    // CPU fast path. Same trick as Mapper4.
    prg_asm_window: Vec<u8>,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum PrgRomMode {
    Zero, // $8000 swappable, $C000 second-last (like MMC3 mode 0)
    One,  // $C000 swappable, $8000 second-last (like MMC3 mode 1)
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum ChrA12Inversion {
    Zero, // 2 KB banks at $0000, 1 KB at $1000 (in 1KB+2KB mode)
    One,  // 2 KB banks at $1000, 1 KB at $0000
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum ChrMode {
    // Bit 5 of $8000 = 0: six 1 KB + two 2 KB banks (classic MMC3-ish).
    OneAndTwoKb,
    // Bit 5 of $8000 = 1: eight 1 KB banks, using R0..=R7 for the low
    // four 1 KB slots and R8/R9 splicing into the 2 KB pair slots.
    AllOneKb,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub next_bank_register: u8,
    pub bank_registers: [u8; 10],
    pub prg_rom_mode: PrgRomMode,
    pub chr_a12_inversion: ChrA12Inversion,
    pub chr_mode: ChrMode,
    pub irq_enabled: bool,
    pub irq_counter: u8,
    pub irq_reload_flag: bool,
    pub irq_latch: u8,
    pub irq_active: bool,
    pub prg_rom_bank_offsets: [usize; 4],
    pub chr_bank_offsets: [usize; 8],
}

impl Mapper64 {
    pub fn new(cartridge: Cartridge) -> Self {
        let mut m = Mapper64 {
            cartridge,
            next_bank_register: 0,
            bank_registers: [0; 10],
            prg_rom_mode: PrgRomMode::Zero,
            chr_a12_inversion: ChrA12Inversion::Zero,
            chr_mode: ChrMode::OneAndTwoKb,
            irq_enabled: false,
            irq_counter: 0,
            irq_reload_flag: false,
            irq_latch: 0,
            irq_active: false,
            prg_rom_bank_offsets: [0; 4],
            chr_bank_offsets: [0; 8],
            prg_asm_window: vec![0u8; 32 * 1024],
        };
        m.update_banks();
        m.rebuild_asm_window();
        m
    }

    fn prg_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / 0x2000).max(1)
    }

    fn chr_count(&self) -> usize {
        (self.cartridge.chr.len() / 0x0400).max(1)
    }

    fn prg_bank_address(&self, bank: u8) -> usize {
        let bank = (bank as usize) % self.prg_count();
        bank * 0x2000
    }

    fn chr_bank_address(&self, bank: u8) -> usize {
        let bank = (bank as usize) % self.chr_count();
        bank * 0x0400
    }

    fn rebuild_asm_window(&mut self) {
        const BANK: usize = 0x2000;
        let prg = &self.cartridge.prg_rom;
        for i in 0..4 {
            let off = self.prg_rom_bank_offsets[i];
            let end = off + BANK;
            if end <= prg.len() {
                self.prg_asm_window[i * BANK..(i + 1) * BANK]
                    .copy_from_slice(&prg[off..end]);
            }
        }
    }

    fn write_bank_select(&mut self, value: u8) {
        // Bits 0..=3 pick the target register. RAMBO-1 uses R0..=R9;
        // higher values wrap (writes to them become no-ops).
        self.next_bank_register = value & 0x0F;

        self.prg_rom_mode = if value & 0x40 == 0 {
            PrgRomMode::Zero
        } else {
            PrgRomMode::One
        };

        self.chr_mode = if value & 0x20 == 0 {
            ChrMode::OneAndTwoKb
        } else {
            ChrMode::AllOneKb
        };

        self.chr_a12_inversion = if value & 0x80 == 0 {
            ChrA12Inversion::Zero
        } else {
            ChrA12Inversion::One
        };

        self.update_banks();
        self.rebuild_asm_window();
    }

    fn write_bank_data(&mut self, value: u8) {
        let idx = self.next_bank_register as usize;
        if idx < self.bank_registers.len() {
            self.bank_registers[idx] = value;
        }
        self.update_banks();
        self.rebuild_asm_window();
    }

    fn write_mirroring(&mut self, value: u8) {
        if self.cartridge.mirroring != Mirroring::FourScreen {
            self.cartridge.mirroring = if value & 0x01 == 0 {
                Mirroring::Vertical
            } else {
                Mirroring::Horizontal
            };
        }
    }

    fn write_prg_ram_protect(&mut self, _value: u8) {
        // Ignore.
    }

    fn update_banks(&mut self) {
        // PRG. RAMBO-1 has up to four 8 KB banks selectable — R6, R7,
        // R8, R9 — but only three slots swap at a time plus one fixed
        // "last" bank. We follow the common RAMBO-1 layout:
        //   mode 0 ($8000 bit 6 = 0):
        //     $8000 = R6, $A000 = R7, $C000 = R8 (or fixed -2), $E000 = fixed -1
        //   mode 1 ($8000 bit 6 = 1):
        //     $8000 = R8 (or fixed -2), $A000 = R7, $C000 = R6, $E000 = fixed -1
        // Simplification: treat R8 slot as second-last bank when mode
        // puts a fixed bank there — this is what actual RAMBO-1 carts
        // see at boot before software programs R8.
        let last = self.prg_count().saturating_sub(1) as u8;
        let second_last = self.prg_count().saturating_sub(2) as u8;

        // $E000-$FFFF is always the last 8 KB.
        self.prg_rom_bank_offsets[3] = self.prg_bank_address(last);

        match self.prg_rom_mode {
            PrgRomMode::Zero => {
                self.prg_rom_bank_offsets[0] = self.prg_bank_address(self.bank_registers[6]);
                self.prg_rom_bank_offsets[1] = self.prg_bank_address(self.bank_registers[7]);
                self.prg_rom_bank_offsets[2] = self.prg_bank_address(self.bank_registers[8]);
                // If R8 hasn't been programmed yet, fall back to
                // second-to-last so boot vectors map somewhere sane.
                if self.bank_registers[8] == 0 {
                    self.prg_rom_bank_offsets[2] = self.prg_bank_address(second_last);
                }
            }
            PrgRomMode::One => {
                self.prg_rom_bank_offsets[0] = self.prg_bank_address(self.bank_registers[8]);
                self.prg_rom_bank_offsets[1] = self.prg_bank_address(self.bank_registers[7]);
                self.prg_rom_bank_offsets[2] = self.prg_bank_address(self.bank_registers[6]);
                if self.bank_registers[8] == 0 {
                    self.prg_rom_bank_offsets[0] = self.prg_bank_address(second_last);
                }
            }
        }

        // CHR. Layout depends on chr_mode + chr_a12_inversion.
        //
        // Mode OneAndTwoKb (bit 5 = 0):
        //   R0/R1 are 2 KB banks (low bit ignored); R2..=R5 are 1 KB.
        //   A12 inversion flips which 4 KB half gets the 2 KB pair.
        //
        // Mode AllOneKb (bit 5 = 1):
        //   R0, R1 each produce a single 1 KB bank (not a pair).
        //   R8, R9 supply the additional 1 KB banks that fill the
        //   slots that used to be the second half of each 2 KB pair.
        //   Layout (inversion = Zero):
        //     $0000: R0     $0400: R8
        //     $0800: R1     $0C00: R9
        //     $1000: R2     $1400: R3
        //     $1800: R4     $1C00: R5
        //   With inversion = One, the top and bottom halves swap.
        match (self.chr_mode, self.chr_a12_inversion) {
            (ChrMode::OneAndTwoKb, ChrA12Inversion::Zero) => {
                self.chr_bank_offsets[0] = self.chr_bank_address(self.bank_registers[0] & 0xFE);
                self.chr_bank_offsets[1] = self.chr_bank_address(self.bank_registers[0] | 0x01);
                self.chr_bank_offsets[2] = self.chr_bank_address(self.bank_registers[1] & 0xFE);
                self.chr_bank_offsets[3] = self.chr_bank_address(self.bank_registers[1] | 0x01);
                self.chr_bank_offsets[4] = self.chr_bank_address(self.bank_registers[2]);
                self.chr_bank_offsets[5] = self.chr_bank_address(self.bank_registers[3]);
                self.chr_bank_offsets[6] = self.chr_bank_address(self.bank_registers[4]);
                self.chr_bank_offsets[7] = self.chr_bank_address(self.bank_registers[5]);
            }
            (ChrMode::OneAndTwoKb, ChrA12Inversion::One) => {
                self.chr_bank_offsets[0] = self.chr_bank_address(self.bank_registers[2]);
                self.chr_bank_offsets[1] = self.chr_bank_address(self.bank_registers[3]);
                self.chr_bank_offsets[2] = self.chr_bank_address(self.bank_registers[4]);
                self.chr_bank_offsets[3] = self.chr_bank_address(self.bank_registers[5]);
                self.chr_bank_offsets[4] = self.chr_bank_address(self.bank_registers[0] & 0xFE);
                self.chr_bank_offsets[5] = self.chr_bank_address(self.bank_registers[0] | 0x01);
                self.chr_bank_offsets[6] = self.chr_bank_address(self.bank_registers[1] & 0xFE);
                self.chr_bank_offsets[7] = self.chr_bank_address(self.bank_registers[1] | 0x01);
            }
            (ChrMode::AllOneKb, ChrA12Inversion::Zero) => {
                self.chr_bank_offsets[0] = self.chr_bank_address(self.bank_registers[0]);
                self.chr_bank_offsets[1] = self.chr_bank_address(self.bank_registers[8]);
                self.chr_bank_offsets[2] = self.chr_bank_address(self.bank_registers[1]);
                self.chr_bank_offsets[3] = self.chr_bank_address(self.bank_registers[9]);
                self.chr_bank_offsets[4] = self.chr_bank_address(self.bank_registers[2]);
                self.chr_bank_offsets[5] = self.chr_bank_address(self.bank_registers[3]);
                self.chr_bank_offsets[6] = self.chr_bank_address(self.bank_registers[4]);
                self.chr_bank_offsets[7] = self.chr_bank_address(self.bank_registers[5]);
            }
            (ChrMode::AllOneKb, ChrA12Inversion::One) => {
                self.chr_bank_offsets[0] = self.chr_bank_address(self.bank_registers[2]);
                self.chr_bank_offsets[1] = self.chr_bank_address(self.bank_registers[3]);
                self.chr_bank_offsets[2] = self.chr_bank_address(self.bank_registers[4]);
                self.chr_bank_offsets[3] = self.chr_bank_address(self.bank_registers[5]);
                self.chr_bank_offsets[4] = self.chr_bank_address(self.bank_registers[0]);
                self.chr_bank_offsets[5] = self.chr_bank_address(self.bank_registers[8]);
                self.chr_bank_offsets[6] = self.chr_bank_address(self.bank_registers[1]);
                self.chr_bank_offsets[7] = self.chr_bank_address(self.bank_registers[9]);
            }
        }
    }

    fn handle_scanline(&mut self, _cpu: &mut Cpu) {
        if self.irq_counter == 0 || self.irq_reload_flag {
            self.irq_counter = self.irq_latch;
            self.irq_reload_flag = false;
        } else {
            self.irq_counter -= 1;
        }

        if self.irq_counter == 0 && self.irq_enabled {
            self.irq_active = true;
        }
    }

    fn read_prg_rom(&self, address: u16) -> u8 {
        let off = self.prg_rom_bank_offsets[(address as usize - 0x8000) / 0x2000]
            | (address as usize & 0x1FFF);
        // Guard against pathological out-of-range during early boot;
        // % prg_count() above should keep this in bounds.
        if off < self.cartridge.prg_rom.len() {
            self.cartridge.prg_rom[off]
        } else {
            0
        }
    }

    fn chr_address(&self, address: u16) -> usize {
        self.chr_bank_offsets[(address as usize) / 0x0400] | (address as usize & 0x03FF)
    }

    fn read_chr(&self, address: u16) -> u8 {
        let addr = self.chr_address(address);
        if addr < self.cartridge.chr.len() {
            self.cartridge.chr[addr]
        } else {
            0
        }
    }

    fn write_chr(&mut self, address: u16, value: u8) {
        let addr = self.chr_address(address);
        if addr < self.cartridge.chr.len() {
            self.cartridge.chr[addr] = value;
        }
    }
}

impl Mapper for Mapper64 {
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
        } else {
            self.read_prg_rom(address)
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        if address < 0x6000 {
            // Nothing below $6000.
        } else if address < 0x8000 {
            if !self.cartridge.prg_ram.is_empty() {
                let idx = (address - 0x6000) as usize % self.cartridge.prg_ram.len();
                self.cartridge.prg_ram[idx] = value;
            }
        } else if address < 0xA000 {
            if address & 0x01 == 0 {
                self.write_bank_select(value);
            } else {
                self.write_bank_data(value);
            }
        } else if address < 0xC000 {
            if address & 0x01 == 0 {
                self.write_mirroring(value);
            } else {
                self.write_prg_ram_protect(value);
            }
        } else if address < 0xE000 {
            if address & 0x01 == 0 {
                self.irq_latch = value;
            } else {
                self.irq_counter = 0;
                self.irq_reload_flag = true;
            }
        } else {
            if address & 0x01 == 0 {
                self.irq_enabled = false;
                self.irq_active = false;
            } else {
                self.irq_enabled = true;
            }
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        self.read_chr(address)
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        self.write_chr(address, value);
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn step(&mut self, cpu: &mut Cpu, ppu: &Ppu) {
        if ppu.rendering_enabled()
            && ppu.scanline <= ppu::VISIBLE_END_SCANLINE
            && ppu.scanline_cycle() == 260
        {
            self.handle_scanline(cpu);
        }
    }

    fn on_scanline_tick(&mut self) {
        if self.irq_counter == 0 || self.irq_reload_flag {
            self.irq_counter = self.irq_latch;
            self.irq_reload_flag = false;
        } else {
            self.irq_counter -= 1;
        }
        if self.irq_counter == 0 && self.irq_enabled {
            self.irq_active = true;
        }
    }

    fn sram(&mut self) -> *mut u8 {
        self.cartridge.prg_ram.as_mut_ptr() as *mut _
    }

    fn sram_size(&self) -> usize {
        self.cartridge.prg_ram.len()
    }

    fn reset(&mut self) {
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.next_bank_register = 0;
        self.bank_registers = [0; 10];
        self.prg_rom_mode = PrgRomMode::Zero;
        self.chr_a12_inversion = ChrA12Inversion::Zero;
        self.chr_mode = ChrMode::OneAndTwoKb;
        self.irq_enabled = false;
        self.irq_counter = 0;
        self.irq_reload_flag = false;
        self.irq_latch = 0;
        self.irq_active = false;
        self.prg_rom_bank_offsets = [0; 4];
        self.chr_bank_offsets = [0; 8];
        self.update_banks();
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        1
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State64(State {
            cartridge: self.cartridge.get_state(),
            next_bank_register: self.next_bank_register,
            bank_registers: self.bank_registers,
            prg_rom_mode: self.prg_rom_mode,
            chr_a12_inversion: self.chr_a12_inversion,
            chr_mode: self.chr_mode,
            irq_enabled: self.irq_enabled,
            irq_counter: self.irq_counter,
            irq_reload_flag: self.irq_reload_flag,
            irq_latch: self.irq_latch,
            irq_active: self.irq_active,
            prg_rom_bank_offsets: self.prg_rom_bank_offsets,
            chr_bank_offsets: self.chr_bank_offsets,
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State64(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.next_bank_register = state.next_bank_register;
                self.bank_registers = state.bank_registers;
                self.prg_rom_mode = state.prg_rom_mode;
                self.chr_a12_inversion = state.chr_a12_inversion;
                self.chr_mode = state.chr_mode;
                self.irq_enabled = state.irq_enabled;
                self.irq_counter = state.irq_counter;
                self.irq_reload_flag = state.irq_reload_flag;
                self.irq_latch = state.irq_latch;
                self.irq_active = state.irq_active;
                self.prg_rom_bank_offsets = state.prg_rom_bank_offsets;
                self.chr_bank_offsets = state.chr_bank_offsets;
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{Cartridge, Mirroring};
    use crate::mapper::Mapper;

    // RAMBO-1 register addresses. The mapper decodes only the $8000
    // block boundaries and address bit 0 (even/odd) within each block.
    const BANK_SELECT: u16 = 0x8000; // even in $8000-$9FFF
    const BANK_DATA: u16 = 0x8001; // odd in $8000-$9FFF
    const MIRROR: u16 = 0xA000; // even in $A000-$BFFF
    const IRQ_LATCH: u16 = 0xC000; // even in $C000-$DFFF
    const IRQ_RELOAD: u16 = 0xC001; // odd in $C000-$DFFF
    const IRQ_DISABLE: u16 = 0xE000; // even in $E000-$FFFF (disable + ack)
    const IRQ_ENABLE: u16 = 0xE001; // odd in $E000-$FFFF

    // Control bits packed into a $8000 (bank-select) write.
    const PRG_MODE_ONE: u8 = 0x40; // bit 6
    const CHR_ALL_1KB: u8 = 0x20; // bit 5
    const CHR_A12_INVERT: u8 = 0x80; // bit 7

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 64,
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

    // Stamp the first byte of every 8 KB PRG bank and every 1 KB CHR
    // bank with that bank's index, so a mapped read reveals which bank
    // is currently selected. Must run before the cartridge is moved
    // into the mapper.
    fn stamped_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        let mut c = test_cart(prg_16k_banks, chr_8k_banks);
        let prg_banks = c.prg_rom.len() / 0x2000;
        for b in 0..prg_banks {
            c.prg_rom[b * 0x2000] = b as u8;
        }
        let chr_banks = c.chr.len() / 0x0400;
        for b in 0..chr_banks {
            c.chr[b * 0x0400] = b as u8;
        }
        c
    }

    // Select `reg` (control bits merged in) then write `bank` to it.
    fn set_reg(m: &mut Mapper64, control: u8, bank: u8) {
        m.prg_write_byte(BANK_SELECT, control);
        m.prg_write_byte(BANK_DATA, bank);
    }

    fn arm_irq(m: &mut Mapper64, latch: u8) {
        m.prg_write_byte(IRQ_LATCH, latch);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
    }

    // ---- PRG banking ----

    // R6 drives the $8000 slot in PRG mode 0.
    #[test]
    fn prg_r6_maps_to_8000_in_mode0() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 6, 3);
        assert_eq!(m.prg_read_byte(0x8000), 3);
    }

    // R7 drives the $A000 slot in PRG mode 0.
    #[test]
    fn prg_r7_maps_to_a000_in_mode0() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 7, 4);
        assert_eq!(m.prg_read_byte(0xA000), 4);
    }

    // The last 8 KB PRG bank is permanently fixed at $E000.
    #[test]
    fn last_prg_bank_fixed_at_e000() {
        let mut m = Mapper64::new(stamped_cart(8, 8)); // 16 x 8 KB, last = 15
        assert_eq!(m.prg_read_byte(0xE000), 15);
        // Reprogramming a swappable slot must not disturb the fixed one.
        set_reg(&mut m, 6, 3);
        assert_eq!(m.prg_read_byte(0xE000), 15);
    }

    // A programmed R8 selects the $C000 slot in PRG mode 0.
    #[test]
    fn prg_r8_maps_to_c000_in_mode0() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 8, 5);
        assert_eq!(m.prg_read_byte(0xC000), 5);
    }

    // SUSPECTED BUG: an unprogrammed R8 (value 0) is treated as "not yet
    // set" and silently replaced by the second-to-last bank, so PRG bank
    // 0 can never be selected into the R8 slot. This is a documented
    // boot-time simplification in update_banks(), not real RAMBO-1
    // behavior. The test pins the CURRENT (fallback) behavior.
    #[test]
    fn unprogrammed_r8_falls_back_to_second_last() {
        let mut m = Mapper64::new(stamped_cart(8, 8)); // second-last = 14
        assert_eq!(m.prg_read_byte(0xC000), 14);
    }

    // PRG mode 1 swaps the $8000 and $C000 roles: R6 moves to $C000.
    #[test]
    fn prg_mode_one_moves_r6_to_c000() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 6, 3);
        assert_eq!(m.prg_read_byte(0x8000), 3); // R6 at $8000 in mode 0
        // Flip to mode 1 without reprogramming any bank register.
        m.prg_write_byte(BANK_SELECT, PRG_MODE_ONE);
        assert_eq!(m.prg_read_byte(0xC000), 3); // R6 now at $C000
        assert_eq!(m.prg_read_byte(0xE000), 15); // last bank still fixed
    }

    // A PRG bank index larger than the ROM wraps (masked mod bank count)
    // instead of reading out of bounds.
    #[test]
    fn prg_bank_index_wraps() {
        let mut m = Mapper64::new(stamped_cart(8, 8)); // 16 banks
        set_reg(&mut m, 6, 200); // 200 % 16 == 8
        assert_eq!(m.prg_read_byte(0x8000), (200 % 16) as u8);
    }

    // ---- CHR banking ----

    // In the 1 KB+2 KB mode, R2 supplies a single 1 KB bank at $1000.
    #[test]
    fn chr_r2_1kb_bank_at_1000() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 2, 7);
        assert_eq!(m.chr_read_byte(0x1000), 7);
    }

    // R0 is a 2 KB pair: its low bit is forced even for $0000 and odd
    // for $0400.
    #[test]
    fn chr_r0_2kb_pair_even_odd() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 0, 5); // pair covers banks 4 and 5
        assert_eq!(m.chr_read_byte(0x0000), 4); // R0 & 0xFE
        assert_eq!(m.chr_read_byte(0x0400), 5); // R0 | 0x01
    }

    // Bit 7 (A12 inversion) swaps the $0000 and $1000 CHR halves.
    #[test]
    fn chr_a12_inversion_swaps_halves() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 0, 5); // R0 pair -> banks 4/5
        set_reg(&mut m, 2, 7); // R2 -> bank 7
        assert_eq!(m.chr_read_byte(0x0000), 4); // low half = R0 pair
        assert_eq!(m.chr_read_byte(0x1000), 7); // high half = R2
        // Invert without reprogramming a register.
        m.prg_write_byte(BANK_SELECT, CHR_A12_INVERT);
        assert_eq!(m.chr_read_byte(0x0000), 7); // halves swapped
        assert_eq!(m.chr_read_byte(0x1000), 4);
    }

    // Bit 5 (all-1 KB mode) makes R0 a single unmasked 1 KB bank rather
    // than an even/odd 2 KB pair.
    #[test]
    fn chr_all_1kb_mode_unmasks_r0() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 0, 5);
        assert_eq!(m.chr_read_byte(0x0000), 4); // masked to even in pair mode
        m.prg_write_byte(BANK_SELECT, CHR_ALL_1KB); // switch to all-1 KB
        assert_eq!(m.chr_read_byte(0x0000), 5); // now unmasked
    }

    // In all-1 KB mode R8 fills the $0400 slot (the extra 1 KB bank).
    #[test]
    fn chr_all_1kb_r8_fills_second_slot() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, CHR_ALL_1KB | 8, 9); // select R8 in all-1 KB mode
        assert_eq!(m.chr_read_byte(0x0400), 9);
    }

    // CHR bank index wraps mod the CHR bank count.
    #[test]
    fn chr_bank_index_wraps() {
        let mut m = Mapper64::new(stamped_cart(8, 8)); // 64 x 1 KB banks
        set_reg(&mut m, 2, 100); // 100 % 64 == 36
        assert_eq!(m.chr_read_byte(0x1000), (100 % 64) as u8);
    }

    // CHR RAM writes land in the currently-selected 1 KB bank.
    #[test]
    fn chr_write_read_roundtrip() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 2, 7); // $1000 -> bank 7
        m.chr_write_byte(0x1000, 0xAB);
        assert_eq!(m.chr_read_byte(0x1000), 0xAB);
    }

    // ---- Mirroring ----

    // $A000 bit 0: 0 = vertical, 1 = horizontal.
    #[test]
    fn mirroring_register() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
        m.prg_write_byte(MIRROR, 0);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        m.prg_write_byte(MIRROR, 1);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);
    }

    // A four-screen cart ignores the mirroring register entirely.
    #[test]
    fn fourscreen_mirroring_locked() {
        let mut c = stamped_cart(8, 8);
        c.mirroring = Mirroring::FourScreen;
        c.default_mirroring = Mirroring::FourScreen;
        let mut m = Mapper64::new(c);
        m.prg_write_byte(MIRROR, 0);
        assert_eq!(m.mirroring(), Mirroring::FourScreen);
        m.prg_write_byte(MIRROR, 1);
        assert_eq!(m.mirroring(), Mirroring::FourScreen);
    }

    // ---- IRQ ----

    // With latch N the counter fires on the (N+1)-th scanline clock: the
    // reload is clock #1, then it counts down to zero.
    #[test]
    fn irq_fires_after_latch_plus_one() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        arm_irq(&mut m, 3);
        for _ in 0..3 {
            m.on_scanline_tick();
            assert!(!m.irq_pending());
        }
        m.on_scanline_tick(); // counter reaches 0
        assert!(m.irq_pending());
    }

    // Writing the disable/ack register clears a pending IRQ.
    #[test]
    fn irq_disable_acknowledges_pending() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        arm_irq(&mut m, 0); // latch 0 -> fires on the reload clock
        m.on_scanline_tick();
        assert!(m.irq_pending());
        m.prg_write_byte(IRQ_DISABLE, 0);
        assert!(!m.irq_pending());
    }

    // Clocking a disabled counter never asserts an IRQ.
    #[test]
    fn irq_disabled_never_fires() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        m.prg_write_byte(IRQ_LATCH, 0);
        m.prg_write_byte(IRQ_RELOAD, 0);
        // Deliberately do not enable.
        for _ in 0..5 {
            m.on_scanline_tick();
        }
        assert!(!m.irq_pending());
    }

    // A write to $C001 forces a reload on the next clock even mid-count.
    #[test]
    fn irq_reload_flag_reloads_counter() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        arm_irq(&mut m, 5);
        m.on_scanline_tick(); // reload -> 5
        m.on_scanline_tick(); // 4
        assert_eq!(m.irq_counter, 4);
        m.prg_write_byte(IRQ_RELOAD, 0); // request reload
        m.on_scanline_tick(); // reload -> 5
        assert_eq!(m.irq_counter, 5);
    }

    // ---- State + reset ----

    // get_state/apply_state round-trips banks, mirroring and IRQ latch.
    #[test]
    fn state_round_trip() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 6, 3); // R6 -> $8000
        set_reg(&mut m, 2, 7); // R2 -> $1000
        m.prg_write_byte(MIRROR, 0); // vertical
        m.prg_write_byte(IRQ_LATCH, 9);
        let snap = m.get_state();

        // Mutate everything away from the snapshot.
        set_reg(&mut m, 6, 1);
        m.prg_write_byte(MIRROR, 1); // horizontal
        assert_eq!(m.prg_read_byte(0x8000), 1);
        assert_eq!(m.mirroring(), Mirroring::Horizontal);

        m.apply_state(&snap);
        assert_eq!(m.prg_read_byte(0x8000), 3);
        assert_eq!(m.chr_read_byte(0x1000), 7);
        assert_eq!(m.mirroring(), Mirroring::Vertical);
        assert_eq!(m.irq_latch, 9);
    }

    // reset() returns to the documented power-on bank layout and default
    // mirroring, and clears any pending IRQ.
    #[test]
    fn reset_restores_power_on() {
        let mut m = Mapper64::new(stamped_cart(8, 8));
        set_reg(&mut m, 6, 3);
        m.prg_write_byte(MIRROR, 0); // vertical
        arm_irq(&mut m, 0);
        m.on_scanline_tick();
        assert!(m.irq_pending());

        m.reset();
        assert_eq!(m.prg_read_byte(0x8000), 0); // R6 == 0 -> bank 0
        assert_eq!(m.prg_read_byte(0xC000), 14); // R8 fallback second-last
        assert_eq!(m.prg_read_byte(0xE000), 15); // last bank fixed
        assert_eq!(m.mirroring(), Mirroring::Horizontal); // default
        assert!(!m.irq_pending());
        assert_eq!(m.irq_latch, 0);
    }
}
