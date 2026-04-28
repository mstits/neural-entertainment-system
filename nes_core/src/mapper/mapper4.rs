use crate::cartridge::{self, Cartridge, Mirroring};
use crate::cpu::Cpu;
use crate::mapper::{self, Mapper};
use crate::ppu::{self, Ppu};

use serde_derive::{Deserialize, Serialize};

pub struct Mapper4 {
    cartridge: Cartridge,

    next_bank_register: u8,
    bank_registers: [u8; 8],

    prg_rom_mode: PrgRomMode,
    chr_a12_inversion: ChrA12Inversion,

    irq_enabled: bool,
    irq_counter: u8,
    irq_reload_flag: bool,
    irq_latch: u8,
    irq_active: bool,

    prg_rom_bank_offsets: [usize; 4],
    chr_bank_offsets: [usize; 8],

    /// 32 KB flat view of the currently-mapped PRG at $8000-$FFFF
    /// (four 8 KB banks), kept in sync with `update_banks`. Used by
    /// the AArch64 ASM CPU fast path.
    prg_asm_window: Vec<u8>,

    /// MMC3 variant (Mmc3 / TxSROM / TQROM). Gates behavior in the few
    /// places TxSROM (118) and TQROM (119) deviate from vanilla MMC3.
    variant: Mapper4Variant,

    /// Per-1KB-slot nametable-page select for TxSROM (mapper 118).
    /// Each entry is 0 or 1; used to compose OneScreenLower/Upper-style
    /// mirroring from the high bit of the last CHR bank value that
    /// mapped into that slot. Unused by Mmc3/TQROM.
    txsrom_nt_select: [u8; 8],

    /// 8 KB CHR-RAM buffer for TQROM (mapper 119). CHR reads/writes
    /// fall back here when the active 1 KB slot has bit 6 set in its
    /// source bank register. Allocated unconditionally — 8 KB is cheap.
    chr_ram: [u8; 8192],

    /// TQROM: which of the 8 1 KB CHR slots currently map to CHR-RAM
    /// (true) vs CHR-ROM (false). Updated from bit 6 of each source
    /// bank register in `update_banks`.
    chr_slot_is_ram: [bool; 8],

    /// Outer-bank PRG window used by multicart wrappers (mappers 37, 47).
    /// MMC3 bank register values get masked into this region so the
    /// inner MMC3 only sees the currently-selected outer chip. Base
    /// and size are in bytes; size == 0 disables masking (regular MMC3).
    outer_prg_base: usize,
    outer_prg_size: usize,

    /// Outer-bank CHR window (see outer_prg_* above).
    outer_chr_base: usize,
    outer_chr_size: usize,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum Mapper4Variant {
    Mmc3,
    TxSROM,
    TQROM,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum PrgRomMode {
    Zero, // $8000-$9FFF swappable, $C000-$DFFF fixed to second-last bank
    One,  // $C000-$DFFF swappable, $8000-$9FFF fixed to second-last bank
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum ChrA12Inversion {
    Zero, // Two 2 KB banks at $0000-$0FFF, four 1 KB banks at $1000-$1FFF
    One,  // Two 2 KB banks at $1000-$1FFF, four 1 KB banks at $0000-$0FFF
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub next_bank_register: u8,
    pub bank_registers: [u8; 8],
    pub prg_rom_mode: PrgRomMode,
    pub chr_a12_inversion: ChrA12Inversion,
    pub irq_enabled: bool,
    pub irq_counter: u8,
    pub irq_reload_flag: bool,
    pub irq_latch: u8,
    pub irq_active: bool,
    pub prg_rom_bank_offsets: [usize; 4],
    pub chr_bank_offsets: [usize; 8],
    #[serde(default = "default_variant")]
    pub variant: Mapper4Variant,
    #[serde(default)]
    pub txsrom_nt_select: [u8; 8],
    #[serde(default = "default_chr_ram")]
    pub chr_ram: Vec<u8>,
}

fn default_variant() -> Mapper4Variant {
    Mapper4Variant::Mmc3
}

fn default_chr_ram() -> Vec<u8> {
    vec![0u8; 8192]
}

impl Mapper4 {
    pub fn new(cartridge: Cartridge) -> Self {
        Self::new_with_variant(cartridge, Mapper4Variant::Mmc3)
    }

    pub fn new_with_variant(cartridge: Cartridge, variant: Mapper4Variant) -> Self {
        let mut m = Mapper4 {
            cartridge,
            next_bank_register: 0,
            bank_registers: [0, 0, 0, 0, 0, 0, 0, 1],
            prg_rom_mode: PrgRomMode::Zero,
            chr_a12_inversion: ChrA12Inversion::Zero,
            irq_enabled: false,
            irq_counter: 0,
            irq_reload_flag: false,
            irq_latch: 0,
            irq_active: false,
            prg_rom_bank_offsets: [0; 4],
            chr_bank_offsets: [0; 8],
            prg_asm_window: vec![0u8; 32 * 1024],
            variant,
            txsrom_nt_select: [0u8; 8],
            chr_ram: [0u8; 8192],
            chr_slot_is_ram: [false; 8],
            outer_prg_base: 0,
            outer_prg_size: 0,
            outer_chr_base: 0,
            outer_chr_size: 0,
        };
        m.update_banks();
        m.rebuild_asm_window();
        m
    }

    /// Select an outer-bank window for multicart wrappers (mappers 37/47).
    /// After this call, MMC3 bank register values are interpreted relative
    /// to the given region (base + size bytes). Use size=0 to clear.
    pub(crate) fn set_outer_region(
        &mut self,
        prg_base: usize,
        prg_size: usize,
        chr_base: usize,
        chr_size: usize,
    ) {
        self.outer_prg_base = prg_base;
        self.outer_prg_size = prg_size;
        self.outer_chr_base = chr_base;
        self.outer_chr_size = chr_size;
        self.update_banks();
        self.rebuild_asm_window();
    }

    fn rebuild_asm_window(&mut self) {
        // Four 8KB banks at $8000-$9FFF / $A000-$BFFF / $C000-$DFFF / $E000-$FFFF.
        // prg_rom_bank_offsets[i] is the byte offset into prg_rom for bank i.
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
        self.next_bank_register = value & 0x07;
        self.prg_rom_mode = if value & 0x40 == 0 {
            PrgRomMode::Zero
        } else {
            PrgRomMode::One
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
        self.bank_registers[self.next_bank_register as usize] = value;
        self.update_banks();
        self.rebuild_asm_window();
        if self.variant == Mapper4Variant::TxSROM {
            self.update_txsrom_mirroring();
        }
    }

    /// TxSROM (mapper 118): derive the effective nametable mirroring
    /// from bit 7 of each CHR bank register. In MMC3 the 8 KB pattern
    /// table is split into 4 KB + 4 KB (with A12 inversion swapping
    /// halves). For TxSROM the "upper 4 KB" half (four 1 KB banks)
    /// drives a 1-bit NT-page select per 1 KB slot. The NT quadrants
    /// are then pages {NT0, NT1, NT2, NT3} -> {page(a), page(b),
    /// page(c), page(d)}. We fold that onto Horizontal / Vertical /
    /// OneScreen{Lower,Upper} which is what the rest of the emulator
    /// supports. This covers the real-game patterns (all four-cart
    /// library uses one of these).
    fn update_txsrom_mirroring(&mut self) {
        if self.cartridge.mirroring == Mirroring::FourScreen {
            return;
        }
        // Per ChrA12Inversion, the four 1 KB slots of the "upper" CHR
        // half come from different bank registers:
        //   Inversion::Zero: $1000-$1FFF <- R2, R3, R4, R5 (1 KB each)
        //   Inversion::One:  $0000-$0FFF <- R2, R3, R4, R5 (same regs)
        // Either way R2..R5 are the four NT drivers.
        let nt = [
            (self.bank_registers[2] >> 7) & 1,
            (self.bank_registers[3] >> 7) & 1,
            (self.bank_registers[4] >> 7) & 1,
            (self.bank_registers[5] >> 7) & 1,
        ];
        self.txsrom_nt_select = [nt[0], nt[1], nt[2], nt[3], nt[0], nt[1], nt[2], nt[3]];
        // Fold the four NT bits onto a supported Mirroring value.
        // NT quadrants are [NT_TL, NT_TR, NT_BL, NT_BR] = [0,1,2,3].
        // Horizontal pairs (0,1) and (2,3); Vertical pairs (0,2) and (1,3).
        let [a, b, c, d] = nt;
        self.cartridge.mirroring = if a == b && c == d && a != c {
            // Two horizontal stripes, distinct pages -> Horizontal
            Mirroring::Horizontal
        } else if a == c && b == d && a != b {
            // Two vertical stripes, distinct pages -> Vertical
            Mirroring::Vertical
        } else if a == 0 && b == 0 && c == 0 && d == 0 {
            Mirroring::OneScreenLower
        } else if a == 1 && b == 1 && c == 1 && d == 1 {
            Mirroring::OneScreenUpper
        } else {
            // Arbitrary pattern — keep last mirroring (best-effort).
            self.cartridge.mirroring
        };
    }

    fn write_mirroring(&mut self, value: u8) {
        // TxSROM (mapper 118) ignores the MMC3 $A000 mirroring register.
        // Mirroring is controlled per-CHR-fetch via bit 7 of the latest
        // CHR bank register value — see `write_bank_data`.
        if self.variant == Mapper4Variant::TxSROM {
            return;
        }
        if self.cartridge.mirroring != Mirroring::FourScreen {
            self.cartridge.mirroring = if value & 0x01 == 0 {
                Mirroring::Vertical
            } else {
                Mirroring::Horizontal
            };
        }
    }

    fn write_prg_ram_protect(&mut self, _value: u8) {
        // Probably don't need to implement this
    }

    fn prg_bank_address(&self, bank: u8) -> usize {
        if self.outer_prg_size > 0 {
            let region_banks = (self.outer_prg_size / 0x2000).max(1);
            let bank = (bank as usize) % region_banks;
            self.outer_prg_base + bank * 0x2000
        } else {
            let bank = (bank as usize) % (self.cartridge.prg_rom.len() / 0x2000);
            bank * 0x2000
        }
    }

    fn chr_bank_address(&self, bank: u8) -> usize {
        if self.outer_chr_size > 0 {
            let region_banks = (self.outer_chr_size / 0x0400).max(1);
            let bank = (bank as usize) % region_banks;
            self.outer_chr_base + bank * 0x0400
        } else {
            let bank = (bank as usize) % (self.cartridge.chr.len() / 0x0400);
            bank * 0x0400
        }
    }

    /// TQROM CHR-RAM address resolver. The 8 KB CHR-RAM holds 8 × 1 KB
    /// "banks"; bits 0..2 of the bank value pick one of them.
    fn chr_ram_bank_address(&self, bank: u8) -> usize {
        ((bank as usize) & 0x07) * 0x0400
    }

    fn update_banks(&mut self) {
        // "Fixed last bank" and "fixed second-to-last bank" refer to the
        // outer region when one is active (multicart wrappers), otherwise
        // the whole PRG ROM. Sizes are in 8 KB units.
        let region_8k_banks: usize = if self.outer_prg_size > 0 {
            (self.outer_prg_size / 0x2000).max(1)
        } else {
            (self.cartridge.prg_rom_num_banks as usize) * 2
        };
        let last_bank = (region_8k_banks - 1) as u8;
        let second_last_bank = (region_8k_banks.saturating_sub(2)) as u8;

        self.prg_rom_bank_offsets[1] = self.prg_bank_address(self.bank_registers[7] & 0x3F);
        self.prg_rom_bank_offsets[3] = self.prg_bank_address(last_bank);

        match self.prg_rom_mode {
            PrgRomMode::Zero => {
                self.prg_rom_bank_offsets[0] = self.prg_bank_address(self.bank_registers[6] & 0x3F);
                self.prg_rom_bank_offsets[2] = self.prg_bank_address(second_last_bank);
            }
            PrgRomMode::One => {
                self.prg_rom_bank_offsets[0] = self.prg_bank_address(second_last_bank);
                self.prg_rom_bank_offsets[2] = self.prg_bank_address(self.bank_registers[6] & 0x3F);
            }
        }

        // CHR slot -> (source_bank_register_index, is_1kb_granularity).
        // For 2 KB banks (R0, R1), the two 1 KB sub-slots share the
        // same source register, so their RAM-vs-ROM flag is identical.
        let chr_slot_src: [u8; 8] = match self.chr_a12_inversion {
            ChrA12Inversion::Zero => [0, 0, 1, 1, 2, 3, 4, 5],
            ChrA12Inversion::One => [2, 3, 4, 5, 0, 0, 1, 1],
        };

        for slot in 0..8 {
            let src = chr_slot_src[slot] as usize;
            let raw = self.bank_registers[src];
            // TQROM: bit 6 selects CHR-RAM for this slot.
            let is_ram = self.variant == Mapper4Variant::TQROM && (raw & 0x40) != 0;
            self.chr_slot_is_ram[slot] = is_ram;
        }

        // Derive the actual bank value passed into chr_bank_address.
        // For TQROM we strip bits 6-7 (bit 7 is unused in TQROM; bit 6
        // is the RAM select). For 2 KB banks the LSB is forced via
        // &0xFE / |0x01 to pick the correct half.
        let strip = |b: u8| if self.variant == Mapper4Variant::TQROM { b & 0x3F } else { b };

        match self.chr_a12_inversion {
            ChrA12Inversion::Zero => {
                let r0 = strip(self.bank_registers[0]);
                let r1 = strip(self.bank_registers[1]);
                self.chr_bank_offsets[0] = self.resolve_chr_offset(0, r0 & 0xFE);
                self.chr_bank_offsets[1] = self.resolve_chr_offset(1, r0 | 0x01);
                self.chr_bank_offsets[2] = self.resolve_chr_offset(2, r1 & 0xFE);
                self.chr_bank_offsets[3] = self.resolve_chr_offset(3, r1 | 0x01);
                self.chr_bank_offsets[4] = self.resolve_chr_offset(4, strip(self.bank_registers[2]));
                self.chr_bank_offsets[5] = self.resolve_chr_offset(5, strip(self.bank_registers[3]));
                self.chr_bank_offsets[6] = self.resolve_chr_offset(6, strip(self.bank_registers[4]));
                self.chr_bank_offsets[7] = self.resolve_chr_offset(7, strip(self.bank_registers[5]));
            }
            ChrA12Inversion::One => {
                let r0 = strip(self.bank_registers[0]);
                let r1 = strip(self.bank_registers[1]);
                self.chr_bank_offsets[0] = self.resolve_chr_offset(0, strip(self.bank_registers[2]));
                self.chr_bank_offsets[1] = self.resolve_chr_offset(1, strip(self.bank_registers[3]));
                self.chr_bank_offsets[2] = self.resolve_chr_offset(2, strip(self.bank_registers[4]));
                self.chr_bank_offsets[3] = self.resolve_chr_offset(3, strip(self.bank_registers[5]));
                self.chr_bank_offsets[4] = self.resolve_chr_offset(4, r0 & 0xFE);
                self.chr_bank_offsets[5] = self.resolve_chr_offset(5, r0 | 0x01);
                self.chr_bank_offsets[6] = self.resolve_chr_offset(6, r1 & 0xFE);
                self.chr_bank_offsets[7] = self.resolve_chr_offset(7, r1 | 0x01);
            }
        }
    }

    /// Resolve the byte offset for a single 1 KB CHR slot. For TQROM
    /// slots that map to CHR-RAM, we store an offset into `chr_ram`
    /// (which `read_chr`/`write_chr` know to consult based on the
    /// `chr_slot_is_ram` flag).
    fn resolve_chr_offset(&self, slot: usize, bank: u8) -> usize {
        if self.chr_slot_is_ram[slot] {
            self.chr_ram_bank_address(bank)
        } else {
            self.chr_bank_address(bank)
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
        let addr = self.prg_rom_bank_offsets[(address as usize - 0x8000) / 0x2000]
            | (address as usize & 0x1FFF);
        self.cartridge.prg_rom[addr]
    }

    fn chr_address(&self, address: u16) -> usize {
        self.chr_bank_offsets[(address as usize) / 0x0400] | (address as usize & 0x03FF)
    }

    fn read_chr(&self, address: u16) -> u8 {
        let slot = (address as usize) / 0x0400;
        let addr = self.chr_address(address);
        if self.variant == Mapper4Variant::TQROM && self.chr_slot_is_ram[slot] {
            self.chr_ram[addr & 0x1FFF]
        } else {
            self.cartridge.chr[addr]
        }
    }

    fn write_chr(&mut self, address: u16, value: u8) {
        let slot = (address as usize) / 0x0400;
        let addr = self.chr_address(address);
        if self.variant == Mapper4Variant::TQROM && self.chr_slot_is_ram[slot] {
            self.chr_ram[addr & 0x1FFF] = value;
        } else {
            self.cartridge.chr[addr] = value;
        }
    }
}

impl Mapper for Mapper4 {
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
        // Dummy cpu arg — handle_scanline doesn't touch it.
        // This path IS called (from PPU tick) regardless of whether
        // the legacy `step` path is wired up, making MMC3 IRQs fire
        // correctly even when the CPU is running in bulk-ASM mode.
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
        self.bank_registers = [0, 0, 0, 0, 0, 0, 0, 1];
        self.prg_rom_mode = PrgRomMode::Zero;
        self.chr_a12_inversion = ChrA12Inversion::Zero;
        self.irq_enabled = false;
        self.irq_counter = 0;
        self.irq_reload_flag = false;
        self.irq_latch = 0;
        self.irq_active = false;
        self.prg_rom_bank_offsets = [0; 4];
        self.chr_bank_offsets = [0; 8];
        self.txsrom_nt_select = [0u8; 8];
        self.chr_slot_is_ram = [false; 8];
        // Preserve `variant` and `chr_ram` contents across reset —
        // chr_ram is user-writable SRAM-like storage whose clearing
        // semantics are not spec-defined; keeping contents is safest.
        self.update_banks();
        self.rebuild_asm_window();
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 { 1 }

    fn get_state(&self) -> mapper::State {
        mapper::State::State4(State {
            cartridge: self.cartridge.get_state(),
            next_bank_register: self.next_bank_register,
            bank_registers: self.bank_registers,
            prg_rom_mode: self.prg_rom_mode,
            chr_a12_inversion: self.chr_a12_inversion,
            irq_enabled: self.irq_enabled,
            irq_counter: self.irq_counter,
            irq_reload_flag: self.irq_reload_flag,
            irq_latch: self.irq_latch,
            irq_active: self.irq_active,
            prg_rom_bank_offsets: self.prg_rom_bank_offsets,
            chr_bank_offsets: self.chr_bank_offsets,
            variant: self.variant,
            txsrom_nt_select: self.txsrom_nt_select,
            chr_ram: self.chr_ram.to_vec(),
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State4(state) => {
                self.cartridge.apply_state(&state.cartridge);
                self.next_bank_register = state.next_bank_register;
                self.bank_registers = state.bank_registers;
                self.prg_rom_mode = state.prg_rom_mode;
                self.chr_a12_inversion = state.chr_a12_inversion;
                self.irq_enabled = state.irq_enabled;
                self.irq_counter = state.irq_counter;
                self.irq_reload_flag = state.irq_reload_flag;
                self.irq_latch = state.irq_latch;
                self.irq_active = state.irq_active;
                self.prg_rom_bank_offsets = state.prg_rom_bank_offsets;
                self.chr_bank_offsets = state.chr_bank_offsets;
                self.variant = state.variant;
                self.txsrom_nt_select = state.txsrom_nt_select;
                if state.chr_ram.len() == self.chr_ram.len() {
                    self.chr_ram.copy_from_slice(&state.chr_ram);
                }
                // Recompute the chr_slot_is_ram flags from the
                // restored bank registers + variant.
                self.update_banks();
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant in apply_state"),
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_active
    }
}
