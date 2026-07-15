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

    // --- Real PPU-A12 IRQ clocking (MMC3 spec) ---
    /// Last PPU A12 level (address bit 12) observed on the CHR address
    /// bus (true = high). Rising edges of this line clock the counter.
    a12_prev_high: bool,
    /// CPU (M2) cycle of the most recent CHR access with A12 high.
    /// Used to measure how long A12 stayed low before a rising edge,
    /// implementing the MMC3 spec's rising-edge M2 filter.
    a12_last_high_cycle: u64,
    /// Current CPU cycle, mirrored from `set_cpu_cycle`.
    cpu_cycle: u64,
    /// CPU cycle of the most recent heuristic `on_scanline_tick` clock.
    /// While this stays fresh (the PPUCTRL heuristic fires every
    /// scanline for the common BG≠sprite-table case) the A12-derived
    /// supplemental clock is suppressed, so the common case is byte-for-
    /// byte unchanged and cannot double-count.
    last_heuristic_cycle: u64,
}

/// MMC3 A12 rising-edge filter: A12 must have been low for at least
/// this many CPU (M2) cycles before a rise clocks the counter. Per the
/// nesdev MMC3 spec: "clocked on a rising edge of PPU A12 after the
/// line has remained low for three falling edges of M2."
const A12_FILTER_CYCLES: u64 = 3;

/// If the PPUCTRL heuristic (`on_scanline_tick`) has clocked within
/// this many CPU cycles, it owns the per-scanline clock and the
/// A12-derived path is suppressed to avoid double-counting. One
/// scanline is ~113.7 CPU cycles; ~2 scanlines of slack guarantees the
/// A12 path only activates when the heuristic is genuinely silent (the
/// 8x16-sprite / same-pattern-table configurations it cannot see).
const HEURISTIC_OWNS_CYCLES: u64 = 240;

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
    #[serde(default)]
    pub a12_prev_high: bool,
    #[serde(default)]
    pub a12_last_high_cycle: u64,
    #[serde(default)]
    pub cpu_cycle: u64,
    #[serde(default)]
    pub last_heuristic_cycle: u64,
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
            a12_prev_high: false,
            a12_last_high_cycle: 0,
            cpu_cycle: 0,
            last_heuristic_cycle: 0,
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

    /// One MMC3 IRQ-counter clock: reload from the latch when the
    /// counter is zero or a reload was requested ($C001), otherwise
    /// decrement; assert the IRQ when the counter reaches zero and IRQs
    /// are enabled ($E001). Shared by every clock source (the PPUCTRL
    /// scanline heuristic and the real-A12 edge tracker) so the counter
    /// semantics live in exactly one place.
    fn clock_irq(&mut self) {
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

    /// Track PPU A12 (address bit 12) off the *actual* CHR address bus
    /// and clock the IRQ counter on a filtered low->high transition, per
    /// the MMC3 spec. This is what catches the scanline IRQs the coarse
    /// PPUCTRL heuristic (`on_scanline_tick`) is blind to:
    ///
    ///   * **8x16 sprites** — the sprite pattern table is chosen per
    ///     sprite by bit 0 of the tile index, so PPUCTRL bit 3 (which
    ///     the heuristic reads) is meaningless and A12 can rise even
    ///     when the heuristic thinks it can't.
    ///   * **Same pattern table** — BG and sprites both selected to
    ///     $0000 (or both $1000): the heuristic emits zero clocks, but
    ///     with 8x16 sprites A12 still rises on real hardware.
    ///
    /// The heuristic stays the authoritative per-scanline clock for the
    /// common (BG≠sprite-table, 8x8) case: this PPU elides garbage
    /// sprite fetches for empty OAM slots, so the CHR-fetch stream alone
    /// would miss the per-scanline A12 rise on sprite-less scanlines,
    /// whereas the heuristic is fetch-independent. To avoid double-
    /// counting, this A12 path only clocks while the heuristic is
    /// *silent* (`last_heuristic_cycle` stale by > ~2 scanlines), i.e.
    /// exactly the configurations the heuristic cannot handle.
    ///
    /// The filter measures the low duration in CPU (M2) cycles via
    /// `cpu_cycle`. Multicart wrappers (mappers 37/47) don't forward
    /// `set_cpu_cycle` to this inner mapper, so `cpu_cycle` stays 0
    /// there; the `heuristic_owns` guard is then always true and this
    /// path is dormant, leaving those wrappers on the unchanged
    /// heuristic clock.
    fn clock_a12(&mut self, address: u16) {
        let high = (address & 0x1000) != 0;
        if high {
            if !self.a12_prev_high {
                let low_dur = self.cpu_cycle.wrapping_sub(self.a12_last_high_cycle);
                let heuristic_owns = self
                    .cpu_cycle
                    .wrapping_sub(self.last_heuristic_cycle)
                    < HEURISTIC_OWNS_CYCLES;
                if low_dur >= A12_FILTER_CYCLES && !heuristic_owns {
                    self.clock_irq();
                }
            }
            self.a12_last_high_cycle = self.cpu_cycle;
        }
        self.a12_prev_high = high;
    }

    fn handle_scanline(&mut self, _cpu: &mut Cpu) {
        self.clock_irq();
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
        self.clock_a12(address);
        self.read_chr(address)
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        self.clock_a12(address);
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

    fn set_cpu_cycle(&mut self, cpu_cycle: u64) {
        self.cpu_cycle = cpu_cycle;
    }

    fn on_scanline_tick(&mut self) {
        // Coarse PPUCTRL heuristic clock, called once per visible +
        // pre-render scanline by the PPU on the rising edge of A12 it
        // infers from the BG/sprite pattern-table selection. This is the
        // authoritative per-scanline clock for the common (BG≠sprite
        // table, 8x8) case; the fetch-driven `clock_a12` path defers to
        // it (see `clock_a12`). It fires regardless of CHR-fetch
        // elision, so sprite-less scanlines still clock here.
        self.last_heuristic_cycle = self.cpu_cycle;
        self.clock_irq();
    }

    fn uses_scanline_irq(&self) -> bool {
        true
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
        self.a12_prev_high = false;
        self.a12_last_high_cycle = 0;
        self.cpu_cycle = 0;
        self.last_heuristic_cycle = 0;
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
            a12_prev_high: self.a12_prev_high,
            a12_last_high_cycle: self.a12_last_high_cycle,
            cpu_cycle: self.cpu_cycle,
            last_heuristic_cycle: self.last_heuristic_cycle,
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
                self.a12_prev_high = state.a12_prev_high;
                self.a12_last_high_cycle = state.a12_last_high_cycle;
                self.cpu_cycle = state.cpu_cycle;
                self.last_heuristic_cycle = state.last_heuristic_cycle;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{Cartridge, Mirroring};
    use crate::mapper::Mapper;

    // MMC3 IRQ register addresses (any even/odd address in the range
    // works; the mapper only looks at bit 0 and the $C000/$E000 split).
    const IRQ_LATCH: u16 = 0xC000; // even in $C000-$DFFF
    const IRQ_RELOAD: u16 = 0xC001; // odd in $C000-$DFFF
    const IRQ_DISABLE_ACK: u16 = 0xE000; // even in $E000-$FFFF
    const IRQ_ENABLE: u16 = 0xE001; // odd in $E000-$FFFF

    fn test_cart(prg_16k_banks: u8, chr_8k_banks: u8) -> Cartridge {
        Cartridge {
            mapper: 4,
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

    fn new_mapper() -> Mapper4 {
        Mapper4::new(test_cart(8, 8))
    }

    /// Arm the IRQ: latch = `latch`, request reload, enable IRQs.
    fn arm(m: &mut Mapper4, latch: u8) {
        m.prg_write_byte(IRQ_LATCH, latch);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
    }

    // ---- Counter semantics (shared by both clock sources) ----

    #[test]
    fn heuristic_clock_fires_irq_after_latch_plus_one() {
        // The PPUCTRL heuristic (`on_scanline_tick`) remains the
        // authoritative per-scanline clock for the common case. latch=3
        // -> reload counts as clock #1, then 3->2->1->0 fires on clock 4.
        let mut m = new_mapper();
        arm(&mut m, 3);
        for _ in 0..3 {
            m.on_scanline_tick();
            assert!(!m.irq_pending());
        }
        m.on_scanline_tick(); // 4th clock: counter hits 0
        assert!(m.irq_pending());
    }

    #[test]
    fn disable_acknowledges_pending_irq() {
        let mut m = new_mapper();
        arm(&mut m, 0); // latch 0 -> fires every clock after reload
        m.on_scanline_tick(); // reload -> 0, enabled -> asserts
        assert!(m.irq_pending());
        m.prg_write_byte(IRQ_DISABLE_ACK, 0);
        assert!(!m.irq_pending());
    }

    // ---- No double-count in the common case ----

    #[test]
    fn a12_rise_is_suppressed_while_heuristic_owns_the_scanline() {
        // Common case: BG and sprites use different pattern tables, so
        // the PPU heuristic clocks every scanline AND the CHR fetch
        // stream produces an A12 rise on the same scanline. The A12 path
        // must NOT add a second clock.
        let mut m = new_mapper();
        m.prg_write_byte(IRQ_LATCH, 200);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);

        m.set_cpu_cycle(1000);
        m.on_scanline_tick(); // reload -> counter = 200
        assert_eq!(m.irq_counter, 200);

        // BG fetches (A12 low) then a sprite fetch (A12 high) a few
        // cycles later — the classic BG=$0000 / sprites=$1000 rise.
        m.set_cpu_cycle(1002);
        m.chr_read_byte(0x0000);
        m.set_cpu_cycle(1004);
        m.chr_read_byte(0x1000); // rising edge, but heuristic fresh

        // Counter unchanged by the A12 path (no double count).
        assert_eq!(m.irq_counter, 200);
    }

    // ---- Same-pattern-table 8x16: heuristic silent, A12 must clock ----

    #[test]
    fn a12_clocks_when_heuristic_is_silent_same_table() {
        // BG and sprites share a pattern table (or 8x16 mode makes
        // PPUCTRL's sprite-table bit meaningless): the PPU never calls
        // `on_scanline_tick`. Real hardware still clocks on genuine A12
        // rises from the sprite pattern fetches — the A12 path must
        // reproduce that.
        let mut m = new_mapper();
        arm(&mut m, 3);

        // Advance well past HEURISTIC_OWNS_CYCLES with no heuristic tick.
        let mut clocks = 0;
        let mut fired_after = None;
        for scan in 0..6u64 {
            let base = 1000 + scan * 120; // ~1 scanline apart
            m.set_cpu_cycle(base);
            m.chr_read_byte(0x0000); // BG region: A12 low
            m.set_cpu_cycle(base + 40);
            m.chr_read_byte(0x0000); // still low, long enough for filter
            m.set_cpu_cycle(base + 80);
            let before = m.irq_counter;
            m.chr_read_byte(0x1000); // sprite fetch: A12 rises -> clock
            if m.irq_counter != before || (before == 0 && m.irq_counter == m.irq_latch) {
                clocks += 1;
            }
            if m.irq_pending() && fired_after.is_none() {
                fired_after = Some(scan);
            }
        }
        assert!(clocks >= 4, "A12 path should clock each silent scanline");
        // latch=3 -> IRQ on the 4th clock (0-indexed scanline 3).
        assert_eq!(fired_after, Some(3));
    }

    #[test]
    fn a12_alternating_tables_produce_multiple_clocks_per_scanline() {
        // 8x16 sprites with alternating pattern tables ($0/$1) and a
        // sufficiently long low gap between the $1xxx groups each cause
        // a filtered rising edge -> more than one clock per scanline.
        let mut m = new_mapper();
        m.prg_write_byte(IRQ_LATCH, 200);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        m.set_cpu_cycle(5000);
        m.on_scanline_tick(); // arm counter to 200, but heuristic goes silent now
        let start = m.irq_counter;
        assert_eq!(start, 200);

        // Move far past the heuristic-owns window.
        let mut cyc = 6000u64;
        // Sequence within one "scanline": low, HIGH, low(>=filter), HIGH ...
        // three well-separated $1xxx groups -> three clocks.
        for _ in 0..3 {
            m.set_cpu_cycle(cyc);
            m.chr_read_byte(0x0000); // low
            cyc += 8; // low gap >= A12_FILTER_CYCLES
            m.set_cpu_cycle(cyc);
            m.chr_read_byte(0x1000); // rising edge -> clock
            cyc += 8;
        }
        assert_eq!(m.irq_counter, start - 3, "three separated rises = three clocks");
    }

    #[test]
    fn a12_filter_rejects_a_rise_after_too_short_a_low() {
        // A rising edge that follows a low period shorter than the M2
        // filter must be ignored (spurious edge rejection).
        let mut m = new_mapper();
        m.prg_write_byte(IRQ_LATCH, 200);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        m.set_cpu_cycle(5000);
        m.on_scanline_tick();
        let start = m.irq_counter; // 200
        let mut cyc = 6000u64;

        // Establish A12 high. A12 was low since init, so this first
        // rise follows a long low period and legitimately clocks once.
        m.set_cpu_cycle(cyc);
        m.chr_read_byte(0x1000);
        assert_eq!(m.irq_counter, start - 1);

        // Drop low for only 1 cycle, then rise: below A12_FILTER_CYCLES.
        cyc += 1;
        m.set_cpu_cycle(cyc);
        m.chr_read_byte(0x0000);
        cyc += 1;
        m.set_cpu_cycle(cyc);
        m.chr_read_byte(0x1000); // rise after 1-cycle low -> filtered out
        assert_eq!(m.irq_counter, start - 1, "short-low rise must not clock");

        // Now a proper long-low rise DOES clock.
        cyc += 1;
        m.set_cpu_cycle(cyc);
        m.chr_read_byte(0x0000);
        cyc += A12_FILTER_CYCLES + 2;
        m.set_cpu_cycle(cyc);
        m.chr_read_byte(0x1000);
        assert_eq!(m.irq_counter, start - 2, "long-low rise must clock once");
    }

    #[test]
    fn no_rise_without_a12_transition() {
        // Repeated fetches that never cross $1000 (BG and sprites both in
        // the low half, 8x8) produce no rising edge and no clock — this
        // is the correct "same table, no IRQ" hardware behavior.
        let mut m = new_mapper();
        m.prg_write_byte(IRQ_LATCH, 200);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        m.set_cpu_cycle(5000);
        m.on_scanline_tick();
        let start = m.irq_counter;
        let mut cyc = 6000u64;
        for _ in 0..20 {
            m.set_cpu_cycle(cyc);
            m.chr_read_byte(0x00A0); // always A12 low
            cyc += 8;
        }
        assert_eq!(m.irq_counter, start, "no A12 rise -> no clock");
    }

    // ---- Multicart wrappers never forward set_cpu_cycle ----

    #[test]
    fn a12_path_dormant_when_cpu_cycle_never_advances() {
        // Mappers 37/47 wrap an inner Mapper4 but don't forward
        // set_cpu_cycle, so cpu_cycle stays 0. `heuristic_owns` is then
        // always true and the A12 path must stay dormant, leaving the
        // wrapper on the unchanged heuristic clock (no lost/double IRQs).
        let mut m = new_mapper();
        m.prg_write_byte(IRQ_LATCH, 200);
        m.prg_write_byte(IRQ_RELOAD, 0);
        m.prg_write_byte(IRQ_ENABLE, 0);
        // NOTE: deliberately never call set_cpu_cycle.
        m.on_scanline_tick(); // heuristic clock still works (reload -> 200)
        assert_eq!(m.irq_counter, 200);
        // Feed A12 rises; they must be ignored (cpu_cycle frozen at 0).
        for _ in 0..5 {
            m.chr_read_byte(0x0000);
            m.chr_read_byte(0x1000);
        }
        assert_eq!(m.irq_counter, 200, "A12 path must be dormant w/o cpu_cycle");
        // Heuristic keeps driving the counter as before.
        m.on_scanline_tick();
        assert_eq!(m.irq_counter, 199);
    }

    // ---- State round-trip ----

    #[test]
    fn a12_state_survives_save_restore() {
        let mut m = new_mapper();
        m.set_cpu_cycle(12345);
        m.chr_read_byte(0x1000); // sets a12_prev_high + a12_last_high_cycle
        m.on_scanline_tick(); // sets last_heuristic_cycle
        let saved = m.get_state();

        let mut m2 = new_mapper();
        m2.apply_state(&saved);
        assert_eq!(m2.a12_prev_high, m.a12_prev_high);
        assert_eq!(m2.a12_last_high_cycle, m.a12_last_high_cycle);
        assert_eq!(m2.cpu_cycle, m.cpu_cycle);
        assert_eq!(m2.last_heuristic_cycle, m.last_heuristic_cycle);
    }
}
