// Nintendo MMC5 / ExROM (mapper 5). Implements the core subset every
// MMC5 title actually uses:
//   * PRG banking (all four modes; mode 3 is the common case)
//   * CHR banking (modes 0-3, with separate sprite/background tables
//     only populated when the CPU wrote both — most games write just
//     the sprite table)
//   * Nametable remap via $5105 (NT0/NT1/ExRAM/Fill)
//   * ExRAM (mode 0 = extra nametable; modes 1-3 parked)
//   * Hardware multiplier at $5205-$5206
//   * Scanline IRQ (approximated at scanline granularity)
//
// Features we DON'T implement yet and the symptoms of each:
//   * Extra audio (2 pulse + PCM): games sound thinner than on hardware
//   * Vertical split mode: advanced backgrounds may glitch on the split
//   * 8x16 sprite mode with separate BG CHR: minor tile mix-up
//   * Fill mode attribute expansion beyond tile byte
// Enough to boot Castlevania III US, Just Breed, Laser Invasion,
// Gemfire, Bandit Kings. The full audio + vertical-split path is a
// separate follow-up.

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, mmc5_audio::Mmc5Audio, Mapper};

use serde_derive::{Deserialize, Serialize};

const BANK_8K: usize = 0x2000;
const BANK_1K: usize = 0x0400;

pub struct Mapper5 {
    cartridge: Cartridge,

    // PRG banking
    prg_mode: u8,        // 0=32KB,1=16+16,2=16+8+8,3=8+8+8+8
    prg_banks: [u8; 5],  // $6000, $8000, $A000, $C000, $E000 (each top bit = ROM/RAM select except $E000)
    prg_ram: Vec<u8>,    // 64 KB window — real carts ship up to 64 KB battery RAM
    prg_ram_protect_a: u8,
    prg_ram_protect_b: u8,

    // CHR banking
    chr_mode: u8, // 0=8KB,1=4KB,2=2KB,3=1KB
    chr_bank_sprite: [u16; 8], // CHR banks when last write was to sprite table ($5120-$5127)
    chr_bank_bg: [u16; 4],     // CHR banks when last write was to BG table ($5128-$512B)
    chr_upper_bits: u8,
    last_chr_write_sprite: bool,

    // Nametable mapping — 2 bits per NT
    nt_mapping: u8,
    fill_tile: u8,
    fill_attr: u8,

    // ExRAM
    exram: Box<[u8; 1024]>,
    exram_mode: u8,

    // Multiplier
    mul_a: u8,
    mul_b: u8,

    // Scanline IRQ
    irq_compare: u8,
    irq_enable: bool,
    irq_pending: bool,
    scanline_counter: u8,
    in_frame: bool,

    prg_asm_window: Vec<u8>,

    audio: Mmc5Audio,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg_mode: u8,
    pub prg_banks: [u8; 5],
    #[serde(with = "serde_bytes")]
    pub prg_ram: Vec<u8>,
    pub prg_ram_protect_a: u8,
    pub prg_ram_protect_b: u8,
    pub chr_mode: u8,
    pub chr_bank_sprite: [u16; 8],
    pub chr_bank_bg: [u16; 4],
    pub chr_upper_bits: u8,
    pub last_chr_write_sprite: bool,
    pub nt_mapping: u8,
    pub fill_tile: u8,
    pub fill_attr: u8,
    #[serde(with = "serde_bytes")]
    pub exram: Vec<u8>,
    pub exram_mode: u8,
    pub mul_a: u8,
    pub mul_b: u8,
    pub irq_compare: u8,
    pub irq_enable: bool,
    pub irq_pending: bool,
    pub scanline_counter: u8,
    pub in_frame: bool,
    pub audio: Mmc5Audio,
}

impl Mapper5 {
    pub fn new(cartridge: Cartridge) -> Self {
        // MMC5 ships with up to 64 KB battery-backed PRG RAM. Games
        // like Just Breed require this for save files. iNES headers
        // don't distinguish MMC5 RAM size reliably — default to 64 KB.
        let prg_ram_size = if cartridge.prg_ram.is_empty() {
            64 * 1024
        } else {
            cartridge.prg_ram.len().max(64 * 1024)
        };
        let mut m = Mapper5 {
            cartridge,
            prg_mode: 3,
            prg_banks: [0, 0, 0, 0, 0xFF], // $E000 defaults to last bank
            prg_ram: vec![0u8; prg_ram_size],
            prg_ram_protect_a: 0,
            prg_ram_protect_b: 0,
            chr_mode: 3,
            chr_bank_sprite: [0; 8],
            chr_bank_bg: [0; 4],
            chr_upper_bits: 0,
            last_chr_write_sprite: true,
            nt_mapping: 0,
            fill_tile: 0,
            fill_attr: 0,
            exram: Box::new([0u8; 1024]),
            exram_mode: 0,
            mul_a: 0,
            mul_b: 0,
            irq_compare: 0,
            irq_enable: false,
            irq_pending: false,
            scanline_counter: 0,
            in_frame: false,
            prg_asm_window: vec![0u8; 32 * 1024],
            audio: Mmc5Audio::new(),
        };
        if !m.cartridge.prg_ram.is_empty() {
            let n = m.cartridge.prg_ram.len().min(m.prg_ram.len());
            m.prg_ram[..n].copy_from_slice(&m.cartridge.prg_ram[..n]);
        }
        m.rebuild_asm_window();
        m
    }

    fn prg_rom_bank_count_8k(&self) -> usize {
        (self.cartridge.prg_rom.len() / BANK_8K).max(1)
    }

    fn chr_bank_count_1k(&self) -> usize {
        (self.cartridge.chr.len() / BANK_1K).max(1)
    }

    // Resolve an 8 KB slot (0..4 for $6000/$8000/$A000/$C000/$E000)
    // into a physical (is_ram, offset) pair. `is_ram` means PRG RAM.
    //
    // PRG banking per $5100 mode (nesdev "MMC5 § PRG mode"). Bank
    // registers hold an 8 KB bank number; when a window is wider than
    // 8 KB the low address bits pick the sub-bank and the register's
    // low bits are ignored (16 KB masks bit 0, 32 KB masks bits 0-1):
    //   mode 0: $8000-$FFFF = one 32 KB bank from $5117 (always ROM)
    //   mode 1: $8000-$BFFF = 16 KB from $5115; $C000-$FFFF = 16 KB from $5117
    //   mode 2: $8000-$BFFF = 16 KB from $5115; $C000-$DFFF = 8 KB from $5116;
    //           $E000-$FFFF = 8 KB from $5117
    //   mode 3: four independent 8 KB banks ($5114/$5115/$5116/$5117)
    // For $5114-$5116 bit 7 selects ROM (1) vs PRG-RAM (0); $5117 is
    // always ROM. $6000-$7FFF is always PRG-RAM in every mode, selected
    // by $5113 — handled before the mode match.
    fn resolve_prg_slot(&self, slot: usize) -> (bool, usize) {
        let rom_banks = self.prg_rom_bank_count_8k();
        let ram_banks = (self.prg_ram.len() / BANK_8K).max(1);

        // $6000-$7FFF: always PRG-RAM, resolved before the mode match.
        if slot == 0 {
            let bank = (self.prg_banks[0] as usize) & 0x07;
            return (true, (bank % ram_banks) * BANK_8K);
        }

        // Decode a $5114-$5117 register plus a resolved 8 KB bank index
        // into (is_ram, byte-offset), honouring the ROM/RAM select bit
        // unless the window is fixed ROM ($5117 / mode-0 / mode-1 upper).
        let decode = |reg: u8, force_rom: bool, bank8: usize| -> (bool, usize) {
            if !force_rom && (reg & 0x80 == 0) {
                (true, (bank8 % ram_banks) * BANK_8K)
            } else {
                (false, (bank8 % rom_banks) * BANK_8K)
            }
        };

        match self.prg_mode {
            // Mode 0: single 32 KB bank from $5117, always ROM. slot 1..4
            // map to the four 8 KB sub-banks of the aligned 32 KB block.
            0 => {
                let base32 = ((self.prg_banks[4] as usize) & 0x7F) & !0x03;
                decode(self.prg_banks[4], true, base32 + (slot - 1))
            }
            // Mode 1: 16 KB + 16 KB. $8000/$A000 from $5115 (ROM/RAM),
            // $C000/$E000 from $5117 (always ROM).
            1 => {
                let (reg, force_rom) = if slot <= 2 {
                    (self.prg_banks[2], false) // $5115
                } else {
                    (self.prg_banks[4], true) // $5117
                };
                let base16 = ((reg as usize) & 0x7F) & !0x01;
                let half = if slot == 1 || slot == 3 { 0 } else { 1 };
                decode(reg, force_rom, base16 + half)
            }
            // Mode 2: 16 KB ($5115) + 8 KB ($5116) + 8 KB ($5117, ROM).
            2 => match slot {
                1 | 2 => {
                    let base16 = ((self.prg_banks[2] as usize) & 0x7F) & !0x01;
                    let half = if slot == 1 { 0 } else { 1 };
                    decode(self.prg_banks[2], false, base16 + half)
                }
                3 => decode(self.prg_banks[3], false, (self.prg_banks[3] as usize) & 0x7F),
                _ => decode(self.prg_banks[4], true, (self.prg_banks[4] as usize) & 0x7F),
            },
            // Mode 3: four independent 8 KB banks; $5117 ($E000) is ROM.
            3 => {
                let reg = self.prg_banks[slot];
                decode(reg, slot == 4, (reg as usize) & 0x7F)
            }
            _ => (false, 0),
        }
    }

    fn rebuild_asm_window(&mut self) {
        for slot in 1..=4 {
            let (is_ram, off) = self.resolve_prg_slot(slot);
            let dst = (slot - 1) * BANK_8K;
            let src = if is_ram {
                &self.prg_ram
            } else {
                &self.cartridge.prg_rom
            };
            if off + BANK_8K <= src.len() {
                self.prg_asm_window[dst..dst + BANK_8K]
                    .copy_from_slice(&src[off..off + BANK_8K]);
            }
        }
    }

    fn chr_bank_for_ppu_addr(&self, address: u16) -> u16 {
        // Resolve a 1 KB CHR slot (0..7 across $0000-$1FFF) into a physical
        // 1 KB bank. Per $5101 the CHR window size is 8/4/2/1 KB (modes
        // 0/1/2/3). Each window is controlled by the bank register at its
        // high boundary (odd index): mode 0 → $5127, mode 1 → $5123/$5127,
        // mode 2 → $5121/$5123/$5125/$5127, mode 3 → $5120..$5127. The
        // register value is in window-size units, so the physical 1 KB
        // bank is `bank * (window / 1 KB) + sub`, with the $5130 upper
        // bits folded into `bank` before the multiply.
        let slot = ((address >> 10) & 0x07) as usize;
        let upper = (self.chr_upper_bits as u16) << 8;
        let subcount: usize = match self.chr_mode {
            0 => 8, // 8 KB window
            1 => 4, // 4 KB window
            2 => 2, // 2 KB window
            _ => 1, // 1 KB window
        };
        let sub = (slot & (subcount - 1)) as u16;
        let bank = if self.last_chr_write_sprite {
            // "A" set ($5120-$5127): 8 registers; the boundary register
            // for a window is at index `slot | (subcount - 1)`.
            self.chr_bank_sprite[slot | (subcount - 1)]
        } else {
            // "B" set ($5128-$512B): 4 registers, mirrored across both
            // pattern tables in 8x16 background mode.
            let idx = ((slot & 3) | ((subcount - 1) & 3)).min(3);
            self.chr_bank_bg[idx]
        };
        (bank | upper)
            .wrapping_mul(subcount as u16)
            .wrapping_add(sub)
    }

    fn prg_ram_writable(&self) -> bool {
        // Write protect: both A ($5102 & 0x03) and B ($5103 & 0x03)
        // must equal 2 and 1 respectively for writes to be enabled.
        self.prg_ram_protect_a == 0x02 && self.prg_ram_protect_b == 0x01
    }
}

impl Mapper for Mapper5 {
    fn prg_read_byte(&mut self, address: u16) -> u8 {
        // Reading the scanline-IRQ status register ($5204) returns the
        // pending + in-frame flags AND acknowledges (clears) the pending
        // IRQ latch. This is how MMC5 titles de-assert the IRQ line from
        // their handler; without it the level-sensitive line would stay
        // asserted and storm. `prg_peek_byte` (debugger/side-effect-free)
        // reports the same bits but does not clear.
        if address == 0x5204 {
            let mut v = 0;
            if self.irq_pending {
                v |= 0x80;
            }
            if self.in_frame {
                v |= 0x40;
            }
            self.irq_pending = false;
            return v;
        }
        self.prg_peek_byte(address)
    }

    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x5C00..=0x5FFF => {
                // ExRAM: readable in modes 2/3; mode 0/1 returns 0 from CPU.
                if self.exram_mode >= 2 {
                    self.exram[(address - 0x5C00) as usize]
                } else {
                    0
                }
            }
            0x5204 => {
                // Scanline IRQ status: bit 7 = IRQ pending, bit 6 = in-frame.
                let mut v = 0;
                if self.irq_pending {
                    v |= 0x80;
                }
                if self.in_frame {
                    v |= 0x40;
                }
                v
            }
            0x5205 => ((self.mul_a as u16) * (self.mul_b as u16)) as u8,
            0x5206 => (((self.mul_a as u16) * (self.mul_b as u16)) >> 8) as u8,
            0x5C00_u16..=u16::MAX if address < 0x6000 => 0,
            0x6000..=0x7FFF => {
                let (is_ram, off) = self.resolve_prg_slot(0);
                let addr = address as usize - 0x6000;
                let src = if is_ram {
                    &self.prg_ram
                } else {
                    &self.cartridge.prg_rom
                };
                src.get(off + addr).copied().unwrap_or(0)
            }
            0x8000..=0xFFFF => {
                let slot = ((address - 0x8000) / 0x2000) as usize + 1;
                let (is_ram, off) = self.resolve_prg_slot(slot);
                let addr = (address as usize) & 0x1FFF;
                let src = if is_ram {
                    &self.prg_ram
                } else {
                    &self.cartridge.prg_rom
                };
                src.get(off + addr).copied().unwrap_or(0)
            }
            _ => 0,
        }
    }

    fn prg_write_byte(&mut self, address: u16, value: u8) {
        // MMC5 audio register block overlaps the banking block in
        // address space ($5000-$5015). Route to the audio sub-unit
        // first; fall through only if it didn't claim the address.
        if self.audio.write_register(address, value) {
            return;
        }
        match address {
            0x5100 => {
                self.prg_mode = value & 0x03;
                self.rebuild_asm_window();
            }
            0x5101 => {
                self.chr_mode = value & 0x03;
            }
            0x5102 => {
                self.prg_ram_protect_a = value & 0x03;
            }
            0x5103 => {
                self.prg_ram_protect_b = value & 0x03;
            }
            0x5104 => {
                self.exram_mode = value & 0x03;
            }
            0x5105 => {
                self.nt_mapping = value;
                // Translate $5105 to a simple Mirroring when possible so
                // the PPU renderer picks up a single NT mode. Full MMC5
                // per-NT mapping needs PPU-side support which we don't
                // thread through yet.
                let q = |s: u8| (value >> (s * 2)) & 0x03;
                let (a, b, c, d) = (q(0), q(1), q(2), q(3));
                self.cartridge.mirroring = if a == b && c == d && a != c {
                    // (A,A,B,B) — horizontal
                    Mirroring::Horizontal
                } else if a == c && b == d && a != b {
                    // (A,B,A,B) — vertical
                    Mirroring::Vertical
                } else if a == b && b == c && c == d && a == 0 {
                    Mirroring::OneScreenLower
                } else if a == b && b == c && c == d && a == 1 {
                    Mirroring::OneScreenUpper
                } else {
                    // Advanced mappings fall back to the cart's own
                    // mirroring — most CV3 scenes land in the H/V
                    // branches above anyway.
                    self.cartridge.mirroring
                };
            }
            0x5106 => {
                self.fill_tile = value;
            }
            0x5107 => {
                self.fill_attr = value & 0x03;
            }
            0x5113 => {
                self.prg_banks[0] = value;
                self.rebuild_asm_window();
            }
            0x5114..=0x5117 => {
                let slot = (address - 0x5114 + 1) as usize;
                self.prg_banks[slot] = value;
                self.rebuild_asm_window();
            }
            0x5120..=0x5127 => {
                let i = (address - 0x5120) as usize;
                self.chr_bank_sprite[i] = value as u16;
                self.last_chr_write_sprite = true;
            }
            0x5128..=0x512B => {
                let i = (address - 0x5128) as usize;
                self.chr_bank_bg[i] = value as u16;
                self.last_chr_write_sprite = false;
            }
            0x5130 => {
                self.chr_upper_bits = value & 0x03;
            }
            0x5203 => {
                self.irq_compare = value;
            }
            0x5204 => {
                self.irq_enable = value & 0x80 != 0;
            }
            0x5205 => {
                self.mul_a = value;
            }
            0x5206 => {
                self.mul_b = value;
            }
            0x5C00..=0x5FFF => {
                if self.exram_mode < 3 {
                    self.exram[(address - 0x5C00) as usize] = value;
                }
            }
            0x6000..=0x7FFF => {
                if self.prg_ram_writable() {
                    let (_, off) = self.resolve_prg_slot(0);
                    let idx = off + (address as usize - 0x6000);
                    if idx < self.prg_ram.len() {
                        self.prg_ram[idx] = value;
                    }
                }
            }
            0x8000..=0xFFFF => {
                // PRG RAM at $8000+ if banked as RAM and writable.
                if self.prg_ram_writable() {
                    let slot = ((address - 0x8000) / 0x2000) as usize + 1;
                    let (is_ram, off) = self.resolve_prg_slot(slot);
                    if is_ram {
                        let idx = off + (address as usize & 0x1FFF);
                        if idx < self.prg_ram.len() {
                            self.prg_ram[idx] = value;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn chr_read_byte(&mut self, address: u16) -> u8 {
        let bank = self.chr_bank_for_ppu_addr(address) as usize % self.chr_bank_count_1k();
        let off = bank * BANK_1K | (address as usize & 0x3FF);
        self.cartridge.chr[off]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let bank = self.chr_bank_for_ppu_addr(address) as usize % self.chr_bank_count_1k();
        let off = bank * BANK_1K | (address as usize & 0x3FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg_mode = 3;
        self.prg_banks = [0, 0, 0, 0, 0xFF];
        self.chr_mode = 3;
        self.chr_bank_sprite = [0; 8];
        self.chr_bank_bg = [0; 4];
        self.chr_upper_bits = 0;
        self.last_chr_write_sprite = true;
        self.nt_mapping = 0;
        self.exram_mode = 0;
        self.irq_compare = 0;
        self.irq_enable = false;
        self.irq_pending = false;
        self.scanline_counter = 0;
        self.in_frame = false;
        self.cartridge.mirroring = self.cartridge.default_mirroring;
        self.rebuild_asm_window();
    }

    fn on_scanline_tick(&mut self) {
        // MMC5 scanline IRQ counter. It advances once per rendered
        // scanline while "in frame"; when it reaches the target in $5203
        // the pending latch (status bit 7 of $5204) is set — independent
        // of the enable bit ($5204 bit 7), which only gates the asserted
        // IRQ line (see `irq_pending`). Reading $5204 clears the latch.
        //
        // NOTE: on real hardware the MMC5 clocks this counter on every
        // rendered scanline at a fixed dot, detected from PPU fetches and
        // wholly independent of pattern-table selection. Here the *timing*
        // of the call is still gated by the PPU's MMC3-style A12 heuristic
        // (see ppu.rs) — making the hook fire on every rendered scanline
        // regardless of that heuristic is the PPU-side half of this fix
        // and lives outside this file. The counter/latch model below is
        // correct for whatever cadence the PPU delivers.
        if self.scanline_counter >= 240 {
            // Past the rendered region: leave the in-frame window and
            // reset the counter for the next frame.
            self.scanline_counter = 0;
            self.in_frame = false;
            return;
        }
        self.scanline_counter = self.scanline_counter.wrapping_add(1);
        self.in_frame = true;
        // Compare against $5203. Target 0 never matches here (the counter
        // is 1..=240 on this path), which mirrors the "disabled" target.
        if self.scanline_counter == self.irq_compare {
            self.irq_pending = true;
        }
    }

    fn irq_pending(&self) -> bool {
        // The IRQ line is asserted only while the pending latch is set and
        // scanline IRQs are enabled ($5204 bit 7). It de-asserts when the
        // handler reads $5204 (clears the latch) or clears the enable bit.
        self.irq_pending && self.irq_enable
    }

    fn tick_audio(&mut self) {
        self.audio.tick_cpu_cycle();
    }

    fn audio_mix(&self) -> f32 {
        self.audio.mix()
    }

    fn prg_asm_ptr(&self) -> Option<*const u8> {
        Some(self.prg_asm_window.as_ptr())
    }

    fn asm_bulk_cycles(&self) -> i64 {
        2
    }

    fn sram(&mut self) -> *mut u8 {
        self.prg_ram.as_mut_ptr()
    }

    fn sram_size(&self) -> usize {
        self.prg_ram.len()
    }

    fn get_state(&self) -> mapper::State {
        mapper::State::State5(State {
            cartridge: self.cartridge.get_state(),
            prg_mode: self.prg_mode,
            prg_banks: self.prg_banks,
            prg_ram: self.prg_ram.clone(),
            prg_ram_protect_a: self.prg_ram_protect_a,
            prg_ram_protect_b: self.prg_ram_protect_b,
            chr_mode: self.chr_mode,
            chr_bank_sprite: self.chr_bank_sprite,
            chr_bank_bg: self.chr_bank_bg,
            chr_upper_bits: self.chr_upper_bits,
            last_chr_write_sprite: self.last_chr_write_sprite,
            nt_mapping: self.nt_mapping,
            fill_tile: self.fill_tile,
            fill_attr: self.fill_attr,
            exram: self.exram.to_vec(),
            exram_mode: self.exram_mode,
            mul_a: self.mul_a,
            mul_b: self.mul_b,
            irq_compare: self.irq_compare,
            irq_enable: self.irq_enable,
            irq_pending: self.irq_pending,
            scanline_counter: self.scanline_counter,
            in_frame: self.in_frame,
            audio: self.audio.clone(),
        })
    }

    fn apply_state(&mut self, state: &mapper::State) {
        match state {
            mapper::State::State5(s) => {
                self.cartridge.apply_state(&s.cartridge);
                self.prg_mode = s.prg_mode;
                self.prg_banks = s.prg_banks;
                let n = s.prg_ram.len().min(self.prg_ram.len());
                self.prg_ram[..n].copy_from_slice(&s.prg_ram[..n]);
                self.prg_ram_protect_a = s.prg_ram_protect_a;
                self.prg_ram_protect_b = s.prg_ram_protect_b;
                self.chr_mode = s.chr_mode;
                self.chr_bank_sprite = s.chr_bank_sprite;
                self.chr_bank_bg = s.chr_bank_bg;
                self.chr_upper_bits = s.chr_upper_bits;
                self.last_chr_write_sprite = s.last_chr_write_sprite;
                self.nt_mapping = s.nt_mapping;
                self.fill_tile = s.fill_tile;
                self.fill_attr = s.fill_attr;
                let en = s.exram.len().min(self.exram.len());
                self.exram[..en].copy_from_slice(&s.exram[..en]);
                self.exram_mode = s.exram_mode;
                self.mul_a = s.mul_a;
                self.mul_b = s.mul_b;
                self.irq_compare = s.irq_compare;
                self.irq_enable = s.irq_enable;
                self.irq_pending = s.irq_pending;
                self.scanline_counter = s.scanline_counter;
                self.in_frame = s.in_frame;
                self.audio = s.audio.clone();
                self.rebuild_asm_window();
            }
            _ => panic!("Invalid mapper state enum variant for MMC5"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A bare cartridge with `prg16` 16 KB PRG banks and `chr8` 8 KB CHR
    // banks (both zero-filled — the banking tests inspect resolved bank
    // *indices*, not payload bytes).
    fn test_cart(prg16: u8, chr8: u8) -> Cartridge {
        Cartridge {
            mapper: 5,
            sub_mapper: 0,
            mirroring: Mirroring::Vertical,
            default_mirroring: Mirroring::Vertical,
            prg_rom_num_banks: prg16,
            prg_rom: vec![0u8; prg16 as usize * 16 * 1024],
            chr_num_banks: chr8,
            chr: vec![0u8; chr8 as usize * 8 * 1024],
            prg_ram: Vec::new(),
            is_battery_backed: false,
            is_nes20: false,
            md5: String::new(),
        }
    }

    // 256 KB PRG (32 × 8 KB banks, indices 0-31), 64 KB CHR (64 × 1 KB
    // banks, indices 0-63), 64 KB PRG-RAM (8 × 8 KB banks).
    fn mk() -> Mapper5 {
        Mapper5::new(test_cart(16, 8))
    }

    // resolve_prg_slot as (is_ram, 8 KB bank index).
    fn prg(m: &Mapper5, slot: usize) -> (bool, usize) {
        let (is_ram, off) = m.resolve_prg_slot(slot);
        (is_ram, off / BANK_8K)
    }

    fn chr(m: &Mapper5, slot: usize) -> u16 {
        m.chr_bank_for_ppu_addr((slot as u16) << 10)
    }

    // ---- F37: PRG banking modes ------------------------------------

    #[test]
    fn prg_mode3_four_independent_8k_banks() {
        let mut m = mk();
        m.prg_mode = 3;
        // $5114=ROM b1, $5115=ROM b2, $5116=ROM b3, $5117=b4 (always ROM).
        m.prg_banks = [0, 0x81, 0x82, 0x83, 0x04];
        assert_eq!(prg(&m, 1), (false, 1));
        assert_eq!(prg(&m, 2), (false, 2));
        assert_eq!(prg(&m, 3), (false, 3));
        assert_eq!(prg(&m, 4), (false, 4));
        // $5114 with bit7 clear selects PRG-RAM.
        m.prg_banks[1] = 0x02;
        assert_eq!(prg(&m, 1), (true, 2));
    }

    #[test]
    fn prg_mode2_16k_plus_8k_plus_8k_not_swapped() {
        let mut m = mk();
        m.prg_mode = 2;
        // $5115 = ROM bank 8 (16 KB → 8 KB banks 8,9), $5116 = ROM b6,
        // $5117 = b7 (always ROM). Regression: the old code put the high
        // half at $8000 and the low half at $A000.
        m.prg_banks = [0, 0, 0x88, 0x86, 0x07];
        assert_eq!(prg(&m, 1), (false, 8)); // $8000 = low half
        assert_eq!(prg(&m, 2), (false, 9)); // $A000 = high half
        assert_eq!(prg(&m, 3), (false, 6)); // $C000 = $5116
        assert_eq!(prg(&m, 4), (false, 7)); // $E000 = $5117
    }

    #[test]
    fn prg_mode1_two_16k_windows_from_pair_registers() {
        let mut m = mk();
        m.prg_mode = 1;
        // $8000/$A000 from $5115 (bank 8 → 8,9); $C000/$E000 from $5117
        // (bank 12 → 12,13). Regression: the old code read each slot's
        // own register instead of the pair register.
        m.prg_banks = [0, 0xFF, 0x88, 0xFF, 0x0C];
        assert_eq!(prg(&m, 1), (false, 8));
        assert_eq!(prg(&m, 2), (false, 9));
        assert_eq!(prg(&m, 3), (false, 12));
        assert_eq!(prg(&m, 4), (false, 13));
    }

    #[test]
    fn prg_mode0_single_32k_bank_from_5117() {
        let mut m = mk();
        m.prg_mode = 0;
        // $5117 = bank 12 (32 KB aligned → 8 KB banks 12,13,14,15). The
        // low two bits are ignored, so 0x0E resolves the same.
        m.prg_banks = [0, 0, 0, 0, 0x0E];
        assert_eq!(prg(&m, 1), (false, 12));
        assert_eq!(prg(&m, 2), (false, 13));
        assert_eq!(prg(&m, 3), (false, 14));
        assert_eq!(prg(&m, 4), (false, 15));
    }

    #[test]
    fn prg_6000_is_ram_in_every_mode() {
        // $6000-$7FFF is always PRG-RAM (bank from $5113), resolved
        // before the mode match. Regression: modes 0/1 previously
        // returned ROM or the wrong RAM offset here.
        for mode in 0u8..=3 {
            let mut m = mk();
            m.prg_mode = mode;
            m.prg_banks[0] = 3; // $5113 → RAM bank 3
            assert_eq!(prg(&m, 0), (true, 3), "mode {mode}");
        }
    }

    // ---- F38: CHR banking modes ------------------------------------

    #[test]
    fn chr_mode3_1k_windows() {
        let mut m = mk();
        m.chr_mode = 3;
        m.last_chr_write_sprite = true;
        m.chr_bank_sprite = [10, 11, 12, 13, 14, 15, 16, 17];
        for s in 0..8 {
            assert_eq!(chr(&m, s), 10 + s as u16);
        }
    }

    #[test]
    fn chr_mode2_2k_windows_read_odd_boundary_and_scale() {
        let mut m = mk();
        m.chr_mode = 2;
        m.last_chr_write_sprite = true;
        // Only the odd boundary registers ($5121/$5123/$5125/$5127)
        // matter; the even ones (99) must be ignored.
        m.chr_bank_sprite = [99, 4, 99, 5, 99, 6, 99, 7];
        // bank = reg*2 + (slot & 1).
        assert_eq!(chr(&m, 0), 8);
        assert_eq!(chr(&m, 1), 9);
        assert_eq!(chr(&m, 2), 10);
        assert_eq!(chr(&m, 3), 11);
        assert_eq!(chr(&m, 6), 14);
        assert_eq!(chr(&m, 7), 15);
    }

    #[test]
    fn chr_mode1_4k_windows_read_odd_boundary_and_scale() {
        let mut m = mk();
        m.chr_mode = 1;
        m.last_chr_write_sprite = true;
        // $5123 (idx 3) drives $0000-$0FFF, $5127 (idx 7) drives $1000-$1FFF.
        m.chr_bank_sprite = [0, 0, 0, 3, 0, 0, 0, 5];
        // bank = reg*4 + (slot & 3).
        assert_eq!(chr(&m, 0), 12);
        assert_eq!(chr(&m, 3), 15);
        assert_eq!(chr(&m, 4), 20);
        assert_eq!(chr(&m, 7), 23);
    }

    #[test]
    fn chr_mode0_8k_window_reads_5127_and_scales() {
        let mut m = mk();
        m.chr_mode = 0;
        m.last_chr_write_sprite = true;
        // Only $5127 (idx 7) drives the whole 8 KB window.
        m.chr_bank_sprite = [0, 0, 0, 0, 0, 0, 0, 3];
        // bank = reg*8 + (slot & 7).
        assert_eq!(chr(&m, 0), 24);
        assert_eq!(chr(&m, 4), 28);
        assert_eq!(chr(&m, 7), 31);
    }

    #[test]
    fn chr_upper_bits_applied_before_the_multiply() {
        let mut m = mk();
        m.chr_mode = 0; // 8 KB → ×8
        m.last_chr_write_sprite = true;
        m.chr_bank_sprite = [0; 8];
        m.chr_upper_bits = 1; // adds 0x100 to the *window* bank number
        // Correct: (0 | 0x100) * 8 = 0x800; NOT (0*8) | 0x100 = 0x100.
        assert_eq!(chr(&m, 0), 0x800);
        assert_eq!(chr(&m, 7), 0x807);
    }

    #[test]
    fn chr_bg_table_mirrors_across_both_pattern_tables() {
        let mut m = mk();
        m.chr_mode = 3; // 1 KB
        m.last_chr_write_sprite = false; // use the "B" set
        m.chr_bank_bg = [40, 41, 42, 43];
        assert_eq!(chr(&m, 0), 40);
        assert_eq!(chr(&m, 3), 43);
        // $1000-$1FFF mirrors the 4-entry B set (no out-of-bounds).
        assert_eq!(chr(&m, 4), 40);
        assert_eq!(chr(&m, 7), 43);
    }

    // ---- F36: scanline IRQ counter / latch model -------------------

    #[test]
    fn irq_latch_set_on_match_line_gated_by_enable() {
        let mut m = mk();
        m.irq_compare = 5;
        m.irq_enable = false;
        for _ in 0..4 {
            m.on_scanline_tick(); // counter 1..4
        }
        assert!(!m.irq_pending, "latch not set before match");
        m.on_scanline_tick(); // counter == 5 → latch
        assert!(m.irq_pending, "latch set on match");
        assert!(!m.irq_pending(), "line stays low while disabled");
        m.irq_enable = true;
        assert!(m.irq_pending(), "line asserts once enabled");
    }

    #[test]
    fn reading_5204_acknowledges_pending_irq() {
        let mut m = mk();
        m.irq_compare = 3;
        m.irq_enable = true;
        for _ in 0..3 {
            m.on_scanline_tick();
        }
        assert!(m.irq_pending());
        let status = m.prg_read_byte(0x5204);
        assert_eq!(status & 0x80, 0x80, "status reports pending");
        assert!(!m.irq_pending, "read clears the latch");
        assert!(!m.irq_pending(), "line de-asserts after ack");
    }

    #[test]
    fn peeking_5204_does_not_clear_pending() {
        let mut m = mk();
        m.irq_compare = 2;
        m.irq_enable = true;
        m.on_scanline_tick();
        m.on_scanline_tick();
        assert!(m.irq_pending);
        let _ = m.prg_peek_byte(0x5204); // side-effect-free
        assert!(m.irq_pending, "peek must not acknowledge");
    }

    #[test]
    fn in_frame_and_counter_reset_past_render_region() {
        let mut m = mk();
        m.irq_compare = 0; // never matches on the increment path
        for _ in 0..240 {
            m.on_scanline_tick();
        }
        assert_eq!(m.scanline_counter, 240);
        assert!(m.in_frame);
        assert!(!m.irq_pending, "target 0 never latches");
        m.on_scanline_tick(); // >= 240 → reset
        assert_eq!(m.scanline_counter, 0);
        assert!(!m.in_frame);
    }
}
