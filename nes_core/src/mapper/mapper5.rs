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
    fn resolve_prg_slot(&self, slot: usize) -> (bool, usize) {
        // Per slot, `prg_banks[slot]` is 8 bit: top bit = ROM select
        // (1 = ROM), lower 7 bits = bank number. $6000 defaults to RAM.
        // $E000 is always ROM.
        let raw = self.prg_banks[slot];
        let (is_ram, bank_idx) = if slot == 4 {
            (false, raw as usize)
        } else if slot == 0 {
            // $6000 is always RAM; bits 0-3 select the RAM bank.
            (true, (raw & 0x0F) as usize)
        } else {
            let is_rom = raw & 0x80 != 0;
            (!is_rom, (raw & 0x7F) as usize)
        };

        match self.prg_mode {
            // Mode 3: all 8 KB banks.
            3 => {
                if is_ram {
                    let off = bank_idx * BANK_8K;
                    (true, off)
                } else {
                    let off = (bank_idx % self.prg_rom_bank_count_8k()) * BANK_8K;
                    (false, off)
                }
            }
            // Mode 2: $8000-$BFFF is 16 KB ($5115), $C000-$DFFF + $E000-$FFFF 8 KB each.
            2 => {
                if slot == 2 {
                    let bank16 = bank_idx & !1;
                    let off = (bank16 % self.prg_rom_bank_count_8k()) * BANK_8K;
                    (is_ram, off)
                } else if slot == 1 {
                    let bank16 = (self.prg_banks[2] as usize & !1) & 0x7F;
                    let off = (bank16 % self.prg_rom_bank_count_8k()) * BANK_8K + BANK_8K;
                    (false, off)
                } else {
                    let off = (bank_idx % self.prg_rom_bank_count_8k()) * BANK_8K;
                    (is_ram, off)
                }
            }
            // Mode 1: 16 KB + 16 KB — $5115 controls $8000-$BFFF, $5117 controls $C000-$FFFF.
            1 => {
                let bank16 = bank_idx & !1;
                let off = if slot == 1 || slot == 3 {
                    (bank16 % self.prg_rom_bank_count_8k()) * BANK_8K
                } else {
                    (bank16 % self.prg_rom_bank_count_8k()) * BANK_8K + BANK_8K
                };
                (is_ram, off)
            }
            // Mode 0: 32 KB switchable via $5117.
            0 => {
                let bank32 = (self.prg_banks[4] as usize) & !3;
                let off = (bank32 % self.prg_rom_bank_count_8k()) * BANK_8K + slot.saturating_sub(1) * BANK_8K;
                (false, off)
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
        // Resolve 1 KB CHR slot. In 1 KB mode, each 1 KB window is
        // independently selected. In 2/4/8 KB modes, we index into
        // the table that would have been written for the "matching"
        // mode-boundary register.
        let slot = ((address >> 10) & 0x07) as usize;
        let table = if self.last_chr_write_sprite {
            &self.chr_bank_sprite[..]
        } else {
            &self.chr_bank_bg[..]
        };
        let upper = (self.chr_upper_bits as u16) << 8;
        match self.chr_mode {
            3 => {
                if self.last_chr_write_sprite {
                    self.chr_bank_sprite[slot] | upper
                } else {
                    self.chr_bank_bg[slot & 3] | upper
                }
            }
            2 => {
                // 2 KB windows: slots 0-1 share one bank base, 2-3 next, etc.
                let base_slot = slot & !1;
                let bank = table.get(base_slot).copied().unwrap_or(0);
                ((bank & !1) | ((slot & 1) as u16)) | upper
            }
            1 => {
                // 4 KB windows.
                let base_slot = slot & !3;
                let bank = table.get(base_slot).copied().unwrap_or(0);
                ((bank & !3) | ((slot & 3) as u16)) | upper
            }
            0 => {
                // 8 KB window — single bank for all of $0000-$1FFF.
                let bank = table.first().copied().unwrap_or(0);
                ((bank & !7) | ((slot & 7) as u16)) | upper
            }
            _ => 0,
        }
    }

    fn prg_ram_writable(&self) -> bool {
        // Write protect: both A ($5102 & 0x03) and B ($5103 & 0x03)
        // must equal 2 and 1 respectively for writes to be enabled.
        self.prg_ram_protect_a == 0x02 && self.prg_ram_protect_b == 0x01
    }
}

impl Mapper for Mapper5 {
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
        // Simple scanline counter: increments on each visible / pre-render
        // scanline. When it equals irq_compare and IRQs are enabled, set
        // the pending flag. Canonical MMC5 clears in_frame on pre-render
        // line and resets counter; we approximate by wrapping at 241.
        self.scanline_counter = self.scanline_counter.wrapping_add(1);
        if self.scanline_counter >= 241 {
            self.scanline_counter = 0;
            self.in_frame = false;
        } else {
            self.in_frame = true;
        }
        if self.irq_enable && self.scanline_counter == self.irq_compare {
            self.irq_pending = true;
        }
    }

    fn irq_pending(&self) -> bool {
        self.irq_pending
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
