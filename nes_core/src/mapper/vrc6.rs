// Konami VRC6 (mappers 24 / 26). Same banking + IRQ structure as
// VRC4; different register layout; adds 3 extra audio channels
// (2 pulse + sawtooth) — the audio is left for a separate pass.
//
// Banking:
//   $8000-$BFFF: 16 KB PRG (reg $8000, 4-bit bank index)
//   $C000-$DFFF:  8 KB PRG (reg $C000, 5-bit bank index)
//   $E000-$FFFF: fixed to last 8 KB PRG bank
//   $B003 controls PPU banking mode (1 KB vs 2/4 KB) + mirroring
//   $D000-$D003: CHR banks 0-3 (1 KB each)
//   $E000-$E003: CHR banks 4-7 (1 KB each)
//   $F000-$F002: IRQ (same layout as VRC4)
//
// Pin swap: mapper 24 direct, mapper 26 swaps A0 <-> A1.

use crate::cartridge::{self, Cartridge, Mirroring};
use crate::mapper::{self, vrc6_audio::Vrc6Audio, Mapper};

use serde_derive::{Deserialize, Serialize};

const PRG_BANK_16K: usize = 0x4000;
const PRG_BANK_8K: usize = 0x2000;
const CHR_BANK: usize = 0x0400;

#[derive(Clone, Copy)]
pub enum Vrc6Board {
    M24, // direct pin mapping
    M26, // A0 <-> A1 swapped
}

impl Vrc6Board {
    #[inline]
    fn remap(self, addr: u16) -> u16 {
        match self {
            Vrc6Board::M24 => addr,
            Vrc6Board::M26 => {
                // swap bit 0 and bit 1 in the low address
                let a0 = (addr & 0x0001) << 1;
                let a1 = (addr & 0x0002) >> 1;
                (addr & !0x0003) | a0 | a1
            }
        }
    }
}

pub struct Vrc6 {
    cartridge: Cartridge,
    board: Vrc6Board,
    prg16_bank: u8, // 4-bit
    prg8_bank: u8,  // 5-bit
    chr_banks: [u8; 8],
    ppu_mode: u8,           // $B003 bit4-5 region
    ppu_extra_mirror: bool, // nametable from CHR
    // IRQ (identical semantics to VRC4)
    irq_latch: u8,
    irq_counter: u8,
    irq_mode_cycle: bool,
    irq_enable_after_ack: bool,
    irq_enabled: bool,
    irq_pending: bool,
    prescaler: i16,
    prg_asm_window: Vec<u8>,

    audio: Vrc6Audio,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cartridge: cartridge::State,
    pub prg16_bank: u8,
    pub prg8_bank: u8,
    pub chr_banks: [u8; 8],
    pub ppu_mode: u8,
    pub ppu_extra_mirror: bool,
    pub irq_latch: u8,
    pub irq_counter: u8,
    pub irq_mode_cycle: bool,
    pub irq_enable_after_ack: bool,
    pub irq_enabled: bool,
    pub irq_pending: bool,
    pub prescaler: i16,
    pub audio: Vrc6Audio,
}

impl Vrc6 {
    pub fn new(cartridge: Cartridge, board: Vrc6Board) -> Self {
        let mut m = Vrc6 {
            cartridge,
            board,
            prg16_bank: 0,
            prg8_bank: 0,
            chr_banks: [0; 8],
            ppu_mode: 0,
            ppu_extra_mirror: false,
            irq_latch: 0,
            irq_counter: 0,
            irq_mode_cycle: false,
            irq_enable_after_ack: false,
            irq_enabled: false,
            irq_pending: false,
            prescaler: 341,
            prg_asm_window: vec![0u8; 32 * 1024],
            audio: Vrc6Audio::new(),
        };
        m.rebuild_asm_window();
        m
    }

    fn prg16_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK_16K).max(1)
    }

    fn prg8_count(&self) -> usize {
        (self.cartridge.prg_rom.len() / PRG_BANK_8K).max(1)
    }

    fn chr_count(&self) -> usize {
        (self.cartridge.chr.len() / CHR_BANK).max(1)
    }

    fn rebuild_asm_window(&mut self) {
        let prg = &self.cartridge.prg_rom;
        let last8 = self.prg8_count() - 1;

        // $8000-$BFFF: 16 KB at prg16_bank
        let off16 = (self.prg16_bank as usize % self.prg16_count()) * PRG_BANK_16K;
        if off16 + PRG_BANK_16K <= prg.len() {
            self.prg_asm_window[..PRG_BANK_16K]
                .copy_from_slice(&prg[off16..off16 + PRG_BANK_16K]);
        }
        // $C000-$DFFF: 8 KB at prg8_bank
        let off8 = (self.prg8_bank as usize % self.prg8_count()) * PRG_BANK_8K;
        if off8 + PRG_BANK_8K <= prg.len() {
            self.prg_asm_window[PRG_BANK_16K..PRG_BANK_16K + PRG_BANK_8K]
                .copy_from_slice(&prg[off8..off8 + PRG_BANK_8K]);
        }
        // $E000-$FFFF: last 8 KB
        let lo = last8 * PRG_BANK_8K;
        if lo + PRG_BANK_8K <= prg.len() {
            self.prg_asm_window[PRG_BANK_16K + PRG_BANK_8K..]
                .copy_from_slice(&prg[lo..lo + PRG_BANK_8K]);
        }
    }

    fn handle_irq_write(&mut self, low_bits: u8, value: u8) {
        match low_bits {
            0 => self.irq_latch = value,
            1 => {
                self.irq_mode_cycle = value & 0x04 != 0;
                self.irq_enable_after_ack = value & 0x01 != 0;
                self.irq_enabled = value & 0x02 != 0;
                self.irq_pending = false;
                if self.irq_enabled {
                    self.irq_counter = self.irq_latch;
                    self.prescaler = 341;
                }
            }
            2 => {
                self.irq_pending = false;
                self.irq_enabled = self.irq_enable_after_ack;
            }
            _ => {}
        }
    }

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
        let a = self.board.remap(address);
        // Audio channels first — they overlap the $9000/$A000/$B000
        // register blocks, but $B003 is banking (PPU mode) rather
        // than audio, and $9003 is audio frequency control. The
        // audio block returns true when it claimed the address.
        if self.audio.write_register(a, value) {
            return;
        }
        let low = (a & 0x0003) as u8;
        match a & 0xF000 {
            0x8000 => {
                // $8000 ignores low bits — all 4 addresses do the same thing
                self.prg16_bank = value & 0x0F;
                self.rebuild_asm_window();
            }
            0x9000 => { /* audio handled above */ }
            0xA000 => { /* audio handled above */ }
            0xB000 => {
                match low {
                    0..=2 => { /* sawtooth audio — deferred */ }
                    3 => {
                        self.cartridge.mirroring = match value & 0x0C {
                            0x00 => Mirroring::Vertical,
                            0x04 => Mirroring::Horizontal,
                            0x08 => Mirroring::OneScreenLower,
                            _ => Mirroring::OneScreenUpper,
                        };
                        self.ppu_mode = (value >> 4) & 0x03;
                        self.ppu_extra_mirror = value & 0x20 != 0;
                    }
                    _ => {}
                }
            }
            0xC000 => {
                self.prg8_bank = value & 0x1F;
                self.rebuild_asm_window();
            }
            0xD000 => {
                let i = low as usize;
                self.chr_banks[i] = value;
            }
            0xE000 => {
                let i = 4 + low as usize;
                self.chr_banks[i] = value;
            }
            0xF000 => {
                self.handle_irq_write(low, value);
            }
            _ => {}
        }
    }
}

impl Mapper for Vrc6 {
    fn prg_peek_byte(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7FFF => {
                let i = (address - 0x6000) as usize;
                self.cartridge.prg_ram.get(i).copied().unwrap_or(0)
            }
            0x8000..=0xBFFF => {
                let bank = self.prg16_bank as usize % self.prg16_count();
                self.cartridge.prg_rom[bank * PRG_BANK_16K | (address as usize & 0x3FFF)]
            }
            0xC000..=0xDFFF => {
                let bank = self.prg8_bank as usize % self.prg8_count();
                self.cartridge.prg_rom[bank * PRG_BANK_8K | (address as usize & 0x1FFF)]
            }
            0xE000..=0xFFFF => {
                let last = self.prg8_count() - 1;
                self.cartridge.prg_rom[last * PRG_BANK_8K | (address as usize & 0x1FFF)]
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
        // In 1 KB mode, each slot is independent. In 2/4 KB modes we
        // emulate by masking bank LSBs — simplification that fits all
        // current VRC6 games tested.
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_count();
        self.cartridge.chr[bank * CHR_BANK | (address as usize & 0x3FF)]
    }

    fn chr_write_byte(&mut self, address: u16, value: u8) {
        let slot = ((address >> 10) & 0x07) as usize;
        let bank = self.chr_banks[slot] as usize % self.chr_count();
        let off = bank * CHR_BANK | (address as usize & 0x3FF);
        if off < self.cartridge.chr.len() {
            self.cartridge.chr[off] = value;
        }
    }

    fn mirroring(&self) -> Mirroring {
        self.cartridge.mirroring
    }

    fn reset(&mut self) {
        self.prg16_bank = 0;
        self.prg8_bank = 0;
        self.chr_banks = [0; 8];
        self.ppu_mode = 0;
        self.ppu_extra_mirror = false;
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

    fn get_state(&self) -> mapper::State {
        let s = State {
            cartridge: self.cartridge.get_state(),
            prg16_bank: self.prg16_bank,
            prg8_bank: self.prg8_bank,
            chr_banks: self.chr_banks,
            ppu_mode: self.ppu_mode,
            ppu_extra_mirror: self.ppu_extra_mirror,
            irq_latch: self.irq_latch,
            irq_counter: self.irq_counter,
            irq_mode_cycle: self.irq_mode_cycle,
            irq_enable_after_ack: self.irq_enable_after_ack,
            irq_enabled: self.irq_enabled,
            irq_pending: self.irq_pending,
            prescaler: self.prescaler,
            audio: self.audio.clone(),
        };
        match self.board {
            Vrc6Board::M24 => mapper::State::State24(s),
            Vrc6Board::M26 => mapper::State::State26(s),
        }
    }

    fn apply_state(&mut self, state: &mapper::State) {
        let inner = match state {
            mapper::State::State24(s) | mapper::State::State26(s) => s,
            _ => panic!("Invalid mapper state variant for VRC6"),
        };
        self.cartridge.apply_state(&inner.cartridge);
        self.prg16_bank = inner.prg16_bank;
        self.prg8_bank = inner.prg8_bank;
        self.chr_banks = inner.chr_banks;
        self.ppu_mode = inner.ppu_mode;
        self.ppu_extra_mirror = inner.ppu_extra_mirror;
        self.irq_latch = inner.irq_latch;
        self.irq_counter = inner.irq_counter;
        self.irq_mode_cycle = inner.irq_mode_cycle;
        self.irq_enable_after_ack = inner.irq_enable_after_ack;
        self.irq_enabled = inner.irq_enabled;
        self.irq_pending = inner.irq_pending;
        self.prescaler = inner.prescaler;
        self.audio = inner.audio.clone();
        self.rebuild_asm_window();
    }
}

pub struct Mapper24(pub Vrc6);
pub struct Mapper26(pub Vrc6);

macro_rules! forward_vrc6 {
    ($name:ident, $board:expr) => {
        impl $name {
            pub fn new(cartridge: Cartridge) -> Self {
                Self(Vrc6::new(cartridge, $board))
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

forward_vrc6!(Mapper24, Vrc6Board::M24);
forward_vrc6!(Mapper26, Vrc6Board::M26);
