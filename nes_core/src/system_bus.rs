use crate::apu::Apu;
use crate::game_genie::Cheat;
use crate::input::Input;
use crate::mapper::{Mapper, MapperEnum};
use crate::memory::{Memory, Ram};
use crate::oam_dma::OamDma;
use crate::ppu::Ppu;

use std::collections::HashMap;

pub const OAMDMA_ADDRESS: u16 = 0x4014;

/// Bus-access trace gate. Env-gated via `NES_TRACE_BUS=1`, frame window
/// via `NES_TRACE_BUS_FMIN` / `NES_TRACE_BUS_FMAX`. Mirrors the nes-py
/// instrumentation (see `/tmp/nespy_instr/nes_py/nes/src/main_bus.cpp`
/// and `scripts/tracing/README.md`).
///
/// Hot-path inline: this is called per MMIO bus read/write (~71M
/// calls/run per PGO profile). Inlining lets LLVM/PGO see that
/// `enabled` is almost always `false` (env var unset in production)
/// and elide the entire trace block via branch prediction at
/// every call site, instead of paying a function-call prologue.
#[inline]
pub fn bus_trace_enabled() -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECK.get_or_init(|| std::env::var("NES_TRACE_BUS").is_ok())
}

pub fn bus_trace_frame_in_range(frame: u64) -> bool {
    static RANGE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (fmin, fmax) = *RANGE.get_or_init(|| {
        let fmin = std::env::var("NES_TRACE_BUS_FMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0u64);
        let fmax = std::env::var("NES_TRACE_BUS_FMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(u64::MAX);
        (fmin, fmax)
    });
    frame >= fmin && frame <= fmax
}

pub struct SystemBus<'a> {
    ram: &'a mut Ram,
    mapper: &'a mut MapperEnum,
    ppu: &'a mut Ppu,
    apu: &'a mut Apu,
    oam_dma: &'a mut OamDma,
    input: &'a mut Input,
    cheats: &'a HashMap<u16, Cheat>,
}

impl<'a> SystemBus<'a> {
    pub fn new(
        ram: &'a mut Ram,
        mapper: &'a mut MapperEnum,
        ppu: &'a mut Ppu,
        apu: &'a mut Apu,
        oam_dma: &'a mut OamDma,
        input: &'a mut Input,
        cheats: &'a HashMap<u16, Cheat>,
    ) -> Self {
        Self {
            ram,
            mapper,
            ppu,
            apu,
            oam_dma,
            input,
            cheats,
        }
    }

    /// Generic cycle-tick that uses the caller's real sinks. Monomorphized
    /// per (V, A) type at call site. Used by the AArch64 ASM CPU's MMIO
    /// callback via a function-pointer vtable so PPU/APU sync BEFORE a
    /// bus read/write happens with the same sinks the frame-render path
    /// uses — no lost pixels, no lost audio samples.
    pub fn tick_cpu_cycles_with<V: crate::sink::VideoSink, A: crate::sink::AudioSink>(
        &mut self,
        cpu_cycles: u32,
        video: &mut V,
        audio: &mut A,
    ) {
        for _ in 0..cpu_cycles {
            let _stall = self.apu.tick(self.mapper, audio);
            self.ppu.tick_three(self.mapper, video);
        }
    }

    /// Tick APU and PPU for `cpu_cycles` CPU cycles. Used by the
    /// AArch64 ASM CPU's MMIO callback to sync PPU timing BEFORE a
    /// register read so $2002 et al. see the correct state. Audio/
    /// video samples produced during these cycles are discarded.
    pub fn tick_cpu_cycles_discard(&mut self, cpu_cycles: u32) {
        struct V;
        impl crate::sink::VideoSink for V {
            fn write_frame(&mut self, _: &[u8]) {}
            fn frame_written(&self) -> bool { false }
            fn pixel_size(&self) -> usize { 4 }
        }
        struct A;
        impl crate::sink::AudioSink for A {
            fn write_sample(&mut self, _: f32) {}
            fn samples_written(&self) -> usize { 0 }
        }
        let mut v = V;
        let mut a = A;
        for _ in 0..cpu_cycles {
            let _stall = self.apu.tick(self.mapper, &mut a);
            self.ppu.tick_three(self.mapper, &mut v);
        }
    }

    pub fn read_byte(&mut self, addr: u16) -> u8 {
        let byte = match addr {
            // PRG ROM first: CPU instruction fetches dominate this
            // match by an order of magnitude. Putting the hottest
            // arm at the top shortens the predicted path.
            0x4020..=0xFFFF => self.mapper.prg_read_byte(addr),
            0x0000..=0x1FFF => self.ram.read_byte(addr),
            0x2000..=0x3FFF => self.ppu.read_byte(self.mapper, addr),
            0x4000..=0x4015 | 0x4018..=0x401F => self.apu.read_byte(addr),
            0x4016..=0x4017 => self.input.read_byte(addr),
        };

        let byte = if let Some(cheat) = self.cheats.get(&addr) {
            let compare = cheat.compare();
            if compare.is_none() || compare.unwrap() == byte {
                cheat.data()
            } else {
                byte
            }
        } else {
            byte
        };

        if bus_trace_enabled()
            && bus_trace_frame_in_range(self.ppu.frame)
        {
            eprintln!(
                "[BUS f={} sl={} slc={} R ${:04X} =0x{:02X}]",
                self.ppu.frame,
                self.ppu.scanline,
                self.ppu.scanline_cycle(),
                addr,
                byte,
            );
        }

        byte
    }

    // Same as read_byte, but without side effects.
    pub fn peek_byte(&self, addr: u16) -> u8 {
        match addr {
            0x4020..=0xFFFF => self.mapper.prg_peek_byte(addr),
            0x0000..=0x1FFF => self.ram.peek_byte(addr),
            0x2000..=0x3FFF => self.ppu.peek_byte(addr),
            0x4000..=0x401F => self.apu.peek_byte(addr),
        }
    }

    pub fn write_byte(&mut self, address: u16, value: u8) {
        if bus_trace_enabled()
            && bus_trace_frame_in_range(self.ppu.frame)
        {
            eprintln!(
                "[BUS f={} sl={} slc={} W ${:04X} =0x{:02X}]",
                self.ppu.frame,
                self.ppu.scanline,
                self.ppu.scanline_cycle(),
                address,
                value,
            );
        }
        if address == OAMDMA_ADDRESS {
            self.oam_dma.activate(value);
        } else if address < 0x2000 {
            self.ram.write_byte(address, value);
        } else if address < 0x4000 {
            self.ppu.write_byte(self.mapper, address, value);
        } else if address < 0x4016 || address == 0x4017 {
            self.apu.write_byte(address, value);
        } else if address < 0x4018 {
            self.input.write_byte(address, value);
        } else {
            // Mapper register write at $4018..=$FFFF. These change
            // CHR bank / mirroring / IRQ state — any of which can
            // shift rendering mid-scanline. Flag the PPU batched
            // path so it falls back to per-pixel for this scanline.
            self.ppu.note_mapper_write_during_render();
            self.mapper.prg_write_byte(address, value);
        }
    }

    pub fn read_word(&mut self, address: u16) -> u16 {
        self.read_byte(address) as u16 | ((self.read_byte(address + 1) as u16) << 8)
    }

    pub fn write_word(&mut self, address: u16, value: u16) {
        self.write_byte(address, (value >> 8) as u8);
        self.write_byte(address + 1, (value & 0xff) as u8);
    }

    pub fn ppu_scanline(&self) -> u16 {
        self.ppu.scanline
    }

    pub fn ppu_scanline_cycle(&self) -> u64 {
        self.ppu.scanline_cycle()
    }

    /// True iff the PPU has both `nmi_occurred` (vblank flag set) AND
    /// `nmi_output` (NMI enabled in PPUCTRL bit 7). Used by the ASM
    /// CPU's MMIO callback to signal a mid-batch NMI rising edge so
    /// the dispatch loop bails for Rust to service the IRQ at the
    /// next instruction boundary.
    pub fn ppu_nmi_pending(&self) -> bool {
        self.ppu.nmi_output && self.ppu.nmi_occurred
    }

    /// True iff a CPU stall is pending or in progress: either OAM DMA
    /// triggered by a write to $4014 (513-514 cycle stall) or a DMC
    /// DMA scheduled by APU (1-4 cycle stall). The ASM CPU's MMIO
    /// write callback signals this so the dispatch loop bails to the
    /// Rust slow path which knows how to handle the stall correctly.
    pub fn cpu_stall_pending(&self) -> bool {
        self.oam_dma.active
    }
}
