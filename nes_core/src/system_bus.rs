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
    ///
    /// Returns the extra cycles burned beyond `cpu_cycles` for DMC DMA
    /// stalls: when `Apu::tick` schedules a sample-fetch stall, the
    /// interpreter path idles the CPU (`Cpu::stall`) while APU/PPU keep
    /// running, so the instruction's bus access lands that many cycles
    /// later. Extending the pre-access catch-up here reproduces that
    /// exactly; the caller must add the returned amount to the CPU
    /// cycle total (see the ASM tick wrapper in `Nes::step`).
    pub fn tick_cpu_cycles_with<V: crate::sink::VideoSink, A: crate::sink::AudioSink>(
        &mut self,
        cpu_cycles: u32,
        video: &mut V,
        audio: &mut A,
    ) -> u32 {
        let mut left = cpu_cycles;
        let mut stall_extra = 0u32;
        while left > 0 {
            let stall = self.apu.tick(self.mapper, audio);
            self.ppu.tick_three(self.mapper, video);
            left -= 1;
            if stall > 0 {
                left += stall as u32;
                stall_extra += stall as u32;
            }
        }
        stall_extra
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

    /// Advance the PPU alone by `dots` (no CPU/APU accounting). Used
    /// by hardware-true reset alignment to establish the power-on
    /// CPU↔PPU phase offset (PPU 4 dots ahead of 3×CPU).
    pub fn tick_ppu_dots_discard(&mut self, dots: u32) {
        struct V;
        impl crate::sink::VideoSink for V {
            fn write_frame(&mut self, _: &[u8]) {}
            fn frame_written(&self) -> bool { false }
            fn pixel_size(&self) -> usize { 4 }
        }
        let mut v = V;
        for _ in 0..dots {
            self.ppu.tick(self.mapper, &mut v);
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

        // Skip the SipHash HashMap probe entirely on the common path:
        // `cheats` is empty for the whole training/solver pipeline, so a
        // `get` per bus read is pure overhead. When empty, `get` would
        // return `None` and leave `byte` unchanged, so the guarded and
        // unguarded paths are behaviourally identical.
        let byte = if !self.cheats.is_empty() {
            if let Some(cheat) = self.cheats.get(&addr) {
                let compare = cheat.compare();
                if compare.is_none() || compare.unwrap() == byte {
                    cheat.data()
                } else {
                    byte
                }
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
            // Controller ports must route to the input unit, not the APU
            // (which returns 0 for anything but $4015). Mirror the
            // `read_byte` routing, but via the side-effect-free peek so
            // the controller shift register is not advanced.
            0x4016..=0x4017 => self.input.peek_byte(addr),
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
        // 6502 is little-endian: low byte at the lower address. This is
        // the inverse of `read_word` above, so a write/read round-trip
        // preserves the value.
        self.write_byte(address, (value & 0xff) as u8);
        self.write_byte(address + 1, (value >> 8) as u8);
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

    /// True iff an OAM DMA is pending or in progress (write to $4014,
    /// 513-514 cycle stall). The ASM CPU's MMIO write callback signals
    /// this so the dispatch loop bails to the Rust slow path, which
    /// runs the DMA cycle-by-cycle. DMC DMA stalls (1-4 cycles) are
    /// NOT signalled here: they surface as `Apu::tick` return values
    /// and are booked inline by the fast-path catch-up loops
    /// (`tick_cpu_cycles_with` and the remainder loops in `Nes::step`).
    pub fn cpu_stall_pending(&self) -> bool {
        self.oam_dma.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::Cartridge;
    use crate::input::BUTTON_A;
    use std::io::Cursor;

    /// Minimal NROM (mapper 0, 1×16 KB PRG, CHR-RAM). Enough to stand up
    /// a `MapperEnum` so the bus can be constructed; PRG reads back 0.
    fn nrom() -> MapperEnum {
        let mut rom = vec![0x4E, 0x45, 0x53, 0x1A]; // "NES\x1A"
        rom.extend_from_slice(&[
            0x01, // prg_lo: 1 PRG bank
            0x00, // chr_lo: 0 => CHR-RAM
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        rom.resize(16 + 16 * 1024, 0);
        let cart = Cartridge::load(&mut Cursor::new(rom)).unwrap();
        MapperEnum::from_cartridge(cart)
    }

    struct Parts {
        ram: Ram,
        mapper: MapperEnum,
        ppu: Ppu,
        apu: Apu,
        oam_dma: OamDma,
        input: Input,
        cheats: HashMap<u16, Cheat>,
    }

    impl Parts {
        fn new() -> Self {
            Self {
                ram: Ram::default(),
                mapper: nrom(),
                ppu: Ppu::new(),
                apu: Apu::new(),
                oam_dma: OamDma::new(),
                input: Input::new(),
                cheats: HashMap::new(),
            }
        }

        fn bus(&mut self) -> SystemBus<'_> {
            SystemBus::new(
                &mut self.ram,
                &mut self.mapper,
                &mut self.ppu,
                &mut self.apu,
                &mut self.oam_dma,
                &mut self.input,
                &self.cheats,
            )
        }
    }

    /// Finding 1: `write_word` must be little-endian so a write/read
    /// round-trip preserves the value. FAILS on the pre-fix big-endian
    /// implementation (0x1234 comes back as 0x3412).
    #[test]
    fn write_word_round_trips_little_endian() {
        let mut parts = Parts::new();
        let mut bus = parts.bus();
        bus.write_word(0x0010, 0x1234);
        // Low byte at the lower address.
        assert_eq!(bus.read_byte(0x0010), 0x34);
        assert_eq!(bus.read_byte(0x0011), 0x12);
        assert_eq!(bus.read_word(0x0010), 0x1234);
    }

    /// Finding 2: the empty-`cheats` guard must not change read results.
    /// Reads with an empty map return the underlying byte; an installed
    /// compare-less cheat overrides it. Identical pre- and post-fix — this
    /// pins that the perf guard is behaviour-preserving.
    #[test]
    fn cheats_guard_preserves_read_semantics() {
        // Empty map: RAM read returns what was written.
        {
            let mut parts = Parts::new();
            let mut bus = parts.bus();
            bus.write_byte(0x0020, 0x5A);
            assert_eq!(bus.read_byte(0x0020), 0x5A);
        }
        // Installed cheat (GOSSIP => addr 0xD1DD, data 0x14, no compare)
        // overrides the underlying PRG byte on read.
        {
            let mut parts = Parts::new();
            let cheat = Cheat::from_code(b"GOSSIP").unwrap();
            parts.cheats.insert(cheat.address(), cheat);
            let mut bus = parts.bus();
            assert_eq!(bus.read_byte(0xD1DD), 0x14);
        }
    }

    /// Finding 3: `peek_byte` must route $4016/$4017 to the controller,
    /// not the APU. Pre-fix, peek returns the APU's 0 while `read_byte`
    /// returns the pressed bit, so the equality FAILS.
    #[test]
    fn peek_byte_matches_read_for_controller_ports() {
        // $4016 -> pad 1.
        {
            let mut parts = Parts::new();
            parts.input.game_pad_1.set_mask(BUTTON_A);
            let mut bus = parts.bus();
            // peek must not advance the shift register, so it observes the
            // same bit the very next read returns.
            let peeked = bus.peek_byte(0x4016);
            let read = bus.read_byte(0x4016);
            assert_eq!(peeked, 1, "$4016 peek should see A pressed");
            assert_eq!(peeked, read, "peek must match read for $4016");
        }
        // $4017 -> pad 2.
        {
            let mut parts = Parts::new();
            parts.input.game_pad_2.set_mask(BUTTON_A);
            let mut bus = parts.bus();
            let peeked = bus.peek_byte(0x4017);
            let read = bus.read_byte(0x4017);
            assert_eq!(peeked, 1, "$4017 peek should see A pressed");
            assert_eq!(peeked, read, "peek must match read for $4017");
        }
    }
}
