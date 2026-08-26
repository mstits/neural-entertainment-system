mod mapper0;
mod mapper1;
mod mapper2;
mod mapper3;
pub(crate) mod mapper4;
mod mapper5;
pub mod mmc5_audio;
mod mapper7;
mod mapper9;
mod mapper10;
mod mapper11;
mod mapper13;
mod mapper19;
mod mapper34;
mod mapper37;
mod mapper40;
mod mapper41;
mod mapper47;
mod mapper64;
mod mapper66;
mod mapper68;
mod mapper69;
mod mapper71;
mod mapper79;
mod mapper85;
mod mapper105;
mod mapper113;
mod mapper228;
mod mapper232;
mod mapper234;
mod vrc24;
mod vrc6;
pub mod vrc6_audio;

use self::mapper0::Mapper0;
use self::mapper1::Mapper1;
use self::mapper2::Mapper2;
use self::mapper3::Mapper3;
use self::mapper4::{Mapper4, Mapper4Variant};
use self::mapper5::Mapper5;
use self::mapper7::Mapper7;
use self::mapper9::Mapper9;
use self::mapper10::Mapper10;
use self::mapper11::Mapper11;
use self::mapper13::Mapper13;
use self::mapper19::Mapper19;
use self::mapper34::Mapper34;
use self::mapper37::Mapper37;
use self::mapper40::Mapper40;
use self::mapper41::Mapper41;
use self::mapper47::Mapper47;
use self::mapper64::Mapper64;
use self::mapper66::Mapper66;
use self::mapper68::Mapper68;
use self::mapper69::Mapper69;
use self::mapper71::Mapper71;
use self::mapper79::Mapper79;
use self::mapper85::Mapper85;
use self::mapper105::Mapper105;
use self::mapper113::Mapper113;
use self::mapper228::Mapper228;
use self::mapper232::Mapper232;
use self::mapper234::Mapper234;
use self::vrc24::{Mapper21, Mapper22, Mapper23, Mapper25};
use self::vrc6::{Mapper24, Mapper26};
use super::cartridge::{Cartridge, Mirroring};
use super::cpu::Cpu;
use super::ppu::Ppu;

use enum_dispatch::enum_dispatch;
use serde_derive::{Deserialize, Serialize};

use std::ptr;

#[enum_dispatch(MapperEnum)]
pub trait Mapper {
    fn prg_read_byte(&mut self, address: u16) -> u8 {
        self.prg_peek_byte(address)
    }
    fn prg_write_byte(&mut self, address: u16, value: u8);
    fn chr_read_byte(&mut self, address: u16) -> u8;
    fn chr_write_byte(&mut self, address: u16, value: u8);

    fn prg_peek_byte(&self, address: u16) -> u8;

    fn mirroring(&self) -> Mirroring;

    // Called for every PPU cycle. Most mappers don't need to do anything.
    fn step(&mut self, _cpu: &mut Cpu, _ppu: &Ppu) {}

    /// Push the current CPU cycle counter to the mapper. Called once
    /// per CPU cycle by `Nes::tick`. Most mappers don't care; MMC1
    /// uses it to detect back-to-back register writes from RMW
    /// opcodes (e.g. `INC $FFFF`) and ignore the second one per the
    /// MMC1 hardware quirk.
    fn set_cpu_cycle(&mut self, _cpu_cycle: u64) {}

    /// Called by the PPU once per scanline (at the canonical MMC3
    /// "cycle 260" boundary for visible + pre-render lines). Mappers
    /// that clock an IRQ counter per scanline (MMC3, MMC5 etc.)
    /// override this. Default impl is a no-op.
    fn on_scanline_tick(&mut self) {}

    /// Whether `on_scanline_tick` does real work (clocks a scanline
    /// IRQ counter that can assert an A12-driven IRQ), as opposed to
    /// the trait's no-op default. Only consulted by the
    /// `ppu_batch_stats` diagnostic (Rung 0 of the event-driven-PPU
    /// campaign) to tell a genuine A12 scanline apart from a no-op
    /// `on_scanline_tick` firing on a mapper without a scanline IRQ
    /// (e.g. NROM/UxROM with split BG/sprite pattern tables). Never
    /// called on the release hot path — it is dead-code-eliminated
    /// unless the feature is enabled. Override to `true` in any mapper
    /// that overrides `on_scanline_tick`.
    fn uses_scanline_irq(&self) -> bool {
        false
    }

    fn sram(&mut self) -> *mut u8 {
        ptr::null_mut()
    }

    fn sram_size(&self) -> usize {
        0
    }

    /// Raw pointer to a flat 32 KB PRG ROM region that's valid as long
    /// as the mapper is alive and its banking is static. Returns
    /// `None` for mappers that rebank at runtime (MMC1, MMC3, …).
    /// Only used by the AArch64 ASM CPU fast path.
    fn prg_asm_ptr(&self) -> Option<*const u8> {
        None
    }

    /// Raw pointer to a flat 8 KB CHR region (ROM or RAM) that's
    /// valid as long as the mapper's CHR banking is static. Returns
    /// `None` for mappers with runtime CHR bank switching (MMC1,
    /// MMC3, …) where the active 8 KB window changes.
    ///
    /// Used by `Ppu::read_chr_byte` to bypass the per-fetch
    /// `MapperEnum` virtual dispatch on PPU CHR reads (~5M reads/
    /// sec/worker for typical games). PPU caches the pointer in
    /// `chr_cache_ptr` at scanline boundary; static-CHR mappers
    /// (NROM, etc.) can use it for the entire scanline's BG +
    /// sprite fetches.
    fn chr_static_ptr(&self) -> Option<*const u8> {
        None
    }

    /// Preferred ASM bulk-step budget in CPU cycles. Mappers whose
    /// banking / MMIO-timing is simple (NROM) can safely run multiple
    /// instructions per ASM call, amortizing setup overhead. Mappers
    /// with runtime bank switching or tight PPU-timing coupling
    /// should return 1 for correctness.
    fn asm_bulk_cycles(&self) -> i64 {
        1
    }

    /// Opt-in runtime override for `asm_bulk_cycles`. Default no-op:
    /// most mappers either already run at their validated ceiling
    /// (NROM = 64) or must stay at 1 for correctness (e.g. MMC3,
    /// whose scanline IRQ can assert mid-batch — not an ASM exit
    /// condition, so batching would delay IRQ service). Only mappers
    /// whose batching is structurally batch-safe implement this:
    /// MMC1 (`Mapper1`) and UxROM (`Mapper2`) — no mapper IRQ line,
    /// bank-switch writes rebuild the ASM window in place. The
    /// default budget stays 1 so emulation timing on default
    /// settings is unchanged; the trainer raises it per-game only
    /// after the lockstep + Mesen-oracle + parity gate passes.
    fn set_asm_bulk_cycles_override(&mut self, _cycles: i64) {}

    fn reset(&mut self);
    fn get_state(&self) -> State;
    fn apply_state(&mut self, state: &State);

    fn irq_pending(&self) -> bool {
        false
    }

    /// Advance mapper-internal audio channels by one CPU cycle.
    /// Mappers with extra sound hardware (MMC5 pulses + PCM, VRC6
    /// pulses + sawtooth, VRC7 FM, N163, FME-7 5B) override this.
    /// Called from `Apu::tick` once per cycle; default is no-op so
    /// mappers without extra audio pay nothing.
    ///
    /// `#[inline(always)]` so that for variants that DON'T override
    /// (NROM, MMC1, MMC3, etc — the vast majority), the
    /// MapperEnum dispatch fold completely to no-ops at the call
    /// site. PGO showed this default no-op was being hit ~3.7B
    /// times/run via per-cycle Apu::tick.
    #[inline(always)]
    fn tick_audio(&mut self) {}

    /// Current audio output sample from the mapper's extra channels,
    /// normalized to [0.0, 1.0]. Summed into APU output in
    /// `Apu::generate_sample`. Default 0.0.
    #[inline(always)]
    fn audio_mix(&self) -> f32 {
        0.0
    }
}

#[enum_dispatch]
pub enum MapperEnum {
    Mapper0,
    Mapper1,
    Mapper2,
    Mapper3,
    Mapper4,
    Mapper5,
    Mapper7,
    Mapper9,
    Mapper10,
    Mapper11,
    Mapper13,
    Mapper19,
    Mapper21,
    Mapper22,
    Mapper23,
    Mapper24,
    Mapper25,
    Mapper26,
    Mapper34,
    Mapper37,
    Mapper40,
    Mapper41,
    Mapper47,
    Mapper64,
    Mapper66,
    Mapper68,
    Mapper69,
    Mapper71,
    Mapper79,
    Mapper85,
    Mapper105,
    Mapper113,
    Mapper228,
    Mapper232,
    Mapper234,
}

impl MapperEnum {
    pub fn from_cartridge(cartridge: Cartridge) -> Result<Self, String> {
        let mapper_num = cartridge.mapper;
        Ok(match cartridge.mapper {
            0 => Mapper0::new(cartridge).into(),
            1 => Mapper1::new(cartridge).into(),
            2 => Mapper2::new(cartridge).into(),
            3 => Mapper3::new(cartridge).into(),
            4 => Mapper4::new(cartridge).into(),
            5 => Mapper5::new(cartridge).into(),
            7 => Mapper7::new(cartridge).into(),
            9 => Mapper9::new(cartridge).into(),
            10 => Mapper10::new(cartridge).into(),
            11 => Mapper11::new(cartridge).into(),
            13 => Mapper13::new(cartridge).into(),
            19 => Mapper19::new(cartridge).into(),
            21 => Mapper21::new(cartridge).into(),
            22 => Mapper22::new(cartridge).into(),
            23 => Mapper23::new(cartridge).into(),
            24 => Mapper24::new(cartridge).into(),
            25 => Mapper25::new(cartridge).into(),
            26 => Mapper26::new(cartridge).into(),
            34 => Mapper34::new(cartridge).into(),
            37 => Mapper37::new(cartridge).into(),
            40 => Mapper40::new(cartridge).into(),
            41 => Mapper41::new(cartridge).into(),
            47 => Mapper47::new(cartridge).into(),
            64 => Mapper64::new(cartridge).into(),
            66 => Mapper66::new(cartridge).into(),
            68 => Mapper68::new(cartridge).into(),
            69 => Mapper69::new(cartridge).into(),
            71 => Mapper71::new(cartridge).into(),
            79 => Mapper79::new(cartridge).into(),
            85 => Mapper85::new(cartridge).into(),
            105 => Mapper105::new(cartridge).into(),
            113 => Mapper113::new(cartridge).into(),
            118 => Mapper4::new_with_variant(cartridge, Mapper4Variant::TxSROM).into(),
            119 => Mapper4::new_with_variant(cartridge, Mapper4Variant::TQROM).into(),
            228 => Mapper228::new(cartridge).into(),
            232 => Mapper232::new(cartridge).into(),
            234 => Mapper234::new(cartridge).into(),
            _ => return Err(format!("Unsupported mapper number: {}", mapper_num)),
        })
    }
}

#[derive(Deserialize, Serialize)]
pub enum State {
    State0(mapper0::State),
    State1(mapper1::State),
    State2(mapper2::State),
    State3(mapper3::State),
    State4(mapper4::State),
    State5(mapper5::State),
    State7(mapper7::State),
    State9(mapper9::State),
    State10(mapper10::State),
    State11(mapper11::State),
    State13(mapper13::State),
    State19(mapper19::State),
    State21(vrc24::State),
    State22(vrc24::State),
    State23(vrc24::State),
    State24(vrc6::State),
    State25(vrc24::State),
    State26(vrc6::State),
    State34(mapper34::State),
    State37(mapper37::State),
    State40(mapper40::State),
    State41(mapper41::State),
    State47(mapper47::State),
    State64(mapper64::State),
    State66(mapper66::State),
    State68(mapper68::State),
    State69(mapper69::State),
    State71(mapper71::State),
    State79(mapper79::State),
    State85(mapper85::State),
    State105(mapper105::State),
    State113(mapper113::State),
    State228(mapper228::State),
    State232(mapper232::State),
    State234(mapper234::State),
}
