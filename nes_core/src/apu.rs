use crate::cpu::CPU_FREQUENCY;
use crate::mapper::{Mapper, MapperEnum};
use crate::memory::Memory;
use crate::sink::*;

use once_cell::sync::Lazy;
use serde_derive::{Deserialize, Serialize};

pub const CPU_CYCLES_PER_SAMPLE: u64 = 41;
pub const SAMPLE_RATE: u32 = (CPU_FREQUENCY / CPU_CYCLES_PER_SAMPLE) as u32;

static DUTY_CYCLE_TABLE: &[[u8; 8]] = &[
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 1],
];

#[rustfmt::skip]
static LENGTH_TABLE: &[u8] = &[
    10, 254, 20,  2, 40,  4, 80,  6, 160,  8, 60, 10, 14, 12, 26, 14,
    12,  16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
];

#[rustfmt::skip]
static TRIANGLE_TABLE: &[u8] = &[
    15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
];

static NOISE_TABLE: &[u16] = &[
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
];

static DMC_TABLE: &[u8] = &[
    214, 190, 170, 160, 143, 127, 113, 107, 95, 80, 71, 64, 53, 42, 36, 27,
];

static PULSE_TABLE: Lazy<[f32; 31]> = Lazy::new(|| {
    let mut pulse_table = [0f32; 31];
    pulse_table
        .iter_mut()
        .enumerate()
        .for_each(|(n, val)| *val = (95.88 / (8128.0 / (n as f64) + 100.0)) as f32);
    pulse_table
});

static TND_TABLE: Lazy<[f32; 203]> = Lazy::new(|| {
    let mut tnd_table = [0f32; 203];
    tnd_table
        .iter_mut()
        .enumerate()
        .for_each(|(n, val)| *val = (163.67 / (24329.0 / (n as f64) + 100.0)) as f32);
    tnd_table
});

pub struct Apu {
    cycles: u64,

    last_sampled_cycles: u64,

    pulse_1: Pulse,
    pulse_2: Pulse,
    triangle: Triangle,
    noise: Noise,
    dmc: Dmc,
    frame_counter: FrameCounter,

    // `Send` bound is required so the parent `Nes` (and the PyO3
    // `NESEnvironment` wrapping it) implement `Send` and can live on
    // a `#[pyclass]`. `Filter` impls in this module are stack-only
    // numerics, so the bound is satisfied at zero cost.
    filter: Box<dyn Filter + Send>,

    pub settings: Settings,

    // When false, `tick` skips sample generation + filter + sink
    // entirely; timers / frame counter still tick, preserving IRQ +
    // DMC integrity. Default true. Flipped off on the 15 non-audio
    // training workers in `Pool` so we don't mix 16 channels of
    // unused samples at 43 KHz each.
    sample_output_enabled: bool,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub cycles: u64,
    pub last_sampled_cycles: u64,
    pub pulse_1: Pulse,
    pub pulse_2: Pulse,
    pub triangle: Triangle,
    pub noise: Noise,
    pub dmc: Dmc,
    pub frame_counter: FrameCounter,
}

impl Apu {
    pub fn new() -> Apu {
        Apu {
            cycles: 0,
            last_sampled_cycles: 0,
            pulse_1: Pulse::new(SweepNegationType::OnesComplement),
            pulse_2: Pulse::new(SweepNegationType::TwosComplement),
            triangle: Triangle::new(),
            noise: Noise::new(),
            dmc: Dmc::new(),
            frame_counter: FrameCounter::new(),
            filter: Box::new(
                LowPassFilter::new(0.815_686)
                    .chain(HighPassFilter::new(0.996_039))
                    .chain(HighPassFilter::new(0.999_835)),
            ),
            settings: Settings {
                pulse_1_enabled: true,
                pulse_2_enabled: true,
                triangle_enabled: true,
                noise_enabled: true,
                dmc_enabled: true,
                filter_enabled: true,
            },
            sample_output_enabled: true,
        }
    }

    /// Toggle whether `tick` generates output samples. When disabled,
    /// channel timers and the frame-counter still advance (IRQ + DMC
    /// integrity preserved), but the mixer/filter/sink path is
    /// skipped entirely. Training pools flip this off for the 15
    /// non-audio workers to avoid ~43 KHz × 15 wasted sink calls/sec.
    pub fn set_sample_output_enabled(&mut self, enabled: bool) {
        self.sample_output_enabled = enabled;
    }

    pub fn peek_byte(&self, address: u16) -> u8 {
        if address == 0x4015 {
            self.read_status()
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.cycles = 0;
        self.pulse_1 = Pulse::new(SweepNegationType::OnesComplement);
        self.pulse_2 = Pulse::new(SweepNegationType::TwosComplement);
        self.triangle = Triangle::new();
        self.noise = Noise::new();
        self.dmc = Dmc::new();
        self.frame_counter = FrameCounter::new();
    }

    pub fn get_state(&self) -> State {
        State {
            cycles: self.cycles,
            last_sampled_cycles: self.last_sampled_cycles,
            pulse_1: self.pulse_1.clone(),
            pulse_2: self.pulse_2.clone(),
            triangle: self.triangle.clone(),
            noise: self.noise.clone(),
            dmc: self.dmc.clone(),
            frame_counter: self.frame_counter,
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.cycles = state.cycles;
        self.last_sampled_cycles = state.last_sampled_cycles;
        self.pulse_1 = state.pulse_1.clone();
        self.pulse_2 = state.pulse_2.clone();
        self.triangle = state.triangle.clone();
        self.noise = state.noise.clone();
        self.dmc = state.dmc.clone();
        self.frame_counter = state.frame_counter;
    }

    // Tried `tick_n(n)` as a bulk-advance helper for the nes::Nes
    // block-interpreter path. Measured -55% regression on canonical
    // bench (1600 → 730 sps) — presumably due to PGO's inlining
    // decisions for `tick` not applying to a wrapper call, or the
    // max-stall tracking adding a branch. Reverted.

    /// Run one APU cycle. There is one APU cycle per CPU cycle.
    /// Returns the number of CPU cycles to stall.
    #[inline(always)]
    pub fn tick<A: AudioSink>(&mut self, mapper: &mut MapperEnum, audio_frame_sink: &mut A) -> u8 {
        self.cycles += 1;

        // Advance any mapper-internal audio hardware (MMC5 pulses +
        // PCM, VRC6 channels, etc.). Gated on sample output so the
        // 15 non-audio training workers skip the per-cycle virtual
        // dispatch entirely — measured ~10% scaling cost on Zelda
        // when this fires unconditionally. Mappers that expose
        // audio-channel status to CPU reads ($5015 on MMC5 etc.)
        // may report stale length counters while audio is muted,
        // but training workloads don't poll those registers.
        if self.sample_output_enabled {
            mapper.tick_audio();
        }

        let cpu_stall_cycles = self.step_timer(mapper);

        if self.cycles.is_multiple_of(2) {
            if self.frame_counter.divider_count == 0 {
                self.frame_counter.divider_count = FrameCounter::DIVIDER_COUNT_RELOAD_VALUE;
                self.step_frame_counter();
            } else {
                self.frame_counter.divider_count -= 1;
            }
        }

        if self.cycles > self.last_sampled_cycles + CPU_CYCLES_PER_SAMPLE {
            self.last_sampled_cycles += CPU_CYCLES_PER_SAMPLE;
            if self.sample_output_enabled {
                let mut sample = self.generate_sample();
                // Mapper-side audio (MMC5 / VRC6 / etc.) summed in
                // AFTER the main APU mix so mappers without extras
                // contribute 0 and don't shift the filter baseline.
                // Safe to call unconditionally — default is 0.0 and
                // we've already paid the virtual dispatch above.
                sample += mapper.audio_mix();
                if sample > 1.0 {
                    sample = 1.0;
                }
                if self.settings.filter_enabled {
                    sample = self.filter.step(sample);
                }
                audio_frame_sink.write_sample(sample);
            }
        }

        cpu_stall_cycles
    }

    #[inline(always)]
    fn generate_sample(&mut self) -> f32 {
        let pulse_1 = if self.settings.pulse_1_enabled {
            self.pulse_1.output()
        } else {
            0
        };
        let pulse_2 = if self.settings.pulse_2_enabled {
            self.pulse_2.output()
        } else {
            0
        };
        let triangle = if self.settings.triangle_enabled {
            self.triangle.output()
        } else {
            0
        };
        let noise = if self.settings.noise_enabled {
            self.noise.output()
        } else {
            0
        };
        let dmc = if self.settings.dmc_enabled {
            self.dmc.output()
        } else {
            0
        };

        let pulse_out = PULSE_TABLE[pulse_1 as usize + pulse_2 as usize];
        let tnd_out = TND_TABLE[3 * triangle as usize + 2 * noise as usize + dmc as usize];

        pulse_out + tnd_out
    }

    /// Per-channel float mix contributions in the order
    /// `[pulse1, pulse2, triangle, noise, dmc]`.
    ///
    /// The NES APU's hardware mix is non-linear (two shared lookup
    /// tables: `PULSE_TABLE[p1+p2]` and `TND_TABLE[3*tri+2*noise+dmc]`),
    /// so the five channels can't be split cleanly after the fact.
    /// We approximate a clean split by evaluating each lookup with the
    /// OTHER channel(s) forced to zero, which is how every audible-
    /// stereo NES mod (Famitracker panning, Mesen-S, etc.) handles it.
    ///
    /// Mathematical properties:
    ///   * `pulse1_f + pulse2_f >= PULSE_TABLE[p1+p2]` (tables are
    ///     concave, so the split gives a slightly hotter sum).
    ///   * `tri_f + noise_f + dmc_f >= TND_TABLE[3t+2n+d]` for the
    ///     same reason. We rescale each triad component by its hardware
    ///     weight (`3/6`, `2/6`, `1/6`) so the sum approximates the
    ///     real curve.
    ///
    /// The per-channel values are meant for panned mixing at playback
    /// time — the caller sums them back (after applying L/R gains) and
    /// clips/filters downstream. Mono-mixing the returned array
    /// produces a waveform very close to `generate_sample()` but not
    /// bit-identical; use `generate_sample()` when exact hardware-match
    /// behavior is required (tests, the non-audio training workers).
    #[allow(dead_code)]
    #[inline(always)]
    pub fn generate_sample_channels(&mut self) -> [f32; 5] {
        let pulse_1 = if self.settings.pulse_1_enabled {
            self.pulse_1.output()
        } else {
            0
        };
        let pulse_2 = if self.settings.pulse_2_enabled {
            self.pulse_2.output()
        } else {
            0
        };
        let triangle = if self.settings.triangle_enabled {
            self.triangle.output()
        } else {
            0
        };
        let noise = if self.settings.noise_enabled {
            self.noise.output()
        } else {
            0
        };
        let dmc = if self.settings.dmc_enabled {
            self.dmc.output()
        } else {
            0
        };

        // Pulse: evaluate the lookup with just that channel present,
        // so each pulse contributes its own value to the mix. At full
        // solo, `PULSE_TABLE[15]` ≈ 0.164. Their sum matches the
        // hardware sum when one pulse is silent (the dominant case in
        // most NES music).
        let p1 = PULSE_TABLE[pulse_1 as usize];
        let p2 = PULSE_TABLE[pulse_2 as usize];

        // TND is weighted 3:2:1 in the lookup index. Split the same
        // way: evaluate each channel's contribution with the others
        // zeroed AND scale by (weight / 6) so the three channels sum
        // into the original non-linear envelope. Prevents the "all
        // three full" case from blowing past the table's max of
        // ~0.44 when naively summed.
        let tri = TND_TABLE[3 * triangle as usize] * (3.0 / 6.0);
        let noi = TND_TABLE[2 * noise as usize] * (2.0 / 6.0);
        let dmc_f = TND_TABLE[dmc as usize] * (1.0 / 6.0);

        [p1, p2, tri, noi, dmc_f]
    }

    fn step_frame_counter(&mut self) {
        // Four Step  Five Step    Function
        // ---------  -----------  -----------------------------
        // - - - f    - - - - -    IRQ (if bit 6 is clear)
        // - l - l    l - l - -    Length counter and sweep
        // e e e e    e e e e -    Envelope and linear counter
        match self.frame_counter.mode {
            FrameCounterMode::FourStep => {
                match self.frame_counter.sequence_frame {
                    0 => {
                        self.step_envelope_and_linear_counter();
                    }
                    1 => {
                        self.step_length_counter();
                        self.step_sweep();
                        self.step_envelope_and_linear_counter();
                    }
                    2 => {
                        self.step_envelope_and_linear_counter();
                    }
                    3 => {
                        if !self.frame_counter.interrupt_inhibit_flag {
                            self.frame_counter.irq_pending = true;
                        }
                        self.step_length_counter();
                        self.step_sweep();
                        self.step_envelope_and_linear_counter();
                    }
                    _ => (),
                }
                self.frame_counter.sequence_frame = (self.frame_counter.sequence_frame + 1) % 4;
            }
            FrameCounterMode::FiveStep => {
                match self.frame_counter.sequence_frame {
                    0 | 2 => {
                        self.step_length_counter();
                        self.step_sweep();
                        self.step_envelope_and_linear_counter();
                    }
                    1 | 3 => {
                        self.step_envelope_and_linear_counter();
                    }
                    _ => (),
                }
                self.frame_counter.sequence_frame = (self.frame_counter.sequence_frame + 1) % 5;
            }
        }
    }

    fn step_length_counter(&mut self) {
        self.pulse_1.length_counter.step();
        self.pulse_2.length_counter.step();
        self.triangle.length_counter.step();
        self.noise.length_counter.step();
    }

    fn step_sweep(&mut self) {
        self.pulse_1.step_sweep();
        self.pulse_2.step_sweep();
    }

    fn step_envelope_and_linear_counter(&mut self) {
        self.pulse_1.envelope.step();
        self.pulse_2.envelope.step();
        self.triangle.step_linear_counter();
        self.noise.envelope.step();
    }

    #[inline(always)]
    fn step_timer(&mut self, mapper: &mut MapperEnum) -> u8 {
        let mut cpu_stall_cycles = 0;
        if self.cycles.is_multiple_of(2) {
            self.pulse_1.step_timer();
            self.pulse_2.step_timer();
            self.noise.step_timer();
            cpu_stall_cycles = self.dmc.step_timer(mapper);
        }

        self.triangle.step_timer();

        cpu_stall_cycles
    }

    fn read_status(&self) -> u8 {
        let mut status = 0x00;

        if self.pulse_1.length_counter.count > 0 {
            status |= 0x01;
        }

        if self.pulse_2.length_counter.count > 0 {
            status |= 0x02;
        }

        if self.triangle.length_counter.count > 0 {
            status |= 0x04;
        }

        if self.noise.length_counter.count > 0 {
            status |= 0x08;
        }

        if self.dmc.current_length > 0 {
            status |= 0x10;
        }

        if self.frame_counter.irq_pending {
            status |= 0x40;
        }

        // $4015 bit 7 must reflect the asserted DMC interrupt, not
        // the IRQ-enable bit set by $4010. Reading irq_flag here made
        // bit 7 read high any time DMC IRQ was enabled (regardless of
        // whether the sample had actually completed), causing games
        // that probe $4015 for IRQ source — Bill & Ted's, sword
        // pickup in some titles, Crash 'n' the Boys — to mis-route
        // their interrupt service routines.
        if self.dmc.irq_pending {
            status |= 0x80;
        }

        status
    }

    fn write_status(&mut self, value: u8) {
        self.dmc.irq_pending = false;

        self.pulse_1.enabled = (value & 0x01) != 0;
        if !self.pulse_1.enabled {
            self.pulse_1.length_counter.reset();
        }

        self.pulse_2.enabled = (value & 0x02) != 0;
        if !self.pulse_2.enabled {
            self.pulse_2.length_counter.reset();
        }

        self.triangle.enabled = (value & 0x04) != 0;
        if !self.triangle.enabled {
            self.triangle.length_counter.reset();
        }

        self.noise.enabled = (value & 0x08) != 0;
        if !self.noise.enabled {
            self.noise.length_counter.reset();
        }

        self.dmc.enable_flag = (value & 0x10) != 0;
        if !self.dmc.enable_flag {
            self.dmc.current_length = 0;
        } else if self.dmc.current_length == 0 {
            self.dmc.restart();
        }
    }

    fn write_frame_counter(&mut self, value: u8) {
        self.frame_counter.mode = if value & 0x80 == 0 {
            FrameCounterMode::FourStep
        } else {
            FrameCounterMode::FiveStep
        };

        self.frame_counter.sequence_frame = 0;
        self.frame_counter.divider_count = FrameCounter::DIVIDER_COUNT_RELOAD_VALUE;

        self.frame_counter.interrupt_inhibit_flag = value & 0x40 != 0;
        // Hardware clears the frame-interrupt flag only when the
        // inhibit bit (bit 6) is set; a write with bit 6 clear leaves
        // a pending frame IRQ asserted.
        if self.frame_counter.interrupt_inhibit_flag {
            self.frame_counter.irq_pending = false;
        }

        if self.frame_counter.mode == FrameCounterMode::FiveStep {
            self.step_length_counter();
            self.step_sweep();
            self.step_envelope_and_linear_counter();
        }
    }

    pub fn irq_pending(&self) -> bool {
        self.frame_counter.irq_pending || self.dmc.irq_pending
    }
}

impl Default for Apu {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for Apu {
    fn read_byte(&mut self, address: u16) -> u8 {
        if address == 0x4015 {
            // Build the status byte FIRST (bit 6 reflects the pending
            // frame IRQ), then clear the frame IRQ flag as the read
            // side effect. Clearing before reading made bit 6 read 0.
            let status = self.read_status();
            self.frame_counter.irq_pending = false;
            status
        } else {
            0
        }
    }

    fn write_byte(&mut self, address: u16, value: u8) {
        match address {
            0x4000 => self.pulse_1.write_control(value),
            0x4001 => self.pulse_1.write_sweep(value),
            0x4002 => self.pulse_1.write_timer_lo(value),
            0x4003 => self.pulse_1.write_timer_hi(value),
            0x4004 => self.pulse_2.write_control(value),
            0x4005 => self.pulse_2.write_sweep(value),
            0x4006 => self.pulse_2.write_timer_lo(value),
            0x4007 => self.pulse_2.write_timer_hi(value),
            0x4008 => self.triangle.write_linear_counter(value),
            0x400A => self.triangle.write_timer_lo(value),
            0x400B => self.triangle.write_length_counter_and_timer_hi(value),
            0x400C => self.noise.write_control(value),
            0x400E => self.noise.write_mode_and_timer_period(value),
            0x400F => self.noise.write_length_counter_and_envelope_restart(value),
            0x4010 => self.dmc.write_control(value),
            0x4011 => self.dmc.write_value(value),
            0x4012 => self.dmc.write_sample_address(value),
            0x4013 => self.dmc.write_sample_length(value),
            0x4015 => self.write_status(value),
            0x4017 => self.write_frame_counter(value),
            _ => (),
        }
    }
}

pub struct Settings {
    pub pulse_1_enabled: bool,
    pub pulse_2_enabled: bool,
    pub triangle_enabled: bool,
    pub noise_enabled: bool,
    pub dmc_enabled: bool,
    pub filter_enabled: bool,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
enum SweepNegationType {
    OnesComplement,
    TwosComplement,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
struct Envelope {
    enabled: bool,
    start: bool,
    loop_flag: bool,
    volume: u8,
    value: u8,
    period: u8,
}

impl Envelope {
    fn new() -> Envelope {
        Envelope {
            enabled: false,
            start: false,
            loop_flag: false,
            volume: 0,
            value: 0,
            period: 0,
        }
    }

    fn step(&mut self) {
        if self.start {
            self.start = false;
            self.volume = 15;
            self.value = self.period;
        } else if self.value > 0 {
            self.value -= 1;
        } else {
            self.value = self.period;

            if self.volume > 0 {
                self.volume -= 1;
            } else if self.loop_flag {
                self.volume = 15;
            }
        }
    }
}

#[derive(Copy, Clone, Deserialize, Serialize)]
struct Sweep {
    enabled: bool,
    negate: bool,
    reload: bool,
    divider: u8,
    period: u8,
    shift_count: u8,
}

impl Sweep {
    fn new() -> Sweep {
        Sweep {
            enabled: false,
            negate: false,
            reload: false,
            divider: 0,
            period: 0,
            shift_count: 0,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct LengthCounter {
    enabled: bool,
    count: u8,
}

impl LengthCounter {
    fn new() -> LengthCounter {
        LengthCounter {
            enabled: true,
            count: 0,
        }
    }

    fn step(&mut self) {
        if self.enabled && self.count > 0 {
            self.count -= 1;
        }
    }

    fn set(&mut self, value: u8) {
        self.count = LENGTH_TABLE[value as usize];
    }

    fn reset(&mut self) {
        self.count = 0;
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Pulse {
    enabled: bool,
    negation_type: SweepNegationType,
    timer_value: u16,
    timer_period: u16,
    duty_mode: u8,
    duty_cycle: u8,
    length_counter: LengthCounter,
    envelope: Envelope,
    sweep: Sweep,
    constant_volume: u8,
}

impl Pulse {
    fn new(negation_type: SweepNegationType) -> Pulse {
        Pulse {
            enabled: false,
            negation_type,
            timer_value: 0,
            timer_period: 0,
            duty_mode: 0,
            duty_cycle: 0,
            length_counter: LengthCounter::new(),
            envelope: Envelope::new(),
            sweep: Sweep::new(),
            constant_volume: 0,
        }
    }

    fn write_control(&mut self, value: u8) {
        // Duty cycle is the TOP TWO bits of $4000/$4004 (D7-D6) — a
        // 2-bit selector into DUTY_CYCLE_TABLE's 4 entries (12.5%,
        // 25%, 50%, 75%-inverted). The previous `value >> 7` captured
        // only the high bit, collapsing modes {0,1} → 0 and {2,3} → 1
        // so half the duty waveforms were never produced.
        self.duty_mode = (value >> 6) & 0x03;
        self.length_counter.enabled = (value & 0x20) == 0;
        self.envelope.loop_flag = !self.length_counter.enabled;
        self.envelope.enabled = (value & 0x10) == 0;
        self.constant_volume = value & 0x0F;
        self.envelope.period = self.constant_volume;
        self.envelope.start = true;
    }

    fn write_sweep(&mut self, value: u8) {
        self.sweep.enabled = (value & 0x80) != 0;
        self.sweep.period = ((value >> 4) & 0x07) + 1;
        self.sweep.negate = (value & 0x08) != 0;
        self.sweep.shift_count = value & 0x07;
        self.sweep.reload = true;
    }

    fn write_timer_lo(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0xFF00) | (value as u16);
    }

    fn write_timer_hi(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x00FF) | (((value & 0x07) as u16) << 8);
        if self.enabled {
            self.length_counter.set(value >> 3);
        }
        self.envelope.start = true;
        self.duty_cycle = 0;
    }

    fn step_sweep(&mut self) {
        if self.sweep.reload {
            if self.sweep.enabled && self.sweep.divider == 0 {
                self.set_timer_period_from_sweep();
            }
            self.sweep.divider = self.sweep.period;
            self.sweep.reload = false;
        } else if self.sweep.divider > 0 {
            self.sweep.divider -= 1;
        } else {
            if self.sweep.enabled {
                self.set_timer_period_from_sweep();
            }
            self.sweep.divider = self.sweep.period;
        }
    }

    fn set_timer_period_from_sweep(&mut self) {
        let delta = self.timer_period >> self.sweep.shift_count;
        if self.sweep.negate {
            match self.negation_type {
                SweepNegationType::OnesComplement => {
                    self.timer_period -= delta + 1;
                }
                SweepNegationType::TwosComplement => {
                    self.timer_period -= delta;
                }
            }
        } else {
            self.timer_period += delta;
        }
    }

    fn step_timer(&mut self) {
        if self.timer_value == 0 {
            self.timer_value = self.timer_period;
            self.duty_cycle = (self.duty_cycle + 1) % 8;
        } else {
            self.timer_value -= 1;
        }
    }

    fn output(&self) -> u8 {
        if !self.enabled
            || self.length_counter.count == 0
            || DUTY_CYCLE_TABLE[self.duty_mode as usize][self.duty_cycle as usize] == 0
            || self.timer_period < 8
            || self.timer_period > 0x7FF
        {
            0
        } else if self.envelope.enabled {
            self.envelope.volume
        } else {
            self.constant_volume
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct LinearCounter {
    period: u8,
    count: u8,
    reload: bool,
}

impl LinearCounter {
    fn new() -> LinearCounter {
        LinearCounter {
            period: 0,
            count: 0,
            reload: false,
        }
    }

    fn step(&mut self, length_counter_enabled: bool) {
        if self.reload {
            self.count = self.period;
        } else if self.count != 0 {
            self.count -= 1;
        }

        if length_counter_enabled {
            self.reload = false;
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Triangle {
    enabled: bool,
    timer_value: u16,
    timer_period: u16,
    length_counter: LengthCounter,
    linear_counter: LinearCounter,
    duty_cycle: u8,
}

impl Triangle {
    fn new() -> Triangle {
        Triangle {
            enabled: false,
            timer_value: 0,
            timer_period: 0,
            length_counter: LengthCounter::new(),
            linear_counter: LinearCounter::new(),
            duty_cycle: 0,
        }
    }

    fn write_linear_counter(&mut self, value: u8) {
        self.length_counter.enabled = value & 0x80 == 0;
        self.linear_counter.period = value & 0x7F;
    }

    fn write_timer_lo(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0xFF00) | (value as u16);
    }

    fn write_length_counter_and_timer_hi(&mut self, value: u8) {
        if self.enabled {
            self.length_counter.set(value >> 3);
        }
        self.timer_period = (self.timer_period & 0x00FF) | (((value & 0x07) as u16) << 8);
        self.timer_value = self.timer_period;
        self.linear_counter.reload = true;
    }

    fn step_timer(&mut self) {
        if self.timer_value == 0 {
            self.timer_value = self.timer_period;
            if self.length_counter.count > 0 && self.linear_counter.count > 0 {
                self.duty_cycle = (self.duty_cycle + 1) % 32;
            }
        } else {
            self.timer_value -= 1;
        }
    }

    fn step_linear_counter(&mut self) {
        self.linear_counter.step(self.length_counter.enabled);
    }

    fn output(&self) -> u8 {
        TRIANGLE_TABLE[self.duty_cycle as usize]
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Noise {
    enabled: bool,
    mode: bool,
    shift_register: u16,
    timer_value: u16,
    timer_period: u16,
    length_counter: LengthCounter,
    envelope: Envelope,
    constant_volume: u8,
}

impl Noise {
    fn new() -> Noise {
        Noise {
            enabled: false,
            mode: false,
            shift_register: 1,
            timer_value: 0,
            timer_period: 0,
            length_counter: LengthCounter::new(),
            envelope: Envelope::new(),
            constant_volume: 0,
        }
    }

    fn write_control(&mut self, value: u8) {
        self.length_counter.enabled = (value & 0x20) == 0;
        self.envelope.loop_flag = !self.length_counter.enabled;
        self.envelope.enabled = (value & 0x10) == 0;
        self.constant_volume = value & 0x0F;
        self.envelope.period = self.constant_volume;
        self.envelope.start = true;
    }

    fn write_mode_and_timer_period(&mut self, value: u8) {
        self.mode = (value & 0x80) != 0;
        self.timer_period = NOISE_TABLE[(value & 0x0F) as usize];
    }

    fn write_length_counter_and_envelope_restart(&mut self, value: u8) {
        if self.enabled {
            self.length_counter.set(value >> 3);
        }
        self.envelope.start = true;
    }

    fn step_timer(&mut self) {
        if self.timer_value == 0 {
            self.timer_value = self.timer_period;
            let shift = if self.mode { 6 } else { 1 };
            let b1 = self.shift_register & 0x0001;
            let b2 = (self.shift_register >> shift) & 0x0001;
            self.shift_register >>= 1;
            self.shift_register |= (b1 ^ b2) << 14;
        } else {
            self.timer_value -= 1;
        }
    }

    fn output(&self) -> u8 {
        if !self.enabled || self.length_counter.count == 0 || self.shift_register & 0x0001 == 1 {
            0
        } else if self.envelope.enabled {
            self.envelope.volume
        } else {
            self.constant_volume
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Dmc {
    enable_flag: bool,
    loop_flag: bool,
    irq_flag: bool,
    irq_pending: bool,
    value: u8,
    sample_address: u16,
    sample_length: u16,
    current_address: u16,
    current_length: u16,
    shift_register: u8,
    bit_count: u8,
    tick_period: u8,
    tick_value: u8,
}

impl Dmc {
    fn new() -> Dmc {
        Dmc {
            enable_flag: false,
            loop_flag: false,
            irq_flag: false,
            irq_pending: false,
            value: 0,
            sample_address: 0,
            sample_length: 0,
            current_address: 0,
            current_length: 0,
            shift_register: 0,
            bit_count: 0,
            tick_period: 0,
            tick_value: 0,
        }
    }

    fn write_control(&mut self, value: u8) {
        self.irq_flag = value & 0x80 != 0;
        self.loop_flag = value & 0x40 != 0;
        self.tick_period = DMC_TABLE[(value & 0x0F) as usize];
    }

    fn write_value(&mut self, value: u8) {
        self.value = value & 0x7F;
    }

    fn write_sample_address(&mut self, value: u8) {
        self.sample_address = 0xC000 | ((value as u16) << 6);
    }

    fn write_sample_length(&mut self, value: u8) {
        self.sample_length = ((value as u16) << 4) | 0x0001;
    }

    fn restart(&mut self) {
        self.current_address = self.sample_address;
        self.current_length = self.sample_length;
    }

    fn step_timer(&mut self, mapper: &mut MapperEnum) -> u8 {
        let mut cpu_stall_cycles = 0;
        if self.enable_flag {
            // DMC interrupt fires when the bytes-remaining counter
            // reaches zero AND the loop flag is clear — handled inside
            // step_reader at the moment current_length hits 0. The
            // previous code asserted irq_pending on every tick when
            // both enable+irq_flag were set, which made $4015 bit 7
            // permanently high for any ROM that enabled DMC IRQ.
            cpu_stall_cycles = self.step_reader(mapper);
            if self.tick_value == 0 {
                self.tick_value = self.tick_period;
                self.step_shifter();
            } else {
                self.tick_value -= 1;
            }
        }

        cpu_stall_cycles
    }

    fn step_reader(&mut self, mapper: &mut MapperEnum) -> u8 {
        let mut cpu_stall_cycles = 0;
        if self.current_length > 0 && self.bit_count == 0 {
            cpu_stall_cycles = 4;
            self.shift_register = mapper.prg_read_byte(self.current_address);
            self.bit_count = 8;
            self.current_address += 1;
            if self.current_address == 0 {
                self.current_address = 0x8000;
            }
            self.current_length -= 1;
            if self.current_length == 0 {
                if self.loop_flag {
                    self.restart();
                } else if self.irq_flag {
                    // Sample finished without looping — assert the IRQ
                    // line. Stays asserted until cleared by a write to
                    // $4015 (write_status sets irq_pending = false).
                    self.irq_pending = true;
                }
            }
        }

        cpu_stall_cycles
    }

    fn step_shifter(&mut self) {
        if self.bit_count == 0 {
            return;
        }

        if self.shift_register & 0x01 != 0 {
            if self.value < 126 {
                self.value += 2;
            }
        } else if self.value > 1 {
            self.value -= 2;
        }

        self.shift_register >>= 1;
        self.bit_count -= 1;
    }

    fn output(&self) -> u8 {
        self.value
    }
}

#[derive(Eq, PartialEq, Clone, Copy, Deserialize, Serialize)]
enum FrameCounterMode {
    FourStep,
    FiveStep,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct FrameCounter {
    divider_count: u16,
    sequence_frame: u8,
    mode: FrameCounterMode,
    irq_pending: bool,
    interrupt_inhibit_flag: bool,
}

impl FrameCounter {
    const DIVIDER_COUNT_RELOAD_VALUE: u16 = 3728;

    fn new() -> FrameCounter {
        FrameCounter {
            divider_count: FrameCounter::DIVIDER_COUNT_RELOAD_VALUE,
            sequence_frame: 0,
            mode: FrameCounterMode::FourStep,
            irq_pending: false,
            // Power-on state of the IRQ-inhibit bit is "unspecified"
            // per NESdev wiki, but Mesen / Nestopia / FCEUX all init
            // it to TRUE because games that use the frame-counter IRQ
            // explicitly enable it via STA $4017 with bit 6 cleared,
            // and games that DON'T want it (the vast majority — every
            // MMC1/AxROM cart that puts its IRQ vector inside ROM
            // instead of RAM) assume IRQ stays inhibited until they
            // initialize it. Initializing to FALSE caused Bill & Ted's
            // (mapper 1) and Crash 'n' the Boys (mapper 4) to crash
            // within 3 frames of cold reset: the spurious frame-IRQ
            // fires before the game has set up its IRQ pointer, the
            // CPU jumps to the IRQ vector, and the indirect target is
            // garbage. See playability_sweep_v2.json for the
            // pre-fix bucket contents.
            interrupt_inhibit_flag: true,
        }
    }
}

trait Filter {
    fn step(&mut self, sample: f32) -> f32;

    fn chain<U>(self, other: U) -> FilterChain<Self, U>
    where
        Self: Sized,
        U: Filter,
    {
        FilterChain { a: self, b: other }
    }
}

struct FilterChain<A, B> {
    a: A,
    b: B,
}

impl<A, B> Filter for FilterChain<A, B>
where
    A: Filter,
    B: Filter,
{
    fn step(&mut self, sample: f32) -> f32 {
        self.b.step(self.a.step(sample))
    }
}

struct LowPassFilter {
    last_out: f32,
    k: f32,
}

impl LowPassFilter {
    fn new(k: f32) -> LowPassFilter {
        LowPassFilter { last_out: 0.0, k }
    }
}

impl Filter for LowPassFilter {
    fn step(&mut self, sample: f32) -> f32 {
        self.last_out = (sample - self.last_out) * self.k;

        self.last_out
    }
}

struct HighPassFilter {
    last_in: f32,
    last_out: f32,
    k: f32,
}

impl HighPassFilter {
    fn new(k: f32) -> HighPassFilter {
        HighPassFilter {
            last_in: 0.0,
            last_out: 0.0,
            k,
        }
    }
}

impl Filter for HighPassFilter {
    fn step(&mut self, sample: f32) -> f32 {
        self.last_out = self.last_out * self.k + sample - self.last_in;
        self.last_in = sample;

        self.last_out
    }
}

#[cfg(test)]
mod frame_irq_tests {
    use super::*;

    // Reading $4015 must report the frame-interrupt flag (bit 6) that
    // was asserted BEFORE the read, then clear it as the documented
    // read side effect. Previously the flag was cleared before the
    // status byte was built, so bit 6 could never read as 1.
    #[test]
    fn read_4015_reports_frame_irq_then_clears_it() {
        let mut apu = Apu::new();
        apu.frame_counter.irq_pending = true;

        let first = apu.read_byte(0x4015);
        assert_eq!(first & 0x40, 0x40, "bit 6 must reflect the pending frame IRQ on read");
        assert!(
            !apu.frame_counter.irq_pending,
            "reading $4015 clears the frame IRQ flag"
        );

        let second = apu.read_byte(0x4015);
        assert_eq!(second & 0x40, 0x00, "flag stays clear on the next read");
    }

    // Reading $4015 clears the frame IRQ but must NOT clear the DMC
    // interrupt flag (bit 7) — only a $4015 write does that.
    #[test]
    fn read_4015_preserves_dmc_irq() {
        let mut apu = Apu::new();
        apu.dmc.irq_pending = true;

        let status = apu.read_byte(0x4015);
        assert_eq!(status & 0x80, 0x80, "bit 7 reflects the DMC IRQ");
        assert!(
            apu.dmc.irq_pending,
            "reading $4015 must not clear the DMC IRQ flag"
        );
    }

    // Writing $4017 clears the frame-interrupt flag only when the
    // interrupt-inhibit bit (bit 6) is set; with bit 6 clear the flag
    // is left untouched.
    #[test]
    fn write_4017_clears_frame_irq_only_when_inhibit_set() {
        let mut apu = Apu::new();

        apu.frame_counter.irq_pending = true;
        apu.write_byte(0x4017, 0x40);
        assert!(
            !apu.frame_counter.irq_pending,
            "writing $4017 with bit 6 set clears the frame IRQ"
        );

        apu.frame_counter.irq_pending = true;
        apu.write_byte(0x4017, 0x00);
        assert!(
            apu.frame_counter.irq_pending,
            "writing $4017 with bit 6 clear must leave the frame IRQ flag set"
        );
    }
}
