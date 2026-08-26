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

/// Bits of the $4015 status byte that describe per-channel activity:
/// pulse 1, pulse 2, triangle, noise (length counter > 0) and DMC
/// (bytes remaining > 0). The two IRQ bits ($40 frame, $80 DMC) are
/// deliberately excluded — they report interrupt plumbing, not
/// whether a voice is sounding.
pub const CHANNEL_ACTIVITY_MASK: u8 = 0x1F;

pub struct Apu {
    cycles: u64,

    last_sampled_cycles: u64,

    /// Hardware-true DMC DMA stall length (config, not savestate).
    /// Legacy charges a flat 4 CPU cycles per DMC byte fetch; real
    /// hardware (and Mesen) charge 3 when the RDY halt lands aligned
    /// with a get cycle — which, in this engine's timing model, is
    /// every normal fetch (DMC steps only on even APU cycles). The
    /// flat 4 over-stalls by ~+1/fetch and drifts long tape replays
    /// vs Mesen at ~3 cycles/frame while DPCM plays (receipted on
    /// the CV block-1 drums). Default OFF: legacy receipts assume 4.
    pub hw_dmc_stall_timing: bool,

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
            hw_dmc_stall_timing: false,
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

    /// Per-channel activity vector: the low 5 bits of $4015
    /// (pulse 1, pulse 2, triangle, noise, DMC), with the frame-IRQ
    /// and DMC-IRQ bits masked off.
    ///
    /// Side-effect-free by construction — it goes through the same
    /// `read_status` builder as `peek_byte`, NOT the `Memory::read_byte`
    /// path, so it never clears the frame-interrupt flag. An observer
    /// can sample this every step without perturbing the game's IRQ
    /// handling, which a bus read of $4015 would.
    pub fn channel_activity(&self) -> u8 {
        self.read_status() & CHANNEL_ACTIVITY_MASK
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
        //
        // Length counters + the frame IRQ flag are CPU-observable
        // ($4015 status bits 0-3 and the IRQ line), so they always
        // step. Sweep + envelope + linear-counter clocking feeds ONLY
        // channel `output()`, so it is skipped on the muted training
        // workers. When audio is enabled every gated block runs in the
        // original order, so the audio-ON path is byte-identical.
        let audio = self.sample_output_enabled;
        match self.frame_counter.mode {
            FrameCounterMode::FourStep => {
                match self.frame_counter.sequence_frame {
                    0 => {
                        if audio {
                            self.step_envelope_and_linear_counter();
                        }
                    }
                    1 => {
                        self.step_length_counter();
                        if audio {
                            self.step_sweep();
                            self.step_envelope_and_linear_counter();
                        }
                    }
                    2 => {
                        if audio {
                            self.step_envelope_and_linear_counter();
                        }
                    }
                    3 => {
                        if !self.frame_counter.interrupt_inhibit_flag {
                            self.frame_counter.irq_pending = true;
                        }
                        self.step_length_counter();
                        if audio {
                            self.step_sweep();
                            self.step_envelope_and_linear_counter();
                        }
                    }
                    _ => (),
                }
                self.frame_counter.sequence_frame = (self.frame_counter.sequence_frame + 1) % 4;
            }
            FrameCounterMode::FiveStep => {
                match self.frame_counter.sequence_frame {
                    0 | 2 => {
                        self.step_length_counter();
                        if audio {
                            self.step_sweep();
                            self.step_envelope_and_linear_counter();
                        }
                    }
                    1 | 3 => {
                        if audio {
                            self.step_envelope_and_linear_counter();
                        }
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
        // Pulse/triangle/noise timer + duty/shift advance feeds ONLY
        // `output()` → `generate_sample()`, which is gated off on the
        // muted training workers. Skip it there. DMC is fidelity-
        // mandatory: its `step_timer` return value steals CPU cycles
        // and it drives $4015 bit 4 + the DMC IRQ, so it always steps.
        // When audio is enabled every branch below is taken, so the
        // audio-ON path (and the parity / Mesen gates, which run
        // audio-ON) stays byte-identical to the pre-skip sequence.
        let audio = self.sample_output_enabled;
        if self.cycles.is_multiple_of(2) {
            if audio {
                self.pulse_1.step_timer();
                self.pulse_2.step_timer();
                self.noise.step_timer();
            }
            cpu_stall_cycles =
                self.dmc.step_timer(mapper, self.hw_dmc_stall_timing);
        }

        if audio {
            self.triangle.step_timer();
        }

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
        // Real hardware only writes the new period when the shift count is
        // non-zero AND the computed target period stays within the 11-bit
        // range (<= 0x7FF). When the target would overflow 0x7FF the
        // channel is muted but the period register is left UNCHANGED. The
        // target is computed in a wide signed type so the u16
        // `timer_period` can never under/overflow (the old unchecked
        // `+= delta` / `-= delta + 1` could wrap on large periods).
        if self.sweep.shift_count == 0 {
            return;
        }
        let delta = (self.timer_period >> self.sweep.shift_count) as i32;
        let target = if self.sweep.negate {
            match self.negation_type {
                SweepNegationType::OnesComplement => self.timer_period as i32 - delta - 1,
                SweepNegationType::TwosComplement => self.timer_period as i32 - delta,
            }
        } else {
            self.timer_period as i32 + delta
        };
        if (0..=0x7FF).contains(&target) {
            self.timer_period = target as u16;
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
        if !self.irq_flag {
            // NESdev APU_DMC: "If clear, the interrupt flag is cleared."
            self.irq_pending = false;
        }
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

    fn step_timer(&mut self, mapper: &mut MapperEnum, hw_stall: bool) -> u8 {
        let mut cpu_stall_cycles = 0;
        if self.enable_flag {
            // DMC interrupt fires when the bytes-remaining counter
            // reaches zero AND the loop flag is clear — handled inside
            // step_reader at the moment current_length hits 0. The
            // previous code asserted irq_pending on every tick when
            // both enable+irq_flag were set, which made $4015 bit 7
            // permanently high for any ROM that enabled DMC IRQ.
            cpu_stall_cycles = self.step_reader(mapper, hw_stall);
            if self.tick_value == 0 {
                self.tick_value = self.tick_period;
                self.step_shifter();
            } else {
                self.tick_value -= 1;
            }
        }

        cpu_stall_cycles
    }

    fn step_reader(&mut self, mapper: &mut MapperEnum, hw_stall: bool) -> u8 {
        let mut cpu_stall_cycles = 0;
        if self.current_length > 0 && self.bit_count == 0 {
            // Hardware: halt + dummy + read = 3 cycles when the RDY
            // halt lands on a get cycle (always true here — DMC steps
            // on even APU cycles only). Legacy flat 4 kept as default.
            cpu_stall_cycles = if hw_stall { 3 } else { 4 };
            self.shift_register = mapper.prg_read_byte(self.current_address);
            self.bit_count = 8;
            self.current_address = self.current_address.wrapping_add(1);
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
        // One-pole low-pass: pull the running output a fraction `k` of the
        // way toward the new sample. The accumulator term (`last_out +=`)
        // is essential — without it (`last_out = (sample - last_out) * k`)
        // the response inverts into a high-pass.
        self.last_out += (sample - self.last_out) * self.k;

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

#[cfg(test)]
mod sweep_lowpass_tests {
    use super::*;

    #[test]
    fn sweep_overflow_does_not_corrupt_period() {
        // Additive sweep whose target exceeds the 11-bit range (0x7FF):
        // hardware mutes the channel but leaves the period register
        // untouched. The old unchecked `+= delta` could wrap the u16.
        let mut p = Pulse::new(SweepNegationType::OnesComplement);
        p.timer_period = 0x7F0;
        p.sweep.negate = false;
        p.sweep.shift_count = 1; // delta = 0x7F0 >> 1 = 0x3F8
        // target = 0x7F0 + 0x3F8 = 0xBE8 > 0x7FF -> must NOT be written.
        p.set_timer_period_from_sweep();
        assert_eq!(
            p.timer_period, 0x7F0,
            "period must be unchanged when the sweep target exceeds 0x7FF"
        );

        // shift_count == 0 never updates the period.
        let before = p.timer_period;
        p.sweep.shift_count = 0;
        p.set_timer_period_from_sweep();
        assert_eq!(
            p.timer_period, before,
            "shift_count 0 must never update the period"
        );

        // An in-range additive target IS written.
        p.timer_period = 0x100;
        p.sweep.shift_count = 2; // delta = 0x40, target = 0x140 <= 0x7FF
        p.set_timer_period_from_sweep();
        assert_eq!(
            p.timer_period, 0x140,
            "in-range additive sweep should update the period"
        );
    }

    #[test]
    fn low_pass_filter_converges_to_dc_input() {
        // A one-pole low-pass fed a constant (DC/step) input must converge
        // TOWARD the input. The buggy `(sample - last_out) * k` form is a
        // high-pass: it settles at k/(1+k)*input (0.2 here), not 1.0.
        let mut lp = LowPassFilter::new(0.25);
        let mut y = 0.0;
        for _ in 0..500 {
            y = lp.step(1.0);
        }
        assert!(
            (y - 1.0).abs() < 1e-3,
            "low-pass should converge to the DC input 1.0, got {}",
            y
        );
        // It must track toward the input, never away from it.
        assert!(
            y > 0.5,
            "output must move toward the input, not away from it, got {}",
            y
        );
    }
}

/// `channel_activity` — the second observation modality exposed to
/// the Python side. It must be (a) exactly the low 5 bits of the
/// $4015 status byte, (b) free of the two IRQ bits, and (c) free of
/// the read side effect that a real $4015 bus read carries.
#[cfg(test)]
mod channel_activity_tests {
    use super::*;

    /// Enable every channel and give each a non-zero length counter /
    /// DMC byte count, so all five activity bits should be set.
    fn enable_all_channels(apu: &mut Apu) {
        // DMC sample length must be programmed BEFORE the $4015
        // enable — `restart()` copies sample_length into
        // current_length, and a zero length leaves bit 4 clear.
        apu.write_byte(0x4013, 0x01); // sample_length = 0x11 bytes
        apu.write_byte(0x4015, 0x1F); // enable pulse1/2, tri, noise, DMC
        // Length-counter loads (top 5 bits index the length table).
        apu.write_byte(0x4003, 0x08); // pulse 1
        apu.write_byte(0x4007, 0x08); // pulse 2
        apu.write_byte(0x400B, 0x08); // triangle
        apu.write_byte(0x400F, 0x08); // noise
    }

    #[test]
    fn silent_apu_reports_no_active_channels() {
        let apu = Apu::new();
        assert_eq!(apu.channel_activity(), 0x00);
    }

    #[test]
    fn all_five_channels_report_active() {
        let mut apu = Apu::new();
        enable_all_channels(&mut apu);
        assert_eq!(
            apu.channel_activity(),
            0x1F,
            "expected all five channel bits set; got {:#04x} (status {:#04x})",
            apu.channel_activity(),
            apu.peek_byte(0x4015),
        );
    }

    /// Each channel owns exactly one bit, in $4015 order. Disabling a
    /// single channel must clear exactly that bit and nothing else —
    /// the property a per-channel observation vector depends on.
    #[test]
    fn each_channel_owns_exactly_one_bit() {
        for (disable_bit, name) in [
            (0x01u8, "pulse 1"),
            (0x02, "pulse 2"),
            (0x04, "triangle"),
            (0x08, "noise"),
            (0x10, "dmc"),
        ] {
            let mut apu = Apu::new();
            enable_all_channels(&mut apu);
            // Re-write $4015 with this one channel's enable cleared;
            // that resets its length counter (or DMC byte count).
            apu.write_byte(0x4015, 0x1F & !disable_bit);
            assert_eq!(
                apu.channel_activity(),
                0x1F & !disable_bit,
                "disabling {name} should clear only bit {disable_bit:#04x}",
            );
        }
    }

    /// Bits 6 (frame IRQ) and 7 (DMC IRQ) are part of $4015 but are
    /// NOT channel activity. They must never leak into the vector.
    #[test]
    fn irq_bits_are_masked_out() {
        let mut apu = Apu::new();
        apu.frame_counter.irq_pending = true;
        apu.dmc.irq_pending = true;

        assert_eq!(
            apu.peek_byte(0x4015) & 0xC0,
            0xC0,
            "precondition: both IRQ bits are set in the raw status byte",
        );
        assert_eq!(
            apu.channel_activity(),
            0x00,
            "IRQ bits leaked into the channel-activity vector",
        );
    }

    /// The vector must agree with `peek_byte(0x4015) & 0x1F` at every
    /// point — it is defined as that projection, and callers that
    /// already peek $4015 must see the same answer.
    #[test]
    fn agrees_with_peek_status_projection() {
        let mut apu = Apu::new();
        assert_eq!(apu.channel_activity(), apu.peek_byte(0x4015) & 0x1F);
        enable_all_channels(&mut apu);
        assert_eq!(apu.channel_activity(), apu.peek_byte(0x4015) & 0x1F);
        apu.frame_counter.irq_pending = true;
        assert_eq!(apu.channel_activity(), apu.peek_byte(0x4015) & 0x1F);
    }

    /// Sampling the vector must NOT clear the frame-interrupt flag —
    /// that side effect belongs to a real bus read of $4015. An
    /// observer polling this every step would otherwise silently eat
    /// the game's frame IRQs.
    #[test]
    fn sampling_does_not_clear_frame_irq() {
        let mut apu = Apu::new();
        apu.frame_counter.irq_pending = true;

        for _ in 0..8 {
            let _ = apu.channel_activity();
        }
        assert!(
            apu.frame_counter.irq_pending,
            "channel_activity must be side-effect-free",
        );

        // Contrast: the real bus read DOES clear it. If this ever
        // stops being true the side-effect-freeness above is vacuous.
        let _ = apu.read_byte(0x4015);
        assert!(!apu.frame_counter.irq_pending);
    }

    /// Channel activity survives a state round trip, so a solver that
    /// restores a savestate observes the same audio modality it saw
    /// when the snapshot was taken.
    #[test]
    fn survives_state_round_trip() {
        let mut apu = Apu::new();
        enable_all_channels(&mut apu);
        let before = apu.channel_activity();
        assert_ne!(before, 0x00);

        let saved = apu.get_state();
        let mut restored = Apu::new();
        restored.apply_state(&saved);
        assert_eq!(restored.channel_activity(), before);
    }
}

#[cfg(test)]
mod apu_coverage_tests {
    use super::*;
    use crate::cartridge::Cartridge;
    use crate::mapper::MapperEnum;

    /// Minimal NROM cart (32 KB PRG / 8 KB CHR of zeros) so the DMC
    /// reader has a mapper to fetch sample bytes from without needing
    /// a real game ROM. Every fetched byte reads back 0x00.
    fn nrom_mapper() -> MapperEnum {
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // 32 KB PRG
        rom.push(1); // 8 KB CHR
        rom.extend_from_slice(&[0u8; 10]);
        rom.extend(vec![0u8; 32 * 1024]);
        rom.extend(vec![0u8; 8 * 1024]);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        MapperEnum::from_cartridge(cart).unwrap()
    }

    // --- GAP 1: DMC sample-address advance + wrap ---------------------

    // Each DMC byte fetch advances the sample pointer by one, consumes
    // one remaining byte, and reloads the 8-bit shifter. If the pointer
    // stopped advancing the channel would replay the same byte forever;
    // if the length stopped counting down the sample would never end.
    #[test]
    fn dmc_sample_pointer_advances_and_counts_down_per_fetch() {
        let mut mapper = nrom_mapper();
        let mut dmc = Dmc::new();
        dmc.enable_flag = true;
        dmc.current_address = 0x8000;
        dmc.current_length = 3;
        dmc.bit_count = 0;

        let stall = dmc.step_reader(&mut mapper, false);
        assert_eq!(dmc.current_address, 0x8001, "each fetch advances the sample pointer by one");
        assert_eq!(dmc.current_length, 2, "each fetch consumes one remaining byte");
        assert_eq!(dmc.bit_count, 8, "a fetched byte reloads the 8-bit shifter");
        assert_eq!(stall, 4, "legacy DMC DMA stall charges 4 CPU cycles per fetch by default");

        // hw_dmc_stall_timing charges the hardware-true 3 cycles instead.
        dmc.bit_count = 0;
        let hw = dmc.step_reader(&mut mapper, true);
        assert_eq!(hw, 3, "hw stall timing charges 3 CPU cycles per fetch");
        assert_eq!(dmc.current_address, 0x8002, "pointer keeps advancing on the second fetch");
    }

    // GAP 1: the sample pointer must wrap $FFFF -> $8000 so the next
    // fetch stays inside cartridge ROM and never drops to $0000 (which
    // would read zero-page RAM). The pointer increment uses wrapping_add
    // so the $FFFF -> $0000 wrap is well-defined in every build profile
    // (not only release, where overflow-checks are off); the guard then
    // rewrites the wrapped 0 to $8000, keeping the fetch inside ROM.
    #[test]
    fn dmc_sample_address_wraps_into_rom_not_zero() {
        let mut mapper = nrom_mapper();
        let mut dmc = Dmc::new();
        dmc.enable_flag = true;
        dmc.loop_flag = false;
        dmc.irq_flag = false;
        dmc.current_address = 0xFFFF;
        dmc.current_length = 4;
        dmc.bit_count = 0;

        // Must not panic under overflow-checks, and must land in ROM.
        dmc.step_reader(&mut mapper, false);
        assert_eq!(
            dmc.current_address, 0x8000,
            "the $FFFF wrap must land in ROM space ($8000), never $0000",
        );
    }

    // --- GAP 2: DMC loop flag + sample-end IRQ ------------------------

    // With the loop flag set, hitting the end of the sample restarts the
    // reader from the PROGRAMMED sample_address / sample_length (not from
    // wherever the pointer happened to stop), and no IRQ is asserted even
    // when the IRQ-enable flag is also set — looping suppresses the IRQ.
    #[test]
    fn dmc_loop_restarts_from_programmed_address_and_suppresses_irq() {
        let mut mapper = nrom_mapper();
        let mut dmc = Dmc::new();
        dmc.enable_flag = true;
        dmc.loop_flag = true;
        dmc.irq_flag = true; // set on purpose: loop must win over IRQ
        dmc.sample_address = 0xC000;
        dmc.sample_length = 0x0010;
        dmc.current_address = 0x9000;
        dmc.current_length = 1;
        dmc.bit_count = 0;

        dmc.step_reader(&mut mapper, false);
        assert_eq!(dmc.current_length, 0x0010, "loop reloads bytes-remaining from sample_length");
        assert_eq!(dmc.current_address, 0xC000, "loop reloads the pointer from sample_address");
        assert!(!dmc.irq_pending, "a looping DMC never asserts the IRQ");
    }

    // Sample end with IRQ enabled and looping OFF asserts the DMC IRQ,
    // which must surface in $4015 bit 7. This drives the flag through the
    // real sample-completion path rather than poking irq_pending directly.
    #[test]
    fn dmc_sample_end_asserts_irq_and_shows_in_4015() {
        let mut mapper = nrom_mapper();
        let mut apu = Apu::new();
        apu.dmc.enable_flag = true;
        apu.dmc.loop_flag = false;
        apu.dmc.irq_flag = true;
        apu.dmc.current_address = 0x8000;
        apu.dmc.current_length = 1;
        apu.dmc.bit_count = 0;
        assert!(!apu.dmc.irq_pending, "precondition: no DMC IRQ pending");

        apu.dmc.step_reader(&mut mapper, false);
        assert!(apu.dmc.irq_pending, "sample end (no loop, IRQ enabled) must assert the DMC IRQ");
        assert_eq!(apu.peek_byte(0x4015) & 0x80, 0x80, "DMC IRQ must surface in $4015 bit 7");
    }

    // Sample end with IRQ DISABLED must not assert — proves the IRQ is
    // gated on irq_flag, not merely on the sample completing.
    #[test]
    fn dmc_sample_end_without_irq_flag_stays_quiet() {
        let mut mapper = nrom_mapper();
        let mut dmc = Dmc::new();
        dmc.enable_flag = true;
        dmc.loop_flag = false;
        dmc.irq_flag = false;
        dmc.current_address = 0x8000;
        dmc.current_length = 1;
        dmc.bit_count = 0;

        dmc.step_reader(&mut mapper, false);
        assert_eq!(dmc.current_length, 0, "sample still completes");
        assert!(!dmc.irq_pending, "IRQ must stay clear when the IRQ-enable flag is off");
    }

    // Rewriting $4010 with bit 7 (IRQ-enable) clear is a hardware-legal
    // ack path, independent of $4015: NESdev APU_DMC says "If clear, the
    // interrupt flag is cleared." A pending DMC IRQ must drop immediately,
    // not linger until something touches $4015 — otherwise the level-
    // triggered IRQ line stays asserted and every RTI re-triggers it.
    #[test]
    fn dmc_control_write_clearing_irq_enable_acks_pending_irq() {
        let mut mapper = nrom_mapper();
        let mut apu = Apu::new();
        apu.write_byte(0x4010, 0x80); // arm: IRQ-enable=1, loop=0
        apu.dmc.enable_flag = true;
        apu.dmc.current_address = 0x8000;
        apu.dmc.current_length = 1;
        apu.dmc.bit_count = 0;

        apu.dmc.step_reader(&mut mapper, false);
        assert!(apu.dmc.irq_pending, "precondition: sample end raised the DMC IRQ");

        apu.write_byte(0x4010, 0x00); // ack: IRQ-enable=0, $4015 untouched
        assert!(!apu.dmc.irq_pending, "clearing $4010 bit 7 must ack a pending DMC IRQ");
        assert!(!apu.irq_pending(), "combined IRQ line must drop once the DMC IRQ is acked");
    }

    // --- GAP 3: frame counter 4-step vs 5-step ($4017 bit 7) ---------

    // 4-step mode: the frame IRQ fires only on the final (4th) sequencer
    // step, and only while the interrupt-inhibit bit is clear. Firing on
    // any earlier step would inject spurious CPU IRQs mid-frame.
    #[test]
    fn four_step_frame_irq_only_on_last_step() {
        let mut apu = Apu::new();
        apu.write_byte(0x4017, 0x00); // FourStep, inhibit cleared, sequence reset

        // Steps 0,1,2 must not raise the frame IRQ.
        for step in 0..3 {
            apu.step_frame_counter();
            assert!(
                !apu.frame_counter.irq_pending,
                "frame IRQ raised early on step {step}",
            );
        }
        // The 4th step is the one that asserts it.
        apu.step_frame_counter();
        assert!(
            apu.frame_counter.irq_pending,
            "frame IRQ must assert on the last step of the 4-step sequence",
        );
    }

    // 4-step with the inhibit bit set: the last step must NOT raise the
    // frame IRQ. This is the flip side of the fire-on-last-step rule.
    #[test]
    fn four_step_inhibited_never_raises_frame_irq() {
        let mut apu = Apu::new();
        apu.write_byte(0x4017, 0x40); // FourStep, inhibit SET
        for _ in 0..8 {
            apu.step_frame_counter();
            assert!(
                !apu.frame_counter.irq_pending,
                "inhibited 4-step must never assert the frame IRQ",
            );
        }
    }

    // 5-step mode: writing $4017 with bit 7 set performs an IMMEDIATE
    // length/sweep/envelope clock, and the 5-step sequence never sets the
    // frame IRQ no matter how long it runs.
    #[test]
    fn five_step_immediate_clock_and_no_frame_irq() {
        let mut apu = Apu::new();
        // Seed a length counter so the immediate clock is observable.
        apu.pulse_1.length_counter.enabled = true;
        apu.pulse_1.length_counter.count = 10;

        apu.write_byte(0x4017, 0x80); // FiveStep -> immediate clock
        assert_eq!(
            apu.pulse_1.length_counter.count, 9,
            "5-step write must immediately clock the length counters once",
        );
        assert!(apu.frame_counter.mode == FrameCounterMode::FiveStep, "bit 7 selects 5-step mode");

        // No amount of stepping asserts the frame IRQ in 5-step mode.
        for _ in 0..12 {
            apu.step_frame_counter();
            assert!(
                !apu.frame_counter.irq_pending,
                "5-step mode must never set the frame IRQ",
            );
        }
    }

    // Contrast: a 4-step write must NOT perform the immediate clock that
    // 5-step does. Pins the mode-specific side effect of the $4017 write.
    #[test]
    fn four_step_write_does_not_immediately_clock() {
        let mut apu = Apu::new();
        apu.pulse_1.length_counter.enabled = true;
        apu.pulse_1.length_counter.count = 10;

        apu.write_byte(0x4017, 0x00); // FourStep -> no immediate clock
        assert_eq!(
            apu.pulse_1.length_counter.count, 10,
            "4-step write must not clock the length counter on write",
        );
    }

    // --- GAP 4: noise LFSR feedback tap (mode 0 = bit 1, mode 1 = bit 6)

    // The noise shift register feeds bit 0 XOR bit 1 in mode 0 and
    // bit 0 XOR bit 6 in mode 1 back into bit 14. Seeding a register
    // whose bit 1 and bit 6 differ makes the two modes diverge on the
    // very first shift, and their full sequences must differ.
    #[test]
    fn noise_lfsr_tap_differs_between_mode_0_and_mode_1() {
        // Seed 0b0000010: bit0=0, bit1=1, bit6=0.
        //   mode 0 feedback = 0 XOR 1 = 1 -> (0x02>>1) | (1<<14) = 0x4001
        //   mode 1 feedback = 0 XOR 0 = 0 -> (0x02>>1) | (0<<14) = 0x0001
        let mut mode0 = Noise::new();
        mode0.mode = false;
        mode0.shift_register = 0x0002;
        mode0.timer_period = 0; // period 0 -> every step_timer shifts
        mode0.timer_value = 0;
        mode0.step_timer();
        assert_eq!(mode0.shift_register, 0x4001, "mode 0 taps bit 1 into the feedback");

        let mut mode1 = Noise::new();
        mode1.mode = true;
        mode1.shift_register = 0x0002;
        mode1.timer_period = 0;
        mode1.timer_value = 0;
        mode1.step_timer();
        assert_eq!(mode1.shift_register, 0x0001, "mode 1 taps bit 6 into the feedback");

        // And the running sequences must not coincide.
        let mut a = Noise::new();
        a.mode = false;
        a.shift_register = 0x0002;
        let mut b = Noise::new();
        b.mode = true;
        b.shift_register = 0x0002;
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..6 {
            a.step_timer();
            b.step_timer();
            seq_a.push(a.shift_register);
            seq_b.push(b.shift_register);
        }
        assert_ne!(seq_a, seq_b, "the two LFSR modes must produce different sequences");
    }

    // The $400E register write selects the mode from bit 7 and loads the
    // timer period from the low nibble via NOISE_TABLE.
    #[test]
    fn noise_write_400e_selects_mode_and_period() {
        let mut apu = Apu::new();
        apu.write_byte(0x400E, 0x00);
        assert!(!apu.noise.mode, "bit 7 clear selects mode 0");
        assert_eq!(apu.noise.timer_period, NOISE_TABLE[0], "low nibble indexes the noise period table");

        apu.write_byte(0x400E, 0x8F);
        assert!(apu.noise.mode, "bit 7 set selects mode 1 (short)");
        assert_eq!(apu.noise.timer_period, NOISE_TABLE[0x0F], "period comes from the low nibble");
    }

    // --- GAP 5: triangle linear-counter reload -----------------------

    // Setting the reload flag ($400B) plus a reload value ($4008) makes
    // the linear counter reload to that value on the next frame clock.
    // With the control/halt bit clear, that clock also clears the reload
    // flag, so a second clock counts DOWN instead of reloading again.
    #[test]
    fn triangle_linear_counter_reloads_then_clears_flag() {
        let mut apu = Apu::new();
        apu.write_byte(0x4008, 0x05); // control clear -> length enabled, period = 5
        apu.write_byte(0x400B, 0x00); // arms the reload flag

        assert!(apu.triangle.linear_counter.reload, "reload flag armed by $400B write");
        apu.step_envelope_and_linear_counter(); // the frame clock
        assert_eq!(apu.triangle.linear_counter.count, 5, "reload loads the period on the clock");
        assert!(
            !apu.triangle.linear_counter.reload,
            "control bit clear clears the reload flag after reloading",
        );

        // Next clock decrements instead of reloading.
        apu.step_envelope_and_linear_counter();
        assert_eq!(apu.triangle.linear_counter.count, 4, "cleared reload flag lets the counter tick down");
    }

    // With the control/halt bit SET ($4008 bit 7), the reload flag is NOT
    // cleared, so every frame clock keeps reloading the counter — the
    // "hold" behavior a game uses to sustain the triangle indefinitely.
    #[test]
    fn triangle_control_bit_holds_reload_flag() {
        let mut apu = Apu::new();
        apu.write_byte(0x4008, 0x87); // control SET -> length disabled, period = 7
        apu.write_byte(0x400B, 0x00); // arm reload

        apu.step_envelope_and_linear_counter();
        assert_eq!(apu.triangle.linear_counter.count, 7, "first clock reloads to the period");
        assert!(
            apu.triangle.linear_counter.reload,
            "control bit set must keep the reload flag armed",
        );

        // Still reloads on the next clock because the flag persisted.
        apu.step_envelope_and_linear_counter();
        assert_eq!(apu.triangle.linear_counter.count, 7, "held reload flag reloads again");
    }

    // --- GAP 6: length counter halt/clock + $4015 disable clears it ---

    // An enabled length counter counts down one per frame clock; setting
    // the halt bit ($4000 bit 5) freezes it in place.
    #[test]
    fn length_counter_clocks_down_unless_halted() {
        let mut apu = Apu::new();
        apu.write_byte(0x4015, 0x01); // enable pulse 1 so length loads take
        apu.write_byte(0x4000, 0x00); // halt bit clear
        apu.write_byte(0x4003, 0x08); // load length via table index 1 -> 254
        assert_eq!(apu.pulse_1.length_counter.count, LENGTH_TABLE[1]);

        apu.step_length_counter();
        assert_eq!(
            apu.pulse_1.length_counter.count,
            LENGTH_TABLE[1] - 1,
            "an un-halted length counter decrements on each clock",
        );

        // Set the halt bit; the counter must freeze.
        apu.write_byte(0x4000, 0x20); // halt bit set
        let frozen = apu.pulse_1.length_counter.count;
        apu.step_length_counter();
        assert_eq!(
            apu.pulse_1.length_counter.count, frozen,
            "a halted length counter must not decrement",
        );
    }

    // Clearing a channel's enable bit in a $4015 write forces its length
    // counter to 0 immediately, silencing the voice — the mechanism games
    // use to cut a note. Bit 4 (DMC) instead zeroes bytes-remaining.
    #[test]
    fn writing_4015_disable_zeroes_length_counter() {
        let mut apu = Apu::new();
        apu.write_byte(0x4015, 0x01); // enable pulse 1
        apu.write_byte(0x4003, 0x08); // load a non-zero length
        assert!(apu.pulse_1.length_counter.count > 0, "precondition: length loaded");

        apu.write_byte(0x4015, 0x00); // disable every channel
        assert_eq!(
            apu.pulse_1.length_counter.count, 0,
            "clearing the enable bit resets the length counter to 0",
        );
    }
}
