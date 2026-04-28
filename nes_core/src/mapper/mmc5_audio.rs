// MMC5 extra audio: two pulse channels + one PCM output channel.
//
// The pulses mirror the NES APU's own pulse channels register layout
// exactly ($5000-$5003 / $5004-$5007) — same duty, envelope, length
// counter — but MMC5 pulses DON'T have the frequency sweep hardware,
// so $5001/$5005 are ignored. They're clocked by the same APU frame
// counter as the main channels; we piggyback on Apu's existing
// envelope/length pulses via the mapper tick hook.
//
// The PCM channel is stupid-simple: $5011 writes a raw 7-bit DAC
// value; $5010 bit 0 toggles read mode (unused by 99% of games).
//
// Output mixing: each MMC5 pulse's `output()` lands in [0,15]; PCM
// in [0,127]. Normalize both and sum. Level is set so CV3's MMC5
// channels roughly match NES channel loudness.

use serde_derive::{Deserialize, Serialize};

const LENGTH_TABLE: &[u8] = &[
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
    12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
];

const DUTY_TABLE: &[[u8; 8]] = &[
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 1],
];

#[derive(Clone, Deserialize, Serialize)]
pub struct Mmc5Pulse {
    enabled: bool,
    duty_mode: u8,
    duty_position: u8,
    timer_period: u16,
    timer_counter: u16,
    length_counter: u8,
    length_halt: bool,
    envelope_start: bool,
    envelope_constant_flag: bool,
    envelope_constant_volume: u8,
    envelope_period: u8,
    envelope_counter: u8,
    envelope_divider: u8,
}

impl Mmc5Pulse {
    pub fn new() -> Self {
        Self {
            enabled: false,
            duty_mode: 0,
            duty_position: 0,
            timer_period: 0,
            timer_counter: 0,
            length_counter: 0,
            length_halt: false,
            envelope_start: false,
            envelope_constant_flag: false,
            envelope_constant_volume: 0,
            envelope_period: 0,
            envelope_counter: 0,
            envelope_divider: 0,
        }
    }

    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
        if !on {
            self.length_counter = 0;
        }
    }

    pub fn write_ctrl(&mut self, value: u8) {
        self.duty_mode = (value >> 6) & 0x03;
        self.length_halt = value & 0x20 != 0;
        self.envelope_constant_flag = value & 0x10 != 0;
        self.envelope_period = value & 0x0F;
        self.envelope_constant_volume = value & 0x0F;
    }

    pub fn write_timer_lo(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x0700) | value as u16;
    }

    pub fn write_length_and_timer_hi(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x00FF) | (((value as u16) & 0x07) << 8);
        if self.enabled {
            let idx = ((value >> 3) & 0x1F) as usize;
            self.length_counter = LENGTH_TABLE[idx];
        }
        self.duty_position = 0;
        self.envelope_start = true;
    }

    /// Clock the 11-bit timer once per APU cycle. Advances the duty
    /// position when the countdown hits zero.
    pub fn tick_timer(&mut self) {
        if self.timer_counter == 0 {
            self.timer_counter = self.timer_period;
            self.duty_position = (self.duty_position + 1) & 7;
        } else {
            self.timer_counter -= 1;
        }
    }

    /// Clock the length counter at quarter-frame / half-frame cadence.
    pub fn tick_length_counter(&mut self) {
        if !self.length_halt && self.length_counter > 0 {
            self.length_counter -= 1;
        }
    }

    /// Clock the envelope at quarter-frame cadence.
    pub fn tick_envelope(&mut self) {
        if self.envelope_start {
            self.envelope_start = false;
            self.envelope_counter = 15;
            self.envelope_divider = self.envelope_period;
        } else if self.envelope_divider == 0 {
            self.envelope_divider = self.envelope_period;
            if self.envelope_counter > 0 {
                self.envelope_counter -= 1;
            } else if self.length_halt {
                self.envelope_counter = 15;
            }
        } else {
            self.envelope_divider -= 1;
        }
    }

    pub fn output(&self) -> u8 {
        if !self.enabled
            || self.length_counter == 0
            || self.timer_period < 8
            || DUTY_TABLE[self.duty_mode as usize][self.duty_position as usize] == 0
        {
            0
        } else if self.envelope_constant_flag {
            self.envelope_constant_volume
        } else {
            self.envelope_counter
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Mmc5Audio {
    pub pulse_1: Mmc5Pulse,
    pub pulse_2: Mmc5Pulse,
    pub pcm_output: u8,
    pub pcm_read_mode: bool,
    // Frame counter emulation — MMC5 audio runs on its own internal
    // timer, but at the same rate as the APU frame counter's quarter-
    // and half-frame events. Simplify: clock both every 7457 CPU cycles
    // (≈ quarter-frame) and halve it for the length tick.
    frame_cycles: u32,
    frame_step: u8,
}

impl Mmc5Audio {
    pub fn new() -> Self {
        Self {
            pulse_1: Mmc5Pulse::new(),
            pulse_2: Mmc5Pulse::new(),
            pcm_output: 0,
            pcm_read_mode: false,
            frame_cycles: 0,
            frame_step: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Handle a write to one of MMC5's audio register addresses.
    /// Returns true if the address was claimed; false otherwise so
    /// the caller can fall through to its own register handling.
    pub fn write_register(&mut self, address: u16, value: u8) -> bool {
        match address {
            0x5000 => {
                self.pulse_1.write_ctrl(value);
                true
            }
            0x5001 => true, // sweep ignored on MMC5
            0x5002 => {
                self.pulse_1.write_timer_lo(value);
                true
            }
            0x5003 => {
                self.pulse_1.write_length_and_timer_hi(value);
                true
            }
            0x5004 => {
                self.pulse_2.write_ctrl(value);
                true
            }
            0x5005 => true,
            0x5006 => {
                self.pulse_2.write_timer_lo(value);
                true
            }
            0x5007 => {
                self.pulse_2.write_length_and_timer_hi(value);
                true
            }
            0x5010 => {
                self.pcm_read_mode = value & 0x01 != 0;
                true
            }
            0x5011 => {
                // Raw 7-bit DAC output; mode bit ignored for now.
                if !self.pcm_read_mode {
                    self.pcm_output = value;
                }
                true
            }
            0x5015 => {
                self.pulse_1.set_enabled(value & 0x01 != 0);
                self.pulse_2.set_enabled(value & 0x02 != 0);
                true
            }
            _ => false,
        }
    }

    /// Advance one CPU cycle. Clocks timers + quarter/half-frame
    /// events.
    pub fn tick_cpu_cycle(&mut self) {
        // NES APUs clock their timers once per APU cycle = every 2
        // CPU cycles for pulses. Match that here.
        self.frame_cycles = self.frame_cycles.wrapping_add(1);
        if self.frame_cycles & 1 == 0 {
            self.pulse_1.tick_timer();
            self.pulse_2.tick_timer();
        }
        // Quarter-frame cadence ≈ every 7457 CPU cycles. Four steps
        // per 4-step frame sequence; steps 0,1,2,3 trigger quarter-
        // frame; steps 1 and 3 also trigger half-frame (length+sweep).
        if self.frame_cycles >= 7457 {
            self.frame_cycles -= 7457;
            self.pulse_1.tick_envelope();
            self.pulse_2.tick_envelope();
            self.frame_step = (self.frame_step + 1) & 3;
            if self.frame_step == 1 || self.frame_step == 3 {
                self.pulse_1.tick_length_counter();
                self.pulse_2.tick_length_counter();
            }
        }
    }

    /// Current normalized output sample in [0.0, 1.0].
    pub fn mix(&self) -> f32 {
        let p1 = self.pulse_1.output() as f32 / 15.0;
        let p2 = self.pulse_2.output() as f32 / 15.0;
        let pcm = self.pcm_output as f32 / 127.0;
        // Match APU's pulse-heavy mix: pulses average, PCM adds
        // directly. Output capped at 1.0.
        let pulses = (p1 + p2) * 0.25; // each pulse contributes up to 0.25
        let sample = pulses + pcm * 0.25;
        sample.min(1.0)
    }

    /// Exposed read state for $5010 / $5015 queries (length status).
    pub fn read_status(&self) -> u8 {
        let mut v = 0;
        if self.pulse_1.length_counter > 0 {
            v |= 0x01;
        }
        if self.pulse_2.length_counter > 0 {
            v |= 0x02;
        }
        v
    }
}
