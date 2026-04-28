// Konami VRC6 extra audio: 2 square channels + 1 sawtooth.
//
// Square channels ($9000-$9002, $A000-$A002) have a 4-bit volume,
// 3-bit duty, and 12-bit period. The "mode" bit (bit 7 of $x000)
// forces the square output to constant volume regardless of duty —
// Konami uses it for PCM-like waveforms.
//
// Sawtooth ($B000-$B002) has a 6-bit accumulator-rate register. Each
// 14-step period it accumulates the rate; output = (accumulator >> 3)
// & 0x1F (5-bit DAC). Every 7 ticks the accumulator resets.
//
// Global control at $9003 ($04 = halt, bits 4-5 = frequency divider
// that shifts period left by 4 or 8 bits — used for bass range).

use serde_derive::{Deserialize, Serialize};

const DUTY_TABLE: &[[u8; 16]] = &[
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
];

#[derive(Clone, Deserialize, Serialize)]
pub struct Vrc6Pulse {
    volume: u8,
    duty_mode: u8,
    pcm_mode: bool,
    timer_period: u16,
    timer_counter: u16,
    duty_position: u8,
    enabled: bool,
}

impl Vrc6Pulse {
    fn new() -> Self {
        Self {
            volume: 0,
            duty_mode: 0,
            pcm_mode: false,
            timer_period: 0,
            timer_counter: 0,
            duty_position: 0,
            enabled: false,
        }
    }

    fn write_ctrl(&mut self, value: u8) {
        self.volume = value & 0x0F;
        self.duty_mode = (value >> 4) & 0x07;
        self.pcm_mode = value & 0x80 != 0;
    }

    fn write_period_lo(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x0F00) | value as u16;
    }

    fn write_period_hi(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x00FF) | (((value as u16) & 0x0F) << 8);
        self.enabled = value & 0x80 != 0;
        if !self.enabled {
            self.duty_position = 0;
        }
    }

    fn tick(&mut self, freq_shift: u8) {
        if !self.enabled {
            return;
        }
        let period = self.timer_period << freq_shift;
        if self.timer_counter == 0 {
            self.timer_counter = period;
            self.duty_position = (self.duty_position + 1) & 0x0F;
        } else {
            self.timer_counter -= 1;
        }
    }

    fn output(&self) -> u8 {
        if !self.enabled {
            return 0;
        }
        if self.pcm_mode
            || DUTY_TABLE[self.duty_mode as usize][self.duty_position as usize] == 1
        {
            self.volume
        } else {
            0
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Vrc6Saw {
    rate: u8,
    timer_period: u16,
    timer_counter: u16,
    accumulator: u8,
    step: u8,
    enabled: bool,
}

impl Vrc6Saw {
    fn new() -> Self {
        Self {
            rate: 0,
            timer_period: 0,
            timer_counter: 0,
            accumulator: 0,
            step: 0,
            enabled: false,
        }
    }

    fn write_rate(&mut self, value: u8) {
        self.rate = value & 0x3F;
    }

    fn write_period_lo(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x0F00) | value as u16;
    }

    fn write_period_hi(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x00FF) | (((value as u16) & 0x0F) << 8);
        self.enabled = value & 0x80 != 0;
        if !self.enabled {
            self.accumulator = 0;
            self.step = 0;
        }
    }

    fn tick(&mut self, freq_shift: u8) {
        if !self.enabled {
            return;
        }
        let period = self.timer_period << freq_shift;
        if self.timer_counter == 0 {
            self.timer_counter = period;
            // 14-step accumulator: every 2 ticks adds rate; every 7
            // pairs resets. Simplified per NesDev reference.
            self.step = (self.step + 1) % 14;
            if self.step == 0 {
                self.accumulator = 0;
            } else if self.step & 1 == 0 {
                self.accumulator = self.accumulator.wrapping_add(self.rate);
            }
        } else {
            self.timer_counter -= 1;
        }
    }

    fn output(&self) -> u8 {
        if !self.enabled {
            0
        } else {
            (self.accumulator >> 3) & 0x1F
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Vrc6Audio {
    pub pulse_1: Vrc6Pulse,
    pub pulse_2: Vrc6Pulse,
    pub sawtooth: Vrc6Saw,
    // $9003 bits: 0 = halt (freeze all channels), 1 = 1x freq
    // (normal), 2 = 16x slower (<<4), 3 = 256x slower (<<8).
    halt: bool,
    freq_shift: u8,
}

impl Vrc6Audio {
    pub fn new() -> Self {
        Self {
            pulse_1: Vrc6Pulse::new(),
            pulse_2: Vrc6Pulse::new(),
            sawtooth: Vrc6Saw::new(),
            halt: false,
            freq_shift: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Handle a write to one of VRC6's audio register addresses
    /// after the mapper has already applied its own pin remap
    /// (board-dependent swap of A0/A1 on mapper 26). Returns true
    /// when the address matched an audio register.
    pub fn write_register(&mut self, canonical_addr: u16, value: u8) -> bool {
        let high = canonical_addr & 0xF000;
        let low = (canonical_addr & 0x0003) as u8;
        match (high, low) {
            (0x9000, 0) => {
                self.pulse_1.write_ctrl(value);
                true
            }
            (0x9000, 1) => {
                self.pulse_1.write_period_lo(value);
                true
            }
            (0x9000, 2) => {
                self.pulse_1.write_period_hi(value);
                true
            }
            (0x9000, 3) => {
                self.halt = value & 0x01 != 0;
                self.freq_shift = match (value >> 1) & 0x03 {
                    0 => 0,
                    1 => 4,
                    _ => 8,
                };
                true
            }
            (0xA000, 0) => {
                self.pulse_2.write_ctrl(value);
                true
            }
            (0xA000, 1) => {
                self.pulse_2.write_period_lo(value);
                true
            }
            (0xA000, 2) => {
                self.pulse_2.write_period_hi(value);
                true
            }
            (0xB000, 0) => {
                self.sawtooth.write_rate(value);
                true
            }
            (0xB000, 1) => {
                self.sawtooth.write_period_lo(value);
                true
            }
            (0xB000, 2) => {
                self.sawtooth.write_period_hi(value);
                true
            }
            _ => false,
        }
    }

    pub fn tick_cpu_cycle(&mut self) {
        if self.halt {
            return;
        }
        self.pulse_1.tick(self.freq_shift);
        self.pulse_2.tick(self.freq_shift);
        self.sawtooth.tick(self.freq_shift);
    }

    pub fn mix(&self) -> f32 {
        // Each pulse outputs 0-15, saw outputs 0-31. Normalize so
        // each contributes roughly 1/3 of full scale; total clipped
        // to 1.0. Matches the well-known Konami mix ratios within
        // the margin of our single-stream output.
        let p1 = self.pulse_1.output() as f32 / 15.0;
        let p2 = self.pulse_2.output() as f32 / 15.0;
        let saw = self.sawtooth.output() as f32 / 31.0;
        ((p1 + p2) * 0.2 + saw * 0.2).min(1.0)
    }
}
