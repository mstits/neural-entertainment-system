pub trait AudioSink {
    fn write_sample(&mut self, sample: f32);
    fn samples_written(&self) -> usize;
}

impl<A: AudioSink + ?Sized> AudioSink for Box<A> {
    fn write_sample(&mut self, sample: f32) {
        (**self).write_sample(sample);
    }

    fn samples_written(&self) -> usize {
        (**self).samples_written()
    }
}

pub struct AudioSinkF32<'a> {
    buffer: &'a mut [(f32, f32)],
    buffer_pos: usize,
}

impl<'a> AudioSinkF32<'a> {
    pub fn new(buffer: &'a mut [(f32, f32)]) -> Self {
        AudioSinkF32 {
            buffer,
            buffer_pos: 0,
        }
    }
}

impl AudioSink for AudioSinkF32<'_> {
    fn write_sample(&mut self, sample: f32) {
        self.buffer[self.buffer_pos] = (sample, sample);
        self.buffer_pos += 1;
    }

    fn samples_written(&self) -> usize {
        self.buffer_pos
    }
}

pub struct AudioSinkI16<'a> {
    buffer: &'a mut [(i16, i16)],
    buffer_pos: usize,
}

impl<'a> AudioSinkI16<'a> {
    pub fn new(buffer: &'a mut [(i16, i16)]) -> Self {
        AudioSinkI16 {
            buffer,
            buffer_pos: 0,
        }
    }
}

impl AudioSink for AudioSinkI16<'_> {
    fn write_sample(&mut self, sample: f32) {
        // Clamp before cast: an `as i16` from an f32 outside [-32768, 32767]
        // saturates on AArch64 but is technically platform-defined for
        // non-finite inputs. Clamp to [-1, 1] then scale to the i16 range
        // (32767, not 32768, so +1.0 maps to i16::MAX without overflow).
        let sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        self.buffer[self.buffer_pos] = (sample, sample);
        self.buffer_pos += 1;
    }

    fn samples_written(&self) -> usize {
        self.buffer_pos
    }
}

pub struct AudioSinkU16<'a> {
    buffer: &'a mut [(u16, u16)],
    buffer_pos: usize,
}

impl<'a> AudioSinkU16<'a> {
    pub fn new(buffer: &'a mut [(u16, u16)]) -> Self {
        AudioSinkU16 {
            buffer,
            buffer_pos: 0,
        }
    }
}

impl AudioSink for AudioSinkU16<'_> {
    fn write_sample(&mut self, sample: f32) {
        // Clamp before cast (see AudioSinkI16). Mid-point biases u16 so
        // 0.0 → 0x8000, ±1.0 → endpoints.
        let scaled = (sample.clamp(-1.0, 1.0) * 32767.0) + 32768.0;
        let sample = scaled.clamp(0.0, u16::MAX as f32) as u16;
        self.buffer[self.buffer_pos] = (sample, sample);
        self.buffer_pos += 1;
    }

    fn samples_written(&self) -> usize {
        self.buffer_pos
    }
}
