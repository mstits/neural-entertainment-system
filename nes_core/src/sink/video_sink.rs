use once_cell::sync::Lazy;

use std::mem;

pub trait VideoSink {
    fn write_frame(&mut self, frame_buffer: &[u8]);
    fn frame_written(&self) -> bool;
    fn pixel_size(&self) -> usize;

    /// Set the PPUMASK color-emphasis bits (`0..=7`; bit 0 = red,
    /// bit 1 = green, bit 2 = blue) to apply on the next `write_frame`.
    /// Default no-op — only sinks that own the emphasis LUT honor it.
    /// Emphasis `0` (the reset default) yields byte-identical output to
    /// the base palette, so sinks that ignore this stay correct for the
    /// overwhelmingly common no-emphasis case.
    fn set_emphasis(&mut self, _emphasis: u8) {}
}

impl<S: VideoSink + ?Sized> VideoSink for Box<S> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        (**self).write_frame(frame_buffer);
    }

    fn frame_written(&self) -> bool {
        (**self).frame_written()
    }

    fn pixel_size(&self) -> usize {
        (**self).pixel_size()
    }

    fn set_emphasis(&mut self, emphasis: u8) {
        (**self).set_emphasis(emphasis);
    }
}

pub struct Rgb565VideoSink<'a> {
    buffer: &'a mut [u16],
    frame_written: bool,
}

impl<'a> Rgb565VideoSink<'a> {
    pub fn new(buffer: &'a mut [u16]) -> Self {
        Rgb565VideoSink {
            buffer,
            frame_written: false,
        }
    }
}

impl VideoSink for Rgb565VideoSink<'_> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        for (i, palette_index) in frame_buffer.iter().enumerate() {
            self.buffer[i] = RGB565_PALETTE[*palette_index as usize];
        }
        self.frame_written = true;
    }

    fn frame_written(&self) -> bool {
        self.frame_written
    }

    fn pixel_size(&self) -> usize {
        mem::size_of::<u16>()
    }
}

pub struct Xrgb1555VideoSink<'a> {
    buffer: &'a mut [u16],
    frame_written: bool,
}

impl<'a> Xrgb1555VideoSink<'a> {
    pub fn new(buffer: &'a mut [u16]) -> Self {
        Xrgb1555VideoSink {
            buffer,
            frame_written: false,
        }
    }
}

impl VideoSink for Xrgb1555VideoSink<'_> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        for (i, palette_index) in frame_buffer.iter().enumerate() {
            self.buffer[i] = XRGB1555_PALETTE[*palette_index as usize];
        }
        self.frame_written = true;
    }

    fn frame_written(&self) -> bool {
        self.frame_written
    }

    fn pixel_size(&self) -> usize {
        mem::size_of::<u16>()
    }
}

// Appropriate for use with javascript ImageData
pub struct WebVideoSink<'a> {
    buffer: &'a mut [u32],
    frame_written: bool,
}

impl<'a> WebVideoSink<'a> {
    pub fn new(buffer: &'a mut [u32]) -> Self {
        WebVideoSink {
            buffer,
            frame_written: false,
        }
    }
}

impl VideoSink for WebVideoSink<'_> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        for (i, palette_index) in frame_buffer.iter().enumerate() {
            self.buffer[i] = WEB_PALETTE[*palette_index as usize];
        }
        self.frame_written = true;
    }

    fn frame_written(&self) -> bool {
        self.frame_written
    }

    fn pixel_size(&self) -> usize {
        mem::size_of::<u32>()
    }
}

pub struct Xrgb8888VideoSink<'a> {
    buffer: &'a mut [u32],
    frame_written: bool,
    /// Color-emphasis selector (0..=7) for the current frame. Selects
    /// the 64-entry slice of `XRGB8888_EMPHASIS_PALETTE`. 0 == base
    /// palette (byte-identical to the historical output).
    emphasis: u8,
}

impl<'a> Xrgb8888VideoSink<'a> {
    pub fn new(buffer: &'a mut [u32]) -> Self {
        Xrgb8888VideoSink {
            buffer,
            frame_written: false,
            emphasis: 0,
        }
    }
}

impl VideoSink for Xrgb8888VideoSink<'_> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        // Select the emphasis slice. For emphasis 0 these 64 entries are
        // byte-identical to XRGB8888_PALETTE, so the default path is
        // unchanged; non-zero slices are the tinted/dimmed variants.
        let palette = &XRGB8888_EMPHASIS_PALETTE
            [(self.emphasis as usize & 0x07) * 64..(self.emphasis as usize & 0x07) * 64 + 64];
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { xrgb8888_write_neon(frame_buffer, self.buffer, palette.as_ptr()) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (i, palette_index) in frame_buffer.iter().enumerate() {
                self.buffer[i] = palette[*palette_index as usize];
            }
        }
        self.frame_written = true;
    }

    fn frame_written(&self) -> bool {
        self.frame_written
    }

    fn pixel_size(&self) -> usize {
        mem::size_of::<u32>()
    }

    fn set_emphasis(&mut self, emphasis: u8) {
        self.emphasis = emphasis & 0x07;
    }
}

/// `palette` must point at a run of at least 64 valid `u32` entries
/// (one 64-color emphasis slice of `XRGB8888_EMPHASIS_PALETTE`, or the
/// base `XRGB8888_PALETTE`). Every index in `frame_buffer` is a 6-bit
/// value (0..=63), so all lookups land inside that run.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xrgb8888_write_neon(frame_buffer: &[u8], buffer: &mut [u32], palette: *const u32) {
    use core::arch::aarch64::*;
    let n = frame_buffer.len().min(buffer.len());
    let full = n & !15;
    let src = frame_buffer.as_ptr();
    let dst = buffer.as_mut_ptr();
    // Gather 16 bytes of palette indices, expand 4x into 4 NEON u32x4
    // vectors by looking up each index in the palette, store.
    let mut i = 0usize;
    while i < full {
        // Load 16 palette indices.
        let indices: uint8x16_t = unsafe { vld1q_u8(src.add(i)) };
        let idx_arr: [u8; 16] = unsafe { core::mem::transmute(indices) };
        let v0 = [
            unsafe { *palette.add(idx_arr[0] as usize) },
            unsafe { *palette.add(idx_arr[1] as usize) },
            unsafe { *palette.add(idx_arr[2] as usize) },
            unsafe { *palette.add(idx_arr[3] as usize) },
        ];
        let v1 = [
            unsafe { *palette.add(idx_arr[4] as usize) },
            unsafe { *palette.add(idx_arr[5] as usize) },
            unsafe { *palette.add(idx_arr[6] as usize) },
            unsafe { *palette.add(idx_arr[7] as usize) },
        ];
        let v2 = [
            unsafe { *palette.add(idx_arr[8] as usize) },
            unsafe { *palette.add(idx_arr[9] as usize) },
            unsafe { *palette.add(idx_arr[10] as usize) },
            unsafe { *palette.add(idx_arr[11] as usize) },
        ];
        let v3 = [
            unsafe { *palette.add(idx_arr[12] as usize) },
            unsafe { *palette.add(idx_arr[13] as usize) },
            unsafe { *palette.add(idx_arr[14] as usize) },
            unsafe { *palette.add(idx_arr[15] as usize) },
        ];
        unsafe {
            vst1q_u32(dst.add(i), vld1q_u32(v0.as_ptr()));
            vst1q_u32(dst.add(i + 4), vld1q_u32(v1.as_ptr()));
            vst1q_u32(dst.add(i + 8), vld1q_u32(v2.as_ptr()));
            vst1q_u32(dst.add(i + 12), vld1q_u32(v3.as_ptr()));
        }
        i += 16;
    }
    // Tail: scalar for the residue (up to 15 pixels).
    while i < n {
        let idx = unsafe { *src.add(i) } as usize;
        unsafe { *dst.add(i) = *palette.add(idx) };
        i += 1;
    }
}

#[allow(clippy::unreadable_literal)]
pub static XRGB8888_PALETTE: &[u32] = &[
    0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC, 0x940084, 0xA80020, 0xA81000, 0x881400, 0x503000,
    0x007800, 0x006800, 0x005800, 0x004058, 0x000000, 0x000000, 0x000000, 0xBCBCBC, 0x0078F8,
    0x0058F8, 0x6844FC, 0xD800CC, 0xE40058, 0xF83800, 0xE45C10, 0xAC7C00, 0x00B800, 0x00A800,
    0x00A844, 0x008888, 0x000000, 0x000000, 0x000000, 0xF8F8F8, 0x3CBCFC, 0x6888FC, 0x9878F8,
    0xF878F8, 0xF85898, 0xF87858, 0xFCA044, 0xF8B800, 0xB8F818, 0x58D854, 0x58F898, 0x00E8D8,
    0x787878, 0x000000, 0x000000, 0xFCFCFC, 0xA4E4FC, 0xB8B8F8, 0xD8B8F8, 0xF8B8F8, 0xF8A4C0,
    0xF0D0B0, 0xFCE0A8, 0xF8D878, 0xD8F878, 0xB8F8B8, 0xB8F8D8, 0x00FCFC, 0xF8D8F8, 0x000000,
    0x000000,
];

/// 512-entry XRGB8888 palette covering all 8 color-emphasis
/// combinations, indexed `emphasis * 64 + palette_index`. Emphasis
/// bit 0 = red, bit 1 = green, bit 2 = blue (matching PPUMASK bits
/// 5/6/7 shifted down). Each active emphasis bit attenuates the two
/// channels it does *not* emphasize; the attenuation is multiplicative,
/// so a single bit tints the picture toward that color while all three
/// bits dim every channel (the "pause dim" / fade effect). Slice
/// `[0..64]` is byte-identical to `XRGB8888_PALETTE`, keeping the
/// no-emphasis render path unchanged.
///
/// The attenuation factor (~0.746) is the Blargg-measured value cited
/// on the NESdev wiki. There is no exact RGB reference to match here:
/// nes-py/LaiNES omits emphasis entirely, and Mesen decodes through a
/// different base palette + NTSC filter, so the LUT is validated
/// structurally (see the unit tests) rather than by pixel-diff.
pub static XRGB8888_EMPHASIS_PALETTE: Lazy<[u32; 512]> = Lazy::new(|| {
    const ATTEN: f64 = 0.746;
    let mut lut = [0u32; 512];
    for emphasis in 0..8usize {
        let mut rf = 1.0f64;
        let mut gf = 1.0f64;
        let mut bf = 1.0f64;
        if emphasis & 0x01 != 0 {
            gf *= ATTEN;
            bf *= ATTEN;
        }
        if emphasis & 0x02 != 0 {
            rf *= ATTEN;
            bf *= ATTEN;
        }
        if emphasis & 0x04 != 0 {
            rf *= ATTEN;
            gf *= ATTEN;
        }
        for idx in 0..64usize {
            let color = XRGB8888_PALETTE[idx];
            let r = (color >> 16) & 0xFF;
            let g = (color >> 8) & 0xFF;
            let b = color & 0xFF;
            // factor == 1.0 for the non-emphasized channels leaves the
            // byte exact, so emphasis 0 reproduces XRGB8888_PALETTE.
            let r = ((r as f64 * rf).round() as u32).min(0xFF);
            let g = ((g as f64 * gf).round() as u32).min(0xFF);
            let b = ((b as f64 * bf).round() as u32).min(0xFF);
            lut[emphasis * 64 + idx] = (r << 16) | (g << 8) | b;
        }
    }
    lut
});

static XRGB1555_PALETTE: Lazy<[u16; 64]> = Lazy::new(|| {
    let mut palette = [0; 64];
    for n in 0..64 {
        let color = XRGB8888_PALETTE[n];
        let r = ((color >> 19) & 0x1F) as u16;
        let g = ((color >> 11) & 0x1F) as u16;
        let b = ((color >> 3) & 0x1F) as u16;
        palette[n] = (r << 10) | (g << 5) | b;
    }
    palette
});

static RGB565_PALETTE: Lazy<[u16; 64]> = Lazy::new(|| {
    let mut palette = [0; 64];
    for n in 0..64 {
        let color = XRGB8888_PALETTE[n];
        let r = ((color >> 19) & 0x1F) as u16;
        let g = ((color >> 10) & 0x3F) as u16;
        let b = ((color >> 3) & 0x1F) as u16;
        palette[n] = (r << 11) | (g << 5) | b;
    }
    palette
});

static WEB_PALETTE: Lazy<[u32; 64]> = Lazy::new(|| {
    let mut palette = [0; 64];
    for n in 0..64 {
        let color = XRGB8888_PALETTE[n];
        let r = (color >> 16) & 0xFF;
        let g = (color >> 8) & 0xFF;
        let b = color & 0xFF;
        palette[n] = 0xFF00_0000 | (b << 16) | (g << 8) | r;
    }
    palette
});

#[cfg(test)]
mod emphasis_tests {
    use super::*;

    fn channels(px: u32) -> (u32, u32, u32) {
        ((px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)
    }

    #[test]
    fn emphasis_zero_slice_is_byte_identical_to_base_palette() {
        for idx in 0..64 {
            assert_eq!(
                XRGB8888_EMPHASIS_PALETTE[idx], XRGB8888_PALETTE[idx],
                "emphasis-0 entry {idx} diverged from base palette"
            );
        }
    }

    #[test]
    fn default_sink_write_matches_base_palette() {
        // A sink with no set_emphasis call must reproduce the historical
        // full-color output exactly (parity-critical default path).
        let frame: Vec<u8> = (0..256u32).map(|i| (i % 64) as u8).collect();
        let mut out = vec![0u32; frame.len()];
        {
            let mut sink = Xrgb8888VideoSink::new(&mut out);
            sink.write_frame(&frame);
        }
        for (i, &idx) in frame.iter().enumerate() {
            assert_eq!(out[i], XRGB8888_PALETTE[idx as usize]);
        }
    }

    #[test]
    fn red_emphasis_preserves_red_and_dims_green_blue() {
        // emphasis bit 0 (red) leaves R untouched, attenuates G and B.
        for idx in 0..64 {
            let base = XRGB8888_PALETTE[idx];
            let emp = XRGB8888_EMPHASIS_PALETTE[64 + idx];
            let (br, bg, bb) = channels(base);
            let (er, eg, eb) = channels(emp);
            assert_eq!(er, br, "red channel changed under red emphasis (idx {idx})");
            assert!(eg <= bg, "green not attenuated under red emphasis (idx {idx})");
            assert!(eb <= bb, "blue not attenuated under red emphasis (idx {idx})");
            // Non-zero channels must actually drop (not just stay equal).
            if bg > 3 {
                assert!(eg < bg, "green unchanged for a bright color (idx {idx})");
            }
        }
    }

    #[test]
    fn green_and_blue_emphasis_target_correct_channels() {
        for idx in 0..64 {
            let base = XRGB8888_PALETTE[idx];
            let (br, bg, bb) = channels(base);
            // green emphasis (bit 1) → slice 2
            let (gr, gg, gb) = channels(XRGB8888_EMPHASIS_PALETTE[2 * 64 + idx]);
            assert_eq!(gg, bg, "green channel changed under green emphasis (idx {idx})");
            assert!(gr <= br && gb <= bb);
            // blue emphasis (bit 2) → slice 4
            let (b2r, b2g, b2b) = channels(XRGB8888_EMPHASIS_PALETTE[4 * 64 + idx]);
            assert_eq!(b2b, bb, "blue channel changed under blue emphasis (idx {idx})");
            assert!(b2r <= br && b2g <= bg);
        }
    }

    #[test]
    fn all_emphasis_dims_every_channel() {
        // All three bits set → each channel attenuated by the two others
        // (the darkening / fade case). No channel may exceed base.
        for idx in 0..64 {
            let base = XRGB8888_PALETTE[idx];
            let emp = XRGB8888_EMPHASIS_PALETTE[7 * 64 + idx];
            let (br, bg, bb) = channels(base);
            let (er, eg, eb) = channels(emp);
            assert!(er <= br && eg <= bg && eb <= bb, "idx {idx} not dimmed");
            if br > 3 {
                assert!(er < br, "red not dimmed for bright color (idx {idx})");
            }
        }
    }

    #[test]
    fn sink_honors_set_emphasis() {
        let frame = vec![0x30u8; 8]; // a bright grey-column entry
        let mut out = vec![0u32; frame.len()];
        {
            let mut sink = Xrgb8888VideoSink::new(&mut out);
            sink.set_emphasis(0b111); // all emphasis
            sink.write_frame(&frame);
        }
        assert_eq!(out[0], XRGB8888_EMPHASIS_PALETTE[7 * 64 + 0x30]);
        assert_ne!(out[0], XRGB8888_PALETTE[0x30]);
    }

    #[test]
    fn set_emphasis_masks_to_three_bits() {
        // Only the low 3 bits select a slice; stray high bits are ignored.
        let frame = vec![0x21u8; 4];
        let mut out = vec![0u32; frame.len()];
        {
            let mut sink = Xrgb8888VideoSink::new(&mut out);
            sink.set_emphasis(0xFF);
            sink.write_frame(&frame);
        }
        assert_eq!(out[0], XRGB8888_EMPHASIS_PALETTE[7 * 64 + 0x21]);
    }
}
