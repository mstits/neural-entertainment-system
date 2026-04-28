use once_cell::sync::Lazy;

use std::mem;

pub trait VideoSink {
    fn write_frame(&mut self, frame_buffer: &[u8]);
    fn frame_written(&self) -> bool;
    fn pixel_size(&self) -> usize;
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
}

impl<'a> Xrgb8888VideoSink<'a> {
    pub fn new(buffer: &'a mut [u32]) -> Self {
        Xrgb8888VideoSink {
            buffer,
            frame_written: false,
        }
    }
}

impl VideoSink for Xrgb8888VideoSink<'_> {
    fn write_frame(&mut self, frame_buffer: &[u8]) {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { xrgb8888_write_neon(frame_buffer, self.buffer) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (i, palette_index) in frame_buffer.iter().enumerate() {
                self.buffer[i] = XRGB8888_PALETTE[*palette_index as usize];
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
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xrgb8888_write_neon(frame_buffer: &[u8], buffer: &mut [u32]) {
    use core::arch::aarch64::*;
    let n = frame_buffer.len().min(buffer.len());
    let full = n & !15;
    let src = frame_buffer.as_ptr();
    let dst = buffer.as_mut_ptr();
    let palette = XRGB8888_PALETTE.as_ptr();
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
