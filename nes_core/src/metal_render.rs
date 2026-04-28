//! Metal compute pipeline for PPU render offload.
//!
//! **Scope (v1, this pass):** palette-expansion kernel only —
//! `&[u8]` palette indices (256×240 per frame) → `&mut [u32]` XRGB8888.
//! Matches the behaviour of the NEON scalar path in `sink/video_sink.rs`
//! exactly; the Rust side stays authoritative for PPU state + frame
//! composition, the GPU just does the final lookup.
//!
//! **Out of scope here** (planned v2): tile-fetch kernel, sprite-0
//! detection, OAM evaluation, priority mixing. Those need CPU-GPU
//! synchronization on scanline-accurate timing (mapper IRQs fire
//! mid-frame); designing that boundary is a separate pass.
//!
//! **Why this small scope is still useful**: it proves the Metal
//! plumbing end-to-end — device selection, pipeline caching, buffer
//! round-trip, byte-exactness vs CPU path — and is extensible. Once
//! the tile-decode kernel lands we can reuse this same `MetalRenderer`
//! pattern.
//!
//! **Correctness check**: every frame emitted by the Metal kernel
//! must be byte-identical to the NEON path for the same palette
//! index array. Test harness in the `tests` module does this against
//! the full XRGB8888 palette.
//!
//! **Performance note (measured 2026-04-20, M4 Max)**: the v1
//! palette-expand kernel ALONE is **10× slower** than the auto-
//! vectorized CPU path (110 μs/frame GPU vs 10.6 μs/frame CPU). The
//! 100 μs per dispatch is almost entirely Metal submission + wait
//! overhead; the actual compute is nanoseconds. Don't use v1 in
//! production — ship v2 (fused tile-decode + sprite-eval + palette
//! in one kernel, one dispatch per frame instead of one per pass)
//! or nothing. See `examples/bench_metal_vs_neon.rs`.

#![cfg(feature = "metal")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::sink::XRGB8888_PALETTE;

/// Embedded Metal Shading Language for the palette-expansion kernel.
/// Input buffer is a flat u8 palette-index array (length = frame
/// pixels). Output buffer is u32 XRGB8888 (same length). The palette
/// is a constant 64-entry u32 array uploaded once at pipeline init.
const PALETTE_EXPAND_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void palette_expand(
    device const uchar *src       [[ buffer(0) ]],
    device uint        *dst       [[ buffer(1) ]],
    constant uint      *palette   [[ buffer(2) ]],
    constant uint      &n_pixels  [[ buffer(3) ]],
    uint               gid        [[ thread_position_in_grid ]]
) {
    if (gid >= n_pixels) { return; }
    dst[gid] = palette[src[gid] & 0x3F];
}
"#;

/// Stateful Metal compute renderer. Owns the device / command queue /
/// pipeline state. Reusable across frames + workers; internally
/// thread-safe via a `Mutex` around the command-queue path (one
/// Metal submission at a time, matching the single-GPU hardware).
pub struct MetalRenderer {
    inner: Mutex<Inner>,
}

struct Inner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    palette_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Reused scratch buffers sized to the max frame we've seen.
    src_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    dst_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_capacity: usize,
}

impl MetalRenderer {
    /// Build a renderer using the system's default Metal device.
    /// Returns `None` if the device / pipeline can't be created; the
    /// caller should then fall back to the CPU render path.
    pub fn new() -> Option<Self> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> = unsafe {
            let ptr = MTLCreateSystemDefaultDevice();
            Retained::retain(ptr)?
        };
        let queue = device.newCommandQueue()?;

        // Compile the MSL source into a library, then fetch the
        // kernel function.
        let source_ns = NSString::from_str(PALETTE_EXPAND_MSL);
        let library = device
            .newLibraryWithSource_options_error(&source_ns, None)
            .ok()?;
        let fn_name = NSString::from_str("palette_expand");
        let function = library.newFunctionWithName(&fn_name)?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;

        // Upload the 64-entry palette once. StorageModeShared keeps
        // the buffer CPU-visible so we don't need an explicit blit;
        // on M-series unified memory this is the right choice.
        let palette_bytes: &[u32] = XRGB8888_PALETTE;
        let palette_buf = unsafe {
            let ptr = NonNull::new(palette_bytes.as_ptr() as *mut c_void)
                .ok_or(())
                .ok()?;
            device.newBufferWithBytes_length_options(
                ptr,
                (palette_bytes.len() * std::mem::size_of::<u32>()) as usize,
                MTLResourceOptions::MTLResourceStorageModeShared,
            )?
        };

        Some(Self {
            inner: Mutex::new(Inner {
                device,
                queue,
                pipeline,
                palette_buf,
                src_buf: None,
                dst_buf: None,
                buf_capacity: 0,
            }),
        })
    }

    /// Expand `src` (u8 palette indices, 0-63) into `dst`
    /// (XRGB8888 u32 pixels). Must satisfy `dst.len() == src.len()`.
    /// Blocks until the GPU completes the kernel.
    pub fn palette_expand(&self, src: &[u8], dst: &mut [u32]) -> Result<(), String> {
        debug_assert_eq!(src.len(), dst.len());
        let mut inner = self.inner.lock().unwrap();
        inner.palette_expand(src, dst)
    }
}

impl Inner {
    fn ensure_buffers(&mut self, n_pixels: usize) -> Result<(), String> {
        if n_pixels <= self.buf_capacity && self.src_buf.is_some() && self.dst_buf.is_some() {
            return Ok(());
        }
        // Grow to 1.5× the requested size to amortize allocation
        // across resolution changes (training doesn't resize, but
        // the renderer is resolution-generic).
        let cap = (n_pixels * 3 / 2).max(256 * 240);
        let src_len = cap * std::mem::size_of::<u8>();
        let dst_len = cap * std::mem::size_of::<u32>();
        self.src_buf = Some(
            self.device
                .newBufferWithLength_options(
                    src_len,
                    MTLResourceOptions::MTLResourceStorageModeShared,
                )
                .ok_or_else(|| "failed to allocate Metal src buffer".to_string())?,
        );
        self.dst_buf = Some(
            self.device
                .newBufferWithLength_options(
                    dst_len,
                    MTLResourceOptions::MTLResourceStorageModeShared,
                )
                .ok_or_else(|| "failed to allocate Metal dst buffer".to_string())?,
        );
        self.buf_capacity = cap;
        Ok(())
    }

    fn palette_expand(&mut self, src: &[u8], dst: &mut [u32]) -> Result<(), String> {
        let n = src.len();
        if n == 0 {
            return Ok(());
        }
        self.ensure_buffers(n)?;
        let src_buf = self.src_buf.as_ref().unwrap();
        let dst_buf = self.dst_buf.as_ref().unwrap();

        // Copy src into shared buffer. On M-series unified memory
        // this is a raw memcpy to mapped GPU memory.
        unsafe {
            let contents = src_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src.as_ptr(), contents, n);
        }

        // Encode + submit the compute dispatch.
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| "commandBuffer nil".to_string())?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| "computeCommandEncoder nil".to_string())?;
        unsafe {
            enc.setComputePipelineState(&self.pipeline);
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.palette_buf), 0, 2);
            let n_u32 = n as u32;
            let n_ptr = NonNull::from(&n_u32).cast::<c_void>();
            enc.setBytes_length_atIndex(n_ptr, std::mem::size_of::<u32>(), 3);
            // Dispatch enough threads to cover every pixel. Use the
            // pipeline's execution-width as threads-per-threadgroup
            // for optimal occupancy.
            let tew = self.pipeline.threadExecutionWidth();
            let tpt = MTLSize {
                width: tew,
                height: 1,
                depth: 1,
            };
            let grid = MTLSize {
                width: n,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tpt);
        }
        enc.endEncoding();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted() };

        // Read dst back. On unified memory this is just a memcpy
        // from GPU-visible mapped RAM into the caller's Vec.
        unsafe {
            let contents = dst_buf.contents().as_ptr() as *const u32;
            std::ptr::copy_nonoverlapping(contents, dst.as_mut_ptr(), n);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU output must byte-match the CPU XRGB8888 palette path for
    /// every palette index. This is the correctness gate — if it
    /// passes, the kernel is safe to use in production.
    #[test]
    fn byte_exact_vs_cpu_palette() {
        let Some(renderer) = MetalRenderer::new() else {
            eprintln!("[metal_render] no Metal device available; skipping test");
            return;
        };
        // Build a deterministic input: every 3rd byte cycles palette
        // 0-63. This exercises every index at least once.
        let n_pixels = 256 * 240;
        let mut src = vec![0u8; n_pixels];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 64) as u8;
        }
        let mut gpu = vec![0u32; n_pixels];
        renderer.palette_expand(&src, &mut gpu).expect("gpu ok");

        let mut cpu = vec![0u32; n_pixels];
        for (i, &idx) in src.iter().enumerate() {
            cpu[i] = XRGB8888_PALETTE[(idx & 0x3F) as usize];
        }

        for i in 0..n_pixels {
            assert_eq!(
                gpu[i], cpu[i],
                "pixel {i}: gpu {:08x} vs cpu {:08x} (palette idx {})",
                gpu[i], cpu[i], src[i]
            );
        }
    }

    /// Two passes on the same renderer exercise buffer reuse.
    #[test]
    fn buffer_reuse_is_byte_exact() {
        let Some(renderer) = MetalRenderer::new() else {
            return;
        };
        let n = 256 * 240;
        let src = vec![13u8; n];
        let mut a = vec![0u32; n];
        let mut b = vec![0u32; n];
        renderer.palette_expand(&src, &mut a).unwrap();
        renderer.palette_expand(&src, &mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0], XRGB8888_PALETTE[13]);
    }
}

// On non-macOS builds / when `metal` feature is off, the module
// evaporates. Callers should cfg-gate their use:
//
//   #[cfg(feature = "metal")]
//   use crate::metal_render::MetalRenderer;
