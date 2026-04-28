// Head-to-head: GPU palette expansion vs NEON palette expansion on
// the same 256×240 frame. The NEON path is already the fast CPU path
// shipped in `sink/video_sink.rs::xrgb8888_write_neon`. This bench
// just confirms my hunch: for palette expansion ALONE, GPU dispatch
// overhead drowns out the compute win. The real Metal value comes
// from fusing tile-decode + palette-expand + sprite-mix into one
// kernel (v2 work).

use nes_core::sink::XRGB8888_PALETTE;
use std::time::Instant;

#[cfg(feature = "metal")]
use nes_core::metal_render::MetalRenderer;

const FRAME_PIXELS: usize = 256 * 240;
const ITERS: usize = 2_000;

fn neon_palette(src: &[u8], dst: &mut [u32]) {
    // Mirror the loop used inside Xrgb8888VideoSink::write_frame on
    // non-aarch64 builds; on aarch64 the compiler auto-vectorizes
    // this tight loop into NEON tbl+store pretty well.
    for (i, &idx) in src.iter().enumerate() {
        dst[i] = XRGB8888_PALETTE[(idx & 0x3F) as usize];
    }
}

fn main() {
    // Deterministic input: cycle 0..63 across the frame.
    let mut src = vec![0u8; FRAME_PIXELS];
    for (i, b) in src.iter_mut().enumerate() {
        *b = (i % 64) as u8;
    }

    // Warm up CPU path.
    let mut dst = vec![0u32; FRAME_PIXELS];
    neon_palette(&src, &mut dst);

    // Bench CPU.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        neon_palette(&src, &mut dst);
    }
    let cpu_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!(
        "CPU (scalar/NEON-auto-vec): {cpu_ns} ns/frame  ({:.0} Mpx/s)",
        FRAME_PIXELS as f64 / cpu_ns as f64 * 1e3
    );

    #[cfg(feature = "metal")]
    {
        let Some(renderer) = MetalRenderer::new() else {
            println!("Metal device unavailable");
            return;
        };
        // Warm up GPU path (first submission is slow — shader
        // compilation, pipeline cache miss).
        renderer.palette_expand(&src, &mut dst).unwrap();

        let t0 = Instant::now();
        for _ in 0..ITERS {
            renderer.palette_expand(&src, &mut dst).unwrap();
        }
        let gpu_ns = t0.elapsed().as_nanos() / ITERS as u128;
        println!(
            "GPU (Metal compute kernel): {gpu_ns} ns/frame  ({:.0} Mpx/s)",
            FRAME_PIXELS as f64 / gpu_ns as f64 * 1e3
        );
        let ratio = cpu_ns as f64 / gpu_ns as f64;
        if ratio >= 1.0 {
            println!("GPU is {ratio:.2}× faster than CPU");
        } else {
            println!("GPU is {:.2}× SLOWER than CPU (submission overhead)", 1.0 / ratio);
        }
    }
}
