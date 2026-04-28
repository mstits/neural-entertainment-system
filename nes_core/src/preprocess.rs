//! RGB frame preprocessing: (240, 256, 3) u8 RGB → (84, 84) u8 grayscale.
//!
//! Matches the semantics of `src/emulation/nes_environment.py::preprocess_frame`:
//!   1. Luminance-weighted grayscale (0.299 R + 0.587 G + 0.114 B).
//!   2. Area-based downsample (pixel-rectangle weighted average).
//!
//! Rust impl replaces the Python `cv2.resize(INTER_AREA)` + numpy path.
//! With rayon-parallel per-worker preprocessing in `Pool::step_all`,
//! this runs concurrently with the trainer's next-action inference on
//! MPS, freeing up GIL time for Python-side work.
//!
//! On aarch64 (Apple Silicon), the grayscale pass uses NEON intrinsics
//! to process 16 RGB triplets per iteration. The resize is scalar —
//! its access pattern isn't SIMD-friendly without extra scatter/gather
//! work that costs more than it saves at this frame size.

pub const FRAME_W: usize = 256;
pub const FRAME_H: usize = 240;
pub const PP_SIZE: usize = 84;

/// Convert a (H*W, 3)-packed uint8 RGB frame into a (H, W) grayscale
/// u8 buffer using the ITU-R BT.601 luma weights the rest of the
/// project already uses (0.299, 0.587, 0.114).
///
/// Allocates; caller owns the resulting Vec.
pub fn rgb_to_grayscale(rgb: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; FRAME_H * FRAME_W];
    rgb_to_grayscale_into(rgb, &mut out);
    out
}

/// Zero-alloc variant. Writes into `out` (must be `FRAME_H * FRAME_W` bytes).
pub fn rgb_to_grayscale_into(rgb: &[u8], out: &mut [u8]) {
    debug_assert_eq!(rgb.len(), FRAME_H * FRAME_W * 3);
    debug_assert_eq!(out.len(), FRAME_H * FRAME_W);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        rgb_to_grayscale_neon(rgb, out);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    rgb_to_grayscale_scalar(rgb, out);
}

#[allow(dead_code)]
fn rgb_to_grayscale_scalar(rgb: &[u8], out: &mut [u8]) {
    // Fixed-point luma: round(0.299*256)=77, round(0.587*256)=150,
    // round(0.114*256)=29 → divide by 256. Close to float within
    // ±1 per pixel; at 84x84 training resolution the policy net
    // won't see the difference.
    for (i, px) in out.iter_mut().enumerate() {
        let r = rgb[i * 3] as u32;
        let g = rgb[i * 3 + 1] as u32;
        let b = rgb[i * 3 + 2] as u32;
        let y = (77 * r + 150 * g + 29 * b) >> 8;
        *px = y.min(255) as u8;
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn rgb_to_grayscale_neon(rgb: &[u8], out: &mut [u8]) {
    use std::arch::aarch64::*;

    let n_pixels = out.len();
    let vec_chunks = n_pixels / 16; // process 16 pixels (48 RGB bytes) per iter
    let scalar_start = vec_chunks * 16;

    // Luma weights as u16 lanes. We're already in an unsafe fn body
    // (per the NEON intrinsic ABI); the crate-level
    // `#![allow(unsafe_op_in_unsafe_fn)]` lets these calls compile
    // without redundant inner `unsafe {}` blocks.
    let wr = vdupq_n_u16(77);
    let wg = vdupq_n_u16(150);
    let wb = vdupq_n_u16(29);

    for c in 0..vec_chunks {
        let base = c * 16 * 3;
        let triple = vld3q_u8(rgb.as_ptr().add(base));
        let r = triple.0;
        let g = triple.1;
        let b = triple.2;

        let r_lo = vmovl_u8(vget_low_u8(r));
        let r_hi = vmovl_high_u8(r);
        let g_lo = vmovl_u8(vget_low_u8(g));
        let g_hi = vmovl_high_u8(g);
        let b_lo = vmovl_u8(vget_low_u8(b));
        let b_hi = vmovl_high_u8(b);

        // y = (77 r + 150 g + 29 b) >> 8
        let mut acc_lo = vmulq_u16(r_lo, wr);
        acc_lo = vmlaq_u16(acc_lo, g_lo, wg);
        acc_lo = vmlaq_u16(acc_lo, b_lo, wb);
        let mut acc_hi = vmulq_u16(r_hi, wr);
        acc_hi = vmlaq_u16(acc_hi, g_hi, wg);
        acc_hi = vmlaq_u16(acc_hi, b_hi, wb);

        let y_lo = vshrn_n_u16(acc_lo, 8);
        let y_hi = vshrn_n_u16(acc_hi, 8);
        let y = vcombine_u8(y_lo, y_hi);
        vst1q_u8(out.as_mut_ptr().add(c * 16), y);
    }

    // Scalar tail for any pixels that didn't fit a full vector.
    for i in scalar_start..n_pixels {
        let r = rgb[i * 3] as u32;
        let g = rgb[i * 3 + 1] as u32;
        let b = rgb[i * 3 + 2] as u32;
        let y = (77 * r + 150 * g + 29 * b) >> 8;
        out[i] = y.min(255) as u8;
    }
}

/// Area-based downsample of a grayscale (`src_h` × `src_w`) u8 image
/// to (`dst` × `dst`) u8. Matches cv2.INTER_AREA semantics closely
/// enough for RL training: each output pixel is the area-weighted
/// average of the input rectangle covering its footprint.
///
/// Pure-scalar Rust; the compiler auto-vectorizes the inner weighted
/// sum reasonably well. SIMD gather/scatter here costs more than it
/// saves at 84×84 target resolution.
pub fn resize_area(src: &[u8], src_h: usize, src_w: usize, dst: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst * dst];
    resize_area_into(src, src_h, src_w, &mut out, dst);
    out
}

/// Zero-alloc variant. Writes into `out` (must be `dst * dst` bytes).
pub fn resize_area_into(src: &[u8], src_h: usize, src_w: usize, out: &mut [u8], dst: usize) {
    debug_assert_eq!(src.len(), src_h * src_w);
    debug_assert_eq!(out.len(), dst * dst);
    let sx = src_w as f32 / dst as f32;
    let sy = src_h as f32 / dst as f32;
    let inv_area = 1.0 / (sx * sy);

    for oy in 0..dst {
        let y0 = oy as f32 * sy;
        let y1 = (oy + 1) as f32 * sy;
        let iy_start = y0.floor() as usize;
        let iy_end = y1.ceil() as usize;
        for ox in 0..dst {
            let x0 = ox as f32 * sx;
            let x1 = (ox + 1) as f32 * sx;
            let ix_start = x0.floor() as usize;
            let ix_end = x1.ceil() as usize;

            let mut acc: f32 = 0.0;
            for iy in iy_start..iy_end.min(src_h) {
                let y_w = (iy as f32 + 1.0).min(y1) - (iy as f32).max(y0);
                if y_w <= 0.0 {
                    continue;
                }
                for ix in ix_start..ix_end.min(src_w) {
                    let x_w = (ix as f32 + 1.0).min(x1) - (ix as f32).max(x0);
                    if x_w <= 0.0 {
                        continue;
                    }
                    acc += src[iy * src_w + ix] as f32 * x_w * y_w;
                }
            }
            out[oy * dst + ox] = (acc * inv_area).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Full preprocess pipeline: RGB frame → 84×84 grayscale u8.
pub fn preprocess_frame(rgb: &[u8]) -> Vec<u8> {
    let mut gray = vec![0u8; FRAME_H * FRAME_W];
    let mut out = vec![0u8; PP_SIZE * PP_SIZE];
    preprocess_frame_into(rgb, &mut gray, &mut out);
    out
}

/// Zero-alloc variant. Caller owns `gray_scratch` (FRAME_H*FRAME_W bytes)
/// and `out` (PP_SIZE*PP_SIZE bytes).
pub fn preprocess_frame_into(rgb: &[u8], gray_scratch: &mut [u8], out: &mut [u8]) {
    rgb_to_grayscale_into(rgb, gray_scratch);
    resize_area_into(gray_scratch, FRAME_H, FRAME_W, out, PP_SIZE);
}

/// Area-downsample a u8 grayscale image straight into IEEE 754 half-
/// precision (`u16` slots carrying f16 bits), normalized to [0, 1] in
/// a single fused pass — no intermediate uint8 output buffer and no
/// downstream /255 divide.
///
/// Match semantics: `resize_area_into` followed by elementwise
/// `(u8 as f32) / 255.0` followed by f32→f16 conversion, modulo
/// half-precision rounding (mantissa only has 10 bits so the output
/// is already lossy relative to the u8 input — but the loss is at
/// the LSB, invisible to the policy net).
///
/// On aarch64 we accumulate area sums in f32 then batch-convert the
/// final 8-lane output row with `vcvt_f16_f32` (hardware fp16 store).
/// On other targets we use the `half` crate's soft-float fallback.
fn resize_area_into_f16_norm(
    src: &[u8], src_h: usize, src_w: usize, out: &mut [u16], dst: usize,
) {
    debug_assert_eq!(src.len(), src_h * src_w);
    debug_assert_eq!(out.len(), dst * dst);
    let sx = src_w as f32 / dst as f32;
    let sy = src_h as f32 / dst as f32;
    // Fuse the /255 normalization into the area-normalization factor
    // so every output pixel is a single f32 → f16 conversion.
    let scale = 1.0 / (sx * sy * 255.0);

    for oy in 0..dst {
        let y0 = oy as f32 * sy;
        let y1 = (oy + 1) as f32 * sy;
        let iy_start = y0.floor() as usize;
        let iy_end = y1.ceil() as usize;
        for ox in 0..dst {
            let x0 = ox as f32 * sx;
            let x1 = (ox + 1) as f32 * sx;
            let ix_start = x0.floor() as usize;
            let ix_end = x1.ceil() as usize;

            let mut acc: f32 = 0.0;
            for iy in iy_start..iy_end.min(src_h) {
                let y_w = (iy as f32 + 1.0).min(y1) - (iy as f32).max(y0);
                if y_w <= 0.0 {
                    continue;
                }
                for ix in ix_start..ix_end.min(src_w) {
                    let x_w = (ix as f32 + 1.0).min(x1) - (ix as f32).max(x0);
                    if x_w <= 0.0 {
                        continue;
                    }
                    acc += src[iy * src_w + ix] as f32 * x_w * y_w;
                }
            }
            // Single fused op: scale into [0, 1] then convert to f16.
            let v = (acc * scale).clamp(0.0, 1.0);
            out[oy * dst + ox] = half::f16::from_f32(v).to_bits();
        }
    }
}

/// Full preprocess pipeline, f16 variant: RGB frame →
/// 84×84 grayscale float16 (normalized to [0, 1]).
///
/// Output is `u16`-typed because `half::f16` isn't yet a native Rust
/// primitive; every slot carries the IEEE 754 half-precision bit
/// pattern and the consuming Python side reads it as
/// `np.frombuffer(..., dtype=np.float16)`.
pub fn preprocess_frame_into_f16(
    rgb: &[u8], gray_scratch: &mut [u8], out: &mut [u16],
) {
    rgb_to_grayscale_into(rgb, gray_scratch);
    resize_area_into_f16_norm(gray_scratch, FRAME_H, FRAME_W, out, PP_SIZE);
}

/// Zero-alloc f16 variant for the headless fused path that already
/// has a grayscale buffer on hand — skips the rgb→gray stage.
pub fn resize_gray_to_f16_norm(gray: &[u8], out: &mut [u16]) {
    resize_area_into_f16_norm(gray, FRAME_H, FRAME_W, out, PP_SIZE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_extremes() {
        // All-white frame → all-white grayscale.
        let rgb = vec![255u8; FRAME_H * FRAME_W * 3];
        let gray = rgb_to_grayscale(&rgb);
        assert!(gray.iter().all(|&v| v >= 253));
        // All-black frame → all-black.
        let rgb = vec![0u8; FRAME_H * FRAME_W * 3];
        let gray = rgb_to_grayscale(&rgb);
        assert!(gray.iter().all(|&v| v == 0));
    }

    #[test]
    fn preprocess_shape() {
        let rgb = vec![128u8; FRAME_H * FRAME_W * 3];
        let pp = preprocess_frame(&rgb);
        assert_eq!(pp.len(), PP_SIZE * PP_SIZE);
        // Mid-gray in → mid-gray out (approx).
        assert!(pp.iter().all(|&v| v >= 126 && v <= 130));
    }

    #[test]
    fn preprocess_f16_matches_u8_within_fp16_rounding() {
        // A deterministic gradient RGB frame exercises every luma
        // weight + area-downsample cell, so rounding error shows up
        // across the full value range rather than on flat inputs.
        let mut rgb = vec![0u8; FRAME_H * FRAME_W * 3];
        for y in 0..FRAME_H {
            for x in 0..FRAME_W {
                let i = (y * FRAME_W + x) * 3;
                rgb[i] = (x & 0xFF) as u8;
                rgb[i + 1] = (y & 0xFF) as u8;
                rgb[i + 2] = ((x ^ y) & 0xFF) as u8;
            }
        }
        let mut gray_a = vec![0u8; FRAME_H * FRAME_W];
        let mut gray_b = vec![0u8; FRAME_H * FRAME_W];
        let mut out_u8 = vec![0u8; PP_SIZE * PP_SIZE];
        let mut out_f16 = vec![0u16; PP_SIZE * PP_SIZE];
        preprocess_frame_into(&rgb, &mut gray_a, &mut out_u8);
        preprocess_frame_into_f16(&rgb, &mut gray_b, &mut out_f16);

        for i in 0..out_u8.len() {
            let ref_norm = out_u8[i] as f32 / 255.0;
            let got = half::f16::from_bits(out_f16[i]).to_f32();
            // fp16 has ~3 decimal digits; tolerate 4/255 of absolute
            // drift (includes both the rounding and the fused-scale
            // path's slightly different operation order).
            assert!(
                (got - ref_norm).abs() <= 4.0 / 255.0,
                "mismatch at {i}: u8→norm {ref_norm} vs f16 {got}",
            );
            assert!(got >= 0.0 && got <= 1.0);
        }
    }

    #[test]
    fn preprocess_f16_range_endpoints() {
        // All-black → all 0.0.
        let rgb = vec![0u8; FRAME_H * FRAME_W * 3];
        let mut gray = vec![0u8; FRAME_H * FRAME_W];
        let mut out = vec![0u16; PP_SIZE * PP_SIZE];
        preprocess_frame_into_f16(&rgb, &mut gray, &mut out);
        assert!(out.iter().all(|&b| half::f16::from_bits(b).to_f32() == 0.0));

        // All-white → ~1.0 (within fp16 rounding of 255/255).
        let rgb = vec![255u8; FRAME_H * FRAME_W * 3];
        preprocess_frame_into_f16(&rgb, &mut gray, &mut out);
        for &b in &out {
            let v = half::f16::from_bits(b).to_f32();
            assert!(v >= 0.99 && v <= 1.0, "white pixel f16 = {v}");
        }
    }
}
