// Suppress Rust 2024 edition's `unsafe_op_in_unsafe_fn` lint for the
// whole crate. PyO3 0.22's macro expansion generates code that
// triggers 74+ instances of this lint per build; PyO3 0.23 fixes the
// macro. Until we upgrade PyO3, silence the noise here so real issues
// in our own code stay visible.
#![allow(unsafe_op_in_unsafe_fn)]

pub mod apu;
pub mod cartridge;
pub mod cpu;
pub mod game_genie;
pub mod input;
pub mod mapper;
pub mod memory;
pub mod nes;
pub mod oam_dma;
pub mod ppu;
pub mod ppu_neon;
pub mod serialize;
pub mod sink;
pub mod system_bus;

// `preprocess`, `rewards`, and `depth_tracker` are pure Rust (no PyO3
// deps) — available to all builds so `cargo test` can exercise them
// without needing a Python interpreter at link time.
mod depth_tracker;
mod narrator;
pub mod preprocess;
mod rewards;
mod smb_tile_extract;

// AArch64 ASM 6502 core — Phase 0 skeleton. Only compiled when the
// `asm_cpu` feature is enabled AND we're on aarch64. See
// `docs/proposals/aarch64_cpu_asm.md`.
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
pub mod cpu_asm;
#[cfg(feature = "python")]
mod audio;
#[cfg(feature = "metal")]
pub mod metal_render;
#[cfg(feature = "python")]
mod pool;
#[cfg(feature = "python")]
mod python;
