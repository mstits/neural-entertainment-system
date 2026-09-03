//! Jackal (USA) boot regression tests, backed by the real ROM dump under
//! `roms/`.
//!
//! Jackal is a UxROM (mapper 2) title whose NMI handler contains a
//! `LDA $2002 / BPL` vblank spin ($D0E0, reached from the $D0D6 nametable
//! upload) that runs with vblank NMI still enabled. Under the crate's
//! LaiNES-parity cycle-0 PPU-register read commit that spin can never
//! observe a set flag: the read resolves at the instruction boundary, and
//! a flag that is up at the boundary means the NMI edge is already
//! latched, so the NMI is serviced first and its own `$2002` read at
//! $C298 drains the flag before the poll runs. The first NMI therefore
//! never returns, the handler's re-entrancy latch $1B stays 1, and
//! `$C2C4 STA $2001` is never reached again, so PPUMASK stays $00: one
//! distinct framebuffer hash across 600 frames.
//!
//! The fix arms `Cpu::hw_mmio_read_timing` (final-cycle read, what real
//! hardware and Mesen do) for this ROM's md5 only, and makes that flag
//! reach the per-cycle path instead of being silently swallowed by the
//! ASM/bulk batchers.
//!
//! Each test opens its ROM with a plain `File::open` and returns early
//! with an `eprintln!` when the file is absent, matching the convention
//! already used by `tests/zelda_real_rom.rs`, `tests/mapper228_boot.rs`
//! and `tests/ppu_shadow_oracle.rs`: these tests must run clean on a
//! checkout without the (git-ignored) ROM library.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use std::fs::File;
use std::io::BufReader;

const JACKAL: &str = "../roms/Jackal (USA).nes";
const CONTRA: &str = "../roms/Contra (USA).nes";

/// PRG+CHR md5 (the `Cartridge::md5` domain, header excluded).
const JACKAL_MD5: &str = "c6c17bf18a51718859f9bd6aacb7ef58";
const CONTRA_MD5: &str = "5a5c2f4f1cafb1f55a8dc0d5ad4550e5";

/// FNV-1a over the 600 per-frame framebuffer hashes, measured on the
/// fixed crate. Jackal was `b5377cb29e9ebaa5` (600 identical blank
/// frames) before the fix.
const JACKAL_600_SEQ_DIGEST: u64 = 0x3ebe_5bdc_af27_35c7;
/// Same digest for Contra, measured on HEAD *and* on the fixed crate and
/// found identical: the per-ROM scoping must not move any other UxROM
/// title's timing.
const CONTRA_600_SEQ_DIGEST: u64 = 0xbf3c_8d56_6851_1024;

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _s: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

fn load(path: &str) -> Option<(Nes, String)> {
    let f = File::open(path).ok()?;
    let cart = Cartridge::load(&mut BufReader::new(f)).ok()?;
    let md5 = cart.md5.clone();
    Some((Nes::new(cart), md5))
}

fn fnv1a_u64(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Step one frame, returning an FNV-1a hash of the XRGB8888 framebuffer.
fn step_frame_hash(nes: &mut Nes) -> u64 {
    let mut buf = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];
    let mut audio = NullAudio;
    let mut video = Xrgb8888VideoSink::new(&mut buf);
    while !video.frame_written() {
        nes.step(&mut video, &mut audio);
    }
    drop(video);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &p in &buf {
        h = fnv1a_u64(h, &p.to_le_bytes());
    }
    h
}

fn frame_hashes(nes: &mut Nes, frames: usize) -> Vec<u64> {
    (0..frames).map(|_| step_frame_hash(nes)).collect()
}

fn seq_digest(hashes: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in hashes {
        h = fnv1a_u64(h, &x.to_le_bytes());
    }
    h
}

/// The iNES header, parsed from the file bytes here rather than through
/// `Cartridge::load`, and the reset vector the mapper serves at power-on.
/// UxROM hard-wires $C000-$FFFF to the LAST PRG bank, so the vector at
/// $FFFC/$FFFD is bank 7's, independent of the switchable register.
#[test]
fn jackal_header_and_power_on_reset_vector_from_rom_bytes() {
    let Ok(bytes) = std::fs::read(JACKAL) else {
        eprintln!("skip: {JACKAL} not present");
        return;
    };
    assert_eq!(&bytes[0..4], b"NES\x1a", "iNES magic");
    let prg16 = bytes[4] as usize;
    let chr8 = bytes[5];
    let flags6 = bytes[6];
    let flags7 = bytes[7];
    let mapper = (flags6 >> 4) | (flags7 & 0xF0);
    assert_eq!(mapper, 2, "Jackal is mapper 2 (UxROM)");
    assert_eq!(prg16, 8, "128 KB PRG = 8 x 16 KB banks");
    assert_eq!(chr8, 0, "CHR RAM, no CHR ROM in the file");
    assert_eq!(flags6 & 0x01, 1, "vertical mirroring");
    assert_eq!(flags6 & 0x04, 0, "no trainer, so PRG starts at offset 16");

    // Reset vector out of the raw file: last bank, +$3FFC.
    let last = (prg16 - 1) * 16 * 1024;
    let lo = bytes[16 + last + 0x3FFC] as u16;
    let hi = bytes[16 + last + 0x3FFD] as u16;
    let reset = (hi << 8) | lo;
    assert_eq!(reset, 0xC23D, "reset vector from the fixed high bank");

    let Some((nes, md5)) = load(JACKAL) else {
        eprintln!("skip: {JACKAL} not loadable");
        return;
    };
    assert_eq!(md5, JACKAL_MD5, "PRG+CHR md5");
    assert_eq!(
        nes.cpu.regs().pc, 0xC23D,
        "power-on PC must be the reset vector the last bank serves"
    );
}

/// The per-ROM quirk arms hardware-true PPUSTATUS read timing for Jackal
/// and for nothing else. Corruption that fails it: drop Jackal's md5 from
/// `HW_STATUS_READ_TIMING_ROMS`, or widen the table to all of mapper 2.
#[test]
fn jackal_arms_hw_status_read_timing_and_contra_does_not() {
    if let Some((nes, md5)) = load(JACKAL) {
        assert_eq!(md5, JACKAL_MD5);
        assert!(
            nes.cpu.hw_mmio_read_timing,
            "Jackal must power on with hardware-true PPUSTATUS read timing"
        );
    } else {
        eprintln!("skip: {JACKAL} not present");
    }
    if let Some((nes, md5)) = load(CONTRA) {
        assert_eq!(md5, CONTRA_MD5);
        assert!(
            !nes.cpu.hw_mmio_read_timing,
            "no other UxROM title may be moved onto the fidelity lane"
        );
    } else {
        eprintln!("skip: {CONTRA} not present");
    }
}

/// Jackal reaches a live, animating screen. Before the fix this ROM
/// produced exactly one distinct framebuffer hash across 600 frames.
#[test]
fn jackal_boots_to_a_live_screen_in_600_frames() {
    let Some((mut nes, _)) = load(JACKAL) else {
        eprintln!("skip: {JACKAL} not present");
        return;
    };
    let hashes = frame_hashes(&mut nes, 600);
    let mut distinct = hashes.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() > 1,
        "Jackal must not render a static screen (got {} distinct hashes in 600 frames)",
        distinct.len()
    );
    assert_eq!(distinct.len(), 201, "measured distinct 600-frame hashes");
    let first_change = (1..hashes.len()).find(|&i| hashes[i] != hashes[i - 1]);
    assert_eq!(
        first_change,
        Some(16),
        "first frame whose framebuffer differs from its predecessor"
    );
    assert_eq!(
        seq_digest(&hashes),
        JACKAL_600_SEQ_DIGEST,
        "600-frame hash sequence digest"
    );
}

/// A sibling UxROM title's 600-frame hash sequence is byte-identical to
/// what HEAD produced. Corruption that fails it: arming the read-timing
/// flag for every mapper-2 cart instead of the listed md5s.
#[test]
fn contra_600_frame_hash_sequence_unchanged() {
    let Some((mut nes, _)) = load(CONTRA) else {
        eprintln!("skip: {CONTRA} not present");
        return;
    };
    let hashes = frame_hashes(&mut nes, 600);
    assert_eq!(
        seq_digest(&hashes),
        CONTRA_600_SEQ_DIGEST,
        "Contra's 600-frame hash sequence must match HEAD's"
    );
}
