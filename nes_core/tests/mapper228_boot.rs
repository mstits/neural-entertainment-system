//! Mapper 228 (Action 52 / Cheetahmen II pirate multicart) boot regression
//! tests, backed by the real ROM dumps under `roms/`.
//!
//! These pin the corrected power-on behavior and chip-select decode from
//! the DO-38 boot fix: 32 KB mode on bank pair 0/1 at power-on (so the
//! reset vector and boot stub, which live in every odd bank, are visible
//! at $C000 before the game's own code takes over), and chip select 3
//! aliasing onto the third 512 KB PRG chip (stored third in the dump)
//! instead of wrapping onto chip 0.
//!
//! Each test opens its ROM with a plain `File::open` and returns early
//! with an `eprintln!` when the file is absent, matching the convention
//! already used by `tests/zelda_real_rom.rs`, `tests/apu_muted_skip_parity.rs`
//! and `tests/ppu_shadow_oracle.rs` in this crate: these tests must be able
//! to run clean on a checkout without the (git-ignored) ROM library.

use nes_core::cartridge::Cartridge;
use nes_core::mapper::Mapper;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use std::fs::File;
use std::io::BufReader;

const ACTION52: &str = "../roms/Action 52 (USA) (Rev A) (Unl).nes";
const CHEETAHMEN2: &str = "../roms/Cheetahmen II (USA) (Unl).nes";

struct NullAudio;
impl AudioSink for NullAudio {
    fn write_sample(&mut self, _s: f32) {}
    fn samples_written(&self) -> usize {
        0
    }
}

fn load(path: &str) -> Option<Nes> {
    let f = File::open(path).ok()?;
    let cart = Cartridge::load(&mut BufReader::new(f)).ok()?;
    Some(Nes::new(cart))
}

/// Step one frame, returning an FNV-1a hash of the XRGB8888 framebuffer.
/// Same hash construction `scratchpad/do38-skeptic/m228_probe.rs` used for
/// the change-point measurements this file pins.
fn step_frame_hash(nes: &mut Nes) -> u64 {
    let mut buf = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];
    let mut audio = NullAudio;
    let mut video = Xrgb8888VideoSink::new(&mut buf);
    while !video.frame_written() {
        nes.step(&mut video, &mut audio);
    }
    drop(video);
    let mut h: u64 = 0xcbf29ce484222325;
    for &p in &buf {
        for b in p.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Minimal SHA-1 (FIPS 180-4), used only to pin the Cheetahmen II
/// change-point sequence to the exact digest the skeptic report measured
/// (`shasum -a1` over the same `  change@<frame> <hash>` lines). Self-checked
/// against the standard test vectors in
/// `sha1_self_check_against_known_vectors` before it is trusted anywhere
/// else in this file.
fn sha1_hex(data: &[u8]) -> String {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let ml_bits: u64 = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&ml_bits.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for i in 0..80 {
            let (f, k) = if i < 20 {
                ((b & c) | ((!b) & d), 0x5A827999u32)
            } else if i < 40 {
                (b ^ c ^ d, 0x6ED9EBA1u32)
            } else if i < 60 {
                ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32)
            } else {
                (b ^ c ^ d, 0xCA62C1D6u32)
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    format!("{:08x}{:08x}{:08x}{:08x}{:08x}", h0, h1, h2, h3, h4)
}

/// Guards `sha1_hex` against FIPS 180-4 test vectors before it is used to
/// pin anything. Not one of the four ROM-backed regression tests; it is
/// infrastructure for `cheetahmen2_first_600_frames_hash_sequence_unchanged`
/// and needs no ROM.
#[test]
fn sha1_self_check_against_known_vectors() {
    assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(
        sha1_hex(b"The quick brown fox jumps over the lazy dog"),
        "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
    );
}

/// (1) Action 52 power-on reset vector reads $FFD8 (32 KB mode, bank pair
/// 0/1, the odd bank carries every reset vector and boot stub) and the
/// CPU's first 16 instruction boundaries match the exact PC sequence
/// measured on the corrected crate: the boot stub's `LDA #$00; STA $8800;
/// JMP $801E` followed by the menu's own code at $8584.
///
/// Corruption that must fail this test: `prg_mode_32k: false` in
/// `Mapper228::new()` (the constructor). `Nes::reset()` runs `cpu.reset()`
/// (which fetches the boot PC from the mapper's CURRENT state) before
/// `mapper.reset()` runs, so a wrong constructor default sends the CPU to
/// $8016 (bank 0, 16 KB mode) even though `mapper.reset()` (untouched)
/// puts the mapper back in 32 KB mode a moment later.
#[test]
fn action52_power_on_reset_vector_and_boot_pcs() {
    let Some(mut nes) = load(ACTION52) else {
        eprintln!("skipping: {} not present", ACTION52);
        return;
    };

    let expected_pcs: [u16; 16] = [
        0xFFD8, 0xFFDA, 0xFFDD, 0xFFDE, 0xFFDF, 0xFFE0, 0xFFE1, 0xFFE2, 0x801E, 0x8584, 0x8586,
        0x8588, 0x8589, 0x858A, 0x858B, 0x858C,
    ];

    let boot_pc = nes.get_state().cpu.regs.pc;
    assert_eq!(
        boot_pc, 0xFFD8,
        "power-on boot PC must be the reset vector $FFD8 (32 KB mode, bank pair 0/1)"
    );

    // Sanity re-peek of the vector bytes through the mapper's settled
    // (post mapper.reset()) state; should agree with boot_pc on a
    // correct crate.
    let lo = nes.mapper.prg_peek_byte(0xFFFC) as u16;
    let hi = nes.mapper.prg_peek_byte(0xFFFD) as u16;
    assert_eq!((hi << 8) | lo, 0xFFD8, "settled mapper state must still serve $FFD8 at the vector");

    let mut audio = NullAudio;
    let mut video_buf = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];
    let mut pcs: Vec<u16> = Vec::new();
    'outer: loop {
        let mut video = Xrgb8888VideoSink::new(&mut video_buf);
        while !video.frame_written() {
            if pcs.len() >= expected_pcs.len() {
                break 'outer;
            }
            pcs.push(nes.get_state().cpu.regs.pc);
            nes.step(&mut video, &mut audio);
        }
    }

    assert_eq!(
        pcs, expected_pcs,
        "first 16 instruction PCs must match the skeptic's measured boot sequence"
    );
}

/// (2) Chip select 3 aliases onto the third 512 KB PRG chip: a write that
/// selects chip 3, inner bank 5, 16 KB mode must serve exactly the ROM's
/// PRG bytes at offset 1 MiB + 5*16 KiB = bank 69, byte-for-byte against
/// the ROM file read directly (bypassing `Cartridge::load` entirely, so
/// this doesn't just check the crate agrees with itself).
///
/// Corruption that must fail this test: deleting the `chip == 3 => 2`
/// remap in `Mapper228::decode_prg_bank` (the DO-36 candidate's decode).
/// Chip 3 then decodes to register value `(3<<5)|5 = 101`, which wraps
/// `101 % 96 == 5`, bank 5 of chip 0, a different 16 KB region with a
/// different SHA-1 (verified: bank 5 != bank 69 in the actual ROM).
#[test]
fn action52_chip_select_3_aliases_third_chip() {
    let Some(mut nes) = load(ACTION52) else {
        eprintln!("skipping: {} not present", ACTION52);
        return;
    };

    let raw = std::fs::read(ACTION52).expect("re-read ROM file directly");
    let flags6 = raw[6];
    let has_trainer = (flags6 & 0x04) != 0;
    let prg_start = 16 + if has_trainer { 512 } else { 0 };

    const INNER: u16 = 5;
    const THIRD_CHIP_BASE_BANK: usize = 1_048_576 / 0x4000; // bank 64
    let target_bank = THIRD_CHIP_BASE_BANK + INNER as usize; // bank 69

    // A12..A11 = chip select (3), A10..A6 = inner bank, A5 set = 16 KB
    // mode (so both $8000-$BFFF and $C000-$FFFF mirror the same bank,
    // no base/base+1 pairing arithmetic to account for).
    let addr: u16 = 0x8000 | (3u16 << 11) | (INNER << 6) | (1 << 5);
    nes.mapper.prg_write_byte(addr, 0);

    let expected_off = prg_start + target_bank * 0x4000;
    let expected = &raw[expected_off..expected_off + 0x4000];

    let mut actual = Vec::with_capacity(0x4000);
    for a in 0x8000u32..0xC000u32 {
        actual.push(nes.mapper.prg_peek_byte(a as u16));
    }

    assert_eq!(
        actual, expected,
        "chip select 3, inner bank {INNER} must read PRG offset 1 MiB + {INNER}*16KiB \
         (bank {target_bank}) byte-for-byte from the ROM file"
    );
}

/// (3) Cheetahmen II's first-600-frame framebuffer-hash change-point
/// sequence is unchanged by the boot fix: this game is a single 256 KB
/// chip, so `(32+inner) mod 16 == inner` makes the chip-select bits
/// irrelevant, and the game's own second instruction (`STA $8800` at
/// $8011) rewrites the register before any high-window read matters. Pins
/// the exact SHA-1 the skeptic report measured (re-measured independently
/// here on this corrected crate before being pinned).
///
/// Corruption that must fail this test: flipping the pinned constant
/// `EXPECTED_SHA1` to a wrong value (see the revert-verify transcript in
/// the report; the corruption is in the test's own expectation, since
/// the crate's Cheetahmen II behavior is provably unchanged by the fix).
#[test]
fn cheetahmen2_first_600_frames_hash_sequence_unchanged() {
    let Some(mut nes) = load(CHEETAHMEN2) else {
        eprintln!("skipping: {} not present", CHEETAHMEN2);
        return;
    };

    const EXPECTED_SHA1: &str = "7d042aa063571b2f203c10e9264bcdd6dee2bce2";

    let mut last: u64 = 0;
    let mut lines = String::new();
    for f in 0..600usize {
        let h = step_frame_hash(&mut nes);
        if f == 0 || h != last {
            lines.push_str(&format!("  change@{} {:016x}\n", f, h));
            last = h;
        }
    }

    let digest = sha1_hex(lines.as_bytes());
    assert_eq!(
        digest, EXPECTED_SHA1,
        "Cheetahmen II 600-frame change-point sequence must match the pinned boot-fix baseline"
    );
}

/// (4) Action 52's $0390 "menu ready" RAM flag goes non-zero by frame 38
/// (measured exactly 38 on the corrected crate) and the multicart menu
/// actually renders: the first 300 frames must produce more than 5
/// distinct framebuffer hashes (a broken boot that never leaves the boot
/// stub renders 1 static frame for the whole run).
///
/// Corruption that must fail this test: reinstating the old `chip == 1 =>
/// 0` remap in `Mapper228::decode_prg_bank` (HEAD's bug, distinct from
/// test 2's corruption: this one substitutes the remap rather than
/// deleting it). The boot stub's `STA $8800` selects chip 1, which remaps
/// to chip 0 / bank 0, the same static bank the ROM powered on into, so
/// the menu never renders: 1 distinct hash, `$0390` never goes non-zero.
#[test]
fn action52_menu_boots_by_frame_38_with_visible_animation() {
    let Some(mut nes) = load(ACTION52) else {
        eprintln!("skipping: {} not present", ACTION52);
        return;
    };

    let mut first_390: Option<usize> = None;
    let mut distinct_first_300: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for f in 0..600usize {
        let h = step_frame_hash(&mut nes);
        if f < 300 {
            distinct_first_300.insert(h);
        }
        if first_390.is_none() {
            let v = nes.get_state().ram[0x0390];
            if v != 0 {
                first_390 = Some(f);
            }
        }
    }

    assert_eq!(
        first_390,
        Some(38),
        "$0390 must go non-zero on exactly frame 38 (the corrected crate's measured boot timing)"
    );
    assert!(
        distinct_first_300.len() > 5,
        "menu must render more than 5 distinct frames in the first 300 (got {})",
        distinct_first_300.len()
    );
}
