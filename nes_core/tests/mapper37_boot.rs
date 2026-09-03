//! Mapper 37 (NES-ZZ multicart: Super Mario Bros. + Tetris + Nintendo
//! World Cup) boot regression tests, backed by the real ROM dumps under
//! `roms/`.
//!
//! These pin the corrected outer-register decode: bit 2 drives A17 on
//! both PRG and CHR, the low pair gates PRG A16 (the value 3 forces it
//! high, anything else ANDs the MMC3's own A16 with bit 2), and the
//! resulting PRG window is 64 KB except for outer 4-6. The window size is
//! what makes this a boot bug rather than a bank-arithmetic bug: the
//! MMC3's fixed last bank at $E000-$FFFF is the last 8 KB of the window,
//! so it is where the 6502 fetches its reset vector. Outer 0 over a
//! 64 KB window reads $0FFFC ($FF00, the multicart menu); over the 128 KB
//! window HEAD used it reads $1FFFC ($8000), Tetris's vector table, and
//! the cart renders one flat grey frame forever.
//!
//! Each test opens its ROM with a plain `File::open` and returns early
//! with an `eprintln!` when the file is absent, matching the convention
//! already used by `tests/mapper228_boot.rs`, `tests/zelda_real_rom.rs`
//! and `tests/ppu_shadow_oracle.rs` in this crate: these tests must be
//! able to run clean on a checkout without the (git-ignored) ROM library.

use nes_core::cartridge::Cartridge;
use nes_core::mapper::Mapper;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};
use std::fs::File;
use std::io::BufReader;

const SMBT: &str = "../roms/Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes";
const SPIKE_NWC: &str = "../roms/Super Spike V'Ball + Nintendo World Cup (USA).nes";

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
/// Same hash construction the scratch probe used for the change-point
/// measurements this file pins.
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

/// Minimal SHA-1 (FIPS 180-4), used only to pin a change-point sequence
/// to an exact digest (`shasum -a1` over the same `  change@<frame>
/// <hash>` lines the scratch probe printed). Self-checked against the
/// standard test vectors in `sha1_self_check_against_known_vectors`
/// before it is trusted anywhere else in this file. Same construction as
/// `tests/mapper228_boot.rs`; duplicated because integration tests are
/// separate crates and this one must stay standalone.
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
/// pin anything. Not one of the ROM-backed regression tests; it is
/// infrastructure for `spike_vball_nwc_600_frame_sequence_unchanged` and
/// needs no ROM.
#[test]
fn sha1_self_check_against_known_vectors() {
    assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(
        sha1_hex(b"The quick brown fox jumps over the lazy dog"),
        "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
    );
}

/// (1) Power-on serves the menu's reset vector. Outer 0 is a 64 KB PRG
/// window, so the MMC3's fixed last bank is PRG $0E000-$0FFFF and the
/// vector at $FFFC reads $FF00. The first instruction PCs are the menu's
/// own `SEI; CLD; LDX #$FF; TXS; ...` entry, measured on the corrected
/// crate.
///
/// Corruption that must fail this test: a 128 KB PRG window at outer 0
/// (HEAD's `chip 0 => (0, 128 * 1024)`). The fixed last bank moves to
/// $1E000-$1FFFF, the vector reads $8000, and the CPU boots Tetris's
/// entry point instead of the menu.
#[test]
fn smbt_power_on_reset_vector_is_the_menu() {
    let Some(mut nes) = load(SMBT) else {
        eprintln!("skipping: {} not present", SMBT);
        return;
    };

    let lo = nes.mapper.prg_peek_byte(0xFFFC) as u16;
    let hi = nes.mapper.prg_peek_byte(0xFFFD) as u16;
    assert_eq!(
        (hi << 8) | lo,
        0xFF00,
        "power-on vector must come from the 64 KB window's last bank ($0FFFC), not $1FFFC"
    );
    assert_eq!(
        nes.get_state().cpu.regs.pc,
        0xFF00,
        "power-on boot PC must be the menu's reset vector $FF00"
    );

    let expected_pcs: [u16; 7] = [0xFF00, 0xFF01, 0xFF02, 0xFF04, 0xFF07, 0xFF0A, 0xFF0D];
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
    assert_eq!(pcs, expected_pcs, "first 7 instruction PCs must be the menu's entry");
}

/// (2) Every outer value maps the window the NESdev wiki's NES-ZZ table
/// gives, checked byte-for-byte against the ROM file read directly
/// (bypassing `Cartridge::load`, so this does not just check the crate
/// agrees with itself). The fixed last 8 KB bank at $E000-$FFFF is the
/// end of the PRG window; CHR $0000 is the base of the 128 KB CHR window.
///
///   outer | PRG window             | last 8 KB bank | CHR base
///   ------+------------------------+----------------+---------
///   0,1,2 | $00000-$0FFFF  (64 KB) | $0E000         | $00000
///   3     | $10000-$1FFFF  (64 KB) | $1E000         | $00000
///   4,5,6 | $20000-$3FFFF (128 KB) | $3E000         | $20000
///   7     | $30000-$3FFFF  (64 KB) | $3E000         | $20000
///
/// Corruption that must fail this test: HEAD's `chip = (val >> 1).min(2)`
/// decode, which puts outer 2 in the same window as outer 3, sizes every
/// window at 128 KB or more, and bases outer >= 4 at PRG offset 256 KB,
/// past the end of a 256 KB dump.
#[test]
fn smbt_outer_windows_match_rom_bytes() {
    let Some(mut nes) = load(SMBT) else {
        eprintln!("skipping: {} not present", SMBT);
        return;
    };

    let raw = std::fs::read(SMBT).expect("re-read ROM file directly");
    let has_trainer = (raw[6] & 0x04) != 0;
    let prg_start = 16 + if has_trainer { 512 } else { 0 };
    let prg_len = raw[4] as usize * 16 * 1024;
    let chr_start = prg_start + prg_len;
    assert_eq!(prg_len, 256 * 1024, "NES-ZZ dump is 256 KB PRG");

    // (outer value, last-8 KB-bank PRG offset, CHR window base)
    let cases: [(u8, usize, usize); 8] = [
        (0, 0x0E000, 0x00000),
        (1, 0x0E000, 0x00000),
        (2, 0x0E000, 0x00000),
        (3, 0x1E000, 0x00000),
        (4, 0x3E000, 0x20000),
        (5, 0x3E000, 0x20000),
        (6, 0x3E000, 0x20000),
        (7, 0x3E000, 0x20000),
    ];

    for (v, prg_off, chr_off) in cases {
        nes.mapper.prg_write_byte(0x6000, v);

        let expected = &raw[prg_start + prg_off..prg_start + prg_off + 0x2000];
        let mut actual = Vec::with_capacity(0x2000);
        for a in 0xE000u32..0x10000u32 {
            actual.push(nes.mapper.prg_peek_byte(a as u16));
        }
        assert_eq!(
            actual, expected,
            "outer {v}: $E000-$FFFF must be PRG offset {prg_off:#07X} byte-for-byte"
        );

        // The CHR window is always 128 KB; check its first 1 KB bank,
        // which the MMC3 powers on with R0 = 0 selecting.
        let expected_chr = &raw[chr_start + chr_off..chr_start + chr_off + 0x0400];
        let mut actual_chr = Vec::with_capacity(0x0400);
        for a in 0x0000u16..0x0400 {
            actual_chr.push(nes.mapper.chr_read_byte(a));
        }
        assert_eq!(
            actual_chr, expected_chr,
            "outer {v}: CHR $0000-$03FF must be CHR offset {chr_off:#07X} byte-for-byte"
        );
    }
}

/// (3) The multicart menu actually boots and renders. `$0033` is the RAM
/// flag the menu's NMI handler sets and its main loop polls (the flag the
/// static-screen diagnosis found stuck at zero forever: HEAD spins on
/// `LDA $33 / BEQ $80DD` inside Tetris's code with NMI never enabled).
/// On the corrected crate it goes non-zero on frame 2 and the first 600
/// frames with no input produce 5 distinct framebuffer hashes.
///
/// Corruption that must fail this test: HEAD's decode, which never
/// reaches the menu: `$0033` stays zero for all 600 frames and the
/// framebuffer is one flat grey hash.
#[test]
fn smbt_menu_boots_and_renders() {
    let Some(mut nes) = load(SMBT) else {
        eprintln!("skipping: {} not present", SMBT);
        return;
    };

    let mut first_33: Option<usize> = None;
    let mut distinct: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for f in 0..600usize {
        distinct.insert(step_frame_hash(&mut nes));
        if first_33.is_none() && nes.get_state().ram[0x0033] != 0 {
            first_33 = Some(f);
        }
    }

    assert_eq!(
        first_33,
        Some(2),
        "the menu's NMI flag $0033 must go non-zero on frame 2"
    );
    assert!(
        distinct.len() >= 5,
        "menu must render at least 5 distinct frames in 600 (got {})",
        distinct.len()
    );
}

/// (4) The other multicart on the same `Mapper4::set_outer_region` path,
/// Super Spike V'Ball + Nintendo World Cup (mapper 47), is unchanged by
/// this fix: same 600-frame change-point sequence, pinned to the digest
/// measured on both HEAD and the corrected crate.
///
/// Corruption that must fail this test: flipping the pinned constant, or
/// any future change to `set_outer_region`'s window arithmetic in
/// `mapper4.rs` (which mapper 47 shares) that moves a bank.
#[test]
fn spike_vball_nwc_600_frame_sequence_unchanged() {
    let Some(mut nes) = load(SPIKE_NWC) else {
        eprintln!("skipping: {} not present", SPIKE_NWC);
        return;
    };

    const EXPECTED_SHA1: &str = "4072bddda88c1a2a65433231faf0daf35cb097ae";

    let mut last: u64 = 0;
    let mut lines = String::new();
    for f in 0..600usize {
        let h = step_frame_hash(&mut nes);
        if f == 0 || h != last {
            lines.push_str(&format!("  change@{} {:016x}\n", f, h));
            last = h;
        }
    }

    assert_eq!(
        sha1_hex(lines.as_bytes()),
        EXPECTED_SHA1,
        "mapper 47 sibling's 600-frame change-point sequence must be unchanged"
    );
}
