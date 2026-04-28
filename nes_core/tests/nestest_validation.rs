//! Single-instruction CPU correctness validation against nestest.nes
//! and the canonical kevtris/nintendulator log.
//!
//! nestest is the de-facto reference for 6502 (NES variant) opcode
//! coverage: it exercises every official opcode, every undocumented
//! opcode, every addressing mode, and every flag-update path. The
//! companion `.log` file records the exact PC, opcode bytes, register
//! values, and cycle counter at every instruction boundary, captured
//! from a known-correct emulator (Nintendulator).
//!
//! Method:
//!   1. Load nestest.nes and force PC=$C000 (the ROM's "automated
//!      mode" entry — bypasses the visual test).
//!   2. Step the CPU until each instruction boundary, format the
//!      trace line, and compare to the corresponding log line.
//!   3. Report the first divergence with both expected and actual
//!      register state — the buggy opcode is named on the previous
//!      line.
//!
//! When this test fails, the diff message immediately points at the
//! incorrect opcode/addressing-mode combination. That's the right
//! debugging entry point for the Zelda cave-sword cycle-accuracy
//! work and the Bill & Ted's / Crash 'n' the Boys boot-time crash
//! — both share root cause: drift in some 6502 opcode our emulator
//! doesn't quite match.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// All 8991 lines of nestest.log validate exactly. Every official
/// 6502 opcode, every undocumented opcode, every addressing mode, AND
/// the cycle counter match Nintendulator byte-for-byte. APU register
/// "= XX" trace annotations are stripped during comparison (see
/// `strip_apu_annotation` for rationale).
const FULL_LOG_LINES: usize = usize::MAX;

fn read_log_lines(limit: usize) -> Vec<String> {
    let f = File::open("../roms/.test_roms/nestest.log")
        .expect("nestest.log present (../roms/.test_roms/)");
    let reader = BufReader::new(f);
    reader
        .lines()
        .map(|l| l.expect("read line"))
        .take(limit)
        .collect()
}

/// Compare two trace lines. We assert PC + opcode bytes + asm + A/X/Y/P/SP
/// AND the CYC (cycle count) — that's the cycle-accuracy ground truth.
///
/// Strip:
///   * `PPU:r,c` numbers — known startup phase offset from Nintendulator,
///     not reflective of any CPU/PPU bug worth gating on.
///   * `= XX` annotation when the operand address is in `$4000-$4017`
///     (APU + I/O registers). Nintendulator's tracer prints `= FF` as a
///     "no-meaningful-value" sentinel for these, regardless of the real
///     register state — it doesn't actually run an APU read for the
///     trace, because that would have side effects (clearing $4015's
///     frame-IRQ-pending flag, for example). nes_core's tracer DOES
///     read APU status for $4015 (returns true APU state), so
///     comparing the annotated value against the canonical log fails
///     for these addresses even though the actual STA executes
///     identically. We strip the annotation rather than emulate
///     Nintendulator's specific sentinel.
fn normalize_trace_line(s: &str) -> String {
    // Drop PPU:r,c but keep CYC:n.
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == 'P' && chars.clone().take(3).collect::<String>() == "PU:" {
            for _ in 0..3 { chars.next(); }
            while let Some(&p) = chars.peek() {
                if p == ' ' || p == '\t' { break; }
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    // Strip "= XX" annotations for absolute APU/IO addresses $4000-$4017.
    // Pattern: `$40[01][0-9A-F] = XX` — keep address, drop ` = XX`.
    // Hand-rolled rather than pulling in regex; the format is fixed.
    let stripped = strip_apu_annotation(&out);
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_apu_annotation(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        // Look for "$40" + hex digit '0'..'1' + hex digit + " = " + 2 hex.
        if i + 11 <= bytes.len()
            && bytes[i] == b'$'
            && bytes[i + 1] == b'4'
            && bytes[i + 2] == b'0'
            && (bytes[i + 3] == b'0' || bytes[i + 3] == b'1')
            && bytes[i + 4].is_ascii_hexdigit()
            && bytes[i + 5] == b' '
            && bytes[i + 6] == b'='
            && bytes[i + 7] == b' '
            && bytes[i + 8].is_ascii_hexdigit()
            && bytes[i + 9].is_ascii_hexdigit()
        {
            // Emit "$40XX" and skip the " = NN" part.
            out.push_str(&s[i..i + 5]);
            i += 10;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn run_nestest_validation(limit: usize, strict: bool) -> Result<usize, String> {
    let path = "../roms/.test_roms/nestest.nes";
    let f = File::open(path).map_err(|_| {
        format!("nestest.nes not present at {path}")
    })?;
    let mut r = BufReader::new(f);
    let cart = Cartridge::load(&mut r).map_err(|e| format!("cart load: {e:?}"))?;

    let mut nes = Nes::new(cart);
    nes.initialize_nestest();

    let log_lines = read_log_lines(limit);
    if log_lines.is_empty() {
        return Err("nestest.log empty".into());
    }
    let _ = strict;

    // Drop video + audio output — we don't need pixels for CPU validation.
    struct SinkV;
    impl VideoSink for SinkV {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool { false }
        fn pixel_size(&self) -> usize { 4 }
    }
    struct SinkA;
    impl AudioSink for SinkA {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize { 0 }
    }
    let mut video = SinkV;
    let mut audio = SinkA;

    // Tick until each instruction boundary, capture trace, advance.
    // Boundary = `cpu.cycle == 0 && cpu.instruction.is_none()`.
    for (i, expected_raw) in log_lines.iter().enumerate() {
        // Capture trace BEFORE the next instruction executes.
        let actual_raw = nes.trace_line();

        let expected = normalize_trace_line(expected_raw);
        let actual = normalize_trace_line(&actual_raw);

        if expected != actual {
            let from = i.saturating_sub(3);
            let mut ctx = String::new();
            for j in from..i {
                ctx.push_str(&format!("  prev {j:>5}: {}\n", log_lines[j]));
            }
            return Err(format!(
                "nestest divergence at line {i}\n\
                {ctx}\
                expected: {expected}\n\
                actual  : {actual}",
            ));
        }

        // Advance one instruction. Bound iterations to detect runaway.
        let mut ticks = 0usize;
        loop {
            nes.tick(&mut video, &mut audio);
            ticks += 1;
            if ticks > 200 {
                return Err(format!("instruction at line {i} took >200 ticks"));
            }
            if nes.cpu.at_instruction_boundary() {
                break;
            }
        }
    }
    Ok(log_lines.len())
}

#[test]
fn nestest_full_log_matches_canonical_byte_exact() {
    match run_nestest_validation(FULL_LOG_LINES, true) {
        Ok(n) => {
            eprintln!(
                "nestest validated: {n} instructions matched canonical log byte-exact \
                (PC + opcode + asm + A/X/Y/P/SP + CYC)"
            );
        }
        Err(e) => {
            if e.contains("not present") || e.contains("empty") {
                eprintln!("skipping: {e}");
                return;
            }
            panic!("CPU regressed: {e}\n\n\
                If you see a divergence here you broke an opcode. Check the prev-3 lines \
                in the diff to see which opcode set up the diverging state.");
        }
    }
}
