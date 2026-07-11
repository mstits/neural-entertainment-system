//! ASM bulk-budget override: API semantics + lockstep gate rungs.
//!
//! MMC1 and UxROM default to `asm_bulk_cycles() == 1` (one 6502
//! instruction per ASM invocation) — the shipped path, so leaving
//! the override untouched is parity-safe by construction. The
//! opt-in override (`set_asm_bulk_cycles_override`, clamped 1..=16)
//! amortizes the per-invocation ASM setup over 2-5 instructions.
//!
//! Tests here come in two tiers:
//!   * Always-on: default budget, clamping, mapper scoping, and
//!     state round-trip.
//!   * `#[ignore]`d ladder rungs (budget 8, 16 on Zelda/MMC1 and
//!     Contra/UxROM), lockstepped against the ASM path at the
//!     default budget: the first stage of the per-game enablement
//!     gate. Run explicitly via
//!     `cargo test --features asm_cpu --test asm_bulk_override -- --ignored --nocapture`
//!     — this command is expected to be fully green: rungs at or
//!     below a game's measured ceiling pass outright, and the one
//!     known-over-ceiling rung (Zelda @ 16) is `should_panic`-pinned
//!     to its documented divergence, so it too reports `ok` today
//!     and fails loudly if a future fix moves the ceiling.
//!     A rung must pass here AND the Mesen oracle + `make parity` +
//!     greedy-eval gates on the main tree before any profile sets
//!     `reinforce.asm_bulk_cycles` to that budget.

use nes_core::cartridge::Cartridge;
use nes_core::input::Button;
use nes_core::mapper::Mapper;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const ZELDA: &str = "../roms/zelda.nes";
const CONTRA: &str = "../roms/Contra (USA).nes";

struct V;
impl VideoSink for V {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
struct A;
impl AudioSink for A {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

/// Minimal in-memory iNES image: 16-byte header, `prg_banks` 16 KB
/// PRG banks, one 8 KB CHR bank. Enough for mapper construction and
/// budget-API checks; nothing executes it.
fn synth_cart(mapper: u8, prg_banks: u8) -> Cartridge {
    let mut img = vec![
        b'N', b'E', b'S', 0x1A,
        prg_banks, 1,
        (mapper & 0x0F) << 4, mapper & 0xF0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    img.extend(std::iter::repeat(0u8).take(prg_banks as usize * 16 * 1024));
    img.extend(std::iter::repeat(0u8).take(8 * 1024));
    Cartridge::load(&mut std::io::Cursor::new(img)).expect("synth cart")
}

fn load_rom(path: &str) -> Nes {
    let f = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    Nes::new(cart)
}

#[test]
fn default_budget_is_one_and_override_clamps() {
    for mapper_id in [1u8, 2u8] {
        let mut nes = Nes::new(synth_cart(mapper_id, 2));
        assert_eq!(
            nes.mapper.asm_bulk_cycles(),
            1,
            "mapper {mapper_id}: default budget must stay 1 (parity-safe)"
        );
        nes.set_asm_bulk_cycles_override(8);
        assert_eq!(nes.mapper.asm_bulk_cycles(), 8);
        nes.set_asm_bulk_cycles_override(16);
        assert_eq!(nes.mapper.asm_bulk_cycles(), 16);
        nes.set_asm_bulk_cycles_override(0);
        assert_eq!(nes.mapper.asm_bulk_cycles(), 1, "clamp floor");
        nes.set_asm_bulk_cycles_override(100);
        assert_eq!(nes.mapper.asm_bulk_cycles(), 16, "clamp ceiling");
    }
}

#[test]
fn override_is_scoped_to_batch_safe_mappers() {
    // NROM already runs its validated ceiling (64); the override
    // must not touch it.
    let mut nes = Nes::new(synth_cart(0, 2));
    assert_eq!(nes.mapper.asm_bulk_cycles(), 64);
    nes.set_asm_bulk_cycles_override(8);
    assert_eq!(nes.mapper.asm_bulk_cycles(), 64, "NROM ignores override");
    // MMC3's scanline IRQ can assert mid-batch (not an ASM exit
    // condition) — it must stay at 1 regardless of the override.
    let mut nes = Nes::new(synth_cart(4, 2));
    assert_eq!(nes.mapper.asm_bulk_cycles(), 1);
    nes.set_asm_bulk_cycles_override(8);
    assert_eq!(nes.mapper.asm_bulk_cycles(), 1, "MMC3 ignores override");
}

#[test]
fn override_survives_state_roundtrip() {
    // The budget is a runtime perf knob, not console state: loading
    // a start-state snapshot (curriculum warm-start path) must not
    // silently reset a trainer-applied budget.
    let mut nes = Nes::new(synth_cart(1, 2));
    let snap = nes.get_state();
    nes.set_asm_bulk_cycles_override(8);
    nes.apply_state(&snap);
    assert_eq!(nes.mapper.asm_bulk_cycles(), 8);
}

/// Deterministic input schedule, applied to both lockstep instances
/// at the same instruction boundary: Start pulses through the title/
/// menu screens for the first ~900 frames, then pseudo-random
/// d-pad + A/B gameplay input per 8-frame block. Gets both ROMs out
/// of attract mode and into gameplay (scrolling, sprite-0 HUD
/// splits, music-driven APU activity) so the rung exercises more
/// than cold boot.
fn schedule_mask(frame: usize) -> u8 {
    if frame < 900 {
        // 2-on / 30-off Start pulse train.
        if frame % 32 < 2 { 0x08 } else { 0x00 }
    } else {
        // xorshift32 on the 8-frame block index; A/B + d-pad only
        // (no Start/Select — pausing would idle the gameplay code).
        let mut x = (frame / 8) as u32 ^ 0x9E37_79B9;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x & 0xF3) as u8
    }
}

fn apply_mask(nes: &mut Nes, mask: u8) {
    let pad = nes.game_pad_1();
    pad.set_button_pressed(Button::A, mask & 0x01 != 0);
    pad.set_button_pressed(Button::B, mask & 0x02 != 0);
    pad.set_button_pressed(Button::Select, mask & 0x04 != 0);
    pad.set_button_pressed(Button::Start, mask & 0x08 != 0);
    pad.set_button_pressed(Button::Up, mask & 0x10 != 0);
    pad.set_button_pressed(Button::Down, mask & 0x20 != 0);
    pad.set_button_pressed(Button::Left, mask & 0x40 != 0);
    pad.set_button_pressed(Button::Right, mask & 0x80 != 0);
}

/// Lockstep-run the ASM path at `budget` against the ASM path at the
/// shipped default budget (1), comparing CPU registers, cycle count,
/// and all 2 KB of system RAM at every instruction boundary. Both
/// sides run the exact same engine (`nes.step`) — the ONLY delta is
/// the batch size — so any divergence is attributable to batching
/// (mid-batch NMI/IRQ service slip, timed-MMIO granularity), not to
/// the pre-existing ASM-vs-interpreter tick-alignment residual
/// (sprite-0 $2002 polling — observed on Zelda at the DEFAULT budget
/// vs `nes.tick`, so an interpreter reference cannot gate the rungs).
fn lockstep_vs_default(rom: &str, budget: i64, max_instr: usize) {
    let mut cand = load_rom(rom);
    let mut base = load_rom(rom);
    cand.reset();
    base.reset();
    cand.set_asm_bulk_cycles_override(budget);
    // `base` keeps the default budget of 1.

    let mut v_c = V; let mut a_c = A;
    let mut v_b = V; let mut a_b = A;

    let mut last_frame = usize::MAX;
    for i in 0..max_instr {
        let cyc_c0 = cand.cycles;
        let cyc_b0 = base.cycles;

        // Both sides are cycle-synchronized here (asserted below),
        // so the schedule flips input at the identical instruction
        // boundary on both instances.
        let frame = cand.cycles / 29781;
        if frame != last_frame {
            last_frame = frame;
            let mask = schedule_mask(frame);
            apply_mask(&mut cand, mask);
            apply_mask(&mut base, mask);
        }

        // One batch (up to `budget` cycles / several instructions),
        // then run to a clean instruction boundary.
        cand.step(&mut v_c, &mut a_c);
        while !cand.cpu.at_instruction_boundary() {
            cand.step(&mut v_c, &mut a_c);
        }

        // Reference catches up one instruction at a time.
        while base.cycles < cand.cycles || !base.cpu.at_instruction_boundary() {
            base.step(&mut v_b, &mut a_b);
        }

        let r_c = cand.cpu.regs();
        let r_b = base.cpu.regs();
        if r_c.pc != r_b.pc || r_c.a != r_b.a || r_c.x != r_b.x
            || r_c.y != r_b.y || r_c.sp != r_b.sp
            || cand.cycles != base.cycles
            || (0..0x800u16).any(|a| cand.system_ram_byte(a) != base.system_ram_byte(a))
        {
            let bad_ram = (0..0x800u16)
                .find(|&a| cand.system_ram_byte(a) != base.system_ram_byte(a));
            panic!(
                "DIVERGENCE rom={rom} budget={budget} iter {i}:\n\
                 cand: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={} (pre {cyc_c0})\n\
                 base: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={} (pre {cyc_b0})\n\
                 first RAM mismatch: {:?}",
                r_c.pc, r_c.a, r_c.x, r_c.y, r_c.sp, cand.cycles,
                r_b.pc, r_b.a, r_b.x, r_b.y, r_b.sp, base.cycles,
                bad_ram.map(|a| format!(
                    "${a:04X}: cand={:02X} base={:02X}",
                    cand.system_ram_byte(a), base.system_ram_byte(a),
                )),
            );
        }
    }
    println!(
        "{rom} budget={budget}: no divergence vs default in {max_instr} \
         batches ({} frames incl. input-driven gameplay)",
        cand.cycles / 29781,
    );
}

// ---- Ladder rungs (run explicitly with --ignored). A failure here
// means that budget is NOT safe for that game; keep the profile at
// the last passing rung. Batch count is env-tunable for longer
// main-tree gate runs: ASM_BULK_LOCKSTEP_BATCHES=50000000. ----

fn max_batches(default: usize) -> usize {
    std::env::var("ASM_BULK_LOCKSTEP_BATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Measured 2026-07-11: PASSES (1852 frames, menus + gameplay).
#[test]
#[ignore]
fn zelda_bulk8_lockstep() {
    lockstep_vs_default(ZELDA, 8, max_batches(6_000_000));
}

/// Measured 2026-07-11: DIVERGES at ~frame 78 (title music) — an
/// interrupt is serviced at a different instruction boundary (SP
/// differs by 2, stack byte $01F7 differs), i.e. the mid-batch
/// IRQ-service slip: `update_irq_line` runs only at end-of-batch
/// (`nes.rs`), so an APU frame/DMC IRQ asserted mid-batch slips by
/// up to ~4 instructions at budget 16. Zelda's ceiling is therefore
/// 8 until an IRQ-predict cap (mirroring the NMI-fire cap) or an
/// IRQ `pending_exit` lands. `should_panic` pins that known ceiling:
/// the `--ignored` gate run stays green today, and the day a fix
/// makes budget 16 lockstep-clean this test fails LOUDLY — the cue
/// to re-measure and promote the rung to a plain pass (and only
/// then consider raising any profile past 8).
#[test]
#[ignore]
#[should_panic(expected = "DIVERGENCE")]
fn zelda_bulk16_lockstep() {
    lockstep_vs_default(ZELDA, 16, max_batches(6_000_000));
}

/// Measured 2026-07-11: PASSES (1875 frames, menus + gameplay).
#[test]
#[ignore]
fn contra_bulk8_lockstep() {
    lockstep_vs_default(CONTRA, 8, max_batches(6_000_000));
}

/// Measured 2026-07-11: PASSES (3639 frames, menus + gameplay).
#[test]
#[ignore]
fn contra_bulk16_lockstep() {
    lockstep_vs_default(CONTRA, 16, max_batches(6_000_000));
}
