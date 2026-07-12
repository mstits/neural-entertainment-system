//! Full-state ASM-vs-slow divergence finder for Gradius (CNROM).
//!
//! Complements `asm_vs_slow_gradius.rs`: the lockstep test asserts
//! CPU-visible equality (registers, cycles, RAM), while this one
//! serializes the ENTIRE emulator state (RAM, mapper, CPU, PPU, APU)
//! and compares field by field — catching internal divergences long
//! before they become CPU-visible. This harness found the indexed-
//! store MMIO commit-cycle misalignment (APU pulse phase diverged
//! ~37k instructions before the CPU-visible skew) and the early-
//! commit read misalignment at the sprite-0 poll.
//!
//! Env overrides: DIAG_EVERY (compare interval, default 1024),
//! DIAG_FROM (first compared iteration), DIAG_MAX (horizon).

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const GRADIUS: &str = "../roms/Gradius (USA).nes";
const MAX_INSTR: usize = 1_500_000;

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

fn load() -> Nes {
    let f = File::open(GRADIUS).expect("Gradius ROM present");
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    Nes::new(cart)
}

fn state_fields(nes: &Nes) -> Vec<(&'static str, Vec<u8>)> {
    let mut s = nes.get_state();
    // ASM path accounts cpu.cycles_total differently (bookkeeping
    // only; DMA alignment consumes nes.cycles). Not a semantic field.
    s.cpu.cycles_total = 0;
    // Per-instruction scratch — re-initialized at every dispatch; the
    // ASM path does not maintain them between instructions.
    s.cpu.opcode = 0;
    s.cpu.cycle = 0;
    s.cpu.addr_abs = 0;
    s.cpu.temp_addr_low = 0;
    s.cpu.base_addr = 0;
    s.cpu.rel_offset = 0;
    s.cpu.fetched_data = 0;
    vec![
        ("ram", bincode::serialize(&s.ram).unwrap()),
        ("mapper", bincode::serialize(&s.mapper).unwrap()),
        ("cpu", bincode::serialize(&s.cpu).unwrap()),
        ("ppu", bincode::serialize(&s.ppu).unwrap()),
        ("apu", bincode::serialize(&s.apu).unwrap()),
        ("cycles", bincode::serialize(&s.cycles).unwrap()),
    ]
}

#[test]
#[cfg_attr(not(feature = "asm_cpu"), ignore)]
fn gradius_first_state_divergence() {
    let mut bulk = load();
    let mut slow = load();
    bulk.reset();
    slow.reset();

    let mut v_b = V; let mut a_b = A;
    let mut v_s = V; let mut a_s = A;

    let check_every: usize = std::env::var("DIAG_EVERY")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let check_from: usize = std::env::var("DIAG_FROM")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let max_instr: usize = std::env::var("DIAG_MAX")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_INSTR);

    for i in 0..max_instr {
        let pre_pc = bulk.cpu.regs().pc;
        let pre_cyc = bulk.cycles;
        bulk.step(&mut v_b, &mut a_b);
        while !bulk.cpu.at_instruction_boundary() {
            bulk.step(&mut v_b, &mut a_b);
        }
        while slow.cycles < bulk.cycles || !slow.cpu.at_instruction_boundary() {
            loop {
                slow.tick(&mut v_s, &mut a_s);
                if slow.cpu.at_instruction_boundary() { break; }
            }
        }

        if i >= check_from && i % check_every == 0 {
            let fb = state_fields(&bulk);
            let fs = state_fields(&slow);
            let mut diffs = Vec::new();
            for ((name, b), (_, s)) in fb.iter().zip(fs.iter()) {
                if b != s {
                    let first = b.iter().zip(s.iter()).position(|(x, y)| x != y);
                    let n = b.iter().zip(s.iter()).filter(|(x, y)| x != y).count();
                    let lo = first.unwrap_or(0).saturating_sub(8);
                    let hi = (lo + 32).min(b.len()).min(s.len());
                    diffs.push(format!(
                        "{name}: {n} bytes differ (len {} vs {}), first at {:?}\n\
                         bulk[{lo}..{hi}]: {:02x?}\n\
                         slow[{lo}..{hi}]: {:02x?}",
                        b.len(), s.len(), first, &b[lo..hi], &s[lo..hi],
                    ));
                }
            }
            if !diffs.is_empty() {
                panic!(
                    "STATE DIVERGENCE at iter {i} (pre PC=${pre_pc:04X} \
                     cyc={pre_cyc} | bulk cyc={} slow cyc={} \
                     bulk PC=${:04X} slow PC=${:04X}):\n{}",
                    bulk.cycles, slow.cycles,
                    bulk.cpu.regs().pc, slow.cpu.regs().pc,
                    diffs.join("\n"),
                );
            }
        }
    }
    println!("No state divergence in {max_instr} instructions.");
}
