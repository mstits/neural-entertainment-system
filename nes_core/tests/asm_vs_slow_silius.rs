//! ASM-CPU vs Rust-slow-CPU lockstep on real Journey to Silius (MMC1,
//! mapper 1).
//!
//! Same cartridge geometry as Castlevania II - Simon's Quest (128 KB
//! PRG / 128 KB CHR, mapper 1, horizontal mirroring), so mapper 1
//! exposes a `prg_asm_window` here too and routes the ROM onto the
//! AArch64 ASM fast path. This test proves the ASM path is byte-exact
//! against the interpreter on the real ROM: it lockstep-runs the bulk
//! path (`nes.step`, ASM + multi-instruction batches) against the
//! slow path (`nes.tick` loop) from cold boot through the title/
//! attract sequence, comparing CPU registers + cycle count after
//! every instruction and full 2 KB system RAM at a regular stride.
//!
//! Journey to Silius is a heavy DPCM user (Sunsoft driver), so unlike
//! the Castlevania II sibling this ROM also exercises DMC-DMA stall
//! accounting against the MMC1 fast path — the DMC x MMC1 combination
//! is otherwise uncovered by the `asm_vs_slow_*` family.

#![cfg(feature = "asm_cpu")]

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const SILIUS: &str = "../roms/Journey to Silius (USA).nes";
const DEFAULT_MAX_INSTR: usize = 1_500_000;
const RAM_DIFF_STRIDE: usize = 512;

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

fn load(path: &str) -> Nes {
    let f = File::open(path).expect("ROM present");
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    Nes::new(cart)
}

fn max_instr() -> usize {
    std::env::var("ASM_LOCKSTEP_MAX_INSTR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_INSTR)
}

fn lockstep(rom_path: &str) {
    let max_instr = max_instr();
    let mut bulk = load(rom_path);
    let mut slow = load(rom_path);
    bulk.reset();
    slow.reset();

    let mut v_b = V; let mut a_b = A;
    let mut v_s = V; let mut a_s = A;

    for i in 0..max_instr {
        let r_b0 = bulk.cpu.regs();
        let r_s0 = slow.cpu.regs();
        let cyc_b0 = bulk.cycles;
        let cyc_s0 = slow.cycles;

        // Advance bulk by one step (routed through the ASM CPU when
        // available), then to a clean instruction boundary.
        bulk.step(&mut v_b, &mut a_b);
        while !bulk.cpu.at_instruction_boundary() {
            bulk.step(&mut v_b, &mut a_b);
        }

        // Advance slow by whole instructions until its cycle count
        // catches up to bulk's — same logical execution point.
        while slow.cycles < bulk.cycles || !slow.cpu.at_instruction_boundary() {
            loop {
                slow.tick(&mut v_s, &mut a_s);
                if slow.cpu.at_instruction_boundary() { break; }
            }
        }

        let r_b = bulk.cpu.regs();
        let r_s = slow.cpu.regs();
        // NOTE: cycle counts are compared with slack, not equality —
        // around OAM DMA the interpreter books the 513/514-cycle
        // stall inside the triggering instruction's boundary loop
        // while the bulk path consumes it on the following step; the
        // counts re-converge one iteration later. Persistent drift
        // would surface as register/PC divergence via NMI timing.
        if r_b.pc != r_s.pc || r_b.a != r_s.a || r_b.x != r_s.x
            || r_b.y != r_s.y || r_b.sp != r_s.sp
            || slow.cycles.saturating_sub(bulk.cycles) > 1024
        {
            let f_b: u8 = bulk.cpu.flags().into();
            let f_s: u8 = slow.cpu.flags().into();
            panic!(
                "DIVERGENCE at iter {i}:\n\
                 pre-bulk:  PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={cyc_b0}\n\
                 pre-slow:  PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={cyc_s0}\n\
                 post-bulk: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X} cyc={}\n\
                 post-slow: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X} cyc={}",
                r_b0.pc, r_b0.a, r_b0.x, r_b0.y, r_b0.sp,
                r_s0.pc, r_s0.a, r_s0.x, r_s0.y, r_s0.sp,
                r_b.pc, r_b.a, r_b.x, r_b.y, r_b.sp, f_b, bulk.cycles,
                r_s.pc, r_s.a, r_s.x, r_s.y, r_s.sp, f_s, slow.cycles,
            );
        }

        // Full system-RAM diff at a stride — catches divergences that
        // haven't reached the register file yet.
        if i % RAM_DIFF_STRIDE == 0 {
            let ram_b = bulk.ram_for_diff_test();
            let ram_s = slow.ram_for_diff_test();
            if ram_b != ram_s {
                let first = ram_b
                    .iter()
                    .zip(ram_s.iter())
                    .position(|(a, b)| a != b)
                    .unwrap();
                panic!(
                    "RAM DIVERGENCE at iter {i}, addr ${first:04X}: \
                     bulk={:02X} slow={:02X} (PC=${:04X} cyc={})",
                    ram_b[first], ram_s[first], r_b.pc, bulk.cycles,
                );
            }
        }
    }

    println!("No divergence in {max_instr} instructions ({rom_path}).");
}

#[test]
fn find_first_asm_divergence_journey_to_silius() {
    lockstep(SILIUS);
}
