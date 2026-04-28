//! Find the first ASM-CPU vs Rust-slow-CPU divergence on real SMB.
//!
//! Headless reproducer for the "Mario falls through floor" bug:
//! Python harness `/tmp/find_first_asm_diverge.py` already proved
//! that bulk-path (`nes.step`, with `asm_cpu` feature on) diverges
//! from slow-path (`nes.tick` loop) at SMB cold-boot frame 2 — at
//! the same PC, Y differs by 1.
//!
//! This test lockstep-runs both paths on the real ROM, comparing CPU
//! state after every instruction, and reports the first divergence
//! with full register state and ROM PC. The diff names the buggy
//! ASM opcode for follow-up.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const SMB: &str = "../roms/Super Mario Bros. (World).nes";
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
    let f = File::open(SMB).expect("SMB ROM present");
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    Nes::new(cart)
}

#[test]
#[cfg_attr(not(feature = "asm_cpu"), ignore)]
fn find_first_asm_divergence_smb() {
    // Two emulators, identical reset state.
    let mut bulk = load();
    let mut slow = load();
    bulk.reset();
    slow.reset();

    let mut v_b = V; let mut a_b = A;
    let mut v_s = V; let mut a_s = A;

    // Walk one instruction at a time. After each instruction,
    // compare CPU state. Bulk uses `nes.step()` (which routes
    // through ASM CPU when available); slow uses `nes.tick()` loop
    // until instruction boundary.
    for i in 0..MAX_INSTR {
        // Capture pre-instruction state for diagnostic.
        let r_b0 = bulk.cpu.regs();
        let r_s0 = slow.cpu.regs();
        let cyc_b0 = bulk.cycles;
        let cyc_s0 = slow.cycles;

        // Advance bulk by one ASM CPU step (which batches up to
        // `mapper.asm_bulk_cycles()` cycles per call — multiple full
        // 6502 instructions for fast opcodes). If that step left a
        // pending NMI/IRQ scheduled (poll_interrupts set
        // `instruction = NMI_INTERRUPT`), keep stepping bulk until
        // we're back at a clean instruction boundary so we compare
        // apples-to-apples with slow's catchup loop (which naturally
        // runs through NMI service inside its tick loop).
        bulk.step(&mut v_b, &mut a_b);
        while !bulk.cpu.at_instruction_boundary() {
            bulk.step(&mut v_b, &mut a_b);
        }

        // Advance slow until its cycle count catches up to bulk's,
        // with each iteration ending exactly on a CPU instruction
        // boundary. At end of this loop, slow.cycles >= bulk.cycles
        // and slow is at an instruction boundary — same logical
        // execution point as bulk.
        while slow.cycles < bulk.cycles || !slow.cpu.at_instruction_boundary() {
            loop {
                slow.tick(&mut v_s, &mut a_s);
                if slow.cpu.at_instruction_boundary() { break; }
            }
        }

        let r_b = bulk.cpu.regs();
        let r_s = slow.cpu.regs();
        // Also compare key zero-page bytes that the original Mario-falls
        // bug surfaced through ($05, $06, $07 — JSR-dispatch ZP slots
        // and the indirect-store target the ground-tile-buffer write
        // uses at PC=$94F7).
        let zp_b = (
            bulk.system_ram_byte(0x05),
            bulk.system_ram_byte(0x06),
            bulk.system_ram_byte(0x07),
        );
        let zp_s = (
            slow.system_ram_byte(0x05),
            slow.system_ram_byte(0x06),
            slow.system_ram_byte(0x07),
        );
        if r_b.pc != r_s.pc || r_b.a != r_s.a || r_b.x != r_s.x
            || r_b.y != r_s.y || r_b.sp != r_s.sp || zp_b != zp_s
        {
            let f_b: u8 = bulk.cpu.flags().into();
            let f_s: u8 = slow.cpu.flags().into();
            panic!(
                "DIVERGENCE at iter {i}:\n\
                 pre-bulk:  PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={cyc_b0}\n\
                 pre-slow:  PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} cyc={cyc_s0}\n\
                 post-bulk: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X} cyc={}\n\
                 post-slow: PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X} P={:02X} cyc={}\n\
                 ZP bulk: $05={:02X} $06={:02X} $07={:02X}\n\
                 ZP slow: $05={:02X} $06={:02X} $07={:02X}",
                r_b0.pc, r_b0.a, r_b0.x, r_b0.y, r_b0.sp,
                r_s0.pc, r_s0.a, r_s0.x, r_s0.y, r_s0.sp,
                r_b.pc, r_b.a, r_b.x, r_b.y, r_b.sp, f_b, bulk.cycles,
                r_s.pc, r_s.a, r_s.x, r_s.y, r_s.sp, f_s, slow.cycles,
                zp_b.0, zp_b.1, zp_b.2,
                zp_s.0, zp_s.1, zp_s.2,
            );
        }
    }

    println!("No divergence in {MAX_INSTR} instructions.");
}
