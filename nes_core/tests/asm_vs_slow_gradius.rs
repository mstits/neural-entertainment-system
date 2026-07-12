//! ASM-CPU vs Rust-slow-CPU lockstep on real Gradius (CNROM/mapper 3).
//!
//! Mapper3 exposes `prg_asm_ptr` + `asm_bulk_cycles` (CNROM PRG is
//! fully static), which moves CNROM games from the per-cycle
//! interpreter onto the ASM engine on default settings. This test
//! guards that the switch is byte-exact: it lockstep-runs the bulk
//! path (`nes.step`, ASM + multi-instruction batches) against the
//! slow path (`nes.tick` loop) on the real ROM through cold boot +
//! attract mode, comparing CPU state after every instruction and
//! full 2 KB system RAM every 512 instructions.
//!
//! Coverage notes: the run crosses the title/attract sequence, which
//! exercises the CNROM CHR-bank register ($8000+ stores) through the
//! ASM MMIO write callback mid-batch, plus NMI delivery under the
//! `cpu_cycles_until_nmi_fire` bulk cap.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::BufReader;

const GRADIUS: &str = "../roms/Gradius (USA).nes";
const MAX_INSTR: usize = 1_500_000;
const RAM_CHECK_EVERY: usize = 512;

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

#[test]
#[cfg_attr(not(feature = "asm_cpu"), ignore)]
fn find_first_asm_divergence_gradius() {
    let mut bulk = load();
    let mut slow = load();
    bulk.reset();
    slow.reset();

    let mut v_b = V; let mut a_b = A;
    let mut v_s = V; let mut a_s = A;

    // Default horizon (~12.9M CPU cycles) crosses boot, title, attract
    // mode, demo gameplay and many OAM DMAs / CHR bank switches.
    // GRADIUS_LOCKSTEP_MAX overrides for longer soak runs.
    let max_instr: usize = std::env::var("GRADIUS_LOCKSTEP_MAX")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_INSTR);

    for i in 0..max_instr {
        let r_b0 = bulk.cpu.regs();
        let r_s0 = slow.cpu.regs();
        let cyc_b0 = bulk.cycles;
        let cyc_s0 = slow.cycles;

        // Advance bulk by one ASM step; drain to a clean instruction
        // boundary if the step left a pending NMI/IRQ scheduled.
        bulk.step(&mut v_b, &mut a_b);
        while !bulk.cpu.at_instruction_boundary() {
            bulk.step(&mut v_b, &mut a_b);
        }

        // Advance slow until its cycle count catches up to bulk's,
        // ending exactly on a CPU instruction boundary.
        while slow.cycles < bulk.cycles || !slow.cpu.at_instruction_boundary() {
            loop {
                slow.tick(&mut v_s, &mut a_s);
                if slow.cpu.at_instruction_boundary() { break; }
            }
        }

        let r_b = bulk.cpu.regs();
        let r_s = slow.cpu.regs();
        if slow.cycles != bulk.cycles {
            panic!(
                "CYCLE SKEW at iter {i}: bulk={} slow={} (pre-bulk PC=${:04X} \
                 post-bulk PC=${:04X} post-slow PC=${:04X})",
                bulk.cycles, slow.cycles, r_b0.pc, r_b.pc, r_s.pc,
            );
        }
        if r_b.pc != r_s.pc || r_b.a != r_s.a || r_b.x != r_s.x
            || r_b.y != r_s.y || r_b.sp != r_s.sp
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

        if i % RAM_CHECK_EVERY == 0 {
            for addr in 0u16..0x800 {
                let b = bulk.system_ram_byte(addr);
                let s = slow.system_ram_byte(addr);
                assert_eq!(
                    b, s,
                    "RAM DIVERGENCE at iter {i} addr ${addr:04X}: \
                     bulk={b:02X} slow={s:02X} (PC=${:04X})",
                    r_b.pc,
                );
            }
        }
    }

    println!("No divergence in {max_instr} instructions.");
}
