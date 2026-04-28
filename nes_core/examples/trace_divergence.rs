//! Compare ASM-path Nes::step vs pure-per-cycle Nes::tick for a ROM,
//! finding the first instruction where the two diverge. Useful for
//! bisecting ASM integration bugs on banked mappers.
//!
//! Run: cargo run --release --features asm_cpu --example trace_divergence -- <rom> <steps>

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};

struct Null;
impl VideoSink for Null {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
impl AudioSink for Null {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn load(path: &str) -> Nes {
    let bytes = std::fs::read(path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    let mut nes = Nes::new(cart);
    nes.reset();
    nes
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../roms/zelda.ines1.nes".into());
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let mut a = load(&path);  // this path uses whatever Nes::step does (ASM if feature on)
    let mut b = load(&path);  // reference: pure per-cycle tick

    let mut va = Null; let mut aa = Null;
    let mut vb = Null; let mut ab = Null;

    let verbose = std::env::args().any(|a| a == "-v" || a == "--verbose");
    // Optional --press-start: holds Start for N outer steps starting at step M.
    // Defaults (when flag present): hold Start between outer steps [1_000, 1_050).
    let press_start = std::env::args().any(|a| a == "--press-start");
    for i in 0..steps {
        if press_start {
            let hold = (1_000..1_050).contains(&i);
            use nes_core::input::Button;
            a.game_pad_1().set_button_pressed(Button::Start, hold);
            b.game_pad_1().set_button_pressed(Button::Start, hold);
        }
        let pre_a = a.get_state();
        let cycles_a = a.step(&mut va, &mut aa);
        for _ in 0..cycles_a {
            b.tick(&mut vb, &mut ab);
        }

        let sa = a.get_state();
        let sb = b.get_state();
        let ra = sa.cpu.regs;
        let rb = sb.cpu.regs;
        let pa: u8 = sa.cpu.flags.into();
        let pb: u8 = sb.cpu.flags.into();

        if verbose && i < 20 {
            println!("step {:4} start_pc={:04X} cycles={} → A pc={:04X} sp={:02X} | B pc={:04X} sp={:02X}",
                i, pre_a.cpu.regs.pc, cycles_a, ra.pc, ra.sp, rb.pc, rb.sp);
        }

        let cpu_diff = ra.pc != rb.pc || ra.a != rb.a || ra.x != rb.x
            || ra.y != rb.y || ra.sp != rb.sp || (pa & 0xCF) != (pb & 0xCF);

        // Compare total elapsed cycles — a delta here means PPU/APU
        // catch-up is running at a different rate between the two
        // paths (ASM may under/over-count mapper-write penalty cycles,
        // for example).
        let cycles_diff = sa.cycles != sb.cycles;

        // Fingerprint mapper state by round-trip serializing via
        // bincode — gives us a cheap opaque blob that diffs bytewise
        // but doesn't depend on inspecting per-mapper fields.
        let ma = bincode::serialize(&sa.mapper).ok();
        let mb = bincode::serialize(&sb.mapper).ok();
        let mapper_diff = ma != mb;

        // Same trick for PPU state — catches palette RAM / OAM /
        // scroll-latch / VRAM divergences that the CPU-only diff
        // would miss. Zelda's first-cave sword-pickup logic reads
        // $2002 in a tight loop; if the PPU internal latches diverge
        // between the two paths even one cycle, the game takes a
        // different branch and Link never picks up the sword.
        let pa_blob = bincode::serialize(&sa.ppu).ok();
        let pb_blob = bincode::serialize(&sb.ppu).ok();
        let ppu_diff = pa_blob != pb_blob;

        // APU state too — frame-counter IRQ timing differs between
        // bulk catch-up (ASM path) and per-cycle tick; could affect
        // any game that IRQ-polls.
        let aa_blob = bincode::serialize(&sa.apu).ok();
        let ab_blob = bincode::serialize(&sb.apu).ok();
        let apu_diff = aa_blob != ab_blob;

        // RAM bytes.
        let ram_diff = sa.ram != sb.ram;

        if cpu_diff || cycles_diff || mapper_diff || ppu_diff || apu_diff || ram_diff {
            println!("DIVERGENCE at outer step {} (start_pc={:04X}, cycles consumed this step: {})",
                i, pre_a.cpu.regs.pc, cycles_a);
            println!("  A (Nes::step):   pc={:04X} a={:02X} x={:02X} y={:02X} sp={:02X} p={:02X} cycles={}",
                ra.pc, ra.a, ra.x, ra.y, ra.sp, pa, sa.cycles);
            println!("  B (per-cycle):   pc={:04X} a={:02X} x={:02X} y={:02X} sp={:02X} p={:02X} cycles={}",
                rb.pc, rb.a, rb.x, rb.y, rb.sp, pb, sb.cycles);
            if cpu_diff { println!("  CPU regs diverged"); }
            if cycles_diff {
                println!("  CPU cycles diverged: A={} B={} (delta {})",
                    sa.cycles, sb.cycles, sa.cycles as i64 - sb.cycles as i64);
            }
            if mapper_diff {
                let la = ma.as_ref().map(|v| v.len()).unwrap_or(0);
                let lb = mb.as_ref().map(|v| v.len()).unwrap_or(0);
                println!("  Mapper state diverged (blob sizes A={} B={})", la, lb);
                if let (Some(a_bytes), Some(b_bytes)) = (ma.as_ref(), mb.as_ref()) {
                    for (i, (x, y)) in a_bytes.iter().zip(b_bytes.iter()).enumerate() {
                        if x != y {
                            println!("    first differing byte at offset {}: A={:02X} B={:02X}",
                                i, x, y);
                            break;
                        }
                    }
                }
            }
            if ppu_diff {
                let la = pa_blob.as_ref().map(|v| v.len()).unwrap_or(0);
                let lb = pb_blob.as_ref().map(|v| v.len()).unwrap_or(0);
                println!("  PPU state diverged (blob sizes A={} B={})", la, lb);
                if let (Some(a_bytes), Some(b_bytes)) = (pa_blob.as_ref(), pb_blob.as_ref()) {
                    for (i, (x, y)) in a_bytes.iter().zip(b_bytes.iter()).enumerate() {
                        if x != y {
                            println!("    first differing byte at offset {}: A={:02X} B={:02X}",
                                i, x, y);
                            break;
                        }
                    }
                }
            }
            if apu_diff {
                println!("  APU state diverged");
            }
            if ram_diff {
                for (i, (x, y)) in sa.ram.iter().zip(sb.ram.iter()).enumerate() {
                    if x != y {
                        println!("  RAM diverged: first at $0{:03X}: A={:02X} B={:02X}", i, x, y);
                        break;
                    }
                }
            }
            return;
        }
    }
    println!("No divergence found in {} outer steps (A cycles total: {})", steps, a.get_state().cycles);
}
