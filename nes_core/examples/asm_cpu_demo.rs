//! AArch64 ASM CPU end-to-end demo.
//!
//! Runs a synthetic NROM ROM through `Nes::step` and reports how many
//! instructions went through the ASM fast path vs fell back to Rust.
//! Requires `--features asm_cpu` (otherwise the path is absent).
//!
//! ```
//! cargo run --release --features asm_cpu --example asm_cpu_demo
//! ```

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::cartridge::Cartridge;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::cpu_asm::ASM_HITS;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::nes::Nes;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use nes_core::sink::{AudioSink, VideoSink};
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
use std::sync::atomic::Ordering;

#[cfg(not(all(target_arch = "aarch64", feature = "asm_cpu")))]
fn main() {
    eprintln!(
        "This example requires aarch64 + --features asm_cpu. \
         Re-run: cargo run --release --features asm_cpu --example asm_cpu_demo"
    );
    std::process::exit(1);
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
struct Null;
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
impl VideoSink for Null {
    fn write_frame(&mut self, _: &[u8]) {}
    fn frame_written(&self) -> bool { false }
    fn pixel_size(&self) -> usize { 4 }
}
#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
impl AudioSink for Null {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn build_rom() -> Vec<u8> {
    // iNES header for 2×16KB PRG (32KB = Mapper 0 with prg_asm_ptr
    // enabled) + 0 CHR banks + mapper 0 + mirroring=0.
    let mut rom = Vec::with_capacity(16 + 32 * 1024);
    rom.extend_from_slice(b"NES\x1a");
    rom.push(2); // PRG 32KB
    rom.push(0); // CHR 0
    rom.push(0); // flags 6 — mapper low nibble 0
    rom.push(0); // flags 7
    rom.extend_from_slice(&[0u8; 8]);

    let mut prg = vec![0u8; 32 * 1024];

    // Reset vector at $FFFC → $C000 (prg[0x3FFC..0x3FFE]).
    let reset_vec = prg.len() - 4;
    prg[reset_vec] = 0x00;
    prg[reset_vec + 1] = 0xC0;

    // Program at $C000 (prg[0x4000]). Uses only ASM-ported opcodes.
    //
    //   loop_start:
    //     LDA #$00        A9 00        (2)
    //     TAX             AA           (2)
    //     TAY             A8           (2)
    //     SEC             38           (2)
    //     CLC             18           (2)
    //   inner:
    //     INX             E8           (2)
    //     INY             C8           (2)
    //     STA $10         85 10        (3)
    //     STA $20         85 20        (3)
    //     LDA $10         A5 10        (3)
    //     ORA #$01        09 01        (2)
    //     AND #$FE        29 FE        (2)
    //     EOR #$55        49 55        (2)
    //     CPX #$0A        E0 0A        (2)
    //     BNE inner       D0 EE        (2/3)
    //     JMP loop_start  4C 00 C0     (3)
    let base = 0x4000usize;
    let program: &[u8] = &[
        0xA9, 0x00,       // LDA #$00
        0xAA,             // TAX
        0xA8,             // TAY
        0x38,             // SEC
        0x18,             // CLC
        // inner (offset +5):
        0xE8,             // INX
        0xC8,             // INY
        0x85, 0x10,       // STA $10
        0x85, 0x20,       // STA $20
        0xA5, 0x10,       // LDA $10
        0x09, 0x01,       // ORA #$01
        0x29, 0xFE,       // AND #$FE
        0x49, 0x55,       // EOR #$55
        0xE0, 0x0A,       // CPX #$0A
        0xD0, 0xEE,       // BNE -0x12 (back to inner)
        0x4C, 0x00, 0xC0, // JMP $C000
    ];
    prg[base..base + program.len()].copy_from_slice(program);

    rom.extend(prg);
    rom
}

#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]
fn main() {
    let arg = std::env::args().nth(1);
    let (cart, label) = match arg {
        Some(path) => {
            let bytes = std::fs::read(&path).expect("read rom");
            let c = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
            (c, format!("{} (mapper {})", path, 0))
        }
        None => {
            let rom = build_rom();
            let c = Cartridge::load(&mut std::io::Cursor::new(rom)).expect("load rom");
            (c, "synthetic 32 KB NROM arith-loop".to_string())
        }
    };

    let mapper_num = cart.mapper;
    let mut nes = Nes::new(cart);
    nes.reset();

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" AArch64 ASM 6502 core — end-to-end integration demo");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("ROM:    {}", label);
    println!("Mapper: {} (ASM engages only on mapper 0 with 16/32 KB PRG)", mapper_num);
    println!();

    let before = ASM_HITS.load(Ordering::Relaxed);

    let mut vs = Null;
    let mut au = Null;
    let target_steps = 1_000_000usize;

    let start = std::time::Instant::now();
    for _ in 0..target_steps {
        nes.step(&mut vs, &mut au);
    }
    let dur = start.elapsed();

    let after = ASM_HITS.load(Ordering::Relaxed);
    let asm_runs = (after - before) as f64;
    let total = target_steps as f64;
    let hit_rate = 100.0 * asm_runs / total;

    println!("Nes::step calls:     {}", target_steps);
    println!("Wall time:           {:.3}s ({:.2} ns/step)",
             dur.as_secs_f64(),
             dur.as_nanos() as f64 / total);
    println!("ASM fast-path hits:  {}  ({:.2}%)",
             after - before, hit_rate);
    println!("Rust fallback hits:  {}  ({:.2}%)",
             target_steps - (after - before) as usize, 100.0 - hit_rate);
    println!("CPU cycles advanced: {}", nes.cycles);
    println!();
    println!("Effective throughput: {:.2}M steps/sec",
             total / dur.as_secs_f64() / 1_000_000.0);
    println!();
    if hit_rate > 99.0 {
        println!("✓ ASM fast path is engaged on every instruction of this ROM.");
    } else if hit_rate > 50.0 {
        println!("✓ ASM fast path handling majority of instructions.");
    } else {
        println!("ASM hit rate is low — most instructions fell back to Rust.");
    }
    println!("═══════════════════════════════════════════════════════════════════");
}
