//! Per-nes.step RAM-write trace for SMB cold boot.
//!
//! Runs `nes.step()` in a loop on real SMB, after each call scans
//! the 2 KB internal RAM for byte changes vs the previous snapshot,
//! and writes one log line per change to `/tmp/smb_writes_<label>.txt`.
//!
//! Usage:
//!   cargo test --release --test smb_write_trace --features asm_cpu
//!   cargo test --release --test smb_write_trace
//!
//! The output filename embeds the build flavor:
//!   * `/tmp/smb_writes_asm.txt`  — when compiled with `--features asm_cpu`
//!   * `/tmp/smb_writes_slow.txt` — when compiled without it
//!
//! Diff the two files to find the first instruction where ASM and
//! Rust slow CPU produce different RAM-write effects on real SMB ROM
//! cold boot. The line format is:
//!   step <N>  cyc <CPU_CYCLE>  PC=$XXXX  $AAAA: 0xBB → 0xCC
//!
//! Why this exists: the residual asm_cpu correctness gap (Mario
//! falls through floor when asm_cpu is enabled by default) survived
//! all the structural fixes shipped in commits 99daf4f / 3e97a94 /
//! 8641db5. The cycle-locked three-way test located divergence at
//! end of 4th env.step (ZP $06 differs by 0x31 vs 0x00) but couldn't
//! discriminate between (a) different control flow vs (b) different
//! write target. This trace produces (a-b) discriminating data.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::sink::{AudioSink, VideoSink};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

const SMB: &str = "../roms/Super Mario Bros. (World).nes";
const MAX_STEPS: usize = 200_000;

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

fn snapshot_ram(nes: &Nes) -> [u8; 2048] {
    let mut buf = [0u8; 2048];
    for i in 0..2048 {
        buf[i] = nes.system_ram_byte(i as u16);
    }
    buf
}

#[cfg(feature = "asm_cpu")]
const LABEL: &str = "asm";
#[cfg(not(feature = "asm_cpu"))]
const LABEL: &str = "slow";

#[test]
fn write_trace_smb_cold_boot() {
    let f = File::open(SMB).expect("SMB ROM present at ../roms/");
    let cart = Cartridge::load(&mut BufReader::new(f)).expect("cart load");
    let mut nes = Nes::new(cart);
    nes.reset();

    let mut v = V; let mut a = A;

    let out_path = format!("/tmp/smb_writes_{LABEL}.txt");
    let out = File::create(&out_path).expect("open trace out");
    let mut w = BufWriter::new(out);

    let mut prev = snapshot_ram(&nes);

    // Within this step range, log EVERY step (including those with no
    // RAM write) — needed to find which instruction in a stretch added
    // an extra cycle. Outside the range, log only writes (keeps file
    // small).
    let log_all_from: usize = std::env::var("LOG_ALL_FROM")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let log_all_to: usize = std::env::var("LOG_ALL_TO")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    // Per-step PC trace mode: log EVERY step's pre-PC + post-PC + A
    // (regardless of writes). Set TRACE_PC=1 env var to enable. Output
    // file gets "_pc" suffix to keep separate from write trace.
    let trace_pc = std::env::var("TRACE_PC").map(|v| v == "1").unwrap_or(false);
    let pc_path = format!("/tmp/smb_pc_{LABEL}.txt");
    let pc_file = if trace_pc { Some(File::create(&pc_path).expect("open pc")) } else { None };
    let mut pcw = pc_file.map(BufWriter::new);

    for step in 0..MAX_STEPS {
        let pc_pre = nes.cpu.regs().pc;
        let r_pre = nes.cpu.regs();
        let cyc_pre = nes.cycles;
        nes.step(&mut v, &mut a);
        let cyc = nes.cycles;
        let r = nes.cpu.regs();
        let cur = snapshot_ram(&nes);
        let mut wrote_any = false;
        for addr in 0..2048 {
            if cur[addr] != prev[addr] {
                wrote_any = true;
                writeln!(
                    w,
                    "step {step}  cyc {cyc}  PC=${pc_pre:04X}  ${addr:04X}: 0x{:02X} → 0x{:02X}",
                    prev[addr], cur[addr]
                ).unwrap();
            }
        }
        if !wrote_any && step >= log_all_from && step <= log_all_to {
            writeln!(
                w,
                "step {step}  cyc {cyc}  PC=${pc_pre:04X}  (no write, +{}cyc from cyc_pre={cyc_pre})",
                cyc - cyc_pre
            ).unwrap();
        }
        if let Some(pcw_) = pcw.as_mut() {
            writeln!(
                pcw_,
                "{step:>7}  cyc{cyc:>9}  pre PC=${pc_pre:04X} A={:02X} X={:02X} Y={:02X} SP={:02X}  post PC=${:04X} A={:02X} X={:02X} Y={:02X} SP={:02X}",
                r_pre.a, r_pre.x, r_pre.y, r_pre.sp,
                r.pc, r.a, r.x, r.y, r.sp,
            ).unwrap();
        }
        prev = cur;
    }

    eprintln!("trace written to {out_path}");
}
