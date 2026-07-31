"""Produce a per-instruction CPU trace of our emulator in the same
format as Mesen's trace_logger.lua. Outputs to stdout or a file.

Format:
    PC OP A:AA X:XX Y:YY P:PP SP:SP PPU:sl,cyc CYC:total

Usage:
    python nes_core_cpu_trace.py [--rom PATH] [--tape PATH]
                                 [--frames N] [--out FILE]

If --tape given, replays tape first then logs from that point.
Default caps output at 200000 instructions (~3000 frames).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nes_core


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", type=Path,
                   default=REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes")
    p.add_argument("--tape", type=Path,
                   default=REPO / "roms" / "zelda_start_419.state.bin")
    p.add_argument("--tape-frames", type=int, default=1010,
                   help="Replay N tape frames before tracing (0 = no tape)")
    p.add_argument("--trace-frames", type=int, default=5,
                   help="Number of frames to trace after tape playback")
    p.add_argument("--out", type=Path, default=Path("/tmp/ours_zelda_trace.txt"))
    p.add_argument("--max-instr", type=int, default=200_000)
    p.add_argument("--hw", action="store_true",
                   help="Enable hardware-true MMIO read timing")
    p.add_argument("--hw-reset", action="store_true",
                   help="Enable hardware-true boot alignment (CYC=7, PPU dot 25)")
    args = p.parse_args()

    env = nes_core.NESEnvironment(rom_path=str(args.rom), frame_skip=1)
    if args.hw_reset:
        env.set_hw_reset_alignment(True)
    # True power-on alignment for cross-emulator diffs: reset() advances a
    # frame before returning (Mesen traces from CYC 7); with no tape we
    # want instruction 1 of the reset vector.
    if args.tape_frames == 0 and hasattr(env, "reset_no_advance"):
        env.reset_no_advance()
    else:
        env.reset()
    if args.hw:
        env.set_hw_mmio_read_timing(True)

    if args.tape_frames and args.tape.exists():
        tape = list(args.tape.read_bytes())
        for b in tape[:args.tape_frames]:
            env.step(int(b) & 0xFF)
        print(f"Replayed {args.tape_frames} tape frames", file=sys.stderr)

    with args.out.open("w") as fh:
        # For each requested trace frame, step one instruction at a
        # time, logging state before each.
        start_frame = env.ppu_state()[1]
        instr_count = 0
        target_frame = start_frame + args.trace_frames
        while env.ppu_state()[1] < target_frame and instr_count < args.max_instr:
            pc, a, x, y, sp, p, nmi = env.cpu_state()
            # Use get_ram for RAM; we don't have a PRG peek helper bound
            # to Python, so show opcode=00 for PRG-space PC. Diff scripts
            # should ignore the opcode byte column; PC + register state
            # is enough to align and compare.
            opcode = env.get_ram(pc) if pc < 0x2000 else 0
            ppu_c, ppu_f, ppu_sl, ppu_slc, cpu_cyc, _ = env.ppu_state()
            fh.write(
                f"{pc:04X} {opcode:02X} A:{a:02X} X:{x:02X} Y:{y:02X} "
                f"P:{p:02X} SP:{sp:02X} PPU:{ppu_sl:3d},{ppu_slc:3d} "
                f"CYC:{cpu_cyc}\n"
            )
            env.step_one_instruction()
            instr_count += 1

        print(f"Logged {instr_count} instructions -> {args.out}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
