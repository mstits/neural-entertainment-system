"""Per-frame RAM dump of a CV input tape through our core — the "ours"
side of the Mesen lockstep comparison (2048 bytes/frame, same layout
mesen_cv_tape_dump.lua emits). All hw flags on; --nmi-subcycle adds the
phi-2 NMI latch on top (the A/B knob for the lockstep gates)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

ROM = str(REPO / "roms" / "Castlevania (USA).nes")

HW_PRE_RESET = [
    "set_hw_reset_alignment",
    "set_hw_dmc_stall_timing",
    "set_hw_frame_anchor",
    "set_hw_nmi_poll_timing",
    "set_hw_mmio_write_timing",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tape", default="/tmp/cv_tape.bin")
    ap.add_argument("--out", default="/tmp/ours_cv_ram.bin")
    ap.add_argument("--nmi-subcycle", action="store_true",
                    help="enable hw_nmi_subcycle_phase (the phi-2 latch)")
    args = ap.parse_args()

    tape = Path(args.tape).read_bytes()
    env = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
    for name in HW_PRE_RESET:
        getattr(env, name)(True)
    if args.nmi_subcycle:
        env.set_hw_nmi_subcycle_phase(True)
    env.reset_no_advance()
    env.set_hw_mmio_read_timing(True)

    with open(args.out, "wb") as f:
        # Tape byte 0 is the reset_no_advance power-on frame itself:
        # dump the post-reset RAM as frame 0, then step bytes 1..N.
        f.write(bytes(env.get_ram_range(0, 2048)))
        for byte in tape[1:]:
            env.step(byte)
            f.write(bytes(env.get_ram_range(0, 2048)))
    print(f"[done] {len(tape)} frames -> {args.out} "
          f"(nmi_subcycle={'ON' if args.nmi_subcycle else 'off'})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
