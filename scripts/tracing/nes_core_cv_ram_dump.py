"""Per-frame RAM dump of a CV input tape through our core — the "ours"
side of the Mesen lockstep comparison (2048 bytes/frame, same layout
mesen_cv_tape_dump.lua emits). All hw flags on; --nmi-subcycle adds the
phi-2 NMI latch on top (the A/B knob for the lockstep gates).

--mesen-align applies each tape byte ONE FRAME LATER. Our env.step
convention polls inputs one frame earlier than mesen_cv_tape_dump.lua's
inputPolled callback; with this flag the two harnesses agree and the
comparison is clean: 2026-08-07 receipt — full 14,401-frame state
lockstep vs Mesen (state_fork_onset.py --cert-end 3400 reports no
persistent divergence, phi-2 flag on or off), where the UNALIGNED run
"forks" at frame 3435. That fork, the OAM-DMA parity anti-phase and the
post-11370 counter chaos were all downstream of the one-frame input
phase, not emulator behavior. Default remains unaligned so the banked
byte-identity receipts stay reproducible."""
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
    ap.add_argument("--mesen-align", action="store_true",
                    help="apply tape bytes one frame later, matching "
                         "mesen_cv_tape_dump.lua's inputPolled phase "
                         "(see module doc)")
    args = ap.parse_args()

    tape = Path(args.tape).read_bytes()
    env = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
    for name in HW_PRE_RESET:
        getattr(env, name)(True)
    if args.nmi_subcycle:
        env.set_hw_nmi_subcycle_phase(True)
    env.reset_no_advance()
    env.set_hw_mmio_read_timing(True)

    feed = tape[1:]
    if args.mesen_align:
        feed = bytes([0]) + feed[:-1]
    with open(args.out, "wb") as f:
        # Tape byte 0 is the reset_no_advance power-on frame itself:
        # dump the post-reset RAM as frame 0, then step bytes 1..N.
        f.write(bytes(env.get_ram_range(0, 2048)))
        for byte in feed:
            env.step(byte)
            f.write(bytes(env.get_ram_range(0, 2048)))
    print(f"[done] {len(tape)} frames -> {args.out} "
          f"(nmi_subcycle={'ON' if args.nmi_subcycle else 'off'}, "
          f"mesen_align={'ON' if args.mesen_align else 'off'})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
