#!/usr/bin/env python3
"""Post-rebuild gate for event-driven-PPU Stage 3 (sub-cycle bus catch-up).

Run AFTER the next attributed maturin rebuild installs the new .so:

    .venv/bin/python scripts/verify_subcycle_offset.py \
        ["roms/Castlevania (USA).nes"] [--frames 6000]

WHAT IT PROVES (without a Mesen trace)
--------------------------------------
1. hw_event_ppu OFF is byte-for-byte unchanged. The Stage 3 diff only
   touches the hw_event_ppu-gated branch of Nes::tick plus new default-2
   offset fields, so the legacy interleave must be identical. Compare the
   OFF trajectory SHA printed below against the SAME script run on the
   PREVIOUS (Stage 2) .so — they MUST match. This is the OFF-vs-Stage2
   gate (mirror of the 6000-frame CV sha gate) the task requires.

2. hw_event_ppu ON at the default offset (read=write=2, end-of-cycle) is
   byte-identical to OFF. Its SHA must equal the OFF SHA on this build.

3. The sub-cycle offset is LIVE. hw_event_ppu ON with read=write=0 under
   the full --hw-all flag set produces a DIFFERENT trajectory SHA (the
   $2002 poll / $2000 NMI-enable now resolve two dots earlier within
   their access cycle). A differing SHA here demonstrates Stage 3 moves
   the machine; an identical SHA means the offset never engaged (check
   that hw_nmi_poll_timing is on so the per-cycle path — the only one
   that positions a deferred access sub-cycle — actually runs).

WHAT IT DOES NOT PROVE
----------------------
Mesen parity. Whether offset 0/1/2 is the CORRECT Mesen master-clock
phase can only be settled against a Mesen trace. See the recipe printed
at the end (CV tape, the ~4118 first-divergence frame the 4-flag fork
currently sits at; Contra desyncs ~screen 5).
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

# The --hw-all flag set (mirrors scripts/tracing/nes_core_cpu_trace.py):
# every hardware-true timing flag. hw_mmio_read_timing is applied AFTER
# reset (matching the trace harness). hw_nmi_poll_timing is the one that
# forces the per-cycle path Stage 3's sub-cycle offset rides on.
HW_ALL_PRE_RESET = [
    "set_hw_reset_alignment",
    "set_hw_dmc_stall_timing",
    "set_hw_frame_anchor",
    "set_hw_nmi_poll_timing",
    "set_hw_mmio_write_timing",
]
HW_ALL_POST_RESET = ["set_hw_mmio_read_timing"]


def _apply(env, names, arg=True):
    for n in names:
        getattr(env, n)(arg)


def trajectory_sha(rom: str, frames: int, *, event_ppu: bool,
                   read_off: int | None, write_off: int | None) -> str:
    """Idle-run the ROM for `frames` frames under the --hw-all flag set
    (+ optional hw_event_ppu / sub-cycle offsets) and SHA256 the full
    per-frame save_state() trajectory."""
    env = nes_core.NESEnvironment(rom_path=rom, frame_skip=1)
    _apply(env, HW_ALL_PRE_RESET)
    env.reset()
    _apply(env, HW_ALL_POST_RESET)
    env.set_hw_event_ppu(event_ppu)
    if read_off is not None and hasattr(env, "set_ppu_read_dot_offset"):
        env.set_ppu_read_dot_offset(read_off)
    if write_off is not None and hasattr(env, "set_ppu_write_dot_offset"):
        env.set_ppu_write_dot_offset(write_off)

    h = hashlib.sha256()
    for _ in range(frames):
        env.step(0)  # idle input — deterministic vblank/NMI trajectory
        h.update(bytes(env.save_state()))
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rom", nargs="?",
                   default=str(REPO / "roms" / "Castlevania (USA).nes"))
    p.add_argument("--frames", type=int, default=6000)
    args = p.parse_args()

    if not Path(args.rom).exists():
        print(f"ROM not found: {args.rom}", file=sys.stderr)
        return 2

    has_offset = hasattr(nes_core.NESEnvironment, "set_ppu_read_dot_offset")
    print(f"rom={args.rom}  frames={args.frames}  "
          f"sub_cycle_setters={'present' if has_offset else 'ABSENT (old .so)'}")

    off = trajectory_sha(args.rom, args.frames,
                         event_ppu=False, read_off=None, write_off=None)
    on2 = trajectory_sha(args.rom, args.frames,
                         event_ppu=True, read_off=2, write_off=2)
    print(f"[1] hw_event_ppu OFF                      sha={off}")
    print(f"[2] hw_event_ppu ON  offset(2,2)          sha={on2}")

    ok = True
    if on2 != off:
        print("    FAIL: ON at default offset diverged from OFF "
              "(Stage 3 default is NOT byte-identical)")
        ok = False
    else:
        print("    ok:   ON default offset == OFF (byte-identical)")

    if has_offset:
        on0 = trajectory_sha(args.rom, args.frames,
                             event_ppu=True, read_off=0, write_off=0)
        print(f"[3] hw_event_ppu ON  offset(0,0)          sha={on0}")
        if on0 == on2:
            print("    WARN: sub-cycle offset produced NO change — the "
                  "per-cycle path likely never ran the deferred access "
                  "(is hw_nmi_poll_timing on?).")
        else:
            print("    ok:   sub-cycle offset is LIVE (trajectory moved).")
    else:
        print("[3] SKIPPED — set_ppu_read/write_dot_offset absent; rebuild "
              "with the Stage 3 .so to exercise the sub-cycle path.")

    print()
    print("OFF-vs-Stage2 gate: run this script on BOTH the previous "
          "(Stage 2) and the new .so; line [1] must be IDENTICAL across "
          "them.")
    print()
    print("MESEN LOCKSTEP EXTENSION (the ON-path fidelity measurement):")
    print("  1. Dump the Mesen ground truth over the CV tape:")
    print("       Mesen --testRunner scripts/tracing/mesen_cv_tape_dump.lua")
    print("  2. Dump ours at each offset (per-cycle path forced by --hw-all):")
    print("       .venv/bin/python scripts/tracing/nes_core_cpu_trace.py \\")
    print("           --hw-all --rom '<CV rom>' --tape '<CV tape>'   # offset 2")
    print("     then re-run with env.set_ppu_read_dot_offset(0/1) and")
    print("     env.set_ppu_write_dot_offset(0/1) wired into the trace script.")
    print("  3. Find the first-divergence frame:")
    print("       .venv/bin/python scripts/tracing/diff_cpu_traces.py \\")
    print("           <mesen_trace> <ours_trace>")
    print("  Baseline: the 4-flag CV fork sits at ~frame 4118 (mesen dawn")
    print("  receipt: 3992->4884). Stage 3 with the calibrated offset must")
    print("  push the first divergence LATER, not earlier. Contra desyncs")
    print("  ~screen 5 and is the harder cross-check once CV is clean.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
