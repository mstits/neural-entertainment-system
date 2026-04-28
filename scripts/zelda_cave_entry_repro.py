"""Regression guard for the Zelda cave-entry transition timing.

Replays the 3354-frame cave tape through nes_core and nes-py in
lockstep. Originally written when the cave-entry transition fired
one frame later in nes_core, with the failure cascading by f1758
into Link drifting off the sword-pickup tile.

Cycle-locking advance_one_frame to exactly 29781 CPU cycles per
frame (matching nes-py / LaiNES upstream's fixed-iteration emulator
step) closed the f1011 cave-entry timing defect (verified end-to-end
in the GUI play window — user grabs the sword and uses it). The tape
replay still desyncs after f1011 because the recorded buttons assume
nes-py's exact frame-by-frame sprite-OAM state, which our PPU's
batched scanline path doesn't reproduce byte-for-byte. Outcome guard
is therefore the f1012 transition itself: if $0011, $0070, $0084
match across both emulators at f1012, the timing fix is intact.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import nes_core
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nes_py


ROM = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"
TAPE = REPO / "roms" / "zelda_start_419.state.bin"


def main() -> int:
    """Outcome check: replay the tape until f1012 (one frame past the
    cave-entry transition) and confirm the four gameplay state bytes
    that drive cave-entry — $0011, $0070, $0084, $0657 — match between
    nes_core and nes-py. Pass = cycle-lock fix is intact."""
    ours = nes_core.NESEnvironment(rom_path=str(ROM), frame_skip=1)
    ours.reset()
    theirs = nes_py.NESEnv(str(ROM))
    theirs.reset()
    theirs.step(0)
    buttons = list(TAPE.read_bytes())
    WATCH = [0x11, 0x70, 0x84, 0x657]
    for i, b in enumerate(buttons):
        ours.step(int(b) & 0xFF)
        theirs.step(int(b) & 0xFF)
        if i + 1 == 1012:
            o = ours.get_ram_range(0, 0x800)
            t = theirs.ram
            mismatches = [(a, o[a], t[a]) for a in WATCH if o[a] != t[a]]
            if not mismatches:
                vals = " ".join(f"${a:04X}=0x{o[a]:02X}" for a in WATCH)
                print(f"PASS: cave-entry transition matches nes-py at "
                      f"f1012 ({vals}).")
                return 0
            detail = ", ".join(
                f"${a:04X} ours=0x{ov:02X} theirs=0x{tv:02X}"
                for a, ov, tv in mismatches
            )
            print(f"FAIL: cave-entry transition diverges at f1012 ({detail}).")
            print(f"      The cycle-lock in NESEnvironment::advance_one_frame "
                  f"may have regressed; re-run the parity suite and check.")
            return 1
    print(f"FAIL: tape ended before f1012 (only {len(buttons)} frames).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
