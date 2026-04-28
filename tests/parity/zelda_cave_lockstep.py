"""Lockstep cold-boot zelda.nes vs Rev A through nes-py during cave
replay. Find first frame where RAM diverges. nes-py does support MMC1
+ this Zelda Rev A is the same PRG/CHR as our zelda.nes. Differences
will name the diverging code path."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nes_core
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nes_py


# Use Rev A on both since nes-py needs the iNES 1.0 header
ROM = "roms/Legend of Zelda, The (USA) (Rev A).nes"
TAPE = "roms/zelda_start_419.state.bin"


def main():
    ours = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
    ours.reset()
    theirs = nes_py.NESEnv(ROM)
    theirs.reset()
    theirs.step(0)  # phase compensation per existing lockstep harness

    buttons = list(open(TAPE, "rb").read())
    print(f"running {len(buttons)} frame replay, watching for first divergence")

    # Find the EXACT frame where diff explodes from <10 to >50 bytes
    explosion_frame = None
    for i, b in enumerate(buttons):
        ours.step(int(b) & 0xFF)
        theirs.step(int(b) & 0xFF)
        ours_ram = bytes(ours.get_ram_range(0, 0x800))
        theirs_ram = bytes(theirs.ram)
        n_diff = sum(1 for a in range(0x800) if ours_ram[a] != theirs_ram[a])
        if i % 50 == 0 or n_diff >= 50 and explosion_frame is None:
            print(f"  frame {i:>5d}: n_diff={n_diff}", flush=True)
        if n_diff >= 50 and explosion_frame is None:
            explosion_frame = i
            diffs = [a for a in range(0x800) if ours_ram[a] != theirs_ram[a]]
            print(f"\n  EXPLOSION at frame {i}: {n_diff} bytes diff", flush=True)
            for a in diffs[:30]:
                print(f"    ${a:04X}: ours=0x{ours_ram[a]:02X}  theirs=0x{theirs_ram[a]:02X}")
            break


if __name__ == "__main__":
    main()
