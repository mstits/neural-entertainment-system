"""Replay zelda_start_419 BUT mask out all Start presses past frame 200
(after the title→file-select Start tap). If the cave-stuck bug
disappears, it's a pause-menu/sub-screen transition bug, not a
cave-specific one."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import nes_core
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nes_py


ROM = "roms/Legend of Zelda, The (USA) (Rev A).nes"
TAPE = "roms/zelda_start_419.state.bin"


def main():
    ours = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
    ours.reset()
    theirs = nes_py.NESEnv(ROM)
    theirs.reset()
    theirs.step(0)

    buttons = list(open(TAPE, "rb").read())
    explosion_frame = None
    for i, b in enumerate(buttons):
        # Mask out Start (0x08) past frame 200 — keep title-screen advance
        m = b
        if i > 200 and (m & 0x08):
            m &= ~0x08
        ours.step(int(m) & 0xFF)
        theirs.step(int(m) & 0xFF)
        ours_ram = bytes(ours.get_ram_range(0, 0x800))
        theirs_ram = bytes(theirs.ram)
        n_diff = sum(1 for a in range(0x800) if ours_ram[a] != theirs_ram[a])
        if i % 100 == 0:
            print(f"  frame {i:>5d}: n_diff={n_diff}", flush=True)
        if n_diff >= 50 and explosion_frame is None:
            explosion_frame = i
            print(f"\n  EXPLOSION at frame {i}: {n_diff} bytes diff (action=0x{m:02X})", flush=True)
            break

    if explosion_frame is None:
        print(f"\n  No explosion in {len(buttons)} frames with Start masked past f=200")
        ours_ram = bytes(ours.get_ram_range(0, 0x800))
        theirs_ram = bytes(theirs.ram)
        n_diff = sum(1 for a in range(0x800) if ours_ram[a] != theirs_ram[a])
        print(f"  final n_diff={n_diff}")


if __name__ == "__main__":
    main()
