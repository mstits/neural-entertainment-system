"""North Star test: does Zelda actually play through the cave
correctly in nes_core, without any comparison to nes-py?

Drive cold-boot Rev A through the existing 3354-frame replay (lands
Link in cave with old man dialog), then append scripted input:
- Wait long enough for dialog to scroll
- Try to advance dialog with A
- Walk UP to the sword
- Check if $0657 (sword inventory) flips from 0 to 1
- Walk DOWN to exit cave
- Check if room changes (screen transition)

If sword pickup works AND cave exit works, the GUI cave-stuck bug
is GUI/input-specific. If either fails here, we have a deterministic
headless reproduction to iterate on.

Also save PNGs at each milestone for visual verification.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nes_core
from PIL import Image


A, B, SELECT, START = 0x01, 0x02, 0x04, 0x08
UP, DOWN, LEFT, RIGHT = 0x10, 0x20, 0x40, 0x80


def snap(env, path: str, label: str):
    ram = env.get_ram_range(0, 0x800)
    pc = env.cpu_state()[0]
    print(
        f"[{label:30s}] PC=0x{pc:04X} "
        f"$10(mode)=0x{ram[0x10]:02X} "
        f"$11(sub)=0x{ram[0x11]:02X} "
        f"$26=0x{ram[0x26]:02X} "
        f"$657(sword)=0x{ram[0x657]:02X} "
        f"$EB(Xtile)=0x{ram[0xEB]:02X} $70=0x{ram[0x70]:02X}",
        flush=True,
    )
    Image.fromarray(env.get_frame(), "RGB").save(path)


def drive(env, mask: int, frames: int):
    for _ in range(frames):
        env.step(mask)


def main() -> int:
    rom = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"
    tape = REPO / "roms" / "zelda_start_419.state.bin"
    env = nes_core.NESEnvironment(rom_path=str(rom), frame_skip=1)
    env.reset()

    # Replay tape (lands Link in cave with old man textbox active)
    buttons = list(tape.read_bytes())
    for b in buttons:
        env.step(int(b) & 0xFF)
    snap(env, "/tmp/zelda_ns_01_after_tape.png", "after tape (in cave?)")

    # Wait for dialog to scroll/dismiss (Zelda cave text is ~4 sec)
    drive(env, 0, 240)
    snap(env, "/tmp/zelda_ns_02_after_wait240.png", "after idle 240f")

    # Mash A to advance dialog (Zelda does auto-advance usually)
    for _ in range(30):
        drive(env, A, 2)
        drive(env, 0, 2)
    snap(env, "/tmp/zelda_ns_03_after_Amash.png", "after A-mash 120f")

    # Idle more (dialog might still be scrolling)
    drive(env, 0, 180)
    snap(env, "/tmp/zelda_ns_04_after_wait180.png", "after wait 180 more")

    # Walk UP toward the sword (old man above Link in cave)
    drive(env, UP, 90)
    snap(env, "/tmp/zelda_ns_05_after_UP90.png", "after UP 90f")

    # Walk UP+A in case sword needs active pickup
    drive(env, UP | A, 30)
    snap(env, "/tmp/zelda_ns_06_after_UPA30.png", "after UP+A 30f")

    # More up
    drive(env, UP, 60)
    snap(env, "/tmp/zelda_ns_07_after_UP60more.png", "after UP 60 more")

    # Idle to let sword pickup animation play
    drive(env, 0, 120)
    snap(env, "/tmp/zelda_ns_08_after_idle120.png", "after idle 120")

    # Walk DOWN to exit cave (south door)
    drive(env, DOWN, 120)
    snap(env, "/tmp/zelda_ns_09_after_DOWN120.png", "after DOWN 120")

    drive(env, DOWN, 120)
    snap(env, "/tmp/zelda_ns_10_after_DOWN_more.png", "after DOWN 240 total")

    # Idle to let cave-exit transition complete
    drive(env, 0, 120)
    snap(env, "/tmp/zelda_ns_11_final.png", "final state")

    # Final check
    ram = env.get_ram_range(0, 0x800)
    sword = ram[0x657]
    print(f"\n=== VERDICT ===")
    print(f"sword inventory $0657 = 0x{sword:02X} ({'picked up' if sword > 0 else 'NOT picked up'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
