"""V2 of North Star test — with CORRECT Link position addresses.
$0084 = Link Y, $008D may be Link X (need to verify).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import nes_core
from PIL import Image


A, UP, DOWN, LEFT, RIGHT = 0x01, 0x10, 0x20, 0x40, 0x80


def snap(env, path: str, label: str):
    ram = env.get_ram_range(0, 0x800)
    pc = env.cpu_state()[0]
    print(
        f"[{label:28s}] PC=0x{pc:04X} "
        f"$84(LinkY)=0x{ram[0x84]:02X} $8D=0x{ram[0x8D]:02X} "
        f"$26=0x{ram[0x26]:02X} $657(sword)=0x{ram[0x657]:02X} "
        f"$11=0x{ram[0x11]:02X}",
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
    buttons = list(tape.read_bytes())
    for b in buttons:
        env.step(int(b) & 0xFF)
    snap(env, "/tmp/zcv2_01_in_cave.png", "in cave, post-tape")

    # Wait for dialog to tick out
    drive(env, 0, 60)
    snap(env, "/tmp/zcv2_02_post_wait.png", "after wait 60")

    # Walk UP a lot toward the sword
    drive(env, UP, 120)
    snap(env, "/tmp/zcv2_03_after_UP120.png", "after UP 120")

    # Pause, Link might have reached sword, give time to pickup animation
    drive(env, 0, 60)
    snap(env, "/tmp/zcv2_04_after_idle60.png", "after idle 60")

    # Walk UP even more in case we're not close enough
    drive(env, UP, 60)
    snap(env, "/tmp/zcv2_05_after_UPmore.png", "UP more")

    drive(env, 0, 120)
    snap(env, "/tmp/zcv2_06_after_idle120.png", "idle 120 after UP")

    # Try walk DOWN to exit
    drive(env, DOWN, 180)
    snap(env, "/tmp/zcv2_07_after_DOWN180.png", "after DOWN 180")

    drive(env, 0, 120)
    snap(env, "/tmp/zcv2_08_final.png", "final")

    ram = env.get_ram_range(0, 0x800)
    print(f"\n=== VERDICT ===")
    print(f"$0657 (sword) = 0x{ram[0x657]:02X}")
    print(f"$0084 (Link Y) = 0x{ram[0x84]:02X}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
