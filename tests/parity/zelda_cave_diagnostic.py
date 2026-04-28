"""Replay zelda_start_419 (lands Link in cave with old man dialog),
then drive the sword pickup + exit. Watch for IRQ trap entry, RAM
freeze (game stuck), and successful sword pickup ($0657 -> 1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nes_core


A, B, SELECT, START = 0x01, 0x02, 0x04, 0x08
UP, DOWN, LEFT, RIGHT = 0x10, 0x20, 0x40, 0x80


def show(env, label):
    ram = env.get_ram_range(0, 0x800)
    pc = env.cpu_state()[0]
    in_trap = 0xFFE0 <= pc <= 0xFFFF
    print(
        f"{label:30s} PC=0x{pc:04X}{'(TRAP)' if in_trap else '       '} "
        f"$11=0x{ram[0x11]:02X} $EB=0x{ram[0xEB]:02X} $70=0x{ram[0x70]:02X} "
        f"$657=0x{ram[0x657]:02X}",
        flush=True,
    )


def step_n(env, n, mask=0):
    pc_history = []
    ram_history = []
    for _ in range(n):
        env.step(mask)
        pc_history.append(env.cpu_state()[0])
        ram_history.append(env.get_ram(0x70))
    return pc_history, ram_history


def main():
    env = nes_core.NESEnvironment(rom_path="roms/zelda.nes", frame_skip=1)
    env.reset()

    # Phase 1: replay the tape to land in cave
    buttons = list(open("roms/zelda_start_419.state.bin", "rb").read())
    for b in buttons:
        env.step(int(b) & 0xFF)
    show(env, "after replay (in cave)")

    # Phase 2: wait through "IT'S DANGEROUS TO GO ALONE" dialog scroll
    # Zelda text scrolls one char per ~4 frames. The message is
    # ~30 chars = ~120 frames. Then Link can press A to dismiss,
    # then walk up to old man + sword.
    pc_h, ram_h = step_n(env, 240)
    trap_entries = sum(1 for pc in pc_h if 0xFFE0 <= pc <= 0xFFFF)
    print(f"  during 240f wait: trap_entries={trap_entries}/240")
    show(env, "after dialog idle 240f")

    # Phase 3: press A repeatedly to dismiss dialog (if any)
    for _ in range(20):
        for _ in range(3): env.step(A)
        for _ in range(3): env.step(0)
    show(env, "after A-mash 120f")

    # Phase 4: walk UP toward the old man + sword
    pc_h, ram_h = step_n(env, 60, UP)
    show(env, "after UP 60f")
    pc_h, ram_h = step_n(env, 60, UP)
    show(env, "after UP 120f")
    pc_h, ram_h = step_n(env, 30, UP|A)
    show(env, "after UP+A 30f (grab?)")
    pc_h, ram_h = step_n(env, 60, 0)
    show(env, "idle 60f")
    pc_h, ram_h = step_n(env, 60, UP)
    show(env, "after UP 60f again")

    # Phase 5: walk DOWN to exit
    pc_h, ram_h = step_n(env, 120, DOWN)
    show(env, "after DOWN 120f (exit?)")
    pc_h, ram_h = step_n(env, 60, 0)
    show(env, "idle 60f")

    # Save final state for inspection
    blob = bytes(env.save_state())
    with open("/tmp/zelda_cave_attempt.bin", "wb") as f:
        f.write(blob)
    from PIL import Image
    Image.fromarray(env.get_frame(),'RGB').save('/tmp/zelda_cave_attempt.png')
    print("artifacts: /tmp/zelda_cave_attempt.{bin,png}")


if __name__ == "__main__":
    main()
