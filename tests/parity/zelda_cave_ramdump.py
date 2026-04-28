"""Land Link in cave, then dump ALL of zero page + key game-state areas
to find what's frozen and what's changing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nes_core


def dump_zp(env, label):
    ram = bytes(env.get_ram_range(0, 0x100))
    print(f"\n=== {label} ===", flush=True)
    for r in range(0, 0x100, 16):
        print(f"  ${r:04X}: " + " ".join(f"{ram[r+i]:02X}" for i in range(16)),
              flush=True)
    return ram


def main():
    env = nes_core.NESEnvironment(rom_path="roms/zelda.nes", frame_skip=1)
    env.reset()
    buttons = list(open("roms/zelda_start_419.state.bin", "rb").read())
    for b in buttons:
        env.step(int(b) & 0xFF)

    a = dump_zp(env, "in cave (replay just finished)")
    # Idle 30 frames
    for _ in range(30): env.step(0)
    b = dump_zp(env, "after idle 30")
    # Step 60 more with UP held
    for _ in range(60): env.step(0x10)
    c = dump_zp(env, "after UP 60")
    # Diff a vs c
    print("\n=== bytes that changed (a -> c, idle then UP) ===", flush=True)
    diffs = [(i, a[i], c[i]) for i in range(0x100) if a[i] != c[i]]
    for i, x, y in diffs:
        print(f"  ${i:04X}: {x:02X} -> {y:02X}")
    print(f"\n  total: {len(diffs)} bytes changed in zero page", flush=True)


if __name__ == "__main__":
    main()
