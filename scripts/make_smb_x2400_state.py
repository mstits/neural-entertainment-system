"""Generate `samples/mario_x2400.state.bin` — an NCST save state of
Super Mario Bros. with Mario standing right before the staircase
obstacle at x≈2400.

Used as a curriculum start-state so the trainer can drop episodes
directly into the staircase region instead of requiring the agent to
walk 2400 pixels every episode just to practice the wall.

The script replays the bundled BC demo recording one NES frame at a
time and snapshots the env state when Mario's world X crosses the
target threshold. The recording goes well past x=2400, so any
sufficiently-clean play will hit the threshold within a few seconds
of game time. Output is the NCST\\x01 format produced by
`env.save_state()` — drop-in compatible with the trainer's
`start_state_path` and the GUI's "Choose State…" picker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nes_core


ROM = "roms/Super Mario Bros. (World).nes"
DEMO = "roms/Super Mario Bros. (World).bc_demo.state.bin"
OUT = "samples/mario_x2400.state.bin"

# RAM addresses for Mario's world X.
RAM_X_PAGE = 0x006D
RAM_X_LOW = 0x0086


def world_x(env: nes_core.NESEnvironment) -> int:
    page = int(env.get_ram_range(RAM_X_PAGE, RAM_X_PAGE + 1)[0])
    low = int(env.get_ram_range(RAM_X_LOW, RAM_X_LOW + 1)[0])
    return (page << 8) | low


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-x", type=int, default=2400)
    parser.add_argument("--rom", default=ROM)
    parser.add_argument("--demo", default=DEMO)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max-frames", type=int, default=10_000)
    args = parser.parse_args()

    rom_path = Path(args.rom)
    demo_path = Path(args.demo)
    out_path = Path(args.out)

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not demo_path.exists():
        print(f"BC demo not found: {demo_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    demo_bytes = demo_path.read_bytes()
    print(f"Loaded demo: {len(demo_bytes)} frames")

    env = nes_core.NESEnvironment(rom_path=str(rom_path), frame_skip=1)
    env.reset()

    target = args.target_x
    last_logged_x = -1
    # Pre-init in case demo_bytes is empty: the for-else trailer below
    # references both `frame_idx` and `x`, which would NameError if the
    # loop never executed.
    frame_idx = -1
    x = world_x(env)
    for frame_idx, raw in enumerate(demo_bytes):
        if frame_idx >= args.max_frames:
            break
        env.step(int(raw))
        x = world_x(env)
        # Periodic progress log so the user sees motion.
        if x >= last_logged_x + 200:
            print(f"  frame {frame_idx:>5}  world_x={x}")
            last_logged_x = x
        if x >= target:
            state = env.save_state()
            out_path.write_bytes(bytes(state))
            print(
                f"\nReached x={x} at frame {frame_idx}. "
                f"Wrote {len(state)} bytes to {out_path}."
            )
            return 0

    print(f"Demo exhausted after {frame_idx + 1} frames; "
          f"target x={target} not reached. Last x={x}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
