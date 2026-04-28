"""Dump a specific frame from a tape's input sequence for visual
inspection. Writes ours.png (always), theirs.png + both.png (when
cross-emulator comparison is possible).

    python -m tests.parity.inspect --tape <path> --frame <n> [--out-dir <dir>]

If --frame is omitted, defaults to the tape's last frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tests.parity.drivers import NESCoreDriver, NESPyDriver
from tests.parity.tape import Tape, TapeError

DEFAULT_OUT = Path(__file__).parent / "failures"


def inspect(tape_path: Path, frame: int | None, out_dir: Path) -> int:
    try:
        tape = Tape.load(tape_path)
    except TapeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    target_frame = tape.frames - 1 if frame is None else frame
    if not 0 <= target_frame < tape.frames:
        print(f"ERROR: --frame {target_frame} out of range [0, {tape.frames})", file=sys.stderr)
        return 1

    buttons = tape.button_sequence()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{tape.name}_inspect_{target_frame:04d}"

    ours_driver = NESCoreDriver(tape.rom)
    theirs_driver = NESPyDriver(tape.rom)
    for i in range(target_frame + 1):
        ours_driver.step(int(buttons[i]))
        theirs_driver.step(int(buttons[i]))

    ours = ours_driver.frame()
    theirs = theirs_driver.frame()
    Image.fromarray(ours).save(out_dir / f"{prefix}_ours.png")
    Image.fromarray(theirs).save(out_dir / f"{prefix}_theirs.png")

    side_by_side = np.concatenate([ours, theirs], axis=1)
    Image.fromarray(side_by_side).save(out_dir / f"{prefix}_both.png")

    print(f"wrote {out_dir}/{prefix}_{{ours,theirs,both}}.png")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump a frame from a tape for inspection.")
    parser.add_argument("--tape", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=None, help="Frame index; defaults to last")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    return inspect(args.tape, args.frame, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
