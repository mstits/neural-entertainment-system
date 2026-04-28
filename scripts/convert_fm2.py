"""
Convert one or more FCEUX `.fm2` movie files into the project's
`.state.bin` format (one byte per NES frame, controller-1 bitmask).

Usage:
    python scripts/convert_fm2.py input.fm2                  # → input.state.bin
    python scripts/convert_fm2.py a.fm2 b.fm2 -o combined.state.bin   # concat
    python scripts/convert_fm2.py a.fm2 b.fm2                # → a_b.state.bin
    python scripts/convert_fm2.py --stitch *.state.bin -o long.state.bin

FM2 format (TASVideos / FCEUX): plaintext. Header lines begin with
non-pipe characters; frame lines look like

    |<cmd>|<pad1 buttons>|<pad2 buttons>|<fds>|

where each pad-buttons field is 8 characters in RLDUTSBA order. `.` is
"not pressed", any other character is "pressed". We read controller-1
(pad1), drop the rest.

The produced byte order matches `src/emulation/frame_utils.py`:
    0x80=Right 0x40=Left 0x20=Down 0x10=Up 0x08=Start 0x04=Select 0x02=B 0x01=A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# FM2 pad-buttons column order, left-to-right.
_FM2_BIT_AT_POS = (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)  # R L D U T S B A


def fm2_to_bytes(fm2_path: Path) -> bytes:
    """Parse an FM2 file and return a byte sequence of controller-1
    bitmasks, one byte per NES frame."""
    out = bytearray()
    with open(fm2_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.startswith("|"):
                continue  # header / metadata line
            parts = line.split("|")
            # parts is ["", <cmd>, <pad1>, <pad2?>, <fds?>, ""]
            if len(parts) < 3:
                continue
            pad1 = parts[2]
            if len(pad1) < 8:
                # malformed row — pad with "not pressed" rather than bail
                pad1 = pad1.ljust(8, ".")
            mask = 0
            for ch, bit in zip(pad1[:8], _FM2_BIT_AT_POS):
                if ch != "." and ch != " ":
                    mask |= bit
            out.append(mask)
    return bytes(out)


def stitch(paths: list[Path]) -> bytes:
    """Concatenate one or more `.state.bin` files in order. Trivial —
    the format has no per-file header, so raw concatenation is correct.
    Useful for splicing e.g. a navigation-menu demo onto a gameplay demo
    so BC sees the full sequence."""
    buf = bytearray()
    for p in paths:
        buf.extend(p.read_bytes())
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".fm2 files (or .state.bin files with --stitch)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .state.bin path. Defaults to <first-input>.state.bin")
    ap.add_argument("--stitch", action="store_true",
                    help="Inputs are already .state.bin files; concatenate them")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    if args.stitch:
        data = stitch(paths)
    else:
        chunks = [fm2_to_bytes(p) for p in paths]
        data = b"".join(chunks)

    if args.output is None:
        first = paths[0]
        out = first.with_suffix("") .with_suffix(".state.bin")
    else:
        out = args.output

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"wrote {out} ({len(data)} frames, ~{len(data) / 60:.1f} s of NES time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
