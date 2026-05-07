"""
Convert NES movie recordings (FCEUX `.fm2` / `.fm3` or BizHawk
`.bk2`) into the project's `.state.bin` format (one byte per NES
frame, controller-1 bitmask).

Usage:
    python scripts/convert_fm2.py input.fm2                  # → input.state.bin
    python scripts/convert_fm2.py input.fm3                  # → input.state.bin
    python scripts/convert_fm2.py input.bk2                  # → input.state.bin
    python scripts/convert_fm2.py a.fm2 b.bk2 -o combined.state.bin   # concat (mixed types ok)
    python scripts/convert_fm2.py --stitch *.state.bin -o long.state.bin

Format details:

    FM2 / FM3 (TASVideos, FCEUX):
        Plaintext. Frame lines: |<cmd>|<pad1>|<pad2>|<fds>|
        pad1 column order: R L D U T(Start) S(Select) B A
        FM3 adds optional savestate / subtitle sections that don't
        start with `|`, so they're naturally filtered out.

    BK2 (BizHawk):
        ZIP container with `Input Log.txt` inside. Frame lines:
            |<system controls>|<P1 buttons>|<P2 buttons>|
        P1 column order: U D L R S(Start) s(select) B A — DIFFERENT
        from FM2. The system-controls column has 10 chars
        (Reset/Power/FDS/VS-coin/etc) — these aren't NES buttons so
        we ignore that column and read only P1.

The produced byte order matches `src/emulation/frame_utils.py`:
    0x80=Right 0x40=Left 0x20=Down 0x10=Up 0x08=Start 0x04=Select 0x02=B 0x01=A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# FM2 pad-buttons column order, left-to-right.
_FM2_BIT_AT_POS = (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)  # R L D U T S B A

# BK2 P1-pad column order, left-to-right. Same bitmask values as FM2
# (we always emit the project's RLDUTSBA bit layout) but the source
# columns are arranged differently in BizHawk's Input Log:
#   col 0 = Up    -> bit 0x10
#   col 1 = Down  -> bit 0x20
#   col 2 = Left  -> bit 0x40
#   col 3 = Right -> bit 0x80
#   col 4 = Start -> bit 0x08
#   col 5 = Select-> bit 0x04
#   col 6 = B     -> bit 0x02
#   col 7 = A     -> bit 0x01
_BK2_BIT_AT_POS = (0x10, 0x20, 0x40, 0x80, 0x08, 0x04, 0x02, 0x01)


def fceux_movie_to_bytes(movie_path: Path) -> bytes:
    """Parse an FCEUX `.fm2` or `.fm3` movie file and return a byte
    sequence of controller-1 bitmasks, one byte per NES frame.

    FM2 and FM3 share the same input-frame line format; the only
    difference is FM3's optional auxiliary sections (savestates,
    subtitles) which our `startswith("|")` filter naturally skips.
    """
    out = bytearray()
    with open(movie_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.startswith("|"):
                continue  # header / metadata / savestate / subtitle line
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


# Backward-compat alias. Existing callers (e.g. GUI _convert_fm2)
# imported `fm2_to_bytes`; keep it working unchanged.
fm2_to_bytes = fceux_movie_to_bytes


def bk2_to_bytes(bk2_path: Path) -> bytes:
    """Parse a BizHawk `.bk2` movie file and return a byte sequence of
    controller-1 bitmasks, one byte per NES frame.

    BK2 is a ZIP archive containing several files; the input sequence
    lives in `Input Log.txt` with a per-frame line format:

        |<system controls>|<P1 buttons>|<P2 buttons>|

    System controls are 10 chars (Reset, Power, FDS controls, VS coin
    inputs, VS service switch) — we ignore them. P1 buttons are 8 chars
    in `UDLRSsBA` order, which is DIFFERENT from FM2's RLDUTSBA — see
    `_BK2_BIT_AT_POS` for the mapping.

    The Input Log section is bracketed by `[Input]` / `[/Input]`
    markers. A `LogKey:` line precedes the frames declaring column
    layout; we skip it (and rely on the documented NES layout instead
    of parsing the LogKey, since variations are rare for NES).
    """
    import zipfile
    out = bytearray()
    with zipfile.ZipFile(str(bk2_path)) as zf:
        # `Input Log.txt` is the canonical name; case-insensitive search
        # in case different BizHawk versions vary capitalization.
        log_name = None
        for name in zf.namelist():
            if name.lower() == "input log.txt":
                log_name = name
                break
        if log_name is None:
            raise ValueError(
                f"{bk2_path.name}: BK2 archive contains no 'Input Log.txt' "
                f"(found: {zf.namelist()})"
            )
        with zf.open(log_name) as f:
            content = f.read().decode("utf-8", errors="replace")
    in_section = False
    for raw in content.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("[Input]"):
            in_section = True
            continue
        if line.startswith("[/Input]"):
            break
        if not in_section:
            # Older BK2s sometimes omit the [Input] header — fall back
            # to "any line starting with |" once we've passed the
            # LogKey declaration. Detect via the LogKey prefix.
            if line.startswith("LogKey:"):
                in_section = True
            continue
        if line.startswith("LogKey:"):
            continue
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        # parts is ["", <system>, <P1>, <P2?>, ""]. Skip system column;
        # read P1 from index 2.
        if len(parts) < 3:
            continue
        pad1 = parts[2]
        if len(pad1) < 8:
            pad1 = pad1.ljust(8, ".")
        mask = 0
        for ch, bit in zip(pad1[:8], _BK2_BIT_AT_POS):
            if ch != "." and ch != " ":
                mask |= bit
        out.append(mask)
    return bytes(out)


def movie_to_bytes(path: Path) -> bytes:
    """Format-agnostic dispatcher: pick the right parser by extension.

    Falls back to FCEUX text-format parsing on unknown extensions —
    most plaintext .nesmov / .fmv variants share the same FM2 frame
    line syntax, so it will usually work. Raises ValueError on a BK2
    archive that's missing its input log.
    """
    suffix = path.suffix.lower()
    if suffix == ".bk2":
        return bk2_to_bytes(path)
    return fceux_movie_to_bytes(path)


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
    ap.add_argument("inputs", nargs="+", help=".fm2 / .fm3 / .bk2 files (or .state.bin files with --stitch)")
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
        chunks = [movie_to_bytes(p) for p in paths]
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
