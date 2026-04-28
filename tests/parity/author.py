"""Author a parity tape by running nes_core headless against a scripted
input sequence.

    python -m tests.parity.author \\
        --rom "roms/Contra (USA).nes" \\
        --out tests/parity/tapes/contra_title.json \\
        --mode golden_hash \\
        --from-file tests/parity/tapes/contra_title.script.txt \\
        --name contra_title \\
        --notes "horizontal scroll-split regression target"

Script file format (one action per line):
    <frame_count> <button_bitmask>    e.g. `120 0` = 120 frames, no input
    <button_bitmask>                  = one frame of that input
    # comment
    blank lines OK

The author always records via nes_core; for golden_hash mode it also
writes a sibling `<out>.golden.bin` of 8-byte BLAKE2b hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tests.parity.diff import frame_hash
from tests.parity.drivers import NESCoreDriver

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WARMUP = 10


def parse_script(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            count, mask = 1, int(parts[0], 0)
        elif len(parts) == 2:
            count, mask = int(parts[0], 0), int(parts[1], 0)
        else:
            raise ValueError(f"line {lineno}: expected '<count> <mask>' or '<mask>', got: {raw!r}")
        if count < 1:
            raise ValueError(f"line {lineno}: frame count must be >= 1, got {count}")
        if not 0 <= mask <= 255:
            raise ValueError(f"line {lineno}: mask must be 0..255, got {mask}")
        ranges.append((count, mask))
    return ranges


def ranges_to_tape_inputs(ranges: list[tuple[int, int]]) -> tuple[list[dict], int]:
    inputs: list[dict] = []
    frame = 0
    for count, mask in ranges:
        inputs.append({"start_frame": frame, "end_frame": frame + count, "buttons": mask})
        frame += count
    return inputs, frame


def record(rom: Path, ranges: list[tuple[int, int]], mode: str) -> np.ndarray | None:
    if mode != "golden_hash":
        return None
    driver = NESCoreDriver(rom)
    total_frames = sum(count for count, _ in ranges)
    hashes = np.empty(total_frames, dtype="<u8")
    i = 0
    for count, mask in ranges:
        for _ in range(count):
            driver.step(mask)
            hashes[i] = frame_hash(driver.frame())
            i += 1
    return hashes


def write_tape(
    out: Path,
    *,
    name: str,
    rom: Path,
    ranges: list[tuple[int, int]],
    mode: str,
    notes: str,
    warmup: int,
    tolerance_scanlines: int,
    tolerance_pixels: int,
) -> None:
    inputs, total = ranges_to_tape_inputs(ranges)
    rom_abs = (REPO / rom).resolve() if not rom.is_absolute() else rom
    rom_rel = rom_abs.relative_to(REPO).as_posix()
    tape_doc = {
        "name": name,
        "rom": rom_rel,
        "rom_md5": hashlib.md5(rom_abs.read_bytes()).hexdigest(),
        "frames": total,
        "frame_skip": 1,
        "mode": mode,
        "warmup_frames": warmup,
        "tolerance_scanlines_diverged": tolerance_scanlines,
        "tolerance_pixels_per_scanline": tolerance_pixels,
        "inputs": inputs,
        "notes": notes,
    }
    if mode == "golden_hash":
        blob_name = f"{out.stem}.golden.bin"
        tape_doc["golden_hashes"] = blob_name
        hashes = record(rom_abs, ranges, mode)
        assert hashes is not None
        hashes.tofile(out.parent / blob_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tape_doc, indent=2) + "\n")


def read_stdin_script() -> str:
    print("Enter script (one action per line, Ctrl-D to finish):", file=sys.stderr)
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Author a parity tape.")
    parser.add_argument("--rom", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["cross_emulator", "golden_hash"], required=True)
    parser.add_argument("--from-file", type=Path)
    parser.add_argument("--name", help="Tape name (default: out filename stem)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--tolerance-scanlines", type=int, default=0)
    parser.add_argument("--tolerance-pixels", type=int, default=0)
    args = parser.parse_args()

    script_text = args.from_file.read_text() if args.from_file else read_stdin_script()
    ranges = parse_script(script_text)
    if not ranges:
        print("ERROR: empty script", file=sys.stderr)
        return 1

    name = args.name or args.out.stem
    write_tape(
        args.out,
        name=name,
        rom=args.rom,
        ranges=ranges,
        mode=args.mode,
        notes=args.notes,
        warmup=args.warmup,
        tolerance_scanlines=args.tolerance_scanlines,
        tolerance_pixels=args.tolerance_pixels,
    )
    total_frames = sum(c for c, _ in ranges)
    print(f"wrote {args.out} ({total_frames} frames, mode={args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
