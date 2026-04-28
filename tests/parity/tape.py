"""Tape: frozen (ROM, inputs, frames, mode, golden-hashes) bundle.

JSON on disk, dataclass in memory. ROM MD5 verified on load. Golden
hashes for `golden_hash` mode live in a sibling binary file
(`<tape>.golden.bin`): one little-endian uint64 per frame.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Mode = Literal["cross_emulator", "golden_hash"]

REPO = Path(__file__).resolve().parents[2]


class TapeError(ValueError):
    pass


@dataclass
class Tape:
    name: str
    rom: Path
    rom_md5: str
    frames: int
    frame_skip: int
    mode: Mode
    tolerance_scanlines_diverged: int
    tolerance_pixels_per_scanline: int
    inputs: list[dict]
    notes: str
    warmup_frames: int = 0
    golden_hashes: np.ndarray | None = field(default=None, repr=False)
    source_path: Path = field(default=Path("."), repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "Tape":
        path = Path(path)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise TapeError(f"{path}: invalid JSON ({e.msg})") from e

        for field_name in (
            "name", "rom", "rom_md5", "frames", "frame_skip",
            "mode", "tolerance_scanlines_diverged",
            "tolerance_pixels_per_scanline", "inputs", "notes",
        ):
            if field_name not in raw:
                raise TapeError(f"{path}: missing field '{field_name}'")

        mode = raw["mode"]
        if mode not in ("cross_emulator", "golden_hash"):
            raise TapeError(f"{path}: mode must be 'cross_emulator' or 'golden_hash', got {mode!r}")

        rom_path = REPO / raw["rom"]
        if not rom_path.exists():
            raise TapeError(f"{path}: ROM not found: {rom_path}")
        actual_md5 = hashlib.md5(rom_path.read_bytes()).hexdigest()
        if actual_md5 != raw["rom_md5"]:
            raise TapeError(
                f"{path}: ROM md5 mismatch for {rom_path.name}. "
                f"expected={raw['rom_md5']} actual={actual_md5}"
            )

        golden = None
        if mode == "golden_hash":
            golden_ref = raw.get("golden_hashes")
            if not golden_ref:
                raise TapeError(f"{path}: mode=golden_hash requires 'golden_hashes' field")
            golden_path = path.parent / golden_ref
            if not golden_path.exists():
                raise TapeError(f"{path}: golden hash blob not found: {golden_path}")
            blob = np.fromfile(golden_path, dtype="<u8")
            if len(blob) != raw["frames"]:
                raise TapeError(
                    f"{path}: golden blob length {len(blob)} != frames {raw['frames']}"
                )
            golden = blob

        return cls(
            name=raw["name"],
            rom=rom_path,
            rom_md5=raw["rom_md5"],
            frames=int(raw["frames"]),
            frame_skip=int(raw["frame_skip"]),
            mode=mode,
            tolerance_scanlines_diverged=int(raw["tolerance_scanlines_diverged"]),
            tolerance_pixels_per_scanline=int(raw["tolerance_pixels_per_scanline"]),
            inputs=list(raw["inputs"]),
            notes=str(raw["notes"]),
            warmup_frames=int(raw.get("warmup_frames", 0)),
            golden_hashes=golden,
            source_path=path,
        )

    def button_sequence(self) -> np.ndarray:
        """Resolve input ranges to a (frames,) uint8 bitmask array.

        Half-open [start, end). Overlapping ranges: last-write-wins.
        """
        seq = np.zeros(self.frames, dtype=np.uint8)
        for i, rng in enumerate(self.inputs):
            for key in ("start_frame", "end_frame", "buttons"):
                if key not in rng:
                    raise TapeError(f"{self.source_path}: inputs[{i}] missing '{key}'")
            start = int(rng["start_frame"])
            end = int(rng["end_frame"])
            btn = int(rng["buttons"]) & 0xFF
            if not 0 <= start <= end <= self.frames:
                raise TapeError(
                    f"{self.source_path}: inputs[{i}] range [{start},{end}) "
                    f"out of bounds for frames={self.frames}"
                )
            seq[start:end] = btn
        return seq
