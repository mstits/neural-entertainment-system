"""Tape loader coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.parity.tape import Tape, TapeError
from tests.skip_gates import requires

REPO = Path(__file__).resolve().parents[2]
ROM_REL = "roms/Mario Bros. (World).nes"
SAMPLE_ROM = REPO / ROM_REL

# Module scope, so the gate is reached: hashing the ROM at import time
# raises FileNotFoundError during COLLECTION on a clean clone, which no
# per-test decorator can catch. Read it inside the helper instead.
pytestmark = requires(ROM_REL)


def _sample_md5() -> str:
    return hashlib.md5(SAMPLE_ROM.read_bytes()).hexdigest()


def _valid_cross_tape(frames: int = 10, inputs: list[dict] | None = None) -> dict:
    return {
        "name": "test_cross",
        "rom": "roms/Mario Bros. (World).nes",
        "rom_md5": _sample_md5(),
        "frames": frames,
        "frame_skip": 1,
        "mode": "cross_emulator",
        "tolerance_scanlines_diverged": 0,
        "tolerance_pixels_per_scanline": 0,
        "inputs": inputs if inputs is not None else [],
        "notes": "test",
    }


@pytest.mark.parity
def test_loads_valid_cross_emulator_tape(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_valid_cross_tape()))
    t = Tape.load(p)
    assert t.name == "test_cross"
    assert t.frames == 10
    assert t.mode == "cross_emulator"
    assert t.golden_hashes is None


@pytest.mark.parity
def test_button_sequence_resolves_ranges(tmp_path):
    p = tmp_path / "t.json"
    tape_data = _valid_cross_tape(
        frames=10,
        inputs=[
            {"start_frame": 0, "end_frame": 10, "buttons": 0},
            {"start_frame": 3, "end_frame": 6, "buttons": 128},  # Right
        ],
    )
    p.write_text(json.dumps(tape_data))
    seq = Tape.load(p).button_sequence()
    assert seq.dtype == np.uint8
    assert seq.shape == (10,)
    assert seq[0] == 0 and seq[2] == 0 and seq[6] == 0 and seq[9] == 0
    assert seq[3] == 128 and seq[4] == 128 and seq[5] == 128


@pytest.mark.parity
def test_overlapping_ranges_last_write_wins(tmp_path):
    p = tmp_path / "t.json"
    tape_data = _valid_cross_tape(
        frames=10,
        inputs=[
            {"start_frame": 0, "end_frame": 10, "buttons": 128},  # Right
            {"start_frame": 5, "end_frame": 8, "buttons": 64},   # Left (overrides)
        ],
    )
    p.write_text(json.dumps(tape_data))
    seq = Tape.load(p).button_sequence()
    assert seq[4] == 128  # original
    assert seq[5] == 64 and seq[7] == 64  # overridden
    assert seq[8] == 128  # back to original


@pytest.mark.parity
def test_md5_mismatch_raises_with_both_hashes(tmp_path):
    p = tmp_path / "t.json"
    bad = _valid_cross_tape()
    bad["rom_md5"] = "0" * 32
    p.write_text(json.dumps(bad))
    with pytest.raises(TapeError) as excinfo:
        Tape.load(p)
    msg = str(excinfo.value)
    assert "0" * 32 in msg and _sample_md5() in msg


@pytest.mark.parity
def test_missing_field_raises_named_error(tmp_path):
    p = tmp_path / "t.json"
    incomplete = _valid_cross_tape()
    del incomplete["mode"]
    p.write_text(json.dumps(incomplete))
    with pytest.raises(TapeError) as excinfo:
        Tape.load(p)
    assert "mode" in str(excinfo.value)


@pytest.mark.parity
def test_malformed_json_raises(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{not json")
    with pytest.raises(TapeError):
        Tape.load(p)


@pytest.mark.parity
def test_invalid_mode_raises(tmp_path):
    p = tmp_path / "t.json"
    bad = _valid_cross_tape()
    bad["mode"] = "whatever"
    p.write_text(json.dumps(bad))
    with pytest.raises(TapeError) as excinfo:
        Tape.load(p)
    assert "whatever" in str(excinfo.value)


@pytest.mark.parity
def test_golden_hash_mode_requires_blob(tmp_path):
    p = tmp_path / "t.json"
    tape_data = _valid_cross_tape()
    tape_data["mode"] = "golden_hash"
    p.write_text(json.dumps(tape_data))
    with pytest.raises(TapeError) as excinfo:
        Tape.load(p)
    assert "golden_hashes" in str(excinfo.value)


@pytest.mark.parity
def test_golden_hash_mode_loads_blob(tmp_path):
    p = tmp_path / "t.json"
    blob = np.arange(10, dtype="<u8")
    blob_path = tmp_path / "t.golden.bin"
    blob.tofile(blob_path)
    tape_data = _valid_cross_tape()
    tape_data["mode"] = "golden_hash"
    tape_data["golden_hashes"] = "t.golden.bin"
    p.write_text(json.dumps(tape_data))
    t = Tape.load(p)
    assert t.mode == "golden_hash"
    assert t.golden_hashes is not None
    assert t.golden_hashes.shape == (10,)
    assert t.golden_hashes[5] == 5


@pytest.mark.parity
def test_golden_blob_length_mismatch_raises(tmp_path):
    p = tmp_path / "t.json"
    blob = np.arange(5, dtype="<u8")  # only 5 hashes
    blob_path = tmp_path / "t.golden.bin"
    blob.tofile(blob_path)
    tape_data = _valid_cross_tape(frames=10)  # but 10 frames requested
    tape_data["mode"] = "golden_hash"
    tape_data["golden_hashes"] = "t.golden.bin"
    p.write_text(json.dumps(tape_data))
    with pytest.raises(TapeError) as excinfo:
        Tape.load(p)
    msg = str(excinfo.value)
    assert "5" in msg and "10" in msg


@pytest.mark.parity
def test_cross_emulator_ignores_golden_hashes_field(tmp_path):
    p = tmp_path / "t.json"
    tape_data = _valid_cross_tape()
    tape_data["golden_hashes"] = "does_not_exist.bin"  # cross_emulator: ignored
    p.write_text(json.dumps(tape_data))
    t = Tape.load(p)
    assert t.golden_hashes is None


@pytest.mark.parity
def test_out_of_bounds_input_range_raises(tmp_path):
    p = tmp_path / "t.json"
    tape_data = _valid_cross_tape(
        frames=10,
        inputs=[{"start_frame": 0, "end_frame": 20, "buttons": 0}],
    )
    p.write_text(json.dumps(tape_data))
    t = Tape.load(p)
    with pytest.raises(TapeError):
        t.button_sequence()
