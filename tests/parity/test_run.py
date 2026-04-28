"""End-to-end runner coverage. Builds ephemeral tapes in tmp_path and
exercises both modes of run.py against real emulators.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.parity.diff import frame_hash
from tests.parity.drivers import NESCoreDriver
from tests.parity.run import run

REPO = Path(__file__).resolve().parents[2]
ROM_REL = "roms/Mario Bros. (World).nes"
ROM_ABS = REPO / ROM_REL
ROM_MD5 = hashlib.md5(ROM_ABS.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _write_tape(tmp_path: Path, *, mode: str, frames: int, inputs: list[dict],
                golden: np.ndarray | None = None, warmup: int = 10) -> Path:
    tape = {
        "name": f"test_{mode}",
        "rom": ROM_REL,
        "rom_md5": ROM_MD5,
        "frames": frames,
        "frame_skip": 1,
        "mode": mode,
        "warmup_frames": warmup,
        "tolerance_scanlines_diverged": 0,
        "tolerance_pixels_per_scanline": 0,
        "inputs": inputs,
        "notes": "ephemeral for test_run",
    }
    if mode == "golden_hash":
        blob_name = "test.golden.bin"
        tape["golden_hashes"] = blob_name
        assert golden is not None
        golden.tofile(tmp_path / blob_name)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(tape))
    return p


@pytest.mark.parity
def test_run_cross_emulator_passes_on_idle_baseline(tmp_path, capsys):
    # 30 frames no input = byte-exact (from T2 + T3 findings)
    p = _write_tape(tmp_path, mode="cross_emulator", frames=30,
                    inputs=[{"start_frame": 0, "end_frame": 30, "buttons": 0}])
    rc = run(p)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "cross_emulator" in out


@pytest.mark.parity
def test_run_cross_emulator_fails_when_tolerance_exceeded(tmp_path, capsys):
    # Force a failure by loosening to 0 then picking inputs that diverge.
    # 60 frames with Start press 10-20 → 6 pixels diverge per T3 finding.
    p = _write_tape(tmp_path, mode="cross_emulator", frames=60,
                    inputs=[
                        {"start_frame": 0, "end_frame": 60, "buttons": 0},
                        {"start_frame": 10, "end_frame": 20, "buttons": 8},
                    ])
    rc = run(p)
    assert rc == 1
    out = capsys.readouterr().out
    assert "scanlines diverged" in out and "cross_emulator" in out
    # Failure artifacts exist
    failures = list((Path(__file__).parent / "failures").glob("test_cross_emulator_*"))
    assert any(f.name.endswith("_diff.png") for f in failures)


@pytest.mark.parity
def test_run_golden_hash_passes_when_hashes_match(tmp_path, capsys):
    # Record golden hashes from the live nes_core, then run against them.
    # golden_hash mode doesn't need warmup — self-reference always converges.
    inputs = [{"start_frame": 0, "end_frame": 20, "buttons": 0}]
    d = NESCoreDriver(ROM_ABS)
    golden = np.zeros(20, dtype="<u8")
    for i in range(20):
        d.step(0)
        golden[i] = frame_hash(d.frame())

    p = _write_tape(tmp_path, mode="golden_hash", frames=20,
                    inputs=inputs, golden=golden, warmup=0)
    rc = run(p)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "golden_hash" in out


@pytest.mark.parity
def test_run_golden_hash_fails_on_hash_drift(tmp_path, capsys):
    # Corrupt the golden blob — any single bit flip must trip the runner.
    inputs = [{"start_frame": 0, "end_frame": 10, "buttons": 0}]
    d = NESCoreDriver(ROM_ABS)
    golden = np.zeros(10, dtype="<u8")
    for i in range(10):
        d.step(0)
        golden[i] = frame_hash(d.frame())
    golden[5] ^= np.uint64(1)  # poison frame 5

    p = _write_tape(tmp_path, mode="golden_hash", frames=10,
                    inputs=inputs, golden=golden, warmup=0)
    rc = run(p)
    assert rc == 1
    out = capsys.readouterr().out
    assert "hash mismatch" in out
