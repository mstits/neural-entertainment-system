"""Run every committed tape. Each tape's pass/fail is a separate test.

Also re-runs each tape twice and compares the nes_core frame-hash
trajectory between runs — catches non-determinism in nes_core that
would make any golden_hash tape flaky.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.parity.diff import frame_hash
from tests.parity.drivers import NESCoreDriver
from tests.parity.run import run
from tests.parity.tape import Tape

TAPES_DIR = Path(__file__).parent / "tapes"
TAPE_PATHS = sorted(p for p in TAPES_DIR.glob("*.json"))


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
@pytest.mark.parametrize("tape_path", TAPE_PATHS, ids=lambda p: p.stem)
def test_tape_passes(tape_path: Path):
    rc = run(tape_path)
    assert rc == 0, f"{tape_path.name} did not pass — see failures/"


@pytest.mark.parity
@pytest.mark.parametrize("tape_path", TAPE_PATHS, ids=lambda p: p.stem)
def test_nes_core_is_deterministic_across_runs(tape_path: Path):
    tape = Tape.load(tape_path)
    buttons = tape.button_sequence()

    def run_and_hash() -> np.ndarray:
        d = NESCoreDriver(tape.rom)
        hashes = np.empty(tape.frames, dtype="<u8")
        for i, btn in enumerate(buttons):
            d.step(int(btn))
            hashes[i] = frame_hash(d.frame())
        return hashes

    h1 = run_and_hash()
    h2 = run_and_hash()
    assert np.array_equal(h1, h2), (
        f"nes_core non-deterministic on {tape.name}: "
        f"frames differ at indices {np.where(h1 != h2)[0].tolist()[:5]}"
    )
