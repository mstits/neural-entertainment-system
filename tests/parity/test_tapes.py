"""Run every committed tape. Each tape's pass/fail is a separate test.

Also re-runs each tape twice and compares the nes_core frame-hash
trajectory between runs — catches non-determinism in nes_core that
would make any golden_hash tape flaky.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.parity.diff import frame_hash
from tests.parity.drivers import NESCoreDriver
from tests.parity.run import run
from tests.parity.tape import Tape
from tests.skip_gates import requires, requires_module

TAPES_DIR = Path(__file__).parent / "tapes"
TAPE_PATHS = sorted(p for p in TAPES_DIR.glob("*.json"))


def _tape_params(need_reference_emulator: bool) -> list:
    """One param per tape, gated on the ROM that tape's JSON names.

    The tapes are committed; the dumps they replay are not (roms/* is
    gitignored, .gitignore:71), so on a clean clone every one of these
    fails with `assert 1 == 0` and a pointer to a `failures/` directory
    that does not exist. Gate each tape on its OWN rom field, so a
    machine holding twelve of the seventeen dumps still runs those
    twelve.

    `need_reference_emulator` additionally gates cross_emulator tapes on
    nes-py, which is quarantined out of requirements.txt: only the tests
    that actually drive the reference emulator take that gate.
    """
    params = []
    for path in TAPE_PATHS:
        raw = json.loads(path.read_text())
        marks = [requires(raw["rom"])]
        if need_reference_emulator and raw["mode"] == "cross_emulator":
            marks.append(requires_module("nes_py"))
        params.append(pytest.param(path, marks=marks, id=path.stem))
    return params


# run() drives nes-py on a cross_emulator tape; the determinism test
# below only ever constructs NESCoreDriver, so it does not take that gate.
TAPE_PARAMS_RUN = _tape_params(need_reference_emulator=True)
TAPE_PARAMS_CORE = _tape_params(need_reference_emulator=False)


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
@pytest.mark.parametrize("tape_path", TAPE_PARAMS_RUN)
def test_tape_passes(tape_path: Path):
    rc = run(tape_path)
    assert rc == 0, f"{tape_path.name} did not pass — see failures/"


@pytest.mark.parity
@pytest.mark.parametrize("tape_path", TAPE_PARAMS_CORE)
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
