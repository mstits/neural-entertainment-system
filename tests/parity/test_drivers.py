"""Driver sanity: button-bit alignment and run-to-run determinism.

These aren't parity tests per se — they prove that both emulator wrappers
honor the same input contract and produce the same output across repeated
runs. A failure here invalidates every downstream tape.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.parity.drivers import NESCoreDriver, NESPyDriver
from tests.skip_gates import requires, requires_module

REPO = Path(__file__).resolve().parents[2]
ROM_REL = "roms/Mario Bros. (World).nes"
ROM = REPO / ROM_REL

# All three tests here drive BOTH wrappers on a real dump: they need the
# gitignored ROM (.gitignore:71) and nes-py, which is deliberately
# quarantined to requirements-legacy-bakeoff.txt and so is absent from a
# default install.
pytestmark = [requires(ROM_REL), requires_module("nes_py")]


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _run(driver_cls, frames: int, buttons_sequence: list[int]) -> np.ndarray:
    assert len(buttons_sequence) == frames
    d = driver_cls(ROM)
    for b in buttons_sequence:
        d.step(b)
    return d.frame()


@pytest.mark.parity
def test_drivers_produce_rgb_frames():
    for cls in (NESCoreDriver, NESPyDriver):
        f = _run(cls, 1, [0])
        assert f.shape == (240, 256, 3), f"{cls.__name__}: {f.shape}"
        assert f.dtype == np.uint8, f"{cls.__name__}: {f.dtype}"


@pytest.mark.parity
def test_no_input_baseline_is_byte_exact_at_30_frames():
    # Duplicates test_palette_parity.py's check, but goes through the driver
    # wrappers — exercises the wrapper contract, not just the raw libs.
    a = _run(NESCoreDriver, 30, [0] * 30)
    b = _run(NESPyDriver, 30, [0] * 30)
    diff = (a != b).any(axis=2).sum()
    assert diff == 0, f"{diff} pixels differ through driver wrappers"


@pytest.mark.parity
def test_each_driver_is_deterministic_run_to_run():
    # Same ROM, same inputs, same hardware — two runs of the same driver
    # must produce the same frame. If this flakes, nes_core or nes-py has
    # a hidden RNG source and the harness can't trust any tape.
    seq = [0] * 120 + [128] * 60  # idle + Right
    for cls in (NESCoreDriver, NESPyDriver):
        f1 = _run(cls, 180, seq)
        f2 = _run(cls, 180, seq)
        diff = (f1 != f2).any(axis=2).sum()
        assert diff == 0, f"{cls.__name__} is non-deterministic: {diff} pixels vary"
