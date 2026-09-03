"""author.py coverage: script parsing + round-trip through run.py."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.parity.author import parse_script, ranges_to_tape_inputs, write_tape
from tests.parity.run import run
from tests.skip_gates import requires, requires_module

REPO = Path(__file__).resolve().parents[2]
ROM_REL = "roms/Mario Bros. (World).nes"
ROM = REPO / ROM_REL


@pytest.fixture(autouse=True)
def _silence_gym_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
def test_parse_script_handles_mixed_forms():
    text = """
    # comment
    120 0
    10 8     # Start held
    128      # one frame of Right
    """
    ranges = parse_script(text)
    assert ranges == [(120, 0), (10, 8), (1, 128)]


@pytest.mark.parity
def test_parse_script_rejects_bad_mask():
    with pytest.raises(ValueError, match="mask"):
        parse_script("1 999\n")


@pytest.mark.parity
def test_parse_script_rejects_bad_count():
    with pytest.raises(ValueError, match="count"):
        parse_script("0 0\n")


@pytest.mark.parity
def test_ranges_to_tape_inputs_are_contiguous():
    inputs, total = ranges_to_tape_inputs([(5, 0), (3, 8), (2, 128)])
    assert total == 10
    assert inputs == [
        {"start_frame": 0, "end_frame": 5, "buttons": 0},
        {"start_frame": 5, "end_frame": 8, "buttons": 8},
        {"start_frame": 8, "end_frame": 10, "buttons": 128},
    ]


# The two round-trip tests below author a tape against a real dump and
# replay it. roms/* is gitignored (.gitignore:71); the cross_emulator
# leg additionally drives nes-py, which requirements.txt does not
# install. The parse_script tests above need neither and stay ungated.
@pytest.mark.parity
@requires(ROM_REL)
@requires_module("nes_py")
def test_authored_cross_emulator_tape_passes_run_py(tmp_path):
    # Author a 30-frame no-input tape → run via run.py → expect pass.
    out = tmp_path / "mario_idle.json"
    write_tape(
        out, name="mario_idle", rom=ROM, ranges=[(30, 0)],
        mode="cross_emulator", notes="round-trip test",
        warmup=10, tolerance_scanlines=0, tolerance_pixels=0,
    )
    assert out.exists()
    assert run(out) == 0


@pytest.mark.parity
@requires(ROM_REL)
def test_authored_golden_hash_tape_passes_run_py(tmp_path):
    # Author a golden-hash tape → immediately run it → hashes must match
    # (same build, no intervening code change).
    out = tmp_path / "mario_golden.json"
    write_tape(
        out, name="mario_golden", rom=ROM, ranges=[(20, 0), (10, 8), (10, 128)],
        mode="golden_hash", notes="round-trip test",
        warmup=0, tolerance_scanlines=0, tolerance_pixels=0,
    )
    assert out.exists()
    assert (tmp_path / "mario_golden.golden.bin").exists()
    assert run(out) == 0
