"""Byte-exact parity between Python (numpy) and Rust SMB tile extractors.

`src/emulation/tile_observations/smb.py::SMBTileObservation.extract`
is the reference implementation. The Rust port at
`nes_core/src/smb_tile_extract.rs` exposed via
`nes_core.extract_smb_tiles(ram_bytes)` must produce a byte-exact
output for ANY input RAM. These tests pin that contract before the
Rust path is wired into the trainer.

Strategy:
  * Static fixtures — a few hand-crafted RAM snapshots that exercise
    each branch (empty, mario at right page edge, enemies, OOB ceiling).
  * Property test — N=2000 randomized RAM snapshots with constrained
    layouts so we hit interesting cases (Mario in valid region, some
    tile RAM populated, some enemies active). The randomness is seeded
    so failures reproduce deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

import nes_core
from src.emulation.tile_observations.smb import SMBTileObservation


_FEATURE_DIM = 175


@pytest.fixture(scope="module")
def py_extractor() -> SMBTileObservation:
    return SMBTileObservation()


def _rust_extract(ram: bytes) -> np.ndarray:
    out = nes_core.extract_smb_tiles(ram)
    assert out.shape == (_FEATURE_DIM,)
    assert out.dtype == np.int8
    return out


def test_empty_ram(py_extractor: SMBTileObservation) -> None:
    ram = bytes(2048)
    assert np.array_equal(py_extractor.extract(ram), _rust_extract(ram))


def test_mario_at_origin_with_solid_tile(py_extractor: SMBTileObservation) -> None:
    ram = bytearray(2048)
    ram[0x006D] = 0   # x_page
    ram[0x0086] = 96  # x_low
    ram[0x00CE] = 96  # y
    # Solid tile right beneath Mario in tile RAM.
    ram[0x0500 + 7 * 16 + 6] = 0x42
    ram = bytes(ram)
    assert np.array_equal(py_extractor.extract(ram), _rust_extract(ram))


def test_mario_with_enemies(py_extractor: SMBTileObservation) -> None:
    ram = bytearray(2048)
    ram[0x006D] = 0
    ram[0x0086] = 96
    ram[0x00CE] = 96
    # Activate two enemies, one in window, one offscreen.
    ram[0x000F] = 1  # enemy slot 0 active
    ram[0x006E] = 0
    ram[0x0087] = 96 + 32
    ram[0x00CF] = 96
    ram[0x0010] = 1  # enemy slot 1 active
    ram[0x006F] = 5  # very far offscreen
    ram[0x0088] = 0
    ram[0x00D0] = 96
    ram = bytes(ram)
    assert np.array_equal(py_extractor.extract(ram), _rust_extract(ram))


def test_left_of_world_origin_is_empty_not_phantom(
    py_extractor: SMBTileObservation,
) -> None:
    """Cells left of world x=0 must be empty, not a phantom page-1 tile.

    Audit F63: the validity mask checked world_px_y >= 0 but not
    world_px_x >= 0, so for Mario near the left edge the floor-mod
    (Python) / rem_euclid (Rust) wrapped a negative world_px_x into a
    valid-looking page-1 sub-tile — reporting phantom terrain. We fill
    page-1 tile RAM densely so any wrap would surface as solid, then
    assert the leftmost grid column is empty.
    """
    ram = bytearray(2048)
    ram[0x006D] = 0   # x_page = 0
    ram[0x0086] = 8   # x_low = 8 -> mario_x_world = 8; left cells are < 0
    ram[0x00CE] = 96  # y
    for i in range(13 * 16):          # fill page-1 region densely
        ram[0x0500 + 13 * 16 + i] = 0x42
    ram = bytes(ram)
    py = py_extractor.extract(ram)
    rust = _rust_extract(ram)
    assert np.array_equal(py, rust), "Rust/Python must still agree at the left edge"
    grid = np.asarray(py[:169]).reshape(13, 13)
    assert (grid[:, 0] == 0).all(), (
        "leftmost column (world_px_x < 0) must be empty, not phantom page-1 solid"
    )


def test_mario_at_page_boundary(py_extractor: SMBTileObservation) -> None:
    """Mario at x=255 (last col of page 0) — grid wraps into page 1."""
    ram = bytearray(2048)
    ram[0x006D] = 0
    ram[0x0086] = 250
    ram[0x00CE] = 96
    # Populate a few tiles in BOTH page 0 and page 1 to ensure the
    # `page * 13*16` offset is correct in Rust.
    for sx in (14, 15):
        ram[0x0500 + 7 * 16 + sx] = 0x77
    for sx in (0, 1):
        ram[0x0500 + 13 * 16 + 7 * 16 + sx] = 0x88  # page 1 row 7
    ram = bytes(ram)
    assert np.array_equal(py_extractor.extract(ram), _rust_extract(ram))


def test_mario_above_ceiling(py_extractor: SMBTileObservation) -> None:
    """Mario at y=8 — most grid cells should fall above the playfield."""
    ram = bytearray(2048)
    ram[0x006D] = 0
    ram[0x0086] = 96
    ram[0x00CE] = 8
    ram = bytes(ram)
    assert np.array_equal(py_extractor.extract(ram), _rust_extract(ram))


def test_velocity_signed_round_trip(py_extractor: SMBTileObservation) -> None:
    ram = bytearray(2048)
    ram[0x0057] = 0xF0  # -16 signed
    ram[0x009F] = 0x40  # +64 signed
    ram = bytes(ram)
    py = py_extractor.extract(ram)
    rs = _rust_extract(ram)
    assert py[169] == -16
    assert py[170] == 64
    assert np.array_equal(py, rs)


def test_lives_and_powerup_clamped(py_extractor: SMBTileObservation) -> None:
    """RAM[$0756] and RAM[$075A] above 127 must clamp to 127 (int8 range)."""
    ram = bytearray(2048)
    ram[0x0756] = 200
    ram[0x075A] = 250
    ram = bytes(ram)
    py = py_extractor.extract(ram)
    rs = _rust_extract(ram)
    assert py[172] == 127
    assert py[173] == 127
    assert np.array_equal(py, rs)


def test_validates_too_short_input() -> None:
    with pytest.raises(ValueError):
        nes_core.extract_smb_tiles(bytes(100))


@pytest.mark.parametrize("seed", list(range(8)))
def test_random_ram_byte_exact(py_extractor: SMBTileObservation, seed: int) -> None:
    """Property test: randomized RAM + constrained Mario position.

    8 seeds × 250 samples = 2000 random RAM snapshots per run.
    """
    rng = np.random.default_rng(seed)
    for _ in range(250):
        ram = bytearray(rng.integers(0, 256, size=2048, dtype=np.uint8).tobytes())
        # Constrain Mario into the valid playfield to avoid trivial all-zero
        # outputs. Random RAM with random Mario position would put him above
        # the ceiling 90% of the time and dilute coverage.
        ram[0x006D] = int(rng.integers(0, 4))   # page 0..3
        ram[0x0086] = int(rng.integers(0, 256)) # any sub-x
        ram[0x00CE] = int(rng.integers(40, 200)) # below ceiling, above floor
        ram = bytes(ram)
        py = py_extractor.extract(ram)
        rs = _rust_extract(ram)
        if not np.array_equal(py, rs):
            diffs = np.where(py != rs)[0]
            raise AssertionError(
                f"seed={seed} mismatch at indices {diffs.tolist()}: "
                f"py={py[diffs].tolist()} rs={rs[diffs].tolist()}"
            )
