"""SMB tile extractor tests — verify RAM decoding produces the
expected feature vector for known game states."""

from __future__ import annotations

import numpy as np
import pytest

from src.emulation.tile_observations import get_extractor
from src.emulation.tile_observations.smb import SMBTileObservation, FEATURE_DIM


def _empty_ram() -> bytearray:
    """2 KB of zeros — no level loaded, no enemies, Mario at origin."""
    return bytearray(2048)


def _ram_with_mario_at(x_world: int, y_screen: int) -> bytearray:
    """Build a synthetic RAM where Mario is at the given world X
    (split across page+low) and screen Y."""
    ram = _empty_ram()
    ram[0x006D] = (x_world >> 8) & 0xFF
    ram[0x0086] = x_world & 0xFF
    ram[0x00CE] = y_screen
    return ram


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def test_factory_returns_smb_extractor() -> None:
    ext = get_extractor("smb_tiles")
    assert isinstance(ext, SMBTileObservation)
    assert ext.feature_dim == FEATURE_DIM


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_extractor("not_a_real_encoder")


# ---------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------


def test_feature_dim_is_175() -> None:
    ext = SMBTileObservation()
    assert ext.feature_dim == 175  # 13*13 grid + 6 scalars


def test_extract_returns_int8_vector() -> None:
    ext = SMBTileObservation()
    out = ext.extract(bytes(_empty_ram()))
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.int8


# ---------------------------------------------------------------------
# Empty RAM behavior
# ---------------------------------------------------------------------


def test_empty_ram_yields_empty_grid() -> None:
    ext = SMBTileObservation()
    out = ext.extract(bytes(_empty_ram()))
    assert np.all(out[:169] == 0)


def test_empty_ram_on_ground_flag_is_one() -> None:
    """The on_ground RAM byte (0x001D) is 0 in empty RAM, which
    SMB encodes as 'on the ground'. The extractor flips this to
    the more natural '1 = on ground' convention."""
    ext = SMBTileObservation()
    out = ext.extract(bytes(_empty_ram()))
    assert out[171] == 1


# ---------------------------------------------------------------------
# Tile decoding
# ---------------------------------------------------------------------


def test_solid_tile_below_mario() -> None:
    """Place a non-zero metatile in the RAM at the position where the
    extractor's Y mapping puts grid (6, 9), then verify it shows up."""
    ext = SMBTileObservation()
    ram = _ram_with_mario_at(x_world=96, y_screen=96)
    # World math (matches extractor):
    #   sub_y for grid dy = (y_screen + (dy-6)*16 - 32) / 16 = dy - 2
    #   So grid (6, 9) has sub_y=7, page=0, sub_x=6.
    ram[0x0500 + 0 * 208 + 7 * 16 + 6] = 0x42
    out = ext.extract(bytes(ram))
    assert out[9 * 13 + 6] == 1  # solid
    assert out[7 * 13 + 6] == 0  # empty


def test_enemy_marks_grid_cell() -> None:
    ext = SMBTileObservation()
    ram = _ram_with_mario_at(x_world=96, y_screen=96)
    # Activate enemy slot 0 at 2 tiles right of Mario.
    ram[0x000F + 0] = 1
    ram[0x006E + 0] = 0
    ram[0x0087 + 0] = 96 + 32  # X = 128 = 2 tiles right of Mario's 96
    ram[0x00CF + 0] = 96       # same Y as Mario
    out = ext.extract(bytes(ram))
    # Grid offset: dx = (128 - 96) / 16 = 2 → grid (8, 6)
    assert out[6 * 13 + 8] == -1


def test_inactive_enemy_slot_ignored() -> None:
    ext = SMBTileObservation()
    ram = _ram_with_mario_at(x_world=96, y_screen=96)
    # Slot 0 inactive (flag=0) but coordinates filled — should be skipped.
    ram[0x000F + 0] = 0
    ram[0x006E + 0] = 0
    ram[0x0087 + 0] = 128
    ram[0x00CF + 0] = 96
    out = ext.extract(bytes(ram))
    # Grid (8, 6) should still be empty.
    assert out[6 * 13 + 8] == 0


# ---------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------


def test_scalar_lives() -> None:
    ext = SMBTileObservation()
    ram = _empty_ram()
    ram[0x075A] = 3  # 3 lives
    out = ext.extract(bytes(ram))
    assert out[173] == 3


def test_scalar_powerup_state() -> None:
    ext = SMBTileObservation()
    ram = _empty_ram()
    ram[0x0756] = 2  # fire flower
    out = ext.extract(bytes(ram))
    assert out[172] == 2


def test_scalar_subtile_x() -> None:
    """Sub-tile X position should be `x_low % 16`, encoding Mario's
    pixel offset within the current tile."""
    ext = SMBTileObservation()
    ram = _empty_ram()
    ram[0x0086] = 99  # x_low % 16 = 3
    out = ext.extract(bytes(ram))
    assert out[174] == 3


def test_scalar_velocity_signed() -> None:
    """Velocities are signed — a backward-moving Mario should produce
    a negative X velocity scalar."""
    ext = SMBTileObservation()
    ram = _empty_ram()
    ram[0x0057] = 0xF0  # -16 as signed byte
    out = ext.extract(bytes(ram))
    assert out[169] == -16
