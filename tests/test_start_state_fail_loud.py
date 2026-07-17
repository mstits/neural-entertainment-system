"""Missing configured start states must fail at startup, not cold-boot.

A `start_state_path` pointing at a file that no longer exists used to
be downgraded to a warning: the trainer nulled the path and the pool
cold-booted to the title screen, silently training the attract-mode
demo (inputs ignored) — or, via the launcher's sidecar fallback, a
different level than the one configured. An entire run was wasted
before anyone noticed. These tests pin the fail-loud contract at both
layers:

  * `Trainer.__init__` — a configured-but-missing path raises
    `FileNotFoundError` naming the path, before any pool spawns.
  * `RustPool.__init__` — the same check at the adapter boundary, so
    callers that bypass the Trainer (dreamer, replay window) fail loud
    at construction too, instead of deep inside the training thread
    when `nes_core.Pool` finally tries to read the file.

`None` (nothing configured) keeps its meaning at both layers: a
deliberate cold boot. The launcher-level contract — the profile's
declared path must not silently fall back to a sidecar — is pinned in
test_rom_resolver.py alongside the other resolver tests.
"""

from __future__ import annotations

import logging

import pytest

from src.emulation.rust_pool_adapter import RustPool


# --------------------------------------------------------------------------
# RustPool adapter boundary
# --------------------------------------------------------------------------


def test_pool_missing_start_state_raises_at_construction(tmp_path):
    """A nonexistent configured path must raise at __init__ — before
    start(), before any worker spawns — and the message must name it."""
    missing = tmp_path / "gone.state.bin"
    with pytest.raises(FileNotFoundError, match="gone.state.bin"):
        RustPool(
            rom_path="unused.nes", num_workers=2,
            start_state_path=str(missing),
        )


def test_pool_directory_start_state_raises(tmp_path):
    """A directory is not a loadable save state — same hard error."""
    with pytest.raises(FileNotFoundError):
        RustPool(
            rom_path="unused.nes", num_workers=2,
            start_state_path=str(tmp_path),
        )


def test_pool_none_start_state_still_constructs():
    """No start state configured => deliberate cold boot, no error."""
    p = RustPool(rom_path="unused.nes", num_workers=2, start_state_path=None)
    assert p.start_state_path is None


def test_pool_existing_start_state_accepted(tmp_path):
    ss = tmp_path / "ok.state.bin"
    ss.write_bytes(b"NCST\x01state")
    p = RustPool(
        rom_path="unused.nes", num_workers=2, start_state_path=str(ss),
    )
    assert p.start_state_path == str(ss)


# --------------------------------------------------------------------------
# Trainer startup validation
# --------------------------------------------------------------------------

# Minimal profile: Trainer.__init__ requires a non-empty action_space;
# everything else falls back to defaults. The ROM path never has to
# exist for these tests — the start-state check fires before any ROM
# access, and the None case only reads the ROM lazily at pool start.
_PROFILE = {"name": "faketest", "action_space": [["right"], ["right", "A"]]}


def _make_trainer(tmp_path, start_state_path):
    from src.training.trainer import Trainer

    return Trainer(
        rom_path=str(tmp_path / "nonexistent.nes"),
        game_profile=dict(_PROFILE),
        num_instances=2,
        population_size=2,
        checkpoint_dir=str(tmp_path / "ckpt"),  # explicit -> isolated
        start_state_path=start_state_path,
    )


def test_trainer_missing_start_state_is_startup_error(tmp_path):
    """The old behavior (warn + fall back to cold boot) silently trained
    the wrong content; now construction must fail, naming the path."""
    missing = tmp_path / "gone.state.bin"
    with pytest.raises(FileNotFoundError, match="gone.state.bin"):
        _make_trainer(tmp_path, str(missing))


def test_trainer_directory_start_state_is_startup_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _make_trainer(tmp_path, str(tmp_path))


def test_trainer_none_start_state_cold_boots(tmp_path, caplog):
    """Nothing configured => construction succeeds (deliberate cold
    boot) and the loud title-screen-demo warning still fires."""
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        t = _make_trainer(tmp_path, None)
    assert t.start_state_path is None
    assert "cold-boot" in caplog.text


def test_trainer_empty_start_state_treated_as_unconfigured(tmp_path):
    """'' (e.g. a cleared GUI field) means unconfigured, not missing."""
    t = _make_trainer(tmp_path, "")
    assert not t.start_state_path
