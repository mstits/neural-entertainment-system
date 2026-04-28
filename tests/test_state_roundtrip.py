"""Regression guard: NCST state blob save/load must round-trip.

History (2026-04-21): `src/gui/play_window.py::_save_state` was
pre-pending `b"NCST\\x01"` to a blob that `env.save_state()` already
prefixed → every saved file was double-prefixed → training Start
failed with `"start-state snapshot unreadable: io error"`.

This test verifies:

1. `env.save_state()` returns a blob with exactly one NCST prefix.
2. Round-trip: save → load on fresh env → RAM matches byte-exact.
3. Save → (append extra NCST prefix manually) → load still succeeds,
   thanks to the belt-and-suspenders multi-prefix-strip in
   `python.rs::load_state`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import nes_core


_NCST_MAGIC = b"NCST\x01"


def _make_synthetic_nrom() -> bytes:
    """Same minimal NROM as test_button_bit_alignment uses."""
    hdr = b"NES\x1a" + bytes([1, 1, 0, 0]) + b"\x00" * 8
    prg = bytearray(b"\xEA" * 0x4000)
    prg[0x0000:0x0003] = b"\x4C\x00\xC0"
    prg[0x3FFC] = 0x00
    prg[0x3FFD] = 0xC0
    return bytes(hdr) + bytes(prg) + b"\x00" * 0x2000


def _fresh_env():
    with tempfile.NamedTemporaryFile(suffix=".nes", delete=False) as f:
        f.write(_make_synthetic_nrom())
        rom = f.name
    env = nes_core.NESEnvironment(rom_path=rom, frame_skip=1)
    env.reset()
    return env, rom


def test_save_state_has_single_ncst_prefix() -> None:
    env, rom = _fresh_env()
    try:
        blob = bytes(env.save_state())
        assert blob.startswith(_NCST_MAGIC), "save_state must start with NCST\\x01"
        # After stripping the prefix once, the remaining body must
        # NOT start with NCST\x01 again. That's the double-prefix
        # bug that bit us in play_window.py; env.save_state should
        # never produce that shape on its own.
        body = blob[len(_NCST_MAGIC):]
        assert not body.startswith(_NCST_MAGIC), (
            "save_state produced double-NCST prefix — regression of the "
            "play_window.py _save_state bug from 2026-04-21"
        )
    finally:
        Path(rom).unlink(missing_ok=True)


def test_save_load_roundtrip_preserves_state() -> None:
    env_a, rom = _fresh_env()
    try:
        # Step a few times so the state isn't reset-default.
        for _ in range(20):
            env_a.step(0)
        blob = bytes(env_a.save_state())

        # Load into a fresh env and verify it's at the same spot.
        env_b = nes_core.NESEnvironment(rom_path=rom, frame_skip=1)
        env_b.reset()
        env_b.load_state(blob)

        # Stepping with the same action should produce the same RAM.
        env_a.step(0)
        env_b.step(0)
        ram_a = bytes(env_a.get_ram_range(0, 0x0800))
        ram_b = bytes(env_b.get_ram_range(0, 0x0800))
        assert ram_a == ram_b, "post-load RAM diverges from pre-load"
    finally:
        Path(rom).unlink(missing_ok=True)


def test_load_state_tolerates_double_ncst_prefix() -> None:
    """Old play_window.py wrote `NCST\\x01 NCST\\x01 <body>`. The
    load_state path now strips repeated prefixes to recover those
    pre-fix user files without manual hand-editing.
    """
    env_a, rom = _fresh_env()
    try:
        for _ in range(10):
            env_a.step(0)
        good_blob = bytes(env_a.save_state())
        # Simulate the historical bug: one extra prefix on the front.
        poisoned = _NCST_MAGIC + good_blob

        env_b = nes_core.NESEnvironment(rom_path=rom, frame_skip=1)
        env_b.reset()
        # Loader must strip both prefixes and deserialize the body.
        env_b.load_state(poisoned)
    finally:
        Path(rom).unlink(missing_ok=True)
