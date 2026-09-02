"""Tests for the DO-10 hash-pin on scripts/probe_unwitnessed_bytes.py.

Mirrors tests/test_transition_witness.py::TestAgainstTheEmulator's
`test_the_rom_still_hashes_to_the_banked_provenance` -- skip when the
ROM is absent (`roms/` is gitignored and not distributable), never
silence the file.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.probe_unwitnessed_bytes import PROBES, probe_one  # noqa: E402

nes_core = pytest.importorskip("nes_core")


def _first_available_probe():
    for spec in PROBES:
        label, rom, state = spec[0], spec[1], spec[2]
        if (REPO / rom).exists() and (REPO / state).exists():
            return spec
    pytest.skip("no probe ROM/start-state pair present in this checkout")


class TestReceiptCarriesTheBytesItMeasured:
    def test_receipt_rom_and_start_state_sha256_match_the_files_on_disk(self):
        """The whole point: a receipt is a claim about *these bytes*, not
        about a filename. Assert the receipt's hashes are exactly
        hashlib.sha256 of the ROM and start-state files it actually read
        -- a swapped ROM with the same name cannot stand in for this
        receipt undetected.
        """
        label, rom, state, addrs, noop_steps, _ = _first_available_probe()
        # Keep this fast: the hash-pin doesn't depend on how many steps
        # are driven, so drive as few as will still exercise the path.
        r = probe_one(label, rom, state, addrs, noop_steps=0,
                       random_steps=1, seed=1)
        assert r["status"] == "OK"
        assert r["rom_sha256"] == hashlib.sha256(
            (REPO / rom).read_bytes()).hexdigest()
        assert r["start_state_sha256"] == hashlib.sha256(
            (REPO / state).read_bytes()).hexdigest()

    def test_receipt_hash_tracks_bytes_not_the_path_string(self, tmp_path):
        """Revert-verify in miniature: point probe_one at a corrupted
        copy of the ROM at a different path and confirm the receipt's
        rom_sha256 is the CORRUPTED file's hash, not the real one's --
        a substituted ROM cannot be laundered through a receipt that
        only ever recorded the filename.
        """
        label, rom, state, addrs, noop_steps, _ = _first_available_probe()
        real_bytes = (REPO / rom).read_bytes()
        real_sha = hashlib.sha256(real_bytes).hexdigest()

        corrupt = tmp_path / "corrupt.nes"
        corrupt.write_bytes(real_bytes[:-1] + bytes([real_bytes[-1] ^ 0xFF]))
        corrupt_sha = hashlib.sha256(corrupt.read_bytes()).hexdigest()
        assert real_sha != corrupt_sha, "fixture didn't actually corrupt anything"

        # probe_one does `REPO / rom`; pathlib resolves an absolute
        # right-hand side by discarding the left, so an absolute Path
        # here is read exactly, not re-based under REPO.
        r = probe_one(label, corrupt, state, addrs, noop_steps=0,
                       random_steps=1, seed=1)
        assert r["status"] == "OK"
        assert r["rom_sha256"] == corrupt_sha
        assert r["rom_sha256"] != real_sha
