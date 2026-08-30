"""The flagship tape's attestation, verified the way it was computed.

docs/receipts/full_run/ banks the complete-game action tape (78261c7,
2026-07-27). Its `tape_sha256` is computed over the RAW uint8 action
bytes (`tape_arr.tobytes()`, assemble_full_run.py:223), NOT over the
.npy file — the file adds a 128-byte numpy header, so a naive
`shasum full_tape.npy` reads as an integrity failure on an intact
artifact. That exact misread triggered a STOP on a 127G archival
decision on 2026-08-29 and cost two sessions a verification round.

This test IS the documented verification recipe, and it turns real
corruption of the banked tape (truncation, byte flips) into a red test
instead of a silent lie under the receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BANK = Path(__file__).resolve().parents[1] / "docs" / "receipts" / "full_run"


def test_banked_tape_matches_its_receipt_in_the_receipts_own_domain():
    receipt = json.loads((BANK / "receipts.json").read_text())
    arr = np.load(BANK / "full_tape.npy")
    assert arr.dtype == np.uint8 and arr.ndim == 1
    assert len(arr) == receipt["steps"], (
        "tape length must equal the receipt's step count — one uint8 "
        "action per step")
    derived = hashlib.sha256(arr.tobytes()).hexdigest()
    assert derived == receipt["tape_sha256"], (
        "banked tape no longer matches its attestation (hash domain: "
        "raw action bytes via arr.tobytes(), NOT the .npy file)")


def test_the_naive_file_hash_is_documented_as_different():
    """Pin the ambiguity that caused the false alarm: the FILE hash is
    NOT the receipt hash, and never was. If numpy's header format ever
    changes such that these coincide, this test says so."""
    receipt = json.loads((BANK / "receipts.json").read_text())
    file_hash = hashlib.sha256((BANK / "full_tape.npy").read_bytes()).hexdigest()
    assert file_hash != receipt["tape_sha256"]
