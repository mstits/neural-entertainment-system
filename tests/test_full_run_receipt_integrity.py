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


def test_every_replay_receipt_matches_the_banked_receipts_json():
    """Each docs/receipts/full_run/replay_*.json is a rerun of the same
    tape against the same receipts.json level marks. This test refuses a
    replay receipt that claims success (`all_ok`) while its own recorded
    levels disagree with the banked attestation — entry for entry, by
    end_step and wd_after/opermode — so a stale or hand-edited replay
    receipt cannot sit next to receipts.json looking authoritative."""
    receipt = json.loads((BANK / "receipts.json").read_text())
    by_end_step = {lv["end_step"]: lv for lv in receipt["levels"]}

    replay_paths = sorted(BANK.glob("replay_*.json"))
    assert replay_paths, "expected at least one docs/receipts/full_run/replay_*.json"

    for path in replay_paths:
        replay = json.loads(path.read_text())
        assert replay["all_ok"] is True, f"{path.name}: all_ok is not True"
        assert replay["opermode"] == 2, f"{path.name}: opermode != 2"
        assert len(replay["levels"]) == len(receipt["levels"]), (
            f"{path.name}: level count {len(replay['levels'])} != "
            f"receipts.json's {len(receipt['levels'])}")
        for lv in replay["levels"]:
            banked = by_end_step.get(lv["end_step"])
            assert banked is not None, (
                f"{path.name}: end_step {lv['end_step']} has no matching "
                f"entry in receipts.json")
            assert lv["level"] == banked["level"], (
                f"{path.name}: level name mismatch at end_step "
                f"{lv['end_step']}")
            assert lv["ok"] is True, (
                f"{path.name}: level {lv['level']} recorded ok=False")
            expected = banked["wd_after"] if "wd_after" in banked else 2
            assert lv["wd_after_expected"] == expected, (
                f"{path.name}: {lv['level']} wd_after_expected "
                f"{lv['wd_after_expected']} != receipts.json's {expected}")
            assert lv["wd_after_observed"] == expected, (
                f"{path.name}: {lv['level']} wd_after_observed "
                f"{lv['wd_after_observed']} != receipts.json's {expected}")
