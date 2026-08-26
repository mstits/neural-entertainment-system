"""Pin the odometer cert's ROM-identity gate.

`scripts/odometer_cert.py` is the trust root for the PPU scroll odometer:
CLAIMS.md and every solver profile that declares `rom_hashes` (rygar,
gradius, megaman, ...) cites its receipt as proof the instrument was
measured against the real cartridge. Before the ROM-identity check, the
cert never validated the loaded ROM's MD5 against the profile's declared
`rom_hashes` at all, and the receipt it wrote recorded no hash — so a
byte-different dump (bad ROM-library merge, translation patch, wrong
revision) left in place of the real ROM, with the profile's `rom_hashes`
untouched, still certified PASS with a receipt indistinguishable from a
real one.

This test reproduces that exact scenario against a live `nes_core.Pool`:
copy the real, committed SMB ROM, flip one byte inside its CHR bank
(gameplay/RAM-invisible — the odometer behaves identically either way),
and certify against the unmodified profile. It must FAIL, and the
receipt must show which hash didn't match.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms" / "Super Mario Bros. (World).nes"
PROFILE = REPO / "configs" / "mario_1_1_backward.yaml"


def _load_odometer_cert():
    script = REPO / "scripts" / "odometer_cert.py"
    spec = importlib.util.spec_from_file_location("odometer_cert_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def oc():
    return _load_odometer_cert()


def _needs_fixtures():
    return not (ROM.exists() and PROFILE.exists())


@pytest.mark.skipif(_needs_fixtures(), reason="real SMB ROM/profile not present in this checkout")
def test_swapped_rom_fails_cert_and_receipt_shows_the_mismatch(oc, tmp_path):
    real_bytes = ROM.read_bytes()
    real_md5 = hashlib.md5(real_bytes).hexdigest()
    profile = yaml.safe_load(PROFILE.read_text())
    declared = [h.lower() for h in profile["rom_hashes"]]
    assert real_md5 in declared  # sanity: the committed ROM matches the profile

    # A byte-different dump: flip one bit deep in the CHR bank (tile data,
    # never read as gameplay/RAM state) so behavior is identical but the
    # file — and its MD5 — is not.
    swapped = bytearray(real_bytes)
    swapped[-200] ^= 0x01
    swapped_path = tmp_path / "swapped.nes"
    swapped_path.write_bytes(bytes(swapped))
    swapped_md5 = hashlib.md5(bytes(swapped)).hexdigest()
    assert swapped_md5 != real_md5
    assert swapped_md5 not in declared

    # The profile's rom_hashes is left exactly as-is; only rom_path swaps.
    profile["rom_path"] = str(swapped_path)
    tmp_profile = tmp_path / "swapped_profile.yaml"
    tmp_profile.write_text(yaml.safe_dump(profile))

    receipt_path = tmp_path / "cert.json"
    rc = oc.run(profile=str(tmp_profile), steps=600, out=str(receipt_path))

    assert rc == 1  # must quarantine, never exit 0 on a mismatched ROM
    verdict = json.loads(receipt_path.read_text())
    assert verdict["passed"] is False
    assert verdict["rom_md5"] == swapped_md5  # receipt records the hash it saw
    assert verdict["checks"]["rom_hash_verified"]["passed"] is False
    assert verdict["checks"]["rom_hash_verified"]["computed_md5"] == swapped_md5
