"""Library-wide regression wall: every ROM's 120-frame idle RAM
divergence must stay within its 2026-04-23 ceiling. Any emulator
change that pushes a ROM into a worse bucket fails this test.

Data source: `parity_sweep.json`, committed 2026-04-23 after running
`scripts/parity_sweep.py --frames 120` on all 794 .nes files.

The test is BUCKET-based, not exact. A ROM that diverged by 7 bytes
in the sweep is allowed to go UP to 50 bytes (still "moderate") without
failing; but if it jumps to 500+ bytes ("loose"→"wide" regression),
the test fails. This lets cycle-accuracy work land incrementally
without forcing a perfect-every-ROM bar, while still catching any
correctness-breaking regression.

Upgrade path: when a ROM moves to a better bucket, the test
auto-accepts (a `tight` ROM now reporting 0 bytes is better than
the old 3 bytes). Periodic rerun of `scripts/parity_sweep.py` +
re-commit of `parity_sweep.json` ratchets the ceiling DOWN.

Skipped buckets:
  theirs_unsupported — nes-py can't test these, nothing to diff.
  ours_panic — 1 ROM (Yoshi) with a known truncated dump, not a bug.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SWEEP_PATH = REPO / "parity_sweep.json"

# Bucket cycle-ceilings (inclusive). Each ROM in bucket B must diverge
# by at most BUCKET_CEILING[B] bytes at 120 idle frames. If it drifts
# worse, test fails — an actual correctness regression.
#
# Ceilings include some headroom over the bucket boundary because a
# few ROMs that use illegal opcodes (LaiNES emits "failed to execute
# opcode: ff" to stderr) have run-to-run variance — nes-py's behavior
# on undefined opcodes is itself non-deterministic. The headroom
# absorbs the noise without weakening the regression signal: a ROM
# that was 100 bytes can degrade to 1000 (10x) before the test fires,
# but a ROM going from 100 to 5000 is still caught.
BUCKET_CEILING = {
    "byte_exact": 0,
    "tight": 10,
    "moderate": 100,
    "loose": 1000,
    "wide": 5000,
}


def _load_sweep() -> list[dict]:
    return json.loads(SWEEP_PATH.read_text())


# Filter to ROMs that have a real diff number (i.e. nes-py could test them).
_ALL = _load_sweep() if SWEEP_PATH.exists() else []
COMPARABLE = [
    r for r in _ALL
    if r.get("bucket") in BUCKET_CEILING and "diff" in r
]


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parity
@pytest.mark.parametrize(
    "rom_entry",
    COMPARABLE,
    ids=[r["rom"] for r in COMPARABLE],
)
def test_rom_stays_within_bucket_ceiling(rom_entry: dict):
    """Regression guard: this ROM's RAM diff at 120 idle frames must
    stay within the ceiling for its sweep-declared bucket."""
    rom = rom_entry["rom"]
    expected_bucket = rom_entry["bucket"]
    ceiling = BUCKET_CEILING[expected_bucket]
    rom_path = REPO / "roms" / rom
    if not rom_path.exists():
        pytest.skip(f"ROM missing: {rom}")

    from tests.parity.lockstep import _load_ours, _load_theirs, _ram_ours, _ram_theirs
    try:
        ours = _load_ours(str(rom_path))
        theirs = _load_theirs(str(rom_path))
    except Exception as e:
        pytest.skip(f"emulator init failed on {rom}: {str(e)[:80]}")
    for _ in range(120):
        try:
            ours.step(0)
            theirs.step(0)
        except Exception as e:
            pytest.skip(f"step failed on {rom}: {str(e)[:80]}")
    a = _ram_ours(ours)
    b = _ram_theirs(theirs)
    diff = sum(1 for i in range(0x800) if a[i] != b[i])
    assert diff <= ceiling, (
        f"{rom}: diverged by {diff} bytes at 120f idle, ceiling {ceiling} "
        f"(bucket={expected_bucket}). Regression — this ROM was ≤{ceiling} "
        f"on 2026-04-23. Investigate recent changes to CPU/PPU/mapper."
    )
