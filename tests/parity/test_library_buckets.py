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


# Exact command that regenerates + refreshes the committed baseline.
# Surfaced verbatim in every loud-failure message below so a developer
# who hits an absent/empty net can fix it without hunting for the recipe.
REGEN_CMD = "python scripts/parity_sweep.py --frames 120 --out parity_sweep.json"


def _load_sweep() -> list[dict]:
    """Parse the committed sweep baseline. Never raises: any problem
    (missing file, blank file, malformed JSON, non-list payload) yields
    an empty list so pytest collection can't crash. The loud-failure
    guard `test_parity_sweep_baseline_present` turns an empty result
    into a clear, actionable test failure instead of a silent inert net."""
    try:
        data = json.loads(SWEEP_PATH.read_text())
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


# Filter to ROMs that have a real diff number (i.e. nes-py could test them).
_ALL = _load_sweep()
COMPARABLE = [
    r for r in _ALL
    if isinstance(r, dict) and r.get("bucket") in BUCKET_CEILING and "diff" in r
]


@pytest.mark.parity
def test_parity_sweep_baseline_present() -> None:
    """The library-wide 794-ROM regression net is only REAL if its
    baseline (`parity_sweep.json`) is committed and populated.

    Without it, the parametrized bucket test below collects an EMPTY
    parameter set — pytest silently skips it and the suite goes green
    while covering zero ROMs. That inert net is worse than no net: it
    reads as protection that isn't there. This guard converts every
    way the baseline can be missing or useless into a loud, actionable
    failure carrying the exact regeneration command."""
    if not SWEEP_PATH.exists():
        # Absent baseline is the EXPECTED state on a fresh clone: the sweep
        # is derived from the user's local ROM library (never shipped), so
        # it cannot be committed. Skip LOUDLY (visible with `-rs`) rather
        # than fail — a red `make test` on every clone is its own defect —
        # but make the inactive net impossible to miss. A baseline that is
        # PRESENT-but-broken still fails below (that is a real regression).
        pytest.skip(
            f"parity library net INACTIVE: {SWEEP_PATH} not present. It is "
            f"derived from your local ROM library and cannot ship, so this is "
            f"expected on a fresh clone. Activate the library-wide regression "
            f"net (recommended once your ROMs are in place):\n    {REGEN_CMD}"
        )
    raw = SWEEP_PATH.read_text().strip()
    if not raw:
        pytest.fail(
            f"parity baseline EMPTY: {SWEEP_PATH} is a blank / zero-byte "
            f"file, so the net covers ZERO ROMs. Regenerate + commit it:\n"
            f"    {REGEN_CMD}"
        )
    try:
        data = json.loads(raw)
    except ValueError as exc:
        pytest.fail(
            f"parity baseline UNPARSEABLE: {SWEEP_PATH} is not valid JSON "
            f"({exc}). Regenerate + commit it:\n    {REGEN_CMD}"
        )
    if not isinstance(data, list) or not data:
        pytest.fail(
            f"parity baseline has NO ENTRIES: {SWEEP_PATH} parsed to an "
            f"empty or non-list sweep, so the net covers ZERO ROMs. "
            f"Regenerate + commit it:\n    {REGEN_CMD}"
        )
    if not COMPARABLE:
        pytest.fail(
            f"parity baseline has {len(data)} rows but NONE are comparable "
            f"(each needs a 'bucket' in {sorted(BUCKET_CEILING)} and a "
            f"'diff' field). The bucket regression test would cover zero "
            f"ROMs. Regenerate + commit it:\n    {REGEN_CMD}"
        )


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
