"""Guard the H1 fix: `scripts/observatory.py` must not fold a profile's
`ram_mapping` block into the pre-probe exclusion set.

Surfaced 2026-08-26 during the quarantine-hardening adjudication.
`observatory.py` used to compute `known = coord_bytes | mapped` and then
`excluded |= known` INLINE IN `main()`, so any address appearing in a
profile's `ram_mapping` — including the six quarantined Zelda addresses
`configs/zelda_gui_tuned.yaml` was carrying — was silently removed from
candidate generation at the `if b in excluded: continue` gate. The
exclusion's stated purpose ("discover NEW bytes only") is a REPORTING
concern; folding it into the set that gates candidate generation
implements it as blindness instead. This is the load-bearing fix:
everything else in the quarantine hardening only matters because bytes
excluded here could never be rediscovered by this project's own
instrument regardless of any YAML cleanup.

WHY THIS FILE LOOKS THE WAY IT DOES. Its first version asserted against a
COPY of the fixed line pasted into the test body (`excluded =
set(coord_bytes)  # mirrors main()`), because the decision lived in
`main()` and `main()` needs a ROM and an archive. Verification restored
the exact original defect in production and all three tests stayed green:
the test certified nothing. The production code was therefore restructured
so the whole decision lives in `pre_probe_exclusions(profile, full)`,
which needs only a synthetic RAM array — every test below drives THAT
function and reads the real `excluded` set it returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from observatory import STACK_LO, pre_probe_exclusions  # noqa: E402

#: The quarantined Zelda win flag (RAM_GANON_DEFEATED). Anything that
#: excludes THIS byte has structurally prevented its own rediscovery.
GANON = 0x0672
SONG = 0x0609


def _profile(ram_mapping: dict | None = None, solve: dict | None = None) -> dict:
    return {
        "solve": solve if solve is not None else {
            "progress": {"lo": 0x0070},
            "y": 0x0084,
            "lives": 0x0670,
            "level_key": [],
        },
        "ram_mapping": ram_mapping or {},
    }


def _full(n: int = 8) -> np.ndarray:
    """A synthetic archive RAM stack with enough per-byte variation that
    the mirror detector (which excludes columns identical across the
    corpus) does not sweep the whole space into `excluded` and make every
    assertion below vacuously true."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, 2048), dtype=np.uint8)


# --------------------------------------------------------------------------
# The defect itself, asserted against the production set.
# --------------------------------------------------------------------------

def test_a_ram_mapping_byte_is_not_in_the_pre_probe_exclusion_set() -> None:
    """A profile carrying the exact quarantined Zelda win-flag address in
    `ram_mapping` must NOT have that byte excluded. Excluding it is
    precisely how an external RAM map steers this project's own discovery
    instrument away from a byte under quarantine."""
    excluded, _, _ = pre_probe_exclusions(
        _profile(ram_mapping={"ganon_defeated": GANON, "song": SONG}), _full())

    assert GANON not in excluded, (
        f"ram_mapping byte 0x{GANON:04X} is in the pre-probe exclusion set — "
        "the `excluded |= mapped` fold this test guards against has "
        "regressed, and the quarantined win flag can no longer be "
        "rediscovered by this project's own instrument.")
    assert SONG not in excluded


def test_a_ram_mapping_byte_still_reaches_the_candidate_loop() -> None:
    """The gate in `main()` is `if b in excluded: continue`. Run that exact
    predicate against the production exclusion set."""
    excluded, _, _ = pre_probe_exclusions(
        _profile(ram_mapping={"ganon_defeated": GANON}), _full())

    def reaches_candidate_loop(byte: int) -> bool:
        return byte not in excluded

    assert reaches_candidate_loop(GANON), (
        "a ram_mapping-only byte must still reach the candidate-predicate "
        "loop — it is annotation, not an instrument blind spot.")


def test_the_caller_never_receives_the_mapping_set() -> None:
    """The structural half of the fix, and the reason a future
    `excluded |= mapped` cannot be written as a one-liner: `main()` gets a
    PREDICATE for receipt tagging, never a set it could union back in.
    Re-introducing the defect now means re-plumbing a return value."""
    _, is_known, _ = pre_probe_exclusions(
        _profile(ram_mapping={"ganon_defeated": GANON}), _full())

    assert callable(is_known), (
        "pre_probe_exclusions must hand back a membership PREDICATE, not a "
        "set — a set is one `|=` away from being an exclusion again.")
    assert not isinstance(is_known, (set, frozenset, dict, list, tuple)), (
        "a container is unionable; that is the whole defect. Keep this a "
        "callable.")
    # It still answers the question the receipt needs answered.
    assert is_known(GANON) is True
    assert is_known(0x0070) is True      # coordinate bytes are known too
    assert is_known(0x0123) is False


def test_the_receipt_records_ram_mapping_as_non_excluding() -> None:
    """Tag, don't hide: a reader of the receipt must be able to tell
    "documented, but still probed" apart from an actual exclusion."""
    _, _, excl_log = pre_probe_exclusions(
        _profile(ram_mapping={"ganon_defeated": GANON}), _full())
    entries = [e for e in excl_log if e["region"] == "profile ram_mapping bytes"]
    assert entries, "ram_mapping bytes must still be logged in the receipt"
    assert entries[0]["excludes"] is False
    assert f"0x{GANON:04X}" in entries[0]["bytes"]


# --------------------------------------------------------------------------
# GUARD-OF-THE-GUARD. These pass before AND after the fix on purpose: they
# prove the exclusion mechanism still works, so "0x0672 is not excluded" is
# a real negative from a live instrument rather than the reading of a
# mechanism that has stopped excluding anything at all.
# --------------------------------------------------------------------------

def test_coordinate_bytes_are_still_excluded() -> None:
    excluded, _, _ = pre_probe_exclusions(_profile(), _full())
    for addr in (0x0070, 0x0084, 0x0670):
        assert addr in excluded, (
            f"0x{addr:04X} is a cell-key observable — it IS the search key "
            "and must stay excluded. If this fails the fix has deleted a "
            "working mechanism instead of correcting it.")


def test_the_stack_page_is_still_excluded() -> None:
    excluded, _, _ = pre_probe_exclusions(_profile(), _full())
    assert STACK_LO in excluded and (STACK_LO + 0xFF) in excluded


def test_level_key_bytes_are_still_excluded() -> None:
    excluded, _, _ = pre_probe_exclusions(
        _profile(solve={"progress": {"lo": 0x0070}, "y": 0x0084,
                        "lives": 0x0670, "level_key": [0x075F, 0x0760]}),
        _full())
    assert 0x075F in excluded and 0x0760 in excluded
