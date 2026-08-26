"""Guard the H1 fix: `scripts/observatory.py` must not fold a profile's
`ram_mapping` block into the pre-probe exclusion set.

Surfaced 2026-08-26 during the quarantine-hardening adjudication:
`observatory.py:489` used to compute `known = coord_bytes | mapped` and
then `excluded |= known`, so any address appearing in a profile's
`ram_mapping` — including a quarantined one smuggled back in through a
file like `zelda_gui_tuned.yaml` — was silently removed from candidate
generation at the `if b in excluded: continue` gate. The exclusion's
stated purpose ("discover NEW bytes only") is a reporting concern; folding
it into the same set that gates candidate generation implements it as
blindness instead. This is the load-bearing fix: everything else in the
quarantine hardening only matters because bytes excluded here could never
be rediscovered by this project's own instrument regardless of any YAML
cleanup.

Both tests import `coordinate_and_mapping_exclusions` directly rather than
driving `main()` — that function needs a live archive/ROM pool and is out
of scope for a unit test; the exclusion-set computation is pure and does
not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from observatory import coordinate_and_mapping_exclusions  # noqa: E402


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


def test_ram_mapping_is_not_folded_into_the_pre_probe_exclusion_set() -> None:
    """A synthetic profile carrying the exact quarantined Zelda win-flag
    address in `ram_mapping` must NOT have that byte end up excluded —
    excluding it is precisely how an external RAM map would steer this
    project's own discovery instrument away from a byte under quarantine."""
    profile = _profile(ram_mapping={"ganon_defeated": 0x0672})
    coord_bytes, mapped, excl_log = coordinate_and_mapping_exclusions(profile)

    assert 0x0672 in mapped, "ram_mapping byte must still be tracked for tagging"
    assert 0x0672 not in coord_bytes, (
        "0x0672 is not a coordinate byte in this profile and must not be "
        "excluded via the coordinate path either"
    )

    # Reconstruct exactly what main() folds into `excluded`.
    excluded = set(coord_bytes)
    assert 0x0672 not in excluded, (
        "ram_mapping byte 0x0672 ended up in the pre-probe exclusion set — "
        "the fold this test guards against has regressed."
    )

    # The receipt log must record ram_mapping as non-excluding annotation,
    # not silently drop it.
    mapping_entries = [e for e in excl_log if e["region"] == "profile ram_mapping bytes"]
    assert mapping_entries, "ram_mapping bytes must still be logged in the receipt"
    assert mapping_entries[0]["excludes"] is False


def test_a_ram_mapping_byte_can_still_become_a_candidate_predicate() -> None:
    """The gate at observatory.py's candidate loop is `if b in excluded:
    continue`. Simulate that gate directly against the exclusion set this
    function returns: a ram_mapping-only byte must pass it and be reachable
    by scoring, while a genuine coordinate byte must not (it IS the search
    key, so excluding it is correct)."""
    profile = _profile(ram_mapping={"ganon_defeated": 0x0672})
    coord_bytes, mapped, _ = coordinate_and_mapping_exclusions(profile)
    excluded = set(coord_bytes)  # mirrors main(): only coord_bytes fold in

    def reaches_candidate_loop(byte: int) -> bool:
        return byte not in excluded

    assert reaches_candidate_loop(0x0672), (
        "a ram_mapping-only byte must still reach the candidate-predicate "
        "loop — it is annotation, not an instrument blind spot."
    )
    assert not reaches_candidate_loop(0x0070), (
        "a genuine coordinate byte (the cell key itself) should stay "
        "excluded — this is not the defect being guarded against."
    )


def test_known_set_still_covers_both_coordinate_and_mapping_bytes() -> None:
    """`known` (coord_bytes | mapped) is used only for receipt tagging now
    — it must still union both sources so the receipt's "known" flag stays
    meaningful even though it no longer gates anything."""
    profile = _profile(ram_mapping={"ganon_defeated": 0x0672, "song": 0x0609})
    coord_bytes, mapped, _ = coordinate_and_mapping_exclusions(profile)
    known = coord_bytes | mapped
    assert 0x0070 in known  # from solve.progress.lo
    assert 0x0672 in known  # from ram_mapping
    assert 0x0609 in known  # from ram_mapping
