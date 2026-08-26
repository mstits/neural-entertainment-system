"""Tree-wide sweep: no `ram_mapping` value in ANY config may equal an
address recorded inside a purity quarantine block anywhere in the tree.

BREACH PATH 2 (quarantine-hardening adjudication, 2026-08-26).
`configs/zelda_gui_tuned.yaml` carried a live, int-valued `ram_mapping:`
block whose 13 entries duplicated `configs/zelda.yaml`'s quarantined
win-chain addresses byte-for-byte (six of them the exact quarantined
values: `dungeon_level`, `current_hearts`, `max_hearts`, `triforce_pieces`,
`ganon_defeated`, `song`). Because the values were plain ints, not the
string-tagged form the quarantine convention uses,
`scripts/observatory.py`'s `int(a) for a in ram_mapping.values()` fold
picked them up and excluded them from the discovery instrument's own
candidate space — an external RAM map steering rediscovery away from
precisely the bytes the quarantine exists to make rediscoverable, the
exact inverse of the mechanism's purpose.

`tests/test_zelda_purity_quarantine.py` guards the Zelda family
specifically. This module is the general form the adjudication asked
for: it does not know which game a quarantine block belongs to, and does
not care — it derives the full set of quarantined addresses by parsing
every `quarantined_external_knowledge:` block already in the tree (so
the guard cannot drift from the record it guards, and a THIRD game's
future quarantine is covered automatically), then checks every config's
`ram_mapping:` values against that set.

Every check here is mutation-tested: `test_the_sweep_rejects_a_synthetic_
contaminated_profile` feeds the checker a value it must catch, so a
predicate that rots into matching nothing cannot pass silently — the
same discipline the metroid quarantine test file already applies to its
own lints.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"

QUARANTINE_KEY = "quarantined_external_knowledge"


def _load(path: Path) -> dict:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _extract_addr_leaves(node) -> set[int]:
    """Pull addresses out of a quarantine block's own structure.

    Only STRING leaves whose stripped content starts with `0x` count —
    that is the shape every quarantine entry actually uses (`"0x0672"`,
    or the annotated `"0x6879 UNVERIFIED-EXTERNAL — claimed live missile
    count"` form). This deliberately does NOT regex-search arbitrary
    prose for hex-looking substrings: `configs/metroid.yaml`'s own
    quarantine block explains, in prose, that `0x0107` is legitimately
    VERIFIED and reachable (the opposite of quarantined) — a naive
    "any 0x token anywhere in the block" scan would misfire on exactly
    that sentence and flag metroid.yaml's own correct `ram_mapping`
    entry for it.
    """
    found: set[int] = set()
    if isinstance(node, dict):
        for v in node.values():
            found |= _extract_addr_leaves(v)
    elif isinstance(node, list):
        for v in node:
            found |= _extract_addr_leaves(v)
    elif isinstance(node, str):
        m = re.match(r"^0x([0-9A-Fa-f]+)\b", node.strip())
        if m:
            found.add(int(m.group(1), 16))
    return found


def _quarantined_addresses() -> set[int]:
    """Union of every address named inside a `quarantined_external_
    knowledge` block anywhere in `configs/`. Sourced from the blocks
    themselves (not a hand-maintained duplicate list) so this guard
    cannot drift from what it guards."""
    addrs: set[int] = set()
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        doc = _load(path)
        block = doc.get(QUARANTINE_KEY)
        if block:
            addrs |= _extract_addr_leaves(block)
    return addrs


QUARANTINED_ADDRESSES_INT = _quarantined_addresses()


def _ram_mapping_hits(doc: dict, quarantined: set[int]) -> list[str]:
    hits = []
    for k, v in (doc.get("ram_mapping") or {}).items():
        try:
            a = int(v)
        except (TypeError, ValueError):
            continue  # the quarantine's own string-tagged form: correct by construction
        if a in quarantined:
            hits.append(f"{k}=0x{a:04X}")
    return hits


def _top_level_profiles() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def test_quarantined_address_set_is_non_empty() -> None:
    # A checker whose source set is empty would pass every profile
    # vacuously. Guard the guard.
    assert QUARANTINED_ADDRESSES_INT, (
        "no quarantined addresses found anywhere under configs/ — either "
        "every quarantine block was deleted, or the parser above stopped "
        "matching the shape those blocks actually use."
    )
    # Sanity: the specific Zelda win-chain address this sweep exists to
    # catch must be in the derived set.
    assert 0x0672 in QUARANTINED_ADDRESSES_INT


@pytest.mark.parametrize("path", _top_level_profiles(), ids=lambda p: p.name)
def test_no_ram_mapping_value_is_a_quarantined_address(path: Path) -> None:
    doc = _load(path)
    hits = _ram_mapping_hits(doc, QUARANTINED_ADDRESSES_INT)
    assert not hits, (
        f"{path.name}: ram_mapping value(s) {hits} equal an address "
        f"recorded in a quarantine block elsewhere in the tree. A "
        f"quarantined byte may only re-enter as a NEW ram_mapping entry "
        f"after independent re-derivation on this ROM, never by copying "
        f"the same address into another profile's map."
    )


def test_recursive_sweep_over_every_yaml_under_configs() -> None:
    """Belt-and-braces: the parametrized test above only covers the
    top-level profiles (the ones `test_profile_configs.py` treats as
    standalone). This walks the WHOLE `configs/` tree — `auto/`,
    `overrides/`, everything — as one aggregated check, so a violation
    tucked into a generated stub or a partial overlay is not exempt just
    because it is not a top-level profile."""
    violations = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        hits = _ram_mapping_hits(_load(path), QUARANTINED_ADDRESSES_INT)
        if hits:
            violations[str(path.relative_to(REPO))] = hits
    assert not violations, (
        f"quarantined address(es) found in ram_mapping outside the "
        f"top-level sweep: {violations}"
    )


def test_the_sweep_rejects_a_synthetic_contaminated_profile() -> None:
    """Mutation control: feed the checker an in-memory profile carrying a
    known-quarantined address and confirm it is flagged. Guards against
    the extraction/comparison logic rotting into matching nothing."""
    addr = sorted(QUARANTINED_ADDRESSES_INT)[0]
    synthetic = {"ram_mapping": {"synthetic": addr}}
    hits = _ram_mapping_hits(synthetic, QUARANTINED_ADDRESSES_INT)
    assert hits, "the sweep failed to flag a synthetically contaminated profile"

    # ...and the string-tagged quarantine form must NOT be flagged (that
    # is the whole point of the string convention: it is correct by
    # construction, not something this sweep should ever touch).
    quarantine_shaped = {"ram_mapping": {"synthetic": f"0x{addr:04X}"}}
    assert not _ram_mapping_hits(quarantine_shaped, QUARANTINED_ADDRESSES_INT), (
        "a string-valued ram_mapping entry was flagged — int(v) should "
        "have raised and been skipped, the same way it does for a real "
        "quarantine block."
    )


def test_extraction_ignores_addresses_named_only_in_quarantine_prose() -> None:
    """Regression guard for the exact false positive this sweep almost
    shipped with: `configs/metroid.yaml`'s quarantine block explains, in
    prose, that `0x0107` is VERIFIED and reachable — the opposite of
    quarantined. A naive "any 0x token in the block" scan would have
    pulled that address in and then flagged metroid.yaml's own correct
    `samus_health_hi: 0x0107` ram_mapping entry."""
    block = {
        "q_real_address": "0x0672",
        "q_prose_mentioning_an_address": (
            'Claimed "upper nibble of 0x0107 = filled energy-tank count". '
            "0x0107 IS reachable and independently verified."
        ),
    }
    found = _extract_addr_leaves(block)
    assert found == {0x0672}, (
        f"prose mentioning an address leaked into the quarantined set: {found}"
    )
