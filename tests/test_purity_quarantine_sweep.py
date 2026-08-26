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


def _rom_key(value) -> str:
    """Normalised identity of a ROM file: its basename, lowercased.
    Profiles cite the same ROM through `rom_path` and `solve.rom`, both
    repo-relative, so the basename is the stable join key."""
    return Path(str(value)).name.strip().lower()


def _profile_rom_keys(doc: dict) -> set[str]:
    keys = set()
    for v in (doc.get("rom_path"), (doc.get("solve") or {}).get("rom")
              if isinstance(doc.get("solve"), dict) else None):
        if v:
            keys.add(_rom_key(v))
    return keys


def _quarantined_addresses() -> tuple[set[int], dict[str, set[int]]]:
    """Every address named inside a `quarantined_external_knowledge`
    block anywhere in `configs/`, sourced from the blocks themselves (not
    a hand-maintained duplicate list) so this guard cannot drift from
    what it guards.

    Returns `(global_union, by_rom)`. RAM addresses are numbered PER ROM:
    Zelda's quarantined `q_dungeon_level: 0x10` says nothing whatever
    about Arkanoid's `solve.progress.hi: 0x0010`, and flagging that
    collision would be a false positive that trains readers to ignore
    this sweep. So each block declares `applies_to_rom:` and is checked
    only against profiles running that ROM. A block that omits the key
    lands in the global union and is checked against EVERYTHING — the
    fail-closed direction, so forgetting the key over-reports rather
    than under-reports.
    """
    by_rom: dict[str, set[int]] = {}
    unscoped: set[int] = set()
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        doc = _load(path)
        block = doc.get(QUARANTINE_KEY)
        if not block:
            continue
        addrs = _extract_addr_leaves(block)
        rom = block.get("applies_to_rom") if isinstance(block, dict) else None
        if rom:
            by_rom.setdefault(_rom_key(rom), set()).update(addrs)
        else:
            unscoped |= addrs
    if unscoped:
        for k in by_rom:
            by_rom[k] |= unscoped
    global_union = set(unscoped)
    for v in by_rom.values():
        global_union |= v
    return global_union, by_rom


QUARANTINED_ADDRESSES_INT, QUARANTINED_BY_ROM = _quarantined_addresses()
#: Addresses from blocks that named no ROM: they apply to every profile.
UNSCOPED_QUARANTINED = set(QUARANTINED_ADDRESSES_INT)
for _s in QUARANTINED_BY_ROM.values():
    UNSCOPED_QUARANTINED -= _s


def quarantined_for(doc: dict) -> set[int]:
    """The quarantined-address set that applies to ONE profile: the
    blocks scoped to its own ROM, plus every unscoped block."""
    addrs = set(UNSCOPED_QUARANTINED)
    for key in _profile_rom_keys(doc):
        addrs |= QUARANTINED_BY_ROM.get(key, set())
    if not _profile_rom_keys(doc):
        # No ROM declared (override-only overlays, synthetic test dicts):
        # fall back to the whole union rather than to nothing.
        return set(QUARANTINED_ADDRESSES_INT)
    return addrs


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
    hits = _ram_mapping_hits(doc, quarantined_for(doc))
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
        doc = _load(path)
        hits = _ram_mapping_hits(doc, quarantined_for(doc))
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


# ==========================================================================
# THE SECOND DOOR INTO THE EXCLUSION SET: solve-block coordinates.
#
# Closing the `ram_mapping` fold (H1) left one path by which a quarantined
# address still reaches `scripts/observatory.py`'s pre-probe exclusion set:
# `pre_probe_exclusions()` legitimately excludes the profile's own solve
# coordinates (progress.lo/hi, y, lives, level_key) because they ARE the
# archive's cell key. A quarantined byte copied into a solve block is
# therefore excluded from candidate generation exactly as it was before the
# fix, by a different door — and the sweep above, which reads only
# `ram_mapping`, never looks there.
#
# Verified 2026-08-26: `configs/legend_of_zelda.yaml` declares
# `solve.progress.lo: 0x0070` and `solve.lives: 0x0609`, both recorded as
# quarantined in `configs/zelda_gui_tuned.yaml` (q_link_x, q_song).
#
# Blanket-banning them is wrong — independent rediscovery is the quarantine's
# documented EXIT, and both of those were genuinely re-derived by measurement.
# What the mechanism cannot do is tell rediscovery from leakage by looking at
# a number. So the rule is: a quarantined address may appear in a solve block
# only alongside a machine-readable `rediscovered_addresses:` entry naming the
# role, the method, and a receipt path that EXISTS IN THE TREE. A prose
# comment is not enough; a receipt under runs/ is not enough either, since
# runs/ is gitignored and would vanish on a fresh checkout.
# ==========================================================================

#: Exactly the fields `scripts/observatory.py::_coordinate_bytes` reads —
#: the guard follows the surface the contamination travels on, not a
#: naming convention. If that function grows a field, this must too.
def _solve_coordinate_addresses(doc: dict) -> dict[int, list[str]]:
    solve = doc.get("solve") or {}
    if not isinstance(solve, dict):
        return {}
    out: dict[int, list[str]] = {}

    def add(value, role: str) -> None:
        try:
            a = int(value)
        except (TypeError, ValueError):
            return
        if a >= 0:
            out.setdefault(a, []).append(role)

    progress = solve.get("progress") or {}
    if isinstance(progress, dict):
        add(progress.get("lo"), "solve.progress.lo")
        add(progress.get("hi"), "solve.progress.hi")
    add(solve.get("y"), "solve.y")
    add(solve.get("lives"), "solve.lives")
    # `area` is not in observatory's exclusion set today, but it IS part of
    # the archive cell key (go_explore_solve.py:2703 keys on
    # `level_key + (area,) + ...`), so a quarantined byte landing here is a
    # live search observable taken from an outside map. Same rule.
    add(solve.get("area"), "solve.area")
    for i, a in enumerate(solve.get("level_key") or []):
        add(a, f"solve.level_key[{i}]")
    return out


def _rediscovery_declarations(doc: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for entry in doc.get("rediscovered_addresses") or []:
        if not isinstance(entry, dict):
            continue
        try:
            a = int(str(entry.get("address")), 0)
        except (TypeError, ValueError):
            continue
        out[a] = entry
    return out


def solve_block_violations(doc: dict, quarantined: set[int]) -> list[str]:
    """Quarantined addresses in this profile's solve block that are not
    covered by a complete, receipted rediscovery declaration."""
    declared = _rediscovery_declarations(doc)
    problems: list[str] = []
    for addr, roles in sorted(_solve_coordinate_addresses(doc).items()):
        if addr not in quarantined:
            continue
        entry = declared.get(addr)
        where = "/".join(roles)
        if entry is None:
            problems.append(
                f"0x{addr:04X} ({where}) is quarantined and has no "
                f"`rediscovered_addresses` entry")
            continue
        if not str(entry.get("method", "")).strip():
            problems.append(f"0x{addr:04X} ({where}) declares no method")
        receipt = str(entry.get("receipt", "")).strip()
        if not receipt:
            problems.append(f"0x{addr:04X} ({where}) declares no receipt")
        elif not (REPO / receipt).exists():
            problems.append(
                f"0x{addr:04X} ({where}) cites receipt {receipt!r}, which "
                f"does not exist in the tree")
        elif receipt.startswith("runs/"):
            problems.append(
                f"0x{addr:04X} ({where}) cites {receipt!r} under runs/, "
                f"which is gitignored — a rediscovery receipt must be "
                f"checked in or the claim is unauditable on a fresh clone")
    return problems


@pytest.mark.parametrize("path", _top_level_profiles(), ids=lambda p: p.name)
def test_quarantined_solve_addresses_carry_a_receipted_rediscovery(
        path: Path) -> None:
    doc = _load(path)
    problems = solve_block_violations(doc, quarantined_for(doc))
    assert not problems, (
        f"{path.name}: {problems}. A solve-block coordinate becomes a "
        f"pre-probe exclusion in scripts/observatory.py, so a quarantined "
        f"address here removes itself from the instrument's own candidate "
        f"space. Declare the independent re-derivation under "
        f"`rediscovered_addresses:` with a checked-in receipt, or stop "
        f"using the address."
    )


def test_the_solve_sweep_rejects_an_undeclared_quarantined_coordinate() -> None:
    """Mutation control. Without this, a checker that stopped extracting
    solve addresses would pass the whole roster silently."""
    addr = sorted(QUARANTINED_ADDRESSES_INT)[0]
    bare = {"solve": {"progress": {"lo": addr}, "y": 1, "lives": 2}}
    assert solve_block_violations(bare, QUARANTINED_ADDRESSES_INT), (
        "an undeclared quarantined solve coordinate was not flagged")

    lives_only = {"solve": {"lives": addr}}
    assert solve_block_violations(lives_only, QUARANTINED_ADDRESSES_INT)

    level_key = {"solve": {"level_key": [addr]}}
    assert solve_block_violations(level_key, QUARANTINED_ADDRESSES_INT)


def test_a_receipted_rediscovery_is_accepted_and_a_missing_receipt_is_not() -> None:
    """Both directions, so the rule is neither a blanket ban (which would
    make the quarantine inescapable) nor a rubber stamp (which would make
    it decorative)."""
    addr = sorted(QUARANTINED_ADDRESSES_INT)[0]
    good = {
        "solve": {"progress": {"lo": addr}},
        "rediscovered_addresses": [{
            "address": f"0x{addr:04X}",
            "role": "solve.progress.lo",
            "method": "measured live: rises under forward, flat under noop",
            # Any checked-in file; this test file itself always exists.
            "receipt": "tests/test_purity_quarantine_sweep.py",
        }],
    }
    assert not solve_block_violations(good, QUARANTINED_ADDRESSES_INT)

    missing = {**good, "rediscovered_addresses": [
        {**good["rediscovered_addresses"][0],
         "receipt": "docs/receipts/rediscovery/does_not_exist.json"}]}
    assert solve_block_violations(missing, QUARANTINED_ADDRESSES_INT), (
        "a rediscovery citing a nonexistent receipt was accepted — the "
        "declaration would then be a line anyone can add, which is the "
        "vacuous-allowlist pattern this rule exists to avoid")

    gitignored = {**good, "rediscovered_addresses": [
        {**good["rediscovered_addresses"][0],
         "receipt": "runs/onboard_wave4/discover_legend_of_zelda_right.json"}]}
    assert solve_block_violations(gitignored, QUARANTINED_ADDRESSES_INT)

    no_method = {**good, "rediscovered_addresses": [
        {"address": f"0x{addr:04X}", "role": "solve.progress.lo",
         "receipt": "tests/test_purity_quarantine_sweep.py"}]}
    assert solve_block_violations(no_method, QUARANTINED_ADDRESSES_INT)


def test_a_non_quarantined_solve_coordinate_is_never_flagged() -> None:
    """The sweep must not demand receipts for ordinary addresses, or every
    profile on the roster would need one and the signal would be noise."""
    clean = {"solve": {"progress": {"lo": 0x0011}, "y": 0x0012, "lives": 0x0013}}
    assert 0x0011 not in QUARANTINED_ADDRESSES_INT
    assert not solve_block_violations(clean, QUARANTINED_ADDRESSES_INT)


def test_every_quarantine_block_names_the_rom_it_applies_to() -> None:
    """Scoping is what keeps this sweep credible (see `_quarantined_
    addresses`). A block without `applies_to_rom` silently reverts to the
    global, cross-game comparison that would flag Arkanoid's
    `solve.progress.hi: 0x0010` against Zelda's `q_dungeon_level: 0x10` —
    a false positive that teaches readers to ignore the sweep."""
    missing = []
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        block = _load(path).get(QUARANTINE_KEY)
        if block and not (isinstance(block, dict)
                          and block.get("applies_to_rom")):
            missing.append(path.name)
    assert not missing, (
        f"quarantine block(s) with no `applies_to_rom:` key: {missing}")


def test_rom_scoping_actually_narrows_the_comparison() -> None:
    """Guard-of-the-guard for the scoping itself, in both directions.

    Anti-vacuity: if `quarantined_for` degenerated into "everything", the
    Arkanoid assertion below would fail; if it degenerated into "nothing",
    the Zelda assertion would. The specific case is real — Arkanoid's
    `solve.progress.hi` is 0x0010 and Zelda's `q_dungeon_level` is 0x10.
    """
    collision = 0x0010
    assert collision in QUARANTINED_ADDRESSES_INT, (
        "the cross-game collision this test is built on has disappeared "
        "from the quarantine record; pick another or delete this test")

    arkanoid = _load(CONFIG_DIR / "arkanoid.yaml")
    assert arkanoid, "configs/arkanoid.yaml must exist for this test"
    assert collision not in quarantined_for(arkanoid), (
        "a Zelda-scoped quarantined address is being compared against an "
        "Arkanoid profile — RAM addresses do not mean the same thing "
        "across ROMs and this sweep would be reporting noise")

    zelda = _load(CONFIG_DIR / "zelda.yaml")
    assert collision in quarantined_for(zelda), (
        "the Zelda quarantine stopped applying to the Zelda profile — "
        "scoping has narrowed to nothing and the sweep is now vacuous")
    assert 0x0672 in quarantined_for(zelda)


def test_an_unscoped_quarantine_block_would_apply_everywhere() -> None:
    """Fail-closed direction: forgetting `applies_to_rom` must over-report,
    never under-report. Asserted on the mechanism rather than on tree
    state, since the test above keeps the tree free of such blocks."""
    assert UNSCOPED_QUARANTINED == set(), (
        "unscoped quarantined addresses exist in the tree; the test above "
        "should have caught that first")
    # Simulate one: an unscoped address is folded into every ROM's set.
    scoped_before = quarantined_for(_load(CONFIG_DIR / "arkanoid.yaml"))
    assert 0xABCD not in scoped_before
