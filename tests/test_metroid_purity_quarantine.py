"""Purity quarantine guard for `configs/metroid.yaml`.

The profile carried external RAM-map material: the cartridge item /
missile / energy-tank addresses with their claimed equipment bitfield,
sitting unlabelled inside `ram_mapping` where they read as our own
measurement, plus a `solve:` comment that justified the progress pairing
by appealing to an outside prescription rather than to our own probe.
The purity line bans knowledge we did not derive ourselves from any path
that solves or scores the game, so that material now lives in a
`quarantined_external_knowledge:` block that nothing reads.

These tests pin the four properties that make the quarantine real rather
than decorative:

  1. the block exists and still states the rediscovery rule;
  2. no external-provenance token and no quarantined cartridge address
     appears anywhere outside it;
  3. nothing in it can be folded into an address map by accident (the
     values are strings, so `int(v)` raises where `ram_mapping` folds
     cleanly), and no source file reads the key at all;
  4. the working profile is untouched — same solve addresses, same
     boot contract, and the in-tree persistence-probe note is intact.

Every lint here is mutation-checked: a companion test feeds the checker
a deliberately contaminated copy and asserts it fails, so a regex that
rots into matching nothing cannot pass silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "metroid.yaml"

QUARANTINE_KEY = "quarantined_external_knowledge"

# Phrases that mark a claim as sourced from an external RAM map / route
# lore rather than from our own differential protocol. They are allowed
# INSIDE the quarantine block (that is the block's whole job: naming the
# contamination) and nowhere else in the profile.
EXTERNAL_PROVENANCE_TOKENS = (
    "datacrystal",
    "data crystal",
    "lore-prescribed",
    "lore prescribes",
    "lore prescribed",
)

# nes_core.get_ram() masks to $07FF, so anything at or above this is
# cartridge PRG-RAM: unreachable, and in this profile also unverified.
SYSTEM_RAM_TOP = 0x07FF
PRG_RAM_BASE = 0x6000


# ---------------------------------------------------------------------------
# helpers — pure text/dict operations, exercised against real and mutated input
# ---------------------------------------------------------------------------

def _quarantine_span(text: str) -> tuple[int, int]:
    """(start, end) line indices of the quarantine block *including* the
    column-0 comment banner directly above it.

    A top-level YAML block runs from its `key:` line to the first later
    line at column 0; the contiguous run of column-0 comments above the
    key is the banner that labels it, and a reader treats banner+block
    as one unit, so the lint does too. Raises if the key is absent —
    "the block vanished" must fail loudly, not vacuously pass.
    """
    lines = text.split("\n")
    start = next(
        (i for i, ln in enumerate(lines)
         if re.match(rf"^{re.escape(QUARANTINE_KEY)}\s*:", ln)),
        None,
    )
    if start is None:
        raise AssertionError(
            f"{PROFILE.name}: top-level `{QUARANTINE_KEY}:` block is gone — "
            f"the quarantined external RAM-map material has no home."
        )
    banner = start
    while banner > 0 and lines[banner - 1].startswith("#"):
        banner -= 1
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and not lines[j][:1].isspace():
            end = j
            break
    return banner, end


def _outside_quarantine(text: str) -> str:
    """Everything in the profile that is NOT the quarantine block."""
    lines = text.split("\n")
    lo, hi = _quarantine_span(text)
    return "\n".join(lines[:lo] + lines[hi:])


def _quarantined_prg_addresses(doc: dict) -> set[int]:
    """Every cartridge PRG-RAM address named inside the quarantine block.

    Scraped from the block's own text so the lint tracks the quarantine
    instead of a hand-maintained duplicate list.
    """
    blob = yaml.safe_dump(doc.get(QUARANTINE_KEY, {}), allow_unicode=True)
    found = {int(m, 16) for m in re.findall(r"0x([0-9A-Fa-f]{4})", blob)}
    return {a for a in found if a >= PRG_RAM_BASE}


def _provenance_leaks(text: str) -> list[str]:
    """Tokens from EXTERNAL_PROVENANCE_TOKENS living outside quarantine."""
    body = _outside_quarantine(text).lower()
    return [tok for tok in EXTERNAL_PROVENANCE_TOKENS if tok in body]


def _address_leaks(text: str, doc: dict) -> list[str]:
    """Quarantined cartridge addresses referenced outside quarantine."""
    body = _outside_quarantine(text).lower()
    leaks = []
    for addr in sorted(_quarantined_prg_addresses(doc)):
        for form in (f"0x{addr:04x}", f"${addr:04x}"):
            if form in body:
                leaks.append(form)
    return leaks


def _profile_text() -> str:
    return PROFILE.read_text()


def _profile_doc() -> dict:
    return yaml.safe_load(_profile_text())


# ---------------------------------------------------------------------------
# 1. the quarantine exists and still says what it is for
# ---------------------------------------------------------------------------

def test_profile_parses_and_declares_the_quarantine_block():
    doc = _profile_doc()
    assert isinstance(doc, dict), "metroid.yaml root is not a mapping"
    block = doc.get(QUARANTINE_KEY)
    assert isinstance(block, dict), (
        f"`{QUARANTINE_KEY}:` must be a mapping; the external item bytes "
        f"have nowhere quarantined to live otherwise."
    )
    for field in ("provenance", "status", "rediscovery_rule"):
        assert block.get(field), f"quarantine block is missing `{field}`"
    assert "UNVERIFIED_EXTERNAL" in block["status"], (
        "the quarantine status must name the material UNVERIFIED_EXTERNAL "
        "so a grep for contaminated profiles finds this one."
    )


def test_rediscovery_rule_states_the_re_derivation_requirement():
    """The rule is the load-bearing part: a byte in here is dead until our
    own differential protocol re-derives it on this ROM, with a receipt.
    A future edit may reword it but may not hollow it out."""
    rule = _profile_doc()[QUARANTINE_KEY]["rediscovery_rule"].lower()
    for phrase in ("re-derived", "differential", "receipt", "solve:"):
        assert phrase in rule, (
            f"rediscovery_rule no longer mentions {phrase!r} — the "
            f"quarantine has been reduced to a label without a rule."
        )


# ---------------------------------------------------------------------------
# 2. nothing contaminated leaks outside the block
# ---------------------------------------------------------------------------

def test_no_external_provenance_token_outside_the_quarantine():
    leaks = _provenance_leaks(_profile_text())
    assert not leaks, (
        f"{PROFILE.name}: external-provenance token(s) {leaks} appear "
        f"outside `{QUARANTINE_KEY}:`. External RAM-map sourcing may be "
        f"named inside the quarantine and nowhere else."
    )


def test_provenance_lint_has_teeth():
    """Mutation check: drop a token into the working part of the profile
    and the lint must catch it. Guards against a span helper that
    accidentally swallows the whole file."""
    text = _profile_text()
    # Anchor on a top-level key the quarantine block never mentions, so
    # the mutation lands in the working part of the file.
    assert text.count("\naction_space:") == 1
    contaminated = text.replace(
        "\naction_space:", "\n# per the datacrystal RAM map\naction_space:", 1
    )
    assert "datacrystal" in _provenance_leaks(contaminated), (
        "the provenance lint passed a deliberately contaminated profile — "
        "it is not actually inspecting the working blocks."
    )
    # ...and the same token INSIDE the quarantine stays legal.
    assert "datacrystal" in text.lower(), "fixture drifted: expected the " \
        "quarantine to name its own source"
    assert not _provenance_leaks(text)


def test_no_quarantined_cartridge_address_outside_the_quarantine():
    text, doc = _profile_text(), _profile_doc()
    addrs = _quarantined_prg_addresses(doc)
    assert addrs, (
        "no cartridge PRG-RAM address found inside the quarantine — the "
        "item/missile bytes were removed rather than quarantined, so the "
        "record of what was contaminated is lost."
    )
    leaks = _address_leaks(text, doc)
    assert not leaks, (
        f"{PROFILE.name}: quarantined cartridge address(es) {leaks} are "
        f"still referenced outside `{QUARANTINE_KEY}:`."
    )


def test_address_lint_has_teeth():
    """Mutation check: paste a quarantined address back into a working
    comment; the lint must flag it. This is the exact failure mode the
    quarantine exists to stop."""
    text, doc = _profile_text(), _profile_doc()
    addr = sorted(_quarantined_prg_addresses(doc))[0]
    contaminated = text.replace(
        "\naction_space:",
        f"\n# repoint the reward at 0x{addr:04X}\naction_space:", 1,
    )
    assert _address_leaks(contaminated, doc), (
        "the address lint passed a profile that pastes a quarantined "
        "cartridge byte back into a working block."
    )


def test_item_bitfield_semantics_left_the_ram_mapping_block():
    """The equipment-bitfield bit assignment was the clearest piece of
    external knowledge in the file. It may be described inside the
    quarantine; it must not sit in ram_mapping looking like ours."""
    body = _outside_quarantine(_profile_text()).lower()
    for claim in ("bit4=", "screwattack", "screw attack", "long beam",
                  "longbeam", "maru mari"):
        assert claim not in body, (
            f"item-bitfield semantics {claim!r} still present outside the "
            f"quarantine block."
        )


# ---------------------------------------------------------------------------
# 3. the block cannot be consumed, by accident or by grep
# ---------------------------------------------------------------------------

def test_quarantined_values_cannot_be_folded_into_an_address_map():
    """scripts/observatory.py folds `ram_mapping` with `int(a)` over its
    values. Anything shaped like that dict is one copy-paste away from
    being consumed the same way, so the quarantined values are strings:
    the fold raises instead of silently yielding banned addresses."""
    block = _profile_doc()[QUARANTINE_KEY]
    byte_block = block["q_unverified_prg_ram_bytes"]
    assert byte_block, "the quarantined byte list is empty"
    for name, val in byte_block.items():
        assert isinstance(val, str), (
            f"{name} is a bare {type(val).__name__} — an address-map fold "
            f"would consume it silently. Keep quarantined values as "
            f"tagged strings."
        )
    with pytest.raises((TypeError, ValueError)):
        {int(v) for v in byte_block.values()}


def test_ram_mapping_still_folds_to_ints_for_the_discovery_tools():
    """The flip side: quarantining must not break the real address map.
    Mirrors observatory's exclusion fold and verify_ram_map's iteration."""
    mapping = _profile_doc()["ram_mapping"]
    folded = {int(a) for a in mapping.values()}
    assert folded, "ram_mapping folded to nothing"
    assert all(a <= SYSTEM_RAM_TOP for a in folded), (
        f"ram_mapping now contains an address above the get_ram mask: "
        f"{sorted(a for a in folded if a > SYSTEM_RAM_TOP)}"
    )


def test_no_source_file_reads_the_quarantine_key():
    """'Solve paths cannot consume it' is only true if nothing reads it.
    Scan the real source trees for the key name (this test file and the
    profile itself are the only legitimate mentions)."""
    readers = []
    for root in ("src", "scripts", "nes_core/src"):
        for path in (REPO / root).rglob("*.py"):
            if QUARANTINE_KEY in path.read_text(errors="ignore"):
                readers.append(str(path.relative_to(REPO)))
        for path in (REPO / root).rglob("*.rs"):
            if QUARANTINE_KEY in path.read_text(errors="ignore"):
                readers.append(str(path.relative_to(REPO)))
    assert not readers, (
        f"{QUARANTINE_KEY} is read by {readers} — quarantined external "
        f"knowledge has become a consumed input."
    )


def test_quarantine_key_is_not_a_registered_profile_key():
    """It is deliberately absent from the schema registry: the validator
    flags it as 'not consumed by the trainer', which is precisely the
    status we want it to advertise at startup."""
    from src.training.config_schema import KNOWN_TOP_KEYS, validate_profile

    assert QUARANTINE_KEY not in KNOWN_TOP_KEYS
    warnings = validate_profile(_profile_doc())
    assert any(QUARANTINE_KEY in w and "silently ignored" in w
               for w in warnings), (
        "the schema validator no longer reports the quarantine block as "
        "unconsumed; something registered it as a live profile key."
    )


# ---------------------------------------------------------------------------
# 4. the working profile is unchanged
# ---------------------------------------------------------------------------

def test_solve_block_consumes_only_verified_system_ram():
    solve = _profile_doc()["solve"]
    banned = _quarantined_prg_addresses(_profile_doc())
    addrs = {"y": solve["y"], "area": solve["area"], "lives": solve["lives"],
             "progress.lo": solve["progress"]["lo"],
             "progress.hi": solve["progress"]["hi"]}
    for label, addr in addrs.items():
        assert isinstance(addr, int), f"solve.{label} is not an int address"
        assert addr <= SYSTEM_RAM_TOP, (
            f"solve.{label} = {addr:#06x} is outside the 2 KB system RAM "
            f"get_ram exposes — it cannot have been observed."
        )
        assert addr not in banned, (
            f"solve.{label} = {addr:#06x} is a quarantined external byte."
        )


def test_solve_addresses_are_the_receipted_ones():
    """Pin the values: the quarantine is a documentation change and must
    stay byte-identical in behaviour. These are the addresses in
    runs/metroid/metroid_receipt.json step2."""
    solve = _profile_doc()["solve"]
    assert solve["progress"] == {"lo": 0x0051, "hi": 0x0050}
    assert solve["y"] == 0x0052
    assert solve["area"] == 0x0074
    assert solve["lives"] == 0x0107
    assert solve["level_key"] == []
    assert solve["progress_cap"] == 16384


def test_persistence_probe_rejection_note_is_preserved():
    """The 0x0008|0x0065 rejection is the profile's record of why the
    higher-range pair was refused: 0x0065 ramps then collapses, so its
    range is transient, not world position. It is the kind of hard-won
    negative that a tidy-up deletes; pin it."""
    text = _profile_text()
    for phrase in ("0x0008|0x0065", "NON-DURABLE",
                   "persistence half of the saturation test"):
        assert phrase in text, (
            f"the saturation/persistence rejection note lost {phrase!r}."
        )


def test_profile_still_meets_the_trainer_boot_contract():
    """Same check test_profile_configs applies to every shipped profile:
    a non-empty action_space that folds to NES bitmasks."""
    from src.training.profile_utils import action_space_to_bitmasks

    action_space = _profile_doc().get("action_space")
    assert action_space, "metroid.yaml lost its action_space"
    action_space_to_bitmasks(action_space)
