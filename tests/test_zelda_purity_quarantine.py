"""Purity quarantine guard for `configs/zelda.yaml`.

Surfaced 2026-08-25 during an external research audit's review (not by a
Learned-ledger run hitting it): the profile's `ram_mapping:` block was a
disassembly-sourced parallel copy of the same addresses `ZeldaReward`
(nes_core/src/rewards.rs) declares as its own hardcoded constants — the
YAML block was never the reward's source, just an undisclosed second
copy with no `[VERIFIED: ...]` receipts. It now lives in a
`quarantined_external_knowledge:` block that nothing reads. Full note:
CLAIMS.md's Quarantine section.

These tests pin the properties that make the quarantine real rather than
decorative: the block exists and states the rediscovery rule; the
quarantined values cannot be folded into an address map by accident
(strings, not ints); no source file reads the quarantine key; the
external-provenance token stays inside the block, not loose in the rest
of the profile; and the working profile still parses and meets the
trainer boot contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "zelda.yaml"

QUARANTINE_KEY = "quarantined_external_knowledge"

QUARANTINED_ADDRESSES = (
    "0x0070", "0x0084", "0x00EB", "0x00EC", "0x10",
    "0x066F", "0x066D", "0x0658", "0x066E", "0x0671",
    "0x0672", "0x0609",
)


@pytest.fixture(scope="module")
def profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


@pytest.fixture(scope="module")
def raw_text() -> str:
    return PROFILE.read_text()


def test_profile_parses_and_declares_the_quarantine_block(profile):
    assert QUARANTINE_KEY in profile
    q = profile[QUARANTINE_KEY]
    assert q["status"] == "UNVERIFIED_EXTERNAL"
    assert "aldonunez" in q["provenance"].lower() or "disassembly" in q["provenance"].lower()


def test_rediscovery_rule_states_the_re_derivation_requirement(profile):
    rule = profile[QUARANTINE_KEY]["rediscovery_rule"]
    assert "re-derived" in rule or "re-derive" in rule
    assert "receipt" in rule


def test_quarantined_values_cannot_be_folded_into_an_address_map(profile):
    q = profile[QUARANTINE_KEY]
    addr_keys = [k for k in q if k.startswith("q_")]
    assert len(addr_keys) >= len(QUARANTINED_ADDRESSES)
    for k in addr_keys:
        assert isinstance(q[k], str), (
            f"{k} is not a string — a tool folding this block the way "
            f"ram_mapping folds (int(v) over .values()) would silently "
            f"consume a quarantined address instead of raising"
        )
        with pytest.raises((ValueError, TypeError)):
            int(q[k])


def test_no_ram_mapping_block_remains(profile):
    # The old top-level `ram_mapping:` key must be gone entirely, not
    # merely renamed alongside a surviving duplicate.
    assert "ram_mapping" not in profile


def test_no_source_file_reads_the_quarantine_key():
    # Excludes this test and the sibling Metroid quarantine test, both of
    # which legitimately name the shared "quarantined_external_knowledge"
    # convention in their own source — not a read of any profile's block.
    excluded = {
        Path(__file__),
        REPO / "tests" / "test_metroid_purity_quarantine.py",
        REPO / "tests" / "test_onboard_game.py",
    }
    hits = []
    for path in REPO.rglob("*.py"):
        if "/.venv/" in str(path) or "/worktrees/" in str(path):
            continue
        if path in excluded:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if QUARANTINE_KEY in text:
            hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"quarantine key read outside the profile: {hits}"


def test_reward_constants_are_independent_of_the_yaml_block():
    # ZeldaReward's addresses are compiled Rust constants, not read from
    # this YAML at all — confirm the claim the quarantine note rests on
    # by checking the ZeldaReward impl block specifically (the file as a
    # whole legitimately mentions "ram_mapping" in unrelated comments
    # about other games' configs, e.g. Contra's).
    rewards_rs = (REPO / "nes_core" / "src" / "rewards.rs").read_text()
    start = rewards_rs.index("impl ZeldaReward")
    end = rewards_rs.index("\nimpl ", start + 1)
    zelda_impl = rewards_rs[start:end]
    assert "ram_mapping" not in zelda_impl


def test_profile_still_meets_the_trainer_boot_contract(profile):
    assert "start_state_path" in profile
    assert "reward_weights" in profile
    assert "solve" in profile
