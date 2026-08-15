"""Guard configs/overrides/online_1_2_escalations.yaml — the inert
next-rung escalation catalog for the 1-2 online campaign.

Three checks, none of which launch an emulator or touch the live
campaign's files:
  - the YAML parses and has the documented top-level shape;
  - each escalation block carries `trigger`, `override`, and `receipt`;
  - deep-merging each block's `override` onto the campaign's base
    profile (configs/mario_1_2_online_v2.yaml, read-only here) produces
    a profile that passes config_schema's STRICT validation — i.e. every
    key the override introduces is a real, registered reinforce knob,
    not a typo that would silently no-op.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.training.config_schema import ConfigSchemaError, check_profile

REPO = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO / "configs" / "overrides" / "online_1_2_escalations.yaml"
BASE_PROFILE_PATH = REPO / "configs" / "mario_1_2_online_v2.yaml"

REQUIRED_BLOCK_KEYS = {"trigger", "override", "receipt"}
EXPECTED_BLOCK_IDS = {"E1_epoch_cut", "E2_entropy_bump", "E3_rung_shift"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto a copy of `base`. Override wins
    on scalars; dict values merge key-by-key so a partial sub-block
    (e.g. reinforce.sil) does not wipe out its siblings."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@pytest.fixture(scope="module")
def catalog() -> dict:
    assert CATALOG_PATH.exists(), f"missing catalog: {CATALOG_PATH}"
    with CATALOG_PATH.open() as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), "catalog must parse to a mapping"
    return doc


@pytest.fixture(scope="module")
def base_profile() -> dict:
    assert BASE_PROFILE_PATH.exists(), f"missing base profile: {BASE_PROFILE_PATH}"
    with BASE_PROFILE_PATH.open() as f:
        profile = yaml.safe_load(f)
    assert isinstance(profile, dict)
    return profile


def test_catalog_yaml_parses(catalog: dict):
    assert "escalations" in catalog, "catalog must have a top-level 'escalations' map"
    assert isinstance(catalog["escalations"], dict)
    assert len(catalog["escalations"]) >= 3


def test_catalog_has_the_three_documented_escalations(catalog: dict):
    ids = set(catalog["escalations"].keys())
    missing = EXPECTED_BLOCK_IDS - ids
    assert not missing, f"catalog is missing expected blocks: {missing}"


@pytest.mark.parametrize("block_id", sorted(EXPECTED_BLOCK_IDS))
def test_each_block_carries_trigger_override_receipt(catalog: dict, block_id: str):
    block = catalog["escalations"][block_id]
    assert isinstance(block, dict), f"{block_id} must be a mapping"
    missing = REQUIRED_BLOCK_KEYS - set(block.keys())
    assert not missing, f"{block_id} is missing required keys: {missing}"
    # trigger + receipt are documentation prose: non-empty strings.
    assert isinstance(block["trigger"], str) and block["trigger"].strip()
    assert isinstance(block["receipt"], str) and block["receipt"].strip()
    # override is a real (non-empty) partial-profile mapping meant to be
    # deep-merged onto the base profile.
    assert isinstance(block["override"], dict) and block["override"]


@pytest.mark.parametrize("block_id", sorted(EXPECTED_BLOCK_IDS))
def test_override_merged_onto_base_profile_passes_strict_schema(
    catalog: dict, base_profile: dict, block_id: str,
):
    block = catalog["escalations"][block_id]
    merged = _deep_merge(base_profile, block["override"])
    # No exception == no unregistered/typo'd key was introduced.
    warnings = check_profile(merged, strict=True)
    assert warnings == []


def test_e1_epoch_cut_override_shape(catalog: dict):
    override = catalog["escalations"]["E1_epoch_cut"]["override"]
    assert override["reinforce"]["steps"] == 4


def test_e2_entropy_bump_override_shape(catalog: dict):
    override = catalog["escalations"]["E2_entropy_bump"]["override"]
    assert override["reinforce"]["entropy_coef"] == 0.02
    assert override["reinforce"]["sil"]["bc_coef"] == 0.8


def test_e3_rung_shift_override_shape(catalog: dict):
    override = catalog["escalations"]["E3_rung_shift"]["override"]
    tau_init = override["reinforce"]["backward_curriculum"]["tau_init"]
    assert tau_init == 4, (
        "tau_init=4 is the x~2000 rung on the base profile's 6-entry "
        "tape (tau 5=x2502 .. tau 0=entrance); see "
        "scripts/select_restart_states.py DEFAULT_TARGETS"
    )


def test_base_profile_itself_is_untouched_and_still_valid(base_profile: dict):
    """Sanity: the base profile this test reads is the real live-campaign
    profile, read-only. Confirm it still passes strict validation
    unmodified, so a future edit to that file that breaks schema is
    caught here too (not just in whatever test already covers it)."""
    warnings = check_profile(copy.deepcopy(base_profile), strict=True)
    assert warnings == []


def test_a_bad_override_is_actually_caught_by_this_harness(base_profile: dict):
    """Negative control: prove the strict-schema check in this file
    would fail on a typo'd override, so a green suite above is not
    green because the assertion is toothless."""
    bad_override = {"reinforce": {"entropy_coeff": 0.02}}  # typo'd key
    merged = _deep_merge(base_profile, bad_override)
    with pytest.raises(ConfigSchemaError):
        check_profile(merged, strict=True)
