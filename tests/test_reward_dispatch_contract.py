"""A profile that declares no `reward_weights` of its own must not
silently inherit a hand-authored, hardcoded reward — including its win
predicate — just because its display name happens to contain a game's
name.

BREACH PATH 1 (quarantine-hardening adjudication, 2026-08-26).
`configs/legend_of_zelda.yaml` is 31 lines, declares zero reward weights,
and describes itself in its own header as "Not a training profile" — yet
it used to resolve to `ZeldaReward` and inherit the disassembly-sourced,
quarantined Ganon-defeated win flag (`0x0672`) purely because its
`name:` contains "Zelda". `configs/zelda_roomfp.yaml` had the identical
defect and was not named in the original adjudication; this test file's
parametrization is what found it. `configs/metroid_roomfp.yaml` silently
acquired `MetroidReward` the same way — a struct CLAIMS.md's Quarantine
section BLOCKS from producing any Learned-ledger claim.

Dispatch is now keyed on the profile's explicit `reward_id`, not its
display name (see `nes_core::rewards::build_reward` and
`RewardFunction.kind`, added specifically so a test can assert dispatch
exactly rather than inferring it from a numeric reward delta several
arms could plausibly produce). This module pins that contract from the
outside, over real profiles, so a future profile that forgets to declare
`reward_id` fails loudly here instead of silently drawing a win predicate
it never asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import nes_core

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    return doc if isinstance(doc, dict) else {}


# The three profiles the adjudication + this hardening pass named
# explicitly, each reproduced live pre-fix.
BREACH_FIX_PROFILES = [
    "legend_of_zelda.yaml",
    "zelda_roomfp.yaml",
    "metroid_roomfp.yaml",
]


@pytest.mark.parametrize("name", BREACH_FIX_PROFILES)
def test_profile_without_reward_weights_resolves_to_generic(name: str) -> None:
    profile = _load(CONFIG_DIR / name)
    assert not profile.get("reward_weights"), (
        f"{name}: test fixture assumption broken — this profile now "
        f"declares reward_weights, so its reward_id is a deliberate "
        f"choice and this test no longer applies to it."
    )
    rf = nes_core.build_reward_function(profile)
    assert rf.kind == "generic", (
        f"{name}: declares no reward_weights but resolved to reward kind "
        f"{rf.kind!r} instead of generic — a profile that asks for "
        f"nothing must not inherit a hand-authored win predicate."
    )


def test_ganon_defeated_byte_does_not_flip_episode_success_on_a_bare_profile() -> None:
    """The exact reproduction from the adjudication: flipping RAM 0x0672
    on an otherwise-zeroed 2KB buffer must NOT flip episode_success() for
    a profile that never asked for ZeldaReward."""
    for name in ("legend_of_zelda.yaml", "zelda_roomfp.yaml"):
        profile = _load(CONFIG_DIR / name)
        rf = nes_core.build_reward_function(profile)
        ram = bytearray(2048)
        rf.compute(bytes(ram), action=0)
        ram[0x0672] = 1
        rf.compute(bytes(ram), action=0)
        assert rf.episode_success() is False, (
            f"{name}: ram[0x0672]=1 flipped episode_success() True on a "
            f"profile that declares no reward_weights — the quarantined "
            f"Zelda win predicate leaked back in."
        )


def test_metroid_roomfp_does_not_acquire_the_blocked_metroid_reward() -> None:
    profile = _load(CONFIG_DIR / "metroid_roomfp.yaml")
    rf = nes_core.build_reward_function(profile)
    assert rf.kind != "metroid", (
        "metroid_roomfp.yaml resolved to MetroidReward, which CLAIMS.md's "
        "Quarantine section blocks from producing any Learned-ledger claim."
    )


def test_a_profile_with_real_reward_weights_still_dispatches_specifically() -> None:
    """The fix must not overcorrect into starving every Zelda profile of
    its reward — zelda.yaml and zelda_gui_tuned.yaml both declare full
    reward_weights blocks and must keep resolving to ZeldaReward."""
    for name in ("zelda.yaml", "zelda_gui_tuned.yaml"):
        profile = _load(CONFIG_DIR / name)
        assert profile.get("reward_weights"), f"{name}: fixture assumption broken"
        rf = nes_core.build_reward_function(profile)
        assert rf.kind == "zelda", (
            f"{name}: declares real reward_weights but resolved to "
            f"{rf.kind!r} instead of zelda."
        )
