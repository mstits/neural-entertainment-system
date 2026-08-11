"""configs/kirby.yaml's discrete-transition gate (`solve: room_advance:`).

The gate machinery (derive_transition_macros + the frontier injection in
Solver.explore) shipped with no profile configuring it, so nothing pinned
what a real config derives. Kirby is the profile that gets it: its own
config already receipts the room-transition lesson (the confluence
detector was rejected there because ordinary room loads fire constantly),
and a room-based stall is exactly what the gate exists to break.

These tests are config-level and pure — no ROM, no Pool, no solver run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.go_explore_solve import GX_BUCKET, derive_transition_macros

REPO = Path(__file__).resolve().parent.parent
KIRBY = REPO / "configs" / "kirby.yaml"


@pytest.fixture(scope="module")
def kirby() -> dict:
    return yaml.safe_load(KIRBY.read_text())


@pytest.fixture(scope="module")
def ra(kirby) -> dict:
    return kirby["solve"]["room_advance"]


def test_profile_parses_and_declares_the_gate(kirby):
    assert kirby["name"].startswith("Kirby")
    assert "room_advance" in kirby["solve"], "gate not configured"


def test_gate_knobs_are_the_specified_values(ra):
    assert ra["steps"] == 20
    assert ra["p"] == pytest.approx(0.05)
    assert ra["near"] == 24


def test_addr_is_the_profiles_own_room_observable(kirby, ra):
    """No new external knowledge: the gate's addr must be a byte this
    profile ALREADY declares. Here that is `area:` — the per-room scene
    id ($004F) whose verification is receipted in the config itself."""
    solve = kirby["solve"]
    assert ra["addr"] == solve["area"] == 0x004F
    declared = {solve["area"], solve["lives"], solve["y"],
                solve["progress"]["lo"], solve["progress"]["hi"]}
    assert ra["addr"] in declared


def test_derives_the_up_hold_from_kirbys_own_action_space(kirby, ra):
    """The shipped derivation, asserted verbatim: one macro, the bare
    ["up"] action held `steps` frames. Only one because this action_space
    declares no up+A / up+right combo — the config comment says so, and
    this is the assertion that keeps that claim honest."""
    space = kirby["action_space"]
    macros = derive_transition_macros(space, ra)
    assert macros == [(6, 20)]
    idx, hold = macros[0]
    assert set(space[idx]) == {"up"}
    assert hold == ra["steps"]
    assert not any(set(c) >= {"up", "A"} or set(c) >= {"up", "right"}
                   for c in space)


def test_gate_is_armed_not_inert(kirby, ra):
    """Solver prints 'discrete-transition gate inert' and skips the
    frontier injection when the derived list is empty; Kirby must not
    land in that branch."""
    assert derive_transition_macros(kirby["action_space"], ra)


def test_steps_knob_actually_drives_the_hold_length(kirby):
    """Mutation check: the 20 in the config is read, not a default that
    happens to match."""
    space = kirby["action_space"]
    assert derive_transition_macros(space, {"addr": 0x004F, "steps": 7}) \
        == [(6, 7)]


def test_derivation_follows_the_action_space_not_a_hardcoded_list():
    """Mutation check on the other side: a space that DOES carry the
    up+jump / up+right combos derives all three, so the single-macro
    result above is a property of Kirby's space, not of the helper."""
    space = [[], ["right"], ["up"], ["up", "A"], ["right", "up"]]
    assert derive_transition_macros(space, {"addr": 0x004F}) == [
        (2, 20), (3, 20), (4, 20)]


def test_near_is_expressed_in_gx_buckets(ra):
    """`near` is compared against gx BUCKETS in Solver.explore, so the
    comment's world-X figure has to track GX_BUCKET."""
    assert ra["near"] * GX_BUCKET == 384


# Profiles that deliberately opt into the discrete-transition gate. This is
# an ALLOWLIST, not a count: the gate changes how the solver spends its
# frontier budget, so arming it is a per-game decision that gets receipted,
# and a profile arriving here by accident must break the build.
#   kirby.yaml — the original room-graph opt-in (configs/kirby.yaml).
#   zelda.yaml — the overworld screen graph, receipted in
#     docs/receipts/games/zelda_onboarding_2026-08-10.md.
GATE_OPTED_IN = ["kirby.yaml", "zelda.yaml"]


def test_every_other_shipped_profile_stays_inert():
    """Default-identical: only the allowlisted profiles configure the gate,
    and a profile without the block derives nothing at all."""
    configured = []
    for path in sorted((REPO / "configs").glob("*.yaml")):
        try:
            prof = yaml.safe_load(path.read_text())
        except yaml.YAMLError:  # pragma: no cover - parse guard
            pytest.fail(f"{path.name} does not parse")
        if not isinstance(prof, dict) or "solve" not in prof:
            continue
        block = (prof["solve"] or {}).get("room_advance")
        if block is None:
            assert derive_transition_macros(
                prof.get("action_space", []), None) == []
        else:
            configured.append(path.name)
    assert configured == GATE_OPTED_IN, (
        "the set of profiles arming `solve: room_advance` changed. Adding a "
        "game here is a real decision — receipt the room observable, then "
        "extend GATE_OPTED_IN. Removing one means a profile lost its gate."
    )


@pytest.mark.parametrize("name", GATE_OPTED_IN)
def test_allowlisted_profiles_are_actually_armed(name):
    """An allowlist entry has to mean something: each opted-in profile must
    derive a non-empty macro list from its OWN action_space, otherwise the
    solver prints 'discrete-transition gate inert' and the entry is a lie."""
    prof = yaml.safe_load((REPO / "configs" / name).read_text())
    ra = prof["solve"]["room_advance"]
    macros = derive_transition_macros(prof["action_space"], ra)
    assert macros, f"{name} is allowlisted but derives no transition macros"
    space = prof["action_space"]
    for idx, hold in macros:
        assert hold == ra.get("steps", 20)
        assert space[idx], f"{name} derived the no-op as a transition macro"


@pytest.mark.parametrize("name", GATE_OPTED_IN)
def test_allowlisted_gate_addr_is_the_profiles_own_observable(name):
    """No new external knowledge, enforced for every opted-in profile: the
    gate's addr must be a byte the profile ALREADY declares and receipts."""
    solve = yaml.safe_load((REPO / "configs" / name).read_text())["solve"]
    declared = {solve["area"], solve["lives"], solve["y"],
                solve["progress"]["lo"]}
    if "hi" in solve["progress"]:
        declared.add(solve["progress"]["hi"])
    assert solve["room_advance"]["addr"] in declared
