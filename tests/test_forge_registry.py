"""src/forge/registry.py -- the arm library and index-only selector.

FORGE_SPEC_2026-09-01.md Section 2c. Each test is revert-verified against
a named corruption (in the test's own docstring).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge.registry import (  # noqa: E402
    ARMS, Arm, audit_violations, is_auditable, select_arm, _eligible,
)


def _bundle(wall_id: str, classes: list[tuple[str, str]],
            arms_tried: list | None = None,
            cell_rate_rows: list | None = None) -> dict:
    return {
        "wall_id": wall_id,
        "mechanism_class": [{"class": c, "certainty": cert} for c, cert in classes],
        "arms_tried": arms_tried or [],
        "cell_rate_history": {"data": cell_rate_rows or []},
    }


# ------------------------------------------------------------------ audit

def test_every_registered_arm_has_activity_counter_or_is_unarmable():
    """Every entry in ARMS is either auditable (armed_signal +
    activity_counter both set) or its own status admits it is not
    (UNAUDITABLE / K0-HALTED) -- never a status that reads as live with
    no counter behind it to audit.

    Revert-verify: register an arm with activity_counter=None and
    status="ARMED" -- audit_violations must name it by name.
    """
    assert audit_violations(ARMS) == []
    assert {a.name: is_auditable(a) for a in ARMS} == {
        "ortho": "AUDITABLE", "lock": "UNAUDITABLE", "gate_opener": "UNAUDITABLE",
    }

    broken_lock = dataclasses.replace(ARMS[1], activity_counter=None, status="ARMED")
    corrupted = (ARMS[0], broken_lock, ARMS[2])
    assert audit_violations(corrupted) == ["lock"]
    # A real counter clears the same corrupted status -- the check
    # reads the counter, not the name "ARMED" as a magic string.
    fixed = dataclasses.replace(broken_lock, activity_counter="lock_selections")
    assert audit_violations((ARMS[0], fixed, ARMS[2])) == []


def test_is_auditable_requires_armed_signal_not_only_a_counter():
    """`is_auditable` reads AUDITABLE only when both `armed_signal` and
    `activity_counter` are set. Against the shipped ARMS this has no
    witness -- ortho carries both, lock and gate_opener carry neither
    (lock's `armed_signal` is set but its `activity_counter` is None,
    which alone already fails the check) -- so a regression that drops
    the `armed_signal` half of the conjunct (`have_both =
    bool(arm.activity_counter)`) leaves every ARMS verdict unchanged.
    This fixture supplies the missing case: a counter with no
    armed_signal.

    Revert-verify: change `is_auditable` to `bool(arm.activity_counter)`
    alone -- this fails (reads AUDITABLE, not UNAUDITABLE).
    """
    counter_no_signal = dataclasses.replace(
        ARMS[0], armed_signal=None, activity_counter="ortho_selections")
    assert is_auditable(counter_no_signal) == "UNAUDITABLE"

    signal_no_counter = dataclasses.replace(
        ARMS[0], armed_signal="ortho_armed()", activity_counter=None)
    assert is_auditable(signal_no_counter) == "UNAUDITABLE"


# --------------------------------------------------------------- select

def test_select_returns_index_in_range_only():
    """select_arm never returns an arm's name -- only an int index into
    the arms tuple it was given, which the caller resolves.

    Revert-verify: change select_arm's returned "index" to
    `arms[index].name` and this fails on the isinstance(int) check.
    """
    bundle = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")])
    result = select_arm(bundle, ARMS)
    assert isinstance(result["index"], int) and not isinstance(result["index"], bool)
    assert 0 <= result["index"] < len(ARMS)
    assert ARMS[result["index"]].name == "ortho"


def test_select_marks_tried_arm_redundant():
    """A bundle whose arms_tried already carries `ortho up` for cv_hall,
    with the selector again proposing ortho's default mode `up` (the
    same mode CLAIMS.md:223-242 records as the real trial) ->
    redundant_with:["ortho"]. A different wall's selection over the
    same registry (no matching history for THAT wall) reads
    redundant_with:[].

    Revert-verify: make select_arm ignore bundle["arms_tried"] and
    arm.history entirely (always return redundant_with=[]) -- the first
    assertion below fails.
    """
    tried = [{"arm": "ortho", "knobs": {"mode": "up"}, "verdict": "FIRED",
              "outcome": "MECHANISM_VALIDATED/PREMISE_STALE",
              "receipt": "CLAIMS.md:223-242"}]
    bundle = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")], arms_tried=tried)
    result = select_arm(bundle, ARMS)
    assert ARMS[result["index"]].name == "ortho"
    assert result["redundant_with"] == ["ortho"]

    # ortho's own seeded history (CLAIMS.md:223-242) fires the same way
    # even when the bundle itself carries no arms_tried -- two
    # independent sources for the same real fact.
    bare = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")])
    assert select_arm(bare, ARMS)["redundant_with"] == ["ortho"]

    # A wall with no matching trial history is never marked redundant.
    other = _bundle("some_other_wall", [("KEY_BLIND", "confirmed_by_receipt")])
    assert select_arm(other, ARMS)["redundant_with"] == []


def test_select_on_key_blind_never_picks_spatial_only_arm():
    """`lock` is registered for SCRIPTED_RELEASE/OBSERVABLE_DEFECT only
    -- positional/survival objective, no interaction-discovery angle on
    a KEY_BLIND wall -- so it must never be eligible for a KEY_BLIND
    selection, and select_arm must never return its index for one.

    Revert-verify: make _eligible ignore `cls` and return every arm
    index regardless of class (dropping the class filter) -- the set
    equality below fails, since `lock`'s index would then appear.
    """
    assert set(_eligible("KEY_BLIND", ARMS)) == {0, 2}  # ortho, gate_opener
    assert ARMS[1].name == "lock"
    assert 1 not in _eligible("KEY_BLIND", ARMS)

    bundle = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")])
    result = select_arm(bundle, ARMS)
    assert ARMS[result["index"]].name != "lock"


# -------------------------------------------------------- extra coverage

def test_select_picks_the_confirmed_class_over_a_merely_candidate_one():
    """contra's real bundle carries two classes at once (SD finding 13):
    SCRIPTED_RELEASE at "candidate", OBSERVABLE_DEFECT at
    "confirmed_by_receipt". Both route to `lock` in this registry, but
    the certainty ranking must still prefer the confirmed reading's
    class_match reason code over the candidate one's.

    Revert-verify: sort ranked classes by bundle order instead of
    certainty (drop _CERTAINTY_RANK) -- with the classes listed
    candidate-first in a bundle, the first reason_code would then read
    "class_match:SCRIPTED_RELEASE:candidate" instead.
    """
    bundle = _bundle("contra_wall", [
        ("SCRIPTED_RELEASE", "candidate"),
        ("OBSERVABLE_DEFECT", "confirmed_by_receipt"),
    ])
    result = select_arm(bundle, ARMS)
    assert ARMS[result["index"]].name == "lock"
    assert result["reason_codes"][0] == "class_match:OBSERVABLE_DEFECT:confirmed_by_receipt"


def test_select_never_picks_the_k0_halted_arm_over_a_live_peer():
    """`gate_opener` shares wall_classes=("KEY_BLIND",) with `ortho` but
    is K0-HALTED -- parked, not pickable tonight (PA open question 1,
    option B). A KEY_BLIND selection must prefer the live arm.

    Revert-verify: drop the live/halted tiebreak (always take
    `eligible[0]` in registration order) -- harmless here only because
    ortho is already index 0; reorder ARMS so gate_opener precedes
    ortho and this fails without the tiebreak.
    """
    bundle = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")])
    reordered = (ARMS[2], ARMS[0], ARMS[1])  # gate_opener, ortho, lock
    result = select_arm(bundle, reordered)
    assert reordered[result["index"]].name == "ortho"


def test_knob_pin_secs_derived_only_from_bundle_cell_rate_history():
    """The `*-pin-secs` knob override comes only from the bundle's own
    cell_rate_history span -- never invented when that data is absent
    (contra's real bundle has none: not_probed).

    Revert-verify: hardcode a pin-secs override regardless of whether
    cell_rate_history has rows -- the second assertion (no override
    key change when rows are empty) fails.
    """
    with_rows = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")],
                         cell_rate_rows=[{"elapsed_s": 60, "cells": 1},
                                         {"elapsed_s": 3601, "cells": 2}])
    result = select_arm(with_rows, ARMS)
    assert result["knobs"]["--ortho-pin-secs"] == 3541.0

    no_rows = _bundle("cv_hall", [("KEY_BLIND", "confirmed_by_receipt")])
    result2 = select_arm(no_rows, ARMS)
    assert result2["knobs"]["--ortho-pin-secs"] == ARMS[0].knobs["--ortho-pin-secs"]


def test_arms_registry_has_no_duplicate_names_and_valid_off_values():
    """Self-check on ARMS's own shape, the same discipline
    tests/test_forge_gates.py runs over FORGE_REGISTRY."""
    names = [a.name for a in ARMS]
    assert len(names) == len(set(names)) == 3
    for a in ARMS:
        assert a.off_value == "off"
        assert isinstance(a.wall_classes, tuple) and a.wall_classes
        assert isinstance(a.history, tuple)
