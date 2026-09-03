"""Designer tests for src/forge/proposal.py's ``design_proposals``.

Covers the one field a ledger entry quotes verbatim from a proposal,
``wall_class_addressed`` (rendered at src/forge/proposal.py:422 and
re-stated in ``why_tried_arms_dont_fit``): it has to name the wall class
that actually made the arm eligible in ``registry.select_arm``, because
that is the claim a reader checks the selection against.

The bundle fixture here is the shape a receipt-shaped wall produces:
``bundle._classify_receipt_shaped`` appends SCRIPTED_RELEASE at
certainty ``candidate`` before OBSERVABLE_DEFECT at
``confirmed_by_receipt``, so the bundle's own first entry is the
*less* certain of the two while ``select_arm`` ranks by certainty. A
one-class bundle (cv_hall's) cannot tell the two orderings apart.
"""
from __future__ import annotations

from src.forge.proposal import design_proposals
from src.forge.registry import ARMS, select_arm

BUDGET = {"max_secs": 1200, "max_steps": 2_000_000, "seed": 1,
          "control": "same seed, flag off"}

#: The measured contra_wall bundle's ``mechanism_class``, in the order
#: src/forge/bundle.py:301-317 emits it: the candidate class first, the
#: receipt-confirmed one second.
CONTRA_CLASSES = [
    {"class": "SCRIPTED_RELEASE", "certainty": "candidate",
     "receipt": "runs/contra_wall/A8/A8_result.json"},
    {"class": "OBSERVABLE_DEFECT", "certainty": "confirmed_by_receipt",
     "receipt": "runs/contra_wall/A6/A6_RECEIPT.json"},
]


def _receipt_shaped_bundle(classes=None) -> dict:
    """A receipt-shaped bundle: no progress member, so no frontier
    shape and no cell_rate_history (src/forge/bundle.py:195-259)."""
    return {
        "wall_id": "contra_wall",
        "frontier_shape": {"certainty": "not_probed", "data": None},
        "cell_rate_history": {"certainty": "not_probed", "data": []},
        "ram_observables": "not_probed",
        "arms_tried": [],
        "mechanism_class": list(CONTRA_CLASSES if classes is None else classes),
        "missing": [],
    }


def _arm(name: str):
    for i, a in enumerate(ARMS):
        if a.name == name:
            return i, a
    raise AssertionError(f"no registered arm named {name!r}")


def _selected_class(selection: dict) -> str:
    """The class from select_arm's own ``class_match:<cls>:<certainty>``
    reason code -- the selector's stated grounds, not a re-derivation."""
    for code in selection.get("reason_codes", []):
        if code.startswith("class_match:"):
            return code.split(":")[1]
    raise AssertionError(f"no class_match reason code in {selection!r}")


def test_the_proposal_names_the_class_that_made_the_arm_eligible():
    """C4. On a two-class receipt-shaped bundle the selector picks the
    arm on OBSERVABLE_DEFECT (confirmed_by_receipt) while the bundle
    lists SCRIPTED_RELEASE (candidate) first. Taking ``classes[0]``
    stamps the class the selection did not use into the field the
    ledger entry quotes."""
    bundle = _receipt_shaped_bundle()
    selection = select_arm(bundle)
    arm_index = selection["index"]
    arm = ARMS[arm_index]
    assert arm.name == "lock", "fixture assumes the lock arm is selected"

    picked = _selected_class(selection)
    assert picked == "OBSERVABLE_DEFECT"
    assert bundle["mechanism_class"][0]["class"] == "SCRIPTED_RELEASE", (
        "fixture must keep the bundle's first entry different from the "
        "selector's, or it cannot see the defect")

    proposals = design_proposals(bundle, arm, arm_index, "contra_wall",
                                 k=3, budget=BUDGET)
    assert proposals
    for p in proposals:
        assert p["wall_class_addressed"] == picked
        assert p["wall_class_addressed"] in arm.wall_classes
        assert picked in p["why_tried_arms_dont_fit"]
        assert "SCRIPTED_RELEASE" not in p["why_tried_arms_dont_fit"]


def test_a_class_the_arm_does_not_address_is_never_named():
    """The certainty ordering alone is not enough: the top-ranked class
    still has to be one this arm registers for, or the proposal claims
    to address a class the registry says the arm never covers."""
    arm_index, arm = _arm("lock")
    bundle = _receipt_shaped_bundle([
        {"class": "KEY_BLIND", "certainty": "confirmed_by_receipt"},
        {"class": "SCRIPTED_RELEASE", "certainty": "candidate"},
    ])
    assert "KEY_BLIND" not in arm.wall_classes

    proposals = design_proposals(bundle, arm, arm_index, "contra_wall",
                                 k=1, budget=BUDGET)
    assert proposals[0]["wall_class_addressed"] == "SCRIPTED_RELEASE"


def test_a_single_class_bundle_is_unchanged():
    """cv_hall's shape: one class, and it is the ortho arm's own. The
    fix must not move this case, which is why cv_hall could not surface
    the defect."""
    arm_index, arm = _arm("ortho")
    bundle = _receipt_shaped_bundle([
        {"class": "KEY_BLIND", "certainty": "confirmed_by_receipt"},
    ])
    proposals = design_proposals(bundle, arm, arm_index, "cv_hall",
                                 k=3, budget=BUDGET)
    assert [p["wall_class_addressed"] for p in proposals] == ["KEY_BLIND"] * 3


def test_no_class_at_all_still_falls_back_to_the_arms_own_first_class():
    """An empty ``mechanism_class`` keeps the pre-existing fallback:
    the arm's first registered class, then the UNKNOWN literal."""
    arm_index, arm = _arm("lock")
    bundle = _receipt_shaped_bundle([])
    proposals = design_proposals(bundle, arm, arm_index, "contra_wall",
                                 k=1, budget=BUDGET)
    assert proposals[0]["wall_class_addressed"] == arm.wall_classes[0]


def test_no_intersection_falls_back_to_the_highest_certainty_class():
    """When nothing the bundle names is in ``arm.wall_classes``, the
    proposal still names a class the bundle actually carries, and the
    most-trusted one rather than whichever happens to be listed first."""
    arm_index, arm = _arm("lock")
    bundle = _receipt_shaped_bundle([
        {"class": "SOMETHING_ELSE", "certainty": "not_probed"},
        {"class": "KEY_BLIND", "certainty": "confirmed_by_receipt"},
    ])
    proposals = design_proposals(bundle, arm, arm_index, "contra_wall",
                                 k=1, budget=BUDGET)
    assert proposals[0]["wall_class_addressed"] == "KEY_BLIND"
