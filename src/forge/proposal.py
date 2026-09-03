"""The Forge proposal object: designer + refuter (FORGE_SPEC_2026-09-01.md
Section 3).

A proposal is the frozen unit the gate judges: a candidate knob setting
(``kind:"knob"``, re-parameterizing an already-registered arm) or a
candidate patch (``kind:"arm"``, a new mechanism, schema-supported and
fixture-tested tonight -- ``why_tried_arms_dont_fit``/patch/tests_added
are carried but no live arm patch ships, per spec Section 3's "live only
if Section 5 finishes early").

The spec's designer and refuter are described as separate agent calls;
tonight both are deterministic pure functions, the same two-way door
piece (c)'s ``select_arm`` takes for its own selector (spec Section 6):
``design_proposals`` derives K knob-shaped candidates from bundle fields
only, and ``refute_proposal`` runs the four fixed checks a reviewer
would run against a designer's draft -- purity, redundancy, vacuity,
budget -- never a model call. Nothing here touches the emulator or
CLAIMS.md.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence

# The selector's certainty ordering, imported rather than re-implemented
# so `wall_class_addressed` and `select_arm`'s `class_match:` reason code
# can never rank the same bundle differently. src/forge/registry.py
# imports nothing from this package, so this direction adds no cycle.
from src.forge.registry import _classes_by_certainty

#: The knob key both this module and src/forge/registry.py's
#: ``_knob_matches`` treat as the primary, coarse-grained axis for
#: redundancy comparisons -- an arm's top-level flag setting
#: (``--ortho up``, ``--lock-objective survival``, ...).
MODE_KEY = "mode"

#: Address-literal detector for the purity refuter check: any hex
#: literal that reads as a memory/ROM address (three or more hex
#: digits after the 0x prefix -- ``0x0450`` is the fixture value the
#: spec names). This is the one purity check the spec's own test names
#: (test_refuter_rejects_address_literal_in_proposal); it does not
#: attempt to enumerate every game-specific-instruction shape.
_ADDRESS_RE = re.compile(r"\b0x[0-9A-Fa-f]{3,}\b")

#: Text fields a refuter reads for purity/vacuity content. Free-text
#: fields only -- never ``patch`` (a diff legitimately contains hex
#: literals for unrelated reasons; the spec's purity boundary is about
#: what informed the DESIGN, not what a patch happens to touch) and
#: never ``knobs`` (numeric CLI values, not prose).
_TEXT_FIELDS = ("why_tried_arms_dont_fit", "wall_class_addressed")

_PIN_SECS_SUFFIX = "-pin-secs"

REFUTER_REASON_PURITY = "purity"
REFUTER_REASON_REDUNDANCY = "redundancy"
REFUTER_REASON_VACUITY = "vacuity"
REFUTER_REASON_BUDGET = "budget"


class FrozenProposalError(ValueError):
    """Raised when a proposal is used in a way its frozen state forbids
    (piloted before freezing was checked, or a hash-mismatch caller
    that wants a hard failure rather than a boolean)."""


def _canonical_json(proposal: Mapping[str, Any]) -> bytes:
    """Canonical bytes for hashing: sorted keys, fixed separators, no
    whitespace ambiguity. The same encoding on every call is what makes
    ``freeze_proposal`` deterministic and ``verify_frozen`` meaningful --
    a dict with the same content but different key order must hash the
    same; a dict with ANY value changed must hash differently."""
    return json.dumps(proposal, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")


def freeze_proposal(proposal: Mapping[str, Any]) -> str:
    """The sha256 hex digest of ``proposal``'s canonical JSON -- spec
    Section 3: "frozen (sha256 recorded) before the pilot." Pure: never
    mutates ``proposal`` or writes anything. A caller records this
    string alongside the proposal (the ``proposal_<k>.md`` frontmatter,
    the cycle log) and re-derives it after the pilot to detect the
    Section 4 row (d) FAIL condition -- "a proposal edited after pilot
    start (hash mismatch)" -- via ``verify_frozen`` below.
    """
    return hashlib.sha256(_canonical_json(proposal)).hexdigest()


def verify_frozen(proposal: Mapping[str, Any], frozen_sha256: str) -> bool:
    """True iff ``proposal`` still hashes to ``frozen_sha256`` -- the
    tamper check a caller runs immediately after a pilot returns, before
    trusting the pilot's reading. False on ANY change: a knob value, a
    reordered list, a single added whitespace-only field -- the
    canonical encoding in ``_canonical_json`` makes all of these visible.
    """
    return freeze_proposal(proposal) == frozen_sha256


def null_candidate(arm, wall_id: str, budget: Mapping[str, Any]) -> dict:
    """The in-situ negative control: a ``kind:"knob"`` proposal whose
    knob equals the arm's ``off_value`` (spec Section 3's gate
    anti-vacuity fixture; Section 3's stop-condition text: "The null
    candidate also runs live once per cycle ... a cycle in which it
    does not read VOID is itself VOID"). Its ``gate.stage2.threshold``
    is a real positive number (never 0) so ``refute_proposal``'s
    vacuity check does not itself reject the null candidate before it
    ever reaches a pilot.
    """
    knobs = {MODE_KEY: arm.off_value}
    metric = arm.gate_fn.split(">=")[0] if arm.gate_fn else "activity"
    threshold = 1
    if arm.gate_fn and ">=" in arm.gate_fn:
        try:
            threshold = float(arm.gate_fn.split(">=", 1)[1])
        except ValueError:
            threshold = 1
    return {
        "kind": "knob",
        "arm_index": None,
        "knobs": knobs,
        "wall_class_addressed": "null",
        "why_tried_arms_dont_fit": (
            f"in-situ negative control for {wall_id}: {arm.name} held at "
            f"its off_value ({arm.off_value!r}) so the activity counter "
            f"is expected to read INERT and stage 1 is expected to read "
            f"VOID; a cycle whose null candidate does not read VOID is "
            f"itself VOID."
        ),
        "patch": None,
        "tests_added": [],
        "inertness_mirror": arm.inertness_test,
        "gate": {
            "stage1": "activity_counter FIRED, n_obs>0",
            "stage2": {"metric": metric, "threshold": threshold,
                       "vs": "matched_control"},
            "stage3": [],
        },
        "kill": ["counter INERT"],
        "budget": dict(budget),
        "tier3_sentence": None,
        "is_null": True,
    }


def _pin_secs_flags(arm) -> list[str]:
    return [f for f in arm.knobs if f.endswith(_PIN_SECS_SUFFIX)]


def design_proposals(bundle: Mapping[str, Any], arm, arm_index: int,
                      wall_id: str, *, k: int = 3,
                      budget: Mapping[str, Any]) -> list[dict]:
    """``K`` deterministic ``kind:"knob"`` proposals for ``arm``,
    varying its registered ``*-pin-secs`` knob (if it has one) around
    the value implied by ``bundle["cell_rate_history"]``'s trailing
    window span -- the same field ``registry.select_arm`` reads,
    expressible from bundle data only (spec Section 3's designer role,
    table-driven tonight rather than an agentic call; the wall's own
    ``mechanism_class`` names ``wall_class_addressed``).

    Every returned proposal carries a real, satisfiable
    ``gate.stage2.threshold`` from the arm's registered ``gate_fn`` and
    a ``max_secs``-respecting pin-secs knob, so a well-formed bundle
    never manufactures a proposal that fails the refuter's vacuity or
    budget checks by construction -- those checks exist for a designer
    that drifts, not to make every proposal here fail on arrival.
    """
    # The class stamped into `wall_class_addressed` has to be the one
    # that made this arm eligible in `registry.select_arm`, because the
    # ledger entry quotes the field back as the grounds for the
    # selection. The bundle's own order is not that class:
    # `bundle._classify_receipt_shaped` appends SCRIPTED_RELEASE at
    # certainty `candidate` before OBSERVABLE_DEFECT at
    # `confirmed_by_receipt`, while the selector ranks by certainty and
    # then filters on the arm's registered classes. So apply the
    # selector's own ordering function (one source of truth for the
    # ranking, not a second copy of _CERTAINTY_RANK) and then the arm's
    # own filter. A one-class bundle is unmoved by both steps.
    ranked = [cls for cls, _certainty in _classes_by_certainty(dict(bundle))]
    wall_class = next(
        (c for c in ranked if c in (arm.wall_classes or ())),
        ranked[0] if ranked else (arm.wall_classes[0]
                                  if arm.wall_classes else "UNKNOWN"))

    metric = arm.gate_fn.split(">=")[0] if arm.gate_fn else "activity"
    threshold = 8
    if arm.gate_fn and ">=" in arm.gate_fn:
        try:
            threshold = float(arm.gate_fn.split(">=", 1)[1])
        except ValueError:
            threshold = 8

    max_secs = float(budget.get("max_secs", 1200))
    pin_flags = _pin_secs_flags(arm)
    base_pin = arm.knobs.get(pin_flags[0]) if pin_flags else None

    crh = bundle.get("cell_rate_history") or {}
    rows = crh.get("data") or []
    span = None
    if len(rows) >= 2:
        first_t, last_t = rows[0].get("elapsed_s"), rows[-1].get("elapsed_s")
        if isinstance(first_t, (int, float)) and isinstance(last_t, (int, float)):
            span = max(1.0, float(last_t) - float(first_t))

    proposals: list[dict] = []
    # Deterministic knob ladder: base value, then one step down and one
    # step up, each clamped well under `max_secs` so no seeded proposal
    # trips the budget refuter check on its own.
    ladder_fracs = (1.0, 0.5, 1.5)
    for i in range(k):
        frac = ladder_fracs[i % len(ladder_fracs)]
        knobs = dict(arm.knobs)
        knobs[MODE_KEY] = {
            "ortho": "up", "lock": "survival", "gate_opener": "enumerate",
        }.get(arm.name, "on")
        if pin_flags:
            candidate = (span if span is not None else base_pin) or 60.0
            pin_val = max(15.0, min(candidate * frac, max_secs * 0.5))
            knobs[pin_flags[0]] = round(pin_val, 1)

        proposals.append({
            "kind": "knob",
            "arm_index": arm_index,
            "knobs": knobs,
            "wall_class_addressed": wall_class,
            "why_tried_arms_dont_fit": (
                f"{arm.name} is the highest-ranked registered arm for "
                f"{wall_class} on {wall_id}'s bundle; proposal {i} "
                f"re-parameterizes its pin-secs knob from the wall's own "
                f"cell_rate_history window span rather than re-using the "
                f"registry default unmodified."
            ),
            "patch": None,
            "tests_added": [],
            "inertness_mirror": arm.inertness_test,
            "gate": {
                "stage1": "activity_counter FIRED, n_obs>0",
                "stage2": {"metric": metric, "threshold": threshold,
                           "vs": "matched_control"},
                "stage3": ["frontier>prior_best replay-verified"],
            },
            "kill": ["throughput cost >40%", "archive >3x cells with no "
                     "frontier motion", "counter INERT"],
            "budget": dict(budget),
            "tier3_sentence": None,
            "is_null": False,
        })
    return proposals


def _purity_violation(proposal: Mapping[str, Any]) -> Optional[str]:
    for field in _TEXT_FIELDS:
        text = proposal.get(field)
        if isinstance(text, str) and _ADDRESS_RE.search(text):
            return (f"{REFUTER_REASON_PURITY}: address literal found in "
                    f"{field!r}")
    return None


#: Tolerance on a shared secondary numeric knob for two knob vectors to
#: count as the same point in knob space (the designer is a
#: deterministic table lookup, so a genuine repeat lands on the exact
#: same value; this only absorbs float round-trip noise).
_REDUNDANCY_TOLERANCE = 1e-6


def _knob_space_distance(a_knobs: Mapping[str, Any],
                          b_knobs: Mapping[str, Any]) -> Optional[float]:
    """Distance between two knob vectors on the axes both actually
    record (spec Section 3: "knob-space distance to arms_tried", not a
    label match on `mode` alone). ``None`` means "not comparable / not
    established as the same point":

    - different ``mode`` -> ``None`` (a different arm setting entirely,
      not a matter of degree).
    - same ``mode`` and at least one shared secondary numeric knob ->
      the max absolute difference over the knobs both sides specify.
    - same ``mode``, no secondary knob shared -> ``0.0`` (the same
      point) only if NEITHER side records any other knob either (an
      arm with no secondary knob, like the registry's `gate_opener`
      entry, has nothing else to vary); if either side has secondary
      knobs the other lacks, ``None`` -- a coarse arm-level history
      entry that recorded only `mode` (e.g. the seeded ortho
      `CLAIMS.md:223-242` trial, ``knobs={"mode": "up"}``) is not
      evidence any particular pin-secs/bias/band point was tried, so a
      knob proposal that varies those must not be treated as the same
      point merely because its primary label matches.
    """
    if a_knobs.get(MODE_KEY) != b_knobs.get(MODE_KEY):
        return None
    a_secondary = {k: v for k, v in a_knobs.items() if k != MODE_KEY}
    b_secondary = {k: v for k, v in b_knobs.items() if k != MODE_KEY}
    shared = [k for k in a_secondary if k in b_secondary
              and isinstance(a_secondary[k], (int, float))
              and isinstance(b_secondary[k], (int, float))]
    if not shared:
        if a_secondary or b_secondary:
            return None
        return 0.0
    return max(abs(a_secondary[k] - b_secondary[k]) for k in shared)


def _redundancy_violation(proposal: Mapping[str, Any],
                           arms_tried: Sequence[Mapping[str, Any]],
                           arm_name: Optional[str]) -> Optional[str]:
    knobs = proposal.get("knobs", {})
    if knobs.get(MODE_KEY) is None:
        return None
    for entry in arms_tried:
        if arm_name is not None and entry.get("arm") != arm_name:
            continue
        distance = _knob_space_distance(knobs, entry.get("knobs", {}))
        if distance is not None and distance <= _REDUNDANCY_TOLERANCE:
            return (f"{REFUTER_REASON_REDUNDANCY}: {arm_name!r} knobs "
                    f"{dict(knobs)!r} within tolerance ({distance}) of "
                    f"an arms_tried/history entry {entry.get('knobs')!r}")
    return None


def _vacuity_violation(proposal: Mapping[str, Any]) -> Optional[str]:
    stage2 = proposal.get("gate", {}).get("stage2", {})
    threshold = stage2.get("threshold")
    metric = stage2.get("metric")
    if not metric:
        return f"{REFUTER_REASON_VACUITY}: stage2 has no metric name"
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        return (f"{REFUTER_REASON_VACUITY}: stage2 threshold "
                f"{threshold!r} can never fail")
    return None


def _budget_violation(proposal: Mapping[str, Any],
                       budget: Mapping[str, Any]) -> Optional[str]:
    max_secs = float(budget.get("max_secs",
                                 proposal.get("budget", {}).get("max_secs", 1200)))
    for flag, value in proposal.get("knobs", {}).items():
        if flag.endswith(_PIN_SECS_SUFFIX) and isinstance(value, (int, float)):
            if value >= max_secs:
                return (f"{REFUTER_REASON_BUDGET}: {flag}={value} cannot "
                         f"arm within max_secs={max_secs}")
    return None


def refute_proposal(proposal: Mapping[str, Any], *,
                     arms_tried: Sequence[Mapping[str, Any]] = (),
                     budget: Mapping[str, Any],
                     arm_name: Optional[str] = None) -> dict:
    """Runs the four fixed checks (purity, redundancy, vacuity, budget)
    against ``proposal``. Returns ``{"rejected": bool, "reasons": [...]}``
    -- every failing check contributes one reason string, never just the
    first; a caller that repairs one issue and re-refutes sees the
    others still listed if they were not also fixed.
    """
    reasons = []
    for check in (_purity_violation(proposal),
                  _redundancy_violation(proposal, arms_tried, arm_name),
                  _vacuity_violation(proposal),
                  _budget_violation(proposal, budget)):
        if check:
            reasons.append(check)
    return {"rejected": bool(reasons), "reasons": reasons}


def repair_proposal(proposal: Mapping[str, Any],
                     reasons: Sequence[str]) -> dict:
    """A deep copy of ``proposal`` with each named failing check's
    surface condition addressed (never blind -- only the specific field
    the matching reason names is touched). A reason with no matching
    repair leaves the proposal unchanged on that axis, so a
    ``refute_proposal`` re-run after ``repair_proposal`` can still
    reject on the same grounds -- this is what lets the repair-round
    cap actually bound the loop instead of guaranteeing eventual
    success.
    """
    repaired = copy.deepcopy(dict(proposal))
    for reason in reasons:
        if reason.startswith(REFUTER_REASON_PURITY):
            for field in _TEXT_FIELDS:
                text = repaired.get(field)
                if isinstance(text, str):
                    repaired[field] = _ADDRESS_RE.sub("[address-redacted]", text)
        elif reason.startswith(REFUTER_REASON_VACUITY):
            stage2 = repaired.setdefault("gate", {}).setdefault("stage2", {})
            if not isinstance(stage2.get("threshold"), (int, float)) \
                    or stage2.get("threshold", 0) <= 0:
                stage2["threshold"] = 1
        elif reason.startswith(REFUTER_REASON_BUDGET):
            max_secs = float(repaired.get("budget", {}).get("max_secs", 1200))
            for flag, value in list(repaired.get("knobs", {}).items()):
                if flag.endswith(_PIN_SECS_SUFFIX) and isinstance(value, (int, float)):
                    if value >= max_secs:
                        repaired["knobs"][flag] = round(max_secs * 0.25, 1)
        elif reason.startswith(REFUTER_REASON_REDUNDANCY):
            knobs = repaired.setdefault("knobs", {})
            for flag, value in list(knobs.items()):
                if flag.endswith(_PIN_SECS_SUFFIX) and isinstance(value, (int, float)):
                    knobs[flag] = round(value * 1.25, 1)
                    break
    return repaired


def render_proposal_markdown(proposal: Mapping[str, Any], *, wall_id: str,
                              cycle_id: str, index: int,
                              frozen_sha256: str) -> str:
    """House-format Markdown for ``runs/forge/<wall>/<cycle>/proposal_<k>.md``
    (spec Section 3: "house format of
    ``docs/proposals/gate_opener_arm_2026-08-11.md`` Sections 1-7").
    Renders the frozen proposal as read-only record -- this function
    never re-serializes ``proposal`` with different field values than
    what ``frozen_sha256`` was computed over; a caller passes the exact
    frozen dict.
    """
    gate = proposal.get("gate", {})
    stage2 = gate.get("stage2", {})
    lines = [
        f"# Forge proposal {index} — {wall_id}/{cycle_id}",
        "",
        f"**Kind.** `{proposal.get('kind')}`, arm index "
        f"`{proposal.get('arm_index')}`.",
        "",
        f"**Wall class addressed.** {proposal.get('wall_class_addressed')}",
        "",
        f"**Why the tried arms don't fit.** "
        f"{proposal.get('why_tried_arms_dont_fit')}",
        "",
        f"**Knobs.** `{json.dumps(proposal.get('knobs', {}), sort_keys=True)}`",
        "",
        f"**Gate.** stage1: {gate.get('stage1')}. "
        f"stage2: metric={stage2.get('metric')} "
        f"threshold={stage2.get('threshold')} vs={stage2.get('vs')}. "
        f"stage3: {gate.get('stage3')}.",
        "",
        f"**Kill conditions.** {proposal.get('kill')}",
        "",
        f"**Budget.** `{json.dumps(proposal.get('budget', {}), sort_keys=True)}`",
        "",
        f"**Inertness mirror.** `{proposal.get('inertness_mirror')}`",
        "",
        f"**Frozen.** sha256 `{frozen_sha256}` (recorded before the pilot; "
        f"any later edit to this proposal's fields changes this hash).",
        "",
    ]
    return "\n".join(lines)
