"""The mechanism registry and index-only arm selection
(FORGE_SPEC_2026-09-01.md Section 2c).

``ARMS`` is the library of search arms a Forge cycle may select from or
extend: a tuple of ``Arm`` entries, one per mechanism that already ships
in ``scripts/go_explore_solve.py`` as an opt-in, default-off flag.
Modeled on the armed-signal + activity-counter pattern of ``Mechanism``
(``scripts/check_mechanism_receipt.py:95-118``) -- same ``activity_kind``
vocabulary (cumulative/nonzero/distinct/ladder), new field names, not the
same class: a registry entry also carries the flag surface, the knob
defaults, which wall classes it addresses, and its own trial history, none
of which ``Mechanism`` models.

``select_arm(bundle, arms=ARMS)`` is the index-only selector (LLM-guided-
exploration rules 1-3: an index cannot name an arm that does not exist).
Tonight's implementation is a deterministic table lookup on the bundle's
``mechanism_class`` list, not an agentic call -- the agentic version is a
two-way door (FORGE_SPEC Section 6).

Every value this module returns is JSON-safe. Nothing here touches the
emulator, the archive, or CLAIMS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Arm:
    """One registry entry. Field order matches the JSON shape pre-
    registered in FORGE_SPEC_2026-09-01.md Section 2c."""

    name: str
    kind: str
    flag: str
    off_value: str
    knobs: dict
    armed_signal: Optional[str]
    activity_counter: Optional[str]
    activity_kind: Optional[str]
    gate_fn: Optional[str]
    inertness_test: str
    forge_entry: Optional[str]
    status: str
    wall_classes: tuple
    history: tuple = field(default_factory=tuple)


#: Certainty ranking a selector reads to prefer a confirmed mechanism
#: class over a merely-candidate one when a bundle lists more than one
#: (contra needs two at once, SD finding 13). Lower is more trusted.
_CERTAINTY_RANK = {"confirmed_by_receipt": 0, "candidate": 1, "not_probed": 2}

#: Statuses that mean "registered for the record, not pickable tonight."
#: An arm carrying one of these is never preferred over a live peer that
#: matches the same wall class, even though it still counts toward the
#: "3 seeded arms, each auditable or marked UNAUDITABLE" inventory.
_HALTED_STATUSES = {"K0-HALTED", "UNAUDITABLE"}

#: Deterministic per-arm primary value -- the top-level flag setting a
#: selection proposes when this arm is picked (ortho's `up`/`down`,
#: gate-opener's `enumerate`, lock's four named objectives). Recorded
#: under the `"mode"` key of the returned `knobs` dict, the same key
#: `arms_tried`/`history` entries use (CLAIMS.md:199-242's `--ortho up`
#: trial), so a redundancy check can compare like against like without
#: needing to know each arm's own flag name.
_DEFAULT_PRIMARY_VALUE = {
    "ortho": "up",
    "lock": "survival",
    "gate_opener": "enumerate",
}

ARMS: tuple[Arm, ...] = (
    Arm(
        name="ortho", kind="arm", flag="--ortho", off_value="off",
        knobs={"--ortho-pin-secs": 120.0, "--ortho-bias": 0.30,
               "--ortho-band": 1, "--ortho-weight": 4.0,
               "--ortho-macro-p": 0.0},
        armed_signal="ortho_armed()", activity_counter="ortho_selections",
        activity_kind="cumulative", gate_fn="ortho_cols_improved>=8",
        inertness_test=(
            "tests/test_go_explore_solve.py::"
            "test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults"),
        forge_entry="CLAIMS.md:199", status="FORGE-PENDING-VALIDATION",
        wall_classes=("KEY_BLIND",),
        # The real, receipted cv_hall trial (CLAIMS.md:223-242): armed
        # (37,345 selections, FIRED), MECHANISM VALIDATED as a selection-
        # pressure treatment, PREMISE STALE overall (the control also
        # reached y-band 7; the pre-registered ortho_cols_improved>=8
        # partial gate read 3, not >=8, in both runs).
        history=({"wall_id": "cv_hall", "knobs": {"mode": "up"},
                   "verdict": "FIRED",
                   "outcome": "MECHANISM_VALIDATED/PREMISE_STALE",
                   "receipt": "CLAIMS.md:223-242"},),
    ),
    Arm(
        # `--lock-objective` (scripts/go_explore_solve.py:9058), not a
        # bare `--lock` -- the four merit functions ('yield'/'survival'/
        # 'novelty', plus 'off') share this one flag. Positional/survival
        # objective, not an interaction-discovery mechanism, so it is
        # never offered for KEY_BLIND -- the spatial-only arm the
        # class filter must keep out of a KEY_BLIND selection.
        name="lock", kind="arm", flag="--lock-objective", off_value="off",
        knobs={"--lock-pin-secs": 300.0, "--lock-band": 0,
               "--lock-weight": 4.0, "--lock-survival-scale": 64.0,
               "--lock-novelty-scale": 4.0},
        armed_signal="lock_armed()",
        # progress_line() (scripts/go_explore_solve.py:8480-8498) emits
        # lock_mode/lock_pinned_secs/lock_armed_secs/lock_band_cells/
        # lock_cells unconditionally under lock_mode!="off", and
        # lock_bursts_tracked only under mode=="yield" with cells in the
        # band -- none of those is a cumulative "the objective selected
        # N cells" counter the way ortho_selections is. Read at build
        # time (2026-09-02) and found absent: UNAUDITABLE and unarmable,
        # logged here rather than patched with a fabricated counter.
        activity_counter=None, activity_kind=None, gate_fn=None,
        inertness_test=(
            "tests/test_go_explore_solve.py::"
            "test_t1_the_default_lock_stream_is_byte_identical_with_the_arm_off"),
        forge_entry=None, status="UNAUDITABLE",
        wall_classes=("SCRIPTED_RELEASE", "OBSERVABLE_DEFECT"),
        history=(),
    ),
    Arm(
        # gate_armed()/gate_armed_secs measure how long the ARMING
        # CONDITION held, not what the mechanism DID to any cell -- the
        # same category error check_mechanism_receipt.py exists to catch
        # (an armed clock ticking is not evidence of activity). The real
        # per-candidate counters (admitted/candidates/sweeps/...) exist
        # in the shadow ledger but were never wired to a registered
        # arming signal for this registry, so neither half is claimed
        # here (PA Section 4; check_mechanism_receipt.py:26-40). K0
        # halted this arm before its own validation ran (K0 verdict:
        # "mechanically SOUND... but NOT K0-certifiable on available
        # targets") -- parked pickable per PA open question 1 option B,
        # sound-but-uncertified, not evidence the registry needs a
        # different interaction-discovery mechanism.
        name="gate_opener", kind="arm", flag="--gate-opener",
        off_value="off", knobs={},
        armed_signal=None, activity_counter=None, activity_kind=None,
        gate_fn=None,
        inertness_test=(
            "tests/test_go_explore_solve.py::"
            "test_gate_telemetry_is_absent_from_a_default_progress_line"),
        forge_entry=None, status="K0-HALTED",
        wall_classes=("KEY_BLIND",),
        history=(),
    ),
)


def is_auditable(arm: Arm) -> str:
    """``"AUDITABLE"`` iff the arm carries both an armed-evidence signal
    and an activity counter (check_mechanism_receipt.py's own rule: a
    mechanism with either half missing is structurally unauditable, no
    matter what its ``status`` field says); ``"UNAUDITABLE"`` otherwise.
    Always one of exactly these two strings -- the "3 seeded arms, each
    auditable or marked UNAUDITABLE" inventory (FORGE_SPEC Section 4 row
    (c)) is this function run over ``ARMS``, not a hand count."""
    have_both = bool(arm.armed_signal) and bool(arm.activity_counter)
    return "AUDITABLE" if have_both else "UNAUDITABLE"


def audit_violations(arms: tuple[Arm, ...] = ARMS) -> list[str]:
    """Names of every arm that has no activity counter yet carries a
    ``status`` that does not admit it -- i.e. a status outside
    ``_HALTED_STATUSES``, which reads as a live/armed claim with no
    counter behind it to audit. Empty on a well-formed registry; the
    exact defect ``check_mechanism_receipt.py``'s UNAUDITABLE verdict
    exists to catch, applied to the registry's own static entries
    instead of a run's log."""
    return [a.name for a in arms
            if a.activity_counter is None and a.status not in _HALTED_STATUSES]


def _classes_by_certainty(bundle: dict) -> list[tuple[str, str]]:
    """``[(class, certainty), ...]`` from ``bundle["mechanism_class"]``,
    most-trusted certainty first; ties keep the bundle's own order."""
    entries = [(c.get("class"), c.get("certainty", "not_probed"))
               for c in bundle.get("mechanism_class", []) if c.get("class")]
    return sorted(entries, key=lambda ce: _CERTAINTY_RANK.get(ce[1], 3))


def _eligible(cls: str, arms: tuple[Arm, ...]) -> list[int]:
    """Indices of every arm whose ``wall_classes`` names ``cls`` -- the
    class filter. Deleting this call (returning every index regardless
    of ``cls``) is exactly the corruption
    test_select_on_key_blind_never_picks_spatial_only_arm guards: a
    KEY_BLIND bundle would then also match `lock` (SCRIPTED_RELEASE,
    OBSERVABLE_DEFECT only), the spatial-only arm the filter exists to
    keep out."""
    return [i for i, a in enumerate(arms) if cls in a.wall_classes]


def _knob_matches(a_knobs: dict, b_knobs: dict) -> bool:
    """Exact match on the `"mode"` axis both `arms_tried`/`history`
    entries and this module's own proposals key on -- the primary,
    coarse-grained knob-space axis (LE rule 14: novelty against tried
    arms). Two proposals with the same mode are the same trial for
    redundancy purposes regardless of any secondary knob's value."""
    return a_knobs.get("mode") == b_knobs.get("mode") and "mode" in a_knobs


def _redundant_with(wall_id: Optional[str], arm: Arm, knobs: dict,
                     bundle: dict) -> list[str]:
    """Arm names whose recorded trial (bundle `arms_tried`, or this
    arm's own registered `history` for the same wall) already carries
    this exact `knobs["mode"]` selection. Checks both sources: the
    bundle's own `arms_tried` (populated by piece (b) when a wall's
    manifest-derived history is wired in) and the registry's own
    per-arm `history` (populated here at seed time from CLAIMS.md,
    independent of whatever a given bundle happens to carry) -- a real
    prior trial is caught whichever side supplies it."""
    hits: list[str] = []
    for entry in bundle.get("arms_tried", []):
        if entry.get("arm") != arm.name:
            continue
        if _knob_matches(entry.get("knobs", {}), knobs):
            hits.append(arm.name)
            break
    if arm.name not in hits:
        for entry in arm.history:
            if wall_id is not None and entry.get("wall_id") != wall_id:
                continue
            if _knob_matches(entry.get("knobs", {}), knobs):
                hits.append(arm.name)
                break
    return hits


def _pin_secs_flag(arm: Arm) -> Optional[str]:
    for flag in arm.knobs:
        if flag.endswith("-pin-secs"):
            return flag
    return None


def select_arm(bundle: dict, arms: tuple[Arm, ...] = ARMS) -> dict:
    """Index-only selection: the most promising arm for ``bundle``, by a
    deterministic table lookup on ``bundle["mechanism_class"]`` (LLM-
    guided-exploration rules 1-3 -- three bounded calls in the agentic
    version this stands in for tonight: index, knobs, novelty).

    Returns ``{"index": i, "knobs": {...}, "redundant_with": [...],
    "reason_codes": [...]}``. ``index`` is always an int in
    ``range(len(arms))``, never a name. ``knobs`` carries the picked
    arm's primary flag value under `"mode"` plus its registered
    secondary knobs, with the `*-pin-secs` knob (if the arm has one)
    overridden from ``bundle["cell_rate_history"]``'s observed window
    span when that data is present -- expressible from bundle fields
    only, never a bundle-independent guess.
    """
    reason_codes: list[str] = []
    ranked_classes = _classes_by_certainty(bundle)

    eligible: list[int] = []
    for cls, certainty in ranked_classes:
        idxs = _eligible(cls, arms)
        if idxs:
            eligible = idxs
            reason_codes.append(f"class_match:{cls}:{certainty}")
            break
        reason_codes.append(f"no_arm_for_class:{cls}:{certainty}")

    if not eligible:
        # No class in this bundle has a registered arm at all. Falls
        # back to the first registered arm rather than raising, so a
        # caller always gets a valid index; not exercised by tonight's
        # two real bundles (both resolve above) or by any enumerated
        # test.
        reason_codes.append("fallback:no_class_match")
        index = 0
    else:
        # Prefer a live (non-halted) arm over a parked one for the same
        # class; ties break on registration order, so the lookup stays
        # a deterministic table read, not a preference model.
        live = [i for i in eligible if arms[i].status not in _HALTED_STATUSES]
        pool = live or eligible
        index = pool[0]
        for i in eligible:
            if i == index:
                continue
            reason_codes.append(
                f"excluded:{arms[i].name}:status={arms[i].status}")

    arm = arms[index]
    reason_codes.append(f"picked:{arm.name}:status={arm.status}")

    knobs = dict(arm.knobs)
    mode = _DEFAULT_PRIMARY_VALUE.get(arm.name)
    if mode is not None:
        knobs["mode"] = mode

    pin_flag = _pin_secs_flag(arm)
    crh = bundle.get("cell_rate_history") or {}
    rows = crh.get("data") or []
    if pin_flag and len(rows) >= 2:
        first_t, last_t = rows[0].get("elapsed_s"), rows[-1].get("elapsed_s")
        if isinstance(first_t, (int, float)) and isinstance(last_t, (int, float)):
            span = round(max(0.0, last_t - first_t), 1)
            if span > 0:
                knobs[pin_flag] = span
                reason_codes.append(f"knob_from_bundle:{pin_flag}={span}")

    wall_id = bundle.get("wall_id")
    redundant_with = _redundant_with(wall_id, arm, knobs, bundle)
    if redundant_with:
        reason_codes.append(f"redundant_with:{','.join(redundant_with)}")

    return {"index": index, "knobs": knobs, "redundant_with": redundant_with,
            "reason_codes": reason_codes}
