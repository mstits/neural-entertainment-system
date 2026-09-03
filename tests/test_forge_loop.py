"""src/forge/loop.py and src/forge/proposal.py -- the propose, pilot,
gate, refute loop (FORGE_SPEC_2026-09-01.md Section 3).

Every pilot here is a stub ``block_runner``/``reading_fn`` pair, never a
real subprocess or a real progress.jsonl -- ``src/forge/block.py``'s own
test file already proves the real launch/watchdog mechanics; this file
proves the loop's ORCHESTRATION: what it does with whatever a block
returns, never re-proving the block runner itself. Each test's own
docstring names the corruption it is revert-verified against.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge import loop as loop_mod  # noqa: E402
from src.forge import proposal as proposal_mod  # noqa: E402
from src.forge.registry import Arm  # noqa: E402

FIXTURE_ARM = Arm(
    name="ortho", kind="arm", flag="--ortho", off_value="off",
    knobs={"--ortho-pin-secs": 120.0, "--ortho-bias": 0.30},
    armed_signal="ortho_armed()", activity_counter="ortho_selections",
    activity_kind="cumulative", gate_fn="ortho_cols_improved>=8",
    inertness_test=("tests/test_go_explore_solve.py::"
                     "test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults"),
    forge_entry="CLAIMS.md:199", status="FORGE-PENDING-VALIDATION",
    wall_classes=("KEY_BLIND",), history=(),
)

#: An arm with no secondary knob at all (mirrors the registry's own
#: `gate_opener` entry, knobs={}) -- design_proposals can only ever
#: produce one point in its knob space (`{"mode": "enumerate"}`), so a
#: prior trial at that same mode is a literal repeat with nothing for
#: `repair_proposal`'s redundancy repair (which only ever touches a
#: `*-pin-secs` flag) to move away from.
FIXTURE_ARM_NO_SECONDARY_KNOBS = Arm(
    name="gate_opener", kind="arm", flag="--gate-opener", off_value="off",
    knobs={}, armed_signal="gate_armed()", activity_counter="gate_armed_secs",
    activity_kind="cumulative", gate_fn="gate_admitted>=1",
    inertness_test=("tests/test_go_explore_solve.py::"
                     "test_gate_telemetry_is_absent_from_a_default_progress_line"),
    forge_entry=None, status="FORGE-PENDING-VALIDATION",
    wall_classes=("KEY_BLIND",), history=(),
)

#: Same shape, but the prior trial lives only in the registry's own
#: per-arm `history` (populated at seed time, independent of any given
#: bundle) -- never in a bundle's `arms_tried`.
FIXTURE_ARM_WITH_HISTORY = Arm(
    name="gate_opener", kind="arm", flag="--gate-opener", off_value="off",
    knobs={}, armed_signal="gate_armed()", activity_counter="gate_armed_secs",
    activity_kind="cumulative", gate_fn="gate_admitted>=1",
    inertness_test=("tests/test_go_explore_solve.py::"
                     "test_gate_telemetry_is_absent_from_a_default_progress_line"),
    forge_entry=None, status="FORGE-PENDING-VALIDATION",
    wall_classes=("KEY_BLIND",),
    history=({"wall_id": "cv_hall", "knobs": {"mode": "enumerate"},
              "verdict": "FIRED", "outcome": "MECHANISM_FAIL/PREMISE_STALE",
              "receipt": "CLAIMS.md:1-1"},),
)

BUNDLE = {
    "wall_id": "cv_hall",
    "mechanism_class": [{"class": "KEY_BLIND", "certainty": "confirmed_by_receipt",
                          "receipt": "docs/proposals/gate_opener_arm_2026-08-11.md:139-215"}],
    "cell_rate_history": {"certainty": "confirmed_by_receipt", "data": []},
    "arms_tried": [],
    "missing": [],
}

BUDGET = {"max_secs": 1200, "max_steps": 2_000_000, "seed": 1,
          "control": "same seed, flag off"}


def _receipt(**overrides) -> dict:
    base = {"stop": "budget", "aborted": False, "watchdog_trips": 0,
            "banked": [], "ratio_ok": False}
    base.update(overrides)
    return base


def _counting_block_runner(receipt_fn=None):
    calls = []

    def run(plan: dict) -> dict:
        calls.append(plan)
        return receipt_fn(plan) if receipt_fn else _receipt()

    run.calls = calls
    return run


# ---------------------------------------------------------------------
# forge_gate -- the deterministic three-stage split verdict
# ---------------------------------------------------------------------

def _gate_proposal(threshold=8, metric="ortho_cols_improved"):
    return {"gate": {"stage1": "activity_counter FIRED, n_obs>0",
                      "stage2": {"metric": metric, "threshold": threshold,
                                 "vs": "matched_control"},
                      "stage3": ["frontier>prior_best replay-verified"]}}


def test_gate_void_when_counter_inert():
    """Stage 1 must gate stage 2 shut: n_obs==0 (counter INERT) reads
    VOID even though the metric reading, taken alone, would clear the
    threshold. Corrupt: skip stage 1 and jump straight to the stage-2
    comparison -- reads MECHANISM_VALIDATED (delta 90 >= threshold 5)
    instead of VOID, failing this test.
    """
    proposal = _gate_proposal(threshold=5)
    pilot = {"n_obs": 0, "metric_value": 100, "stage3_true": False}
    control = {"n_obs": 5, "metric_value": 10, "stage3_true": False}
    gate = loop_mod.forge_gate(proposal, pilot, control)
    assert gate["stage1"] == "VOID"
    assert gate["stage2"] is None
    assert gate["verdict"] == "VOID"


def test_gate_validated_only_over_threshold():
    """The delta must clear the threshold with `>=`, at the exact
    boundary -- 8 >= 8 validates, 7 does not. Corrupt: `>=` to `>` ->
    the exact-threshold case (delta==8) now reads MECHANISM_FAIL
    instead of MECHANISM_VALIDATED, failing this test.
    """
    proposal = _gate_proposal(threshold=8)
    control = {"n_obs": 5, "metric_value": 10, "stage3_true": False}

    at_threshold = {"n_obs": 5, "metric_value": 18, "stage3_true": False}
    gate = loop_mod.forge_gate(proposal, at_threshold, control)
    assert gate["stage2"] == "MECHANISM_VALIDATED"
    assert gate["verdict"] == "MECHANISM_VALIDATED"

    below_threshold = {"n_obs": 5, "metric_value": 17, "stage3_true": False}
    gate = loop_mod.forge_gate(proposal, below_threshold, control)
    assert gate["stage2"] == "MECHANISM_FAIL"
    assert gate["verdict"] == "MECHANISM_FAIL"


def test_gate_reports_control_crossing_as_falsifier():
    """A stage-3 clause the matched control ALSO crosses is reported as
    a falsifier, never credited to the candidate as a pass -- `verdict`
    stays derived from stage 2 alone. Corrupt: credit the candidate
    (e.g. `if falsifier: verdict = MECHANISM_VALIDATED`) -- the FAIL-
    shaped stage-2 reading below would then read MECHANISM_VALIDATED,
    failing this test.
    """
    proposal = _gate_proposal(threshold=5)
    pilot = {"n_obs": 5, "metric_value": 10, "stage3_true": True}
    control = {"n_obs": 5, "metric_value": 10, "stage3_true": True}
    gate = loop_mod.forge_gate(proposal, pilot, control)
    assert gate["stage2"] == "MECHANISM_FAIL"
    assert gate["stage3"] == "PREMISE_CROSSED"
    assert gate["control_crossed"] is True
    assert gate["falsifier"] is True
    assert gate["verdict"] == "MECHANISM_FAIL"


# ---------------------------------------------------------------------
# refute_proposal -- the four fixed refuter checks
# ---------------------------------------------------------------------

def test_refuter_rejects_address_literal_in_proposal():
    """A ROM-address literal in a proposal's free-text field is a
    purity violation -- the refuter must reject it regardless of how
    clean the rest of the proposal is. Corrupt: drop the address-
    literal regex -- the fixture below stops being rejected, failing
    this test.
    """
    prop = {
        "knobs": {"mode": "up", "--ortho-pin-secs": 60.0},
        "why_tried_arms_dont_fit": (
            "the archive stalls near the boundary at 0x0450, which the "
            "existing arms do not probe"),
        "gate": {"stage2": {"metric": "ortho_cols_improved",
                             "threshold": 8, "vs": "matched_control"}},
        "budget": BUDGET,
    }
    refute = proposal_mod.refute_proposal(prop, arms_tried=[], budget=BUDGET,
                                           arm_name="ortho")
    assert refute["rejected"] is True
    assert any(r.startswith("purity") for r in refute["reasons"])


# ---------------------------------------------------------------------
# forge_cycle -- orchestration
# ---------------------------------------------------------------------

def test_cycle_void_when_null_candidate_not_void(tmp_path):
    """The null candidate (flag at off_value) must read VOID or the
    whole cycle reads VOID before any proposal is ever designed or
    piloted. Corrupt: drop the null run entirely (jump straight to the
    control/proposal loop) -- with a reading_fn that always reports
    n_obs>0 (below), the corrupted code would proceed past the null
    check the real code stops on, and `runner.calls` would exceed 1
    instead of stopping at exactly one block launch.
    """
    runner = _counting_block_runner()

    def always_armed_reading(receipt, *, arm, prop, out_dir):
        # Simulates a bug where the off-value run still reads as
        # armed -- the exact anti-vacuity failure the null candidate
        # exists to catch.
        return {"n_obs": 5, "metric_value": 1, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c1", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=always_armed_reading)

    assert result["cycle_verdict"] == "VOID"
    assert result["stop_reason"] == "null_candidate_not_void"
    assert len(runner.calls) == 1  # only the null block ever launched


def test_cycle_stops_after_two_repair_rounds(tmp_path, monkeypatch):
    """A proposal the refuter rejects on every attempt -- an arm with
    no secondary knob to vary (`knobs={}`, like the registry's own
    `gate_opener` entry), whose sole `mode` axis is already in
    arms_tried, so `repair_proposal`'s redundancy repair (which only
    ever touches a `*-pin-secs` flag) has nothing to change -- is
    refuted exactly ``1 + max_repair_rounds`` times, then discarded
    without ever reaching a pilot. Corrupt: loop to 3 repair rounds
    (widen the cap by one) -- the call count below becomes 4 instead
    of 3, failing this test.
    """
    calls = {"n": 0}
    real_refute = proposal_mod.refute_proposal

    def counting_refute(prop, **kwargs):
        calls["n"] += 1
        return real_refute(prop, **kwargs)

    monkeypatch.setattr(proposal_mod, "refute_proposal", counting_refute)

    runner = _counting_block_runner()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    # arms_tried already carries `gate_opener mode=enumerate` -- the
    # same, and only, knob-space point design_proposals can ever
    # produce for an arm with no secondary knob.
    arms_tried = [{"arm": "gate_opener", "knobs": {"mode": "enumerate"}}]

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM_NO_SECONDARY_KNOBS], 0, cycle_id="c2",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=null_reads_void,
        arms_tried=arms_tried, k=1, max_repair_rounds=2)

    assert calls["n"] == 3  # 1 initial + 2 repair rounds
    assert len(runner.calls) == 2  # null + control only, no pilot
    assert result["proposals"][0]["discard_stage"] == "refuter"
    assert result["cycle_verdict"] == "ALL_VOID_OR_FAIL"

    discards_path = tmp_path / "cv_hall" / "c2" / "discards.jsonl"
    rows = [json.loads(l) for l in discards_path.read_text().splitlines()]
    assert rows[0]["stage"] == "refuter"
    assert rows[0]["rounds"] == 3
    assert rows[0]["status"] == "FORGE-VOID"


def test_arms_tried_at_arm_level_does_not_block_knob_reparameterization(tmp_path):
    """Section 4 row (d) / step 3: cv_hall's arms_tried already carries
    `{arm: ortho, knobs: {mode: up}}` (the ortho ARM-level trial,
    CLAIMS.md:223-242) -- a coarse mode-only record with no secondary
    knob value, not evidence any particular pin-secs/bias/band point
    was tried. Redundancy is knob-space DISTANCE (spec Section 3), not
    a `mode`-label match: `design_proposals`'s knob proposals (which
    also carry `--ortho-pin-secs`, absent from this history entry)
    must not be rejected as redundant against it, so the cv_hall live
    plan's knob pilots on `ortho` can actually run. Corrupt: revert
    `_redundancy_violation` to compare `mode` alone -- all three
    proposals would be rejected for redundancy and `runner.calls`
    would stay at 2 (null + control only), never reaching a pilot.
    """
    runner = _counting_block_runner()

    def reading_fn(receipt, *, arm, prop, out_dir):
        if prop.get("is_null"):
            return {"n_obs": 0, "metric_value": 0, "stage3_true": False}
        return {"n_obs": 5, "metric_value": 20, "stage3_true": False}

    arms_tried = [{"arm": "ortho", "knobs": {"mode": "up"}}]

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c6", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=reading_fn, arms_tried=arms_tried, k=3)

    assert len(runner.calls) == 3  # null + control + a pilot actually ran
    assert result["cycle_verdict"] == "MECHANISM_VALIDATED"


def test_refuter_checks_arm_history_not_only_arms_tried(tmp_path):
    """Spec Section 3: redundancy is checked against arms_tried/history
    -- the registry's OWN per-arm `history` (populated at seed time,
    independent of whatever a bundle's `arms_tried` happens to carry;
    registry.py's `_redundant_with` already reads both) must also gate
    the refuter, not only the bundle's `arms_tried`. Here `arms_tried`
    is empty; only `FIXTURE_ARM_WITH_HISTORY.history` records the prior
    `mode=enumerate` trial. Corrupt: drop the `arm.history` merge in
    `forge_cycle` (pass only `arms_tried` to `refute_proposal`) --
    nothing would block the pilot, `runner.calls` would become 3
    instead of 2, and `discard_stage` would read `None`.
    """
    runner = _counting_block_runner()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM_WITH_HISTORY], 0, cycle_id="c7",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=null_reads_void,
        arms_tried=[], k=1, max_repair_rounds=2)

    assert len(runner.calls) == 2  # null + control only, no pilot ever ran
    assert result["proposals"][0]["discard_stage"] == "refuter"
    assert any(r.startswith("redundancy")
               for r in result["proposals"][0]["discard_reason"])


def test_arm_proposal_requires_inertness_mirror_before_pilot(tmp_path, monkeypatch):
    """A ``kind:"arm"`` proposal carrying a patch with no
    ``tests_added`` must never reach a pilot -- Section 4 row (d)'s own
    FAIL condition. Corrupt: pilot anyway -- ``runner.calls`` would
    include the pilot block (3 calls instead of 2), and
    ``discard_stage`` would read ``None`` (gated normally) instead of
    ``"arm_precheck"``, failing this test.
    """
    bad_arm_proposal = {
        "kind": "arm", "arm_index": 0, "knobs": {"mode": "up"},
        "wall_class_addressed": "KEY_BLIND",
        "why_tried_arms_dont_fit": "existing arms miss this class",
        "patch": "diff --git a/x b/x\n+ pass\n",
        "tests_added": [],  # missing -- must block the pilot
        "inertness_mirror": FIXTURE_ARM.inertness_test,
        "gate": {"stage1": "activity_counter FIRED, n_obs>0",
                 "stage2": {"metric": "ortho_cols_improved", "threshold": 8,
                            "vs": "matched_control"},
                 "stage3": []},
        "kill": ["counter INERT"], "budget": BUDGET, "tier3_sentence": None,
        "is_null": False,
    }
    monkeypatch.setattr(proposal_mod, "design_proposals",
                         lambda *a, **k: [bad_arm_proposal])

    runner = _counting_block_runner()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c3", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=null_reads_void, k=1)

    assert len(runner.calls) == 2  # null + control only, no pilot
    assert result["proposals"][0]["discard_stage"] == "arm_precheck"
    assert result["cycle_verdict"] == "ALL_VOID_OR_FAIL"


def test_arm_proposal_with_no_patch_never_reaches_pilot(tmp_path, monkeypatch):
    """A ``kind:"arm"`` proposal with ``patch: None`` must be blocked
    the same as one with an empty inertness mirror or empty
    ``tests_added`` -- there is nothing for the byte-identity or
    inertness-mirror check to run against a patch that does not exist.
    Corrupt: revert the precheck condition to ``if prop.get("patch")
    and (missing_mirror or no_tests)`` -- a patchless arm proposal
    would then skip the precheck entirely (the ``and`` short-circuits
    on the falsy ``patch``) and reach a pilot: ``runner.calls`` would
    become 3 instead of 2, and ``discard_stage`` would read ``None``.
    """
    bad_arm_proposal = {
        "kind": "arm", "arm_index": 0, "knobs": {"mode": "up"},
        "wall_class_addressed": "KEY_BLIND",
        "why_tried_arms_dont_fit": "existing arms miss this class",
        "patch": None,  # no patch at all
        "tests_added": ["tests/test_x.py::test_y"],
        "inertness_mirror": FIXTURE_ARM.inertness_test,
        "gate": {"stage1": "activity_counter FIRED, n_obs>0",
                 "stage2": {"metric": "ortho_cols_improved", "threshold": 8,
                            "vs": "matched_control"},
                 "stage3": []},
        "kill": ["counter INERT"], "budget": BUDGET, "tier3_sentence": None,
        "is_null": False,
    }
    monkeypatch.setattr(proposal_mod, "design_proposals",
                         lambda *a, **k: [bad_arm_proposal])

    runner = _counting_block_runner()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c3b", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=null_reads_void, k=1)

    assert len(runner.calls) == 2  # null + control only, no pilot
    assert result["proposals"][0]["discard_stage"] == "arm_precheck"
    assert result["cycle_verdict"] == "ALL_VOID_OR_FAIL"


def test_gate_discard_rows_also_carry_forge_void_marker(tmp_path):
    """Spec Section 3: 'Discards are logged as FORGE-VOID rows with
    stage and reason' -- covers the gate-stage discard call site too
    (a proposal that reaches the gate but does not validate), not only
    the refuter-stage one `test_cycle_stops_after_two_repair_rounds`
    already checks. Corrupt: drop the `{"status": "FORGE-VOID", **row}`
    merge in `_write_discard` -- the `status` assertion below fails.
    """
    runner = _counting_block_runner()

    def reading_fn(receipt, *, arm, prop, out_dir):
        if prop.get("is_null"):
            return {"n_obs": 0, "metric_value": 0, "stage3_true": False}
        # Pilot metric never clears the threshold -> MECHANISM_FAIL.
        return {"n_obs": 5, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c11", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=reading_fn, k=1)

    assert result["cycle_verdict"] == "ALL_VOID_OR_FAIL"
    discards_path = tmp_path / "cv_hall" / "c11" / "discards.jsonl"
    rows = [json.loads(l) for l in discards_path.read_text().splitlines()]
    assert rows[0]["stage"] == "gate"
    assert rows[0]["status"] == "FORGE-VOID"


# ---------------------------------------------------------------------
# Extra coverage (not spec-named): freeze/tamper detection and the
# stop-on-first-validated path, both load-bearing for Section 4 row (d).
# ---------------------------------------------------------------------

def test_freeze_then_edit_changes_hash():
    """`verify_frozen` must catch ANY post-freeze edit -- the Section 4
    row (d) FAIL condition ("a proposal edited after pilot start (hash
    mismatch)"). Corrupt: freeze only a subset of fields (e.g. only
    `knobs`) -- editing `why_tried_arms_dont_fit` after freezing would
    then NOT change the hash, failing this test.
    """
    prop = proposal_mod.null_candidate(FIXTURE_ARM, "cv_hall", BUDGET)
    sha = proposal_mod.freeze_proposal(prop)
    assert proposal_mod.verify_frozen(prop, sha) is True

    prop["why_tried_arms_dont_fit"] = prop["why_tried_arms_dont_fit"] + " edited"
    assert proposal_mod.verify_frozen(prop, sha) is False


def test_cycle_registers_first_validated_proposal_and_stops(tmp_path):
    """Stop condition (1): the cycle stops at the first
    MECHANISM_VALIDATED proposal and never pilots the remaining ones.
    Corrupt: keep piloting after a validated proposal -- `runner.calls`
    would exceed 3 (null + control + 1 pilot) instead of stopping
    there, failing this test.
    """
    runner_state = {"n": 0}

    def receipt_fn(plan):
        runner_state["n"] += 1
        return _receipt()

    runner = _counting_block_runner(receipt_fn)

    def reading_fn(receipt, *, arm, prop, out_dir):
        if prop.get("is_null"):
            return {"n_obs": 0, "metric_value": 0, "stage3_true": False}
        # control (no arm_index, is_null True already handled above);
        # any pilot proposal reads a comfortably validating metric.
        return {"n_obs": 5, "metric_value": 20, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c4", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=reading_fn, k=3)

    assert result["cycle_verdict"] == "MECHANISM_VALIDATED"
    assert len(runner.calls) == 3  # null + control + exactly one pilot
    assert result["registration"] is not None


def test_freeze_verify_raises_on_mutation_during_pilot(tmp_path, monkeypatch):
    """The in-cycle hash-mismatch check must actually fire inside
    `forge_cycle`'s own control flow, not only when called directly on
    a proposal object in isolation: if the exact object `design_
    proposals` returned (and `forge_cycle` froze) is mutated while its
    pilot block "runs" -- here, by a block_runner that closes over the
    same monkeypatched proposals list -- `forge_cycle` must raise
    `FrozenProposalError` rather than silently gating a proposal that
    was not the one piloted. Corrupt: replace the `if not proposal_mod.
    verify_frozen(prop, frozen_sha256):` check with `if False:` --
    this test would then not raise, failing.
    """
    tampered_prop = {
        "kind": "knob", "arm_index": 0,
        "knobs": {"mode": "up", "--ortho-pin-secs": 60.0},
        "wall_class_addressed": "KEY_BLIND",
        "why_tried_arms_dont_fit": "x", "patch": None, "tests_added": [],
        "inertness_mirror": FIXTURE_ARM.inertness_test,
        "gate": {"stage1": "activity_counter FIRED, n_obs>0",
                 "stage2": {"metric": "ortho_cols_improved", "threshold": 8,
                            "vs": "matched_control"},
                 "stage3": []},
        "kill": [], "budget": BUDGET, "tier3_sentence": None, "is_null": False,
    }
    monkeypatch.setattr(proposal_mod, "design_proposals",
                         lambda *a, **k: [tampered_prop])

    def tampering_runner(plan):
        if plan["cycle_id"].endswith("-pilot_0"):
            # The same object `forge_cycle` froze and is about to
            # verify -- a real mutation, not a hand-corrupted hash.
            tampered_prop["knobs"]["mode"] = "tampered-after-freeze"
        return _receipt()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    with pytest.raises(proposal_mod.FrozenProposalError):
        loop_mod.forge_cycle(
            "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c5", budget=BUDGET,
            cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
            block_runner=tampering_runner, reading_fn=null_reads_void, k=1)


def test_null_and_control_are_frozen_before_their_pilots(tmp_path):
    """Section 4 row (d) PASS: 'every proposal frozen before its
    pilot' -- `null_candidate()` returns a `kind:"knob"` proposal too,
    and the matched control pilots that exact object at the arm's
    off_value, so both must carry a `frozen_sha256` the same way a
    designed proposal does. Corrupt: skip freezing the null and
    control blocks -- `result["null"]["frozen_sha256"]` and
    `result["control"]["frozen_sha256"]` would be falsy, and
    `proposal_null.md`/`proposal_control.md` would not exist, failing
    the assertions below.
    """
    runner = _counting_block_runner()

    def null_reads_void(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c9", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=null_reads_void, k=1)

    assert result["null"]["frozen_sha256"]
    assert result["control"]["frozen_sha256"]
    cycle_dir = tmp_path / "cv_hall" / "c9"
    assert (cycle_dir / "proposal_null.md").exists()
    assert (cycle_dir / "proposal_control.md").exists()


def test_null_candidate_freeze_verify_raises_on_mutation(tmp_path, monkeypatch):
    """The same tamper check applies to the null candidate's own pilot,
    not only a designed proposal's. Corrupt: drop the null-candidate
    `verify_frozen` check (or its `control` counterpart) -- this test
    would then not raise.
    """
    tampered = {
        "kind": "knob", "arm_index": None, "knobs": {"mode": "off"},
        "wall_class_addressed": "null", "why_tried_arms_dont_fit": "x",
        "patch": None, "tests_added": [],
        "inertness_mirror": FIXTURE_ARM.inertness_test,
        "gate": {"stage1": "activity_counter FIRED, n_obs>0",
                 "stage2": {"metric": "ortho_cols_improved", "threshold": 8,
                            "vs": "matched_control"},
                 "stage3": []},
        "kill": ["counter INERT"], "budget": BUDGET, "tier3_sentence": None,
        "is_null": True,
    }
    monkeypatch.setattr(proposal_mod, "null_candidate", lambda *a, **k: tampered)

    def tampering_runner(plan):
        if plan["cycle_id"].endswith("-null"):
            tampered["knobs"]["mode"] = "tampered"
        return _receipt()

    def reading_fn(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    with pytest.raises(proposal_mod.FrozenProposalError):
        loop_mod.forge_cycle(
            "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c10", budget=BUDGET,
            cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
            block_runner=tampering_runner, reading_fn=reading_fn, k=1)


def test_knob_registration_writes_receipt_file(tmp_path):
    """Spec Section 3: a `kind:"knob"` proposal "appends its setting to
    the registry entry's history and the run's receipt." `Arm` is a
    frozen dataclass (registry.py), so nothing in this process can
    mutate `arm.history` in place -- the receipt FILE is what makes
    that sentence true, not only the in-memory return value. Corrupt:
    drop the `receipt_path.write_text(...)` call in `_register` --
    `registration["history_receipt_path"]` would point at a path that
    does not exist, failing the `.exists()` assertion below.
    """
    runner = _counting_block_runner()

    def reading_fn(receipt, *, arm, prop, out_dir):
        if prop.get("is_null"):
            return {"n_obs": 0, "metric_value": 0, "stage3_true": False}
        return {"n_obs": 5, "metric_value": 20, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c8", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path,
        block_runner=runner, reading_fn=reading_fn, k=1)

    assert result["cycle_verdict"] == "MECHANISM_VALIDATED"
    receipt_path = Path(result["registration"]["history_receipt_path"])
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text())
    assert payload["arm"] == "ortho"
    assert payload["history_entry"]["wall_id"] == "cv_hall"


def test_arm_registration_appends_row_to_proposed_claims_queue(tmp_path):
    """Spec Section 2e "Output paths": a rendered FORGE entry lands at
    ``runs/forge/<wall>/<cycle>/CLAIMS_ENTRY.md`` "plus a row in
    runs/engine/proposed_claims.jsonl" -- the queue
    ``scripts/engine_driver.py``'s own ``PROPOSED`` constant names.
    ``_register`` wrote only the entry file and never that row
    (FORGE-FIX-25). Corrupt: drop the
    ``with open(proposed_claims_path, "a")`` append in ``_register`` --
    the queue file would not exist, failing the first assertion below.
    """
    prop = {
        "kind": "arm", "arm_index": 0,
        "knobs": {"--ortho-pin-secs": 120.0, "--ortho-bias": 0.30},
        "gate": {"stage2": {"metric": "ortho_cols_improved",
                             "threshold": 8}},
        "inertness_mirror": FIXTURE_ARM.inertness_test,
    }
    gate = {"stage1": "ARMED", "stage2": "MECHANISM_VALIDATED",
            "stage3": "PREMISE_STALE", "control_crossed": False,
            "falsifier": False, "verdict": "MECHANISM_VALIDATED",
            "reason": None}
    cycle_dir = tmp_path / "runs" / "cv_hall" / "c1"
    queue_path = tmp_path / "engine" / "proposed_claims.jsonl"

    registration = loop_mod._register(
        prop, FIXTURE_ARM, 0, gate, wall_id="cv_hall", cycle_id="c1",
        cycle_dir=cycle_dir, commit="abc123", date="2026-09-02",
        frozen_sha256="deadbeef", proposed_claims_path=queue_path)

    assert queue_path.exists()
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["wall_id"] == "cv_hall"
    assert row["cycle_id"] == "c1"
    assert row["arm"] == "ortho"
    assert row["arm_index"] == 0
    assert row["frozen_sha256"] == "deadbeef"
    assert row["status"] == "FORGE-VALIDATED-MECHANISM"
    assert row["entry_path"] == registration["entry_path"]
    assert row["citable_as"] == (
        "ortho MECHANISM_VALIDATED on cv_hall; no clear may be "
        "attributed to it.")

    # A second validated registration on the same path appends, never
    # overwrites: the queue is a log the engine's landing step drains,
    # not a single-slot file.
    loop_mod._register(
        prop, FIXTURE_ARM, 0, gate, wall_id="cv_hall", cycle_id="c2",
        cycle_dir=tmp_path / "runs" / "cv_hall" / "c2", commit="abc124",
        date="2026-09-02", frozen_sha256="deadbeef2",
        proposed_claims_path=queue_path)
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["cycle_id"] == "c2"


def test_knob_registration_does_not_queue_a_proposed_claims_row(tmp_path):
    """A ``kind:"knob"`` proposal returns from ``_register`` before the
    ledger writer ever runs (no rendered entry exists for it to
    match), so it must not create the queue file either. Corrupt: move
    the queue-append above the knob branch's early return -- this test
    would then find a queue file that should not exist.
    """
    runner = _counting_block_runner()

    def reading_fn(receipt, *, arm, prop, out_dir):
        if prop.get("is_null"):
            return {"n_obs": 0, "metric_value": 0, "stage3_true": False}
        return {"n_obs": 5, "metric_value": 20, "stage3_true": False}

    queue_path = tmp_path / "engine" / "proposed_claims.jsonl"
    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c11", budget=BUDGET,
        cmd_prefix=["python3", "fixture.py"], out_base=tmp_path / "runs",
        block_runner=runner, reading_fn=reading_fn, k=1,
        proposed_claims_path=queue_path)

    assert result["cycle_verdict"] == "MECHANISM_VALIDATED"
    assert result["registration"]["kind"] == "knob"
    assert not queue_path.exists()


# ---------------------------------------------------------------------
# The cycle receipt on disk (FORGE-FIX-1)
# ---------------------------------------------------------------------

def _receipt_writing_block_runner(tmp_path, receipt_fn=None):
    """A stub block_runner that writes a real ``block_receipt.json``
    under the plan's own ``--out`` directory, the way ``run_block`` now
    does, so the cycle receipt has real paths to name. Still no
    subprocess: this file proves orchestration, never the runner."""
    calls = []

    def run(plan: dict) -> dict:
        calls.append(plan)
        out_dir = Path(plan["cmd"][plan["cmd"].index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        receipt = (receipt_fn(plan) if receipt_fn else _receipt())
        receipt = dict(receipt)
        receipt.update({
            "wall_id": plan["wall_id"], "cycle_id": plan["cycle_id"],
            "grant_entry": plan.get("grant_entry"),
            "attended_hours": 0.5, "run_lock_hours": 0.25,
            "fabricated_clears_unretracted": 0,
            "positive_control": {"injected": False, "caught": False,
                                  "banked_from_reset": 0},
        })
        path = out_dir / "block_receipt.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        receipt["receipt_path"] = str(path)
        return receipt

    run.calls = calls
    return run


def test_cycle_receipt_is_written_naming_every_block_receipt(tmp_path):
    """A cycle writes ``cycle_receipt.json`` beside its proposals,
    naming every block receipt path it ran, the cycle verdicts, and the
    grant fields a cycle is judged on (spec Section 4, live-validation
    point 4): the watchdog-trip sum, attended hours beside run-lock
    hours with the ratio, the positive control and the banked list.
    Before this, nothing anywhere wrote this file.

    Attended hours are the maximum across blocks, not the sum -- every
    block reads the same attended log, so summing would multiply one
    operator's hours by the block count.

    Revert-verify (live, this session): make ``forge_cycle``'s
    ``_finish`` return ``result`` without calling
    ``write_cycle_receipt`` and this fails by name on the first
    assertion -- the file does not exist. Second corruption: sum the
    blocks' attended hours instead of taking the max, and the
    ``attended_hours == 0.5`` assertion fails at 1.5. Both restored,
    re-passed.
    """
    runner = _receipt_writing_block_runner(tmp_path)

    def inert_reading(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c_receipt",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"],
        out_base=tmp_path, block_runner=runner, reading_fn=inert_reading,
        grant_entry="CLAIMS.md#FORGE-GRANT-cv_hall-2026-09-01",
        root_state_sha="00a93d0aae5b27d2", k=1, commit="4174d76",
        date="2026-09-02")

    path = Path(tmp_path) / "cv_hall" / "c_receipt" / "cycle_receipt.json"
    assert path.exists(), "the cycle ran and wrote no cycle receipt"
    assert result["cycle_receipt_path"] == str(path)
    receipt = json.loads(path.read_text())

    # Every block that ran is named by its own receipt's path, and each
    # of those files is really there.
    tags = [b["tag"] for b in receipt["blocks"]]
    assert tags == ["null", "control", "pilot_0"], tags
    assert receipt["block_receipt_paths"] == [b["receipt_path"]
                                              for b in receipt["blocks"]]
    for named in receipt["block_receipt_paths"]:
        assert named is not None and Path(named).exists(), named

    assert receipt["cycle_verdict"] == "ALL_VOID_OR_FAIL"
    assert receipt["stop_reason"] == "all_proposals_void_or_fail"
    assert receipt["null_gate"]["verdict"] == "VOID"
    assert receipt["grant_entry"] == "CLAIMS.md#FORGE-GRANT-cv_hall-2026-09-01"
    assert receipt["root_state_sha"] == "00a93d0aae5b27d2"
    assert receipt["commit"] == "4174d76"

    assert receipt["watchdog_trips_sum"] == 0
    assert receipt["fabricated_clears_unretracted"] == 0
    assert receipt["banked"] == []
    assert receipt["run_lock_hours"] == 0.75      # three blocks, summed
    assert receipt["attended_hours"] == 0.5       # one log, the max
    assert receipt["ratio_machine_per_attended"] == 1.5
    assert receipt["ratio_ok"] is False           # reported, never a refusal
    assert len(receipt["positive_control"]) == 3


def test_cycle_result_keeps_every_block_receipt_field(tmp_path):
    """The cycle result carries each block's whole receipt, not four
    fields of it. The grant judges a cycle on attended hours beside
    run-lock hours, the positive control and the banked list; keeping
    only ``stop``/``aborted``/``watchdog_trips`` threw all of those
    away before the caller ever saw them.

    Revert-verify (live, this session): put back the four-key literal
    (``{"receipt_stop": ..., "aborted": ..., "watchdog_trips": ...,
    "frozen_sha256": ...}``) for ``result["null"]`` and this fails by
    name on ``attended_hours``. Restored, re-passed.
    """
    runner = _receipt_writing_block_runner(tmp_path)

    def inert_reading(receipt, *, arm, prop, out_dir):
        return {"n_obs": 0, "metric_value": 0, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM], 0, cycle_id="c_fields",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"],
        out_base=tmp_path, block_runner=runner, reading_fn=inert_reading,
        k=1)

    for where in (result["null"], result["control"]):
        for field in ("attended_hours", "run_lock_hours", "banked",
                      "positive_control", "grant_entry", "receipt_path",
                      "fabricated_clears_unretracted"):
            assert field in where, f"{field} dropped from the cycle result"
        # The four keys the old shape kept are still there.
        assert where["receipt_stop"] == "budget"
        assert where["watchdog_trips"] == 0
        assert where["frozen_sha256"]

    pilot = result["proposals"][0]
    assert pilot["receipt"]["attended_hours"] == 0.5
    assert Path(pilot["receipt_path"]).exists()


# ---------------------------------------------------------------------
# FORGE-FIX-5: an unauditable arm is refused before any block launches
# ---------------------------------------------------------------------

#: An arm with no activity counter at all -- registry.is_auditable reads
#: UNAUDITABLE, and forge_gate(auditable=False) returns VOID before
#: n_obs is consulted, which is what makes the null control vacuous
#: unless the cycle refuses first.
FIXTURE_ARM_NO_COUNTER = Arm(
    name="lock", kind="arm", flag="--lock", off_value="off",
    knobs={"--lock-pin-secs": 120.0},
    armed_signal="lock_armed()", activity_counter=None,
    activity_kind="cumulative", gate_fn="lock_cols_improved>=8",
    inertness_test=("tests/test_go_explore_solve.py::"
                     "test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults"),
    forge_entry="CLAIMS.md:199", status="K0-HALTED",
    wall_classes=("KEY_BLIND",), history=(),
)

#: The registry's other half: a counter with no armed-evidence signal.
FIXTURE_ARM_NO_SIGNAL = Arm(
    name="lock", kind="arm", flag="--lock", off_value="off",
    knobs={"--lock-pin-secs": 120.0},
    armed_signal=None, activity_counter="lock_selections",
    activity_kind="cumulative", gate_fn="lock_cols_improved>=8",
    inertness_test=("tests/test_go_explore_solve.py::"
                     "test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults"),
    forge_entry="CLAIMS.md:199", status="K0-HALTED",
    wall_classes=("KEY_BLIND",), history=(),
)


def test_gate_reads_unauditable_before_n_obs_is_consulted():
    """auditable=False is a stage-1 VOID with reason UNAUDITABLE, taken
    on a reading that would otherwise clear stage 2 outright. Corrupt:
    drop `not auditable or` from the stage-1 condition at loop.py:95 --
    the reading reads MECHANISM_VALIDATED (delta 90 >= threshold 5) and
    reason None, failing both asserts.
    """
    proposal = _gate_proposal(threshold=5)
    pilot = {"n_obs": 7, "metric_value": 100, "stage3_true": False}
    control = {"n_obs": 7, "metric_value": 10, "stage3_true": False}

    gate = loop_mod.forge_gate(proposal, pilot, control, auditable=False)

    assert gate["verdict"] == "VOID"
    assert gate["stage1"] == "VOID"
    assert gate["reason"] == "UNAUDITABLE"


def test_cycle_refuses_an_unauditable_arm_before_any_block_launches(tmp_path):
    """An arm with no activity counter cannot be audited, so its null
    control cannot fail: forge_gate returns VOID for it no matter what
    the block reads, and the "null must read VOID" check passes
    vacuously on exactly the arm where the control matters most. The
    cycle must refuse up front instead. Corrupt: delete the
    `arm_unauditable` refusal block in forge_cycle -- with a reading_fn
    that always reports n_obs>0 (the anti-vacuity failure the null
    exists to catch), the cycle walks past the null check and launches
    blocks, so `runner.calls` is non-empty and `stop_reason` is not
    `arm_unauditable`.
    """
    runner = _counting_block_runner()

    def always_armed_reading(receipt, *, arm, prop, out_dir):
        return {"n_obs": 5, "metric_value": 1, "stage3_true": False}

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM_NO_COUNTER], 0, cycle_id="c1",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"],
        out_base=tmp_path, block_runner=runner,
        reading_fn=always_armed_reading)

    assert result["cycle_verdict"] == "VOID"
    assert result["stop_reason"] == "arm_unauditable"
    assert runner.calls == []  # no block launched at all
    # The refusal is still a receipted cycle, not a bare return.
    assert Path(result["cycle_receipt_path"]).exists()


def test_cycle_refuses_an_arm_with_a_counter_but_no_armed_signal(tmp_path):
    """The registry's predicate is both halves. forge_cycle read only
    `bool(arm.activity_counter)`, so an arm the registry calls
    UNAUDITABLE was piloted as auditable. Corrupt: put
    `auditable = bool(arm.activity_counter)` back in place of the
    registry call -- this arm reads auditable, no refusal fires, and
    the cycle launches its null block.
    """
    runner = _counting_block_runner()
    from src.forge import registry as registry_mod
    assert registry_mod.is_auditable(FIXTURE_ARM_NO_SIGNAL) == "UNAUDITABLE"

    result = loop_mod.forge_cycle(
        "cv_hall", BUNDLE, [FIXTURE_ARM_NO_SIGNAL], 0, cycle_id="c1",
        budget=BUDGET, cmd_prefix=["python3", "fixture.py"],
        out_base=tmp_path, block_runner=runner)

    assert result["stop_reason"] == "arm_unauditable"
    assert runner.calls == []
