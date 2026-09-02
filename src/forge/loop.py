"""The Forge loop: propose, pilot, gate, review (FORGE_SPEC_2026-09-01.md
Section 3). Piece (d) calls piece (f)'s ``run_block`` for every pilot;
``run_block`` never calls back into this module.

``forge_gate`` is the deterministic three-stage split-verdict gate
(never a model): stage 1 ARMED/VOID on the arm's own activity counter,
stage 2 MECHANISM_VALIDATED/MECHANISM_FAIL on the pre-registered metric
beating a matched control by a pre-registered threshold, stage 3
PREMISE_CROSSED/PREMISE_STALE reported but never required for
registration. ``forge_cycle`` orchestrates one cycle over a wall's
registry selection: a null candidate (in-situ negative control, must
read VOID or the whole cycle is VOID), up to ``k`` designed proposals
each refuted/repaired/piloted/gated, and registration of the first
MECHANISM_VALIDATED survivor.

Every proposal is frozen (``proposal.freeze_proposal``) immediately
before its pilot and re-verified (``proposal.verify_frozen``)
immediately after -- a mismatch is the Section 4 row (d) FAIL
condition ("a proposal edited after pilot start") and raises rather
than silently gating a proposal that was not the one piloted.

``forge_cycle`` writes ``runs/forge/<wall>/<cycle>/cycle_receipt.json``
on every path that returns a result, naming every block receipt's own
path and the cycle's verdicts (spec Section 4, live-validation point
4). It also carries each block's whole receipt into the returned
result rather than four fields of it: the grant judges a cycle on
attended hours beside run-lock hours, the watchdog-trip sum, the
positive control and the banked list, and dropping those here left a
cycle unable to answer for itself.

Nothing here writes CLAIMS.md. Registration of a validated ``arm``
proposal renders a FORGE ledger entry (piece (e)) to
``runs/forge/<wall>/<cycle>/CLAIMS_ENTRY.md``; landing that text into
CLAIMS.md is a separate commit under the owner's name (spec Section
2e). Discards -- refuter rejections and non-validated gate readings
alike -- are appended to ``runs/forge/<wall>/<cycle>/discards.jsonl``
with their stage and reason, per spec Section 3's "never silently
retried under a new name."
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from src.forge import proposal as proposal_mod
from src.forge.block import RATIO_FLOOR as _RATIO_FLOOR
from src.forge.block import run_block as _real_run_block
from src.forge.ledger import render_entry as _real_render_entry

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_FORGE_RUNS_DIR = REPO / "runs" / "forge"

#: Stage-3 clause presence does not change registration -- a candidate
#: that crosses stage 3 while its matched control ALSO crosses it is
#: the taxonomy's own falsifier of the wall's classification, reported
#: as such, never credited to the candidate as a pass (spec Section 3;
#: test_gate_reports_control_crossing_as_falsifier).
STAGE3_CROSSED = "PREMISE_CROSSED"
STAGE3_STALE = "PREMISE_STALE"

VERDICT_VOID = "VOID"
VERDICT_VALIDATED = "MECHANISM_VALIDATED"
VERDICT_FAIL = "MECHANISM_FAIL"

CYCLE_VALIDATED = "MECHANISM_VALIDATED"
CYCLE_ALL_VOID_OR_FAIL = "ALL_VOID_OR_FAIL"
CYCLE_VOID = "VOID"


def forge_gate(proposal: Mapping[str, Any], pilot_reading: Mapping[str, Any],
                control_reading: Mapping[str, Any], *,
                auditable: bool = True) -> dict:
    """The three-stage split verdict (spec Section 3).

    ``pilot_reading``/``control_reading``: ``{"n_obs": int,
    "metric_value": number, "stage3_true": bool}`` -- readings this
    function treats as already-extracted telemetry (never itself reads
    a file); ``forge_cycle`` below is what extracts them from a block
    receipt.

    ``auditable=False`` (the arm has no ``activity_counter`` to read at
    all -- registry.py's UNAUDITABLE) forces stage 1 VOID before
    ``n_obs`` is even consulted, the same category error
    ``check_mechanism_receipt.py`` exists to catch: an arming clock
    ticking, or a knob simply being set, is not evidence of activity.

    Returns ``{"stage1": "ARMED"|"VOID", "stage2": ...|None,
    "stage3": ...|None, "control_crossed": bool, "falsifier": bool,
    "verdict": "VOID"|"MECHANISM_VALIDATED"|"MECHANISM_FAIL"}``.
    """
    n_obs = pilot_reading.get("n_obs", 0)
    if not auditable or not isinstance(n_obs, (int, float)) or n_obs <= 0:
        return {
            "stage1": VERDICT_VOID, "stage2": None, "stage3": None,
            "control_crossed": False, "falsifier": False,
            "verdict": VERDICT_VOID,
            "reason": "UNAUDITABLE" if not auditable else "INERT",
        }
    stage1 = "ARMED"

    stage2_cfg = proposal.get("gate", {}).get("stage2", {})
    threshold = stage2_cfg.get("threshold", 0)
    pilot_metric = pilot_reading.get("metric_value", 0) or 0
    control_metric = control_reading.get("metric_value", 0) or 0
    delta = pilot_metric - control_metric
    if isinstance(threshold, (int, float)) and delta >= threshold:
        stage2 = VERDICT_VALIDATED
    else:
        stage2 = VERDICT_FAIL
    verdict = stage2

    pilot_crossed = bool(pilot_reading.get("stage3_true"))
    control_crossed = bool(control_reading.get("stage3_true"))
    stage3 = STAGE3_CROSSED if pilot_crossed else STAGE3_STALE
    # A crossing the matched control also produces is the taxonomy's own
    # falsifier -- reported, and it never upgrades `verdict` above,
    # which is derived from stage 2 alone.
    falsifier = pilot_crossed and control_crossed

    return {
        "stage1": stage1, "stage2": stage2, "stage3": stage3,
        "control_crossed": control_crossed, "falsifier": falsifier,
        "verdict": verdict, "reason": None,
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _out_dir_for(wall_id: str, cycle_id: str, tag: str, base: Path) -> Path:
    return base / wall_id / cycle_id / tag


def _default_block_plan(*, wall_id: str, cycle_id: str, tag: str,
                         arm, prop: dict, budget: Mapping[str, Any],
                         grant_entry: Optional[str], root_state_sha: Optional[str],
                         out_dir: Path, cmd_prefix: Sequence[str]) -> dict:
    """The block plan for one pilot/control/null run. ``cmd_prefix`` is
    the caller-supplied solver invocation (script path, ``--config``,
    ``--root-state``, ``--seed``) this function appends the arm's flag
    and the proposal's knob values to -- callers own everything
    game/wall-specific; this function only knows the registry/proposal
    shapes.
    """
    cmd = list(cmd_prefix) + ["--out", str(out_dir)]
    mode = prop.get("knobs", {}).get(proposal_mod.MODE_KEY)
    if mode is not None and mode != arm.off_value:
        cmd += [arm.flag, str(mode)]
    for flag, value in prop.get("knobs", {}).items():
        if flag == proposal_mod.MODE_KEY:
            continue
        cmd += [flag, str(value)]
    return {
        "wall_id": wall_id, "cycle_id": f"{cycle_id}-{tag}",
        "cmd": cmd, "root_state_sha": root_state_sha,
        "max_secs": budget.get("max_secs", 1200),
        "max_steps": budget.get("max_steps", 2_000_000),
        "grant_entry": grant_entry,
        "attended_log": str(DEFAULT_FORGE_RUNS_DIR / "attended.jsonl"),
        "inject_wrongful_reset": False,
    }


def _default_reading_fn(block_receipt: dict, *, arm, prop: dict,
                         out_dir: Path) -> dict:
    """Real-use default: tails ``<out_dir>/progress.jsonl`` for the
    arm's ``activity_counter`` and the proposal's ``stage2.metric``
    field. Never consulted by the pure gate-logic tests below, which
    hand-build readings directly; exists so ``forge_cycle`` is usable
    against a real solver child once a grant exists, without every
    caller having to write its own extractor.
    """
    progress = out_dir / "progress.jsonl"
    row: dict = {}
    try:
        for line in reversed(Path(progress).read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                break
            except ValueError:
                continue
    except OSError:
        row = {}
    n_obs = row.get(arm.activity_counter, 0) if arm.activity_counter else 0
    metric_name = prop.get("gate", {}).get("stage2", {}).get("metric")
    metric_value = row.get(metric_name, 0) if metric_name else 0
    return {"n_obs": n_obs, "metric_value": metric_value,
            "stage3_true": False}


def _write_discard(discards_path: Path, row: dict) -> None:
    """Appends one FORGE-VOID row (spec Section 3: "Discards are logged
    as FORGE-VOID rows with stage and reason") -- the ``status`` field
    is set here, once, so every call site gets it and none can forget
    it."""
    discards_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"status": "FORGE-VOID", **row}
    with open(discards_path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


CYCLE_RECEIPT_NAME = "cycle_receipt.json"


def _block_row(tag: str, receipt: Mapping[str, Any]) -> dict:
    """One row of the cycle receipt's ``blocks`` list: the tag, the
    block receipt's own path on disk, and the fields the grant judges a
    cycle on. ``receipt_path`` is None for a caller-supplied
    ``block_runner`` that writes no receipt of its own -- reported as
    None, never silently omitted, so a cycle receipt cannot read as if
    every block had left one."""
    return {
        "tag": tag,
        "receipt_path": receipt.get("receipt_path"),
        "cycle_id": receipt.get("cycle_id"),
        "stop": receipt.get("stop"),
        "aborted": receipt.get("aborted"),
        "watchdog_trips": receipt.get("watchdog_trips"),
        "attended_hours": receipt.get("attended_hours"),
        "run_lock_hours": receipt.get("run_lock_hours"),
        "ratio_ok": receipt.get("ratio_ok"),
        "positive_control": receipt.get("positive_control"),
        "banked": receipt.get("banked"),
        "fabricated_clears_unretracted": receipt.get(
            "fabricated_clears_unretracted"),
        "root_state_sha256_before": receipt.get("root_state_sha256_before"),
        "root_state_sha256_after": receipt.get("root_state_sha256_after"),
    }


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def write_cycle_receipt(result: Mapping[str, Any], blocks: Sequence[tuple],
                         cycle_dir: Path, *, grant_entry: Optional[str],
                         root_state_sha: Optional[str], commit: str,
                         date: str) -> Path:
    """Write ``<cycle_dir>/cycle_receipt.json`` and return its path.

    Section 4's live-validation point 4 names this file and what it must
    carry: attended hours beside run-lock hours with the ratio check,
    the watchdog-trip sum, the positive control, zero unretracted
    fabricated clears, and the grant anchor. It also names every block
    receipt's own path, so the cycle receipt is an index into the block
    receipts rather than a summary that replaces them.

    Attended hours are taken as the MAXIMUM across the blocks, not the
    sum: every block in a cycle reads the same ``attended_log``, so
    summing would multiply one operator's hours by the block count. The
    maximum is that log's latest reading.
    """
    rows = [_block_row(tag, receipt) for tag, receipt in blocks]
    run_lock_hours = round(sum(_num(r["run_lock_hours"]) for r in rows), 4)
    attended_hours = max([_num(r["attended_hours"]) for r in rows] or [0.0])
    ratio = (round(run_lock_hours / attended_hours, 4)
             if attended_hours > 0 else None)
    receipt = {
        "wall_id": result.get("wall_id"),
        "cycle_id": result.get("cycle_id"),
        "arm": result.get("arm"),
        "grant_entry": grant_entry,
        "root_state_sha": root_state_sha,
        "commit": commit,
        "date": date,
        "written": _now_iso(),
        "blocks": rows,
        "block_receipt_paths": [r["receipt_path"] for r in rows],
        "attended_hours": attended_hours,
        "run_lock_hours": run_lock_hours,
        "ratio_machine_per_attended": ratio,
        "ratio_ok": bool(ratio is not None and ratio >= _RATIO_FLOOR),
        "watchdog_trips_sum": sum(int(_num(r["watchdog_trips"])) for r in rows),
        "fabricated_clears_unretracted": sum(
            int(_num(r["fabricated_clears_unretracted"])) for r in rows),
        "banked": [b for r in rows for b in (r["banked"] or [])],
        "positive_control": [
            {"tag": r["tag"], "positive_control": r["positive_control"]}
            for r in rows],
        "cycle_verdict": result.get("cycle_verdict"),
        "stop_reason": result.get("stop_reason"),
        "null_gate": (result.get("null") or {}).get("gate"),
        "control_stop": (result.get("control") or {}).get("stop"),
        "proposals": result.get("proposals"),
        "registration": result.get("registration"),
    }
    cycle_dir = Path(cycle_dir)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    path = cycle_dir / CYCLE_RECEIPT_NAME
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def _write_proposal_file(base: Path, wall_id: str, cycle_id: str, index: int,
                          prop: dict, frozen_sha256: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"proposal_{index}.md"
    path.write_text(proposal_mod.render_proposal_markdown(
        prop, wall_id=wall_id, cycle_id=cycle_id, index=index,
        frozen_sha256=frozen_sha256))
    return path


def forge_cycle(
    wall_id: str,
    bundle: Mapping[str, Any],
    arms: Sequence[Any],
    arm_index: int,
    *,
    cycle_id: str,
    budget: Mapping[str, Any],
    cmd_prefix: Sequence[str],
    grant_entry: Optional[str] = None,
    root_state_sha: Optional[str] = None,
    arms_tried: Sequence[Mapping[str, Any]] = (),
    k: int = 3,
    max_repair_rounds: int = 2,
    out_base: Optional[Path] = None,
    claims_path: Optional[Path] = None,
    grant_state_path: Optional[Path] = None,
    block_runner: Optional[Callable[[dict], dict]] = None,
    reading_fn: Optional[Callable[..., dict]] = None,
    commit: str = "PENDING",
    date: Optional[str] = None,
) -> dict:
    """Runs one full propose→pilot→gate→refute cycle against
    ``arms[arm_index]`` for ``wall_id``. Returns a JSON-safe result:

    ``{"wall_id", "cycle_id", "arm": name, "cycle_verdict":
    "MECHANISM_VALIDATED"|"ALL_VOID_OR_FAIL"|"VOID",
    "stop_reason": str, "null": {...}, "control": {...},
    "proposals": [{"index", "frozen_sha256", "repair_rounds",
    "gate": {...}|None, "discard_stage": str|None, "receipt": {...},
    "receipt_path": str|None}, ...],
    "registration": {...}|None, "cycle_receipt_path": str}``

    Stop conditions (spec Section 3): (1) a proposal reads
    MECHANISM_VALIDATED -- stop, register, return. (2) every proposal
    is VOID/FAIL after its repair rounds -- ``ALL_VOID_OR_FAIL``. (3)
    any block reports ``watchdog_trips>0`` or ``aborted:true`` -- the
    cycle is VOID, nothing registers, no later proposal is piloted.
    (5, folded into 3) ``grant_state.json`` reading ``GRANT_ENDED`` is
    exactly a block that refuses and reports ``aborted:true`` --
    ``run_block`` already encodes this, so no separate check is needed
    here. The null candidate not reading VOID is a distinct anti-vacuity
    failure of the cycle itself, checked first, before any proposal is
    even designed.
    """
    block_runner = block_runner or _real_run_block
    reading_fn = reading_fn or _default_reading_fn
    out_base = Path(out_base) if out_base is not None else DEFAULT_FORGE_RUNS_DIR
    cycle_dir = out_base / wall_id / cycle_id
    discards_path = cycle_dir / "discards.jsonl"
    arm = arms[arm_index]

    result: dict = {
        "wall_id": wall_id, "cycle_id": cycle_id, "arm": arm.name,
        "cycle_verdict": None, "stop_reason": None,
        "null": None, "control": None, "proposals": [],
        "registration": None, "cycle_receipt_path": None,
    }

    #: (tag, block receipt) for every block this cycle ran, in order.
    #: The cycle receipt is built from these, so a cycle that stops
    #: early still names the blocks that did run.
    blocks: list = []

    def _finish() -> dict:
        path = write_cycle_receipt(
            result, blocks, cycle_dir, grant_entry=grant_entry,
            root_state_sha=root_state_sha, commit=commit,
            date=date or time.strftime("%Y-%m-%d"))
        result["cycle_receipt_path"] = str(path)
        return result

    # --- null candidate: in-situ negative control ----------------------
    # Frozen before its pilot exactly like a designed proposal (spec
    # Section 4 row (d) PASS: "every proposal frozen before its pilot"
    # -- null_candidate() returns a kind:"knob" proposal too).
    null_prop = proposal_mod.null_candidate(arm, wall_id, budget)
    null_frozen_sha256 = proposal_mod.freeze_proposal(null_prop)
    _write_proposal_file(cycle_dir, wall_id, cycle_id, "null", null_prop,
                          null_frozen_sha256)
    null_out = _out_dir_for(wall_id, cycle_id, "null", out_base)
    null_plan = _default_block_plan(
        wall_id=wall_id, cycle_id=cycle_id, tag="null", arm=arm,
        prop=null_prop, budget=budget, grant_entry=grant_entry,
        root_state_sha=root_state_sha, out_dir=null_out,
        cmd_prefix=cmd_prefix)
    null_receipt = block_runner(null_plan)
    blocks.append(("null", null_receipt))
    if not proposal_mod.verify_frozen(null_prop, null_frozen_sha256):
        raise proposal_mod.FrozenProposalError(
            f"null candidate proposal was mutated during its pilot "
            f"(sha256 no longer matches {null_frozen_sha256})")
    # The whole block receipt, not four fields of it: everything the
    # grant fixes -- attended and run-lock hours, the ratio, the
    # positive control, banked, the root-state digests -- was dropped
    # here before, so a cycle result could not answer the questions the
    # grant asks even when the block had answered them.
    result["null"] = dict(null_receipt)
    result["null"].update({"receipt_stop": null_receipt.get("stop"),
                            "frozen_sha256": null_frozen_sha256})
    if null_receipt.get("aborted") or (null_receipt.get("watchdog_trips") or 0) > 0:
        result["cycle_verdict"] = CYCLE_VOID
        result["stop_reason"] = "null_block_aborted"
        return _finish()

    auditable = bool(arm.activity_counter)
    null_reading = reading_fn(null_receipt, arm=arm, prop=null_prop,
                               out_dir=null_out)
    null_gate = forge_gate(null_prop, null_reading, null_reading,
                            auditable=auditable)
    result["null"]["gate"] = null_gate
    if null_gate["verdict"] != VERDICT_VOID:
        result["cycle_verdict"] = CYCLE_VOID
        result["stop_reason"] = "null_candidate_not_void"
        return _finish()

    # --- shared matched control (flag at off_value, same seed/budget) --
    # Pilots the same frozen proposal object as the null candidate (the
    # control IS the arm held at off_value); frozen and verified again
    # around its own, separate pilot.
    control_frozen_sha256 = proposal_mod.freeze_proposal(null_prop)
    _write_proposal_file(cycle_dir, wall_id, cycle_id, "control", null_prop,
                          control_frozen_sha256)
    control_out = _out_dir_for(wall_id, cycle_id, "control", out_base)
    control_plan = _default_block_plan(
        wall_id=wall_id, cycle_id=cycle_id, tag="control", arm=arm,
        prop=null_prop, budget=budget, grant_entry=grant_entry,
        root_state_sha=root_state_sha, out_dir=control_out,
        cmd_prefix=cmd_prefix)
    control_receipt = block_runner(control_plan)
    blocks.append(("control", control_receipt))
    if not proposal_mod.verify_frozen(null_prop, control_frozen_sha256):
        raise proposal_mod.FrozenProposalError(
            f"matched control proposal was mutated during its pilot "
            f"(sha256 no longer matches {control_frozen_sha256})")
    result["control"] = dict(control_receipt)
    result["control"].update({"receipt_stop": control_receipt.get("stop"),
                               "frozen_sha256": control_frozen_sha256})
    if control_receipt.get("aborted") or (control_receipt.get("watchdog_trips") or 0) > 0:
        result["cycle_verdict"] = CYCLE_VOID
        result["stop_reason"] = "control_block_aborted"
        return _finish()
    control_reading = reading_fn(control_receipt, arm=arm, prop=null_prop,
                                  out_dir=control_out)

    # --- designed proposals ---------------------------------------------
    proposals = proposal_mod.design_proposals(bundle, arm, arm_index,
                                               wall_id, k=k, budget=budget)

    # Redundancy is checked against BOTH sources spec Section 3 names:
    # the bundle's own arms_tried (populated by piece (b)) AND the
    # registry entry's own `history` (populated at seed time,
    # independent of any given bundle -- registry.py's
    # `_redundant_with` already reads both; the refuter previously read
    # only `arms_tried`, silently missing a real prior trial recorded
    # solely in `arm.history`, e.g. the seeded ortho `CLAIMS.md:223-242`
    # entry when a bundle carries no `arms_tried` of its own).
    history_as_tried = [{"arm": arm.name, "knobs": h.get("knobs", {})}
                         for h in arm.history]
    combined_tried = list(arms_tried) + history_as_tried

    for index, prop in enumerate(proposals):
        rounds = 0
        while True:
            refute = proposal_mod.refute_proposal(
                prop, arms_tried=combined_tried, budget=budget,
                arm_name=arm.name)
            if not refute["rejected"]:
                break
            rounds += 1
            if rounds > max_repair_rounds:
                entry = {"proposal_index": index, "stage": "refuter",
                         "reason": refute["reasons"], "rounds": rounds,
                         "t": _now_iso()}
                _write_discard(discards_path, entry)
                result["proposals"].append({
                    "index": index, "frozen_sha256": None,
                    "repair_rounds": rounds, "gate": None,
                    "discard_stage": "refuter",
                    "discard_reason": refute["reasons"],
                })
                prop = None
                break
            prop = proposal_mod.repair_proposal(prop, refute["reasons"])

        if prop is None:
            continue  # exhausted repair rounds; next proposal

        # Frozen immediately before the pilot, per spec Section 3.
        frozen_sha256 = proposal_mod.freeze_proposal(prop)
        _write_proposal_file(cycle_dir, wall_id, cycle_id, index, prop,
                              frozen_sha256)

        if prop.get("kind") == "arm":
            # Presence-only precheck (spec Section 3: "must pass
            # default-off byte-identity and the inertness mirror before
            # any pilot" -- schema-supported and fixture-tested
            # tonight, per Section 3's own "live only if Section 5
            # finishes early": this never RUNS the named inertness
            # test or a byte-identity diff, it only refuses to pilot a
            # kind:"arm" proposal missing any of the three fields that
            # would make either check meaningful). A missing `patch`
            # blocks the pilot exactly like a missing mirror or empty
            # `tests_added` -- an arm proposal with no patch at all has
            # nothing for either check to run against, so it must not
            # skip the precheck and reach a pilot by default.
            missing_mirror = not prop.get("inertness_mirror")
            no_tests = not prop.get("tests_added")
            no_patch = not prop.get("patch")
            if no_patch or missing_mirror or no_tests:
                entry = {"proposal_index": index, "stage": "arm_precheck",
                          "reason": ["arm proposal missing patch, inertness "
                                     "mirror, or tests_added before pilot"],
                          "rounds": rounds, "t": _now_iso()}
                _write_discard(discards_path, entry)
                result["proposals"].append({
                    "index": index, "frozen_sha256": frozen_sha256,
                    "repair_rounds": rounds, "gate": None,
                    "discard_stage": "arm_precheck",
                    "discard_reason": entry["reason"],
                })
                continue

        pilot_out = _out_dir_for(wall_id, cycle_id, f"pilot_{index}", out_base)
        pilot_plan = _default_block_plan(
            wall_id=wall_id, cycle_id=cycle_id, tag=f"pilot_{index}",
            arm=arm, prop=prop, budget=budget, grant_entry=grant_entry,
            root_state_sha=root_state_sha, out_dir=pilot_out,
            cmd_prefix=cmd_prefix)
        pilot_receipt = block_runner(pilot_plan)
        blocks.append((f"pilot_{index}", pilot_receipt))

        # Defense-in-depth, not (yet) reachable through the default
        # block_runner/reading_fn pair (block.py's real run_block gets
        # only `pilot_plan` -- a dict of cmd/paths/budget derived from
        # `prop`, never `prop` itself -- and nothing re-reads
        # proposal_<k>.md after this point), per the Section 4 row (d)
        # FAIL condition ("a proposal edited after pilot start (hash
        # mismatch)"): a caller-supplied block_runner or reading_fn
        # that DOES hold a reference into the same object graph
        # design_proposals returned (see
        # test_freeze_verify_raises_on_mutation_during_pilot) is caught
        # here rather than silently gating a proposal that was not the
        # one piloted.
        if not proposal_mod.verify_frozen(prop, frozen_sha256):
            raise proposal_mod.FrozenProposalError(
                f"proposal {index} was mutated after freeze "
                f"(sha256 no longer matches {frozen_sha256})")

        if pilot_receipt.get("aborted") or (pilot_receipt.get("watchdog_trips") or 0) > 0:
            result["cycle_verdict"] = CYCLE_VOID
            result["stop_reason"] = f"proposal_{index}_block_aborted"
            result["proposals"].append({
                "index": index, "frozen_sha256": frozen_sha256,
                "repair_rounds": rounds, "gate": None,
                "discard_stage": "block_abort", "discard_reason": None,
                "receipt": dict(pilot_receipt),
                "receipt_path": pilot_receipt.get("receipt_path"),
            })
            return _finish()

        pilot_reading = reading_fn(pilot_receipt, arm=arm, prop=prop,
                                    out_dir=pilot_out)
        gate = forge_gate(prop, pilot_reading, control_reading,
                           auditable=auditable)

        entry_row = {
            "index": index, "frozen_sha256": frozen_sha256,
            "repair_rounds": rounds, "gate": gate,
            "discard_stage": None, "discard_reason": None,
            "receipt": dict(pilot_receipt),
            "receipt_path": pilot_receipt.get("receipt_path"),
        }
        result["proposals"].append(entry_row)

        if gate["verdict"] == VERDICT_VALIDATED:
            registration = _register(
                prop, arm, arm_index, gate, wall_id=wall_id,
                cycle_id=cycle_id, cycle_dir=cycle_dir, commit=commit,
                date=date or time.strftime("%Y-%m-%d"),
                frozen_sha256=frozen_sha256)
            result["registration"] = registration
            result["cycle_verdict"] = CYCLE_VALIDATED
            result["stop_reason"] = f"proposal_{index}_validated"
            return _finish()

        stage = "gate"
        entry_row["discard_stage"] = stage
        entry_row["discard_reason"] = [gate.get("reason") or gate["verdict"]]
        _write_discard(discards_path, {
            "proposal_index": index, "stage": stage,
            "reason": entry_row["discard_reason"],
            "frozen_sha256": frozen_sha256, "t": _now_iso(),
        })

    result["cycle_verdict"] = CYCLE_ALL_VOID_OR_FAIL
    result["stop_reason"] = "all_proposals_void_or_fail"
    return _finish()


def _register(prop: dict, arm, arm_index: int, gate: dict, *,
              wall_id: str, cycle_id: str, cycle_dir: Path, commit: str,
              date: str, frozen_sha256: str,
              render_entry_fn: Optional[Callable[[Mapping[str, Any]], str]] = None) -> dict:
    """Renders the FORGE ledger entry for a MECHANISM_VALIDATED
    proposal via piece (e) and writes it to
    ``runs/forge/<wall>/<cycle>/CLAIMS_ENTRY.md`` -- never to
    CLAIMS.md itself (spec Section 2e: landing is a separate commit
    under the owner's name). A ``kind:"arm"`` proposal is additionally
    recorded as an ``ARMS`` registry entry appendix
    (``registration["arms_entry"]``) for the landing step to apply. A
    ``kind:"knob"`` proposal's ``history`` addendum is written to a
    receipt file, ``runs/forge/<wall>/<cycle>/knob_history_receipt.json``
    (spec Section 3: "a knob proposal appends its setting to the
    registry entry's history and the run's receipt") -- ``Arm`` is a
    frozen dataclass (registry.py), so nothing in this process can
    mutate ``arm.history`` in place; the receipt file, not an in-memory
    mutation, is what a later landing step reads to actually apply the
    addendum, the same division of labor as ``arms_entry`` below for a
    validated ``kind:"arm"`` proposal.
    """
    registration: dict = {
        "kind": prop.get("kind"), "arm": arm.name, "gate": gate,
        "frozen_sha256": frozen_sha256,
    }
    if prop.get("kind") == "knob":
        history_entry = {
            "wall_id": wall_id, "knobs": prop.get("knobs", {}),
            "verdict": gate["stage1"],
            "outcome": f"{gate['stage2']}/{gate['stage3']}",
            "receipt": f"runs/forge/{wall_id}/{cycle_id}/",
        }
        cycle_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = cycle_dir / "knob_history_receipt.json"
        receipt_path.write_text(json.dumps({
            "arm": arm.name, "history_entry": history_entry,
            "frozen_sha256": frozen_sha256,
        }, sort_keys=True, indent=2) + "\n")
        registration["history_entry"] = history_entry
        registration["history_receipt_path"] = str(receipt_path)
        return registration

    render_entry_fn = render_entry_fn or _real_render_entry
    telemetry = [wall_id, arm.activity_counter or "n/a",
                 prop.get("gate", {}).get("stage2", {}).get("metric", "n/a")]
    entry_text = render_entry_fn({
        "status": "FORGE-VALIDATED-MECHANISM",
        "arm": arm.name, "flag": arm.flag, "commit": commit, "date": date,
        "detection": f"telemetry read for {wall_id}; {arm.name} selected "
                      f"by registry.select_arm on the wall's bundle.",
        "mechanism": f"knobs {prop.get('knobs', {})}",
        "review": "refuter checks (purity, redundancy, vacuity, budget) "
                  "passed before this proposal was frozen and piloted.",
        "gate": f"stage1={gate['stage1']} stage2={gate['stage2']} "
               f"inertness_mirror={prop.get('inertness_mirror')}",
        "telemetry": telemetry,
        "gate_met": True,
        "gate_measure": f"{prop.get('gate', {}).get('stage2', {}).get('metric')} "
                        f">= {prop.get('gate', {}).get('stage2', {}).get('threshold')} "
                        f"vs matched control",
        "citable_as": f"{arm.name} MECHANISM_VALIDATED on {wall_id}; no "
                      f"clear may be attributed to it.",
        "addenda": [],
    })
    cycle_dir.mkdir(parents=True, exist_ok=True)
    entry_path = cycle_dir / "CLAIMS_ENTRY.md"
    entry_path.write_text(entry_text)
    registration["entry_path"] = str(entry_path)
    registration["arms_entry"] = {
        "name": arm.name, "history_addendum": {
            "wall_id": wall_id, "knobs": prop.get("knobs", {}),
            "verdict": gate["stage1"],
            "outcome": f"{gate['stage2']}/{gate['stage3']}",
            "receipt": str(entry_path),
        },
    }
    return registration
