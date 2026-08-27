#!/usr/bin/env python3
"""Did the mechanisms a run ARMED ever actually FIRE?

Six vacuous gates shipped in this project before anyone noticed, and every
one of them passed a text rule. A treatment that is declared in the config,
printed in the banner, and named in the pre-registration can still do
literally nothing for an entire run — and nothing downstream will say so.
The measured cases:

  * ReDo logged `dormant fc1 0/96 fc2 0/32 recycled 0 cum 0` on ~2,000
    per-iteration checks across all 8 v27+v28 runs. It was registered as one
    of two variables. It was never applied. Both experiments were
    single-variable arms and were not described as such.
  * `demo_anchor_loss` is 0.0 on 250/250 rows in all eight of those runs —
    a number that was already sitting in `metrics.jsonl`, unread.
  * `symlog_rewards: true` and an entropy floor were armed and inert.

The rule this script enforces is mechanical and needs no prose:

    A MECHANISM THAT IS ARMED AND WHOSE OWN COUNTER NEVER MOVES FOR AN
    ENTIRE RUN MAKES THAT RUN **VOID** FOR ANY CLAIM THAT MECHANISM WAS
    A VARIABLE IN.

VOID is not FAIL (the ledger is explicit about this). A VOID run's other
arms may be perfectly sound — v27 and v28 both keep their FAIL verdicts,
because neither verdict depended on ReDo doing anything. What VOID forbids
is the SENTENCE: "we tested X and it did not help" when X never ran.

The checker refuses to certify what it cannot see. Four verdicts:

    NOT_ARMED     the mechanism was not enabled in this run. Nothing to say.
    FIRED         armed, and its counter moved. Certified live.
    INERT         armed, its counter was observed N>0 times, and it never
                  moved. -> VOID
    UNAUDITABLE   armed, and this run emits NO counter for it at all. This is
                  NOT a pass. `HazardMask` accumulates
                  `actions_vetoed`/`n_fully_vetoed` on every apply() and its
                  own docstring says "a veto you cannot audit is a veto you
                  cannot trust" — and nothing reads them, so an armed run
                  prints `[hazard-mask] ARMED ...` and then never reports a
                  single veto. An armed mechanism with no counter is the raw
                  material of the next vacuous gate.

The INERT/UNAUDITABLE split is the checker's own anti-vacuity property. A
counter at zero is only evidence when the counter was actually read: zero
observations of a counter is ignorance, not a null result, and this script
reports it as ignorance.

Usage
-----
    check_mechanism_receipt.py RUN_DIR [RUN_DIR ...] [--require redo,sil]
                              [--json] [--log PATH]

RUN_DIR may be a checkpoint dir (holding `metrics.jsonl` and/or `run.log`)
or a run dir holding `train_seed*.log`. Exit 0 when every armed mechanism
fired and every `--require`d mechanism fired; exit 2 otherwise (VOID);
exit 1 on a usage/IO error, including "found nothing to read", which must
never be reported as a pass.

Examples that reproduce known findings:

    # ReDo, on the campaign that registered it as a variable:
    scripts/check_mechanism_receipt.py runs/v27_fresh_recovery
      -> redo INERT (recycles 0 over 251 observations)  => VOID

    # The 1-2 campaign, whose registered mechanisms were all genuinely live:
    scripts/check_mechanism_receipt.py checkpoints/mario_1_2_online_v2
      -> sil FIRED, kl_anchor FIRED, backward FIRED
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Mechanism registry
# ---------------------------------------------------------------------------
#
# Each entry names, for ONE treatment, the evidence that it was armed and the
# COUNTER that proves it did something. Both halves are required. A mechanism
# with no counter is registered anyway, with `activity_*` left None, so it
# reports UNAUDITABLE instead of silently disappearing from the report --
# absence from a checker's output is exactly how an unaudited mechanism stays
# unaudited.
#
# `activity_log` regexes must capture exactly ONE numeric group.


@dataclass(frozen=True)
class Mechanism:
    name: str
    what: str
    # --- armed evidence -----------------------------------------------------
    armed_log: Optional[str] = None
    armed_metric: Optional[str] = None
    disarmed_log: Optional[str] = None
    # --- activity evidence (the counter that must move) ---------------------
    # "cumulative": the captured number is a running total; it moved if the
    #   maximum observed is > 0.
    # "nonzero": each observation is an instantaneous value; it moved if any
    #   observation is non-zero.
    # "distinct": it moved if more than one distinct value was observed (for
    #   schedules, where "did it walk?" is the question, not "was it > 0?").
    # "ladder": a schedule that counts DOWN to a terminal 0. Live if it
    #   either walked (>1 distinct rung) or is sitting at its terminal rung.
    #   Entrance-pinned consolidation legitimately arms the backward ladder
    #   and holds it at tau=0 for the whole run -- armed-and-arrived, not
    #   armed-and-inert. A ladder frozen at a NON-zero rung for an entire run
    #   is the real defect and still reads INERT.
    activity_log: Optional[str] = None
    activity_metric: Optional[str] = None
    activity_kind: str = "cumulative"
    note: str = ""


MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        name="redo",
        what="ReDo dormant-neuron recycling",
        armed_log=r"\[redo\] ENABLED",
        disarmed_log=r"\[redo\] disabled",
        activity_log=r"\[redo\] iter \d+:.*?\bcum (\d+)\b",
        activity_kind="cumulative",
        note=(
            "Registered as one of two variables in v27 AMENDMENT 1. Logged "
            "cum 0 on every check in all 8 v27+v28 runs. Its dormancy "
            "statistic normalizes post-activation magnitudes by the layer "
            "mean, but TilePolicyNetwork LayerNorms immediately BEFORE the "
            "SiLU, holding the statistic near 1 -- the registered tau=0.025 "
            "was below the reachable range before launch."
        ),
    ),
    Mechanism(
        name="demo_anchor",
        what="behaviour-cloning anchor term in the PPO loss",
        armed_metric="demo_anchor_coef",
        activity_metric="demo_anchor_loss",
        activity_kind="nonzero",
        note="0.0 on 250/250 rows in all eight v27+v28 runs, already in "
             "metrics.jsonl and unread until the 2026-08-27 audit.",
    ),
    Mechanism(
        name="sil",
        what="self-imitation on the agent's own clears",
        armed_metric="sil_loss",
        activity_metric="sil_clears_total",
        activity_kind="cumulative",
        note="Load-bearing for the 1-2 campaign; verifiably live there "
             "(sil_clears_total reached 3,596).",
    ),
    Mechanism(
        name="kl_anchor",
        what="KL anchor to the offline prior",
        armed_metric="kl_anchor_beta",
        activity_metric="kl_anchor_div",
        activity_kind="nonzero",
    ),
    Mechanism(
        name="wave_terminal",
        what="wavefront terminal/truncation rule (NOT the PBRS reward term)",
        armed_log=r"\[wavefront\] \d+ cells",
        activity_metric="wave_truncations",
        activity_kind="nonzero",
        note=(
            "NAMED FOR WHAT ITS COUNTER MEASURES. `wave_truncations` counts "
            "the wave TERMINAL rule firing; it says nothing about the "
            "monotone-invariant PBRS shaping term, which emits no counter "
            "and is therefore unaudited. A FIRED here must not be read as "
            "certifying the shaping. (This entry originally used "
            "wave_truncations as BOTH its arming and its activity signal, "
            "which made it structurally incapable of ever returning INERT "
            "-- a vacuous entry inside an anti-vacuity checker. The "
            "`[wavefront] N cells` build line is a genuinely independent "
            "arming signal.)"
        ),
    ),
    Mechanism(
        name="backward",
        what="backward-curriculum restart ladder",
        armed_log=r"\[backward\] ENABLED",
        activity_log=r"\[backward\] iter \d+: tau=(\d+)/\d+",
        activity_kind="ladder",
        note="'Did the ladder WALK, or is it already home?' -- a ladder "
             "frozen at a NON-zero rung for a whole run is armed and inert, "
             "which a plain >0 test would miss; one held at tau=0 is "
             "entrance-pinned consolidation working as designed.",
    ),
    Mechanism(
        name="kernel_adv",
        what="kernel adversary (phase-4 hardening)",
        armed_log=r"\[kernel_adv\] ON",
        activity_log=r"\[kernel_adv\] iter \d+: repeat_frac=([0-9.]+)",
        activity_kind="nonzero",
    ),
    Mechanism(
        name="hazard_mask",
        what="hazard-probability action veto",
        armed_log=r"\[hazard-mask\] ARMED",
        activity_log=None,  # deliberately absent -- see below
        note=(
            "NO COUNTER IS EMITTED. HazardMask.stats accumulates "
            "actions_vetoed / n_fully_vetoed / veto_fraction on every "
            "apply(), and nothing in trainer.py or eval_game.py reads "
            ".stats or calls .as_dict(). An armed run announces itself and "
            "then never reports a single veto. Reported UNAUDITABLE, not "
            "FIRED: this is a defect, not a pass."
        ),
    ),
    Mechanism(
        name="commitment",
        what="action-commitment options",
        armed_log=r"\[commitment\] ARMED",
        activity_log=None,  # deliberately absent
        note="Armed with no counter, same shape as hazard_mask.",
    ),
)

MECHANISMS_BY_NAME = {m.name: m for m in MECHANISMS}

NOT_ARMED = "NOT_ARMED"
FIRED = "FIRED"
INERT = "INERT"
UNAUDITABLE = "UNAUDITABLE"

# The verdicts that make a run VOID for any claim naming that mechanism.
VOIDING = (INERT, UNAUDITABLE)


@dataclass
class Reading:
    """What one mechanism's counter did across one run."""

    name: str
    verdict: str
    armed_evidence: Optional[str] = None
    observations: int = 0
    peak: Optional[float] = None
    detail: str = ""
    note: str = ""


@dataclass
class RunReport:
    run: str
    sources: list[str] = field(default_factory=list)
    readings: list[Reading] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def void_for(self) -> list[str]:
        return [r.name for r in self.readings if r.verdict in VOIDING]


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

def discover_sources(run_dir: Path, extra_log: Optional[Path] = None) -> dict:
    """Find the log text and the metrics rows for a run.

    Deliberately permissive about layout (`run.log`, `train_seed*.log`,
    `checkpoints/*/metrics.jsonl`) and deliberately strict about the result:
    a run with neither a log nor metrics is an ERROR, never a pass.
    """
    run_dir = Path(run_dir)
    logs: list[Path] = []
    metrics_files: list[Path] = []
    if extra_log is not None:
        logs.append(Path(extra_log))
    if run_dir.is_file():
        (metrics_files if run_dir.suffix == ".jsonl" else logs).append(run_dir)
    elif run_dir.is_dir():
        logs.extend(sorted(run_dir.glob("*.log")))
        metrics_files.extend(sorted(run_dir.glob("metrics.jsonl")))

    log_text = ""
    used: list[Path] = []
    for p in logs:
        try:
            log_text += p.read_text(errors="replace")
            used.append(p)
        except OSError:
            continue

    rows: list[dict] = []
    for p in metrics_files:
        try:
            for line in p.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            used.append(p)
        except OSError:
            continue

    return {"log_text": log_text, "rows": rows, "sources": used}


def _numeric(value) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def read_mechanism(mech: Mechanism, log_text: str, rows: list[dict]) -> Reading:
    """One mechanism's verdict for one run."""
    armed_evidence: Optional[str] = None

    if mech.disarmed_log and re.search(mech.disarmed_log, log_text, re.M):
        return Reading(mech.name, NOT_ARMED,
                       detail="explicitly disabled in the log", note=mech.note)

    if mech.armed_log:
        m = re.search(mech.armed_log, log_text, re.M)
        if m:
            armed_evidence = log_text[m.start():m.start() + 120].splitlines()[0]
    if armed_evidence is None and mech.armed_metric:
        for row in rows:
            v = _numeric(row.get(mech.armed_metric))
            if v is not None and v != 0.0:
                armed_evidence = f"{mech.armed_metric}={v} in metrics.jsonl"
                break

    if armed_evidence is None:
        return Reading(mech.name, NOT_ARMED, note=mech.note)

    # Armed. Now: is there a counter at all, and did it move?
    if not mech.activity_log and not mech.activity_metric:
        return Reading(
            mech.name, UNAUDITABLE, armed_evidence=armed_evidence,
            detail="armed, but this mechanism emits no counter to read",
            note=mech.note)

    values: list[float] = []
    if mech.activity_log:
        for m in re.finditer(mech.activity_log, log_text, re.M):
            try:
                values.append(float(m.group(1)))
            except (TypeError, ValueError):
                continue
    if mech.activity_metric:
        for row in rows:
            v = _numeric(row.get(mech.activity_metric))
            if v is not None:
                values.append(v)

    if not values:
        # Armed, a counter is defined, and the run produced ZERO readings of
        # it. That is ignorance, not a null result -- refuse to call it INERT.
        return Reading(
            mech.name, UNAUDITABLE, armed_evidence=armed_evidence,
            observations=0,
            detail="armed, counter defined, but the run recorded no "
                   "observation of it",
            note=mech.note)

    if mech.activity_kind in ("distinct", "ladder"):
        n_distinct = len({round(v, 9) for v in values})
        walked = n_distinct > 1
        arrived = mech.activity_kind == "ladder" and min(values) == 0.0
        moved = walked or arrived
        peak = float(max(values))
        summary = f"{n_distinct} distinct values"
        if arrived and not walked:
            summary += " (held at the terminal rung 0 -- armed and arrived)"
    elif mech.activity_kind == "nonzero":
        moved = any(v != 0.0 for v in values)
        peak = float(max(values, key=abs))
        summary = f"{sum(1 for v in values if v != 0.0)} non-zero observations"
    else:  # cumulative
        peak = float(max(values))
        moved = peak > 0.0
        summary = f"peak {peak:g}"

    return Reading(
        mech.name, FIRED if moved else INERT, armed_evidence=armed_evidence,
        observations=len(values), peak=peak,
        detail=f"{summary} over {len(values)} observations", note=mech.note)


def check_run(run_dir: Path, extra_log: Optional[Path] = None) -> RunReport:
    src = discover_sources(run_dir, extra_log)
    report = RunReport(run=str(run_dir),
                       sources=[str(p) for p in src["sources"]])
    if not src["log_text"] and not src["rows"]:
        # ANTI-VACUITY: an empty scan is an error, not a clean bill of health.
        # `tests/test_purity_quarantine_sweep.py` refuses to certify an empty
        # scan for the same reason; a checker that passes when it read nothing
        # is the defect it exists to catch.
        report.error = (
            f"no readable log or metrics.jsonl under {run_dir} -- refusing "
            "to certify a run whose artifacts could not be read")
        return report
    for mech in MECHANISMS:
        report.readings.append(read_mechanism(mech, src["log_text"], src["rows"]))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_report(report: RunReport, require: list[str]) -> str:
    lines = [f"== {report.run}"]
    if report.error:
        lines.append(f"   ERROR: {report.error}")
        return "\n".join(lines)
    for p in report.sources:
        lines.append(f"   read: {p}")
    for r in sorted(report.readings, key=lambda x: (x.verdict != INERT, x.name)):
        if r.verdict == NOT_ARMED and r.name not in require:
            continue
        mark = {INERT: "VOID", UNAUDITABLE: "VOID", FIRED: "ok  ",
                NOT_ARMED: "----"}[r.verdict]
        lines.append(f"   [{mark}] {r.name:<12} {r.verdict:<12} {r.detail}")
        if r.verdict in VOIDING and r.note:
            lines.append(f"          {r.note}")
    for name in require:
        rd = next((x for x in report.readings if x.name == name), None)
        if rd is None:
            lines.append(f"   [VOID] {name:<12} UNKNOWN      not a registered "
                         "mechanism in this checker")
        elif rd.verdict != FIRED:
            lines.append(f"   [VOID] {name:<12} required by --require but "
                         f"read as {rd.verdict}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail a run whose armed mechanisms never fired.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--require", default="",
                    help="comma-separated mechanism names a registration "
                         "named as variables; each must read FIRED")
    ap.add_argument("--log", type=Path, default=None,
                    help="additional log file to read")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    require = [s.strip() for s in args.require.split(",") if s.strip()]
    reports = [check_run(d, args.log) for d in args.run_dirs]

    if args.json:
        print(json.dumps([
            {"run": r.run, "sources": r.sources, "error": r.error,
             "void_for": r.void_for,
             "readings": [vars(x) for x in r.readings]}
            for r in reports], indent=2))
    else:
        for r in reports:
            print(format_report(r, require))

    if any(r.error for r in reports):
        return 1
    bad = False
    for r in reports:
        if r.void_for:
            bad = True
        for name in require:
            rd = next((x for x in r.readings if x.name == name), None)
            if rd is None or rd.verdict != FIRED:
                bad = True
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
