#!/usr/bin/env python3
"""Phase R — the bottom-k preflight, at exactly the registered operating
point. V32_REDO_BOTTOM_K_2026-08-28.md §7.

Reads a 60-iteration `run.log` from `configs/mario_1_1_v32_redo_bk_seed0
.yaml` at seed 0, `num_envs 60`, `--no-resume --no-supervise
--strict-config` (12 cadenced checks), and decides GO/NO-GO on R1-R4 —
the same B1-B4 conditions `scripts/redo_arm_gate.py` uses at verdict
time, at Phase R's lower event-count floor (>= 10 of 12 checks instead
of the full campaign's >= 48 of 50).

Also banks the RECOVERY CURVE (§7, the measurement no prior ReDo run
could make): for every recycled unit, its dormancy score at each of the
next four cadenced checks — whether a re-initialized trunk unit climbs
out of the rank-bottom when left alone. This is recorded in every
branch, GO or STOP, because it is a mechanism finding independent of
the verdict.

A9 (log integrity): this script only READS the log; it is the caller's
job to have copied it first if the trainer process that wrote it might
still be running (the trainer's FileHandler truncates on (re)open).

Usage:
    scripts/adjudicate_phase_r.py --log runs/v32_.../phase_r/run.log \\
        --out runs/v32_.../phase_r/adjudication.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.redo_arm_gate import adjudicate_bottom_k, parse_log_bottom_k  # noqa: E402

# Phase R's own floor (§7 R1): 60 iterations / cadence 5 = 12 cadenced
# checks; a rank rule fires by construction so R1 asks for >= 10 of 12,
# not the full campaign's >= 48 of 50.
PHASE_R_ITERS = 60
PHASE_R_MIN_EVENTS = 10
PHASE_R_EXPECTED_CHECKS = PHASE_R_ITERS // 5  # 12, at the registered cadence


def recovery_curve(rep, *, horizon: int = 4) -> list[dict]:
    """For each recycled unit at event i, its fc2 score at events
    i+1 .. i+horizon (whatever the trainer's own selection did at those
    checks — the unit may or may not be recycled again). `None` where
    the run ended before the horizon. This is the direct measurement of
    §1.3: does a re-initialized unit ever climb back out of the bottom,
    or does it stay at 0.027-0.054 forever."""
    curve = []
    for i, ev in enumerate(rep.events):
        for unit in ev.logged_fc2_indices:
            row = {
                "recycled_at_iter": ev.iter,
                "unit": unit,
                "score_at_recycle": (
                    ev.fc2_scores[unit] if unit < len(ev.fc2_scores) else None
                ),
            }
            for step in range(1, horizon + 1):
                j = i + step
                if j < len(rep.events) and unit < len(rep.events[j].fc2_scores):
                    row[f"score_after_{step}_check"] = rep.events[j].fc2_scores[unit]
                    row[f"still_bottom_k_after_{step}_check"] = (
                        unit in rep.events[j].logged_fc2_indices
                    )
                else:
                    row[f"score_after_{step}_check"] = None
                    row[f"still_bottom_k_after_{step}_check"] = None
            curve.append(row)
    return curve


def adjudicate_phase_r(
    log_path: Path, *, k: int = 2, cadence: int = 5,
    min_events: int = PHASE_R_MIN_EVENTS,
) -> dict:
    rep = adjudicate_bottom_k(
        parse_log_bottom_k(log_path), k=k, cadence=cadence,
        min_events=min_events,
    )
    # Re-derive R1-R4 individually (not just the combined verdict) so a
    # STOP-vs-escalate decision can be made per §8: R1/R2/R3 failing is
    # an implementation defect (STOP, fix, re-register); R4 alone
    # failing is the ladder's one licensed trigger.
    bottomk_lines = [
        line for line in rep.enabled_lines
        if "mode=bottom_k" in line
    ]
    r1 = (
        len(rep.enabled_lines) == 1 and len(bottomk_lines) == 1
        and not rep.saw_disabled
        and rep.mode == "bottom_k" and rep.k == k and rep.cadence == cadence
        and rep.recycle_events >= min_events
    )
    frac = rep.artifact_match_frac
    r2 = frac is not None and frac == 1.0
    expected_frac = k / 32.0
    r3 = (
        rep.fc1_recycled_total == 0
        and not rep.saw_overdose
        and all(abs(d - expected_frac) < 1e-9 for d in rep.dose_fractions)
    )
    r4 = rep.repeat_rate is not None and rep.repeat_rate < 1.00

    go = r1 and r2 and r3 and r4
    if go:
        decision = "GO"
    elif r1 and r2 and r3 and not r4:
        decision = (
            "NO-GO-R4: cadence bought no recovery — escalate ONCE to "
            "(k=4, C=10) per the registered ladder and re-run Phase R "
            "in full"
        )
    else:
        failed = [name for name, ok in (
            ("R1", r1), ("R2", r2), ("R3", r3),
        ) if not ok]
        decision = (
            f"STOP: {', '.join(failed)} failed — implementation "
            "defect(s), not a dose question. No rung fixes code; fix, "
            "re-register, start over."
        )

    return {
        "log": str(log_path),
        "operating_point": {"k": k, "cadence": cadence},
        "expected_checks": PHASE_R_EXPECTED_CHECKS,
        "r1_reached": r1,
        "r2_artifact_match": r2,
        "r3_dose": r3,
        "r4_turnover": r4,
        "artifact_match_frac": frac,
        "repeat_rate": rep.repeat_rate,
        "recycle_events": rep.recycle_events,
        "cum_recycled": rep.cum_recycled,
        "distinct_fc2_indices": rep.distinct_fc2_indices,
        "max_index_share": rep.max_index_share,
        "arm_gate_verdict": rep.verdict,
        "arm_gate_reasons": rep.reasons,
        "decision": decision,
        "go": go,
        "recovery_curve": recovery_curve(rep),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cadence", type=int, default=5)
    ap.add_argument("--min-events", type=int, default=PHASE_R_MIN_EVENTS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.log.is_file():
        print(f"error: no such log: {args.log}", file=sys.stderr)
        return 1

    result = adjudicate_phase_r(
        args.log, k=args.k, cadence=args.cadence, min_events=args.min_events,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if result["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
