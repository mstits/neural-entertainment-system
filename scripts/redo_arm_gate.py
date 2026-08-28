#!/usr/bin/env python3
"""ReDo arming gate — decide whether a run may be given a verdict at all.

A run in which ReDo does not fire is VOID, not FAIL. v27 and v28 each
burned four seeds x ~7h at ``redo_tau: 0.025`` — an order of magnitude
below the firing threshold of a Linear->LayerNorm->SiLU trunk — and then
reported FAIL verdicts (0.530 and 0.670) that the treatment could not
have produced. Telemetry from both campaigns reads
``dormant fc1 0/N fc2 0/32 recycled 0 cum 0`` on every one of ~2000
per-iteration checks. This script exists so that class of run cannot
reach a verdict again.

It reads a training ``run.log`` and answers exactly one question: was the
treatment ARMED AND FIRING? It is structurally incapable of printing PASS
or FAIL — those words never appear in an output path. Its only verdicts
are ARMED (the run may now be scored by the honest-eval gate) and one of
the VOID reasons.

Registered conditions, all of which must hold for ARMED:

  V1  exactly one ``[redo] ENABLED tau=<tau>`` line, and its tau equals
      the registered operating point (``--tau``, default 0.25). A
      preflight or run evaluated at any other tau voids the arm it
      certified — the structural fix for the V7 defect, where the pilot
      ran at tau 0.50 while the runs ran at 0.025. Note that "the sweep
      showed 0 recycles at tau X" is only evidence about the iterations
      the sweep covered: the banked isolation sweep ran 2 iterations, and
      at orthogonal init the score distribution is still tight. Measured
      over 20 iterations, tau 0.15 fires from iter 4.
  V2  no ``[redo] disabled`` line. The armed-but-unsupported-architecture
      path prints exactly that, so a silent no-op cannot pass as a
      treatment.
  V3  cum_recycled > 0 at the end of the run.
  V4  at least ``--min-events`` (default 10) iterations with recycled >= 1
      and a final cum of at least ``--min-units`` (default 20) units.
      This closes the "technically fired once" loophole, which is how the
      eighth vacuous gate gets written.
  V5  median greedy-argmax agreement over the first ``--agree-window``
      (default 50) recycle events >= ``--min-agree`` (default 0.60). Below
      that the step is a partial network reset rather than a surgical
      intervention and the arm is uninterpretable.
  V6  median recycled fraction of the trunk per firing event <=
      ``--max-frac`` (default 0.25). V5 alone CANNOT catch over-recycling:
      the recycle zeroes the outgoing actor and critic columns by
      construction, so the network's output is approximately preserved no
      matter how many hidden units were re-initialized. Measured on a
      20-iteration pilot at tau 0.25, agreement sits at median 0.856 —
      comfortably past V5 — while 20 of 32 trunk units are re-initialized
      every single iteration. Agreement is structurally insensitive to the
      damage; the recycled fraction is not. A gate that checks only the
      former is vacuous against exactly the "network reset" family this
      experiment must avoid.

Exit codes: 0 ARMED, 2 VOID, 1 usage/parse error.

Usage:
    scripts/redo_arm_gate.py checkpoints/<run>/run.log [--tau 0.25]
    scripts/redo_arm_gate.py <log> [<log> ...] --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# `[redo] ENABLED tau=0.25 every_iters=1 scope=fc1,fc2 sample=4096 ...`
ENABLED_RE = re.compile(r"\[redo\] ENABLED tau=([0-9.eE+-]+)\b")
DISABLED_RE = re.compile(r"\[redo\] disabled\b")
# `[redo] iter 7: dormant fc1 0/64 fc2 5/32 recycled 5 cum 5 agree 0.8142 ...`
ITER_RE = re.compile(
    r"\[redo\] iter (\d+): dormant fc1 (\d+)/(\d+) fc2 (\d+)/(\d+) "
    r"recycled (\d+) cum (\d+) agree ([0-9.]+) max_dlogit ([0-9.eE+-]+)"
)
# `[redo] recycled unit indices: fc1=[] fc2=[3, 7, 19]` — promoted DEBUG ->
# INFO for V31 (trainer.py) specifically so F3 below can see it; a log
# captured at the old DEBUG level has no such lines and F3 correctly
# cannot ARM it (distinct count reads 0).
INDICES_RE = re.compile(
    r"\[redo\] recycled unit indices: fc1=(\[[^\]]*\]) fc2=(\[[^\]]*\])"
)


@dataclass
class ArmReport:
    """What one run.log says about whether ReDo was armed and firing."""

    log: str
    enabled_taus: list[float] = field(default_factory=list)
    saw_disabled: bool = False
    checks: int = 0
    recycle_events: int = 0
    cum_recycled: int = 0
    first_recycle_iter: int | None = None
    last_iter: int | None = None
    median_agree: float | None = None
    min_agree: float | None = None
    median_recycled_frac: float | None = None
    max_recycled_frac: float | None = None
    trunk_dim: int | None = None
    # F3 (V31_REDO_SURGICAL_2026-08-27.md §4.1) — distinctness of the
    # recycled fc2 units over the whole run. A treatment that resets the
    # SAME two or three units forever is a permanent partial lesion of a
    # 32-unit trunk, not a recycle (v30 observed exactly this pathology
    # at tau=0.50: fc2=[1,2,4,5,7,9,13,16,...] identical at iters 0-3).
    distinct_fc2_indices: int = 0
    max_index_share: float | None = None
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def armed(self) -> bool:
        return self.verdict == "ARMED"


def parse_log(path: Path) -> ArmReport:
    rep = ArmReport(log=str(path))
    agrees: list[float] = []
    fracs: list[float] = []
    fc2_index_counts: dict[int, int] = {}
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if (m := ENABLED_RE.search(line)) is not None:
                rep.enabled_taus.append(float(m.group(1)))
                continue
            if DISABLED_RE.search(line) is not None:
                rep.saw_disabled = True
                continue
            if (m := INDICES_RE.search(line)) is not None:
                try:
                    fc2_list = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    fc2_list = []
                for idx in fc2_list:
                    fc2_index_counts[idx] = fc2_index_counts.get(idx, 0) + 1
                continue
            if (m := ITER_RE.search(line)) is not None:
                it = int(m.group(1))
                recycled = int(m.group(6))
                rep.checks += 1
                rep.last_iter = it
                rep.cum_recycled = int(m.group(7))
                hidden, trunk = int(m.group(3)), int(m.group(5))
                rep.trunk_dim = trunk
                if recycled >= 1:
                    rep.recycle_events += 1
                    if rep.first_recycle_iter is None:
                        rep.first_recycle_iter = it
                    agrees.append(float(m.group(8)))
                    # Dose = the WORST-HIT LAYER's re-initialized
                    # fraction, never the pooled one. Pooling is itself a
                    # vacuity: fc1 never goes dormant on this
                    # architecture (0/64 at every tau from 0.025 to 0.25,
                    # across 66 measured iterations), so a pooled
                    # denominator carries a permanent 64-unit ballast that
                    # can never contribute to the numerator. It reports
                    # 20/96 = 21% for an event that re-initialized 20 of
                    # the 32 trunk units — 62%. Per-layer max is the
                    # statistic that tracks the damage.
                    d1, d2 = int(m.group(2)), int(m.group(4))
                    fracs.append(max(
                        d1 / hidden if hidden else 0.0,
                        d2 / trunk if trunk else 0.0,
                    ))
    rep.distinct_fc2_indices = len(fc2_index_counts)
    if fc2_index_counts:
        rep.max_index_share = max(fc2_index_counts.values()) / sum(
            fc2_index_counts.values()
        )
    rep._agrees = agrees  # type: ignore[attr-defined]
    rep._fracs = fracs  # type: ignore[attr-defined]
    return rep


def adjudicate(
    rep: ArmReport,
    *,
    tau: float,
    min_events: int,
    min_units: int,
    min_agree: float,
    agree_window: int,
    max_frac: float = 0.25,
    min_distinct_fc2: int = 6,
    max_index_share: float = 0.60,
) -> ArmReport:
    agrees: list[float] = getattr(rep, "_agrees", [])
    window = agrees[:agree_window]
    if window:
        rep.median_agree = float(statistics.median(window))
        rep.min_agree = float(min(window))
    fracs: list[float] = getattr(rep, "_fracs", [])
    if fracs:
        rep.median_recycled_frac = float(statistics.median(fracs))
        rep.max_recycled_frac = float(max(fracs))

    # Every condition is evaluated — no early return — so one violation
    # can never mask another. The verdict is then the highest-priority
    # violation, but the report always lists all of them.
    violations: list[tuple[str, str]] = []

    # V3 — the lesson of v27/v28, stated as a number. Checked first
    # because "never fired" is the headline, whatever else is also wrong.
    if rep.cum_recycled == 0:
        violations.append((
            "VOID-NEVER-FIRED",
            f"cum_recycled == 0 over {rep.checks} dormancy checks — ReDo "
            "never fired, so this run is VOID, not FAIL, and may not be "
            "cited for or against the plasticity-loss hypothesis",
        ))
    elif rep.recycle_events < min_events or rep.cum_recycled < min_units:
        # V4 — armed but effectively inert.
        violations.append((
            "VOID-MINIMAL-DOSE",
            f"{rep.recycle_events} recycle events (need >= {min_events}) "
            f"and cum {rep.cum_recycled} units (need >= {min_units}) — "
            "technically armed but effectively inert; its FAIL would be "
            "non-informative about plasticity",
        ))

    # F3 (V31 §4.1) — distinctness. A treatment that resets the SAME two
    # or three units every iteration forever is a permanent partial
    # lesion, not a recycle, and no earlier condition catches it: V3/V4
    # only see counts, and V5/V6 are structurally insensitive to WHICH
    # units were touched. Requires the `[redo] recycled unit indices:`
    # line at INFO level (trainer.py); a log without it reads 0 distinct
    # and correctly cannot ARM, since distinctness is then unverifiable.
    if rep.cum_recycled > 0:
        if rep.distinct_fc2_indices < min_distinct_fc2:
            violations.append((
                "VOID-MINIMAL-DOSE",
                f"only {rep.distinct_fc2_indices} distinct fc2 index/indices "
                f"recycled across the run (need >= {min_distinct_fc2}) — "
                "either a permanent partial lesion of a few units, or the "
                "'[redo] recycled unit indices:' INFO line is missing from "
                "this log and distinctness cannot be verified",
            ))
        elif (
            rep.max_index_share is not None
            and rep.max_index_share > max_index_share
        ):
            violations.append((
                "VOID-MINIMAL-DOSE",
                f"one fc2 index accounts for {rep.max_index_share:.1%} of "
                f"all recycled-unit-events (> {max_index_share:.0%}) across "
                f"{rep.distinct_fc2_indices} distinct indices — a "
                "self-sustaining single-unit lesion, not a recycle "
                "(v30 saw this pathology at tau=0.50: fc2=[1,2,4,5,7,9,13,"
                "16,...] identical at iters 0-3)",
            ))

    # V1/V2 — armed at the registered operating point, and not disabled.
    if len(rep.enabled_taus) != 1:
        violations.append((
            "VOID-NOT-ARMED",
            f"expected exactly one '[redo] ENABLED tau=' line, found "
            f"{len(rep.enabled_taus)}",
        ))
    elif abs(rep.enabled_taus[0] - tau) > 1e-12:
        # The V7 vacuity class: certified at one tau, run at another.
        violations.append((
            "VOID-WRONG-TAU",
            f"armed at tau={rep.enabled_taus[0]:g} but the registered "
            f"operating point is tau={tau:g}; a run or preflight evaluated "
            f"at any other tau voids the arm it certified",
        ))
    if rep.saw_disabled:
        violations.append((
            "VOID-NOT-ARMED",
            "log contains '[redo] disabled' — the mechanism was inert",
        ))

    # V5 — surgical intervention, or a partial reset?
    if rep.median_agree is not None and rep.median_agree < min_agree:
        violations.append((
            "VOID-IDENTITY",
            f"median greedy-argmax agree {rep.median_agree:.4f} over the "
            f"first {len(window)} recycle events < {min_agree:.2f} — the "
            "step is a partial network reset, not a surgical "
            "intervention, and the arm is uninterpretable",
        ))

    # V6 — over-recycling. The condition V5 cannot see.
    if rep.median_recycled_frac is not None and (
        rep.median_recycled_frac > max_frac
    ):
        violations.append((
            "VOID-OVERDOSE",
            f"median {rep.median_recycled_frac:.1%} of the worst-hit "
            f"layer re-initialized per firing event (max "
            f"{rep.max_recycled_frac:.1%}), over the {max_frac:.0%} "
            "ceiling — this is a per-iteration partial network reset, not "
            "a surgical recycle, and the 'network reset' family is ruled "
            "incompatible with this experiment. Note the identity check "
            "does NOT catch this: zeroing the outgoing columns preserves "
            "the output whatever the dose",
        ))

    if violations:
        order = [
            "VOID-NEVER-FIRED", "VOID-NOT-ARMED", "VOID-WRONG-TAU",
            "VOID-MINIMAL-DOSE", "VOID-OVERDOSE", "VOID-IDENTITY",
        ]
        rep.verdict = min(
            (v for v, _ in violations), key=lambda v: order.index(v)
        )
        rep.reasons.extend(r for _, r in violations)
    else:
        rep.verdict = "ARMED"
        rep.reasons.append(
            f"{rep.recycle_events} recycle events / {rep.cum_recycled} units "
            f"at tau={tau:g}, first at iter {rep.first_recycle_iter}, median "
            f"agree {rep.median_agree:.4f}, median dose "
            f"{rep.median_recycled_frac:.1%} of the worst-hit layer"
        )
    return rep


def _render(rep: ArmReport) -> str:
    head = f"VERDICT: {rep.verdict}"
    if rep.verdict == "VOID-NEVER-FIRED":
        head = "VERDICT: VOID (redo never fired)"
    lines = [
        f"{head}  [{rep.log}]",
        f"  checks={rep.checks} recycle_events={rep.recycle_events} "
        f"cum_recycled={rep.cum_recycled} "
        f"first_recycle_iter={rep.first_recycle_iter} "
        f"last_iter={rep.last_iter}",
        f"  enabled_taus={rep.enabled_taus} saw_disabled={rep.saw_disabled} "
        f"median_agree={rep.median_agree} min_agree={rep.min_agree}",
        f"  median_recycled_frac={rep.median_recycled_frac} "
        f"max_recycled_frac={rep.max_recycled_frac} "
        f"trunk_dim={rep.trunk_dim}",
        f"  distinct_fc2_indices={rep.distinct_fc2_indices} "
        f"max_index_share={rep.max_index_share}",
    ]
    lines += [f"  - {r}" for r in rep.reasons]
    if not rep.armed:
        lines.append(
            "  This run yields NO verdict. It is not counted in best-of-4 "
            "and may not be cited as evidence for or against the "
            "plasticity-loss hypothesis."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("logs", nargs="+", help="run.log path(s), one per seed")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="registered operating point (default 0.25)")
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--min-units", type=int, default=20)
    ap.add_argument("--min-agree", type=float, default=0.60)
    ap.add_argument("--agree-window", type=int, default=50)
    ap.add_argument("--max-frac", type=float, default=0.25,
                    help="max median fraction of recyclable hidden units "
                         "one firing event may re-initialize (default 0.25)")
    ap.add_argument("--min-distinct-fc2", type=int, default=6,
                    help="F3: min distinct fc2 indices recycled over the "
                         "whole run (default 6)")
    ap.add_argument("--max-index-share", type=float, default=0.60,
                    help="F3: max share of recycled-unit-events one fc2 "
                         "index may account for (default 0.60)")
    ap.add_argument("--json", type=str, default=None,
                    help="also write the per-log reports as JSON")
    args = ap.parse_args(argv)

    reports: list[ArmReport] = []
    for raw in args.logs:
        path = Path(raw)
        if not path.is_file():
            print(f"error: no such log: {path}", file=sys.stderr)
            return 1
        rep = adjudicate(
            parse_log(path),
            tau=args.tau,
            min_events=args.min_events,
            min_units=args.min_units,
            min_agree=args.min_agree,
            agree_window=args.agree_window,
            max_frac=args.max_frac,
            min_distinct_fc2=args.min_distinct_fc2,
            max_index_share=args.max_index_share,
        )
        reports.append(rep)
        print(_render(rep))

    if args.json:
        payload = []
        for rep in reports:
            d = asdict(rep)
            d["armed"] = rep.armed
            payload.append(d)
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")

    n_void = sum(1 for r in reports if not r.armed)
    if n_void:
        print(f"\n{n_void}/{len(reports)} log(s) VOID — no verdict may be "
              "issued for them.")
        return 2
    print(f"\n{len(reports)}/{len(reports)} log(s) ARMED — the honest-eval "
          "gate may now score them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
