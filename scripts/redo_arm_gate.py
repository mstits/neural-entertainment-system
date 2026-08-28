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
  F3  (V31_REDO_SURGICAL_2026-08-27.md §4.1) distinctness: at least
      ``--min-distinct-fc2`` (default 6) distinct fc2 indices recycled
      over the whole run, and no single index accounting for more than
      ``--max-index-share`` (default 0.60) of all recycled-unit-events.
      A surgical dose (2-4 units/event) that resets the SAME two or
      three units every firing event forever is a permanent partial
      lesion of the trunk, not a recycle — v30 observed exactly this at
      tau=0.50 (``fc2=[1,2,4,5,7,9,13,16,...]`` identical at iters 0-3)
      and none of V1-V6 can see it. Requires the `[redo] recycled unit
      indices:` line at INFO level (trainer.py); a log without it reads
      0 distinct and correctly cannot ARM.

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
# `[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096
#  reset_moments=true mode=bottom_k k=2 recycle_scope=fc2` — the full
# bottom-k line (V32 §3 B1), one physical line in the log.
ENABLED_BOTTOMK_RE = re.compile(
    r"\[redo\] ENABLED tau=([0-9.eE+-]+) every_iters=(\d+) "
    r"scope=(\S+) sample=(\d+) reset_moments=(\S+) mode=(\S+) k=(\d+) "
    r"recycle_scope=(\S+)"
)
# `[redo] fc2 scores: [2.61425, 1.192299, ...]` — the B2 artifact-match
# input, logged immediately after the matching `recycled unit indices:`
# line for every recycle event (trainer.py logs both inside the same
# `if _rd.recycled:` block).
FC2_SCORES_RE = re.compile(r"\[redo\] fc2 scores: (\[[^\]]*\])")
# `[redo] VOID-OVERDOSE: ...` — the in-run dose ceiling trip (V31 §3,
# abort A4). Structurally unreachable at k=2 but B3/R3 check its
# absence rather than assuming it.
OVERDOSE_RE = re.compile(r"\[redo\] VOID-OVERDOSE")


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


##############################################################################
# Bottom-k (rank-rule) arming gate — V32_REDO_BOTTOM_K_2026-08-28.md §3.
#
# A rank rule fires by construction every cadence, so inheriting F1/F2
# unchanged would install a gate that cannot fail (the tenth vacuous
# gate). B1-B4 REPLACE F1/F2/F3 wholesale for a bottom_k run — there is
# exactly one arming gate for this mode, and it is this one. Phase R's
# R1-R4 (scripts/adjudicate_phase_r.py) are the same four checks at a
# lower event-count floor, run before any seed training begins.
##############################################################################


@dataclass
class BottomKEvent:
    """One recycle event: what was logged, and what recomputing bottom-k
    OFFLINE from the logged score vector alone would have selected."""

    iter: int
    logged_fc2_indices: list[int]
    fc2_scores: list[float]
    recomputed_fc2_indices: list[int]
    artifact_match: bool
    separation_margin: float | None  # min(non-recycled) - max(recycled)


@dataclass
class BottomKArmReport:
    """What one run.log says about a bottom_k (rank-rule) ReDo arm."""

    log: str
    enabled_lines: list[str] = field(default_factory=list)
    saw_disabled: bool = False
    saw_overdose: bool = False
    mode: str | None = None
    k: int | None = None
    cadence: int | None = None
    tau: float | None = None
    checks: int = 0
    recycle_events: int = 0
    cum_recycled: int = 0
    dose_fractions: list[float] = field(default_factory=list)
    fc1_recycled_total: int = 0
    events: list[BottomKEvent] = field(default_factory=list)
    repeat_rate: float | None = None
    distinct_fc2_indices: int = 0
    max_index_share: float | None = None
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def armed(self) -> bool:
        return self.verdict == "ARMED"

    @property
    def artifact_match_frac(self) -> float | None:
        if not self.events:
            return None
        return sum(e.artifact_match for e in self.events) / len(self.events)


def _bottom_k_from_scores(scores: list[float], k: int) -> list[int]:
    """Offline recomputation of `bottom_k_indices` (src/training/redo.py)
    from plain-Python logged floats: stable ascending sort, take the
    `k` lowest, ties toward the LOWER index (a stable sort already
    guarantees this), result sorted ascending — a pure re-implementation
    with no dependency on torch, so this gate certifies the ARTIFACT
    (the logged bytes) independently of the code that produced it."""
    k = max(0, min(k, len(scores)))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return sorted(order[:k])


def parse_log_bottom_k(path: Path) -> BottomKArmReport:
    rep = BottomKArmReport(log=str(path))
    pending_fc2_indices: list[int] | None = None
    pending_iter: int | None = None
    fc2_index_counts: dict[int, int] = {}
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if (m := ENABLED_BOTTOMK_RE.search(line)) is not None:
                rep.enabled_lines.append(line.strip())
                rep.tau = float(m.group(1))
                rep.cadence = int(m.group(2))
                rep.mode = m.group(6)
                rep.k = int(m.group(7))
                continue
            if ENABLED_RE.search(line) is not None and (
                ENABLED_BOTTOMK_RE.search(line) is None
            ):
                # A threshold-mode ENABLED line in what should be a
                # bottom_k log — recorded as a bare enabled_line so B1's
                # "exactly one" count and mode check both fail loudly
                # rather than this line being silently invisible.
                rep.enabled_lines.append(line.strip())
                continue
            if DISABLED_RE.search(line) is not None:
                rep.saw_disabled = True
                continue
            if OVERDOSE_RE.search(line) is not None:
                rep.saw_overdose = True
                continue
            if (m := ITER_RE.search(line)) is not None:
                rep.checks += 1
                pending_iter = int(m.group(1))
                recycled = int(m.group(6))
                rep.cum_recycled = int(m.group(7))
                d1, hidden, d2, trunk = (
                    int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)),
                )
                rep.fc1_recycled_total += d1
                rep.dose_fractions.append(max(
                    d1 / hidden if hidden else 0.0,
                    d2 / trunk if trunk else 0.0,
                ))
                if recycled >= 1:
                    rep.recycle_events += 1
                continue
            if (m := INDICES_RE.search(line)) is not None:
                try:
                    pending_fc2_indices = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    pending_fc2_indices = []
                for idx in pending_fc2_indices:
                    fc2_index_counts[idx] = fc2_index_counts.get(idx, 0) + 1
                continue
            if (m := FC2_SCORES_RE.search(line)) is not None:
                try:
                    scores = ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    scores = []
                logged = pending_fc2_indices if pending_fc2_indices is not None else []
                k_for_event = rep.k if rep.k is not None else len(logged)
                recomputed = _bottom_k_from_scores(scores, k_for_event)
                match = recomputed == sorted(logged)
                margin = None
                if scores and logged:
                    recycled_scores = [scores[i] for i in logged]
                    other_scores = [
                        s for i, s in enumerate(scores) if i not in logged
                    ]
                    if other_scores:
                        margin = min(other_scores) - max(recycled_scores)
                rep.events.append(BottomKEvent(
                    iter=(pending_iter if pending_iter is not None else -1),
                    logged_fc2_indices=sorted(logged),
                    fc2_scores=scores, recomputed_fc2_indices=recomputed,
                    artifact_match=match, separation_margin=margin,
                ))
                pending_fc2_indices = None
                pending_iter = None
                continue
    rep.distinct_fc2_indices = len(fc2_index_counts)
    if fc2_index_counts:
        rep.max_index_share = max(fc2_index_counts.values()) / sum(
            fc2_index_counts.values()
        )
    # F3' TURNOVER (§6.2): fraction of events after the first sharing
    # >= 1 index with the IMMEDIATELY PRECEDING event's index set.
    seqs = [set(e.logged_fc2_indices) for e in rep.events]
    if len(seqs) >= 2:
        shared = sum(
            1 for prev, cur in zip(seqs, seqs[1:]) if prev & cur
        )
        rep.repeat_rate = shared / (len(seqs) - 1)
    return rep


def adjudicate_bottom_k(
    rep: BottomKArmReport,
    *,
    k: int,
    cadence: int,
    min_events: int,
    require_artifact_match: float = 1.0,
) -> BottomKArmReport:
    """B1-B4 (§3), evaluated in full — every condition checked, no early
    return, so one violation can never mask another. `min_events` is
    the campaign's `>= 48` for a full 250-iteration seed or Phase R's
    `>= 10` of 12 (adjudicate_phase_r.py supplies its own)."""
    violations: list[tuple[str, str]] = []

    # B1 — REACHED: exactly one bottom_k ENABLED line at the registered
    # (k, cadence); no `[redo] disabled`; enough events; exact dose.
    bottomk_lines = [
        line for line in rep.enabled_lines
        if ENABLED_BOTTOMK_RE.search(line) is not None
    ]
    if len(rep.enabled_lines) != 1 or len(bottomk_lines) != 1:
        violations.append((
            "VOID-NOT-ARMED",
            f"expected exactly one bottom_k '[redo] ENABLED ... "
            f"mode=bottom_k' line, found {len(bottomk_lines)} bottom_k "
            f"and {len(rep.enabled_lines) - len(bottomk_lines)} other "
            "ENABLED line(s)",
        ))
    elif rep.mode != "bottom_k" or rep.k != k or rep.cadence != cadence:
        violations.append((
            "VOID-WRONG-POINT",
            f"armed at mode={rep.mode} k={rep.k} every_iters={rep.cadence} "
            f"but the registered operating point is mode=bottom_k k={k} "
            f"every_iters={cadence}",
        ))
    if rep.saw_disabled:
        violations.append((
            "VOID-NOT-ARMED",
            "log contains '[redo] disabled' — the mechanism was inert",
        ))
    if rep.recycle_events < min_events:
        violations.append((
            "VOID-NOT-REACHED",
            f"{rep.recycle_events} recycle events (need >= {min_events})",
        ))
    elif rep.cum_recycled != k * rep.recycle_events:
        violations.append((
            "VOID-NOT-REACHED",
            f"cum_recycled={rep.cum_recycled} != k*events="
            f"{k}*{rep.recycle_events}={k * rep.recycle_events}",
        ))

    # B2 — GENUINELY THE BOTTOM (artifact-match). Requires the offline
    # recomputation from the logged score vector to reproduce the
    # logged index set on every event checked.
    frac = rep.artifact_match_frac
    if frac is None or frac < require_artifact_match:
        violations.append((
            "VOID-ARTIFACT-MISMATCH",
            f"offline bottom-k recomputation matches logged indices on "
            f"{'no events' if frac is None else f'{frac:.1%} of events'} "
            f"(need {require_artifact_match:.0%}) — the logged selection "
            "does not match the logged score vector",
        ))

    # B3 — DOSE (structural; asserted anyway).
    expected_frac = k / 32.0
    bad_dose = [d for d in rep.dose_fractions if abs(d - expected_frac) > 1e-9]
    if bad_dose or rep.fc1_recycled_total != 0:
        violations.append((
            "VOID-DOSE",
            f"dose_fraction != {expected_frac:g} on {len(bad_dose)} "
            f"check(s), fc1_recycled_total={rep.fc1_recycled_total} "
            "(must be 0 — bottom_k scope is fc2 only)",
        ))
    if rep.saw_overdose:
        violations.append((
            "VOID-DOSE",
            "log contains '[redo] VOID-OVERDOSE' — the in-run dose "
            "ceiling tripped, which is structurally unreachable at "
            "this k and signals a registration or code defect",
        ))

    # B4 — RECOVERY (F3' TURNOVER, §6.2). VOID at exactly 1.00; nothing
    # else about repeat_rate is gated.
    if rep.repeat_rate is not None and rep.repeat_rate == 1.00:
        violations.append((
            "VOID-NO-TURNOVER",
            f"repeat_rate == 1.00 over {len(rep.events)} events — the "
            "recycled set never turned over even once; cadence bought "
            "no recovery",
        ))

    if violations:
        order = [
            "VOID-NOT-ARMED", "VOID-WRONG-POINT", "VOID-NOT-REACHED",
            "VOID-ARTIFACT-MISMATCH", "VOID-DOSE", "VOID-NO-TURNOVER",
        ]
        rep.verdict = min(
            (v for v, _ in violations), key=lambda v: order.index(v)
        )
        rep.reasons.extend(r for _, r in violations)
    else:
        rep.verdict = "ARMED"
        rep.reasons.append(
            f"{rep.recycle_events} recycle events / {rep.cum_recycled} "
            f"units at k={k}, every_iters={cadence}, artifact_match="
            f"{frac:.1%}, repeat_rate={rep.repeat_rate}, "
            f"distinct_fc2_indices={rep.distinct_fc2_indices}"
        )
    return rep


def _render_bottom_k(rep: BottomKArmReport) -> str:
    lines = [
        f"VERDICT: {rep.verdict}  [{rep.log}]",
        f"  mode={rep.mode} k={rep.k} every_iters={rep.cadence} "
        f"tau={rep.tau} (provenance only)",
        f"  checks={rep.checks} recycle_events={rep.recycle_events} "
        f"cum_recycled={rep.cum_recycled} saw_disabled={rep.saw_disabled} "
        f"saw_overdose={rep.saw_overdose}",
        f"  artifact_match_frac={rep.artifact_match_frac} "
        f"repeat_rate={rep.repeat_rate}",
        f"  distinct_fc2_indices={rep.distinct_fc2_indices} "
        f"max_index_share={rep.max_index_share} "
        f"fc1_recycled_total={rep.fc1_recycled_total}",
    ]
    lines += [f"  - {r}" for r in rep.reasons]
    if not rep.armed:
        lines.append(
            "  This run yields NO verdict. It is not counted in best-of-4 "
            "and may not be cited as evidence for or against the "
            "plasticity-loss hypothesis."
        )
    return "\n".join(lines)


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
    ap.add_argument("--bottom-k", action="store_true",
                    help="adjudicate under the V32 bottom_k B1-B4 gate "
                         "(REPLACES F1/F2/F3) instead of the legacy "
                         "threshold-mode V1-V6/F3 gate")
    ap.add_argument("--k", type=int, default=2,
                    help="bottom_k mode: registered k (default 2)")
    ap.add_argument("--cadence", type=int, default=5,
                    help="bottom_k mode: registered every_iters (default 5)")
    ap.add_argument("--bottom-k-min-events", type=int, default=48,
                    help="bottom_k mode: min recycle events to be "
                         "REACHED (default 48, the full-campaign floor; "
                         "Phase R uses adjudicate_phase_r.py's own 10)")
    args = ap.parse_args(argv)

    if args.bottom_k:
        bk_reports: list[BottomKArmReport] = []
        for raw in args.logs:
            path = Path(raw)
            if not path.is_file():
                print(f"error: no such log: {path}", file=sys.stderr)
                return 1
            rep = adjudicate_bottom_k(
                parse_log_bottom_k(path),
                k=args.k, cadence=args.cadence,
                min_events=args.bottom_k_min_events,
            )
            bk_reports.append(rep)
            print(_render_bottom_k(rep))
        if args.json:
            payload = []
            for rep in bk_reports:
                d = asdict(rep)
                d["armed"] = rep.armed
                d["artifact_match_frac"] = rep.artifact_match_frac
                payload.append(d)
            Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        n_void = sum(1 for r in bk_reports if not r.armed)
        if n_void:
            print(f"\n{n_void}/{len(bk_reports)} log(s) VOID — no verdict "
                  "may be issued for them.")
            return 2
        print(f"\n{len(bk_reports)}/{len(bk_reports)} log(s) ARMED — the "
              "honest-eval gate may now score them.")
        return 0

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
