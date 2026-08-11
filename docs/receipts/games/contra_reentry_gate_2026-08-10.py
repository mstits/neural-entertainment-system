"""Contra re-entry gate readout — the registered stopping rule, executable.

This file IS the pre-commitment. It is frozen at registration time and its
sha256 is recorded in §5 of `contra_reentry_2026-08-10.md`. Reading a run
with a modified copy is not the registered gate.

WHY A FILE AND NOT AN INLINE SNIPPET
------------------------------------
The first draft of this gate was an inline `python -c` block, and it
could not emit the outcome the registration predicted. It computed
`'VOID' if not g0 else 'PASS' if gA else 'PARTIAL' if gB else 'FAIL'`,
which has no ELIMINATE branch at all, so the pre-registered failure
(frontier stays pinned, nothing moves) printed PARTIAL — a partial
success. Verified against the banked prior: `poweron_to_wall`, the
archetypal nothing-happened campaign, printed PARTIAL under it. A
stopping rule that cannot print its own predicted outcome is not a
stopping rule, and a decision procedure that has to be re-typed into a
shell each time is not frozen. Hence: one file, hashed, self-testable.

WHY THE FOOTPRINT AND NOT THE VERDICT WORD
------------------------------------------
The obvious reading of "the wall is still there" is "wall_taxonomy still
returns GATED". That reading is wrong here, for a measured reason.
`gated_wall_verdict` reaches GATED on this profile through
`CONCENTRATION_GATED_MIN = 25.0`, and concentration is
`cells / distinct_spatial`. On Contra the spatial footprint is frozen at
~8,000 buckets across a 13x growth in cells, so concentration is very
nearly `cells / 8000` — a monotone function of how long the run went.

    campaign                 cells      distinct_spatial  span   conc   class
    stage1_v3_kk_saturated   154,896    7,654             385    20.24  coverage_limited
    poweron_to_wall          212,367    7,427             385    28.59  gated
    stage1_v5_bosstyped      289,945    8,046             385    36.04  gated
    stage1_v2                313,666    7,983             385    39.29  gated
    stage1_v4_localkk        432,521    7,962             385    54.32  gated
    (five rows re-measured from the archives on 2026-08-10; the remaining
     four campaigns are tabulated in §1 of the receipt)

A fresh 30-minute run lands at 122k-206k cells (measured from the five
fresh campaigns' own progress series at t=1740s), i.e. concentration
~15-26. It straddles the threshold. So a gate keyed on the word GATED
would be deciding an elimination on a coin-flip about run length, and
the most likely single outcome — a `coverage_limited` read like v3's,
whose remedy is literally "give it more wall-clock" — would have printed
PARTIAL, i.e. "the arm changed the shape of the search", when nothing of
the kind was shown.

The statistics that actually carry the load are the ones that DON'T
move: `distinct_spatial` and `spatial_span`. Those saturate fast — a
fresh 24-minute run (`poweron_to_wall`) already reached 7,427 buckets,
and the entire remaining 20.7 hours of prior search bought +14% and
exactly zero change in `spatial_span`, which is 385 in all five
re-measured campaigns and in all ten per §1. That is a footprint
reachable, and therefore falsifiable, inside a 30-minute budget.

So the gate is defined on the footprint, and the taxonomy verdict is
retained only as a guard (did the arm BREAK the search?) and as a
reported diagnostic. `CONCENTRATION_GATED_MIN` does not appear in any
branch condition below.

RUNTIME STATUS: out-of-band analysis only. This file lives under `docs/`
precisely because `tests/test_wall_taxonomy.py::
test_module_is_not_wired_into_any_runtime_dispatch` forbids anything
under src/ scripts/ nes_core/ configs/ from importing `wall_taxonomy`.
Do not move it into `scripts/`, and do not wire it into the solver.

USAGE
    python docs/receipts/games/contra_reentry_gate_2026-08-10.py <run_dir>
    python docs/receipts/games/contra_reentry_gate_2026-08-10.py --self-test
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.training.wall_taxonomy import (  # noqa: E402
    gated_wall_verdict,
    telemetry_from_paths,
)

# --------------------------------------------------------------------------
# Pre-registered constants. Every one is a measurement from §1, not a knob.
# --------------------------------------------------------------------------

#: The frontier (`max_gx_in_max_area`) receipted in nine of ten campaigns.
BASE_FRONTIER = 3072

#: `spatial_span` — distinct gx buckets inside the deepest area — in every
#: campaign measured, across a 2.8x range of cell counts. Zero variance.
#: Any increase means the search reached horizontal extent no prior Contra
#: campaign ever reached.
SPAN_BASE = 385

#: The smallest footprint any §1 campaign reached: `poweron_to_wall`,
#: fresh from power-on, 24 minutes, 205,826 cells at its last progress
#: record. This is the ELIMINATE PRECONDITION. A run that does not
#: re-cover this much footprint has not earned the right to conclude the
#: footprint is immovable — its failure to exceed the prior is
#: uninformative, because it did not even reproduce the prior.
FOOTPRINT_FLOOR = 7427

#: (cells, distinct_spatial) for the nine campaigns whose frontier reached
#: BASE_FRONTIER. The tenth (`stage1_baseline_collapsed_cells`, 13 cells)
#: was a control with deliberately broken observables and is excluded.
#: Five of these rows were re-measured from the archives on 2026-08-10 and
#: reproduced §1 exactly; the four largest are as tabulated in §1.
FOOTPRINT_CURVE = (
    (154_896, 7_654),
    (212_367, 7_427),
    (289_945, 8_046),
    (313_666, 7_983),
    (432_521, 7_962),
    (652_059, 8_182),
    (1_443_275, 8_281),
    (2_087_424, 8_444),
    (2_248_480, 8_477),
)

#: How far above the matched-cell-count prior the footprint must land
#: before we are willing to say the arm changed the SHAPE of the search.
#: Derivation: the upper envelope of FOOTPRINT_CURVE runs 7,654 -> 8,477,
#: a factor of 1.108. So +10% at matched coverage means a 30-minute run
#: found as much additional footprint as the ENTIRE 20.7 hours of prior
#: search did. The residual scatter of the curve at comparable cell counts
#: is ~3% (7,427 at 212k vs 7,654 at 155k), so this is ~3x noise.
FOOTPRINT_MARGIN = 0.10

#: Taxonomy classes that mean the arm damaged the search rather than
#: testing it. §5 carries G-B only as this guard.
BROKEN_CLASSES = ("barren", "key_blind")

#: Taxonomy classes that mean the telemetry could not be classified at
#: all — too few records, or too little compute in the window.
UNCLASSIFIABLE_CLASSES = ("insufficient", "indeterminate")

GATE_HASH_NOTE = "frozen at registration; sha256 recorded in receipt §5"


def footprint_envelope(cells: int) -> int:
    """Best `distinct_spatial` any prior campaign reached at <= `cells`.

    The upper envelope, not a fit: the raw curve is non-monotone
    (`poweron_to_wall` sits BELOW `stage1_v3` despite 37% more cells), and
    taking the running maximum both restores monotonicity and picks the
    conservative direction — the new run is measured against the best the
    prior ever did at that much coverage, never against its average.

    Outside the measured range the envelope CLAMPS rather than
    extrapolates. Below 154,896 cells that means comparing against 7,654,
    which is a HIGHER bar than a downward extrapolation would set; a
    thin run therefore cannot earn PARTIAL by being compared against a
    conveniently low expectation.
    """
    pts = sorted(FOOTPRINT_CURVE)
    env, best = [], 0
    for c, d in pts:
        best = max(best, d)
        env.append((c, best))
    if cells <= env[0][0]:
        return env[0][1]
    if cells >= env[-1][0]:
        return env[-1][1]
    for (c0, d0), (c1, d1) in zip(env, env[1:]):
        if cells <= c1:
            if c1 == c0:
                return max(d0, d1)
            return int(round(d0 + (cells - c0) / (c1 - c0) * (d1 - d0)))
    return env[-1][1]


def decide(*, ortho_selections: int, frontier: int, wall_class: str | None,
           cells: int | None, distinct_spatial: int | None,
           spatial_span: int | None) -> dict:
    """The registered stopping rule. Pure — this is the whole decision.

    Exactly four outcomes, matching §5 one for one: VOID, PASS, PARTIAL,
    ELIMINATE. VOID carries a machine-readable `reason` because there are
    four distinct ways for an attempt to test nothing and they call for
    four different repairs.

    Order matters and is deliberate:
      1. the arm never engaged            -> VOID   (nothing was tested)
      2. the frontier passed the base     -> PASS   (needs only progress.jsonl)
      3. no archive to measure footprint  -> VOID
      4. the arm broke the search         -> VOID
      5. the telemetry cannot be classed  -> VOID
      6. the run under-covered the prior  -> VOID   (ELIMINATE precondition)
      7. the footprint moved              -> PARTIAL
      8. otherwise                        -> ELIMINATE

    G-A is checked before the archive-dependent guards on purpose: passing
    a frontier that stood across 20.7 hours is the headline regardless of
    what the classifier thinks of the archive, and it is a pure
    high-water-mark over progress.jsonl.
    """
    if ortho_selections <= 0:
        return {"gate": "VOID", "reason": "arm_never_engaged",
                "detail": "ortho_selections == 0 for the whole run: the "
                          "orthogonal arm never armed, so the attempt tested "
                          "nothing. Repairable and re-runnable once."}

    if frontier > BASE_FRONTIER:
        return {"gate": "PASS", "reason": "frontier_past_base",
                "detail": f"max_gx_in_max_area {frontier} > {BASE_FRONTIER}: "
                          f"the frontier moved past the wall receipted in "
                          f"nine of ten prior campaigns."}

    if cells is None or distinct_spatial is None or spatial_span is None:
        return {"gate": "VOID", "reason": "no_archive",
                "detail": "no archive.pkl, so the footprint the gate is "
                          "defined on cannot be measured."}

    if wall_class in BROKEN_CLASSES:
        return {"gate": "VOID", "reason": "search_broken",
                "detail": f"wall_class={wall_class}: the arm degraded the "
                          f"search rather than testing it. This is the only "
                          f"thing G-B is carried for. Repairable."}

    if wall_class in UNCLASSIFIABLE_CLASSES:
        return {"gate": "VOID", "reason": "unclassifiable",
                "detail": f"wall_class={wall_class}: not enough records or "
                          f"compute in the window to classify the telemetry."}

    if distinct_spatial < FOOTPRINT_FLOOR:
        return {"gate": "VOID", "reason": "under_covered",
                "detail": f"distinct_spatial {distinct_spatial} < "
                          f"FOOTPRINT_FLOOR {FOOTPRINT_FLOOR}: the run did not "
                          f"re-cover the smallest footprint any prior campaign "
                          f"reached (poweron_to_wall, fresh, 24 min). Failing "
                          f"to EXCEED a prior you did not REPRODUCE is not "
                          f"evidence the prior is immovable. Give it more "
                          f"budget; this is not an elimination."}

    expected = footprint_envelope(cells)
    threshold = expected * (1.0 + FOOTPRINT_MARGIN)
    span_moved = spatial_span > SPAN_BASE
    volume_moved = distinct_spatial >= threshold

    if span_moved or volume_moved:
        why = []
        if span_moved:
            why.append(f"spatial_span {spatial_span} > {SPAN_BASE} (unchanged "
                       f"in every prior campaign)")
        if volume_moved:
            why.append(f"distinct_spatial {distinct_spatial} >= "
                       f"{threshold:.0f} = {expected} x "
                       f"{1.0 + FOOTPRINT_MARGIN:.2f}, the matched-coverage "
                       f"prior envelope at {cells} cells")
        return {"gate": "PARTIAL", "reason": "footprint_moved",
                "detail": "the frontier held but the shape of the search "
                          "changed: " + "; ".join(why)}

    return {"gate": "ELIMINATE", "reason": "footprint_frozen",
            "detail": f"frontier pinned at {frontier} and the footprint did "
                      f"not move: distinct_spatial {distinct_spatial} < "
                      f"{threshold:.0f} (matched-coverage prior envelope "
                      f"{expected} at {cells} cells, +"
                      f"{FOOTPRINT_MARGIN:.0%}), spatial_span "
                      f"{spatial_span} == {SPAN_BASE}. The orthogonal arm — "
                      f"the last mechanism the discriminator names — has been "
                      f"tried and crossed off. Re-shelve Contra and record it."}


def _last_segment(raw: list[dict]) -> list[dict]:
    """Split a progress.jsonl on `elapsed_s` going backwards, take the last.

    `telemetry_from_paths` already does this for the typed records, so the
    raw pass must too or a resumed run would read `ortho_selections` from a
    different attempt than the one being graded.
    """
    segs: list[list[dict]] = [[]]
    prev = None
    for r in raw:
        e = int(r.get("elapsed_s", 0))
        if prev is not None and e < prev:
            segs.append([])
        segs[-1].append(r)
        prev = e
    return segs[-1]


def read_run(run: str) -> dict:
    """Score one run directory. Read-only."""
    prog = os.path.join(run, "progress.jsonl")
    arc = os.path.join(run, "archive.pkl")
    has_arc = os.path.exists(arc)

    with open(prog) as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    seg = _last_segment(raw)

    tel = telemetry_from_paths(prog, archive_path=arc if has_arc else None,
                               label=run)
    verdict = gated_wall_verdict(tel).as_dict()

    frontier = max((r.max_gx for r in tel.records), default=0)
    sel = max((int(r.get("ortho_selections", 0) or 0) for r in seg), default=0)
    a = tel.archive

    result = decide(
        ortho_selections=sel,
        frontier=frontier,
        wall_class=verdict["wall_class"],
        cells=a.cells if a else None,
        distinct_spatial=a.distinct_spatial if a else None,
        spatial_span=a.spatial_span if a else None,
    )

    out = {
        "run": run,
        "GATE": result["gate"],
        "gate_reason": result["reason"],
        "gate_detail": result["detail"],
        # --- the three gate inputs ---
        "ortho_selections": sel,
        "frontier_max_gx": frontier,
        "receipted_base_wall": BASE_FRONTIER,
        "cells": a.cells if a else None,
        "distinct_spatial": a.distinct_spatial if a else None,
        "spatial_span": a.spatial_span if a else None,
        "footprint_floor": FOOTPRINT_FLOOR,
        "footprint_envelope_at_this_coverage": (
            footprint_envelope(a.cells) if a else None),
        "footprint_partial_threshold": (
            round(footprint_envelope(a.cells) * (1.0 + FOOTPRINT_MARGIN), 1)
            if a else None),
        # --- reported only; NOT read by any branch above ---
        "records": len(tel.records),
        "wall_class": verdict["wall_class"],
        "degraded": verdict["degraded"],
        "concentration_REPORTED_ONLY": (
            round(a.concentration, 2) if a else None),
        "taxonomy_remedy_REPORTED_ONLY": verdict["remedy"],
        "solutions": max((r.solutions for r in tel.records), default=0),
        "calibration": verdict["calibration"],
        "gate_provenance": GATE_HASH_NOTE,
    }
    if out["solutions"] > 0:
        out["NOTE_CLEAR_FIRED"] = (
            "A clear fired. It does NOT by itself pass this gate (§5). It "
            "requires its own receipt: --verify-bank replay plus the "
            "counterfactual gate."
        )
    return out


# --------------------------------------------------------------------------
# Self-test: the pre-registered branch table, executable.
#
# Every row is (name, kwargs, expected_gate). These ARE the branches §5
# commits to; if this table and §5 disagree, §5 wins and this file is wrong.
# --------------------------------------------------------------------------

_FROZEN = dict(wall_class="gated", cells=200_000, distinct_spatial=7_600,
               spatial_span=385)

SELF_TEST = (
    # The registration's own predicted outcome. This is the row the
    # original inline snippet got wrong: it printed PARTIAL.
    ("predicted failure: frontier pinned, footprint frozen",
     dict(ortho_selections=5_000, frontier=3072, **_FROZEN), "ELIMINATE"),

    # blocker 2's boundary case, with stage1_v3's real numbers: a
    # coverage_limited read whose remedy is "more wall-clock". Must still
    # eliminate, because the FOOTPRINT is what is frozen, not the verdict.
    ("v3-like: coverage_limited but footprint frozen",
     dict(ortho_selections=5_000, frontier=3072, wall_class="coverage_limited",
          cells=154_896, distinct_spatial=7_654, spatial_span=385),
     "ELIMINATE"),

    # poweron_to_wall's real numbers: the archetypal nothing-happened run.
    ("poweron_to_wall-like: gated, footprint frozen",
     dict(ortho_selections=5_000, frontier=3072, wall_class="gated",
          cells=212_367, distinct_spatial=7_427, spatial_span=385),
     "ELIMINATE"),

    ("arm never engaged",
     dict(ortho_selections=0, frontier=3072, **_FROZEN), "VOID"),
    ("arm never engaged even if frontier moved",
     dict(ortho_selections=0, frontier=4096, **_FROZEN), "VOID"),

    ("frontier past base",
     dict(ortho_selections=1, frontier=3073, **_FROZEN), "PASS"),
    ("frontier equal to base is NOT past it",
     dict(ortho_selections=1, frontier=3072, **_FROZEN), "ELIMINATE"),
    ("frontier past base outranks a broken classifier",
     dict(ortho_selections=1, frontier=5000, wall_class="barren",
          cells=200_000, distinct_spatial=7_600, spatial_span=385), "PASS"),

    ("no archive",
     dict(ortho_selections=1, frontier=3072, wall_class="gated", cells=None,
          distinct_spatial=None, spatial_span=None), "VOID"),
    ("arm broke the search (barren)",
     dict(ortho_selections=1, frontier=3072, wall_class="barren",
          cells=200_000, distinct_spatial=7_600, spatial_span=385), "VOID"),
    ("arm broke the search (key_blind)",
     dict(ortho_selections=1, frontier=3072, wall_class="key_blind",
          cells=200_000, distinct_spatial=7_600, spatial_span=385), "VOID"),
    ("unclassifiable telemetry",
     dict(ortho_selections=1, frontier=3072, wall_class="insufficient",
          cells=200_000, distinct_spatial=7_600, spatial_span=385), "VOID"),

    # The ELIMINATE precondition blocker 2 asked for.
    ("under-covered: did not reproduce the smallest prior footprint",
     dict(ortho_selections=1, frontier=3072, wall_class="coverage_limited",
          cells=60_000, distinct_spatial=5_000, spatial_span=385), "VOID"),
    ("just under the floor",
     dict(ortho_selections=1, frontier=3072, wall_class="gated",
          cells=200_000, distinct_spatial=7_426, spatial_span=385), "VOID"),
    ("exactly at the floor is covered enough",
     dict(ortho_selections=1, frontier=3072, wall_class="gated",
          cells=200_000, distinct_spatial=7_427, spatial_span=385),
     "ELIMINATE"),

    # PARTIAL, both routes.
    ("footprint moved: spatial_span grew by one bucket",
     dict(ortho_selections=1, frontier=3072, wall_class="gated",
          cells=200_000, distinct_spatial=7_600, spatial_span=386), "PARTIAL"),
    ("footprint moved: distinct_spatial past the +10% envelope",
     dict(ortho_selections=1, frontier=3072, wall_class="gated",
          cells=200_000, distinct_spatial=8_420, spatial_span=385), "PARTIAL"),
    ("just under the +10% envelope is still an elimination",
     dict(ortho_selections=1, frontier=3072, wall_class="gated",
          cells=200_000, distinct_spatial=8_419, spatial_span=385),
     "ELIMINATE"),
)


def self_test() -> int:
    """Run the branch table, plus the invariant that the whole §1 prior
    eliminates. Returns a process exit code."""
    failures = []
    for name, kw, expected in SELF_TEST:
        got = decide(**kw)["gate"]
        ok = got == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {got}")
        if not ok:
            failures.append(f"{name}: expected {expected}, got {got}")

    # Retrospective sanity: applied to the prior it is built from, the rule
    # must return ELIMINATE on every §1 campaign. If any prior campaign read
    # PARTIAL, the rule would be calling the prior's own null result a shape
    # change, and a new run reading PARTIAL would mean nothing.
    print("\n  retrospective: the rule applied to the §1 prior")
    for cells, spatial in FOOTPRINT_CURVE:
        got = decide(ortho_selections=1, frontier=3072, wall_class="gated",
                     cells=cells, distinct_spatial=spatial,
                     spatial_span=385)["gate"]
        ok = got == "ELIMINATE"
        print(f"  [{'ok ' if ok else 'FAIL'}] {cells:>9,} cells, "
              f"{spatial:,} buckets -> {got}")
        if not ok:
            failures.append(f"prior {cells} cells: expected ELIMINATE, got {got}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"{len(failures)} failure(s)")
        return 1
    print(f"all {len(SELF_TEST) + len(FOOTPRINT_CURVE)} branch assertions pass")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-3].strip(), file=sys.stderr)
        print("usage: contra_reentry_gate_2026-08-10.py <run_dir>|--self-test",
              file=sys.stderr)
        return 2
    if argv[1] == "--self-test":
        return self_test()
    print(json.dumps(read_run(argv[1]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
