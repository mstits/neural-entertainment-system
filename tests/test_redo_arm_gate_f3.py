"""Anti-vacuity tests for gate condition F3 — distinctness of recycled
fc2 units (V31_REDO_SURGICAL_2026-08-27.md §4.1).

No earlier condition catches the pathology F3 exists for: V3/V4 only
count events and units, and V5/V6 are structurally insensitive to WHICH
units were touched (`recycle()` zeroes the outgoing actor+critic columns
regardless of index). A treatment that resets the SAME two or three
fc2 units forever is a permanent partial lesion of a 32-unit trunk, not a
recycle — v30 observed exactly this at tau=0.50
(`fc2=[1,2,4,5,7,9,13,16,...]` identical at iters 0-3) and no gate caught
it at the time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.redo_arm_gate import adjudicate, parse_log  # noqa: E402

ENABLED = "[redo] ENABLED tau={tau} every_iters=1 scope=fc1,fc2 sample=4096"
ITER = (
    "[redo] iter {it}: dormant fc1 0/64 fc2 {n}/32 recycled {n} cum {cum} "
    "agree {agree:.4f} max_dlogit 0.050000"
)


def _write(tmp_path: Path, lines: list[str], name: str = "run.log") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _adj(path: Path, **kw):
    kw.setdefault("tau", 0.10)
    kw.setdefault("min_events", 40)
    kw.setdefault("min_units", 64)
    kw.setdefault("min_agree", 0.60)
    kw.setdefault("agree_window", 50)
    kw.setdefault("max_frac", 0.25)
    kw.setdefault("min_distinct_fc2", 6)
    kw.setdefault("max_index_share", 0.60)
    return adjudicate(parse_log(path), **kw)


def _build(
    tmp_path: Path, *, per_event_indices, agree: float = 0.97,
    name: str = "run.log",
) -> Path:
    """One `[redo] ENABLED tau=0.10` run with one firing event per entry
    of `per_event_indices` (a list of fc2-index-lists), enough events to
    clear F1/F2 (>=40 events, >=64 units) so ONLY F3 can be at issue.
    """
    lines = [ENABLED.format(tau=0.10)]
    cum = 0
    for it, idxs in enumerate(per_event_indices):
        n = len(idxs)
        cum += n
        lines.append(ITER.format(it=it, n=n, cum=cum, agree=agree))
        lines.append(f"[redo] recycled unit indices: fc1=[] fc2={idxs}")
    return _write(tmp_path, lines, name=name)


def test_single_index_lesion_voids_on_f3(tmp_path):
    """The exact v30 tau=0.50 pathology, shrunk to the surgical dose: 2
    units recycled every event, but ALWAYS the same 2 indices. 60 events
    x 2 units clears F1 (>=40) and F2 (>=64 = 2.0x32), so nothing but F3
    is at issue.

    REVERT-VERIFIED FAILURE: see
    test_deleting_f3_lets_the_single_index_lesion_through below — this
    assertion is what that deletion breaks.
    """
    log = _build(tmp_path, per_event_indices=[[3, 7]] * 60)
    rep = _adj(log)
    assert rep.distinct_fc2_indices == 2
    assert rep.max_index_share == pytest.approx(0.5)
    assert rep.verdict == "VOID-MINIMAL-DOSE"
    assert any("distinct" in r or "share" in r for r in rep.reasons)


def test_two_index_alternation_still_voids_on_share(tmp_path):
    """Even with two indices contributing, if together they still fail
    the >=6-distinct floor the run is VOID — distinctness, not just
    "more than one index", is what F3 requires.
    """
    log = _build(
        tmp_path,
        per_event_indices=[[1, 2] if it % 2 == 0 else [2, 3] for it in range(60)],
    )
    rep = _adj(log)
    assert rep.distinct_fc2_indices == 3
    assert rep.verdict == "VOID-MINIMAL-DOSE"


def test_healthy_multi_index_run_is_armed(tmp_path):
    """A genuine recycle — the dormant tail moves around the trunk over
    training, touching many different units, no single one dominating —
    is ARMED. The ceiling is not vacuous in the direction that would
    void every real run.
    """
    # 60 events, indices rotate through all 32 trunk slots, 2/event.
    per_event = [[(it * 2) % 32, (it * 2 + 1) % 32] for it in range(60)]
    log = _build(tmp_path, per_event_indices=per_event)
    rep = _adj(log)
    assert rep.distinct_fc2_indices == 32
    assert rep.max_index_share is not None and rep.max_index_share < 0.10
    assert rep.verdict == "ARMED"


def test_index_share_just_under_and_over_the_60_percent_line(tmp_path):
    """Boundary check on the share half of F3, independent of the
    distinct-count half: construct exactly 10 distinct indices where one
    index accounts for exactly 60% (survives, non-strict at the
    threshold per `> max_index_share`) vs 61% (voids).
    """
    # 100 total unit-events, 9 distinct "other" indices with 4 events
    # each (36) plus index 0 carrying 64 -> 64% > 60%.
    events = [[0]] * 64 + [[i] for i in range(1, 10) for _ in range(4)]
    log = _build(tmp_path, per_event_indices=events)
    rep = _adj(log)
    assert rep.distinct_fc2_indices == 10
    assert rep.max_index_share == pytest.approx(0.64)
    assert rep.verdict == "VOID-MINIMAL-DOSE"

    # Flatten index 0's share to exactly 60/100 by trimming 4 of its
    # events and redistributing them as a brand-new 11th index.
    events2 = (
        [[0]] * 60 + [[i] for i in range(1, 10) for _ in range(4)] + [[10]] * 4
    )
    log2 = _build(tmp_path, per_event_indices=events2, name="run2.log")
    rep2 = _adj(log2)
    assert rep2.distinct_fc2_indices == 11
    assert rep2.max_index_share == pytest.approx(0.60)
    assert rep2.verdict == "ARMED", (
        "exactly at the 60% share must survive the strict '>' threshold"
    )


def test_no_index_lines_cannot_verify_distinctness_and_voids(tmp_path):
    """A log recycling real units but missing the (promoted-to-INFO)
    `[redo] recycled unit indices:` lines cannot have its distinctness
    verified and correctly VOIDs rather than being given the benefit of
    the doubt — this is the live consequence of requiring the DEBUG ->
    INFO promotion in trainer.py.
    """
    lines = [ENABLED.format(tau=0.10)]
    cum = 0
    for it in range(60):
        cum += 2
        lines.append(ITER.format(it=it, n=2, cum=cum, agree=0.97))
        # Deliberately no "[redo] recycled unit indices:" line.
    log = _write(tmp_path, lines)
    rep = _adj(log)
    assert rep.distinct_fc2_indices == 0
    assert rep.verdict == "VOID-MINIMAL-DOSE"


def test_deleting_f3_lets_the_single_index_lesion_through(tmp_path):
    """Executed, in-process revert-verification: with F3's two violation
    branches removed, the exact v30-tau-0.50-shaped lesion in
    `test_single_index_lesion_voids_on_f3` is called ARMED — reproducing
    the eighth vacuous gate one condition later. This is the deletion
    check `docs/proposals/V31_REDO_SURGICAL_2026-08-27.md` §4.1 requires,
    executed as code so it cannot silently rot, rather than only being
    run by hand once and described in a document.
    """
    import scripts.redo_arm_gate as gate_mod

    log = _build(tmp_path, per_event_indices=[[3, 7]] * 60)
    rep_with_f3 = _adj(log)
    assert rep_with_f3.verdict == "VOID-MINIMAL-DOSE"

    real_adjudicate = gate_mod.adjudicate

    def _adjudicate_without_f3(rep, **kw):
        kw = dict(kw)
        kw["min_distinct_fc2"] = 0
        kw["max_index_share"] = 1.0
        return real_adjudicate(rep, **kw)

    rep_without_f3 = _adjudicate_without_f3(
        gate_mod.parse_log(log), tau=0.10, min_events=40, min_units=64,
        min_agree=0.60, agree_window=50, max_frac=0.25,
    )
    assert rep_without_f3.verdict == "ARMED", (
        "with F3 neutered, the same permanent-2-unit-lesion log that "
        "must VOID is instead certified ARMED — this is the failure F3 "
        "exists to prevent"
    )
