"""FORGE_REGISTRY -- the Forge's own anti-vacuity discipline.

Mirrors the (verdict key, positive, negative) shape of
tests/test_anti_vacuity_gates.py:235-244 (REGISTRY), adapted for the
Forge's verdicts: those are multi-way strings (STALLED/ADVANCING/...),
not the binary passed/ok/valid/clear anti_vacuity_scan.py's AST scanner
looks for, so the polarity check below compares against the registered
target string rather than truthiness.

scripts/anti_vacuity_scan.py already scans all of src/ (SCAN_DIRS =
("scripts", "src")), so src/forge/ is in its sweep the moment it exists;
this file additionally proves that every site the scanner CAN find under
src/forge/ is accounted for, so a future Forge piece that introduces the
scanner's exact `passed = not X` / `len(X) == 0` shape cannot land
without a registered positive/negative pair here.

Grows across pieces (a)-(f); tonight it carries the two entries piece
(a) contributes (archive_verdict, campaign_verdict).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.anti_vacuity_scan import scan_repo  # noqa: E402
from src.forge.stall import archive_verdict, campaign_verdict  # noqa: E402

#: Scratch dir for the positive/negative fixtures below. A tempdir
#: (never a repo path) so this file never touches real wall data.
_GATE_TMP = Path(tempfile.mkdtemp(prefix="forge_gate_fixtures_"))


# ---------------------------------------------------------------------
# (a) archive_verdict / campaign_verdict
# ---------------------------------------------------------------------

def _archive_verdict_positive() -> dict:
    """12 flat rows on a primed baseline, steps clearing EFFORT_MIN_STEPS
    -> STALLED. Depends on the real update_stall() replay, not a
    hardcoded answer: see test_forge_stall.py's own revert-verify."""
    tail = [{"cells": 500, "steps": 100}]
    for i in range(12):
        tail.append({"cells": 500, "steps": 100 + (i + 1) * 25_000})
    return archive_verdict(tail, wall_id="gate_fixture")


def _archive_verdict_negative() -> dict:
    """A short, still-growing tail -> ADVANCING, never STALLED."""
    tail = [{"cells": c, "steps": 1000 * c} for c in (10, 20, 30)]
    return archive_verdict(tail, wall_id="gate_fixture")


def _campaign_verdict_positive() -> dict:
    """Three receipt members, all beat_3072: false, at MIN_TERMINAL ->
    STALLED."""
    manifest = {
        "wall_id": "gate_fixture_pos", "prior_best": 3072,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": f"gate_pos_{i}", "shape": "receipt",
             "receipt": "r.json", "terminal_field": "beat",
             "best_field": "score", "root_family": "x"}
            for i in range(3)
        ],
    }
    for i in range(3):
        d = _GATE_TMP / f"gate_pos_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "r.json").write_text('{"beat": false, "score": 3072}')
    return campaign_verdict(manifest, repo=_GATE_TMP)


def _campaign_verdict_negative() -> dict:
    """One member beats prior_best -> ADVANCING, never STALLED."""
    manifest = {
        "wall_id": "gate_fixture_neg", "prior_best": 3072,
        "prior_best_replay_verified": True,
        "members": [
            {"dir": "gate_neg_0", "shape": "receipt", "receipt": "r.json",
             "terminal_field": "beat", "best_field": "score",
             "root_family": "x"},
        ],
    }
    d = _GATE_TMP / "gate_neg_0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r.json").write_text('{"beat": true, "score": 4000}')
    return campaign_verdict(manifest, repo=_GATE_TMP)


#: (file, func) -> (verdict key, target value, positive case, negative case).
#: `positive()[key] == target` and `negative()[key] != target` must both
#: hold, proven fresh on every run (test_forge_registry_entries_report_
#: both_polarities below), the same discipline REGISTRY enforces for the
#: scanner's own idiom.
FORGE_REGISTRY = {
    ("src/forge/stall.py", "archive_verdict"): (
        "verdict", "STALLED", _archive_verdict_positive, _archive_verdict_negative),
    ("src/forge/stall.py", "campaign_verdict"): (
        "verdict", "STALLED", _campaign_verdict_positive, _campaign_verdict_negative),
}


def test_forge_registry_entries_report_both_polarities():
    """Every FORGE_REGISTRY entry is re-proven on every run: a refactor
    that quietly makes the gate unable to fail (drops the branch that
    can return the negative reading) is caught here.
    """
    failures = []
    for (file, func), (key, target, positive, negative) in FORGE_REGISTRY.items():
        pos_result = positive()
        neg_result = negative()
        if pos_result.get(key) != target:
            failures.append(
                f"{file}:{func} — positive case did not yield {key}="
                f"{target!r} (got {pos_result.get(key)!r}); the "
                f"demonstrated-pass case for this gate is broken")
        if neg_result.get(key) == target:
            failures.append(
                f"{file}:{func} — negative case DID yield {key}="
                f"{target!r}; this gate can no longer be observed to "
                f"fail")
    assert not failures, "\n".join(failures)


def test_forge_registry_has_no_duplicate_or_orphaned_entries():
    """Self-check on the registry's own shape, mirroring
    test_anti_vacuity_gates.py's test_registry_has_no_duplicate_or_
    orphaned_entries: catches a copy-paste registry entry before either
    test above would."""
    for key, value in FORGE_REGISTRY.items():
        assert isinstance(key, tuple) and len(key) == 2
        file, func = key
        assert file and func
        assert isinstance(value, tuple) and len(value) == 4
        verdict_key, target, positive, negative = value
        assert isinstance(verdict_key, str) and verdict_key
        assert isinstance(target, str) and target
        assert callable(positive) and callable(negative)
        assert positive is not negative


def test_no_unregistered_vacuous_shaped_gate_in_src_forge():
    """scripts/anti_vacuity_scan.py already sweeps all of src/ (so
    src/forge/ is covered the moment the package exists). This is the
    drift catch scoped to the Forge's own directory: any site the
    scanner finds under src/forge/ that is not accounted for in the
    project-wide REGISTRY (tests/test_anti_vacuity_gates.py) is a defect
    a future piece must fix by registering it there, per the drift test
    at tests/test_anti_vacuity_gates.py:247-263.

    Vacuously true tonight (piece (a) writes no `passed`/`ok`/`valid`/
    `clear`/`success` = `not X` shape) -- this exists so it stops being
    vacuous the moment a later piece adds one and forgets to register
    it, rather than that omission surfacing only in the top-level
    project-wide test.
    """
    from tests.test_anti_vacuity_gates import REGISTRY

    hits = [h for h in scan_repo() if h.file.startswith("src/forge/")]
    unregistered = [h for h in hits if (h.file, h.func) not in REGISTRY]
    assert not unregistered, (
        "New vacuous-shaped gate(s) found under src/forge/ with no "
        "registered proof in tests/test_anti_vacuity_gates.py::REGISTRY:\n"
        + "\n".join(f"  {h.file}:{h.line} func={h.func}" for h in unregistered))
