"""scripts/progress_gate_sweep.py — the roster re-run that produced the
"7 of 45 verdicts change" number.

A sweep whose own decision logic has no tests is how a headline number
becomes unfalsifiable, so the four decisions it makes are driven here:
which profiles are in the roster, which are out of the gate's domain,
what the calibration constant is derived from, and how two probes that
each see things the other cannot are composed into one reading.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.progress_gate_sweep import (
    LEGACY_MIN_WINDOW,
    calibration,
    compose,
    probe_plan,
    roster,
)
from scripts.progress_signal_gate import MIN_ASSESSABLE_STEPS

REPO = Path(__file__).resolve().parent.parent
HOLD = REPO / "docs/receipts/progress_gate_window_sweep_2026-08-26.json"
RANDOM = REPO / "docs/receipts/progress_gate_random_probe_2026-08-26.json"


def _row(profile, *, distinct, live, faults=(), verdict="v", probe="hold"):
    return {"profile": profile, "applicable": True, "probe": probe,
            "live_steps": live,
            "after": {"distinct": distinct, "verdict": verdict,
                      "instrument_findings": list(faults),
                      "steps_to_min_distinct": None}}


def test_the_roster_is_derived_not_listed():
    """Every configs/*.yaml with a `solve:` block, and nothing else. A
    hardcoded list would stop covering a profile the day one is
    onboarded, silently."""
    import yaml
    names = roster()
    assert len(names) >= 45
    for n in names:
        d = yaml.safe_load((REPO / n).read_text())
        assert d.get("solve"), f"{n} has no solve block but is in the roster"
    with_solve = {
        f"configs/{p.name}" for p in (REPO / "configs").glob("*.yaml")
        if isinstance(yaml.safe_load(p.read_text()), dict)
        and (yaml.safe_load(p.read_text()) or {}).get("solve")}
    assert set(names) == with_solve


def test_a_profile_with_no_progress_signal_is_inapplicable_not_failed():
    """punchout declares `source: fight_gate` and no `progress.lo`. The
    gate has nothing to assess there. Scoring that as a FAIL would pad
    the roster's failure count with profiles the gate was never defined
    on — the same conflation the INCONCLUSIVE band exists to stop."""
    plan = probe_plan("configs/punchout.yaml")
    assert plan["applicable"] is False
    assert "declares no scalar progress signal" in plan["reason"]


def test_an_ordinary_profile_is_applicable():
    """Anti-vacuity for the check above: if `applicable` were always
    False the sweep would score nothing and report zero changes."""
    plan = probe_plan("configs/contra.yaml")
    assert plan["applicable"] is True
    assert plan["forward"] == "right"
    assert plan["odometer"] is False


def test_forward_policy_default_matches_the_gates_own_default():
    """The banked verdicts were all measured with `--forward right`,
    including 1942's, whose declared axis is `-y`. The sweep must
    reproduce them, so `default` holds right on everything and only
    `axis` derives the hold from the declaration."""
    assert probe_plan("configs/1942.yaml", "default")["forward"] == "right"
    assert probe_plan("configs/1942.yaml", "axis")["forward"] == "up"


def test_calibration_population_is_signals_that_reach_the_bar():
    """Not "profiles that pass". Kung Fu reaches 91 distinct levels and
    fails on the unpaired-wrap check, which says nothing about
    resolution; its 187 steps to 32 levels is the roster's slowest and
    is exactly the evidence the floor is set from. Restricting to
    passing profiles drops it and yields 102."""
    rows = [
        {"profile": "a", "applicable": True,
         "after": {"passed": True, "steps_to_min_distinct": 40}},
        {"profile": "kungfu", "applicable": True,
         "after": {"passed": False, "steps_to_min_distinct": 187}},
        {"profile": "coarse", "applicable": True,
         "after": {"passed": False, "steps_to_min_distinct": None}},
        {"profile": "n/a", "applicable": False, "after": {}},
    ]
    cal = calibration(rows)
    assert cal["max"] == 187
    assert cal["slowest_profile"] == "kungfu"
    assert cal["profiles_reaching_min_distinct"] == 2


def test_compose_keeps_a_fault_either_probe_demonstrated():
    """Kung Fu's shape: the directed hold reaches 200 on an unpaired byte
    and reports the wrap; the random rollout never travels far enough to
    reach 200 and sees no fault at all. A composition that took the
    'better' verdict would launder a demonstrated fault away."""
    out = compose(
        [_row("configs/kungfu.yaml", distinct=91, live=1200,
              faults=["reaches >=200 with no paired high byte"])],
        [_row("configs/kungfu.yaml", distinct=35, live=1200, probe="random")])
    assert out[0]["verdict"] == "SIGNAL UNUSABLE"
    assert out[0]["faults"] == ["reaches >=200 with no paired high byte"]


def test_compose_takes_the_best_resolution_evidence():
    """Contra's shape: 20 distinct in the hold's 69 live steps, 346 in
    the random probe's 721, no fault from either."""
    out = compose(
        [_row("configs/contra.yaml", distinct=20, live=69)],
        [_row("configs/contra.yaml", distinct=346, live=721, probe="random")])
    assert out[0]["verdict"] == "SIGNAL SOUND"
    assert out[0]["best_probe"] == "random"


def test_compose_stays_inconclusive_when_neither_probe_got_a_window():
    """blaster_master: 22 live steps held forward, 102 under random,
    fewer than 32 levels either way and no fault demonstrated. Neither
    condemned nor certified."""
    out = compose(
        [_row("configs/blaster_master.yaml", distinct=15, live=22)],
        [_row("configs/blaster_master.yaml", distinct=28, live=102,
              probe="random")])
    assert out[0]["verdict"].startswith("INCONCLUSIVE")
    assert out[0]["faults"] == []


def test_compose_passes_inapplicable_through():
    out = compose([{"profile": "configs/punchout.yaml", "applicable": False}],
                  [])
    assert out[0]["verdict"] == "GATE INAPPLICABLE"


# --- the banked receipts, re-read as assertions ---------------------------

def test_the_banked_sweep_says_the_fix_never_unblocked_anything():
    """The safety property the whole design rests on: raising the window
    floor can only move a verdict from FAIL to VOID. If a future change
    made `passed` flip anywhere, the floor would have become an amnesty."""
    d = json.loads(HOLD.read_text())
    assert d["n_profiles"] == 45
    assert d["legacy_min_window"] == LEGACY_MIN_WINDOW == 0
    assert d["min_window"] == MIN_ASSESSABLE_STEPS
    flipped = [r["profile"] for r in d["rows"] if r.get("passed_changed")]
    assert flipped == [], flipped
    changed = [r for r in d["rows"] if r["changed"]]
    assert len(changed) == d["verdicts_changed"] == 7
    for r in changed:
        assert r["before"]["verdict"].startswith("SIGNAL UNUSABLE")
        assert r["after"]["verdict"].startswith("INCONCLUSIVE")
        assert r["live_steps"] < MIN_ASSESSABLE_STEPS


def test_the_banked_sweep_reproduces_the_earlier_receipt_exactly():
    """The `before` column is only a baseline if it IS the old gate. All
    43 profiles the previous sweep covered must come back with the same
    verdict and the same live-step count."""
    new = {r["profile"][8:]: r for r in json.loads(HOLD.read_text())["rows"]}
    old = json.loads(
        (REPO / "docs/receipts/progress_gate_stasis_sweep_2026-08-26.json")
        .read_text())
    assert len(old) == 43
    for o in old:
        r = new[o["profile"]]
        assert r["before"]["verdict"] == o["after_verdict"], o["profile"]
        assert r["live_steps"] == o["after_steps"], o["profile"]


def test_the_random_probe_bought_contra_a_real_window():
    """The probe half of the finding, as a number rather than a claim."""
    hold = {r["profile"]: r for r in json.loads(HOLD.read_text())["rows"]}
    rand = {r["profile"]: r for r in json.loads(RANDOM.read_text())["rows"]}
    c = "configs/contra.yaml"
    assert hold[c]["live_steps"] == 69 and hold[c]["after"]["distinct"] == 20
    assert rand[c]["live_steps"] == 721
    assert rand[c]["after"]["distinct"] == 346
    assert rand[c]["after"]["verdict"].startswith("SIGNAL SOUND")


def test_the_random_probe_is_not_uniformly_better():
    """The honest half. It loses windows on some games and would
    manufacture a shortfall on others if it were allowed to — so the
    receipt must keep showing at least one profile where the directed
    hold saw more."""
    hold = {r["profile"]: r for r in json.loads(HOLD.read_text())["rows"]}
    rand = {r["profile"]: r for r in json.loads(RANDOM.read_text())["rows"]}
    worse = [p for p in hold
             if hold[p].get("applicable")
             and rand[p]["live_steps"] < hold[p]["live_steps"]]
    assert worse, "no profile where the random probe got a shorter window"
    fewer = [p for p in hold
             if hold[p].get("applicable")
             and rand[p]["after"]["distinct"] < hold[p]["after"]["distinct"]]
    assert fewer, "no profile where the random probe saw fewer levels"
