"""scripts/phase3_adjudicate.py — the Phase-3 gate, written before results.

The tests pin the boundaries most likely to be argued with after the
fact: relative rather than absolute improvement, a zero control being
unscorable rather than infinite, and short or single-seed evidence being
refused rather than accepted with a caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.phase3_adjudicate import (  # noqa: E402
    GATE_RELATIVE_IMPROVEMENT, adjudicate, pooled_rate)


def rec(seed, n=50, rate=0.4, sticky=0.25):
    return {"n_episodes": n, "sticky_prob": sticky, "eval_seed": seed,
            "clear_rate": rate}


def two(rate):
    return [rec(7, rate=rate), rec(101, rate=rate)]


def test_gate_is_relative_not_absolute():
    """0.38 -> 0.45 is +7 points and still a FAIL: +18.4% relative."""
    assert adjudicate(two(0.38), two(0.46))["verdict"] == "PASS"
    assert adjudicate(two(0.38), two(0.45))["verdict"] == "FAIL"


def test_exactly_at_the_threshold_passes():
    ctrl, treat = 0.40, 0.40 * (1 + GATE_RELATIVE_IMPROVEMENT)
    v = adjudicate(two(ctrl), two(treat))
    assert v["verdict"] == "PASS" and v["relative_improvement"] >= 0.20


def test_a_zero_control_is_unscorable_not_infinite():
    v = adjudicate(two(0.0), two(0.10))
    assert v["verdict"] == "UNSCORABLE"
    assert any("undefined" in p for p in v["problems"])
    assert v["relative_improvement"] is None


def test_a_regression_fails():
    v = adjudicate(two(0.40), two(0.20))
    assert v["verdict"] == "FAIL" and v["relative_improvement"] < 0


def test_too_few_episodes_is_unscorable():
    v = adjudicate([rec(7, n=50), rec(101, n=20)], two(0.9))
    assert v["verdict"] == "UNSCORABLE"


def test_a_single_seed_is_unscorable():
    v = adjudicate([rec(7, n=200)], [rec(7, n=200, rate=0.9)])
    assert v["verdict"] == "UNSCORABLE"
    assert any("seed" in p for p in v["problems"])


def test_mismatched_eval_seeds_are_unscorable():
    """Control and masked must be scored on the same eval seeds; two
    equally-sized but disjoint seed sets are not a matched-seed gate."""
    v = adjudicate([rec(1, rate=0.40), rec(2, rate=0.40)],
                   [rec(3, rate=0.90), rec(4, rate=0.90)])
    assert v["verdict"] == "UNSCORABLE"
    assert any("eval seeds differ" in p for p in v["problems"])


def test_non_honest_records_are_not_admissible():
    """A deterministic (sticky=0) eval is not the honest protocol."""
    cl, n, seeds = pooled_rate([rec(7, sticky=0.0), rec(101, sticky=0.0)])
    assert n == 0 and cl == 0 and seeds == []


def test_repeated_rows_for_a_seed_are_deduped_not_pooled():
    """eval.jsonl is append-only: a seed can carry a sanity probe, a
    retry, and a final measurement. Only the last admissible row for a
    given seed counts — a stale re-probe must not add its own episodes
    and clears on top of the intended final measurement."""
    cl, n, seeds = pooled_rate([rec(7, rate=0.28), rec(7, rate=0.04)])
    assert n == 50 and cl == 2 and seeds == [7]


def test_failure_carries_the_synthesis_instruction():
    """The gate forbids rescuing a null by retuning."""
    v = adjudicate(two(0.40), two(0.41))
    assert v["verdict"] == "FAIL"
    assert "terminate" in v["instruction"]
    assert "tune the threshold" in v["instruction"]


def test_a_mostly_inert_veto_is_flagged_beside_the_verdict():
    v = adjudicate(two(0.40), two(0.40),
                   veto_stats={"fully_vetoed_fraction": 0.82})
    assert v["veto_caveat"] and "second control" in v["veto_caveat"]


def test_a_live_veto_carries_no_caveat():
    v = adjudicate(two(0.40), two(0.50),
                   veto_stats={"fully_vetoed_fraction": 0.13})
    assert v["veto_caveat"] is None


def test_identical_arm_fingerprints_are_void_not_null():
    """Two independently trained policies do not tie to the digit on rate
    AND mean length at both seeds; when they do, nothing trained."""
    def rec2(seed, rate, ml):
        return {"n_episodes": 50, "sticky_prob": 0.25, "eval_seed": seed,
                "clear_rate": rate, "mean_length": ml}
    same = [rec2(7, 0.28, 653.3), rec2(101, 0.34, 805.0)]
    v = adjudicate(list(same), list(same))
    assert v["verdict"] == "UNSCORABLE"
    assert any("frozen-actor" in p for p in v["problems"])
    # genuinely different arms still adjudicate normally
    other = [rec2(7, 0.30, 640.0), rec2(101, 0.36, 790.2)]
    v2 = adjudicate(same, other)
    assert v2["verdict"] in ("PASS", "FAIL")
