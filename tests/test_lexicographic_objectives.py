"""Tests for src/training/lexicographic_objectives.py (the learnfun port).

Three things are being protected here:

  1. THE PORT IS CORRECT. Tom7's objective.cc ships a self-check —
     CheckOrdering — that every emitted ordering must satisfy: over the
     look set it was enumerated on, the first RAM location where two
     consecutive memories differ never goes DOWN. With `decrease_tol=0`
     a correct port reports exactly zero violations, so that property is
     the invariant every ranking test runs behind.

  2. THE RANKING FINDS STRUCTURE, AND ALSO FINDS THE TRAP. A synthetic
     trace with a progress byte, a score digit, a stage counter and a
     free-running timer reproduces both halves of the real measurement:
     the structural bytes take the top lead ranks, and the timer takes
     the heaviest MASS — outranking real progress, exactly as SMB's
     $07C7 does on the banked 1-1 tape (lead-rank 6, mass 43.7). Nothing
     in this module can reject it; only Gate 1 (idling the emulator) can,
     so the gate is exercised here too.

  3. THE QUALITY STATISTIC IS CONCENTRATION, NOT WEIGHT. The negative
     control inverts total objective weight, so `concentration_verdict`
     is what adjudicates trajectory validity. It is tested against the
     measured receipts below.

A NOTE FOR WHOEVER TOUCHES THE CONCENTRATION TESTS NEXT: the
concentration effect does NOT reproduce on a toy trace, and the fixtures
here deliberately do not pretend otherwise. Measured while writing these
tests, over seven seeds of a 32-location synthetic: the STRUCTURELESS
variant comes out MORE concentrated than the solved one every time
(d_leaders +1..+8, d_entropy +0.14..+1.95 bits). The reason is scale — a
real broken tape still has all 2048 bytes alive and churning, so its
lead mass scatters, whereas a toy's broken variant simply has fewer
bytes able to lead. Do not "fix" a failing concentration test by
asserting the toy's direction; the real numbers are in
`MEASURED_CONCENTRATION` and they are what the statistic was validated
on.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.discover_observables import _col_stats, _flat_under_noop
from src.training.lexicographic_objectives import (
    NEVER_WIRE_AS_REWARD,
    Objective,
    build_scorer,
    changing_frames,
    check_ordering,
    concentration,
    concentration_verdict,
    curve_stats,
    learn_objectives,
    rank_locations,
    score_trace,
    skip_until_first_input,
    value_fraction_weight,
)

# Synthetic RAM layout. Addresses are arbitrary; the roles are the ones
# the real receipts turn on.
PROGRESS, SCORE, STAGE, TIMER = 3, 5, 6, 9
QUIET = (11, 13, 17, 19, 23)      # sparse steppers (room/level-ish)
NOISY = (2, 8, 14, 20, 26)        # churn (velocity/RNG-ish)
N_LOC = 32

#: Lead-mass concentration as MEASURED by the validation run (2026-08-10),
#: all three at the same 903-memory length so they are comparable.
#: `total_weight` is carried only to show that it moves the WRONG way.
MEASURED_CONCENTRATION = {
    #                        n_leaders  entropy_bits  total_weight
    "smb_1_1_correct":      (77,        5.01,         82.1),
    "smb_1_1_wrong_root":   (99,        6.07,         205.2),
    "smb_1_1_shuffled":     (148,       6.92,         218.7),
}
#: The 32-tape SMB chain — same game, 31590 memories. Concentration is
#: strongly length-dependent (31 leaders / 3.85 bits), which is why a
#: cross-length comparison must be refused rather than reported.
MEASURED_SMB_CHAIN = (31590, 31, 3.85)


def _conc(name: str) -> dict:
    n_leaders, bits, _ = MEASURED_CONCENTRATION[name]
    return {"n_leaders": n_leaders, "lead_entropy_bits": bits,
            "lead_total": 0.0, "top1_share": 0.0, "top3_share": 0.0}


def solved_trace(n: int = 240, seed: int = 7) -> np.ndarray:
    """A trace that really is progress: monotone position, score and stage."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n, N_LOC), dtype=np.uint8)
    t = np.arange(n)
    M[:, TIMER] = t.astype(np.uint8)                 # ticks regardless
    M[:, PROGRESS] = (t // 4).astype(np.uint8)
    M[:, SCORE] = (t // 40).astype(np.uint8)
    M[:, STAGE] = (t // 90).astype(np.uint8)
    for c in QUIET:
        k = int(rng.integers(2, 6))
        edges = np.sort(rng.choice(n, size=k, replace=False))
        M[:, c] = np.searchsorted(edges, t).astype(np.uint8)
    for c in NOISY:
        M[:, c] = rng.integers(0, 40, n).astype(np.uint8)
    return M


def structureless_trace(n: int = 240, seed: int = 7) -> np.ndarray:
    """Same bytes, same activity, no progress: the structural counters
    wander instead of climbing. The timer ticks identically — a timer is
    cadence-invariant, which is exactly why it survives this control."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n, N_LOC), dtype=np.uint8)
    t = np.arange(n)
    M[:, TIMER] = t.astype(np.uint8)
    for c, step in ((PROGRESS, 4), (SCORE, 2), (STAGE, 1)):
        M[:, c] = np.clip(40 + np.cumsum(rng.integers(-step, step + 1, n)),
                          0, 255).astype(np.uint8)
    for c in QUIET:
        k = int(rng.integers(2, 6))
        edges = np.sort(rng.choice(n, size=k, replace=False))
        M[:, c] = rng.permutation(np.searchsorted(edges, t)).astype(np.uint8)
    for c in NOISY:
        M[:, c] = rng.integers(0, 40, n).astype(np.uint8)
    return M


# ---------------------------------------------------------------------
# 1. Port correctness: Tom7's CheckOrdering property.
# ---------------------------------------------------------------------

def test_check_ordering_counts_only_leading_decreases():
    """A pair violates when the FIRST differing location goes down; a
    lower-significance byte dropping under a rising leader does not."""
    M = np.array([[1, 9], [1, 3], [2, 0], [1, 9]], dtype=np.uint8)
    look = np.arange(4)
    # [0, 1]: row1->2 leads with 1->2 (up, fine) even though the second
    # byte falls; row0->1 leads with the SECOND byte falling 9->3.
    n_bad, n_pairs = check_ordering(M, look, [0, 1])
    assert (n_bad, n_pairs) == (2, 3)          # rows 0->1 and 2->3
    # Reversed significance: now the falling byte leads more often.
    assert check_ordering(M, look, [1, 0])[0] == 2
    # An ordering over a constant location can never violate.
    assert check_ordering(np.zeros((5, 2), np.uint8), np.arange(5), [0]) == (0, 4)


def test_every_learned_ordering_is_valid_over_its_own_look_set():
    """THE port invariant. `learn_objectives(verify=True)` runs
    CheckOrdering over every emitted ordering against the look set it was
    enumerated on; at decrease_tol=0 a correct port reports zero."""
    M = solved_trace()
    res = learn_objectives(M, whole=12, verify=True)
    bad = res["invalid_orderings"]
    assert bad["violating_pairs"] == 0, bad
    assert bad["orderings_over_tol"] == 0, bad
    assert bad["total_pairs"] > 0            # the check actually ran
    assert res["n_unique"] > 0 and res["n_positive"] > 0


def test_enumerated_orderings_are_independently_valid():
    """Re-derive the property outside `learn_objectives`, so a bug in its
    bookkeeping cannot hide a bug in the enumeration."""
    M = solved_trace(seed=3)
    obj = Objective(M, decrease_tol=0.0)
    look = changing_frames(M)
    orderings = [o for i in range(8)
                 for o in obj.enumerate_full(look, 1, i)]
    assert orderings, "enumeration produced nothing to check"
    for o in orderings:
        assert len(set(o)) == len(o), "a location repeats in one ordering"
        assert check_ordering(M, look, o)[0] == 0


def test_decrease_tol_admits_a_single_frame_blip():
    """The one documented deviation from Tom7: a real progress counter
    that blips DOWN for one frame (Bubble Bobble's round byte reads
    67->66->68 across its clear) is rejected outright at tol=0 and
    admitted with a small tolerance."""
    n = 60
    M = np.zeros((n, 4), dtype=np.uint8)
    M[:, 0] = np.arange(n) // 2 + 60    # the round counter
    M[30, 0] = M[29, 0] - 1             # the blip: one frame BELOW its run
    M[:, 1] = np.arange(n) // 20        # a clean companion
    look = changing_frames(M)
    # The tolerance is a FRACTION of the equal-prefix transitions, so the
    # budget only opens up on a trace with enough of them: 30 pairs here,
    # int(0.10 * 30) = 3 permitted dips against the one that occurs.
    assert look.size >= 20
    strict = Objective(M, decrease_tol=0.0).enumerate_full(look, 4, 0)
    loose = Objective(M, decrease_tol=0.10).enumerate_full(look, 4, 0)
    assert all(0 not in o for o in strict), "tol=0 must reject the blipper"
    assert any(0 in o for o in loose), "a small tol must admit it"


def test_value_fraction_weight_zeroes_a_losing_objective():
    """WeightByExamples: VF(last) - VF(first), floored at zero."""
    up = np.array([[0], [1], [2], [3]], dtype=np.uint8)
    w, n_values, vf_begin, vf_end = value_fraction_weight(up, [0])
    assert n_values == 4 and vf_begin == 0.0 and vf_end == 0.75
    assert w == pytest.approx(0.75)
    assert value_fraction_weight(up[::-1], [0])[0] == 0.0   # lost > gained


def test_skip_until_first_input_drops_the_boot_prefix():
    M = np.arange(10, dtype=np.uint8).reshape(10, 1)
    masks = np.array([0, 0, 0, 2, 0, 8, 0, 0, 0, 0])
    out, start = skip_until_first_input(M, masks)
    assert start == 3 and out.shape[0] == 7 and out[0, 0] == 3
    # No input at all: keep the whole trace rather than emptying it.
    assert skip_until_first_input(M, np.zeros(10, int))[1] == 0


# ---------------------------------------------------------------------
# 2. Ranking: structure is recovered, and so is the trap.
# ---------------------------------------------------------------------

def test_ranking_recovers_the_structural_progress_bytes():
    """The instrument's whole job: shortlist the bytes worth adjudicating."""
    M = solved_trace()
    ranked = rank_locations(learn_objectives(M, whole=12, verify=False),
                            memories=M)
    top8 = [r["addr"] for r in ranked[:8]]
    for addr, role in ((PROGRESS, "progress"), (SCORE, "score"),
                       (STAGE, "stage")):
        assert addr in top8, f"{role} byte ${addr:04X} missed: {top8}"
    # Ranks are 1-based, dense and ordered by the requested statistic.
    assert [r["rank"] for r in ranked[:5]] == [1, 2, 3, 4, 5]
    leads = [r["lead"] for r in ranked]
    assert leads == sorted(leads, reverse=True)
    by_mass = rank_locations(learn_objectives(M, whole=12, verify=False),
                             by="mass", memories=M)
    masses = [r["mass"] for r in by_mass]
    assert masses == sorted(masses, reverse=True)


def test_a_free_running_timer_outranks_true_progress():
    """The $07C7 trap, reproduced. The timer carries the heaviest mass of
    any location — more than the byte that actually is progress — because
    it climbs monotonically on every single frame. This is a statement
    about the LIMIT of lexicographic ranking, and it is why nothing here
    may be trusted without Gate 1."""
    M = solved_trace()
    ranked = rank_locations(learn_objectives(M, whole=12, verify=False),
                            memories=M)
    mass = {r["addr"]: r["mass"] for r in ranked}
    assert mass[TIMER] == max(mass.values())
    assert mass[TIMER] > mass[PROGRESS]


def test_gate1_noop_flatness_kills_the_timer_the_ranking_loved():
    """Gate 1 is the only thing that can reject it, and it does.

    The NOOP probe is the discriminator: idle the emulator and a timer
    keeps ticking while position/score/stage sit still. Fed a synthetic
    idle log with exactly that shape, `_flat_under_noop` rejects the
    top-mass byte and passes every structural one."""
    n = 240
    idle = np.zeros((n, N_LOC), dtype=np.uint8)
    idle[:, TIMER] = np.arange(n).astype(np.uint8)      # still ticking
    idle[:, PROGRESS] = 40                              # parked
    idle[:, SCORE] = 3
    idle[:, STAGE] = 1
    sn = _col_stats(idle)
    assert not _flat_under_noop(sn, TIMER)
    for addr in (PROGRESS, SCORE, STAGE, *QUIET):
        assert _flat_under_noop(sn, addr), f"${addr:04X} wrongly rejected"


# ---------------------------------------------------------------------
# 3. Concentration is the quality statistic — total weight is not.
# ---------------------------------------------------------------------

def test_concentration_reads_a_lead_distribution():
    """Exact arithmetic against hand-built lead mass."""
    flat = {i: {"lead": 1.0, "mass": 1.0, "disc": 1.0, "n": 1,
                "best_pos": 0} for i in range(8)}
    c = concentration(flat)
    assert c["n_leaders"] == 8
    assert c["lead_entropy_bits"] == pytest.approx(3.0)      # log2(8)
    assert c["top1_share"] == pytest.approx(0.125)
    assert c["lead_total"] == pytest.approx(8.0)

    peaked = {0: {"lead": 9.0, "mass": 9.0, "disc": 9.0, "n": 1, "best_pos": 0},
              1: {"lead": 1.0, "mass": 1.0, "disc": 1.0, "n": 1, "best_pos": 0}}
    p = concentration(peaked)
    assert p["n_leaders"] == 2 and p["top1_share"] == pytest.approx(0.9)
    assert p["lead_entropy_bits"] < c["lead_entropy_bits"]

    # Locations that never lead are not leaders, and an all-zero table is
    # reported as empty rather than dividing by zero.
    tail = {0: {"lead": 0.0, "mass": 5.0, "disc": 5.0, "n": 3, "best_pos": 2}}
    assert concentration(tail)["n_leaders"] == 0


def test_concentration_verdict_ranks_the_measured_controls():
    """The receipt, encoded. Same tape, same length, three trajectories:
    the real solve, the wrong root, and the shuffled actions."""
    n = 903
    correct, wrong, shuf = (_conc("smb_1_1_correct"),
                            _conc("smb_1_1_wrong_root"),
                            _conc("smb_1_1_shuffled"))
    for broken in (wrong, shuf):
        v = concentration_verdict(correct, broken, n_fit=n, n_reference=n)
        assert v["verdict"] == "concentrated", v
        assert v["length_matched"] is True
        assert v["d_leaders"] < 0 and v["d_entropy_bits"] < 0
        # ... and the comparison is antisymmetric.
        back = concentration_verdict(broken, correct, n_fit=n, n_reference=n)
        assert back["verdict"] == "diffuse", back


def test_total_weight_would_have_ranked_the_controls_BACKWARDS():
    """Why `concentration` exists at all. The broken trajectories score
    2.5x and 2.7x MORE total objective weight than the real solve, so any
    gate built on weight would have called them the better runs."""
    correct = MEASURED_CONCENTRATION["smb_1_1_correct"][2]
    for name in ("smb_1_1_wrong_root", "smb_1_1_shuffled"):
        assert MEASURED_CONCENTRATION[name][2] > 2.4 * correct
    # And the statistic that IS used never reports weight as quality.
    v = concentration_verdict(_conc("smb_1_1_correct"),
                              _conc("smb_1_1_shuffled"),
                              n_fit=903, n_reference=903)
    assert "total" not in v["statistic"].lower()
    assert "total objective weight" in v["not_used"]


def test_concentration_verdict_refuses_a_length_mismatch():
    """Concentration is length-dependent — the 31590-memory SMB chain
    sits on 31 leaders where the 903-memory single level spreads over 77.
    Comparing across lengths is refused, not reported."""
    n_chain, leaders, bits = MEASURED_SMB_CHAIN
    chain = {"n_leaders": leaders, "lead_entropy_bits": bits,
             "lead_total": 0.0, "top1_share": 0.0, "top3_share": 0.0}
    v = concentration_verdict(chain, _conc("smb_1_1_correct"),
                              n_fit=n_chain, n_reference=903)
    assert v["verdict"] == "inconclusive_length_mismatch"
    assert v["length_matched"] is False
    # Within tolerance the same pair adjudicates normally.
    ok = concentration_verdict(chain, chain, n_fit=n_chain,
                               n_reference=n_chain)
    assert ok["length_matched"] is True and ok["verdict"] == "mixed"


def test_concentration_verdict_without_a_reference_says_so():
    v = concentration_verdict(_conc("smb_1_1_correct"), None, n_fit=903)
    assert v["verdict"] == "no_reference"
    assert v["d_leaders"] is None and v["length_matched"] is None


def test_concentration_runs_end_to_end_on_learned_objectives():
    """Shape/wiring check on real learned output (see the module note:
    the toy inverts the DIRECTION, so only the plumbing is asserted)."""
    M = solved_trace()
    c = concentration(learn_objectives(M, whole=12, verify=False))
    assert c["n_leaders"] >= 1
    assert 0.0 < c["top1_share"] <= 1.0
    assert c["top1_share"] <= c["top3_share"] <= 1.0 + 1e-9
    assert c["lead_entropy_bits"] >= 0.0
    other = concentration(learn_objectives(structureless_trace(),
                                           whole=12, verify=False))
    assert other["n_leaders"] >= 1
    v = concentration_verdict(c, other, n_fit=240, n_reference=240)
    assert v["verdict"] in ("concentrated", "diffuse", "mixed")


# ---------------------------------------------------------------------
# 4. The scorer: measurement only, never a reward.
# ---------------------------------------------------------------------

def test_scorer_tracks_time_in_sample_and_drops_zero_weight_objectives():
    M = solved_trace()
    res = learn_objectives(M, whole=12, verify=False)
    entries = build_scorer(M, res)
    assert entries, "no positive-weight objectives to freeze"
    assert len(entries) == res["n_positive"]
    assert all(w > 0 for _, _, _, w in entries)
    unweighted, weighted = score_trace(entries, M)
    assert unweighted.shape == weighted.shape == (M.shape[0],)
    st = curve_stats(weighted)
    # In-sample rho is what the fit maximizes — high here proves the
    # ladder was frozen and replayed correctly, and proves nothing about
    # transfer. That is the point of NEVER_WIRE_AS_REWARD.
    assert st["spearman_vs_time"] > 0.9
    assert st["delta_end_begin"] > 0
    assert "-0.771" in NEVER_WIRE_AS_REWARD


def test_scorer_does_not_transfer_to_a_structureless_trace():
    """The held-out failure in miniature: freeze the ladder on one
    trajectory, score another, and the in-sample agreement evaporates."""
    fit = solved_trace()
    entries = build_scorer(fit, learn_objectives(fit, whole=12, verify=False))
    in_sample = curve_stats(score_trace(entries, fit)[1])["spearman_vs_time"]
    held_out = curve_stats(score_trace(entries, structureless_trace())[1])
    assert in_sample > 0.9
    assert held_out["spearman_vs_time"] < in_sample - 0.2
