"""Stub-data tests for the eval-RNG-regime forensics statistics helpers.

Pure-math functions only — no emulator, no torch, no eval runs. Reference
values are hand-checked closed forms (hypergeometric sums via math.comb,
Wald-Wolfowitz normal approximation, empirical bootstrap quantiles).
"""

import math

import pytest

from scripts.eval_regime_stats import (
    bootstrap_rate_ci,
    fisher_exact_two_sided,
    lag1_autocorr,
    runs_test_z,
)


class TestFisherExact:
    def test_tea_tasting_table(self):
        # Fisher's classic [[3,1],[1,3]]: two-sided p = 0.485714...
        p = fisher_exact_two_sided(3, 1, 1, 3)
        assert p == pytest.approx(0.4857142857, abs=1e-9)

    def test_no_association_uniform(self):
        # Identical rates -> p == 1.0 (every table at least as extreme).
        p = fisher_exact_two_sided(5, 5, 5, 5)
        assert p == pytest.approx(1.0, abs=1e-12)

    def test_extreme_association(self):
        # [[10,0],[0,10]]: only the two corner tables are as extreme.
        # p = 2 / C(20,10) = 2/184756
        p = fisher_exact_two_sided(10, 0, 0, 10)
        assert p == pytest.approx(2.0 / math.comb(20, 10), rel=1e-9)

    def test_symmetry(self):
        assert fisher_exact_two_sided(1, 89, 2, 98) == pytest.approx(
            fisher_exact_two_sided(2, 98, 1, 89), rel=1e-12
        )


class TestBootstrapCI:
    def test_deterministic_given_seed(self):
        a = bootstrap_rate_ci(13, 90, n_boot=2000, seed=1)
        b = bootstrap_rate_ci(13, 90, n_boot=2000, seed=1)
        assert a == b

    def test_ci_brackets_point_estimate(self):
        lo, hi = bootstrap_rate_ci(13, 90, n_boot=5000, seed=7)
        assert lo <= 13 / 90 <= hi
        assert 0.0 <= lo < hi <= 1.0

    def test_degenerate_zero_successes(self):
        lo, hi = bootstrap_rate_ci(0, 50, n_boot=1000, seed=3)
        assert lo == 0.0
        assert hi == 0.0  # resampling zeros only ever yields zeros

    def test_wider_at_smaller_n(self):
        lo_s, hi_s = bootstrap_rate_ci(3, 30, n_boot=5000, seed=11)
        lo_l, hi_l = bootstrap_rate_ci(30, 300, n_boot=5000, seed=11)
        assert (hi_s - lo_s) > (hi_l - lo_l)


class TestRunsTest:
    def test_alternating_sequence_excess_runs(self):
        # 0101...: maximal runs -> strongly positive z.
        seq = [0, 1] * 20
        z = runs_test_z(seq)
        assert z is not None and z > 3.0

    def test_blocked_sequence_deficit_runs(self):
        # 000...111: 2 runs, far fewer than expected -> strongly negative z.
        seq = [0] * 20 + [1] * 20
        z = runs_test_z(seq)
        assert z is not None and z < -3.0

    def test_degenerate_single_class(self):
        assert runs_test_z([1, 1, 1, 1]) is None

    def test_known_value(self):
        # n1=n2=5, runs=6: mu = 2*5*5/10+1 = 6 -> z = 0 exactly.
        seq = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
        assert runs_test_z(seq) == pytest.approx(0.0, abs=1e-12)


class TestLag1Autocorr:
    def test_alternating_negative(self):
        # Biased (denominator-n) estimator: perfect alternation of length n
        # reads exactly -(n-1)/n, not -1.
        r = lag1_autocorr([0, 1] * 25)
        assert r == pytest.approx(-49.0 / 50.0, abs=1e-12)

    def test_blocked_positive(self):
        r = lag1_autocorr([0] * 25 + [1] * 25)
        assert r is not None and r > 0.9

    def test_constant_is_none(self):
        assert lag1_autocorr([1] * 10) is None
