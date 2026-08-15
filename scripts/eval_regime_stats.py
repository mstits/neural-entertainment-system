"""Pure-math statistics helpers for the eval-RNG-regime forensics.

Used by the 2026-08-15 protocol-gap analysis (14.4% campaign-probe rate vs
2.0% definitive-eval rate on the 1-2 line). No emulator, no torch, no eval
launches — these operate only on already-recorded per-episode outcomes read
from eval.jsonl / final_eval*.json / campaign.jsonl.

Functions
---------
fisher_exact_two_sided(a, b, c, d)
    Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]] (successes /
    failures per protocol), via the exact hypergeometric distribution.
bootstrap_rate_ci(k, n, ...)
    Percentile bootstrap CI for a binomial rate k/n.
runs_test_z(seq)
    Wald-Wolfowitz runs-test z statistic for a binary sequence — the streak
    detector for episode-order outcome sequences (shared-stream correlation
    probe). None when the sequence is single-class.
lag1_autocorr(seq)
    Lag-1 Pearson autocorrelation of a numeric sequence; None when variance
    is zero.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def _hypergeom_pmf(k: int, row1: int, col1: int, n: int) -> float:
    """P(X = k) for X ~ Hypergeometric(n, row1, col1) via math.comb."""
    return (
        math.comb(col1, k)
        * math.comb(n - col1, row1 - k)
        / math.comb(n, row1)
    )


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for the table [[a, b], [c, d]].

    Sums the probabilities of every table (with the same margins) whose
    point probability does not exceed the observed table's, the standard
    two-sided convention. A tiny relative tolerance absorbs float noise in
    the "as extreme" comparison.
    """
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    k_lo = max(0, row1 + col1 - n)
    k_hi = min(row1, col1)
    p_obs = _hypergeom_pmf(a, row1, col1, n)
    tol = p_obs * 1e-12
    total = 0.0
    for k in range(k_lo, k_hi + 1):
        p_k = _hypergeom_pmf(k, row1, col1, n)
        if p_k <= p_obs + tol:
            total += p_k
    return min(1.0, total)


def bootstrap_rate_ci(
    k: int,
    n: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap (1 - alpha) CI for a binomial rate k/n.

    Resamples n Bernoulli outcomes (k ones, n-k zeros) with replacement
    n_boot times and returns the (alpha/2, 1 - alpha/2) percentiles of the
    resampled rates. Deterministic for a given seed.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    p = k / n
    rates = sorted(
        sum(1 for _ in range(n) if rng.random() < p) / n
        for _ in range(n_boot)
    )
    lo_idx = int(math.floor((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)
    return rates[lo_idx], rates[hi_idx]


def runs_test_z(seq: Sequence[int]) -> Optional[float]:
    """Wald-Wolfowitz runs-test z statistic for a binary sequence.

    z > 0: more runs than an exchangeable (independent) ordering expects
    (alternation); z < 0: fewer runs (streaks / positive serial dependence).
    Returns None for sequences with fewer than two distinct values.
    """
    xs = [1 if x else 0 for x in seq]
    n1 = sum(xs)
    n2 = len(xs) - n1
    if n1 == 0 or n2 == 0:
        return None
    runs = 1 + sum(1 for i in range(1, len(xs)) if xs[i] != xs[i - 1])
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1.0))
    if var <= 0.0:
        return None
    return (runs - mu) / math.sqrt(var)


def lag1_autocorr(seq: Sequence[float]) -> Optional[float]:
    """Lag-1 Pearson autocorrelation; None when the sequence is constant."""
    n = len(seq)
    if n < 2:
        return None
    mean = sum(seq) / n
    var = sum((x - mean) ** 2 for x in seq)
    if var == 0.0:
        return None
    cov = sum((seq[i] - mean) * (seq[i + 1] - mean) for i in range(n - 1))
    return cov / var
