"""Comparison-outcome detector: transfer entropy over recorded RAM traces.

~140 of 789 library titles (sports, racing, board/quiz) cannot be scored
by a state-transition win predicate because success is a COMPARISON —
player score > opponent score, finishing position <= target, elapsed
time under a target. This module finds the RAM field(s) that implement
that comparison, from OUR OWN recorded play, and classifies each field
into one of four pre-registered outcome types or ABSTAINs.

The discriminator (research round v17, see
docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md) is TRANSFER ENTROPY
FROM THE CONTROLLER: a player's score responds to our input; an
opponent's does not; a timer responds to nobody. Nothing here is
authored per-game. No disassembly, no RAM maps, no wikis, no
walkthroughs, no genre priors baked into thresholds by name — every
verdict is a statistic computed from the trace itself: a discrete
transfer-entropy estimator with a permutation-null significance test, a
structural symmetry detector over update statistics, and a monotone-
decrement test. Where evidence is thin the classifier ABSTAINs
(UNSCORABLE) rather than guess — a false SCORE_WIN_VS_OPPONENT claim is
worse than reporting nothing, so every significance test below is
FDR-corrected across the fields it is applied to, and the default
thresholds are conservative on purpose.

This module drives NO emulator. It consumes RECORDED traces only — see
`Trace`/`load_trace` for the two supported formats (.npz, .jsonl). The
one function that *can* touch a live Pool, `record_trace_live`, is
gated behind an explicit `allow=True` / `--i-know-this-drives-the-
emulator` flag and imports nes_core lazily inside its own body, so
importing this module, running its classifier, or running its test
suite never touches the emulator. See that function's docstring for the
exact command to run once the 2-1 campaign frees the emulator.

Trace format
------------
A trace is `T` RAM frames plus the `T-1` controller inputs that produced
them, using the tape-replay convention already established by
src/training/tape_replay.py: `ram[n]` is memory BEFORE `inputs[n]` runs,
so `inputs[n]` is the action that transitions `ram[n] -> ram[n+1]`. The
controller byte layout matches scripts/discover_observables.py: bit0=A,
bit1=B, bit2=SELECT, bit3=START, bit4=UP, bit5=DOWN, bit6=LEFT,
bit7=RIGHT (this module never reads the layout, it is only mentioned
for the deferred recorder's callers).

  .npz  — keys `ram` (T, N) uint8, `inputs` (T-1,) uint8, optional
          `meta` (a JSON string).
  .jsonl — one frame per line, `{"ram": [...], "input": <int>}`; the
          last line's "input" (if present) is ignored since there is no
          T-th transition.

No trace format existed in-repo before this file; this is the minimal
one this lane needs, documented here rather than invented silently.

Usage
-----
  python scripts/outcome_probe.py classify --trace path/to/trace.npz
  python scripts/outcome_probe.py selftest      # 5 synthetic ground-truth cases
  python scripts/outcome_probe.py record --rom ... --i-know-this-drives-the-emulator
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field as _dc_field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent

RAM_SIZE = 0x800

# ---------------------------------------------------------------------
# Trace container + I/O.
# ---------------------------------------------------------------------


@dataclass
class Trace:
    """`ram[n]` is the memory BEFORE `inputs[n]` runs; `inputs[n]` is the
    action that transitions `ram[n] -> ram[n+1]`. `len(inputs) ==
    len(ram) - 1` is enforced so every downstream index is unambiguous —
    the same discipline src/training/tape_replay.py uses for tapes."""

    ram: np.ndarray
    inputs: np.ndarray
    meta: dict = _dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ram = np.asarray(self.ram, dtype=np.uint8)
        self.inputs = np.asarray(self.inputs, dtype=np.uint8).reshape(-1)
        if self.ram.ndim != 2:
            raise ValueError(f"ram must be a 2D (T, N) array; got shape {self.ram.shape}")
        if len(self.inputs) != len(self.ram) - 1:
            raise ValueError(
                "inputs must have exactly len(ram)-1 entries (one action per "
                f"transition); got {len(self.inputs)} inputs for {len(self.ram)} ram frames"
            )

    def __len__(self) -> int:
        return int(self.ram.shape[0])


def load_trace(path) -> Trace:
    """Load a `Trace` from `.npz` or `.jsonl` — see module docstring for
    the exact shapes. Raises on any other suffix rather than guessing."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            ram = z["ram"]
            inputs = z["inputs"]
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return Trace(ram=ram, inputs=inputs, meta=meta)
    if path.suffix in (".jsonl", ".ndjson"):
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"{path}: empty trace")
        ram = np.array([r["ram"] for r in rows], dtype=np.uint8)
        inputs = np.array([r.get("input", 0) for r in rows[:-1]], dtype=np.uint8)
        meta = rows[0]["meta"] if isinstance(rows[0].get("meta"), dict) else {}
        return Trace(ram=ram, inputs=inputs, meta=meta)
    raise ValueError(f"unsupported trace format: {path.suffix!r} (use .npz or .jsonl)")


def save_trace_npz(path, ram, inputs, meta: Optional[dict] = None) -> None:
    """Write a trace in the `.npz` shape `load_trace` reads back. Used by
    `record_trace_live` and by tests that round-trip synthetic traces
    through disk."""
    ram = np.asarray(ram, dtype=np.uint8)
    inputs = np.asarray(inputs, dtype=np.uint8)
    kwargs = {"ram": ram, "inputs": inputs}
    if meta is not None:
        kwargs["meta"] = json.dumps(meta)
    np.savez(path, **kwargs)


# ---------------------------------------------------------------------
# Fields: a byte, or an adjacent 2-byte little/big-endian pair.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """A candidate observable: one byte, or two bytes combined into a
    16-bit value. `offsets` and `endianness` are the only inputs to
    `extract` — nothing about *what the field means* is ever assumed."""

    name: str
    offsets: tuple
    endianness: str = "le"

    def __post_init__(self) -> None:
        if len(self.offsets) not in (1, 2):
            raise ValueError("Field.offsets must have length 1 (byte) or 2 (multi-byte)")
        if self.endianness not in ("le", "be"):
            raise ValueError("endianness must be 'le' or 'be'")

    @property
    def width_bytes(self) -> int:
        return len(self.offsets)

    def extract(self, ram: np.ndarray) -> np.ndarray:
        ram = np.asarray(ram)
        if ram.ndim != 2:
            raise ValueError("ram must be a (T, N) array")
        if len(self.offsets) == 1:
            return ram[:, self.offsets[0]].astype(np.int64)
        a, b = self.offsets
        lo, hi = (a, b) if self.endianness == "le" else (b, a)
        return ram[:, lo].astype(np.int64) | (ram[:, hi].astype(np.int64) << 8)


def enumerate_fields(n_bytes: int, include_pairs: bool = True) -> list:
    """Every single byte, plus every ADJACENT 2-byte pair in both byte
    orders. Non-adjacent or wider fields are out of scope for the
    auto-scan (an O(N^2) sweep of arbitrary-width fields over a 2KB RAM
    image is not a defensible default); pass an explicit `--fields` file
    (see `load_fields`) for those."""
    fields = [Field(name=f"b{i}", offsets=(i,)) for i in range(n_bytes)]
    if include_pairs:
        for i in range(n_bytes - 1):
            fields.append(Field(name=f"u16le_{i}_{i + 1}", offsets=(i, i + 1), endianness="le"))
            fields.append(Field(name=f"u16be_{i}_{i + 1}", offsets=(i, i + 1), endianness="be"))
    return fields


def load_fields(path) -> list:
    """`[{"name": ..., "offsets": [..], "endianness": "le"}, ...]`."""
    data = json.loads(Path(path).read_text())
    return [
        Field(name=row["name"], offsets=tuple(int(x) for x in row["offsets"]),
              endianness=row.get("endianness", "le"))
        for row in data
    ]


# ---------------------------------------------------------------------
# Discrete transfer entropy, TE(controller -> field), with a permutation
# null. Plug-in TE estimators are known to carry a positive small-sample
# bias (worse the sparser the joint table); rather than trust the raw
# bits figure we compare it to the SAME estimator run on `n_surrogates`
# random permutations of the controller stream, which reproduces the
# same finite-sample bias under the null of "no dependence" — the raw
# number is never the decision, the permutation p-value is.
# ---------------------------------------------------------------------

DEFAULT_STATE_BINS = 8
DEFAULT_X_BINS = 16


def _digitize(values: np.ndarray, max_bins: int) -> np.ndarray:
    """Compact-alphabet remap: exact index if the field takes at most
    `max_bins` distinct values in-trace, else quantile bins. Applied
    identically to the controller stream and to every field's values —
    nothing here is field- or game-specific."""
    v = np.asarray(values).astype(np.int64)
    uniq = np.unique(v)
    if uniq.size <= max_bins:
        return np.searchsorted(uniq, v).astype(np.int64)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, max_bins + 1)))
    if edges.size < 3:
        return np.searchsorted(uniq, v).astype(np.int64)
    return np.digitize(v, edges[1:-1], right=False).astype(np.int64)


def transfer_entropy(controller: np.ndarray, values: np.ndarray,
                      state_bins: int = DEFAULT_STATE_BINS,
                      x_bins: int = DEFAULT_X_BINS, lag: int = 1) -> float:
    """TE(X->Y) in bits, order-`lag`: `sum p(x_{n-lag+1}, y_n, y_{n+1}) *
    log2[p(y_{n+1}|x_{n-lag+1},y_n) / p(y_{n+1}|y_n)]`. `controller[n]` is
    the action that produced `values[n+1]` from `values[n]` (the trace's
    own order-1 convention), so `len(controller) == len(values) - 1` is
    required regardless of `lag`. `lag=1` (the default) reproduces the
    original order-1 test exactly: the predictor is `controller[n]`
    itself. `lag>1` tests whether the transition was instead driven by a
    controller sample taken `lag-1` steps earlier than the order-1
    convention assumes — a response that lands one or more recorded
    steps later than immediate (see `best_lag_te` / `MAX_LAG`, and
    the module docstring: an order-1-only test has no power to see
    this, and "no power to see it" is not evidence it isn't there)."""
    if lag < 1:
        raise ValueError("lag must be >= 1")
    controller = np.asarray(controller)
    values = np.asarray(values)
    if len(controller) != len(values) - 1:
        raise ValueError(
            "controller must have len(values)-1 entries (one action per "
            f"transition); got {len(controller)} vs {len(values)} values"
        )
    y = _digitize(values, state_bins)
    x = _digitize(controller, x_bins)

    start = lag - 1  # index of the first valid transition n = start
    y_n = y[start:-1]
    y_np1 = y[start + 1:]
    x_pred = x[: len(x) - start]

    n = len(x_pred)
    if n < 8:
        return 0.0

    nx = int(x_pred.max()) + 1 if x_pred.size else 1
    ny = int(max(y_n.max(), y_np1.max())) + 1 if y_n.size else 1

    idx = (x_pred.astype(np.int64) * ny + y_n.astype(np.int64)) * ny + y_np1.astype(np.int64)
    counts = np.bincount(idx, minlength=nx * ny * ny).astype(np.float64)
    joint = counts.reshape(nx, ny, ny) / n

    p_xy = joint.sum(axis=2)   # (nx, ny)     p(x, y_n)
    p_yy = joint.sum(axis=0)   # (ny, ny)     p(y_n, y_{n+1})
    p_y = p_xy.sum(axis=0)     # (ny,)        p(y_n)

    with np.errstate(divide="ignore", invalid="ignore"):
        num = joint * p_y[None, :, None]
        den = p_xy[:, :, None] * p_yy[None, :, :]
        ratio = np.where(joint > 0, num / np.where(den > 0, den, 1.0), 1.0)
        terms = np.where(joint > 0, joint * np.log2(np.clip(ratio, 1e-300, None)), 0.0)

    return max(float(terms.sum()), 0.0)


@dataclass
class TEResult:
    te_bits: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    n_surrogates: int
    significant: bool = False  # filled in by classify_trace's BH correction


def te_significance(controller: np.ndarray, values: np.ndarray, *,
                     n_surrogates: int = 200, seed: int = 0,
                     state_bins: int = DEFAULT_STATE_BINS,
                     x_bins: int = DEFAULT_X_BINS, lag: int = 1) -> TEResult:
    """Observed TE (at the given `lag`, see `transfer_entropy`) plus a
    permutation null built by shuffling the controller stream
    `n_surrogates` times (destroys the temporal alignment, keeps the
    marginal symbol distribution — the standard surrogate construction
    for TE significance testing; shuffling wipes out dependence at every
    lag simultaneously, so the null does not itself need to vary with
    `lag`)."""
    rng = np.random.default_rng(seed)
    ctrl = np.asarray(controller)
    obs = transfer_entropy(ctrl, values, state_bins=state_bins, x_bins=x_bins, lag=lag)
    null = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        null[i] = transfer_entropy(rng.permutation(ctrl), values, state_bins=state_bins, x_bins=x_bins, lag=lag)
    mu = float(null.mean())
    sigma = float(null.std())
    z = (obs - mu) / sigma if sigma > 1e-12 else (float("inf") if obs > mu else 0.0)
    p = float((np.sum(null >= obs) + 1) / (n_surrogates + 1))
    return TEResult(te_bits=obs, null_mean=mu, null_std=sigma, z_score=float(z),
                     p_value=p, n_surrogates=n_surrogates)


MAX_LAG = 3  # generic order-k search width: a controller-driven response
# may land more than one recorded step after the order-1 convention
# assumes (score-tally/animation delay inside a probed step, or spilling
# into the next one — see the deferred recorder's frame_skip). Not tuned
# to any specific game; it is a statement about how many recorded steps
# a cause-effect pair may be allowed to straddle before the search gives
# up, nothing about what the effect means.


def best_lag_te(controller: np.ndarray, values: np.ndarray, *,
                 max_lag: int = MAX_LAG,
                 state_bins: int = DEFAULT_STATE_BINS,
                 x_bins: int = DEFAULT_X_BINS) -> Tuple[int, float]:
    """Cheap (no-surrogate) raw TE at every lag in `1..max_lag`; returns
    `(best_lag, best_te_bits)`. A prefilter only, exactly like the
    lag-1-only raw TE it generalizes — never the significance decision,
    just which single lag's permutation test is worth paying for."""
    best_lag, best_te = 1, 0.0
    for lag in range(1, max_lag + 1):
        te = transfer_entropy(controller, values, state_bins=state_bins, x_bins=x_bins, lag=lag)
        if te > best_te:
            best_lag, best_te = lag, te
    return best_lag, best_te


# ---------------------------------------------------------------------
# Delta-coupling: mutual information between the controller and a
# field's own STEP-TO-STEP CHANGE, rather than its raw level. This is a
# second, complementary confirmatory test used only where the timer
# test is deciding whether a monotone/near-monotone field genuinely
# "responds to nobody" -- see classify_trace stage 4.
#
# A near-monotone counter's raw level is a running total: after a few
# hundred steps it has visited most of its range, so a fixed quantile
# binning of the LEVEL (what `transfer_entropy` uses) spends almost all
# of its state alphabet distinguishing "early in the trace" from "late
# in the trace" -- the level is highly collinear with elapsed time
# itself. Per-step controller dependence shows up in whether the field
# changed on a given step, not in which coarse level-quantile it has
# accumulated into; testing MI(controller, delta) directly is the
# standard way to avoid burying that signal under a time-collinear
# level encoding. This is a generic property of monotone counters, not
# anything authored about a specific game.
# ---------------------------------------------------------------------


def delta_mutual_information(controller: np.ndarray, values: np.ndarray,
                              delta_bins: int = DEFAULT_STATE_BINS,
                              x_bins: int = DEFAULT_X_BINS, lag: int = 1) -> float:
    """MI(controller_{n-lag+1}; values[n+1]-values[n]) in bits, over the
    same `n` range and `lag` convention as `transfer_entropy`."""
    if lag < 1:
        raise ValueError("lag must be >= 1")
    controller = np.asarray(controller)
    values = np.asarray(values)
    if len(controller) != len(values) - 1:
        raise ValueError(
            "controller must have len(values)-1 entries (one action per "
            f"transition); got {len(controller)} vs {len(values)} values"
        )
    start = lag - 1
    delta = np.diff(values.astype(np.int64))[start:]
    x = _digitize(controller, x_bins)
    x_pred = x[: len(x) - start]

    n = len(x_pred)
    if n < 8:
        return 0.0

    d = _digitize(delta, delta_bins)
    nx = int(x_pred.max()) + 1 if x_pred.size else 1
    nd = int(d.max()) + 1 if d.size else 1

    idx = x_pred.astype(np.int64) * nd + d.astype(np.int64)
    counts = np.bincount(idx, minlength=nx * nd).astype(np.float64)
    joint = counts.reshape(nx, nd) / n

    px = joint.sum(axis=1, keepdims=True)
    pd = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = px * pd
        ratio = np.where(joint > 0, joint / np.where(denom > 0, denom, 1.0), 1.0)
        terms = np.where(joint > 0, joint * np.log2(np.clip(ratio, 1e-300, None)), 0.0)

    return max(float(terms.sum()), 0.0)


def best_lag_delta_mi(controller: np.ndarray, values: np.ndarray, *,
                       max_lag: int = MAX_LAG,
                       delta_bins: int = DEFAULT_STATE_BINS,
                       x_bins: int = DEFAULT_X_BINS) -> Tuple[int, float]:
    """Cheap (no-surrogate) delta-MI at every lag in `1..max_lag`;
    returns `(best_lag, best_bits)` — same shape and role as
    `best_lag_te`, over the delta-coupling estimator instead."""
    best_lag, best_mi = 1, 0.0
    for lag in range(1, max_lag + 1):
        mi = delta_mutual_information(controller, values, delta_bins=delta_bins, x_bins=x_bins, lag=lag)
        if mi > best_mi:
            best_lag, best_mi = lag, mi
    return best_lag, best_mi


def delta_coupling_significance(controller: np.ndarray, values: np.ndarray, *,
                                 n_surrogates: int = 200, seed: int = 0,
                                 delta_bins: int = DEFAULT_STATE_BINS,
                                 x_bins: int = DEFAULT_X_BINS, lag: int = 1) -> TEResult:
    """Observed delta-MI (at `lag`) plus a permutation null built by
    shuffling the controller stream, exactly like `te_significance` —
    same surrogate construction, applied to `delta_mutual_information`
    instead of `transfer_entropy`. Reuses `TEResult`: its `te_bits`
    field holds delta-MI bits in this context."""
    rng = np.random.default_rng(seed)
    ctrl = np.asarray(controller)
    obs = delta_mutual_information(ctrl, values, delta_bins=delta_bins, x_bins=x_bins, lag=lag)
    null = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        null[i] = delta_mutual_information(rng.permutation(ctrl), values,
                                            delta_bins=delta_bins, x_bins=x_bins, lag=lag)
    mu = float(null.mean())
    sigma = float(null.std())
    z = (obs - mu) / sigma if sigma > 1e-12 else (float("inf") if obs > mu else 0.0)
    p = float((np.sum(null >= obs) + 1) / (n_surrogates + 1))
    return TEResult(te_bits=obs, null_mean=mu, null_std=sigma, z_score=float(z),
                     p_value=p, n_surrogates=n_surrogates)


def _max_lag_significance(estimator, controller: np.ndarray, values: np.ndarray,
                           lags: Sequence[int], n_surrogates: int, seed: int) -> Tuple[int, TEResult]:
    """Shared machinery for `te_significance_over_lags` and
    `delta_significance_over_lags`: test "best of several lags" AS A
    SINGLE STATISTIC, with the SAME max-over-lags reduction applied to
    every surrogate as to the real data.

    This is not optional bookkeeping. Picking whichever lag scores
    highest on the real (unpermuted) data and then running a
    significance test against a null that was only ever computed at
    that one fixed lag re-introduces exactly the bug this module exists
    to avoid: the null never got the same "best of `len(lags)` tries"
    advantage the observed statistic did, so its mean/std understate the
    true null spread and z-scores come out inflated for every field,
    including genuinely independent ones (verified: this under-corrected
    version pushed a genuinely-independent synthetic timer's z-score
    over 1.0 in roughly half of 30 trials — a false-abstain rate the
    naive single-lag null does not warn you about). Maximizing inside
    both the observed statistic AND every one of the `n_surrogates`
    permutation replicates is the standard scan-statistic correction for
    "which of several lags/windows looks best" and is what keeps the
    multi-lag search from being a free lunch."""
    rng = np.random.default_rng(seed)
    ctrl = np.asarray(controller)
    obs_per_lag = {lag: estimator(ctrl, values, lag) for lag in lags}
    best_lag = max(obs_per_lag, key=obs_per_lag.get)
    obs = obs_per_lag[best_lag]
    null = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        perm = rng.permutation(ctrl)
        null[i] = max(estimator(perm, values, lag) for lag in lags)
    mu = float(null.mean())
    sigma = float(null.std())
    z = (obs - mu) / sigma if sigma > 1e-12 else (float("inf") if obs > mu else 0.0)
    p = float((np.sum(null >= obs) + 1) / (n_surrogates + 1))
    return best_lag, TEResult(te_bits=obs, null_mean=mu, null_std=sigma, z_score=float(z),
                               p_value=p, n_surrogates=n_surrogates)


def te_significance_over_lags(controller: np.ndarray, values: np.ndarray, *,
                               lags: Sequence[int] = (1,), n_surrogates: int = 200, seed: int = 0,
                               state_bins: int = DEFAULT_STATE_BINS,
                               x_bins: int = DEFAULT_X_BINS) -> Tuple[int, TEResult]:
    """`te_significance` maximized over a SET of candidate lags at once
    (see `_max_lag_significance` for why the null must be maximized too).
    Returns `(best_lag, TEResult)`. This is the actual confirmatory test
    `classify_trace` runs — `best_lag_te`'s cheap point estimate only
    decides whether it is worth running at all."""
    return _max_lag_significance(
        lambda c, v, lag: transfer_entropy(c, v, state_bins=state_bins, x_bins=x_bins, lag=lag),
        controller, values, lags, n_surrogates, seed,
    )


def delta_significance_over_lags(controller: np.ndarray, values: np.ndarray, *,
                                  lags: Sequence[int] = (1,), n_surrogates: int = 200, seed: int = 0,
                                  delta_bins: int = DEFAULT_STATE_BINS,
                                  x_bins: int = DEFAULT_X_BINS) -> Tuple[int, TEResult]:
    """`delta_coupling_significance` maximized over a SET of candidate
    lags at once — the delta-coupling counterpart of
    `te_significance_over_lags`, same rationale."""
    return _max_lag_significance(
        lambda c, v, lag: delta_mutual_information(c, v, delta_bins=delta_bins, x_bins=x_bins, lag=lag),
        controller, values, lags, n_surrogates, seed,
    )


def bh_correct(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Scanning hundreds or thousands
    of candidate fields at a per-field alpha would produce dozens of
    false positives by chance alone — precision-over-recall means every
    multi-field scan runs through this, never a raw per-field p<=alpha."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[int(np.max(np.where(below)[0]))]
    return p <= cutoff


# ---------------------------------------------------------------------
# Monotone-timer test: strict autoregressive decrement (or increment),
# independent of the controller.
# ---------------------------------------------------------------------


def monotone_timer_score(values: np.ndarray) -> dict:
    v = np.asarray(values).astype(np.int64)
    deltas = np.diff(v)
    nz = deltas[deltas != 0]
    if nz.size == 0:
        return {"is_monotone": False, "direction": 0, "step_consistency": 0.0, "frac_nonzero": 0.0}
    pos = float((nz > 0).mean())
    neg = float((nz < 0).mean())
    direction = 1 if pos >= neg else -1
    agree = max(pos, neg)
    same_dir = nz[nz > 0] if direction == 1 else nz[nz < 0]
    _, counts = np.unique(np.abs(same_dir), return_counts=True)
    step_consistency = float(counts.max()) / float(same_dir.size) if same_dir.size else 0.0
    return {
        "is_monotone": bool(agree >= 0.98),
        "direction": int(direction),
        "step_consistency": step_consistency,
        "frac_nonzero": float(nz.size) / float(deltas.size),
    }


# ---------------------------------------------------------------------
# Symmetry detector: candidate player/opponent field pairs, by STRUCTURE
# (byte width, update-magnitude histogram, update rate, value range) —
# never by address adjacency or any authored per-game layout guess.
# ---------------------------------------------------------------------

MIN_UPDATE_RATE_FOR_PAIRING = 0.02


@dataclass
class FieldStats:
    field: Field
    values: np.ndarray
    delta_hist: dict
    update_rate: float
    value_range: tuple


def compute_field_stats(trace: Trace, field: Field) -> FieldStats:
    values = field.extract(trace.ram)
    deltas = np.diff(values)
    nz = deltas[deltas != 0]
    update_rate = float(nz.size) / max(deltas.size, 1)
    if nz.size:
        mags, counts = np.unique(np.abs(nz), return_counts=True)
        total = float(counts.sum())
        delta_hist = {int(m): float(c) / total for m, c in zip(mags, counts)}
    else:
        delta_hist = {}
    value_range = (int(values.min()), int(values.max()))
    return FieldStats(field=field, values=values, delta_hist=delta_hist,
                       update_rate=update_rate, value_range=value_range)


def _hist_intersection(a: dict, b: dict) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    return float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys))


def symmetry_score(a: FieldStats, b: FieldStats) -> float:
    """1.0 = structurally identical update statistics (mirrored layout);
    0.0 = different byte width, i.e. not even candidate-comparable."""
    if a.field.width_bytes != b.field.width_bytes:
        return 0.0
    hist_sim = _hist_intersection(a.delta_hist, b.delta_hist)
    rsum = a.update_rate + b.update_rate
    rate_sim = 1.0 - abs(a.update_rate - b.update_rate) / rsum if rsum > 1e-9 else 1.0
    ra, rb = a.value_range[1] - a.value_range[0], b.value_range[1] - b.value_range[0]
    rsum2 = ra + rb
    range_sim = 1.0 - abs(ra - rb) / rsum2 if rsum2 > 0 else 1.0
    return float(np.clip((hist_sim + rate_sim + range_sim) / 3.0, 0.0, 1.0))


def find_symmetric_pairs(stats_list: Sequence[FieldStats], min_score: float = 0.6) -> list:
    pairs = []
    n = len(stats_list)
    for i in range(n):
        a = stats_list[i]
        if a.update_rate < MIN_UPDATE_RATE_FOR_PAIRING:
            continue
        for j in range(i + 1, n):
            b = stats_list[j]
            if b.update_rate < MIN_UPDATE_RATE_FOR_PAIRING:
                continue
            if set(a.field.offsets) & set(b.field.offsets):
                continue  # overlapping bytes are not a real "pair"
            score = symmetry_score(a, b)
            if score >= min_score:
                pairs.append((a.field, b.field, score))
    return pairs


def best_partners(stats_list: Sequence[FieldStats], min_score: float = 0.6) -> dict:
    """field -> (best-matching partner field, symmetry score)."""
    best: dict = {}
    for a, b, score in find_symmetric_pairs(stats_list, min_score):
        if a not in best or score > best[a][1]:
            best[a] = (b, score)
        if b not in best or score > best[b][1]:
            best[b] = (a, score)
    return best


# ---------------------------------------------------------------------
# Classification.
# ---------------------------------------------------------------------


class Outcome(str, Enum):
    SCORE_WIN_VS_OPPONENT = "SCORE_WIN_VS_OPPONENT"
    POSITION_TARGET_MET = "POSITION_TARGET_MET"
    TIME_TARGET_BEATEN = "TIME_TARGET_BEATEN"
    THRESHOLD_REACHED = "THRESHOLD_REACHED"
    UNSCORABLE = "UNSCORABLE"


@dataclass
class Verdict:
    field: Field
    outcome: Outcome
    confidence: float
    evidence: dict
    paired_with: Optional[Field] = None


MIN_TRACE_LEN = 32

TIMER_NULL_Z_MAX = 1.0  # A field may only be called "responds to nobody"
# (TIME_TARGET_BEATEN) when its best-lag controller-TE sits within one
# null standard deviation of the permutation-null mean. Failing the
# FDR-corrected shortlist test is not the same claim as "z-score near
# zero" -- BH correction exists precisely to make the significance bar
# stricter than a raw z-test when scanning many fields, so a field can
# be unremarkable-by-BH and still visibly elevated over the null. Only
# the latter (z < TIMER_NULL_Z_MAX) is actual evidence of independence;
# anything more elevated is insufficient power to rule out a weak or
# lagged score coupling, and must ABSTAIN rather than commit.


def _te_result_to_dict(r: Optional[TEResult]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "te_bits": r.te_bits, "null_mean": r.null_mean, "null_std": r.null_std,
        "z_score": r.z_score, "p_value": r.p_value, "n_surrogates": r.n_surrogates,
        "significant": r.significant,
    }


def classify_trace(trace: Trace, fields: Optional[Sequence[Field]] = None, *,
                    alpha: float = 0.01, min_te_bits: float = 0.02,
                    n_surrogates: int = 200, symmetry_threshold: float = 0.6,
                    position_max_states: int = 10, seed: int = 0,
                    state_bins: int = DEFAULT_STATE_BINS,
                    x_bins: int = DEFAULT_X_BINS, max_lag: int = MAX_LAG) -> list:
    """Classify every field into one of the four pre-registered outcome
    types, or UNSCORABLE. Precision over recall throughout:

      1. Cheap raw-TE prefilter (no surrogates), searched over every lag
         in `1..max_lag` (see `best_lag_te`) so a controller response
         that lands more than one recorded step after the order-1
         convention assumes is not invisible to the prefilter — still a
         speed optimization only, never the significance decision.
      2. Confirmatory permutation test at each shortlisted field's best
         lag, BH-corrected across the shortlist — THIS decides "responds
         to the controller".
      3. Structural symmetry pairing (checked BEFORE the timer test, since
         a short-trace one-directional score can otherwise look like a
         monotone counter): controller-significant field with a
         structurally mirrored, NOT-significant partner ->
         SCORE_WIN_VS_OPPONENT. Both sides significant -> ambiguous,
         UNSCORABLE. Paired field itself not significant -> UNSCORABLE
         (it is the opponent side, not a win condition).
      4. Monotone-timer test (strict decrement/increment + no qualifying
         partner + NOT controller-significant): "not controller-
         significant" is confirmed directly rather than inferred from
         "missed the FDR-corrected bar" -- a field that never reached
         the stage-2 shortlist is tested here for the first time, at its
         best lag, so the decision always rests on a computed z-score.
         A second, complementary test checks the field's own
         step-to-step CHANGE against the controller too (see
         `delta_coupling_significance` — a monotone counter's raw level
         is collinear with elapsed time, which can hide real per-step
         dependence under coarse quantile binning of the level alone).
         Only when BOTH signals sit within `TIMER_NULL_Z_MAX` of their
         permutation nulls (genuine no-coupling evidence) does the field
         get TIME_TARGET_BEATEN; either one elevated-but-not-BH-
         significant is insufficient power to rule out a weak or lagged
         score, so it ABSTAINs (UNSCORABLE) instead of guessing.
      5. Controller-significant, no qualifying partner, small bounded
         alphabet with genuinely bidirectional motion -> POSITION_TARGET_MET
         (a rank/place field). Otherwise -> THRESHOLD_REACHED.
      6. Anything left -> UNSCORABLE.
    """
    if fields is None:
        fields = enumerate_fields(trace.ram.shape[1])
    stats = [compute_field_stats(trace, f) for f in fields]
    stats_by_field = {s.field: s for s in stats}

    if len(trace.inputs) < MIN_TRACE_LEN:
        return [
            Verdict(f, Outcome.UNSCORABLE, 0.0,
                    {"reason": f"trace too short ({len(trace.inputs)} steps < {MIN_TRACE_LEN})"})
            for f in fields
        ]

    # Stage 1: cheap raw-TE prefilter, searched over every lag 1..max_lag.
    raw_te = {}
    best_lag = {}
    for s in stats:
        if s.update_rate <= 0.0:
            raw_te[s.field] = 0.0
            best_lag[s.field] = 1
        else:
            lag, te = best_lag_te(trace.inputs, s.values, max_lag=max_lag,
                                   state_bins=state_bins, x_bins=x_bins)
            raw_te[s.field] = te
            best_lag[s.field] = lag
    shortlist = [f for f in fields if raw_te[f] >= min_te_bits]

    # Stage 2: confirmatory surrogate test, maximized over 1..max_lag in
    # BOTH the observed statistic and every surrogate (see
    # `te_significance_over_lags` / `_max_lag_significance` — testing
    # only the lag that happened to score highest on the real data,
    # against a null that was never given the same "best of several
    # tries" advantage, inflates significance under the null; Stage 1's
    # `best_lag[f]` is discarded here for exactly that reason and only
    # used to decide whether this field is worth the expense at all).
    # BH-corrected across the shortlist.
    te_results: dict = {}
    for f in shortlist:
        _, te_results[f] = te_significance_over_lags(
            trace.inputs, stats_by_field[f].values, lags=range(1, max_lag + 1),
            n_surrogates=n_surrogates, seed=seed, state_bins=state_bins, x_bins=x_bins)
    if te_results:
        flist = list(te_results)
        reject = bh_correct([te_results[f].p_value for f in flist], alpha)
        for f, sig in zip(flist, reject):
            te_results[f].significant = bool(sig) and te_results[f].te_bits >= min_te_bits

    def is_sig(f: Field) -> bool:
        r = te_results.get(f)
        return bool(r and r.significant)

    partners = best_partners(stats, symmetry_threshold)

    verdicts = []
    for f in fields:
        s = stats_by_field[f]
        r = te_results.get(f)
        sig = is_sig(f)

        # Structural pairing is checked BEFORE the monotone-timer test: a
        # field that is a candidate opponent-score companion (paired with a
        # controller-significant field) is explained by that pairing even
        # when, taken alone, its own trajectory also happens to look like a
        # monotone counter (a score that has only ever gone up during a
        # short trace is structurally indistinguishable from a timer by
        # monotonicity alone — the pairing evidence is what disambiguates
        # it, so it must win the tie).
        partner_info = partners.get(f)
        if partner_info is not None:
            partner_field, sym_score = partner_info
            partner_sig = is_sig(partner_field)
            if sig and not partner_sig:
                z = r.z_score if r and np.isfinite(r.z_score) else 0.0
                conf = float(np.clip(sym_score * min(1.0, max(z, 0.0) / 5.0), 0.0, 1.0))
                verdicts.append(Verdict(f, Outcome.SCORE_WIN_VS_OPPONENT, conf,
                                         {"paired_field": partner_field.name, "symmetry_score": sym_score,
                                          "te": _te_result_to_dict(r)},
                                         paired_with=partner_field))
                continue
            if sig and partner_sig:
                verdicts.append(Verdict(f, Outcome.UNSCORABLE, 0.0,
                                         {"reason": "both paired fields respond to the controller — ambiguous "
                                                     "ownership, refusing to name a winner",
                                          "paired_field": partner_field.name, "symmetry_score": sym_score}))
                continue
            if not sig:
                verdicts.append(Verdict(f, Outcome.UNSCORABLE, 0.0,
                                         {"reason": "symmetric partner found but this field shows no significant "
                                                     "controller TE itself (looks like the opponent side of a pair)",
                                          "paired_field": partner_field.name, "symmetry_score": sym_score}))
                continue

        timer = monotone_timer_score(s.values)
        if timer["is_monotone"] and timer["step_consistency"] >= 0.8 and timer["frac_nonzero"] >= 0.05 and not sig:
            # "not sig" only means the BH-corrected shortlist test didn't
            # clear the bar (or the field never reached that test at
            # all) -- neither is evidence of independence on its own.
            # Confirm "responds to nobody" directly: reuse the stage-2
            # result if this field already has one, otherwise run the
            # permutation test now (at its best lag) so the timer/abstain
            # split always rests on an actual computed z-score.
            timer_r = r if r is not None else te_significance_over_lags(
                trace.inputs, s.values, lags=range(1, max_lag + 1), n_surrogates=n_surrogates,
                seed=seed, state_bins=state_bins, x_bins=x_bins)[1]
            level_elevated = (not np.isfinite(timer_r.z_score)) or timer_r.z_score >= TIMER_NULL_Z_MAX

            # Second, complementary confirmation: a near-monotone counter's
            # raw LEVEL is collinear with elapsed time itself, which can
            # bury real per-step controller dependence under coarse
            # quantile binning (see delta_coupling_significance docstring).
            # Test the field's own step-to-step CHANGE too, and require
            # both signals to look like genuine independence before
            # committing to TIME_TARGET_BEATEN. The cheap (no-surrogate)
            # best_lag_delta_mi is only the "is it worth testing at all"
            # gate; the actual test is max-over-lags in both directions,
            # same rationale as the level-TE confirmation above.
            _, delta_raw = best_lag_delta_mi(trace.inputs, s.values, max_lag=max_lag,
                                              delta_bins=state_bins, x_bins=x_bins)
            delta_r = (delta_significance_over_lags(trace.inputs, s.values, lags=range(1, max_lag + 1),
                                                      n_surrogates=n_surrogates, seed=seed,
                                                      delta_bins=state_bins, x_bins=x_bins)[1]
                       if delta_raw >= min_te_bits else None)
            delta_elevated = delta_r is not None and (
                (not np.isfinite(delta_r.z_score)) or delta_r.z_score >= TIMER_NULL_Z_MAX
            )

            if level_elevated or delta_elevated:
                reason = ("monotone trajectory, but controller-coupling evidence is elevated without "
                          "clearing the FDR-corrected significance bar (")
                if level_elevated:
                    reason += f"level-TE z={timer_r.z_score:.3g}"
                if level_elevated and delta_elevated:
                    reason += ", "
                if delta_elevated:
                    reason += f"delta-MI z={delta_r.z_score:.3g}"
                reason += (") — not-significant is not evidence of independence, so this is insufficient "
                           "power to call it a pure timer rather than a weak or lagged score coupling")
                verdicts.append(Verdict(f, Outcome.UNSCORABLE, 0.0,
                                         {"reason": reason, "timer": timer,
                                          "te": _te_result_to_dict(timer_r),
                                          "delta_te": _te_result_to_dict(delta_r)}))
                continue
            conf = float(np.clip(timer["step_consistency"], 0.0, 1.0))
            verdicts.append(Verdict(f, Outcome.TIME_TARGET_BEATEN, conf,
                                     {"timer": timer, "te": _te_result_to_dict(timer_r)}))
            continue

        if sig:
            n_states = int(len(np.unique(s.values)))
            d = np.diff(s.values.astype(np.int64))
            pos_share = float((d > 0).mean()) if d.size else 0.0
            neg_share = float((d < 0).mean()) if d.size else 0.0
            bidirectional = min(pos_share, neg_share) >= 0.15
            conf = float(np.clip(1.0 - r.p_value, 0.0, 1.0))
            if n_states <= position_max_states and bidirectional:
                verdicts.append(Verdict(f, Outcome.POSITION_TARGET_MET, conf,
                                         {"n_states": n_states, "pos_share": pos_share, "neg_share": neg_share,
                                          "te": _te_result_to_dict(r)}))
            else:
                verdicts.append(Verdict(f, Outcome.THRESHOLD_REACHED, conf,
                                         {"n_states": n_states, "te": _te_result_to_dict(r)}))
            continue

        verdicts.append(Verdict(f, Outcome.UNSCORABLE, 0.0,
                                 {"reason": "no significant controller TE, not a monotone timer, no qualifying "
                                             "symmetric partner"}))

    return verdicts


def verdict_to_dict(v: Verdict) -> dict:
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(val) for k, val in o.items()}
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    return {
        "field": v.field.name,
        "offsets": list(v.field.offsets),
        "endianness": v.field.endianness,
        "outcome": v.outcome.value,
        "confidence": float(v.confidence),
        "evidence": _clean(v.evidence),
        "paired_with": v.paired_with.name if v.paired_with else None,
    }


# ---------------------------------------------------------------------
# Synthetic ground-truth traces — used by both `--selftest` and
# tests/test_outcome_probe.py, so the ground truth is defined exactly
# once. Every quantity here is a generic statistical construction (a
# byte that increments on a button, a byte that decrements every frame,
# uniform noise) — nothing about any real game.
# ---------------------------------------------------------------------


def synth_driven_field_trace(t: int = 500, seed: int = 0, n_bytes: int = 8,
                              offset: int = 2, drive_bit: int = 0x01,
                              drive_p: float = 0.35) -> Trace:
    """Byte at `offset` increments by exactly 1 whenever `drive_bit` is
    set in the controller input on the preceding frame, else holds.
    Ground truth: HIGH transfer entropy from the controller; no paired
    field exists in this trace."""
    rng = np.random.default_rng(seed)
    inputs = np.where(rng.random(t - 1) < drive_p, drive_bit, 0x00).astype(np.uint8)
    ram = np.zeros((t, n_bytes), dtype=np.uint8)
    ram[:, 5] = 77  # constant junk byte, present to exercise the "no update" path
    score = 0
    for i in range(t - 1):
        if inputs[i] & drive_bit:
            score += 1
        ram[i + 1, offset] = score & 0xFF
    return Trace(ram=ram, inputs=inputs, meta={"synthetic": "driven_field", "offset": offset})


def synth_mirrored_pair_trace(t: int = 500, seed: int = 1, n_bytes: int = 8,
                               player_off: int = 2, opp_off: int = 3,
                               drive_bit: int = 0x01, drive_p: float = 0.35,
                               opp_p: float = 0.35) -> Trace:
    """Two same-width byte fields with matched update-rate and delta-
    magnitude statistics (mirrored layout): `player_off` increments
    exactly when `drive_bit` is held (controller-driven), `opp_off`
    increments independently at a matched marginal rate (NOT
    controller-driven). Ground truth: player -> SCORE_WIN_VS_OPPONENT
    paired with opp; opp -> UNSCORABLE (not the winner side)."""
    rng = np.random.default_rng(seed)
    inputs = np.where(rng.random(t - 1) < drive_p, drive_bit, 0x00).astype(np.uint8)
    ram = np.zeros((t, n_bytes), dtype=np.uint8)
    player = opp = 0
    for i in range(t - 1):
        if inputs[i] & drive_bit:
            player += 1
        if rng.random() < opp_p:
            opp += 1
        ram[i + 1, player_off] = player & 0xFF
        ram[i + 1, opp_off] = opp & 0xFF
    return Trace(ram=ram, inputs=inputs,
                 meta={"synthetic": "mirrored_pair", "player_offset": player_off, "opponent_offset": opp_off})


def synth_timer_trace(t: int = 500, seed: int = 2, n_bytes: int = 8,
                       offset: int = 4, start: int = 600, step: int = 1) -> Trace:
    """Byte at `offset` decrements by a constant `step` every single
    frame, independent of a fully random controller stream. Ground
    truth: strict monotone decrement, near-zero controller TE ->
    TIME_TARGET_BEATEN."""
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, 256, size=t - 1).astype(np.uint8)
    ram = np.zeros((t, n_bytes), dtype=np.uint8)
    val = start
    ram[0, offset] = val & 0xFF
    for i in range(t - 1):
        val = max(0, val - step)
        ram[i + 1, offset] = val & 0xFF
    return Trace(ram=ram, inputs=inputs, meta={"synthetic": "timer", "offset": offset})


def synth_noise_trace(t: int = 500, seed: int = 3, n_bytes: int = 8, offset: int = 6) -> Trace:
    """Every byte, including `offset`, is IID uniform noise, unrelated to
    the (also random) controller stream. Ground truth: UNSCORABLE."""
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, 256, size=t - 1).astype(np.uint8)
    ram = rng.integers(0, 256, size=(t, n_bytes)).astype(np.uint8)
    return Trace(ram=ram, inputs=inputs, meta={"synthetic": "noise", "offset": offset})


def synth_ambiguous_pair_trace(t: int = 500, seed: int = 4, n_bytes: int = 8,
                                off_a: int = 2, off_b: int = 3,
                                drive_bit: int = 0x01, drive_p: float = 0.35) -> Trace:
    """Deliberately ambiguous: TWO structurally identical fields both
    increment together on the SAME controller bit (e.g. a capture where
    our one controller stream cannot disambiguate which of two
    identically-responsive fields is 'ours'). Ground truth: both fields
    are controller-significant AND structurally symmetric, so the
    classifier must ABSTAIN on both rather than pick a winner."""
    rng = np.random.default_rng(seed)
    inputs = np.where(rng.random(t - 1) < drive_p, drive_bit, 0x00).astype(np.uint8)
    ram = np.zeros((t, n_bytes), dtype=np.uint8)
    a = b = 0
    for i in range(t - 1):
        if inputs[i] & drive_bit:
            a += 1
            b += 1
        ram[i + 1, off_a] = a & 0xFF
        ram[i + 1, off_b] = b & 0xFF
    return Trace(ram=ram, inputs=inputs, meta={"synthetic": "ambiguous_pair", "a": off_a, "b": off_b})


def _selftest() -> int:
    results = []

    v1 = classify_trace(synth_driven_field_trace(), [Field("score", (2,))], n_surrogates=150, seed=0)[0]
    results.append(("driven field, no partner -> THRESHOLD_REACHED",
                     v1.outcome == Outcome.THRESHOLD_REACHED, v1))

    v2p, v2o = classify_trace(synth_mirrored_pair_trace(),
                               [Field("player_score", (2,)), Field("opp_score", (3,))],
                               n_surrogates=150, seed=1)
    results.append(("mirrored player -> SCORE_WIN_VS_OPPONENT", v2p.outcome == Outcome.SCORE_WIN_VS_OPPONENT, v2p))
    results.append(("mirrored opponent -> UNSCORABLE (not the winner side)", v2o.outcome == Outcome.UNSCORABLE, v2o))

    v3 = classify_trace(synth_timer_trace(), [Field("timer", (4,))], n_surrogates=150, seed=2)[0]
    results.append(("monotone timer -> TIME_TARGET_BEATEN", v3.outcome == Outcome.TIME_TARGET_BEATEN, v3))

    v4 = classify_trace(synth_noise_trace(), [Field("noise", (6,))], n_surrogates=150, seed=3)[0]
    results.append(("noise field -> UNSCORABLE (abstain)", v4.outcome == Outcome.UNSCORABLE, v4))

    v5a, v5b = classify_trace(synth_ambiguous_pair_trace(), [Field("a", (2,)), Field("b", (3,))],
                               n_surrogates=150, seed=4)
    results.append(("ambiguous identical pair -> UNSCORABLE (abstain, both respond)",
                     v5a.outcome == Outcome.UNSCORABLE and v5b.outcome == Outcome.UNSCORABLE, (v5a, v5b)))

    ok = True
    for name, passed, detail in results:
        print(("PASS" if passed else "FAIL") + f" — {name}")
        if not passed:
            ok = False
            print(f"    got: {detail}")
    return 0 if ok else 1


# ---------------------------------------------------------------------
# DEFERRED: live recording. Never called by this module's import, its
# CLI classify/selftest paths, or any test — the 2-1 campaign owns the
# emulator and this lane is token-bound (code/config/tests only).
# ---------------------------------------------------------------------


def record_trace_live(rom: str, state, out_path, *, n_steps: int = 1800,
                       frame_skip: int = 4, forward: str = "right",
                       seed: int = 1, allow: bool = False) -> None:
    """DEFERRED capability — drives the real emulator via nes_core.Pool.
    Not exercised by this module's import, its `classify`/`selftest` CLI
    paths, or tests/test_outcome_probe.py; this lane records nothing
    live while a campaign owns the emulator.

    EXACT DEFERRED COMMAND — run once the emulator frees, to record the
    decisive v17 racing-title trace (R.C. Pro-Am, per the research
    round's suggested first target; Excitebike / Mach Rider are the
    documented fallbacks):

        .venv/bin/python scripts/outcome_probe.py record \\
            --rom "roms/R.C. Pro-Am (USA).nes" \\
            --state "roms/R.C. Pro-Am (USA)_start.state.bin" \\
            --out runs/outcome_probe/rcproam_trace.npz \\
            --steps 1800 --frame-skip 4 --forward right \\
            --i-know-this-drives-the-emulator

    then classify the result with:

        .venv/bin/python scripts/outcome_probe.py classify \\
            --trace runs/outcome_probe/rcproam_trace.npz

    `allow=True` (the CLI's `--i-know-this-drives-the-emulator`) is
    required before a single Pool step runs, so a stray import or a
    fuzzed CLI invocation cannot touch the emulator through this path.
    """
    if not allow:
        raise RuntimeError(
            "record_trace_live() refuses to run without allow=True — see this "
            "function's docstring for the deferred command. This lane does not "
            "step the emulator (the 2-1 campaign owns it)."
        )
    from nes_core import Pool  # deferred import — see module docstring.

    if isinstance(state, (str, Path)):
        state = Path(state).read_bytes()

    pool = Pool(rom_path=rom, num_workers=1, frame_skip=int(frame_skip))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, state)

    dir_bits = {"right": 0x80, "left": 0x40, "up": 0x10, "down": 0x20}
    fwd = dir_bits[forward.lower()]

    ram_log = np.empty((n_steps + 1, RAM_SIZE), dtype=np.uint8)
    input_log = np.empty((n_steps,), dtype=np.uint8)
    r0 = pool.step_all(np.array([0x00], dtype=np.uint8))
    ram_log[0] = np.frombuffer(bytes(r0[0][2]), dtype=np.uint8)[:RAM_SIZE]
    rng = np.random.default_rng(seed)
    for i in range(n_steps):
        mask = fwd if rng.random() < 0.8 else 0x00
        input_log[i] = mask
        r = pool.step_all(np.array([mask], dtype=np.uint8))
        ram_log[i + 1] = np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]

    save_trace_npz(out_path, ram_log, input_log,
                    meta={"rom": str(rom), "frame_skip": frame_skip, "forward": forward, "seed": seed})


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="outcome_probe.py",
                                  description="Comparison-outcome detector over recorded RAM traces.")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("classify", help="classify RAM fields in a recorded trace")
    c.add_argument("--trace", required=True)
    c.add_argument("--fields", help="JSON file of explicit Field descriptors (see load_fields)")
    c.add_argument("--alpha", type=float, default=0.01)
    c.add_argument("--min-te-bits", type=float, default=0.02)
    c.add_argument("--surrogates", type=int, default=200)
    c.add_argument("--symmetry-threshold", type=float, default=0.6)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", help="write the JSON report here")
    c.add_argument("--only-scorable", action="store_true", help="omit UNSCORABLE fields from the report")

    sub.add_parser("selftest", help="run the 5 synthetic ground-truth cases and print PASS/FAIL")

    r = sub.add_parser("record", help="DEFERRED — drives the real emulator; refuses to run without the flag below")
    r.add_argument("--rom", required=True)
    r.add_argument("--state", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=1800)
    r.add_argument("--frame-skip", type=int, default=4)
    r.add_argument("--forward", default="right")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--i-know-this-drives-the-emulator", action="store_true", dest="allow")

    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    if args.cmd is None:
        ap.print_help()
        return 2

    if args.cmd == "selftest":
        return _selftest()

    if args.cmd == "record":
        record_trace_live(args.rom, args.state, args.out, n_steps=args.steps,
                           frame_skip=args.frame_skip, forward=args.forward,
                           seed=args.seed, allow=args.allow)
        return 0

    if args.cmd == "classify":
        trace = load_trace(args.trace)
        fields = load_fields(args.fields) if args.fields else None
        verdicts = classify_trace(trace, fields, alpha=args.alpha, min_te_bits=args.min_te_bits,
                                   n_surrogates=args.surrogates, symmetry_threshold=args.symmetry_threshold,
                                   seed=args.seed)
        rows = [verdict_to_dict(v) for v in verdicts]
        if args.only_scorable:
            rows = [row for row in rows if row["outcome"] != Outcome.UNSCORABLE.value]
        report = json.dumps(rows, indent=2)
        if args.out:
            Path(args.out).write_text(report)
        print(report)
        return 0

    ap.error("no command given")
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
