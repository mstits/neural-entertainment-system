#!/usr/bin/env python3
"""Offline instrument sweep over banked training iterates — NO training.

WHY THIS EXISTS. `configs/mario_1_1_backward_v4.yaml` closes with a
registered action: *"Score the late `vanilla_ppo_iter_*.pt` checkpoints too —
the binary accept gate can only re-snapshot once, so the winner file is not
guaranteed to be the best policy the run produced."* One of them ever got
scored (iter 260 -> greedy 0.00, in
`checkpoints/mario_1_1_backward_v4/eval.jsonl`). The other eighteen, plus
twenty-four B4 iterates
(`checkpoints/mario_1_1_backward/prevrun_20260808_124336/`, iters 10-240),
are still sitting on disk unmeasured. This script is that sweep.

SIX INSTRUMENTS, five of them free (no emulator, no ROM, no training):

  1. greedy/sampled paired clear-rate curve, under ONE named harness.
     THIS ONE COSTS EMULATION, so it is OPT-IN (`--eval`). By default the
     script STUB-COSTS it: it reports the episode count and the projected
     wall clock instead of running it, so the free instruments can be read
     in seconds and the paid one is a deliberate decision.
  2. top-two logit margin `M(s)` at supplied states — "is the argmax a tie?"
     measured, rather than asserted from an unsourced threshold.
  3. dormant-unit fraction (Sokar et al., ICML 2023) — the plasticity story,
     testable against the collapse arc rather than assumed.
  4. effective rank of the representation (or of a weight matrix) — the other
     half of the plasticity read.
  5. ``||theta_t - theta_0||`` parameter drift — and, as a by-product, the
     honest discriminator for the "the instrument was the bug" hypothesis:
     if a scored network is parameter-identical to another, no eval result
     can differ between them, and drift = 0 says the weights never moved.
  6. ``V_adv = E_s[Var_a(A-hat)]`` — the discriminator `v15_d1` ADOPTED and
     nobody built, which is why a never-retracted "REAL CAPABILITY WALL"
     (`docs/research/B5_PREREG_2026-08-08.md:414`) has been standing on
     evidence that both of its rival hypotheses predict. Free: it reads the
     CRITIC over banked offline transitions, so no emulator is involved.
     Registration: `docs/proposals/VADV_PREREG_2026-08-27.md`.

Everything that can be a pure function of arrays is one, so the arithmetic is
unit-testable with duck-typed stubs and no checkpoints. The torch- and
emulator-touching parts are thin shells around those functions.

USAGE

    # free instruments over both banked runs, JSON to stdout
    .venv/bin/python scripts/score_banked_iterates.py \\
        --iterates 'checkpoints/mario_1_1_backward/prevrun_20260808_124336/vanilla_ppo_iter_*.pt' \\
        --iterates 'checkpoints/mario_1_1_backward_v4/vanilla_ppo_iter_*.pt' \\
        --states runs/instruments/stall_states.npy \\
        --out runs/instruments/b4_v4_sweep.json

    # ...and the paid one (subprocesses scripts/eval_game.py per iterate)
    ... --eval --profile configs/mario_1_1_backward_v4.yaml \\
        --start-state runs/live_show/smb_4_4_micro/entrance_start.state \\
        --episodes 30 --sticky-prob 0.25 --start-jitter 16 --eval-seed 7
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ===========================================================================
# PURE INSTRUMENTS — numpy in, numbers out. No torch, no I/O, no globals.
# ===========================================================================

def top_two_margin(logits: Any) -> np.ndarray:
    """Per-state gap between the best and second-best action logit.

    `logits` is `(N, A)` (a single `(A,)` row is accepted and treated as
    `N=1`). Returns an `(N,)` float array of ``max - second_max``, which is
    exactly the quantity an "argmax tie" claim is about: a near-zero margin
    means the greedy action is decided by numerical noise, while the sampled
    policy still visits both branches.

    A single-action space has no second-best; the margin is then +inf, since
    an argmax over one action cannot tie.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"logits must be (N, A) or (A,), got {arr.shape}")
    if arr.shape[1] < 2:
        return np.full(arr.shape[0], np.inf)
    part = np.partition(arr, -2, axis=1)
    return part[:, -1] - part[:, -2]


def margin_summary(logits: Any, *, tie_threshold: float = 0.15) -> dict:
    """Distributional read of :func:`top_two_margin` (pure).

    `tie_threshold` is REPORTED, not believed: the fraction of states under
    it is emitted alongside the quantiles so the threshold can be calibrated
    against a healthy iterate instead of asserted a priori.
    """
    m = top_two_margin(logits)
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return {"n_states": int(m.size), "mean": None, "median": None,
                "p05": None, "p25": None, "min": None,
                "tie_threshold": float(tie_threshold), "tie_fraction": None}
    return {
        "n_states": int(m.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "min": float(np.min(finite)),
        "tie_threshold": float(tie_threshold),
        "tie_fraction": float(np.mean(finite < float(tie_threshold))),
    }


def dormant_fraction(activations: Any, *, tau: float = 0.0) -> float:
    """Fraction of units that are DORMANT under the Sokar et al. score.

    (Sokar, Agarwal, Castro & Evci, "The Dormant Neuron Phenomenon in Deep
    Reinforcement Learning", ICML 2023.) For a layer with activations `(N,
    U)`, unit i's score is its mean absolute activation divided by the layer's
    mean over all units:

        s_i = E_x|h_i(x)| / ( (1/U) * sum_j E_x|h_j(x)| )

    and unit i is tau-dormant when ``s_i <= tau``. `tau = 0.0` is the strict
    definition (the unit is exactly dead). A layer whose activations are all
    zero has an undefined normalizer; every unit is then dormant by
    definition, which is what this returns (1.0) rather than a NaN.
    """
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] == 0:
        raise ValueError(f"activations must be (N, U), got {a.shape}")
    per_unit = np.mean(np.abs(a), axis=0)
    denom = float(np.mean(per_unit))
    if denom <= 0.0:
        return 1.0
    scores = per_unit / denom
    return float(np.mean(scores <= float(tau)))


def effective_rank(matrix: Any, *, delta: float = 0.01) -> dict:
    """Two effective-rank readings of a matrix (pure).

    * ``srank`` — the smallest k whose top-k singular values carry at least
      ``1 - delta`` of the total singular mass (Kumar et al.'s implicit
      under-parameterisation measure). Integer, easy to read across a run.
    * ``erank`` — ``exp(H(p))`` where ``p`` is the singular spectrum
      normalized to a distribution (Roy & Vetterli). Continuous, so it moves
      before ``srank`` snaps.

    An all-zero matrix has no spectrum: both are 0.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {m.shape}")
    sv = np.linalg.svd(m, compute_uv=False)
    total = float(np.sum(sv))
    if total <= 0.0:
        return {"srank": 0, "erank": 0.0, "n_singular_values": int(sv.size)}
    cumulative = np.cumsum(sv) / total
    srank = int(np.searchsorted(cumulative, 1.0 - float(delta)) + 1)
    p = sv / total
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "srank": min(srank, int(sv.size)),
        "erank": float(np.exp(entropy)),
        "n_singular_values": int(sv.size),
    }


def param_drift(theta_t: dict, theta_0: dict) -> dict:
    """``||theta_t - theta_0||`` over the shared tensors (pure-ish).

    Duck-typed: values may be numpy arrays or anything with a
    ``detach().cpu().numpy()`` chain (i.e. torch tensors). Returns the global
    L2 distance, the relative drift against ``||theta_0||``, the count of
    compared tensors, and the single largest per-tensor contribution — which
    is what tells you whether a run moved its trunk or only its heads.

    A tensor present in both state_dicts but with a different shape (e.g. a
    resized hidden layer) cannot be compared and is excluded from the sum —
    but it is named in ``skipped_tensors`` rather than silently vanishing, so
    a drift computed over a shrunken trunk cannot be mistaken for one over
    the whole network.

    ``identical`` is the discriminator for the "was the scored network the
    trained network?" question: a drift of exactly 0 means the two
    state_dicts are the same weights, and no eval difference between them can
    be real. It is never true while a tensor was skipped for a shape
    mismatch — networks that disagree on shape are not identical, whatever
    the comparable tensors say.
    """
    total_sq = 0.0
    base_sq = 0.0
    per_tensor: dict = {}
    skipped: list = []
    shared = [k for k in theta_t if k in theta_0]
    for k in shared:
        a = _to_numpy(theta_t[k]).astype(np.float64).ravel()
        b = _to_numpy(theta_0[k]).astype(np.float64).ravel()
        if a.shape != b.shape:
            skipped.append(k)
            continue
        d = float(np.sum((a - b) ** 2))
        per_tensor[k] = float(np.sqrt(d))
        total_sq += d
        base_sq += float(np.sum(b ** 2))
    l2 = float(np.sqrt(total_sq))
    base = float(np.sqrt(base_sq))
    largest = max(per_tensor.items(), key=lambda kv: kv[1]) if per_tensor else None
    return {
        "l2": l2,
        "relative": (l2 / base) if base > 0 else None,
        "n_tensors": len(per_tensor),
        "identical": bool(l2 == 0.0 and per_tensor and not skipped),
        "largest_tensor": largest[0] if largest else None,
        "largest_tensor_l2": largest[1] if largest else None,
        "skipped_tensors": sorted(skipped),
        "n_skipped": len(skipped),
    }


# ---------------------------------------------------------------------------
# INSTRUMENT 6 — V_adv, the discriminator `v15_d1` adopted and nobody built.
#
# `docs/research/B5_PREREG_2026-08-08.md:414` still says "THIS IS A REAL
# CAPABILITY WALL at gx ~2674-2872", on evidence (0/717 entrance successes)
# that BOTH live hypotheses predict:
#
#   CAPABILITY        — the policy cannot execute the traversal.
#   MIS-SPECIFICATION — the reward gives no gradient there, so nothing could
#                       have learned it.
#
# `V_adv = E_s[Var_a(A-hat)]` separates them: where the reward carries no
# signal the critic cannot tell the actions apart and V_adv collapses; where
# the agent is merely incapable, the actions still differ in value.
#
# A-HAT COMES FROM THE CRITIC, over REAL banked successors:
#
#     A-hat(s, a) = gamma * V(s') * (1 - done) - V(s)
#
# and NOT from the actor's logits. Centred logits would be a monotone
# re-reading of `top_two_margin` above — and B6 measured margins RISING 14x
# through the entropy collapse, so an actor-sourced V_adv would report LIVE at
# the wall for the wrong reason (confident-wrong noop sharpening). That is the
# fifth vacuous instrument this exists to avoid.
#
# The full registration — bands, controls, thresholds, admissibility — is
# `docs/proposals/VADV_PREREG_2026-08-27.md`, written before any number here
# was computed from a checkpoint.
# ---------------------------------------------------------------------------

# v2 tile observation (`SMBTileObservationV2`): 178 features per frame, four
# frames stacked = 712. The encoder's own progress scalars live at 175/176 of
# each frame; nothing new is read off RAM here.
SMB_V2_FRAME_DIM = 178
SMB_V2_GX_PAGE = 175
SMB_V2_GX_FINE = 176
SMB_V2_PHASE = 177


def decode_gx(obs: Any, *, frame_dim: int = SMB_V2_FRAME_DIM,
              frame: int = -1) -> np.ndarray:
    """Absolute world x, decoded from the observation's OWN progress scalars.

    `SMBTileObservationV2` appends ``out[175] = global_x >> 8`` and
    ``out[176] = (global_x & 0xFF) >> 1`` to every frame
    (`src/emulation/tile_observations/smb.py`), so gx is recoverable from a
    banked observation with no RAM map and no emulator — this reads the
    encoder's existing contract, it does not add an address.

    `obs` is `(N, frames*frame_dim)`; `frame` selects which stacked frame to
    read (-1 = the current one). Returns `(N,)` int, quantised to 2 pixels by
    the encoder's own ``>> 1``.
    """
    arr = np.asarray(obs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"obs must be (N, D) or (D,), got {arr.shape}")
    width = arr.shape[1]
    if frame_dim <= 0 or width % frame_dim != 0:
        raise ValueError(
            f"observation width {width} is not a multiple of frame_dim "
            f"{frame_dim}; this is not a stacked v2 tile observation")
    n_frames = width // frame_dim
    idx = (frame % n_frames) * frame_dim
    page = arr[:, idx + SMB_V2_GX_PAGE].astype(np.int64)
    fine = arr[:, idx + SMB_V2_GX_FINE].astype(np.int64)
    return page * 256 + fine * 2


def state_cell_keys(obs: Any, *, frame_dim: int = SMB_V2_FRAME_DIM,
                    frame: int = -1, exact: bool = False) -> np.ndarray:
    """Integer cell ids grouping observations into "the same state".

    Default (registered) key: the CURRENT stacked frame minus the animation-
    phase byte — tiles, velocities, on_ground, powerup, lives, sub-x and the
    two gx scalars. History frames are dropped from the KEY only; A-hat is
    always computed from the full observation the network consumes.

    `exact=True` keys on the whole observation instead (the registered
    sensitivity check). Exact keys are stricter, so they yield fewer and
    purer cells; the loose key trades purity for coverage and pays for it in
    ``SS_within``, which is why `eta2` and not `raw` is the primary reading.
    """
    arr = np.asarray(obs)
    if arr.ndim != 2:
        raise ValueError(f"obs must be (N, D), got {arr.shape}")
    if exact:
        sub = arr
    else:
        width = arr.shape[1]
        if frame_dim <= 0 or width % frame_dim != 0:
            raise ValueError(
                f"observation width {width} is not a multiple of frame_dim "
                f"{frame_dim}; pass exact=True to key on the raw vector")
        n_frames = width // frame_dim
        idx = (frame % n_frames) * frame_dim
        sub = arr[:, idx:idx + SMB_V2_PHASE]
    contiguous = np.ascontiguousarray(sub)
    view = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1])
    _, cells = np.unique(view.ravel(), return_inverse=True)
    return np.asarray(cells, dtype=np.int64)


def critic_advantage(v_state: Any, v_next: Any, *, done: Any = None,
                     gamma: float = 0.99) -> np.ndarray:
    """``A-hat = gamma * V(s') * (1 - done) - V(s)`` (pure).

    `done` rows are absorbing: their successor bootstrap is zeroed rather
    than carried, matching how the offline banks record a death
    (`scripts/smodice_data.py`).
    """
    vs = np.asarray(v_state, dtype=np.float64).ravel()
    vn = np.asarray(v_next, dtype=np.float64).ravel()
    if vs.shape != vn.shape:
        raise ValueError(f"v_state {vs.shape} != v_next {vn.shape}")
    d = (np.zeros_like(vs) if done is None
         else np.asarray(done, dtype=np.float64).ravel())
    if d.shape != vs.shape:
        raise ValueError(f"done {d.shape} != v_state {vs.shape}")
    return float(gamma) * vn * (1.0 - d) - vs


def qualifying_rows(cells: Any, actions: Any, *, min_rows_per_action: int = 2,
                    min_actions: int = 2, min_rows: int = 6) -> np.ndarray:
    """Boolean mask of rows in cells that can carry an action contrast.

    A cell qualifies when at least `min_actions` distinct actions each have
    at least `min_rows_per_action` rows and those rows total `min_rows`.
    Rows whose action is under the per-action minimum are dropped from the
    cell — a singleton action contributes no within-cell variance, so it
    would inflate the between-action term with pure noise.
    """
    c = np.asarray(cells, dtype=np.int64).ravel()
    a = np.asarray(actions, dtype=np.int64).ravel()
    if c.shape != a.shape:
        raise ValueError(f"cells {c.shape} != actions {a.shape}")
    keep = np.zeros(c.size, dtype=bool)
    order = np.argsort(c, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(c[order]) != 0])
    for lo, hi in zip(starts, np.r_[starts[1:], c.size] if starts.size else []):
        idx = order[lo:hi]
        acts, counts = np.unique(a[idx], return_counts=True)
        good = set(acts[counts >= int(min_rows_per_action)].tolist())
        if len(good) < int(min_actions):
            continue
        sel = idx[np.isin(a[idx], list(good))]
        if sel.size >= int(min_rows):
            keep[sel] = True
    return keep


def advantage_variance(advantages: Any, cells: Any, actions: Any) -> dict:
    """**V_adv** — ``E_s[Var_a(A-hat)]`` and its pooled between-action share.

    `raw` is the literal registered quantity: the mean over state cells of the
    population variance, across actions, of the per-action MEAN advantage.
    It carries critic units and is NEVER comparable across checkpoints —
    ``V -> cV + b`` scales it by ``c**2``, which is exactly how an
    unnormalised V_adv becomes a fifth vacuous instrument.

    `eta2` is the primary reading: the pooled between-action share of
    advantage variance with the state cell as a blocking factor,

        eta2 = SS_between / (SS_between + SS_within)

    a ratio of two variances of the SAME critic, so it is exactly invariant
    under ``V -> cV + b``.

    `eta2_null_analytic` is ``df_between / (df_between + df_within)`` — the
    value `eta2` takes IN EXPECTATION UNDER NO ACTION EFFECT. It is emitted on
    every reading because it is not zero and it is not small: a 2-action cell
    with 3 rows nulls at 0.5. An absolute threshold on `eta2` would have sat
    outside its own acting range, which is the defect that left ReDo's tau at
    0.025 against a >=0.25 firing threshold, never firing in v27 or v28.
    """
    adv = np.asarray(advantages, dtype=np.float64).ravel()
    c = np.asarray(cells, dtype=np.int64).ravel()
    a = np.asarray(actions, dtype=np.int64).ravel()
    if not (adv.shape == c.shape == a.shape):
        raise ValueError(
            f"advantages {adv.shape}, cells {c.shape}, actions {a.shape} "
            "must be row-aligned")
    empty = {
        "n_rows": int(adv.size), "n_cells": 0, "n_action_cells": 0,
        "raw": None, "eta2": None, "eta2_null_analytic": None,
        "ss_between": 0.0, "ss_within": 0.0,
        "df_between": 0, "df_within": 0, "per_cell_var": [],
    }
    if adv.size == 0:
        return empty
    ss_b = 0.0
    ss_w = 0.0
    df_b = 0
    df_w = 0
    per_cell: list = []
    n_cells = 0
    n_action_cells = 0
    order = np.argsort(c, kind="stable")
    cs = c[order]
    starts = np.flatnonzero(np.r_[True, np.diff(cs) != 0])
    for lo, hi in zip(starts, np.r_[starts[1:], cs.size]):
        idx = order[lo:hi]
        acts = np.unique(a[idx])
        if acts.size < 2:
            continue
        n_cells += 1
        cell_mean = float(np.mean(adv[idx]))
        means = []
        for act in acts:
            rows = adv[idx[a[idx] == act]]
            means.append(float(np.mean(rows)))
            ss_b += rows.size * (means[-1] - cell_mean) ** 2
            ss_w += float(np.sum((rows - means[-1]) ** 2))
            df_w += rows.size - 1
            n_action_cells += 1
        df_b += acts.size - 1
        per_cell.append(float(np.var(np.asarray(means))))
    if not per_cell:
        return empty
    # A batch with NO variance at all (ss_b == ss_w == 0) has eta2 = 0/0.
    # It is reported as 0.0, not None: ss_b is exactly 0, so there is
    # provably no between-action variance — the clearest COLLAPSED there is.
    # (`normalise_vadv` still refuses the raw ratio when the CRITIC is flat;
    # that is a different failure and it reads VOID, not COLLAPSED.)
    #
    # The tolerance is load-bearing, not cosmetic. Summing squared deviations
    # of a constant leaves float dust of order 1e-31, and the RATIO of two
    # dust terms is an arbitrary number in [0, 1] — a constant-advantage batch
    # measured eta2 = 0.889 (a confident LIVE) before this guard existed.
    # Anything below the accumulated rounding floor of the data's own scale is
    # not a signal.
    total = ss_b + ss_w
    dust = 1e-12 * max(1.0, float(np.sum(adv ** 2)))
    if total <= dust:
        ss_b, ss_w, total = 0.0, 0.0, 0.0
    return {
        "n_rows": int(adv.size),
        "n_cells": n_cells,
        "n_action_cells": n_action_cells,
        "raw": float(np.mean(per_cell)),
        "eta2": (float(ss_b / total) if total > 0 else 0.0),
        "eta2_null_analytic": (float(df_b / (df_b + df_w))
                               if (df_b + df_w) > 0 else None),
        "ss_between": float(ss_b),
        "ss_within": float(ss_w),
        "df_between": int(df_b),
        "df_within": int(df_w),
        "per_cell_var": [float(v) for v in per_cell],
    }


def normalise_vadv(raw: Optional[float], value_variance: float) -> dict:
    """`raw / Var_batch[V]` — the cross-checkpoint-comparable form.

    A constant critic has ``Var_batch[V] == 0`` and the ratio is 0/0. That
    returns ``None`` with ``degenerate_critic: True``, NOT 0.0: a dead critic
    must read VOID, never COLLAPSED. Reporting 0.0 there would manufacture the
    mis-specification signature out of a broken checkpoint.
    """
    var = float(value_variance)
    if not np.isfinite(var) or var <= 0.0:
        return {"raw_norm": None, "value_variance": var,
                "degenerate_critic": True}
    if raw is None:
        return {"raw_norm": None, "value_variance": var,
                "degenerate_critic": False}
    return {"raw_norm": float(raw) / var, "value_variance": var,
            "degenerate_critic": False}


def permutation_null(advantages: Any, cells: Any, actions: Any, *,
                     n_perm: int = 1000, seed: int = 20260827) -> dict:
    """NC-c: the reference distribution of `eta2` under NO action effect.

    Permutes the action labels WITHIN each cell. That destroys the
    action-successor pairing while preserving every marginal the reading
    depends on — the cell structure, the successor-value distribution, the
    critic, and the critic's scale. It is the only null that answers "is this
    instrument returning the same shape everywhere?" using the data itself
    rather than an asserted constant.
    """
    c = np.asarray(cells, dtype=np.int64).ravel()
    a = np.asarray(actions, dtype=np.int64).ravel()
    rng = np.random.default_rng(int(seed))
    order = np.argsort(c, kind="stable")
    cs = c[order]
    starts = np.flatnonzero(np.r_[True, np.diff(cs) != 0]) if cs.size else []
    blocks = [order[lo:hi] for lo, hi
              in zip(starts, np.r_[starts[1:], cs.size] if len(starts) else [])]
    draws = []
    for _ in range(int(n_perm)):
        shuffled = a.copy()
        for idx in blocks:
            shuffled[idx] = rng.permutation(a[idx])
        e = advantage_variance(advantages, c, shuffled)["eta2"]
        if e is not None:
            draws.append(e)
    if not draws:
        return {"n_perm": int(n_perm), "n_valid": 0, "median": None,
                "q025": None, "q975": None, "degenerate": True,
                "zero_spread": True}
    arr = np.asarray(draws, dtype=np.float64)
    q025, q975 = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
    return {
        "n_perm": int(n_perm), "n_valid": int(arr.size),
        "median": float(np.median(arr)), "q025": q025, "q975": q975,
        "degenerate": False,
        # A permutation that cannot move the statistic has no power to reject
        # anything; `classify_vadv` turns that into VOID, never COLLAPSED.
        "zero_spread": bool(q975 <= q025),
    }


def bootstrap_eta2(advantages: Any, cells: Any, actions: Any, *,
                   n_boot: int = 2000, seed: int = 20260827) -> dict:
    """95% CI of `eta2`, resampling CELLS (the independent unit), not rows."""
    adv = np.asarray(advantages, dtype=np.float64).ravel()
    c = np.asarray(cells, dtype=np.int64).ravel()
    a = np.asarray(actions, dtype=np.int64).ravel()
    uniq = np.unique(c)
    if uniq.size == 0:
        return {"lo": None, "hi": None, "n_boot": int(n_boot)}
    by_cell = {int(u): np.flatnonzero(c == u) for u in uniq}
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx, relabel = [], []
        for j, p in enumerate(pick):
            rows = by_cell[int(uniq[p])]
            idx.append(rows)
            relabel.append(np.full(rows.size, j, dtype=np.int64))
        idx = np.concatenate(idx)
        e = advantage_variance(adv[idx], np.concatenate(relabel), a[idx])["eta2"]
        if e is not None:
            draws.append(e)
    if not draws:
        return {"lo": None, "hi": None, "n_boot": int(n_boot)}
    lo, hi = (float(x) for x in np.percentile(np.asarray(draws), [2.5, 97.5]))
    return {"lo": lo, "hi": hi, "n_boot": int(n_boot), "n_valid": len(draws)}


def classify_vadv(observed: dict, null: dict, *, live_ratio: float = 1.5) -> str:
    """LIVE / COLLAPSED / VOID against the registered rule (§5 of the prereg).

    LIVE      : eta2 above the null's 97.5th percentile AND >= `live_ratio`x
                its median.
    COLLAPSED : eta2 at or below the null's 97.5th percentile.
    VOID      : no eta2, no null draws, or a permutation that cannot move the
                statistic at all — a test with no power must never be allowed
                to report COLLAPSED, because COLLAPSED is the more interesting
                conclusion and would otherwise be free.

    One identity carve-out: when the batch has NO variance whatsoever
    (`ss_between == ss_within == 0`) the absence of an action effect is an
    arithmetic fact rather than an inference, and that reads COLLAPSED.
    """
    e = observed.get("eta2")
    med, q975 = null.get("median"), null.get("q975")
    if e is None or med is None or q975 is None or int(null.get("n_valid", 0)) == 0:
        return "VOID"
    if e > q975 and (med <= 0.0 or e / med >= float(live_ratio)):
        return "LIVE"
    no_variance = (float(observed.get("ss_between", 1.0)) == 0.0
                   and float(observed.get("ss_within", 1.0)) == 0.0)
    if null.get("zero_spread") and not no_variance:
        return "VOID"
    return "COLLAPSED"


def injected_power(advantages: Any, cells: Any, actions: Any, *,
                   effect: float, n_trials: int = 200, n_perm: int = 200,
                   seed: int = 20260827, live_ratio: float = 1.5) -> dict:
    """A7: could this region have DETECTED a positive-control-sized effect?

    Adds a per-action offset of scale `effect` to the region's OWN rows and
    asks how often the registered LIVE rule fires. A thin region that cannot
    detect a real effect makes "collapsed here" unfalsifiable, and the honest
    answer is then VOID rather than a verdict. Low power biases toward
    reading COLLAPSED — the more interesting conclusion — so this gate guards
    the direction that would flatter the finding.
    """
    adv = np.asarray(advantages, dtype=np.float64).ravel()
    c = np.asarray(cells, dtype=np.int64).ravel()
    a = np.asarray(actions, dtype=np.int64).ravel()
    rng = np.random.default_rng(int(seed))
    fired = 0
    for _ in range(int(n_trials)):
        offsets = {int(k): rng.normal(0.0, float(effect))
                   for k in np.unique(a)}
        boosted = adv + np.array([offsets[int(x)] for x in a])
        obs = advantage_variance(boosted, c, a)
        null = permutation_null(boosted, c, a, n_perm=int(n_perm),
                                seed=int(rng.integers(1, 2 ** 31)))
        if classify_vadv(obs, null, live_ratio=live_ratio) == "LIVE":
            fired += 1
    return {"effect": float(effect), "n_trials": int(n_trials),
            "detected": int(fired), "power": fired / float(max(1, n_trials))}


def _to_numpy(v: Any) -> np.ndarray:
    """numpy view of an array-like or a torch tensor (duck-typed)."""
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy())
    return np.asarray(v)


def iterate_number(path: Any) -> Optional[int]:
    """The iteration number encoded in a ``vanilla_ppo_iter_NNNNN.pt`` name."""
    m = re.search(r"iter[_-]?(\d+)", str(path))
    return int(m.group(1)) if m else None


def sort_iterates(paths: Iterable[Any]) -> list:
    """Iterate paths ordered by their iteration number (unnumbered last)."""
    return sorted(paths, key=lambda p: (iterate_number(p) is None,
                                        iterate_number(p) or 0, str(p)))


def eval_cost_estimate(
    n_iterates: int, episodes: int, *, modes: int = 2,
    seconds_per_episode: float = 4.5, workers: int = 1,
) -> dict:
    """STUB COST for the paid instrument (pure arithmetic, no emulation).

    `seconds_per_episode` defaults to the measured worst case for this
    lineage: a stuck argmax burning the 1500-step cap
    (`configs/mario_1_1_backward_v4.yaml`, ~4.5 s). Reported so the decision
    to spend the emulation is made against a number rather than a vibe.
    """
    eps = int(n_iterates) * int(episodes) * int(modes)
    serial = eps * float(seconds_per_episode)
    return {
        "n_iterates": int(n_iterates),
        "episodes_per_cell": int(episodes),
        "modes": int(modes),
        "total_episodes": eps,
        "seconds_per_episode": float(seconds_per_episode),
        "serial_seconds": serial,
        "workers": max(1, int(workers)),
        "wall_seconds_estimate": serial / max(1, int(workers)),
    }


# ===========================================================================
# THIN SHELLS — torch / emulator. Everything above stays importable without
# either, so the pure suite runs with numpy alone.
# ===========================================================================

def load_iterate(path: Any) -> dict:
    """The policy `state_dict` inside a trainer checkpoint (CPU tensors)."""
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        for key in ("net_state_dict", "state_dict", "model_state_dict"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
        if all(hasattr(v, "shape") for v in blob.values()):
            return blob
    raise ValueError(f"{path}: no recognisable policy state_dict in checkpoint")


def build_tile_net(state_dict: dict):
    """Reconstruct a `TilePolicyNetwork` sized from the weights themselves.

    The trainer's checkpoints carry only `net_state_dict`, so the widths are
    read off `fc1/fc2/actor` rather than from a profile — which also means the
    instruments cannot be silently run against a differently-shaped net.
    """
    from src.models.tile_policy import TilePolicyNetwork

    try:
        hidden, feature = tuple(_to_numpy(state_dict["fc1.weight"]).shape)
        trunk, _ = tuple(_to_numpy(state_dict["fc2.weight"]).shape)
        actions, _ = tuple(_to_numpy(state_dict["actor.weight"]).shape)
    except KeyError as e:  # pragma: no cover - guarded by the caller
        raise ValueError(
            f"state_dict is not a TilePolicyNetwork (missing {e})"
        ) from e
    net = TilePolicyNetwork(
        num_actions=int(actions), feature_dim=int(feature),
        hidden_dim=int(hidden), trunk_dim=int(trunk),
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def tile_forward_with_trunk(net, states: Any) -> tuple:
    """`(logits, trunk_activations)` for `(N, feature_dim)` states.

    The trunk activation is the post-SiLU representation the heads read, i.e.
    the layer whose dormancy and effective rank the plasticity story is about.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.asarray(states, dtype=np.float32))
    with torch.no_grad():
        h1 = F.silu(net.norm1(net.fc1(x)))
        h2 = F.silu(net.norm2(net.fc2(h1)))
        logits = net.actor(h2)
    return _to_numpy(logits), _to_numpy(h2), _to_numpy(h1)


def tile_critic_values(net, states: Any, *, batch: int = 8192) -> np.ndarray:
    """`V(s)` for `(N, feature_dim)` states — the critic head, not the actor."""
    import torch

    arr = np.asarray(states, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, arr.shape[0], int(batch)):
            x = torch.as_tensor(arr[i:i + int(batch)])
            out.append(_to_numpy(net.forward_ac(x)[1]))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def load_transition_bank(paths: Sequence[str]) -> dict:
    """Pool offline `(s, a, s', done)` banks. NO emulator, NO training.

    Two accepted layouts, both already on disk:

    * ``state / action / next_state / done`` — a true transition bank
      (`runs/smodice_1_2/transitions.npz`, `runs/iq_1_2/transitions.npz`).
    * ``obs_0 / act_0 / traj_len`` — trajectory rows with explicit episode
      boundaries (`runs/interference/success_1_1.npz`); successors are the
      next row WITHIN a trajectory, so no transition is ever spliced across
      an episode boundary.

    A bank with `obs_0` but no `traj_len` is REFUSED rather than silently
    spliced: consecutive rows across an unknown boundary are not transitions.
    """
    S, A, N, D = [], [], [], []
    for p in paths:
        z = np.load(str(p), allow_pickle=True)
        keys = set(z.files)
        if {"state", "action", "next_state"} <= keys:
            S.append(np.asarray(z["state"]))
            A.append(np.asarray(z["action"], dtype=np.int64))
            N.append(np.asarray(z["next_state"]))
            D.append(np.asarray(z["done"], dtype=np.int64) if "done" in keys
                     else np.zeros(len(z["action"]), dtype=np.int64))
        elif {"obs_0", "act_0"} <= keys:
            if "traj_len" not in keys:
                raise ValueError(
                    f"{p}: obs_0/act_0 without traj_len — episode boundaries "
                    "are unknown, so consecutive rows are not transitions")
            obs = np.asarray(z["obs_0"])
            act = np.asarray(z["act_0"], dtype=np.int64)
            bounds = np.cumsum(np.r_[0, np.asarray(z["traj_len"], dtype=np.int64)])
            idx = np.concatenate([np.arange(b0, b1 - 1)
                                  for b0, b1 in zip(bounds[:-1], bounds[1:])
                                  if b1 - b0 >= 2])
            S.append(obs[idx])
            A.append(act[idx])
            N.append(obs[idx + 1])
            D.append(np.zeros(idx.size, dtype=np.int64))
        else:
            raise ValueError(f"{p}: unrecognised bank schema {sorted(keys)}")
    return {
        "state": np.concatenate(S), "action": np.concatenate(A),
        "next_state": np.concatenate(N), "done": np.concatenate(D),
        "sources": [str(p) for p in paths],
    }


def parse_bands(spec: str) -> dict:
    """``"WALL=2674:2872,PC=2872:3266"`` -> ``{"WALL": (2674, 2872), ...}``.

    Half-open on the right, matching the registered band table.
    """
    bands: dict = {}
    for part in [s for s in str(spec).split(",") if s.strip()]:
        name, _, rng = part.partition("=")
        lo, _, hi = rng.partition(":")
        bands[name.strip()] = (int(lo), int(hi))
    return bands


def score_vadv(bank: dict, values_fn: Callable, *, bands: dict,
               gamma: float = 0.99, exact_key: bool = False,
               min_rows_per_action: int = 2, min_actions: int = 2,
               min_rows: int = 6, n_perm: int = 1000, n_boot: int = 2000,
               seed: int = 20260827, gx_frozen_band: Optional[str] = None) -> dict:
    """V_adv per registered band for ONE checkpoint. Emulator-free.

    `values_fn` is duck-typed (``states -> V(s)``) so the whole path is
    testable without torch. ``Var_batch[V]`` for the normaliser is computed
    ONCE over the pooled union of the scored bands, so every band shares one
    denominator and regional readings stay comparable.
    """
    state = bank["state"]
    action = np.asarray(bank["action"], dtype=np.int64)
    gx = decode_gx(state)
    cells = state_cell_keys(state, exact=exact_key)
    in_any = np.zeros(gx.size, dtype=bool)
    for lo, hi in bands.values():
        in_any |= (gx >= lo) & (gx < hi)
    if not in_any.any():
        return {"error": "no rows in any registered band", "bands": bands}

    v_state = np.asarray(values_fn(state[in_any]), dtype=np.float64)
    v_next = np.asarray(values_fn(bank["next_state"][in_any]), dtype=np.float64)
    adv_all = critic_advantage(v_state, v_next,
                               done=np.asarray(bank["done"])[in_any],
                               gamma=gamma)
    gx_next_all = decode_gx(bank["next_state"][in_any])
    sub = np.flatnonzero(in_any)
    norm = normalise_vadv(None, float(np.var(v_state)))

    def _region(mask: np.ndarray) -> dict:
        keep = qualifying_rows(cells[sub][mask], action[sub][mask],
                               min_rows_per_action=min_rows_per_action,
                               min_actions=min_actions, min_rows=min_rows)
        a_, c_, act_ = (adv_all[mask][keep], cells[sub][mask][keep],
                        action[sub][mask][keep])
        obs = advantage_variance(a_, c_, act_)
        null = permutation_null(a_, c_, act_, n_perm=n_perm, seed=seed)
        row = {
            "n_rows_in_band": int(mask.sum()),
            "n_qualifying_rows": int(keep.sum()),
            "observed": obs,
            "null": null,
            "bootstrap": bootstrap_eta2(a_, c_, act_, n_boot=n_boot, seed=seed),
            "verdict": classify_vadv(obs, null),
        }
        row.update(normalise_vadv(obs.get("raw"), norm["value_variance"]))
        if row["degenerate_critic"]:
            # A constant critic makes EVERY advantage exactly zero, so the
            # arithmetic carve-out in `classify_vadv` would hand back a
            # confident COLLAPSED — manufacturing the mis-specification
            # signature out of a broken checkpoint. The critic is the
            # instrument here; a dead one reads VOID.
            row["verdict"] = "VOID"
            row["void_reason"] = "degenerate_critic"
        row["coverage_ok"] = bool(obs.get("n_cells", 0) >= 20
                                  and row["n_qualifying_rows"] >= 400)
        return row

    out = {"bands": {k: list(v) for k, v in bands.items()},
           "gamma": float(gamma), "exact_key": bool(exact_key),
           "value_variance": norm["value_variance"],
           "degenerate_critic": norm["degenerate_critic"],
           "regions": {}}
    for name, (lo, hi) in bands.items():
        g = gx[sub]
        out["regions"][name] = _region((g >= lo) & (g < hi))
    if gx_frozen_band and gx_frozen_band in bands:
        lo, hi = bands[gx_frozen_band]
        g = gx[sub]
        band = (g >= lo) & (g < hi)
        # NC-b: cells where NO tried action moves gx, so a progress-based
        # reward is provably flat across the actions actually tried.
        frozen = band & (gx_next_all == g)
        cid = cells[sub]
        moving = set(cid[band & (gx_next_all != g)].tolist())
        out["regions"]["NEG_gx_frozen"] = _region(
            frozen & ~np.isin(cid, list(moving)))
    return out


def score_one_iterate(
    path: Any,
    *,
    states: Optional[np.ndarray],
    theta_0: Optional[dict],
    tie_threshold: float,
    dormant_tau: float,
    srank_delta: float,
    forward: Optional[Callable] = None,
    bank: Optional[dict] = None,
    vadv_kwargs: Optional[dict] = None,
    values_fn: Optional[Callable] = None,
) -> dict:
    """Every FREE instrument for one iterate. Emulator-free by construction.

    `forward` is injectable (duck-typed: ``states -> (logits, trunk, hidden)``)
    so the sweep can be exercised end to end without torch or a checkpoint.
    """
    row: dict = {"path": str(path), "iter": iterate_number(path)}
    sd = load_iterate(path) if forward is None else {}
    if theta_0 is not None and sd:
        row["drift"] = param_drift(sd, theta_0)
    if bank is not None:
        vfn = values_fn
        if vfn is None:
            net = build_tile_net(sd)
            width = int(_to_numpy(sd["fc1.weight"]).shape[1])
            if width != int(bank["state"].shape[1]):
                # A8: never let load_state_dict(strict=False) hand back a
                # half-random first layer that scores clean.
                raise ValueError(
                    f"checkpoint input width {width} != bank observation "
                    f"width {bank['state'].shape[1]}")

            def vfn(s, _net=net):
                return tile_critic_values(_net, s)

        row["vadv"] = score_vadv(bank, vfn, **(vadv_kwargs or {}))
    if states is None:
        return row
    fwd = forward
    if fwd is None:
        net = build_tile_net(sd)

        def fwd(s, _net=net):
            return tile_forward_with_trunk(_net, s)

    logits, trunk, hidden = fwd(states)
    row["margin"] = margin_summary(logits, tie_threshold=tie_threshold)
    row["dormant"] = {
        "trunk": dormant_fraction(trunk, tau=dormant_tau),
        "hidden": dormant_fraction(hidden, tau=dormant_tau),
        "tau": float(dormant_tau),
    }
    row["effective_rank"] = {
        "trunk": effective_rank(trunk, delta=srank_delta),
        "hidden": effective_rank(hidden, delta=srank_delta),
        "delta": float(srank_delta),
    }
    return row


def score_curve_cell(
    path: Any, *, profile: str, rom: Optional[str], start_state: Optional[str],
    episodes: int, max_steps: int, sticky_prob: float, start_jitter: int,
    eval_seed: int, eval_workers: int, action_select: str,
) -> dict:
    """ONE paid cell: subprocess `eval_game.py` for this iterate + mode.

    Uses the same harness for every cell (``--sequential --level-clear`` from
    an explicit start state) because the point of the curve is that greedy and
    sampled are PAIRED — a curve assembled from two harnesses is not a curve.
    """
    import yaml

    from src.training import cold_probe

    cfg = yaml.safe_load(Path(profile).read_text())
    sd = load_iterate(path)
    net = build_tile_net(sd)
    res = cold_probe.probe(
        net, cfg, episodes=episodes, sequential=True, level_clear=True,
        start_state=start_state, max_steps=max_steps,
        rom_path=rom or cfg.get("rom_path"),
        game=str(cfg.get("name", "mario")),
        sticky_prob=sticky_prob, start_jitter=start_jitter,
        eval_seed=eval_seed, eval_workers=eval_workers,
        eval_rng="per-episode" if eval_workers > 1 else None,
        action_select=action_select,
    )
    return {
        "action_select": action_select,
        "clear_rate": res.get("cold_seq_clear_rate"),
        "n_episodes": res.get("cold_n_episodes"),
        "mean_length": res.get("cold_mean_length"),
        "status": res.get("cold_status"),
        "error": res.get("cold_error"),
        "protocol": {
            "sticky_prob": float(sticky_prob),
            "start_jitter": int(start_jitter),
            "eval_seed": int(eval_seed),
            "eval_workers": int(eval_workers),
            "max_steps": int(max_steps),
            "harness": "eval_game.py --sequential --level-clear",
        },
    }


def expand_iterates(patterns: Sequence[str]) -> list:
    """Glob every `--iterates` pattern, de-duplicate, order by iteration."""
    hits: list = []
    seen: set = set()
    for pat in patterns:
        for p in glob.glob(pat):
            rp = str(Path(p).resolve())
            if rp not in seen:
                seen.add(rp)
                hits.append(Path(p))
    return sort_iterates(hits)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline instrument sweep over banked iterates "
                    "(no training; emulation is opt-in).",
    )
    ap.add_argument("--iterates", action="append", default=[], metavar="GLOB",
                    help="Glob of checkpoints to score. Repeatable.")
    ap.add_argument("--baseline", default=None, metavar="CKPT",
                    help="theta_0 for the drift instrument. Default: the "
                         "lowest-numbered iterate in the sweep.")
    ap.add_argument("--states", default=None, metavar="NPY",
                    help="(N, feature_dim) float array of observations — the "
                         "states the logit margin / dormancy / effective rank "
                         "are measured AT. Omit to run drift only.")
    ap.add_argument("--tie-threshold", type=float, default=0.15,
                    help="Reported (not believed) margin tie threshold; "
                         "calibrate it against a healthy iterate.")
    ap.add_argument("--dormant-tau", type=float, default=0.0,
                    help="Sokar tau. 0.0 = the strict (exactly dead) rule.")
    ap.add_argument("--srank-delta", type=float, default=0.01)
    # --- instrument 6: V_adv (offline; still no emulator) ----------------
    ap.add_argument("--transitions", action="append", default=[], metavar="NPZ",
                    help="Offline (s,a,s') bank for V_adv. Repeatable; banks "
                         "are pooled. Omit to skip V_adv entirely.")
    ap.add_argument("--vadv-bands", default="WALL=2674:2872,PC_B5=2872:3267",
                    help="gx bands, NAME=lo:hi (half-open). Registered "
                         "defaults are B5's own wall and the rungs its "
                         "curriculum advanced through.")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="Discount for A-hat = gamma*V(s')*(1-done) - V(s).")
    ap.add_argument("--vadv-exact-key", action="store_true",
                    help="Registered sensitivity check: key cells on the "
                         "whole observation instead of the current frame.")
    ap.add_argument("--vadv-perm", type=int, default=1000)
    ap.add_argument("--vadv-boot", type=int, default=2000)
    ap.add_argument("--vadv-seed", type=int, default=20260827)
    ap.add_argument("--vadv-frozen-band", default=None, metavar="NAME",
                    help="Band to draw the NC-b gx-frozen negative control "
                         "from (a region where a progress-based reward is "
                         "provably flat across the actions tried).")
    ap.add_argument("--out", default=None, metavar="JSON",
                    help="Write the sweep here as well as to stdout.")
    # --- the paid instrument -------------------------------------------
    ap.add_argument("--eval", action="store_true",
                    help="Actually run the greedy/sampled curve (subprocesses "
                         "eval_game.py per iterate). OFF by default: the "
                         "sweep stub-costs it instead.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--sticky-prob", type=float, default=0.25)
    ap.add_argument("--start-jitter", type=int, default=16)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--seconds-per-episode", type=float, default=4.5,
                    help="Cost model for the stub (measured worst case).")
    args = ap.parse_args(argv)

    paths = expand_iterates(args.iterates)
    if not paths:
        print("no iterates matched", file=sys.stderr)
        return 2

    states = None
    if args.states:
        states = np.asarray(np.load(args.states), dtype=np.float32)

    theta_0 = None
    base_path = args.baseline or (str(paths[0]) if paths else None)
    if base_path:
        theta_0 = load_iterate(base_path)

    bank = load_transition_bank(args.transitions) if args.transitions else None
    vadv_kwargs = None if bank is None else {
        "bands": parse_bands(args.vadv_bands), "gamma": args.gamma,
        "exact_key": args.vadv_exact_key, "n_perm": args.vadv_perm,
        "n_boot": args.vadv_boot, "seed": args.vadv_seed,
        "gx_frozen_band": args.vadv_frozen_band,
    }

    rows = []
    for p in paths:
        try:
            row = score_one_iterate(
                p, states=states, theta_0=theta_0,
                tie_threshold=args.tie_threshold, dormant_tau=args.dormant_tau,
                srank_delta=args.srank_delta,
                bank=bank, vadv_kwargs=vadv_kwargs,
            )
            if args.eval:
                if not args.profile:
                    print("--eval needs --profile", file=sys.stderr)
                    return 2
                row["curve"] = [
                    score_curve_cell(
                        p, profile=args.profile, rom=args.rom,
                        start_state=args.start_state, episodes=args.episodes,
                        max_steps=args.max_steps, sticky_prob=args.sticky_prob,
                        start_jitter=args.start_jitter, eval_seed=args.eval_seed,
                        eval_workers=args.eval_workers, action_select=mode,
                    )
                    for mode in ("greedy", "sampled")
                ]
        except Exception as exc:
            # One bad iterate (truncated .pt, unrecognised architecture) must
            # not cost every iterate already scored this run: report it in
            # its own row and keep going, so stdout/--out still land.
            print(f"{p}: {exc}", file=sys.stderr)
            row = {"path": str(p), "iter": iterate_number(p), "error": str(exc)}
        rows.append(row)

    out = {
        "baseline": base_path,
        "n_iterates": len(rows),
        "states": args.states,
        "n_states": int(states.shape[0]) if states is not None else 0,
        "eval_ran": bool(args.eval),
        "iterates": rows,
    }
    if bank is not None:
        out["vadv_bank"] = {
            "sources": bank["sources"], "n_transitions": int(bank["state"].shape[0]),
            "obs_width": int(bank["state"].shape[1]),
            "prereg": "docs/proposals/VADV_PREREG_2026-08-27.md",
        }
    if not args.eval:
        out["curve_stub_cost"] = eval_cost_estimate(
            len(rows), args.episodes,
            seconds_per_episode=args.seconds_per_episode,
            workers=args.eval_workers,
        )
        out["curve_stub_note"] = (
            "greedy/sampled curve NOT run (--eval omitted); the cost above is "
            "an estimate, not a measurement"
        )
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
