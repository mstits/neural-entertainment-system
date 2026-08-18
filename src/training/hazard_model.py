"""Discrete-time survival (hazard) model — Phase 2 of the hazard substrate.

Context (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md, v18/v19/v20):
three independent research rounds converged on the same artifact — a
discrete-time hazard model trained by survival analysis on our own
deaths, predicting the per-interval conditional probability of death
given survival so far, from the tile observation plus the action taken.
This module is Phase 2 only: model + loss + the gate metric. It never
touches the emulator, a rollout, or a checkpoint from any live campaign
-- every array it consumes is a plain numpy/torch tensor the caller
already owns (loaded from an npz on disk, or built synthetically in
tests). Phase 3 (masking actions above a hazard threshold during PPO)
is explicitly out of scope until this phase's gate is met.

THE GATE (from the same synthesis doc, non-negotiable): Uno's C-index
with inverse-probability-of-censoring weighting (IPCW) >= 0.85 on a
held-out set, split by SOURCE STATE (see scripts/train_hazard.py), not
by row -- rows forked from the same saved state are not independent
samples, and splitting by row would leak the source state's outcome
across train/val. Below 0.85 the observation does not resolve threats
well enough to act on and Phase 3 does not start.

Discrete-time survival likelihood (the "Nnet-survival" formulation,
Gensheimer & Narasimhan 2019 -- a standard, publicly documented
statistical method, not game-specific knowledge):

  The observation horizon is chopped into `n_bins` intervals. The model
  emits one hazard logit per interval: h_k = sigmoid(logit_k) is the
  model's estimate of P(event in interval k | survived through interval
  k-1). For a sample whose event/censoring falls in bin y (0-indexed):

    uncensored (the event happened):
      log-lik = sum_{j<y} log(1 - h_j)  +  log(h_y)
              = "survived every earlier bin, then failed in bin y"

    censored (observation window ended before any event):
      log-lik = sum_{j<=y} log(1 - h_j)
              = "survived every bin we actually observed, including y"

  The two cases share the same "survived the earlier bins" term and
  differ only in what happens AT bin y: fail (uncensored) vs. also
  survive it (censored). That is the whole mechanism; everything below
  is numerically-stable bookkeeping around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matches the hazard_collect.py schema this lane was told to write
# against: obs is int8 (N, 712) -- the stacked tile observation used
# everywhere else in this codebase (see src/training/cold_probe.py,
# scripts/eval_game.py). Action is a scalar index into a 6-action space
# (matches the SMB tile-mode action set used by the live 2-1 campaign
# and configs/mario_tiles.yaml).
OBS_DIM = 712
NUM_ACTIONS = 6
INPUT_DIM = OBS_DIM + NUM_ACTIONS

DEFAULT_HIDDEN = 128
DEFAULT_HIDDEN_LAYERS = 2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HazardMLP(nn.Module):
    """Tile-obs + one-hot-action -> per-bin hazard logits.

    ~100k params at the defaults (input_dim=718, hidden=128, 2 hidden
    layers, n_bins~20): layer 1 dominates the count (718*128+128 ~=
    92k), so `n_bins` barely moves the total -- the "~100k params"
    target in the spec is a property of the trunk, not the head.

    forward() returns RAW LOGITS, not probabilities. Every consumer
    (the loss, the risk score, `hazard_probs`/`survival_probs` below)
    is written against logits directly via `F.logsigmoid`, which is
    numerically stable at the extremes where log(sigmoid(x)) or
    log(1 - sigmoid(x)) would otherwise underflow to log(0).
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = DEFAULT_HIDDEN,
                 n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS, n_bins: int = 20):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.n_hidden_layers = int(n_hidden_layers)
        self.n_bins = int(n_bins)

        layers: list[nn.Module] = []
        prev = self.input_dim
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(prev, self.hidden))
            layers.append(nn.ReLU())
            prev = self.hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.n_bins)

        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) float. Returns (N, n_bins) raw hazard logits."""
        return self.head(self.trunk(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def encode_input(obs: np.ndarray, action: np.ndarray,
                  num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """(N, OBS_DIM) int8 obs + (N,) int action -> (N, OBS_DIM+num_actions) float32.

    Action is one-hot encoded, matching the "712-dim tile observation
    plus the action (one-hot, 6 actions)" input contract. Obs is cast
    to float32 as-is (tile ids are small integers; no normalization is
    imposed here so this stays a drop-in match for whatever scale the
    rest of the codebase already uses for the same 712-dim vector).
    """
    obs = np.asarray(obs)
    action = np.asarray(action)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"expected obs shape (N, {OBS_DIM}), got {obs.shape}")
    if action.shape[0] != obs.shape[0]:
        raise ValueError(
            f"obs has {obs.shape[0]} rows but action has {action.shape[0]}")
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    action_idx = torch.as_tensor(action.astype(np.int64), dtype=torch.long)
    if action_idx.numel() and (action_idx.min() < 0 or action_idx.max() >= num_actions):
        raise ValueError(
            f"action indices must be in [0, {num_actions}), got range "
            f"[{int(action_idx.min())}, {int(action_idx.max())}]")
    onehot = F.one_hot(action_idx, num_classes=num_actions).to(torch.float32)
    return torch.cat([obs_t, onehot], dim=1)


# ---------------------------------------------------------------------------
# Time discretization
# ---------------------------------------------------------------------------


def build_time_bin_edges(horizon: float, n_bins: int) -> np.ndarray:
    """Equal-width bin edges covering [0, horizon] in n_bins intervals."""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    return np.linspace(0.0, float(horizon), int(n_bins) + 1)


def discretize_time(steps_to_event: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Continuous steps-to-event -> 0-indexed bin index in [0, n_bins-1].

    `np.searchsorted(edges, t, side="right") - 1` puts t on the bin
    [edges[k], edges[k+1]); values at or beyond the horizon clip into
    the final bin (right-censoring beyond the modeled horizon is still
    "survived through the last bin we can represent").
    """
    steps_to_event = np.asarray(steps_to_event, dtype=np.float64)
    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, steps_to_event, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# Discrete-time survival loss
# ---------------------------------------------------------------------------


def discrete_time_survival_nll(logits: torch.Tensor, bin_idx: torch.Tensor,
                                censored: torch.Tensor,
                                reduction: str = "mean") -> torch.Tensor:
    """Negative log-likelihood of the discrete-time survival model.

    logits:   (N, n_bins) raw hazard logits (model output).
    bin_idx:  (N,) long, the bin the event/censoring fell in, in
              [0, n_bins-1] (see `discretize_time`).
    censored: (N,) float/bool/long, 1 if the sample is right-censored
              (no event observed), 0 if the event (death) happened in
              bin_idx.

    Per-sample log-likelihood, vectorized with a bin mask so no Python
    loop over the batch is needed:

      log_surv[j]  = log(1 - h_j) = F.logsigmoid(-logits[j])   (all bins)
      log_fail[j]  = log(h_j)     = F.logsigmoid(logits[j])    (all bins)

      before_mask[i, j] = 1 if j < bin_idx[i]           (bins strictly
                                                           before the
                                                           event/censor
                                                           bin: always
                                                           "survived")
      at_mask[i, j]     = 1 if j == bin_idx[i]           (the bin the
                                                           event/censor
                                                           fell in)

      ll_i = sum_j before_mask[i,j] * log_surv[i,j]
           + at_mask[i,:] . ( uncensored[i] * log_fail[i,:]
                               + censored[i]   * log_surv[i,:] )

    which is exactly the two cases in the module docstring: censored
    rows get log_surv at the boundary bin too (survived it, no
    failure observed); uncensored rows get log_fail there instead
    (failed in it). Bins after bin_idx never contribute -- they were
    never observed for either the uncensored or censored case.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N, n_bins), got shape {tuple(logits.shape)}")
    n, n_bins = logits.shape
    bin_idx = bin_idx.to(torch.long)
    if bin_idx.shape[0] != n:
        raise ValueError(f"bin_idx has {bin_idx.shape[0]} rows, logits has {n}")
    if (bin_idx < 0).any() or (bin_idx >= n_bins).any():
        raise ValueError(f"bin_idx out of range [0, {n_bins})")
    censored = censored.to(torch.float32).reshape(-1)
    if censored.shape[0] != n:
        raise ValueError(f"censored has {censored.shape[0]} rows, logits has {n}")

    log_surv = F.logsigmoid(-logits)  # log(1 - h_j), (N, n_bins)
    log_fail = F.logsigmoid(logits)   # log(h_j),     (N, n_bins)

    col = torch.arange(n_bins, device=logits.device).unsqueeze(0)  # (1, n_bins)
    row = bin_idx.unsqueeze(1)                                     # (N, 1)
    before_mask = (col < row).to(logits.dtype)
    at_mask = (col == row).to(logits.dtype)

    survived_earlier = (before_mask * log_surv).sum(dim=1)
    uncensored = 1.0 - censored
    at_bin_term = (at_mask * (uncensored.unsqueeze(1) * log_fail
                               + censored.unsqueeze(1) * log_surv)).sum(dim=1)

    log_lik = survived_earlier + at_bin_term
    nll = -log_lik

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def hazard_probs(logits: torch.Tensor) -> torch.Tensor:
    """(N, n_bins) hazard logits -> (N, n_bins) per-bin hazard probabilities."""
    return torch.sigmoid(logits)


def survival_probs(logits: torch.Tensor) -> torch.Tensor:
    """(N, n_bins) hazard logits -> (N, n_bins) cumulative survival S(k),
    S(k) = P(survive through bin k) = prod_{j<=k} (1 - h_j)."""
    log_surv = F.logsigmoid(-logits)
    return torch.exp(torch.cumsum(log_surv, dim=1))


def cumulative_risk_score(logits: torch.Tensor) -> torch.Tensor:
    """Scalar risk score per row: total hazard accumulated across every
    modeled bin, i.e. -log S(n_bins-1) = sum_j -log(1 - h_j).

    Higher = the model thinks this (obs, action) pair is more likely to
    end in death somewhere within the modeled horizon; used as the risk
    score fed to the concordance index below. This integrates over the
    whole horizon rather than committing to one evaluation time tau, so
    it does not need a tau chosen ahead of the C-index call.
    """
    log_surv = F.logsigmoid(-logits)
    return (-log_surv).sum(dim=1)


# ---------------------------------------------------------------------------
# Kaplan-Meier + Uno's C-index with IPCW (the gate metric)
# ---------------------------------------------------------------------------


@dataclass
class KaplanMeier:
    """Kaplan-Meier survival curve, fit from scratch (no lifelines/sksurv
    dependency in this environment). `times`/`events` follow the usual
    survival-analysis convention: event=1 means the event of interest
    was observed at that time, event=0 means censored at that time."""

    unique_times: np.ndarray = field(default_factory=lambda: np.array([]))
    survival: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def fit(cls, times: np.ndarray, events: np.ndarray) -> "KaplanMeier":
        times = np.asarray(times, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)
        order = np.argsort(times)
        times, events = times[order], events[order]

        uniq = np.unique(times)
        surv = np.empty_like(uniq)
        s = 1.0
        n_at_risk = len(times)
        idx = 0
        for k, t in enumerate(uniq):
            # everyone with time == t is either an event or censored at t;
            # both leave the risk set after this step, only events shrink S.
            mask = times == t
            d = int(events[mask].sum())          # events at t
            n = n_at_risk                          # at risk just before t
            if n > 0 and d > 0:
                s *= (1.0 - d / n)
            surv[k] = s
            n_at_risk -= int(mask.sum())
        return cls(unique_times=uniq, survival=surv)

    def eval_left(self, query_times: np.ndarray) -> np.ndarray:
        """G(t-): survival probability strictly before each query time
        (step function evaluated just to the left of any jump AT that
        time, so a subject's own censoring/event time still counts as
        'at risk' going into that instant). Times before the first
        observed time get probability 1.0."""
        query_times = np.asarray(query_times, dtype=np.float64)
        if len(self.unique_times) == 0:
            return np.ones_like(query_times)
        # index of the last unique_time strictly less than query_times
        pos = np.searchsorted(self.unique_times, query_times, side="left") - 1
        out = np.where(pos < 0, 1.0, self.survival[np.clip(pos, 0, None)])
        return out


def concordance_index_ipcw(times: np.ndarray, events: np.ndarray,
                            risk_scores: np.ndarray, tau: Optional[float] = None,
                            eps: float = 1e-3) -> dict:
    """Uno's C-index with inverse-probability-of-censoring weighting.

    Standard survival-analysis metric (Uno, Cai, Pencina, D'Agostino &
    Wei, 2011) -- generic statistics, not game-specific. Implemented
    from scratch because neither lifelines nor sksurv is installed in
    this environment; every step below is the published formula, no
    shortcuts.

    Why IPCW at all: an unweighted (Harrell's) C-index restricts
    comparable pairs to those where the earlier time is an observed
    event, which silently under-represents subjects who survive long
    (heavily censored, i.e. exactly our successful trajectories) in the
    denominator's implicit population. IPCW corrects this by weighting
    each comparable pair by the inverse SQUARED probability of the
    earlier subject remaining uncensored up to their own time, using a
    Kaplan-Meier estimate of the CENSORING distribution (fit with the
    censoring indicator, 1-event, playing the role of "event").

    times:       (N,) observed time (steps_to_event OR the censoring
                 time, whichever came first for that row).
    events:      (N,) 1 = event (death) observed at `times`, 0 =
                 censored at `times`.
    risk_scores: (N,) higher = model predicts higher risk (shorter
                 survival). See `cumulative_risk_score`.
    tau:         optional truncation time; pairs with the earlier time
                 beyond tau are excluded (matches Uno's definition of a
                 restricted C-index). Defaults to the max observed time
                 (no truncation).
    eps:         floor on G_hat(t-) before inverting, so a subject
                 observed at or after the last censoring time (where
                 G_hat could legitimately hit 0) does not blow the
                 weight up to infinity.

    Returns a dict: {"c_index", "n_pairs", "n_comparable", "n_concordant"}.
    n_comparable/n_concordant are WEIGHTED sums (not raw counts), since
    that is what the ratio is actually built from; a degenerate
    dataset (fewer than 2 events, or zero comparable pairs) returns
    c_index=float("nan") rather than raising, since "no gate signal
    yet" is a legitimate state for a script to report and move past.
    """
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    n = len(times)
    if not (len(events) == n and len(risk_scores) == n):
        raise ValueError("times, events, risk_scores must be the same length")
    if tau is None:
        tau = float(times.max()) if n else 0.0

    censoring_km = KaplanMeier.fit(times, 1.0 - events)
    g_at_times = censoring_km.eval_left(times)
    g_at_times = np.clip(g_at_times, eps, 1.0)
    weights = 1.0 / (g_at_times ** 2)

    comparable = 0.0
    concordant = 0.0
    n_pairs_evaluated = 0
    event_idx = np.where((events == 1) & (times <= tau))[0]
    for i in event_idx:
        # j is a valid comparison partner for i iff j's time is strictly
        # later than i's (i's death is known to precede j's outcome,
        # whether j is itself an event or still-censored at that later
        # time -- both tell us i truly died first).
        later = times > times[i]
        if not later.any():
            continue
        w = weights[i]
        n_j = int(later.sum())
        n_pairs_evaluated += n_j
        comparable += w * n_j
        ri = risk_scores[i]
        rj = risk_scores[later]
        concordant += w * (np.sum(ri > rj) + 0.5 * np.sum(ri == rj))

    c_index = float(concordant / comparable) if comparable > 0 else float("nan")
    return {
        "c_index": c_index,
        "n_pairs": n_pairs_evaluated,
        "n_comparable": comparable,
        "n_concordant": concordant,
    }


# ---------------------------------------------------------------------------
# Source-state grouping (for the train/val split — see scripts/train_hazard.py)
# ---------------------------------------------------------------------------

# Column names checked, in priority order, for an explicit source-state
# id in a collector npz. `source_state_idx` matches the field
# scripts/hazard_collect.py's build_fork_jobs/collect_labels actually
# write (the index into its `source_states` restore-point list); the
# rest are defensive synonyms in case a future collector revision uses
# a different name. hazard_collect.py's schema was summarized in this
# lane's brief as (obs, action, died, steps_to_event, censored) without
# it, so the hash-based fallback below stays in place for any npz that
# genuinely lacks an id column, but the live collector's own column is
# checked first and is what real data uses.
SOURCE_ID_KEYS: Sequence[str] = (
    "source_state_idx", "source_state_id", "state_id", "fork_id",
    "source_id", "episode_id",
)


def infer_source_groups(npz) -> np.ndarray:
    """Recover a per-row source-state group id from an open npz (or any
    mapping with the same key surface as hazard_collect.py's output).

    Preferred path: an explicit id column, if the collector schema
    grows one (see SOURCE_ID_KEYS above).

    Fallback path (the one this schema currently requires): micro-
    forking logs the PRE-FORK observation as `obs` for every child
    forked from the same saved state -- the branch action is what
    differs, applied AFTER that observation is captured, so all rows
    forked from one state share a byte-identical `obs` row. Hashing
    each row's raw bytes therefore recovers exactly the grouping the
    "split by source state, not by sample" rule needs, without
    depending on the collector adding a dedicated id field. This is a
    property of the logging order (state observed, then action forked),
    not a guess -- but it silently degrades to per-row groups (i.e. no
    grouping at all) if some future collector revision perturbs `obs`
    per-child (e.g. logs post-action rather than pre-action), so
    scripts/train_hazard.py prints the group-count diagnostic so that
    silent degradation would be visible, not hidden.
    """
    for key in SOURCE_ID_KEYS:
        if key in npz:
            ids = np.asarray(npz[key])
            return ids

    obs = np.asarray(npz["obs"])
    if obs.ndim != 2:
        raise ValueError(f"expected 2D obs for grouping, got shape {obs.shape}")
    # Bytes -> a stable hash per row without materializing an object
    # array; np.void view of each row's raw bytes is hashable.
    contiguous = np.ascontiguousarray(obs)
    row_bytes = contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).reshape(-1)
    _, inverse = np.unique(row_bytes, return_inverse=True)
    return inverse.astype(np.int64)
