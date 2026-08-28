"""ReDo — Recycling Dormant neurons for the feedforward tile policy.

Registered mechanism: docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md,
AMENDMENT 1 (B3/B4/B6). Sokar et al. 2023 ("The Dormant Neuron
Phenomenon in Deep Reinforcement Learning"), Definition 1, applied to
`TilePolicyNetwork`'s two hidden layers only:

- Dormancy statistic (B3.1): for hidden unit i of a layer, over a
  sample batch drawn from the just-collected rollout,
      s_i = E_x[|h_i(x)|] / ((1/H) * sum_k E_x[|h_k(x)|])
  where h is the POST-SiLU activation. Dormant iff s_i <= tau. The
  normalizer is the LAYER MEAN (the paper's Definition 1), a registered
  deliberate correction of the DR consultation's "layer's maximum"
  phrasing — the 0.025-0.1 threshold range is calibrated to the mean.
- Recycle (B3.5), per dormant unit:
    * incoming weight row + bias re-sampled Kaiming-uniform at the
      layer's fan-in (nn.Linear's default reset distribution — the
      original orthogonal init has no per-row re-sample);
    * that unit's LayerNorm affine reset to weight=1 / bias=0;
    * outgoing weight column set exactly to 0.0 — for an fc2 unit that
      is BOTH the actor and the critic column. Head units themselves
      are never recycled.
- Optimizer handling (B3.6): Adam `exp_avg` / `exp_avg_sq` slices for
  every touched entry are zeroed; per-tensor `step` counters are left
  alone (they are per-tensor, not per-element).
- Identity caveat (B3.7): with PRE-ACTIVATION LayerNorm the classic
  "zero outgoing => identical output" argument is only approximate
  (re-initializing row i shifts the layer statistics seen by the other
  units), so every recycle event measures pre/post greedy-argmax
  agreement and max|delta logit| on the same sample batch. Those two
  numbers ride the registered `[redo]` log line and gate the V7
  forced-recycle preflight.

`maybe_check_and_recycle` is the trainer-facing gate: with
`enabled=False` it returns before touching the network, the optimizer,
or ANY RNG stream, so a redo-off run is byte-identical to a build
without this module (unit-tested in tests/test_redo_mechanism.py).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# In-run dose ceiling (V31_REDO_SURGICAL_2026-08-27.md §3). A fixed tau's
# dormant tail drifts DOWN across training (measured, v30 §1.3/§5), so a
# check that is surgical at iter 30 can be a partial network reset by
# iter 200. This is the early-abort: the SAME numeral V6 uses at
# verdict-time, checked every dormancy check instead of only at the end,
# so a mis-dosed run is caught in ~15 min instead of burning to iter 250.
DOSE_CEILING_WINDOW = 10
DOSE_CEILING_FRAC = 0.25


def dose_fraction(
    dormant_fc1: int, hidden_dim: int, dormant_fc2: int, trunk_dim: int,
) -> float:
    """Worst-hit-layer recycled fraction for one dormancy check.

    Never pooled: fc1 is 0/64 dormant at every measured tau (v30 §1.1),
    so a pooled `(dormant_fc1+dormant_fc2)/(hidden_dim+trunk_dim)` carries
    permanent ballast that can never contribute to the numerator — it
    would report 20/96 = 21% for an event that re-initialized 20 of 32
    trunk units (62%). `max()` of the two per-layer fractions is the
    statistic that tracks the actual damage.
    """
    f1 = (dormant_fc1 / hidden_dim) if hidden_dim else 0.0
    f2 = (dormant_fc2 / trunk_dim) if trunk_dim else 0.0
    return max(f1, f2)


def dose_ceiling_trips(
    history: list[float] | tuple[float, ...],
    *,
    window: int = DOSE_CEILING_WINDOW,
    ceiling: float = DOSE_CEILING_FRAC,
) -> bool:
    """True iff the trailing-`window`-CHECK median dose exceeds `ceiling`.

    `history` is one `dose_fraction()` value per dormancy check —
    including zero-recycle checks, appended in order — never per firing
    event: a median over firing events only is blind to how OFTEN the
    treatment fires, and this ceiling exists specifically to see that.
    Fewer than `window` checks so far always returns False (there is
    nothing to abort on yet); strict `>` so exactly `ceiling` (e.g.
    8/32 = 0.25) survives and only a breach (9/32 = 0.28125) trips.
    """
    if len(history) < window:
        return False
    return statistics.median(history[-window:]) > ceiling


@dataclass
class RedoStats:
    """One dormancy check's outcome (one `[redo] iter` log line)."""

    dormant_fc1: int
    dormant_fc2: int
    hidden_dim: int
    trunk_dim: int
    recycled: int
    agree: float
    max_dlogit: float
    fc1_indices: list[int] = field(default_factory=list)
    fc2_indices: list[int] = field(default_factory=list)
    # Left tail of the dormancy-score distribution per layer (min, p5,
    # p10). Telemetry only — never read by the recycle path. This is the
    # number that says how far the population sits from any candidate
    # tau, which is what separates "tau fires by luck near init" from
    # "a real dormant tail exists on this architecture".
    fc1_tail: tuple[float, float, float] = (float("nan"),) * 3
    fc2_tail: tuple[float, float, float] = (float("nan"),) * 3


def dormancy_scores(h: torch.Tensor) -> torch.Tensor:
    """Per-unit dormancy scores of one layer per Sokar et al. Definition 1.

    `h`: (B, H) post-activation sample batch. Returns the (H,) score
    tensor s_i = E_x|h_i(x)| / mean_k E_x|h_k(x)|. A layer whose every
    unit is exactly silent (mean of means == 0) scores all-zero, which
    makes it entirely dormant at any tau >= 0.
    """
    if h.dim() != 2:
        raise ValueError(f"expected (B, H) activations, got {tuple(h.shape)}")
    per_unit = h.abs().mean(dim=0)  # E_x |h_i(x)|, shape (H,)
    denom = float(per_unit.mean())
    if denom == 0.0:
        return torch.zeros_like(per_unit)
    return per_unit / denom


def score_tail(scores: torch.Tensor) -> tuple[float, float, float]:
    """(min, p5, p10) of a layer's dormancy scores — telemetry only."""
    if scores.numel() == 0:
        return (float("nan"),) * 3
    s = scores.detach().to(torch.float32).flatten()
    q = torch.quantile(s, torch.tensor([0.05, 0.10], dtype=s.dtype))
    return float(s.min()), float(q[0]), float(q[1])


def dormant_indices(h: torch.Tensor, tau: float) -> torch.Tensor:
    """Dormant unit indices of one layer per Sokar et al. Definition 1.

    `h`: (B, H) post-activation sample batch. Returns a 1-D int64
    tensor of unit indices with s_i <= tau, where s_i is the mean
    absolute activation normalized by the LAYER MEAN of those means.
    A layer whose every unit is exactly silent (mean of means == 0)
    is entirely dormant.
    """
    score = dormancy_scores(h)
    return (score <= tau).nonzero(as_tuple=False).flatten().to(torch.int64)


@torch.no_grad()
def hidden_activations(
    net, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-SiLU activations of both hidden layers plus the actor logits.

    Mirrors `TilePolicyNetwork.forward_ac` exactly (Linear -> LayerNorm
    -> SiLU twice); re-implemented here so the intermediate h1/h2 are
    observable without hooks.
    """
    h1 = F.silu(net.norm1(net.fc1(x)))
    h2 = F.silu(net.norm2(net.fc2(h1)))
    logits = net.actor(h2)
    return h1, h2, logits


def _kaiming_uniform_rows(weight: torch.Tensor, rows: torch.Tensor) -> None:
    """Re-sample the given rows of a Linear weight Kaiming-uniform at the
    layer's fan-in — nn.Linear's own reset distribution
    (kaiming_uniform_ with a=sqrt(5) => U(-1/sqrt(fan_in), +1/sqrt(fan_in)))."""
    fan_in = weight.shape[1]
    fresh = torch.empty(
        int(rows.numel()), fan_in, dtype=weight.dtype, device=weight.device
    )
    nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
    weight[rows] = fresh


def _uniform_bias_entries(
    bias: torch.Tensor, idx: torch.Tensor, fan_in: int
) -> None:
    """Re-sample bias entries from nn.Linear's reset distribution:
    U(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    bound = 1.0 / math.sqrt(fan_in)
    fresh = torch.empty(int(idx.numel()), dtype=bias.dtype, device=bias.device)
    fresh.uniform_(-bound, bound)
    bias[idx] = fresh


def _zero_moment_slices(optimizer, param: torch.Tensor, dim: int,
                        idx: torch.Tensor) -> None:
    """Zero the Adam moments (`exp_avg`, `exp_avg_sq`) of one parameter
    at the rows (dim=0) or columns (dim=1) named by `idx`, in place via
    `index_fill_` — fancy getitem returns a COPY, so `m[idx].zero_()`
    would silently do nothing. Per-tensor `step` is deliberately
    untouched (B3.6). No-op when the optimizer holds no state for the
    param yet."""
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        moment = state.get(key)
        if moment is not None:
            moment.index_fill_(dim, idx.to(moment.device), 0.0)


@torch.no_grad()
def recycle(
    net,
    optimizer,
    fc1_idx: torch.Tensor,
    fc2_idx: torch.Tensor,
    reset_optimizer_moments: bool = True,
) -> None:
    """Recycle the given dormant hidden units in place (B3.5/B3.6).

    Order matters for exactness at the boundary: fc2's dormant incoming
    rows are re-sampled BEFORE fc1's outgoing columns are zeroed, so an
    entry at the intersection (fc2.weight[j, i] with i dormant in fc1
    and j dormant in fc2) ends exactly 0.0 — the outgoing-column zero
    is the final write on every such entry.
    """
    fc1_idx = fc1_idx.to(torch.int64)
    fc2_idx = fc2_idx.to(torch.int64)

    if fc2_idx.numel():
        # fc2 dormant units: incoming row + bias, LN affine, then their
        # outgoing columns in BOTH heads (actor AND critic — B3.5).
        _kaiming_uniform_rows(net.fc2.weight, fc2_idx)
        _uniform_bias_entries(net.fc2.bias, fc2_idx, net.fc2.weight.shape[1])
        net.norm2.weight[fc2_idx] = 1.0
        net.norm2.bias[fc2_idx] = 0.0
    if fc1_idx.numel():
        # fc1 dormant units: incoming row + bias, LN affine, outgoing
        # column in fc2. Column zeroing AFTER the fc2 row re-sample
        # above (see docstring).
        _kaiming_uniform_rows(net.fc1.weight, fc1_idx)
        _uniform_bias_entries(net.fc1.bias, fc1_idx, net.fc1.weight.shape[1])
        net.norm1.weight[fc1_idx] = 1.0
        net.norm1.bias[fc1_idx] = 0.0
        net.fc2.weight[:, fc1_idx] = 0.0
    if fc2_idx.numel():
        net.actor.weight[:, fc2_idx] = 0.0
        net.critic.weight[:, fc2_idx] = 0.0

    if not reset_optimizer_moments or optimizer is None:
        return
    # Zero the Adam moments for exactly the touched slices (B3.6).
    if fc1_idx.numel():
        _zero_moment_slices(optimizer, net.fc1.weight, 0, fc1_idx)
        _zero_moment_slices(optimizer, net.fc1.bias, 0, fc1_idx)
        _zero_moment_slices(optimizer, net.norm1.weight, 0, fc1_idx)
        _zero_moment_slices(optimizer, net.norm1.bias, 0, fc1_idx)
        _zero_moment_slices(optimizer, net.fc2.weight, 1, fc1_idx)
    if fc2_idx.numel():
        _zero_moment_slices(optimizer, net.fc2.weight, 0, fc2_idx)
        _zero_moment_slices(optimizer, net.fc2.bias, 0, fc2_idx)
        _zero_moment_slices(optimizer, net.norm2.weight, 0, fc2_idx)
        _zero_moment_slices(optimizer, net.norm2.bias, 0, fc2_idx)
        _zero_moment_slices(optimizer, net.actor.weight, 1, fc2_idx)
        _zero_moment_slices(optimizer, net.critic.weight, 1, fc2_idx)


@torch.no_grad()
def check_and_recycle(
    *,
    net,
    optimizer,
    sample_obs: torch.Tensor,
    tau: float,
    reset_optimizer_moments: bool = True,
) -> RedoStats:
    """One full dormancy check + recycle on a prepared sample batch.

    Measures the B3.7 identity diagnostics (greedy-argmax agreement and
    max|delta logit| pre/post recycle on the SAME batch) whenever at
    least one unit was recycled; a zero-recycle check is exact by
    construction (agree 1.0, max_dlogit 0.0) and skips the second
    forward pass.
    """
    h1, h2, logits_pre = hidden_activations(net, sample_obs)
    s1, s2 = dormancy_scores(h1), dormancy_scores(h2)
    fc1_tail, fc2_tail = score_tail(s1), score_tail(s2)
    fc1_idx = (s1 <= tau).nonzero(as_tuple=False).flatten().to(torch.int64)
    fc2_idx = (s2 <= tau).nonzero(as_tuple=False).flatten().to(torch.int64)
    n_recycled = int(fc1_idx.numel() + fc2_idx.numel())
    agree, max_dlogit = 1.0, 0.0
    if n_recycled:
        recycle(
            net, optimizer, fc1_idx, fc2_idx,
            reset_optimizer_moments=reset_optimizer_moments,
        )
        _, _, logits_post = hidden_activations(net, sample_obs)
        agree = float(
            (logits_pre.argmax(dim=-1) == logits_post.argmax(dim=-1))
            .float().mean()
        )
        max_dlogit = float((logits_post - logits_pre).abs().max())
    return RedoStats(
        dormant_fc1=int(fc1_idx.numel()),
        dormant_fc2=int(fc2_idx.numel()),
        hidden_dim=int(h1.shape[1]),
        trunk_dim=int(h2.shape[1]),
        recycled=n_recycled,
        agree=agree,
        max_dlogit=max_dlogit,
        fc1_indices=fc1_idx.tolist(),
        fc2_indices=fc2_idx.tolist(),
        fc1_tail=fc1_tail,
        fc2_tail=fc2_tail,
    )


def maybe_check_and_recycle(
    *,
    enabled: bool,
    net,
    optimizer,
    obs_all: torch.Tensor | None,
    valid_indices: np.ndarray,
    tau: float,
    sample_batch: int,
    check_every_iters: int,
    global_it: int,
    reset_optimizer_moments: bool = True,
) -> RedoStats | None:
    """Trainer-facing gate (the end-of-iteration hook, B3.3).

    Disabled or off-cadence calls return None BEFORE any sampling —
    they consume no numpy/torch RNG and touch nothing, which is what
    keeps a `redo_enabled: false` run byte-identical to a build
    without the mechanism. The sample batch is min(sample_batch, valid)
    rollout steps drawn uniformly without replacement from the
    just-collected rollout's valid (non-padded) rows (B3.4).
    """
    if not enabled:
        return None
    if check_every_iters > 1 and (global_it % check_every_iters) != 0:
        return None
    if obs_all is None:
        raise ValueError(
            "[redo] enabled but no rollout obs tensor is available — "
            "ReDo requires the feedforward tile-mode PPO path"
        )
    n = int(min(sample_batch, valid_indices.size))
    if n < 1:
        return None
    rows = np.random.choice(valid_indices, size=n, replace=False)
    sample = obs_all[torch.from_numpy(rows).to(obs_all.device)]
    return check_and_recycle(
        net=net,
        optimizer=optimizer,
        sample_obs=sample,
        tau=tau,
        reset_optimizer_moments=reset_optimizer_moments,
    )
