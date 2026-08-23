"""Semi-MDP GAE over commitment decisions — the v22 mechanics, testable.

Every formula here is the one v22 grounded in the action-repetition /
HRL literature, implemented as pure functions over plain tensors so each
property can be pinned by a unit test before any trainer wiring:

  delta_t = sum_{i<k} gamma^i r_{t+i} + gamma^k V(s_{t+k}) - V(s_t)
  A_t     = delta_t + (gamma * lambda)^{k_t} A_{t+k_t}

Lambda is exponentiated by the duration. Applied per-decision instead,
long options would stretch the estimator's effective horizon and bias
the advantage — the exact class of quiet correctness bug that voided the
first Phase-3 run.

Truncation semantics: a macro interrupted at k' < k (death or timeout)
contributes its realized discounted reward over k' steps; a DEATH cut
bootstraps 0, a TIMEOUT cut bootstraps V(s_cut). The decision's identity
remains the INTENDED pair — relabeling to the realized duration would
penalize a choice the network never made.

Entropy: `scale_entropy_by_duration` multiplies each decision's entropy
by its k, equalizing the per-env-step footprint. Without it a k=1 policy
farms 4x the bonus of a k=4 policy by deciding 4x as often, and the
optimizer itself collapses the policy back to per-step decisions.
"""
from __future__ import annotations

import torch


def discounted_partial_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """sum_i gamma^i r_i over one commitment's realized steps. (k',) -> scalar."""
    k = rewards.shape[0]
    if k == 0:
        return torch.tensor(0.0)
    w = gamma ** torch.arange(k, dtype=torch.float32)
    return (w * rewards.float()).sum()


def smdp_deltas(seg_returns: torch.Tensor, seg_k: torch.Tensor,
                values: torch.Tensor, boot_values: torch.Tensor,
                boot_mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-decision TD errors.

    seg_returns: (N,) discounted in-commitment reward sums
    seg_k:       (N,) REALIZED durations k' (env steps actually run)
    values:      (N,) V at each decision state
    boot_values: (N,) V at each commitment's end state
    boot_mask:   (N,) 1.0 where bootstrapping is allowed (not a death) —
                 a timeout cut keeps 1.0 with boot_values at the cut
                 state; a death forces the bootstrap term to zero.
    """
    return (seg_returns
            + (gamma ** seg_k.float()) * boot_mask * boot_values
            - values)


def smdp_gae(deltas: torch.Tensor, seg_k: torch.Tensor, done: torch.Tensor,
             gamma: float, lam: float) -> torch.Tensor:
    """A_t = delta_t + (gamma*lam)^{k_t} * A_{t+1}, cut at episode ends.

    Decisions are ordered; `done[i]` marks the last decision of an
    episode, after which the recursion restarts. The decay uses each
    decision's own realized duration, so a k=4 commitment discounts the
    future advantage as 4 env steps, not 1 decision.
    """
    n = deltas.shape[0]
    adv = torch.zeros_like(deltas)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if bool(done[i]):
            running = 0.0
        decay = (gamma * lam) ** float(seg_k[i])
        running = float(deltas[i]) + decay * running
        adv[i] = running
    return adv


def scale_entropy_by_duration(entropy: torch.Tensor,
                              seg_k: torch.Tensor) -> torch.Tensor:
    """Per-decision entropy * k = per-env-step entropy footprint.

    Uses the INTENDED duration, matching the decision the bonus
    regularizes; a truncated commitment still chose k.
    """
    return entropy * seg_k.float()
