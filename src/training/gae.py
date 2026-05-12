"""Generalized Advantage Estimation (GAE-λ).

Extracted from `trainer._reinforce_update` so the recurrence can be
unit-tested directly (the truncation-vs-natural-termination branch,
in particular, is correctness-critical and historically broken).
The function is pure: device routing, gamma, and the max-episode-
steps cutoff are arguments rather than `self.*` lookups.
"""

from __future__ import annotations

import numpy as np
import torch


def gae(
    rewards: torch.Tensor,
    values_pred: torch.Tensor,
    traj_len_full: int,
    max_episode_steps: int,
    gamma: float,
    gae_lambda: float = 0.95,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, value_targets) for a single trajectory.

    `rewards` and `values_pred` must be 1-D tensors of equal length T,
    on the same device. `traj_len_full` is the full pre-windowing
    trajectory length used to decide whether the final step was a
    natural termination (death, level-clear: bootstrap with 0) or a
    truncation by `max_episode_steps` (bootstrap with V(s_T)).

    Returned tensors land on `values_pred.device`. With `normalize=True`
    (default), advantages are mean/std normalized — standard PPO.

    The recurrence runs CPU-side on a numpy view to avoid an MPSGraph
    leak: per-step `advantages[t] = python_scalar` on an MPS tensor
    spawns a fresh cached graph PER CALL (~5 MB each) and the index
    rotates the cache key out. Measured at >150 MB/s in 16-worker
    training before the workaround. Computing the running sum on
    numpy and pulling a single `torch.from_numpy().to(device)` at the
    end keeps the MPS-side tensor alloc to one per trajectory.
    """
    device = values_pred.device
    with torch.no_grad():
        T = rewards.numel()
        # Next-state values: shift values_pred by one and fill the
        # last slot based on how the trajectory ended (see below).
        values_next = torch.empty_like(values_pred)
        if T > 1:
            values_next[:-1] = values_pred[1:]
        terminated_naturally = traj_len_full < max_episode_steps
        if terminated_naturally:
            # V(s_{T+1}) = 0 — no future after death/level-clear.
            # The previous unconditional bootstrap muted the death
            # signal ~100× at γ=0.99 (true Δ_T = r_T − V(s_T),
            # bootstrap gave r_T + (γ−1)·V(s_T)).
            values_next[-1] = 0.0
        else:
            # Truncation by max_episode_steps — bootstrap with the
            # critic's own estimate of "what comes next."
            values_next[-1] = values_pred[-1]

        deltas = rewards + gamma * values_next - values_pred
        deltas_np = deltas.detach().cpu().numpy()
        advantages_np = np.empty_like(deltas_np)
        running_adv = 0.0
        for t in range(T - 1, -1, -1):
            running_adv = float(deltas_np[t]) + gamma * gae_lambda * running_adv
            advantages_np[t] = running_adv
        advantages = torch.from_numpy(advantages_np).to(device)

        # V-targets: discounted-return estimate the critic learns toward.
        value_targets = advantages + values_pred.detach()

        if normalize and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return advantages, value_targets
