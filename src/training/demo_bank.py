"""Fixed demonstration bank for demo-anchored PPO (DQfD-style).

A device-resident, immutable store of (obs, action) pairs from winning
demo trajectories. The vanilla-PPO minibatch loop draws a demo minibatch
alongside every on-policy minibatch and adds a supervised anchor loss
(see `src.training.ppo.demo_anchor_loss`), keeping the policy on the
demo manifold through a hard seam while the reward gradient sculpts the
parts behavior cloning alone cannot fit. The bank never receives policy
rollouts — self-collected data stays in the PPO buffers.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class DemoBank:
    """Immutable bank of demo (obs float32 (M, D), act int64 (M,))."""

    def __init__(self, obs: torch.Tensor, acts: torch.Tensor, device) -> None:
        self.obs = obs.to(device)
        self.acts = acts.to(device)
        self.n = int(obs.shape[0])

    @classmethod
    def from_npz(cls, paths: Sequence[str], feature_dim: int,
                 num_actions: int, device) -> "DemoBank":
        """Load per-demo npz banks (keys obs_i (Li, D) int8 / act_i (Li,)).

        Guards (all bitten in practice): obs width must equal the net's
        input dim; actions must fit the action table; and each demo's obs
        rows must be substantially unique — a collector that stores views
        into a reused feature buffer produces a bank of identical rows,
        which is unfittable by construction (one input, many labels).
        """
        O, A = [], []
        for p in paths:
            d = np.load(str(p))
            for k in sorted(k for k in d.files if k.startswith("obs_")):
                j = k.split("_", 1)[1]
                o, a = d[k], d[f"act_{j}"]
                assert o.ndim == 2 and o.shape[1] == feature_dim, \
                    f"{p}:{k} shape {o.shape} != (*, {feature_dim})"
                assert int(a.max()) < num_actions, \
                    f"{p}:act_{j} max {a.max()} >= {num_actions}"
                uniq = len(np.unique(o, axis=0))
                assert uniq > 0.5 * len(o), \
                    f"{p}:{k} only {uniq}/{len(o)} unique obs rows — " \
                    "buffer-aliased collection (store copies, not views)"
                O.append(o)
                A.append(a)
        assert O, f"no demos found in {list(paths)}"
        obs = torch.from_numpy(np.concatenate(O)).float()
        acts = torch.from_numpy(np.concatenate(A).astype(np.int64))
        return cls(obs, acts, device)

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.n, (batch,), device=self.obs.device)
        return self.obs[idx], self.acts[idx]
