"""Actor-critic networks for the Dreamer trainer.

Distinct from `policy_network.PolicyNetwork`:

* The Dreamer policy operates on the **world-model latent** (deter +
  stoch), not on raw observations. The encoder lives in the world model,
  not here.
* The critic uses a separate target network for stability (standard
  Q-learning trick — without it the bootstrap value chases its own
  prediction and diverges).
* Actor optimization uses lambda-returns from imagined rollouts plus an
  entropy bonus; the gradient flows back through the imagined latent
  trajectory via the RSSM's reparameterized stochastic latents.

Both heads are small MLPs — the heavy representation work is done by
the world model. Default sizes match the world model's `hidden_dim`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DreamerActor(nn.Module):
    """Categorical policy over the discrete action space, conditioned
    on `(deter, stoch)`."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape `(..., num_actions)`."""
        return self.net(latent)

    def sample(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a one-hot action with straight-through gradient.

        Returns `(action_one_hot, log_prob_chosen, entropy)`:

        * `action_one_hot` — discrete forward, straight-through backward.
          Lets gradients flow through the world model's action input
          during imagined-rollout backprop.
        * `log_prob_chosen` — log p(action | latent) at the sampled
          index. Used for the policy gradient surrogate.
        * `entropy` — analytic Shannon entropy of the categorical
          distribution `H = -Σ p log p`. The PPO/Dreamer actor
          maximizes this for exploration; using the analytic form
          (instead of approximating with `-log p(a)`) gives a much
          stronger gradient when the policy is near-uniform and
          a near-zero gradient once it commits, which is exactly
          the desired exploration-vs-exploitation pressure.
        """
        logits = self.forward(latent)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Sample categorical: forward sees discrete one-hot, backward
        # passes through `probs` via the straight-through residual.
        idx = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1,
        ).view(*probs.shape[:-1])
        one_hot = F.one_hot(idx, num_classes=probs.size(-1)).float()
        action = one_hot + (probs - probs.detach())
        log_prob_chosen = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        # Analytic Shannon entropy of the categorical distribution.
        # Negative-sum-over-classes of p * log p.
        entropy = -(probs * log_probs).sum(dim=-1)
        return action, log_prob_chosen, entropy


class DreamerCritic(nn.Module):
    """Value head with a slow-tracking target network.

    Target update is a simple Polyak EMA (`update_target(tau)` blends
    target = tau * online + (1 - tau) * target).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.online = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Target = exact copy at init; updated via EMA during training.
        # `requires_grad=False` so gradients never leak through the
        # bootstrap path.
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Online value prediction. Use this for the critic's regression
        loss against lambda-returns."""
        return self.online(latent).squeeze(-1)

    def target_value(self, latent: torch.Tensor) -> torch.Tensor:
        """Bootstrap value from the slow target network. Always called
        with no_grad in the trainer's hot path."""
        with torch.no_grad():
            return self.target(latent).squeeze(-1)

    def update_target(self, tau: float = 0.02) -> None:
        """Polyak EMA: target ← (1 - tau) * target + tau * online.

        The standard Dreamer setting is τ ≈ 0.02 (i.e. ~50-step half-
        life, slow enough that the bootstrap is stable but fast enough
        to track real value improvements within a training run).
        """
        with torch.no_grad():
            for p_online, p_target in zip(
                self.online.parameters(), self.target.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)


def lambda_returns(
    rewards: torch.Tensor,    # (B, H)
    values: torch.Tensor,     # (B, H)
    bootstrap: torch.Tensor,  # (B,) — value at H
    continues: torch.Tensor,  # (B, H) ∈ [0, 1]
    discount: float = 0.99,
    lambd: float = 0.95,
) -> torch.Tensor:
    """λ-returns over an imagined rollout.

    `continues[t]` ∈ [0, 1] gates discounting (predicted by the
    continue-head — when the world model thinks the episode ends, the
    return collapses to the immediate reward). `discount` is multiplied
    inside for the standard `r + γ * c * V'` recurrence.

    Returns a tensor shaped `(B, H)` of lambda-targets.
    """
    H = rewards.size(1)
    next_values = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    # Standard λ-returns recursion: Δ = r + γc * ((1-λ) V' + λ G_{t+1}).
    # Compute backwards in time so `running` accumulates correctly.
    targets = torch.zeros_like(rewards)
    running = bootstrap
    for t in reversed(range(H)):
        running = (
            rewards[:, t]
            + discount * continues[:, t] * (
                (1.0 - lambd) * next_values[:, t] + lambd * running
            )
        )
        targets[:, t] = running
    return targets
