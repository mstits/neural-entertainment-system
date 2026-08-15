"""KL-anchored warm start + critic warmup for the vanilla-PPO conductor.

`reinforce.kl_anchor_checkpoint` names a solved-level policy checkpoint
(`net_state_dict`, tile MLP — widths shape-inferred). At startup the live
net receives the checkpoint's ACTOR weights (trunk + actor head; the critic
head keeps its fresh orthogonal init), and a frozen full copy is kept as the
prior. Two phases follow:

* While global env-steps < `actor_freeze_steps`, actor gradients are
  dropped before each optimizer step (critic-only warmup) — the loaded
  actor stays bit-identical while the fresh critic learns its values.
* After unfreeze, beta * KL(prior(.|s) || pi_theta(.|s)) is subtracted
  per-step from the reward stream (reward-level penalty), with beta
  decayed linearly `kl_beta_start` -> `kl_beta_end` over
  `kl_beta_decay_steps` env-steps.

D_KL(prior || pi) is exposed in metrics every iteration — the campaign's
kill criterion (KL > 0.15 sustained 2M steps) reads it from there.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# "actor set" = everything but the critic head, matching the SHAPO actor
# param set in ppo_updater (`not n.startswith("critic")`): the trunk is
# part of the loaded/frozen policy, not the fresh critic.
_CRITIC_PREFIX = "critic"


class KLAnchor:
    """Frozen prior + freeze/penalty schedule for one vanilla-PPO run."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        beta_start: float,
        beta_end: float,
        beta_decay_steps: float,
        actor_freeze_steps: float,
        num_actions: int,
        feature_dim: int,
        device: torch.device,
    ) -> None:
        from src.models.tile_policy import build_tile_policy_from_checkpoint

        payload = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        sd = payload
        if isinstance(payload, dict):
            sd = (
                payload.get("net_state_dict")
                or payload.get("state_dict")
                or payload
            )
        prior, is_recurrent = build_tile_policy_from_checkpoint(
            payload, num_actions=num_actions, feature_dim=feature_dim
        )
        if is_recurrent:
            raise ValueError(
                f"kl_anchor_checkpoint {checkpoint_path} holds a recurrent "
                f"policy — the KL anchor supports the stateless tile MLP only"
            )
        missing, unexpected = prior.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(
                f"kl_anchor_checkpoint {checkpoint_path} is missing "
                f"{len(missing)} parameter(s): {missing} — refusing a "
                f"partial prior"
            )
        if unexpected:
            log.warning(
                "[kl_anchor] unexpected keys in %s ignored: %s",
                checkpoint_path, unexpected,
            )
        prior.to(device)
        prior.eval()
        prior.requires_grad_(False)
        self.prior = prior
        self.checkpoint_path = str(checkpoint_path)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_decay_steps = float(beta_decay_steps)
        self.actor_freeze_steps = float(actor_freeze_steps)
        # Live schedule state, refreshed once per iter via set_env_steps.
        self.beta = self.beta_start
        self.frozen = self.actor_freeze_steps > 0

    # ----- startup -----------------------------------------------------

    def load_actor_into(self, net: torch.nn.Module) -> list[str]:
        """Copy the prior's actor set (all non-critic weights) into `net`.

        The critic head keeps `net`'s own fresh orthogonal init. Raises on
        any shape mismatch — a half-loaded actor is worse than a loud stop.
        Returns the loaded key list.
        """
        live_sd = net.state_dict()
        actor_sd = {}
        for k, v in self.prior.state_dict().items():
            if k.startswith(_CRITIC_PREFIX):
                continue
            if k not in live_sd or live_sd[k].shape != v.shape:
                raise ValueError(
                    f"kl_anchor actor weight {k!r} does not fit the live net "
                    f"(checkpoint {tuple(v.shape)} vs live "
                    f"{tuple(live_sd[k].shape) if k in live_sd else 'absent'}) "
                    f"— set tile_hidden_dim/tile_trunk_dim to match "
                    f"{self.checkpoint_path}"
                )
            actor_sd[k] = v
        net.load_state_dict(actor_sd, strict=False)
        return sorted(actor_sd)

    # ----- schedule ----------------------------------------------------

    def set_env_steps(self, env_steps: float) -> None:
        """Refresh beta + the freeze flag from the global env-step count."""
        frac = min(1.0, max(0.0, float(env_steps)) / max(1.0, self.beta_decay_steps))
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)
        self.frozen = float(env_steps) < self.actor_freeze_steps

    # ----- per-step penalty ---------------------------------------------

    def kl_divergence(
        self, obs_t: torch.Tensor, policy_log_probs: torch.Tensor
    ) -> np.ndarray:
        """Per-row D_KL(prior(.|s) || pi(.|s)) as float32 (N,).

        `policy_log_probs` is the CPU log-softmax the rollout already
        computed for sampling; only the prior forward is added here.
        """
        with torch.no_grad():
            prior_logits, _ = self.prior.forward_ac(obs_t)
            prior_logp = F.log_softmax(prior_logits.float().cpu(), dim=-1)
            kl = (
                prior_logp.exp() * (prior_logp - policy_log_probs)
            ).sum(dim=-1)
        return kl.numpy().astype(np.float32)

    # ----- freeze phase --------------------------------------------------

    def zero_actor_grads(self, net: torch.nn.Module) -> None:
        """Drop the actor set's gradients so the optimizer step is
        critic-only (Adam skips params whose grad is None — no moment
        update, no step-count bump: the actor stays bit-identical)."""
        for name, p in net.named_parameters():
            if not name.startswith(_CRITIC_PREFIX) and p.grad is not None:
                p.grad = None
