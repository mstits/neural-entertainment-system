"""Pure building blocks for the vanilla_ppo trainer loop.

Extracted from `Trainer._run_vanilla_ppo` (a ~600-line method) so the
two correctness-critical, side-effect-free pieces can be unit-tested
in isolation and reused when Phase 1 wires in intrinsic exploration:

* `batched_gae` — the (rollout_steps × num_envs) GAE-λ backward sweep
  with per-step done masking. Distinct from `gae.gae`, which is a
  single-trajectory estimator that decides natural-termination vs.
  truncation once at the end. The batched form handles MULTIPLE
  episode boundaries inside one env's rollout — required now that
  auto-reset-on-death produces several dones per env per rollout.

* `ppo_losses` — the PPO clipped-surrogate + value + entropy loss
  for one minibatch. The total loss is assembled here so the Phase 1
  RND intrinsic-loss term has exactly one place to plug in.

Neither function touches `self`, the optimizer, the pool, or any I/O.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def batched_gae(
    reward_buf: np.ndarray,
    value_buf: np.ndarray,
    done_buf: np.ndarray,
    final_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE-λ backward sweep over a (T, N) rollout buffer.

    Args:
        reward_buf:   (T, N) float32 per-step rewards.
        value_buf:    (T, N) float32 critic values V(s_t).
        done_buf:     (T, N) bool — True where step t was terminal.
        final_values: (N,) float32 bootstrap V(s_T) for the step after
                      the last rollout step.
        gamma, gae_lambda: PPO discount and GAE trace-decay.

    Returns `(advantages, value_targets)`, both (T, N) float32 and
    UN-normalized. `value_targets = advantages + value_buf` (the
    critic regression target). The caller normalizes advantages for
    the policy loss separately so value_targets stay on the reward
    scale.

    A done at step t zeroes both the bootstrapped next-value AND the
    running GAE accumulator for that env, breaking the episode
    boundary so advantage never leaks across a death/level-clear.
    """
    rollout_steps, num_envs = reward_buf.shape
    advantages = np.zeros_like(reward_buf)
    gae_running = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            next_value = final_values
        else:
            next_value = value_buf[t + 1]
        not_done = (~done_buf[t]).astype(np.float32)
        delta = reward_buf[t] + gamma * next_value * not_done - value_buf[t]
        gae_running = delta + gamma * gae_lambda * gae_running * not_done
        advantages[t] = gae_running
    value_targets = advantages + value_buf
    return advantages, value_targets


def ppo_losses(
    logits: torch.Tensor,
    values_pred: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    value_loss_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped-surrogate + value + entropy loss for one minibatch.

    Args:
        logits:        (B, A) policy logits from the current network.
        values_pred:   (B,) critic predictions for the minibatch.
        actions:       (B,) int64 action indices that were executed.
        log_probs_old: (B,) log-probs under the behavior policy.
        advantages:    (B,) normalized advantages.
        value_targets: (B,) critic regression targets.
        clip_eps:      PPO clip range (falls back to 0.2 if <= 0).
        value_coef:    value-loss weight.
        entropy_coef:  entropy-bonus weight (subtracted from the loss).
        value_loss_kind: "mse" for squared error, else Huber (smooth_l1).

    Returns `(loss, policy_loss, value_loss, entropy)`. Only `loss`
    carries grad for `.backward()`; the other three are reported as
    metrics. No optimizer step happens here.
    """
    log_probs_all = F.log_softmax(logits.float(), dim=-1)
    log_probs_new = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    values_pred = values_pred.float()

    # PPO clipped surrogate.
    ratio = torch.exp(log_probs_new - log_probs_old)
    clip = clip_eps if clip_eps > 0 else 0.2
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    policy_obj = torch.min(ratio * advantages, clipped * advantages)
    policy_loss = -policy_obj.mean()

    # Entropy bonus.
    probs = log_probs_all.exp()
    entropy = -(probs * log_probs_all).sum(dim=-1).mean()

    # Value loss.
    if value_loss_kind == "mse":
        value_loss = F.mse_loss(values_pred, value_targets)
    else:
        value_loss = F.smooth_l1_loss(values_pred, value_targets)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, policy_loss, value_loss, entropy
