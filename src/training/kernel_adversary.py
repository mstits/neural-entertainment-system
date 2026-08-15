"""Kernel-matched binary adversary (`reinforce.adversary`, mode
``kernel_sticky``) — the PR-MDP machinery refit to the honest-eval kernel.

The v7 PR-MDP adversary picked arbitrary override actions from the full
action set on Bernoulli(alpha) steps and collapsed into a uniform noise
generator (entropy pinned at ln|A|). The honest eval protocol's only noise
is STICKY: with some probability the previous executed action repeats. This
adversary is matched to exactly that kernel:

* Two actions: PASS (protagonist's sample executes) or REPEAT (the previous
  EXECUTED action replaces it). It decides EVERY step — pass is a decision,
  so its behavior distribution covers the whole rollout, not an alpha draw.
* Reward: ``-protagonist_reward - budget_penalty * I(repeat)`` — zero-sum
  core plus a budget price per intervention, so "repeat always" is only
  optimal where repeats actually hurt the protagonist more than they cost.
* Training: negated-reward GAE on its own decision steps, epochs matched to
  the protagonist's (10:10, not the 10:2 imbalance that drove the v7 GDA
  limit cycle), entropy coefficient 0.01 against the ln 2 binary ceiling.

The adversary maintains the previous-executed-action vector itself (post
override), so the repeat kernel matches eval's sticky semantics even with
``sticky_action_prob: 0``. At an episode's first step the "previous
executed action" is whatever the tracker holds (0/no-op at process start)
— same permissive boundary the eval kernel has, not worth a special case.

Not persisted in the iter checkpoint (a resumed campaign restarts the
adversary fresh; the protagonist's robustness is the artifact that matters).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.training.ppo import batched_gae, ppo_losses

log = logging.getLogger(__name__)


def adversary_rewards(
    reward_buf: np.ndarray, repeat_buf: np.ndarray, budget_penalty: float
) -> np.ndarray:
    """Adversary reward stream: negated protagonist reward, minus the
    budget price on every step it chose REPEAT."""
    return (
        -reward_buf - float(budget_penalty) * repeat_buf.astype(np.float32)
    ).astype(np.float32)


class KernelStickyAdversary:
    """Pass/repeat adversary head co-trained against the vanilla-PPO
    protagonist on the shared rollout."""

    PASS = 0
    REPEAT = 1

    def __init__(
        self,
        *,
        net: torch.nn.Module,
        num_envs: int,
        rollout_steps: int,
        budget_penalty: float,
        entropy_coef: float,
        epochs: int,
        clip: float,
        lr: float,
        gamma: float,
        gae_lambda: float,
        value_coef: float,
        value_loss_kind: str,
        grad_clip: float,
        device: torch.device,
        preprocess_f16: bool,
        is_tile_mode: bool,
    ) -> None:
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=float(lr))
        self.budget_penalty = float(budget_penalty)
        self.entropy_coef = float(entropy_coef)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.value_loss_kind = str(value_loss_kind)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.preprocess_f16 = bool(preprocess_f16)
        self.is_tile_mode = bool(is_tile_mode)
        self.action_buf = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.repeat_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        self._value_steps: list[torch.Tensor] = []
        self._final_values_np = np.zeros(num_envs, dtype=np.float32)
        self._ent_sum = 0.0
        self._ent_n = 0

    # ----- rollout side --------------------------------------------------

    def begin_iter(self) -> None:
        self.repeat_buf[:] = False
        self._value_steps = []
        self._ent_sum = 0.0
        self._ent_n = 0

    def decide(
        self,
        t: int,
        batch_t: torch.Tensor,
        actions: torch.Tensor,
        log_probs_all: torch.Tensor,
        log_probs_taken: torch.Tensor,
        prev_exec_action: np.ndarray,
    ) -> None:
        """One rollout step: sample pass/repeat per env, apply REPEAT
        overrides in place, and advance the executed-action tracker.

        `actions` / `log_probs_taken` are the protagonist's CPU tensors;
        on REPEAT rows the executed action becomes the previous executed
        one and the recorded log-prob follows it (clamped — the house
        sticky pattern, or a near-deterministic protagonist NaNs the PPO
        ratio).
        """
        logits, values = self.net.forward_ac(batch_t)
        lp_all = F.log_softmax(logits.float().cpu(), dim=-1)
        probs = lp_all.exp()
        adv_actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # Decision-time policy entropy, against the ln 2 binary ceiling.
        self._ent_sum += float(-(probs * lp_all).sum(dim=-1).mean())
        self._ent_n += 1

        repeat_rows = np.nonzero(adv_actions.numpy() == self.REPEAT)[0]
        if repeat_rows.size:
            rows_t = torch.from_numpy(repeat_rows)
            prev_t = torch.from_numpy(prev_exec_action[repeat_rows])
            actions[rows_t] = prev_t
            log_probs_taken[rows_t] = torch.clamp(
                log_probs_all[rows_t, prev_t], min=-13.0
            )
            self.repeat_buf[t, repeat_rows] = True
        self.action_buf[t] = adv_actions.numpy()
        self.logp_buf[t] = torch.clamp(
            lp_all.gather(1, adv_actions.unsqueeze(1)).squeeze(1), min=-13.0
        ).numpy()
        self._value_steps.append(values)
        # The kernel repeats the previous EXECUTED action, so the tracker
        # advances on the post-override result every step.
        prev_exec_action[:] = actions.numpy()

    def drain_values(self) -> None:
        n = len(self._value_steps)
        if n:
            self.value_buf[:n] = (
                torch.stack(self._value_steps, dim=0).cpu().numpy()
            )

    def compute_final_values(self, final_batch_t: torch.Tensor) -> None:
        _, final_values = self.net.forward_ac(final_batch_t)
        self._final_values_np = final_values.detach().cpu().numpy()

    # ----- metrics --------------------------------------------------------

    def mean_entropy(self) -> float:
        return self._ent_sum / self._ent_n if self._ent_n else 0.0

    def repeat_fraction(self, valid_buf: np.ndarray) -> float:
        n_valid = int(valid_buf.sum())
        if n_valid == 0:
            return 0.0
        return float((self.repeat_buf & valid_buf).sum()) / n_valid

    # ----- update side ----------------------------------------------------

    def update(
        self,
        *,
        reward_buf: np.ndarray,
        done_buf: np.ndarray,
        valid_buf: np.ndarray,
        obs_all: torch.Tensor | None,
        obs_flat: np.ndarray,
        mb_size: int,
        trunc_buf: np.ndarray | None = None,
    ) -> dict:
        """PPO on the shared rollout with the adversary's own reward stream
        and critic, over every valid step (its whole behavior
        distribution). Mirrors the legacy PR-MDP update block, with the
        epochs/entropy matched per the spec."""
        stats = {
            "adversary_entropy": self.mean_entropy(),
            "adversary_repeat_frac": self.repeat_fraction(valid_buf),
            "adversary_policy_loss": 0.0,
        }
        adv_rew = adversary_rewards(
            reward_buf, self.repeat_buf, self.budget_penalty
        )
        advantages, targets = batched_gae(
            adv_rew, self.value_buf, done_buf, self._final_values_np,
            self.gamma, self.gae_lambda, trunc_buf=trunc_buf,
        )
        valid_rows = np.where(valid_buf.reshape(-1))[0]
        if valid_rows.size < 2:
            return stats
        flat_adv = advantages.reshape(-1)
        v_adv = flat_adv[valid_rows]
        a_mean = float(v_adv.mean())
        a_std = float(v_adv.std()) + 1e-8
        adv_all = torch.from_numpy(
            ((flat_adv - a_mean) / a_std).astype(np.float32)
        ).to(self.device)
        tgt_all = torch.from_numpy(
            targets.reshape(-1).astype(np.float32)
        ).to(self.device)
        act_all = torch.from_numpy(self.action_buf.reshape(-1)).to(self.device)
        lp_old_all = torch.from_numpy(self.logp_buf.reshape(-1)).to(self.device)
        last_pl = None
        for _ in range(self.epochs):
            perm = np.random.permutation(valid_rows)
            for mb0 in range(0, perm.shape[0], mb_size):
                mb = perm[mb0:mb0 + mb_size]
                if mb.size < 2:
                    continue
                mb_t = torch.from_numpy(mb).to(self.device)
                if obs_all is not None:
                    st = obs_all[mb_t]
                else:
                    st = torch.from_numpy(
                        np.ascontiguousarray(obs_flat[mb])
                    ).to(self.device).float()
                    if not self.preprocess_f16:
                        st = st.div_(255.0)
                lg, vp = self.net.forward_ac(st)
                l, pl, vl, en = ppo_losses(
                    lg, vp, act_all[mb_t], lp_old_all[mb_t],
                    adv_all[mb_t], tgt_all[mb_t],
                    clip_eps=self.clip,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                    value_loss_kind=self.value_loss_kind,
                )
                if not torch.isfinite(l):
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip
                )
                self.opt.step()
                last_pl = pl.detach()
        if last_pl is not None:
            stats["adversary_policy_loss"] = float(last_pl.item())
        return stats
