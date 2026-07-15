"""Random Network Distillation (RND) for intrinsic motivation.

The standard sparse-reward RL trick from Burda et al. (2018). Two CNNs:

* A **target** network with random, frozen weights. It maps observations
  to a fixed feature embedding.
* A **predictor** network with the same architecture, trained online to
  match the target's output via MSE.

Per-state prediction error becomes the intrinsic reward. States the
predictor has seen many times are easy to mimic → low error → low bonus.
Novel states the predictor hasn't seen → high error → high bonus,
which pulls the policy toward exploration.

Two normalization layers wrapping the raw RND mechanic, both from the
paper and both critical for stable training:

* **Observation normalization** — running mean/std over the raw obs
  pixels. Both target and predictor see `(obs - μ) / σ` clipped to
  ±5 so prediction error doesn't depend on absolute pixel intensity.
* **Intrinsic-reward normalization** — running std of the bonus,
  used to divide the per-state error before returning. Keeps the
  bonus's scale stable as the predictor improves.

In this codebase RND opt-in lives at `reinforce.rnd_intrinsic_coef` (the
weight applied to the bonus when summing into the extrinsic reward) and
`reinforce.rnd_loss_coef` (the weight on the predictor MSE in the PPO
update's total loss). Set the intrinsic coef to 0 to fully disable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RunningMeanStd(nn.Module):
    """Welford-style running mean / variance over arbitrary tensor
    shapes. Stats are stored as buffers so they roundtrip through
    `state_dict()` and survive checkpoint save/load.

    `update(x)` accepts a batch of values; the running stats are
    updated incrementally without ever materializing the full history.
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        # Init variance at 1 so the first normalize call doesn't divide
        # by zero. Once a batch arrives the actual stats kick in.
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor) -> None:
        """`x`: `(N, *shape)` — flatten leading dims into the batch
        before calling. Must be detached; we never want stats to track
        the gradient graph."""
        with torch.no_grad():
            x = x.detach()
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.size(0)

            tot = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * (batch_count / tot)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            new_var = (
                m_a + m_b + delta.pow(2) * self.count * batch_count / tot
            ) / tot

            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.fill_(tot.item())

    @property
    def std(self) -> torch.Tensor:
        # Floor at sqrt(eps) so divides stay well-behaved even before
        # the first update arrives.
        return (self.var + 1e-8).sqrt()


class _RNDEncoder(nn.Module):
    """Compact CNN matching the Nature-DQN encoder shape but with a
    smaller projection at the end. Both target and predictor use this
    architecture; the target's weights are frozen at random init."""

    def __init__(self, in_channels: int = 4, feat_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 84 → 20 → 9 → 7 ; 64 * 7 * 7 = 3136
        self.proj = nn.Linear(3136, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        return self.proj(x)


class RND(nn.Module):
    """Wraps a frozen target + a trainable predictor.

    Usage in the PPO update:
        per_sample_err = rnd(obs)            # shape (B,) — squared error per obs
        intrinsic = per_sample_err.detach()  # bonus signal, no gradient flow
        loss_rnd = per_sample_err.mean()     # train predictor

    `forward` returns the per-sample error so the caller can decide which
    reduction to use for reward vs. loss. Predictor gradients only — the
    target's `requires_grad` is False from init.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_dim: int = 512,
        obs_size: int = 84,
        obs_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.target = _RNDEncoder(in_channels=in_channels, feat_dim=feat_dim)
        self.predictor = _RNDEncoder(in_channels=in_channels, feat_dim=feat_dim)
        # Freeze the target. The whole point of RND is that the
        # prediction target is fixed; if it drifted, "novelty" would be
        # circular. Disable gradient tracking AND switch to eval mode
        # so any future BatchNorm-style buffers don't get updated.
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()
        self.obs_clip = float(obs_clip)
        # Per-pixel obs running stats — normalize each pixel independently.
        # Both target and predictor see normalized inputs, so their
        # representations stay scale-invariant as training progresses.
        self.obs_rms = _RunningMeanStd(shape=(in_channels, obs_size, obs_size))
        # Scalar running stats over the per-sample intrinsic error.
        # Used to divide the bonus before returning, keeping its scale
        # roughly unit-variance regardless of predictor accuracy.
        self.reward_rms = _RunningMeanStd(shape=())

    def train(self, mode: bool = True) -> "RND":
        """Override so the target stays in eval mode even when the
        outer module is set to train. Predictor follows the requested
        mode normally."""
        super().train(mode)
        self.target.eval()
        return self

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Subtract running per-pixel mean, divide by std, clip to
        ±obs_clip. The clip prevents single rare frames (e.g. game-over
        flash, all-white screens) from blowing the predictor's MSE
        scale before the running stats catch up."""
        norm = (obs - self.obs_rms.mean) / self.obs_rms.std
        return norm.clamp(-self.obs_clip, self.obs_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the **raw** per-sample MSE, shape `(B,)`.

        obs is normalized (running mean/std, clipped) before going into
        both target and predictor; the returned value is the raw squared
        prediction error. The caller reduces it with `.mean()` for the
        predictor loss and passes it to `normalize_bonus()` for the
        intrinsic-reward signal. Keeping the bonus normalization OUT of
        forward() is deliberate: `update_normalization()` must be fed the
        raw error, otherwise `reward_rms` would track `err / std` — a
        self-referential loop that mis-scales the bonus.
        """
        normalized = self._normalize_obs(obs)
        with torch.no_grad():
            target_feat = self.target(normalized)
        pred_feat = self.predictor(normalized)
        return ((pred_feat - target_feat) ** 2).mean(dim=-1)

    def target_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Frozen-target embedding `target(normalize_obs(obs))`, no grad.

        The constant half of `forward`'s per-sample MSE. The target is
        frozen and, for the whole of a K-epoch PPO update, `obs_rms` is
        held fixed — so this embedding is identical every epoch for a
        given observation. The trainer computes it once per iteration and
        reuses it across the minibatch loop (via `predictor_loss`) instead
        of re-running the target CNN K times per observation.
        """
        with torch.no_grad():
            return self.target(self._normalize_obs(obs))

    def predictor_loss(
        self, obs: torch.Tensor, target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample MSE, shape `(B,)`, against a precomputed target
        embedding. Equivalent to `forward(obs)` but takes the (cached)
        frozen target features instead of recomputing them; the predictor
        half still forwards with grad. `target_feat` MUST be
        `target_features(obs)` under the SAME `obs_rms` — valid only while
        `obs_rms` is unchanged (the vanilla_ppo update loop). Paths that
        mutate `obs_rms` mid-update (the GA `_reinforce_update`) must keep
        using `forward`.
        """
        pred_feat = self.predictor(self._normalize_obs(obs))
        return ((pred_feat - target_feat) ** 2).mean(dim=-1)

    def normalize_bonus(self, per_sample_err: torch.Tensor) -> torch.Tensor:
        """Scale a raw per-sample error into the intrinsic-reward bonus.

        Divides by the running std of the bonus so its scale stays
        stable as the predictor improves. Detached — this is reward, not
        a gradient path. Call BEFORE `update_normalization()` so the
        divisor reflects prior batches, not the current one.
        """
        return (per_sample_err / self.reward_rms.std).detach()

    def update_normalization(
        self,
        obs: torch.Tensor,
        intrinsic_err: torch.Tensor,
    ) -> None:
        """Update running stats with the latest batch.

        Trainer should call this once per gradient step with the
        unnormalized obs and the **raw** per-sample error returned from
        `forward()` (not the normalized bonus). Cheap (Welford
        increment); no perf concern.
        """
        # Flatten the leading time/batch dims so RunningMeanStd sees
        # `(N, *obs_shape)`.
        obs_flat = obs.reshape(-1, *obs.shape[1:])
        self.obs_rms.update(obs_flat)
        self.reward_rms.update(intrinsic_err.reshape(-1))

    @property
    def feat_dim(self) -> int:
        """Width of the target/predictor embedding — the row size of the
        per-iter target-feature cache the update loop builds."""
        return self.target.proj.out_features

    @property
    def num_params(self) -> int:
        """Trainable predictor params only (target is frozen)."""
        return sum(p.numel() for p in self.predictor.parameters())
