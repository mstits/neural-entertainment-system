"""Random Network Distillation for tile-feature observations.

Companion to `src/models/rnd.py` (which uses a CNN encoder for pixel
obs). This module's encoder is a small MLP sized for the 175-feature
SMB tile vector. The `TileRND` class implements the same protocol as
`RND` — `forward(obs)` returns a per-sample MSE bonus,
`update_normalization(obs, err)` pushes Welford stats — so the trainer
dispatches transparently based on `_is_tile_mode`.

Why bother with a separate class:

* The `_RNDEncoder` in `rnd.py` is a 3-layer Conv2d stack expecting a
  `(B, 4, 84, 84)` input. Tile features arrive as `(B, 175)` — a Linear
  stack is the right shape.
* Tile features are int8 in [-128, 127] (signed enemy markers, signed
  velocities). The pixel RND normalizes to [0, 255]-ish after the
  running-stat divide. Different scale, different clip range.
* The whole point of tile mode is "small everything" — mirroring the
  pixel RND's 1.5M-param encoder would make RND ~10x bigger than the
  policy itself. The MLP version is ~30k params total.

Why RND matters for tile mode specifically:

The tile mode trainer's symptom across multiple runs is "PPO has near-
zero policy_loss because advantages collapse once the value head fits
the policy's typical return." That collapse happens even faster on
tile mode because the value head (one Linear from 32-dim trunk) is
trivial to fit. RND's intrinsic-reward bonus keeps advantages alive
when extrinsic reward stalls — the agent sees novel states (further
into 1-1, past the staircase) and the predictor's high error gives
PPO real gradient signal even when no flag has been touched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing running-stats module; it works on any tensor shape.
from src.models.rnd import _RunningMeanStd


class _TileRNDEncoder(nn.Module):
    """Small MLP that produces a fixed-size embedding for an arbitrary
    tile-feature vector. Same shape on target and predictor; target is
    frozen, predictor learns to mimic.

    Default hidden widths chosen so total params (target + predictor)
    are roughly comparable to the policy network — neither dwarfs the
    other. At feat_dim=64 + hidden=128 + input=175: ~30k params per
    encoder, ~60k for both.
    """

    def __init__(
        self,
        in_features: int = 175,
        feat_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TileRND(nn.Module):
    """RND for tile-feature observations. Drop-in for the trainer's
    multi-genome `_rnd` slot when `_is_tile_mode` is True.

    Same external surface as `RND`:

        rnd = TileRND(feature_dim=175)
        per_sample_err = rnd(obs)            # (B,) intrinsic-reward bonus
        rnd.update_normalization(obs, err)   # push Welford stats
    """

    def __init__(
        self,
        feature_dim: int = 175,
        feat_dim: int = 64,
        obs_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.obs_clip = float(obs_clip)
        self.target = _TileRNDEncoder(in_features=feature_dim, feat_dim=feat_dim)
        self.predictor = _TileRNDEncoder(in_features=feature_dim, feat_dim=feat_dim)
        # Freeze target. RND requires a fixed reference; if the target
        # drifted the "novelty" signal would be circular.
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()
        # Per-feature observation stats. Tile feature scale varies wildly
        # (tile bytes are 0/1/-1, velocities are -127..127, lives is
        # 0..N) so per-feature standardization keeps the predictor's
        # MSE on a comparable scale across input dimensions.
        self.obs_rms = _RunningMeanStd(shape=(feature_dim,))
        # Scalar reward stats — divides the per-sample error before
        # returning. Same role as the pixel RND.
        self.reward_rms = _RunningMeanStd(shape=())

    def train(self, mode: bool = True) -> "TileRND":
        """Override so the target stays in eval mode even when the
        outer module is set to train. Same pattern as `RND.train`."""
        super().train(mode)
        self.target.eval()
        return self

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        norm = (obs - self.obs_rms.mean) / self.obs_rms.std
        return norm.clamp(-self.obs_clip, self.obs_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the **raw** per-sample squared MSE, shape `(B,)`.

        Mirrors `RND.forward`: the bonus normalization is applied
        separately via `normalize_bonus()` so `update_normalization()`
        receives the raw error rather than a self-referential
        `err / std`.
        """
        normalized = self._normalize_obs(obs)
        with torch.no_grad():
            target_feat = self.target(normalized)
        pred_feat = self.predictor(normalized)
        return ((pred_feat - target_feat) ** 2).mean(dim=-1)

    def target_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Frozen-target embedding `target(normalize_obs(obs))`, no grad.

        Same contract as `RND.target_features`: constant across the
        K-epoch PPO update while `obs_rms` is held fixed, so the trainer
        computes it once per iteration and reuses it via
        `predictor_loss` instead of re-running the target MLP per
        minibatch per epoch.
        """
        with torch.no_grad():
            return self.target(self._normalize_obs(obs))

    def predictor_loss(
        self, obs: torch.Tensor, target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample MSE, shape `(B,)`, against a precomputed target
        embedding. Same contract as `RND.predictor_loss`: `target_feat`
        MUST be `target_features(obs)` under the SAME `obs_rms` — valid
        only while `obs_rms` is unchanged (the vanilla_ppo update loop);
        paths that mutate `obs_rms` mid-update keep using `forward`.
        """
        pred_feat = self.predictor(self._normalize_obs(obs))
        return ((pred_feat - target_feat) ** 2).mean(dim=-1)

    @property
    def feat_dim(self) -> int:
        """Width of the target/predictor embedding — the row size of the
        per-iter target-feature cache the update loop builds."""
        return self.target.net[-1].out_features

    def normalize_bonus(self, per_sample_err: torch.Tensor) -> torch.Tensor:
        """Scale a raw per-sample error into the intrinsic-reward bonus
        by the running std of the bonus. Detached. Call BEFORE
        `update_normalization()`."""
        return (per_sample_err / self.reward_rms.std).detach()

    def update_normalization(
        self,
        obs: torch.Tensor,
        intrinsic_err: torch.Tensor,
    ) -> None:
        """Push the latest batch into the running-stats buffers.

        `obs` shape: `(N, feature_dim)` — flatten any leading time/batch
        dims before calling. `intrinsic_err` shape: `(N,)` — the **raw**
        per-sample error from `forward()`, not the normalized bonus.
        """
        obs_flat = obs.reshape(-1, obs.shape[-1])
        self.obs_rms.update(obs_flat)
        self.reward_rms.update(intrinsic_err.reshape(-1))

    @property
    def num_params(self) -> int:
        """Trainable predictor params only (target is frozen)."""
        return sum(p.numel() for p in self.predictor.parameters())
