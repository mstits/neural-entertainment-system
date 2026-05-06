"""DreamerV3 world model — RSSM, encoder, decoder, prediction heads.

This is the architectural core of the world-model trainer. The model
learns to compress observations into a small latent state and predict
how that state evolves under the policy's actions, plus auxiliary heads
for reward and episode-continue. Once the model is accurate enough,
the policy is trained on imagined rollouts in latent space — much
cheaper than emulator steps and lets the agent "plan ahead" implicitly.

Architecture (all on a configurable device, defaults to MPS):

* `ObservationEncoder` — CNN, observation → flat features. Uses
  GroupNorm + SiLU per the V3 paper. 4-stage downsample mirrors the
  decoder.
* `ObservationDecoder` — transpose-CNN, latent → reconstructed obs.
  Reconstruction loss anchors the latent space to actual observations.
* `RSSM` (Recurrent State-Space Model) — the heart. A GRU runs a
  deterministic state `h_t`; each step produces both a posterior
  distribution `q(z_t | h_t, encoder(o_t))` and a prior distribution
  `p(z_t | h_t)`. KL between posterior and prior is the dynamics-loss
  term that teaches the prior to predict the posterior.
* `RewardHead` / `ContinueHead` — small MLPs from `(h_t, z_t)` to
  reward and not-done predictions.

V3 vs V1/V2: the proper V3 latent is a **categorical** mixture (32
classes × 32 dims) with straight-through gradients. This scaffold
uses a continuous Gaussian latent for v1 simplicity and numerical
stability on MPS — the categorical upgrade is a natural follow-up
that swaps the `_StochHead` block without touching anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class WorldModelConfig:
    """All sizing knobs for the world model in one place. Defaults
    match DreamerV3: 32 stochastic dims × 32 categories. Smaller than
    the paper for Apple-Silicon-friendly memory but large enough for
    DQN-class policies."""

    obs_channels: int = 4
    obs_size: int = 84
    num_actions: int = 8
    # Deterministic GRU state width.
    deter_dim: int = 256
    # Stochastic latent. Categorical (V3-proper) is `latent_kind="categorical"`
    # with stoch_dim independent categorical heads, each of `stoch_classes`
    # classes. Total flat representation = stoch_dim * stoch_classes.
    # `latent_kind="gaussian"` falls back to a diagonal Gaussian over
    # `stoch_dim` continuous dims, kept for ablation comparisons.
    latent_kind: str = "categorical"
    stoch_dim: int = 32
    stoch_classes: int = 32
    # Hidden width inside the RSSM cell + small MLP heads.
    hidden_dim: int = 256
    # Encoder feature dim coming out of the CNN.
    encoder_feat_dim: int = 256
    # Free-bits floor on the KL term; below this the dynamics loss
    # contributes zero gradient. Stops the prior from over-fitting on
    # easy steady-state regions of the latent space.
    kl_free_nats: float = 1.0

    @property
    def stoch_flat(self) -> int:
        """Flat representation size of the stochastic latent.

        Categorical: each of `stoch_dim` heads emits a one-hot over
        `stoch_classes`. Concatenated, that's `stoch_dim * stoch_classes`
        bits. Gaussian: simply `stoch_dim`.
        """
        if self.latent_kind == "categorical":
            return self.stoch_dim * self.stoch_classes
        return self.stoch_dim


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _conv_norm_act(in_c: int, out_c: int, k: int, s: int, p: int) -> nn.Sequential:
    """Conv → GroupNorm → SiLU. The DreamerV3 default activation block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
    )


def _deconv_norm_act(in_c: int, out_c: int, k: int, s: int, p: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
    )


# ---------------------------------------------------------------------
# Encoder + Decoder
# ---------------------------------------------------------------------


class ObservationEncoder(nn.Module):
    """4-stage strided CNN. 84 → 42 → 21 → 11 → 6 spatial; channels
    32 → 64 → 128 → 256. Output flattened then projected to
    `encoder_feat_dim` so downstream heads see a fixed-size vector."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stages = nn.Sequential(
            _conv_norm_act(cfg.obs_channels, 32, k=4, s=2, p=1),  # 84 → 42
            _conv_norm_act(32, 64, k=4, s=2, p=1),                # 42 → 21
            _conv_norm_act(64, 128, k=4, s=2, p=1),               # 21 → 10
            _conv_norm_act(128, 256, k=4, s=2, p=1),              # 10 → 5
        )
        # Compute flat dim once via a dry run; spatial output for 84
        # input is 5 (84 → 42 → 21 → 10 → 5), but we let the conv stack
        # decide so changing the input size doesn't require manual math.
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.obs_channels, cfg.obs_size, cfg.obs_size)
            flat = self.stages(dummy).flatten(1).size(1)
        self.proj = nn.Linear(flat, cfg.encoder_feat_dim)
        self._flat_dim = flat

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs `(B, C, H, W)` → features `(B, encoder_feat_dim)`."""
        h = self.stages(obs)
        h = h.flatten(1)
        return self.proj(h)


class ObservationDecoder(nn.Module):
    """Mirror of the encoder. Decodes latent state → reconstructed obs.

    Input: `(B, deter_dim + stoch_flat)`. Output: `(B, C, H, W)` with
    pixels in [0, 1]; the trainer uses MSE against the normalized obs.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.deter_dim + cfg.stoch_flat, 256 * 5 * 5)
        self.stages = nn.Sequential(
            _deconv_norm_act(256, 128, k=4, s=2, p=1),  # 5 → 10
            _deconv_norm_act(128, 64, k=4, s=2, p=1),   # 10 → 20
            _deconv_norm_act(64, 32, k=4, s=2, p=2),    # 20 → 38
        )
        # Final transpose conv lands at 84×84; output channel = obs_channels.
        # ConvTranspose math: out = (in-1)*s - 2*p + k. Tune k/s/p so we
        # land precisely at obs_size. Settled on k=10, s=2, p=0:
        # (38-1)*2 - 0 + 10 = 84.
        self.final = nn.ConvTranspose2d(
            32, cfg.obs_channels, kernel_size=10, stride=2, padding=0
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.proj(latent).view(-1, 256, 5, 5)
        h = self.stages(h)
        # Sigmoid maps to [0, 1]. The trainer normalizes obs to that
        # same range, so MSE here is a proper reconstruction loss.
        return torch.sigmoid(self.final(h))


# ---------------------------------------------------------------------
# Stochastic head — Gaussian latent (V3 categorical is a follow-up)
# ---------------------------------------------------------------------


class _StochHead(nn.Module):
    """Stochastic-latent head. Either:

    * **categorical** (V3-proper) — emits `stoch_dim * stoch_classes`
      logits, reshaped to `(B, stoch_dim, stoch_classes)`. The actual
      sampling and KL computation are done on the RSSM (this module
      only exposes the parameters).
    * **gaussian** — emits `2 * stoch_dim` channels split into
      `(mean, std)` for a diagonal Gaussian. `std` uses softplus + 0.1
      floor so the posterior can't collapse to a delta.

    Output shape is always `(B, *, ...)` where the trailing dims encode
    the distribution params for the chosen `latent_kind`. RSSM owns the
    sampling/KL math.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cfg: WorldModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.latent_kind == "categorical":
            out_dim = cfg.stoch_dim * cfg.stoch_classes
        elif cfg.latent_kind == "gaussian":
            out_dim = 2 * cfg.stoch_dim
        else:
            raise ValueError(
                f"unknown latent_kind {cfg.latent_kind!r}; "
                "expected 'categorical' or 'gaussian'"
            )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw distribution parameters. Caller (RSSM)
        interprets according to `cfg.latent_kind`."""
        return self.net(x)


# ---------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------


@dataclass
class RSSMState:
    """One time step of RSSM state.

    * `deter` — the GRU's deterministic hidden vector.
    * `stoch` — the **flat** stochastic representation `(B, stoch_flat)`.
      Heads concatenate `[deter, stoch]` regardless of latent kind.
    * `prior_params` / `post_params` — opaque distribution parameter
      tensors, shape depends on `latent_kind`:
        - categorical: `(B, stoch_dim, stoch_classes)` logits
        - gaussian:    `(B, 2 * stoch_dim)` to split into mean/std

    Distribution params are populated by `RSSM.observe()` (both prior
    and posterior) and `RSSM.imagine()` (prior only).
    """

    deter: torch.Tensor
    stoch: torch.Tensor
    prior_params: torch.Tensor | None = None
    post_params: torch.Tensor | None = None


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Three sub-modules:

    * `state_action_to_hidden`: combines `(prev_stoch, prev_action)` into
      a hidden vector, then a GRU steps the deterministic state forward.
    * `prior_head` (`_StochHead`): produces `p(z_t | h_t)`.
    * `posterior_head` (`_StochHead`): produces `q(z_t | h_t, embed_t)`
      where `embed_t` is the encoder's output for the current obs.

    Two main entry points:

    * `observe(embed_seq, action_seq)` → list of states with both prior
      AND posterior info. Used during world-model training.
    * `imagine(start_state, action_seq)` → list of states with prior
      info only. Used for actor/critic training on imagined rollouts.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Linear(cfg.num_actions, cfg.hidden_dim)

        # Flat stochastic dim (categorical: dim*classes; gaussian: dim).
        stoch_flat = cfg.stoch_flat
        self.state_action_to_hidden = nn.Sequential(
            nn.Linear(stoch_flat + cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)
        self.prior_head = _StochHead(cfg.deter_dim, cfg.hidden_dim, cfg)
        self.posterior_head = _StochHead(
            cfg.deter_dim + cfg.encoder_feat_dim, cfg.hidden_dim, cfg
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Zero-init initial deter/stoch state for a batch of B
        sequences. Imagination + observation both consume one of these."""
        return RSSMState(
            deter=torch.zeros(batch_size, self.cfg.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.cfg.stoch_flat, device=device),
        )

    def _step_deter(
        self,
        prev_stoch: torch.Tensor,
        prev_action_onehot: torch.Tensor,
        prev_deter: torch.Tensor,
    ) -> torch.Tensor:
        action_h = self.action_embed(prev_action_onehot)
        # Combine prev_stoch and action embedding into the GRU input.
        gru_in = self.state_action_to_hidden(
            torch.cat([prev_stoch, action_h], dim=-1)
        )
        return self.gru(gru_in, prev_deter)

    def _sample_stoch(self, params: torch.Tensor) -> torch.Tensor:
        """Sample a flat stochastic latent from raw head parameters,
        with gradients propagated for backprop:

        * Gaussian: reparameterized `mean + eps * std`.
        * Categorical: per-dim multinomial → one-hot, with the
          straight-through trick `one_hot + probs - probs.detach()` so
          the forward is discrete but gradients flow as if through
          `probs`. This is what makes V3's discrete latent trainable.

        Returns shape `(B, stoch_flat)`.
        """
        if self.cfg.latent_kind == "gaussian":
            mean, raw_std = params.chunk(2, dim=-1)
            std = F.softplus(raw_std) + 0.1
            eps = torch.randn_like(mean)
            return mean + eps * std

        # categorical
        B = params.size(0)
        D, C = self.cfg.stoch_dim, self.cfg.stoch_classes
        logits = params.view(B, D, C)
        # Sample categorical per-dim: multinomial on flattened (B*D, C).
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs.reshape(B * D, C), num_samples=1).view(B, D)
        one_hot = F.one_hot(idx, num_classes=C).float()
        # Straight-through: forward sees `one_hot` (discrete bits), the
        # gradient passes through `probs` as if we'd used soft sampling.
        sample = one_hot + probs - probs.detach()
        return sample.view(B, D * C)

    def observe(
        self,
        embed_seq: torch.Tensor,
        action_seq: torch.Tensor,
        init_state: RSSMState | None = None,
    ) -> list[RSSMState]:
        """Run the RSSM forward over a sequence using the encoder's
        embedding at each step. Returns one `RSSMState` per timestep
        with both posterior and prior distributions populated.

        embed_seq:  (B, L, encoder_feat_dim)
        action_seq: (B, L, num_actions) one-hot
        """
        B, L, _ = embed_seq.shape
        device = embed_seq.device
        state = init_state if init_state is not None else self.initial_state(B, device)
        states: list[RSSMState] = []
        for t in range(L):
            # NB: at t=0 we use a zero "prev action". The replay buffer
            # provides the action that *led to* obs_t, so action_seq[:,t]
            # is actually `a_{t-1}`. To stay aligned with the standard
            # Dreamer convention (predict z_t from h_t which depends on
            # a_{t-1}), shift: at t=0, prev_action is zero; at t>0,
            # prev_action is action_seq[:, t-1].
            prev_action = (
                torch.zeros(B, self.cfg.num_actions, device=device)
                if t == 0
                else action_seq[:, t - 1]
            )
            deter = self._step_deter(state.stoch, prev_action, state.deter)
            prior_params = self.prior_head(deter)
            post_in = torch.cat([deter, embed_seq[:, t]], dim=-1)
            post_params = self.posterior_head(post_in)
            # Posterior sample is what the trainer feeds to the heads —
            # uses the actual observation. Prior is supervised against
            # the posterior via KL.
            stoch = self._sample_stoch(post_params)
            state = RSSMState(
                deter=deter,
                stoch=stoch,
                prior_params=prior_params,
                post_params=post_params,
            )
            states.append(state)
        return states

    def imagine(
        self,
        start_state: RSSMState,
        actions: torch.Tensor,
    ) -> list[RSSMState]:
        """Roll the RSSM forward using only the prior — no observations
        needed. Used for actor/critic training: starting from the
        encoded posterior of a real observation, imagine H steps of
        latent trajectory.

        actions: (B, H, num_actions) one-hot
        """
        B, H, _ = actions.shape
        state = start_state
        states: list[RSSMState] = []
        for t in range(H):
            deter = self._step_deter(state.stoch, actions[:, t], state.deter)
            prior_params = self.prior_head(deter)
            stoch = self._sample_stoch(prior_params)
            state = RSSMState(
                deter=deter,
                stoch=stoch,
                prior_params=prior_params,
            )
            states.append(state)
        return states

    def kl_loss(
        self,
        states: list[RSSMState],
        free_nats: float = 1.0,
    ) -> torch.Tensor:
        """KL(posterior || prior), per-sample free-nats.

        * Categorical: `KL(post || prior) = Σ_c p_post * log(p_post / p_prior)`,
          summed over classes for each of the `stoch_dim` independent heads
          and then summed across heads. Computed in log-space for numerical
          stability.
        * Gaussian: standard analytic form
          `log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²) / (2σ_p²) - 0.5`,
          summed over dim.

        Free-nats is applied per-sample (per (B, t)), not just on the
        global mean — matches the V3 paper's "balance" treatment and
        prevents one outlier sample from dominating the floor logic.
        """
        kls = []
        for s in states:
            assert s.post_params is not None and s.prior_params is not None, (
                "RSSMState missing distribution params (must come from observe())"
            )
            if self.cfg.latent_kind == "categorical":
                B = s.post_params.size(0)
                D, C = self.cfg.stoch_dim, self.cfg.stoch_classes
                post_logits = s.post_params.view(B, D, C)
                prior_logits = s.prior_params.view(B, D, C)
                post_lp = F.log_softmax(post_logits, dim=-1)
                prior_lp = F.log_softmax(prior_logits, dim=-1)
                post_p = post_lp.exp()
                # KL per (B, dim) = Σ_c p (log p - log q); sum over dims.
                kl_per_dim = (post_p * (post_lp - prior_lp)).sum(dim=-1)
                kl = kl_per_dim.sum(dim=-1)  # (B,)
            else:
                mu_q, raw_q = s.post_params.chunk(2, dim=-1)
                mu_p, raw_p = s.prior_params.chunk(2, dim=-1)
                sigma_q = F.softplus(raw_q) + 0.1
                sigma_p = F.softplus(raw_p) + 0.1
                kl_per_dim = (
                    torch.log(sigma_p / sigma_q)
                    + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
                    - 0.5
                )
                kl = kl_per_dim.sum(dim=-1)
            kls.append(kl)
        kl_bt = torch.stack(kls, dim=1)  # (B, L)
        # Per-sample free-nats: clamp at the floor BEFORE averaging so
        # samples already below the floor contribute zero gradient
        # while above-floor samples still backprop. Subtract the floor
        # so a fully-below-floor batch reads as 0 loss.
        return (torch.clamp(kl_bt, min=free_nats) - free_nats).mean()


# ---------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------


class RewardHead(nn.Module):
    """Predicts scalar reward from `(deter, stoch)`. Symlog-transformed
    target is more stable with high-dynamic-range rewards (this codebase
    routinely sees rewards spanning 5+ orders of magnitude)."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.stoch_flat, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class ContinueHead(nn.Module):
    """Predicts P(not done). Logit; trainer applies BCEWithLogits."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.stoch_flat, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


# ---------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------


class WorldModel(nn.Module):
    """Bundles the encoder, RSSM, decoder, and heads.

    Use `train_step()` for the world-model gradient step. `imagine()`
    and `encode()` are exposed for the actor-critic training loop.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ObservationEncoder(cfg)
        self.rssm = RSSM(cfg)
        self.decoder = ObservationDecoder(cfg)
        self.reward_head = RewardHead(cfg)
        self.continue_head = ContinueHead(cfg)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs `(B, C, H, W)` → embedding `(B, encoder_feat_dim)`."""
        return self.encoder(obs)

    @staticmethod
    def latent(state: RSSMState) -> torch.Tensor:
        """Concat `(deter, stoch)` for downstream heads."""
        return torch.cat([state.deter, state.stoch], dim=-1)

    def loss(
        self,
        obs: torch.Tensor,        # (B, L, C, H, W) float in [0, 1]
        actions: torch.Tensor,    # (B, L, num_actions) one-hot
        rewards: torch.Tensor,    # (B, L) symlog-transformed
        continues: torch.Tensor,  # (B, L) bool tensor (1 - done)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the full world-model loss (recon + reward + continue
        + KL). Returns (loss, stats_dict)."""
        B, L = obs.shape[:2]
        # Encode the entire sequence in one shot — encoder is stateless.
        flat_obs = obs.view(B * L, *obs.shape[2:])
        embeddings = self.encoder(flat_obs).view(B, L, -1)
        states = self.rssm.observe(embeddings, actions)
        latents = torch.stack([self.latent(s) for s in states], dim=1)
        flat_latents = latents.view(B * L, -1)

        recon = self.decoder(flat_latents).view(B, L, *obs.shape[2:])
        recon_loss = F.mse_loss(recon, obs)

        reward_pred = self.reward_head(flat_latents).view(B, L)
        reward_loss = F.mse_loss(reward_pred, rewards)

        continue_logits = self.continue_head(flat_latents).view(B, L)
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits, continues.float(),
        )

        kl_loss = self.rssm.kl_loss(states, free_nats=self.cfg.kl_free_nats)

        # DreamerV3 typically uses 1.0 for all four; the kl_free_nats
        # floor handles the dynamics term. Tunable via subclass.
        total = recon_loss + reward_loss + continue_loss + kl_loss
        stats = {
            "wm_recon": float(recon_loss.detach().item()),
            "wm_reward": float(reward_loss.detach().item()),
            "wm_continue": float(continue_loss.detach().item()),
            "wm_kl": float(kl_loss.detach().item()),
        }
        return total, stats
