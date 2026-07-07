"""Small actor-critic MLP for tile-based observations.

The whole point of tile mode is shrinking the policy parameter count
to a regime where:

* GA mutation produces meaningful behavioral change per generation
  (moving 14k weights by 0.005 std covers a noticeable chunk of
  behavior space).
* PPO gradients are large relative to the parameter count, so the
  policy actually moves between updates instead of being lost in
  noise.

So this network is intentionally tiny — `Linear → LayerNorm → SiLU`
twice (pre-activation norm; bounds the linear output before the
nonlinearity to prevent variance blowup in a 14k-param net), then
split actor + critic heads. ~14k params total at the default 64/32
hidden widths, vs 1.7M for the pixel CNN.

Same `forward()` / `forward_ac()` / `act()` / `save()` / `load()`
surface as `PolicyNetwork` so the trainer can dispatch on encoder
without touching the call sites.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    """In-place orthogonal weight init with the given gain; bias to 0.

    Same gains as PolicyNetwork: √2 for relu/SiLU trunks, 0.01 for the
    actor logits (small so initial policy is near-uniform without
    committing), 1.0 for the critic head."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.orthogonal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


class TilePolicyNetwork(nn.Module):
    """Tile-based actor-critic. ~14k params at default widths.

    Construction:
        net = TilePolicyNetwork(num_actions=8, feature_dim=175)
        logits, value = net.forward_ac(features)  # features: (B, 175)
    """

    ARCH_VERSION = 1

    def __init__(
        self,
        num_actions: int,
        feature_dim: int = 175,
        hidden_dim: int = 64,
        trunk_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trunk_dim = trunk_dim

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, trunk_dim)
        self.norm2 = nn.LayerNorm(trunk_dim)
        # Actor head: logits over the discrete action space.
        self.actor = nn.Linear(trunk_dim, num_actions)
        # Critic head: scalar V(s). PPO without a value baseline is
        # mathematically broken at this scale (raw returns have wild
        # variance) — the critic is non-optional.
        self.critic = nn.Linear(trunk_dim, 1)

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _orthogonal_init(m, gain=math.sqrt(2.0))
        # PPO-paper gains for the heads.
        _orthogonal_init(self.actor, gain=0.01)
        _orthogonal_init(self.critic, gain=1.0)

    # ----- forward -----------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actor-only path (kept for callers that don't need V)."""
        return self.forward_ac(x)[0]

    def forward_ac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One-pass actor-critic.

        `x` shape: `(B, feature_dim)`. Tile features are int8 in the
        replay buffer; the trainer casts to float32 before this call.
        No /255 normalization since tile features are already small
        signed ints (the LayerNorm + first linear handle scale).
        """
        h = F.silu(self.norm1(self.fc1(x)))
        h = F.silu(self.norm2(self.fc2(h)))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_ac(x)[1]

    # ----- bookkeeping -------------------------------------------------

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        torch.save({
            "arch_version": self.ARCH_VERSION,
            "kind": "tile_mlp",
            "num_actions": self.num_actions,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "trunk_dim": self.trunk_dim,
            "state_dict": self.state_dict(),
        }, str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "TilePolicyNetwork":
        data = torch.load(str(path), map_location=map_location, weights_only=False)
        net = cls(
            num_actions=data["num_actions"],
            feature_dim=data.get("feature_dim", 175),
            hidden_dim=data.get("hidden_dim", 64),
            trunk_dim=data.get("trunk_dim", 32),
        )
        import logging as _log
        _logger = _log.getLogger(__name__)
        ckpt_arch = data.get("arch_version")
        if ckpt_arch is not None and ckpt_arch != cls.ARCH_VERSION:
            raise ValueError(
                f"checkpoint {path} has arch_version {ckpt_arch}, but this "
                f"TilePolicyNetwork is ARCH_VERSION {cls.ARCH_VERSION} — the "
                f"architecture is incompatible. Re-train or migrate the checkpoint."
            )
        if ckpt_arch is None:
            _logger.warning(
                "checkpoint %s predates arch_version; loading as legacy "
                "ARCH_VERSION %d — verify it behaves as expected.",
                path, cls.ARCH_VERSION,
            )
        missing, unexpected = net.load_state_dict(data["state_dict"], strict=False)
        if missing:
            raise ValueError(
                f"checkpoint {path} is missing {len(missing)} required "
                f"parameter(s): {missing}. Loading it would leave those weights "
                f"randomly initialized; refusing to return a partial policy."
            )
        if unexpected:
            _logger.warning(
                "unexpected keys in checkpoint %s: %s", path, unexpected,
            )
        return net

    def act(self, features: torch.Tensor, deterministic: bool = False) -> int:
        """Select an action from a single feature vector."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(features)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            probs = F.softmax(logits, dim=1)
            return int(torch.multinomial(probs, 1).item())


class TileRecurrentPolicyNetwork(nn.Module):
    """Stateful actor-critic with a small GRU over the tile feature
    vector. Drop-in shape-compatible with `TilePolicyNetwork` for the
    inference path; PPO update path needs hidden-state propagation
    through replay (not yet wired in `trainer.py`).

    Why a recurrent net for tile mode: the stateless `TilePolicyNetwork`
    sees only the current step's tile grid + scalars. It has no memory
    of recent decisions or velocities-over-time. For credit assignment
    over long trajectories (where the bad jump at step T-30 caused the
    death at step T), a recurrent layer's hidden state carries that
    context forward. GAE with γ=0.99 over 600-step trajectories is
    theoretically enough for credit-assignment, but in practice the
    signal washes out without a learnable carry-state.

    Construction:
        net = TileRecurrentPolicyNetwork(num_actions=8, feature_dim=175)
        h0 = net.initial_hidden(batch_size=16, device=torch.device('cpu'))
        logits, value, h_next = net.forward_ac_recurrent(features, h0)
        # h_next propagates to the next step

    On `done`: caller resets the hidden state for that worker via
    `net.initial_hidden(1, device)` and replaces row `i` of the batch
    hidden with the zero state.
    """

    ARCH_VERSION = 1

    def __init__(
        self,
        num_actions: int,
        feature_dim: int = 175,
        hidden_dim: int = 64,
        gru_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim

        # Project tile features to a fixed embedding before recurrence.
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        # Single GRU cell. Smaller than LSTM (no separate cell state)
        # and almost always equivalent quality on RL tasks.
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        self.norm_h = nn.LayerNorm(gru_dim)
        self.actor = nn.Linear(gru_dim, num_actions)
        self.critic = nn.Linear(gru_dim, 1)

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _orthogonal_init(m, gain=math.sqrt(2.0))
        # GRU has its own initialization scheme; let PyTorch handle.
        _orthogonal_init(self.actor, gain=0.01)
        _orthogonal_init(self.critic, gain=1.0)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def initial_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Zero hidden state. Caller initializes once per episode and
        passes it through `forward_ac_recurrent` step by step,
        resetting individual rows on done."""
        return torch.zeros(batch_size, self.gru_dim, device=device)

    def forward_ac_recurrent(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns `(logits, value, h_next)`.

        `x` shape: `(B, feature_dim)` — single timestep, one tile vector
        per worker. `h` shape: `(B, gru_dim)` — hidden state to extend.

        For the trainer's collection path: B = num_workers; h flows
        across episode steps with worker-row resets on done.
        For the PPO update path: replay the trajectory step-by-step
        through this same forward, accumulating gradients across the
        unrolled GRU. Truncated BPTT — a standard recurrent-PPO pattern.
        """
        proj = self.input_proj(x)
        h_next = self.gru(proj, h)
        h_norm = self.norm_h(h_next)
        logits = self.actor(h_norm)
        value = self.critic(h_norm).squeeze(-1)
        return logits, value, h_next

    # Stateless-shaped fallback for callers that don't carry hidden
    # state. Initializes a fresh hidden each call — equivalent to
    # `TilePolicyNetwork.forward_ac` but goes through one GRU step.
    # Use only for unit tests / sanity checks; the trainer's hot path
    # MUST use `forward_ac_recurrent` to get the recurrence benefit.
    def forward_ac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.initial_hidden(x.size(0), x.device)
        logits, value, _ = self.forward_ac_recurrent(x, h)
        return logits, value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_ac(x)[0]
