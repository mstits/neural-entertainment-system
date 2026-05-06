"""
Policy network for NES game playing.

Two encoders are available, picked by the `encoder` constructor arg
(threaded from the game profile's `reinforce.encoder` key):

* `nature_dqn` (default) — the same 3-conv stack used in DQN and most
  Atari PPO papers. ~1.7M params. Fast forward pass, well understood.
* `impala` — IMPALA-style ResNet (3 stages × 2 residual blocks). ~3-4×
  the params, slower forward pass, but transfers across levels much
  better and learns sample-efficiently from sparse rewards. Worth
  the perf hit when training Zelda / Metroid-style exploration games.

Other improvements applied unconditionally:

* Orthogonal weight init with PPO-standard gains: √2 for relu trunks,
  0.01 for the actor logits, 1.0 for the critic. Stops the early-training
  gradient explosion that vanilla PyTorch init can produce.
* Optional `LayerNorm` on the trunk feature vector (`use_layernorm=True`,
  default). Stabilizes the value head when reward magnitude varies wildly
  across signals — exactly the case in this project where exploration
  rewards routinely span 5+ orders of magnitude.

The actor + critic share the trunk (one conv pass per step). PPO without
a value baseline has mathematically invalid advantage estimates, so the
critic head is non-optional.

Runs on MPS (Apple Silicon) when available.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------


class NatureDQNEncoder(nn.Module):
    """The classic 3-conv DQN encoder. 84x84 → 7x7 spatial, 64 channels."""

    def __init__(self, frame_stack: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)


class _ImpalaResBlock(nn.Module):
    """Two-conv residual block from IMPALA. Identity skip; relu before
    each conv (pre-activation pattern). Channel count is preserved."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h


class _ImpalaStage(nn.Module):
    """One IMPALA stage: 3x3 conv → maxpool 2x2 → two residual blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ImpalaResBlock(out_channels)
        self.res2 = _ImpalaResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA ResNet encoder. 3 stages with channel widths [16, 32, 32].

    Spatial path on 84×84 input: 84 → 42 → 21 → 11 (rounded by maxpool
    `stride=2,padding=1`). Flattened feature size = 32 × 11 × 11 = 3872,
    close to Nature-DQN's 3136 but with substantially more capacity.
    """

    CHANNELS = (16, 32, 32)

    def __init__(self, frame_stack: int = 4) -> None:
        super().__init__()
        in_c = frame_stack
        stages = []
        for out_c in self.CHANNELS:
            stages.append(_ImpalaStage(in_c, out_c))
            in_c = out_c
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = F.relu(x)
        return x.flatten(1)


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    """In-place orthogonal weight init with the given gain; bias to 0."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.orthogonal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """Actor-critic CNN. Encoder is either Nature-DQN or IMPALA ResNet.

    `forward(x)` → actor logits (kept for callers that don't need V).
    `forward_ac(x)` → `(logits, value)` — the canonical inference path.
    `value(x)` → V only (used at trajectory tail to bootstrap GAE).
    """

    # Bumped when state_dict layout changes in a non-additive way so
    # `load` can detect and refuse a clearly-incompatible checkpoint.
    ARCH_VERSION = 2

    def __init__(
        self,
        num_actions: int,
        frame_stack: int = 4,
        frame_size: int = 84,
        encoder: str = "nature_dqn",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.encoder_kind = encoder
        self.use_layernorm = use_layernorm

        if encoder == "nature_dqn":
            # Keep the legacy `conv1/conv2/conv3` attribute names so
            # checkpoints saved before the encoder split still load
            # without a key-rename adapter. Old state_dicts contain
            # `conv1.weight`, `conv2.weight`, `conv3.weight`; the
            # nested NatureDQNEncoder uses the same names under
            # `encoder.conv1.weight`. Resolved in `_load_legacy_keys`.
            self.encoder = NatureDQNEncoder(frame_stack)
        elif encoder == "impala":
            self.encoder = ImpalaEncoder(frame_stack)
        else:
            raise ValueError(
                f"unknown encoder {encoder!r}; expected 'nature_dqn' or 'impala'"
            )

        conv_out_size = self._compute_conv_output_size(frame_size, frame_stack)

        self.fc1 = nn.Linear(conv_out_size, 512)
        # LayerNorm on the trunk activation. Cheap (1024 params) and
        # measurably stabilizes the value head when reward magnitudes
        # span orders of magnitude, which is the norm in this project.
        # Identity by default if the user disables it.
        self.trunk_norm: nn.Module = (
            nn.LayerNorm(512) if use_layernorm else nn.Identity()
        )
        self.fc2 = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

        self._apply_orthogonal_init()

    def _compute_conv_output_size(self, frame_size: int, frame_stack: int) -> int:
        """Dry-run a tensor through the encoder to get the flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, frame_size, frame_size)
            return int(self.encoder(dummy).size(1))

    def _apply_orthogonal_init(self) -> None:
        """PPO-standard orthogonal init.

        Gains:
          * `sqrt(2)` for relu-followed conv + trunk FC
          * `0.01` for the actor logits (small to keep early policy near
            uniform; otherwise the network commits hard to a single
            action before any signal arrives)
          * `1.0` for the critic head (no nonlinearity afterwards)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                _orthogonal_init(m, gain=math.sqrt(2.0))
        # Override actor + critic with their PPO-paper-recommended gains.
        _orthogonal_init(self.fc2, gain=0.01)
        _orthogonal_init(self.value_head, gain=1.0)

    # ----- forward paths ------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_ac(x)[0]

    def forward_ac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        trunk = F.relu(self.trunk_norm(self.fc1(feats)))
        logits = self.fc2(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return logits, value

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_ac(x)[1]

    # ----- bookkeeping --------------------------------------------------

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        """Self-describing checkpoint. Carries enough hparams to
        reconstruct the network without consulting the profile YAML."""
        torch.save({
            "arch_version": self.ARCH_VERSION,
            "num_actions": self.num_actions,
            "frame_size": self.frame_size,
            "frame_stack": self.frame_stack,
            "encoder": self.encoder_kind,
            "use_layernorm": self.use_layernorm,
            "state_dict": self.state_dict(),
        }, str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "PolicyNetwork":
        data = torch.load(str(path), map_location=map_location, weights_only=False)
        net = cls(
            num_actions=data["num_actions"],
            frame_stack=data["frame_stack"],
            frame_size=data["frame_size"],
            # Older checkpoints predate these fields; default to the legacy
            # config (Nature-DQN, no LayerNorm) so they keep loading.
            encoder=data.get("encoder", "nature_dqn"),
            use_layernorm=data.get("use_layernorm", False),
        )
        sd = cls._migrate_legacy_state_dict(data["state_dict"])
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if unexpected:
            import logging as _log
            _log.getLogger(__name__).warning(
                "unexpected keys in checkpoint %s: %s", path, unexpected,
            )
        return net

    @staticmethod
    def _migrate_legacy_state_dict(sd: dict) -> dict:
        """Rewrite pre-encoder-split state_dict keys.

        Old format: top-level `conv1.weight`, `conv2.weight`, `conv3.weight`.
        New format: nested `encoder.conv1.weight`, etc.

        Anything that already uses the new layout (or never had the
        legacy keys, e.g. IMPALA checkpoints) is returned unchanged.
        """
        if any(k.startswith("encoder.") for k in sd):
            return sd
        if not any(k.startswith(("conv1.", "conv2.", "conv3.")) for k in sd):
            return sd
        out = {}
        for k, v in sd.items():
            if k.startswith(("conv1.", "conv2.", "conv3.")):
                out[f"encoder.{k}"] = v
            else:
                out[k] = v
        return out

    def act(self, frame_stack: torch.Tensor, deterministic: bool = False) -> int:
        """Select an action from a single observation."""
        if frame_stack.dim() == 3:
            frame_stack = frame_stack.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(frame_stack)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            probs = F.softmax(logits, dim=1)
            return int(torch.multinomial(probs, 1).item())


def get_best_device() -> torch.device:
    """Pick MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
