"""
Policy network for NES game playing.

Architecture: Nature-DQN style CNN.
- Input: (batch, 4, 84, 84) — 4 stacked grayscale frames
- 3 conv layers with ReLU
- 2 fully-connected layers
- Output: logits over the game's action space

This is deliberately conservative — it's the same architecture that works well
on Atari and has been battle-tested. An IMPALA-style ResNet encoder is a
plausible future swap for better transfer across levels.

Runs on MPS backend for Apple Silicon acceleration.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Nature-DQN CNN adapted for policy gradient / genetic algorithm use.

    Actor-Critic: `forward(x)` returns the actor logits. `value(x)`
    (or the combined `forward_ac(x)` → `(logits, value)`) returns the
    scalar state-value baseline. PPO without a value baseline has
    mathematically invalid advantage estimates (raw returns have
    pathological variance) — the critic head closes that gap.
    """

    def __init__(
        self,
        num_actions: int,
        frame_stack: int = 4,
        frame_size: int = 84,
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.frame_size = frame_size

        # Convolutional feature extractor (Nature-DQN)
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute conv output size: 84 → 20 → 9 → 7, so 64 * 7 * 7 = 3136
        conv_out_size = self._compute_conv_output_size(frame_size, frame_stack)

        # Shared trunk → actor head (policy logits) + critic head (V).
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        # Critic head: one scalar V(s). Shares the conv + fc1 trunk
        # so the value and policy are computed in one forward pass.
        # Orthogonal init at 1.0 (standard PPO practice) would be a
        # nice-to-have; stock PyTorch init works well enough in
        # practice and matches the rest of the net.
        self.value_head = nn.Linear(512, 1)

    def _compute_conv_output_size(self, frame_size: int, frame_stack: int) -> int:
        """Dry-run a tensor through the conv layers to get the flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, frame_size, frame_size)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            return int(dummy.view(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Policy-only forward (backwards-compat with callers that
        don't care about the value baseline — GA population
        selection, replay). Returns actor logits only.

        Args:
            x: (batch, frame_stack, 84, 84) float tensor in range [0, 1]

        Returns:
            (batch, num_actions) logits
        """
        return self.forward_ac(x)[0]

    def forward_ac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor-critic forward — one pass, two heads.

        Returns:
            (logits: (batch, num_actions), value: (batch,))
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        trunk = F.relu(self.fc1(x))
        logits = self.fc2(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return logits, value

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Value-only forward (useful for bootstrap at trajectory end)."""
        return self.forward_ac(x)[1]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        """Self-describing checkpoint: architecture hparams + weights."""
        torch.save({
            "num_actions": self.num_actions,
            "frame_size": self.frame_size,
            "frame_stack": self.conv1.in_channels,
            "state_dict": self.state_dict(),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PolicyNetwork":
        data = torch.load(str(path), map_location=map_location, weights_only=False)
        net = cls(
            num_actions=data["num_actions"],
            frame_stack=data["frame_stack"],
            frame_size=data["frame_size"],
        )
        # strict=False so pre-Critic checkpoints (no value_head weights)
        # load cleanly — the value head stays at its random init,
        # untrained until the next REINFORCE pass touches it.
        missing, unexpected = net.load_state_dict(data["state_dict"], strict=False)
        if unexpected:
            import logging as _log
            _log.getLogger(__name__).warning(
                "unexpected keys in checkpoint %s: %s", path, unexpected,
            )
        return net

    def act(self, frame_stack: torch.Tensor, deterministic: bool = False) -> int:
        """
        Select an action from a single observation.

        Args:
            frame_stack: (frame_stack, 84, 84) or (1, frame_stack, 84, 84) tensor
            deterministic: if True, pick argmax; else sample from softmax

        Returns:
            Action index into the game's action space
        """
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
