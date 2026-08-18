"""Shared-trunk, per-level multi-head tile policy.

Context: the naive version of "one net for every level" was already
falsified (see `scripts/interference_falsifier.py`) -- pooling 1-1 and
1-2 successes into one undifferentiated net captured 1-1 and destroyed
1-2. The untested form is trunk-plus-heads: one shared representation
feeding N independent actor/critic heads, so gradient from one level's
loss cannot overwrite another level's output layer, only the shared
trunk underneath it.

This module is architecture + import/export only. It does not run the
emulator, train anything, or touch any checkpoint on disk by itself --
every torch.load/torch.save call is the caller's choice, made against
whatever checkpoint path they pass in.

Layout, matched key-for-key to `src.models.tile_policy.TilePolicyNetwork`
so the shared trunk is a drop-in slice of that net's state_dict:

    fc1 -> norm1 -> SiLU -> fc2 -> norm2 -> SiLU        (shared trunk)
    -> actor_heads[level] / critic_heads[level]          (per-level, N of each)

`export_head(level)` turns one head + the shared trunk back into that
exact single-head TilePolicyNetwork state_dict layout, so a trained
head evaluates through `scripts/eval_game.py` unmodified -- the eval
stack shape-infers hidden/trunk dims from `fc1.weight` / `fc2.weight`
and neither knows nor cares that the checkpoint's trunk was shared
across other levels during training.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keys shared 1:1 with TilePolicyNetwork's state_dict. Order is the
# encoder's fc1 -> norm1 -> fc2 -> norm2 stack; eval_game's shape-infer
# helper reads fc1.weight / fc2.weight specifically, so these names are
# load-bearing, not cosmetic.
TRUNK_KEYS = (
    "fc1.weight", "fc1.bias",
    "norm1.weight", "norm1.bias",
    "fc2.weight", "fc2.bias",
    "norm2.weight", "norm2.bias",
)
HEAD_KEYS = ("actor.weight", "actor.bias", "critic.weight", "critic.bias")

LevelKey = Union[int, str]


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    """Same init recipe as TilePolicyNetwork: orthogonal weight, zero
    bias. Applied per-head below so every head starts from the same
    distribution a from-scratch specialist would."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.orthogonal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def _resolve_state_dict(checkpoint, map_location: str = "cpu") -> dict:
    """Unwrap a checkpoint (path, on-disk payload, or bare state_dict)
    down to a raw {param_name: tensor} mapping.

    Mirrors the unwrap order `build_tile_policy_from_checkpoint` uses in
    `src/models/tile_policy.py` (net_state_dict, then state_dict, then a
    bare tensor mapping) so anything that loads there for eval also
    loads here for warm-starting -- one convention, not two.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(
            str(checkpoint), map_location=map_location, weights_only=False,
        )
    if isinstance(checkpoint, dict):
        if "net_state_dict" in checkpoint:
            return checkpoint["net_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        f"could not find a state_dict in checkpoint of type {type(checkpoint)!r}; "
        "expected a path, a {'net_state_dict': ...} / {'state_dict': ...} "
        "payload, or a bare tensor mapping."
    )


class MultiHeadTilePolicy(nn.Module):
    """Tile-based actor-critic with one shared trunk and N per-level
    actor/critic head pairs.

    Construction:
        net = MultiHeadTilePolicy(num_actions=8, num_levels=4,
                                   feature_dim=175,
                                   levels=["1-1", "1-2", "1-3", "1-4"])
        logits, value = net.forward_ac(features, level_ids)

    `level_ids` is either a single int (whole-batch shortcut) or a
    `(B,)` long tensor of per-sample level indices/labels resolved via
    `levels`. A mixed batch -- some rows level 0, some level 2 -- is the
    normal case, not an edge case: PPO minibatches are shuffled across
    every level's rollouts once they share one buffer.
    """

    ARCH_VERSION = 1

    def __init__(
        self,
        num_actions: int,
        num_levels: int,
        feature_dim: int = 175,
        hidden_dim: int = 256,
        trunk_dim: int = 64,
        levels: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if levels is not None and len(levels) != num_levels:
            raise ValueError(
                f"levels has {len(levels)} entries but num_levels={num_levels}"
            )
        self.num_actions = num_actions
        self.num_levels = num_levels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trunk_dim = trunk_dim
        self.levels = list(levels) if levels is not None else [
            str(i) for i in range(num_levels)
        ]

        # Shared trunk -- identical module names/shapes to
        # TilePolicyNetwork so a trunk-only slice of this state_dict
        # copies straight into a single-head net (see export_head).
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, trunk_dim)
        self.norm2 = nn.LayerNorm(trunk_dim)

        # Per-level heads. ModuleList index == the level index passed
        # at forward time and accepted by export_head/init_head_from.
        self.actor_heads = nn.ModuleList(
            nn.Linear(trunk_dim, num_actions) for _ in range(num_levels)
        )
        self.critic_heads = nn.ModuleList(
            nn.Linear(trunk_dim, 1) for _ in range(num_levels)
        )

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self) -> None:
        for m in (self.fc1, self.fc2):
            _orthogonal_init(m, gain=math.sqrt(2.0))
        # PPO-paper gains for the heads, same as TilePolicyNetwork:
        # small actor gain so the initial policy is near-uniform,
        # unit critic gain.
        for head in self.actor_heads:
            _orthogonal_init(head, gain=0.01)
        for head in self.critic_heads:
            _orthogonal_init(head, gain=1.0)

    # ----- level resolution --------------------------------------------

    def _level_index(self, level: LevelKey) -> int:
        if isinstance(level, str):
            try:
                return self.levels.index(level)
            except ValueError:
                raise KeyError(
                    f"unknown level label {level!r}; known levels: {self.levels}"
                ) from None
        idx = int(level)
        if not (0 <= idx < self.num_levels):
            raise IndexError(
                f"level index {idx} out of range [0, {self.num_levels})"
            )
        return idx

    def _resolve_param(self, dotted_name: str) -> nn.Parameter:
        mod_name, param_name = dotted_name.split(".")
        return getattr(getattr(self, mod_name), param_name)

    # ----- forward -------------------------------------------------------

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.fc1(x)))
        h = F.silu(self.norm2(self.fc2(h)))
        return h

    def forward_ac(
        self, x: torch.Tensor, level_ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-pass actor-critic with PER-SAMPLE head routing.

        Every head is evaluated over the whole batch and the right row
        picked out with `gather`, rather than boolean-masked
        assignment, so autograd gives exact gradient isolation for
        free: gather's backward scatters gradient only into the rows
        that were actually selected, so a loss confined to one level's
        rows produces an exact-zero gradient at every OTHER head's
        parameters -- not a numerically-small one, an exact zero, which
        is what makes an untouched head bit-identical after an
        optimizer step.
        """
        if x.dim() != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected x shape (B, {self.feature_dim}), got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        if isinstance(level_ids, int):
            level_ids = torch.full(
                (batch,), level_ids, dtype=torch.long, device=x.device,
            )
        elif not torch.is_tensor(level_ids):
            level_ids = torch.as_tensor(level_ids, dtype=torch.long, device=x.device)
        level_ids = level_ids.to(device=x.device, dtype=torch.long)
        if level_ids.shape != (batch,):
            raise ValueError(
                f"level_ids must have shape ({batch},), got {tuple(level_ids.shape)}"
            )
        if bool((level_ids < 0).any()) or bool((level_ids >= self.num_levels).any()):
            raise IndexError(f"level_ids out of range [0, {self.num_levels})")

        trunk_out = self._trunk(x)  # (B, trunk_dim)

        all_logits = torch.stack(
            [head(trunk_out) for head in self.actor_heads], dim=0,
        )  # (num_levels, B, num_actions)
        all_values = torch.stack(
            [head(trunk_out).squeeze(-1) for head in self.critic_heads], dim=0,
        )  # (num_levels, B)

        actor_idx = level_ids.view(1, batch, 1).expand(1, batch, self.num_actions)
        logits = torch.gather(all_logits, 0, actor_idx).squeeze(0)
        critic_idx = level_ids.view(1, batch)
        value = torch.gather(all_values, 0, critic_idx).squeeze(0)
        return logits, value

    def forward(self, x: torch.Tensor, level_ids) -> torch.Tensor:
        return self.forward_ac(x, level_ids)[0]

    def value(self, x: torch.Tensor, level_ids) -> torch.Tensor:
        return self.forward_ac(x, level_ids)[1]

    def act(
        self, features: torch.Tensor, level: LevelKey, deterministic: bool = False,
    ) -> int:
        """Select an action from a single feature vector for one level."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        idx = self._level_index(level)
        with torch.no_grad():
            logits = self.forward(features, idx)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            probs = F.softmax(logits, dim=1)
            return int(torch.multinomial(probs, 1).item())

    # ----- export ----------------------------------------------------------

    def export_head(self, level: LevelKey) -> dict:
        """Return a state_dict in the EXACT single-head TilePolicyNetwork
        layout (shared trunk + this level's actor/critic), detached and
        moved to CPU. This is what makes the experiment evaluable: the
        trunk-plus-heads split is invisible on the far side of export --
        `scripts/eval_game.py` shape-infers hidden/trunk dims from
        `fc1.weight` / `fc2.weight` and loads this directly, with zero
        changes to the eval stack.
        """
        idx = self._level_index(level)
        sd: dict[str, torch.Tensor] = {}
        for k in TRUNK_KEYS:
            sd[k] = self._resolve_param(k).detach().cpu().clone()
        actor = self.actor_heads[idx]
        critic = self.critic_heads[idx]
        sd["actor.weight"] = actor.weight.detach().cpu().clone()
        sd["actor.bias"] = actor.bias.detach().cpu().clone()
        sd["critic.weight"] = critic.weight.detach().cpu().clone()
        sd["critic.bias"] = critic.bias.detach().cpu().clone()
        return sd

    def export_checkpoint(self, level: LevelKey, **meta) -> dict:
        """Wrap `export_head`'s state_dict in the on-disk payload shape
        the trainer / bc_distill checkpoints use (`{"net_state_dict":
        ...}`), so the result can be `torch.save`d straight to a `.pt`
        that `eval_game.py` reads unmodified. Does not write to disk --
        saving is the caller's call, against a path the caller chooses.
        """
        idx = self._level_index(level)
        payload = {"net_state_dict": self.export_head(idx), "level": self.levels[idx]}
        payload.update(meta)
        return payload

    # ----- warm start --------------------------------------------------

    def init_trunk_from(self, checkpoint) -> None:
        """Copy fc1/norm1/fc2/norm2 weights from an existing specialist
        checkpoint (path, on-disk payload, or bare state_dict) into the
        shared trunk. Shapes must match exactly -- a specialist trained
        at a different hidden/trunk width can't warm-start this trunk,
        and this raises rather than silently reshaping."""
        sd = _resolve_state_dict(checkpoint)
        missing = [k for k in TRUNK_KEYS if k not in sd]
        if missing:
            raise ValueError(
                f"checkpoint is missing trunk key(s) {missing}; not a "
                "TilePolicyNetwork-layout specialist"
            )
        with torch.no_grad():
            for k in TRUNK_KEYS:
                own = self._resolve_param(k)
                if sd[k].shape != own.shape:
                    raise ValueError(
                        f"trunk shape mismatch at {k}: checkpoint "
                        f"{tuple(sd[k].shape)} vs this net {tuple(own.shape)}"
                    )
                own.copy_(sd[k])

    def init_head_from(self, level: LevelKey, checkpoint) -> None:
        """Copy actor/critic weights from an existing specialist
        checkpoint into ONE level's head, leaving every other head and
        the trunk untouched."""
        idx = self._level_index(level)
        sd = _resolve_state_dict(checkpoint)
        missing = [k for k in HEAD_KEYS if k not in sd]
        if missing:
            raise ValueError(
                f"checkpoint is missing head key(s) {missing}; not a "
                "TilePolicyNetwork-layout specialist"
            )
        actor = self.actor_heads[idx]
        critic = self.critic_heads[idx]
        with torch.no_grad():
            for own, key in (
                (actor.weight, "actor.weight"), (actor.bias, "actor.bias"),
                (critic.weight, "critic.weight"), (critic.bias, "critic.bias"),
            ):
                if sd[key].shape != own.shape:
                    raise ValueError(
                        f"head shape mismatch at {key}: checkpoint "
                        f"{tuple(sd[key].shape)} vs this net {tuple(own.shape)}"
                    )
                own.copy_(sd[key])

    # ----- bookkeeping ---------------------------------------------------

    @property
    def trunk_param_count(self) -> int:
        return sum(
            p.numel()
            for mod_name in ("fc1", "norm1", "fc2", "norm2")
            for p in getattr(self, mod_name).parameters()
        )

    @property
    def head_param_count(self) -> int:
        """Params in ONE actor+critic head pair (every head is the same
        shape, so any index gives the per-head count)."""
        return sum(p.numel() for p in self.actor_heads[0].parameters()) + sum(
            p.numel() for p in self.critic_heads[0].parameters()
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
