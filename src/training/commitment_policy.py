"""Action-commitment options: the policy decides (what, for how long).

Pre-registration: docs/proposals/OPTIONS_PREREG_2026-08-22.md. The
mechanism in one line: sticky noise only hurts when the repeat differs
from the chosen action, so committing to one primitive for k steps makes
the noise a no-op inside the commitment instead of fighting it.

The policy's output head is over PAIRS (primitive a, duration k) — 6x3 =
18 decisions — but the environment interface stays primitive-by-step:
each forward emits logits over the 6 primitives, near-deterministic on
the committed one while a commitment is running. The harness (sticky,
jitter, greedy/sampled selection) is untouched; only the policy's
internal structure changes. That is what keeps every banked number
comparable.

Warm start: `from_flat_policy` copies trunk and critic from a banked
6-action policy and initializes the 18-way head by replicating each
primitive's actor row across its durations — the initial distribution
over PRIMITIVES matches the seed policy exactly; durations start
uniform. A warm start that changed the primitive distribution would be
confounded before the first update.

Training integration (semi-MDP GAE, per-decision log-probs) follows the
v22 consultation; this module already records everything that update
needs: which steps were decisions, the pair chosen, and its log-prob.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DURATIONS = (1, 2, 4)
# The committed primitive's exposed logit margin. Large enough that both
# greedy and sampled harness selection follow the commitment; finite so
# nothing NaNs — the exact lesson of the Phase-3 -inf defect.
COMMIT_LOGIT = 30.0


@dataclass
class CommitState:
    """Per-worker commitment state. Tensors, batched over workers."""
    pair: torch.Tensor        # (B,) long — committed pair index, -1 if none
    remaining: torch.Tensor   # (B,) long — env steps left in the commitment

    @classmethod
    def initial(cls, batch: int) -> "CommitState":
        return cls(pair=torch.full((batch,), -1, dtype=torch.long),
                   remaining=torch.zeros(batch, dtype=torch.long))

    def clone(self) -> "CommitState":
        return CommitState(self.pair.clone(), self.remaining.clone())


class CommitmentPolicy(nn.Module):
    """18-way (action, duration) head over a shared trunk, with a timer.

    `step()` is the single entry point for both rollout and (later)
    update — the Phase-3 rule that behaviour and target must be the same
    object. It returns primitive logits for the harness AND the decision
    record the semi-MDP update needs.
    """

    def __init__(self, trunk: nn.Module, trunk_dim: int, num_primitives: int,
                 durations: tuple = DEFAULT_DURATIONS):
        super().__init__()
        if not durations or any(d < 1 for d in durations):
            raise ValueError(f"durations must be >=1, got {durations}")
        self.trunk = trunk
        self.num_primitives = int(num_primitives)
        self.durations = tuple(int(d) for d in durations)
        self.num_pairs = self.num_primitives * len(self.durations)
        self.pair_actor = nn.Linear(trunk_dim, self.num_pairs)
        self.critic = nn.Linear(trunk_dim, 1)

    # ---- pair index arithmetic (pure, no tensors needed) ----
    def pair_of(self, primitive: int, k_idx: int) -> int:
        return primitive * len(self.durations) + k_idx

    def primitive_of(self, pair: torch.Tensor) -> torch.Tensor:
        return pair // len(self.durations)

    def duration_of(self, pair: torch.Tensor) -> torch.Tensor:
        d = torch.tensor(self.durations, device=pair.device)
        return d[pair % len(self.durations)]

    def step(self, obs: torch.Tensor, state: CommitState,
             sample: bool = True):
        """One env step for a batch of workers.

        Returns (prim_logits, values, new_state, record) where record has
        `decision` (B,) bool — True where a NEW pair was chosen this step
        — plus `pair` and `log_prob` valid at decision positions. Workers
        mid-commitment get COMMIT_LOGIT on their committed primitive and
        no decision record; their timer decrements.
        """
        b = obs.shape[0]
        h = self.trunk(obs)
        values = self.critic(h).squeeze(-1)
        pair_logits = self.pair_actor(h)

        deciding = state.remaining <= 0
        new_state = state.clone()

        # Decisions where the timer has run out.
        pair = torch.full((b,), -1, dtype=torch.long)
        log_prob = torch.zeros(b)
        if deciding.any():
            lp_all = F.log_softmax(pair_logits[deciding], dim=-1)
            if sample:
                chosen = torch.multinomial(lp_all.exp(), 1).squeeze(-1)
            else:
                chosen = lp_all.argmax(-1)
            pair[deciding] = chosen
            log_prob[deciding] = lp_all.gather(
                1, chosen.unsqueeze(1)).squeeze(1)
            new_state.pair[deciding] = chosen
            new_state.remaining[deciding] = self.duration_of(chosen)

        # Everyone now has a live commitment; emit its primitive and tick.
        prim = self.primitive_of(new_state.pair.clamp(min=0))
        prim_logits = torch.zeros(b, self.num_primitives)
        prim_logits.scatter_(1, prim.unsqueeze(1), COMMIT_LOGIT)
        new_state.remaining = new_state.remaining - 1

        record = {"decision": deciding, "pair": pair, "log_prob": log_prob,
                  "pair_logits": pair_logits}
        return prim_logits, values, new_state, record

    @classmethod
    def from_flat_policy(cls, flat: nn.Module, trunk_dim: int,
                         num_primitives: int,
                         durations: tuple = DEFAULT_DURATIONS):
        """Warm start from a banked single-step policy.

        The flat policy must expose `.trunk_forward` or be a
        TilePolicyNetwork-shaped module (fc1/norm1/fc2/norm2 + actor +
        critic). Actor row for primitive a is REPLICATED across a's
        durations, so softmax over pairs marginalizes to the seed
        policy's distribution over primitives times a uniform over
        durations. The critic copies unchanged.
        """
        import copy
        trunk = _TrunkView(copy.deepcopy(flat))
        p = cls(trunk, trunk_dim, num_primitives, durations)
        with torch.no_grad():
            aw, ab = flat.actor.weight, flat.actor.bias
            for a in range(num_primitives):
                for ki in range(len(durations)):
                    i = p.pair_of(a, ki)
                    p.pair_actor.weight[i] = aw[a]
                    p.pair_actor.bias[i] = ab[a]
            p.critic.weight.copy_(flat.critic.weight)
            p.critic.bias.copy_(flat.critic.bias)
        return p


class _TrunkView(nn.Module):
    """The banked TilePolicyNetwork minus its heads."""

    def __init__(self, flat: nn.Module):
        super().__init__()
        self.flat = flat

    def forward(self, x):
        f = self.flat
        h = F.silu(f.norm1(f.fc1(x)))
        return F.silu(f.norm2(f.fc2(h)))
