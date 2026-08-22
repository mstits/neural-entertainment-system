"""Phase 3: veto actions the hazard model says are lethal.

The synthesis's third gated phase, and the first time the hazard model is
asked to change behaviour rather than describe it. Phases 1 and 2 are
banked: 104,640 causal micro-fork labels, and a discrete-time survival
model scoring an IPCW C-index of 0.9170 against a pre-registered 0.85
threshold.

MECHANISM. At each step, every candidate action is scored for the
probability of death within the model's horizon. Any action above
`threshold` has its logit set to -inf, so the policy cannot select it and
the gradient never flows through it. The substrate is FROZEN — this
changes which actions are available, not what the hazard model believes.

THE ESCAPE HATCH IS NOT OPTIONAL. If every action is vetoed the mask is
dropped entirely for that step and the policy chooses freely. A state
where death is likely whatever you do is common in a platformer — mid-air
over a pit, already falling — and masking everything there would leave a
policy with no legal action at exactly the moments that decide an
episode. `n_fully_vetoed` counts those steps, because a run where that
number is large is one where the veto is mostly inert and any result
should be read accordingly.

DEFAULT OFF. `enabled=False` reproduces the unmasked policy bit for bit,
which is what makes the Phase-3 control arm a true control: the same code
path runs in both arms and only the flag differs.

Gate (pre-registered, docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md):
strict honest clear rate over >=100 episodes must improve >=20% relative
to the unmasked control. Below that, Phase 3 fails and the synthesis says
terminate the substrate experiment rather than tune the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.training.hazard_model import HazardMLP, NUM_ACTIONS, OBS_DIM

DEFAULT_THRESHOLD = 0.90

# A large finite negative, NOT -inf, and the difference decided a whole
# experiment. With -inf, log_softmax yields -inf for a vetoed action and
# its probability is 0, so the entropy term computes 0 * -inf = NaN. The
# loss goes NaN, the gradients go NaN, and the trainer's NaN guard skips
# the actor update -- silently. The first Phase-3 masked arm ran 200
# iterations that way: only critic.weight and critic.bias differed from
# the control at the end, the actor was byte-identical, and both arms
# then evaluated to exactly 0.28 / 0.34 because they WERE the same actor.
# Read at face value that is "hazard masking gives 0% improvement", which
# would have terminated the substrate experiment on a bug.
#
# -1e9 gives a vetoed action probability 0.0 to float precision -- the
# same behaviour -- while keeping the entropy finite and the gradients
# clean.
NEG_MASK = -1.0e9
NEG_INF = NEG_MASK          # retained name; no longer literally infinite


@dataclass
class MaskStats:
    """Telemetry. A veto you cannot audit is a veto you cannot trust."""
    steps: int = 0
    actions_vetoed: int = 0
    n_fully_vetoed: int = 0
    per_action: list = field(default_factory=lambda: [0] * NUM_ACTIONS)

    def as_dict(self) -> dict:
        frac = self.actions_vetoed / max(1, self.steps * NUM_ACTIONS)
        return {"steps": self.steps, "actions_vetoed": self.actions_vetoed,
                "veto_fraction": round(frac, 4),
                "n_fully_vetoed": self.n_fully_vetoed,
                "fully_vetoed_fraction": round(
                    self.n_fully_vetoed / max(1, self.steps), 4),
                "per_action": list(self.per_action)}


def survival_to_death_prob(logits: torch.Tensor) -> torch.Tensor:
    """P(death within the horizon) from per-bin discrete hazard logits.

    The model emits a hazard per time bin. Survival across the horizon is
    the product of (1 - h_i), so death is one minus that. Computed in log
    space via logsigmoid for the same numerical reason the training loss
    is: a product of twenty near-one terms underflows cheerfully in float.
    """
    log_surv = torch.nn.functional.logsigmoid(-logits).sum(dim=-1)
    return 1.0 - torch.exp(log_surv)


class HazardMask:
    """Scores every action for lethality and vetoes the worst.

    Frozen and in eval mode: Phase 3 tests the model as a fixed prior, so
    letting PPO's gradients reach it would be testing something else.
    """

    def __init__(self, model: HazardMLP, threshold: float = DEFAULT_THRESHOLD,
                 enabled: bool = False, device: str = "cpu"):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.threshold = float(threshold)
        self.enabled = bool(enabled)
        self.device = device
        self.stats = MaskStats()

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kw) -> "HazardMask":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        # train_hazard writes {state_dict, config, provenance}. The shape
        # is read from the checkpoint's own config rather than assumed,
        # so a model trained at a different width still loads.
        state = blob.get("state_dict", blob.get("model_state_dict", blob))
        cfg = blob.get("config", {}) if isinstance(blob, dict) else {}
        model = HazardMLP(hidden=int(cfg.get("hidden", 128)),
                          n_hidden_layers=int(cfg.get("hidden_layers", 2)),
                          n_bins=int(cfg.get("bins", cfg.get("n_bins", 20))))
        model.load_state_dict(state)
        return cls(model, **kw)

    @torch.no_grad()
    def death_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, NUM_ACTIONS) death probability for every action from obs.

        One batched pass over B*NUM_ACTIONS rows rather than a loop: the
        mask runs on every environment step, so a Python loop over actions
        would tax the rollout far more than the model itself does.
        """
        b = obs.shape[0]
        rep = obs.unsqueeze(1).expand(b, NUM_ACTIONS, OBS_DIM)
        eye = torch.eye(NUM_ACTIONS, device=obs.device).unsqueeze(0).expand(
            b, NUM_ACTIONS, NUM_ACTIONS)
        x = torch.cat([rep, eye], dim=-1).reshape(b * NUM_ACTIONS, -1)
        return survival_to_death_prob(self.model(x)).reshape(b, NUM_ACTIONS)

    def apply(self, logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Return logits with lethal actions vetoed. Disabled = identity."""
        if not self.enabled:
            return logits
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(
                f"hazard mask expects obs of width {OBS_DIM}, got "
                f"{obs.shape[-1]} — the mask must see the same observation "
                f"the model was trained on")
        risky = self.death_probs(obs.to(self.device)) > self.threshold
        # Escape hatch: never leave a state with no legal action.
        all_bad = risky.all(dim=-1)
        self.stats.steps += int(logits.shape[0])
        self.stats.n_fully_vetoed += int(all_bad.sum().item())
        risky = risky & ~all_bad.unsqueeze(-1)
        self.stats.actions_vetoed += int(risky.sum().item())
        counts = risky.sum(dim=0).tolist()
        self.stats.per_action = [a + int(b) for a, b
                                 in zip(self.stats.per_action, counts)]
        return logits.masked_fill(risky.to(logits.device), NEG_INF)


class HazardMaskedPolicy(torch.nn.Module):
    """A policy whose lethal actions are vetoed, wrapping the real net.

    The veto belongs to the POLICY, not to the sampling site. Masking only
    where actions are drawn would leave PPO's update recomputing logits
    from the unmasked network, so the behaviour distribution and the
    target distribution would differ and the importance ratio would be
    wrong — the update would be correcting for a policy that never acted.

    Both the rollout and the updater call `net.forward_ac(states)`, so
    applying the mask inside that one method makes the two consistent by
    construction. The hazard model is frozen and deterministic, so the
    same observation yields the same veto in both passes.

    Everything else delegates to the wrapped net, deliberately including
    `state_dict`. A checkpoint written through this wrapper must be
    loadable by the plain policy: the mask is an experiment arm, not a
    change to the artifact, and a run that could only be resumed through
    the wrapper would quietly make the arm permanent.
    """

    def __init__(self, net: torch.nn.Module, mask: HazardMask):
        super().__init__()
        self.net = net
        self.mask = mask

    def forward_ac(self, states: torch.Tensor):
        logits, values = self.net.forward_ac(states)
        if not self.mask.enabled:
            return logits, values
        obs = states.reshape(states.shape[0], -1).float()
        return self.mask.apply(logits, obs), values

    def forward(self, *a, **k):
        return self.net.forward(*a, **k)

    # --- delegation: the wrapper must be invisible to everything else ---
    def state_dict(self, *a, **k):
        return self.net.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.net.load_state_dict(*a, **k)

    def parameters(self, *a, **k):
        return self.net.parameters(*a, **k)

    def named_parameters(self, *a, **k):
        return self.net.named_parameters(*a, **k)

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def eval(self):
        self.net.eval()
        return self

    def to(self, *a, **k):
        self.net.to(*a, **k)
        return self

    def __getattr__(self, name):
        # nn.Module.__getattr__ handles registered submodules first; this
        # only fires for genuinely unknown names, which belong to the net.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["net"], name)
