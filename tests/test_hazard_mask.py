"""src/training/hazard_mask.py — Phase 3's action veto.

The properties that matter for a fair experiment: disabled must be a bit-
exact no-op so the control arm shares the treatment's code path, a state
where everything is lethal must not be left with no legal action, and the
veto must be auditable after the fact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.hazard_mask import (  # noqa: E402
    HazardMask, MaskStats, NEG_INF, NEG_MASK, survival_to_death_prob)
from src.training.hazard_model import HazardMLP, NUM_ACTIONS, OBS_DIM


class _Fixed(torch.nn.Module):
    """A stand-in whose death probability is dictated per row."""

    def __init__(self, probs):
        super().__init__()
        self.probs = probs
        self.n_bins = 1

    def forward(self, x):
        n = x.shape[0]
        p = torch.tensor([self.probs[i % len(self.probs)] for i in range(n)])
        p = p.clamp(1e-6, 1 - 1e-6)
        # invert survival_to_death_prob for a single bin
        return torch.log(p / (1 - p)).unsqueeze(-1)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())


def test_disabled_is_bit_exact_identity():
    """The control arm must run the same code path, differing only by flag."""
    m = HazardMask(HazardMLP(n_bins=4), enabled=False)
    logits = torch.randn(5, NUM_ACTIONS)
    assert torch.equal(m.apply(logits, torch.zeros(5, OBS_DIM)), logits)
    assert m.stats.steps == 0, "disabled mask must not even record"


def test_lethal_actions_are_vetoed():
    m = HazardMask(_Fixed([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.9,
                   enabled=True)
    out = m.apply(torch.zeros(1, NUM_ACTIONS), torch.zeros(1, OBS_DIM))
    assert out[0, 0] == NEG_INF
    assert (out[0, 1:] == 0).all()


def test_a_fully_lethal_state_drops_the_mask_entirely():
    """Never leave a state with no legal action.

    Mid-air over a pit every action may be fatal, and those are exactly
    the moments that decide an episode.
    """
    m = HazardMask(_Fixed([0.99]), threshold=0.9, enabled=True)
    logits = torch.zeros(1, NUM_ACTIONS)
    out = m.apply(logits, torch.zeros(1, OBS_DIM))
    assert torch.equal(out, logits), "policy left with no legal action"
    assert m.stats.n_fully_vetoed == 1
    assert m.stats.actions_vetoed == 0, "fully-vetoed steps veto nothing"


def test_fully_vetoed_states_are_counted_for_audit():
    """A run where this is large is one where the veto was mostly inert."""
    m = HazardMask(_Fixed([0.99]), threshold=0.9, enabled=True)
    for _ in range(3):
        m.apply(torch.zeros(2, NUM_ACTIONS), torch.zeros(2, OBS_DIM))
    s = m.stats.as_dict()
    assert s["n_fully_vetoed"] == 6 and s["fully_vetoed_fraction"] == 1.0


def test_threshold_is_exclusive_and_validated():
    m = HazardMask(_Fixed([0.9]), threshold=0.9, enabled=True)
    out = m.apply(torch.zeros(1, NUM_ACTIONS), torch.zeros(1, OBS_DIM))
    assert (out == 0).all(), "p == threshold must not be vetoed"
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            HazardMask(HazardMLP(n_bins=2), threshold=bad)


def test_wrong_observation_width_is_refused():
    """The mask must see the observation the model was trained on."""
    m = HazardMask(HazardMLP(n_bins=2), enabled=True)
    with pytest.raises(ValueError, match="obs of width"):
        m.apply(torch.zeros(1, NUM_ACTIONS), torch.zeros(1, OBS_DIM - 3))


def test_substrate_is_frozen():
    """Phase 3 tests a fixed prior; PPO gradients must not reach it."""
    model = HazardMLP(n_bins=3)
    HazardMask(model, enabled=True)
    assert all(not p.requires_grad for p in model.parameters())


def test_survival_to_death_prob_matches_the_product_form():
    logits = torch.tensor([[0.0, 0.0]])          # hazard 0.5 per bin
    got = float(survival_to_death_prob(logits)[0])
    assert abs(got - (1 - 0.25)) < 1e-6


def test_death_prob_is_monotone_in_hazard():
    lo = survival_to_death_prob(torch.tensor([[-4.0, -4.0]]))
    hi = survival_to_death_prob(torch.tensor([[4.0, 4.0]]))
    assert float(lo) < float(hi)


def test_stats_round_trip_as_dict():
    s = MaskStats(steps=10, actions_vetoed=6, n_fully_vetoed=2,
                  per_action=[1, 2, 3, 0, 0, 0])
    d = s.as_dict()
    assert d["veto_fraction"] == round(6 / 60, 4)
    assert d["fully_vetoed_fraction"] == 0.2


# ---- the wrapper: the veto belongs to the policy, not the sampler ----

class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(OBS_DIM, NUM_ACTIONS)
        self.v = torch.nn.Linear(OBS_DIM, 1)
        self.tag = "base"

    def forward_ac(self, s):
        return self.lin(s), self.v(s)

    def forward(self, s):
        return self.lin(s)


def test_wrapper_masks_inside_forward_ac_so_both_passes_agree():
    """Rollout and update both call forward_ac; if only the sampler
    masked, PPO's ratio would correct for a policy that never acted."""
    from src.training.hazard_mask import HazardMaskedPolicy
    net = _Net()
    m = HazardMask(_Fixed([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.9,
                   enabled=True)
    p = HazardMaskedPolicy(net, m)
    obs = torch.zeros(3, OBS_DIM)
    a, _ = p.forward_ac(obs)
    b, _ = p.forward_ac(obs)
    assert (a[:, 0] == NEG_INF).all()
    assert torch.equal(a, b), "same obs must yield the same veto"


def test_wrapper_is_transparent_when_disabled():
    from src.training.hazard_mask import HazardMaskedPolicy
    net = _Net()
    p = HazardMaskedPolicy(net, HazardMask(HazardMLP(n_bins=2),
                                           enabled=False))
    obs = torch.zeros(2, OBS_DIM)
    assert torch.equal(p.forward_ac(obs)[0], net.forward_ac(obs)[0])


def test_checkpoint_written_through_the_wrapper_loads_into_a_plain_net():
    """The mask is an experiment arm, not a change to the artifact."""
    from src.training.hazard_mask import HazardMaskedPolicy
    net = _Net()
    p = HazardMaskedPolicy(net, HazardMask(HazardMLP(n_bins=2), enabled=True))
    fresh = _Net()
    fresh.load_state_dict(p.state_dict())
    assert torch.equal(fresh.lin.weight, net.lin.weight)
    assert all(not k.startswith("net.") for k in p.state_dict())


def test_wrapper_delegates_unknown_attributes():
    from src.training.hazard_mask import HazardMaskedPolicy
    p = HazardMaskedPolicy(_Net(), HazardMask(HazardMLP(n_bins=2)))
    assert p.tag == "base"


def test_wrapper_refuses_forward_ac_recurrent_instead_of_bypassing_the_veto():
    """__getattr__ delegates unknown names to the raw net; without an
    explicit override, that would run the recurrent step path fully
    unmasked with no exception, warning, or log. This must fail loud
    instead."""
    from src.training.hazard_mask import HazardMaskedPolicy

    class _RecurrentNet(_Net):
        def forward_ac_recurrent(self, x, h):
            return self.lin(x), self.v(x), h

    net = _RecurrentNet()
    m = HazardMask(_Fixed([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.9,
                   enabled=True)
    p = HazardMaskedPolicy(net, m)
    with pytest.raises(NotImplementedError, match="forward_ac_recurrent"):
        p.forward_ac_recurrent(torch.zeros(1, OBS_DIM), torch.zeros(1, 4))
    assert m.stats.steps == 0, "the veto must never have run"


# ---- the -inf trap that voided the first Phase-3 masked arm ----

def test_the_veto_value_is_finite():
    """-inf makes entropy 0 * -inf = NaN, the loss NaN, and the trainer's
    NaN guard skips every actor update. 200 iterations produced an actor
    byte-identical to the control."""
    import math
    assert math.isfinite(NEG_MASK)
    assert NEG_MASK < -1e6, "must still drive the probability to zero"


def test_masked_logits_keep_entropy_and_gradients_finite():
    import torch.nn.functional as F
    m = HazardMask(_Fixed([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.9,
                   enabled=True)
    logits = torch.zeros(1, NUM_ACTIONS, requires_grad=True)
    out = m.apply(logits, torch.zeros(1, OBS_DIM))
    lp = F.log_softmax(out, dim=-1)
    ent = -(lp.exp() * lp).sum(-1)
    assert torch.isfinite(ent).all(), "entropy went non-finite"
    ent.sum().backward()
    assert torch.isfinite(logits.grad).all(), "gradients went non-finite"


def test_a_vetoed_action_still_has_zero_probability():
    import torch.nn.functional as F
    m = HazardMask(_Fixed([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.9,
                   enabled=True)
    out = m.apply(torch.zeros(1, NUM_ACTIONS), torch.zeros(1, OBS_DIM))
    assert float(F.softmax(out, dim=-1)[0, 0]) == 0.0
