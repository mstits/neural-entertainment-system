"""src/training/commitment_policy.py — the commitment state machine.

The properties that make the experiment readable: the warm start must
not change the primitive distribution, a commitment must actually hold
for its duration, and the decision record must be exactly what a
semi-MDP update needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.commitment_policy import (  # noqa: E402
    COMMIT_LOGIT, CommitState, CommitmentPolicy)


class _Flat(nn.Module):
    """Minimal TilePolicyNetwork-shaped module."""

    def __init__(self, obs=32, hid=16, trunk=8, acts=6):
        super().__init__()
        self.fc1 = nn.Linear(obs, hid)
        self.norm1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, trunk)
        self.norm2 = nn.LayerNorm(trunk)
        self.actor = nn.Linear(trunk, acts)
        self.critic = nn.Linear(trunk, 1)


def _policy(**kw):
    trunk = nn.Sequential(nn.Linear(32, 8), nn.SiLU())
    return CommitmentPolicy(trunk, 8, 6, **kw)


def test_pair_arithmetic_round_trips():
    p = _policy()
    for a in range(6):
        for ki in range(3):
            i = p.pair_of(a, ki)
            assert int(p.primitive_of(torch.tensor([i]))) == a
            assert int(p.duration_of(torch.tensor([i]))) == p.durations[ki]


def test_a_commitment_holds_for_its_full_duration():
    torch.manual_seed(0)
    p = _policy()
    obs = torch.randn(1, 32)
    st = CommitState.initial(1)
    lg, _, st, rec = p.step(obs, st)
    assert bool(rec["decision"][0])
    pair = int(rec["pair"][0])
    k = int(p.duration_of(torch.tensor([pair])))
    prim = int(p.primitive_of(torch.tensor([pair])))
    assert int(lg.argmax(-1)) == prim
    for _ in range(k - 1):                      # rest of the commitment
        lg, _, st, rec = p.step(torch.randn(1, 32), st)
        assert not bool(rec["decision"][0]), "decided mid-commitment"
        assert int(lg.argmax(-1)) == prim, "primitive changed mid-commitment"
    _, _, st, rec = p.step(torch.randn(1, 32), st)
    assert bool(rec["decision"][0]), "no new decision after expiry"


def test_committed_logits_dominate_for_greedy_and_sampled_selection():
    p = _policy()
    lg, _, _, _ = p.step(torch.randn(3, 32), CommitState.initial(3))
    probs = F.softmax(lg, dim=-1)
    assert float(probs.max(-1).values.min()) > 0.999999
    assert torch.isfinite(lg).all(), "the Phase-3 -inf lesson"


def test_greedy_mode_is_deterministic():
    p = _policy()
    obs = torch.randn(5, 32)
    r1 = p.step(obs, CommitState.initial(5), sample=False)[3]["pair"]
    r2 = p.step(obs, CommitState.initial(5), sample=False)[3]["pair"]
    assert torch.equal(r1, r2)


def test_log_prob_is_recorded_at_decisions_and_zero_elsewhere():
    torch.manual_seed(1)
    p = _policy(durations=(2,))
    st = CommitState.initial(2)
    _, _, st, rec = p.step(torch.randn(2, 32), st)
    assert (rec["log_prob"] < 0).all()          # real log-probs
    _, _, st, rec = p.step(torch.randn(2, 32), st)
    assert not rec["decision"].any()
    assert (rec["log_prob"] == 0).all()
    assert (rec["pair"] == -1).all()


def test_batch_members_decide_independently():
    torch.manual_seed(2)
    p = _policy(durations=(1, 4))
    st = CommitState.initial(8)
    _, _, st, rec = p.step(torch.randn(8, 32), st)
    # force a mix: k=1 workers decide next step, k=4 workers do not
    ks = p.duration_of(rec["pair"])
    _, _, st, rec2 = p.step(torch.randn(8, 32), st)
    for i in range(8):
        assert bool(rec2["decision"][i]) == (int(ks[i]) == 1)


def test_warm_start_marginalizes_to_the_seed_primitive_distribution():
    """The load-bearing init property: replicating actor rows across
    durations means P(primitive) is unchanged and P(duration) uniform —
    a warm start that shifted primitives would confound the arms."""
    torch.manual_seed(3)
    flat = _Flat()
    p = CommitmentPolicy.from_flat_policy(flat, trunk_dim=8,
                                          num_primitives=6)
    x = torch.randn(4, 32)
    h = F.silu(flat.norm2(flat.fc2(F.silu(flat.norm1(flat.fc1(x))))))
    flat_probs = F.softmax(flat.actor(h), dim=-1)
    pair_probs = F.softmax(p.pair_actor(p.trunk(x)), dim=-1)
    marg = pair_probs.reshape(4, 6, 3).sum(-1)
    assert torch.allclose(marg, flat_probs, atol=1e-5)
    within = pair_probs.reshape(4, 6, 3)
    within = within / within.sum(-1, keepdim=True)
    assert torch.allclose(within, torch.full_like(within, 1 / 3), atol=1e-5)


def test_warm_start_copies_the_critic():
    flat = _Flat()
    p = CommitmentPolicy.from_flat_policy(flat, trunk_dim=8,
                                          num_primitives=6)
    x = torch.randn(3, 32)
    h = F.silu(flat.norm2(flat.fc2(F.silu(flat.norm1(flat.fc1(x))))))
    assert torch.allclose(p.critic(p.trunk(x)), flat.critic(h), atol=1e-5)


def test_invalid_durations_are_rejected():
    for bad in ((), (0,), (-1, 2)):
        with pytest.raises(ValueError):
            _policy(durations=bad)
