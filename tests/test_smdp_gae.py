"""Semi-MDP GAE math — each v22 mechanic pinned before trainer wiring."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.smdp_gae import (  # noqa: E402
    discounted_partial_return, scale_entropy_by_duration, smdp_deltas,
    smdp_gae)


def test_partial_return_discounts_within_the_commitment():
    r = torch.tensor([1.0, 1.0, 1.0])
    assert torch.isclose(discounted_partial_return(r, 0.9),
                         torch.tensor(1 + 0.9 + 0.81))
    assert float(discounted_partial_return(torch.empty(0), 0.9)) == 0.0


def test_delta_reduces_to_standard_td_for_k_equals_1():
    d = smdp_deltas(torch.tensor([2.0]), torch.tensor([1]),
                    torch.tensor([5.0]), torch.tensor([4.0]),
                    torch.tensor([1.0]), gamma=0.9)
    assert torch.isclose(d[0], torch.tensor(2.0 + 0.9 * 4.0 - 5.0))


def test_delta_discounts_the_bootstrap_by_realized_duration():
    d = smdp_deltas(torch.tensor([0.0]), torch.tensor([4]),
                    torch.tensor([0.0]), torch.tensor([1.0]),
                    torch.tensor([1.0]), gamma=0.5)
    assert torch.isclose(d[0], torch.tensor(0.5 ** 4))


def test_death_cut_bootstraps_zero_timeout_bootstraps_value():
    death = smdp_deltas(torch.tensor([1.0]), torch.tensor([2]),
                        torch.tensor([0.0]), torch.tensor([9.0]),
                        torch.tensor([0.0]), gamma=0.9)
    timeout = smdp_deltas(torch.tensor([1.0]), torch.tensor([2]),
                          torch.tensor([0.0]), torch.tensor([9.0]),
                          torch.tensor([1.0]), gamma=0.9)
    assert torch.isclose(death[0], torch.tensor(1.0))
    assert torch.isclose(timeout[0], torch.tensor(1.0 + 0.81 * 9.0))


def test_gae_decay_is_exponentiated_by_duration():
    """The v22 point: lambda per-decision would stretch the horizon."""
    deltas = torch.tensor([1.0, 1.0])
    done = torch.tensor([False, True])
    g, l = 0.9, 0.95
    a_k1 = smdp_gae(deltas, torch.tensor([1, 1]), done, g, l)
    a_k4 = smdp_gae(deltas, torch.tensor([4, 4]), done, g, l)
    assert torch.isclose(a_k1[0], torch.tensor(1.0 + (g * l) * 1.0))
    assert torch.isclose(a_k4[0], torch.tensor(1.0 + (g * l) ** 4 * 1.0))
    assert float(a_k4[0]) < float(a_k1[0])


def test_gae_restarts_across_episode_boundaries():
    deltas = torch.tensor([1.0, 100.0])
    done = torch.tensor([True, True])       # first decision ends its episode
    a = smdp_gae(deltas, torch.tensor([2, 2]), done, 0.9, 0.95)
    assert torch.isclose(a[0], torch.tensor(1.0)), "leaked across episodes"


def test_gae_matches_flat_gae_when_all_durations_are_1():
    """Sanity anchor: k==1 everywhere must reproduce ordinary GAE."""
    torch.manual_seed(0)
    n, g, l = 6, 0.99, 0.9
    deltas = torch.randn(n)
    done = torch.tensor([False] * (n - 1) + [True])
    ours = smdp_gae(deltas, torch.ones(n, dtype=torch.long), done, g, l)
    ref = torch.zeros(n)
    run = 0.0
    for i in range(n - 1, -1, -1):
        run = float(deltas[i]) + g * l * run
        ref[i] = run
    assert torch.allclose(ours, ref, atol=1e-5)


def test_entropy_scaling_equalizes_the_per_step_footprint():
    """1000 env steps of k=1 vs k=4 must earn the same total bonus."""
    ent = torch.tensor(0.7)
    k1_total = scale_entropy_by_duration(
        ent.repeat(1000), torch.ones(1000, dtype=torch.long)).sum()
    k4_total = scale_entropy_by_duration(
        ent.repeat(250), torch.full((250,), 4, dtype=torch.long)).sum()
    assert torch.isclose(k1_total, k4_total)
