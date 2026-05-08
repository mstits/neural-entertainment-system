"""Unit tests for behavior_cloning module. No ROM required."""

from __future__ import annotations

import torch

from src.training.behavior_cloning import (
    _action_space_bitmasks,
    _nearest_action_index,
    pretrain,
    seed_population_from_weights,
)


def test_action_bitmasks_mario_style() -> None:
    action_space = [
        [],               # NOOP
        ["right"],
        ["right", "A"],
        ["A"],
        ["left"],
    ]
    masks = _action_space_bitmasks(action_space)
    # right = 0x80, A = 0x01
    assert masks == [0x00, 0x80, 0x81, 0x01, 0x40]


def test_nearest_action_index_exact_match() -> None:
    masks = [0x00, 0x80, 0x81, 0x01]
    assert _nearest_action_index(0x80, masks) == 1
    assert _nearest_action_index(0x81, masks) == 2


def test_nearest_action_index_closest_hamming() -> None:
    masks = [0x00, 0x80, 0x01]  # NOOP, RIGHT, A
    # 0x02 (B) is not in the space. Hamming:
    #   vs 0x00: 1 bit diff
    #   vs 0x80: 2 bits diff
    #   vs 0x01: 2 bits diff
    assert _nearest_action_index(0x02, masks) == 0


def test_pretrain_reduces_loss_on_separable_data() -> None:
    """Synthetic states differ strongly by action; pretrain should learn them."""
    torch.manual_seed(0)
    n_per_class = 32
    n_actions = 3

    # Class 0: all-zero states
    # Class 1: all-128 states
    # Class 2: all-255 states
    states = torch.cat([
        torch.zeros((n_per_class, 4, 84, 84), dtype=torch.uint8),
        torch.full((n_per_class, 4, 84, 84), 128, dtype=torch.uint8),
        torch.full((n_per_class, 4, 84, 84), 255, dtype=torch.uint8),
    ])
    actions = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_per_class, dtype=torch.long),
        torch.full((n_per_class,), 2, dtype=torch.long),
    ])

    from src.models.policy_network import PolicyNetwork
    net = PolicyNetwork(num_actions=n_actions)

    final_loss = pretrain(
        net,
        (states, actions),
        epochs=3,
        batch_size=16,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    # Random classifier over 3 classes: -log(1/3) ≈ 1.0986. We want
    # meaningfully below that — the prior `< 1.0` bar passed at 0.99,
    # which is barely-better-than-random and would let a regression
    # through. Halved the threshold for a sharper signal.
    assert final_loss < 0.5


def test_seed_population_from_weights_copies_elite() -> None:
    """Index-0 genome gets clean weights; others get noisy copies."""
    from src.models.policy_network import PolicyNetwork
    from src.training.genetic_algorithm import Genome

    net = PolicyNetwork(num_actions=4)
    src_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    population = [Genome(genome_id=i, state_dict={}) for i in range(4)]
    seed_population_from_weights(population, src_state, noise_std=0.01)

    # Index 0: clean copy — matches exactly.
    first_key = next(iter(src_state.keys()))
    assert torch.allclose(population[0].state_dict[first_key], src_state[first_key])
    # Index 1+: noisy — differs.
    assert not torch.allclose(population[1].state_dict[first_key], src_state[first_key])
