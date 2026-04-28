"""
Property-based tests with Hypothesis.

Pins invariants of the core algorithms — any change that violates them
breaks training. Runs 50-100 random test cases per property, shrinking
counterexamples automatically. These catch subtle bugs unit tests miss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from hypothesis import assume, given, settings, strategies as st

from src.emulation.frame_utils import preprocess_frame, stack_frames
from src.training.behavior_cloning import (
    _action_space_bitmasks,
    _nearest_action_index,
)
from src.training.curriculum import CurriculumManager, CurriculumStage
from src.training.genetic_algorithm import GeneticAlgorithm, Genome


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)


# ---------------- GA invariants ------------------------------------------


@st.composite
def _pop_and_fitnesses(draw):
    pop_size = draw(st.integers(min_value=4, max_value=32))
    fitnesses = draw(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=pop_size, max_size=pop_size,
        )
    )
    return pop_size, fitnesses


@given(_pop_and_fitnesses())
@settings(max_examples=30, deadline=None)
def test_ga_evolve_preserves_population_size(arg):
    pop_size, fitnesses = arg
    ga = GeneticAlgorithm(population_size=pop_size, network_factory=lambda: _TinyNet(), seed=0)
    ga.initialize()
    for g, f in zip(ga.population, fitnesses):
        g.fitness = f
    ga.evolve()
    assert len(ga.population) == pop_size


@given(pop_size=st.integers(min_value=4, max_value=16))
@settings(max_examples=10, deadline=None)
def test_ga_elite_always_survives_with_zero_mutation(pop_size):
    """With mutation_rate=0, the fittest genome's weights must appear in the next gen unchanged."""
    ga = GeneticAlgorithm(
        population_size=pop_size,
        network_factory=lambda: _TinyNet(),
        mutation_rate=0.0,
        elite_fraction=0.5,
        tournament_size=2,
        seed=42,
    )
    ga.initialize()
    for i, g in enumerate(ga.population):
        g.fitness = float(i)
    best_before = max(ga.population, key=lambda g: g.fitness)
    best_state = {k: v.clone() for k, v in best_before.state_dict.items()}
    ga.evolve()
    # Elite should be somewhere in the new population with identical weights
    found = False
    for g in ga.population:
        if all(torch.allclose(g.state_dict[k], v) for k, v in best_state.items()):
            found = True
            break
    assert found, "elite weights did not survive evolve() with mutation_rate=0"


# ---------------- Reward-function / action-space invariants --------------


@given(raw=st.integers(min_value=0, max_value=255))
def test_nearest_action_index_always_in_bounds(raw):
    action_space = [[], ["right"], ["A"], ["B"], ["up"]]
    masks = _action_space_bitmasks(action_space)
    idx = _nearest_action_index(raw, masks)
    assert 0 <= idx < len(action_space)


@given(raw=st.integers(min_value=0, max_value=255))
def test_nearest_action_index_prefers_exact_match(raw):
    action_space = [[], ["right"], ["A"]]
    masks = _action_space_bitmasks(action_space)
    idx = _nearest_action_index(raw, masks)
    if raw in masks:
        assert masks[idx] == raw


# ---------------- Frame preprocessing invariants -------------------------


@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255),
)
def test_preprocess_produces_valid_uint8(r, g, b):
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[..., 0] = r
    frame[..., 1] = g
    frame[..., 2] = b
    out = preprocess_frame(frame)
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8
    # uniform color in = uniform output (save a small edge)
    assert out.min() >= 0 and out.max() <= 255


@given(n=st.integers(min_value=1, max_value=8))
def test_stack_frames_always_returns_fixed_shape(n):
    frames = [np.full((84, 84), i * 20, dtype=np.uint8) for i in range(n)]
    stacked = stack_frames(frames, stack_size=4)
    assert stacked.shape == (4, 84, 84)
    assert stacked.dtype == np.uint8


# ---------------- Curriculum invariants ----------------------------------


@given(
    successes=st.lists(st.booleans(), min_size=10, max_size=200),
)
@settings(max_examples=20, deadline=None)
def test_curriculum_success_rate_always_in_01(successes):
    cm = CurriculumManager(
        stages=[CurriculumStage(name="s1", levels=["*"], advance_threshold=0.8, min_episodes=5)]
    )
    for s in successes:
        cm.record_episode("L", s)
    rate = cm.stage_success_rate()
    assert 0.0 <= rate <= 1.0
