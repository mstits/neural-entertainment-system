"""Tests for the genome-naming system."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.genetic_algorithm import GeneticAlgorithm
from src.training.genome_names import GenomeNamer, _base_names


def _tiny_net_factory():
    return nn.Sequential(nn.Linear(4, 2))


def test_namer_issues_unique_names_within_pool() -> None:
    n = GenomeNamer(seed=42)
    names = [n.next_name() for _ in range(50)]
    # First 50 draws should all be unique (pool is >100).
    assert len(set(names)) == 50


def test_namer_seed_reproducible() -> None:
    a = GenomeNamer(seed=7)
    b = GenomeNamer(seed=7)
    assert [a.next_name() for _ in range(10)] == [b.next_name() for _ in range(10)]


def test_namer_falls_back_to_suffixes_when_pool_exhausted() -> None:
    pool_size = len(_base_names())
    n = GenomeNamer(seed=0)
    drawn = [n.next_name() for _ in range(pool_size + 5)]
    # Pool exhausted → next names carry a numeric suffix.
    suffixed = [x for x in drawn if "-" in x]
    assert len(suffixed) == 5
    # And they're all still unique across the whole draw.
    assert len(set(drawn)) == pool_size + 5


def test_ga_assigns_unique_names_to_initial_population() -> None:
    ga = GeneticAlgorithm(
        population_size=12,
        network_factory=_tiny_net_factory,
        seed=123,
    )
    ga.initialize()
    names = [g.name for g in ga.population]
    assert all(n for n in names), "every genome must have a non-empty name"
    assert len(set(names)) == len(names), "names must be unique within the pop"


def test_ga_elite_keeps_name_across_generations() -> None:
    ga = GeneticAlgorithm(
        population_size=6,
        network_factory=_tiny_net_factory,
        elite_fraction=0.5,
        seed=1,
    )
    ga.initialize()
    for g in ga.population:
        g.fitness = float(g.genome_id)  # deterministic ranking
    elite_before = sorted(ga.population, key=lambda g: g.fitness, reverse=True)[:3]
    elite_names_before = {g.name for g in elite_before}
    ga.evolve()
    elite_names_after = {g.name for g in ga.population[:3]}
    # Top-3 keep their names; new children below them get fresh names.
    assert elite_names_before == elite_names_after


def test_ga_checkpoint_roundtrips_names(tmp_path) -> None:
    ga = GeneticAlgorithm(
        population_size=4,
        network_factory=_tiny_net_factory,
        seed=9,
    )
    ga.initialize()
    for g in ga.population:
        g.fitness = 1.0
    path = tmp_path / "ga.pt"
    ga.save_checkpoint(str(path))

    ga2 = GeneticAlgorithm(
        population_size=4,
        network_factory=_tiny_net_factory,
        seed=9,
    )
    ga2.load_checkpoint(str(path))
    names_before = [g.name for g in ga.population]
    names_after = [g.name for g in ga2.population]
    assert names_before == names_after
