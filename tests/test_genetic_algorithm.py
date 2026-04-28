"""Tests for the weights-as-genome genetic algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.genetic_algorithm import GeneticAlgorithm, Genome


class TinyNet(nn.Module):
    """Deterministic factory produces the same fresh random net each call."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3, bias=True)


def _factory() -> TinyNet:
    return TinyNet()


def test_initialize_creates_population() -> None:
    ga = GeneticAlgorithm(population_size=8, network_factory=_factory, seed=42)
    ga.initialize()
    assert len(ga.population) == 8
    assert all(isinstance(g, Genome) for g in ga.population)
    # Genome IDs are unique
    ids = {g.genome_id for g in ga.population}
    assert len(ids) == 8


def test_evolve_preserves_population_size_and_bumps_generation() -> None:
    ga = GeneticAlgorithm(population_size=8, network_factory=_factory, seed=0)
    ga.initialize()
    for i, g in enumerate(ga.population):
        g.fitness = float(i)
    assert ga.generation == 0
    ga.evolve()
    assert ga.generation == 1
    assert len(ga.population) == 8


def test_elite_survives_unchanged() -> None:
    ga = GeneticAlgorithm(
        population_size=4,
        network_factory=_factory,
        elite_fraction=0.5,  # top 2 survive
        mutation_rate=0.0,   # no mutation
        seed=1,
    )
    ga.initialize()
    for i, g in enumerate(ga.population):
        g.fitness = float(i)  # genome 3 is best, genome 2 second
    best_state = {k: v.clone() for k, v in ga.population[-1].state_dict.items()}

    ga.evolve()
    # After sort+elite, position 0 should be the old top genome
    top_state = ga.population[0].state_dict
    for k, v in best_state.items():
        assert torch.allclose(top_state[k], v), f"elite weight {k} changed"


def test_mutation_applied_when_rate_is_one() -> None:
    ga = GeneticAlgorithm(
        population_size=4,
        network_factory=_factory,
        elite_fraction=0.25,   # only 1 elite — the rest must be mutated
        mutation_rate=1.0,
        mutation_std=1.0,
        seed=7,
    )
    ga.initialize()
    originals = [{k: v.clone() for k, v in g.state_dict.items()} for g in ga.population]
    for i, g in enumerate(ga.population):
        g.fitness = float(i)
    ga.evolve()

    # Non-elite children (indexes 1..3) should differ from every original.
    for child in ga.population[1:]:
        differs_from_all = True
        for orig in originals:
            if all(torch.allclose(child.state_dict[k], orig[k]) for k in orig):
                differs_from_all = False
                break
        assert differs_from_all


def test_stale_restart_replaces_bottom_after_no_improvement() -> None:
    """If best-fitness doesn't move for `stale_gens_before_restart` gens,
    the bottom `restart_fraction` of the population gets fresh random
    weights. Elites are preserved."""
    ga = GeneticAlgorithm(
        population_size=8,
        network_factory=_factory,
        elite_fraction=0.25,          # 2 elites
        mutation_rate=0.0,
        stale_gens_before_restart=2,  # trigger quickly
        restart_fraction=0.5,         # wipe 4 genomes
        seed=123,
    )
    ga.initialize()

    # Fitness is constant across all genomes every gen → best never improves.
    for _ in range(2):
        for g in ga.population:
            g.fitness = 1.0
        pre_ids = {g.genome_id for g in ga.population}
        ga.evolve()

    # Third evolve: gens_since_improvement has been incremented twice,
    # hits the threshold, restart should fire on THIS evolve.
    for g in ga.population:
        g.fitness = 1.0
    ids_before = [g.genome_id for g in ga.population]
    ga.evolve()

    # Bottom half should have fresh (never-seen-before) IDs.
    fresh = [g for g in ga.population if g.genome_id not in ids_before]
    assert len(fresh) >= 4, f"expected ≥4 fresh genomes, got {len(fresh)}"


def test_stale_restart_disabled_by_default() -> None:
    ga = GeneticAlgorithm(population_size=4, network_factory=_factory, seed=7)
    assert ga.stale_gens_before_restart == 0  # off unless explicitly set


def test_stale_restart_counter_resets_on_fitness_improvement() -> None:
    ga = GeneticAlgorithm(
        population_size=4,
        network_factory=_factory,
        elite_fraction=0.5,
        mutation_rate=0.0,
        stale_gens_before_restart=3,
        restart_fraction=0.5,
        seed=9,
    )
    ga.initialize()
    # Gen 0: fitness 1.0 — improves from -inf, counter stays 0.
    for g in ga.population:
        g.fitness = 1.0
    ga.evolve()
    assert ga._gens_since_improvement == 0
    # Gen 1: same fitness — counter ticks to 1.
    for g in ga.population:
        g.fitness = 1.0
    ga.evolve()
    assert ga._gens_since_improvement == 1
    # Gen 2: improved — counter resets.
    for g in ga.population:
        g.fitness = 5.0
    ga.evolve()
    assert ga._gens_since_improvement == 0


def test_best_genome_returns_highest_fitness() -> None:
    ga = GeneticAlgorithm(population_size=4, network_factory=_factory, seed=3)
    ga.initialize()
    fits = [10.0, 5.0, 99.0, 1.0]
    for g, f in zip(ga.population, fits):
        g.fitness = f
    best = ga.best_genome()
    assert best.fitness == 99.0


def test_save_load_round_trip(tmp_path) -> None:
    ga = GeneticAlgorithm(population_size=4, network_factory=_factory, seed=5)
    ga.initialize()
    for i, g in enumerate(ga.population):
        g.fitness = float(i * 3)
    ga.evolve()

    ckpt = tmp_path / "ga.pt"
    ga.save_checkpoint(str(ckpt))

    ga2 = GeneticAlgorithm(population_size=4, network_factory=_factory)
    ga2.load_checkpoint(str(ckpt))

    assert ga2.generation == ga.generation
    assert len(ga2.population) == len(ga.population)
    assert [g.genome_id for g in ga2.population] == [g.genome_id for g in ga.population]
